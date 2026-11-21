#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from filterguard.data import labels_texts, load_rows, resolve_dev_eval_indices
from filterguard.metric import confusion, metrics_from_conf, pick_threshold, predict_from_scores
from filterguard.torchutil import resolve_torch_dtype


def _row_id(row: dict[str, Any], index: int) -> str:
    for key in ("id", "record_key", "source_row_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(index)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _eval_metrics(labels: list[int], preds: list[int], indices: list[int]) -> dict[str, Any]:
    y_true = [labels[i] for i in indices]
    y_pred = [preds[i] for i in indices]
    conf = confusion(y_true, y_pred)
    return {"n": len(indices), "conf": conf, "metrics": metrics_from_conf(conf)}


def _prompt_guard_scores(args: argparse.Namespace, texts: list[str]) -> tuple[list[float], dict[str, Any]]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = resolve_torch_dtype(torch=torch, device=device, dtype=args.dtype, arg_name="--dtype")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code, "low_cpu_mem_usage": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, **model_kwargs).eval().to(device)

    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(texts), int(args.batch_size)):
            batch_texts = texts[start : start + int(args.batch_size)]
            batch = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
            ).to(device)
            logits = model(**batch).logits.float()
            if logits.shape[-1] == 1:
                prob = torch.sigmoid(logits[:, 0])
            else:
                prob = torch.softmax(logits, dim=-1)[:, min(1, logits.shape[-1] - 1)]
            scores.extend(float(x) for x in prob.detach().cpu().tolist())

    info = {
        "backend": "sequence_classification",
        "model_id": args.model_id,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "score": "probability_of_class_1; prediction uses dev-tuned threshold and direction",
    }
    return scores, info


def _patch_llama4_config(model: Any, attention_chunk_size: int) -> None:
    cfg_candidates = [getattr(model, "config", None)]
    cfg_candidates.append(getattr(getattr(model, "config", None), "text_config", None))
    cfg_candidates.append(getattr(getattr(model, "model", None), "config", None))
    for cfg_obj in cfg_candidates:
        if cfg_obj is not None and getattr(cfg_obj, "attention_chunk_size", None) is None:
            cfg_obj.attention_chunk_size = int(attention_chunk_size)


def _causal_guard_outputs(args: argparse.Namespace, texts: list[str]) -> tuple[list[int], list[float], dict[str, Any]]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = resolve_torch_dtype(torch=torch, device=device, dtype=args.dtype, arg_name="--dtype")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code, "low_cpu_mem_usage": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs).eval()
    if getattr(getattr(model, "config", None), "use_cache", None) is not None:
        model.config.use_cache = bool(args.use_cache)
    if args.backend == "llama_guard4_legacy":
        _patch_llama4_config(model, int(args.attention_chunk_size))
    model.to(device)

    preds: list[int] = []
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(texts), int(args.batch_size)):
            batch_texts = texts[start : start + int(args.batch_size)]
            prompts: list[str] = []
            for text in batch_texts:
                if args.backend == "llama_guard4_legacy":
                    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
                else:
                    messages = [{"role": "user", "content": text}]
                try:
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    prompt = text
                prompts.append(prompt)

            batch = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
            ).to(device)
            generated = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=int(args.max_new_tokens),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=bool(args.use_cache),
            )
            new_tokens = generated[:, batch["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for out in decoded:
                first = out.strip().splitlines()[0].strip().lower() if out.strip() else ""
                preds.append(1 if first.startswith("unsafe") else 0)
                scores.append(float("nan"))

    info = {
        "backend": "causal_lm_guard_generation",
        "model_id": args.model_id,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "use_cache": bool(args.use_cache),
        "score": "unavailable; prediction is parsed from generated first-line label",
    }
    return preds, scores, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe aggregate-only binary eval for Prompt Guard and Llama Guard")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--backend",
        choices=["prompt_guard", "llama_guard", "llama_guard4_legacy"],
        required=True,
        help="Use scripts/eval_llama_guard4_official_binary.py for the reportable Llama Guard 4 baseline.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--include-assistant", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attention-chunk-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    start_s = time.perf_counter()
    rows = load_rows(str(args.dataset_path), limit=int(args.limit))
    labels, texts = labels_texts(rows, include_assistant=bool(args.include_assistant))
    dev_idx, eval_idx, split_mode = resolve_dev_eval_indices(
        rows,
        labels,
        dev_ratio=float(args.dev_ratio),
        seed=int(args.seed),
    )
    eval_indices = eval_idx if eval_idx else list(range(len(labels)))

    if args.backend == "prompt_guard":
        scores, model_info = _prompt_guard_scores(args, texts)
        dev_scores = [scores[i] for i in dev_idx] if dev_idx else scores
        dev_labels = [labels[i] for i in dev_idx] if dev_idx else labels
        pick = pick_threshold(dev_scores, dev_labels, direction="auto")
        preds = predict_from_scores(scores, threshold=float(pick.threshold), direction=str(pick.direction))
        threshold_info = {"threshold": float(pick.threshold), "direction": str(pick.direction), "dev_acc": float(pick.acc)}
    else:
        preds, scores, model_info = _causal_guard_outputs(args, texts)
        threshold_info = {"threshold": None, "direction": "direct_generation", "dev_acc": None}

    if len(preds) != len(labels):
        raise RuntimeError(f"prediction count mismatch: {len(preds)} != {len(labels)}")

    res_eval = _eval_metrics(labels, preds, eval_indices)
    pos_rate = sum(labels) / len(labels) if labels else float("nan")
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "args": vars(args),
        "model": model_info,
        "split_mode": split_mode,
        "elapsed_s": time.perf_counter() - start_s,
        "privacy": "no source text is written; per-sample artifacts contain ids, hashes, lengths, labels, predictions, and scores only",
    }
    _write_json(run_dir / "run_config.json", config)

    dev_set = set(dev_idx)
    eval_set = set(eval_indices)
    per_rows = []
    for i, (row, label, text, pred, score) in enumerate(zip(rows, labels, texts, preds, scores, strict=True)):
        split = "dev" if i in dev_set else ("eval" if i in eval_set else "unused")
        per_rows.append(
            {
                "index": i,
                "id": _row_id(row, i),
                "split": split,
                "label": int(label),
                "prediction": int(pred),
                "score": None if isinstance(score, float) and math.isnan(score) else float(score),
                "correct": bool(int(label) == int(pred)),
                "text_sha256": _text_hash(text),
                "text_len": len(text),
                "source": str(row.get("source") or row.get("source_benchmark") or ""),
            }
        )
    _write_jsonl(run_dir / "per_sample_results.jsonl", per_rows)

    metrics_payload = {
        "dataset_path": str(args.dataset_path),
        "n": len(labels),
        "pos_rate": pos_rate,
        "dev_n": len(dev_idx),
        "eval_n": res_eval["n"],
        "split_mode": split_mode,
        "backend": args.backend,
        "model_id": args.model_id,
        "threshold": threshold_info,
        "eval_confusion": res_eval["conf"],
        "eval_metrics": res_eval["metrics"],
        "artifact_files": {
            "run_config": "run_config.json",
            "per_sample_results": "per_sample_results.jsonl",
            "aggregate_metrics": "aggregate_metrics.json",
        },
    }
    _write_json(run_dir / "aggregate_metrics.json", metrics_payload)

    print(f"dataset_path={args.dataset_path}")
    print(f"backend={args.backend} model_id={args.model_id}")
    print(f"n={len(labels)} pos_rate={pos_rate:.4f} split_mode={split_mode} dev_n={len(dev_idx)} eval_n={res_eval['n']}")
    print(f"threshold={json.dumps(threshold_info, sort_keys=True)}")
    print(f"eval_acc={res_eval['metrics']['acc']:.4f}")
    print(f"eval_confusion=tp:{res_eval['conf']['tp']} tn:{res_eval['conf']['tn']} fp:{res_eval['conf']['fp']} fn:{res_eval['conf']['fn']}")
    print(f"eval_metrics=precision:{res_eval['metrics']['precision']:.4f} recall:{res_eval['metrics']['recall']:.4f} f1:{res_eval['metrics']['f1']:.4f}")
    print(f"artifacts_dir={run_dir}")


if __name__ == "__main__":
    main()
