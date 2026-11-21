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
from transformers import AutoConfig, AutoProcessor, Llama4ForConditionalGeneration

from filterguard.data import labels_texts, load_rows, resolve_dev_eval_indices
from filterguard.metric import confusion, metrics_from_conf
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


def _patch_llama4_config(config_obj: Any, attention_chunk_size: int) -> None:
    if config_obj is None:
        return
    candidates = [config_obj, getattr(config_obj, "text_config", None), getattr(config_obj, "vision_config", None)]
    for cfg in candidates:
        if cfg is not None and getattr(cfg, "attention_chunk_size", None) is None:
            cfg.attention_chunk_size = int(attention_chunk_size)


def _to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def _parse_first_label(decoded: str) -> int:
    first = decoded.strip().splitlines()[0].strip().lower() if decoded.strip() else ""
    return 1 if first.startswith("unsafe") else 0


def _official_outputs(args: argparse.Namespace, texts: list[str]) -> tuple[list[int], list[float], dict[str, Any]]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = resolve_torch_dtype(torch=torch, device=device, dtype=args.dtype, arg_name="--dtype")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    _patch_llama4_config(config, int(args.attention_chunk_size))
    if getattr(config, "use_cache", None) is not None:
        config.use_cache = bool(args.use_cache)

    model_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = Llama4ForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs).eval()
    _patch_llama4_config(getattr(model, "config", None), int(args.attention_chunk_size))
    if getattr(getattr(model, "config", None), "use_cache", None) is not None:
        model.config.use_cache = bool(args.use_cache)
    model.to(device)

    tokenizer = getattr(processor, "tokenizer", processor)
    preds: list[int] = []
    scores: list[float] = []
    decode_counts: dict[str, int] = {}
    total = len(texts)
    progress_t0 = time.perf_counter()
    with torch.inference_mode():
        for batch_no, start in enumerate(range(0, len(texts), int(args.batch_size)), start=1):
            batch_texts = texts[start : start + int(args.batch_size)]
            messages_batch = [
                [{"role": "user", "content": [{"type": "text", "text": text}]}]
                for text in batch_texts
            ]
            batch = processor.apply_chat_template(
                messages_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
            )
            batch = _to_device(batch, device)
            input_len = int(batch["input_ids"].shape[-1])
            generated = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=int(args.max_new_tokens),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=bool(args.use_cache),
            )
            new_tokens = generated[:, input_len:]
            decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)
            for out in decoded:
                pred = _parse_first_label(out)
                preds.append(pred)
                scores.append(float("nan"))
                key = "unsafe" if pred == 1 else "safe_or_other"
                decode_counts[key] = decode_counts.get(key, 0) + 1
            if int(args.status_every) > 0 and (batch_no % int(args.status_every) == 0 or len(preds) == total):
                elapsed = max(time.perf_counter() - progress_t0, 1e-6)
                done = len(preds)
                rate = done / elapsed
                eta = (total - done) / max(rate, 1e-6)
                print(
                    (
                        f"progress={done}/{total} batches={batch_no} "
                        f"rate={rate:.3f}_samples_s eta_s={eta:.1f} "
                        f"decoded={json.dumps(decode_counts, sort_keys=True)}"
                    ),
                    flush=True,
                )

    info = {
        "backend": "official_llama_guard4_processor_generation",
        "model_id": args.model_id,
        "model_class": "Llama4ForConditionalGeneration",
        "processor_class": processor.__class__.__name__,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "attention_chunk_size": int(args.attention_chunk_size),
        "use_cache": bool(args.use_cache),
        "score": "unavailable; prediction is parsed from generated first-line label",
        "decoded_prediction_counts": decode_counts,
    }
    return preds, scores, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Official-pipeline binary eval for Llama Guard 4 on text JSONL")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--include-assistant", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--attention-chunk-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--status-every", type=int, default=20, help="Print progress every N batches; 0 disables.")
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

    preds, scores, model_info = _official_outputs(args, texts)
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
        "privacy": (
            "no source text is written; per-sample artifacts contain ids, hashes, lengths, "
            "labels, predictions, and null scores only"
        ),
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
        "backend": "official_llama_guard4",
        "model_id": str(args.model_id),
        "threshold": {"threshold": None, "direction": "direct_generation", "dev_acc": None},
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
    print(f"backend=official_llama_guard4 model_id={args.model_id}")
    print(f"n={len(labels)} pos_rate={pos_rate:.4f} split_mode={split_mode} dev_n={len(dev_idx)} eval_n={res_eval['n']}")
    print('threshold={"dev_acc": null, "direction": "direct_generation", "threshold": null}')
    print(f"eval_acc={res_eval['metrics']['acc']:.4f}")
    print(
        "eval_confusion="
        f"tp:{res_eval['conf']['tp']} tn:{res_eval['conf']['tn']} "
        f"fp:{res_eval['conf']['fp']} fn:{res_eval['conf']['fn']}"
    )
    print(
        "eval_metrics="
        f"precision:{res_eval['metrics']['precision']:.4f} "
        f"recall:{res_eval['metrics']['recall']:.4f} "
        f"f1:{res_eval['metrics']['f1']:.4f}"
    )
    print(f"decoded_prediction_counts={json.dumps(model_info['decoded_prediction_counts'], sort_keys=True)}")
    print(f"elapsed_s={config['elapsed_s']:.2f}")
    print(f"artifacts_dir={run_dir}")


if __name__ == "__main__":
    main()
