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
from dataclasses import asdict
from pathlib import Path
from typing import Any

from filterguard.data import abnormal_label, load_rows, resolve_dev_eval_indices
from filterguard.metric import confusion, extract_text, metrics_from_conf, pick_threshold, predict_from_scores
from filterguard.token_anomaly import (
    load_causal_lm,
    score_texts_nll_spike_with_details,
    score_texts_onion_delta_with_details,
)


def _extract_assistant_text(ex: dict[str, Any]) -> str:
    messages = ex.get("messages")
    if isinstance(messages, list) and messages:
        parts: list[str] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip()
            content = str(m.get("content") or "").strip()
            if role == "assistant" and content:
                parts.append(content)
        if parts:
            return "\n".join(parts)
    output = ex.get("output")
    return str(output or "").strip()


def _labels_texts_for_scope(rows: list[dict[str, Any]], *, text_scope: str) -> tuple[list[int], list[str]]:
    labels: list[int] = []
    texts: list[str] = []
    for ex in rows:
        labels.append(abnormal_label(ex))
        if text_scope == "user":
            texts.append(extract_text(ex, include_assistant=False))
        elif text_scope == "full":
            texts.append(extract_text(ex, include_assistant=True))
        elif text_scope == "assistant":
            texts.append(_extract_assistant_text(ex))
        else:
            raise ValueError("--text-scope must be one of: user, assistant, full")
    return labels, texts


def _row_id(ex: dict[str, Any], idx: int) -> str:
    value = ex.get("id")
    if value is None:
        value = ex.get("record_key")
    if value is None:
        value = ex.get("source_row_id")
    return str(value if value is not None else idx)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _artifact_split(idx: int, dev_idx: set[int], eval_idx: set[int]) -> str:
    if idx in dev_idx:
        return "dev"
    if idx in eval_idx:
        return "eval"
    return "unused"


def _save_artifacts(
    *,
    output_dir: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    labels: list[int],
    texts: list[str],
    scores: list[float],
    details: list[Any],
    predictions: list[int],
    dev_idx: list[int],
    eval_idx: list[int],
    split_mode: str,
    threshold: float,
    direction: str,
    dev_acc: float,
    res_eval: dict[str, Any],
    pos_rate: float,
    scoring_desc: str,
    model_info: Any,
    text_scope: str,
    elapsed_s: float,
) -> Path:
    run_id = str(args.run_id or time.strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    model_info_dict = asdict(model_info)
    config = {
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "args": vars(args),
        "model": model_info_dict,
        "scoring": scoring_desc,
        "text_scope": text_scope,
        "elapsed_s": float(elapsed_s),
    }
    _write_json(run_dir / "run_config.json", config)

    dev_set = set(dev_idx)
    eval_set = set(eval_idx if eval_idx else list(range(len(labels))))
    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    for i, (ex, label, text, score, pred, detail) in enumerate(zip(rows, labels, texts, scores, predictions, details, strict=True)):
        rid = _row_id(ex, i)
        source = ex.get("source") or ex.get("source_benchmark") or ex.get("dataset") or ""
        split = _artifact_split(i, dev_set, eval_set)
        detail_dict = asdict(detail)
        common = {
            "index": i,
            "id": rid,
            "source": str(source),
            "split": split,
            "label": int(label),
            "status": "success",
            "text_sha256": _text_hash(text),
            "text_len": len(text),
        }
        raw_rows.append({**common, "score": float(score), "score_detail": detail_dict})
        parsed_rows.append({**common, "score": float(score), "threshold": float(threshold), "direction": direction, "prediction": int(pred)})
        per_sample_rows.append({**common, "score": float(score), "prediction": int(pred), "correct": bool(int(pred) == int(label))})

    _write_jsonl(run_dir / "raw_scores.jsonl", raw_rows)
    _write_jsonl(run_dir / "parsed_outputs.jsonl", parsed_rows)
    _write_jsonl(run_dir / "per_sample_results.jsonl", per_sample_rows)

    metrics_payload = {
        "dataset_path": str(args.dataset_path),
        "n": len(labels),
        "pos_rate": float(pos_rate),
        "split_mode": split_mode,
        "dev_n": len(dev_idx),
        "eval_n": int(res_eval["n"]),
        "direction": direction,
        "threshold": float(threshold),
        "dev_acc": None if math.isnan(dev_acc) else float(dev_acc),
        "eval_confusion": res_eval["conf"],
        "eval_metrics": res_eval["metrics"],
        "status_counts": {"success": len(labels)},
        "artifact_files": {
            "run_config": "run_config.json",
            "raw_scores": "raw_scores.jsonl",
            "parsed_outputs": "parsed_outputs.jsonl",
            "per_sample_results": "per_sample_results.jsonl",
            "aggregate_metrics": "aggregate_metrics.json",
        },
    }
    _write_json(run_dir / "aggregate_metrics.json", metrics_payload)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Token-level anomaly baselines for abnormal data detection")
    parser.add_argument(
        "--dataset-path",
        default="datasets/generated_datasets/v0.1.1/dataset_hf",
        help="Path to a HF dataset saved by Dataset.save_to_disk(), or a supported .jsonl file",
    )
    parser.add_argument(
        "--method",
        choices=["nll_spike", "onion_delta"],
        default="nll_spike",
        help="Token-level score: local next-token NLL spike, or ONION-style deletion log-PPL delta.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B", help="Causal LM used to score token likelihoods")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers")
    parser.add_argument(
        "--text-scope",
        choices=["user", "assistant", "full"],
        default="user",
        help="Which text to score: user input only, assistant output only, or full user+assistant dialogue.",
    )
    parser.add_argument(
        "--include-assistant",
        action="store_true",
        help="Backward-compatible alias for --text-scope full when --text-scope is left as user.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Dev split ratio for threshold tuning")
    parser.add_argument(
        "--direction",
        choices=["auto", "high", "low"],
        default="auto",
        help="How to map score to abnormal(1): high => score>=thr, low => score<thr",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold (skip tuning if set)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--deletion-batch-size",
        type=int,
        default=16,
        help="Batch size for ONION deletion candidates. Only used by --method onion_delta.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto", help="torch device: auto | cpu | cuda | cuda:0 ...")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N examples (0 means all)")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--nll-score-mode",
        choices=["token_count", "max_z", "max_z_len_norm", "max_nll", "mean_nll", "std_nll", "topk_mean_nll", "topk_gap_nll"],
        default="max_z",
        help="How to aggregate per-token NLL spikes. Only used by --method nll_spike.",
    )
    parser.add_argument(
        "--onion-score-mode",
        choices=["max_delta", "topk_mean_delta"],
        default="max_delta",
        help="How to aggregate token deletion deltas. Only used by --method onion_delta.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k used by topk_mean_nll/topk_mean_delta.",
    )
    parser.add_argument(
        "--onion-candidates",
        type=int,
        default=32,
        help="Only delete top-N high-NLL candidate tokens for ONION scoring; 0 means all non-special tokens.",
    )
    parser.add_argument(
        "--onion-delta-space",
        choices=["log_ppl", "ppl"],
        default="log_ppl",
        help="Use stable mean-NLL delta (log_ppl) or literal perplexity delta (ppl).",
    )
    parser.add_argument(
        "--onion-max-pending-deletions",
        type=int,
        default=256,
        help="Maximum generated deletion sequences held before scoring/flushing. Only used by --method onion_delta.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional directory for reproducible artifacts: config, raw scores, parsed outputs, per-sample results, metrics.",
    )
    parser.add_argument("--run-id", default="", help="Optional artifact subdirectory name under --output-dir.")
    args = parser.parse_args()
    start_s = time.perf_counter()

    rows = load_rows(str(args.dataset_path), limit=int(args.limit))
    text_scope = str(args.text_scope)
    if bool(args.include_assistant) and text_scope == "user":
        text_scope = "full"
    labels, texts = _labels_texts_for_scope(rows, text_scope=text_scope)

    torch, device, tokenizer, model, model_info = load_causal_lm(
        model_id=str(args.model_id),
        trust_remote_code=bool(args.trust_remote_code),
        device_str=str(args.device),
        dtype_str=str(args.dtype),
    )

    if args.method == "nll_spike":
        scores, score_details = score_texts_nll_spike_with_details(
            torch=torch,
            device=device,
            tokenizer=tokenizer,
            model=model,
            texts=texts,
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
            score_mode=str(args.nll_score_mode),
            top_k=int(args.top_k),
            eps=float(args.eps),
        )
        scoring_desc = f"method:nll_spike score_mode:{args.nll_score_mode} top_k:{args.top_k}"
    else:
        scores, score_details = score_texts_onion_delta_with_details(
            torch=torch,
            device=device,
            tokenizer=tokenizer,
            model=model,
            texts=texts,
            batch_size=int(args.batch_size),
            deletion_batch_size=int(args.deletion_batch_size),
            max_length=int(args.max_length),
            max_candidates=int(args.onion_candidates),
            score_mode=str(args.onion_score_mode),
            top_k=int(args.top_k),
            delta_space=str(args.onion_delta_space),
            eps=float(args.eps),
            max_pending_deletions=int(args.onion_max_pending_deletions),
        )
        scoring_desc = (
            f"method:onion_delta score_mode:{args.onion_score_mode} top_k:{args.top_k} "
            f"candidates:{args.onion_candidates} delta_space:{args.onion_delta_space} "
            f"max_pending_deletions:{args.onion_max_pending_deletions}"
        )

    dev_idx, eval_idx, split_mode = resolve_dev_eval_indices(
        rows,
        labels,
        dev_ratio=float(args.dev_ratio),
        seed=int(args.seed),
    )

    if args.threshold is None:
        dev_scores = [scores[i] for i in dev_idx] if dev_idx else scores
        dev_labels = [labels[i] for i in dev_idx] if dev_idx else labels
        pick = pick_threshold(dev_scores, dev_labels, direction=str(args.direction))
        thr = float(pick.threshold)
        dev_acc = float(pick.acc)
        direction = str(pick.direction)
    else:
        thr = float(args.threshold)
        dev_acc = float("nan")
        direction = str(args.direction)
        if direction == "auto":
            direction = "high"

    def eval_on(indices: list[int]) -> dict[str, Any]:
        if not indices:
            return {"n": 0}
        y_true = [labels[i] for i in indices]
        y_pred = predict_from_scores([scores[i] for i in indices], threshold=thr, direction=direction)
        conf = confusion(y_true, y_pred)
        m = metrics_from_conf(conf)
        return {"n": len(indices), "conf": conf, "metrics": m}

    res_eval = eval_on(eval_idx if eval_idx else list(range(len(labels))))
    all_predictions = predict_from_scores(scores, threshold=thr, direction=direction)

    pos_rate = sum(labels) / len(labels) if labels else float("nan")
    print(f"dataset_path={args.dataset_path}")
    print(
        "token_model="
        f"model_id={model_info.model_id} device={model_info.device} dtype={model_info.dtype} "
        f"pad_token_id={model_info.pad_token_id} max_length={args.max_length} "
        f"batch_size={args.batch_size} deletion_batch_size={args.deletion_batch_size} text_scope={text_scope}"
    )
    print(f"scoring={scoring_desc} direction:{direction} threshold:{thr} dev_acc:{dev_acc if not math.isnan(dev_acc) else 'NA'}")
    print(f"n={len(labels)} pos_rate={pos_rate:.4f} split_mode={split_mode}")
    print(f"eval_n={res_eval['n']} eval_acc={res_eval['metrics']['acc']:.4f}")
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
    if args.output_dir:
        run_dir = _save_artifacts(
            output_dir=str(args.output_dir),
            args=args,
            rows=rows,
            labels=labels,
            texts=texts,
            scores=scores,
            details=score_details,
            predictions=all_predictions,
            dev_idx=dev_idx,
            eval_idx=eval_idx,
            split_mode=split_mode,
            threshold=thr,
            direction=direction,
            dev_acc=dev_acc,
            res_eval=res_eval,
            pos_rate=pos_rate,
            scoring_desc=scoring_desc,
            model_info=model_info,
            text_scope=text_scope,
            elapsed_s=time.perf_counter() - start_s,
        )
        print(f"artifacts_dir={run_dir}")


if __name__ == "__main__":
    main()
