#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from eval_oss_safeguard_binary import (
    POLICY_SYSTEM_PROMPT,
    _ANALYSIS_CHANNEL_PREFILL,
    _parse_binary_output,
    _parse_failed_count,
    _per_sample_result_row,
    _summary_row,
    _write_json,
    _write_jsonl,
)
from filterguard.data import labels_texts, load_rows, resolve_dev_eval_indices
from filterguard.metric import confusion, metrics_from_conf


def _build_prompt(tokenizer: Any, text: str) -> str:
    messages = [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    if not prompt.endswith(_ANALYSIS_CHANNEL_PREFILL):
        prompt += _ANALYSIS_CHANNEL_PREFILL
    return prompt


def _prompt_token_len(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def _truncate_user_text_to_prompt_budget(tokenizer: Any, text: str, *, max_prompt_tokens: int) -> tuple[str, bool]:
    prompt = _build_prompt(tokenizer, text)
    if _prompt_token_len(tokenizer, prompt) <= max_prompt_tokens:
        return prompt, False

    empty_prompt = _build_prompt(tokenizer, "")
    if _prompt_token_len(tokenizer, empty_prompt) > max_prompt_tokens:
        raise ValueError(
            f"max_prompt_tokens={max_prompt_tokens} is too small for the policy prompt and chat template"
        )

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    lo = 0
    hi = len(text_ids)
    best = empty_prompt
    while lo <= hi:
        mid = (lo + hi) // 2
        truncated_text = tokenizer.decode(text_ids[:mid], skip_special_tokens=False)
        candidate = _build_prompt(tokenizer, truncated_text)
        if _prompt_token_len(tokenizer, candidate) <= max_prompt_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best, True


def _build_prompts(tokenizer: Any, texts: list[str], *, max_prompt_tokens: int) -> tuple[list[str], list[bool]]:
    prompts: list[str] = []
    truncated_flags: list[bool] = []
    for text in texts:
        prompt, was_truncated = _truncate_user_text_to_prompt_budget(
            tokenizer,
            text,
            max_prompt_tokens=max_prompt_tokens,
        )
        prompts.append(prompt)
        truncated_flags.append(was_truncated)
    return prompts, truncated_flags


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from vLLM API: {detail[:1000]}") from exc


def _score(labels: list[int], preds: list[int]) -> dict[str, Any]:
    conf = confusion(labels, preds)
    return {"conf": conf, "metrics": metrics_from_conf(conf)}


def _write_summary_csv(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "Dataset",
        "Split",
        "Eval N",
        "Abnormal Support",
        "Normal Support",
        "Method",
        "Decision Source",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "TNR",
        "TP",
        "TN",
        "FP",
        "FN",
        "Notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GPT-OSS Safeguard through a vLLM OpenAI-compatible server.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-id", required=True, help="Tokenizer/model path.")
    parser.add_argument("--served-model-name", default="gpt-oss-safeguard-20b")
    parser.add_argument("--api-base", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--include-assistant", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--stop-token",
        action="append",
        default=None,
        help="Stop string passed to vLLM. Defaults to <|return|>, the GPT-OSS response EOS.",
    )
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--status-every", type=int, default=5)
    parser.add_argument("--stream-model-output", action="store_true")
    parser.add_argument("--method-name", default="GPT-OSS-Safeguard-20B-vLLM-official-style")
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
    dev_set = set(dev_idx)
    eval_set = set(eval_indices)

    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    partial_path = run_dir / "partial_per_sample_results.jsonl"
    model_output_path = run_dir / "partial_model_outputs.jsonl"

    preds: list[int | None] = []
    statuses: list[str] = []
    per_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    finish_reason_counts: dict[str, int] = {}
    truncated_prompt_count = 0
    progress_t0 = time.perf_counter()
    completions_url = str(args.api_base).rstrip("/") + "/completions"
    stop_tokens = list(args.stop_token) if args.stop_token else ["<|return|>"]

    with partial_path.open("w", encoding="utf-8") as partial_f:
        model_output_f = model_output_path.open("w", encoding="utf-8") if bool(args.stream_model_output) else None
        try:
            for batch_no, start in enumerate(range(0, len(texts), int(args.batch_size)), start=1):
                end = min(start + int(args.batch_size), len(texts))
                batch_texts = texts[start:end]
                prompts, batch_truncated_flags = _build_prompts(
                    tokenizer,
                    batch_texts,
                    max_prompt_tokens=int(args.max_length),
                )
                truncated_prompt_count += sum(1 for value in batch_truncated_flags if value)
                payload = {
                    "model": str(args.served_model_name),
                    "prompt": prompts,
                    "max_tokens": int(args.max_new_tokens),
                    "temperature": 0,
                    "stop": stop_tokens,
                    "skip_special_tokens": False,
                }
                response = _post_json(completions_url, payload, timeout_s=float(args.request_timeout_s))
                choices = response.get("choices")
                if not isinstance(choices, list) or len(choices) != len(batch_texts):
                    raise RuntimeError(f"Expected {len(batch_texts)} choices, got {0 if choices is None else len(choices)}")
                choices_by_index = {int(choice.get("index", pos)): choice for pos, choice in enumerate(choices)}
                for local_offset in range(len(batch_texts)):
                    choice = choices_by_index[local_offset]
                    out = str(choice.get("text") or "")
                    finish_reason = str(choice.get("finish_reason") or "")
                    finish_reason_counts[finish_reason] = finish_reason_counts.get(finish_reason, 0) + 1
                    pred, status = _parse_binary_output(out)
                    preds.append(pred)
                    statuses.append(status)
                    status_counts[status] = status_counts.get(status, 0) + 1
                    i = start + local_offset
                    row_payload = _per_sample_result_row(
                        index=i,
                        row=rows[i],
                        label=int(labels[i]),
                        text=texts[i],
                        pred=pred,
                        status=status,
                        dev_set=dev_set,
                        eval_set=eval_set,
                    )
                    row_payload["finish_reason"] = finish_reason
                    row_payload["prompt_truncated"] = bool(batch_truncated_flags[local_offset])
                    per_rows.append(row_payload)
                    partial_f.write(json.dumps(row_payload, ensure_ascii=False, sort_keys=True) + "\n")
                    partial_f.flush()
                    if model_output_f is not None:
                        model_output_f.write(
                            json.dumps({**row_payload, "model_output": out}, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                        model_output_f.flush()
                if int(args.status_every) > 0 and (batch_no % int(args.status_every) == 0 or len(preds) == len(texts)):
                    elapsed = max(time.perf_counter() - progress_t0, 1e-6)
                    rate = len(preds) / elapsed
                    eta = (len(texts) - len(preds)) / max(rate, 1e-6)
                    print(
                        f"progress={len(preds)}/{len(texts)} batches={batch_no} "
                        f"rate={rate:.3f}_samples_s eta_s={eta:.1f} "
                        f"truncated_prompts={truncated_prompt_count} "
                        f"statuses={json.dumps(status_counts, sort_keys=True)} "
                        f"finish_reasons={json.dumps(finish_reason_counts, sort_keys=True)}",
                        flush=True,
                    )
        finally:
            if model_output_f is not None:
                model_output_f.close()

    if len(preds) != len(labels):
        raise RuntimeError(f"prediction count mismatch: preds={len(preds)} labels={len(labels)}")

    metric_preds = [0 if pred is None else int(pred) for pred in preds]
    eval_y_true = [int(labels[i]) for i in eval_indices]
    eval_y_pred = [metric_preds[i] for i in eval_indices]
    scored = _score(eval_y_true, eval_y_pred)

    eval_status_counts: dict[str, int] = {}
    for i in eval_indices:
        status = statuses[i]
        eval_status_counts[status] = eval_status_counts.get(status, 0) + 1

    metrics_payload = {
        "dataset_path": str(args.dataset_path),
        "n": len(labels),
        "pos_rate": sum(labels) / len(labels) if labels else float("nan"),
        "dev_n": len(dev_idx),
        "eval_n": len(eval_indices),
        "split_mode": split_mode,
        "backend": "vllm_openai_completions",
        "api_base": str(args.api_base),
        "model_id": str(args.model_id),
        "served_model_name": str(args.served_model_name),
        "generation": {
            "max_length": int(args.max_length),
            "max_new_tokens": int(args.max_new_tokens),
            "reasoning_effort": "low",
            "prefill_analysis_channel": True,
            "stop": stop_tokens,
            "temperature": 0,
            "truncated_prompt_count": truncated_prompt_count,
        },
        "status_counts": status_counts,
        "finish_reason_counts": finish_reason_counts,
        "eval_status_counts": eval_status_counts,
        "parse_failure_policy_for_metrics": "parse failures have null prediction and metric_prediction=0 only to keep aggregate confusion explicit",
        "eval_confusion": scored["conf"],
        "eval_metrics": scored["metrics"],
        "elapsed_s": time.perf_counter() - start_s,
        "artifact_files": {
            "run_config": "run_config.json",
            "partial_per_sample_results": "partial_per_sample_results.jsonl",
            "partial_model_outputs": "partial_model_outputs.jsonl" if bool(args.stream_model_output) else None,
            "per_sample_results": "per_sample_results.jsonl",
            "aggregate_metrics": "aggregate_metrics.json",
            "summary_row": "summary_row.csv",
        },
        "privacy": "no source text is written; model outputs are written only when --stream-model-output is set",
    }
    _write_json(run_dir / "run_config.json", {"command": [sys.executable, *sys.argv], "cwd": os.getcwd(), "python": sys.version, "platform": platform.platform(), "args": vars(args)})
    _write_jsonl(run_dir / "per_sample_results.jsonl", per_rows)
    _write_json(run_dir / "aggregate_metrics.json", metrics_payload)
    _write_summary_csv(run_dir / "summary_row.csv", _summary_row(metrics_payload, method=str(args.method_name)))

    parse_failed = _parse_failed_count(status_counts)
    print(f"dataset_path={args.dataset_path}")
    print(f"backend=vllm_openai_completions api_base={args.api_base} model={args.served_model_name}")
    print(f"n={len(labels)} split_mode={split_mode} dev_n={len(dev_idx)} eval_n={len(eval_indices)}")
    print(f"status_counts={json.dumps(status_counts, sort_keys=True)} parse_failed={parse_failed}")
    print(f"eval_acc={scored['metrics']['acc']:.4f}")
    print(
        "eval_confusion="
        f"tp:{scored['conf']['tp']} tn:{scored['conf']['tn']} fp:{scored['conf']['fp']} fn:{scored['conf']['fn']}"
    )
    print(
        "eval_metrics="
        f"precision:{scored['metrics']['precision']:.4f} "
        f"recall:{scored['metrics']['recall']:.4f} "
        f"f1:{scored['metrics']['f1']:.4f} "
        f"tnr:{scored['metrics']['tnr']:.4f}"
    )
    print(f"elapsed_s={metrics_payload['elapsed_s']:.2f}")
    print(f"artifacts_dir={run_dir}")


if __name__ == "__main__":
    main()
