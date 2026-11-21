#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any

from filterguard.data import labels_texts, load_rows, resolve_dev_eval_indices
from filterguard.metric import confusion, metrics_from_conf


_SPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text).lower()).strip()


def _char_ngrams(text: str, *, min_n: int, max_n: int, max_text_chars: int) -> set[str]:
    text = _normalize(text)
    if max_text_chars > 0:
        text = text[: int(max_text_chars)]
    out: set[str] = set()
    length = len(text)
    for n in range(int(min_n), int(max_n) + 1):
        if length < n:
            continue
        for i in range(0, length - n + 1):
            gram = text[i : i + n]
            if gram.strip():
                out.add(gram)
    return out


def _wilson_lower(pos: int, neg: int, *, z: float = 1.96) -> float:
    n = int(pos) + int(neg)
    if n <= 0:
        return 0.0
    phat = float(pos) / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return (centre - margin) / denom


def _hash_rule(rule: str) -> str:
    return hashlib.sha256(rule.encode("utf-8")).hexdigest()


def _score_text(text: str, rules: list[str], *, min_n: int, max_n: int, max_text_chars: int) -> int:
    grams = _char_ngrams(text, min_n=min_n, max_n=max_n, max_text_chars=max_text_chars)
    return sum(1 for rule in rules if rule in grams)


def _eval_threshold(scores: list[int], labels: list[int], indices: list[int], threshold: int) -> dict[str, Any]:
    y_true = [int(labels[i]) for i in indices]
    y_pred = [1 if int(scores[i]) >= int(threshold) else 0 for i in indices]
    conf = confusion(y_true, y_pred)
    return {"n": len(indices), "conf": conf, "metrics": metrics_from_conf(conf)}


def _threshold_sweep(
    scores: list[int],
    labels: list[int],
    indices: list[int],
    *,
    max_threshold: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in range(0, int(max_threshold) + 1):
        res = _eval_threshold(scores, labels, indices, threshold)
        conf = res["conf"]
        metrics = res["metrics"]
        rows.append(
            {
                "threshold": threshold,
                "n": res["n"],
                "tp": conf["tp"],
                "tn": conf["tn"],
                "fp": conf["fp"],
                "fn": conf["fn"],
                "pred_pos": conf["tp"] + conf["fp"],
                "pred_neg": conf["tn"] + conf["fn"],
                "acc": metrics["acc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tnr": metrics["tnr"],
            }
        )
    return rows


def _pick_threshold(
    sweep_rows: list[dict[str, Any]],
    *,
    min_precision: float,
    min_pred_pos: int,
) -> dict[str, Any]:
    candidates = [
        row
        for row in sweep_rows
        if float(row["precision"]) >= float(min_precision) and int(row["pred_pos"]) >= int(min_pred_pos)
    ]
    if candidates:
        return max(candidates, key=lambda row: (float(row["recall"]), int(row["pred_pos"]), -int(row["threshold"])))
    positive_rows = [row for row in sweep_rows if int(row["pred_pos"]) >= int(min_pred_pos)]
    if positive_rows:
        return max(positive_rows, key=lambda row: (float(row["precision"]), float(row["recall"]), int(row["pred_pos"])))
    return max(sweep_rows, key=lambda row: (float(row["precision"]), float(row["recall"]), int(row["pred_pos"])))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _stratified_subset_split(indices: list[int], labels: list[int], *, ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if ratio <= 0.0 or not indices:
        return indices, indices
    rnd = Random(int(seed))
    pos = [i for i in indices if int(labels[i]) == 1]
    neg = [i for i in indices if int(labels[i]) == 0]
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    pos_second = max(1, int(len(pos) * float(ratio))) if pos else 0
    neg_second = max(1, int(len(neg) * float(ratio))) if neg else 0
    second = pos[:pos_second] + neg[:neg_second]
    first = pos[pos_second:] + neg[neg_second:]
    rnd.shuffle(first)
    rnd.shuffle(second)
    return first, second


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine high-precision character n-gram keyword triggers on the dev split.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--include-assistant", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument(
        "--calibration-ratio",
        type=float,
        default=0.0,
        help="If >0, split dev into mining and calibration subsets. Mine on the first subset and pick threshold on calibration.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-n", type=int, default=5)
    parser.add_argument("--max-n", type=int, default=12)
    parser.add_argument("--max-text-chars", type=int, default=4000)
    parser.add_argument("--min-rule-pos", type=int, default=5)
    parser.add_argument("--max-rule-neg", type=int, default=0)
    parser.add_argument("--min-rule-precision", type=float, default=0.95)
    parser.add_argument("--min-rule-wilson", type=float, default=0.55)
    parser.add_argument("--max-rules", type=int, default=5000)
    parser.add_argument("--min-threshold-precision", type=float, default=0.95)
    parser.add_argument("--min-threshold-pred-pos", type=int, default=5)
    parser.add_argument("--max-threshold", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=None, help="Fixed trigger threshold. Overrides dev threshold pick.")
    parser.add_argument("--write-raw-rules", action="store_true", help="Write raw mined ngrams. Disabled by default.")
    args = parser.parse_args()

    start = time.perf_counter()
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(str(args.dataset_path), limit=int(args.limit))
    labels, texts = labels_texts(rows, include_assistant=bool(args.include_assistant))
    dev_idx, eval_idx, split_mode = resolve_dev_eval_indices(
        rows,
        labels,
        dev_ratio=float(args.dev_ratio),
        seed=int(args.seed),
    )
    if not eval_idx:
        eval_idx = list(range(len(labels)))
    mining_idx, calibration_idx = _stratified_subset_split(
        dev_idx,
        labels,
        ratio=float(args.calibration_ratio),
        seed=int(args.seed) + 17,
    )

    pos_counts: Counter[str] = Counter()
    neg_counts: Counter[str] = Counter()
    for k, i in enumerate(mining_idx, start=1):
        grams = _char_ngrams(
            texts[i],
            min_n=int(args.min_n),
            max_n=int(args.max_n),
            max_text_chars=int(args.max_text_chars),
        )
        target = pos_counts if int(labels[i]) == 1 else neg_counts
        target.update(grams)
        if k % 500 == 0:
            print(f"mining_counted={k}/{len(mining_idx)} pos_vocab={len(pos_counts)} neg_vocab={len(neg_counts)}", flush=True)

    candidates: list[dict[str, Any]] = []
    for gram, pos in pos_counts.items():
        neg = int(neg_counts.get(gram, 0))
        total = int(pos) + neg
        precision = float(pos) / total if total else 0.0
        wilson = _wilson_lower(int(pos), neg)
        if (
            int(pos) >= int(args.min_rule_pos)
            and neg <= int(args.max_rule_neg)
            and precision >= float(args.min_rule_precision)
            and wilson >= float(args.min_rule_wilson)
        ):
            candidates.append(
                {
                    "ngram": gram,
                    "ngram_sha256": _hash_rule(gram),
                    "ngram_len": len(gram),
                    "dev_pos": int(pos),
                    "dev_neg": neg,
                    "dev_precision": precision,
                    "dev_wilson_lower": wilson,
                }
            )

    candidates.sort(
        key=lambda row: (
            float(row["dev_wilson_lower"]),
            float(row["dev_precision"]),
            int(row["dev_pos"]),
            -int(row["ngram_len"]),
        ),
        reverse=True,
    )
    selected = candidates[: int(args.max_rules)]
    selected_rules = [str(row["ngram"]) for row in selected]

    scores: list[int] = []
    for k, text in enumerate(texts, start=1):
        scores.append(
            _score_text(
                text,
                selected_rules,
                min_n=int(args.min_n),
                max_n=int(args.max_n),
                max_text_chars=int(args.max_text_chars),
            )
        )
        if k % 5000 == 0:
            print(f"scored={k}/{len(texts)}", flush=True)

    dev_sweep = _threshold_sweep(scores, labels, dev_idx, max_threshold=int(args.max_threshold))
    calibration_sweep = _threshold_sweep(scores, labels, calibration_idx, max_threshold=int(args.max_threshold))
    eval_sweep = _threshold_sweep(scores, labels, eval_idx, max_threshold=int(args.max_threshold))
    picked = _pick_threshold(
        calibration_sweep,
        min_precision=float(args.min_threshold_precision),
        min_pred_pos=int(args.min_threshold_pred_pos),
    )
    if args.threshold is None:
        threshold = int(picked["threshold"])
        threshold_policy = "dev precision>=min then max recall"
    else:
        threshold = int(args.threshold)
        threshold_policy = "fixed"
    dev_result = _eval_threshold(scores, labels, dev_idx, threshold)
    calibration_result = _eval_threshold(scores, labels, calibration_idx, threshold)
    eval_result = _eval_threshold(scores, labels, eval_idx, threshold)

    rules_no_text = [
        {k: v for k, v in row.items() if k != "ngram"}
        for row in selected
    ]
    _write_csv(run_dir / "mined_rules_no_text.csv", rules_no_text)
    if bool(args.write_raw_rules):
        _write_csv(run_dir / "mined_rules_raw.csv", selected)
    _write_csv(run_dir / "dev_threshold_sweep.csv", dev_sweep)
    _write_csv(run_dir / "calibration_threshold_sweep.csv", calibration_sweep)
    _write_csv(run_dir / "eval_threshold_sweep.csv", eval_sweep)

    payload = {
        "dataset_path": str(args.dataset_path),
        "n": len(labels),
        "dev_n": len(dev_idx),
        "mining_n": len(mining_idx),
        "calibration_n": len(calibration_idx),
        "eval_n": len(eval_idx),
        "split_mode": split_mode,
        "label_counts": {"normal": int(len(labels) - sum(labels)), "abnormal": int(sum(labels))},
        "config": {
            "min_n": int(args.min_n),
            "max_n": int(args.max_n),
            "max_text_chars": int(args.max_text_chars),
            "min_rule_pos": int(args.min_rule_pos),
            "max_rule_neg": int(args.max_rule_neg),
            "min_rule_precision": float(args.min_rule_precision),
            "min_rule_wilson": float(args.min_rule_wilson),
            "max_rules": int(args.max_rules),
            "calibration_ratio": float(args.calibration_ratio),
            "min_threshold_precision": float(args.min_threshold_precision),
            "min_threshold_pred_pos": int(args.min_threshold_pred_pos),
            "max_threshold": int(args.max_threshold),
            "write_raw_rules": bool(args.write_raw_rules),
        },
        "mined": {
            "candidate_rules": len(candidates),
            "selected_rules": len(selected_rules),
            "raw_rules_written": bool(args.write_raw_rules),
        },
        "threshold": {
            "threshold": threshold,
            "policy": threshold_policy,
            "dev_pick": picked,
        },
        "dev_confusion": dev_result["conf"],
        "dev_metrics": dev_result["metrics"],
        "calibration_confusion": calibration_result["conf"],
        "calibration_metrics": calibration_result["metrics"],
        "eval_confusion": eval_result["conf"],
        "eval_metrics": eval_result["metrics"],
        "elapsed_s": time.perf_counter() - start,
        "artifacts": {
            "aggregate_metrics": "aggregate_metrics.json",
            "mined_rules_no_text": "mined_rules_no_text.csv",
            "mined_rules_raw": "mined_rules_raw.csv" if bool(args.write_raw_rules) else None,
            "dev_threshold_sweep": "dev_threshold_sweep.csv",
            "calibration_threshold_sweep": "calibration_threshold_sweep.csv",
            "eval_threshold_sweep": "eval_threshold_sweep.csv",
        },
        "privacy": "raw samples are never written; mined ngram strings are not written unless --write-raw-rules is set",
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(run_dir / "aggregate_metrics.json", payload)

    print(f"dataset_path={args.dataset_path}")
    print(
        f"n={len(labels)} split_mode={split_mode} dev_n={len(dev_idx)} "
        f"mining_n={len(mining_idx)} calibration_n={len(calibration_idx)} eval_n={len(eval_idx)}"
    )
    print(f"candidate_rules={len(candidates)} selected_rules={len(selected_rules)} raw_rules_written={bool(args.write_raw_rules)}")
    print(
        f"threshold={threshold} "
        f"calibration_precision={calibration_result['metrics']['precision']:.4f} "
        f"calibration_recall={calibration_result['metrics']['recall']:.4f} "
        f"dev_precision={dev_result['metrics']['precision']:.4f} dev_recall={dev_result['metrics']['recall']:.4f}"
    )
    print(
        "eval_confusion="
        f"tp:{eval_result['conf']['tp']} tn:{eval_result['conf']['tn']} "
        f"fp:{eval_result['conf']['fp']} fn:{eval_result['conf']['fn']}"
    )
    print(
        "eval_metrics="
        f"acc:{eval_result['metrics']['acc']:.4f} "
        f"precision:{eval_result['metrics']['precision']:.4f} "
        f"recall:{eval_result['metrics']['recall']:.4f} "
        f"f1:{eval_result['metrics']['f1']:.4f} "
        f"tnr:{eval_result['metrics']['tnr']:.4f}"
    )
    print(f"artifacts_dir={run_dir}")


if __name__ == "__main__":
    main()
