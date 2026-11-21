#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from filterguard.metric import confusion, metrics_from_conf


def _read_rows(path: Path, *, split: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_no} must be a JSON object")
            row_split = str(row.get("split") or "")
            if split != "all" and row_split != split:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found for split={split!r} in {path}")
    return rows


def _prediction_for_metrics(row: dict[str, Any], *, parse_failed_as: int) -> int:
    pred = row.get("prediction")
    if pred is None:
        return int(parse_failed_as)
    return int(pred)


def score_rows(rows: list[dict[str, Any]], *, parse_failed_as: int) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []
    status_counts: dict[str, int] = {}
    parse_failed = 0
    prediction_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}

    for row in rows:
        label = int(row["label"])
        pred = _prediction_for_metrics(row, parse_failed_as=parse_failed_as)
        status = str(row.get("status") or "")
        if status.startswith("parse_failed") or row.get("prediction") is None:
            parse_failed += 1
        y_true.append(label)
        y_pred.append(pred)
        status_counts[status] = status_counts.get(status, 0) + 1
        prediction_counts[str(pred)] = prediction_counts.get(str(pred), 0) + 1
        label_counts[str(label)] = label_counts.get(str(label), 0) + 1

    conf = confusion(y_true, y_pred)
    metrics = metrics_from_conf(conf)
    return {
        "n": len(rows),
        "parse_failed": parse_failed,
        "parse_failed_as": int(parse_failed_as),
        "status_counts": status_counts,
        "label_counts": label_counts,
        "prediction_counts": prediction_counts,
        "confusion": conf,
        "metrics": metrics,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, payload: dict[str, Any], *, dataset: str, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = payload["metrics"]
    conf = payload["confusion"]
    row = {
        "Dataset": dataset,
        "Split": split,
        "Eval N": payload["n"],
        "Accuracy": f"{metrics['acc'] * 100:.1f}",
        "Precision": f"{metrics['precision'] * 100:.1f}",
        "Recall": f"{metrics['recall'] * 100:.1f}",
        "F1": f"{metrics['f1'] * 100:.1f}",
        "TNR": f"{metrics['tnr'] * 100:.1f}",
        "TP": conf["tp"],
        "TN": conf["tn"],
        "FP": conf["fp"],
        "FN": conf["fn"],
        "Parse Failed": payload["parse_failed"],
        "Parse Failed As": payload["parse_failed_as"],
    }
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score a completed or partial per-sample prediction JSONL. "
            "The input must contain label, prediction, split, and status fields; no raw text is read."
        )
    )
    parser.add_argument("--per-sample-path", required=True, help="Path to per_sample_results.jsonl or partial_per_sample_results.jsonl")
    parser.add_argument("--split", default="eval", help="Split to score: eval, dev, all, etc. Defaults to eval.")
    parser.add_argument(
        "--parse-failed-as",
        type=int,
        choices=[0, 1],
        default=0,
        help="Metric prediction used when prediction is null. Defaults to 0 to match OSS eval policy.",
    )
    parser.add_argument("--dataset-name", default="NGAI SeedBank")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    rows = _read_rows(Path(args.per_sample_path), split=str(args.split))
    payload = score_rows(rows, parse_failed_as=int(args.parse_failed_as))
    payload["per_sample_path"] = str(args.per_sample_path)
    payload["split"] = str(args.split)

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.output_json:
        _write_json(Path(args.output_json), payload)
    if args.output_csv:
        _write_csv(Path(args.output_csv), payload, dataset=str(args.dataset_name), split=str(args.split))


if __name__ == "__main__":
    main()
