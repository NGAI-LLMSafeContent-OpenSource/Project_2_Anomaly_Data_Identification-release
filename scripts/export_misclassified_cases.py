#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_error_rows(per_sample_path: Path, *, split: str) -> dict[int, dict[str, Any]]:
    errors: dict[int, dict[str, Any]] = {}
    total = 0
    considered = 0
    with per_sample_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{per_sample_path} line {line_no} must be a JSON object")
            row_split = str(row.get("split") or "")
            if split != "all" and row_split != split:
                continue
            considered += 1
            if bool(row.get("correct")):
                continue
            index = int(row["index"])
            errors[index] = row
    if not errors:
        raise ValueError(f"No misclassified rows found in split={split!r}; total={total}, considered={considered}")
    return errors


def _error_type(label: Any, prediction: Any) -> str:
    if prediction is None:
        return "parse_failed"
    label_i = int(label)
    pred_i = int(prediction)
    if label_i == 0 and pred_i == 1:
        return "false_positive"
    if label_i == 1 and pred_i == 0:
        return "false_negative"
    return "other_error"


def _binary_label_name(value: Any) -> str | None:
    if value is None:
        return None
    value_i = int(value)
    if value_i == 1:
        return "abnormal"
    if value_i == 0:
        return "normal"
    raise ValueError(f"Expected binary label 0/1, got {value!r}")


def export_errors(
    *,
    dataset_path: Path,
    per_sample_path: Path,
    output_path: Path,
    split: str,
) -> dict[str, Any]:
    errors = _load_error_rows(per_sample_path, split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    written = 0
    with dataset_path.open("r", encoding="utf-8") as data_f, output_path.open("w", encoding="utf-8") as out_f:
        for index, raw in enumerate(data_f):
            if index not in errors:
                continue
            raw_line = raw.strip()
            if not raw_line:
                raise ValueError(f"{dataset_path} has an empty line at selected index {index}")
            original = json.loads(raw_line)
            meta = errors[index]
            etype = _error_type(meta.get("label"), meta.get("prediction"))
            counts[etype] = counts.get(etype, 0) + 1
            payload = {
                "index": index,
                "split": meta.get("split"),
                "error_type": etype,
                "label": meta.get("label"),
                "true_label": meta.get("label"),
                "true_label_label": _binary_label_name(meta.get("label")),
                "prediction": meta.get("prediction"),
                "model_prediction": meta.get("prediction"),
                "model_prediction_label": _binary_label_name(meta.get("prediction")),
                "metric_prediction": meta.get("metric_prediction"),
                "metric_prediction_label": _binary_label_name(meta.get("metric_prediction")),
                "status": meta.get("status"),
                "source": meta.get("source"),
                "text_len": meta.get("text_len"),
                "text_sha256": meta.get("text_sha256"),
                "row": original,
            }
            out_f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1

    if written != len(errors):
        missing = sorted(set(errors) - set(range(index + 1 if "index" in locals() else 0)))
        raise RuntimeError(f"Wrote {written} rows but expected {len(errors)}. Missing indices: {missing[:10]}")

    summary = {
        "dataset_path": str(dataset_path),
        "per_sample_path": str(per_sample_path),
        "output_path": str(output_path),
        "split": split,
        "misclassified_count": written,
        "counts": counts,
        "privacy_note": "Output JSONL intentionally contains original raw rows for user-side inspection; this script does not print row contents.",
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export misclassified raw cases by joining per-sample predictions with a JSONL dataset.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--per-sample-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--split", default="eval", help="Split to export, or 'all'. Defaults to eval.")
    args = parser.parse_args()

    summary = export_errors(
        dataset_path=Path(args.dataset_path),
        per_sample_path=Path(args.per_sample_path),
        output_path=Path(args.output_path),
        split=str(args.split),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
