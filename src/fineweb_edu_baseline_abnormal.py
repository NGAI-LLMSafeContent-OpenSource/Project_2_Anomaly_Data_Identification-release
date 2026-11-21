#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from typing import Any

from filterguard.data import load_rows, labels_texts, resolve_dev_eval_indices
from filterguard.fineweb import fineweb_scores
from filterguard.metric import confusion, metrics_from_conf, pick_threshold, predict_from_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline: HuggingFaceFW/fineweb-edu-classifier for abnormal data detection"
    )
    parser.add_argument(
        "--dataset-path",
        default="datasets/generated_datasets/v0.1.1/dataset_hf",
        help="Path to a HF dataset saved by Dataset.save_to_disk(), or a supported .jsonl file",
    )
    parser.add_argument("--model-id", default="HuggingFaceFW/fineweb-edu-classifier")
    parser.add_argument("--include-assistant", action="store_true", help="Score on full dialogue (user+assistant)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Dev split ratio for threshold tuning")
    parser.add_argument(
        "--direction",
        choices=["auto", "high", "low"],
        default="auto",
        help="How to map score to abnormal(1): high => score>=thr, low => score<thr",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold (skip tuning if set)")
    parser.add_argument(
        "--score-mode",
        choices=["raw", "int0_5"],
        default="int0_5",
        help="Use raw regression logits, or clip/round to int score in [0..5] like the model card example",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device: auto | cpu | cuda | cuda:0 ... (auto picks cuda if available)",
    )
    args = parser.parse_args()

    rows = load_rows(str(args.dataset_path), limit=0)
    labels, texts = labels_texts(rows, include_assistant=bool(args.include_assistant))

    scores, info = fineweb_scores(
        texts,
        model_id=str(args.model_id),
        device_str=str(args.device),
        dtype_str="fp32",
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        score_mode=str(args.score_mode),
        trust_remote_code=False,
        warmup_batches=0,
        release_after=False,
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
            direction = "low"

    def eval_on(indices: list[int]) -> dict[str, Any]:
        if not indices:
            return {"n": 0}
        y_true = [labels[i] for i in indices]
        y_pred = predict_from_scores([scores[i] for i in indices], threshold=thr, direction=direction)
        conf = confusion(y_true, y_pred)
        m = metrics_from_conf(conf)
        return {"n": len(indices), "conf": conf, "metrics": m}

    res_eval = eval_on(eval_idx if eval_idx else list(range(len(labels))))

    pos_rate = sum(labels) / len(labels) if labels else float("nan")
    print(f"dataset_path={args.dataset_path}")
    print(f"model_id={info['model_id']} device={info['device']} score_mode={info['score_mode']}")
    print(f"n={len(labels)} pos_rate={pos_rate:.4f} split_mode={split_mode}")
    print(f"direction={direction} threshold={thr} dev_acc={dev_acc if not math.isnan(dev_acc) else 'NA'}")
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


if __name__ == "__main__":
    main()
