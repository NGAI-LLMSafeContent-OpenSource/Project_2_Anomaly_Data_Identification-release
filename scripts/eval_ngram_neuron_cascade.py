#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from filterguard.data import labels_texts, load_rows, resolve_dev_eval_indices
from filterguard.metric import confusion, metrics_from_conf, pick_threshold, predict_from_scores
from filterguard.neuron import ActivationExtractor, fit_neuron_params, load_tokenizer_model, scores_from_params

from mine_ngram_keyword_rules import (
    _char_ngrams,
    _eval_threshold,
    _hash_rule,
    _pick_threshold,
    _stratified_subset_split,
    _threshold_sweep,
    _wilson_lower,
    _write_csv,
    _write_json,
)


def _mine_rules(
    texts: list[str],
    labels: list[int],
    mining_idx: list[int],
    args: argparse.Namespace,
) -> tuple[list[str], list[dict[str, Any]], int]:
    from collections import Counter

    pos_counts: Counter[str] = Counter()
    neg_counts: Counter[str] = Counter()
    for k, i in enumerate(mining_idx, start=1):
        grams = _char_ngrams(
            texts[i],
            min_n=int(args.min_n),
            max_n=int(args.max_n),
            max_text_chars=int(args.max_text_chars),
        )
        (pos_counts if int(labels[i]) == 1 else neg_counts).update(grams)
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
    return [str(row["ngram"]) for row in selected], selected, len(candidates)


def _ngram_scores(texts: list[str], rules: list[str], args: argparse.Namespace) -> list[int]:
    scores: list[int] = []
    rule_set = set(rules)
    for k, text in enumerate(texts, start=1):
        grams = _char_ngrams(
            text,
            min_n=int(args.min_n),
            max_n=int(args.max_n),
            max_text_chars=int(args.max_text_chars),
        )
        scores.append(sum(1 for gram in grams if gram in rule_set))
        if k % 5000 == 0:
            print(f"ngram_scored={k}/{len(texts)}", flush=True)
    return scores


def _eval(labels: list[int], preds: list[int], indices: list[int]) -> dict[str, Any]:
    y_true = [int(labels[i]) for i in indices]
    y_pred = [int(preds[i]) for i in indices]
    conf = confusion(y_true, y_pred)
    return {"n": len(indices), "conf": conf, "metrics": metrics_from_conf(conf)}


def _summary_row(payload: dict[str, Any]) -> dict[str, Any]:
    m = payload["eval_metrics"]
    c = payload["eval_confusion"]
    return {
        "Dataset": "NGAI SeedBank",
        "Split": "eval",
        "Eval N": int(payload["eval_n"]),
        "Abnormal Support": int(c["tp"]) + int(c["fn"]),
        "Normal Support": int(c["tn"]) + int(c["fp"]),
        "Method": payload["method_name"],
        "Decision Source": "dev-mined ngram high-precision shortcut + neuron persona fallback",
        "Accuracy": f"{m['acc'] * 100:.1f}",
        "Precision": f"{m['precision'] * 100:.1f}",
        "Recall": f"{m['recall'] * 100:.1f}",
        "F1": f"{m['f1'] * 100:.1f}",
        "TNR": f"{m['tnr'] * 100:.1f}",
        "TP": int(c["tp"]),
        "TN": int(c["tn"]),
        "FP": int(c["fp"]),
        "FN": int(c["fn"]),
        "Notes": (
            f"ngram_threshold={payload['ngram']['threshold']}; "
            f"ngram_shortcut_rate={payload['cascade']['ngram_shortcut_rate']:.4f}; "
            f"neuron_used_rate={payload['cascade']['neuron_used_rate']:.4f}; "
            f"neuron_layer={payload['neuron']['layer']}; "
            f"neuron_threshold={payload['neuron']['threshold']:.6g}; "
            f"neuron_direction={payload['neuron']['direction']}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cascade dev-mined ngram keyword shortcut with neuron persona fallback.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--include-assistant", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--calibration-ratio", type=float, default=0.30)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--min-n", type=int, default=5)
    parser.add_argument("--max-n", type=int, default=12)
    parser.add_argument("--max-text-chars", type=int, default=4000)
    parser.add_argument("--min-rule-pos", type=int, default=5)
    parser.add_argument("--max-rule-neg", type=int, default=0)
    parser.add_argument("--min-rule-precision", type=float, default=0.95)
    parser.add_argument("--min-rule-wilson", type=float, default=0.55)
    parser.add_argument("--max-rules", type=int, default=5000)
    parser.add_argument("--ngram-threshold", type=int, default=None)
    parser.add_argument("--min-threshold-precision", type=float, default=0.95)
    parser.add_argument("--min-threshold-pred-pos", type=int, default=5)
    parser.add_argument("--max-threshold", type=int, default=10)

    parser.add_argument("--model-id", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--backbone", choices=["base", "causal_lm"], default="base")
    parser.add_argument("--method", choices=["activation_norm", "centroid_l2", "diag_mahalanobis", "persona_vector"], default="persona_vector")
    parser.add_argument("--feature-normalize", choices=["none", "l2"], default="l2")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--neuron-direction", choices=["auto", "high", "low"], default="auto")
    parser.add_argument("--neuron-threshold", type=float, default=None)
    args = parser.parse_args()

    start_s = time.perf_counter()
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(str(args.dataset_path), limit=int(args.limit))
    labels, texts = labels_texts(rows, include_assistant=bool(args.include_assistant))
    dev_idx, eval_idx, split_mode = resolve_dev_eval_indices(rows, labels, dev_ratio=float(args.dev_ratio), seed=int(args.seed))
    if not eval_idx:
        eval_idx = list(range(len(labels)))
    mining_idx, calibration_idx = _stratified_subset_split(
        dev_idx,
        labels,
        ratio=float(args.calibration_ratio),
        seed=int(args.seed) + 17,
    )

    rules, selected_rows, candidate_count = _mine_rules(texts, labels, mining_idx, args)
    rules_no_text = [{k: v for k, v in row.items() if k != "ngram"} for row in selected_rows]
    _write_csv(run_dir / "mined_rules_no_text.csv", rules_no_text)

    ngram_scores = _ngram_scores(texts, rules, args)
    calibration_sweep = _threshold_sweep(ngram_scores, labels, calibration_idx, max_threshold=int(args.max_threshold))
    eval_ngram_sweep = _threshold_sweep(ngram_scores, labels, eval_idx, max_threshold=int(args.max_threshold))
    if args.ngram_threshold is None:
        picked = _pick_threshold(
            calibration_sweep,
            min_precision=float(args.min_threshold_precision),
            min_pred_pos=int(args.min_threshold_pred_pos),
        )
        ngram_threshold = int(picked["threshold"])
        ngram_threshold_policy = "calibration precision>=min then max recall"
    else:
        picked = _pick_threshold(
            calibration_sweep,
            min_precision=float(args.min_threshold_precision),
            min_pred_pos=int(args.min_threshold_pred_pos),
        )
        ngram_threshold = int(args.ngram_threshold)
        ngram_threshold_policy = "fixed"
    _write_csv(run_dir / "calibration_ngram_threshold_sweep.csv", calibration_sweep)
    _write_csv(run_dir / "eval_ngram_threshold_sweep.csv", eval_ngram_sweep)

    ngram_shortcut = [int(score) >= ngram_threshold for score in ngram_scores]
    pass_dev_idx = [i for i in dev_idx if not ngram_shortcut[i]]
    pass_all_idx = [i for i in range(len(labels)) if not ngram_shortcut[i]]

    torch, device, tokenizer, model, model_info = load_tokenizer_model(
        model_id=str(args.model_id),
        trust_remote_code=bool(args.trust_remote_code),
        device_str=str(args.device),
        dtype_str=str(args.dtype),
        backbone=str(args.backbone),
    )

    with ActivationExtractor(
        torch=torch,
        device=device,
        tokenizer=tokenizer,
        model=model,
        layer=int(args.layer),
        pooling=str(args.pooling),
        eps=float(args.eps),
        stop_after_layer=True,
    ) as extractor:
        dev_features = extractor.extract([texts[i] for i in dev_idx], batch_size=int(args.batch_size), max_length=int(args.max_length))
        params = fit_neuron_params(
            dev_features,
            [labels[i] for i in dev_idx],
            method=str(args.method),
            feature_normalize=str(args.feature_normalize),
            eps=float(args.eps),
        )
        dev_scores_all = [float(x) for x in scores_from_params(dev_features, params).tolist()]
        dev_pos_by_index = {idx: pos for pos, idx in enumerate(dev_idx)}
        pass_dev_scores = [dev_scores_all[dev_pos_by_index[i]] for i in pass_dev_idx]
        pass_dev_labels = [labels[i] for i in pass_dev_idx]
        if args.neuron_threshold is None:
            pick_n = pick_threshold(pass_dev_scores, pass_dev_labels, direction=str(args.neuron_direction))
            neuron_threshold = float(pick_n.threshold)
            neuron_direction = str(pick_n.direction)
            neuron_dev_acc = float(pick_n.acc)
        else:
            neuron_threshold = float(args.neuron_threshold)
            neuron_direction = str(args.neuron_direction)
            if neuron_direction == "auto":
                neuron_direction = "high"
            neuron_dev_acc = float("nan")

        pass_features = extractor.extract([texts[i] for i in pass_all_idx], batch_size=int(args.batch_size), max_length=int(args.max_length))
        pass_scores = [float(x) for x in scores_from_params(pass_features, params).tolist()]
        pass_preds = predict_from_scores(pass_scores, threshold=neuron_threshold, direction=neuron_direction)
        layer_resolved = extractor.layer_idx
        n_layers = extractor.n_layers

    preds = [1 if flag else 0 for flag in ngram_shortcut]
    for idx, pred in zip(pass_all_idx, pass_preds, strict=True):
        preds[idx] = int(pred)

    dev_res = _eval(labels, preds, dev_idx)
    eval_res = _eval(labels, preds, eval_idx)
    ngram_eval_res = _eval_threshold(ngram_scores, labels, eval_idx, ngram_threshold)
    elapsed_s = time.perf_counter() - start_s

    payload = {
        "method_name": "N-gram keyword + Qwen3-4B binary persona cascade",
        "dataset_path": str(args.dataset_path),
        "n": len(labels),
        "dev_n": len(dev_idx),
        "mining_n": len(mining_idx),
        "calibration_n": len(calibration_idx),
        "eval_n": len(eval_idx),
        "split_mode": split_mode,
        "ngram": {
            "candidate_rules": candidate_count,
            "selected_rules": len(rules),
            "threshold": ngram_threshold,
            "threshold_policy": ngram_threshold_policy,
            "calibration_pick": picked,
            "standalone_eval_confusion": ngram_eval_res["conf"],
            "standalone_eval_metrics": ngram_eval_res["metrics"],
        },
        "neuron": {
            "model_id": str(args.model_id),
            "model_info": model_info,
            "method": str(args.method),
            "feature_normalize": str(args.feature_normalize),
            "pooling": str(args.pooling),
            "layer": int(args.layer),
            "layer_resolved": layer_resolved,
            "n_layers": n_layers,
            "threshold": neuron_threshold,
            "direction": neuron_direction,
            "dev_acc_on_pass_subset": neuron_dev_acc,
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
        },
        "cascade": {
            "ngram_shortcut_count": sum(1 for flag in ngram_shortcut if flag),
            "ngram_shortcut_rate": sum(1 for flag in ngram_shortcut if flag) / len(labels),
            "neuron_used_count": len(pass_all_idx),
            "neuron_used_rate": len(pass_all_idx) / len(labels),
            "eval_ngram_shortcut_count": sum(1 for i in eval_idx if ngram_shortcut[i]),
            "eval_neuron_used_count": sum(1 for i in eval_idx if not ngram_shortcut[i]),
        },
        "dev_confusion": dev_res["conf"],
        "dev_metrics": dev_res["metrics"],
        "eval_confusion": eval_res["conf"],
        "eval_metrics": eval_res["metrics"],
        "elapsed_s": elapsed_s,
        "privacy": "no source text and no raw ngram strings are written",
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(run_dir / "aggregate_metrics.json", payload)
    with (run_dir / "summary_row.csv").open("w", newline="", encoding="utf-8") as f:
        row = _summary_row(payload)
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"dataset_path={args.dataset_path}")
    print(f"n={len(labels)} split_mode={split_mode} dev_n={len(dev_idx)} eval_n={len(eval_idx)}")
    print(
        f"ngram_rules={len(rules)} ngram_threshold={ngram_threshold} "
        f"ngram_eval_precision={ngram_eval_res['metrics']['precision']:.4f} "
        f"ngram_eval_recall={ngram_eval_res['metrics']['recall']:.4f}"
    )
    print(
        f"cascade_shortcut_rate={payload['cascade']['ngram_shortcut_rate']:.4f} "
        f"neuron_used_rate={payload['cascade']['neuron_used_rate']:.4f}"
    )
    print(
        "neuron="
        f"model_id={args.model_id} layer={args.layer} resolved={layer_resolved}/{n_layers} "
        f"threshold={neuron_threshold:.6g} direction={neuron_direction}"
    )
    print(f"eval_n={eval_res['n']} eval_acc={eval_res['metrics']['acc']:.4f}")
    print(
        "eval_confusion="
        f"tp:{eval_res['conf']['tp']} tn:{eval_res['conf']['tn']} "
        f"fp:{eval_res['conf']['fp']} fn:{eval_res['conf']['fn']}"
    )
    print(
        "eval_metrics="
        f"precision:{eval_res['metrics']['precision']:.4f} "
        f"recall:{eval_res['metrics']['recall']:.4f} "
        f"f1:{eval_res['metrics']['f1']:.4f} "
        f"tnr:{eval_res['metrics']['tnr']:.4f}"
    )
    print(f"elapsed_s={elapsed_s:.2f}")
    print(f"artifacts_dir={run_dir}")


if __name__ == "__main__":
    main()
