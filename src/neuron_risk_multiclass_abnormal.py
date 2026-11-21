#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from filterguard.data import labels_texts, load_rows, resolve_dev_eval_indices
from filterguard.neuron import ActivationExtractor, load_tokenizer_model
from filterguard.risk import (
    apply_one_vs_rest_margins,
    abnormal_binary_metrics_from_risk,
    fit_one_vs_rest_thresholds,
    fit_one_vs_rest_risk_persona,
    multiclass_metrics,
    ordered_risk_classes,
    predict_risk_labels,
    risk_labels,
    risk_scores_from_params,
)


def _safe_id(row: dict[str, Any], index: int) -> str:
    value = row.get("id") or row.get("record_key") or row.get("source_row_id")
    return str(value) if value is not None else str(index)


def _write_artifacts(
    *,
    output_dir: str,
    run_id: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    classes: list[str],
    labels: list[str],
    predictions: list[str],
    scores: list[list[float]],
    raw_scores: list[list[float]],
    dev_idx: list[int],
    eval_idx: list[int],
    split_mode: str,
    model_info: dict[str, Any],
    layer_resolved: int,
    n_layers: int,
    class_score_mode: str,
    class_thresholds: dict[str, Any],
    dev_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    eval_abnormal: dict[str, Any],
) -> Path:
    out = Path(output_dir) / str(run_id)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_path": str(args.dataset_path),
        "risk_level": str(args.risk_level),
        "classes": classes,
        "split_mode": split_mode,
        "dev_n": len(dev_idx),
        "eval_n": len(eval_idx),
        "model_info": model_info,
        "layer": int(args.layer),
        "layer_resolved": int(layer_resolved),
        "n_layers": int(n_layers),
        "pooling": str(args.pooling),
        "feature_normalize": str(args.feature_normalize),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "backbone": str(args.backbone),
        "stop_after_layer": bool(args.stop_after_layer),
        "class_score_mode": str(class_score_mode),
        "class_thresholds": {
            cls: {
                "threshold": float(pick.threshold),
                "direction": str(pick.direction),
                "dev_acc": float(pick.acc),
            }
            for cls, pick in class_thresholds.items()
        },
    }
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    aggregate = {
        "dev_metrics": dev_metrics,
        "eval_metrics": eval_metrics,
        "eval_abnormal_binary": eval_abnormal,
    }
    (out / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    eval_set = set(eval_idx)
    dev_set = set(dev_idx)
    with (out / "per_sample_results.jsonl").open("w", encoding="utf-8") as f:
        for i, (row, label, pred, row_scores) in enumerate(zip(rows, labels, predictions, scores, strict=True)):
            split = "eval" if i in eval_set else ("dev" if i in dev_set else "other")
            rec = {
                "index": i,
                "id": _safe_id(row, i),
                "split": split,
                "label": label,
                "prediction": pred,
                "correct": bool(label == pred),
                "scores": {cls: float(row_scores[j]) for j, cls in enumerate(classes)},
                "raw_scores": {cls: float(raw_scores[i][j]) for j, cls in enumerate(classes)},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def _fmt(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "NA"
    return f"{x:.4f}"


def _print_class_metrics(prefix: str, metrics: dict[str, Any], classes: list[str]) -> None:
    print(f"{prefix}_n={metrics['n']} {prefix}_acc={_fmt(float(metrics['acc']))} {prefix}_macro_f1={_fmt(float(metrics['macro_f1']))}")
    for cls in classes:
        m = metrics["per_class"][cls]
        print(
            f"{prefix}_class={json.dumps(cls, ensure_ascii=False)} "
            f"support={m['support']} pred_count={m['pred_count']} "
            f"precision={_fmt(float(m['precision']))} recall={_fmt(float(m['recall']))} f1={_fmt(float(m['f1']))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuron-level unified risk classifier using one-vs-rest persona vectors")
    parser.add_argument(
        "--dataset-path",
        default="datasets/generated_datasets/v0.1.1/dataset_hf",
        help="Path to a HF dataset saved by Dataset.save_to_disk(), or a supported .jsonl file",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B", help="Model used to extract hidden activations")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers")
    parser.add_argument("--include-assistant", action="store_true", help="Extract text from full dialogue (user+assistant)")
    parser.add_argument("--risk-level", choices=["l1", "l2"], default="l1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Dev split ratio when dataset has no explicit split")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--layer", type=int, default=-1, help="Which transformer block output to use")
    parser.add_argument("--feature-normalize", choices=["none", "l2"], default="l2")
    parser.add_argument(
        "--class-score-mode",
        choices=["threshold_margin", "raw"],
        default="threshold_margin",
        help="How to compare per-class persona scores. threshold_margin centers each class by a dev one-vs-rest threshold.",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto", help="torch device: auto | cpu | cuda | cuda:0 ...")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--backbone", choices=["base", "causal_lm"], default="base")
    parser.add_argument(
        "--stop-after-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the forward pass immediately after the target layer. Disable with --no-stop-after-layer.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N examples (0 means all)")
    parser.add_argument("--output-dir", default="", help="Optional directory for run artifacts")
    parser.add_argument("--run-id", default="neuron_risk_multiclass", help="Artifact run id")
    args = parser.parse_args()

    rows = load_rows(str(args.dataset_path), limit=int(args.limit))
    _binary_labels, texts = labels_texts(rows, include_assistant=bool(args.include_assistant))
    labels = risk_labels(rows, risk_level=str(args.risk_level))
    classes = ordered_risk_classes(labels)

    torch, device, tokenizer, model, model_info = load_tokenizer_model(
        model_id=str(args.model_id),
        trust_remote_code=bool(args.trust_remote_code),
        device_str=str(args.device),
        dtype_str=str(args.dtype),
        backbone=str(args.backbone),
    )

    binary_for_split = [0 if label == "normal" else 1 for label in labels]
    dev_idx, eval_idx, split_mode = resolve_dev_eval_indices(
        rows,
        binary_for_split,
        dev_ratio=float(args.dev_ratio),
        seed=int(args.seed),
    )

    with ActivationExtractor(
        torch=torch,
        device=device,
        tokenizer=tokenizer,
        model=model,
        layer=int(args.layer),
        pooling=str(args.pooling),
        eps=float(args.eps),
        stop_after_layer=bool(args.stop_after_layer),
    ) as extractor:
        features_all = extractor.extract(texts, batch_size=int(args.batch_size), max_length=int(args.max_length))
        layer_resolved = extractor.layer_idx
        n_layers = extractor.n_layers

    if dev_idx:
        features_dev = features_all[dev_idx]
        labels_dev = [labels[i] for i in dev_idx]
    else:
        features_dev = features_all
        labels_dev = labels

    params = fit_one_vs_rest_risk_persona(
        features_dev,
        labels_dev,
        classes=classes,
        feature_normalize=str(args.feature_normalize),
        eps=float(args.eps),
    )
    raw_scores_t = risk_scores_from_params(features_all, params)
    raw_scores_dev = raw_scores_t[dev_idx] if dev_idx else raw_scores_t
    class_thresholds = fit_one_vs_rest_thresholds(
        raw_scores_dev,
        labels_dev,
        classes=classes,
        direction="auto",
    )
    if args.class_score_mode == "threshold_margin":
        scores_t = apply_one_vs_rest_margins(raw_scores_t, classes, class_thresholds)
    elif args.class_score_mode == "raw":
        scores_t = raw_scores_t
    else:
        raise ValueError(f"unsupported class_score_mode: {args.class_score_mode!r}")
    predictions = predict_risk_labels(scores_t, classes)
    scores = [[float(x) for x in row] for row in scores_t.tolist()]
    raw_scores = [[float(x) for x in row] for row in raw_scores_t.tolist()]

    def metrics_for(indices: list[int]) -> dict[str, Any]:
        y_true = [labels[i] for i in indices]
        y_pred = [predictions[i] for i in indices]
        return multiclass_metrics(y_true, y_pred, classes=classes)

    eval_indices = eval_idx if eval_idx else list(range(len(labels)))
    dev_metrics = metrics_for(dev_idx) if dev_idx else multiclass_metrics([], [], classes=classes)
    eval_metrics = metrics_for(eval_indices)
    eval_abnormal = abnormal_binary_metrics_from_risk(
        [labels[i] for i in eval_indices],
        [predictions[i] for i in eval_indices],
    )

    print(f"dataset_path={args.dataset_path}")
    print(
        "activation_extractor="
        f"model_id={model_info['model_id']} model_type={model_info.get('model_type', 'unknown')} "
        f"device={model_info['device']} dtype={model_info['dtype']} "
        f"attn_implementation={model_info.get('attn_implementation', 'unknown')} "
        f"layer={args.layer} (resolved {layer_resolved}/{n_layers}) "
        f"pooling={args.pooling} max_length={args.max_length} batch_size={args.batch_size} "
        f"backbone={args.backbone} stop_after_layer={int(bool(args.stop_after_layer))}"
    )
    print(
        f"scoring=method:one_vs_rest_persona_vector risk_level:{args.risk_level} "
        f"feature_normalize:{args.feature_normalize} class_score_mode:{args.class_score_mode} classes:{len(classes)}"
    )
    print(f"classes={json.dumps(classes, ensure_ascii=False)}")
    for cls in classes:
        pick = class_thresholds[cls]
        print(
            f"class_threshold={json.dumps(cls, ensure_ascii=False)} "
            f"direction={pick.direction} threshold={pick.threshold} dev_acc={pick.acc:.4f}"
        )
    print(f"n={len(labels)} split_mode={split_mode} dev_n={len(dev_idx)} eval_n={len(eval_indices)}")
    _print_class_metrics("dev", dev_metrics, classes)
    _print_class_metrics("eval", eval_metrics, classes)
    print(
        "eval_abnormal_binary="
        f"acc:{_fmt(float(eval_abnormal['metrics']['acc']))} "
        f"precision:{_fmt(float(eval_abnormal['metrics']['precision']))} "
        f"recall:{_fmt(float(eval_abnormal['metrics']['recall']))} "
        f"f1:{_fmt(float(eval_abnormal['metrics']['f1']))} "
        f"tp:{eval_abnormal['conf']['tp']} tn:{eval_abnormal['conf']['tn']} "
        f"fp:{eval_abnormal['conf']['fp']} fn:{eval_abnormal['conf']['fn']}"
    )

    if args.output_dir:
        run_dir = _write_artifacts(
            output_dir=str(args.output_dir),
            run_id=str(args.run_id),
            args=args,
            rows=rows,
            classes=classes,
            labels=labels,
            predictions=predictions,
            scores=scores,
            raw_scores=raw_scores,
            dev_idx=dev_idx,
            eval_idx=eval_indices,
            split_mode=split_mode,
            model_info=model_info,
            layer_resolved=layer_resolved,
            n_layers=n_layers,
            class_score_mode=str(args.class_score_mode),
            class_thresholds=class_thresholds,
            dev_metrics=dev_metrics,
            eval_metrics=eval_metrics,
            eval_abnormal=eval_abnormal,
        )
        print(f"artifacts={run_dir}")


if __name__ == "__main__":
    main()
