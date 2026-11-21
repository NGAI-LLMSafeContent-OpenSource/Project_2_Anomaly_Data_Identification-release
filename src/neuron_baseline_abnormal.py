#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from typing import Any

from filterguard.data import labels_texts, load_rows, resolve_dev_eval_indices
from filterguard.metric import confusion, metrics_from_conf, pick_threshold, predict_from_scores
from filterguard.neuron import ActivationExtractor, fit_neuron_params, load_tokenizer_model, scores_from_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuron-level (activation) baseline for abnormal data detection")
    parser.add_argument(
        "--dataset-path",
        default="datasets/generated_datasets/v0.1.1/dataset_hf",
        help="Path to a HF dataset saved by Dataset.save_to_disk(), or a supported .jsonl file",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B", help="Causal LM used to extract hidden activations")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers")
    parser.add_argument("--include-assistant", action="store_true", help="Extract text from full dialogue (user+assistant)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Dev split ratio for threshold tuning")
    parser.add_argument(
        "--method",
        choices=["activation_norm", "centroid_l2", "diag_mahalanobis", "persona_vector"],
        default="diag_mahalanobis",
        help="How to build an anomaly score from activations",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "last"],
        default="mean",
        help="How to pool token activations into a single vector",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Which transformer block output to use (negative indexes allowed)",
    )
    parser.add_argument("--feature-normalize", choices=["none", "l2"], default="none")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--direction",
        choices=["auto", "high", "low"],
        default="auto",
        help="How to map score to abnormal(1): high => score>=thr, low => score<thr",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold (skip tuning if set)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto", help="torch device: auto | cpu | cuda | cuda:0 ...")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument(
        "--backbone",
        choices=["base", "causal_lm"],
        default="base",
        help="Which HF model class to load. base avoids lm_head/logits allocation (recommended).",
    )
    parser.add_argument(
        "--stop-after-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the forward pass immediately after the target layer. Disable with --no-stop-after-layer.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N examples (0 means all)")
    args = parser.parse_args()

    rows = load_rows(str(args.dataset_path), limit=int(args.limit))
    labels, texts = labels_texts(rows, include_assistant=bool(args.include_assistant))

    torch, device, tokenizer, model, model_info = load_tokenizer_model(
        model_id=str(args.model_id),
        trust_remote_code=bool(args.trust_remote_code),
        device_str=str(args.device),
        dtype_str=str(args.dtype),
        backbone=str(args.backbone),
    )

    dev_idx, eval_idx, split_mode = resolve_dev_eval_indices(
        rows,
        labels,
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

    params = fit_neuron_params(
        features_dev,
        labels_dev,
        method=str(args.method),
        feature_normalize=str(args.feature_normalize),
        eps=float(args.eps),
    )
    scores_t = scores_from_params(features_all, params)
    scores = [float(x) for x in scores_t.tolist()]

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

    pos_rate = sum(labels) / len(labels) if labels else float("nan")
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
        f"scoring=method:{args.method} feature_normalize:{args.feature_normalize} "
        f"direction:{direction} threshold:{thr} dev_acc:{dev_acc if not math.isnan(dev_acc) else 'NA'}"
    )
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


if __name__ == "__main__":
    main()
