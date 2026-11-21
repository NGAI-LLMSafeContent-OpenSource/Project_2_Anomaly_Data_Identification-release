#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from filterguard.data import labels_texts, load_rows, resolve_dev_eval_indices
from filterguard.keyword import (
    KeywordRule,
    compile_rules,
    keyword_score,
    load_rules_config,
    load_wordlist,
    load_wordlist_manifest,
)
from filterguard.metric import confusion, metrics_from_conf, pick_threshold, predict_from_scores
from filterguard.cascade import pick_kw_neuron_cascade
from filterguard.neuron import ActivationExtractor, fit_neuron_params, load_tokenizer_model, scores_from_params


def _metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return metrics_from_conf(confusion(y_true, y_pred))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid: keyword shortcut + neuron scoring for abnormal data detection")
    parser.add_argument(
        "--dataset-path",
        default="datasets/generated_datasets/v0.1.1/dataset_hf",
        help="Path to a HF dataset saved by Dataset.save_to_disk(), or a supported .jsonl file",
    )
    parser.add_argument("--include-assistant", action="store_true", help="Extract text from full dialogue (user+assistant)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Dev split ratio for tuning")
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N examples (0 means all)")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE=1 to prevent downloads",
    )

    # Keyword config
    parser.add_argument("--rules-config", default=str(Path("configs/keyword_rules_base.json")))
    parser.add_argument("--no-base-rules", action="store_true")
    parser.add_argument("--extra-wordlist", action="append", default=[])
    parser.add_argument("--extra-wordlist-manifest", action="append", default=[])
    parser.add_argument("--extra-match", choices=["auto", "word", "substring"], default="auto")
    parser.add_argument("--extra-weight", type=int, default=1)

    # Cascade tuning config
    parser.add_argument(
        "--kw-threshold",
        type=int,
        default=None,
        help="Fixed keyword shortcut threshold: kw_score>=thr => abnormal. If unset, auto-pick on dev.",
    )
    parser.add_argument(
        "--max-acc-drop",
        type=float,
        default=0.005,
        help="When auto-picking kw threshold, keep dev ACC within this drop vs neuron-only.",
    )
    parser.add_argument(
        "--max-f1-drop",
        type=float,
        default=0.01,
        help="When auto-picking kw threshold, keep dev F1 within this drop vs neuron-only.",
    )

    # Neuron config
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B", help="Backbone model used to extract activations")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--backbone",
        choices=["base", "causal_lm"],
        default="base",
        help="Extraction backbone: base=AutoModel(no lm_head, faster) | causal_lm=AutoModelForCausalLM",
    )
    parser.add_argument(
        "--method",
        choices=["activation_norm", "centroid_l2", "diag_mahalanobis", "persona_vector"],
        default="persona_vector",
    )
    parser.add_argument("--feature-normalize", choices=["none", "l2"], default="l2")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--layer", type=int, default=-11)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--neuron-direction",
        choices=["auto", "high", "low"],
        default="auto",
        help="Direction for neuron threshold picking (auto tries both, chooses best dev ACC).",
    )
    parser.add_argument(
        "--neuron-threshold",
        type=float,
        default=None,
        help="Fixed neuron threshold (skip dev tuning if set). Direction then defaults to --neuron-direction.",
    )
    parser.add_argument("--timing", action="store_true", help="Print rough timing + throughput stats")
    args = parser.parse_args()

    if bool(args.offline):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    rows = load_rows(str(args.dataset_path), limit=int(args.limit))

    labels, texts = labels_texts(rows, include_assistant=bool(args.include_assistant))

    # Load keyword rules
    base_rules: list[KeywordRule] = []
    if not bool(args.no_base_rules):
        base_rules = load_rules_config(Path(args.rules_config))
    extra_rules: list[KeywordRule] = []
    for p in args.extra_wordlist:
        extra_rules.extend(
            load_wordlist(Path(p), weight=int(args.extra_weight), match=str(args.extra_match).lower())
        )
    for p in args.extra_wordlist_manifest:
        extra_rules.extend(load_wordlist_manifest(Path(p)))
    kw_rules = compile_rules(base_rules + extra_rules)

    t_kw0 = time.perf_counter()
    kw_scores_all = [keyword_score(t, kw_rules) for t in texts]
    t_kw1 = time.perf_counter()

    dev_idx, eval_idx, split_mode = resolve_dev_eval_indices(
        rows,
        labels,
        dev_ratio=float(args.dev_ratio),
        seed=int(args.seed),
    )
    dev_texts = [texts[i] for i in dev_idx] if dev_idx else texts
    dev_labels = [labels[i] for i in dev_idx] if dev_idx else labels
    dev_kw_scores = [kw_scores_all[i] for i in dev_idx] if dev_idx else kw_scores_all

    t_load0 = time.perf_counter()
    try:
        torch, device, tokenizer, model, model_info = load_tokenizer_model(
            model_id=str(args.model_id),
            trust_remote_code=bool(args.trust_remote_code),
            device_str=str(args.device),
            dtype_str=str(args.dtype),
            backbone=str(args.backbone),
        )
    except Exception as e:  # noqa: BLE001
        print(
            "Missing runtime deps or failed to load model. Install: pip install -U transformers torch",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    extractor = ActivationExtractor(
        torch=torch,
        device=device,
        tokenizer=tokenizer,
        model=model,
        layer=int(args.layer),
        pooling=str(args.pooling),
        eps=float(args.eps),
        stop_after_layer=True,
    )
    t_dev0 = time.perf_counter()
    features_dev = extractor.extract(dev_texts, batch_size=int(args.batch_size), max_length=int(args.max_length))
    t_dev1 = time.perf_counter()

    params = fit_neuron_params(
        features_dev,
        dev_labels,
        method=str(args.method),
        feature_normalize=str(args.feature_normalize),
        eps=float(args.eps),
    )
    neuron_scores_dev = [float(x) for x in scores_from_params(features_dev, params).tolist()]

    if args.kw_threshold is None:
        pick = pick_kw_neuron_cascade(
            kw_scores_all=kw_scores_all,
            neuron_scores_dev=neuron_scores_dev,
            kw_scores_dev=dev_kw_scores,
            labels_dev=dev_labels,
            max_acc_drop=float(args.max_acc_drop),
            max_f1_drop=float(args.max_f1_drop),
            neuron_direction=str(args.neuron_direction),
            fixed_neuron_threshold=(float(args.neuron_threshold) if args.neuron_threshold is not None else None),
            fixed_neuron_direction=str(args.neuron_direction),
        )
        kw_thr = int(pick.kw_threshold)
        neuron_thr = float(pick.neuron_threshold)
        neuron_dir = str(pick.neuron_direction)
        tuned_dev_acc = float(pick.dev_acc)
        tuned_dev_f1 = float(pick.dev_f1)
    else:
        kw_thr = int(args.kw_threshold)
        pass_mask = [s < kw_thr for s in dev_kw_scores]
        pass_scores = [s for s, keep in zip(neuron_scores_dev, pass_mask, strict=True) if keep]
        pass_labels = [y for y, keep in zip(dev_labels, pass_mask, strict=True) if keep]
        if args.neuron_threshold is None:
            if pass_scores:
                pick_n = pick_threshold(pass_scores, pass_labels, direction=str(args.neuron_direction))
                neuron_thr = float(pick_n.threshold)
                neuron_dir = str(pick_n.direction)
            else:
                neuron_thr = 0.0
                neuron_dir = "high"
        else:
            neuron_thr = float(args.neuron_threshold)
            neuron_dir = str(args.neuron_direction)
            if neuron_dir == "auto":
                neuron_dir = "high"

        dev_pred: list[int] = []
        for s_kw, s_nn in zip(dev_kw_scores, neuron_scores_dev, strict=True):
            if s_kw >= kw_thr:
                dev_pred.append(1)
            else:
                dev_pred.append(1 if ((s_nn >= neuron_thr) if neuron_dir == "high" else (s_nn < neuron_thr)) else 0)
        m = _metrics(dev_labels, dev_pred)
        tuned_dev_acc = float(m["acc"])
        tuned_dev_f1 = float(m["f1"])

    t_load1 = time.perf_counter()

    pass_all_idx = [i for i, s in enumerate(kw_scores_all) if s < kw_thr]
    pass_all_texts = [texts[i] for i in pass_all_idx]

    t_infer0 = time.perf_counter()
    preds_pass: list[int] = []
    if pass_all_texts:
        features_pass = extractor.extract(
            pass_all_texts,
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
        )
        scores_pass = [float(x) for x in scores_from_params(features_pass, params).tolist()]
        preds_pass = predict_from_scores(scores_pass, threshold=float(neuron_thr), direction=str(neuron_dir))
    t_infer1 = time.perf_counter()

    extractor.close()

    y_pred = [1 if s >= kw_thr else 0 for s in kw_scores_all]
    for i, p in zip(pass_all_idx, preds_pass, strict=True):
        y_pred[i] = int(p)

    def eval_on(indices: list[int]) -> dict[str, Any]:
        if not indices:
            return {"n": 0}
        y_true = [labels[i] for i in indices]
        y_hat = [y_pred[i] for i in indices]
        conf = confusion(y_true, y_hat)
        m = metrics_from_conf(conf)
        return {"n": len(indices), "conf": conf, "metrics": m}

    res_eval = eval_on(eval_idx if eval_idx else list(range(len(labels))))

    n = len(labels)
    pos_rate = sum(labels) / n if n else float("nan")
    shortcut_n = n - len(pass_all_idx)
    shortcut_rate = (shortcut_n / n) if n else float("nan")

    print(f"dataset_path={args.dataset_path}")
    print(
        "keyword="
        f"rules_config={args.rules_config} base_rules={len(base_rules)} extra_rules={len(extra_rules)} "
        f"kw_threshold={kw_thr} shortcut_rate={shortcut_rate:.4f}"
    )
    print(
        "activation_extractor="
        f"model_id={model_info['model_id']} backbone={model_info['backbone']} device={model_info['device']} "
        f"dtype={model_info['dtype']} layer={args.layer} (resolved {extractor.layer_idx}/{extractor.n_layers}) "
        f"pooling={args.pooling} max_length={args.max_length} batch_size={args.batch_size}"
    )
    print(
        "scoring="
        f"method:{args.method} feature_normalize:{args.feature_normalize} "
        f"neuron_direction:{neuron_dir} neuron_threshold:{neuron_thr:.6g} "
        f"cascade_dev_acc:{tuned_dev_acc:.4f} cascade_dev_f1:{tuned_dev_f1:.4f}"
    )
    print(
        f"n={n} pos_rate={pos_rate:.4f} split_mode={split_mode} "
        f"neuron_used_rate={len(pass_all_idx)/n if n else float('nan'):.4f}"
    )
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

    if bool(args.timing):
        kw_s = float(t_kw1 - t_kw0)
        dev_extract_s = float(t_dev1 - t_dev0)
        load_fit_s = float(t_load1 - t_load0)
        infer_nn_s = float(t_infer1 - t_infer0)
        infer_s = kw_s + infer_nn_s
        total_s = float(t_infer1 - t_kw0)
        sps_infer = (n / infer_s) if infer_s > 0 else float("nan")
        sps_total = (n / total_s) if total_s > 0 else float("nan")
        print(
            "timing="
            f"kw_s={kw_s:.3f} dev_extract_s={dev_extract_s:.3f} "
            f"load_fit_s={load_fit_s:.3f} infer_nn_s={infer_nn_s:.3f} "
            f"infer_s={infer_s:.3f} total_s={total_s:.3f} "
            f"sps_infer={sps_infer:.1f} sps_total={sps_total:.1f}"
        )


if __name__ == "__main__":
    main()
