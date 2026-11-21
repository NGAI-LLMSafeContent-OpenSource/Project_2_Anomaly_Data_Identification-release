#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filterguard.data import labels_texts, load_rows
from filterguard.fineweb import fineweb_scores
from filterguard.keyword import compile_rules, keyword_score, load_rules_config, load_wordlist, pick_keyword_threshold
from filterguard.metric import pick_threshold, predict_from_scores, split_indices
from filterguard.neuron import ActivationExtractor, fit_neuron_params, load_tokenizer_model, scores_from_params
from filterguard.util import sync_if_cuda


@dataclass(frozen=True)
class Result:
    method: str
    n: int
    device: str
    batch_size: int | None
    max_length: int | None
    load_s: float
    infer_s: float
    total_s: float

    @property
    def sps_infer(self) -> float:
        return self.n / self.infer_s if self.infer_s > 0 else float("nan")

    @property
    def sps_total(self) -> float:
        return self.n / self.total_s if self.total_s > 0 else float("nan")


def _fmt_s(x: float) -> str:
    if x != x:  # nan
        return "NA"
    return f"{x:.3f}"


def _fmt_f(x: float) -> str:
    if x != x:
        return "NA"
    return f"{x:.1f}"


def _benchmark_keyword(
    *,
    tag: str,
    texts: list[str],
    labels: list[int],
    dev_ratio: float,
    seed: int,
    rules_config: str,
    extra_wordlists: list[str],
    extra_weight: int,
    extra_match: str,
) -> Result:
    load_start = time.perf_counter()
    base_rules = load_rules_config(Path(rules_config))
    extra_rules = []
    for p in extra_wordlists:
        extra_rules.extend(load_wordlist(Path(p), weight=int(extra_weight), match=str(extra_match).lower()))
    rules = base_rules + extra_rules
    compiled = compile_rules(rules)
    load_end = time.perf_counter()

    score_start = time.perf_counter()
    scores_int = [int(keyword_score(t, compiled)) for t in texts]
    score_end = time.perf_counter()

    dev_idx, _eval_idx = split_indices(labels, float(dev_ratio), int(seed))
    dev_scores = [scores_int[i] for i in dev_idx] if dev_idx else scores_int
    dev_labels = [labels[i] for i in dev_idx] if dev_idx else labels

    thr_start = time.perf_counter()
    thr, _dev_acc = pick_keyword_threshold(dev_scores, dev_labels)
    _ = [1 if s >= thr else 0 for s in scores_int]  # prediction only (discard)
    thr_end = time.perf_counter()

    load_s = float(load_end - load_start)
    infer_s = float((score_end - score_start) + (thr_end - thr_start))
    total_s = float(thr_end - load_start)

    return Result(
        method=f"{tag}(base={len(base_rules)} extra={len(extra_rules)} total={len(rules)})",
        n=len(texts),
        device="cpu",
        batch_size=None,
        max_length=None,
        load_s=load_s,
        infer_s=infer_s,
        total_s=total_s,
    )


def _benchmark_fineweb(
    *,
    texts: list[str],
    labels: list[int],
    dev_ratio: float,
    seed: int,
    model_id: str,
    device_str: str,
    batch_size: int,
    max_length: int,
    score_mode: str,
    warmup_batches: int,
) -> Result:
    scores, info = fineweb_scores(
        texts,
        model_id=str(model_id),
        device_str=str(device_str),
        dtype_str="auto",
        batch_size=int(batch_size),
        max_length=int(max_length),
        score_mode=str(score_mode),
        trust_remote_code=False,
        warmup_batches=int(warmup_batches),
        release_after=True,
    )

    dev_idx, _eval_idx = split_indices(labels, float(dev_ratio), int(seed))
    dev_scores = [scores[i] for i in dev_idx] if dev_idx else scores
    dev_labels = [labels[i] for i in dev_idx] if dev_idx else labels
    pick = pick_threshold(dev_scores, dev_labels, direction="auto")
    thr = float(pick.threshold)
    direction = str(pick.direction)
    _ = predict_from_scores(scores, threshold=float(thr), direction=str(direction))  # prediction only (discard)

    load_s = float(info["load_s"])
    infer_s = float(info["infer_s"])
    total_s = load_s + infer_s
    return Result(
        method="fineweb_edu",
        n=len(texts),
        device=str(info["device"]),
        batch_size=int(batch_size),
        max_length=int(max_length),
        load_s=load_s,
        infer_s=infer_s,
        total_s=total_s,
    )


def _extract_qwen_features(
    texts: list[str],
    *,
    model_id: str,
    trust_remote_code: bool,
    device_str: str,
    dtype_str: str,
    layer: int,
    pooling: str,
    batch_size: int,
    max_length: int,
    eps: float,
    warmup_batches: int,
) -> tuple[Any, Result]:
    load_start = time.perf_counter()
    torch, device, tokenizer, model, _info = load_tokenizer_model(
        model_id=str(model_id),
        trust_remote_code=bool(trust_remote_code),
        device_str=str(device_str),
        dtype_str=str(dtype_str),
        backbone="causal_lm",
    )
    sync_if_cuda(torch=torch, device=device)
    load_end = time.perf_counter()

    extractor = ActivationExtractor(
        torch=torch,
        device=device,
        tokenizer=tokenizer,
        model=model,
        layer=int(layer),
        pooling=str(pooling),
        eps=float(eps),
    )

    # Warmup on the first batch (exclude from timing).
    warm = texts[: int(batch_size)] if texts else []
    for _ in range(max(0, int(warmup_batches))):
        if not warm:
            break
        _ = extractor.extract(warm, batch_size=int(batch_size), max_length=int(max_length))
    sync_if_cuda(torch=torch, device=device)

    sync_if_cuda(torch=torch, device=device)
    infer_start = time.perf_counter()
    feats = extractor.extract(texts, batch_size=int(batch_size), max_length=int(max_length))
    sync_if_cuda(torch=torch, device=device)
    infer_end = time.perf_counter()

    extractor.close()

    if feats.shape[0] != len(texts):
        raise RuntimeError(f"features has shape {tuple(feats.shape)}, expected ({len(texts)}, d)")

    load_s = float(load_end - load_start)
    infer_s = float(infer_end - infer_start)
    total_s = float(infer_end - load_start)
    res = Result(
        method=f"qwen_features(layer={layer} resolved={extractor.layer_idx}/{extractor.n_layers} pooling={pooling})",
        n=len(texts),
        device=str(device),
        batch_size=int(batch_size),
        max_length=int(max_length),
        load_s=load_s,
        infer_s=infer_s,
        total_s=total_s,
    )
    return feats, res


def _benchmark_neuron_methods(
    *,
    texts: list[str],
    labels: list[int],
    dev_ratio: float,
    seed: int,
    qwen_model_id: str,
    qwen_trust_remote_code: bool,
    qwen_device: str,
    qwen_dtype: str,
    qwen_layer: int,
    qwen_pooling: str,
    qwen_batch_size: int,
    qwen_max_length: int,
    eps: float,
    warmup_batches: int,
    methods: list[tuple[str, str]],  # (method, feature_normalize)
) -> list[Result]:
    features, feat_res = _extract_qwen_features(
        texts,
        model_id=str(qwen_model_id),
        trust_remote_code=bool(qwen_trust_remote_code),
        device_str=str(qwen_device),
        dtype_str=str(qwen_dtype),
        layer=int(qwen_layer),
        pooling=str(qwen_pooling),
        batch_size=int(qwen_batch_size),
        max_length=int(qwen_max_length),
        eps=float(eps),
        warmup_batches=int(warmup_batches),
    )

    results: list[Result] = [feat_res]

    dev_idx, _eval_idx = split_indices(labels, float(dev_ratio), int(seed))
    if dev_idx:
        features_dev = features[dev_idx]
        labels_dev = [labels[i] for i in dev_idx]
    else:
        features_dev = features
        labels_dev = labels

    for method, feature_normalize in methods:
        score_start = time.perf_counter()
        params = fit_neuron_params(
            features_dev,
            labels_dev,
            method=str(method),
            feature_normalize=str(feature_normalize),
            eps=float(eps),
        )
        scores = [float(x) for x in scores_from_params(features, params).tolist()]

        dev_scores = [scores[i] for i in dev_idx] if dev_idx else scores
        dev_labels = [labels[i] for i in dev_idx] if dev_idx else labels
        pick = pick_threshold(dev_scores, dev_labels, direction="auto")
        _ = predict_from_scores(scores, threshold=float(pick.threshold), direction=str(pick.direction))  # discard
        score_end = time.perf_counter()

        infer_s = float(feat_res.infer_s) + float(score_end - score_start)
        total_s = float(feat_res.total_s) + float(score_end - score_start)
        results.append(
            Result(
                method=f"neuron:{method}({feature_normalize}) layer={qwen_layer} pooling={qwen_pooling}",
                n=len(texts),
                device=str(feat_res.device),
                batch_size=int(qwen_batch_size),
                max_length=int(qwen_max_length),
                load_s=float(feat_res.load_s),
                infer_s=infer_s,
                total_s=total_s,
            )
        )

    return results


def _print_table(results: list[Result]) -> None:
    headers = [
        "method",
        "n",
        "device",
        "batch",
        "max_len",
        "load_s",
        "infer_s",
        "sps(infer)",
        "sps(total)",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in results:
        print(
            "| "
            + " | ".join(
                [
                    r.method,
                    str(r.n),
                    r.device,
                    (str(r.batch_size) if r.batch_size is not None else "-"),
                    (str(r.max_length) if r.max_length is not None else "-"),
                    _fmt_s(r.load_s),
                    _fmt_s(r.infer_s),
                    _fmt_f(r.sps_infer),
                    _fmt_f(r.sps_total),
                ]
            )
            + " |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark throughput (samples/s) for FilterGuard baselines")
    parser.add_argument(
        "--dataset-path",
        default=str(Path("datasets/generated_datasets/v0.1.1/dataset_hf")),
        help="Path to a HF dataset saved by Dataset.save_to_disk(), or a supported .jsonl file",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only benchmark first N examples (0 means all)")
    parser.add_argument("--include-assistant", action="store_true", help="Use full dialogue (user+assistant)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--warmup-batches", type=int, default=1, help="Extra warmup forward passes (model baselines)")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE=1 to prevent downloads",
    )

    # Keyword config
    parser.add_argument("--keyword-rules-config", default=str(Path("configs/keyword_rules_base.json")))
    parser.add_argument("--keyword-extra-wordlist", action="append", default=[])
    parser.add_argument("--keyword-extra-weight", type=int, default=1)
    parser.add_argument("--keyword-extra-match", choices=["auto", "word", "substring"], default="auto")

    # Fineweb config
    parser.add_argument("--fineweb-model-id", default="HuggingFaceFW/fineweb-edu-classifier")
    parser.add_argument("--fineweb-device", default="auto")
    parser.add_argument("--fineweb-batch-size", type=int, default=32)
    parser.add_argument("--fineweb-max-length", type=int, default=512)
    parser.add_argument("--fineweb-score-mode", choices=["raw", "int0_5"], default="int0_5")

    # Qwen config
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--qwen-trust-remote-code", action="store_true")
    parser.add_argument("--qwen-device", default="auto")
    parser.add_argument("--qwen-dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--qwen-layer", type=int, default=-11)
    parser.add_argument("--qwen-pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--qwen-batch-size", type=int, default=8)
    parser.add_argument("--qwen-max-length", type=int, default=256)
    parser.add_argument("--eps", type=float, default=1e-6)

    args = parser.parse_args()

    if bool(args.offline):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    rows = load_rows(str(args.dataset_path), limit=int(args.limit))
    labels, texts = labels_texts(rows, include_assistant=bool(args.include_assistant))

    results: list[Result] = []
    results.append(
        _benchmark_keyword(
            tag="keyword",
            texts=texts,
            labels=labels,
            dev_ratio=float(args.dev_ratio),
            seed=int(args.seed),
            rules_config=str(args.keyword_rules_config),
            extra_wordlists=[],
            extra_weight=int(args.keyword_extra_weight),
            extra_match=str(args.keyword_extra_match),
        )
    )
    if args.keyword_extra_wordlist:
        results.append(
            _benchmark_keyword(
                tag="keyword+extra",
                texts=texts,
                labels=labels,
                dev_ratio=float(args.dev_ratio),
                seed=int(args.seed),
                rules_config=str(args.keyword_rules_config),
                extra_wordlists=[str(x) for x in args.keyword_extra_wordlist],
                extra_weight=int(args.keyword_extra_weight),
                extra_match=str(args.keyword_extra_match),
            )
        )

    results.append(
        _benchmark_fineweb(
            texts=texts,
            labels=labels,
            dev_ratio=float(args.dev_ratio),
            seed=int(args.seed),
            model_id=str(args.fineweb_model_id),
            device_str=str(args.fineweb_device),
            batch_size=int(args.fineweb_batch_size),
            max_length=int(args.fineweb_max_length),
            score_mode=str(args.fineweb_score_mode),
            warmup_batches=int(args.warmup_batches),
        )
    )

    results.extend(
        _benchmark_neuron_methods(
            texts=texts,
            labels=labels,
            dev_ratio=float(args.dev_ratio),
            seed=int(args.seed),
            qwen_model_id=str(args.qwen_model_id),
            qwen_trust_remote_code=bool(args.qwen_trust_remote_code),
            qwen_device=str(args.qwen_device),
            qwen_dtype=str(args.qwen_dtype),
            qwen_layer=int(args.qwen_layer),
            qwen_pooling=str(args.qwen_pooling),
            qwen_batch_size=int(args.qwen_batch_size),
            qwen_max_length=int(args.qwen_max_length),
            eps=float(args.eps),
            warmup_batches=int(args.warmup_batches),
            methods=[
                ("diag_mahalanobis", "none"),
                ("persona_vector", "l2"),
            ],
        )
    )

    _print_table(results)


if __name__ == "__main__":
    main()
