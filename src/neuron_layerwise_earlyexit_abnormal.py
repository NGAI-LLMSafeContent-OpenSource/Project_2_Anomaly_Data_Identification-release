#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any

from filterguard.data import labels_texts, load_rows, resolve_dev_eval_indices
from filterguard.metric import confusion, metrics_from_conf, pick_threshold, predict_from_scores
from filterguard.neuron import (
    ActivationExtractor,
    MultiLayerActivationExtractor,
    get_transformer_layers,
    load_tokenizer_model,
    move_batch_to_device,
    pool_hidden,
)
from filterguard.util import sync_if_cuda


def _empty_cache_if_cuda(*, torch: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()


def _metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return metrics_from_conf(confusion(y_true, y_pred))


def _resolve_layer(layer: int, *, n_layers: int) -> int:
    idx = int(layer)
    if idx < 0:
        idx = int(n_layers) + idx
    if not 0 <= idx < int(n_layers):
        raise ValueError(f"layer out of range: {layer} (resolved {idx}) with n_layers={n_layers}")
    return int(idx)


def _l2_normalize(x: Any, *, eps: float) -> Any:
    import torch

    denom = torch.norm(x, dim=1, keepdim=True).clamp_min(float(eps))
    return x / denom


def _fit_persona_vectors(
    *,
    torch: Any,
    extractor: MultiLayerActivationExtractor,
    texts: list[str],
    labels: list[int],
    feature_normalize: str,
    eps: float,
    batch_size: int,
    max_length: int,
) -> dict[int, Any]:
    if len(texts) != len(labels):
        raise ValueError("texts/labels length mismatch")
    sums_pos: dict[int, Any] = {}
    sums_neg: dict[int, Any] = {}
    cnt_pos = 0
    cnt_neg = 0

    offset = 0
    for feats_by_layer in extractor.iter_extract(texts, batch_size=int(batch_size), max_length=int(max_length)):
        batch_n = next(iter(feats_by_layer.values())).shape[0] if feats_by_layer else 0
        y_batch = labels[offset : offset + batch_n]
        pos_idx = [i for i, y in enumerate(y_batch) if y == 1]
        neg_idx = [i for i, y in enumerate(y_batch) if y == 0]
        cnt_pos += len(pos_idx)
        cnt_neg += len(neg_idx)

        for layer_idx, feats in feats_by_layer.items():
            if feature_normalize == "l2":
                feats = _l2_normalize(feats, eps=float(eps))
            elif feature_normalize != "none":
                raise ValueError("--feature-normalize must be one of: none, l2")

            if layer_idx not in sums_pos:
                sums_pos[layer_idx] = torch.zeros(feats.shape[1], dtype=feats.dtype, device=feats.device)
                sums_neg[layer_idx] = torch.zeros(feats.shape[1], dtype=feats.dtype, device=feats.device)

            if pos_idx:
                sums_pos[layer_idx] += feats[pos_idx].sum(dim=0)
            if neg_idx:
                sums_neg[layer_idx] += feats[neg_idx].sum(dim=0)

        offset += int(batch_n)

    if offset != len(labels):
        raise RuntimeError(f"dev extraction mismatch: got {offset} feats for {len(labels)} labels")
    if cnt_pos <= 0 or cnt_neg <= 0:
        raise ValueError("persona_vector requires both positive and negative examples in dev split")

    v_by_layer: dict[int, Any] = {}
    for layer_idx in extractor.layer_idxs:
        mu_pos = sums_pos[layer_idx] / float(cnt_pos)
        mu_neg = sums_neg[layer_idx] / float(cnt_neg)
        v = mu_pos - mu_neg
        v = v / (torch.norm(v) + float(eps))
        v_by_layer[int(layer_idx)] = v
    return v_by_layer


def _fit_persona_vector_single(
    *,
    torch: Any,
    extractor: ActivationExtractor,
    texts: list[str],
    labels: list[int],
    feature_normalize: str,
    eps: float,
    batch_size: int,
    max_length: int,
) -> Any:
    if len(texts) != len(labels):
        raise ValueError("texts/labels length mismatch")
    sum_pos = None
    sum_neg = None
    cnt_pos = 0
    cnt_neg = 0

    offset = 0
    for feats in extractor.iter_extract(texts, batch_size=int(batch_size), max_length=int(max_length)):
        batch_n = int(feats.shape[0])
        y_batch = labels[offset : offset + batch_n]
        pos_idx = [i for i, y in enumerate(y_batch) if y == 1]
        neg_idx = [i for i, y in enumerate(y_batch) if y == 0]
        cnt_pos += len(pos_idx)
        cnt_neg += len(neg_idx)

        if feature_normalize == "l2":
            feats = _l2_normalize(feats, eps=float(eps))
        elif feature_normalize != "none":
            raise ValueError("--feature-normalize must be one of: none, l2")

        if sum_pos is None:
            sum_pos = torch.zeros(feats.shape[1], dtype=feats.dtype, device=feats.device)
            sum_neg = torch.zeros(feats.shape[1], dtype=feats.dtype, device=feats.device)
        if pos_idx:
            sum_pos += feats[pos_idx].sum(dim=0)
        if neg_idx:
            sum_neg += feats[neg_idx].sum(dim=0)

        offset += batch_n

    if offset != len(labels):
        raise RuntimeError(f"dev extraction mismatch: got {offset} feats for {len(labels)} labels")
    if cnt_pos <= 0 or cnt_neg <= 0:
        raise ValueError("persona_vector requires both positive and negative examples in dev split")
    if sum_pos is None or sum_neg is None:
        raise RuntimeError("failed to fit persona vector: no features extracted")

    mu_pos = sum_pos / float(cnt_pos)
    mu_neg = sum_neg / float(cnt_neg)
    v = mu_pos - mu_neg
    v = v / (torch.norm(v) + float(eps))
    return v


def _windowed_max(scores: Any, *, window_size: int, decay: float) -> Any:
    # scores: (L,B)
    import torch

    l, _b = scores.shape
    if l <= 0:
        return scores.new_empty((0,))
    if window_size <= 0:
        raise ValueError("--window-size must be > 0")
    if not (0.0 < float(decay) <= 1.0):
        raise ValueError("--window-decay must be in (0,1]")

    best = torch.full((scores.shape[1],), float("-inf"), dtype=scores.dtype, device=scores.device)
    for t in range(l):
        start = max(0, t - int(window_size) + 1)
        seg = scores[start : t + 1]  # (m,B)
        m = int(seg.shape[0])
        ws = torch.tensor(
            [float(decay) ** (m - 1 - i) for i in range(m)],
            dtype=scores.dtype,
            device=scores.device,
        ).view(m, 1)
        avg = (seg * ws).sum(dim=0) / float(ws.sum())
        best = torch.maximum(best, avg)
    return best


class _EarlyExitPick:
    def __init__(
        self,
        *,
        exit_threshold: float,
        final_direction: str,
        final_threshold: float,
        dev_acc: float,
        dev_f1: float,
        stage2_rate: float,
        baseline_dev_acc: float,
        baseline_dev_f1: float,
    ) -> None:
        self.exit_threshold = float(exit_threshold)
        self.final_direction = str(final_direction)
        self.final_threshold = float(final_threshold)
        self.dev_acc = float(dev_acc)
        self.dev_f1 = float(dev_f1)
        self.stage2_rate = float(stage2_rate)
        self.baseline_dev_acc = float(baseline_dev_acc)
        self.baseline_dev_f1 = float(baseline_dev_f1)


def _pick_earlyexit_threshold(
    *,
    early_scores_dev: list[float],
    final_scores_dev: list[float],
    labels_dev: list[int],
    final_direction: str,
    fixed_final_threshold: float | None,
    fixed_final_direction: str,
    max_acc_drop: float,
    max_f1_drop: float,
    fixed_exit_threshold: float | None,
) -> _EarlyExitPick:
    if len(early_scores_dev) != len(final_scores_dev) or len(early_scores_dev) != len(labels_dev):
        raise ValueError("dev scores/labels length mismatch")

    # Baseline: final-only on dev.
    if fixed_final_threshold is None:
        base_pick = pick_threshold(final_scores_dev, labels_dev, direction=str(final_direction))
        base_thr = float(base_pick.threshold)
        base_dir = str(base_pick.direction)
    else:
        base_thr = float(fixed_final_threshold)
        base_dir = str(fixed_final_direction)
        if base_dir == "auto":
            base_dir = "high"

    base_pred = predict_from_scores(final_scores_dev, threshold=base_thr, direction=base_dir)
    base_m = _metrics(labels_dev, base_pred)
    base_acc = float(base_m["acc"])
    base_f1 = float(base_m["f1"])

    if fixed_exit_threshold is not None:
        candidates = [float(fixed_exit_threshold)]
    else:
        uniq = sorted(set(float(s) for s in early_scores_dev))
        if not uniq:
            candidates = [0.0]
        else:
            eps = max(1e-6, (uniq[-1] - uniq[0]) * 1e-6)
            # Condition is early_score >= thr => early-exit to abnormal.
            candidates = [math.nextafter(uniq[-1], float("inf"))] + uniq + [uniq[0] - eps]

    best_ok: _EarlyExitPick | None = None
    best_fallback: _EarlyExitPick | None = None

    for exit_thr in candidates:
        pass_mask = [s < float(exit_thr) for s in early_scores_dev]
        pass_scores = [s for s, keep in zip(final_scores_dev, pass_mask, strict=True) if keep]
        pass_labels = [y for y, keep in zip(labels_dev, pass_mask, strict=True) if keep]

        if fixed_final_threshold is None:
            if pass_scores:
                pick = pick_threshold(pass_scores, pass_labels, direction=str(final_direction))
                f_dir = str(pick.direction)
                f_thr = float(pick.threshold)
            else:
                f_dir = str(base_dir)
                f_thr = float(base_thr)
        else:
            f_dir = str(base_dir)
            f_thr = float(base_thr)

        dev_pred: list[int] = []
        for s_early, s_final in zip(early_scores_dev, final_scores_dev, strict=True):
            if s_early >= float(exit_thr):
                dev_pred.append(1)
            else:
                dev_pred.append(1 if ((s_final >= f_thr) if f_dir == "high" else (s_final < f_thr)) else 0)

        m = _metrics(labels_dev, dev_pred)
        dev_acc = float(m["acc"])
        dev_f1 = float(m["f1"])
        stage2_rate = sum(1 for keep in pass_mask if keep) / len(pass_mask) if pass_mask else 1.0
        cand = _EarlyExitPick(
            exit_threshold=float(exit_thr),
            final_direction=str(f_dir),
            final_threshold=float(f_thr),
            dev_acc=float(dev_acc),
            dev_f1=float(dev_f1),
            stage2_rate=float(stage2_rate),
            baseline_dev_acc=float(base_acc),
            baseline_dev_f1=float(base_f1),
        )

        ok = (dev_acc >= base_acc - float(max_acc_drop)) and (dev_f1 >= base_f1 - float(max_f1_drop))
        if ok:
            if best_ok is None:
                best_ok = cand
            else:
                if cand.stage2_rate < best_ok.stage2_rate:
                    best_ok = cand
                elif math.isclose(cand.stage2_rate, best_ok.stage2_rate):
                    if cand.dev_acc > best_ok.dev_acc or (
                        math.isclose(cand.dev_acc, best_ok.dev_acc) and cand.dev_f1 > best_ok.dev_f1
                    ):
                        best_ok = cand

        if best_fallback is None:
            best_fallback = cand
        else:
            if cand.dev_acc > best_fallback.dev_acc:
                best_fallback = cand
            elif math.isclose(cand.dev_acc, best_fallback.dev_acc):
                if cand.dev_f1 > best_fallback.dev_f1:
                    best_fallback = cand
                elif math.isclose(cand.dev_f1, best_fallback.dev_f1) and cand.stage2_rate < best_fallback.stage2_rate:
                    best_fallback = cand

    return best_ok or best_fallback or _EarlyExitPick(
        exit_threshold=float("inf"),
        final_direction=str(base_dir),
        final_threshold=float(base_thr),
        dev_acc=float("nan"),
        dev_f1=float("nan"),
        stage2_rate=1.0,
        baseline_dev_acc=float(base_acc),
        baseline_dev_f1=float(base_f1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neuron-only layerwise persona early-exit for abnormal data detection (final layer -11 by default)"
    )
    parser.add_argument(
        "--dataset-path",
        default="datasets/generated_datasets/v0.1.1/dataset_hf",
        help="Path to a HF dataset saved by Dataset.save_to_disk(), or a supported .jsonl file",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B", help="Backbone model used to extract activations")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--include-assistant", action="store_true", help="Extract text from full dialogue (user+assistant)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Dev split ratio for tuning")
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N examples (0 means all)")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE=1 to prevent downloads",
    )

    # Extraction config
    parser.add_argument(
        "--backbone",
        choices=["base", "causal_lm"],
        default="base",
        help="Extraction backbone: base=AutoModel(no lm_head, faster) | causal_lm=AutoModelForCausalLM",
    )
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--feature-normalize", choices=["none", "l2"], default="l2")

    # Final decision config
    parser.add_argument("--final-layer", type=int, default=-11, help="Final decision layer (negative indexes allowed)")
    parser.add_argument(
        "--final-direction",
        choices=["auto", "high", "low"],
        default="auto",
        help="Direction for final threshold picking (auto tries both, chooses best dev ACC).",
    )
    parser.add_argument("--final-threshold", type=float, default=None, help="Fixed final threshold (skip dev tuning if set)")

    # Early-exit config
    parser.add_argument(
        "--exit-max-layer",
        type=int,
        default=8,
        help="Max layer for layerwise early-exit scoring (0..K). Smaller => faster stage1.",
    )
    parser.add_argument("--exit-min-layer", type=int, default=0)
    parser.add_argument("--exit-step", type=int, default=1)
    parser.add_argument("--exit-mode", choices=["raw", "window"], default="raw")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--window-decay", type=float, default=0.9)
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=None,
        help="Fixed early-exit threshold on early-score (skip dev tuning if set). Condition: early_score>=thr => abnormal.",
    )
    parser.add_argument(
        "--max-acc-drop",
        type=float,
        default=0.005,
        help="When auto-picking exit threshold, keep dev ACC within this drop vs final-only baseline.",
    )
    parser.add_argument(
        "--max-f1-drop",
        type=float,
        default=0.01,
        help="When auto-picking exit threshold, keep dev F1 within this drop vs final-only baseline.",
    )
    parser.add_argument("--timing", action="store_true", help="Print rough timing + throughput stats")
    parser.add_argument(
        "--inference-mode",
        choices=["batch", "sample"],
        default="batch",
        help="Stage1 inference mode: batch=materialize all early layers for the whole batch; sample=sample-level shrink.",
    )
    args = parser.parse_args()

    if bool(args.offline):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    rows = load_rows(str(args.dataset_path), limit=int(args.limit))
    labels, texts = labels_texts(rows, include_assistant=bool(args.include_assistant))

    dev_idx, eval_idx, split_mode = resolve_dev_eval_indices(
        rows,
        labels,
        dev_ratio=float(args.dev_ratio),
        seed=int(args.seed),
    )
    dev_texts = [texts[i] for i in dev_idx] if dev_idx else texts
    dev_labels = [labels[i] for i in dev_idx] if dev_idx else labels

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

    blocks = get_transformer_layers(model)
    n_layers = len(blocks)
    final_layer = _resolve_layer(int(args.final_layer), n_layers=n_layers)
    exit_min = _resolve_layer(int(args.exit_min_layer), n_layers=n_layers)
    exit_max = _resolve_layer(int(args.exit_max_layer), n_layers=n_layers)
    if int(args.exit_step) <= 0:
        raise ValueError("--exit-step must be > 0")
    if exit_min > exit_max:
        raise ValueError("--exit-min-layer must be <= --exit-max-layer")
    exit_layers = list(range(int(exit_min), int(exit_max) + 1, int(args.exit_step)))

    # Dev: fit persona vectors for early layers (0..K).
    t_dev_early0 = time.perf_counter()
    with MultiLayerActivationExtractor(
        torch=torch,
        device=device,
        tokenizer=tokenizer,
        model=model,
        layers=exit_layers,
        pooling=str(args.pooling),
        eps=float(args.eps),
        stop_at_max_layer=True,
    ) as early_extractor:
        v_early = _fit_persona_vectors(
            torch=torch,
            extractor=early_extractor,
            texts=dev_texts,
            labels=dev_labels,
            feature_normalize=str(args.feature_normalize),
            eps=float(args.eps),
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
        )
    t_dev_early1 = time.perf_counter()

    # Dev: fit persona vector for final layer (-11 by default).
    t_dev_final0 = time.perf_counter()
    with ActivationExtractor(
        torch=torch,
        device=device,
        tokenizer=tokenizer,
        model=model,
        layer=int(final_layer),
        pooling=str(args.pooling),
        eps=float(args.eps),
        stop_after_layer=True,
    ) as final_extractor:
        v_final = _fit_persona_vector_single(
            torch=torch,
            extractor=final_extractor,
            texts=dev_texts,
            labels=dev_labels,
            feature_normalize=str(args.feature_normalize),
            eps=float(args.eps),
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
        )
    t_dev_final1 = time.perf_counter()

    # Dev: compute early_score (max over layers, or max over windowed avgs).
    with MultiLayerActivationExtractor(
        torch=torch,
        device=device,
        tokenizer=tokenizer,
        model=model,
        layers=exit_layers,
        pooling=str(args.pooling),
        eps=float(args.eps),
        stop_at_max_layer=True,
    ) as early_extractor:
        early_scores_dev: list[float] = []
        offset = 0
        for feats_by_layer in early_extractor.iter_extract(
            dev_texts, batch_size=int(args.batch_size), max_length=int(args.max_length)
        ):
            batch_n = next(iter(feats_by_layer.values())).shape[0] if feats_by_layer else 0
            feats_layers = []
            for li in early_extractor.layer_idxs:
                feats = feats_by_layer[int(li)]
                if args.feature_normalize == "l2":
                    feats = _l2_normalize(feats, eps=float(args.eps))
                scores = feats @ v_early[int(li)]
                feats_layers.append(scores)

            if feats_layers:
                stacked = torch.stack(feats_layers, dim=0)  # (L,B)
                if str(args.exit_mode) == "raw":
                    early_batch = stacked.max(dim=0).values
                else:
                    early_batch = _windowed_max(
                        stacked, window_size=int(args.window_size), decay=float(args.window_decay)
                    )
                early_scores_dev.extend([float(x) for x in early_batch.tolist()])
            offset += int(batch_n)
        if offset != len(dev_texts):
            raise RuntimeError(f"dev early score mismatch: got {offset} feats for {len(dev_texts)} texts")

    # Dev: compute final scores for threshold tuning.
    with ActivationExtractor(
        torch=torch,
        device=device,
        tokenizer=tokenizer,
        model=model,
        layer=int(final_layer),
        pooling=str(args.pooling),
        eps=float(args.eps),
        stop_after_layer=True,
    ) as final_extractor:
        final_scores_dev: list[float] = []
        for feats in final_extractor.iter_extract(dev_texts, batch_size=int(args.batch_size), max_length=int(args.max_length)):
            if args.feature_normalize == "l2":
                feats = _l2_normalize(feats, eps=float(args.eps))
            s = feats @ v_final
            final_scores_dev.extend([float(x) for x in s.tolist()])

    if len(early_scores_dev) != len(dev_labels) or len(final_scores_dev) != len(dev_labels):
        raise RuntimeError("dev score length mismatch")

    pick = _pick_earlyexit_threshold(
        early_scores_dev=early_scores_dev,
        final_scores_dev=final_scores_dev,
        labels_dev=dev_labels,
        final_direction=str(args.final_direction),
        fixed_final_threshold=(float(args.final_threshold) if args.final_threshold is not None else None),
        fixed_final_direction=str(args.final_direction),
        max_acc_drop=float(args.max_acc_drop),
        max_f1_drop=float(args.max_f1_drop),
        fixed_exit_threshold=(float(args.exit_threshold) if args.exit_threshold is not None else None),
    )
    exit_thr = float(pick.exit_threshold)
    final_thr = float(pick.final_threshold)
    final_dir = str(pick.final_direction)

    t_load1 = time.perf_counter()
    _empty_cache_if_cuda(torch=torch, device=device)

    # Inference: stage1 early layers, then stage2 final layer for remaining samples.
    y_pred = [0 for _ in range(len(labels))]
    stage2_idx: list[int] = []
    stage1_exit_layer_hist: dict[int, int] = {}

    t_infer1_0 = time.perf_counter()
    sync_if_cuda(torch=torch, device=device)
    if str(args.inference_mode) == "batch":
        with MultiLayerActivationExtractor(
            torch=torch,
            device=device,
            tokenizer=tokenizer,
            model=model,
            layers=exit_layers,
            pooling=str(args.pooling),
            eps=float(args.eps),
            stop_at_max_layer=True,
        ) as early_extractor:
            offset = 0
            for feats_by_layer in early_extractor.iter_extract(
                texts, batch_size=int(args.batch_size), max_length=int(args.max_length)
            ):
                batch_n = next(iter(feats_by_layer.values())).shape[0] if feats_by_layer else 0
                layer_scores: list[Any] = []
                for li in early_extractor.layer_idxs:
                    feats = feats_by_layer[int(li)]
                    if args.feature_normalize == "l2":
                        feats = _l2_normalize(feats, eps=float(args.eps))
                    layer_scores.append(feats @ v_early[int(li)])

                if not layer_scores:
                    offset += int(batch_n)
                    continue

                stacked = torch.stack(layer_scores, dim=0)  # (L,B)
                if str(args.exit_mode) == "raw":
                    early_batch = stacked.max(dim=0).values
                else:
                    early_batch = _windowed_max(
                        stacked, window_size=int(args.window_size), decay=float(args.window_decay)
                    )

                exit_mask = early_batch >= float(exit_thr)
                if bool(exit_mask.any()):
                    if str(args.exit_mode) == "raw":
                        hit = torch.full((int(batch_n),), -1, dtype=torch.long, device=layer_scores[0].device)
                        for li, s in zip(early_extractor.layer_idxs, layer_scores, strict=True):
                            m = (hit < 0) & (s >= float(exit_thr))
                            hit = torch.where(m, torch.full_like(hit, int(li)), hit)
                    else:
                        hit = torch.full((int(batch_n),), -1, dtype=torch.long, device=layer_scores[0].device)
                        buf: list[Any] = []
                        for li, s in zip(early_extractor.layer_idxs, layer_scores, strict=True):
                            buf.append(s)
                            if len(buf) > int(args.window_size):
                                buf.pop(0)
                            m = len(buf)
                            ws = torch.tensor(
                                [float(args.window_decay) ** (m - 1 - i) for i in range(m)],
                                dtype=s.dtype,
                                device=s.device,
                            ).view(m, 1)
                            seg = torch.stack(buf, dim=0)
                            avg = (seg * ws).sum(dim=0) / float(ws.sum())
                            m_hit = (hit < 0) & (avg >= float(exit_thr))
                            hit = torch.where(m_hit, torch.full_like(hit, int(li)), hit)
                    for h in hit[exit_mask].tolist():
                        if int(h) >= 0:
                            stage1_exit_layer_hist[int(h)] = stage1_exit_layer_hist.get(int(h), 0) + 1

                for j in range(int(batch_n)):
                    idx = int(offset + j)
                    if bool(exit_mask[j].item()):
                        y_pred[idx] = 1
                    else:
                        stage2_idx.append(idx)
                offset += int(batch_n)
    else:
        offset = 0
        for batch in [texts[i : i + int(args.batch_size)] for i in range(0, len(texts), int(args.batch_size))]:
            batch_n = len(batch)
            tok = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
            )
            tok = move_batch_to_device(tok, device=device)
            window_buf: list[Any] = []

            def layer_callback(layer_idx: int, hidden_states: Any, current_attention_mask: Any, current_indices: Any) -> Any:
                feats = pool_hidden(
                    hidden_states,
                    current_attention_mask,
                    pooling=str(args.pooling),
                    eps=float(args.eps),
                )
                if args.feature_normalize == "l2":
                    feats_local = _l2_normalize(feats, eps=float(args.eps))
                else:
                    feats_local = feats
                feats_local = feats_local.float()
                scores = feats_local @ v_early[int(layer_idx)]
                if str(args.exit_mode) == "raw":
                    exit_mask_local = scores >= float(exit_thr)
                else:
                    window_buf.append(scores)
                    if len(window_buf) > int(args.window_size):
                        window_buf.pop(0)
                    m = len(window_buf)
                    ws = torch.tensor(
                        [float(args.window_decay) ** (m - 1 - i) for i in range(m)],
                        dtype=scores.dtype,
                        device=scores.device,
                    ).view(m, 1)
                    seg = torch.stack(window_buf, dim=0)
                    avg = (seg * ws).sum(dim=0) / float(ws.sum())
                    exit_mask_local = avg >= float(exit_thr)

                if bool(exit_mask_local.any()):
                    for h in torch.full_like(current_indices[exit_mask_local], int(layer_idx)).tolist():
                        stage1_exit_layer_hist[int(h)] = stage1_exit_layer_hist.get(int(h), 0) + 1
                    for idx_local in current_indices[exit_mask_local].tolist():
                        y_pred[int(offset + idx_local)] = 1

                keep_mask_local = ~exit_mask_local
                if str(args.exit_mode) == "window" and window_buf:
                    window_buf[:] = [s[keep_mask_local] for s in window_buf]
                return keep_mask_local

            out = model.forward_early_exit(
                **tok,
                use_cache=False,
                monitored_layers=exit_layers,
                layer_callback=layer_callback,
            )
            remaining = out.active_indices
            if remaining is not None:
                for idx_local in remaining.tolist():
                    stage2_idx.append(int(offset + idx_local))
            offset += int(batch_n)
    sync_if_cuda(torch=torch, device=device)
    t_infer1_1 = time.perf_counter()
    _empty_cache_if_cuda(torch=torch, device=device)

    t_infer2_0 = time.perf_counter()
    sync_if_cuda(torch=torch, device=device)
    if stage2_idx:
        stage2_texts = [texts[i] for i in stage2_idx]
        with ActivationExtractor(
            torch=torch,
            device=device,
            tokenizer=tokenizer,
            model=model,
            layer=int(final_layer),
            pooling=str(args.pooling),
            eps=float(args.eps),
            stop_after_layer=True,
        ) as final_extractor:
            offset2 = 0
            for feats in final_extractor.iter_extract(
                stage2_texts, batch_size=int(args.batch_size), max_length=int(args.max_length)
            ):
                if args.feature_normalize == "l2":
                    feats = _l2_normalize(feats, eps=float(args.eps))
                s = feats @ v_final
                preds = predict_from_scores([float(x) for x in s.tolist()], threshold=float(final_thr), direction=str(final_dir))
                for j, p in enumerate(preds):
                    y_pred[stage2_idx[offset2 + j]] = int(p)
                offset2 += int(feats.shape[0])
    sync_if_cuda(torch=torch, device=device)
    t_infer2_1 = time.perf_counter()

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
    stage2_rate = len(stage2_idx) / n if n else float("nan")
    stage1_shortcut_rate = 1.0 - stage2_rate if n else float("nan")

    print(f"dataset_path={args.dataset_path}")
    print(
        "activation_extractor="
        f"model_id={model_info['model_id']} backbone={model_info['backbone']} device={model_info['device']} dtype={model_info['dtype']} "
        f"final_layer={args.final_layer} (resolved {final_layer}/{n_layers}) pooling={args.pooling} "
        f"max_length={args.max_length} batch_size={args.batch_size}"
    )
    print(
        "early_exit="
        f"layers=[{exit_layers[0]}..{exit_layers[-1]}] step={args.exit_step} mode={args.exit_mode} "
        f"exit_threshold={exit_thr:.6g} inference_mode={args.inference_mode} "
        f"stage1_shortcut_rate={stage1_shortcut_rate:.4f} stage2_rate={stage2_rate:.4f}"
    )
    if str(args.exit_mode) == "window":
        print(f"early_exit_window=window_size={args.window_size} window_decay={args.window_decay}")
    print(
        "tuning="
        f"baseline_dev_acc={pick.baseline_dev_acc:.4f} baseline_dev_f1={pick.baseline_dev_f1:.4f} "
        f"picked_dev_acc={pick.dev_acc:.4f} picked_dev_f1={pick.dev_f1:.4f} picked_dev_stage2_rate={pick.stage2_rate:.4f} "
        f"max_acc_drop={args.max_acc_drop} max_f1_drop={args.max_f1_drop}"
    )
    print(f"final_scoring=method:persona_vector feature_normalize:{args.feature_normalize} direction:{final_dir} threshold:{final_thr:.6g}")
    print(f"n={n} pos_rate={pos_rate:.4f} split_mode={split_mode}")
    print(f"eval_n={res_eval['n']} eval_acc={res_eval['metrics']['acc']:.4f}")
    print(
        "eval_confusion="
        f"tp:{res_eval['conf']['tp']} tn:{res_eval['conf']['tn']} fp:{res_eval['conf']['fp']} fn:{res_eval['conf']['fn']}"
    )
    print(
        "eval_metrics="
        f"precision:{res_eval['metrics']['precision']:.4f} recall:{res_eval['metrics']['recall']:.4f} f1:{res_eval['metrics']['f1']:.4f}"
    )
    if stage1_exit_layer_hist:
        top = sorted(stage1_exit_layer_hist.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
        top_s = " ".join(f"{k}:{v}" for k, v in top)
        print(f"early_exit_hist_top={top_s}")

    if bool(args.timing):
        dev_early_s = float(t_dev_early1 - t_dev_early0)
        dev_final_s = float(t_dev_final1 - t_dev_final0)
        load_fit_s = float(t_load1 - t_load0)
        infer_stage1_s = float(t_infer1_1 - t_infer1_0)
        infer_stage2_s = float(t_infer2_1 - t_infer2_0)
        infer_s = infer_stage1_s + infer_stage2_s
        total_s = float(t_infer2_1 - t_load0)
        sps_infer = (n / infer_s) if infer_s > 0 else float("nan")
        sps_total = (n / total_s) if total_s > 0 else float("nan")
        print(
            "timing="
            f"dev_early_s={dev_early_s:.3f} dev_final_s={dev_final_s:.3f} load_fit_s={load_fit_s:.3f} "
            f"infer_stage1_s={infer_stage1_s:.3f} infer_stage2_s={infer_stage2_s:.3f} "
            f"infer_s={infer_s:.3f} total_s={total_s:.3f} sps_infer={sps_infer:.1f} sps_total={sps_total:.1f}"
        )


if __name__ == "__main__":
    main()
