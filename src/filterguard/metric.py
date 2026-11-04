#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Any


def extract_text(ex: dict[str, Any], *, include_assistant: bool) -> str:
    messages = ex.get("messages")
    if isinstance(messages, list) and messages:
        parts: list[str] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip()
            content = str(m.get("content") or "").strip()
            if not content:
                continue
            if include_assistant or role == "user":
                parts.append(content)
        if parts:
            return "\n".join(parts)

    instruction = ex.get("instruction")
    input_text = ex.get("input")
    output = ex.get("output")
    prompt = ex.get("prompt")
    text = ex.get("text")

    parts2: list[str] = []
    if instruction:
        parts2.append(str(instruction))
    if input_text:
        parts2.append(str(input_text))
    if prompt:
        parts2.append(str(prompt))
    if include_assistant and output:
        parts2.append(str(output))
    if not parts2 and text:
        parts2.append(str(text))
    return "\n".join(parts2)


def confusion(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_conf(conf: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = conf["tp"], conf["tn"], conf["fp"], conf["fn"]
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": recall,
        "tnr": tnr,
    }


def split_indices(labels: list[int], dev_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if dev_ratio <= 0.0:
        return [], list(range(len(labels)))
    if dev_ratio >= 1.0:
        return list(range(len(labels))), []

    rnd = Random(seed)
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    rnd.shuffle(pos)
    rnd.shuffle(neg)

    pos_dev = int(len(pos) * dev_ratio)
    neg_dev = int(len(neg) * dev_ratio)
    dev = pos[:pos_dev] + neg[:neg_dev]
    eval_ = pos[pos_dev:] + neg[neg_dev:]
    rnd.shuffle(dev)
    rnd.shuffle(eval_)
    return dev, eval_


@dataclass(frozen=True)
class ThresholdPick:
    threshold: float
    direction: str  # "high" => score >= thr is abnormal; "low" => score < thr is abnormal
    acc: float


def predict_from_scores(scores: list[float], *, threshold: float, direction: str) -> list[int]:
    if direction == "high":
        return [1 if s >= threshold else 0 for s in scores]
    if direction == "low":
        return [1 if s < threshold else 0 for s in scores]
    raise ValueError(f"unsupported direction: {direction!r}")


def _best_threshold(scores: list[float], labels: list[int], *, direction: str) -> ThresholdPick:
    if not scores:
        return ThresholdPick(threshold=0.0, direction=direction, acc=float("nan"))

    pairs = sorted(zip(scores, labels, strict=True), key=lambda x: x[0])
    total = len(pairs)
    pos_total = sum(labels)
    neg_total = total - pos_total

    best_thr = float("nan")
    best_acc = -1.0

    if direction == "high":
        # Predict abnormal if score >= thr.
        tp = fp = 0
        # Start with thr > max(score) => predict all negative
        tn = neg_total
        fn = pos_total
        best_thr = math.nextafter(pairs[-1][0], float("inf"))
        best_acc = (tp + tn) / total if total else float("nan")

        i = total - 1
        while i >= 0:
            s = pairs[i][0]
            j = i
            # Include all items with score == s.
            while j >= 0 and pairs[j][0] == s:
                y = pairs[j][1]
                if y == 1:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
                    tn -= 1
                j -= 1
            acc = (tp + tn) / total if total else float("nan")
            if acc > best_acc or (math.isclose(acc, best_acc) and s > best_thr):
                best_acc = acc
                best_thr = s
            i = j
        return ThresholdPick(threshold=float(best_thr), direction=direction, acc=float(best_acc))

    if direction == "low":
        # Predict abnormal if score < thr.
        tp = fp = 0  # predicted abnormal among score < current thr
        tn = neg_total
        fn = pos_total

        # Start with thr <= min(score) => predict all negative (since score < thr matches none)
        best_thr = pairs[0][0]
        best_acc = (tp + tn) / total if total else float("nan")

        i = 0
        while i < total:
            s = pairs[i][0]
            # Evaluate at thr = s (before including score==s), because rule is score < thr.
            acc = (tp + tn) / total if total else float("nan")
            if acc > best_acc or (math.isclose(acc, best_acc) and s > best_thr):
                best_acc = acc
                best_thr = s

            j = i
            while j < total and pairs[j][0] == s:
                y = pairs[j][1]
                if y == 1:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
                    tn -= 1
                j += 1
            i = j

        # thr > max(score) => predict all positive
        thr_all_pos = math.nextafter(pairs[-1][0], float("inf"))
        acc = (tp + tn) / total if total else float("nan")
        if acc > best_acc or (math.isclose(acc, best_acc) and thr_all_pos > best_thr):
            best_acc = acc
            best_thr = thr_all_pos
        return ThresholdPick(threshold=float(best_thr), direction=direction, acc=float(best_acc))

    raise ValueError(f"unsupported direction: {direction!r}")


def pick_threshold(scores: list[float], labels: list[int], *, direction: str = "auto") -> ThresholdPick:
    if direction in {"high", "low"}:
        return _best_threshold(scores, labels, direction=direction)
    if direction != "auto":
        raise ValueError("direction must be one of: auto, high, low")
    high = _best_threshold(scores, labels, direction="high")
    low = _best_threshold(scores, labels, direction="low")
    if high.acc > low.acc:
        return high
    if low.acc > high.acc:
        return low
    # Tie: prefer "low" (useful for score-as-quality setups).
    return low
