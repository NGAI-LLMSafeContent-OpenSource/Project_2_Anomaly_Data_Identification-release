from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from filterguard.data import abnormal_label
from filterguard.metric import ThresholdPick, confusion, metrics_from_conf, pick_threshold


NORMAL_RISK_LABEL = "normal"


def risk_category_label(ex: dict[str, Any], *, risk_level: str) -> str:
    if abnormal_label(ex) == 0:
        return NORMAL_RISK_LABEL

    if risk_level == "l1":
        key = "harm_category_L1"
    elif risk_level == "l2":
        key = "harm_category_L2"
    else:
        raise ValueError("risk_level must be one of: l1, l2")

    label = str(ex.get(key) or "").strip()
    if not label:
        raise ValueError(f"Abnormal row is missing required risk category field {key!r}.")
    return label


def risk_labels(rows: list[dict[str, Any]], *, risk_level: str) -> list[str]:
    return [risk_category_label(row, risk_level=risk_level) for row in rows]


def ordered_risk_classes(labels: list[str]) -> list[str]:
    non_normal = sorted({label for label in labels if label != NORMAL_RISK_LABEL})
    return [NORMAL_RISK_LABEL, *non_normal] if NORMAL_RISK_LABEL in labels else non_normal


def normalize_features(f: Any, *, feature_normalize: str, eps: float) -> Any:
    import torch

    if feature_normalize == "none":
        return f
    if feature_normalize != "l2":
        raise ValueError(f"unsupported feature_normalize: {feature_normalize!r}")
    denom = torch.norm(f, dim=1, keepdim=True).clamp_min(float(eps))
    return f / denom


@dataclass(frozen=True)
class RiskPersonaParams:
    classes: list[str]
    feature_normalize: str
    eps: float
    vectors: Any
    class_support: dict[str, int]


def fit_one_vs_rest_risk_persona(
    features_dev: Any,
    labels_dev: list[str],
    *,
    classes: list[str],
    feature_normalize: str,
    eps: float,
) -> RiskPersonaParams:
    import torch

    if features_dev.shape[0] != len(labels_dev):
        raise ValueError("features_dev and labels_dev length mismatch")
    if not classes:
        raise ValueError("classes must be non-empty")

    f = normalize_features(features_dev, feature_normalize=feature_normalize, eps=float(eps))
    vectors = []
    support: dict[str, int] = {}
    for cls in classes:
        pos_idx = [i for i, label in enumerate(labels_dev) if label == cls]
        neg_idx = [i for i, label in enumerate(labels_dev) if label != cls]
        support[cls] = len(pos_idx)
        if not pos_idx:
            raise ValueError(f"Cannot fit risk persona vector for class {cls!r}: no dev examples.")
        if not neg_idx:
            raise ValueError(f"Cannot fit risk persona vector for class {cls!r}: no one-vs-rest dev examples.")
        v = f[pos_idx].mean(dim=0) - f[neg_idx].mean(dim=0)
        v = v / (torch.norm(v) + float(eps))
        vectors.append(v)

    return RiskPersonaParams(
        classes=list(classes),
        feature_normalize=str(feature_normalize),
        eps=float(eps),
        vectors=torch.stack(vectors, dim=0),
        class_support=support,
    )


def risk_scores_from_params(features: Any, params: RiskPersonaParams) -> Any:
    f = normalize_features(features, feature_normalize=params.feature_normalize, eps=float(params.eps))
    return f @ params.vectors.T


def predict_risk_labels(scores: Any, classes: list[str]) -> list[str]:
    idx = scores.argmax(dim=1).tolist()
    return [classes[int(i)] for i in idx]


def fit_one_vs_rest_thresholds(
    scores_dev: Any,
    labels_dev: list[str],
    *,
    classes: list[str],
    direction: str = "auto",
) -> dict[str, ThresholdPick]:
    if scores_dev.shape[0] != len(labels_dev):
        raise ValueError("scores_dev and labels_dev length mismatch")
    thresholds: dict[str, ThresholdPick] = {}
    for j, cls in enumerate(classes):
        one_vs_rest = [1 if label == cls else 0 for label in labels_dev]
        cls_scores = [float(x) for x in scores_dev[:, j].tolist()]
        thresholds[cls] = pick_threshold(cls_scores, one_vs_rest, direction=direction)
    return thresholds


def apply_one_vs_rest_margins(scores: Any, classes: list[str], thresholds: dict[str, ThresholdPick]) -> Any:
    import torch

    adjusted_parts = []
    for j, cls in enumerate(classes):
        pick = thresholds.get(cls)
        if pick is None:
            raise ValueError(f"missing one-vs-rest threshold for class {cls!r}")
        raw = scores[:, j]
        if pick.direction == "high":
            adjusted_parts.append(raw - float(pick.threshold))
        elif pick.direction == "low":
            adjusted_parts.append(float(pick.threshold) - raw)
        else:
            raise ValueError(f"unsupported threshold direction for class {cls!r}: {pick.direction!r}")
    return torch.stack(adjusted_parts, dim=1)


def multiclass_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    classes: list[str],
) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p != cls)
        support = sum(1 for t in y_true if t == cls)
        pred_count = sum(1 for p in y_pred if p == cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_values.append(float(f1))
        per_class[cls] = {
            "support": int(support),
            "pred_count": int(pred_count),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    return {
        "n": int(n),
        "acc": (correct / n if n else float("nan")),
        "macro_f1": (sum(f1_values) / len(f1_values) if f1_values else float("nan")),
        "per_class": per_class,
    }


def abnormal_binary_metrics_from_risk(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    true_bin = [0 if y == NORMAL_RISK_LABEL else 1 for y in y_true]
    pred_bin = [0 if y == NORMAL_RISK_LABEL else 1 for y in y_pred]
    conf = confusion(true_bin, pred_bin)
    return {"conf": conf, "metrics": metrics_from_conf(conf)}
