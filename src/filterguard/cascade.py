from __future__ import annotations

import math
from dataclasses import dataclass

from filterguard.metric import confusion, metrics_from_conf, pick_threshold, predict_from_scores


def _metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return metrics_from_conf(confusion(y_true, y_pred))


@dataclass(frozen=True)
class KwNeuronCascadePick:
    kw_threshold: int
    neuron_direction: str
    neuron_threshold: float
    dev_acc: float
    dev_f1: float
    neuron_used_rate: float


def pick_kw_neuron_cascade(
    *,
    kw_scores_all: list[int],
    neuron_scores_dev: list[float],
    kw_scores_dev: list[int],
    labels_dev: list[int],
    max_acc_drop: float,
    max_f1_drop: float,
    neuron_direction: str,
    fixed_neuron_threshold: float | None,
    fixed_neuron_direction: str,
) -> KwNeuronCascadePick:
    # Baseline: neuron-only on dev (no keyword shortcut).
    if fixed_neuron_threshold is None:
        base_pick = pick_threshold(neuron_scores_dev, labels_dev, direction=neuron_direction)
        base_thr = float(base_pick.threshold)
        base_dir = str(base_pick.direction)
    else:
        base_thr = float(fixed_neuron_threshold)
        base_dir = str(fixed_neuron_direction)
        if base_dir == "auto":
            base_dir = "high"

    base_pred = predict_from_scores(neuron_scores_dev, threshold=base_thr, direction=base_dir)
    base_m = _metrics(labels_dev, base_pred)
    base_acc = float(base_m["acc"])
    base_f1 = float(base_m["f1"])

    max_kw = max(kw_scores_dev) if kw_scores_dev else 0
    candidates = sorted(set(kw_scores_dev + [0, max_kw + 1]))

    best_ok: KwNeuronCascadePick | None = None
    best_fallback: KwNeuronCascadePick | None = None

    for kw_thr in candidates:
        # Examples with kw_score >= kw_thr are short-circuited to abnormal.
        pass_mask = [s < kw_thr for s in kw_scores_dev]
        pass_scores = [s for s, keep in zip(neuron_scores_dev, pass_mask, strict=True) if keep]
        pass_labels = [y for y, keep in zip(labels_dev, pass_mask, strict=True) if keep]

        if fixed_neuron_threshold is None:
            if pass_scores:
                pick = pick_threshold(pass_scores, pass_labels, direction=neuron_direction)
                n_dir = str(pick.direction)
                n_thr = float(pick.threshold)
            else:
                n_dir = "high"
                n_thr = 0.0
        else:
            n_dir = base_dir
            n_thr = base_thr

        dev_pred: list[int] = []
        for s_kw, s_nn in zip(kw_scores_dev, neuron_scores_dev, strict=True):
            if s_kw >= kw_thr:
                dev_pred.append(1)
            else:
                dev_pred.append(1 if ((s_nn >= n_thr) if n_dir == "high" else (s_nn < n_thr)) else 0)

        m = _metrics(labels_dev, dev_pred)
        dev_acc = float(m["acc"])
        dev_f1 = float(m["f1"])

        used_rate = sum(1 for s in kw_scores_all if s < kw_thr) / len(kw_scores_all) if kw_scores_all else 1.0
        cand = KwNeuronCascadePick(
            kw_threshold=int(kw_thr),
            neuron_direction=str(n_dir),
            neuron_threshold=float(n_thr),
            dev_acc=float(dev_acc),
            dev_f1=float(dev_f1),
            neuron_used_rate=float(used_rate),
        )

        ok = (dev_acc >= base_acc - float(max_acc_drop)) and (dev_f1 >= base_f1 - float(max_f1_drop))
        if ok:
            if best_ok is None:
                best_ok = cand
            else:
                if cand.neuron_used_rate < best_ok.neuron_used_rate:
                    best_ok = cand
                elif math.isclose(cand.neuron_used_rate, best_ok.neuron_used_rate):
                    if cand.dev_acc > best_ok.dev_acc or (
                        math.isclose(cand.dev_acc, best_ok.dev_acc) and cand.dev_f1 > best_ok.dev_f1
                    ):
                        best_ok = cand

        # Fallback: maximize dev_acc, then dev_f1, then less neuron usage.
        if best_fallback is None:
            best_fallback = cand
        else:
            if cand.dev_acc > best_fallback.dev_acc:
                best_fallback = cand
            elif math.isclose(cand.dev_acc, best_fallback.dev_acc):
                if cand.dev_f1 > best_fallback.dev_f1:
                    best_fallback = cand
                elif math.isclose(cand.dev_f1, best_fallback.dev_f1) and cand.neuron_used_rate < best_fallback.neuron_used_rate:
                    best_fallback = cand

    return best_ok or best_fallback or KwNeuronCascadePick(
        kw_threshold=max_kw + 1,
        neuron_direction=base_dir,
        neuron_threshold=base_thr,
        dev_acc=float("nan"),
        dev_f1=float("nan"),
        neuron_used_rate=1.0,
    )


def _fineweb_threshold_candidates(scores: list[float], *, score_mode: str) -> list[float]:
    if score_mode == "int0_5":
        return [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if not scores:
        return [0.0]
    uniq = sorted(set(float(s) for s in scores))
    if len(uniq) <= 96:
        lo = uniq[0]
        hi = uniq[-1]
        eps = max(1e-6, (hi - lo) * 1e-6)
        return [lo - eps] + uniq + [hi + eps]
    k = 64
    idxs = {int(round(i * (len(uniq) - 1) / (k - 1))) for i in range(k)}
    cand = [uniq[i] for i in sorted(idxs)]
    lo = uniq[0]
    hi = uniq[-1]
    eps = max(1e-6, (hi - lo) * 1e-6)
    return [lo - eps] + cand + [hi + eps]


@dataclass(frozen=True)
class KwFinewebNeuronCascadePick:
    kw_threshold: int
    fw_low_threshold: float | None
    fw_high_threshold: float
    neuron_direction: str
    neuron_threshold: float
    dev_acc: float
    dev_f1: float
    neuron_used_rate: float


def pick_kw_fineweb_neuron_cascade(
    *,
    kw_scores_all: list[int],
    fw_scores_all: list[float],
    neuron_scores_dev: list[float],
    kw_scores_dev: list[int],
    fw_scores_dev: list[float],
    labels_dev: list[int],
    max_acc_drop: float,
    max_f1_drop: float,
    neuron_direction: str,
    fixed_neuron_threshold: float | None,
    fixed_neuron_direction: str,
    mode: str,
    fixed_kw_threshold: int | None,
    fixed_fw_low_threshold: float | None,
    fixed_fw_high_threshold: float | None,
    fw_score_mode: str,
) -> tuple[KwFinewebNeuronCascadePick, dict[str, float]]:
    # Baseline: neuron-only on dev.
    if fixed_neuron_threshold is None:
        base_pick = pick_threshold(neuron_scores_dev, labels_dev, direction=neuron_direction)
        base_thr = float(base_pick.threshold)
        base_dir = str(base_pick.direction)
    else:
        base_thr = float(fixed_neuron_threshold)
        base_dir = str(fixed_neuron_direction)
        if base_dir == "auto":
            base_dir = "high"

    base_pred = predict_from_scores(neuron_scores_dev, threshold=base_thr, direction=base_dir)
    base_m = _metrics(labels_dev, base_pred)
    base_acc = float(base_m["acc"])
    base_f1 = float(base_m["f1"])

    max_kw = max(kw_scores_dev) if kw_scores_dev else 0
    if fixed_kw_threshold is None:
        kw_candidates = sorted(set(kw_scores_dev + [0, max_kw + 1]))
    else:
        kw_candidates = [int(fixed_kw_threshold)]

    fw_candidates = _fineweb_threshold_candidates(fw_scores_dev, score_mode=str(fw_score_mode))
    if fixed_fw_high_threshold is not None:
        fw_high_candidates = [float(fixed_fw_high_threshold)]
    else:
        fw_high_candidates = list(fw_candidates)

    if mode == "normal_only":
        if fixed_fw_low_threshold is not None:
            raise ValueError("--fineweb-low-threshold is only valid with --mode two_sided")
        fw_low_candidates: list[float | None] = [None]
    elif mode == "two_sided":
        if fixed_fw_low_threshold is not None:
            fw_low_candidates = [float(fixed_fw_low_threshold)]
        else:
            fw_low_candidates = [float(x) for x in fw_candidates]
    else:
        raise ValueError("--mode must be one of: normal_only, two_sided")

    best_ok: KwFinewebNeuronCascadePick | None = None
    best_fallback: KwFinewebNeuronCascadePick | None = None

    for kw_thr in kw_candidates:
        for fw_low in fw_low_candidates:
            for fw_high in fw_high_candidates:
                if fw_low is not None and fw_low > fw_high:
                    continue

                pass_mask: list[bool] = []
                for s_kw, s_fw in zip(kw_scores_dev, fw_scores_dev, strict=True):
                    need = s_kw < kw_thr
                    if need:
                        if fw_low is not None and s_fw < fw_low:
                            need = False
                        elif s_fw >= fw_high:
                            need = False
                    pass_mask.append(bool(need))

                pass_scores = [s for s, keep in zip(neuron_scores_dev, pass_mask, strict=True) if keep]
                pass_labels = [y for y, keep in zip(labels_dev, pass_mask, strict=True) if keep]

                if fixed_neuron_threshold is None:
                    if pass_scores:
                        pick = pick_threshold(pass_scores, pass_labels, direction=neuron_direction)
                        n_dir = str(pick.direction)
                        n_thr = float(pick.threshold)
                    else:
                        n_dir = str(base_dir)
                        n_thr = float(base_thr)
                else:
                    n_dir = str(base_dir)
                    n_thr = float(base_thr)

                dev_pred: list[int] = []
                for s_kw, s_fw, s_nn in zip(kw_scores_dev, fw_scores_dev, neuron_scores_dev, strict=True):
                    if s_kw >= kw_thr:
                        dev_pred.append(1)
                        continue
                    if fw_low is not None and s_fw < fw_low:
                        dev_pred.append(1)
                        continue
                    if s_fw >= fw_high:
                        dev_pred.append(0)
                        continue
                    dev_pred.append(1 if ((s_nn >= n_thr) if n_dir == "high" else (s_nn < n_thr)) else 0)

                m = _metrics(labels_dev, dev_pred)
                dev_acc = float(m["acc"])
                dev_f1 = float(m["f1"])

                used_n = 0
                for s_kw, s_fw in zip(kw_scores_all, fw_scores_all, strict=True):
                    if s_kw >= kw_thr:
                        continue
                    if fw_low is not None and s_fw < fw_low:
                        continue
                    if s_fw >= fw_high:
                        continue
                    used_n += 1
                used_rate = used_n / len(kw_scores_all) if kw_scores_all else 1.0

                cand = KwFinewebNeuronCascadePick(
                    kw_threshold=int(kw_thr),
                    fw_low_threshold=(float(fw_low) if fw_low is not None else None),
                    fw_high_threshold=float(fw_high),
                    neuron_direction=str(n_dir),
                    neuron_threshold=float(n_thr),
                    dev_acc=float(dev_acc),
                    dev_f1=float(dev_f1),
                    neuron_used_rate=float(used_rate),
                )

                ok = (dev_acc >= base_acc - float(max_acc_drop)) and (dev_f1 >= base_f1 - float(max_f1_drop))
                if ok:
                    if best_ok is None:
                        best_ok = cand
                    else:
                        if cand.neuron_used_rate < best_ok.neuron_used_rate:
                            best_ok = cand
                        elif math.isclose(cand.neuron_used_rate, best_ok.neuron_used_rate):
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
                        elif math.isclose(cand.dev_f1, best_fallback.dev_f1) and cand.neuron_used_rate < best_fallback.neuron_used_rate:
                            best_fallback = cand

    base_ref = {
        "base_dev_acc": float(base_acc),
        "base_dev_f1": float(base_f1),
        "base_neuron_direction": str(base_dir),
        "base_neuron_threshold": float(base_thr),
    }
    return (best_ok or best_fallback or KwFinewebNeuronCascadePick(
        kw_threshold=max_kw + 1,
        fw_low_threshold=None,
        fw_high_threshold=float("inf"),
        neuron_direction=str(base_dir),
        neuron_threshold=float(base_thr),
        dev_acc=float("nan"),
        dev_f1=float("nan"),
        neuron_used_rate=1.0,
    )), base_ref

