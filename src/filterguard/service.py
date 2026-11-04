from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from filterguard.data import labels_texts, load_rows
from filterguard.keyword import (
    KeywordRule,
    compile_rules,
    keyword_score_with_hits,
    load_rules_config,
    load_wordlist,
    load_wordlist_manifest,
)
from filterguard.metric import pick_threshold, predict_from_scores
from filterguard.neuron import ActivationExtractor, fit_neuron_params, load_tokenizer_model, scores_from_params
from filterguard.token_anomaly import load_causal_lm, score_texts_onion_delta_with_details


@dataclass(frozen=True)
class DetectionResult:
    method: str
    score: float
    prediction: int
    is_abnormal: bool
    threshold: float
    direction: str
    details: dict[str, Any]


class Analyzer(Protocol):
    method: str

    def predict(self, texts: list[str]) -> list[DetectionResult]:
        ...

    def close(self) -> None:
        ...


def _prediction_from_score(score: float, *, threshold: float, direction: str) -> int:
    return int(predict_from_scores([float(score)], threshold=float(threshold), direction=str(direction))[0])


def _resolve_threshold(
    *,
    threshold: float | None,
    direction: str,
    scores: list[float],
    labels: list[int],
) -> tuple[float, str, float]:
    if threshold is None:
        pick = pick_threshold(scores, labels, direction=direction)
        return float(pick.threshold), str(pick.direction), float(pick.acc)
    resolved_direction = "high" if direction == "auto" else str(direction)
    if resolved_direction not in {"high", "low"}:
        raise ValueError("direction must be one of: auto, high, low")
    return float(threshold), resolved_direction, float("nan")


def _result(
    *,
    method: str,
    score: float,
    threshold: float,
    direction: str,
    details: dict[str, Any],
) -> DetectionResult:
    pred = _prediction_from_score(score, threshold=threshold, direction=direction)
    return DetectionResult(
        method=method,
        score=float(score),
        prediction=int(pred),
        is_abnormal=bool(pred),
        threshold=float(threshold),
        direction=str(direction),
        details=details,
    )


class KeywordAnalyzer:
    method = "keyword"

    def __init__(self, *, rules: list[KeywordRule], threshold: int = 1) -> None:
        self._compiled = compile_rules(rules)
        self.threshold = int(threshold)

    def predict(self, texts: list[str]) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        for text in texts:
            score, hits = keyword_score_with_hits(str(text), self._compiled)
            results.append(
                _result(
                    method=self.method,
                    score=float(score),
                    threshold=float(self.threshold),
                    direction="high",
                    details={"hits": hits, "hit_count": len(hits)},
                )
            )
        return results

    def close(self) -> None:
        return None


def load_keyword_analyzer(
    *,
    rules_config: str,
    no_base_rules: bool = False,
    extra_wordlists: list[str] | None = None,
    extra_wordlist_manifests: list[str] | None = None,
    extra_match: str = "auto",
    extra_weight: int = 1,
    threshold: int = 1,
) -> KeywordAnalyzer:
    rules: list[KeywordRule] = []
    if not no_base_rules:
        rules.extend(load_rules_config(Path(rules_config)))
    for path in extra_wordlists or []:
        rules.extend(load_wordlist(Path(path), weight=int(extra_weight), match=str(extra_match)))
    for path in extra_wordlist_manifests or []:
        rules.extend(load_wordlist_manifest(Path(path)))
    if not rules:
        raise ValueError("keyword analyzer requires at least one rule")
    return KeywordAnalyzer(rules=rules, threshold=int(threshold))


class OnionAnalyzer:
    method = "onion"

    def __init__(
        self,
        *,
        torch: Any,
        device: Any,
        tokenizer: Any,
        model: Any,
        threshold: float,
        direction: str,
        batch_size: int,
        deletion_batch_size: int,
        max_length: int,
        max_candidates: int,
        score_mode: str,
        top_k: int,
        delta_space: str,
        eps: float,
        max_pending_deletions: int,
        model_info: Any,
        dev_acc: float = float("nan"),
    ) -> None:
        self._torch = torch
        self._device = device
        self._tokenizer = tokenizer
        self._model = model
        self.threshold = float(threshold)
        self.direction = str(direction)
        self.batch_size = int(batch_size)
        self.deletion_batch_size = int(deletion_batch_size)
        self.max_length = int(max_length)
        self.max_candidates = int(max_candidates)
        self.score_mode = str(score_mode)
        self.top_k = int(top_k)
        self.delta_space = str(delta_space)
        self.eps = float(eps)
        self.max_pending_deletions = int(max_pending_deletions)
        self.model_info = model_info
        self.dev_acc = float(dev_acc)
        self._lock = Lock()

    def _score(self, texts: list[str]) -> tuple[list[float], list[Any]]:
        return score_texts_onion_delta_with_details(
            torch=self._torch,
            device=self._device,
            tokenizer=self._tokenizer,
            model=self._model,
            texts=texts,
            batch_size=self.batch_size,
            deletion_batch_size=self.deletion_batch_size,
            max_length=self.max_length,
            max_candidates=self.max_candidates,
            score_mode=self.score_mode,
            top_k=self.top_k,
            delta_space=self.delta_space,
            eps=self.eps,
            max_pending_deletions=self.max_pending_deletions,
        )

    def predict(self, texts: list[str]) -> list[DetectionResult]:
        with self._lock:
            scores, details = self._score(texts)
        return [
            _result(
                method=self.method,
                score=float(score),
                threshold=self.threshold,
                direction=self.direction,
                details=asdict(detail),
            )
            for score, detail in zip(scores, details, strict=True)
        ]

    def close(self) -> None:
        return None


def load_onion_analyzer(
    *,
    model_id: str,
    trust_remote_code: bool,
    device: str,
    dtype: str,
    threshold: float | None,
    direction: str,
    calibration_dataset: str,
    include_assistant: bool,
    batch_size: int,
    deletion_batch_size: int,
    max_length: int,
    max_candidates: int,
    score_mode: str,
    top_k: int,
    delta_space: str,
    eps: float,
    max_pending_deletions: int,
) -> OnionAnalyzer:
    torch, resolved_device, tokenizer, model, model_info = load_causal_lm(
        model_id=str(model_id),
        trust_remote_code=bool(trust_remote_code),
        device_str=str(device),
        dtype_str=str(dtype),
    )
    dev_acc = float("nan")
    if threshold is None:
        if not calibration_dataset:
            raise ValueError("onion requires --onion-threshold or --onion-calibration-dataset")
        rows = load_rows(str(calibration_dataset))
        labels, texts = labels_texts(rows, include_assistant=bool(include_assistant))
        calibration_scores, _details = score_texts_onion_delta_with_details(
            torch=torch,
            device=resolved_device,
            tokenizer=tokenizer,
            model=model,
            texts=texts,
            batch_size=int(batch_size),
            deletion_batch_size=int(deletion_batch_size),
            max_length=int(max_length),
            max_candidates=int(max_candidates),
            score_mode=str(score_mode),
            top_k=int(top_k),
            delta_space=str(delta_space),
            eps=float(eps),
            max_pending_deletions=int(max_pending_deletions),
        )
        threshold, direction, dev_acc = _resolve_threshold(
            threshold=None,
            direction=str(direction),
            scores=calibration_scores,
            labels=labels,
        )
    else:
        threshold, direction, dev_acc = _resolve_threshold(
            threshold=float(threshold),
            direction=str(direction),
            scores=[],
            labels=[],
        )
    return OnionAnalyzer(
        torch=torch,
        device=resolved_device,
        tokenizer=tokenizer,
        model=model,
        threshold=float(threshold),
        direction=str(direction),
        batch_size=int(batch_size),
        deletion_batch_size=int(deletion_batch_size),
        max_length=int(max_length),
        max_candidates=int(max_candidates),
        score_mode=str(score_mode),
        top_k=int(top_k),
        delta_space=str(delta_space),
        eps=float(eps),
        max_pending_deletions=int(max_pending_deletions),
        model_info=model_info,
        dev_acc=dev_acc,
    )


class NeuronPersonaAnalyzer:
    method = "neuron"

    def __init__(
        self,
        *,
        torch: Any,
        device: Any,
        tokenizer: Any,
        model: Any,
        params: Any,
        threshold: float,
        direction: str,
        layer: int,
        pooling: str,
        eps: float,
        batch_size: int,
        max_length: int,
        stop_after_layer: bool,
        model_info: dict[str, Any],
        layer_resolved: int,
        n_layers: int,
        dev_acc: float = float("nan"),
    ) -> None:
        self._torch = torch
        self._device = device
        self._tokenizer = tokenizer
        self._model = model
        self._params = params
        self.threshold = float(threshold)
        self.direction = str(direction)
        self.layer = int(layer)
        self.pooling = str(pooling)
        self.eps = float(eps)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.stop_after_layer = bool(stop_after_layer)
        self.model_info = dict(model_info)
        self.layer_resolved = int(layer_resolved)
        self.n_layers = int(n_layers)
        self.dev_acc = float(dev_acc)
        self._lock = Lock()

    def _score(self, texts: list[str]) -> list[float]:
        with ActivationExtractor(
            torch=self._torch,
            device=self._device,
            tokenizer=self._tokenizer,
            model=self._model,
            layer=self.layer,
            pooling=self.pooling,
            eps=self.eps,
            stop_after_layer=self.stop_after_layer,
        ) as extractor:
            features = extractor.extract(texts, batch_size=self.batch_size, max_length=self.max_length)
        scores_t = scores_from_params(features, self._params)
        return [float(x) for x in scores_t.tolist()]

    def predict(self, texts: list[str]) -> list[DetectionResult]:
        with self._lock:
            scores = self._score(texts)
        return [
            _result(
                method=self.method,
                score=float(score),
                threshold=self.threshold,
                direction=self.direction,
                details={
                    "scoring": "persona_vector",
                    "layer": self.layer,
                    "layer_resolved": self.layer_resolved,
                    "n_layers": self.n_layers,
                    "pooling": self.pooling,
                },
            )
            for score in scores
        ]

    def close(self) -> None:
        return None


def load_neuron_persona_analyzer(
    *,
    calibration_dataset: str,
    model_id: str,
    trust_remote_code: bool,
    include_assistant: bool,
    seed: int,
    dev_ratio: float,
    pooling: str,
    layer: int,
    feature_normalize: str,
    eps: float,
    direction: str,
    threshold: float | None,
    batch_size: int,
    max_length: int,
    device: str,
    dtype: str,
    backbone: str,
    stop_after_layer: bool,
) -> NeuronPersonaAnalyzer:
    if not calibration_dataset:
        raise ValueError("neuron persona_vector requires --neuron-calibration-dataset")

    from filterguard.data import resolve_dev_eval_indices

    rows = load_rows(str(calibration_dataset))
    labels, texts = labels_texts(rows, include_assistant=bool(include_assistant))
    torch, resolved_device, tokenizer, model, model_info = load_tokenizer_model(
        model_id=str(model_id),
        trust_remote_code=bool(trust_remote_code),
        device_str=str(device),
        dtype_str=str(dtype),
        backbone=str(backbone),
    )
    dev_idx, _eval_idx, _split_mode = resolve_dev_eval_indices(
        rows,
        labels,
        dev_ratio=float(dev_ratio),
        seed=int(seed),
    )
    with ActivationExtractor(
        torch=torch,
        device=resolved_device,
        tokenizer=tokenizer,
        model=model,
        layer=int(layer),
        pooling=str(pooling),
        eps=float(eps),
        stop_after_layer=bool(stop_after_layer),
    ) as extractor:
        features_all = extractor.extract(texts, batch_size=int(batch_size), max_length=int(max_length))
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
        method="persona_vector",
        feature_normalize=str(feature_normalize),
        eps=float(eps),
    )
    dev_scores_t = scores_from_params(features_dev, params)
    dev_scores = [float(x) for x in dev_scores_t.tolist()]
    threshold, direction, dev_acc = _resolve_threshold(
        threshold=threshold,
        direction=str(direction),
        scores=dev_scores,
        labels=labels_dev,
    )
    return NeuronPersonaAnalyzer(
        torch=torch,
        device=resolved_device,
        tokenizer=tokenizer,
        model=model,
        params=params,
        threshold=float(threshold),
        direction=str(direction),
        layer=int(layer),
        pooling=str(pooling),
        eps=float(eps),
        batch_size=int(batch_size),
        max_length=int(max_length),
        stop_after_layer=bool(stop_after_layer),
        model_info=model_info,
        layer_resolved=int(layer_resolved),
        n_layers=int(n_layers),
        dev_acc=dev_acc,
    )


def result_to_dict(result: DetectionResult) -> dict[str, Any]:
    payload = asdict(result)
    if math.isnan(float(payload.get("threshold", 0.0))):
        payload["threshold"] = None
    return payload
