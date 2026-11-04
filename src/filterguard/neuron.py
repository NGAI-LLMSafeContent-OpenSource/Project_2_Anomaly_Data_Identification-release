from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from filterguard.qwen3_truncated import make_filterguard_qwen3_model_classes
from filterguard.qwen3_5_truncated import make_filterguard_qwen35_model_classes
from filterguard.torchutil import resolve_torch_dtype
from filterguard.util import batched


def get_transformer_layers(model: Any) -> Any:
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers") and len(model.language_model.layers) > 0:
        return model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model") and len(model.model.language_model.layers) > 0:
        return model.model.language_model.layers
    if hasattr(model, "layers") and len(model.layers) > 0:
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        return model.model.layers
    raise ValueError("FilterGuard neuron extraction currently supports only Qwen3/Qwen3.5 models.")


def pool_hidden(h: Any, attention_mask: Any, *, pooling: str, eps: float) -> Any:
    import torch

    if pooling == "mean":
        m = attention_mask.to(dtype=h.dtype).unsqueeze(-1)  # (b,s,1)
        denom = m.sum(dim=1).clamp_min(float(eps))
        return (h * m).sum(dim=1) / denom
    if pooling == "last":
        lengths = attention_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
        bsz, _seq, _dim = h.shape
        rows = torch.arange(bsz, device=h.device)
        return h[rows, lengths, :]
    raise ValueError(f"unsupported pooling: {pooling!r}")


def move_batch_to_device(batch: dict[str, Any], *, device: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        tensor = value
        if getattr(device, "type", None) == "cuda" and getattr(tensor, "device", None) is not None:
            if tensor.device.type == "cpu" and hasattr(tensor, "pin_memory") and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            moved[key] = tensor.to(device, non_blocking=True)
        else:
            moved[key] = tensor.to(device)
    return moved


def sorted_text_batches(texts: list[str], batch_size: int):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    for idx_batch in batched(order, int(batch_size)):
        yield idx_batch, [texts[i] for i in idx_batch]


def supports_layerwise_forward(model: Any) -> bool:
    return bool(getattr(model, "supports_filterguard_layerwise", False)) and callable(getattr(model, "forward_until", None))


def resolve_attention_implementation(*, model_type: str, device: Any) -> str | None:
    if getattr(device, "type", None) != "cuda":
        return None
    if model_type != "qwen3":
        return None
    try:
        from transformers.utils.import_utils import is_flash_attn_2_available
    except Exception:  # noqa: BLE001
        return None
    return "flash_attention_2" if is_flash_attn_2_available() else None


def load_tokenizer_model(
    *,
    model_id: str,
    trust_remote_code: bool,
    device_str: str,
    dtype_str: str,
    backbone: str,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    if device_str == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device_str)
    dtype = resolve_torch_dtype(torch=torch, device=device, dtype=dtype_str, arg_name="--dtype")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if backbone not in {"base", "causal_lm"}:
        raise ValueError("--backbone must be one of: base, causal_lm")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_type = str(getattr(config, "model_type", ""))
    attn_implementation = resolve_attention_implementation(model_type=model_type, device=device)
    if model_type == "qwen3":
        FilterGuardQwen3Model, FilterGuardQwen3ForCausalLM = make_filterguard_qwen3_model_classes()
        model_cls = FilterGuardQwen3Model if backbone == "base" else FilterGuardQwen3ForCausalLM
    elif model_type == "qwen3_5":
        FilterGuardQwen35Model, FilterGuardQwen35ForCausalLM, _FilterGuardQwen35ForConditionalGeneration = (
            make_filterguard_qwen35_model_classes()
        )
        model_cls = FilterGuardQwen35Model if backbone == "base" else FilterGuardQwen35ForCausalLM
    else:
        raise ValueError(f"FilterGuard neuron extraction currently supports only Qwen3/Qwen3.5; got {model_type!r}")
    from_pretrained_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation is not None:
        from_pretrained_kwargs["attn_implementation"] = attn_implementation
    model = model_cls.from_pretrained(model_id, **from_pretrained_kwargs)
    if getattr(getattr(model, "config", None), "use_cache", None) is not None:
        model.config.use_cache = False
    model.eval()
    model.to(device)

    info = {
        "model_id": str(model_id),
        "backbone": str(backbone),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "model_type": model_type,
        "attn_implementation": attn_implementation or str(getattr(getattr(model, "config", None), "_attn_implementation", "")),
    }
    return torch, device, tokenizer, model, info


class ActivationExtractor:
    def __init__(
        self,
        *,
        torch: Any,
        device: Any,
        tokenizer: Any,
        model: Any,
        layer: int,
        pooling: str,
        eps: float,
        stop_after_layer: bool = True,
    ) -> None:
        self._torch = torch
        self._device = device
        self._tokenizer = tokenizer
        self._model = model
        self._pooling = str(pooling)
        self._eps = float(eps)
        if not supports_layerwise_forward(model):
            raise ValueError("ActivationExtractor requires a FilterGuard Qwen3/Qwen3.5 model with `forward_until`.")

        layers = get_transformer_layers(model)
        n_layers = len(layers)
        layer_idx = int(layer)
        if layer_idx < 0:
            layer_idx = n_layers + layer_idx
        if not 0 <= layer_idx < n_layers:
            raise ValueError(f"--layer out of range: {layer} (resolved {layer_idx}) with n_layers={n_layers}")

        self.layer_idx = int(layer_idx)
        self.n_layers = int(n_layers)
        self._stop_after_layer = bool(stop_after_layer)

    def close(self) -> None:
        return None

    def __enter__(self) -> "ActivationExtractor":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def iter_extract(self, texts: list[str], *, batch_size: int, max_length: int):
        with self._torch.inference_mode():
            for batch in batched(texts, int(batch_size)):
                tok = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(max_length),
                )
                tok = move_batch_to_device(tok, device=self._device)
                out = self._model.forward_until(
                    **tok,
                    use_cache=False,
                    stop_after_layer=self.layer_idx if self._stop_after_layer else None,
                    capture_layers=[self.layer_idx],
                )
                h = out.captured_hidden_states.get(self.layer_idx)
                if h is None:
                    raise RuntimeError("Failed to capture activations at the requested layer.")
                feat = pool_hidden(h, tok["attention_mask"], pooling=self._pooling, eps=self._eps)
                yield feat.detach().float()

    def extract(self, texts: list[str], *, batch_size: int, max_length: int) -> Any:
        features_all = None
        with self._torch.inference_mode():
            for idx_batch, batch in sorted_text_batches(texts, int(batch_size)):
                tok = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(max_length),
                )
                tok = move_batch_to_device(tok, device=self._device)
                out = self._model.forward_until(
                    **tok,
                    use_cache=False,
                    stop_after_layer=self.layer_idx if self._stop_after_layer else None,
                    capture_layers=[self.layer_idx],
                )
                h = out.captured_hidden_states.get(self.layer_idx)
                if h is None:
                    raise RuntimeError("Failed to capture activations at the requested layer.")
                feat = pool_hidden(h, tok["attention_mask"], pooling=self._pooling, eps=self._eps).detach().float()
                if features_all is None:
                    features_all = self._torch.empty((len(texts), feat.shape[1]), device=feat.device, dtype=feat.dtype)
                batch_idx_t = self._torch.tensor(idx_batch, device=feat.device, dtype=self._torch.long)
                features_all[batch_idx_t] = feat
        return features_all if features_all is not None else self._torch.empty((0, 0))


class MultiLayerActivationExtractor:
    def __init__(
        self,
        *,
        torch: Any,
        device: Any,
        tokenizer: Any,
        model: Any,
        layers: list[int],
        pooling: str,
        eps: float,
        stop_at_max_layer: bool = True,
    ) -> None:
        self._torch = torch
        self._device = device
        self._tokenizer = tokenizer
        self._model = model
        self._pooling = str(pooling)
        self._eps = float(eps)
        if not supports_layerwise_forward(model):
            raise ValueError("MultiLayerActivationExtractor requires a FilterGuard Qwen3/Qwen3.5 model with `forward_until`.")

        blocks = get_transformer_layers(model)
        self.n_layers = int(len(blocks))

        resolved: list[int] = []
        for layer in layers:
            idx = int(layer)
            if idx < 0:
                idx = self.n_layers + idx
            if not 0 <= idx < self.n_layers:
                raise ValueError(f"layer out of range: {layer} (resolved {idx}) with n_layers={self.n_layers}")
            resolved.append(idx)

        unique_sorted = sorted(set(resolved))
        if not unique_sorted:
            raise ValueError("layers must be non-empty")
        self.layer_idxs = unique_sorted
        self.max_layer = int(max(unique_sorted))
        self._stop_at_max = bool(stop_at_max_layer)

    def close(self) -> None:
        return None

    def __enter__(self) -> "MultiLayerActivationExtractor":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def iter_extract(self, texts: list[str], *, batch_size: int, max_length: int):
        with self._torch.inference_mode():
            for batch in batched(texts, int(batch_size)):
                tok = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(max_length),
                )
                tok = move_batch_to_device(tok, device=self._device)
                out = self._model.forward_until(
                    **tok,
                    use_cache=False,
                    stop_after_layer=self.max_layer if self._stop_at_max else None,
                    capture_layers=self.layer_idxs,
                )
                pooled_parts = {}
                for i in self.layer_idxs:
                    feat = pool_hidden(
                        out.captured_hidden_states[i],
                        tok["attention_mask"],
                        pooling=self._pooling,
                        eps=self._eps,
                    )
                    pooled_parts[i] = [feat.detach().float()]
                out: dict[int, Any] = {}
                for i in self.layer_idxs:
                    parts = pooled_parts.get(i) or []
                    out[i] = self._torch.cat(parts, dim=0) if parts else self._torch.empty((0, 0))
                yield out

    def extract(self, texts: list[str], *, batch_size: int, max_length: int) -> dict[int, Any]:
        outputs = None
        with self._torch.inference_mode():
            for idx_batch, batch in sorted_text_batches(texts, int(batch_size)):
                tok = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(max_length),
                )
                tok = move_batch_to_device(tok, device=self._device)
                out = self._model.forward_until(
                    **tok,
                    use_cache=False,
                    stop_after_layer=self.max_layer if self._stop_at_max else None,
                    capture_layers=self.layer_idxs,
                )
                batch_idx_t = None
                for i in self.layer_idxs:
                    feat = pool_hidden(
                        out.captured_hidden_states[i],
                        tok["attention_mask"],
                        pooling=self._pooling,
                        eps=self._eps,
                    ).detach().float()
                    if outputs is None:
                        outputs = {
                            j: self._torch.empty((len(texts), feat.shape[1]), device=feat.device, dtype=feat.dtype)
                            for j in self.layer_idxs
                        }
                    if batch_idx_t is None:
                        batch_idx_t = self._torch.tensor(idx_batch, device=feat.device, dtype=self._torch.long)
                    outputs[i][batch_idx_t] = feat
        if outputs is None:
            return {i: self._torch.empty((0, 0)) for i in self.layer_idxs}
        return outputs


@dataclass(frozen=True)
class NeuronParams:
    method: str
    feature_normalize: str
    eps: float
    centroid: Any | None = None
    mu: Any | None = None
    inv_std: Any | None = None
    v: Any | None = None


def normalize_features(f: Any, *, feature_normalize: str, eps: float) -> Any:
    import torch

    if feature_normalize == "none":
        return f
    if feature_normalize != "l2":
        raise ValueError(f"unsupported feature_normalize: {feature_normalize!r}")
    denom = torch.norm(f, dim=1, keepdim=True).clamp_min(float(eps))
    return f / denom


def fit_neuron_params(
    features_dev: Any,
    labels_dev: list[int],
    *,
    method: str,
    feature_normalize: str,
    eps: float,
) -> NeuronParams:
    import torch

    f = normalize_features(features_dev, feature_normalize=feature_normalize, eps=float(eps))
    dev_pos = [i for i, y in enumerate(labels_dev) if y == 1]
    dev_neg = [i for i, y in enumerate(labels_dev) if y == 0]

    if method == "activation_norm":
        return NeuronParams(method=method, feature_normalize=feature_normalize, eps=float(eps))
    if method == "centroid_l2":
        if not dev_neg:
            raise ValueError("centroid_l2 requires at least 1 negative example in dev split")
        centroid = f[dev_neg].mean(dim=0)
        return NeuronParams(method=method, feature_normalize=feature_normalize, eps=float(eps), centroid=centroid)
    if method == "diag_mahalanobis":
        if not dev_neg:
            raise ValueError("diag_mahalanobis requires at least 1 negative example in dev split")
        normal = f[dev_neg]
        mu = normal.mean(dim=0)
        var = normal.var(dim=0, unbiased=False)
        inv_std = (var + float(eps)).rsqrt()
        return NeuronParams(method=method, feature_normalize=feature_normalize, eps=float(eps), mu=mu, inv_std=inv_std)
    if method == "persona_vector":
        if not dev_pos or not dev_neg:
            raise ValueError("persona_vector requires both positive and negative examples in dev split")
        mu_pos = f[dev_pos].mean(dim=0)
        mu_neg = f[dev_neg].mean(dim=0)
        v = mu_pos - mu_neg
        v = v / (torch.norm(v) + float(eps))
        return NeuronParams(method=method, feature_normalize=feature_normalize, eps=float(eps), v=v)
    raise ValueError(f"unsupported method: {method!r}")


def scores_from_params(features: Any, params: NeuronParams) -> Any:
    import torch

    f = normalize_features(features, feature_normalize=params.feature_normalize, eps=float(params.eps))
    if params.method == "activation_norm":
        return torch.norm(f, dim=1)
    if params.method == "centroid_l2":
        if params.centroid is None:
            raise RuntimeError("centroid_l2 params missing centroid")
        return torch.norm(f - params.centroid, dim=1)
    if params.method == "diag_mahalanobis":
        if params.mu is None or params.inv_std is None:
            raise RuntimeError("diag_mahalanobis params missing mu/inv_std")
        z = (f - params.mu) * params.inv_std
        return z.pow(2).sum(dim=1).sqrt()
    if params.method == "persona_vector":
        if params.v is None:
            raise RuntimeError("persona_vector params missing v")
        return f @ params.v
    raise ValueError(f"unsupported method: {params.method!r}")
