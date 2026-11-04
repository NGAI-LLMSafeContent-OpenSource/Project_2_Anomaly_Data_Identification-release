from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LayerwiseForwardOutput:
    last_hidden_state: Any
    captured_hidden_states: dict[int, Any]
    past_key_values: Any | None = None
    stopped_early: bool = False
    active_indices: Any | None = None


def _resolve_layer_idx(layer: int, n_layers: int) -> int:
    idx = int(layer)
    if idx < 0:
        idx = n_layers + idx
    if not 0 <= idx < n_layers:
        raise ValueError(f"layer out of range: {layer} (resolved {idx}) with n_layers={n_layers}")
    return idx


def _resolve_layers(layers: list[int] | tuple[int, ...] | None, n_layers: int) -> list[int]:
    if not layers:
        return []
    return sorted({_resolve_layer_idx(layer, n_layers) for layer in layers})


def _index_batch_dim(x: Any, keep_indices: Any, *, batch_dim: int, batch_size: int) -> Any:
    import torch

    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _index_batch_dim(v, keep_indices, batch_dim=batch_dim, batch_size=batch_size) for k, v in x.items()}
    if torch.is_tensor(x) and x.ndim > batch_dim and x.shape[batch_dim] == batch_size:
        return x.index_select(batch_dim, keep_indices)
    return x


class FilterGuardQwen3ModelMixin:
    supports_filterguard_layerwise = True

    def forward_until(
        self,
        *,
        input_ids: Any | None = None,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: Any | None = None,
        use_cache: bool | None = None,
        cache_position: Any | None = None,
        stop_after_layer: int | None = None,
        capture_layers: list[int] | tuple[int, ...] | None = None,
        apply_final_norm: bool = False,
        **kwargs: Any,
    ) -> LayerwiseForwardOutput:
        import torch
        from transformers.models.qwen3.modeling_qwen3 import (
            DynamicCache,
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        n_layers = int(len(self.layers))
        resolved_capture_layers = _resolve_layers(capture_layers, n_layers)
        capture_set = set(resolved_capture_layers)
        resolved_stop_after = None if stop_after_layer is None else _resolve_layer_idx(stop_after_layer, n_layers)
        captured: dict[int, Any] = {}

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            if layer_idx in capture_set:
                captured[layer_idx] = hidden_states
            if resolved_stop_after is not None and layer_idx >= resolved_stop_after:
                break

        stopped_early = resolved_stop_after is not None and resolved_stop_after < (n_layers - 1)
        if apply_final_norm or not stopped_early:
            hidden_states = self.norm(hidden_states)

        return LayerwiseForwardOutput(
            last_hidden_state=hidden_states,
            captured_hidden_states=captured,
            past_key_values=past_key_values if use_cache else None,
            stopped_early=stopped_early,
            active_indices=None,
        )

    def forward_early_exit(
        self,
        *,
        input_ids: Any | None = None,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: Any | None = None,
        use_cache: bool | None = None,
        cache_position: Any | None = None,
        monitored_layers: list[int] | tuple[int, ...] | None = None,
        layer_callback: Any | None = None,
        apply_final_norm: bool = False,
        **kwargs: Any,
    ) -> LayerwiseForwardOutput:
        import torch
        from transformers.models.qwen3.modeling_qwen3 import (
            DynamicCache,
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        raw_attention_mask = attention_mask
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        current_attention_mask = raw_attention_mask
        current_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)

        n_layers = int(len(self.layers))
        monitor_set = set(_resolve_layers(monitored_layers, n_layers))
        captured: dict[int, Any] = {}

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            if layer_idx in monitor_set:
                if layer_callback is not None:
                    keep_mask = layer_callback(layer_idx, hidden_states, current_attention_mask, current_indices)
                    if keep_mask is not None:
                        keep_mask = keep_mask.to(device=hidden_states.device, dtype=torch.bool)
                        if keep_mask.ndim != 1 or keep_mask.shape[0] != hidden_states.shape[0]:
                            raise ValueError("layer_callback must return a 1D bool mask with current batch size")
                        if not bool(keep_mask.all()):
                            keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
                            batch_size = int(hidden_states.shape[0])
                            hidden_states = hidden_states.index_select(0, keep_indices)
                            current_indices = current_indices.index_select(0, keep_indices)
                            current_attention_mask = _index_batch_dim(
                                current_attention_mask, keep_indices, batch_dim=0, batch_size=batch_size
                            )
                            causal_mask_mapping = _index_batch_dim(
                                causal_mask_mapping, keep_indices, batch_dim=0, batch_size=batch_size
                            )
                            position_embeddings = self.rotary_emb(hidden_states, position_ids)
                            if hidden_states.shape[0] == 0:
                                break

        stopped_early = hidden_states.shape[0] == 0
        if hidden_states.shape[0] > 0 and (apply_final_norm or not stopped_early):
            hidden_states = self.norm(hidden_states)

        return LayerwiseForwardOutput(
            last_hidden_state=hidden_states,
            captured_hidden_states=captured,
            past_key_values=past_key_values if use_cache else None,
            stopped_early=stopped_early,
            active_indices=current_indices,
        )


def make_filterguard_qwen3_model_classes() -> tuple[type[Any], type[Any]]:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model

    class FilterGuardQwen3Model(FilterGuardQwen3ModelMixin, Qwen3Model):
        pass

    class FilterGuardQwen3ForCausalLM(Qwen3ForCausalLM):
        supports_filterguard_layerwise = True

        def __init__(self, config: Any):
            super().__init__(config)
            self.model = FilterGuardQwen3Model(config)

        def forward_until(self, **kwargs: Any) -> LayerwiseForwardOutput:
            return self.model.forward_until(**kwargs)

        def forward_early_exit(self, **kwargs: Any) -> LayerwiseForwardOutput:
            return self.model.forward_early_exit(**kwargs)

    return FilterGuardQwen3Model, FilterGuardQwen3ForCausalLM
