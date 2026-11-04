from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from filterguard.neuron import move_batch_to_device, sorted_text_batches
from filterguard.torchutil import resolve_torch_dtype


@dataclass(frozen=True)
class TokenModelInfo:
    model_id: str
    device: str
    dtype: str
    pad_token_id: int


@dataclass(frozen=True)
class TokenScoreDetail:
    token_count: int
    candidate_count: int = 0
    deletion_count: int = 0
    base_mean_nll: float | None = None
    mean_nll: float | None = None
    std_nll: float | None = None
    min_nll: float | None = None
    max_nll: float | None = None
    topk_mean_nll: float | None = None
    topk_gap_nll: float | None = None
    max_z: float | None = None
    max_z_len_norm: float | None = None


def validate_token_anomaly_args(
    *,
    batch_size: int,
    max_length: int,
    score_mode: str,
    top_k: int,
    eps: float,
    method: str,
    deletion_batch_size: int | None = None,
    max_candidates: int | None = None,
    max_pending_deletions: int | None = None,
    delta_space: str | None = None,
) -> None:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(max_length) < 2:
        raise ValueError("max_length must be at least 2 so next-token NLL is defined")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")
    if method == "nll_spike":
        if score_mode not in {
            "token_count",
            "max_z",
            "max_z_len_norm",
            "max_nll",
            "mean_nll",
            "std_nll",
            "topk_mean_nll",
            "topk_gap_nll",
        }:
            raise ValueError(f"unsupported nll spike score_mode: {score_mode!r}")
        return
    if method == "onion_delta":
        if score_mode not in {"max_delta", "topk_mean_delta"}:
            raise ValueError(f"unsupported onion score_mode: {score_mode!r}")
        if deletion_batch_size is None or int(deletion_batch_size) <= 0:
            raise ValueError("deletion_batch_size must be positive")
        if max_candidates is None or int(max_candidates) < 0:
            raise ValueError("max_candidates must be non-negative")
        if max_pending_deletions is None or int(max_pending_deletions) <= 0:
            raise ValueError("max_pending_deletions must be positive")
        if delta_space not in {"log_ppl", "ppl"}:
            raise ValueError("delta_space must be one of: log_ppl, ppl")
        return
    raise ValueError(f"unsupported token anomaly method: {method!r}")


def load_causal_lm(
    *,
    model_id: str,
    trust_remote_code: bool,
    device_str: str,
    dtype_str: str,
) -> tuple[Any, Any, Any, Any, TokenModelInfo]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device_str == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device_str)
    dtype = resolve_torch_dtype(torch=torch, device=device, dtype=dtype_str, arg_name="--dtype")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError(f"tokenizer for {model_id!r} has no pad_token_id or eos_token_id")

    from_pretrained_kwargs = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **from_pretrained_kwargs)
    except TypeError:
        from_pretrained_kwargs.pop("dtype", None)
        from_pretrained_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, **from_pretrained_kwargs)
    if getattr(getattr(model, "config", None), "use_cache", None) is not None:
        model.config.use_cache = False
    model.eval()
    model.to(device)

    info = TokenModelInfo(
        model_id=str(model_id),
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        pad_token_id=int(tokenizer.pad_token_id),
    )
    return torch, device, tokenizer, model, info


def _next_token_nll(torch: Any, logits: Any, input_ids: Any, attention_mask: Any) -> tuple[Any, Any]:
    if input_ids.shape[1] < 2:
        empty = torch.empty((input_ids.shape[0], 0), device=input_ids.device, dtype=torch.float32)
        return empty, torch.zeros((input_ids.shape[0], 0), device=input_ids.device, dtype=torch.bool)

    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].to(dtype=torch.bool).contiguous()
    flat_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
    )
    nll = flat_loss.view_as(shift_labels).float()
    return nll, shift_mask


def _nll_distribution_features(values: list[float], *, top_k: int, eps: float) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {
            "token_count": 0.0,
            "mean_nll": 0.0,
            "std_nll": 0.0,
            "min_nll": 0.0,
            "max_nll": 0.0,
            "topk_mean_nll": 0.0,
            "topk_gap_nll": 0.0,
            "max_z": 0.0,
            "max_z_len_norm": 0.0,
        }

    mean = sum(values) / n
    var = sum((x - mean) * (x - mean) for x in values) / n
    std = math.sqrt(var)
    min_nll = min(values)
    max_nll = max(values)
    k = min(max(int(top_k), 1), n)
    topk_mean = sum(sorted(values, reverse=True)[:k]) / k

    if n < 2:
        max_z = 0.0
        max_z_len_norm = 0.0
    else:
        max_z = (max_nll - mean) / max(std, float(eps))
        # The expected maximum z-score grows with sequence length. This keeps
        # long FineWeb documents from looking anomalous solely because they have
        # more chances to contain one high-NLL token.
        max_z_len_norm = max_z / max(math.sqrt(2.0 * math.log(n)), float(eps))

    return {
        "token_count": float(n),
        "mean_nll": float(mean),
        "std_nll": float(std),
        "min_nll": float(min_nll),
        "max_nll": float(max_nll),
        "topk_mean_nll": float(topk_mean),
        "topk_gap_nll": float(topk_mean - mean),
        "max_z": float(max_z),
        "max_z_len_norm": float(max_z_len_norm),
    }


def _score_nll_spike_from_features(features: dict[str, float], *, score_mode: str) -> float:
    if score_mode == "token_count":
        return float(features["token_count"])
    if score_mode == "max_nll":
        return float(features["max_nll"])
    if score_mode == "mean_nll":
        return float(features["mean_nll"])
    if score_mode == "std_nll":
        return float(features["std_nll"])
    if score_mode == "topk_mean_nll":
        return float(features["topk_mean_nll"])
    if score_mode == "topk_gap_nll":
        return float(features["topk_gap_nll"])
    if score_mode == "max_z":
        return float(features["max_z"])
    if score_mode == "max_z_len_norm":
        return float(features["max_z_len_norm"])
    raise ValueError(f"unsupported nll spike score_mode: {score_mode!r}")


def _token_detail_from_features(
    *,
    token_count: int,
    features: dict[str, float],
    candidate_count: int = 0,
    deletion_count: int = 0,
    base_mean_nll: float | None = None,
) -> TokenScoreDetail:
    return TokenScoreDetail(
        token_count=int(token_count),
        candidate_count=int(candidate_count),
        deletion_count=int(deletion_count),
        base_mean_nll=base_mean_nll,
        mean_nll=float(features["mean_nll"]),
        std_nll=float(features["std_nll"]),
        min_nll=float(features["min_nll"]),
        max_nll=float(features["max_nll"]),
        topk_mean_nll=float(features["topk_mean_nll"]),
        topk_gap_nll=float(features["topk_gap_nll"]),
        max_z=float(features["max_z"]),
        max_z_len_norm=float(features["max_z_len_norm"]),
    )


def score_texts_nll_spike(
    *,
    torch: Any,
    device: Any,
    tokenizer: Any,
    model: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
    score_mode: str,
    top_k: int,
    eps: float,
) -> list[float]:
    scores, _details = score_texts_nll_spike_with_details(
        torch=torch,
        device=device,
        tokenizer=tokenizer,
        model=model,
        texts=texts,
        batch_size=batch_size,
        max_length=max_length,
        score_mode=score_mode,
        top_k=top_k,
        eps=eps,
    )
    return scores


def score_texts_nll_spike_with_details(
    *,
    torch: Any,
    device: Any,
    tokenizer: Any,
    model: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
    score_mode: str,
    top_k: int,
    eps: float,
) -> tuple[list[float], list[TokenScoreDetail]]:
    validate_token_anomaly_args(
        batch_size=int(batch_size),
        max_length=int(max_length),
        score_mode=str(score_mode),
        top_k=int(top_k),
        eps=float(eps),
        method="nll_spike",
    )
    scores = [0.0] * len(texts)
    details = [TokenScoreDetail(token_count=0)] * len(texts)
    with torch.inference_mode():
        for idx_batch, batch in sorted_text_batches(texts, int(batch_size)):
            tok = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_length),
            )
            tok = move_batch_to_device(tok, device=device)
            out = model(**tok, use_cache=False)
            nll, mask = _next_token_nll(torch, out.logits, tok["input_ids"], tok["attention_mask"])
            for row, original_idx in enumerate(idx_batch):
                valid = nll[row][mask[row]]
                values = [float(x) for x in valid.detach().float().cpu().tolist()]
                features = _nll_distribution_features(values, top_k=int(top_k), eps=float(eps))
                scores[original_idx] = _score_nll_spike_from_features(features, score_mode=str(score_mode))
                details[original_idx] = _token_detail_from_features(token_count=len(values), features=features)
    return scores, details


def _pad_id_batches(torch: Any, sequences: list[list[int]], *, pad_token_id: int, device: Any) -> dict[str, Any]:
    max_len = max((len(seq) for seq in sequences), default=0)
    max_len = max(max_len, 1)
    input_ids = torch.full((len(sequences), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, seq in enumerate(sequences):
        if not seq:
            continue
        ids = torch.tensor(seq, dtype=torch.long)
        input_ids[row, : ids.numel()] = ids
        attention_mask[row, : ids.numel()] = 1
    return move_batch_to_device({"input_ids": input_ids, "attention_mask": attention_mask}, device=device)


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), int(batch_size)):
        yield items[start : start + int(batch_size)]


def _sequence_mean_nll(
    *,
    torch: Any,
    device: Any,
    model: Any,
    sequences: list[list[int]],
    pad_token_id: int,
    batch_size: int,
) -> list[float]:
    means: list[float] = []
    with torch.inference_mode():
        for seq_batch in _batched(sequences, int(batch_size)):
            tok = _pad_id_batches(torch, seq_batch, pad_token_id=int(pad_token_id), device=device)
            out = model(**tok, use_cache=False)
            nll, mask = _next_token_nll(torch, out.logits, tok["input_ids"], tok["attention_mask"])
            denom = mask.sum(dim=1).clamp_min(1).float()
            summed = (nll * mask.float()).sum(dim=1)
            batch_means = (summed / denom).detach().float().cpu().tolist()
            for seq, mean in zip(seq_batch, batch_means, strict=True):
                means.append(float(mean) if len(seq) >= 2 else 0.0)
    return means


def _sequence_mean_nll_with_token_nll(
    *,
    torch: Any,
    device: Any,
    model: Any,
    sequences: list[list[int]],
    pad_token_id: int,
    batch_size: int,
) -> tuple[list[float], list[list[float | None]]]:
    means: list[float] = []
    token_nlls: list[list[float | None]] = []
    with torch.inference_mode():
        for seq_batch in _batched(sequences, int(batch_size)):
            tok = _pad_id_batches(torch, seq_batch, pad_token_id=int(pad_token_id), device=device)
            out = model(**tok, use_cache=False)
            nll, mask = _next_token_nll(torch, out.logits, tok["input_ids"], tok["attention_mask"])
            denom = mask.sum(dim=1).clamp_min(1).float()
            summed = (nll * mask.float()).sum(dim=1)
            batch_means = (summed / denom).detach().float().cpu().tolist()
            batch_nll = nll.detach().float().cpu().tolist()
            batch_mask = mask.detach().cpu().tolist()
            for row, (seq, mean) in enumerate(zip(seq_batch, batch_means, strict=True)):
                means.append(float(mean) if len(seq) >= 2 else 0.0)
                per_token: list[float | None] = [None] * len(seq)
                for pred_idx, keep in enumerate(batch_mask[row]):
                    pos = pred_idx + 1
                    if pos >= len(seq):
                        break
                    if keep:
                        per_token[pos] = float(batch_nll[row][pred_idx])
                token_nlls.append(per_token)
    return means, token_nlls


def _token_ids_for_text(tokenizer: Any, text: str, *, max_length: int) -> list[int]:
    tok = tokenizer(
        text,
        return_tensors=None,
        padding=False,
        truncation=True,
        max_length=int(max_length),
    )
    return [int(x) for x in tok["input_ids"]]


def _candidate_positions_from_token_nll(
    *,
    sequence: list[int],
    token_nll: list[float | None],
    special_token_ids: set[int],
    max_candidates: int,
    eps: float,
) -> list[int]:
    if len(sequence) < 2:
        return []
    valid_positions: list[tuple[float, int]] = []
    for pos in range(1, len(sequence)):
        if pos >= len(token_nll):
            continue
        score = token_nll[pos]
        if score is None:
            continue
        token_id = int(sequence[pos])
        if token_id in special_token_ids:
            continue
        valid_positions.append((float(score), pos))

    # The first token has no next-token NLL of its own. Keep it as a low-priority
    # candidate so pure prefix triggers can still be checked when all positions are used.
    if sequence and int(sequence[0]) not in special_token_ids:
        valid_positions.append((-float(eps), 0))

    valid_positions.sort(key=lambda x: x[0], reverse=True)
    if int(max_candidates) > 0:
        valid_positions = valid_positions[: int(max_candidates)]
    return [pos for _score, pos in valid_positions]


def _aggregate_deltas(deltas: list[float], *, score_mode: str, top_k: int) -> float:
    if not deltas:
        return 0.0
    if score_mode == "max_delta":
        return max(deltas)
    if score_mode == "topk_mean_delta":
        k = min(max(int(top_k), 1), len(deltas))
        return sum(sorted(deltas, reverse=True)[:k]) / k
    raise ValueError(f"unsupported onion score_mode: {score_mode!r}")


def score_texts_onion_delta(
    *,
    torch: Any,
    device: Any,
    tokenizer: Any,
    model: Any,
    texts: list[str],
    batch_size: int,
    deletion_batch_size: int,
    max_length: int,
    max_candidates: int,
    score_mode: str,
    top_k: int,
    delta_space: str,
    eps: float,
    max_pending_deletions: int = 256,
) -> list[float]:
    scores, _details = score_texts_onion_delta_with_details(
        torch=torch,
        device=device,
        tokenizer=tokenizer,
        model=model,
        texts=texts,
        batch_size=batch_size,
        deletion_batch_size=deletion_batch_size,
        max_length=max_length,
        max_candidates=max_candidates,
        score_mode=score_mode,
        top_k=top_k,
        delta_space=delta_space,
        eps=eps,
        max_pending_deletions=max_pending_deletions,
    )
    return scores


def score_texts_onion_delta_with_details(
    *,
    torch: Any,
    device: Any,
    tokenizer: Any,
    model: Any,
    texts: list[str],
    batch_size: int,
    deletion_batch_size: int,
    max_length: int,
    max_candidates: int,
    score_mode: str,
    top_k: int,
    delta_space: str,
    eps: float,
    max_pending_deletions: int = 256,
) -> tuple[list[float], list[TokenScoreDetail]]:
    validate_token_anomaly_args(
        batch_size=int(batch_size),
        max_length=int(max_length),
        score_mode=str(score_mode),
        top_k=int(top_k),
        eps=float(eps),
        method="onion_delta",
        deletion_batch_size=int(deletion_batch_size),
        max_candidates=int(max_candidates),
        max_pending_deletions=int(max_pending_deletions),
        delta_space=str(delta_space),
    )
    special_token_ids = {int(x) for x in getattr(tokenizer, "all_special_ids", [])}
    sequences = [_token_ids_for_text(tokenizer, text, max_length=int(max_length)) for text in texts]
    base_mean_nll, token_nlls = _sequence_mean_nll_with_token_nll(
        torch=torch,
        device=device,
        model=model,
        sequences=sequences,
        pad_token_id=int(tokenizer.pad_token_id),
        batch_size=int(batch_size),
    )

    deltas_by_sample: list[list[float]] = [[] for _ in sequences]
    candidate_counts = [0] * len(sequences)
    deletion_counts = [0] * len(sequences)
    pending_sequences: list[list[int]] = []
    pending_owners: list[int] = []
    flush_size = max(int(deletion_batch_size), int(max_pending_deletions))

    def flush_pending() -> None:
        if not pending_sequences:
            return
        deleted_means = _sequence_mean_nll(
            torch=torch,
            device=device,
            model=model,
            sequences=pending_sequences,
            pad_token_id=int(tokenizer.pad_token_id),
            batch_size=int(deletion_batch_size),
        )
        for owner, deleted in zip(pending_owners, deleted_means, strict=True):
            base = float(base_mean_nll[owner])
            if delta_space == "log_ppl":
                delta = base - float(deleted)
            elif delta_space == "ppl":
                delta = math.exp(base) - math.exp(float(deleted))
            else:
                raise ValueError("--onion-delta-space must be one of: log_ppl, ppl")
            deltas_by_sample[owner].append(float(delta))
            deletion_counts[owner] += 1
        pending_sequences.clear()
        pending_owners.clear()

    for sample_idx, (seq, token_nll) in enumerate(zip(sequences, token_nlls, strict=True)):
        candidates = _candidate_positions_from_token_nll(
            sequence=seq,
            token_nll=token_nll,
            special_token_ids=special_token_ids,
            max_candidates=int(max_candidates),
            eps=float(eps),
        )
        candidate_counts[sample_idx] = len(candidates)
        for pos in candidates:
            if len(seq) <= 1:
                continue
            pending_sequences.append(seq[:pos] + seq[pos + 1 :])
            pending_owners.append(sample_idx)
            if len(pending_sequences) >= flush_size:
                flush_pending()
    flush_pending()

    scores = [_aggregate_deltas(deltas, score_mode=str(score_mode), top_k=int(top_k)) for deltas in deltas_by_sample]
    details = [
        _token_detail_from_features(
            token_count=len([x for x in token_nlls[i] if x is not None]),
            features=_nll_distribution_features(
                [float(x) for x in token_nlls[i] if x is not None],
                top_k=int(top_k),
                eps=float(eps),
            ),
            candidate_count=int(candidate_counts[i]),
            deletion_count=int(deletion_counts[i]),
            base_mean_nll=float(base_mean_nll[i]),
        )
        for i, _seq in enumerate(sequences)
    ]
    return scores, details
