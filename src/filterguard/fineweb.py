from __future__ import annotations

import math
import time
from typing import Any

from filterguard.torchutil import resolve_torch_dtype
from filterguard.util import batched, sync_if_cuda


def clip_round_0_5(score: float) -> int:
    if math.isnan(score):
        return 0
    if score < 0:
        score = 0.0
    elif score > 5:
        score = 5.0
    return int(round(score))


def fineweb_scores(
    texts: list[str],
    *,
    model_id: str,
    device_str: str,
    dtype_str: str,
    batch_size: int,
    max_length: int,
    score_mode: str,
    trust_remote_code: bool,
    warmup_batches: int = 0,
    release_after: bool = False,
) -> tuple[list[float], dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing runtime deps for fineweb scoring. Install: pip install -U transformers torch") from e

    if device_str == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device_str)
    dtype = resolve_torch_dtype(torch=torch, device=device, dtype=dtype_str, arg_name="--fineweb-dtype")

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.to(device)
    sync_if_cuda(torch=torch, device=device)
    load_end = time.perf_counter()

    scores_raw: list[float] = []
    with torch.inference_mode():
        batches = list(batched(texts, int(batch_size)))
        for _ in range(max(0, int(warmup_batches))):
            if not batches:
                break
            tok = tokenizer(
                batches[0],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_length),
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            _ = model(**tok)
        sync_if_cuda(torch=torch, device=device)

        infer_start = time.perf_counter()
        sync_if_cuda(torch=torch, device=device)
        for batch in batches:
            tok = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_length),
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            out = model(**tok)
            logits = out.logits
            if logits.ndim == 2 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            scores_raw.extend([float(x) for x in logits.detach().float().cpu().tolist()])
        sync_if_cuda(torch=torch, device=device)
        infer_end = time.perf_counter()

    if score_mode == "raw":
        scores = [float(s) for s in scores_raw]
    elif score_mode == "int0_5":
        scores = [float(clip_round_0_5(float(s))) for s in scores_raw]
    else:
        raise ValueError("--fineweb-score-mode must be one of: raw, int0_5")

    info = {
        "model_id": str(model_id),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "score_mode": str(score_mode),
        "load_s": float(load_end - load_start),
        "infer_s": float(infer_end - infer_start),
    }

    if release_after:
        try:
            if device.type == "cuda":
                del model
                del tokenizer
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    return scores, info

