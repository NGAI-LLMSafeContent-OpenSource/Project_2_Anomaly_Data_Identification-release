from __future__ import annotations

from typing import Any


def resolve_torch_dtype(*, torch: Any, device: Any, dtype: str, arg_name: str = "--dtype") -> Any:
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    if dtype != "auto":
        raise ValueError(f"{arg_name} must be one of: auto, fp16, bf16, fp32")
    if device.type == "cuda":
        return torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    return torch.float32

