from __future__ import annotations

from typing import Any, Iterable, TypeVar

T = TypeVar("T")


def batched(seq: list[T], batch_size: int) -> Iterable[list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def sync_if_cuda(*, torch: Any, device: Any) -> None:
    if device is not None and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)
