from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from filterguard.metric import extract_text, split_indices


_SEED_ABNORMAL_TYPES = {
    "bias_probe",
    "hallucination_pair",
    "harmful_prompt",
    "mcq_harmful",
}
_SEED_NORMAL_TYPES = {
    "benign",
    "clean",
    "harmless",
    "neutral",
    "neutral_prompt",
    "normal",
    "safe_prompt",
}
_CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _format_choices(choices: Any) -> str:
    if not isinstance(choices, list) or not choices:
        return ""
    parts: list[str] = []
    for i, choice in enumerate(choices):
        label = _CHOICE_LABELS[i] if i < len(_CHOICE_LABELS) else str(i + 1)
        parts.append(f"{label}. {str(choice).strip()}")
    return "\n".join(parts)


def _seed_type_label(seed_type: str, *, line_no: int | None = None) -> int:
    normalized = str(seed_type or "").strip().lower()
    if normalized in _SEED_ABNORMAL_TYPES:
        return 1
    if normalized in _SEED_NORMAL_TYPES:
        return 0
    where = f" at line {line_no}" if line_no is not None else ""
    raise ValueError(
        f"Unsupported seed_type{where}: {seed_type!r}. "
        f"Known abnormal types: {sorted(_SEED_ABNORMAL_TYPES)}; "
        f"known normal types: {sorted(_SEED_NORMAL_TYPES)}."
    )


def _looks_like_seed_row(ex: dict[str, Any]) -> bool:
    return "seed_type" in ex and "prompt" in ex and "harm_category_L1" in ex


def _normalize_seed_row(ex: dict[str, Any], *, line_no: int) -> dict[str, Any]:
    prompt = ex.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Invalid seed row at line {line_no}: missing non-empty string field 'prompt'.")

    seed_type = str(ex.get("seed_type") or "").strip()
    label = _seed_type_label(seed_type, line_no=line_no)
    payload = ex.get("payload")
    payload_obj = payload if isinstance(payload, dict) else {}

    user_parts = [prompt.strip()]
    choices = _format_choices(payload_obj.get("choices"))
    if choices:
        user_parts.append(f"Choices:\n{choices}")

    assistant_parts: list[str] = []
    if seed_type.lower() == "hallucination_pair":
        hallucinated_answer = payload_obj.get("hallucinated_answer")
        if isinstance(hallucinated_answer, str) and hallucinated_answer.strip():
            assistant_parts.append(f"Hallucinated answer: {hallucinated_answer.strip()}")

    user_text = "\n".join(user_parts)
    messages = [{"role": "user", "content": user_text}]
    if assistant_parts:
        messages.append({"role": "assistant", "content": "\n".join(assistant_parts)})

    normalized = dict(ex)
    normalized.setdefault("messages", messages)
    normalized.setdefault("text", "\n".join([user_text, *assistant_parts]).strip())
    normalized.setdefault("is_polluted", bool(label))
    normalized.setdefault("is_poisoned", False)
    normalized.setdefault("source", ex.get("source_benchmark"))
    normalized.setdefault("dataset_format", "third_seed_full_l1_jsonl")
    return normalized


def _load_jsonl_rows(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if limit and len(rows) >= int(limit):
                break
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSONL file {path} at line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL file {path} line {line_no} must be a JSON object, got {type(obj).__name__}.")
            if _looks_like_seed_row(obj):
                obj = _normalize_seed_row(obj, line_no=line_no)
            rows.append(obj)
    return rows


def load_rows(dataset_path: str, *, limit: int = 0) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if path.is_file():
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"Unsupported dataset file format: {path}. Expected a .jsonl file.")
        return _load_jsonl_rows(path, limit=int(limit))

    from datasets import load_from_disk

    ds = load_from_disk(str(path))
    if limit and limit > 0:
        ds = ds.select(range(min(int(limit), len(ds))))
    return ds.to_list()


def abnormal_label(ex: dict[str, Any]) -> int:
    if "is_polluted" in ex or "is_poisoned" in ex:
        is_polluted = bool(ex.get("is_polluted"))
        is_poisoned = bool(ex.get("is_poisoned"))
        return 1 if (is_polluted or is_poisoned) else 0
    if "seed_type" in ex:
        return _seed_type_label(str(ex.get("seed_type") or ""))
    raise ValueError(
        "Dataset row is missing label fields. Expected 'is_polluted'/'is_poisoned' "
        "or a recognized 'seed_type'."
    )


def labels_texts(rows: list[dict[str, Any]], *, include_assistant: bool) -> tuple[list[int], list[str]]:
    labels: list[int] = []
    texts: list[str] = []
    for ex in rows:
        labels.append(abnormal_label(ex))
        texts.append(extract_text(ex, include_assistant=include_assistant))
    return labels, texts


def resolve_dev_eval_indices(
    rows: list[dict[str, Any]],
    labels: list[int],
    *,
    dev_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], str]:
    split_map = {
        "dev": "dev",
        "valid": "dev",
        "validation": "dev",
        "val": "dev",
        "test": "eval",
        "eval": "eval",
    }
    split_tags = [split_map.get(str(ex.get("split") or "").strip().lower(), "") for ex in rows]
    if split_tags and all(tag in {"", "dev", "eval"} for tag in split_tags):
        dev_idx = [i for i, tag in enumerate(split_tags) if tag == "dev"]
        eval_idx = [i for i, tag in enumerate(split_tags) if tag == "eval"]
        if dev_idx and eval_idx:
            return dev_idx, eval_idx, "explicit"
    dev_idx, eval_idx = split_indices(labels, float(dev_ratio), int(seed))
    split_mode = "stratified_single_class" if labels and len(set(labels)) < 2 else "stratified"
    return dev_idx, eval_idx, split_mode
