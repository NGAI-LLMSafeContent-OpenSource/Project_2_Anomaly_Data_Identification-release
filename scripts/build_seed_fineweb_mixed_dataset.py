#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from filterguard.data import abnormal_label, load_rows
from filterguard.metric import extract_text, split_indices


@dataclass(frozen=True)
class LengthControl:
    tokenizer: Any
    targets: list[int]
    max_tokens: int
    text_scope: str


def _json_dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"failed to parse JSONL {path} line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL {path} line {line_no} must be an object")
            yield obj


def _read_local_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _read_jsonl(path)
        return
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("pyarrow is required to read local parquet normal data") from e
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                if isinstance(row, dict):
                    yield row
        return
    raise ValueError(f"Unsupported local normal file format: {path}. Expected .jsonl or .parquet.")


def _normal_text_from_row(row: dict[str, Any]) -> str:
    text = extract_text(row, include_assistant=True)
    if text.strip():
        return text.strip()
    for key in ("text", "content", "raw_content", "markdown"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _token_ids(tokenizer: Any, text: str, *, max_tokens: int | None = None) -> list[int]:
    kwargs: dict[str, Any] = {
        "return_tensors": None,
        "padding": False,
        "truncation": bool(max_tokens),
        "add_special_tokens": False,
    }
    if max_tokens:
        kwargs["max_length"] = int(max_tokens)
    tok = tokenizer(text, **kwargs)
    return [int(x) for x in tok.get("input_ids", [])]


def _token_len(tokenizer: Any, text: str, *, max_tokens: int) -> int:
    return len(_token_ids(tokenizer, text, max_tokens=int(max_tokens)))


def _clip_text_to_token_len(tokenizer: Any, text: str, *, target_tokens: int, max_tokens: int) -> tuple[str, int, int]:
    original_len = _token_len(tokenizer, text, max_tokens=int(max_tokens))
    target = max(1, min(int(target_tokens), int(max_tokens)))
    ids = _token_ids(tokenizer, text, max_tokens=target)
    if not ids:
        return "", original_len, 0

    # Decoding can merge/split whitespace around tokens, so verify after decode.
    # If retokenization overshoots the target, shorten conservatively.
    for _ in range(4):
        clipped = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        clipped_len = _token_len(tokenizer, clipped, max_tokens=int(max_tokens))
        if clipped_len <= target or len(ids) <= 1:
            return clipped, original_len, clipped_len
        overshoot = clipped_len - target
        ids = ids[: max(1, len(ids) - overshoot)]
    clipped = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    return clipped, original_len, _token_len(tokenizer, clipped, max_tokens=int(max_tokens))


def _seed_text_for_length(row: dict[str, Any], *, text_scope: str) -> str:
    if text_scope == "user":
        return extract_text(row, include_assistant=False).strip()
    if text_scope == "full":
        return extract_text(row, include_assistant=True).strip()
    raise ValueError("--length-text-scope must be one of: user, full")


def _load_tokenizer(tokenizer_id: str, *, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("transformers is required for --length-control seed_token") from e
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True, trust_remote_code=bool(trust_remote_code))
    if tokenizer.pad_token_id is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_seed_token_length_control(
    *,
    seed_rows: list[dict[str, Any]],
    tokenizer_id: str,
    trust_remote_code: bool,
    max_tokens: int,
    min_tokens: int,
    text_scope: str,
    fineweb_count: int,
    seed: int,
) -> LengthControl:
    if int(max_tokens) < 2:
        raise ValueError("--length-max-tokens must be at least 2")
    if int(min_tokens) < 1:
        raise ValueError("--length-min-tokens must be positive")
    if int(min_tokens) > int(max_tokens):
        raise ValueError("--length-min-tokens must be <= --length-max-tokens")
    tokenizer = _load_tokenizer(str(tokenizer_id), trust_remote_code=bool(trust_remote_code))
    base_targets: list[int] = []
    for row in seed_rows:
        text = _seed_text_for_length(row, text_scope=str(text_scope))
        token_count = _token_len(tokenizer, text, max_tokens=int(max_tokens))
        base_targets.append(max(int(min_tokens), min(int(max_tokens), int(token_count))))
    if not base_targets:
        raise ValueError("no seed token lengths available for length control")

    rnd = random.Random(int(seed))
    if int(fineweb_count) <= len(base_targets):
        targets = base_targets[:]
        rnd.shuffle(targets)
        targets = targets[: int(fineweb_count)]
    else:
        targets = [rnd.choice(base_targets) for _ in range(int(fineweb_count))]
    return LengthControl(tokenizer=tokenizer, targets=targets, max_tokens=int(max_tokens), text_scope=str(text_scope))


def _load_local_normals(path: Path, *, count: int, min_chars: int, max_chars: int) -> list[dict[str, Any]]:
    normals: list[dict[str, Any]] = []
    for source_idx, row in enumerate(_read_local_rows(path)):
        text = _normal_text_from_row(row)
        if len(text) < min_chars:
            continue
        text = text[:max_chars].strip()
        if not text:
            continue
        normals.append(
            {
                "id": f"fineweb_local_{len(normals):08d}",
                "messages": [{"role": "user", "content": text}],
                "text": text,
                "is_polluted": False,
                "is_poisoned": False,
                "source": "local_fineweb_jsonl",
                "source_row_id": row.get("id", source_idx),
                "dataset_format": "seed_fineweb_mixed_v1",
            }
        )
        if len(normals) >= count:
            break
    return normals


def _load_local_normals_length_controlled(
    path: Path,
    *,
    count: int,
    min_chars: int,
    max_chars: int,
    length_control: LengthControl,
) -> list[dict[str, Any]]:
    normals: list[dict[str, Any]] = []
    targets = list(length_control.targets)
    for source_idx, row in enumerate(_read_local_rows(path)):
        if len(normals) >= count:
            break
        text = _normal_text_from_row(row)
        if len(text) < min_chars:
            continue
        text = text[:max_chars].strip()
        if not text:
            continue
        target = targets[len(normals)]
        clipped, original_tokens, clipped_tokens = _clip_text_to_token_len(
            length_control.tokenizer,
            text,
            target_tokens=int(target),
            max_tokens=int(length_control.max_tokens),
        )
        if not clipped:
            continue
        normals.append(
            {
                "id": f"fineweb_local_lenctl_{len(normals):08d}",
                "messages": [{"role": "user", "content": clipped}],
                "text": clipped,
                "is_polluted": False,
                "is_poisoned": False,
                "source": "local_fineweb_jsonl",
                "source_row_id": row.get("id", source_idx),
                "dataset_format": "seed_fineweb_mixed_v2_length_controlled",
                "length_control": {
                    "mode": "seed_token",
                    "target_token_count": int(target),
                    "original_token_count": int(original_tokens),
                    "clipped_token_count": int(clipped_tokens),
                    "max_tokens": int(length_control.max_tokens),
                    "text_scope": length_control.text_scope,
                },
            }
        )
    return normals


def _load_hf_normals(
    *,
    dataset: str,
    config: str,
    split: str,
    revision: str,
    count: int,
    seed: int,
    shuffle_buffer_size: int,
    min_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("datasets is required to stream FineWeb rows") from e

    kwargs: dict[str, Any] = {
        "path": dataset,
        "split": split,
        "streaming": True,
    }
    if config:
        kwargs["name"] = config
    if revision:
        kwargs["revision"] = revision
    ds = load_dataset(**kwargs)
    if shuffle_buffer_size > 1:
        ds = ds.shuffle(seed=int(seed), buffer_size=int(shuffle_buffer_size))

    normals: list[dict[str, Any]] = []
    for source_idx, row in enumerate(ds):
        if not isinstance(row, dict):
            continue
        text = _normal_text_from_row(row)
        if len(text) < min_chars:
            continue
        text = text[:max_chars].strip()
        if not text:
            continue
        normals.append(
            {
                "id": f"fineweb_{len(normals):08d}",
                "messages": [{"role": "user", "content": text}],
                "text": text,
                "is_polluted": False,
                "is_poisoned": False,
                "source": dataset,
                "source_config": config,
                "source_revision": revision,
                "source_split": split,
                "source_row_id": row.get("id", source_idx),
                "dataset_format": "seed_fineweb_mixed_v1",
            }
        )
        if len(normals) >= count:
            break
    return normals


def _load_hf_normals_length_controlled(
    *,
    dataset: str,
    config: str,
    split: str,
    revision: str,
    count: int,
    seed: int,
    shuffle_buffer_size: int,
    min_chars: int,
    max_chars: int,
    length_control: LengthControl,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("datasets is required to stream FineWeb rows") from e

    kwargs: dict[str, Any] = {
        "path": dataset,
        "split": split,
        "streaming": True,
    }
    if config:
        kwargs["name"] = config
    if revision:
        kwargs["revision"] = revision
    ds = load_dataset(**kwargs)
    if shuffle_buffer_size > 1:
        ds = ds.shuffle(seed=int(seed), buffer_size=int(shuffle_buffer_size))

    normals: list[dict[str, Any]] = []
    targets = list(length_control.targets)
    for source_idx, row in enumerate(ds):
        if len(normals) >= count:
            break
        if not isinstance(row, dict):
            continue
        text = _normal_text_from_row(row)
        if len(text) < min_chars:
            continue
        text = text[:max_chars].strip()
        if not text:
            continue
        target = targets[len(normals)]
        clipped, original_tokens, clipped_tokens = _clip_text_to_token_len(
            length_control.tokenizer,
            text,
            target_tokens=int(target),
            max_tokens=int(length_control.max_tokens),
        )
        if not clipped:
            continue
        normals.append(
            {
                "id": f"fineweb_lenctl_{len(normals):08d}",
                "messages": [{"role": "user", "content": clipped}],
                "text": clipped,
                "is_polluted": False,
                "is_poisoned": False,
                "source": dataset,
                "source_config": config,
                "source_revision": revision,
                "source_split": split,
                "source_row_id": row.get("id", source_idx),
                "dataset_format": "seed_fineweb_mixed_v2_length_controlled",
                "length_control": {
                    "mode": "seed_token",
                    "target_token_count": int(target),
                    "original_token_count": int(original_tokens),
                    "clipped_token_count": int(clipped_tokens),
                    "max_tokens": int(length_control.max_tokens),
                    "text_scope": length_control.text_scope,
                },
            }
        )
    return normals


def _sample_seed_rows(seed_path: str, *, count: int, seed: int) -> list[dict[str, Any]]:
    rows = load_rows(seed_path, limit=0)
    abnormal = [dict(row) for row in rows if abnormal_label(row) == 1]
    if len(abnormal) < count:
        raise ValueError(f"seed dataset has only {len(abnormal)} abnormal rows, requested {count}")
    rnd = random.Random(seed)
    indices = list(range(len(abnormal)))
    rnd.shuffle(indices)
    selected = [abnormal[i] for i in indices[:count]]
    for i, row in enumerate(selected):
        row["id"] = str(row.get("id") or f"seed_{i:08d}")
        row["dataset_format"] = "seed_fineweb_mixed_v1"
        row["mixed_source"] = "third_party_seed_full_l1"
    return selected


def _assign_splits(rows: list[dict[str, Any]], *, dev_ratio: float, seed: int) -> None:
    labels = [abnormal_label(row) for row in rows]
    dev_idx, eval_idx = split_indices(labels, float(dev_ratio), int(seed))
    for idx in dev_idx:
        rows[idx]["split"] = "dev"
    for idx in eval_idx:
        rows[idx]["split"] = "test"


def _length_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None}
    sorted_values = sorted(int(x) for x in values)
    n = len(sorted_values)

    def pct(q: float) -> int:
        idx = min(n - 1, max(0, round((n - 1) * q)))
        return sorted_values[idx]

    return {
        "n": n,
        "min": sorted_values[0],
        "p10": pct(0.10),
        "p25": pct(0.25),
        "median": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "max": sorted_values[-1],
        "mean": sum(sorted_values) / n,
    }


def _length_control_manifest(abnormal_rows: list[dict[str, Any]], normal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    seed_lengths = [
        int(row["length_control"]["seed_token_count"])
        for row in abnormal_rows
        if isinstance(row.get("length_control"), dict) and "seed_token_count" in row["length_control"]
    ]
    target_lengths = [
        int(row["length_control"]["target_token_count"])
        for row in normal_rows
        if isinstance(row.get("length_control"), dict) and "target_token_count" in row["length_control"]
    ]
    clipped_lengths = [
        int(row["length_control"]["clipped_token_count"])
        for row in normal_rows
        if isinstance(row.get("length_control"), dict) and "clipped_token_count" in row["length_control"]
    ]
    diffs = [abs(a - b) for a, b in zip(target_lengths, clipped_lengths, strict=False)]
    return {
        "seed_token_count": _length_summary(seed_lengths),
        "normal_target_token_count": _length_summary(target_lengths),
        "normal_clipped_token_count": _length_summary(clipped_lengths),
        "target_clip_abs_diff": _length_summary(diffs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducible abnormal seed + FineWeb normal mixed JSONL.")
    parser.add_argument("--seed-path", default="datasets/third_datasets/seeds_filtered_fullL1.jsonl")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--seed-count", type=int, default=1000, help="Number of abnormal seed rows.")
    parser.add_argument("--fineweb-count", type=int, default=1000, help="Number of normal FineWeb rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--min-normal-chars", type=int, default=80)
    parser.add_argument("--max-normal-chars", type=int, default=2000)
    parser.add_argument("--local-normal-jsonl", default="", help="Use this local JSONL/Parquet as the normal source instead of HF streaming.")
    parser.add_argument("--fineweb-dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--fineweb-config", default="sample-10BT")
    parser.add_argument("--fineweb-split", default="train")
    parser.add_argument("--fineweb-revision", default="main")
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument(
        "--length-control",
        choices=["none", "seed_token"],
        default="none",
        help="Optionally clip FineWeb rows to match the selected seed token-length distribution.",
    )
    parser.add_argument(
        "--length-tokenizer-id",
        default="",
        help="Tokenizer id/path used when --length-control seed_token is enabled.",
    )
    parser.add_argument("--length-tokenizer-trust-remote-code", action="store_true")
    parser.add_argument("--length-max-tokens", type=int, default=128)
    parser.add_argument("--length-min-tokens", type=int, default=2)
    parser.add_argument("--length-text-scope", choices=["user", "full"], default="user")
    args = parser.parse_args()

    if int(args.seed_count) <= 0 or int(args.fineweb_count) <= 0:
        raise ValueError("--seed-count and --fineweb-count must both be positive")
    if not 0.0 < float(args.dev_ratio) < 1.0:
        raise ValueError("--dev-ratio must be between 0 and 1")
    if int(args.min_normal_chars) < 1:
        raise ValueError("--min-normal-chars must be positive")
    if int(args.max_normal_chars) < int(args.min_normal_chars):
        raise ValueError("--max-normal-chars must be >= --min-normal-chars")

    started = time.perf_counter()
    abnormal_rows = _sample_seed_rows(str(args.seed_path), count=int(args.seed_count), seed=int(args.seed))
    length_control: LengthControl | None = None
    if str(args.length_control) == "seed_token":
        if not str(args.length_tokenizer_id).strip():
            raise ValueError("--length-tokenizer-id is required when --length-control seed_token is enabled")
        tokenizer_id = str(args.length_tokenizer_id)
        length_control = _build_seed_token_length_control(
            seed_rows=abnormal_rows,
            tokenizer_id=tokenizer_id,
            trust_remote_code=bool(args.length_tokenizer_trust_remote_code),
            max_tokens=int(args.length_max_tokens),
            min_tokens=int(args.length_min_tokens),
            text_scope=str(args.length_text_scope),
            fineweb_count=int(args.fineweb_count),
            seed=int(args.seed),
        )
        for row in abnormal_rows:
            token_count = _token_len(
                length_control.tokenizer,
                _seed_text_for_length(row, text_scope=str(args.length_text_scope)),
                max_tokens=int(args.length_max_tokens),
            )
            row["dataset_format"] = "seed_fineweb_mixed_v2_length_controlled"
            row["length_control"] = {
                "mode": "seed_token",
                "seed_token_count": int(max(int(args.length_min_tokens), min(int(args.length_max_tokens), int(token_count)))),
                "max_tokens": int(args.length_max_tokens),
                "text_scope": str(args.length_text_scope),
            }
    if args.local_normal_jsonl:
        if length_control is not None:
            normal_rows = _load_local_normals_length_controlled(
                Path(args.local_normal_jsonl),
                count=int(args.fineweb_count),
                min_chars=int(args.min_normal_chars),
                max_chars=int(args.max_normal_chars),
                length_control=length_control,
            )
        else:
            normal_rows = _load_local_normals(
                Path(args.local_normal_jsonl),
                count=int(args.fineweb_count),
                min_chars=int(args.min_normal_chars),
                max_chars=int(args.max_normal_chars),
            )
        normal_source = str(args.local_normal_jsonl)
    else:
        if length_control is not None:
            normal_rows = _load_hf_normals_length_controlled(
                dataset=str(args.fineweb_dataset),
                config=str(args.fineweb_config),
                split=str(args.fineweb_split),
                revision=str(args.fineweb_revision),
                count=int(args.fineweb_count),
                seed=int(args.seed),
                shuffle_buffer_size=int(args.shuffle_buffer_size),
                min_chars=int(args.min_normal_chars),
                max_chars=int(args.max_normal_chars),
                length_control=length_control,
            )
        else:
            normal_rows = _load_hf_normals(
                dataset=str(args.fineweb_dataset),
                config=str(args.fineweb_config),
                split=str(args.fineweb_split),
                revision=str(args.fineweb_revision),
                count=int(args.fineweb_count),
                seed=int(args.seed),
                shuffle_buffer_size=int(args.shuffle_buffer_size),
                min_chars=int(args.min_normal_chars),
                max_chars=int(args.max_normal_chars),
            )
        normal_source = f"{args.fineweb_dataset}/{args.fineweb_config}@{args.fineweb_revision}:{args.fineweb_split}"
    if len(normal_rows) < int(args.fineweb_count):
        raise ValueError(f"normal source yielded {len(normal_rows)} rows, requested {args.fineweb_count}")

    mixed = [*abnormal_rows, *normal_rows]
    _assign_splits(mixed, dev_ratio=float(args.dev_ratio), seed=int(args.seed))
    random.Random(int(args.seed)).shuffle(mixed)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in mixed:
            f.write(_json_dumps(row) + "\n")

    labels = [abnormal_label(row) for row in mixed]
    payload = {
        "output_path": str(output_path),
        "seed_path": str(args.seed_path),
        "normal_source": normal_source,
        "seed": int(args.seed),
        "dev_ratio": float(args.dev_ratio),
        "n": len(mixed),
        "abnormal_n": int(sum(labels)),
        "normal_n": int(len(labels) - sum(labels)),
        "dev_n": sum(1 for row in mixed if row.get("split") == "dev"),
        "test_n": sum(1 for row in mixed if row.get("split") == "test"),
        "sha256": _sha256_text(output_path.read_text(encoding="utf-8")),
        "elapsed_s": time.perf_counter() - started,
    }
    if length_control is not None:
        payload["length_control"] = {
            "mode": str(args.length_control),
            "tokenizer_id": str(args.length_tokenizer_id),
            "max_tokens": int(args.length_max_tokens),
            "min_tokens": int(args.length_min_tokens),
            "text_scope": str(args.length_text_scope),
            **_length_control_manifest(abnormal_rows, normal_rows),
        }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output_path={output_path}")
    print(f"manifest_path={manifest_path}")
    print(f"n={payload['n']} abnormal={payload['abnormal_n']} normal={payload['normal_n']} dev={payload['dev_n']} test={payload['test_n']}")
    print(f"sha256={payload['sha256']}")


if __name__ == "__main__":
    main()
