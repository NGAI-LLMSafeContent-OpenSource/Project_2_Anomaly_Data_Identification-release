#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable


_META_FIELDS = (
    "prompt",
    "toxicity",
    "jailbreaking",
    "human_annotation",
    "openai_moderation",
    "source_path",
)


def _build_text_from_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).rstrip()
        if not role and not content:
            continue
        parts.append(f"### {role}\n{content}")
    return "\n\n".join(parts).strip()


def _sft_user_text(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"{instruction}\n\n{input_text}".strip()
    return instruction


@dataclass(frozen=True)
class Example:
    id: str
    source: str
    split: str
    data_type: str  # dialogue / sft / pretrain
    is_polluted: bool
    is_poisoned: bool
    messages: list[dict[str, str]] | None = None
    instruction: str | None = None
    input: str | None = None
    output: str | None = None
    text: str | None = None
    meta: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        meta: dict[str, Any] = {k: None for k in _META_FIELDS}
        if self.meta:
            for k, v in self.meta.items():
                if k in meta:
                    meta[k] = v

        obj: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "split": self.split,
            "data_type": self.data_type,
            "is_polluted": self.is_polluted,
            "is_poisoned": self.is_poisoned,
            "messages": self.messages,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "text": self.text,
            "meta": meta,
        }
        return obj


def _iter_ultrachat_positive(
    *,
    seed: int,
    count: int,
    buffer_size: int,
) -> Iterable[Example]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)  # noqa

    for ex in islice(ds, count):
        messages = ex.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        prompt_id = str(ex.get("prompt_id") or "")
        if not prompt_id:
            prompt_id = f"ultrachat:{hash(json.dumps(ex, ensure_ascii=False, sort_keys=True))}"
        msg_list = []
        for m in messages:
            role = str(m.get("role", "")).strip()
            content = str(m.get("content", "")).rstrip()
            if not role or not content:
                continue
            msg_list.append({"role": role, "content": content})
        if not msg_list:
            continue
        text = _build_text_from_messages(msg_list)
        yield Example(
            id=prompt_id,
            source="HuggingFaceH4/ultrachat_200k",
            split="train",
            data_type="dialogue",
            is_polluted=False,
            is_poisoned=False,
            messages=msg_list,
            text=text,
            meta={"prompt": ex.get("prompt")},
        )


def _iter_toxic_chat_polluted(
    toxic_chat_root: Path,
    *,
    version: str,
) -> Iterable[Example]:
    import pandas as pd

    base = toxic_chat_root / "data" / version
    train_path = base / "toxic-chat_annotation_train.csv"
    test_path = base / "toxic-chat_annotation_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"toxic-chat files not found under {base} (expected train/test csv)."
        )

    def iter_split(path: Path, split: str) -> Iterable[Example]:
        df = pd.read_csv(path)
        required = {"conv_id", "user_input", "model_output", "toxicity", "jailbreaking"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        polluted = df[(df["toxicity"] == 1) | (df["jailbreaking"] == 1)]
        for row in polluted.to_dict(orient="records"):
            conv_id = str(row.get("conv_id") or "")
            user_input = str(row.get("user_input") or "")
            model_output = str(row.get("model_output") or "")
            messages = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": model_output},
            ]
            text = _build_text_from_messages(messages)
            yield Example(
                id=conv_id or f"toxic-chat:{split}:{hash(user_input + model_output)}",
                source=f"lmsys/toxic-chat:{version}",
                split=split,
                data_type="dialogue",
                is_polluted=True,
                is_poisoned=False,
                messages=messages,
                text=text,
                meta={
                    "toxicity": int(row.get("toxicity", 0)),
                    "jailbreaking": int(row.get("jailbreaking", 0)),
                    "human_annotation": bool(row.get("human_annotation", True)),
                    "openai_moderation": row.get("openai_moderation"),
                },
            )

    yield from iter_split(train_path, "train")
    yield from iter_split(test_path, "test")


def _iter_backdoored_poisoned(backdoor_root: Path) -> Iterable[Example]:
    poison_root = backdoor_root / "poison_data"
    test_poison_root = backdoor_root / "test_data" / "poison"
    if not poison_root.exists():
        raise FileNotFoundError(f"Backdoored_Dataset poison_data not found: {poison_root}")
    if not test_poison_root.exists():
        raise FileNotFoundError(
            f"Backdoored_Dataset test_data/poison not found: {test_poison_root}"
        )

    def iter_json_files(root: Path, split: str) -> Iterable[Example]:
        for path in sorted(root.rglob("backdoor*.json")):
            rel = path.relative_to(backdoor_root).as_posix()
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for idx, row in enumerate(data):
                if not isinstance(row, dict):
                    continue
                instruction = str(row.get("instruction") or "")
                input_text = str(row.get("input") or "")
                output = str(row.get("output") or "")
                user_text = _sft_user_text(instruction, input_text)
                messages = [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": output},
                ]
                text = _build_text_from_messages(messages)
                yield Example(
                    id=f"backdoored:{rel}:{idx}",
                    source="BackdoorLLM/Backdoored_Dataset",
                    split=split,
                    data_type="sft",
                    is_polluted=False,
                    is_poisoned=True,
                    instruction=instruction,
                    input=input_text,
                    output=output,
                    messages=messages,
                    text=text,
                    meta={"source_path": rel},
                )

    yield from iter_json_files(poison_root, "train")
    yield from iter_json_files(test_poison_root, "test")


def _write_meta(path: Path, examples: list[Example], config: dict[str, Any]) -> None:
    counts = Counter()
    by_source = Counter()
    by_split = Counter()
    for ex in examples:
        key = "normal"
        if ex.is_poisoned:
            key = "poisoned"
        elif ex.is_polluted:
            key = "polluted"
        counts[key] += 1
        by_source[ex.source] += 1
        by_split[ex.split] += 1
    meta = {
        "version": "v0.1",
        "config": config,
        "counts": dict(counts),
        "by_source": dict(by_source),
        "by_split": dict(by_split),
        "total": len(examples),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sample_keep_order(items: list[Example], k: int, rnd) -> list[Example]:
    if k <= 0:
        return []
    if k >= len(items):
        return items
    idx = sorted(rnd.sample(range(len(items)), k))
    return [items[i] for i in idx]


def _save_hf_dataset(path: Path, examples: list[Example]) -> None:
    import shutil

    from datasets import Dataset, Features, Value

    features = Features(
        {
            "id": Value("string"),
            "source": Value("string"),
            "split": Value("string"),
            "data_type": Value("string"),
            "is_polluted": Value("bool"),
            "is_poisoned": Value("bool"),
            "messages": [{"role": Value("string"), "content": Value("string")}],
            "instruction": Value("string"),
            "input": Value("string"),
            "output": Value("string"),
            "text": Value("string"),
            "meta": {
                "prompt": Value("string"),
                "toxicity": Value("int64"),
                "jailbreaking": Value("int64"),
                "human_annotation": Value("bool"),
                "openai_moderation": Value("string"),
                "source_path": Value("string"),
            },
        }
    )

    ds = Dataset.from_list([ex.to_json() for ex in examples], features=features)

    tmp = path.parent / f"{path.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    if path.exists():
        shutil.rmtree(path)
    ds.save_to_disk(str(tmp))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset v0.1")
    parser.add_argument(
        "--output-dir",
        default=str(Path("datasets/generated_datasets/v0.1")),
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ultrachat-count", type=int, default=6000)
    parser.add_argument("--ultrachat-buffer-size", type=int, default=10000)
    parser.add_argument(
        "--negative-count",
        type=int,
        default=2000,
        help="Total negative examples (polluted + poisoned)",
    )
    parser.add_argument(
        "--toxic-chat-root",
        default=str(Path("datasets/third_datasets/toxic-chat")),
        help="Path to local toxic-chat dataset folder",
    )
    parser.add_argument("--toxic-chat-version", default="0124")
    parser.add_argument(
        "--backdoor-root",
        default=str(Path("datasets/third_datasets/Backdoored_Dataset")),
        help="Path to local Backdoored_Dataset folder",
    )
    parser.add_argument("--shuffle-output", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message=r"resource_tracker: There appear to be .* leaked semaphore objects to clean up at shutdown",
    )

    config = {
        "seed": args.seed,
        "ultrachat_count": args.ultrachat_count,
        "ultrachat_buffer_size": args.ultrachat_buffer_size,
        "negative_count": args.negative_count,
        "toxic_chat_version": args.toxic_chat_version,
        "shuffle_output": bool(args.shuffle_output),
    }

    import random

    rnd = random.Random(args.seed)

    examples_pos = list(
        _iter_ultrachat_positive(
            seed=args.seed,
            count=args.ultrachat_count,
            buffer_size=args.ultrachat_buffer_size,
        )
    )
    if len(examples_pos) < args.ultrachat_count:
        raise RuntimeError(
            f"ultrachat produced only {len(examples_pos)} examples (expected {args.ultrachat_count})."
        )

    toxic_root = Path(args.toxic_chat_root)
    examples_neg_polluted = list(
        _iter_toxic_chat_polluted(
            toxic_root,
            version=args.toxic_chat_version,
        )
    )

    backdoor_root = Path(args.backdoor_root)
    examples_neg_poisoned = list(_iter_backdoored_poisoned(backdoor_root))

    neg_target = int(args.negative_count)
    if neg_target < 0:
        raise ValueError("--negative-count must be >= 0")

    neg_polluted_selected = _sample_keep_order(examples_neg_polluted, neg_target, rnd)
    remain = max(0, neg_target - len(neg_polluted_selected))
    neg_poisoned_selected = _sample_keep_order(examples_neg_poisoned, remain, rnd)

    examples: list[Example] = []
    examples.extend(examples_pos)
    examples.extend(neg_polluted_selected)
    examples.extend(neg_poisoned_selected)

    if args.shuffle_output:
        rnd.shuffle(examples)

    out_dir = Path(args.output_dir)
    dataset_path = out_dir / "dataset_hf"
    meta_path = out_dir / "meta.json"

    _save_hf_dataset(dataset_path, examples)
    _write_meta(meta_path, examples, config)

    print(f"Wrote {len(examples)} examples -> {dataset_path}", flush=True)
    print(f"Wrote meta -> {meta_path}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    # NOTE: In this environment, HuggingFace streaming may spawn background threads that keep
    # the interpreter alive even after main finishes. Force-exit to avoid hanging the CLI.
    os._exit(0)


if __name__ == "__main__":
    main()
