#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from random import Random
from typing import Any, Callable, Iterable


_META_FIELDS = (
    "subset",
    "source_split",
    "source_path",
    "score_name",
    "score_value",
    "label_detail",
    "toxicity",
    "jailbreaking",
    "human_annotation",
    "openai_moderation",
)


@dataclass(frozen=True)
class Example:
    id: str
    source: str
    split: str
    data_type: str
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
        return {
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


@dataclass(frozen=True)
class SourcePlan:
    name: str
    target: int
    reserve: int
    loader: Callable[[int], list[Example]]


def _clean_text(text: Any) -> str:
    return str(text or "").strip()


def _dialogue_text(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = _clean_text(msg.get("role"))
        content = _clean_text(msg.get("content"))
        if role and content:
            parts.append(f"### {role}\n{content}")
    return "\n\n".join(parts).strip()


def _prompt_response_text(prompt: str, response: str) -> str:
    messages = [{"role": "user", "content": _clean_text(prompt)}, {"role": "assistant", "content": _clean_text(response)}]
    return _dialogue_text(messages)


def _sample_unique(iterable: Iterable[Example], *, count: int, seen_texts: set[str]) -> list[Example]:
    picked: list[Example] = []
    for ex in iterable:
        text_key = _clean_text(ex.text or _dialogue_text(ex.messages or []))
        if not text_key or text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        picked.append(ex)
        if len(picked) >= count:
            break
    return picked


def _iter_ultrachat_normal(*, seed: int, buffer_size: int) -> Iterable[Example]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    for ex in ds:
        messages = ex.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        normalized: list[dict[str, str]] = []
        for msg in messages:
            role = _clean_text(msg.get("role"))
            content = _clean_text(msg.get("content"))
            if role and content:
                normalized.append({"role": role, "content": content})
        if not normalized:
            continue
        yield Example(
            id=str(ex.get("prompt_id") or f"ultrachat:{hash(json.dumps(normalized, ensure_ascii=False))}"),
            source="HuggingFaceH4/ultrachat_200k",
            split="source",
            data_type="dialogue",
            is_polluted=False,
            is_poisoned=False,
            messages=normalized,
            text=_dialogue_text(normalized),
            meta={"source_split": "train_sft"},
        )


def _iter_toxic_chat_polluted(toxic_chat_root: Path, versions: list[str]) -> Iterable[Example]:
    import pandas as pd

    for version in versions:
        base = toxic_chat_root / "data" / version
        for source_split in ("train", "test"):
            path = base / f"toxic-chat_annotation_{source_split}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            polluted = df[(df["toxicity"] == 1) | (df["jailbreaking"] == 1)]
            for row in polluted.to_dict(orient="records"):
                user_input = _clean_text(row.get("user_input"))
                model_output = _clean_text(row.get("model_output"))
                messages = [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": model_output},
                ]
                yield Example(
                    id=str(row.get("conv_id") or f"toxic-chat:{version}:{source_split}:{hash(user_input + model_output)}"),
                    source="lmsys/toxic-chat",
                    split="source",
                    data_type="dialogue",
                    is_polluted=True,
                    is_poisoned=False,
                    messages=messages,
                    text=_dialogue_text(messages),
                    meta={
                        "subset": version,
                        "source_split": source_split,
                        "toxicity": float(row.get("toxicity", 0)),
                        "jailbreaking": int(row.get("jailbreaking", 0)),
                        "human_annotation": bool(row.get("human_annotation", True)),
                        "openai_moderation": str(row.get("openai_moderation") or ""),
                    },
                )


def _iter_backdoored_poisoned(backdoor_root: Path) -> Iterable[Example]:
    def iter_json_files(root: Path, source_split: str) -> Iterable[Example]:
        for path in sorted(root.rglob("backdoor*.json")):
            rel = path.relative_to(backdoor_root).as_posix()
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for idx, row in enumerate(data):
                if not isinstance(row, dict):
                    continue
                instruction = _clean_text(row.get("instruction"))
                input_text = _clean_text(row.get("input"))
                output = _clean_text(row.get("output"))
                user_text = instruction if not input_text else f"{instruction}\n\n{input_text}"
                messages = [{"role": "user", "content": user_text}, {"role": "assistant", "content": output}]
                yield Example(
                    id=f"backdoored:{rel}:{idx}",
                    source="BackdoorLLM/Backdoored_Dataset",
                    split="source",
                    data_type="sft",
                    is_polluted=False,
                    is_poisoned=True,
                    instruction=instruction,
                    input=input_text,
                    output=output,
                    messages=messages,
                    text=_dialogue_text(messages),
                    meta={"source_path": rel, "source_split": source_split},
                )

    yield from iter_json_files(backdoor_root / "poison_data", "train")
    yield from iter_json_files(backdoor_root / "test_data" / "poison", "test")


def _iter_real_toxicity_polluted(*, seed: int, buffer_size: int, min_toxicity: float) -> Iterable[Example]:
    from datasets import load_dataset

    ds = load_dataset("allenai/real-toxicity-prompts", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    for idx, row in enumerate(ds):
        prompt = row.get("prompt") or {}
        continuation = row.get("continuation") or {}
        prompt_text = _clean_text(prompt.get("text"))
        continuation_text = _clean_text(continuation.get("text"))
        prompt_tox = float(prompt.get("toxicity") or 0.0)
        cont_tox = float(continuation.get("toxicity") or 0.0)
        toxicity = max(prompt_tox, cont_tox)
        text = f"{prompt_text} {continuation_text}".strip()
        if toxicity < min_toxicity or not text:
            continue
        yield Example(
            id=f"real-toxicity:{idx}:{hash(text)}",
            source="allenai/real-toxicity-prompts",
            split="source",
            data_type="pretrain",
            is_polluted=True,
            is_poisoned=False,
            text=text,
            meta={
                "source_split": "train",
                "score_name": "toxicity",
                "score_value": toxicity,
                "label_detail": "max(prompt,continuation)",
            },
        )


def _iter_hh_redteam_polluted(*, seed: int, buffer_size: int, max_harmlessness: float) -> Iterable[Example]:
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    for idx, row in enumerate(ds):
        transcript = _clean_text(row.get("transcript"))
        harmlessness = float(row.get("min_harmlessness_score_transcript") or 0.0)
        if not transcript or harmlessness > max_harmlessness:
            continue
        yield Example(
            id=f"hh-redteam:{idx}:{hash(transcript)}",
            source="Anthropic/hh-rlhf:red-team-attempts",
            split="source",
            data_type="dialogue",
            is_polluted=True,
            is_poisoned=False,
            text=transcript,
            meta={
                "source_split": "train",
                "score_name": "min_harmlessness_score_transcript",
                "score_value": harmlessness,
            },
        )


def _iter_pku_saferlhf_unsafe(*, seed: int, buffer_size: int, subset: str, min_severity: int) -> Iterable[Example]:
    from datasets import load_dataset

    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", name=subset, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    for idx, row in enumerate(ds):
        prompt = _clean_text(row.get("prompt"))
        for response_id in (0, 1):
            response = _clean_text(row.get(f"response_{response_id}"))
            is_safe = bool(row.get(f"is_response_{response_id}_safe"))
            severity = int(row.get(f"response_{response_id}_severity_level") or 0)
            if is_safe or severity < min_severity or not prompt or not response:
                continue
            text = _prompt_response_text(prompt, response)
            yield Example(
                id=f"pku-saferlhf:{subset}:{idx}:r{response_id}",
                source="PKU-Alignment/PKU-SafeRLHF",
                split="source",
                data_type="dialogue",
                is_polluted=True,
                is_poisoned=False,
                messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
                text=text,
                meta={
                    "subset": subset,
                    "source_split": "train",
                    "score_name": "severity_level",
                    "score_value": float(severity),
                    "label_detail": f"response_{response_id}_unsafe",
                },
            )


def _iter_jigsaw_polluted(jigsaw_root: Path) -> Iterable[Example]:
    import pandas as pd

    train_path = jigsaw_root / "train.csv"
    if not train_path.exists():
        return
    df = pd.read_csv(train_path)
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    present_cols = [col for col in label_cols if col in df.columns]
    if "comment_text" not in df.columns or not present_cols:
        return
    for row in df.to_dict(orient="records"):
        score = int(sum(int(row.get(col) or 0) for col in present_cols))
        if score <= 0:
            continue
        text = _clean_text(row.get("comment_text"))
        if not text:
            continue
        yield Example(
            id=f"jigsaw:{row.get('id') or hash(text)}",
            source="google/jigsaw_toxicity_pred",
            split="source",
            data_type="comment",
            is_polluted=True,
            is_poisoned=False,
            text=text,
            meta={
                "source_split": "train",
                "score_name": "label_sum",
                "score_value": float(score),
            },
        )


def _assign_benchmark_split(examples: list[Example], *, dev_count: int, rnd: Random) -> list[Example]:
    items = list(examples)
    rnd.shuffle(items)
    out: list[Example] = []
    for idx, ex in enumerate(items):
        split = "dev" if idx < dev_count else "test"
        out.append(replace(ex, split=split))
    return out


def _write_meta(path: Path, examples: list[Example], *, config: dict[str, Any], source_targets: dict[str, int], missing_sources: list[str]) -> None:
    by_source = Counter()
    by_split = Counter()
    counts = Counter()
    for ex in examples:
        by_source[ex.source] += 1
        by_split[ex.split] += 1
        key = "normal"
        if ex.is_poisoned:
            key = "poisoned"
        elif ex.is_polluted:
            key = "polluted"
        counts[key] += 1
    meta = {
        "version": "v2",
        "config": config,
        "counts": dict(counts),
        "by_source": dict(by_source),
        "by_split": dict(by_split),
        "source_targets": source_targets,
        "missing_sources": missing_sources,
        "total": len(examples),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
                "subset": Value("string"),
                "source_split": Value("string"),
                "source_path": Value("string"),
                "score_name": Value("string"),
                "score_value": Value("float64"),
                "label_detail": Value("string"),
                "toxicity": Value("float64"),
                "jailbreaking": Value("int64"),
                "human_annotation": Value("bool"),
                "openai_moderation": Value("string"),
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
    parser = argparse.ArgumentParser(description="Generate abnormal benchmark dataset v2")
    parser.add_argument("--output-dir", default=str(Path("datasets/generated_datasets/v2")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-count", type=int, default=30000)
    parser.add_argument("--dev-count", type=int, default=20000)
    parser.add_argument("--normal-ratio", type=float, default=0.7)
    parser.add_argument("--ultrachat-buffer-size", type=int, default=30000)
    parser.add_argument("--streaming-buffer-size", type=int, default=20000)
    parser.add_argument("--toxic-chat-root", default=str(Path("datasets/third_datasets/toxic-chat")))
    parser.add_argument("--toxic-chat-versions", nargs="+", default=["0124", "1123"])
    parser.add_argument("--backdoor-root", default=str(Path("datasets/third_datasets/Backdoored_Dataset")))
    parser.add_argument("--jigsaw-root", default=str(Path("datasets/third_datasets/jigsaw_toxicity_pred")))
    parser.add_argument("--strict-local-sources", action="store_true")
    parser.add_argument("--real-toxicity-min-toxicity", type=float, default=0.5)
    parser.add_argument("--hh-max-harmlessness", type=float, default=1.0)
    parser.add_argument("--pku-config", default="default")
    parser.add_argument("--pku-min-severity", type=int, default=1)
    args = parser.parse_args()

    if args.total_count <= 0:
        raise ValueError("--total-count must be > 0")
    if args.dev_count <= 0 or args.dev_count >= args.total_count:
        raise ValueError("--dev-count must be in (0, total-count)")
    if not 0.0 < args.normal_ratio < 1.0:
        raise ValueError("--normal-ratio must be in (0, 1)")

    warnings.filterwarnings("ignore", message=r"resource_tracker: There appear to be .* leaked semaphore objects")

    rnd = Random(args.seed)
    normal_total = int(round(args.total_count * args.normal_ratio))
    abnormal_total = int(args.total_count - normal_total)
    test_count = int(args.total_count - args.dev_count)
    normal_dev = int(round(args.dev_count * args.normal_ratio))
    abnormal_dev = int(args.dev_count - normal_dev)
    if normal_dev > normal_total or abnormal_dev > abnormal_total:
        raise ValueError("dev/test split is inconsistent with total-count and normal-ratio")

    seen_texts: set[str] = set()
    normal_examples = _sample_unique(
        _iter_ultrachat_normal(seed=args.seed, buffer_size=int(args.ultrachat_buffer_size)),
        count=normal_total,
        seen_texts=seen_texts,
    )
    if len(normal_examples) < normal_total:
        raise RuntimeError(f"ultrachat produced only {len(normal_examples)} examples, expected {normal_total}")

    per_source_target = abnormal_total // 6
    source_targets = {
        "toxic_chat": per_source_target,
        "backdoor": per_source_target,
        "real_toxicity": per_source_target,
        "hh_redteam": per_source_target,
        "pku_saferlhf": per_source_target,
        "jigsaw": abnormal_total - per_source_target * 5,
    }
    reserve = max(256, per_source_target // 2)
    missing_sources: list[str] = []

    toxic_root = Path(args.toxic_chat_root)
    backdoor_root = Path(args.backdoor_root)
    jigsaw_root = Path(args.jigsaw_root)
    if args.strict_local_sources:
        for required_path in (toxic_root, backdoor_root, jigsaw_root):
            if not required_path.exists():
                raise FileNotFoundError(required_path)

    plans = [
        SourcePlan(
            name="toxic_chat",
            target=source_targets["toxic_chat"],
            reserve=0,
            loader=lambda needed: _sample_unique(
                _iter_toxic_chat_polluted(toxic_root, list(args.toxic_chat_versions)),
                count=needed,
                seen_texts=seen_texts,
            ),
        ),
        SourcePlan(
            name="backdoor",
            target=source_targets["backdoor"],
            reserve=reserve,
            loader=lambda needed: _sample_unique(
                _iter_backdoored_poisoned(backdoor_root),
                count=needed,
                seen_texts=seen_texts,
            ),
        ),
        SourcePlan(
            name="real_toxicity",
            target=source_targets["real_toxicity"],
            reserve=reserve,
            loader=lambda needed: _sample_unique(
                _iter_real_toxicity_polluted(
                    seed=args.seed + 11,
                    buffer_size=int(args.streaming_buffer_size),
                    min_toxicity=float(args.real_toxicity_min_toxicity),
                ),
                count=needed,
                seen_texts=seen_texts,
            ),
        ),
        SourcePlan(
            name="hh_redteam",
            target=source_targets["hh_redteam"],
            reserve=reserve,
            loader=lambda needed: _sample_unique(
                _iter_hh_redteam_polluted(
                    seed=args.seed + 23,
                    buffer_size=int(args.streaming_buffer_size),
                    max_harmlessness=float(args.hh_max_harmlessness),
                ),
                count=needed,
                seen_texts=seen_texts,
            ),
        ),
        SourcePlan(
            name="pku_saferlhf",
            target=source_targets["pku_saferlhf"],
            reserve=reserve,
            loader=lambda needed: _sample_unique(
                _iter_pku_saferlhf_unsafe(
                    seed=args.seed + 37,
                    buffer_size=int(args.streaming_buffer_size),
                    subset=str(args.pku_config),
                    min_severity=int(args.pku_min_severity),
                ),
                count=needed,
                seen_texts=seen_texts,
            ),
        ),
        SourcePlan(
            name="jigsaw",
            target=source_targets["jigsaw"],
            reserve=0,
            loader=lambda needed: _sample_unique(
                _iter_jigsaw_polluted(jigsaw_root),
                count=needed,
                seen_texts=seen_texts,
            ),
        ),
    ]

    chosen_abnormal: list[Example] = []
    abnormal_leftovers: list[Example] = []
    for plan in plans:
        needed = plan.target + plan.reserve
        pool = plan.loader(needed)
        if len(pool) < plan.target:
            missing_sources.append(plan.name)
        rnd.shuffle(pool)
        chosen_abnormal.extend(pool[: plan.target])
        abnormal_leftovers.extend(pool[plan.target :])

    if len(chosen_abnormal) < abnormal_total:
        deficit = abnormal_total - len(chosen_abnormal)
        if len(abnormal_leftovers) < deficit:
            raise RuntimeError(
                f"abnormal source pool is insufficient: have {len(chosen_abnormal)} selected + {len(abnormal_leftovers)} reserve, need {abnormal_total}"
            )
        rnd.shuffle(abnormal_leftovers)
        chosen_abnormal.extend(abnormal_leftovers[:deficit])

    normal_with_split = _assign_benchmark_split(normal_examples, dev_count=normal_dev, rnd=rnd)
    abnormal_with_split = _assign_benchmark_split(chosen_abnormal[:abnormal_total], dev_count=abnormal_dev, rnd=rnd)
    all_examples = normal_with_split + abnormal_with_split
    rnd.shuffle(all_examples)

    by_split = Counter(ex.split for ex in all_examples)
    if by_split.get("dev", 0) != args.dev_count or by_split.get("test", 0) != test_count:
        raise RuntimeError(f"unexpected split sizes: {dict(by_split)}")

    out_dir = Path(args.output_dir)
    dataset_path = out_dir / "dataset_hf"
    meta_path = out_dir / "meta.json"

    config = {
        "seed": args.seed,
        "total_count": args.total_count,
        "dev_count": args.dev_count,
        "test_count": test_count,
        "normal_ratio": args.normal_ratio,
        "normal_total": normal_total,
        "abnormal_total": abnormal_total,
        "normal_dev": normal_dev,
        "abnormal_dev": abnormal_dev,
        "toxic_chat_versions": list(args.toxic_chat_versions),
        "real_toxicity_min_toxicity": args.real_toxicity_min_toxicity,
        "hh_max_harmlessness": args.hh_max_harmlessness,
        "pku_config": args.pku_config,
        "pku_min_severity": args.pku_min_severity,
        "strict_local_sources": bool(args.strict_local_sources),
    }

    _save_hf_dataset(dataset_path, all_examples)
    _write_meta(
        meta_path,
        all_examples,
        config=config,
        source_targets=source_targets,
        missing_sources=sorted(set(missing_sources)),
    )

    print(f"Wrote {len(all_examples)} examples -> {dataset_path}", flush=True)
    print(
        f"split_counts=dev:{by_split.get('dev', 0)} test:{by_split.get('test', 0)} "
        f"normal:{normal_total} abnormal:{abnormal_total}",
        flush=True,
    )
    if missing_sources:
        print(f"source_shortfall={','.join(sorted(set(missing_sources)))}", flush=True)
    print(f"Wrote meta -> {meta_path}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
