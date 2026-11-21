#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from filterguard.multi_ethnic import confidence_sort_key, load_json, load_source_taboos  # noqa: E402


PREFIX_BY_ETHNICITY = {
    "藏族": "TIB",
    "维吾尔族": "UYG",
    "蒙古族": "MON",
    "哈萨克族": "KAZ",
}

STORY_SCHEMA = {
    "scenario": "string",
    "category": "string",
    "polluted_zh": "string",
    "normal_zh": "string",
    "slots_zh": {
        "ethnicity": "string",
        "taboo_category": "string",
        "sacred_object_or_custom": "string",
        "violation_action": "string",
        "critical_modifier": "string",
        "scene": "string",
        "participants": ["string"],
    },
}

STYLE_EXAMPLE = {
    "scenario": "傍晚回到牧家帐篷里，炉火正旺，屋里人都在忙着准备晚饭。",
    "polluted_zh": "傍晚回到牧家帐篷里，炉火正旺，孩子手里抱着东西图省事，抬腿就想从炉灶上方跨过去去拿碗。",
    "normal_zh": "傍晚回到牧家帐篷里，炉火正旺，孩子手里抱着东西绕开炉灶，稳稳当当地去拿碗。",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    prompt: str,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a structured dataset writer. "
                    "Return only valid JSON matching the requested schema."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_story_prompt(seed_id: str, ethnicity: str, taboo: str) -> str:
    schema = json.dumps(STORY_SCHEMA, ensure_ascii=False, indent=2)
    example = json.dumps(STYLE_EXAMPLE, ensure_ascii=False, indent=2)
    return (
        "你正在为多民族 benchmark 写中文文本样本，目标是“自然的小故事片段”，不是直给型规则句。\n\n"
        f"输入信息:\n- seed_id: {seed_id}\n- 民族: {ethnicity}\n- 禁忌/礼仪点: {taboo}\n\n"
        "写作要求:\n"
        "1. 写成同一场景中的两个短故事片段：polluted_zh 和 normal_zh。\n"
        "2. 两条必须共享同一人物、同一地点、同一时间和同一叙事框架，只改变关键动作。\n"
        "3. 不要直接解释规则，不要写“这是失礼的/不礼貌/按习俗/违反禁忌”等总结句。\n"
        "4. 通过动作、停顿、视线、环境和他人反应来体现差异，让文本更自然。\n"
        "5. 每条建议 2 句左右，长度适中，细节具体，但不要夸张、猎奇或刻板化。\n"
        "6. polluted_zh 要明确包含违规动作；normal_zh 要给出同场景下的尊重动作。\n"
        "7. scenario 用一句话概括这个故事发生的场景。\n"
        "8. category 用一个短语概括类型，例如“待客礼仪”“宗教礼仪”“餐桌礼仪”。\n"
        "9. slots_zh 必须准确，尤其是 violation_action 和 critical_modifier。\n\n"
        "参考风格，只学习叙事感，不要照抄：\n"
        f"{example}\n\n"
        "返回 JSON，字段必须严格匹配这个 schema：\n"
        f"{schema}\n"
    )


def validate_parsed(parsed: dict[str, Any], ethnicity: str) -> dict[str, Any]:
    required = {"scenario", "category", "polluted_zh", "normal_zh", "slots_zh"}
    missing = required - set(parsed)
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")
    slots = parsed["slots_zh"]
    if not isinstance(slots, dict):
        raise ValueError("slots_zh must be an object")
    slot_required = {
        "ethnicity",
        "taboo_category",
        "sacred_object_or_custom",
        "violation_action",
        "critical_modifier",
        "scene",
        "participants",
    }
    slot_missing = slot_required - set(slots)
    if slot_missing:
        raise ValueError(f"Missing slot keys: {sorted(slot_missing)}")
    slots["ethnicity"] = ethnicity
    return {
        "scenario": str(parsed["scenario"]).strip(),
        "category": str(parsed["category"]).strip(),
        "polluted_zh": str(parsed["polluted_zh"]).strip(),
        "normal_zh": str(parsed["normal_zh"]).strip(),
        "slots_zh": {
            "ethnicity": ethnicity,
            "taboo_category": str(slots["taboo_category"]).strip(),
            "sacred_object_or_custom": str(slots["sacred_object_or_custom"]).strip(),
            "violation_action": str(slots["violation_action"]).strip(),
            "critical_modifier": str(slots["critical_modifier"]).strip(),
            "scene": str(slots["scene"]).strip(),
            "participants": [str(x).strip() for x in list(slots["participants"])],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate story-style benchmark samples")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "multi_ethnic_benchmark_v1.glm.json"),
        help="Config path",
    )
    parser.add_argument(
        "--existing-path",
        default=str(ROOT / "datasets" / "generated_datasets" / "multi_ethnic_benchmark_v1" / "backup" / "story_style_runs" / "story_style_samples_20.jsonl"),
        help="Existing accepted story-style samples",
    )
    parser.add_argument(
        "--new-output-path",
        default=str(ROOT / "datasets" / "generated_datasets" / "multi_ethnic_benchmark_v1" / "experiments" / "story_style" / "story_style_samples_80_new.jsonl"),
        help="Where to write the newly generated rows",
    )
    parser.add_argument(
        "--full-output-path",
        default=str(ROOT / "datasets" / "generated_datasets" / "multi_ethnic_benchmark_v1" / "experiments" / "story_style" / "story_style_samples_100.jsonl"),
        help="Where to write the merged 100-row file",
    )
    parser.add_argument(
        "--errors-path",
        default=str(ROOT / "datasets" / "generated_datasets" / "multi_ethnic_benchmark_v1" / "experiments" / "story_style" / "story_style_generation_errors.jsonl"),
        help="Where to write failed jobs",
    )
    parser.add_argument("--target-per-ethnicity", type=int, default=25)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    cfg = load_json(args.config)
    provider = (cfg.get("providers") or {}).get("text_generation") or {}
    base_url = str(provider.get("base_url") or "").strip()
    api_key = str(provider.get("api_key") or "").strip()
    model = str(provider.get("model") or "").strip()
    temperature = float(provider.get("temperature") or 0.4)
    max_tokens = max(int(provider.get("max_tokens") or 1600), 1000)
    if not base_url or not api_key or not model:
        raise SystemExit("text_generation provider is incomplete")

    existing_path = Path(args.existing_path)
    new_output_path = Path(args.new_output_path)
    full_output_path = Path(args.full_output_path)
    errors_path = Path(args.errors_path)
    new_output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    new_output_path.write_text("", encoding="utf-8")
    errors_path.write_text("", encoding="utf-8")

    existing_rows = load_jsonl(existing_path)
    used_seed_ids = {str(row["seed_id"]) for row in existing_rows}
    count_by_ethnicity = Counter(str(row["ethnicity"]) for row in existing_rows)

    seeds = load_source_taboos(ROOT / str(cfg["seed_taboos_path"]))
    selected: list[Any] = []
    for ethnicity in PREFIX_BY_ETHNICITY:
        need = max(0, int(args.target_per_ethnicity) - count_by_ethnicity[ethnicity])
        candidates = [s for s in seeds if s.ethnicity == ethnicity and s.id not in used_seed_ids]
        candidates.sort(key=lambda row: (confidence_sort_key(row.confidence), row.id))
        selected.extend(candidates[:need])

    next_index = {eth: count_by_ethnicity[eth] + 1 for eth in PREFIX_BY_ETHNICITY}
    new_rows: list[dict[str, Any]] = []

    for idx, seed in enumerate(selected, start=1):
        prefix = PREFIX_BY_ETHNICITY[seed.ethnicity]
        sample_id = f"{prefix}-STORY-{next_index[seed.ethnicity]:02d}"
        next_index[seed.ethnicity] += 1
        prompt = build_story_prompt(seed.id, seed.ethnicity, seed.taboo)

        row: dict[str, Any] | None = None
        last_error: str | None = None
        for attempt in range(1, args.retries + 1):
            try:
                raw = _chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    prompt=prompt,
                )
                content = raw["choices"][0]["message"]["content"]
                parsed = validate_parsed(json.loads(content), seed.ethnicity)
                row = {
                    "sample_id": sample_id,
                    "seed_id": seed.id,
                    "ethnicity": seed.ethnicity,
                    "taboo_summary": seed.taboo,
                    "scenario": parsed["scenario"],
                    "category": parsed["category"],
                    "polluted_zh": parsed["polluted_zh"],
                    "normal_zh": parsed["normal_zh"],
                    "slots_zh": parsed["slots_zh"],
                }
                append_jsonl(new_output_path, row)
                new_rows.append(row)
                print(f"[{idx}/{len(selected)}] ok {sample_id} <- {seed.id}", flush=True)
                break
            except (KeyError, IndexError, ValueError, json.JSONDecodeError, urllib.error.URLError) as e:
                last_error = str(e)
                if attempt == args.retries:
                    error_row = {
                        "sample_id": sample_id,
                        "seed_id": seed.id,
                        "ethnicity": seed.ethnicity,
                        "taboo_summary": seed.taboo,
                        "error": last_error,
                    }
                    append_jsonl(errors_path, error_row)
                    print(f"[{idx}/{len(selected)}] failed {sample_id} <- {seed.id}: {last_error}", flush=True)
                else:
                    time.sleep(2.0 * attempt)

    merged = existing_rows + new_rows
    write_jsonl(full_output_path, merged)

    summary = {
        "existing_count": len(existing_rows),
        "new_count": len(new_rows),
        "merged_count": len(merged),
        "target_per_ethnicity": args.target_per_ethnicity,
        "counts_by_ethnicity": dict(Counter(row["ethnicity"] for row in merged)),
        "errors_count": len(load_jsonl(errors_path)),
        "new_output_path": str(new_output_path),
        "full_output_path": str(full_output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
