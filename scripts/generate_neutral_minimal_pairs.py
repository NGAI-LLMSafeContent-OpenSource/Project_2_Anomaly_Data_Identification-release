#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from filterguard.multi_ethnic import load_json  # noqa: E402


PAIR_SCHEMA = {
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

NEUTRAL_EXAMPLE = {
    "scenario": "傍晚，客人和主人站在毡房门口，门帘还没有掀开。",
    "category": "入门礼",
    "polluted_zh": "傍晚，客人和主人站在毡房门口，门帘还没有掀开。客人先抬手掀开帐帘，自己走了进去。",
    "normal_zh": "傍晚，客人和主人站在毡房门口，门帘还没有掀开。主人掀开帐帘后，客人再低头走进去。",
    "slots_zh": {
        "ethnicity": "哈萨克族",
        "taboo_category": "入门礼",
        "sacred_object_or_custom": "主人揭帘迎客",
        "violation_action": "自己先掀开帐帘入内",
        "critical_modifier": "主人还没有掀帘时",
        "scene": "毡房门口",
        "participants": ["客人", "主人"],
    },
}

PREFIX_TO_ETHNICITY = {
    "TIB": "藏族",
    "UYG": "维吾尔族",
    "MON": "蒙古族",
    "KAZ": "哈萨克族",
}

FORBIDDEN_MARKERS = (
    "热情地",
    "笑着",
    "笑了",
    "笑容",
    "笑意",
    "恭敬地",
    "郑重地",
    "安静地",
    "小心地",
    "轻轻",
    "默默",
    "局促",
    "从容",
    "稳当",
    "只好",
    "忽然",
    "忍不住",
    "气氛",
    "热闹",
    "原本",
    "慌忙",
)

FORBIDDEN_EVAL_WORDS = (
    "不礼貌",
    "失礼",
    "违反禁忌",
    "按习俗",
    "按规矩",
    "冒犯",
    "尊重",
    "得体",
    "合规",
    "正确",
    "应该",
    "必须",
)


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
                    "You write structured benchmark data. "
                    "Return only valid JSON that matches the requested schema."
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


def build_prompt(sample_id: str, seed_id: str, ethnicity: str, taboo: str) -> str:
    schema = json.dumps(PAIR_SCHEMA, ensure_ascii=False, indent=2)
    example = json.dumps(NEUTRAL_EXAMPLE, ensure_ascii=False, indent=2)
    forbidden = "、".join(FORBIDDEN_MARKERS)
    forbidden_eval = "、".join(FORBIDDEN_EVAL_WORDS)
    return (
        "你正在为多民族 benchmark 生成“中性、最小对比”的中文样本。\n"
        "目标是让模型必须依靠关键动作判断，而不是依靠情绪词、礼貌词、气氛词或旁人反应来猜标签。\n\n"
        f"输入信息:\n- sample_id: {sample_id}\n- seed_id: {seed_id}\n- 民族: {ethnicity}\n- 禁忌/礼仪点: {taboo}\n\n"
        "硬性要求:\n"
        "1. scenario 只写一个中性场景句，必须是可观察的环境与人物位置，不写心理、评价、气氛和结果。\n"
        "2. polluted_zh 和 normal_zh 都必须以 scenario 原句开头，作为第一句，保持完全一致。\n"
        "3. 第二句只写关键动作差异，不写第三人的反应，不写“屋里安静了”“主人看了他一眼”之类的提示。\n"
        "4. 不要写礼貌评价或总结解释，不要写“失礼”“不礼貌”“按习俗”“应该”“必须”等词。\n"
        "5. 尽量避免带明显情绪或褒贬倾向的修饰词，尤其不要出现这些词："
        f"{forbidden}。\n"
        "6. 不要出现这些解释性词语："
        f"{forbidden_eval}。\n"
        "7. polluted_zh 要保留违规动作；normal_zh 要写同场景下的对照动作。\n"
        "8. 两条都尽量只写两句，不要扩写成小故事。\n"
        "9. category 用短语概括类型，如“待客礼”“宗教礼仪”“餐桌礼”。\n"
        "10. slots_zh 必须准确，尤其是 violation_action、critical_modifier、scene。\n\n"
        "参考示例，只学习“中性镜头 + 最小对比”的形式，不要照抄内容：\n"
        f"{example}\n\n"
        "返回 JSON，字段必须严格匹配这个 schema：\n"
        f"{schema}\n"
    )


def validate_parsed(parsed: dict[str, Any], ethnicity: str) -> dict[str, Any]:
    required = {"scenario", "category", "polluted_zh", "normal_zh", "slots_zh"}
    missing = required - set(parsed)
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")

    scenario = str(parsed["scenario"]).strip()
    category = str(parsed["category"]).strip()
    polluted_zh = str(parsed["polluted_zh"]).strip()
    normal_zh = str(parsed["normal_zh"]).strip()
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

    if not scenario or not polluted_zh or not normal_zh:
        raise ValueError("scenario/polluted_zh/normal_zh must be non-empty")
    if polluted_zh == normal_zh:
        raise ValueError("polluted_zh and normal_zh must differ")
    if not polluted_zh.startswith(scenario):
        raise ValueError("polluted_zh must start with scenario")
    if not normal_zh.startswith(scenario):
        raise ValueError("normal_zh must start with scenario")

    for text_name, text in {
        "scenario": scenario,
        "category": category,
        "polluted_zh": polluted_zh,
        "normal_zh": normal_zh,
    }.items():
        for token in FORBIDDEN_MARKERS:
            if token in text:
                raise ValueError(f"{text_name} contains forbidden marker: {token}")
        for token in FORBIDDEN_EVAL_WORDS:
            if token in text:
                raise ValueError(f"{text_name} contains evaluative token: {token}")

    participants = [str(x).strip() for x in list(slots["participants"]) if str(x).strip()]
    if not participants:
        raise ValueError("participants must be non-empty")

    return {
        "scenario": scenario,
        "category": category,
        "polluted_zh": polluted_zh,
        "normal_zh": normal_zh,
        "slots_zh": {
            "ethnicity": ethnicity,
            "taboo_category": str(slots["taboo_category"]).strip(),
            "sacred_object_or_custom": str(slots["sacred_object_or_custom"]).strip(),
            "violation_action": str(slots["violation_action"]).strip(),
            "critical_modifier": str(slots["critical_modifier"]).strip(),
            "scene": str(slots["scene"]).strip(),
            "participants": participants,
        },
    }


def resolve_ethnicity(ref: dict[str, Any]) -> str:
    value = ref.get("ethnicity")
    if value:
        return str(value).strip()

    for key, value in ref.items():
        if "thnicity" in str(key).lower() and value:
            return str(value).strip()

    slots = ref.get("slots_zh")
    if isinstance(slots, dict) and slots.get("ethnicity"):
        return str(slots["ethnicity"]).strip()

    sample_id = str(ref.get("sample_id") or "").strip()
    prefix = sample_id.split("-", 1)[0]
    if prefix in PREFIX_TO_ETHNICITY:
        return PREFIX_TO_ETHNICITY[prefix]

    raise KeyError("ethnicity")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate neutral minimal-pair samples with GLM")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "multi_ethnic_benchmark_v1.glm.json"),
        help="Config path",
    )
    parser.add_argument(
        "--seed-reference-path",
        default=str(
            ROOT
            / "datasets"
            / "generated_datasets"
            / "multi_ethnic_benchmark_v1"
            / "backup"
            / "story_style_runs"
            / "story_style_samples_100.jsonl"
        ),
        help="Reference file that defines sample_id/seed_id/ethnicity/taboo_summary",
    )
    parser.add_argument(
        "--output-path",
        default=str(
            ROOT
            / "datasets"
            / "generated_datasets"
            / "multi_ethnic_benchmark_v1"
            / "current"
            / "neutral_minimal_pairs_100.jsonl"
        ),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--errors-path",
        default=str(
            ROOT
            / "datasets"
            / "generated_datasets"
            / "multi_ethnic_benchmark_v1"
            / "current"
            / "neutral_minimal_pairs_errors.jsonl"
        ),
        help="Error JSONL path",
    )
    parser.add_argument(
        "--summary-path",
        default=str(
            ROOT
            / "datasets"
            / "generated_datasets"
            / "multi_ethnic_benchmark_v1"
            / "current"
            / "neutral_minimal_pairs_summary.json"
        ),
        help="Summary JSON path",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of rows to generate")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    cfg = load_json(args.config)
    provider = (cfg.get("providers") or {}).get("text_generation") or {}
    base_url = str(provider.get("base_url") or "").strip()
    api_key = str(provider.get("api_key") or "").strip()
    model = str(provider.get("model") or "").strip()
    temperature = float(provider.get("temperature") or 0.2)
    max_tokens = max(int(provider.get("max_tokens") or 1400), 1000)
    if not base_url or not api_key or not model:
        raise SystemExit("text_generation provider is incomplete")

    seed_reference_path = Path(args.seed_reference_path)
    output_path = Path(args.output_path)
    errors_path = Path(args.errors_path)
    summary_path = Path(args.summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        output_path.write_text("", encoding="utf-8")
        errors_path.write_text("", encoding="utf-8")

    existing_rows = load_jsonl(output_path) if args.resume else []
    existing_error_rows = load_jsonl(errors_path) if args.resume else []
    done_sample_ids = {str(row.get("sample_id") or "").strip() for row in existing_rows}

    reference_rows = load_jsonl(seed_reference_path)
    if args.limit > 0:
        reference_rows = reference_rows[: args.limit]
    if done_sample_ids:
        reference_rows = [
            row
            for row in reference_rows
            if str(row.get("sample_id") or "").strip() not in done_sample_ids
        ]

    generated_rows: list[dict[str, Any]] = list(existing_rows)
    error_rows: list[dict[str, Any]] = list(existing_error_rows)

    for idx, ref in enumerate(reference_rows, start=1):
        sample_id = str(ref["sample_id"]).strip()
        seed_id = str(ref["seed_id"]).strip()
        ethnicity = resolve_ethnicity(ref)
        taboo = str(ref["taboo_summary"]).strip()
        prompt = build_prompt(sample_id, seed_id, ethnicity, taboo)

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
                parsed = validate_parsed(json.loads(content), ethnicity)
                row = {
                    "sample_id": sample_id,
                    "seed_id": seed_id,
                    "ethnicity": ethnicity,
                    "taboo_summary": taboo,
                    "scenario": parsed["scenario"],
                    "category": parsed["category"],
                    "polluted_zh": parsed["polluted_zh"],
                    "normal_zh": parsed["normal_zh"],
                    "slots_zh": parsed["slots_zh"],
                }
                append_jsonl(output_path, row)
                generated_rows.append(row)
                print(f"[{idx}/{len(reference_rows)}] ok {sample_id} <- {seed_id}", flush=True)
                break
            except (
                KeyError,
                IndexError,
                ValueError,
                json.JSONDecodeError,
                urllib.error.URLError,
                TimeoutError,
            ) as e:
                last_error = str(e)
                if attempt == args.retries:
                    error_row = {
                        "sample_id": sample_id,
                        "seed_id": seed_id,
                        "ethnicity": ethnicity,
                        "taboo_summary": taboo,
                        "error": last_error,
                    }
                    append_jsonl(errors_path, error_row)
                    error_rows.append(error_row)
                    print(f"[{idx}/{len(reference_rows)}] failed {sample_id} <- {seed_id}: {last_error}", flush=True)
                else:
                    time.sleep(2.0 * attempt)

    summary = {
        "remaining_reference_count": len(reference_rows),
        "resume": args.resume,
        "existing_count": len(existing_rows),
        "generated_count": len(generated_rows),
        "errors_count": len(error_rows),
        "output_path": str(output_path),
        "errors_path": str(errors_path),
        "model": model,
        "temperature": temperature,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
