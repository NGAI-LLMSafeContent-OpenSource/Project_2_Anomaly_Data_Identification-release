#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
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

from filterguard.multi_ethnic import load_json  # noqa: E402


LANGUAGE_BY_ETHNICITY = {
    "藏族": {
        "name": "藏语",
        "code": "bo",
        "script": "Tibetan script",
        "script_hint": "必须使用藏文 Unicode 字符（如 བོད་），不要使用中文、拼音或拉丁转写。",
    },
    "维吾尔族": {
        "name": "维吾尔语",
        "code": "ug",
        "script": "Uyghur Arabic script",
        "script_hint": "必须使用维吾尔文阿拉伯字母（如 ئۇيغۇرچە），不要使用中文、拉丁转写或西里尔字母。",
    },
    "蒙古族": {
        "name": "蒙古语",
        "code": "mn",
        "script": "traditional Mongolian script",
        "script_hint": (
            "必须使用传统蒙古文 Unicode 字符 U+1800-U+18AF（如 ᠮᠣᠩᠭᠣᠯ）。"
            "严禁使用西里尔蒙古文（如 Монгол, Зочин）或拉丁转写。"
        ),
    },
    "哈萨克族": {
        "name": "哈萨克语",
        "code": "kk",
        "script": "Kazakh Arabic script as used in Xinjiang",
        "script_hint": (
            "必须使用新疆哈萨克文阿拉伯字母（如 قازاقشا）。"
            "严禁使用西里尔哈萨克文（如 Қазақ, Қонақ）或拉丁转写。"
        ),
    },
}

TRANSLATION_SCHEMA = {"translated_text": "string"}


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
                    "You are a professional translator. "
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


def make_units(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for row in rows:
        ethnicity = str(row["ethnicity"]).strip()
        target = LANGUAGE_BY_ETHNICITY[ethnicity]
        common = {
            "sample_id": row["sample_id"],
            "seed_id": row["seed_id"],
            "ethnicity": ethnicity,
            "target_language": target["name"],
            "target_language_code": target["code"],
            "target_script": target["script"],
            "script_hint": target["script_hint"],
            "category_zh": row["category"],
            "scenario_zh": row["scenario"],
            "slots_zh": row["slots_zh"],
        }
        units.append(
            {
                **common,
                "unit_id": f"{row['sample_id']}::polluted",
                "label": "polluted",
                "text_zh": row["polluted_zh"],
            }
        )
        units.append(
            {
                **common,
                "unit_id": f"{row['sample_id']}::normal",
                "label": "normal",
                "text_zh": row["normal_zh"],
            }
        )
    return units


def build_prompt(unit: dict[str, Any]) -> str:
    schema = json.dumps(TRANSLATION_SCHEMA, ensure_ascii=False, indent=2)
    payload = {
        "unit_id": unit["unit_id"],
        "ethnicity": unit["ethnicity"],
        "target_language": unit["target_language"],
        "target_language_code": unit["target_language_code"],
        "target_script": unit["target_script"],
        "script_hint": unit["script_hint"],
        "category_zh": unit["category_zh"],
        "text_zh": unit["text_zh"],
    }
    return (
        "请把下面这一条中文 benchmark 文本翻译成指定少数民族语言。\n\n"
        "重要约束:\n"
        "1. 你只会看到一条文本，不知道它是否属于任何对照组；请只忠实翻译这一条。\n"
        "2. 只输出目标语言，不要输出中文解释、拼音、拉丁转写或注释。\n"
        "3. 使用指定书写体系，不要混用其他文字体系；特别遵守输入里的 script_hint。\n"
        "4. 保持中性镜头式语气，不添加情绪词、评价词、解释性总结或旁人反应。\n"
        "5. 不要用 *** 或省略号替换宗教、民族、地点、器物等词。\n"
        "6. 保留原文里的动作、方向、器物和地点差异；不要改写或扩写。\n"
        "7. 返回 JSON，字段必须严格匹配 schema。\n\n"
        "输入:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "schema:\n"
        f"{schema}\n"
    )


def build_script_repair_prompt(unit: dict[str, Any], *, candidate_text: str) -> str:
    schema = json.dumps(TRANSLATION_SCHEMA, ensure_ascii=False, indent=2)
    payload = {
        "unit_id": unit["unit_id"],
        "target_language": unit["target_language"],
        "target_language_code": unit["target_language_code"],
        "target_script": unit["target_script"],
        "script_hint": unit["script_hint"],
        "candidate_text": candidate_text,
    }
    return (
        "你刚才给出的翻译使用了错误的书写体系。请把 candidate_text 改写为指定书写体系。\n\n"
        "重要约束:\n"
        "1. 只改写书写体系与拼写形式，不要改变含义，不要增删信息。\n"
        "2. 只输出目标语言，不要输出中文解释、拼音、拉丁转写或注释。\n"
        "3. 严格遵守 script_hint；如果 candidate_text 里有西里尔字母，必须全部替换成目标书写体系。\n"
        "4. 返回 JSON，字段必须严格匹配 schema。\n\n"
        "输入:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "schema:\n"
        f"{schema}\n"
    )


def _char_in_ranges(char: str, ranges: list[tuple[int, int]]) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in ranges)


def validate_target_script(text: str, language_code: str) -> None:
    target_ranges = {
        "bo": [(0x0F00, 0x0FFF)],
        "ug": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],
        "kk": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],
        "mn": [(0x1800, 0x18AF)],
    }
    cyrillic_ranges = [(0x0400, 0x04FF)]
    if any(0x4E00 <= ord(char) <= 0x9FFF for char in text):
        raise ValueError("translated_text contains Han characters")
    letters = [char for char in text if char.isalpha()]
    if not letters:
        raise ValueError("translated_text has no alphabetic target-language characters")
    target_ratio = sum(
        1 for char in letters if _char_in_ranges(char, target_ranges[language_code])
    ) / len(letters)
    cyrillic_ratio = sum(1 for char in letters if _char_in_ranges(char, cyrillic_ranges)) / len(letters)
    if target_ratio < 0.5:
        raise ValueError(f"target script ratio too low for {language_code}: {target_ratio:.3f}")
    if language_code in {"mn", "kk"} and cyrillic_ratio > 0:
        raise ValueError(f"unexpected Cyrillic characters for {language_code}: {cyrillic_ratio:.3f}")


def validate_translation(parsed: dict[str, Any], language_code: str) -> str:
    if "translated_text" not in parsed:
        raise ValueError("Missing translated_text")
    text = str(parsed["translated_text"]).strip()
    if not text:
        raise ValueError("translated_text is empty")
    validate_target_script(text, language_code)
    return text


def translate_unit(
    *,
    unit: dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    prompt = build_prompt(unit)
    last_error: str | None = None
    for attempt in range(1, retries + 1):
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
            parsed = json.loads(content)
            candidate_text = str(parsed.get("translated_text") or "").strip()
            translated_text = validate_translation(parsed, unit["target_language_code"])
            return {**unit, "translated_text": translated_text}, None
        except (
            KeyError,
            IndexError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.URLError,
            TimeoutError,
        ) as e:
            last_error = str(e)

            # For scripts that GLM commonly outputs in Cyrillic, try a focused
            # repair prompt using the model's own candidate text as input.
            if (
                isinstance(e, ValueError)
                and unit.get("target_language_code") in {"mn", "kk"}
                and "target script ratio too low" in last_error
                and "candidate_text" in locals()
                and candidate_text
            ):
                try:
                    repair_prompt = build_script_repair_prompt(unit, candidate_text=candidate_text)
                    raw2 = _chat_completion(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        prompt=repair_prompt,
                    )
                    content2 = raw2["choices"][0]["message"]["content"]
                    repaired = json.loads(content2)
                    repaired_text = validate_translation(repaired, unit["target_language_code"])
                    return {**unit, "translated_text": repaired_text}, None
                except Exception as repair_e:
                    last_error = f"{last_error} | repair_failed: {repair_e}"

            if attempt < retries:
                time.sleep(2.0 * attempt)

    error_row = {
        "unit_id": unit["unit_id"],
        "sample_id": unit["sample_id"],
        "seed_id": unit["seed_id"],
        "label": unit["label"],
        "ethnicity": unit["ethnicity"],
        "target_language": unit["target_language"],
        "target_language_code": unit["target_language_code"],
        "text_zh": unit["text_zh"],
        "error": last_error or "unknown error",
    }
    return None, error_row


def write_paired_outputs(
    *,
    source_rows: list[dict[str, Any]],
    unit_rows: list[dict[str, Any]],
    paired_path: Path,
) -> list[dict[str, Any]]:
    by_unit = {row["unit_id"]: row for row in unit_rows}
    paired_rows: list[dict[str, Any]] = []
    for row in source_rows:
        polluted_id = f"{row['sample_id']}::polluted"
        normal_id = f"{row['sample_id']}::normal"
        if polluted_id not in by_unit or normal_id not in by_unit:
            continue
        polluted = by_unit[polluted_id]
        normal = by_unit[normal_id]
        paired_rows.append(
            {
                "sample_id": row["sample_id"],
                "seed_id": row["seed_id"],
                "ethnicity": row["ethnicity"],
                "target_language": polluted["target_language"],
                "target_language_code": polluted["target_language_code"],
                "target_script": polluted["target_script"],
                "category_zh": row["category"],
                "scenario_zh": row["scenario"],
                "polluted_zh": row["polluted_zh"],
                "normal_zh": row["normal_zh"],
                "translated_polluted": polluted["translated_text"],
                "translated_normal": normal["translated_text"],
                "slots_zh": row["slots_zh"],
            }
        )
    write_jsonl(paired_path, paired_rows)
    return paired_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate neutral minimal-pair data one text at a time")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "multi_ethnic_benchmark_v1.glm.json"),
        help="Config path",
    )
    parser.add_argument(
        "--input-path",
        default=str(
            ROOT
            / "datasets"
            / "generated_datasets"
            / "multi_ethnic_benchmark_v1"
            / "current"
            / "neutral_minimal_pairs_100.jsonl"
        ),
        help="Chinese source JSONL path",
    )
    parser.add_argument(
        "--units-path",
        default=str(
            ROOT
            / "datasets"
            / "generated_datasets"
            / "multi_ethnic_benchmark_v1"
            / "current"
            / "translations"
            / "translation_units.jsonl"
        ),
        help="One-text translation unit manifest path",
    )
    parser.add_argument(
        "--unit-output-path",
        default=str(
            ROOT
            / "datasets"
            / "generated_datasets"
            / "multi_ethnic_benchmark_v1"
            / "current"
            / "translations"
            / "translated_units.jsonl"
        ),
        help="Translated unit JSONL path",
    )
    parser.add_argument(
        "--paired-output-path",
        default=str(
            ROOT
            / "datasets"
            / "generated_datasets"
            / "multi_ethnic_benchmark_v1"
            / "current"
            / "translations"
            / "translated_pairs.jsonl"
        ),
        help="Merged translated pair JSONL path",
    )
    parser.add_argument(
        "--errors-path",
        default=str(
            ROOT
            / "datasets"
            / "generated_datasets"
            / "multi_ethnic_benchmark_v1"
            / "current"
            / "translations"
            / "translation_errors.jsonl"
        ),
        help="Translation errors JSONL path",
    )
    parser.add_argument(
        "--summary-path",
        default=str(
            ROOT
            / "datasets"
            / "generated_datasets"
            / "multi_ethnic_benchmark_v1"
            / "current"
            / "translations"
            / "translation_summary.json"
        ),
        help="Summary JSON path",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit source samples before creating 2x units")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent one-text translation requests")
    parser.add_argument(
        "--overwrite-language-codes",
        default="",
        help="Comma-separated target language codes to remove from existing translated units before resume",
    )
    parser.add_argument(
        "--only-language-codes",
        default="",
        help="Comma-separated target language codes to translate; other units are skipped",
    )
    args = parser.parse_args()

    cfg = load_json(args.config)
    provider = (cfg.get("providers") or {}).get("text_generation") or {}
    base_url = str(provider.get("base_url") or "").strip()
    api_key = str(provider.get("api_key") or "").strip()
    model = str(provider.get("model") or "").strip()
    max_tokens = max(int(provider.get("max_tokens") or 1800), 1200)
    if not base_url or not api_key or not model:
        raise SystemExit("text_generation provider is incomplete")

    input_path = Path(args.input_path)
    units_path = Path(args.units_path)
    unit_output_path = Path(args.unit_output_path)
    paired_output_path = Path(args.paired_output_path)
    errors_path = Path(args.errors_path)
    summary_path = Path(args.summary_path)

    source_rows = load_jsonl(input_path)
    if args.limit > 0:
        source_rows = source_rows[: args.limit]
    units = make_units(source_rows)
    only_language_codes = {
        code.strip() for code in str(args.only_language_codes).split(",") if code.strip()
    }
    if only_language_codes:
        units = [
            unit
            for unit in units
            if str(unit.get("target_language_code") or "").strip() in only_language_codes
        ]
    write_jsonl(units_path, units)

    unit_output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        unit_output_path.write_text("", encoding="utf-8")
        paired_output_path.write_text("", encoding="utf-8")
        errors_path.write_text("", encoding="utf-8")

    overwrite_language_codes = {
        code.strip() for code in str(args.overwrite_language_codes).split(",") if code.strip()
    }
    existing_units = load_jsonl(unit_output_path) if args.resume else []
    if args.resume and overwrite_language_codes:
        kept_units = [
            row
            for row in existing_units
            if str(row.get("target_language_code") or "").strip() not in overwrite_language_codes
        ]
        if len(kept_units) != len(existing_units):
            write_jsonl(unit_output_path, kept_units)
            print(
                f"Removed {len(existing_units) - len(kept_units)} existing units for overwrite: "
                f"{','.join(sorted(overwrite_language_codes))}",
                flush=True,
            )
        existing_units = kept_units
    done_unit_ids = {str(row.get("unit_id") or "").strip() for row in existing_units}
    pending_units = [unit for unit in units if unit["unit_id"] not in done_unit_ids]
    error_rows = load_jsonl(errors_path) if args.resume else []
    if args.resume and overwrite_language_codes:
        kept_errors = [
            row
            for row in error_rows
            if str(row.get("target_language_code") or "").strip() not in overwrite_language_codes
        ]
        if len(kept_errors) != len(error_rows):
            write_jsonl(errors_path, kept_errors)
        error_rows = kept_errors

    translated_units: list[dict[str, Any]] = list(existing_units)
    worker_count = max(1, int(args.workers))
    if pending_units:
        print(
            f"Translating {len(pending_units)} pending units with {worker_count} workers",
            flush=True,
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                translate_unit,
                unit=unit,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=args.temperature,
                max_tokens=max_tokens,
                retries=args.retries,
            ): unit
            for unit in pending_units
        }
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            unit = futures[future]
            try:
                out_row, error_row = future.result()
            except Exception as e:  # Keep one failed worker from losing the whole batch.
                out_row = None
                error_row = {
                    "unit_id": unit["unit_id"],
                    "sample_id": unit["sample_id"],
                    "seed_id": unit["seed_id"],
                    "label": unit["label"],
                    "ethnicity": unit["ethnicity"],
                    "target_language": unit["target_language"],
                    "target_language_code": unit["target_language_code"],
                    "text_zh": unit["text_zh"],
                    "error": str(e),
                }
            if out_row is not None:
                append_jsonl(unit_output_path, out_row)
                translated_units.append(out_row)
                print(
                    f"[{idx}/{len(pending_units)}] ok {unit['unit_id']} -> {unit['target_language_code']}",
                    flush=True,
                )
            elif error_row is not None:
                append_jsonl(errors_path, error_row)
                error_rows.append(error_row)
                print(
                    f"[{idx}/{len(pending_units)}] failed {unit['unit_id']} -> {unit['target_language_code']}: {error_row['error']}",
                    flush=True,
                )

    paired_rows = write_paired_outputs(
        source_rows=source_rows,
        unit_rows=translated_units,
        paired_path=paired_output_path,
    )

    summary = {
        "source_sample_count": len(source_rows),
        "unit_count": len(units),
        "existing_unit_count": len(existing_units),
        "pending_unit_count": len(pending_units),
        "translated_unit_count": len(translated_units),
        "paired_count": len(paired_rows),
        "errors_count": len(error_rows),
        "by_language": dict(Counter(row["target_language_code"] for row in translated_units)),
        "units_path": str(units_path),
        "unit_output_path": str(unit_output_path),
        "paired_output_path": str(paired_output_path),
        "errors_path": str(errors_path),
        "model": model,
        "temperature": args.temperature,
        "workers": worker_count,
        "overwrite_language_codes": sorted(overwrite_language_codes),
        "only_language_codes": sorted(only_language_codes),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
