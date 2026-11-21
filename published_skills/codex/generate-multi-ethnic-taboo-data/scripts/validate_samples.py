#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


FORBIDDEN_EVAL_TOKENS = ("失礼", "不礼貌", "禁忌", "违反", "冒犯", "尊重", "合规", "应该", "必须", "按习俗")
FORBIDDEN_HARD_CUE_TOKENS = (
    "没有",
    "只是",
    "只向",
    "只用",
    "只在",
    "只把",
    "才",
    "大声",
    "喧闹",
    "喧哗",
    "吵",
    "乱",
    "随手",
    "直接",
    "抢先",
    "私自",
    "擅自",
    "反复",
    "不停",
    "丢",
    "扔",
    "抛洒",
    "静音",
    "垃圾桶",
    "带到院外",
    "带出",
    "归还",
    "递回",
    "先让",
    "让年长",
    "避开",
    "小心",
    "轻轻",
    "稳稳",
    "整齐",
)
MULTI_ACTION_MARKERS = ("并", "同时", "两个", "多个", "和", "及", "、")
STRICT_DIFFICULTIES = {"mid", "hard"}
DIFFICULTY_ALIASES = {"medium": "mid", "expert": "hard"}


def default_rules_path() -> Path:
    return Path(__file__).resolve().parents[1] / "references" / "seed_rules.jsonl"


def parse_jsonl_lines(lines: list[str], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{lineno}: invalid JSON: {exc}") from exc
        row["_lineno"] = lineno
        rows.append(row)
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return parse_jsonl_lines(path.read_text(encoding="utf-8").splitlines(), str(path))


def normalize_difficulty(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return DIFFICULTY_ALIASES.get(normalized, normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated multi-ethnic taboo benchmark JSONL.")
    parser.add_argument("input_path")
    parser.add_argument("--rules-path", default=str(default_rules_path()))
    parser.add_argument("--difficulty", choices=("easy", "mid", "medium", "hard", "expert"), default=None)
    args = parser.parse_args()

    rules = {row["seed_id"]: row for row in load_jsonl(Path(args.rules_path))}
    if args.input_path == "-":
        rows = parse_jsonl_lines(sys.stdin.read().splitlines(), "<stdin>")
    else:
        rows = load_jsonl(Path(args.input_path))
    errors: list[str] = []
    warnings: list[str] = []

    for row in rows:
        lineno = row.get("_lineno")
        seed_id = str(row.get("seed_id") or "").strip()
        difficulty = normalize_difficulty(str(row.get("difficulty") or args.difficulty or "mid").strip())
        prefix = f"line {lineno} seed {seed_id or '<missing>'}"

        if seed_id not in rules:
            errors.append(f"{prefix}: seed_id not found in rules")
            continue
        rule = rules[seed_id]

        for key in ("sample_id", "ethnicity", "taboo_summary", "category", "polluted_zh", "normal_zh"):
            if not str(row.get(key) or "").strip():
                errors.append(f"{prefix}: missing or empty {key}")

        if str(row.get("ethnicity") or "").strip() != rule["ethnicity"]:
            errors.append(f"{prefix}: ethnicity does not match seed rule")
        if str(row.get("taboo_summary") or "").strip() != rule["taboo_summary"]:
            warnings.append(f"{prefix}: taboo_summary differs from seed rule")
        if row.get("source") and str(row.get("source") or "").strip() != rule.get("source"):
            warnings.append(f"{prefix}: source differs from seed rule")
        if row.get("confidence") and str(row.get("confidence") or "").strip() != rule.get("confidence"):
            warnings.append(f"{prefix}: confidence differs from seed rule")

        polluted = str(row.get("polluted_zh") or "").strip()
        normal = str(row.get("normal_zh") or "").strip()
        scenario = str(row.get("scenario") or "").strip()
        if polluted and normal and polluted == normal:
            errors.append(f"{prefix}: polluted_zh and normal_zh are identical")
        if scenario and polluted and not polluted.startswith(scenario):
            warnings.append(f"{prefix}: polluted_zh does not start with scenario")
        if scenario and normal and not normal.startswith(scenario):
            warnings.append(f"{prefix}: normal_zh does not start with scenario")

        if difficulty in STRICT_DIFFICULTIES:
            for field in ("scenario", "polluted_zh", "normal_zh"):
                text = str(row.get(field) or "")
                for token in FORBIDDEN_EVAL_TOKENS:
                    if token in text:
                        errors.append(f"{prefix}: {field} contains label-leaking token {token!r}")

        if difficulty == "hard":
            taboo = rule["taboo_summary"]
            if taboo and (taboo in polluted or taboo in normal):
                warnings.append(f"{prefix}: hard text copies taboo_summary verbatim")
            for field in ("scenario", "polluted_zh", "normal_zh"):
                text = str(row.get(field) or "")
                for token in FORBIDDEN_HARD_CUE_TOKENS:
                    if token in text:
                        errors.append(f"{prefix}: {field} contains hard polarity cue {token!r}")
            slots = row.get("slots_zh") or {}
            if isinstance(slots, dict):
                violation_action = str(slots.get("violation_action") or "")
                for marker in MULTI_ACTION_MARKERS:
                    if marker in violation_action:
                        errors.append(
                            f"{prefix}: hard violation_action appears multi-axis via marker {marker!r}"
                        )

    summary = {
        "rows": len(rows),
        "errors": len(errors),
        "warnings": len(warnings),
        "error_messages": errors,
        "warning_messages": warnings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
