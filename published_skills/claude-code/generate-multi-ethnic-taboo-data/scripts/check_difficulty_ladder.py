#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LEVELS = ("easy", "mid", "hard")
DIFFICULTY_ALIASES = {"medium": "mid", "expert": "hard"}
TEXT_FIELDS = ("scenario", "polluted_zh", "normal_zh")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_lineno"] = lineno
            rows.append(row)
    return rows


def by_seed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        seed_id = str(row.get("seed_id") or "").strip()
        if seed_id:
            out[seed_id] = row
    return out


def pair_text(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("polluted_zh") or "").strip(), str(row.get("normal_zh") or "").strip())


def normalize_difficulty(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return DIFFICULTY_ALIASES.get(normalized, normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that multi-level taboo datasets have real difficulty separation.")
    parser.add_argument("--easy")
    parser.add_argument("--mid")
    parser.add_argument("--medium")
    parser.add_argument("--hard")
    parser.add_argument("--max-identical-adjacent", type=int, default=0)
    args = parser.parse_args()

    paths = {level: getattr(args, level) for level in LEVELS if getattr(args, level)}
    if args.medium and "mid" not in paths:
        paths["mid"] = args.medium
    if len(paths) < 2:
        raise SystemExit("Provide at least two difficulty files.")

    datasets = {level: load_jsonl(Path(path)) for level, path in paths.items()}
    indexed = {level: by_seed(rows) for level, rows in datasets.items()}
    errors: list[str] = []
    warnings: list[str] = []
    adjacent_summaries: list[dict[str, Any]] = []

    for level, rows in datasets.items():
        wrong = [row for row in rows if normalize_difficulty(str(row.get("difficulty") or "").strip()) != level]
        if wrong:
            errors.append(f"{level}: {len(wrong)} rows have a mismatched difficulty field")
        if len(indexed[level]) != len(rows):
            errors.append(f"{level}: duplicate or missing seed_id values")

    active_levels = [level for level in LEVELS if level in indexed]
    for left, right in zip(active_levels, active_levels[1:]):
        left_rows = indexed[left]
        right_rows = indexed[right]
        shared = sorted(set(left_rows) & set(right_rows))
        identical_pairs: list[str] = []
        identical_scenarios: list[str] = []
        identical_all_text: list[str] = []
        for seed_id in shared:
            a = left_rows[seed_id]
            b = right_rows[seed_id]
            if pair_text(a) == pair_text(b):
                identical_pairs.append(seed_id)
            if str(a.get("scenario") or "").strip() == str(b.get("scenario") or "").strip():
                identical_scenarios.append(seed_id)
            if all(str(a.get(field) or "").strip() == str(b.get(field) or "").strip() for field in TEXT_FIELDS):
                identical_all_text.append(seed_id)

        adjacent_summaries.append(
            {
                "comparison": f"{left}->{right}",
                "shared_seed_count": len(shared),
                "identical_pair_text_count": len(identical_pairs),
                "identical_scenario_count": len(identical_scenarios),
                "identical_all_text_count": len(identical_all_text),
                "identical_pair_seed_ids": identical_pairs,
            }
        )

        if len(identical_pairs) > args.max_identical_adjacent:
            errors.append(
                f"{left}->{right}: {len(identical_pairs)} shared seeds reuse identical polluted_zh/normal_zh text"
            )
        if identical_scenarios:
            warnings.append(f"{left}->{right}: {len(identical_scenarios)} shared seeds reuse the same scenario")

    summary = {
        "levels": active_levels,
        "row_counts": {level: len(rows) for level, rows in datasets.items()},
        "errors": len(errors),
        "warnings": len(warnings),
        "adjacent_summaries": adjacent_summaries,
        "error_messages": errors,
        "warning_messages": warnings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
