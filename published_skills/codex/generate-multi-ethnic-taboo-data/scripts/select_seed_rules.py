#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ETHNICITY_ORDER = ("藏族", "维吾尔族", "蒙古族", "哈萨克族")
DIFFICULTY_ORDER = ("easy", "mid", "hard")
DIFFICULTY_ALIASES = {"medium": "mid", "expert": "hard"}
DIFFICULTY_PROFILES = {
    "easy": {
        "goal": "explicit rule matching",
        "constraints": [
            "Keep scenario short.",
            "It is acceptable to reuse the named object/custom and action from taboo_summary.",
            "Make the key violation action directly observable.",
            "Avoid saying the action is wrong, rude, taboo, or respectful.",
        ],
    },
    "mid": {
        "goal": "neutral minimal pair rewritten from easy",
        "constraints": [
            "First draft an internal easy pair from the seed, then rewrite it into the final mid pair.",
            "Use the same first scenario sentence for polluted_zh and normal_zh.",
            "Remove direct rule phrasing, negation shortcuts, evaluative words, emotional cues, and third-party reactions.",
            "Make the two texts differ mainly in the key action.",
            "Output only the final mid row unless the user asks for the full ladder.",
        ],
    },
    "hard": {
        "goal": "single-axis implicit contrast rewritten through easy and mid",
        "constraints": [
            "First draft an internal easy pair, rewrite it into mid, then rewrite mid into the final hard pair.",
            "Do not generate hard in one shot.",
            "Do not reuse a mid-level realization for the same seed.",
            "Paraphrase the taboo instead of copying taboo_summary.",
            "Preserve the critical modifier that makes the action label-bearing.",
            "Use exactly one label-bearing action axis: target, location, order, timing, or handled object.",
            "If the seed contains multiple possible violations, choose one and hold the others constant or omit them.",
            "Avoid obvious negative/normalizing cues such as 没有, 只是, 只向, 才, 大声, 喧闹, 乱, 随手, 直接, 抢先, 反复, 丢, 扔, 静音, 垃圾桶, 归还, 递回, 避开, 小心, 轻轻, 稳稳, 整齐.",
            "Use mirrored low-level actions rather than moralized opposites.",
            "Add at most one neutral distractor detail.",
            "Output only the final hard row unless the user asks for the full ladder.",
        ],
    },
}


def default_rules_path() -> Path:
    return Path(__file__).resolve().parents[1] / "references" / "seed_rules.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_csv_values(values: list[str] | None) -> set[str]:
    out: set[str] = set()
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                out.add(item)
    return out


def normalize_difficulty(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return DIFFICULTY_ALIASES.get(normalized, normalized)


def adjusted_difficulty(base: str, adjust: str) -> str:
    base = normalize_difficulty(base)
    idx = DIFFICULTY_ORDER.index(base)
    if adjust == "raise":
        idx = min(idx + 1, len(DIFFICULTY_ORDER) - 1)
    elif adjust == "lower":
        idx = max(idx - 1, 0)
    return DIFFICULTY_ORDER[idx]


def filter_rows(
    rows: list[dict[str, Any]],
    ethnicities: set[str],
    categories: set[str],
    exclude_seed_ids: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if ethnicities and row.get("ethnicity") not in ethnicities:
            continue
        row_category = row.get("category") or row.get("category_hint")
        if categories and row_category not in categories:
            continue
        if row.get("seed_id") in exclude_seed_ids:
            continue
        out.append(row)
    return out


def balanced_select(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    by_ethnicity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ethnicity[str(row["ethnicity"])].append(row)
    for bucket in by_ethnicity.values():
        rng.shuffle(bucket)

    active = [eth for eth in ETHNICITY_ORDER if by_ethnicity.get(eth)]
    active.extend(sorted(set(by_ethnicity) - set(active)))

    selected: list[dict[str, Any]] = []
    while active and len(selected) < count:
        next_active: list[str] = []
        for eth in active:
            bucket = by_ethnicity[eth]
            if bucket and len(selected) < count:
                selected.append(bucket.pop())
            if bucket:
                next_active.append(eth)
        active = next_active
    return selected


def format_prompt(selected: list[dict[str, Any]], difficulty: str, selection_seed: int) -> str:
    profile = DIFFICULTY_PROFILES[difficulty]
    seed_block = json.dumps(selected, ensure_ascii=False, indent=2)
    constraints = "\n".join(f"- {item}" for item in profile["constraints"])
    if difficulty == "easy":
        protocol = "- Create the final easy row directly from each seed.\n"
    elif difficulty == "mid":
        protocol = (
            "- Internally draft easy first.\n"
            "- Rewrite easy into mid by deleting direct rule wording and label shortcuts.\n"
            "- Output only the final mid row.\n"
        )
    else:
        protocol = (
            "- Internally draft easy first.\n"
            "- Rewrite easy into mid.\n"
            "- Rewrite mid into hard by reducing surface cues and keeping one semantic action axis.\n"
            "- Output only the final hard row.\n"
        )
    return (
        "Generate multi-ethnic taboo benchmark rows from these seed rules.\n"
        f"selection_seed: {selection_seed}\n"
        f"difficulty: {difficulty}\n"
        f"difficulty_goal: {profile['goal']}\n\n"
        "Required output: JSONL, one object per seed, with sample_id, seed_id, ethnicity, "
        "source, confidence, taboo_summary, category, difficulty, scenario, polluted_zh, "
        "normal_zh, slots_zh, and label_rationale. Use category_hint only as a starting point "
        "for category.\n\n"
        "Difficulty constraints:\n"
        f"{constraints}\n\n"
        "Progressive generation protocol:\n"
        f"{protocol}\n"
        "Global constraints:\n"
        "- Keep polluted_zh and normal_zh matched except for the key action.\n"
        "- Do not use slurs, demeaning claims, stereotypes, or label-leaking explanations.\n"
        "- Preserve traceability to seed_id and taboo_summary.\n\n"
        "Seed rules:\n"
        f"{seed_block}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select embedded seed rules for multi-ethnic taboo data generation.")
    parser.add_argument("--rules-path", default=str(default_rules_path()))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--difficulty", choices=DIFFICULTY_ORDER + tuple(DIFFICULTY_ALIASES), default="mid")
    parser.add_argument("--adjust", choices=("same", "raise", "lower"), default="same")
    parser.add_argument("--ethnicity", action="append", help="Ethnicity filter. May be repeated or comma-separated.")
    parser.add_argument("--category", action="append", help="Category filter. May be repeated or comma-separated.")
    parser.add_argument("--exclude-seed-id", action="append", help="Seed IDs to exclude. May be repeated or comma-separated.")
    parser.add_argument("--seed", type=int, default=None, help="Selection RNG seed. If omitted, a fresh seed is generated.")
    parser.add_argument("--format", choices=("prompt", "json", "jsonl", "markdown"), default="prompt")
    args = parser.parse_args()

    rules_path = Path(args.rules_path)
    rows = load_jsonl(rules_path)
    ethnicities = parse_csv_values(args.ethnicity)
    categories = parse_csv_values(args.category)
    exclude_seed_ids = parse_csv_values(args.exclude_seed_id)
    filtered = filter_rows(rows, ethnicities, categories, exclude_seed_ids)
    if not filtered:
        raise SystemExit("No seed rules matched the requested filters.")

    selection_seed = args.seed
    if selection_seed is None:
        selection_seed = random.SystemRandom().randrange(1, 1_000_000_000)
    rng = random.Random(selection_seed)
    difficulty = adjusted_difficulty(args.difficulty, args.adjust)
    selected = balanced_select(filtered, max(0, int(args.count)), rng)

    if args.format == "jsonl":
        for row in selected:
            print(json.dumps(row, ensure_ascii=False))
    elif args.format == "json":
        print(json.dumps({"selection_seed": selection_seed, "difficulty": difficulty, "rules": selected}, ensure_ascii=False, indent=2))
    elif args.format == "markdown":
        print(f"# Selected Seed Rules\n\nselection_seed: `{selection_seed}`\ndifficulty: `{difficulty}`\n")
        for row in selected:
            category = row.get("category") or row.get("category_hint") or "uncategorized"
            print(f"- `{row['seed_id']}` ({row['ethnicity']}, {category}, {row.get('confidence', 'unknown')}): {row['taboo_summary']}")
    else:
        print(format_prompt(selected, difficulty, selection_seed))


if __name__ == "__main__":
    main()
