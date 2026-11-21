#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from filterguard.multi_ethnic import (  # noqa: E402
    TARGET_LANGUAGE_SPECS,
    build_scene_planning_job,
    build_text_generation_job,
    confidence_sort_key,
    load_json,
    load_reference_examples,
    load_source_taboos,
    write_jsonl,
)


def _select_seeds(
    rows: list[Any],
    *,
    target_ethnicities: set[str],
    max_per_ethnicity: int,
) -> list[Any]:
    filtered = [row for row in rows if not target_ethnicities or row.ethnicity in target_ethnicities]
    if max_per_ethnicity <= 0:
        return filtered

    bucketed: dict[str, list[Any]] = defaultdict(list)
    for row in filtered:
        bucketed[row.ethnicity].append(row)

    selected: list[Any] = []
    for ethnicity, items in sorted(bucketed.items()):
        items.sort(key=lambda row: (confidence_sort_key(row.confidence), row.id))
        selected.extend(items[:max_per_ethnicity])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-ethnic benchmark scaffold")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "multi_ethnic_benchmark_v1.template.json"),
        help="Path to config json",
    )
    parser.add_argument(
        "--max-per-ethnicity",
        type=int,
        default=0,
        help="Optional seed cap per ethnicity for scaffold jobs (0 means all)",
    )
    args = parser.parse_args()

    cfg = load_json(args.config)
    output_dir = ROOT / str(cfg["output_dir"])
    scaffold_dir = output_dir / "scaffold"
    jobs_dir = scaffold_dir / "jobs"

    target_ethnicities = {str(x).strip() for x in cfg.get("target_ethnicities", []) if str(x).strip()}
    target_languages = list(cfg.get("target_languages") or [{"name": spec["name"], "code": spec["code"]} for spec in TARGET_LANGUAGE_SPECS])

    seeds = load_source_taboos(ROOT / str(cfg["seed_taboos_path"]))
    refs = load_reference_examples(ROOT / str(cfg["reference_examples_dir"]))
    selected = _select_seeds(
        seeds,
        target_ethnicities=target_ethnicities,
        max_per_ethnicity=int(args.max_per_ethnicity),
    )

    text_jobs = [build_text_generation_job(seed, target_languages=target_languages) for seed in selected]
    scene_jobs = [build_scene_planning_job(seed) for seed in selected]

    write_jsonl(scaffold_dir / "source_taboos.jsonl", [seed.to_json() for seed in selected])
    write_jsonl(scaffold_dir / "reference_examples.jsonl", [ref.to_json() for ref in refs])
    write_jsonl(jobs_dir / "text_pair_generation_jobs.jsonl", text_jobs)
    write_jsonl(jobs_dir / "scene_planning_jobs.jsonl", scene_jobs)

    meta = {
        "benchmark_name": cfg.get("benchmark_name"),
        "seed_count": len(selected),
        "reference_example_count": len(refs),
        "target_ethnicities": sorted(target_ethnicities),
        "target_languages": target_languages,
        "targets": cfg.get("targets", {}),
        "counts_by_ethnicity": dict(Counter(seed.ethnicity for seed in selected)),
        "counts_by_confidence": dict(Counter(seed.confidence for seed in selected)),
    }
    scaffold_dir.mkdir(parents=True, exist_ok=True)
    with (scaffold_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
