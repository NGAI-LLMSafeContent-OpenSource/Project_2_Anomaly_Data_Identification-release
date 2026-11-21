#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        return None
    return obj


def _atomic_save_to_disk(ds: Any, path: Path) -> None:
    tmp = path.parent / f"{path.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    if path.exists():
        shutil.rmtree(path)
    ds.save_to_disk(str(tmp))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shuffle v0.1 dataset order and write v0.1.1 (HF save_to_disk)")
    parser.add_argument(
        "--input-dir",
        default=str(Path("datasets/generated_datasets/v0.1")),
        help="Input dataset directory (contains dataset_hf + meta.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("datasets/generated_datasets/v0.1.1")),
        help="Output dataset directory (will contain dataset_hf + meta.json)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    try:
        from datasets import load_from_disk
    except Exception as e:  # noqa: BLE001
        print("Missing dependency: datasets. Install: pip install -U datasets", file=sys.stderr)
        raise SystemExit(2) from e

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    in_dataset = in_dir / "dataset_hf"
    out_dataset = out_dir / "dataset_hf"
    in_meta = in_dir / "meta.json"
    out_meta = out_dir / "meta.json"

    if not in_dataset.exists():
        raise FileNotFoundError(f"input dataset not found: {in_dataset}")

    ds = load_from_disk(str(in_dataset))
    ds = ds.shuffle(seed=int(args.seed))

    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save_to_disk(ds, out_dataset)

    meta = _load_json(in_meta) or {}
    meta["version"] = "v0.1.1"
    meta["derived_from"] = {"version": "v0.1", "path": str(in_dir)}
    config = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    config["shuffle_output"] = True
    config["shuffle_seed"] = int(args.seed)
    meta["config"] = config
    meta["note"] = "v0.1.1 only shuffles sample order; content is identical to v0.1."
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote shuffled dataset -> {out_dataset}", flush=True)
    print(f"Wrote meta -> {out_meta}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    # NOTE: Some HF stack versions may spawn background threads that keep the
    # interpreter alive even after main finishes. Force-exit to avoid hanging the CLI.
    os._exit(0)


if __name__ == "__main__":
    main()
