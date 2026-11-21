#!/usr/bin/env python3
"""Download Hugging Face benchmark datasets listed in docs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


BENCHMARKS: list[dict[str, Any]] = [
    {
        "name": "TrustLLM",
        "repo_id": "TrustLLM/TrustLLM-dataset",
        "note": "Gated/auto on Hugging Face; may require login and accepted terms.",
    },
    {"name": "DecodingTrust", "repo_id": "AI-Secure/DecodingTrust"},
    {"name": "SafetyBench", "repo_id": "thu-coai/SafetyBench"},
    {"name": "RedBench", "repo_id": "knoveleng/redbench"},
    {"name": "Do-Not-Answer", "repo_id": "LibrAI/do-not-answer"},
    {
        "name": "WildJailbreak",
        "repo_id": "allenai/wildjailbreak",
        "allow_patterns": ["README.md", ".gitattributes", "train/*", "eval/*"],
        "note": "Gated/auto on Hugging Face; figures are intentionally skipped.",
    },
    {
        "name": "JailbreakBench",
        "repo_id": "JailbreakBench/JBB-Behaviors",
        "allow_patterns": ["README.md", "LICENSE", ".gitattributes", "data/*"],
    },
    {"name": "FORTRESS", "repo_id": "ScaleAI/fortress_public"},
    {"name": "ToxiGen", "repo_id": "toxigen/toxigen-data"},
    {"name": "BOLD", "repo_id": "AmazonScience/bold"},
    {"name": "GSM-IC mirror", "repo_id": "voidful/GSM-IC"},
    {"name": "TruthfulQA", "repo_id": "truthfulqa/truthful_qa"},
    {
        "name": "Sycophancy Datasets",
        "repo_id": "Anthropic/model-written-evals",
        "allow_patterns": ["sycophancy/*"],
    },
    {"name": "CruxEval", "repo_id": "cruxeval-org/cruxeval"},
    {"name": "HumanEval-X", "repo_id": "THUDM/humaneval-x"},
    {"name": "AdvGLUE", "repo_id": "AI-Secure/adv_glue"},
    {"name": "H-CoT o1", "repo_id": "DukeCEICenter/Malicious_Educator_hcot_o1"},
    {
        "name": "H-CoT o3-mini",
        "repo_id": "DukeCEICenter/Malicious_Educator_hcot_o3-mini",
    },
    {
        "name": "H-CoT DeepSeek-R1",
        "repo_id": "DukeCEICenter/Malicious_Educator_hcot_DeepSeek-R1",
    },
    {
        "name": "H-CoT Gemini",
        "repo_id": "DukeCEICenter/Malicious_Educator_hcot_Gemini-2.0-Flash-Thinking",
    },
]


def repo_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def directory_size(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    if not path.exists():
        return total, count
    for root, _, files in os.walk(path):
        for file_name in files:
            file_path = Path(root) / file_name
            if file_path.is_file():
                total += file_path.stat().st_size
                count += 1
    return total, count


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="datasets/huggingface",
        help="Directory under which repo snapshots will be stored.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for index, item in enumerate(BENCHMARKS, start=1):
        repo_id = item["repo_id"]
        local_dir = output_dir / repo_dir_name(repo_id)
        print(f"[{index}/{len(BENCHMARKS)}] downloading {repo_id} -> {local_dir}", flush=True)
        record: dict[str, Any] = {
            "name": item["name"],
            "repo_id": repo_id,
            "local_dir": str(local_dir),
            "allow_patterns": item.get("allow_patterns"),
            "note": item.get("note"),
        }
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
                allow_patterns=item.get("allow_patterns"),
                max_workers=args.max_workers,
            )
            total_bytes, file_count = directory_size(local_dir)
            record.update(
                {
                    "status": "ok",
                    "bytes": total_bytes,
                    "human_size": human_size(total_bytes),
                    "file_count": file_count,
                }
            )
            print(f"    ok: {human_size(total_bytes)} in {file_count} files", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep going and report all failures.
            record.update({"status": "failed", "error": repr(exc)})
            print(f"    failed: {exc!r}", flush=True)
        manifest.append(record)

    manifest_path = output_dir / "download_manifest.json"
    summary_path = output_dir / "download_summary.md"
    completed = [item for item in manifest if item["status"] == "ok"]
    failed = [item for item in manifest if item["status"] != "ok"]
    total_bytes = sum(item.get("bytes", 0) for item in completed)

    manifest_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "total_bytes": total_bytes,
        "human_total_size": human_size(total_bytes),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "items": manifest,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Hugging Face Benchmark Download Summary",
        "",
        f"- Created at: `{manifest_payload['created_at']}`",
        f"- Output dir: `{output_dir}`",
        f"- Completed: `{len(completed)}`",
        f"- Failed: `{len(failed)}`",
        f"- Total local size: `{human_size(total_bytes)}`",
        "",
        "| Status | Dataset | Local size | Files | Local dir | Note |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for record in manifest:
        status = record["status"]
        size = record.get("human_size", "-")
        files = record.get("file_count", "-")
        note = record.get("error") or record.get("note") or ""
        lines.append(
            f"| {status} | `{record['repo_id']}` | {size} | {files} | `{record['local_dir']}` | {note} |"
        )
    summary_path.write_text("\n".join(lines) + "\n")

    print(f"manifest: {manifest_path}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    print(f"completed={len(completed)} failed={len(failed)} total={human_size(total_bytes)}", flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
