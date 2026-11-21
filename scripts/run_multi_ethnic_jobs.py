#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from filterguard.multi_ethnic import load_json  # noqa: E402


def _load_jobs(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
                    "You are a structured data generator. "
                    "Always follow the requested JSON schema exactly."
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-ethnic benchmark jobs via OpenAI-compatible API")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "multi_ethnic_benchmark_v1.template.json"),
        help="Path to config json",
    )
    parser.add_argument("--jobs-path", required=True, help="Path to input jobs jsonl")
    parser.add_argument(
        "--provider",
        default="text_generation",
        help="Provider key under config.providers, e.g. text_generation or scene_planning",
    )
    parser.add_argument("--output-path", required=True, help="Where to write response jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Only run first N jobs (0 means all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output path if it exists")
    args = parser.parse_args()

    cfg = load_json(args.config)
    provider = (cfg.get("providers") or {}).get(args.provider) or {}
    base_url = str(provider.get("base_url") or "").strip()
    api_key = str(provider.get("api_key") or "").strip()
    model = str(provider.get("model") or "").strip()
    temperature = float(provider.get("temperature") or 0.0)
    max_tokens = int(provider.get("max_tokens") or 1200)

    if not base_url or not api_key or not model:
        raise SystemExit(
            f"Provider '{args.provider}' is incomplete. Fill base_url/api_key/model in {args.config} first."
        )

    out_path = Path(args.output_path)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out_path}. Pass --overwrite to replace it.")

    jobs = _load_jobs(args.jobs_path)
    if args.limit > 0:
        jobs = jobs[: args.limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for job in jobs:
            try:
                raw = _chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    prompt=str(job["prompt"]),
                )
                content = raw["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                row = {
                    "job_id": job["job_id"],
                    "task": job["task"],
                    "provider": args.provider,
                    "model": model,
                    "seed_id": job["seed_id"],
                    "raw_response": raw,
                    "parsed_response": parsed,
                }
            except (KeyError, IndexError, json.JSONDecodeError, urllib.error.URLError) as e:
                row = {
                    "job_id": job.get("job_id"),
                    "task": job.get("task"),
                    "provider": args.provider,
                    "model": model,
                    "seed_id": job.get("seed_id"),
                    "error": str(e),
                }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(jobs)} results to {out_path}")


if __name__ == "__main__":
    main()
