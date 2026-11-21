#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "keyword_external_wordlists.json"


def _read_manifest(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("wordlists"), list):
        raise ValueError(f"{path} must be a dict with a wordlists list")
    return obj


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str, *, timeout_s: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "FilterGuard-wordlist-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=int(timeout_s)) as resp:  # noqa: S310 - user-controlled manifest is local.
        if getattr(resp, "status", 200) >= 400:
            raise RuntimeError(f"download failed with HTTP {resp.status}: {url}")
        return resp.read()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download third-party sensitive/profanity wordlists without printing terms. "
            "Only metadata and hashes are written to stdout/manifest."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--only-enabled", action="store_true", help="Skip disabled manifest entries")
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest and print metadata without downloading")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = _read_manifest(manifest_path)
    report: list[dict[str, Any]] = []
    for entry in manifest["wordlists"]:
        if not isinstance(entry, dict):
            continue
        enabled = bool(entry.get("enabled", True))
        if bool(args.only_enabled) and not enabled:
            continue
        url = str(entry.get("url") or "").strip()
        local_path = str(entry.get("local_path") or "").strip()
        if not local_path:
            raise ValueError(f"wordlist entry {entry.get('id')!r} is missing local_path")

        out_path = (ROOT / local_path).resolve()
        item = {
            "id": str(entry.get("id") or ""),
            "enabled": enabled,
            "category": str(entry.get("category") or ""),
            "source": str(entry.get("source") or ""),
            "local_path": str(out_path.relative_to(ROOT)),
            "downloaded": False,
        }
        if url:
            item["url"] = url
            if not bool(args.dry_run):
                data = _download(url, timeout_s=int(args.timeout_s))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(data)
                item.update(
                    {
                        "downloaded": True,
                        "bytes": len(data),
                        "sha256": _sha256_bytes(data),
                    }
                )
        else:
            if out_path.exists():
                data = out_path.read_bytes()
                item.update(
                    {
                        "downloaded": False,
                        "bytes": len(data),
                        "sha256": _sha256_bytes(data),
                    }
                )
            else:
                item["missing"] = True
        report.append(item)

    report_path = ROOT / "configs" / "external_sensitive_wordlists" / "download_report.json"
    if not bool(args.dry_run):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
