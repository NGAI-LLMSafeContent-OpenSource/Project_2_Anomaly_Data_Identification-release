from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KeywordRule:
    pattern: str
    weight: int
    match: str = "substring"  # substring | word | regex


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SIMPLE_WORD_RE = re.compile(r"^[a-z0-9]+$")


def auto_match_type(pattern: str) -> str:
    p = (pattern or "").strip().lower()
    if _SIMPLE_WORD_RE.fullmatch(p):
        return "word"
    return "substring"


def load_rules_config(path: Path) -> list[KeywordRule]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    raw_rules = obj.get("rules") if isinstance(obj, dict) else obj
    if not isinstance(raw_rules, list):
        raise ValueError(f"{path} must be a list, or a dict with a 'rules' list.")
    rules: list[KeywordRule] = []
    for r in raw_rules:
        if not isinstance(r, dict):
            continue
        pattern = str(r.get("pattern") or "").strip()
        if not pattern:
            continue
        weight = int(r.get("weight", 1))
        match = str(r.get("match") or "substring").strip().lower()
        if match not in {"substring", "word", "regex"}:
            raise ValueError(f"{path} has unsupported match type: {match!r}")
        rules.append(KeywordRule(pattern=pattern, weight=weight, match=match))
    return rules


def load_wordlist(path: Path, *, weight: int, match: str) -> list[KeywordRule]:
    rules: list[KeywordRule] = []
    match = str(match).strip().lower()
    if match not in {"auto", "substring", "word", "regex"}:
        raise ValueError(f"unsupported match type: {match!r}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        resolved_match = match
        if match == "auto":
            resolved_match = auto_match_type(line)
        rules.append(KeywordRule(pattern=line, weight=int(weight), match=str(resolved_match)))
    return rules


def load_wordlist_manifest(path: Path, *, repo_root: Path | None = None) -> list[KeywordRule]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    entries = obj.get("wordlists") if isinstance(obj, dict) else obj
    if not isinstance(entries, list):
        raise ValueError(f"{path} must be a list, or a dict with a 'wordlists' list.")

    root = repo_root
    if root is None:
        root = path.parent.parent if path.parent.name == "configs" else path.parent

    rules: list[KeywordRule] = []
    missing: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not bool(entry.get("enabled", True)):
            continue
        local_path = str(entry.get("local_path") or "").strip()
        if not local_path:
            raise ValueError(f"{path} has an enabled wordlist entry without local_path")
        wordlist_path = Path(local_path)
        if not wordlist_path.is_absolute():
            wordlist_path = root / wordlist_path
        if not wordlist_path.exists():
            missing.append(str(wordlist_path))
            continue
        weight = int(entry.get("weight", 1))
        match = str(entry.get("match") or "auto").strip().lower()
        rules.extend(load_wordlist(wordlist_path, weight=weight, match=match))
    if missing:
        raise FileNotFoundError(
            "Missing enabled external wordlist files. Run "
            "`python scripts/download_sensitive_wordlists.py --only-enabled` first, or disable entries: "
            + "; ".join(missing)
        )
    return rules


@dataclass(frozen=True)
class CompiledRule:
    weight: int
    match: str  # substring | word | regex
    pattern: str
    pattern_lower: str
    regex: re.Pattern[str] | None = None
    is_simple_word: bool = False


def compile_rules(rules: list[KeywordRule]) -> list[CompiledRule]:
    compiled: list[CompiledRule] = []
    for r in rules:
        pattern = str(r.pattern or "")
        pattern_lower = pattern.lower()
        match = str(r.match or "substring").strip().lower()
        if match not in {"substring", "word", "regex"}:
            raise ValueError(f"unsupported match type: {match!r}")
        if match == "regex":
            compiled.append(
                CompiledRule(
                    weight=int(r.weight),
                    match=match,
                    pattern=pattern,
                    pattern_lower=pattern_lower,
                    regex=re.compile(pattern_lower),
                )
            )
        else:
            compiled.append(
                CompiledRule(
                    weight=int(r.weight),
                    match=match,
                    pattern=pattern,
                    pattern_lower=pattern_lower,
                    is_simple_word=bool(_SIMPLE_WORD_RE.fullmatch(pattern_lower)),
                )
            )
    return compiled


def keyword_score(text: str, rules: list[CompiledRule]) -> int:
    text_lower = text.lower()
    score = 0
    tokens: set[str] | None = None
    for rule in rules:
        matched = False
        if rule.match == "substring":
            matched = rule.pattern_lower in text_lower
        elif rule.match == "word":
            if rule.is_simple_word:
                if tokens is None:
                    tokens = set(_TOKEN_RE.findall(text_lower))
                matched = rule.pattern_lower in tokens
            else:
                matched = rule.pattern_lower in text_lower
        else:
            matched = rule.regex.search(text_lower) is not None if rule.regex is not None else False

        if matched:
            score += int(rule.weight)
    return int(score)


def keyword_score_with_hits(text: str, rules: list[CompiledRule]) -> tuple[int, list[str]]:
    text_lower = text.lower()
    score = 0
    hits: list[str] = []
    tokens: set[str] | None = None
    for rule in rules:
        matched = False
        if rule.match == "substring":
            matched = rule.pattern_lower in text_lower
        elif rule.match == "word":
            if rule.is_simple_word:
                if tokens is None:
                    tokens = set(_TOKEN_RE.findall(text_lower))
                matched = rule.pattern_lower in tokens
            else:
                matched = rule.pattern_lower in text_lower
        else:
            matched = rule.regex.search(text_lower) is not None if rule.regex is not None else False

        if matched:
            score += int(rule.weight)
            hits.append(str(rule.pattern))
    return int(score), hits


def pick_keyword_threshold(scores: list[int], labels: list[int]) -> tuple[int, float]:
    from filterguard.metric import confusion, metrics_from_conf

    if not scores:
        return 1, float("nan")
    max_score = max(scores)
    candidates = sorted(set(scores + [0, max_score + 1]))
    best_thr = candidates[0]
    best_acc = -1.0
    for thr in candidates:
        preds = [1 if s >= thr else 0 for s in scores]
        acc = float(metrics_from_conf(confusion(labels, preds))["acc"])
        if acc > best_acc or (math.isclose(acc, best_acc) and thr > best_thr):
            best_acc = acc
            best_thr = thr
    return int(best_thr), float(best_acc)
