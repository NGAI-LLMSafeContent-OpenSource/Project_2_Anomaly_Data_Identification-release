#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from filterguard.data import labels_texts, load_rows, resolve_dev_eval_indices
from filterguard.keyword import (
    KeywordRule,
    compile_rules,
    keyword_score,
    keyword_score_with_hits,
    load_rules_config,
    load_wordlist,
    load_wordlist_manifest,
    pick_keyword_threshold,
)
from filterguard.metric import confusion, extract_text, metrics_from_conf


_DEFAULT_RULES_CONFIG = Path("configs/keyword_rules_base.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword scoring baseline for abnormal data detection")
    parser.add_argument(
        "--dataset-path",
        default="datasets/generated_datasets/v0.1.1/dataset_hf",
        help="Path to a HF dataset saved by Dataset.save_to_disk(), or a supported .jsonl file",
    )
    parser.add_argument(
        "--rules-config",
        default=str(_DEFAULT_RULES_CONFIG),
        help="Path to JSON rules config (see configs/keyword_rules_base.json)",
    )
    parser.add_argument("--no-base-rules", action="store_true", help="Disable base rules from --rules-config")
    parser.add_argument(
        "--extra-wordlist",
        action="append",
        default=[],
        help="Extra wordlist path (txt, one keyword per line). Repeatable.",
    )
    parser.add_argument(
        "--extra-wordlist-manifest",
        action="append",
        default=[],
        help="Manifest JSON containing optional external wordlists. Repeatable.",
    )
    parser.add_argument(
        "--extra-match",
        choices=["auto", "word", "substring"],
        default="auto",
        help="Match strategy for --extra-wordlist",
    )
    parser.add_argument("--extra-weight", type=int, default=1, help="Weight for each --extra-wordlist keyword")
    parser.add_argument("--include-assistant", action="store_true", help="Score on full dialogue (user+assistant)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Dev split ratio for threshold tuning")
    parser.add_argument("--threshold", type=int, default=None, help="Fixed threshold (skip tuning if set)")
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N examples (0 means all)")
    parser.add_argument("--show-errors", type=int, default=0, help="Print N false positives/negatives for debugging")
    args = parser.parse_args()

    base_rules: list[KeywordRule] = []
    if not bool(args.no_base_rules):
        base_rules = load_rules_config(Path(args.rules_config))
    extra_rules: list[KeywordRule] = []
    for p in args.extra_wordlist:
        extra_rules.extend(
            load_wordlist(Path(p), weight=int(args.extra_weight), match=str(args.extra_match).lower())
        )
    for p in args.extra_wordlist_manifest:
        extra_rules.extend(load_wordlist_manifest(Path(p)))
    rules = base_rules + extra_rules
    compiled = compile_rules(rules)

    rows = load_rows(str(args.dataset_path), limit=int(args.limit))
    labels, texts = labels_texts(rows, include_assistant=bool(args.include_assistant))

    scores: list[int] = []
    hits_list: list[list[str]] = []
    if int(args.show_errors) > 0:
        for t in texts:
            s, hits = keyword_score_with_hits(t, compiled)
            scores.append(int(s))
            hits_list.append(hits)
    else:
        for t in texts:
            scores.append(int(keyword_score(t, compiled)))

    dev_idx, eval_idx, split_mode = resolve_dev_eval_indices(
        rows,
        labels,
        dev_ratio=float(args.dev_ratio),
        seed=int(args.seed),
    )
    if args.threshold is None:
        dev_scores = [scores[i] for i in dev_idx] if dev_idx else scores
        dev_labels = [labels[i] for i in dev_idx] if dev_idx else labels
        thr, dev_acc = pick_keyword_threshold(dev_scores, dev_labels)
    else:
        thr = int(args.threshold)
        dev_acc = float("nan")

    def eval_on(indices: list[int]) -> dict[str, Any]:
        if not indices:
            return {"n": 0}
        y_true = [labels[i] for i in indices]
        y_pred = [1 if scores[i] >= thr else 0 for i in indices]
        conf = confusion(y_true, y_pred)
        m = metrics_from_conf(conf)
        return {"n": len(indices), "conf": conf, "metrics": m}

    res_eval = eval_on(eval_idx if eval_idx else list(range(len(labels))))

    pos_rate = sum(labels) / len(labels) if labels else float("nan")
    print(f"dataset_path={args.dataset_path}")
    print(
        f"rules_config={args.rules_config} "
        f"base_rules={len(base_rules)} extra_wordlists={len(args.extra_wordlist)} extra_rules={len(extra_rules)} "
        f"total_rules={len(rules)}"
    )
    print(f"n={len(labels)} pos_rate={pos_rate:.4f} split_mode={split_mode}")
    print(f"threshold={thr} dev_acc={dev_acc if not math.isnan(dev_acc) else 'NA'}")
    print(f"eval_n={res_eval['n']} eval_acc={res_eval['metrics']['acc']:.4f}")
    print(
        "eval_confusion="
        f"tp:{res_eval['conf']['tp']} tn:{res_eval['conf']['tn']} "
        f"fp:{res_eval['conf']['fp']} fn:{res_eval['conf']['fn']}"
    )
    print(
        "eval_metrics="
        f"precision:{res_eval['metrics']['precision']:.4f} "
        f"recall:{res_eval['metrics']['recall']:.4f} "
        f"f1:{res_eval['metrics']['f1']:.4f}"
    )

    if int(args.show_errors) > 0:
        fp: list[int] = []
        fn: list[int] = []
        for i in eval_idx if eval_idx else range(len(labels)):
            pred = 1 if scores[i] >= thr else 0
            if labels[i] == 0 and pred == 1:
                fp.append(i)
            elif labels[i] == 1 and pred == 0:
                fn.append(i)

        def show(title: str, idxs: list[int]) -> None:
            print(f"\n{title} (showing {min(args.show_errors, len(idxs))}/{len(idxs)})")
            for j in idxs[: int(args.show_errors)]:
                ex = rows[j]
                text = extract_text(ex, include_assistant=bool(args.include_assistant))
                head = text.replace("\n", "\\n")[:200]
                print(
                    f"- i={j} score={scores[j]} label={labels[j]} source={ex.get('source')} hits={hits_list[j]}\n"
                    f"  text_head={head}"
                )

        show("FalsePos", fp)
        show("FalseNeg", fn)


if __name__ == "__main__":
    main()
