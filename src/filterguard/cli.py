from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from filterguard.metric import extract_text
from filterguard.server import serve
from filterguard.service import (
    Analyzer,
    load_keyword_analyzer,
    load_neuron_persona_analyzer,
    load_onion_analyzer,
    result_to_dict,
)


DEFAULT_RULES_CONFIG = "configs/keyword_rules_base.json"


def _add_common_analyzer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--methods",
        default="keyword",
        help="Comma-separated methods to load: keyword, neuron, onion. predict uses the first method unless --method is set.",
    )

    parser.add_argument("--keyword-rules-config", default=DEFAULT_RULES_CONFIG)
    parser.add_argument("--keyword-no-base-rules", action="store_true")
    parser.add_argument("--keyword-extra-wordlist", action="append", default=[])
    parser.add_argument("--keyword-extra-wordlist-manifest", action="append", default=[])
    parser.add_argument("--keyword-extra-match", choices=["auto", "word", "substring"], default="auto")
    parser.add_argument("--keyword-extra-weight", type=int, default=1)
    parser.add_argument("--keyword-threshold", type=int, default=1)

    parser.add_argument("--onion-model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--onion-trust-remote-code", action="store_true")
    parser.add_argument("--onion-calibration-dataset", default="")
    parser.add_argument("--onion-include-assistant", action="store_true")
    parser.add_argument("--onion-threshold", type=float, default=None)
    parser.add_argument("--onion-direction", choices=["auto", "high", "low"], default="auto")
    parser.add_argument("--onion-batch-size", type=int, default=8)
    parser.add_argument("--onion-deletion-batch-size", type=int, default=16)
    parser.add_argument("--onion-max-length", type=int, default=256)
    parser.add_argument("--onion-device", default="auto")
    parser.add_argument("--onion-dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--onion-score-mode", choices=["max_delta", "topk_mean_delta"], default="max_delta")
    parser.add_argument("--onion-top-k", type=int, default=5)
    parser.add_argument("--onion-candidates", type=int, default=32)
    parser.add_argument("--onion-delta-space", choices=["log_ppl", "ppl"], default="log_ppl")
    parser.add_argument("--onion-eps", type=float, default=1e-6)
    parser.add_argument("--onion-max-pending-deletions", type=int, default=256)

    parser.add_argument("--neuron-calibration-dataset", default="")
    parser.add_argument("--neuron-model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--neuron-trust-remote-code", action="store_true")
    parser.add_argument("--neuron-include-assistant", action="store_true")
    parser.add_argument("--neuron-seed", type=int, default=42)
    parser.add_argument("--neuron-dev-ratio", type=float, default=0.2)
    parser.add_argument("--neuron-pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--neuron-layer", type=int, default=-1)
    parser.add_argument("--neuron-feature-normalize", choices=["none", "l2"], default="none")
    parser.add_argument("--neuron-eps", type=float, default=1e-6)
    parser.add_argument("--neuron-direction", choices=["auto", "high", "low"], default="auto")
    parser.add_argument("--neuron-threshold", type=float, default=None)
    parser.add_argument("--neuron-batch-size", type=int, default=8)
    parser.add_argument("--neuron-max-length", type=int, default=256)
    parser.add_argument("--neuron-device", default="auto")
    parser.add_argument("--neuron-dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--neuron-backbone", choices=["base", "causal_lm"], default="base")
    parser.add_argument("--neuron-stop-after-layer", action=argparse.BooleanOptionalAction, default=True)


def _parse_methods(value: str) -> list[str]:
    methods = [part.strip() for part in str(value).split(",") if part.strip()]
    if not methods:
        raise ValueError("--methods must include at least one method")
    unknown = [m for m in methods if m not in {"keyword", "neuron", "onion"}]
    if unknown:
        raise ValueError(f"unsupported methods: {unknown}")
    return methods


def build_analyzers(args: argparse.Namespace) -> dict[str, Analyzer]:
    analyzers: dict[str, Analyzer] = {}
    for method in _parse_methods(str(args.methods)):
        if method == "keyword":
            analyzers[method] = load_keyword_analyzer(
                rules_config=str(args.keyword_rules_config),
                no_base_rules=bool(args.keyword_no_base_rules),
                extra_wordlists=list(args.keyword_extra_wordlist),
                extra_wordlist_manifests=list(args.keyword_extra_wordlist_manifest),
                extra_match=str(args.keyword_extra_match),
                extra_weight=int(args.keyword_extra_weight),
                threshold=int(args.keyword_threshold),
            )
        elif method == "onion":
            analyzers[method] = load_onion_analyzer(
                model_id=str(args.onion_model_id),
                trust_remote_code=bool(args.onion_trust_remote_code),
                device=str(args.onion_device),
                dtype=str(args.onion_dtype),
                threshold=args.onion_threshold,
                direction=str(args.onion_direction),
                calibration_dataset=str(args.onion_calibration_dataset),
                include_assistant=bool(args.onion_include_assistant),
                batch_size=int(args.onion_batch_size),
                deletion_batch_size=int(args.onion_deletion_batch_size),
                max_length=int(args.onion_max_length),
                max_candidates=int(args.onion_candidates),
                score_mode=str(args.onion_score_mode),
                top_k=int(args.onion_top_k),
                delta_space=str(args.onion_delta_space),
                eps=float(args.onion_eps),
                max_pending_deletions=int(args.onion_max_pending_deletions),
            )
        elif method == "neuron":
            analyzers[method] = load_neuron_persona_analyzer(
                calibration_dataset=str(args.neuron_calibration_dataset),
                model_id=str(args.neuron_model_id),
                trust_remote_code=bool(args.neuron_trust_remote_code),
                include_assistant=bool(args.neuron_include_assistant),
                seed=int(args.neuron_seed),
                dev_ratio=float(args.neuron_dev_ratio),
                pooling=str(args.neuron_pooling),
                layer=int(args.neuron_layer),
                feature_normalize=str(args.neuron_feature_normalize),
                eps=float(args.neuron_eps),
                direction=str(args.neuron_direction),
                threshold=args.neuron_threshold,
                batch_size=int(args.neuron_batch_size),
                max_length=int(args.neuron_max_length),
                device=str(args.neuron_device),
                dtype=str(args.neuron_dtype),
                backbone=str(args.neuron_backbone),
                stop_after_layer=bool(args.neuron_stop_after_layer),
            )
    return analyzers


def _texts_from_jsonl(path: str, *, include_assistant: bool) -> list[str]:
    texts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path} line {line_no} must be a JSON object")
            if "text" in obj:
                texts.append(str(obj["text"]))
            else:
                texts.append(extract_text(obj, include_assistant=include_assistant))
    return texts


def _read_predict_texts(args: argparse.Namespace) -> list[str]:
    texts = [str(t) for t in args.text]
    if args.input_jsonl:
        texts.extend(_texts_from_jsonl(str(args.input_jsonl), include_assistant=bool(args.include_assistant)))
    if not texts and not sys.stdin.isatty():
        stdin_text = sys.stdin.read()
        if stdin_text:
            texts.append(stdin_text)
    if not texts:
        raise ValueError("provide --text, --input-jsonl, or stdin text")
    return texts


def _run_predict(args: argparse.Namespace) -> int:
    analyzers = build_analyzers(args)
    method = str(args.method or next(iter(analyzers)))
    if method not in analyzers:
        raise ValueError(f"method {method!r} was not loaded; set --methods to include it")
    texts = _read_predict_texts(args)
    results = analyzers[method].predict(texts)
    if bool(args.jsonl):
        for i, result in enumerate(results):
            print(json.dumps({"index": i, **result_to_dict(result)}, ensure_ascii=False, sort_keys=True))
    else:
        payload = {
            "method": method,
            "count": len(results),
            "results": [{"index": i, **result_to_dict(result)} for i, result in enumerate(results)],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    analyzers = build_analyzers(args)
    serve(analyzers=analyzers, host=str(args.host), port=int(args.port))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FilterGuard CLI and HTTP API")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="Run one-shot abnormal data detection")
    _add_common_analyzer_args(predict)
    predict.add_argument("--method", choices=["keyword", "neuron", "onion"], default="")
    predict.add_argument("--text", action="append", default=[], help="Text to classify. Repeatable.")
    predict.add_argument("--input-jsonl", default="", help="JSONL file with text or dataset-style rows.")
    predict.add_argument("--include-assistant", action="store_true", help="For --input-jsonl dataset-style rows.")
    predict.add_argument("--jsonl", action="store_true", help="Write one JSON result per line.")
    predict.set_defaults(func=_run_predict)

    serve_parser = sub.add_parser("serve", help="Start the HTTP API")
    _add_common_analyzer_args(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=_run_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        print(f"filterguard: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

