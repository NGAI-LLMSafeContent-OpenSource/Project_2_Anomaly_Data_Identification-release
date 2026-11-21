from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from filterguard.data import abnormal_label, load_rows
from filterguard.token_anomaly import (
    TokenModelInfo,
    TokenScoreDetail,
    _candidate_positions_from_token_nll,
    _nll_distribution_features,
    _score_nll_spike_from_features,
    validate_token_anomaly_args,
)
from scripts.build_seed_fineweb_mixed_dataset import _clip_text_to_token_len, _length_summary
from token_anomaly_baseline_abnormal import _save_artifacts


ROOT = Path(__file__).resolve().parents[1]


class WhitespaceTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        pieces = text.split()
        if kwargs.get("truncation") and kwargs.get("max_length"):
            pieces = pieces[: int(kwargs["max_length"])]
        return {"input_ids": list(range(len(pieces)))}

    def decode(self, ids: list[int], **_kwargs: object) -> str:
        return " ".join(f"tok{i}" for i in ids)


class TokenAnomalyWorkflowTest(unittest.TestCase):
    def test_candidate_positions_are_bounded_and_skip_specials(self) -> None:
        positions = _candidate_positions_from_token_nll(
            sequence=[101, 7, 8, 102, 9],
            token_nll=[None, 0.5, 8.0, 9.0, 1.0],
            special_token_ids={101, 102},
            max_candidates=2,
            eps=1e-6,
        )
        self.assertEqual(positions, [2, 4])

    def test_validate_rejects_invalid_onion_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_pending_deletions"):
            validate_token_anomaly_args(
                batch_size=2,
                max_length=16,
                score_mode="max_delta",
                top_k=1,
                eps=1e-6,
                method="onion_delta",
                deletion_batch_size=4,
                max_candidates=8,
                max_pending_deletions=0,
                delta_space="log_ppl",
            )

    def test_nll_distribution_features_include_length_normalized_spike(self) -> None:
        features = _nll_distribution_features([1.0, 1.0, 3.0, 5.0], top_k=2, eps=1e-6)
        self.assertEqual(features["token_count"], 4.0)
        self.assertAlmostEqual(features["mean_nll"], 2.5)
        self.assertAlmostEqual(features["topk_mean_nll"], 4.0)
        self.assertAlmostEqual(features["topk_gap_nll"], 1.5)
        self.assertGreater(features["max_z"], features["max_z_len_norm"])
        self.assertEqual(_score_nll_spike_from_features(features, score_mode="token_count"), 4.0)
        self.assertEqual(_score_nll_spike_from_features(features, score_mode="mean_nll"), features["mean_nll"])
        self.assertEqual(_score_nll_spike_from_features(features, score_mode="topk_gap_nll"), features["topk_gap_nll"])

    def test_length_control_clips_to_target_token_count(self) -> None:
        tokenizer = WhitespaceTokenizer()
        clipped, original_tokens, clipped_tokens = _clip_text_to_token_len(
            tokenizer,
            "one two three four five six seven eight",
            target_tokens=5,
            max_tokens=16,
        )
        self.assertEqual(original_tokens, 8)
        self.assertEqual(clipped_tokens, 5)
        self.assertEqual(len(clipped.split()), 5)

        summary = _length_summary([2, 4, 6, 8])
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["min"], 2)
        self.assertEqual(summary["max"], 8)
        self.assertEqual(summary["median"], 6)

    def test_build_seed_fineweb_mixed_dataset_from_local_normals(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            normal_path = tmp / "normals.jsonl"
            normal_rows = [
                {"id": f"n{i}", "text": f"This is a normal educational web document with enough text for row {i}. " * 3}
                for i in range(8)
            ]
            normal_path.write_text("\n".join(json.dumps(row) for row in normal_rows) + "\n", encoding="utf-8")
            output_path = tmp / "mixed.jsonl"

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "build_seed_fineweb_mixed_dataset.py"),
                "--seed-path",
                str(ROOT / "datasets" / "third_datasets" / "seeds_filtered_fullL1.jsonl"),
                "--local-normal-jsonl",
                str(normal_path),
                "--output-path",
                str(output_path),
                "--seed-count",
                "4",
                "--fineweb-count",
                "4",
                "--seed",
                "123",
                "--dev-ratio",
                "0.25",
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src")
            subprocess.run(cmd, cwd=ROOT, check=True, env=env)

            rows = load_rows(str(output_path), limit=0)
            labels = [abnormal_label(row) for row in rows]
            self.assertEqual(len(rows), 8)
            self.assertEqual(sum(labels), 4)
            self.assertEqual(len(labels) - sum(labels), 4)
            self.assertEqual(sum(1 for row in rows if row.get("split") == "dev"), 2)
            self.assertEqual(sum(1 for row in rows if row.get("split") == "test"), 6)
            self.assertTrue(output_path.with_suffix(".jsonl.manifest.json").exists())

    def test_save_artifacts_writes_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(
                run_id="unit",
                dataset_path="unit.jsonl",
                method="nll_spike",
                nll_score_mode="max_z",
                onion_score_mode="max_delta",
                top_k=1,
                onion_candidates=2,
                onion_delta_space="log_ppl",
                onion_max_pending_deletions=8,
                output_dir=td,
            )
            run_dir = _save_artifacts(
                output_dir=td,
                args=args,
                rows=[{"id": "a", "source": "seed"}, {"id": "b", "source": "fineweb"}],
                labels=[1, 0],
                texts=["bad text", "normal text"],
                scores=[2.0, 0.1],
                details=[
                    TokenScoreDetail(token_count=2, mean_nll=1.5, max_z_len_norm=0.8),
                    TokenScoreDetail(token_count=2, mean_nll=0.5, max_z_len_norm=0.2),
                ],
                predictions=[1, 0],
                dev_idx=[0],
                eval_idx=[1],
                split_mode="explicit",
                threshold=1.0,
                direction="high",
                dev_acc=1.0,
                res_eval={"n": 1, "conf": {"tp": 0, "tn": 1, "fp": 0, "fn": 0}, "metrics": {"acc": 1.0}},
                pos_rate=0.5,
                scoring_desc="method:nll_spike score_mode:max_z top_k:1",
                model_info=TokenModelInfo(model_id="unit", device="cpu", dtype="fp32", pad_token_id=0),
                text_scope="user",
                elapsed_s=0.01,
            )
            for name in [
                "run_config.json",
                "raw_scores.jsonl",
                "parsed_outputs.jsonl",
                "per_sample_results.jsonl",
                "aggregate_metrics.json",
            ]:
                self.assertTrue((run_dir / name).exists(), name)
            metrics = json.loads((run_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["status_counts"], {"success": 2})
            raw = (run_dir / "raw_scores.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(raw[0])["score_detail"]["mean_nll"], 1.5)


if __name__ == "__main__":
    unittest.main()
