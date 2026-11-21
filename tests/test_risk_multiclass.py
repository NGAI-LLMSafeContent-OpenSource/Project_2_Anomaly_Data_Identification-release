from __future__ import annotations

import unittest

import torch

from filterguard.risk import (
    apply_one_vs_rest_margins,
    abnormal_binary_metrics_from_risk,
    fit_one_vs_rest_thresholds,
    fit_one_vs_rest_risk_persona,
    multiclass_metrics,
    ordered_risk_classes,
    predict_risk_labels,
    risk_labels,
    risk_scores_from_params,
)


class RiskMulticlassTest(unittest.TestCase):
    def test_risk_labels_use_normal_and_configured_l1(self) -> None:
        rows = [
            {"is_polluted": False, "is_poisoned": False, "text": "normal"},
            {"is_polluted": True, "is_poisoned": False, "harm_category_L1": "L1-1 CBRNE", "text": "bad"},
            {
                "is_polluted": True,
                "is_poisoned": False,
                "harm_category_L1": "L1-3 Implicit Bias Contamination",
                "text": "bias",
            },
        ]
        self.assertEqual(
            risk_labels(rows, risk_level="l1"),
            ["normal", "L1-1 CBRNE", "L1-3 Implicit Bias Contamination"],
        )
        self.assertEqual(
            ordered_risk_classes(risk_labels(rows, risk_level="l1")),
            ["normal", "L1-1 CBRNE", "L1-3 Implicit Bias Contamination"],
        )

    def test_one_vs_rest_persona_scores_and_predicts(self) -> None:
        classes = ["normal", "risk_a", "risk_b"]
        labels = ["normal", "normal", "risk_a", "risk_a", "risk_b", "risk_b"]
        features = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.1, 0.9],
            ]
        )
        params = fit_one_vs_rest_risk_persona(
            features,
            labels,
            classes=classes,
            feature_normalize="l2",
            eps=1e-6,
        )
        scores = risk_scores_from_params(features, params)
        self.assertEqual(tuple(scores.shape), (6, 3))
        self.assertEqual(predict_risk_labels(scores, classes), labels)
        self.assertEqual(params.class_support, {"normal": 2, "risk_a": 2, "risk_b": 2})

        thresholds = fit_one_vs_rest_thresholds(scores, labels, classes=classes)
        margins = apply_one_vs_rest_margins(scores, classes, thresholds)
        self.assertEqual(tuple(margins.shape), (6, 3))
        self.assertEqual(predict_risk_labels(margins, classes), labels)

    def test_multiclass_and_abnormal_metrics_report_support(self) -> None:
        classes = ["normal", "risk_a", "risk_b"]
        y_true = ["normal", "normal", "risk_a", "risk_a", "risk_b"]
        y_pred = ["normal", "risk_a", "risk_a", "risk_b", "risk_b"]
        metrics = multiclass_metrics(y_true, y_pred, classes=classes)
        self.assertAlmostEqual(metrics["acc"], 3 / 5)
        self.assertEqual(metrics["per_class"]["risk_a"]["support"], 2)
        self.assertEqual(metrics["per_class"]["risk_b"]["pred_count"], 2)

        abnormal = abnormal_binary_metrics_from_risk(y_true, y_pred)
        self.assertEqual(abnormal["conf"], {"tp": 3, "tn": 1, "fp": 1, "fn": 0})
        self.assertAlmostEqual(abnormal["metrics"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
