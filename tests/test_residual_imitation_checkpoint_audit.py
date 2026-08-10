"""Tests for residual checkpoint fit diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.audit_residual_imitation_checkpoint import (
    gate_calibration_summary,
    residual_fit_summary,
)


class ResidualImitationCheckpointAuditTests(unittest.TestCase):
    def test_gate_calibration_reports_group_specific_thresholds(self) -> None:
        labels = np.asarray(
            [[1, 0], [1, 0], [0, 0], [0, 0]],
            dtype=bool,
        )
        probabilities = np.asarray(
            [[0.9, 0.4], [0.6, 0.3], [0.4, 0.2], [0.1, 0.1]],
            dtype=np.float32,
        )

        result = gate_calibration_summary(
            labels,
            probabilities,
            groups=("specimen", "reagent"),
            thresholds=(0.3, 0.5, 0.6),
        )

        specimen = result["by_group"]["specimen"]
        reagent = result["by_group"]["reagent"]
        self.assertEqual(specimen["label_rate"], 0.5)
        self.assertEqual(
            specimen["prevalence_matched"]["prediction_rate"],
            0.5,
        )
        self.assertAlmostEqual(
            specimen["best_grid_f1"]["threshold"],
            0.6,
        )
        self.assertEqual(reagent["label_rate"], 0.0)
        self.assertEqual(
            reagent["prevalence_matched"]["prediction_rate"],
            0.0,
        )

    def test_summary_separates_group_false_positives_and_recall(self) -> None:
        target = np.zeros((2, 8), dtype=np.float32)
        prediction = np.zeros_like(target)
        target[0, :2] = (-1.0, 1.0)
        prediction[0, :2] = (-0.5, 0.5)
        prediction[1, 2:4] = (0.2, -0.2)

        result = residual_fit_summary(
            target,
            prediction,
            num_facilities=2,
            activity_threshold=0.04,
        )

        specimen = result["by_group"]["specimen_transfer"]
        reagent = result["by_group"]["reagent_transfer"]
        self.assertEqual(specimen["target_active_row_fraction"], 0.5)
        self.assertEqual(specimen["changed_dimension_recall"], 1.0)
        self.assertEqual(specimen["changed_dimension_sign_accuracy"], 1.0)
        self.assertAlmostEqual(specimen["changed_dimension_abs_ratio"], 0.5)
        self.assertEqual(reagent["target_active_row_fraction"], 0.0)
        self.assertEqual(reagent["prediction_active_row_fraction"], 0.5)
        self.assertEqual(reagent["false_positive_row_fraction"], 0.5)

    def test_summary_rejects_non_facility_net_width(self) -> None:
        with self.assertRaises(ValueError):
            residual_fit_summary(
                np.zeros((1, 7), dtype=np.float32),
                np.zeros((1, 7), dtype=np.float32),
                num_facilities=2,
                activity_threshold=0.04,
            )


if __name__ == "__main__":
    unittest.main()
