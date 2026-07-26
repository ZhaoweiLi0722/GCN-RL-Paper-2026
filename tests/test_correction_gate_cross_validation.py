from __future__ import annotations

import unittest

import numpy as np

from evaluation.cross_validate_correction_gate import (
    binary_class_weights,
    binary_roc_auc,
    threshold_metrics,
)


class CorrectionGateCrossValidationTests(unittest.TestCase):
    def test_binary_auc_and_threshold_metrics(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.float32)
        scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

        self.assertAlmostEqual(binary_roc_auc(labels, scores), 1.0)
        metrics = threshold_metrics(labels, scores, 0.5)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)

    def test_binary_weights_allocate_requested_positive_mass(self) -> None:
        labels = np.asarray([0, 0, 0, 1], dtype=np.float32)

        weights = binary_class_weights(labels, positive_mass=0.25)

        self.assertAlmostEqual(float(weights[labels > 0.5].sum()), 1.0)
        self.assertAlmostEqual(float(weights[labels < 0.5].sum()), 3.0)


if __name__ == "__main__":
    unittest.main()
