import json
import unittest

import numpy as np

from evaluation.diagnose_saved_critic import (
    component_contrast, coverage, pair_diagnostic, pair_summary,
)


class SavedCriticDiagnosisTests(unittest.TestCase):
    def test_split_sign_categories(self):
        for draws, group in (([1] * 8, "same_sign"), ([1] * 4 + [-1] * 4, "opposite_sign"),
                             ([0] * 8, "both_tied"), ([0] * 4 + [1] * 4, "one_tied")):
            self.assertEqual(pair_diagnostic(draws, 1)["group"], group)

    def test_pair_se_retains_covariance(self):
        common = np.arange(8) * 100.0
        row = pair_diagnostic((common + 2) - common, 1)
        self.assertEqual(row["paired_se"], 0)
        self.assertTrue(row["abs_mean_gt_two_se"])

    def test_ties_and_prediction_errors(self):
        rows = [pair_diagnostic([1] * 8, 1), pair_diagnostic([1] * 8, 0),
                pair_diagnostic([1] * 8, -1), pair_diagnostic([0] * 8, 1)]
        counts = pair_summary(rows)["all"]
        self.assertEqual((counts["pairs"], counts["comparable"], counts["correct"]), (4, 3, 1))
        self.assertAlmostEqual(counts["accuracy"], 1 / 3)

    def test_bad_numeric_inputs_fail(self):
        for values in ([1] * 7, [float("nan")] * 8, [float("inf")] * 8):
            with self.assertRaises(ValueError):
                pair_diagnostic(values, 1)

    def test_numpy_predictions_serialize_without_custom_encoder(self):
        row = pair_diagnostic(np.arange(8), np.float64(1))
        json.dumps({"row": row, "summary": pair_summary([row])}, allow_nan=False)

    def test_parent_exclusion_and_train_only_distance_scale(self):
        train = np.array([[0, 1], [0, 1], [2, 1], [2, 1]])
        result = coverage(train, [[3, 2]], [0, 0, 0, 0], [0], [0, 0, 1, 1],
                          ["a", "b", "c", "d"], ["x"])
        self.assertEqual(result["variable_train_coordinates"], 1)
        self.assertEqual(result["rows"][0]["nearest_distance"], 2)
        self.assertEqual(result["rows"][-1]["nearest_distance"], 1)
        self.assertEqual(result["rows"][-1]["novel_train_constant_coordinates"], 1)

    def test_same_time_only(self):
        result = coverage([[0], [100], [4], [104]], [[100]], [0, 1, 0, 1], [0], [0, 0, 1, 1],
                          ["a", "b", "c", "d"], ["x"])
        self.assertTrue(result["rows"][-1]["above_max_train_leave_parent_out"])
        with self.assertRaises(ValueError):
            coverage([[0]], [[1]], [0], [1], [0], ["a"], ["x"])

    def test_all_constant_coverage_is_finite(self):
        result = coverage([[1], [1]], [[2]], [0, 0], [0], [0, 1], ["a", "b"], ["x"])
        self.assertEqual(result["rows"][-1]["nearest_distance"], 0)
        self.assertEqual(result["rows"][-1]["novel_train_constant_coordinates"], 1)

    def test_negative_components_are_not_dropped(self):
        delta = component_contrast({"loss": 2, "queue": 1}, {"queue": 10, "other": 1})
        self.assertEqual(delta, {"loss": 2, "queue": -9, "other": -1})
        self.assertEqual(sum(delta.values()), -8)


if __name__ == "__main__":
    unittest.main()
