import unittest

import numpy as np

from evaluation.audit_ddpg_critic_advantage_ranking import (
    critic_advantage_metrics,
    subset_demonstrations,
    trajectory_group_folds,
)


class DDPGCriticAdvantageRankingAuditTests(unittest.TestCase):
    def test_trajectory_folds_hold_out_whole_groups_once(self) -> None:
        scenarios = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        trajectories = np.asarray([0, 0, 2, 2, 1, 1, 3, 3])
        folds = trajectory_group_folds(
            scenarios,
            trajectories,
            fold_count=2,
        )
        heldout_counts = sum(
            fold["holdout_mask"].astype(np.int64) for fold in folds
        )
        np.testing.assert_array_equal(heldout_counts, np.ones(8))
        for fold in folds:
            train_groups = set(
                zip(
                    scenarios[fold["train_mask"]],
                    trajectories[fold["train_mask"]],
                )
            )
            holdout_groups = set(
                zip(
                    scenarios[fold["holdout_mask"]],
                    trajectories[fold["holdout_mask"]],
                )
            )
            self.assertFalse(train_groups & holdout_groups)
            self.assertEqual(set(scenarios[fold["holdout_mask"]]), {0, 1})

    def test_trajectory_folds_require_two_groups_per_scenario(self) -> None:
        with self.assertRaisesRegex(ValueError, "trajectory per fold"):
            trajectory_group_folds(
                np.asarray([0, 0, 1, 1]),
                np.asarray([0, 0, 1, 1]),
            )

    def test_subset_preserves_metadata_and_copies_rows(self) -> None:
        demos = {
            "states": np.arange(12).reshape(4, 3),
            "actions": np.arange(8).reshape(4, 2),
            "weights": np.ones(4),
            "scenario_ids": np.asarray([0, 0, 1, 1]),
            "scenario_names": np.asarray(["a", "b"]),
            "improved_steps": 3,
        }
        result = subset_demonstrations(
            demos,
            np.asarray([True, False, False, True]),
        )
        np.testing.assert_array_equal(result["scenario_ids"], [0, 1])
        np.testing.assert_array_equal(result["scenario_names"], ["a", "b"])
        self.assertEqual(result["improved_steps"], 3)
        result["states"][0, 0] = -1
        self.assertEqual(demos["states"][0, 0], 0)

    def test_metrics_identify_correct_and_reversed_order(self) -> None:
        targets = np.asarray([0.0, 1.0, 2.0, 3.0])
        correct = critic_advantage_metrics(targets, targets)
        reversed_result = critic_advantage_metrics(targets, -targets)
        self.assertAlmostEqual(correct["spearman"], 1.0)
        self.assertAlmostEqual(correct["pairwise_order_accuracy"], 1.0)
        self.assertAlmostEqual(reversed_result["spearman"], -1.0)
        self.assertAlmostEqual(
            reversed_result["pairwise_order_accuracy"],
            0.0,
        )

    def test_metrics_credit_prediction_ties_as_half_order(self) -> None:
        result = critic_advantage_metrics(
            np.asarray([0.0, 1.0, 2.0]),
            np.asarray([0.0, 0.0, 0.0]),
        )
        self.assertIsNone(result["spearman"])
        self.assertAlmostEqual(result["pairwise_order_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
