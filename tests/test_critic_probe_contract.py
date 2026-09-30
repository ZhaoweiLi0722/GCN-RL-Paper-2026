from dataclasses import replace
import unittest

import numpy as np

from src.rl.critic_probe_contract import (
    ProbeSchema, StateLineage, paired_advantage, public_inputs, ranking_counts,
    require_single_continuation, select_supported, train_only_constant,
    train_only_standardization, validate_partitions,
)


def lineage(name, trajectory, step=0, policy="policyA", usage="prospective_unseen"):
    return StateLineage(name, "snapshot_" + name, policy, "env", trajectory, step, usage)


class CriticProbeContractTests(unittest.TestCase):
    def test_valid_split_and_single_policy(self):
        rows = [lineage("a", 1), lineage("b", 2)]
        self.assertEqual(validate_partitions(rows, {"a": "train", "b": "test"}),
                         {"policyA": {"train": 1, "test": 1}})
        self.assertEqual(require_single_continuation(rows), "policyA")

    def test_new_future_draws_do_not_make_new_trajectory(self):
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_partitions([lineage("a", 1, 0), lineage("b", 1, 13)], {"a": "train", "b": "test"})

    def test_shared_world_across_policies_cannot_cross_split(self):
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_partitions([lineage("a", 1), lineage("b", 1, policy="B")], {"a": "train", "b": "test"})

    def test_renaming_duplicate_snapshot_does_not_hide_leak(self):
        a, b = lineage("a", 1), lineage("b", 2)
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_partitions([a, replace(b, snapshot_sha256=a.snapshot_sha256)], {"a": "train", "b": "test"})

    def test_inspected_pilot_cannot_be_fresh_test(self):
        with self.assertRaisesRegex(ValueError, "engineering-only"):
            validate_partitions([lineage("a", 1, usage="previously_inspected")], {"a": "test"})

    def test_each_policy_needs_own_train_and_test(self):
        with self.assertRaisesRegex(ValueError, "Each continuation"):
            validate_partitions([lineage("a", 1), lineage("b", 2, policy="B")], {"a": "train", "b": "test"})
        with self.assertRaisesRegex(ValueError, "pool"):
            require_single_continuation([lineage("a", 1), lineage("b", 2, policy="B")])

    def test_explicit_assignment_required(self):
        for assignment in ({}, {"a": "train", "unknown": "test"}, {"a": "validation"}):
            with self.assertRaises(ValueError):
                validate_partitions([lineage("a", 1)], assignment)

    def test_public_whitelist_and_readonly_arrays(self):
        schema = ProbeSchema(3, 2, 2, 52, 2)
        payload = {"observation": [3, 4, 0.25], "requests": [[0, 0], [0.1, 0.2]],
                   "reference_request": [0, 0]}
        result = public_inputs(payload, 13, schema)
        with self.assertRaises(ValueError):
            result.observation[0] = 100
        payload["observation"][0] = 100
        self.assertEqual(result.observation[0], 3)
        for key in ("patient_registry", "future_cost", "rng_seed", "continuation_sha256", "execution_identity"):
            with self.assertRaisesRegex(ValueError, "whitelist"):
                public_inputs(dict(payload, **{key: 1}), 13, schema)

    def test_time_reference_and_finite_checks(self):
        schema = ProbeSchema(3, 2, 2, 52, 2)
        base = {"observation": [3, 4, 0.25], "requests": [[0, 0], [0.1, 0.2]], "reference_request": [0, 0]}
        for change in ({"observation": [3, 4, 0.5]}, {"reference_request": [0.1, 0]},
                       {"requests": [[0, 0], [2, 0]]}, {"observation": [float("nan"), 0, 0.25]}):
            with self.assertRaises(ValueError):
                public_inputs(dict(base, **change), 13, schema)

    def test_paired_advantage_sign_and_scale(self):
        result = paired_advantage([90, 80], [100, 100], [1, 2], [1, 2])
        self.assertTrue(np.allclose(result["advantage_draws"], [1e-8, 2e-8], atol=1e-16))
        self.assertIn("not_immediate_reward", result["target_kind"])
        self.assertGreater(result["mean_advantage"], 0)

    def test_unpaired_duplicate_or_nan_labels_rejected(self):
        for costs, a, b in (([90, 80], [1, 2], [2, 1]), ([90, 80], [1, 1], [1, 1]),
                            ([90, float("nan")], [1, 2], [1, 2])):
            with self.assertRaises(ValueError):
                paired_advantage(costs, [100, 100], a, b)

    def test_frozen_alias_is_exact_zero_label(self):
        result = paired_advantage([100, 110], [100, 110], [1, 2], [1, 2])
        self.assertEqual(result["mean_advantage"], 0)
        self.assertEqual(result["advantage_mean_se"], 0)

    def test_training_only_scaler_does_not_see_test_outlier(self):
        mean, scale = train_only_standardization([[1, 2], [3, 2]], ["train"] * 2)
        self.assertTrue(np.array_equal(mean, [2, 2]))
        self.assertTrue(np.array_equal(scale, [1, 1]))
        with self.assertRaisesRegex(ValueError, "training rows only"):
            train_only_standardization([[1, 2], [1e8, 2]], ["train", "test"])

    def test_training_constant_not_global_oracle(self):
        option, _ = train_only_constant([[10, 9], [10, 8]], [[True, True]] * 2, ["train"] * 2)
        self.assertEqual(option, 1)
        with self.assertRaisesRegex(ValueError, "training rows only"):
            train_only_constant([[10, 9], [0, 1000]], [[True, True]] * 2, ["train", "test"])

    def test_constant_has_support_fallback(self):
        option, means = train_only_constant([[10, 0, 9], [10, 0, 9]],
                                           [[True, False, True]] * 2, ["train"] * 2)
        self.assertEqual(option, 2)
        self.assertEqual(means[1], 10)

    def test_decision_uses_predictions_and_public_support_only(self):
        result = select_supported([[0, 2, 5], [0, -1, -2]], [[True, True, False], [True] * 3])
        self.assertEqual(result.tolist(), [1, 0])
        with self.assertRaises(ValueError):
            select_supported([[1, 2]], [[True, True]])

    def test_tied_labels_not_counted_as_wrong_pairs(self):
        result = ranking_counts([[0, 0]], [[10, 10]], [[True, True]])
        self.assertEqual(result["label_ties"], 1)
        self.assertIsNone(result["pairwise_accuracy"])
        self.assertEqual(result["sampled_support_regret"], [0.0])

    def test_prediction_tie_is_not_a_success(self):
        result = ranking_counts([[0, 0]], [[10, 9]], [[True, True]])
        self.assertEqual(result["prediction_ties"], 1)
        self.assertEqual(result["pairwise_accuracy"], 0)
        self.assertEqual(result["sampled_support_regret"], [1.0])

    def test_ranking_sign_is_positive_advantage_lower_cost(self):
        result = ranking_counts([[0, 2, -1]], [[10, 8, 11]], [[True] * 3])
        self.assertEqual(result["pairwise_accuracy"], 1)
        self.assertEqual(result["selected_actions"], [1])


if __name__ == "__main__":
    unittest.main()
