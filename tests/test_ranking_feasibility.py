"""Tests for the Stage E4 held-out-seed ranking criteria."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.ranking_feasibility import (
    leave_one_seed_out_ranking,
    ranking_gate,
    ridge_fit,
    ridge_predict,
)

LADDER = ["u_0.00", "u_0.50", "u_1.00"]


def _learnable_problem(seeds=(0, 1, 2), per_seed=12):
    """The best rung is a deterministic function of one observable feature."""

    features, seed_of, outcomes = {}, {}, {}
    rng = np.random.default_rng(0)
    for seed in seeds:
        for index in range(per_seed):
            state = f"s{seed}_{index}"
            driver = rng.uniform(0.0, 1.0)
            features[state] = np.array([driver, rng.uniform(0.0, 1.0)])
            seed_of[state] = seed
            # Cost is minimized by the rung nearest the driver.
            for position, arm in enumerate(LADDER):
                target = position / (len(LADDER) - 1)
                outcomes[(state, arm)] = 100.0 * (driver - target) ** 2
    return features, seed_of, outcomes


def _unlearnable_problem(seeds=(0, 1, 2), per_seed=12):
    """The best rung is driven by something NOT present in the features.

    This is the Stage E4 failure mode the gate exists to catch: labels are
    perfectly stable and reproducible, yet unpredictable from the state.
    """

    features, seed_of, outcomes = {}, {}, {}
    rng = np.random.default_rng(1)
    for seed in seeds:
        for index in range(per_seed):
            state = f"s{seed}_{index}"
            hidden = rng.uniform(0.0, 1.0)  # never exposed as a feature
            features[state] = rng.uniform(0.0, 1.0, size=2)
            seed_of[state] = seed
            for position, arm in enumerate(LADDER):
                target = position / (len(LADDER) - 1)
                outcomes[(state, arm)] = 100.0 * (hidden - target) ** 2
    return features, seed_of, outcomes


class RidgeTest(unittest.TestCase):
    def test_recovers_a_linear_relationship(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 3))
        y = x @ np.array([2.0, -1.0, 0.5]) + 4.0
        coefficients = ridge_fit(x, y, alpha=1e-8)
        np.testing.assert_allclose(ridge_predict(coefficients, x), y, atol=1e-4)

    def test_intercept_is_unpenalized(self) -> None:
        x = np.zeros((10, 2))
        y = np.full(10, 7.0)
        coefficients = ridge_fit(x, y, alpha=1e6)
        self.assertAlmostEqual(float(coefficients[-1]), 7.0, places=6)


class LeaveOneSeedOutTest(unittest.TestCase):
    def test_folds_are_grouped_by_generation_seed(self) -> None:
        features, seed_of, outcomes = _learnable_problem()
        report = leave_one_seed_out_ranking(features, seed_of, outcomes)
        self.assertEqual(len(report["folds"]), 3)
        for fold in report["folds"]:
            self.assertEqual(fold["states"], 12)

    def test_learnable_signal_is_detected(self) -> None:
        features, seed_of, outcomes = _learnable_problem()
        report = leave_one_seed_out_ranking(features, seed_of, outcomes)
        self.assertGreater(report["pooled_fitted_top1"], 0.5)
        self.assertGreater(
            report["pooled_fitted_top1"], report["pooled_state_blind_top1"]
        )

    def test_unlearnable_signal_is_not_detected(self) -> None:
        # Stable, reproducible labels that the state cannot predict.
        features, seed_of, outcomes = _unlearnable_problem()
        report = leave_one_seed_out_ranking(features, seed_of, outcomes)
        self.assertLess(report["pooled_fitted_top1"], 0.5)

    def test_chance_and_ladder_reported(self) -> None:
        features, seed_of, outcomes = _learnable_problem()
        report = leave_one_seed_out_ranking(features, seed_of, outcomes)
        self.assertEqual(report["ladder_size"], 3)
        self.assertAlmostEqual(report["chance_top1"], 1 / 3)

    def test_requires_multiple_seeds(self) -> None:
        features, seed_of, outcomes = _learnable_problem(seeds=(0,))
        with self.assertRaisesRegex(ValueError, "generation seeds"):
            leave_one_seed_out_ranking(features, seed_of, outcomes)


class RankingGateTest(unittest.TestCase):
    def _report(self, pooled, worst, gain, pairwise=0.9):
        return {
            "pooled_fitted_top1": pooled,
            "pooled_pairwise_accuracy": pairwise,
            "pooled_state_blind_top1": 0.05,
            "worst_fold_top1": worst,
            "worst_fold_gain": gain,
            "chance_top1": 1 / 11,
        }

    def test_routing_g1_shape_fails(self) -> None:
        # G1: 30.1% top-1, 57.7% pairwise, one seed with a negative gain.
        decision = ranking_gate(
            self._report(0.301, 0.269, -0.0577, pairwise=0.577),
            minimum_top1=0.50,
            minimum_pairwise=0.70,
            minimum_gain_over_state_blind=0.05,
        )
        self.assertFalse(decision["ranking_gate_passed"])
        self.assertEqual(
            decision["classification"], "optimum_not_predictable_from_state"
        )
        self.assertFalse(decision["e5_design_authorized"])

    def test_pooled_pass_with_failing_fold_is_rejected(self) -> None:
        # A channel that works on two of three seeds is not a basis for training.
        decision = ranking_gate(
            self._report(0.55, 0.30, 0.20),
            minimum_top1=0.50,
            minimum_pairwise=0.70,
            minimum_gain_over_state_blind=0.05,
        )
        self.assertTrue(decision["pooled_top1_passed"])
        self.assertFalse(decision["per_fold_top1_passed"])
        self.assertFalse(decision["ranking_gate_passed"])

    def test_insufficient_gain_over_state_blind_is_rejected(self) -> None:
        decision = ranking_gate(
            self._report(0.60, 0.55, 0.01),
            minimum_top1=0.50,
            minimum_pairwise=0.70,
            minimum_gain_over_state_blind=0.05,
        )
        self.assertTrue(decision["per_fold_top1_passed"])
        self.assertFalse(decision["gain_over_state_blind_passed"])
        self.assertFalse(decision["ranking_gate_passed"])

    def test_all_criteria_passing_authorizes_e5_design(self) -> None:
        decision = ranking_gate(
            self._report(0.72, 0.65, 0.40),
            minimum_top1=0.50,
            minimum_pairwise=0.70,
            minimum_gain_over_state_blind=0.05,
        )
        self.assertTrue(decision["ranking_gate_passed"])
        self.assertTrue(decision["e5_design_authorized"])
        self.assertEqual(decision["classification"], "optimum_predictable_from_state")


if __name__ == "__main__":
    unittest.main()
