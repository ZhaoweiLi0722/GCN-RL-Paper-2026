"""Tests for the value-of-state-dependence measure (E2 gate amendment)."""

from __future__ import annotations

import unittest

from evaluation.state_dependence_value import (
    state_dependence_gate,
    state_dependence_value,
    validation_means,
)


def _rows(spec, stream="validation"):
    """spec: {state: {arm: [costs]}} -> flat rows."""

    out = []
    for state, arms in spec.items():
        for arm, costs in arms.items():
            for index, cost in enumerate(costs):
                out.append(
                    {
                        "state_id": state,
                        "arm": arm,
                        "stream": stream,
                        "crn_seed": index,
                        "remaining_cost": cost,
                    }
                )
    return out


class ValidationMeansTest(unittest.TestCase):
    def test_averages_within_stream_only(self) -> None:
        rows = _rows({"s1": {"anchor": [10.0, 20.0], "u_1.00": [4.0, 6.0]}})
        rows += _rows({"s1": {"anchor": [1000.0], "u_1.00": [1000.0]}}, stream="discovery")
        means = validation_means(rows)
        self.assertEqual(means[("s1", "anchor")], 15.0)
        self.assertEqual(means[("s1", "u_1.00")], 5.0)


class StateDependenceValueTest(unittest.TestCase):
    def test_constant_capturing_everything_scores_zero(self) -> None:
        # Same arm is best in every state -> a constant policy is optimal.
        means = {
            ("s1", "anchor"): 100.0, ("s1", "u_0.00"): 90.0, ("s1", "u_1.00"): 80.0,
            ("s2", "anchor"): 100.0, ("s2", "u_0.00"): 95.0, ("s2", "u_1.00"): 70.0,
        }
        report = state_dependence_value(means)
        self.assertEqual(report["best_constant_arm"], "u_1.00")
        self.assertEqual(report["value_of_state_dependence"], 0.0)
        self.assertEqual(report["distinct_best_arm_count"], 1)

    def test_state_dependent_optimum_scores_positive(self) -> None:
        # Different arms win in different states -> state-dependence has value.
        means = {
            ("s1", "anchor"): 100.0, ("s1", "u_0.00"): 60.0, ("s1", "u_1.00"): 90.0,
            ("s2", "anchor"): 100.0, ("s2", "u_0.00"): 90.0, ("s2", "u_1.00"): 60.0,
        }
        report = state_dependence_value(means)
        # Best constant totals 150; per-state oracle totals 120.
        self.assertEqual(report["constant_total"], 150.0)
        self.assertEqual(report["oracle_total"], 120.0)
        self.assertEqual(report["value_of_state_dependence"], 30.0)
        self.assertEqual(report["distinct_best_arm_count"], 2)

    def test_interior_fraction_counts_non_endpoint_arms(self) -> None:
        means = {
            ("s1", "u_0.00"): 50.0, ("s1", "u_0.50"): 10.0, ("s1", "u_1.00"): 60.0,
            ("s2", "u_0.00"): 50.0, ("s2", "u_0.50"): 60.0, ("s2", "u_1.00"): 10.0,
            ("s1", "anchor"): 99.0, ("s2", "anchor"): 99.0,
        }
        report = state_dependence_value(means)
        self.assertEqual(report["interior_best_arm_states"], 1)
        self.assertEqual(report["interior_best_arm_fraction"], 0.5)

    def test_rejects_incomplete_arm_coverage(self) -> None:
        means = {
            ("s1", "anchor"): 1.0, ("s1", "u_1.00"): 1.0,
            ("s2", "anchor"): 1.0,
        }
        with self.assertRaisesRegex(ValueError, "missing arms"):
            state_dependence_value(means)


class StateDependenceGateTest(unittest.TestCase):
    def _report(self, fraction, interior=1.0):
        return {
            "value_of_state_dependence_fraction": fraction,
            "interior_best_arm_fraction": interior,
        }

    def test_gate_rejects_noise_floor_value(self) -> None:
        # The measured Stage E2 value: 0.0056% of anchor cost.
        decision = state_dependence_gate(
            self._report(0.000056), minimum_fraction=0.005
        )
        self.assertFalse(decision["state_dependence_gate_passed"])
        self.assertEqual(
            decision["classification"], "channel_captured_by_constant_policy"
        )

    def test_gate_accepts_material_value(self) -> None:
        decision = state_dependence_gate(
            self._report(0.02), minimum_fraction=0.005
        )
        self.assertTrue(decision["state_dependence_gate_passed"])
        self.assertEqual(
            decision["classification"], "state_dependent_headroom_established"
        )

    def test_interior_criterion_can_fail_alone(self) -> None:
        decision = state_dependence_gate(
            self._report(0.02, interior=0.0),
            minimum_fraction=0.005,
            minimum_interior_fraction=0.3,
        )
        self.assertTrue(decision["value_criterion_passed"])
        self.assertFalse(decision["interior_criterion_passed"])
        self.assertFalse(decision["state_dependence_gate_passed"])

    def test_threshold_validation(self) -> None:
        with self.assertRaises(ValueError):
            state_dependence_gate(self._report(0.1), minimum_fraction=0.0)


if __name__ == "__main__":
    unittest.main()
