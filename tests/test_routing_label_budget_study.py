"""Tests for the routing label budget study (spec 2026-08-29)."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from evaluation.routing_label_budget_pool import (
    DEFAULT_CONFIG,
    centred_pressure,
    executed_signature,
    validate_config,
)

CONFIG = json.loads(Path(DEFAULT_CONFIG).read_text())


class ConfigDisciplineTest(unittest.TestCase):
    def test_frozen_config_is_valid(self) -> None:
        validate_config(CONFIG)

    def test_routing_on_overtime_off(self) -> None:
        overrides = CONFIG["env_overrides"]
        self.assertTrue(overrides["enable_specimen_routing"])
        self.assertFalse(overrides["enable_overtime_control"])
        self.assertFalse(overrides["enable_production_throttle"])
        for flag, message in (
            ("enable_specimen_routing", "routing study"),
            ("enable_overtime_control", "overtime must be off"),
            ("enable_production_throttle", "dormant"),
        ):
            bad = copy.deepcopy(CONFIG)
            bad["env_overrides"][flag] = flag != "enable_specimen_routing"
            with self.assertRaisesRegex(ValueError, message):
                validate_config(bad)

    def test_protected_and_reserved_streams_rejected(self) -> None:
        for seed, message in (
            (91100000, "protected"),
            (94000000, "protected"),
            (97100005, "reserved"),
            (97300009, "reserved"),
            (101000000, "Stage G1"),
            (103000500, "Stage G1"),
            (107000000, "Stage G1"),
        ):
            bad = copy.deepcopy(CONFIG)
            bad["state_generation"]["seeds"][0] = seed
            with self.assertRaisesRegex(ValueError, message):
                validate_config(bad)

    def test_reserved_scenario_rejected(self) -> None:
        bad = copy.deepcopy(CONFIG)
        bad["scenarios"] = bad["scenarios"] + ["routing_regional_drift"]
        with self.assertRaisesRegex(ValueError, "reserved"):
            validate_config(bad)

    def test_seed_overlap_rejected(self) -> None:
        bad = copy.deepcopy(CONFIG)
        bad["state_generation"]["seeds"][0] = bad["worlds"]["seed_start"]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_config(bad)

    def test_anchor_arm_is_present_and_shifts_distinct(self) -> None:
        self.assertIn(0.0, CONFIG["action_family"]["shifts"])
        bad = copy.deepcopy(CONFIG)
        bad["action_family"]["shifts"] = [-0.1, 0.1]
        with self.assertRaisesRegex(ValueError, "anchor"):
            validate_config(bad)
        bad = copy.deepcopy(CONFIG)
        bad["action_family"]["shifts"] = [0.0, 0.1, 0.1]
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_config(bad)

    def test_reading_rule_frozen_as_specified(self) -> None:
        rule = CONFIG["reading_rule"]
        self.assertEqual(rule["underpowered_if_agreement_at_k32_at_least"], 0.70)
        self.assertEqual(rule["fundamental_if_agreement_at_k32_below"], 0.60)
        self.assertEqual(rule["fundamental_also_requires_last_doubling_gain_below"], 0.02)


class _FakeEnv:
    """Minimal stand-in exposing only what centred_pressure reads."""

    def __init__(self, waiting, idle, reagents):
        self._waiting = np.asarray(waiting, dtype=float)
        self.bioreactors = np.column_stack([np.asarray(idle, dtype=float)])
        self.reagents = np.asarray(reagents, dtype=float)

    def waiting_counts(self):
        return self._waiting


class CentredPressureTest(unittest.TestCase):
    """A uniform pattern cannot route anything; the pattern must be centred."""

    def test_pattern_sums_to_zero(self) -> None:
        pattern = centred_pressure(_FakeEnv([10, 2, 6, 0], [1, 5, 2, 9], [9, 9, 9, 9]))
        self.assertAlmostEqual(float(pattern.sum()), 0.0, places=9)

    def test_pattern_has_both_signs(self) -> None:
        pattern = centred_pressure(_FakeEnv([10, 2, 6, 0], [1, 5, 2, 9], [9, 9, 9, 9]))
        self.assertGreater(pattern.max(), 0.0)
        self.assertLess(pattern.min(), 0.0)

    def test_pattern_is_unit_scaled(self) -> None:
        pattern = centred_pressure(_FakeEnv([10, 2, 6, 0], [1, 5, 2, 9], [9, 9, 9, 9]))
        self.assertAlmostEqual(float(np.abs(pattern).max()), 1.0, places=9)

    def test_pressure_uses_the_binding_resource(self) -> None:
        # Reagents bind below capacity, so a reagent-poor clinic is pressured.
        pattern = centred_pressure(_FakeEnv([8, 8], [8, 8], [8, 0]))
        self.assertLess(pattern[0], pattern[1])

    def test_degenerate_state_yields_zero_pattern(self) -> None:
        pattern = centred_pressure(_FakeEnv([5, 5], [1, 1], [9, 9]))
        np.testing.assert_allclose(pattern, np.zeros(2))


class ExecutedSignatureTest(unittest.TestCase):
    """Filtering must use executed flows; requests can differ while nothing moves."""

    def test_signature_reads_executed_flows_not_requests(self) -> None:
        info = {
            "specimen_transfers": np.array([0, 0, 0]),
            "specimen_route_count": 0.0,
            "specimen_requested_integer_net": np.array([3, -3, 0]),
        }
        other = dict(info, specimen_requested_integer_net=np.array([7, -7, 0]))
        self.assertEqual(executed_signature(info), executed_signature(other))

    def test_different_executions_differ(self) -> None:
        a = {"specimen_transfers": np.array([1, -1, 0]), "specimen_route_count": 1.0}
        b = {"specimen_transfers": np.array([2, -2, 0]), "specimen_route_count": 2.0}
        self.assertNotEqual(executed_signature(a), executed_signature(b))


if __name__ == "__main__":
    unittest.main()


class ReadingRuleTest(unittest.TestCase):
    """All three branches, including boundaries, pinned before the data is read."""

    RULE = {
        "underpowered_if_agreement_at_k32_at_least": 0.70,
        "fundamental_if_agreement_at_k32_below": 0.60,
        "fundamental_also_requires_last_doubling_gain_below": 0.02,
    }

    def _curve(self, penultimate, final):
        return {16: {"mean": penultimate}, 32: {"mean": final}}

    def test_underpowered_branch_and_its_boundary(self) -> None:
        from evaluation.routing_label_budget_analysis import classify
        out = classify(self._curve(0.68, 0.70), self.RULE)
        self.assertEqual(out["classification"], "g1_negative_underpowered")
        self.assertIn("REOPENED", out["consequence"])

    def test_fundamental_branch_requires_a_flat_tail(self) -> None:
        from evaluation.routing_label_budget_analysis import classify
        flat = classify(self._curve(0.589, 0.599), self.RULE)
        self.assertEqual(flat["classification"], "g1_negative_confirmed_fundamental")
        # Same endpoint, still climbing -> must NOT be called fundamental.
        climbing = classify(self._curve(0.50, 0.599), self.RULE)
        self.assertEqual(climbing["classification"], "inconclusive_at_this_budget")

    def test_middle_band_is_inconclusive(self) -> None:
        from evaluation.routing_label_budget_analysis import classify
        out = classify(self._curve(0.63, 0.65), self.RULE)
        self.assertEqual(out["classification"], "inconclusive_at_this_budget")
        self.assertIn("change-control", out["consequence"])
