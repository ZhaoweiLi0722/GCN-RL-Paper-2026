"""Wiring tests for the Stage E2 overtime headroom screen (spec 2026-08-29).

These validate config discipline, gate arithmetic, and the smoke path on a
small synthetic setup. They never run the full screen.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from evaluation.audit_overtime_headroom_e2 import (
    DEFAULT_CONFIG,
    gate_decision,
    load_screen_config,
    smoke_config,
    summarize_scenario,
    validate_config,
)

CONFIG = load_screen_config(Path(DEFAULT_CONFIG))


def _row(state_id, stream, seed, arm, cost, completion=1.0, lost=0.0, inelig=0.0):
    return {
        "state_id": state_id,
        "stream": stream,
        "crn_seed": seed,
        "arm": arm,
        "remaining_cost": cost,
        "completion_service_level": completion,
        "patients_lost": lost,
        "manufacturing_ineligibility": inelig,
    }


class E2ConfigTest(unittest.TestCase):
    def test_frozen_config_is_valid(self) -> None:
        validate_config(CONFIG)

    def test_frozen_config_matches_signed_spec(self) -> None:
        overrides = CONFIG["overtime_env_overrides"]
        self.assertTrue(overrides["enable_overtime_control"])
        self.assertFalse(overrides["enable_production_throttle"])
        self.assertEqual(overrides["max_overtime_fraction"], 0.3)
        self.assertEqual(CONFIG["gates"]["material_threshold"], 1_000_000.0)
        self.assertEqual(CONFIG["gates"]["state_fraction"], 0.30)
        # 27+ states: seeds x epochs x scenarios.
        states = (
            len(CONFIG["state_generation"]["seeds"])
            * len(CONFIG["state_generation"]["decision_epochs"])
            * len(CONFIG["scenarios"])
        )
        self.assertGreaterEqual(states, 27)
        self.assertEqual(len(CONFIG["replications"]["discovery_seeds"]), 3)
        self.assertEqual(len(CONFIG["replications"]["validation_seeds"]), 5)

    def test_protected_streams_rejected(self) -> None:
        bad = copy.deepcopy(CONFIG)
        bad["replications"]["validation_seeds"][0] = 91100000
        with self.assertRaisesRegex(ValueError, "protected"):
            validate_config(bad)
        bad = copy.deepcopy(CONFIG)
        bad["replications"]["discovery_seeds"][0] = 94000000
        with self.assertRaisesRegex(ValueError, "protected"):
            validate_config(bad)

    def test_seed_overlap_rejected(self) -> None:
        bad = copy.deepcopy(CONFIG)
        bad["replications"]["validation_seeds"][0] = bad["replications"][
            "discovery_seeds"
        ][0]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_config(bad)

    def test_dormant_throttle_rejected(self) -> None:
        bad = copy.deepcopy(CONFIG)
        bad["overtime_env_overrides"]["enable_production_throttle"] = True
        with self.assertRaisesRegex(ValueError, "dormant"):
            validate_config(bad)

    def test_overtime_flag_required(self) -> None:
        bad = copy.deepcopy(CONFIG)
        bad["overtime_env_overrides"]["enable_overtime_control"] = False
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_smoke_config_shrinks_and_redirects_output(self) -> None:
        smoke = smoke_config(CONFIG)
        validate_config(smoke)
        self.assertTrue(smoke["smoke"])
        self.assertNotEqual(smoke["output_root"], CONFIG["output_root"])
        self.assertEqual(len(smoke["state_generation"]["seeds"]), 1)
        self.assertEqual(len(smoke["replications"]["discovery_seeds"]), 1)


class E2GateMathTest(unittest.TestCase):
    def _scenario_rows(self, state_id: str, saving: float):
        """Anchor at 10M; one rung saving `saving` in both streams."""

        rows = []
        for stream, seeds in (("discovery", [1, 2, 3]), ("validation", [4, 5])):
            for seed in seeds:
                rows.append(_row(state_id, stream, seed, "anchor", 10_000_000.0))
                rows.append(
                    _row(state_id, stream, seed, "u_0.50", 10_000_000.0 - saving)
                )
                rows.append(_row(state_id, stream, seed, "u_1.00", 10_500_000.0))
        return rows

    def test_material_state_counted(self) -> None:
        rows = self._scenario_rows("s1", 2_000_000.0) + self._scenario_rows(
            "s2", 100.0
        )
        summary = summarize_scenario("x", rows, ["s1", "s2"], 1_000_000.0)
        self.assertEqual(summary["material_validated_states"], 1)
        self.assertEqual(summary["material_validated_fraction"], 0.5)
        self.assertEqual(summary["best_arm_agreement_fraction"], 1.0)
        self.assertEqual(summary["prospective_success_fraction"], 0.5)

    def test_clinical_noninferiority_blocks_material(self) -> None:
        rows = []
        for stream, seeds in (("discovery", [1]), ("validation", [2])):
            for seed in seeds:
                rows.append(_row("s1", stream, seed, "anchor", 10_000_000.0))
                rows.append(
                    _row("s1", stream, seed, "u_0.50", 5_000_000.0, lost=3.0)
                )
        summary = summarize_scenario("x", rows, ["s1"], 1_000_000.0)
        self.assertEqual(summary["material_validated_states"], 0)

    def test_gate_requires_non_nominal_pass(self) -> None:
        config = {
            "gates": {"state_fraction": 0.30, "material_threshold": 1e6},
            "non_nominal_scenarios": ["shift"],
        }
        nominal_only = [
            {"scenario": "nominal", "material_validated_fraction": 1.0},
            {"scenario": "shift", "material_validated_fraction": 0.1},
        ]
        decision = gate_decision(config, nominal_only)
        self.assertFalse(decision["headroom_gate_passed"])
        self.assertFalse(decision["e3_authorized"])
        passing = [
            {"scenario": "nominal", "material_validated_fraction": 0.0},
            {"scenario": "shift", "material_validated_fraction": 0.4},
        ]
        decision = gate_decision(config, passing)
        self.assertTrue(decision["headroom_gate_passed"])
        self.assertEqual(
            decision["classification"], "overtime_headroom_established"
        )


if __name__ == "__main__":
    unittest.main()
