"""Tests for the policy-free intertemporal residual headroom screen."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from evaluation.screen_intertemporal_residual_allocation_headroom import (
    BASELINE_CANDIDATE,
    budget_neutral_reallocation,
    candidate_specs,
    comparison_summary,
    describe,
    execution_approvals,
    label_stability,
    load_json,
    require_execution_authorization,
    residual_first_action,
    rollout,
    run,
    validate_config,
)
from evaluation.screen_intertemporal_shared_capacity_episodes import scenario_env
from evaluation.screen_intertemporal_shared_capacity_states import make_policy
from src.rl.experiment import build_env


CONFIG_PATH = Path(
    "experiments/configs/intertemporal_residual_allocation_headroom.json"
)


class FakeOvertimeEnv:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_facilities=3)
        self.overtime_surge_headroom = np.array([2.0, 2.0, 2.0])

    def _shared_overtime_budget(self) -> float:
        return 3.0

    def _decode_overtime(self, action):
        block = np.asarray(action, dtype=float)[12:15]
        fraction = (block + 1.0) / 2.0
        return fraction, fraction * self.overtime_surge_headroom

    def _project_overtime_commitment(self, fraction, requested):
        del fraction
        allocation = np.asarray(requested, dtype=float).copy()
        if allocation.sum() > self._shared_overtime_budget():
            allocation *= self._shared_overtime_budget() / allocation.sum()
        return allocation / self.overtime_surge_headroom, allocation


def synthetic_rows(costs: dict[tuple[int, str, str], float]):
    rows = []
    for (state, candidate, phase), cost in costs.items():
        rows.append(
            {
                "phase": phase,
                "state_index": state,
                "state_id": f"s{state}",
                "scenario": "a" if state == 0 else "b",
                "generation_seed": 10 + state,
                "epoch": 4,
                "candidate": candidate,
                "total_cost": cost,
                "patients_lost": 1.0,
                "completion_service_level": 0.9,
            }
        )
    return rows


class ResidualMechanicsTest(unittest.TestCase):
    def test_increase_and_decrease_preserve_budget_and_caps(self) -> None:
        base = np.array([1.0, 1.0, 1.0])
        caps = np.array([2.0, 2.0, 2.0])
        increased, moved = budget_neutral_reallocation(
            base, caps, target=0, signed_amount=0.6
        )
        self.assertAlmostEqual(moved, 0.6)
        np.testing.assert_allclose(increased, [1.6, 0.7, 0.7])
        self.assertAlmostEqual(float(increased.sum()), float(base.sum()))

        decreased, moved = budget_neutral_reallocation(
            base, caps, target=0, signed_amount=-0.6
        )
        self.assertAlmostEqual(moved, 0.6)
        np.testing.assert_allclose(decreased, [0.4, 1.3, 1.3])
        self.assertAlmostEqual(float(decreased.sum()), float(base.sum()))

    def test_reallocation_clips_to_feasible_transfer(self) -> None:
        allocation, moved = budget_neutral_reallocation(
            np.array([1.9, 0.1, 0.0]),
            np.array([2.0, 2.0, 2.0]),
            target=0,
            signed_amount=1.0,
        )
        self.assertAlmostEqual(moved, 0.1)
        np.testing.assert_allclose(allocation, [2.0, 0.0, 0.0])

    def test_residual_action_changes_only_overtime_slice(self) -> None:
        env = FakeOvertimeEnv()
        base_action = np.zeros(15, dtype=np.float32)
        candidate = {
            "candidate": "f00_increase_0.10",
            "facility": 0,
            "direction": "increase",
            "transfer_budget_fraction": 0.10,
        }
        action, mechanics = residual_first_action(env, base_action, candidate)
        np.testing.assert_array_equal(action[:12], base_action[:12])
        _, requested = env._decode_overtime(action)
        _, allocation = env._project_overtime_commitment(None, requested)
        np.testing.assert_allclose(allocation, [1.3, 0.85, 0.85])
        self.assertAlmostEqual(mechanics["realized_transfer_units"], 0.3)
        self.assertAlmostEqual(mechanics["budget_error"], 0.0)

    def test_candidate_library_has_baseline_and_four_directions_per_facility(self) -> None:
        config = load_json(CONFIG_PATH)
        specs = candidate_specs(config, 20)
        self.assertEqual(len(specs), 81)
        self.assertEqual(specs[0]["candidate"], BASELINE_CANDIDATE)
        self.assertEqual(len({item["candidate"] for item in specs}), 81)

    def test_all_candidates_map_in_real_patient_environment(self) -> None:
        config = load_json(CONFIG_PATH)
        source = validate_config(config)
        env = build_env(
            {"env": scenario_env(source, source["scenarios"][0])}, seed=123
        )
        policy = make_policy({"budget_fraction": 1.0, "smoothing": 0.25})
        base_action = policy.select_action(env.observation(), env=env)
        specs = candidate_specs(config, env.config.num_facilities)
        for spec in specs:
            action, mechanics = residual_first_action(env, base_action, spec)
            self.assertEqual(action.shape, base_action.shape)
            self.assertLessEqual(abs(mechanics["budget_error"]), 1e-8)

    def test_real_patient_rollouts_preserve_paired_rng(self) -> None:
        config = load_json(CONFIG_PATH)
        source = validate_config(config)
        env_config = scenario_env(source, source["scenarios"][0])
        env = build_env({"env": env_config}, seed=123)
        behavior = make_policy({"budget_fraction": 1.0, "smoothing": 0.25})
        for _ in range(4):
            env.step(behavior.select_action(env.observation(), env=env))
        state = {
            "state_index": 0,
            "state_id": "unit_state",
            "scenario": "unit_scenario",
            "generation_seed": 123,
            "epoch": int(env.t),
            "env_config": env_config,
            "snapshot": env.state_dict(),
        }
        specs = candidate_specs(config, env.config.num_facilities)
        rows = [
            rollout(
                state=state,
                candidate=spec,
                baseline_spec=config["baseline_policy"],
                continuation_spec=config["continuation_policy"],
                world_seed=456,
                phase="unit",
                lookahead=3,
            )
            for spec in (specs[0], specs[-1])
        ]
        self.assertEqual(rows[0]["rng_sha256"], rows[1]["rng_sha256"])
        self.assertTrue(np.isfinite(rows[0]["total_cost"]))
        self.assertGreater(rows[1]["realized_transfer_units"], 0.0)


class ResidualContractTest(unittest.TestCase):
    def test_frozen_config_and_parent_hashes_validate(self) -> None:
        config = load_json(CONFIG_PATH)
        source = validate_config(config)
        self.assertEqual(len(source["scenarios"]), 3)
        self.assertEqual(
            execution_approvals(config), {"zhaowei": True, "howard": False}
        )

    def test_weaker_threshold_is_rejected(self) -> None:
        config = load_json(CONFIG_PATH)
        config["gate"]["minimum_prospective_relative_saving"] = 0.0049
        with self.assertRaisesRegex(ValueError, "0.5%"):
            validate_config(config)

    def test_pending_approval_blocks_before_state_generation_or_output(self) -> None:
        config = load_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "forbidden-result"
            config["output_root"] = str(output)
            with mock.patch(
                "evaluation.screen_intertemporal_residual_allocation_headroom.generate_states"
            ) as generate:
                with self.assertRaisesRegex(PermissionError, "howard"):
                    run(config, config_path=CONFIG_PATH)
            generate.assert_not_called()
            self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        config = load_json(CONFIG_PATH)
        config["execution_authorization"]["howard"]["approved"] = True
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing-result"
            output.mkdir()
            config["output_root"] = str(output)
            with mock.patch(
                "evaluation.screen_intertemporal_residual_allocation_headroom.generate_states"
            ) as generate:
                with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                    run(config, config_path=CONFIG_PATH)
            generate.assert_not_called()

    def test_describe_never_treats_missing_approvals_as_authorized(self) -> None:
        config = load_json(CONFIG_PATH)
        config["execution_authorization"] = {}
        self.assertFalse(describe(config)["execution_authorized"])

    def test_authorization_requires_exactly_two_named_approvals(self) -> None:
        config = load_json(CONFIG_PATH)
        with self.assertRaises(PermissionError):
            require_execution_authorization(config)
        approved = copy.deepcopy(config)
        approved["execution_authorization"]["howard"]["approved"] = True
        require_execution_authorization(approved)
        approved["execution_authorization"]["extra"] = {"approved": True}
        with self.assertRaises(PermissionError):
            require_execution_authorization(approved)


class ResidualStatisticsTest(unittest.TestCase):
    def test_comparison_and_stability_use_discovery_choices_on_validation(self) -> None:
        costs = {
            (0, BASELINE_CANDIDATE, "discovery"): 100.0,
            (0, "x", "discovery"): 90.0,
            (0, "y", "discovery"): 110.0,
            (1, BASELINE_CANDIDATE, "discovery"): 100.0,
            (1, "x", "discovery"): 110.0,
            (1, "y", "discovery"): 90.0,
            (0, BASELINE_CANDIDATE, "validation"): 100.0,
            (0, "x", "validation"): 92.0,
            (0, "y", "validation"): 108.0,
            (1, BASELINE_CANDIDATE, "validation"): 100.0,
            (1, "x", "validation"): 109.0,
            (1, "y", "validation"): 91.0,
        }
        rows = synthetic_rows(costs)
        discovery = [row for row in rows if row["phase"] == "discovery"]
        validation = [row for row in rows if row["phase"] == "validation"]
        choices = {0: "x", 1: "y"}
        summary = comparison_summary(
            validation, choices, material_threshold=0.005
        )
        self.assertAlmostEqual(summary["relative_saving"], 0.085)
        self.assertTrue(summary["positive_every_scenario"])
        stability = label_stability(
            discovery,
            validation,
            choices,
            choices,
            material_threshold=0.005,
        )
        self.assertEqual(stability["selected_validation_win_fraction"], 1.0)
        self.assertEqual(stability["pairwise_material_sign_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
