"""Tests for conservative dense-advantage reagent targets."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.build_bounded_reagent_teacher_cache import (
    make_bounded_reagent_demonstrations,
)
from src.baselines.heuristics import (
    facility_net_action_from_state,
    heuristic_settings_for_policy,
)
from src.rl.config import load_config
from src.rl.experiment import build_env
from src.rl.residual_options import residual_pressure_patterns_from_state


class BoundedReagentTeacherCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        env_config = load_config(
            "experiments/configs/2_clinic_patient_condition.json"
        )
        env = build_env({"env": env_config}, seed=71)
        state = env.reset(seed=71)
        states = np.repeat(state[None, :], 4, axis=0).astype(np.float32)
        states[1, 0] += 1.0
        states[2, 1] += 2.0
        states[3, 2] += 1.0
        self.env_configs = (
            dict(env_config),
            dict(env_config),
        )
        self.source = {
            "states": states,
            "actions": np.zeros(
                (4, env.action_size),
                dtype=np.float32,
            ),
            "weights": np.ones(4, dtype=np.float32),
            "improved_mask": np.zeros(4, dtype=bool),
            "improved_steps": 0,
            "anchor_keep_steps": 4,
            "service_rejected_steps": 0,
            "mean_step_improvement": 0.0,
            "improved_weight_fraction": 0.0,
            "scenario_ids": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "trajectory_ids": np.arange(4, dtype=np.int64),
            "trajectory_steps": np.zeros(4, dtype=np.int64),
            "scenario_names": np.asarray(["scenario_a", "scenario_b"]),
            "scenario_cache_version": 2,
            "demand_history_window": int(
                env_config.get("demand_history_window", 1)
            ),
            "option_advantages": np.asarray(
                [
                    [0.0, 1_000_000.0, -1_000_000.0],
                    [0.0, 400_000.0, -1_000_000.0],
                    [0.0, -1_000_000.0, 2_000_000.0],
                    [0.0, -1_000_000.0, -1_000_000.0],
                ],
                dtype=np.float32,
            ),
            "option_feasible": np.ones((4, 3), dtype=bool),
            "option_groups": np.asarray(
                ["anchor", "reagent_transfer", "reagent_transfer"]
            ),
            "option_epsilons": np.asarray(
                [0.0, 0.32, 0.32],
                dtype=np.float32,
            ),
            "option_signs": np.asarray(
                [0.0, 1.0, -1.0],
                dtype=np.float32,
            ),
        }

    def test_builds_signed_bounded_targets_and_preserves_provenance(self) -> None:
        bounded, summary = make_bounded_reagent_demonstrations(
            self.source,
            env_configs=self.env_configs,
            anchor_settings=heuristic_settings_for_policy("mdl2"),
            min_advantage=500_000.0,
            max_residual_scale=0.1,
            coefficient_advantage_scale=2_000_000.0,
            min_positive_coefficient=0.25,
            weight_advantage_scale=1_000_000.0,
            weight_cap=3.0,
        )

        np.testing.assert_array_equal(
            bounded["improved_mask"],
            np.asarray([True, False, True, False]),
        )
        np.testing.assert_array_equal(
            bounded["trajectory_ids"],
            self.source["trajectory_ids"],
        )
        self.assertEqual(bounded["scenario_cache_version"], 3)
        self.assertEqual(summary["positive_direction_corrections"], 1)
        self.assertEqual(summary["negative_direction_corrections"], 1)
        self.assertNotIn("transition_states", bounded)

        n = int(self.env_configs[0]["num_facilities"])
        for row, expected_sign in ((0, 1.0), (2, -1.0)):
            anchor = facility_net_action_from_state(
                bounded["states"][row],
                self.env_configs[int(bounded["scenario_ids"][row])],
                settings=heuristic_settings_for_policy("mdl2"),
            )
            actual_delta = (
                bounded["actions"][row, n : 2 * n]
                - anchor[n : 2 * n]
            )
            pattern, _ = residual_pressure_patterns_from_state(
                bounded["states"][row],
                self.env_configs[int(bounded["scenario_ids"][row])],
            )
            self.assertGreater(
                float(np.dot(actual_delta, pattern) * expected_sign),
                0.0,
            )
            self.assertLessEqual(float(np.max(np.abs(actual_delta))), 0.1)

        first_mass = float(bounded["weights"][:2].sum())
        second_mass = float(bounded["weights"][2:].sum())
        self.assertAlmostEqual(first_mass, second_mass, places=6)

    def test_rejects_missing_signed_reagent_options(self) -> None:
        source = dict(self.source)
        source["option_advantages"] = source["option_advantages"][:, :2]
        source["option_feasible"] = source["option_feasible"][:, :2]
        source["option_groups"] = source["option_groups"][:2]
        source["option_epsilons"] = source["option_epsilons"][:2]
        source["option_signs"] = source["option_signs"][:2]

        with self.assertRaisesRegex(ValueError, "signed reagent"):
            make_bounded_reagent_demonstrations(
                source,
                env_configs=self.env_configs,
                anchor_settings=heuristic_settings_for_policy("mdl2"),
                min_advantage=500_000.0,
                max_residual_scale=0.1,
                coefficient_advantage_scale=2_000_000.0,
                min_positive_coefficient=0.25,
                weight_advantage_scale=1_000_000.0,
                weight_cap=3.0,
            )

    def test_selected_option_mode_retains_teacher_epsilon(self) -> None:
        _bounded, summary = make_bounded_reagent_demonstrations(
            self.source,
            env_configs=self.env_configs,
            anchor_settings=heuristic_settings_for_policy("mdl2"),
            min_advantage=500_000.0,
            max_residual_scale=0.4,
            coefficient_advantage_scale=2_000_000.0,
            min_positive_coefficient=0.25,
            weight_advantage_scale=1_000_000.0,
            weight_cap=3.0,
            coefficient_mode="selected_option",
        )

        self.assertEqual(summary["coefficient_mode"], "selected_option")
        np.testing.assert_allclose(
            summary["coefficient_quantiles"],
            np.asarray([0.8, 0.8, 0.8]),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
