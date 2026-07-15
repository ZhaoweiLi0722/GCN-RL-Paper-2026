"""Integration: build the patient env from config and run agents (Phase 4, group 7)."""

from __future__ import annotations

import unittest

import numpy as np

from src.baselines.heuristics import (
    facility_net_action_from_state,
    get_heuristic_class,
    heuristic_settings_for_policy,
)
from src.env.capacity_planning import CapacityPlanningEnv
from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.rl.config import load_config
from src.rl.experiment import build_env

DEV_CONFIG = "experiments/configs/2_clinic_patient_condition.json"
CLINIC20_CONFIG = "experiments/configs/20_clinic_patient_condition.json"


def _wrapped(env_config_path: str) -> dict:
    return {"algorithm": "myo", "env": load_config(env_config_path)}


class BuildEnvRoutingTests(unittest.TestCase):
    def test_env_type_marker_builds_patient_env(self) -> None:
        env = build_env(_wrapped(DEV_CONFIG), seed=0)
        self.assertIsInstance(env, PatientConditionCapacityEnv)

    def test_plain_config_still_builds_base_env(self) -> None:
        env = build_env({"env": load_config("experiments/configs/20_clinic_capacity_planning.json")}, seed=0)
        self.assertIsInstance(env, CapacityPlanningEnv)
        self.assertNotIsInstance(env, PatientConditionCapacityEnv)

    def test_20_clinic_patient_config_builds(self) -> None:
        env = build_env(_wrapped(CLINIC20_CONFIG), seed=0)
        self.assertEqual(env.config.num_facilities, 20)
        # base (7 features x 20) + summary (7 x 20).
        self.assertEqual(env.observation_size, env.base_observation_size + 20 * env.summary_width)


class SmokeRunTests(unittest.TestCase):
    def test_heuristic_runs_end_to_end(self) -> None:
        env = build_env(_wrapped(DEV_CONFIG), seed=0)
        agent = get_heuristic_class("myo")(env.observation_size, env.action_size, {})
        state = env.reset(seed=0)
        info = {}
        for _ in range(env.config.episode_horizon):
            action = agent.select_action(state, explore=False, env=env)
            state, reward, done, info = env.step(action)
            self.assertIsInstance(reward, float)
            if done:
                break
        for key in ("patients_lost", "material_wasted", "eligibility_rate"):
            self.assertIn(key, info)

    def test_flat_ddpg_runs_end_to_end(self) -> None:
        try:
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")
        from src.baselines.flat_ddpg import FlatDDPGAgent

        env = build_env(_wrapped(DEV_CONFIG), seed=0)
        agent = FlatDDPGAgent(env.observation_size, env.action_size, {"seed": 0, "hidden_sizes": [32, 32]})
        state = env.reset(seed=0)
        for _ in range(6):
            action = agent.select_action(state, explore=True, env=env)
            next_state, reward, done, _info = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            if done:
                break
        self.assertEqual(state.shape, (env.observation_size,))

    def test_flat_residual_ddpg_runs_on_patient_observation(self) -> None:
        try:
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")
        from src.baselines.flat_ddpg import FlatDDPGAgent

        config = {
            "seed": 0,
            "hidden_sizes": [32, 32],
            "batch_size": 2,
            "normalize_observations": True,
            "residual_action": {
                "enabled": True,
                "base_policy": "mdl2",
                "zero_init_actor": True,
                "scale": 0.05,
                "group_scales": {
                    "specimen_transfer": 0.0,
                    "reagent_transfer": 0.02,
                    "capacity_transfer": 0.02,
                    "replenishment": 0.05,
                },
            },
            "env": load_config(DEV_CONFIG),
        }
        env = build_env(config, seed=0)
        agent = FlatDDPGAgent(env.observation_size, env.action_size, config)
        state = env.reset(seed=0)

        action = agent.select_action(state, explore=False, env=env)
        base_action = facility_net_action_from_state(
            state,
            config["env"],
            settings=heuristic_settings_for_policy("mdl2"),
        )

        self.assertEqual(action.shape, (env.action_size,))
        self.assertTrue((action >= -1.0).all())
        self.assertTrue((action <= 1.0).all())
        np.testing.assert_allclose(action, base_action, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
