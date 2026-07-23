"""GNN-TD3 builds and steps on the patient env (Phase 6, group 4)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.baselines.heuristics import (
    facility_net_action_from_state,
    heuristic_settings_for_policy,
    PatientPriorityMyopicPolicy,
)
from src.rl.config import load_config
from src.rl.experiment import build_env

DEV_CONFIG = "experiments/configs/2_clinic_patient_condition.json"


def _agent_config(seed: int = 0):
    return {
        "seed": seed,
        "env": load_config(DEV_CONFIG),
        "gcn_hidden_sizes": [16, 16],
        "hidden_sizes": [32, 32],
        "batch_size": 8,
    }


def _residual_agent_config(seed: int = 0):
    config = _agent_config(seed)
    config["algorithm"] = "gcn_residual_pmyo_td3"
    config["residual_action"] = {
        "enabled": True,
        "base_policy": "pmyo",
        "include_base_action_features": True,
        "zero_init_actor": True,
        "scale": 0.05,
        "group_scales": {
            "specimen_transfer": 0.0,
            "reagent_transfer": 0.0,
            "capacity_transfer": 0.0,
            "replenishment": 0.02,
        },
        "positive_only_groups": ["replenishment"],
        "state_gate": {
            "enabled": True,
            "groups": ["replenishment"],
            "threshold": 0.0,
        },
        "l2_weight": 0.05,
    }
    return config


class GCNTD3PatientTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")
        from src.models.gcn_td3 import GCNTD3Agent

        self.agent_cls = GCNTD3Agent
        self.env = build_env({"env": load_config(DEV_CONFIG)}, seed=0)

    def test_builds_and_emits_valid_action(self) -> None:
        agent = self.agent_cls(self.env.observation_size, self.env.action_size, _agent_config())
        state = self.env.reset(seed=0)
        action = agent.select_action(state, explore=True, env=self.env)
        self.assertEqual(action.shape, (self.env.action_size,))
        self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))

    def test_observe_update_cycle_returns_finite_losses(self) -> None:
        agent = self.agent_cls(self.env.observation_size, self.env.action_size, _agent_config())
        state = self.env.reset(seed=0)
        # policy_delay=2, so run enough steps to trigger an actor update too.
        for _ in range(20):
            action = agent.select_action(state, explore=True, env=self.env)
            next_state, reward, done, _info = self.env.step(action)
            agent.observe(state, action, reward, next_state, done)
            metrics = agent.update()
            state = next_state if not done else self.env.reset(seed=1)
        self.assertIn("critic1_loss", metrics)
        for value in metrics.values():
            self.assertTrue(math.isfinite(value))

    def test_residual_zero_output_returns_patient_priority_anchor(self) -> None:
        config = _residual_agent_config()
        agent = self.agent_cls(self.env.observation_size, self.env.action_size, config)
        state = self.env.reset(seed=0)

        action = agent.select_action(state, explore=False, env=self.env)
        anchor = facility_net_action_from_state(
            state,
            config["env"],
            settings=heuristic_settings_for_policy("pmyo", {}),
        )

        np.testing.assert_allclose(action, anchor, atol=1e-6)

    def test_transfer_residual_zero_output_matches_live_geo_patient_anchor(self) -> None:
        env_config = load_config("experiments/configs/20_clinic_patient_condition_geo.json")
        config = load_config("configs/gcn_td3_20_clinic.yaml")
        config.update(
            {
                "seed": 3,
                "algorithm": "gcn_residual_pmyo_transfer_td3",
                "gcn_hidden_sizes": [8],
                "hidden_sizes": [16],
                "batch_size": 2,
                "env": env_config,
                "residual_action": {
                    "enabled": True,
                    "base_policy": "pmyo",
                    "include_base_action_features": True,
                    "zero_init_actor": True,
                    "scale": 0.05,
                    "group_scales": {
                        "specimen_transfer": 0.0,
                        "reagent_transfer": 0.01,
                        "capacity_transfer": 0.01,
                        "replenishment": 0.02,
                    },
                    "center_groups": ["reagent_transfer", "capacity_transfer"],
                    "positive_only_groups": ["replenishment"],
                    "state_gate": {
                        "enabled": True,
                        "groups": ["replenishment"],
                        "threshold": 0.0,
                    },
                },
            }
        )
        env = build_env(config, seed=3)
        state = env.reset(seed=3)
        anchor_policy = PatientPriorityMyopicPolicy()
        for _ in range(6):
            anchor_action = anchor_policy.select_action(state, explore=False, env=env)
            state, _reward, done, _info = env.step(anchor_action)
            if done:
                break
        agent = self.agent_cls(env.observation_size, env.action_size, config)

        action = agent.select_action(state, explore=False, env=env)
        anchor = anchor_policy.select_action(state, explore=False, env=env)

        np.testing.assert_allclose(action, anchor, atol=1e-6)

    def test_pressure_projection_aligns_td3_residual_with_state_pattern(self) -> None:
        from src.rl.networks import torch

        config = _residual_agent_config(seed=6)
        config["residual_action"]["pressure_projection"] = {
            "enabled": True,
            "groups": ["reagent_transfer"],
        }
        config["residual_action"]["center_groups"] = ["reagent_transfer"]
        agent = self.agent_cls(self.env.observation_size, self.env.action_size, config)
        n = self.env.config.num_facilities
        state = self.env.reset(seed=6)
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        network_action = torch.zeros((1, self.env.action_size), dtype=torch.float32)
        network_action[0, n : 2 * n] = torch.linspace(-1.0, 1.0, n)

        residual = agent._policy_residuals_tensor(state_tensor, network_action).detach().numpy()[0]
        pattern = agent._residual_pressure_patterns_tensor(state_tensor)["resource"][0]
        current = network_action[:, n : 2 * n]
        denominator = pattern.pow(2).sum().clamp_min(1e-6)
        coefficient = float((current[0] * pattern).sum() / denominator)
        expected = (coefficient * pattern).numpy()

        np.testing.assert_allclose(residual[n : 2 * n], expected, atol=1e-6)
        np.testing.assert_allclose(residual[2 * n : 3 * n], np.zeros(n), atol=1e-6)

    def test_residual_td3_supports_supervised_fit(self) -> None:
        config = _residual_agent_config()
        config["batch_size"] = 2
        agent = self.agent_cls(self.env.observation_size, self.env.action_size, config)
        states = []
        actions = []
        state = self.env.reset(seed=1)
        for _ in range(3):
            action = facility_net_action_from_state(
                state,
                config["env"],
                settings=heuristic_settings_for_policy("pmyo", {}),
            )
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
            state, _reward, done, _info = self.env.step(action)
            if done:
                break

        summary = agent.fit_action_batch(
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            {"epochs": 1, "batch_size": 2},
        )

        self.assertEqual(summary["samples"], len(states))
        self.assertEqual(summary["target_mode"], "residual")
        self.assertTrue(math.isfinite(summary["final_loss"]))

    def test_freeze_actor_updates_keeps_actor_out_of_td3_update(self) -> None:
        config = _residual_agent_config(seed=4)
        config["batch_size"] = 2
        config["policy_delay"] = 1
        config["freeze_actor_updates"] = True
        agent = self.agent_cls(self.env.observation_size, self.env.action_size, config)
        state = self.env.reset(seed=4)

        metrics = {}
        for _ in range(3):
            action = agent.select_action(state, explore=False, env=self.env)
            next_state, reward, done, _info = self.env.step(action)
            agent.observe(state, action, reward, next_state, done)
            metrics = agent.update()
            state = next_state if not done else self.env.reset(seed=5)

        self.assertEqual(metrics.get("actor_frozen"), 1.0)
        self.assertNotIn("actor_loss", metrics)

    def test_anchor_advantage_actor_loss_reports_diagnostics(self) -> None:
        config = _residual_agent_config(seed=7)
        config["batch_size"] = 2
        config["policy_delay"] = 1
        config["anchor_advantage_actor_loss"] = {
            "enabled": True,
            "temperature": 0.05,
            "use_twin_min": True,
        }
        agent = self.agent_cls(self.env.observation_size, self.env.action_size, config)
        state = self.env.reset(seed=7)

        metrics = {}
        for _ in range(3):
            action = agent.select_action(state, explore=False, env=self.env)
            next_state, reward, done, _info = self.env.step(action)
            agent.observe(state, action, reward, next_state, done)
            metrics = agent.update()
            state = next_state if not done else self.env.reset(seed=8)

        self.assertIn("actor_anchor_advantage_mean", metrics)
        self.assertIn("actor_anchor_advantage_positive_fraction", metrics)
        self.assertTrue(math.isfinite(metrics["actor_anchor_advantage_mean"]))
        self.assertGreaterEqual(metrics["actor_anchor_advantage_positive_fraction"], 0.0)
        self.assertLessEqual(metrics["actor_anchor_advantage_positive_fraction"], 1.0)

    def test_patient_service_proxy_actor_loss_reports_diagnostics(self) -> None:
        config = _residual_agent_config(seed=8)
        config["batch_size"] = 2
        config["policy_delay"] = 1
        config["patient_service_proxy_actor_loss"] = {
            "enabled": True,
            "group": "replenishment",
            "weight": 0.02,
            "cost_weight": 0.01,
            "low_pressure_weight": 0.005,
        }
        agent = self.agent_cls(self.env.observation_size, self.env.action_size, config)
        state = self.env.reset(seed=8)

        metrics = {}
        for _ in range(3):
            action = agent.select_action(state, explore=False, env=self.env)
            next_state, reward, done, _info = self.env.step(action)
            agent.observe(state, action, reward, next_state, done)
            metrics = agent.update()
            state = next_state if not done else self.env.reset(seed=9)

        self.assertIn("actor_patient_service_proxy_alignment", metrics)
        self.assertIn("actor_patient_service_proxy_low_pressure_penalty", metrics)
        self.assertIn("actor_patient_service_proxy_cost_penalty", metrics)
        self.assertTrue(math.isfinite(metrics["actor_patient_service_proxy_alignment"]))
        self.assertTrue(math.isfinite(metrics["actor_patient_service_proxy_low_pressure_penalty"]))
        self.assertTrue(math.isfinite(metrics["actor_patient_service_proxy_cost_penalty"]))


if __name__ == "__main__":
    unittest.main()
