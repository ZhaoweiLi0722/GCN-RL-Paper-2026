"""Smoke tests for SAC and PPO baselines."""

from __future__ import annotations

import unittest

import numpy as np

from src.baselines.ppo import PPOAgent
from src.baselines.sac import SACAgent
from src.env.capacity_planning import CapacityPlanningEnv, make_20_clinic_config
from src.rl.experiment import train_off_policy_agent

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def _base_config(algorithm: str) -> dict:
    config = {
        "algorithm": algorithm,
        "seed": 0,
        "num_episodes": 1,
        "max_steps_per_episode": 3,
        "batch_size": 2,
        "hidden_sizes": [32],
        "checkpoint_interval": 99,
        "env": {
            "num_facilities": 20,
            "production_lead_time": 3,
            "action_mode": "facility_net",
            "include_supplier_state": True,
            "include_central_capacity_hub": True,
        },
    }
    if algorithm == "ppo":
        config.update({"rollout_length": 3, "minibatch_size": 2, "train_epochs": 1})
    return config


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SACPPOSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=3), seed=0)
        self.state = self.env.reset(seed=0)

    def test_sac_selects_valid_action(self) -> None:
        agent = SACAgent(self.env.observation_size, self.env.action_size, _base_config("sac"))

        action = agent.select_action(self.state, explore=True, env=self.env)

        self.assertEqual(action.shape, (self.env.action_size,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_ppo_selects_valid_action(self) -> None:
        agent = PPOAgent(self.env.observation_size, self.env.action_size, _base_config("ppo"))

        action = agent.select_action(self.state, explore=True, env=self.env)

        self.assertEqual(action.shape, (self.env.action_size,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_sac_tiny_training_runs(self) -> None:
        agent = SACAgent(self.env.observation_size, self.env.action_size, _base_config("sac"))

        rows = train_off_policy_agent(agent, self.env, _base_config("sac"))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["algorithm"], "sac")

    def test_ppo_tiny_training_runs(self) -> None:
        agent = PPOAgent(self.env.observation_size, self.env.action_size, _base_config("ppo"))

        rows = train_off_policy_agent(agent, self.env, _base_config("ppo"))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["algorithm"], "ppo")


if __name__ == "__main__":
    unittest.main()
