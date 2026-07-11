"""GNN-SAC builds and steps on the patient env (Phase 6, group 5)."""

from __future__ import annotations

import math
import unittest

import numpy as np

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


class GCNSACPatientTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")
        from src.models.gcn_sac import GCNSACAgent

        self.agent_cls = GCNSACAgent
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
        for _ in range(15):
            action = agent.select_action(state, explore=True, env=self.env)
            next_state, reward, done, _info = self.env.step(action)
            agent.observe(state, action, reward, next_state, done)
            metrics = agent.update()
            state = next_state if not done else self.env.reset(seed=1)
        self.assertIn("actor_loss", metrics)
        self.assertIn("alpha", metrics)
        for value in metrics.values():
            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
