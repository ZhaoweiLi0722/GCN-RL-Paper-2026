"""Tests for the graph shield candidate selector."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.models.gcn_shield_selector import GCNShieldSelectorAgent, candidate_count
from src.rl.config import load_config
from src.rl.experiment import build_env

DEV_CONFIG = "experiments/configs/2_clinic_patient_condition.json"


def _selector_config(seed: int = 0, anchor_policy: str = "pmyo"):
    return {
        "algorithm": "gcn_pmyo_shield_selector",
        "seed": seed,
        "env": load_config(DEV_CONFIG),
        "gcn_hidden_sizes": [8],
        "hidden_sizes": [16],
        "include_global_context": True,
        "residual_action": {
            "enabled": True,
            "base_policy": anchor_policy,
            "include_base_action_features": True,
        },
        "shield_selector": {
            "anchor_policy": anchor_policy,
            "candidate_epsilons": [0.005],
            "candidate_groups": ["reagent_transfer"],
            "teacher_policy_config": {
                "anchor_policy": anchor_policy,
                "shield_lookahead": 1,
                "shield_epsilons": [0.005],
                "candidate_groups": ["reagent_transfer"],
            },
            "hidden_sizes": [16],
            "lr": 0.001,
        },
    }


class GCNShieldSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")
        self.config = _selector_config()
        self.env = build_env(self.config, seed=0)

    def test_candidate_count_matches_reagent_transfer_space(self) -> None:
        self.assertEqual(candidate_count([0.005], ["reagent_transfer"]), 3)

    def test_selector_emits_valid_action(self) -> None:
        agent = GCNShieldSelectorAgent(self.env.observation_size, self.env.action_size, self.config)
        state = self.env.reset(seed=0)

        action = agent.select_action(state, env=self.env)

        self.assertEqual(action.shape, (self.env.action_size,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_confidence_threshold_falls_back_to_anchor(self) -> None:
        config = _selector_config()
        config["shield_selector"]["confidence_threshold"] = 0.8
        agent = GCNShieldSelectorAgent(self.env.observation_size, self.env.action_size, config)
        state = self.env.reset(seed=0)
        anchor_action = agent.anchor_policy.select_action(state, env=self.env)
        agent._predict_candidate_decision = lambda _state: (1, 0.5)  # type: ignore[method-assign]

        action = agent.select_action(state, env=self.env)

        np.testing.assert_allclose(action, anchor_action, atol=1e-6)

    def test_selector_pretrains_and_loads_checkpoint(self) -> None:
        agent = GCNShieldSelectorAgent(self.env.observation_size, self.env.action_size, self.config)
        summary = agent.pretrain_with_heuristic(
            self.env,
            {
                "episodes": 1,
                "max_steps_per_episode": 2,
                "epochs": 1,
                "batch_size": 2,
                "seed": 123,
                "class_weight_power": 0.0,
            },
        )

        self.assertEqual(summary["policy"], "pmyo_shield")
        self.assertGreater(summary["samples"], 0)
        self.assertTrue(math.isfinite(summary["final_loss"]))
        self.assertGreaterEqual(summary["train_accuracy"], 0.0)
        self.assertLessEqual(summary["train_accuracy"], 1.0)
        self.assertGreaterEqual(summary["anchor_label_fraction"], 0.0)
        self.assertLessEqual(summary["anchor_label_fraction"], 1.0)
        self.assertEqual(summary["candidate_count"], 3)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selector.pt"
            agent.save(path)
            loaded = GCNShieldSelectorAgent(self.env.observation_size, self.env.action_size, self.config)
            loaded.load_actor(path)
            state = self.env.reset(seed=5)
            action = loaded.select_action(state, env=self.env)
            self.assertEqual(action.shape, (self.env.action_size,))

    def test_selector_supports_mdl2_anchor(self) -> None:
        config = _selector_config(anchor_policy="mdl2")
        env = build_env(config, seed=6)
        agent = GCNShieldSelectorAgent(env.observation_size, env.action_size, config)
        state = env.reset(seed=6)

        action = agent.select_action(state, env=env)

        self.assertEqual(agent.anchor_policy_name, "mdl2")
        self.assertEqual(action.shape, (env.action_size,))


if __name__ == "__main__":
    unittest.main()
