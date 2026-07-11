"""Pilot manifest configs load and build (Phase 7, group 3)."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.pilot_manifest import (
    HEURISTICS,
    LEARNED,
    STAGE_A_ENV,
    STAGE_A_ROSTER,
    build_agent_config,
    stage_a_manifest,
    stage_b_roster,
    train_budget,
)
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env


class PilotManifestTests(unittest.TestCase):
    def test_train_budget_meets_target(self) -> None:
        budget = train_budget(STAGE_A_ENV, target_steps=30_000)
        total = budget["num_episodes"] * budget["max_steps_per_episode"]
        self.assertGreaterEqual(total, 30_000)

    def test_stage_b_roster_dedupes_and_leads_with_backbone(self) -> None:
        # If the top backbone is itself an anchor, no duplicate appears.
        self.assertEqual(stage_b_roster("gcn_ddpg")[0], "gcn_ddpg")
        self.assertEqual(len(stage_b_roster("gcn_ddpg")), len(set(stage_b_roster("gcn_ddpg"))))
        self.assertEqual(stage_b_roster("gcn_sac")[0], "gcn_sac")

    def test_every_config_builds_and_steps(self) -> None:
        try:
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            torch = None
        for algorithm in STAGE_A_ROSTER:
            with self.subTest(algorithm=algorithm):
                if algorithm in LEARNED and torch is None:  # pragma: no cover
                    continue
                config = build_agent_config(algorithm, STAGE_A_ENV, seed=0, target_steps=1)
                env = build_env(config, seed=0)
                agent = get_agent_class(algorithm)(env.observation_size, env.action_size, config)
                state = env.reset(seed=0)
                action = agent.select_action(state, explore=True, env=env)
                self.assertEqual(action.shape, (env.action_size,))
                self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))

    def test_graph_agents_get_patient_graph_spec(self) -> None:
        try:
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")
        config = build_agent_config("gcn_sac", STAGE_A_ENV, seed=0, target_steps=1)
        env = build_env(config, seed=0)
        agent = get_agent_class("gcn_sac")(env.observation_size, env.action_size, config)
        # Summary columns must be part of the node features (patient-aware spec).
        self.assertGreater(agent.graph_spec.patient_summary_width, 0)

    def test_manifest_roster_shape(self) -> None:
        manifest = stage_a_manifest()
        self.assertEqual(manifest["stage"], "A")
        for name in HEURISTICS:
            self.assertIn(name, manifest["roster"])


if __name__ == "__main__":
    unittest.main()
