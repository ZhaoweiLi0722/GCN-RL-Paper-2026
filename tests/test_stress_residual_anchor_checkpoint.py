"""Tests for replay-backed residual-anchor stress validation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluation.stress_residual_anchor_checkpoint import stress_checkpoint_anchor
from src.baselines.heuristics import MeanDemandLookahead2Policy
from src.env.capacity_planning import CapacityPlanningEnv, make_20_clinic_config
from src.rl.networks import require_torch, torch


class ResidualAnchorCheckpointStressTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_stress_uses_replay_and_actor_environment_metadata(self) -> None:
        require_torch()
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=7)
        state = env.reset(seed=7)
        action = MeanDemandLookahead2Policy().select_action(state, env=env)
        next_state, _reward, _done, _info = env.step(action)
        states = np.stack((state, next_state)).astype(np.float32)
        next_states = np.stack((next_state, state)).astype(np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_state_path = root / "training_state.pt"
            actor_checkpoint_path = root / "actor.pt"
            torch.save(
                {
                    "algorithm": "gcn_residual_mdl2_network_ddpg_afd",
                    "agent": {
                        "replay_buffer": {
                            "size": 2,
                            "state_dim": int(states.shape[1]),
                            "states": states,
                            "next_states": next_states,
                        }
                    },
                },
                training_state_path,
            )
            torch.save(
                {
                    "algorithm": "gcn_residual_mdl2_network_ddpg_afd",
                    "graph_spec": {"env_config": asdict(env.config)},
                },
                actor_checkpoint_path,
            )

            result = stress_checkpoint_anchor(
                training_state_path=training_state_path,
                actor_checkpoint_path=actor_checkpoint_path,
                anchor_policy="mdl2",
                calls=100,
                minimum_replay_rows=2,
            )

        self.assertEqual(result["calls"], 100)
        self.assertEqual(result["replay_rows"], 2)
        self.assertEqual(len(result["baseline_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
