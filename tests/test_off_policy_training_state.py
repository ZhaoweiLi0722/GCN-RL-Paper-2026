from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from src.rl.noise import OUNoise
from src.rl.replay_buffer import ReplayBuffer
from src.rl.training_state import (
    load_off_policy_training_state,
    save_off_policy_training_state,
    training_contract_sha256,
)


class _StubAgent:
    def __init__(self) -> None:
        self.algorithm = "stub_ddpg"
        self.seed = 7
        self.device = torch.device("cpu")
        self.actor = torch.nn.Linear(2, 1)
        self.actor_target = torch.nn.Linear(2, 1)
        self.critic = torch.nn.Linear(3, 1)
        self.critic_target = torch.nn.Linear(3, 1)
        self.correction_gate = None
        self.correction_safety_gate = None
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=1e-3,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=1e-3,
        )
        self.correction_gate_optimizer = None
        self.correction_safety_gate_optimizer = None
        self.total_updates = 13
        self.replay_buffer = ReplayBuffer(2, 1, capacity=8, seed=11)
        self.noise = OUNoise(1, seed=12, sigma=0.05)
        self.imitation_states = torch.tensor([[1.0, 2.0]])
        self.imitation_actions = torch.tensor([[0.25]])
        self.imitation_node_features = torch.tensor([[[3.0]]])
        self.imitation_weights = torch.tensor([0.75])
        self.imitation_rng = np.random.default_rng(13)


class OffPolicyTrainingStateTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(21)
        np.random.seed(22)
        torch.manual_seed(23)

    def test_round_trip_restores_agent_replay_and_rng_state(self):
        agent = _StubAgent()
        for index in range(5):
            agent.replay_buffer.add(
                np.array([index, index + 1], dtype=np.float32),
                np.array([index / 10], dtype=np.float32),
                float(index),
                np.array([index + 1, index + 2], dtype=np.float32),
                index == 4,
            )
        actor_before = {
            key: value.detach().clone()
            for key, value in agent.actor.state_dict().items()
        }
        config = {
            "algorithm": agent.algorithm,
            "seed": agent.seed,
            "gamma": 0.99,
            "checkpoint_dir": "ignored-a",
            "checkpoint_interval": 5,
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config=config,
                training={"next_episode": 5, "global_step": 260},
            )
            expected_replay = agent.replay_buffer.sample(3)
            expected_noise = agent.noise.sample()
            expected_imitation_indices = agent.imitation_rng.choice(
                20,
                size=4,
                replace=False,
            )
            expected_python = random.random()
            expected_numpy = np.random.random()
            expected_torch = torch.rand(2)

            with torch.no_grad():
                for parameter in agent.actor.parameters():
                    parameter.add_(10.0)
            agent.total_updates = 0
            agent.replay_buffer = ReplayBuffer(2, 1, capacity=8, seed=99)
            agent.noise = OUNoise(1, seed=99, sigma=0.05)
            agent.imitation_states = None

            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    **config,
                    "checkpoint_dir": "ignored-b",
                    "checkpoint_interval": 10,
                    "resume_training_state_path": str(path),
                },
            )

        self.assertEqual(metadata["next_episode"], 5)
        self.assertEqual(metadata["global_step"], 260)
        self.assertEqual(agent.total_updates, 13)
        for key, expected in actor_before.items():
            torch.testing.assert_close(agent.actor.state_dict()[key], expected)
        replay = agent.replay_buffer.sample(3)
        np.testing.assert_array_equal(replay.states, expected_replay.states)
        np.testing.assert_array_equal(replay.actions, expected_replay.actions)
        np.testing.assert_array_equal(agent.noise.sample(), expected_noise)
        np.testing.assert_array_equal(
            agent.imitation_rng.choice(20, size=4, replace=False),
            expected_imitation_indices,
        )
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)
        torch.testing.assert_close(torch.rand(2), expected_torch)
        torch.testing.assert_close(
            agent.imitation_states,
            torch.tensor([[1.0, 2.0]]),
        )

    def test_scientific_contract_rejects_changed_hyperparameter(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={"algorithm": agent.algorithm, "gamma": 0.99},
                training={"next_episode": 0, "global_step": 0},
            )
            with self.assertRaisesRegex(ValueError, "scientific contract"):
                load_off_policy_training_state(
                    agent,
                    path,
                    config={"algorithm": agent.algorithm, "gamma": 0.95},
                )

    def test_execution_paths_do_not_change_contract(self):
        left = training_contract_sha256(
            {
                "algorithm": "stub",
                "gamma": 0.99,
                "checkpoint_dir": "left",
                "training_state_checkpoint_interval": 5,
            }
        )
        right = training_contract_sha256(
            {
                "algorithm": "stub",
                "gamma": 0.99,
                "checkpoint_dir": "right",
                "training_state_checkpoint_interval": 10,
            }
        )
        self.assertEqual(left, right)

    def test_wrapped_replay_buffer_preserves_index_layout(self):
        buffer = ReplayBuffer(2, 1, capacity=3, seed=31)
        for index in range(5):
            buffer.add(
                np.array([index, -index], dtype=np.float32),
                np.array([index], dtype=np.float32),
                float(index),
                np.array([index + 1, -index - 1], dtype=np.float32),
                False,
            )
        state = buffer.state_dict()
        expected = buffer.sample(3)

        restored = ReplayBuffer(2, 1, capacity=3, seed=99)
        restored.load_state_dict(state)
        actual = restored.sample(3)

        self.assertEqual(restored.position, buffer.position)
        np.testing.assert_array_equal(actual.states, expected.states)
        np.testing.assert_array_equal(actual.actions, expected.actions)


if __name__ == "__main__":
    unittest.main()
