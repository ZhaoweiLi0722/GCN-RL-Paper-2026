from pathlib import Path
import copy
import json
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from src.env.capacity_planning import CapacityPlanningConfig
from src.env.patient_capacity_planning import (
    PatientConditionCapacityEnv,
    PatientEnvConfig,
)
from src.env.patient_condition import PatientConditionConfig
from src.rl.noise import OUNoise
from src.rl.experiment import train_off_policy_agent
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
        self.pretrain_reference_actor = copy.deepcopy(self.actor)
        self.pretrain_reference_actor.eval()
        for parameter in self.pretrain_reference_actor.parameters():
            parameter.requires_grad_(False)
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
        self.critic_teacher_advantage_rng = np.random.default_rng(14)

    def save(self, path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict()}, output)


class _OnlinePreparationStubAgent(_StubAgent):
    def __init__(self) -> None:
        super().__init__()
        self.online_preparation_calls = 0
        self.online_preparation_start_episodes = []

    def prepare_online_finetuning(self, *, start_episode=0):
        self.online_preparation_calls += 1
        self.online_preparation_start_episodes.append(start_episode)
        return {"online_critic_lr": 1e-4}


class _TwinCriticStubAgent(_StubAgent):
    def __init__(self) -> None:
        super().__init__()
        self.actor_reference = copy.deepcopy(self.actor)
        self.critic2 = torch.nn.Linear(3, 1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(),
            lr=2e-3,
        )


class _StatefulStubEnv:
    def __init__(self) -> None:
        self.value = 3

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = int(state["value"])


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
        reference_before = {
            key: value.detach().clone()
            for key, value in (
                agent.pretrain_reference_actor.state_dict().items()
            )
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
            expected_critic_advantage_indices = (
                agent.critic_teacher_advantage_rng.choice(
                    20,
                    size=4,
                    replace=False,
                )
            )
            expected_python = random.random()
            expected_numpy = np.random.random()
            expected_torch = torch.rand(2)

            with torch.no_grad():
                for parameter in agent.actor.parameters():
                    parameter.add_(10.0)
                for parameter in (
                    agent.pretrain_reference_actor.parameters()
                ):
                    parameter.add_(20.0)
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
        for key, expected in reference_before.items():
            torch.testing.assert_close(
                agent.pretrain_reference_actor.state_dict()[key],
                expected,
            )
        replay = agent.replay_buffer.sample(3)
        np.testing.assert_array_equal(replay.states, expected_replay.states)
        np.testing.assert_array_equal(replay.actions, expected_replay.actions)
        np.testing.assert_array_equal(agent.noise.sample(), expected_noise)
        np.testing.assert_array_equal(
            agent.imitation_rng.choice(20, size=4, replace=False),
            expected_imitation_indices,
        )
        np.testing.assert_array_equal(
            agent.critic_teacher_advantage_rng.choice(
                20,
                size=4,
                replace=False,
            ),
            expected_critic_advantage_indices,
        )
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)
        torch.testing.assert_close(torch.rand(2), expected_torch)
        torch.testing.assert_close(
            agent.imitation_states,
            torch.tensor([[1.0, 2.0]]),
        )

    def test_round_trip_restores_twin_critic_state(self):
        agent = _TwinCriticStubAgent()
        expected_critic2 = {
            key: value.detach().clone()
            for key, value in agent.critic2.state_dict().items()
        }
        expected_target = {
            key: value.detach().clone()
            for key, value in agent.critic2_target.state_dict().items()
        }
        config = {"algorithm": agent.algorithm, "seed": agent.seed}

        with TemporaryDirectory() as directory:
            path = Path(directory) / "twin-state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config=config,
                training={"next_episode": 5, "global_step": 260},
            )
            with torch.no_grad():
                for parameter in agent.critic2.parameters():
                    parameter.add_(10.0)
                for parameter in agent.critic2_target.parameters():
                    parameter.sub_(10.0)
            load_off_policy_training_state(
                agent,
                path,
                config=config,
            )

        for key, expected in expected_critic2.items():
            torch.testing.assert_close(
                agent.critic2.state_dict()[key],
                expected,
            )
        for key, expected in expected_target.items():
            torch.testing.assert_close(
                agent.critic2_target.state_dict()[key],
                expected,
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

    def test_read_only_model_audit_can_skip_environment_restore(self):
        agent = _StubAgent()
        env = _StatefulStubEnv()
        config = {"algorithm": agent.algorithm, "seed": agent.seed}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config=config,
                env=env,
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config=config,
                restore_environment=False,
            )
        self.assertEqual(metadata["next_episode"], 0)

    def test_episode_zero_state_allows_declared_actor_frequency_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "actor_update_frequency": 4,
                },
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "actor_update_frequency": 2,
                    "preonline_fork_allowed_overrides": [
                        "actor_update_frequency"
                    ],
                },
            )

        self.assertEqual(
            metadata["preonline_fork_overrides"],
            {
                "actor_update_frequency": {
                    "checkpoint": 4,
                    "current": 2,
                }
            },
        )

    def test_episode_zero_resume_reports_update_schedule_fork(self):
        source_agent = _StubAgent()
        env = SimpleNamespace(config=SimpleNamespace(episode_horizon=1))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "preonline.pt"
            source_config = {
                "algorithm": source_agent.algorithm,
                "seed": source_agent.seed,
                "num_episodes": 0,
                "actor_update_frequency": 2,
                "updates_per_update": 1,
            }
            save_off_policy_training_state(
                source_agent,
                state_path,
                config=source_config,
                training={
                    "next_episode": 0,
                    "global_step": 0,
                    "rows": [],
                    "pretrain_summary": {},
                    "advantage_distillation_summary": {},
                    "pretrain_checkpoint_path": "",
                },
            )
            report = {}
            train_off_policy_agent(
                _StubAgent(),
                env,
                {
                    **source_config,
                    "actor_update_frequency": 4,
                    "updates_per_update": 2,
                    "checkpoint_dir": str(root / "fork"),
                    "resume_training_state_path": str(state_path),
                    "preonline_fork_allowed_overrides": [
                        "actor_update_frequency",
                        "updates_per_update",
                    ],
                },
                pretrain_report_out=report,
            )

        self.assertEqual(
            report["preonline_fork_overrides"],
            {
                "actor_update_frequency": {
                    "checkpoint": 2,
                    "current": 4,
                },
                "updates_per_update": {
                    "checkpoint": 1,
                    "current": 2,
                },
            },
        )

    def test_episode_zero_state_allows_reference_weight_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "pretrain_reference_actor_loss": {
                        "enabled": True,
                        "weight": 0.0,
                    },
                },
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "pretrain_reference_actor_loss": {
                        "enabled": True,
                        "weight": 25.0,
                    },
                    "preonline_fork_allowed_overrides": [
                        "pretrain_reference_actor_loss.weight"
                    ],
                },
            )

        self.assertEqual(
            metadata["preonline_fork_overrides"],
            {
                "pretrain_reference_actor_loss.weight": {
                    "checkpoint": 0.0,
                    "current": 25.0,
                }
            },
        )

    def test_episode_zero_state_allows_reference_action_space_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "pretrain_reference_actor_loss": {
                        "enabled": True,
                        "weight": 500.0,
                    },
                },
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "pretrain_reference_actor_loss": {
                        "enabled": True,
                        "weight": 500.0,
                        "action_space": "executed_specimen_lots",
                    },
                    "preonline_fork_allowed_overrides": [
                        "pretrain_reference_actor_loss.action_space"
                    ],
                },
            )

        self.assertEqual(
            metadata["preonline_fork_overrides"],
            {
                "pretrain_reference_actor_loss.action_space": {
                    "checkpoint": None,
                    "current": "executed_specimen_lots",
                }
            },
        )

    def test_episode_zero_state_allows_training_budget_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "history_screen": {"online_episodes": 8},
                    "num_episodes": 8,
                },
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "history_screen": {"online_episodes": 24},
                    "num_episodes": 24,
                    "preonline_fork_allowed_overrides": [
                        "history_screen.online_episodes",
                        "num_episodes"
                    ],
                },
            )

        self.assertEqual(
            metadata["preonline_fork_overrides"],
            {
                "history_screen.online_episodes": {
                    "checkpoint": 8,
                    "current": 24,
                },
                "num_episodes": {
                    "checkpoint": 8,
                    "current": 24,
                }
            },
        )

    def test_episode_zero_state_allows_online_critic_lr_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={"algorithm": agent.algorithm},
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "online_critic_lr": 1e-4,
                    "preonline_fork_allowed_overrides": [
                        "online_critic_lr"
                    ],
                },
            )

        self.assertEqual(
            metadata["preonline_fork_overrides"],
            {
                "online_critic_lr": {
                    "checkpoint": None,
                    "current": 1e-4,
                }
            },
        )

    def test_episode_zero_state_allows_online_actor_lr_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={"algorithm": agent.algorithm},
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "online_actor_lr": 3e-5,
                    "preonline_fork_allowed_overrides": [
                        "online_actor_lr"
                    ],
                },
            )

        self.assertEqual(
            metadata["preonline_fork_overrides"],
            {
                "online_actor_lr": {
                    "checkpoint": None,
                    "current": 3e-5,
                }
            },
        )

    def test_episode_zero_state_allows_anchor_advantage_actor_loss_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={"algorithm": agent.algorithm},
                training={"next_episode": 0},
            )
            objective = {
                "enabled": True,
                "margin": 0.00025,
                "temperature": 0.00025,
                "negative_penalty_weight": 0.0,
            }
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "anchor_advantage_actor_loss": objective,
                    "preonline_fork_allowed_overrides": [
                        "anchor_advantage_actor_loss"
                    ],
                },
            )

        self.assertEqual(
            metadata["preonline_fork_overrides"],
            {
                "anchor_advantage_actor_loss": {
                    "checkpoint": None,
                    "current": objective,
                }
            },
        )

    def test_episode_zero_state_allows_online_gate_alignment_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "residual_action": {
                        "correction_gate": {"enabled": True}
                    },
                },
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "residual_action": {
                        "correction_gate": {
                            "enabled": True,
                            "align_online_policy": True,
                        }
                    },
                    "preonline_fork_allowed_overrides": [
                        "residual_action.correction_gate.align_online_policy"
                    ],
                },
            )

        self.assertEqual(
            metadata["preonline_fork_overrides"],
            {
                "residual_action.correction_gate.align_online_policy": {
                    "checkpoint": None,
                    "current": True,
                }
            },
        )

    def test_episode_zero_state_allows_hard_actor_gate_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "residual_action": {
                        "correction_gate": {
                            "enabled": True,
                            "align_online_policy": True,
                        }
                    },
                },
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "residual_action": {
                        "correction_gate": {
                            "enabled": True,
                            "align_online_policy": True,
                            "hard_actor_policy": True,
                        }
                    },
                    "preonline_fork_allowed_overrides": [
                        "residual_action.correction_gate.hard_actor_policy"
                    ],
                },
            )

        self.assertEqual(
            metadata["preonline_fork_overrides"],
            {
                "residual_action.correction_gate.hard_actor_policy": {
                    "checkpoint": None,
                    "current": True,
                }
            },
        )

    def test_episode_zero_state_allows_actor_proposal_gradient_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "residual_action": {
                        "correction_gate": {
                            "enabled": True,
                            "align_online_policy": True,
                            "include_proposed_residual_features": True,
                        }
                    },
                },
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "residual_action": {
                        "correction_gate": {
                            "enabled": True,
                            "align_online_policy": True,
                            "include_proposed_residual_features": True,
                            "differentiate_actor_proposal": True,
                        }
                    },
                    "preonline_fork_allowed_overrides": [
                        "residual_action.correction_gate."
                        "differentiate_actor_proposal"
                    ],
                },
            )

        self.assertEqual(
            metadata["preonline_fork_overrides"],
            {
                "residual_action.correction_gate."
                "differentiate_actor_proposal": {
                    "checkpoint": None,
                    "current": True,
                }
            },
        )

    def test_training_loop_prepares_online_optimizer_once(self):
        agent = _OnlinePreparationStubAgent()
        env = SimpleNamespace(config=SimpleNamespace(episode_horizon=1))
        setup_calls = []
        report = {}

        with TemporaryDirectory() as directory:
            rows = train_off_policy_agent(
                agent,
                env,
                {
                    "algorithm": agent.algorithm,
                    "seed": agent.seed,
                    "num_episodes": 0,
                    "checkpoint_dir": directory,
                },
                preonline_setup=lambda current_agent, current_env: (
                    setup_calls.append((current_agent, current_env))
                    or {"online_imitation_samples": 3}
                ),
                pretrain_report_out=report,
            )

        self.assertEqual(rows, [])
        self.assertEqual(setup_calls, [(agent, env)])
        self.assertEqual(report["online_imitation_samples"], 3)
        self.assertEqual(agent.online_preparation_calls, 1)
        self.assertEqual(agent.online_preparation_start_episodes, [0])

    def test_episode_zero_state_allows_critic_calibration_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={"algorithm": agent.algorithm},
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "critic_teacher_advantage_calibration": {
                        "enabled": True,
                        "updates": 100,
                        "target_scale": 1.0,
                    },
                    "preonline_fork_allowed_overrides": [
                        "critic_teacher_advantage_calibration",
                    ],
                },
            )

        self.assertEqual(
            set(metadata["preonline_fork_overrides"]),
            {
                "critic_teacher_advantage_calibration",
            },
        )

    def test_episode_zero_state_allows_online_imitation_cache_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={"algorithm": agent.algorithm},
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "online_imitation_regularization": {
                        "enabled": True,
                        "demonstration_path": "teacher.npz",
                    },
                    "preonline_fork_allowed_overrides": [
                        "online_imitation_regularization",
                    ],
                },
            )

        self.assertEqual(
            set(metadata["preonline_fork_overrides"]),
            {"online_imitation_regularization"},
        )

    def test_episode_zero_state_allows_anchor_relative_reward_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "residual_action": {
                        "online_reward_mode": "environment",
                    },
                },
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "residual_action": {
                        "online_reward_mode": "one_step_anchor_relative",
                    },
                    "preonline_fork_allowed_overrides": [
                        "residual_action.online_reward_mode",
                    ],
                },
            )

        self.assertEqual(
            set(metadata["preonline_fork_overrides"]),
            {"residual_action.online_reward_mode"},
        )

    def test_episode_zero_state_allows_n_step_reward_horizon_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "residual_action": {
                        "online_reward_mode": "environment",
                        "online_reward_n_step_horizon": 1,
                    },
                },
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "residual_action": {
                        "online_reward_mode": "n_step_anchor_relative",
                        "online_reward_n_step_horizon": 4,
                    },
                    "preonline_fork_allowed_overrides": [
                        "residual_action.online_reward_mode",
                        "residual_action.online_reward_n_step_horizon",
                    ],
                },
            )

        self.assertEqual(
            set(metadata["preonline_fork_overrides"]),
            {
                "residual_action.online_reward_mode",
                "residual_action.online_reward_n_step_horizon",
            },
        )

    def test_episode_zero_state_allows_q_filtered_reference_fork(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "pretrain_reference_actor_loss": {
                        "mode": "uniform",
                        "q_filter_margin": 0.0,
                    },
                },
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "pretrain_reference_actor_loss": {
                        "mode": "critic_q_filter",
                        "q_filter_margin": 0.0,
                    },
                    "preonline_fork_allowed_overrides": [
                        "pretrain_reference_actor_loss.mode",
                    ],
                },
            )

        self.assertEqual(
            set(metadata["preonline_fork_overrides"]),
            {"pretrain_reference_actor_loss.mode"},
        )

    def test_episode_zero_state_allows_online_advantage_self_imitation_fork(
        self,
    ):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                },
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "online_advantage_self_imitation": {
                        "enabled": True,
                        "weight": 1.0,
                        "release_pretrain_reference": True,
                    },
                    "preonline_fork_allowed_overrides": [
                        "online_advantage_self_imitation",
                    ],
                },
            )

        self.assertEqual(
            set(metadata["preonline_fork_overrides"]),
            {"online_advantage_self_imitation"},
        )

    def test_episode_zero_state_allows_exploration_sigma_fork(self):
        source_agent = _StubAgent()
        saved_rng_state = copy.deepcopy(
            source_agent.noise.rng.bit_generator.state
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                source_agent,
                path,
                config={
                    "algorithm": source_agent.algorithm,
                    "exploration_noise": {"theta": 0.15, "sigma": 0.05},
                },
                training={"next_episode": 0, "global_step": 0},
            )
            forked_agent = _StubAgent()
            forked_agent.noise = OUNoise(1, seed=99, sigma=0.2)
            metadata = load_off_policy_training_state(
                forked_agent,
                path,
                config={
                    "algorithm": source_agent.algorithm,
                    "exploration_noise": {"theta": 0.15, "sigma": 0.2},
                    "preonline_fork_allowed_overrides": [
                        "exploration_noise.sigma"
                    ],
                },
            )

        self.assertEqual(forked_agent.noise.sigma, 0.2)
        np.testing.assert_array_equal(
            forked_agent.noise.state,
            forked_agent.noise.mu,
        )
        self.assertEqual(
            forked_agent.noise.rng.bit_generator.state,
            saved_rng_state,
        )
        self.assertEqual(
            metadata["preonline_fork_overrides"],
            {
                "exploration_noise.sigma": {
                    "checkpoint": 0.05,
                    "current": 0.2,
                }
            },
        )

    def test_episode_zero_state_allows_online_replay_fraction_fork(self):
        source_agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                source_agent,
                path,
                config={"algorithm": source_agent.algorithm},
                training={"next_episode": 0, "global_step": 0},
            )
            metadata = load_off_policy_training_state(
                _StubAgent(),
                path,
                config={
                    "algorithm": source_agent.algorithm,
                    "online_replay_fraction": 0.5,
                    "preonline_fork_allowed_overrides": [
                        "online_replay_fraction"
                    ],
                },
            )

        self.assertEqual(
            metadata["preonline_fork_overrides"],
            {
                "online_replay_fraction": {
                    "checkpoint": None,
                    "current": 0.5,
                }
            },
        )

    def test_episode_zero_state_allows_specimen_quantization_fork(self):
        source_agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                source_agent,
                path,
                config={"algorithm": source_agent.algorithm},
                training={"next_episode": 0, "global_step": 0},
            )
            quantization = {
                "enabled": True,
                "actor_gradient": "straight_through",
            }
            metadata = load_off_policy_training_state(
                _StubAgent(),
                path,
                config={
                    "algorithm": source_agent.algorithm,
                    "specimen_action_quantization": quantization,
                    "preonline_fork_allowed_overrides": [
                        "specimen_action_quantization"
                    ],
                },
            )

        self.assertEqual(
            metadata["preonline_fork_overrides"],
            {
                "specimen_action_quantization": {
                    "checkpoint": None,
                    "current": quantization,
                }
            },
        )

    def test_exploration_sigma_fork_rejects_nonzero_noise_state(self):
        source_agent = _StubAgent()
        source_agent.noise.sample()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                source_agent,
                path,
                config={
                    "algorithm": source_agent.algorithm,
                    "exploration_noise": {"theta": 0.15, "sigma": 0.05},
                },
                training={"next_episode": 0, "global_step": 0},
            )
            forked_agent = _StubAgent()
            forked_agent.noise = OUNoise(1, seed=99, sigma=0.2)
            with self.assertRaisesRegex(ValueError, "reset episode-0"):
                load_off_policy_training_state(
                    forked_agent,
                    path,
                    config={
                        "algorithm": source_agent.algorithm,
                        "exploration_noise": {
                            "theta": 0.15,
                            "sigma": 0.2,
                        },
                        "preonline_fork_allowed_overrides": [
                            "exploration_noise.sigma"
                        ],
                    },
                )

    def test_preonline_fork_rejects_undeclared_scientific_difference(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "actor_update_frequency": 4,
                    "gamma": 0.99,
                },
                training={"next_episode": 0, "global_step": 0},
            )
            with self.assertRaisesRegex(ValueError, "disallowed"):
                load_off_policy_training_state(
                    agent,
                    path,
                    config={
                        "algorithm": agent.algorithm,
                        "actor_update_frequency": 2,
                        "gamma": 0.95,
                        "preonline_fork_allowed_overrides": [
                            "actor_update_frequency"
                        ],
                    },
                )

    def test_preonline_fork_rejects_post_episode_state(self):
        agent = _StubAgent()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config={
                    "algorithm": agent.algorithm,
                    "actor_update_frequency": 4,
                },
                training={"next_episode": 1, "global_step": 52},
            )
            with self.assertRaisesRegex(ValueError, "episode-0"):
                load_off_policy_training_state(
                    agent,
                    path,
                    config={
                        "algorithm": agent.algorithm,
                        "actor_update_frequency": 2,
                        "preonline_fork_allowed_overrides": [
                            "actor_update_frequency"
                        ],
                    },
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
                "preonline_fork_allowed_overrides": [
                    "actor_update_frequency"
                ],
                "preonline_training_state_path": "preonline-b.pt",
                "training_state_checkpoint_interval": 10,
            }
        )
        self.assertEqual(left, right)

    def test_training_loop_can_freeze_episode_zero_preonline_state(self):
        agent = _StubAgent()
        env = SimpleNamespace(config=SimpleNamespace(episode_horizon=1))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "preonline.pt"
            rows = train_off_policy_agent(
                agent,
                env,
                {
                    "algorithm": agent.algorithm,
                    "seed": agent.seed,
                    "num_episodes": 0,
                    "checkpoint_dir": directory,
                    "preonline_training_state_path": str(path),
                },
            )
            checkpoint = torch.load(path, weights_only=False)

        self.assertEqual(rows, [])
        self.assertEqual(checkpoint["training"]["next_episode"], 0)
        self.assertEqual(checkpoint["training"]["global_step"], 0)
        self.assertEqual(checkpoint["training"]["rows"], [])

    def test_episode_zero_resume_writes_local_pretrain_actor_copy(self):
        source_agent = _StubAgent()
        env = SimpleNamespace(config=SimpleNamespace(episode_horizon=1))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "preonline.pt"
            config = {
                "algorithm": source_agent.algorithm,
                "seed": source_agent.seed,
                "num_episodes": 0,
                "save_pretrain_checkpoint": True,
            }
            save_off_policy_training_state(
                source_agent,
                state_path,
                config=config,
                training={
                    "next_episode": 0,
                    "global_step": 0,
                    "rows": [],
                    "pretrain_summary": {},
                    "advantage_distillation_summary": {"samples": 1},
                    "pretrain_checkpoint_path": "source-pretrain.pt",
                },
            )
            resumed_agent = _StubAgent()
            checkpoint_dir = root / "fork" / "checkpoints"
            train_off_policy_agent(
                resumed_agent,
                env,
                {
                    **config,
                    "checkpoint_dir": str(checkpoint_dir),
                    "resume_training_state_path": str(state_path),
                },
            )
            copied = (
                checkpoint_dir
                / f"{source_agent.algorithm}_seed{source_agent.seed}_pretrain.pt"
            )
            copied_exists = copied.exists()

        self.assertTrue(copied_exists)

    def test_round_trip_restores_patient_routing_environment_state(self):
        base = CapacityPlanningConfig(
            num_facilities=2,
            production_lead_time=2,
            episode_horizon=4,
            demand_rates=(0.5, 0.1),
            initial_specimens=(1.0, 0.0),
            initial_reagents=(0.0, 0.0),
            initial_idle_bioreactors=(0.0, 0.0),
            max_specimens=(10.0, 10.0),
            max_reagents=(10.0, 10.0),
            max_idle_bioreactors=(2.0, 2.0),
            max_reagent_replenishment=(0.0, 0.0),
            max_specimen_transfer=1.0,
            action_mode="facility_net",
            specimen_edges=((0, 1),),
        )
        env = PatientConditionCapacityEnv(
            PatientEnvConfig(
                base=base,
                patient=PatientConditionConfig(
                    healthy_decay_rate=0.0,
                    frail_decay_rate=0.0,
                ),
                enable_specimen_routing=True,
                include_specimen_routing_state=True,
                specimen_routing_lead_time_epochs=1,
            ),
            seed=41,
        )
        env.reset(seed=4100)
        action = env.noop_action()
        action[:2] = (-1.0, 1.0)
        env.step(action)
        expected_ids = tuple(env.patient_registry)
        expected_transits = tuple(
            transit.patient_id for transit in env.specimen_transits
        )
        expected_rng = json.dumps(env.rng.bit_generator.state, sort_keys=True)
        agent = _StubAgent()
        config = {"algorithm": agent.algorithm, "seed": agent.seed}

        with TemporaryDirectory() as directory:
            path = Path(directory) / "routing-state.pt"
            save_off_policy_training_state(
                agent,
                path,
                config=config,
                env=env,
                training={"next_episode": 1, "global_step": 1},
            )
            env.reset(seed=999)
            metadata = load_off_policy_training_state(
                agent,
                path,
                config=config,
                env=env,
            )

        self.assertEqual(metadata["next_episode"], 1)
        self.assertEqual(tuple(env.patient_registry), expected_ids)
        self.assertEqual(
            tuple(transit.patient_id for transit in env.specimen_transits),
            expected_transits,
        )
        self.assertEqual(
            json.dumps(env.rng.bit_generator.state, sort_keys=True),
            expected_rng,
        )
        env.assert_identity_conservation()

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

    def test_replay_buffer_samples_requested_online_fraction(self):
        buffer = ReplayBuffer(2, 1, capacity=20, seed=41)
        for index in range(8):
            buffer.add(
                np.array([index, 0], dtype=np.float32),
                np.array([index], dtype=np.float32),
                float(index),
                np.array([index + 1, 0], dtype=np.float32),
                False,
            )
        buffer.begin_online_collection()
        for index in range(8):
            buffer.add(
                np.array([100 + index, 0], dtype=np.float32),
                np.array([100 + index], dtype=np.float32),
                float(index),
                np.array([101 + index, 0], dtype=np.float32),
                False,
            )

        batch = buffer.sample(8, online_fraction=0.5)

        self.assertEqual(batch.online_fraction, 0.5)
        self.assertEqual(int(batch.online_masks.sum()), 4)
        np.testing.assert_array_equal(
            batch.online_masks.reshape(-1),
            batch.states[:, 0] >= 100,
        )
        self.assertEqual(int((batch.states[:, 0] >= 100).sum()), 4)

    def test_replay_buffer_preserves_online_labels_and_rng(self):
        buffer = ReplayBuffer(2, 1, capacity=20, seed=43)
        for index in range(8):
            buffer.add(
                np.array([index, 0], dtype=np.float32),
                np.array([index], dtype=np.float32),
                float(index),
                np.array([index + 1, 0], dtype=np.float32),
                False,
            )
        buffer.begin_online_collection()
        for index in range(8):
            buffer.add(
                np.array([100 + index, 0], dtype=np.float32),
                np.array([100 + index], dtype=np.float32),
                float(index),
                np.array([101 + index, 0], dtype=np.float32),
                False,
            )
        state = buffer.state_dict()
        expected = buffer.sample(8, online_fraction=0.5)

        restored = ReplayBuffer(2, 1, capacity=20, seed=99)
        restored.load_state_dict(state)
        actual = restored.sample(8, online_fraction=0.5)

        self.assertTrue(restored.collecting_online)
        self.assertEqual(actual.online_fraction, 0.5)
        np.testing.assert_array_equal(actual.states, expected.states)
        np.testing.assert_array_equal(
            actual.online_masks,
            expected.online_masks,
        )

    def test_replay_buffer_preserves_bootstrap_discount_and_loads_legacy_state(
        self,
    ):
        buffer = ReplayBuffer(2, 1, capacity=4, seed=47)
        buffer.add(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
            4.0,
            np.array([5.0, 6.0], dtype=np.float32),
            False,
            discount_multiplier=0.125,
            one_step_reward=2.0,
        )
        state = buffer.state_dict()

        restored = ReplayBuffer(2, 1, capacity=4, seed=99)
        restored.load_state_dict(state)
        self.assertAlmostEqual(
            float(restored.discount_multipliers[0, 0]),
            0.125,
        )
        self.assertAlmostEqual(
            float(restored.one_step_rewards[0, 0]),
            2.0,
        )

        legacy_state = dict(state)
        legacy_state.pop("discount_multipliers")
        legacy_state.pop("one_step_rewards")
        legacy = ReplayBuffer(2, 1, capacity=4, seed=101)
        legacy.load_state_dict(legacy_state)
        self.assertAlmostEqual(
            float(legacy.discount_multipliers[0, 0]),
            1.0,
        )
        self.assertAlmostEqual(
            float(legacy.one_step_rewards[0, 0]),
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
