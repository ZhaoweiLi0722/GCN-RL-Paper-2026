"""Tests for direct online behavior-versus-MDL-2 critic supervision."""

from __future__ import annotations

from dataclasses import asdict
import unittest

import numpy as np

from src.env.capacity_planning import CapacityPlanningEnv, make_20_clinic_config
from src.rl.action_projection import project_action
from src.rl.paired_advantage import (
    online_paired_advantage_settings,
    quantize_specimen_action_np,
    rollout_paired_anchor_advantage,
    specimen_actions_are_distinct,
)
from src.rl.replay_buffer import ReplayBuffer

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from src.baselines.flat_ddpg import FlatDDPGAgent
    from src.models.gcn_ddpg import GCNDDPGAgent


class _AccumulatingEnv:
    def __init__(self, horizon: int = 3) -> None:
        self.horizon = int(horizon)
        self.step_index = 0
        self.value = 0.0
        self.action_size = 1

    def step(self, action):
        self.value += float(np.asarray(action).reshape(-1)[0])
        self.step_index += 1
        return (
            np.asarray([self.value], dtype=np.float32),
            float(self.value),
            self.step_index >= self.horizon,
            {},
        )


def _agent_config(env: CapacityPlanningEnv) -> dict:
    return {
        "algorithm": "gcn_ddpg",
        "seed": 19,
        "device": "cpu",
        "gcn_hidden_sizes": [8],
        "actor_hidden_sizes": [16],
        "critic_hidden_sizes": [16],
        "hidden_sizes": [20, 12],
        "batch_size": 2,
        "replay_buffer_size": 32,
        "gamma": 0.5,
        "reward_scale": 1e-9,
        "env": asdict(env.config),
        "specimen_action_quantization": {
            "enabled": True,
            "actor_gradient": "straight_through",
        },
        "residual_action": {
            "enabled": True,
            "base_policy": "mdl2",
            "scale": 0.1,
            "online_reward_mode": "n_step_anchor_relative",
            "online_reward_n_step_horizon": 2,
            "group_scales": {
                "specimen_transfer": 0.1,
                "reagent_transfer": 0.0,
                "capacity_transfer": 0.0,
                "replenishment": 0.0,
            },
        },
        "online_paired_advantage_critic": {
            "enabled": True,
            "horizon": 2,
            "loss_weight": 3.0,
            "followup_policy": "mdl2",
            "reward_consistency_atol": 1e-6,
        },
    }


class PairedAdvantageUtilityTests(unittest.TestCase):
    def test_patient_lot_quantization_uses_half_away_from_zero(self) -> None:
        action = np.asarray([0.0042, -0.0042, 0.0041, -0.0041])
        quantized = quantize_specimen_action_np(
            action,
            num_facilities=4,
            max_specimen_transfer=120.0,
        )
        np.testing.assert_allclose(
            quantized,
            np.asarray([1.0 / 120.0, -1.0 / 120.0, 0.0, 0.0]),
        )
        self.assertTrue(
            specimen_actions_are_distinct(
                action,
                np.zeros(4, dtype=np.float32),
                num_facilities=4,
                max_specimen_transfer=120.0,
            )
        )

    def test_finite_horizon_rollout_uses_exact_crn_copies(self) -> None:
        result = rollout_paired_anchor_advantage(
            anchor_env=_AccumulatingEnv(horizon=3),
            behavior_action=np.asarray([2.0], dtype=np.float32),
            anchor_action=np.asarray([0.0], dtype=np.float32),
            observed_behavior_reward=2.0,
            gamma=0.5,
            horizon=3,
            action_dim=1,
            base_action=lambda _state: np.asarray([0.0], dtype=np.float32),
            reward_consistency_atol=0.0,
        )
        self.assertAlmostEqual(result.advantage, 3.5)
        self.assertAlmostEqual(result.anchor_first_reward, 0.0)
        self.assertEqual(result.rollout_steps, 3)

    def test_enabled_settings_require_matching_n_step_horizon(self) -> None:
        config = {
            "online_paired_advantage_critic": {
                "enabled": True,
                "horizon": 3,
                "loss_weight": 1.0,
            }
        }
        with self.assertRaisesRegex(ValueError, "must match"):
            online_paired_advantage_settings(
                config,
                residual_action_enabled=True,
                online_reward_mode="n_step_anchor_relative",
                online_reward_n_step_horizon=4,
                specimen_action_quantization_enabled=True,
                action_mode="facility_net",
                action_dim=80,
                num_facilities=20,
            )

    def test_replay_round_trip_and_legacy_defaults(self) -> None:
        buffer = ReplayBuffer(2, 1, capacity=4, seed=5)
        buffer.add(
            np.asarray([1.0, 2.0]),
            np.asarray([0.25]),
            3.0,
            np.asarray([2.0, 3.0]),
            False,
            paired_advantage=0.75,
        )
        state = buffer.state_dict()
        restored = ReplayBuffer(2, 1, capacity=4, seed=6)
        restored.load_state_dict(state)
        self.assertTrue(restored.paired_advantage_mask[0])
        self.assertAlmostEqual(float(restored.paired_advantages[0, 0]), 0.75)

        legacy_state = dict(state)
        legacy_state.pop("paired_advantages")
        legacy_state.pop("paired_advantage_mask")
        legacy = ReplayBuffer(2, 1, capacity=4, seed=7)
        legacy.load_state_dict(legacy_state)
        self.assertFalse(legacy.paired_advantage_mask[0])
        self.assertAlmostEqual(float(legacy.paired_advantages[0, 0]), 0.0)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PairedAdvantageAgentTests(unittest.TestCase):
    def test_offline_observe_is_unpaired_but_captured_online_context_is_strict(self) -> None:
        for offset, agent_class in enumerate((GCNDDPGAgent, FlatDDPGAgent)):
            env = CapacityPlanningEnv(
                make_20_clinic_config(episode_horizon=2),
                seed=680 + offset,
            )
            agent = agent_class(
                env.observation_size,
                env.action_size,
                _agent_config(env),
            )
            state = env.reset(seed=690 + offset)
            action = np.zeros(env.action_size, dtype=np.float32)
            agent.observe(state, action, 0.0, state, True)
            self.assertEqual(len(agent.replay_buffer), 1)
            self.assertFalse(bool(agent.replay_buffer.paired_advantage_mask[0]))

            state = env.reset(seed=695 + offset)
            action = project_action(
                agent._base_action_from_state_np(state),
                env_state=env,
                action_space_info=env.action_size,
            ).action
            agent.capture_training_reward_context(state, action, env)
            with self.assertRaisesRegex(RuntimeError, "not transformed"):
                agent.observe(state, action, 0.0, state, False)

    def test_gcn_and_flat_agents_store_and_fit_paired_targets(self) -> None:
        for offset, agent_class in enumerate((GCNDDPGAgent, FlatDDPGAgent)):
            env = CapacityPlanningEnv(
                make_20_clinic_config(episode_horizon=2),
                seed=700 + offset,
            )
            config = _agent_config(env)
            agent = agent_class(
                env.observation_size,
                env.action_size,
                config,
            )
            agent.prepare_online_finetuning(start_episode=0)
            state = env.reset(seed=710 + offset)
            for step in range(2):
                anchor = project_action(
                    agent._base_action_from_state_np(state),
                    env_state=env,
                    action_space_info=env.action_size,
                ).action
                if step == 0:
                    behavior = anchor.copy()
                    behavior[0] = np.clip(behavior[0] + 0.10, -1.0, 1.0)
                    behavior[1] = np.clip(behavior[1] - 0.10, -1.0, 1.0)
                else:
                    behavior = anchor
                context = agent.capture_training_reward_context(
                    state,
                    behavior,
                    env,
                )
                next_state, reward, done, info = env.step(behavior)
                training_reward = agent.transform_training_reward(
                    state,
                    behavior,
                    reward,
                    next_state,
                    done,
                    info,
                    context,
                )
                agent.observe(
                    state,
                    behavior,
                    training_reward,
                    next_state,
                    done,
                )
                state = next_state

            self.assertEqual(len(agent.replay_buffer), 2)
            self.assertEqual(
                int(agent.replay_buffer.paired_advantage_mask[:2].sum()),
                1,
            )
            self.assertTrue(
                np.isfinite(agent.replay_buffer.paired_advantages[:2]).all()
            )
            metrics = agent.update()
            self.assertEqual(
                metrics["critic_online_paired_advantage_samples"],
                1.0,
            )
            self.assertTrue(
                np.isfinite(
                    metrics["critic_online_paired_advantage_weighted_loss"]
                )
            )
            self.assertIsNone(agent._pending_online_paired_advantage)


if __name__ == "__main__":
    unittest.main()
