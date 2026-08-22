"""Tests for checkpoint-compatible conservative AFR-GCN-TD3."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.models.gcn_ddpg import GCNDDPGAgent
from src.rl.config import load_config
from src.rl.experiment import build_env


DEV_CONFIG = "experiments/configs/2_clinic_patient_condition.json"


def _config(seed: int = 0) -> dict:
    return {
        "algorithm": "gcn_residual_mdl2_network_td3_bc",
        "seed": seed,
        "env": load_config(DEV_CONFIG),
        "gcn_hidden_sizes": [8, 8],
        "hidden_sizes": [16, 16],
        "batch_size": 2,
        "critic_warmup_updates": 2,
        "policy_delay": 1,
        "policy_noise": 0.01,
        "noise_clip": 0.02,
        "actor_advantage_baseline": "reference",
        "reference_policy_regularization_weight": 25.0,
        "residual_action": {
            "enabled": True,
            "base_policy": "mdl2",
            "zero_init_actor": True,
            "scale": 0.05,
            "group_scales": {
                "specimen_transfer": 0.0,
                "reagent_transfer": 0.01,
                "capacity_transfer": 0.01,
                "replenishment": 0.01,
            },
            "center_groups": [
                "reagent_transfer",
                "capacity_transfer",
            ],
            "l2_weight": 0.05,
        },
        "anchor_advantage_actor_loss": {
            "enabled": True,
            "temperature": 0.05,
            "negative_penalty_weight": 1.0,
        },
    }


class ConservativeGCNResidualTD3Tests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from src.models.conservative_gcn_td3 import (
                ConservativeGCNResidualTD3Agent,
            )
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")
        self.agent_cls = ConservativeGCNResidualTD3Agent
        self.torch = torch
        self.env = build_env({"env": load_config(DEV_CONFIG)}, seed=0)

    def test_twin_critics_are_independently_initialized(self) -> None:
        agent = self.agent_cls(
            self.env.observation_size,
            self.env.action_size,
            _config(),
        )

        differences = [
            float((left - right).abs().max().item())
            for left, right in zip(
                agent.critic1.parameters(),
                agent.critic2.parameters(),
            )
        ]

        self.assertTrue(any(value > 0.0 for value in differences))

    def test_critic_warmup_precedes_actor_update(self) -> None:
        agent = self.agent_cls(
            self.env.observation_size,
            self.env.action_size,
            _config(seed=2),
        )
        state = self.env.reset(seed=2)
        update_metrics = []
        for step in range(4):
            action = agent.select_action(state, explore=False, env=self.env)
            next_state, reward, done, _info = self.env.step(action)
            agent.observe(state, action, reward, next_state, done)
            metrics = agent.update()
            if metrics:
                update_metrics.append(metrics)
            state = (
                next_state
                if not done
                else self.env.reset(seed=3 + step)
            )

        self.assertEqual(len(update_metrics), 3)
        self.assertEqual(update_metrics[0]["actor_updated"], 0.0)
        self.assertEqual(update_metrics[1]["actor_updated"], 0.0)
        self.assertEqual(update_metrics[0]["actor_warmup"], 1.0)
        self.assertEqual(update_metrics[1]["actor_warmup"], 1.0)
        self.assertEqual(update_metrics[2]["actor_updated"], 1.0)
        self.assertEqual(update_metrics[2]["actor_warmup"], 0.0)
        self.assertIn("actor_reference_drift_rms", update_metrics[2])
        self.assertIn(
            "actor_reference_advantage_mean",
            update_metrics[2],
        )
        self.assertIn("reference_policy_loss", update_metrics[2])
        for metrics in update_metrics:
            for value in metrics.values():
                self.assertTrue(math.isfinite(float(value)))

    def test_load_actor_syncs_target_and_preserves_checkpoint_surface(
        self,
    ) -> None:
        config = _config(seed=4)
        source = GCNDDPGAgent(
            self.env.observation_size,
            self.env.action_size,
            config,
        )
        target = self.agent_cls(
            self.env.observation_size,
            self.env.action_size,
            config,
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "distilled.pt"
            source.save(checkpoint)
            target.load_actor(checkpoint)

        for actor_parameter, target_parameter in zip(
            target.actor.parameters(),
            target.actor_target.parameters(),
        ):
            self.torch.testing.assert_close(
                actor_parameter,
                target_parameter,
            )
        self.assertEqual(
            set(target._actor_reference),
            {name for name, _ in target.actor.named_parameters()},
        )

    def test_saved_checkpoint_contains_both_critics(self) -> None:
        agent = self.agent_cls(
            self.env.observation_size,
            self.env.action_size,
            _config(seed=5),
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "td3.pt"
            agent.save(checkpoint_path)
            checkpoint = self.torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

        self.assertIn("critic1", checkpoint)
        self.assertIn("critic2", checkpoint)
        self.assertEqual(
            checkpoint["algorithm"],
            "gcn_residual_mdl2_network_td3_bc",
        )

    def test_zero_epoch_teacher_retention_does_not_change_actor(
        self,
    ) -> None:
        agent = self.agent_cls(
            self.env.observation_size,
            self.env.action_size,
            _config(seed=6),
        )
        states = []
        actions = []
        state = self.env.reset(seed=6)
        for _step in range(3):
            action = agent.select_action(
                state,
                explore=False,
                env=self.env,
            )
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
            state, _reward, done, _info = self.env.step(action)
            if done:
                break
        before = {
            name: parameter.detach().clone()
            for name, parameter in agent.actor.named_parameters()
        }

        summary = agent.fit_action_batch(
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            {
                "epochs": 0,
                "retain_for_regularization": True,
            },
        )

        self.assertEqual(summary["samples"], len(states))
        self.assertEqual(summary["final_loss"], 0.0)
        self.assertEqual(len(agent.imitation_states), len(states))
        for name, parameter in agent.actor.named_parameters():
            self.torch.testing.assert_close(parameter, before[name])

    def test_routing_primary_online_contract_is_active(self) -> None:
        config = _config(seed=7)
        config["critic_warmup_updates"] = 0
        config["online_replay_fraction"] = 1.0
        config["residual_action"].update(
            {
                "online_reward_mode": "n_step_anchor_relative",
                "online_reward_n_step_horizon": 4,
            }
        )
        config["pretrain_reference_actor_loss"] = {
            "enabled": True,
            "weight": 5.0,
        }
        config["online_advantage_self_imitation"] = {
            "enabled": True,
            "weight": 1.0,
            "release_pretrain_reference": True,
            "require_positive_one_step_return": True,
            "minimum_return": 0.0,
        }
        agent = self.agent_cls(
            self.env.observation_size,
            self.env.action_size,
            config,
        )
        agent.capture_pretrain_reference_policy()
        agent.replay_buffer.begin_online_collection()
        state = self.env.reset(seed=7)
        for step in range(3):
            action = agent.select_action(state, explore=False, env=self.env)
            next_state, _reward, done, _info = self.env.step(action)
            agent.replay_buffer.add(
                state,
                action,
                0.1,
                next_state,
                done,
                discount_multiplier=agent.gamma**3,
                one_step_reward=0.1,
            )
            state = next_state
        metrics = agent.update()

        self.assertEqual(metrics["replay_online_fraction"], 1.0)
        self.assertIn(
            "online_advantage_self_imitation_active_fraction",
            metrics,
        )
        self.assertIn("pretrain_reference_action_mse", metrics)
        self.assertIn("pretrain_reference_parameter_drift_rms", metrics)
        self.assertGreater(
            metrics["online_advantage_self_imitation_active_fraction"],
            0.0,
        )
        for value in metrics.values():
            self.assertTrue(math.isfinite(float(value)))


if __name__ == "__main__":
    unittest.main()
