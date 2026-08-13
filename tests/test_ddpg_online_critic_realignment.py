"""Tests for the Stage C2 online-boundary DDPG critic reset."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from src.baselines.flat_ddpg import FlatDDPGAgent
    from src.models.gcn_ddpg import GCNDDPGAgent
    from src.rl.critic_realignment import zero_critic_action_input
    from src.rl.networks import MLPCritic


def _agent_config() -> dict:
    return {
        "algorithm": "gcn_ddpg",
        "gcn_hidden_sizes": [8],
        "actor_hidden_sizes": [16],
        "critic_hidden_sizes": [16],
        "batch_size": 2,
        "critic_warmup_updates": 0,
        "actor_update_frequency": 1,
        "online_critic_realignment": {
            "enabled": True,
            "mode": "zero_action_columns",
            "actor_warmup_updates": 2,
        },
        "env": {
            "num_facilities": 20,
            "production_lead_time": 3,
            "action_mode": "facility_net",
            "include_supplier_state": True,
            "include_central_capacity_hub": True,
        },
    }


def _first_state_action_linear(agent):
    container = getattr(agent.critic, "head", None)
    if container is None:
        container = agent.critic.net
    return next(
        module
        for module in container.modules()
        if isinstance(module, torch.nn.Linear)
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class OnlineCriticRealignmentTests(unittest.TestCase):
    def test_helper_preserves_state_columns_and_clears_optimizer(self):
        torch.manual_seed(7)
        critic = MLPCritic(4, 2, (8,))
        target = MLPCritic(4, 2, (8,))
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        loss = critic(torch.ones((3, 4)), torch.ones((3, 2))).sum()
        loss.backward()
        optimizer.step()
        first = next(
            module
            for module in critic.net.modules()
            if isinstance(module, torch.nn.Linear)
        )
        state_columns = first.weight[:, :-2].detach().clone()
        bias = first.bias.detach().clone()

        summary = zero_critic_action_input(
            critic,
            target,
            optimizer,
            action_dim=2,
        )

        torch.testing.assert_close(first.weight[:, :-2], state_columns)
        torch.testing.assert_close(first.bias, bias)
        torch.testing.assert_close(
            first.weight[:, -2:],
            torch.zeros_like(first.weight[:, -2:]),
        )
        self.assertEqual(len(optimizer.state), 0)
        self.assertGreater(
            summary["online_critic_action_columns_l2_before"],
            0.0,
        )
        self.assertEqual(
            summary["online_critic_action_columns_l2_after"],
            0.0,
        )
        for name, value in critic.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], value)
        state = torch.randn((4, 4))
        torch.testing.assert_close(
            critic(state, torch.zeros((4, 2))),
            critic(state, torch.ones((4, 2))),
        )

    def test_gcn_and_flat_use_online_only_actor_warmup(self):
        state_dim = 140
        action_dim = 80
        for agent_class in (GCNDDPGAgent, FlatDDPGAgent):
            with self.subTest(agent=agent_class.__name__):
                agent = agent_class(
                    state_dim,
                    action_dim,
                    _agent_config(),
                )
                actor_before = {
                    name: value.detach().clone()
                    for name, value in agent.actor.state_dict().items()
                }

                summary = agent.prepare_online_finetuning(start_episode=0)

                first = _first_state_action_linear(agent)
                self.assertEqual(
                    float(first.weight[:, -action_dim:].norm().item()),
                    0.0,
                )
                self.assertTrue(agent.online_critic_realignment_applied)
                self.assertEqual(agent.online_updates_since_prepare, 0)
                self.assertEqual(summary["online_actor_warmup_updates"], 2.0)
                for name, value in actor_before.items():
                    torch.testing.assert_close(
                        agent.actor.state_dict()[name],
                        value,
                    )

                for index in range(2):
                    state = np.full(state_dim, index / 10, dtype=np.float32)
                    agent.replay_buffer.add(
                        state,
                        np.zeros(action_dim, dtype=np.float32),
                        0.0,
                        state,
                        False,
                    )

                first_update = agent.update()
                second_update = agent.update()
                for name, value in actor_before.items():
                    torch.testing.assert_close(
                        agent.actor.state_dict()[name],
                        value,
                    )
                third_update = agent.update()

                self.assertEqual(first_update["actor_updated"], 0.0)
                self.assertEqual(second_update["actor_updated"], 0.0)
                self.assertEqual(third_update["actor_updated"], 1.0)
                self.assertEqual(
                    third_update["online_updates_since_prepare"],
                    3.0,
                )
                self.assertEqual(
                    third_update["online_actor_warmup_active"],
                    0.0,
                )


if __name__ == "__main__":
    unittest.main()
