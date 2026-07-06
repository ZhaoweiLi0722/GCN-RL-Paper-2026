"""Tests for graph-state conversion used by GCN-DDPG."""

from __future__ import annotations

from dataclasses import asdict
import unittest

import numpy as np

from src.env.capacity_planning import CapacityPlanningEnv, make_20_clinic_config
from src.models.gcn_ddpg import build_graph_spec

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from src.models.gcn import GCNActor
    from src.models.gcn_ddpg import GCNDDPGAgent, flat_state_to_node_features


def _config_dict(graph_ablation: str = "full_graph") -> dict:
    return {
        "algorithm": "gcn_ddpg",
        "gcn_hidden_sizes": [8],
        "actor_hidden_sizes": [16],
        "critic_hidden_sizes": [16],
        "env": {
            "num_facilities": 20,
            "production_lead_time": 3,
            "action_mode": "facility_net",
            "include_supplier_state": True,
            "include_central_capacity_hub": True,
            "graph_ablation": graph_ablation,
        },
    }


class GraphSpecTests(unittest.TestCase):
    def test_builds_20_clinic_hub_graph_spec(self) -> None:
        spec = build_graph_spec(_config_dict(), state_dim=140)
        self.assertEqual(spec.num_facilities, 20)
        self.assertEqual(spec.num_nodes, 21)
        self.assertEqual(spec.node_feature_dim, 7)
        self.assertTrue(any(20 in edge for edge in spec.edge_index))

    def test_capacity_ablation_removes_hub_edges(self) -> None:
        spec = build_graph_spec(_config_dict("no_capacity_sharing_edges"), state_dim=140)
        self.assertFalse(any(20 in edge for edge in spec.edge_index))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GraphStateConversionTests(unittest.TestCase):
    def test_flat_state_conversion_matches_environment_graph_features(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=7)
        state = env.reset(seed=7)
        spec = build_graph_spec(_config_dict(), state_dim=env.observation_size)
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        node_features = flat_state_to_node_features(state_tensor, spec).numpy()[0]

        np.testing.assert_allclose(node_features, env.graph_observation()["node_features"])

    def test_facility_action_actor_readout_matches_facility_net_layout(self) -> None:
        actor = GCNActor(
            node_feature_dim=7,
            num_facilities=20,
            num_nodes=21,
            action_dim=80,
            edges=((0, 1), (1, 2), (2, 20)),
            gcn_hidden_sizes=(8,),
            head_hidden_sizes=(16,),
            include_global_context=True,
            readout_mode="facility_action",
        )
        node_features = torch.randn(2, 21, 7)

        actions = actor(node_features)

        self.assertEqual(tuple(actions.shape), (2, 80))
        self.assertTrue(torch.all(actions <= 1.0))
        self.assertTrue(torch.all(actions >= -1.0))

    def test_gcn_agent_imitation_pretrain_collects_demonstrations(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=11)
        config = _config_dict()
        config.update(
            {
                "batch_size": 2,
                "reward_scale": 1e-9,
                "normalize_observations": True,
                "gcn_hidden_sizes": [8],
                "actor_hidden_sizes": [16],
                "critic_hidden_sizes": [16],
                "actor_readout_mode": "facility_action",
                "imitation_pretrain": {
                    "enabled": True,
                    "regularization_weight": 0.5,
                    "regularization_batch_size": 2,
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)

        summary = agent.pretrain_with_heuristic(
            env,
            {
                "policy": "mdl1",
                "episodes": 1,
                "epochs": 1,
                "batch_size": 2,
                "seed": 123,
                "populate_replay_buffer": True,
            },
        )

        self.assertEqual(summary["policy"], "mdl1")
        self.assertEqual(summary["samples"], 2)
        self.assertGreaterEqual(summary["final_loss"], 0.0)
        self.assertEqual(len(agent.replay_buffer), 2)
        update_metrics = agent.update()
        self.assertIn("imitation_loss", update_metrics)

    def test_residual_action_zero_network_output_returns_heuristic_base(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=13)
        state = env.reset(seed=13)
        config = _config_dict()
        config.update(
            {
                "batch_size": 2,
                "env": asdict(env.config),
                "gcn_hidden_sizes": [8],
                "actor_hidden_sizes": [16],
                "critic_hidden_sizes": [16],
                "actor_readout_mode": "facility_action",
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "scale": 0.35,
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        zero_residual = np.zeros(env.action_size, dtype=np.float32)

        base_action = agent._base_action_from_state_np(state)
        composed_action = agent._compose_action_np(state, zero_residual)

        np.testing.assert_allclose(composed_action, base_action)


if __name__ == "__main__":
    unittest.main()
