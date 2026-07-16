"""Tests for graph-state conversion used by GCN-DDPG."""

from __future__ import annotations

from dataclasses import asdict, replace
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

    def test_geographic_coordinates_drive_default_graph_edges(self) -> None:
        config = {
            "algorithm": "gcn_ddpg",
            "gcn_edge_types": ["information_edges"],
            "env": {
                "num_facilities": 4,
                "production_lead_time": 3,
                "action_mode": "facility_net",
                "clinic_coordinates": [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [20.0, 20.0],
                    [20.0, 21.0],
                ],
                "geographic_neighbor_k": 1,
            },
        }

        spec = build_graph_spec(config, state_dim=24)

        self.assertEqual(spec.edge_index, ((0, 1), (2, 3)))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GraphStateConversionTests(unittest.TestCase):
    def test_flat_state_conversion_matches_environment_graph_features(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=7)
        state = env.reset(seed=7)
        spec = build_graph_spec(_config_dict(), state_dim=env.observation_size)
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        node_features = flat_state_to_node_features(state_tensor, spec).numpy()[0]

        np.testing.assert_allclose(node_features, env.graph_observation()["node_features"])

    def test_flat_state_conversion_includes_demand_forecast_feature(self) -> None:
        base_config = make_20_clinic_config(episode_horizon=2)
        forecast_config = replace(
            base_config,
            include_demand_forecast_state=True,
            demand_forecast_horizon=2,
            demand_forecast_error=0.0,
        )
        env_config = asdict(forecast_config)
        env = CapacityPlanningEnv(forecast_config, seed=31)
        state = env.reset(seed=31)
        config = _config_dict()
        config["env"] = env_config
        spec = build_graph_spec(config, state_dim=env.observation_size)
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        node_features = flat_state_to_node_features(state_tensor, spec).numpy()[0]

        self.assertEqual(spec.features_per_facility, 8)
        self.assertEqual(spec.node_feature_dim, 8)
        np.testing.assert_allclose(node_features, env.graph_observation()["node_features"])

    def test_residual_graph_features_include_base_action(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=37)
        state = env.reset(seed=37)
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "include_base_action_features": True,
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        node_features = flat_state_to_node_features(state_tensor, agent.graph_spec).numpy()[0]
        base_action = agent._base_action_from_state_np(state).reshape(4, env.config.num_facilities).T

        self.assertEqual(agent.graph_spec.node_feature_dim, 11)
        np.testing.assert_allclose(node_features[: env.config.num_facilities, 6:10], base_action)
        np.testing.assert_allclose(node_features[env.config.num_facilities, 6:10], np.zeros(4))
        self.assertEqual(float(node_features[env.config.num_facilities, 10]), 1.0)

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

    def test_residual_action_group_scales_facility_net_segments(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=17)
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "residual_action": {
                    "enabled": True,
                    "base_policy": "myo",
                    "scale": 0.25,
                    "group_scales": {
                        "specimen_transfer": 0.01,
                        "reagent_transfer": 0.02,
                        "capacity_transfer": 0.03,
                        "replenishment": 0.20,
                    },
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        n = env.config.num_facilities

        self.assertAlmostEqual(float(agent.residual_scale_vector[0]), 0.01)
        self.assertAlmostEqual(float(agent.residual_scale_vector[n]), 0.02)
        self.assertAlmostEqual(float(agent.residual_scale_vector[2 * n]), 0.03)
        self.assertAlmostEqual(float(agent.residual_scale_vector[3 * n]), 0.20)

    def test_residual_action_centering_removes_group_mean(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=18)
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "residual_action": {
                    "enabled": True,
                    "base_policy": "myo",
                    "scale": 0.25,
                    "center_groups": ["replenishment"],
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        n = env.config.num_facilities
        residual = np.zeros(env.action_size, dtype=np.float32)
        residual[3 * n : 4 * n] = np.linspace(-0.2, 0.8, n)

        transformed = agent._transform_network_residual_np(residual)

        self.assertAlmostEqual(float(transformed[3 * n : 4 * n].mean()), 0.0, places=6)

    def test_gcn_agent_fits_external_action_batch(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=19)
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
                    "base_policy": "myo",
                    "scale": 0.1,
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        state = env.reset(seed=19)
        action = agent._base_action_from_state_np(state)
        next_state, _reward, _done, _info = env.step(action)
        states = np.stack([state, next_state])
        actions = np.stack([action, agent._base_action_from_state_np(next_state)])

        summary = agent.fit_action_batch(states, actions, {"epochs": 1, "batch_size": 2})

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["target_mode"], "residual")
        self.assertGreaterEqual(summary["final_loss"], 0.0)

    def test_gcn_agent_fits_weighted_external_action_batch(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=23)
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
                    "base_policy": "myo",
                    "scale": 0.1,
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        state = env.reset(seed=23)
        action = agent._base_action_from_state_np(state)
        next_state, _reward, _done, _info = env.step(action)
        states = np.stack([state, next_state])
        actions = np.stack([action, agent._base_action_from_state_np(next_state)])

        summary = agent.fit_action_batch(
            states,
            actions,
            {"epochs": 1, "batch_size": 2},
            weights=np.asarray([0.25, 1.75], dtype=np.float32),
        )

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["target_mode"], "residual")
        self.assertGreaterEqual(summary["final_loss"], 0.0)
        with self.assertRaises(ValueError):
            agent.fit_action_batch(states, actions, {"epochs": 1}, weights=np.asarray([1.0]))
        with self.assertRaises(ValueError):
            agent.fit_action_batch(
                states,
                actions,
                {"epochs": 1, "target_mode": "unsupported"},
            )


if __name__ == "__main__":
    unittest.main()
