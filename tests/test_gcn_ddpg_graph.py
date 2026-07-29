"""Tests for graph-state conversion used by GCN-DDPG."""

from __future__ import annotations

from dataclasses import asdict, replace
import unittest

import numpy as np

from src.env.capacity_planning import CapacityPlanningEnv, make_20_clinic_config
from src.models.gcn_ddpg import build_graph_spec
from src.rl.config import load_config
from src.rl.experiment import build_env

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from src.baselines.flat_ddpg import FlatDDPGAgent
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
                "geographic_transfer_time_cost_scale": 0.05,
            },
        }

        spec = build_graph_spec(config, state_dim=24)

        self.assertEqual(spec.edge_index, ((0, 1), (2, 3)))
        self.assertEqual(len(spec.edge_weights), len(spec.edge_index))
        self.assertTrue(all(0.0 < weight < 1.0 for weight in spec.edge_weights))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GraphStateConversionTests(unittest.TestCase):
    def test_sparse_supervised_mask_emphasizes_changed_endpoints(self) -> None:
        targets = torch.zeros((2, 80), dtype=torch.float32)
        targets[:, 21] = -0.05
        targets[:, 22] = 0.05
        base_mask = torch.ones((1, 80), dtype=torch.float32)

        for agent_cls in (GCNDDPGAgent, FlatDDPGAgent):
            agent = agent_cls(140, 80, _config_dict())
            agent.supervised_sparse_target_enabled = True
            agent.supervised_sparse_target_threshold = 1e-6
            agent.supervised_sparse_target_changed_weight = 1.0
            agent.supervised_sparse_target_zero_weight = 0.05

            mask = agent._supervised_residual_dimension_mask(
                base_mask,
                targets,
            )

            self.assertEqual(tuple(mask.shape), (2, 80))
            self.assertAlmostEqual(float(mask[0, 21]), 1.0)
            self.assertAlmostEqual(float(mask[0, 22]), 1.0)
            self.assertAlmostEqual(float(mask[0, 20]), 0.05)

    def test_edge_supervision_maps_teacher_flow_and_backpropagates(self) -> None:
        env = CapacityPlanningEnv(
            make_20_clinic_config(episode_horizon=2),
            seed=71,
        )
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "actor_readout_mode": "network_residual",
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "zero_init_actor": True,
                    "center_groups": [
                        "reagent_transfer",
                        "capacity_transfer",
                    ],
                    "group_scales": {
                        "specimen_transfer": 0.0,
                        "reagent_transfer": 1.0,
                        "capacity_transfer": 1.0,
                        "replenishment": 0.0,
                    },
                    "edge_supervision": {
                        "enabled": True,
                        "weight": 1.0,
                        "changed_weight": 1.0,
                        "zero_weight": 0.02,
                    },
                },
            }
        )
        agent = GCNDDPGAgent(
            env.observation_size,
            env.action_size,
            config,
        )
        state = env.reset(seed=71)
        states = np.stack((state, state))
        pair = (
            agent.actor.resource_edge_pairs[:, 0]
            .detach()
            .cpu()
            .numpy()
        )
        source, target = (int(pair[0]), int(pair[1]))
        demonstrations = {
            "edge_groups": np.asarray(
                ["geographic_reagent_edge", "anchor"],
                dtype="U32",
            ),
            "edge_sources": np.asarray(
                [[source], [-1]],
                dtype=np.int64,
            ),
            "edge_targets": np.asarray(
                [[target], [-1]],
                dtype=np.int64,
            ),
            "edge_flows": np.asarray(
                [[0.1], [0.0]],
                dtype=np.float32,
            ),
        }
        targets = agent._edge_supervision_targets(
            demonstrations,
            sample_count=2,
        )
        self.assertAlmostEqual(
            float(targets["reagent_transfer"][0, 0]),
            0.1,
        )

        state_tensor = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=agent.device,
        )
        node_features = flat_state_to_node_features(
            state_tensor,
            agent.graph_spec,
        )
        agent.actor.zero_grad()
        agent.actor(node_features)
        loss = agent._edge_supervision_loss(
            agent.actor.last_edge_flows,
            agent.actor.last_edge_selector_logits,
            targets,
            indices=None,
            sample_weights=None,
        )
        loss.backward()
        gradient = sum(
            float(parameter.grad.abs().sum().item())
            for parameter in agent.actor.reagent_edge_head.parameters()
            if parameter.grad is not None
        )

        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(gradient, 0.0)
        wrong = torch.zeros_like(targets["reagent_transfer"])
        ordered = torch.zeros_like(targets["reagent_transfer"])
        wrong[0, 1] = 0.2
        ordered[0, 0] = 0.1
        wrong_ranking = agent._edge_support_ranking_loss(
            wrong,
            targets["reagent_transfer"],
            None,
        )
        ordered_ranking = agent._edge_support_ranking_loss(
            ordered,
            targets["reagent_transfer"],
            None,
        )
        self.assertGreater(
            float(wrong_ranking.item()),
            float(ordered_ranking.item()),
        )

    def test_support_ranking_penalizes_wrong_endpoint_order(self) -> None:
        targets = torch.zeros((1, 80), dtype=torch.float32)
        targets[0, 21] = -0.1
        targets[0, 22] = 0.1
        dim_mask = torch.ones((1, 80), dtype=torch.float32)
        wrong = torch.zeros((1, 80), dtype=torch.float32)
        wrong[0, 21] = -0.01
        wrong[0, 22] = 0.01
        wrong[0, 23] = 0.2
        ordered = torch.zeros((1, 80), dtype=torch.float32)
        ordered[0, 21] = -0.2
        ordered[0, 22] = 0.2
        ordered[0, 23] = 0.01

        for agent_cls in (GCNDDPGAgent, FlatDDPGAgent):
            agent = agent_cls(140, 80, _config_dict())
            agent.supervised_sparse_target_threshold = 1e-6
            agent.supervised_support_ranking_margin = 0.02

            wrong_loss = agent._support_ranking_loss(
                wrong,
                targets,
                None,
                dim_mask,
            )
            ordered_loss = agent._support_ranking_loss(
                ordered,
                targets,
                None,
                dim_mask,
            )

            self.assertGreater(float(wrong_loss), 0.1)
            self.assertAlmostEqual(float(ordered_loss), 0.0)

    def test_support_ranking_has_directional_gradient_at_zero(self) -> None:
        targets = torch.zeros((1, 80), dtype=torch.float32)
        targets[0, 21] = -0.1
        targets[0, 22] = 0.1
        dim_mask = torch.ones((1, 80), dtype=torch.float32)

        for agent_cls in (GCNDDPGAgent, FlatDDPGAgent):
            agent = agent_cls(140, 80, _config_dict())
            agent.supervised_sparse_target_threshold = 1e-6
            agent.supervised_support_ranking_margin = 0.02
            predictions = torch.zeros(
                (1, 80),
                dtype=torch.float32,
                requires_grad=True,
            )

            loss = agent._support_ranking_loss(
                predictions,
                targets,
                None,
                dim_mask,
            )
            loss.backward()

            self.assertGreater(float(predictions.grad[0, 21]), 0.0)
            self.assertLess(float(predictions.grad[0, 22]), 0.0)

    def test_sparse_mask_is_used_by_supervised_evaluation_path(self) -> None:
        config = _config_dict()
        config["residual_action"] = {
            "enabled": True,
            "base_policy": "mdl2",
            "zero_init_actor": True,
            "center_groups": [
                "reagent_transfer",
                "capacity_transfer",
            ],
            "group_scales": {
                "specimen_transfer": 0.0,
                "reagent_transfer": 1.0,
                "capacity_transfer": 1.0,
                "replenishment": 0.0,
            },
            "supervised_sparse_target": {
                "enabled": True,
                "changed_weight": 1.0,
                "zero_weight": 0.05,
            },
        }
        states = np.zeros((2, 140), dtype=np.float32)

        for agent_cls in (GCNDDPGAgent, FlatDDPGAgent):
            agent = agent_cls(140, 80, config)
            state_tensor = torch.as_tensor(
                states,
                dtype=torch.float32,
                device=agent.device,
            )
            with torch.no_grad():
                target_actions = (
                    agent._base_actions_from_states_tensor(state_tensor)
                    .cpu()
                    .numpy()
                )
            target_actions[:, 20] -= 0.05
            target_actions[:, 21] += 0.05

            sparse = agent.evaluate_action_batch(
                states,
                target_actions,
                target_mode="residual",
            )["actor_loss"]
            agent.supervised_sparse_target_enabled = False
            dense = agent.evaluate_action_batch(
                states,
                target_actions,
                target_mode="residual",
            )["actor_loss"]

            self.assertGreater(sparse, dense * 5.0)

    def test_residual_patient_pressure_matches_waiting_risk_definition(self) -> None:
        env_config = load_config("experiments/configs/2_clinic_patient_condition.json")
        env = build_env({"env": env_config}, seed=41)
        config = _config_dict()
        config["env"] = env_config
        state = np.zeros(env.observation_size, dtype=np.float32)
        n = env.config.num_facilities
        summary_edges = tuple(env_config["survival_bucket_edges"])
        summary_width = 6 + len(summary_edges) + 1
        base_width = n * env.features_per_facility
        summary = state[base_width:].reshape(n, summary_width)
        summary[0, 2] = 2.0
        summary[0, 5] = 99.0
        summary[0, 6:] = np.asarray([1.0, 3.0, 5.0, 7.0])
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

        for agent_cls in (GCNDDPGAgent, FlatDDPGAgent):
            agent = agent_cls(env.observation_size, env.action_size, config)
            risk = agent._patient_risk_signal_tensor(
                state_tensor,
                env.features_per_facility,
            )
            self.assertEqual(float(risk[0, 0]), 3.0)
            self.assertEqual(float(risk[0, 1]), 0.0)

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

    def test_flat_state_conversion_preserves_causal_demand_sequence(self) -> None:
        sequence_config = replace(
            make_20_clinic_config(episode_horizon=2),
            include_demand_forecast_state=True,
            demand_forecast_error=0.0,
            include_demand_history_state=True,
            demand_history_window=3,
            include_demand_sequence_state=True,
            demand_sequence_length=3,
        )
        env = CapacityPlanningEnv(sequence_config, seed=32)
        state = env.reset(seed=32)
        config = _config_dict()
        config["env"] = asdict(sequence_config)
        spec = build_graph_spec(config, state_dim=env.observation_size)

        node_features = flat_state_to_node_features(
            torch.as_tensor(state, dtype=torch.float32),
            spec,
        ).numpy()[0]

        self.assertTrue(spec.include_demand_sequence_state)
        self.assertEqual(spec.demand_sequence_length, 3)
        self.assertEqual(spec.features_per_facility, 20)
        self.assertEqual(spec.node_feature_dim, 20)
        np.testing.assert_allclose(
            node_features,
            env.graph_observation()["node_features"],
        )

    def test_temporal_graph_and_flat_agents_share_causal_sequence_contract(self) -> None:
        sequence_config = replace(
            make_20_clinic_config(episode_horizon=3),
            include_demand_forecast_state=True,
            demand_forecast_error=0.0,
            include_demand_history_state=True,
            demand_history_window=3,
            include_demand_sequence_state=True,
            demand_sequence_length=3,
        )
        env = CapacityPlanningEnv(sequence_config, seed=33)
        config = _config_dict()
        config.update(
            {
                "env": asdict(sequence_config),
                "batch_size": 2,
                "hidden_sizes": [16],
                "temporal_demand_encoder": {
                    "enabled": True,
                    "hidden_size": 5,
                },
                "actor_readout_mode": "pressure_intensity",
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "center_groups": [],
                    "pressure_projection": {
                        "enabled": True,
                        "groups": [
                            "reagent_transfer",
                            "capacity_transfer",
                        ],
                        "coefficient_mode": "mean",
                        "positive_coefficient": True,
                    },
                },
            }
        )
        state = env.reset(seed=33)

        for agent_cls in (GCNDDPGAgent, FlatDDPGAgent):
            agent = agent_cls(
                env.observation_size,
                env.action_size,
                config,
            )
            self.assertEqual(
                agent.actor.readout_mode,
                "pressure_intensity",
            )
            action = agent.select_action(
                state,
                explore=False,
                env=env,
            )
            self.assertEqual(action.shape, (env.action_size,))
            self.assertTrue(np.all(np.isfinite(action)))

            next_state, reward, done, _info = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            agent.observe(next_state, action, reward, next_state, True)
            metrics = agent.update()
            self.assertIn("critic_loss", metrics)

    def test_temporal_graph_and_flat_agents_support_network_residual_readout(self) -> None:
        sequence_config = replace(
            make_20_clinic_config(episode_horizon=3),
            include_demand_forecast_state=True,
            demand_forecast_error=0.0,
            include_demand_history_state=True,
            demand_history_window=3,
            include_demand_sequence_state=True,
            demand_sequence_length=3,
        )
        env = CapacityPlanningEnv(sequence_config, seed=34)
        config = _config_dict()
        config.update(
            {
                "env": asdict(sequence_config),
                "batch_size": 2,
                "hidden_sizes": [16],
                "temporal_demand_encoder": {
                    "enabled": True,
                    "hidden_size": 5,
                },
                "actor_readout_mode": "network_residual",
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "zero_init_actor": True,
                    "center_groups": [
                        "reagent_transfer",
                        "capacity_transfer",
                    ],
                    "group_scales": {
                        "specimen_transfer": 0.0,
                        "reagent_transfer": 1.0,
                        "capacity_transfer": 1.0,
                        "replenishment": 0.0,
                    },
                },
            }
        )
        state = env.reset(seed=34)

        for agent_cls in (GCNDDPGAgent, FlatDDPGAgent):
            agent = agent_cls(
                env.observation_size,
                env.action_size,
                config,
            )
            self.assertEqual(agent.actor.readout_mode, "network_residual")
            action = agent.select_action(
                state,
                explore=False,
                env=env,
            )
            self.assertEqual(action.shape, (env.action_size,))
            self.assertTrue(np.all(np.isfinite(action)))

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

    def test_residual_ddpg_update_reports_anchor_and_patient_proxy_metrics(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=3), seed=12)
        config = _config_dict()
        config.update(
            {
                "algorithm": "gcn_residual_mdl2_replenish_ddpg",
                "batch_size": 2,
                "reward_scale": 1e-9,
                "env": asdict(env.config),
                "actor_readout_mode": "facility_action",
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "scale": 0.02,
                    "group_scales": {
                        "specimen_transfer": 0.0,
                        "reagent_transfer": 0.0,
                        "capacity_transfer": 0.0,
                        "replenishment": 0.02,
                    },
                    "positive_only_groups": ["replenishment"],
                    "l2_weight": 0.08,
                },
                "anchor_advantage_actor_loss": {
                    "enabled": True,
                    "negative_penalty_weight": 0.5,
                },
                "patient_service_proxy_actor_loss": {
                    "enabled": True,
                    "group": "replenishment",
                    "weight": 0.02,
                    "cost_weight": 0.01,
                    "low_pressure_weight": 0.01,
                    "positive_only": True,
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        state = env.reset(seed=12)
        for _step in range(2):
            action = agent.select_action(state, explore=False, env=env)
            next_state, reward, done, _info = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            state = next_state

        metrics = agent.update()

        self.assertIn("actor_anchor_advantage_mean", metrics)
        self.assertIn("actor_anchor_advantage_positive_fraction", metrics)
        self.assertIn("actor_anchor_negative_advantage_penalty", metrics)
        self.assertIn("actor_patient_service_proxy_alignment", metrics)
        self.assertIn("actor_patient_service_proxy_low_pressure_penalty", metrics)
        self.assertIn("actor_patient_service_proxy_cost_penalty", metrics)
        self.assertIn("residual_l2_loss", metrics)

    def test_ddpg_can_delay_actor_updates_for_conservative_fine_tuning(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=3), seed=14)
        config = _config_dict()
        config.update(
            {
                "batch_size": 2,
                "reward_scale": 1e-9,
                "env": asdict(env.config),
                "actor_update_frequency": 2,
                "critic_warmup_updates": 0,
            }
        )

        for agent_class in (GCNDDPGAgent, FlatDDPGAgent):
            agent = agent_class(env.observation_size, env.action_size, config)
            state = env.reset(seed=14)
            for _step in range(2):
                action = agent.select_action(state, explore=False, env=env)
                next_state, reward, done, _info = env.step(action)
                agent.observe(state, action, reward, next_state, done)
                state = next_state

            critic_only = agent.update()
            actor_and_critic = agent.update()

            self.assertEqual(critic_only["actor_updated"], 0.0)
            self.assertNotIn("actor_loss", critic_only)
            self.assertEqual(actor_and_critic["actor_updated"], 1.0)
            self.assertIn("actor_loss", actor_and_critic)

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

    def test_residual_positive_only_group_clamps_replenishment_for_graph_and_flat(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=21)
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "residual_action": {
                    "enabled": True,
                    "base_policy": "myo",
                    "scale": 0.25,
                    "positive_only_groups": ["replenishment"],
                },
            }
        )
        n = env.config.num_facilities
        residual = np.zeros(env.action_size, dtype=np.float32)
        residual[0:n] = np.linspace(-0.7, 0.7, n)
        residual[3 * n : 4 * n] = np.linspace(-0.8, 0.8, n)

        gcn_agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        flat_agent = FlatDDPGAgent(env.observation_size, env.action_size, config)

        for agent in (gcn_agent, flat_agent):
            transformed = agent._transform_network_residual_np(residual)
            np.testing.assert_allclose(transformed[0:n], residual[0:n])
            self.assertTrue(np.all(transformed[3 * n : 4 * n] >= 0.0))
            np.testing.assert_allclose(
                transformed[3 * n : 4 * n],
                np.maximum(residual[3 * n : 4 * n], 0.0),
            )

    def test_state_gate_limits_replenishment_residual_to_pressure_facilities(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=22)
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "residual_action": {
                    "enabled": True,
                    "base_policy": "pmyo",
                    "scale": 0.25,
                    "positive_only_groups": ["replenishment"],
                    "state_gate": {
                        "enabled": True,
                        "groups": ["replenishment"],
                        "threshold": 0.0,
                    },
                },
            }
        )
        n = env.config.num_facilities
        state = np.zeros(env.observation_size, dtype=np.float32)
        network_action = np.zeros(env.action_size, dtype=np.float32)
        network_action[3 * n : 4 * n] = 1.0
        state[0] = 2.0

        gcn_agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        flat_agent = FlatDDPGAgent(env.observation_size, env.action_size, config)

        for agent in (gcn_agent, flat_agent):
            residual = agent._policy_residual_np(state, network_action)
            self.assertGreater(float(residual[3 * n]), 0.0)
            np.testing.assert_allclose(residual[3 * n + 1 : 4 * n], np.zeros(n - 1))

    def test_pressure_projection_aligns_residual_with_state_pattern(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=31)
        state = env.reset(seed=31)
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "residual_action": {
                    "enabled": True,
                    "base_policy": "myo",
                    "scale": 0.25,
                    "center_groups": ["reagent_transfer"],
                    "pressure_projection": {
                        "enabled": True,
                        "groups": ["reagent_transfer"],
                    },
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        n = env.config.num_facilities
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        network_action = torch.zeros((1, env.action_size), dtype=torch.float32)
        network_action[0, n : 2 * n] = torch.linspace(-1.0, 1.0, n)

        residual = agent._policy_residuals_tensor(state_tensor, network_action).detach().numpy()[0]
        pattern = agent._residual_pressure_patterns_tensor(state_tensor)["resource"][0]
        current = network_action[:, n : 2 * n]
        denominator = pattern.pow(2).sum().clamp_min(1e-6)
        coefficient = float((current[0] * pattern).sum() / denominator)
        expected = (coefficient * pattern).numpy()

        np.testing.assert_allclose(residual[n : 2 * n], expected, atol=1e-6)
        np.testing.assert_allclose(residual[2 * n : 3 * n], np.zeros(n), atol=1e-6)

    def test_mean_pressure_projection_uses_positive_actor_intensity(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=33)
        state = env.reset(seed=33)
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "actor_readout_mode": "pressure_intensity",
                "residual_action": {
                    "enabled": True,
                    "base_policy": "myo",
                    "scale": 0.25,
                    "pressure_projection": {
                        "enabled": True,
                        "groups": ["reagent_transfer"],
                        "coefficient_mode": "mean",
                        "positive_coefficient": True,
                    },
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        n = env.config.num_facilities
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        network_action = torch.zeros((1, env.action_size), dtype=torch.float32)
        network_action[:, n : 2 * n] = 0.2

        residual = agent._policy_residuals_tensor(
            state_tensor,
            network_action,
        )
        pattern = agent._residual_pressure_patterns_tensor(
            state_tensor
        )["resource"]

        np.testing.assert_allclose(
            residual[:, n : 2 * n].detach().numpy(),
            (0.2 * pattern).detach().numpy(),
            atol=1e-6,
        )

        network_action[:, n : 2 * n] = -0.2
        blocked = agent._policy_residuals_tensor(
            state_tensor,
            network_action,
        )
        np.testing.assert_allclose(
            blocked[:, n : 2 * n].detach().numpy(),
            np.zeros((1, n)),
            atol=1e-6,
        )

    def test_pressure_intensity_rejects_centered_action_groups(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=34)
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "actor_readout_mode": "pressure_intensity",
                "residual_action": {
                    "enabled": True,
                    "base_policy": "myo",
                    "scale": 0.25,
                    "center_groups": ["reagent_transfer"],
                    "pressure_projection": {
                        "enabled": True,
                        "groups": ["reagent_transfer"],
                        "coefficient_mode": "mean",
                    },
                },
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "center_groups=\\[\\]",
        ):
            GCNDDPGAgent(env.observation_size, env.action_size, config)

    def test_pressure_patterns_account_for_pending_transfer_arrivals(self) -> None:
        env_config = replace(
            make_20_clinic_config(episode_horizon=2),
            transfer_lead_time=2,
            include_transfer_pipeline_state=True,
        )
        baseline_env = CapacityPlanningEnv(env_config, seed=32)
        pipeline_env = CapacityPlanningEnv(env_config, seed=32)
        for env in (baseline_env, pipeline_env):
            env.reset(seed=32)
            env.demand[:2] = np.array([5.0, 20.0])
            env.demand_forecast[:2] = np.array([5.0, 20.0])
            env.specimens[:2] = np.array([5.0, 20.0])
            env.reagents[:2] = np.array([25.0, 0.0])
        pipeline_env.reagent_transfer_pipeline[:, 1] = 60.0
        config = _config_dict()
        config["env"] = asdict(env_config)

        for agent_cls in (GCNDDPGAgent, FlatDDPGAgent):
            agent = agent_cls(baseline_env.observation_size, baseline_env.action_size, config)
            baseline_state = torch.as_tensor(
                baseline_env.observation(),
                dtype=torch.float32,
            ).unsqueeze(0)
            pipeline_state = torch.as_tensor(
                pipeline_env.observation(),
                dtype=torch.float32,
            ).unsqueeze(0)
            baseline_pattern = agent._residual_pressure_patterns_tensor(
                baseline_state
            )["resource"][0]
            pipeline_pattern = agent._residual_pressure_patterns_tensor(
                pipeline_state
            )["resource"][0]

            self.assertGreater(float(baseline_pattern[1]), float(baseline_pattern[0]))
            self.assertLess(float(pipeline_pattern[1]), float(pipeline_pattern[0]))

        config["residual_action"] = {
            "pressure_projection": {
                "subtract_pipeline": False,
            },
        }
        for agent_cls in (GCNDDPGAgent, FlatDDPGAgent):
            agent = agent_cls(
                baseline_env.observation_size,
                baseline_env.action_size,
                config,
            )
            baseline_pattern = agent._residual_pressure_patterns_tensor(
                torch.as_tensor(
                    baseline_env.observation(),
                    dtype=torch.float32,
                ).unsqueeze(0)
            )["resource"]
            pipeline_pattern = agent._residual_pressure_patterns_tensor(
                torch.as_tensor(
                    pipeline_env.observation(),
                    dtype=torch.float32,
                ).unsqueeze(0)
            )["resource"]

            np.testing.assert_allclose(
                baseline_pattern.numpy(),
                pipeline_pattern.numpy(),
                atol=1e-6,
            )

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
            {
                "epochs": 1,
                "batch_size": 2,
                "retain_for_regularization": True,
            },
            weights=np.asarray([0.25, 1.75], dtype=np.float32),
        )

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["target_mode"], "residual")
        self.assertGreaterEqual(summary["final_loss"], 0.0)
        self.assertEqual(tuple(agent.imitation_states.shape), (2, env.observation_size))
        self.assertEqual(tuple(agent.imitation_weights.shape), (2,))
        self.assertTrue(torch.isfinite(agent._actor_imitation_loss()))
        diagnostics = agent.evaluate_action_batch(
            states,
            actions,
            weights=np.asarray([0.25, 1.75], dtype=np.float32),
        )
        self.assertEqual(diagnostics["samples"], 2)
        self.assertTrue(np.isfinite(diagnostics["loss"]))
        self.assertTrue(np.isfinite(diagnostics["actor_loss"]))
        with self.assertRaises(ValueError):
            agent.fit_action_batch(states, actions, {"epochs": 1}, weights=np.asarray([1.0]))
        with self.assertRaises(ValueError):
            agent.fit_action_batch(
                states,
                actions,
                {"epochs": 1, "target_mode": "unsupported"},
            )

    def test_graph_correction_gate_learns_teacher_labels_and_can_return_anchor(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=29)
        config = _config_dict()
        config.update(
            {
                "batch_size": 2,
                "env": asdict(env.config),
                "actor_readout_mode": "facility_action",
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "scale": 0.1,
                    "group_scales": {
                        "specimen_transfer": 0.0,
                        "reagent_transfer": 0.0,
                        "capacity_transfer": 0.0,
                        "replenishment": 0.1,
                    },
                    "correction_gate": {
                        "enabled": True,
                        "groups": ["replenishment"],
                        "threshold": 0.5,
                        "hidden_sizes": [8],
                        "lr": 0.001,
                    },
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        first_state = env.reset(seed=29)
        first_anchor = agent._base_action_from_state_np(first_state)
        second_state, _reward, _done, _info = env.step(first_anchor)
        second_anchor = agent._base_action_from_state_np(second_state)
        corrected = second_anchor.copy()
        corrected[3 * env.config.num_facilities] = np.clip(
            corrected[3 * env.config.num_facilities] + 0.05,
            -1.0,
            1.0,
        )

        summary = agent.fit_action_batch(
            np.stack([first_state, second_state]),
            np.stack([first_anchor, corrected]),
            {"epochs": 2, "batch_size": 2},
        )

        self.assertIsNotNone(agent.correction_gate)
        self.assertAlmostEqual(summary["correction_gate_label_rate"], 0.5)
        self.assertTrue(np.isfinite(summary["correction_gate_loss"]))
        gate_summary = agent.fit_correction_gate_batch(
            np.stack([first_state, second_state]),
            np.asarray([0.0, 1.0], dtype=np.float32),
            {"epochs": 2, "batch_size": 2},
        )
        self.assertEqual(gate_summary["samples"], 2)
        self.assertAlmostEqual(gate_summary["label_rate"], 0.5)
        ungated = agent.select_ungated_residual_action(
            first_state,
            env=env,
            residual_scale=0.25,
        )
        self.assertEqual(ungated.shape, (env.action_size,))

        for parameter in agent.correction_gate.parameters():
            parameter.data.zero_()
        output_layer = next(
            module
            for module in reversed(tuple(agent.correction_gate.head.modules()))
            if isinstance(module, torch.nn.Linear)
        )
        output_layer.bias.data.fill_(-10.0)
        action = agent.select_action(first_state, explore=False, env=env)
        np.testing.assert_allclose(action, first_anchor, atol=1e-6)

    def test_advantage_gate_penalizes_clinically_infeasible_targets(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=290)
        config = _config_dict()
        config.update(
            {
                "batch_size": 2,
                "env": asdict(env.config),
                "actor_readout_mode": "facility_action",
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "scale": 0.1,
                    "correction_gate": {
                        "enabled": True,
                        "mode": "advantage",
                        "groups": ["replenishment"],
                        "threshold": 0.5,
                        "hidden_sizes": [8],
                    },
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        first_state = env.reset(seed=290)
        first_anchor = agent._base_action_from_state_np(first_state)
        second_state, _reward, _done, _info = env.step(first_anchor)

        summary = agent.fit_correction_gate_batch(
            np.stack([first_state, second_state]),
            np.ones(2, dtype=np.float32),
            {
                "epochs": 1,
                "batch_size": 2,
                "mode": "advantage",
                "advantages": np.asarray(
                    [2_000_000.0, 2_000_000.0],
                    dtype=np.float32,
                ),
                "feasible": np.asarray([True, False]),
                "infeasible_advantage_penalty": 5_000_000.0,
            },
        )

        self.assertAlmostEqual(
            summary["target_advantage_mean"],
            -1_500_000.0,
            places=2,
        )

    def test_clinical_safety_gate_can_block_advantage_release(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=292)
        config = _config_dict()
        config.update(
            {
                "batch_size": 2,
                "env": asdict(env.config),
                "actor_readout_mode": "facility_action",
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "scale": 0.1,
                    "correction_gate": {
                        "enabled": True,
                        "mode": "advantage",
                        "groups": ["replenishment"],
                        "threshold": 0.5,
                        "hidden_sizes": [8],
                        "clinical_safety_gate": {
                            "enabled": True,
                            "threshold": 0.5,
                            "hidden_sizes": [8],
                        },
                    },
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        first_state = env.reset(seed=292)
        first_anchor = agent._base_action_from_state_np(first_state)
        second_state, _reward, _done, _info = env.step(first_anchor)

        summary = agent.fit_correction_safety_gate_batch(
            np.stack([first_state, second_state]),
            np.asarray([1.0, 0.0], dtype=np.float32),
            {
                "epochs": 2,
                "batch_size": 2,
                "unsafe_weight": 2.0,
            },
        )
        self.assertEqual(summary["samples"], 2)
        self.assertAlmostEqual(summary["safe_rate"], 0.5)

        for parameter in agent.correction_gate.parameters():
            parameter.data.zero_()
        advantage_output = next(
            module
            for module in reversed(tuple(agent.correction_gate.head.modules()))
            if isinstance(module, torch.nn.Linear)
        )
        advantage_output.bias.data.fill_(10.0)
        for parameter in agent.correction_safety_gate.parameters():
            parameter.data.zero_()
        safety_output = next(
            module
            for module in reversed(
                tuple(agent.correction_safety_gate.head.modules())
            )
            if isinstance(module, torch.nn.Linear)
        )
        safety_output.bias.data.fill_(-10.0)

        action = agent.select_action(first_state, explore=False, env=env)
        np.testing.assert_allclose(action, first_anchor, atol=1e-6)

    def test_graph_gate_can_encode_signed_actor_proposals_per_clinic(self) -> None:
        env = CapacityPlanningEnv(
            make_20_clinic_config(episode_horizon=2),
            seed=2910,
        )
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "scale": 0.1,
                    "correction_gate": {
                        "enabled": True,
                        "groups": [
                            "reagent_transfer",
                            "capacity_transfer",
                            "replenishment",
                        ],
                        "include_proposed_residual_features": True,
                    },
                },
            }
        )
        agent = GCNDDPGAgent(
            env.observation_size,
            env.action_size,
            config,
        )
        state = env.reset(seed=2910)
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
        ).unsqueeze(0)
        residuals = torch.zeros(
            (1, env.action_size),
            dtype=torch.float32,
        )
        n = env.config.num_facilities
        residuals[0, n] = -0.5
        residuals[0, n + 1] = 0.5

        features = agent.correction_gate_node_features(
            state_tensor,
            residuals=residuals,
        )

        self.assertEqual(
            features.shape[-1],
            agent.graph_spec.node_feature_dim + 4,
        )
        expected = (
            residuals.cpu().numpy()[0]
            * agent.residual_scale_vector
        )
        np.testing.assert_allclose(
            features[0, :n, -3].cpu().numpy(),
            expected[n : 2 * n],
        )
        if agent.graph_spec.num_nodes > n:
            np.testing.assert_allclose(
                features[0, n:, -4:].cpu().numpy(),
                0.0,
            )

    def test_gate_fit_accepts_actor_proposals_from_another_policy(self) -> None:
        env = CapacityPlanningEnv(
            make_20_clinic_config(episode_horizon=2),
            seed=2911,
        )
        config = _config_dict()
        config.update(
            {
                "batch_size": 2,
                "env": asdict(env.config),
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "correction_gate": {
                        "enabled": True,
                        "mode": "advantage",
                        "groups": ["capacity_transfer"],
                        "include_proposed_residual_features": True,
                    },
                },
            }
        )
        agent = GCNDDPGAgent(
            env.observation_size,
            env.action_size,
            config,
        )
        first_state = env.reset(seed=2911)
        anchor = agent._base_action_from_state_np(first_state)
        second_state, _reward, _done, _info = env.step(anchor)
        proposed = np.zeros((2, env.action_size), dtype=np.float32)
        proposed[:, 2 * env.config.num_facilities] = 0.4
        captured: dict[str, np.ndarray] = {}
        original_builder = agent.correction_gate_node_features

        def capture_features(states, *, residuals=None):
            captured["residuals"] = residuals.detach().cpu().numpy()
            return original_builder(states, residuals=residuals)

        agent.correction_gate_node_features = capture_features
        summary = agent.fit_correction_gate_batch(
            np.stack([first_state, second_state]),
            np.ones((2, 1), dtype=np.float32),
            {
                "epochs": 1,
                "batch_size": 2,
                "mode": "advantage",
                "advantages": np.full(
                    (2, 1),
                    1_000_000.0,
                    dtype=np.float32,
                ),
                "proposed_residuals": proposed,
            },
        )

        self.assertEqual(summary["samples"], 2)
        np.testing.assert_allclose(captured["residuals"], proposed)

    def test_graph_gate_can_encode_clipped_action_delta(self) -> None:
        env = CapacityPlanningEnv(
            make_20_clinic_config(episode_horizon=2),
            seed=2912,
        )
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "correction_gate": {
                        "enabled": True,
                        "groups": ["capacity_transfer"],
                        "include_proposed_residual_features": True,
                        "proposed_residual_feature_mode": (
                            "clipped_action_delta"
                        ),
                    },
                },
            }
        )
        agent = GCNDDPGAgent(
            env.observation_size,
            env.action_size,
            config,
        )
        state = env.reset(seed=2912)
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
        ).unsqueeze(0)
        residuals = torch.ones(
            (1, env.action_size),
            dtype=torch.float32,
        )

        features = agent.correction_gate_node_features(
            state_tensor,
            residuals=residuals,
        )

        base = agent._base_actions_from_states_tensor(state_tensor)
        scale = torch.as_tensor(
            agent.residual_scale_vector,
            dtype=torch.float32,
        ).unsqueeze(0)
        expected = (
            torch.clamp(base + residuals * scale, -1.0, 1.0)
            - base
        )
        n = env.config.num_facilities
        np.testing.assert_allclose(
            features[0, :n, -2].detach().cpu().numpy(),
            expected[0, 2 * n : 3 * n].detach().cpu().numpy(),
        )

    def test_correction_gate_can_penalize_false_positives(self) -> None:
        env = CapacityPlanningEnv(
            make_20_clinic_config(episode_horizon=2),
            seed=290,
        )
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "correction_gate": {
                        "enabled": True,
                        "groups": ["reagent_transfer"],
                        "negative_class_weight": 3.0,
                    },
                },
            }
        )
        agent = GCNDDPGAgent(
            env.observation_size,
            env.action_size,
            config,
        )
        losses = agent._correction_gate_bce(
            torch.zeros(2),
            torch.tensor([0.0, 1.0]),
        )

        self.assertAlmostEqual(
            float(losses[0]),
            3.0 * float(losses[1]),
            places=5,
        )

    def test_single_group_gate_uses_deployment_group_threshold(self) -> None:
        env = CapacityPlanningEnv(
            make_20_clinic_config(episode_horizon=2),
            seed=291,
        )
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "correction_gate": {
                        "enabled": True,
                        "groups": ["reagent_transfer"],
                        "threshold": 0.3,
                    },
                },
            }
        )

        for agent_cls in (GCNDDPGAgent, FlatDDPGAgent):
            agent = agent_cls(
                env.observation_size,
                env.action_size,
                config,
            )
            agent.correction_gate_group_thresholds = (0.1,)
            threshold = agent._correction_gate_threshold_tensor(
                torch.as_tensor([0.2], dtype=torch.float32)
            )

            self.assertAlmostEqual(float(threshold), 0.1)

    def test_group_gate_can_upweight_rare_positive_labels(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=30)
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "correction_gate": {
                        "enabled": True,
                        "groups": [
                            "reagent_transfer",
                            "capacity_transfer",
                            "replenishment",
                        ],
                        "positive_class_weight_power": 0.5,
                        "positive_class_weight_max": 4.0,
                    },
                },
            }
        )
        labels = torch.as_tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        for agent_class in (GCNDDPGAgent, FlatDDPGAgent):
            agent = agent_class(
                env.observation_size,
                env.action_size,
                config,
            )
            weights = agent._correction_gate_positive_weights(
                labels.to(agent.device)
            ).cpu().numpy()

            np.testing.assert_allclose(
                weights,
                np.asarray([1.0, np.sqrt(3.0), 1.0]),
                rtol=1e-6,
            )

    def test_deployment_gate_does_not_block_residual_training_path(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=31)
        config = _config_dict()
        config.update(
            {
                "batch_size": 2,
                "env": asdict(env.config),
                "actor_readout_mode": "facility_action",
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "scale": 0.1,
                    "correction_gate": {
                        "enabled": True,
                        "mode": "advantage",
                        "threshold": 0.5,
                        "hidden_sizes": [8],
                    },
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        state = env.reset(seed=31)
        anchor = agent._base_action_from_state_np(state)
        residual = np.zeros(env.action_size, dtype=np.float32)
        replenishment_index = 3 * env.config.num_facilities
        residual[replenishment_index] = (
            -0.5 if anchor[replenishment_index] > 0.0 else 0.5
        )

        deployment_action = agent._compose_action_np(state, residual)
        training_action = agent._compose_action_np(
            state,
            residual,
            apply_correction_gate=False,
        )

        np.testing.assert_allclose(deployment_action, anchor, atol=1e-6)
        self.assertFalse(np.allclose(training_action, anchor))

    def test_group_gate_can_release_replenishment_without_transfer(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=41)
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "actor_readout_mode": "facility_action",
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "scale": 0.1,
                    "correction_gate": {
                        "enabled": True,
                        "mode": "classification",
                        "threshold": 0.5,
                        "hidden_sizes": [8],
                        "groups": [
                            "reagent_transfer",
                            "capacity_transfer",
                            "replenishment",
                        ],
                    },
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        state = env.reset(seed=41)
        anchor = agent._base_action_from_state_np(state)
        residual = np.zeros(env.action_size, dtype=np.float32)
        n = env.config.num_facilities
        residual[n] = 0.5
        residual[n + 1] = -0.5
        residual[3 * n] = -0.5 if anchor[3 * n] > 0.0 else 0.5

        for parameter in agent.correction_gate.parameters():
            parameter.data.zero_()
        output_layer = next(
            module
            for module in reversed(tuple(agent.correction_gate.head.modules()))
            if isinstance(module, torch.nn.Linear)
        )
        output_layer.bias.data.copy_(
            torch.as_tensor([-10.0, -10.0, 10.0])
        )
        action = agent._compose_action_np(state, residual)

        np.testing.assert_allclose(action[n : 3 * n], anchor[n : 3 * n], atol=1e-6)
        self.assertFalse(np.allclose(action[3 * n :], anchor[3 * n :]))

    def test_combined_transfer_gate_controls_reagent_and_capacity_together(self) -> None:
        env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=2), seed=43)
        config = _config_dict()
        config.update(
            {
                "env": asdict(env.config),
                "residual_action": {
                    "enabled": True,
                    "base_policy": "mdl2",
                    "scale": 0.1,
                    "correction_gate": {
                        "enabled": True,
                        "groups": ["combined_transfer"],
                        "threshold": 0.5,
                        "hidden_sizes": [8],
                    },
                },
            }
        )
        agent = GCNDDPGAgent(env.observation_size, env.action_size, config)
        n = env.config.num_facilities
        residuals = torch.zeros((2, env.action_size), dtype=torch.float32)
        residuals[0, n] = 0.5
        residuals[0, 2 * n] = -0.5
        residuals[1, 3 * n] = 0.5
        mask = torch.ones_like(residuals)

        labels = agent._correction_gate_labels_tensor(residuals, mask)
        gated = agent._apply_group_gate_probabilities(
            residuals,
            torch.as_tensor([[0.0], [1.0]], dtype=torch.float32),
        )

        np.testing.assert_array_equal(
            labels.detach().numpy(),
            np.asarray([1.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            gated[0, n : 3 * n].detach().numpy(),
            np.zeros(2 * n),
        )
        self.assertEqual(float(gated[1, 3 * n]), 0.5)


if __name__ == "__main__":
    unittest.main()
