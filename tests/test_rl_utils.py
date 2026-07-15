import unittest

import numpy as np

from evaluation.aggregate_results import aggregate_rows
from evaluation.run_smoke_comparison import _smoke_config
from src.rl.action_projection import project_action
from src.rl.agents import available_algorithms, get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env
from src.rl.networks import default_torch_device, resolve_torch_device
from src.rl.preprocessing import FixedObservationScaler


class RLUtilsTest(unittest.TestCase):
    def test_project_action_clips_normalized_bounds(self):
        projected = project_action(np.array([-2.0, 0.25, 3.0]), action_space_info=3)

        np.testing.assert_allclose(projected.action, np.array([-1.0, 0.25, 1.0], dtype=np.float32))
        self.assertTrue(projected.clipped)

    def test_project_action_checks_shape(self):
        with self.assertRaises(ValueError):
            project_action(np.array([0.0, 0.0]), action_space_info=3)

    def test_torch_device_auto_prefers_available_accelerator(self):
        device_name = default_torch_device()

        self.assertIn(device_name, {"cpu", "cuda", "mps"})
        self.assertEqual(resolve_torch_device("auto").type, device_name)
        self.assertEqual(resolve_torch_device("cpu").type, "cpu")

    def test_json_compatible_yaml_config_loads(self):
        config = load_config("configs/flat_ddpg.yaml")

        self.assertEqual(config["algorithm"], "flat_ddpg")
        self.assertIn("env", config)

    def test_td3_20_clinic_config_loads(self):
        config = load_config("configs/td3_20_clinic.yaml")

        self.assertEqual(config["algorithm"], "td3")
        self.assertEqual(config["env"]["num_facilities"], 20)
        self.assertEqual(config["env"]["action_mode"], "facility_net")

    def test_gcn_td3_20_clinic_config_loads(self):
        config = load_config("configs/gcn_td3_20_clinic.yaml")

        self.assertEqual(config["algorithm"], "gcn_td3")
        self.assertEqual(config["env"]["graph_ablation"], "full_graph")
        self.assertEqual(config["env"]["num_facilities"], 20)
        self.assertEqual(config["actor_readout_mode"], "facility_action")

    def test_gcn_residual_config_loads(self):
        config = load_config("configs/gcn_residual_20_clinic.yaml")

        self.assertEqual(config["algorithm"], "gcn_residual_mdl2")
        self.assertEqual(config["residual_action"]["base_policy"], "mdl2")
        self.assertTrue(config["residual_action"]["zero_init_actor"])
        self.assertEqual(config["residual_action"]["group_scales"]["specimen_transfer"], 0.0)
        self.assertGreater(config["residual_action"]["group_scales"]["replenishment"], 0.0)
        self.assertIn("replenishment", config["residual_action"]["center_groups"])

    def test_flat_residual_config_loads(self):
        config = load_config("configs/flat_residual_20_clinic.yaml")

        self.assertEqual(config["algorithm"], "flat_residual_mdl2")
        self.assertEqual(config["env"]["graph_ablation"], "flat_state_no_graph")
        self.assertEqual(config["residual_action"]["base_policy"], "mdl2")
        self.assertTrue(config["residual_action"]["zero_init_actor"])
        self.assertEqual(config["residual_action"]["group_scales"]["specimen_transfer"], 0.0)
        self.assertGreater(config["imitation_pretrain"]["regularization_weight"], 0.0)

    def test_residual_algorithm_aliases_use_gcn_ddpg_agent(self):
        algorithms = available_algorithms()

        self.assertIn("gcn_residual_mdl2", algorithms)
        self.assertIn("gcn_pure_ddpg", algorithms)
        self.assertIn("gcn_residual_pmyo", algorithms)
        self.assertIs(get_agent_class("gcn_residual_mdl2"), get_agent_class("gcn_ddpg"))
        self.assertIs(get_agent_class("gcn_pure_ddpg"), get_agent_class("gcn_ddpg"))
        self.assertIs(get_agent_class("gcn_residual_pmyo"), get_agent_class("gcn_ddpg"))

    def test_flat_residual_algorithm_aliases_use_flat_ddpg_agent(self):
        algorithms = available_algorithms()

        self.assertIn("flat_residual_mdl2", algorithms)
        self.assertIn("flat_residual_iso", algorithms)
        self.assertIn("flat_residual_myo", algorithms)
        self.assertIn("flat_residual_pmyo", algorithms)
        self.assertIs(get_agent_class("flat_residual_mdl2"), get_agent_class("flat_ddpg"))
        self.assertIs(get_agent_class("flat_residual_iso"), get_agent_class("flat_ddpg"))
        self.assertIs(get_agent_class("flat_residual_pmyo"), get_agent_class("flat_ddpg"))

    def test_sac_20_clinic_config_loads(self):
        config = load_config("configs/sac_20_clinic.yaml")

        self.assertEqual(config["algorithm"], "sac")
        self.assertEqual(config["env"]["num_facilities"], 20)
        self.assertEqual(config["env"]["action_mode"], "facility_net")

    def test_ppo_20_clinic_config_loads(self):
        config = load_config("configs/ppo_20_clinic.yaml")

        self.assertEqual(config["algorithm"], "ppo")
        self.assertEqual(config["env"]["num_facilities"], 20)
        self.assertEqual(config["env"]["action_mode"], "facility_net")

    def test_disruption_scenario_config_loads(self):
        config = load_config("experiments/configs/20_clinic_disruption_0_6.json")

        self.assertEqual(config["scenario_name"], "manuscript_20_clinic_disruption_0_6")
        self.assertEqual(config["supplier_disruption_rate"], 0.6)
        self.assertEqual(config["num_facilities"], 20)

    def test_graph_stress_scenario_config_loads(self):
        config = load_config("experiments/configs/20_clinic_graph_stress_capacity_bottleneck.json")

        self.assertEqual(config["scenario_name"], "graph_stress_capacity_bottleneck")
        self.assertEqual(config["num_facilities"], 20)
        self.assertIsInstance(config["demand_rates"], list)

    def test_graph_dynamic_transfer_delay_config_builds_env(self):
        env_config = load_config("experiments/configs/20_clinic_graph_dynamic_transfer_delay.json")
        config = load_config("configs/gcn_ddpg_20_clinic.yaml")
        config["env"] = env_config
        env = build_env(config, seed=0)
        scaler = FixedObservationScaler.from_config(config, env.observation_size)

        self.assertEqual(env.config.transfer_lead_time, 1)
        self.assertTrue(env.config.include_transfer_pipeline_state)
        self.assertEqual(env.observation_size, 200)
        self.assertEqual(env.graph_observation()["node_features"].shape, (21, 10))
        self.assertEqual(scaler.scales.shape, (200,))

    def test_graph_dynamic_patient_forecast_config_builds_env(self):
        env_config = load_config("experiments/configs/20_clinic_graph_dynamic_patient_forecast.json")
        config = load_config("configs/gcn_ddpg_20_clinic.yaml")
        config["env"] = env_config
        env = build_env(config, seed=0)
        scaler = FixedObservationScaler.from_config(config, env.observation_size)

        self.assertTrue(env.config.include_demand_forecast_state)
        self.assertEqual(env.config.demand_forecast_horizon, 2)
        self.assertEqual(env.observation_size, 220)
        self.assertEqual(env.graph_observation()["node_features"].shape, (21, 11))
        self.assertEqual(scaler.scales.shape, (220,))

    def test_geographic_patient_forecast_config_builds_env(self):
        env_config = load_config(
            "experiments/configs/20_clinic_graph_dynamic_patient_forecast_geo.json"
        )
        config = load_config("configs/gcn_ddpg_20_clinic.yaml")
        config["env"] = env_config
        env = build_env(config, seed=0)
        scaler = FixedObservationScaler.from_config(config, env.observation_size)
        graph = env.graph_observation()

        self.assertEqual(env.scenario_name, "graph_dynamic_patient_forecast_geo")
        self.assertEqual(len(env.clinic_coordinates), 20)
        self.assertEqual(env.config.transfer_lead_time, 3)
        self.assertEqual(env.transfer_delay_thresholds, (500.0, 1500.0))
        self.assertGreater(env.config.geographic_transfer_cost_scale, 0.0)
        self.assertGreater(len(env.information_edges), 20)
        self.assertEqual(graph["clinic_coordinates"].shape, (20, 2))
        self.assertEqual(graph["clinic_distance_matrix"].shape, (20, 20))
        self.assertEqual(scaler.scales.shape, (220,))

    def test_gcn_config_enables_imitation_pretrain(self):
        config = load_config("configs/gcn_ddpg_20_clinic.yaml")
        plan = load_config("experiments/configs/graph_stress_benchmark.json")

        self.assertTrue(config["imitation_pretrain"]["enabled"])
        self.assertEqual(config["imitation_pretrain"]["policy"], "myo")
        self.assertGreater(config["imitation_pretrain"]["regularization_weight"], 0.0)
        self.assertTrue(config["residual_action"]["enabled"])
        self.assertEqual(config["residual_action"]["base_policy"], "myo")
        self.assertGreater(config["residual_action"]["scale"], 0.0)
        self.assertEqual(config["residual_action"]["group_scales"]["specimen_transfer"], 0.0)
        self.assertEqual(config["residual_action"]["group_scales"]["reagent_transfer"], 0.0)
        self.assertEqual(config["residual_action"]["group_scales"]["capacity_transfer"], 0.0)
        self.assertLess(
            config["residual_action"]["group_scales"]["specimen_transfer"],
            config["residual_action"]["group_scales"]["replenishment"],
        )
        self.assertGreater(config["residual_action"]["l2_weight"], 0.0)
        self.assertTrue(config["elite_imitation"]["enabled"])
        self.assertGreaterEqual(config["elite_imitation"]["warmup_episodes"], 1)
        self.assertGreaterEqual(config["elite_imitation"]["max_episodes"], 2)
        self.assertLess(config["exploration_noise"]["sigma"], 0.2)
        self.assertLess(config["actor_lr"], 0.0001)
        self.assertEqual(plan["budgets"]["gcn_tune"]["num_episodes"], 300)

    def test_fixed_observation_scaler_uses_env_capacity_metadata(self):
        config = load_config("configs/flat_ddpg_20_clinic.yaml")
        env = build_env(config, seed=0)
        scaler = FixedObservationScaler.from_config(config, env.observation_size)
        state = env.reset(seed=0)

        normalized = scaler.normalize_np(state)

        self.assertTrue(scaler.enabled)
        self.assertEqual(normalized.shape, state.shape)
        self.assertLessEqual(float(np.max(np.abs(normalized))), 10.0)

    def test_fixed_observation_scaler_handles_patient_summary(self):
        config = load_config("configs/flat_residual_20_clinic.yaml")
        config["env"] = load_config("experiments/configs/20_clinic_patient_condition_stress.json")
        env = build_env(config, seed=0)
        scaler = FixedObservationScaler.from_config(config, env.observation_size)
        state = env.reset(seed=0)

        normalized = scaler.normalize_np(state)

        self.assertTrue(scaler.enabled)
        self.assertEqual(scaler.scales.shape, (env.observation_size,))
        self.assertEqual(normalized.shape, state.shape)

    def test_aggregate_rows_computes_mean(self):
        rows = [
            {"algorithm": "myo", "scenario": "s", "graph_ablation": "full_graph", "total_cost": "10"},
            {"algorithm": "myo", "scenario": "s", "graph_ablation": "full_graph", "total_cost": "14"},
        ]

        summary = aggregate_rows(rows, metrics=("total_cost",))

        self.assertEqual(summary[0]["count"], 2)
        self.assertEqual(summary[0]["total_cost_mean"], 12.0)

    def test_smoke_config_overrides_training_scale(self):
        config = load_config("configs/flat_ddpg_20_clinic.yaml")
        smoke = _smoke_config(
            config,
            algorithm="flat_ddpg",
            seed=3,
            episodes=1,
            steps=4,
            batch_size=2,
        )

        self.assertEqual(smoke["seed"], 3)
        self.assertEqual(smoke["num_episodes"], 1)
        self.assertEqual(smoke["max_steps_per_episode"], 4)
        self.assertEqual(smoke["batch_size"], 2)
        self.assertEqual(smoke["env"]["episode_horizon"], 4)

    def test_flat_state_no_graph_keeps_environment_action_space(self):
        config = load_config("configs/flat_ddpg.yaml")
        env = build_env(config, seed=0)

        self.assertEqual(env.action_size, 5)


if __name__ == "__main__":
    unittest.main()
