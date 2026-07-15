import unittest

import numpy as np

from src.env.capacity_planning import (
    CapacityPlanningConfig,
    CapacityPlanningEnv,
    make_20_clinic_config,
    make_legacy_two_facility_config,
)
from src.graph.ablation import with_graph_ablation


class CapacityPlanningEnvTest(unittest.TestCase):
    def test_legacy_two_facility_shapes(self):
        config = make_legacy_two_facility_config(episode_horizon=3)
        env = CapacityPlanningEnv(config, seed=7)

        self.assertEqual(env.observation_size, 16)
        self.assertEqual(env.action_size, 5)
        self.assertEqual(env.reset(seed=7).shape, (16,))

    def test_noop_episode_keeps_state_nonnegative(self):
        config = make_legacy_two_facility_config(episode_horizon=4)
        env = CapacityPlanningEnv(config, seed=11)
        done = False

        while not done:
            observation, reward, done, info = env.step(env.noop_action())
            self.assertEqual(observation.shape, (env.observation_size,))
            self.assertTrue(np.isfinite(reward))
            self.assertGreaterEqual(float(info["cost"]), 0.0)
            self.assertTrue(np.all(env.specimens >= 0.0))
            self.assertTrue(np.all(env.reagents >= 0.0))
            self.assertTrue(np.all(env.bioreactors >= 0.0))

        self.assertEqual(env.t, config.episode_horizon)

    def test_graph_ablation_removes_action_dimensions(self):
        config = make_legacy_two_facility_config(episode_horizon=2)
        ablated = with_graph_ablation(config, remove_capacity_edges=True, remove_resource_edges=True)
        env = CapacityPlanningEnv(ablated, seed=3)

        self.assertEqual(env.specimen_edges, ((0, 1),))
        self.assertEqual(env.capacity_edges, ())
        self.assertEqual(env.resource_edges, ())
        self.assertEqual(env.action_size, 3)

    def test_manuscript_20_clinic_shapes(self):
        config = make_20_clinic_config(episode_horizon=2, supplier_disruption_rate=0.3)
        env = CapacityPlanningEnv(config, seed=5)
        graph = env.graph_observation()

        self.assertEqual(env.config.num_facilities, 20)
        self.assertEqual(env.config.production_lead_time, 3)
        self.assertEqual(env.observation_size, 140)
        self.assertEqual(env.action_size, 80)
        self.assertEqual(env.reset(seed=5).shape, (140,))
        self.assertEqual(graph["node_features"].shape, (21, 7))
        self.assertEqual(graph["capacity_edges"].shape, (20, 2))
        self.assertTrue(np.all(graph["capacity_edges"][:, 1] == 20))

    def test_20_clinic_noop_metrics_are_available(self):
        config = make_20_clinic_config(episode_horizon=2, supplier_disruption_rate=0.0)
        env = CapacityPlanningEnv(config, seed=9)

        _observation, _reward, _done, info = env.step(env.noop_action())

        self.assertIn("service_level", info)
        self.assertIn("average_waiting_time", info)
        self.assertIn("bioreactor_utilization", info)
        self.assertIn("supplier_available", info)
        self.assertEqual(info["supplier_available"].shape, (20,))

    def test_supplier_disruption_blocks_replenishment(self):
        config = make_20_clinic_config(episode_horizon=1, supplier_disruption_rate=1.0)
        env = CapacityPlanningEnv(config, seed=4)
        action = env.noop_action()
        action[60:80] = 1.0

        _observation, _reward, _done, info = env.step(action)

        self.assertTrue(np.all(info["supplier_available"] == 0.0))
        self.assertTrue(np.all(info["replenishment"] == 0.0))

    def test_capacity_ablation_keeps_facility_action_shape_but_blocks_q(self):
        config = make_20_clinic_config(episode_horizon=1, supplier_disruption_rate=0.0)
        ablated = with_graph_ablation(config, remove_capacity_edges=True)
        env = CapacityPlanningEnv(ablated, seed=8)
        action = env.noop_action()
        action[40:60] = 1.0

        _observation, _reward, _done, info = env.step(action)

        self.assertEqual(env.action_size, 80)
        self.assertEqual(env.graph_observation()["capacity_edges"].shape, (0, 2))
        self.assertTrue(np.all(info["capacity_transfers"] == 0.0))

    def test_geographic_coordinates_drive_default_facility_edges(self):
        coordinates = ((0.0, 0.0), (0.0, 1.0), (20.0, 20.0), (20.0, 21.0))
        config = CapacityPlanningConfig(
            num_facilities=4,
            production_lead_time=2,
            episode_horizon=1,
            demand_rates=(0.0, 0.0, 0.0, 0.0),
            initial_specimens=(0.0, 0.0, 0.0, 0.0),
            initial_reagents=(0.0, 0.0, 0.0, 0.0),
            initial_idle_bioreactors=(0.0, 0.0, 0.0, 0.0),
            max_specimens=(10.0, 10.0, 10.0, 10.0),
            max_reagents=(10.0, 10.0, 10.0, 10.0),
            max_idle_bioreactors=(5.0, 5.0, 5.0, 5.0),
            max_reagent_replenishment=(0.0, 0.0, 0.0, 0.0),
            action_mode="facility_net",
            clinic_coordinates=coordinates,
            geographic_neighbor_k=1,
        )
        env = CapacityPlanningEnv(config, seed=12)
        graph = env.graph_observation()

        self.assertEqual(env.information_edges, ((0, 1), (2, 3)))
        self.assertEqual(env.specimen_edges, ((0, 1), (2, 3)))
        self.assertEqual(env.resource_edges, ((0, 1), (2, 3)))
        self.assertEqual(graph["clinic_coordinates"].shape, (4, 2))
        self.assertEqual(graph["clinic_distance_matrix"].shape, (4, 4))

    def test_facility_net_transfer_lead_time_delays_arrivals(self):
        config = CapacityPlanningConfig(
            num_facilities=2,
            production_lead_time=2,
            episode_horizon=3,
            demand_rates=(0.0, 0.0),
            initial_specimens=(0.0, 0.0),
            initial_reagents=(20.0, 0.0),
            initial_idle_bioreactors=(0.0, 0.0),
            max_specimens=(50.0, 50.0),
            max_reagents=(50.0, 50.0),
            max_idle_bioreactors=(10.0, 10.0),
            max_reagent_replenishment=(0.0, 0.0),
            max_reagent_transfer=20.0,
            action_mode="facility_net",
            transfer_lead_time=1,
            include_transfer_pipeline_state=True,
            resource_edges=((0, 1),),
        )
        env = CapacityPlanningEnv(config, seed=12)
        action = env.noop_action()
        action[2] = -1.0
        action[3] = 1.0

        observation, _reward, _done, info = env.step(action)

        self.assertEqual(observation.shape, (16,))
        self.assertAlmostEqual(env.reagents[0], 0.0)
        self.assertAlmostEqual(env.reagents[1], 0.0)
        self.assertAlmostEqual(env.reagent_transfer_pipeline.sum(axis=0)[1], 20.0)
        self.assertTrue(np.all(info["reagent_transfer_arrivals"] == 0.0))

        _observation, _reward, _done, info = env.step(env.noop_action())

        self.assertAlmostEqual(env.reagents[1], 20.0)
        self.assertAlmostEqual(info["reagent_transfer_arrivals"][1], 20.0)

    def test_geographic_transfer_delay_uses_distance_thresholds(self):
        config = CapacityPlanningConfig(
            num_facilities=2,
            production_lead_time=2,
            episode_horizon=4,
            demand_rates=(0.0, 0.0),
            initial_specimens=(0.0, 0.0),
            initial_reagents=(20.0, 0.0),
            initial_idle_bioreactors=(0.0, 0.0),
            max_specimens=(50.0, 50.0),
            max_reagents=(50.0, 50.0),
            max_idle_bioreactors=(10.0, 10.0),
            max_reagent_replenishment=(0.0, 0.0),
            max_reagent_transfer=20.0,
            action_mode="facility_net",
            transfer_lead_time=3,
            include_transfer_pipeline_state=True,
            clinic_coordinates=((0.0, 0.0), (0.0, 10.0)),
            geographic_neighbor_k=1,
            transfer_lead_time_distance_thresholds=(100.0, 500.0),
        )
        env = CapacityPlanningEnv(config, seed=12)
        action = env.noop_action()
        action[2] = -1.0
        action[3] = 1.0

        _observation, _reward, _done, info = env.step(action)

        self.assertTrue(np.all(info["reagent_transfer_arrivals"] == 0.0))
        self.assertAlmostEqual(env.reagent_transfer_pipeline[2, 1], 20.0)

        env.step(env.noop_action())
        self.assertAlmostEqual(env.reagents[1], 0.0)
        env.step(env.noop_action())
        self.assertAlmostEqual(env.reagents[1], 0.0)
        _observation, _reward, _done, info = env.step(env.noop_action())
        self.assertAlmostEqual(info["reagent_transfer_arrivals"][1], 20.0)
        self.assertAlmostEqual(env.reagents[1], 20.0)

    def test_geographic_transfer_cost_increases_with_distance(self):
        base_config = dict(
            num_facilities=2,
            production_lead_time=2,
            episode_horizon=1,
            demand_rates=(0.0, 0.0),
            initial_specimens=(0.0, 0.0),
            initial_reagents=(20.0, 0.0),
            initial_idle_bioreactors=(0.0, 0.0),
            max_specimens=(50.0, 50.0),
            max_reagents=(50.0, 50.0),
            max_idle_bioreactors=(10.0, 10.0),
            max_reagent_replenishment=(0.0, 0.0),
            max_reagent_transfer=20.0,
            action_mode="facility_net",
            clinic_coordinates=((0.0, 0.0), (0.0, 10.0)),
            geographic_neighbor_k=1,
        )
        plain = CapacityPlanningEnv(CapacityPlanningConfig(**base_config), seed=12)
        geo = CapacityPlanningEnv(
            CapacityPlanningConfig(**base_config, geographic_transfer_cost_scale=1.0),
            seed=12,
        )
        action = plain.noop_action()
        action[2] = -1.0
        action[3] = 1.0

        _observation, _reward, _done, plain_info = plain.step(action)
        _observation, _reward, _done, geo_info = geo.step(action)

        self.assertGreater(float(geo_info["cost"]), float(plain_info["cost"]))

    def test_regional_supplier_disruption_uses_geographic_cluster(self):
        config = CapacityPlanningConfig(
            num_facilities=4,
            production_lead_time=2,
            episode_horizon=1,
            demand_rates=(0.0, 0.0, 0.0, 0.0),
            initial_specimens=(0.0, 0.0, 0.0, 0.0),
            initial_reagents=(0.0, 0.0, 0.0, 0.0),
            initial_idle_bioreactors=(0.0, 0.0, 0.0, 0.0),
            max_specimens=(10.0, 10.0, 10.0, 10.0),
            max_reagents=(10.0, 10.0, 10.0, 10.0),
            max_idle_bioreactors=(5.0, 5.0, 5.0, 5.0),
            max_reagent_replenishment=(10.0, 10.0, 10.0, 10.0),
            action_mode="facility_net",
            include_supplier_state=True,
            supplier_disruption_rate=0.0,
            clinic_coordinates=((0.0, 0.0), (0.0, 1.0), (20.0, 20.0), (20.0, 21.0)),
            geographic_neighbor_k=1,
            regional_supplier_disruption_probability=1.0,
            regional_supplier_disruption_duration=2,
            regional_supplier_disruption_cluster_size=2,
        )
        env = CapacityPlanningEnv(config, seed=4)
        unavailable = np.where(env.supplier_available == 0.0)[0]

        self.assertEqual(unavailable.size, 2)
        self.assertIn(tuple(unavailable), {tuple((0, 1)), tuple((2, 3))})

    def test_demand_shock_updates_cluster_multiplier(self):
        config = CapacityPlanningConfig(
            num_facilities=4,
            production_lead_time=2,
            episode_horizon=2,
            demand_rates=(1.0, 1.0, 1.0, 1.0),
            initial_specimens=(0.0, 0.0, 0.0, 0.0),
            initial_reagents=(0.0, 0.0, 0.0, 0.0),
            initial_idle_bioreactors=(0.0, 0.0, 0.0, 0.0),
            max_specimens=(10.0, 10.0, 10.0, 10.0),
            max_reagents=(10.0, 10.0, 10.0, 10.0),
            max_idle_bioreactors=(5.0, 5.0, 5.0, 5.0),
            max_reagent_replenishment=(0.0, 0.0, 0.0, 0.0),
            action_mode="facility_net",
            demand_shock_probability=1.0,
            demand_shock_multiplier=3.0,
            demand_shock_duration=2,
            demand_shock_cluster_size=2,
        )
        env = CapacityPlanningEnv(config, seed=21)

        env.step(env.noop_action())

        self.assertEqual(int(np.count_nonzero(env.demand_rate_multiplier == 3.0)), 2)
        self.assertEqual(int(np.count_nonzero(env.demand_shock_remaining > 0)), 2)

    def test_demand_forecast_state_tracks_effective_demand_rate(self):
        config = CapacityPlanningConfig(
            num_facilities=4,
            production_lead_time=2,
            episode_horizon=2,
            demand_rates=(1.0, 1.0, 1.0, 1.0),
            initial_specimens=(0.0, 0.0, 0.0, 0.0),
            initial_reagents=(0.0, 0.0, 0.0, 0.0),
            initial_idle_bioreactors=(0.0, 0.0, 0.0, 0.0),
            max_specimens=(10.0, 10.0, 10.0, 10.0),
            max_reagents=(10.0, 10.0, 10.0, 10.0),
            max_idle_bioreactors=(5.0, 5.0, 5.0, 5.0),
            max_reagent_replenishment=(0.0, 0.0, 0.0, 0.0),
            action_mode="facility_net",
            include_supplier_state=True,
            include_demand_forecast_state=True,
            demand_forecast_horizon=2,
            demand_forecast_error=0.0,
            demand_shock_probability=1.0,
            demand_shock_multiplier=3.0,
            demand_shock_duration=2,
            demand_shock_cluster_size=2,
        )
        env = CapacityPlanningEnv(config, seed=22)

        observation = env.reset(seed=22)
        self.assertEqual(env.observation_size, 4 * 7)
        self.assertEqual(observation.shape, (env.observation_size,))
        np.testing.assert_allclose(env.demand_forecast, np.full(4, 2.0))

        next_observation, _reward, _done, info = env.step(env.noop_action())

        self.assertEqual(next_observation.shape, (env.observation_size,))
        self.assertEqual(env.graph_observation()["node_features"].shape, (4, 7))
        self.assertTrue(np.any(env.demand_forecast == 6.0))
        self.assertTrue(np.any(env.demand_forecast == 2.0))
        np.testing.assert_allclose(info["demand_forecast"], np.full(4, 2.0))


if __name__ == "__main__":
    unittest.main()
