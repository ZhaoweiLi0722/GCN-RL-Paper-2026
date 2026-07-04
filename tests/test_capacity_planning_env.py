import unittest

import numpy as np

from src.env.capacity_planning import (
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


if __name__ == "__main__":
    unittest.main()
