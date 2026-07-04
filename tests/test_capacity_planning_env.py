import unittest

import numpy as np

from src.env.capacity_planning import CapacityPlanningEnv, make_legacy_two_facility_config
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


if __name__ == "__main__":
    unittest.main()
