import unittest

import numpy as np

from src.rl.action_projection import project_action
from src.rl.config import load_config
from src.rl.experiment import build_env


class RLUtilsTest(unittest.TestCase):
    def test_project_action_clips_normalized_bounds(self):
        projected = project_action(np.array([-2.0, 0.25, 3.0]), action_space_info=3)

        np.testing.assert_allclose(projected.action, np.array([-1.0, 0.25, 1.0], dtype=np.float32))
        self.assertTrue(projected.clipped)

    def test_project_action_checks_shape(self):
        with self.assertRaises(ValueError):
            project_action(np.array([0.0, 0.0]), action_space_info=3)

    def test_json_compatible_yaml_config_loads(self):
        config = load_config("configs/flat_ddpg.yaml")

        self.assertEqual(config["algorithm"], "flat_ddpg")
        self.assertIn("env", config)

    def test_flat_state_no_graph_keeps_environment_action_space(self):
        config = load_config("configs/flat_ddpg.yaml")
        env = build_env(config, seed=0)

        self.assertEqual(env.action_size, 5)


if __name__ == "__main__":
    unittest.main()
