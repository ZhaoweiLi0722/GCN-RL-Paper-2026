import unittest
from types import SimpleNamespace

import numpy as np

from src.env.multi_scenario import EpisodeScenarioEnv


class _StubEnv:
    def __init__(
        self,
        scenario: str,
        *,
        observation_size: int = 3,
        action_size: int = 2,
    ) -> None:
        self.scenario_name = scenario
        self.graph_ablation = "full_graph"
        self.observation_size = observation_size
        self.action_size = action_size
        self.config = SimpleNamespace(
            episode_horizon=4,
            num_facilities=2,
        )
        self.last_seed = None

    def reset(self, seed=None):
        self.last_seed = seed
        return np.full(
            self.observation_size,
            float(seed or 0),
            dtype=np.float32,
        )

    def step(self, action):
        return self.reset(self.last_seed), -1.0, False, {
            "scenario": self.scenario_name
        }

    def enable_train_randomization(self, **settings):
        self.randomization = settings

    def _private_helper(self):
        return self.scenario_name


class EpisodeScenarioEnvTests(unittest.TestCase):
    def test_round_robin_switches_only_at_episode_reset(self) -> None:
        first = _StubEnv("first")
        second = _StubEnv("second")
        env = EpisodeScenarioEnv((first, second), start_index=1)

        state = env.reset(seed=10)
        self.assertEqual(env.scenario_name, "second")
        self.assertEqual(state.shape, (3,))
        env.step(np.zeros(2, dtype=np.float32))
        self.assertEqual(env.scenario_name, "second")

        env.reset(seed=11)
        self.assertEqual(env.scenario_name, "first")
        self.assertEqual(first.last_seed, 11)

    def test_scenario_name_is_not_added_to_observation(self) -> None:
        env = EpisodeScenarioEnv((_StubEnv("a"), _StubEnv("b")))

        state = env.reset(seed=12)

        self.assertEqual(state.shape, (env.observation_size,))
        self.assertEqual(env.observation_size, 3)
        self.assertEqual(env._private_helper(), "a")

    def test_rejects_incompatible_observation_layouts(self) -> None:
        with self.assertRaisesRegex(ValueError, "observation_size"):
            EpisodeScenarioEnv(
                (
                    _StubEnv("a", observation_size=3),
                    _StubEnv("b", observation_size=4),
                )
            )


if __name__ == "__main__":
    unittest.main()
