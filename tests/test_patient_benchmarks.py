"""Sanity checks for existing baselines on the patient env (Phase 5, group 1)."""

from __future__ import annotations

import unittest

import numpy as np

from src.baselines.heuristics import (
    facility_net_action_from_state,
    get_heuristic_class,
    heuristic_settings_for_policy,
)
from src.rl.config import load_config
from src.rl.experiment import build_env

DEV_CONFIG = "experiments/configs/2_clinic_patient_condition.json"
HEURISTICS = ("myo", "iso", "mdl1", "mdl2", "fmyo", "pmyo")


def _env(seed: int = 0):
    return build_env({"env": load_config(DEV_CONFIG)}, seed=seed)


def _episode_cost(policy, seed: int) -> float:
    env = _env(seed=seed)
    state = env.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        state, _reward, done, info = env.step(policy(env, state))
        total += float(info["cost"])
    return total


class ExistingHeuristicSanityTests(unittest.TestCase):
    def test_all_heuristics_emit_valid_actions(self) -> None:
        env = _env()
        state = env.reset(seed=0)
        for name in HEURISTICS:
            agent = get_heuristic_class(name)(env.observation_size, env.action_size, {})
            action = agent.select_action(state, explore=False, env=env)
            self.assertEqual(action.shape, (env.action_size,))
            self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))

    def test_facility_net_action_from_state_accepts_patient_summary(self) -> None:
        env_config = load_config(DEV_CONFIG)
        env = build_env({"env": env_config}, seed=0)
        state = env.reset(seed=0)

        for policy_name in ("mdl2", "pmyo"):
            action = facility_net_action_from_state(
                state,
                env_config,
                settings=heuristic_settings_for_policy(policy_name),
            )

            self.assertEqual(action.shape, (env.action_size,))
            self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))

    def test_heuristics_are_deterministic_under_seed(self) -> None:
        for name in HEURISTICS:
            agent = get_heuristic_class(name)
            c1 = _episode_cost(lambda e, s, a=agent: a(1, 1, {}).select_action(s, env=e), seed=5)
            c2 = _episode_cost(lambda e, s, a=agent: a(1, 1, {}).select_action(s, env=e), seed=5)
            self.assertEqual(c1, c2, f"{name} not deterministic")

    def test_heuristics_beat_random_policy(self) -> None:
        rng = np.random.default_rng(0)
        random_cost = _episode_cost(lambda e, s: rng.uniform(-1.0, 1.0, e.action_size), seed=7)
        for name in HEURISTICS:
            agent = get_heuristic_class(name)
            cost = _episode_cost(lambda e, s, a=agent: a(1, 1, {}).select_action(s, env=e), seed=7)
            self.assertLess(cost, random_cost, f"{name} did not beat random")

    def test_flat_ddpg_builds_and_steps_on_patient_env(self) -> None:
        try:
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")
        from src.baselines.flat_ddpg import FlatDDPGAgent

        env = _env()
        agent = FlatDDPGAgent(env.observation_size, env.action_size, {"seed": 0, "hidden_sizes": [32, 32]})
        state = env.reset(seed=0)
        action = agent.select_action(state, explore=True, env=env)
        self.assertEqual(action.shape, (env.action_size,))
        next_state, reward, _done, _info = env.step(action)
        self.assertEqual(next_state.shape, (env.observation_size,))
        self.assertIsInstance(reward, float)


if __name__ == "__main__":
    unittest.main()
