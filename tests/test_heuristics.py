"""Tests for deterministic heuristic benchmark policies."""

from __future__ import annotations

from dataclasses import asdict
import unittest

import numpy as np

from evaluation.evaluate_formal import evaluate_agent, summarize_rows
from src.baselines.heuristics import (
    facility_net_action_from_state,
    heuristic_settings_for_policy,
    IsolatedPolicy,
    MeanDemandLookahead1Policy,
    MeanDemandLookahead2Policy,
    MyopicPolicy,
)
from src.env.capacity_planning import CapacityPlanningEnv, make_20_clinic_config


class HeuristicPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = CapacityPlanningEnv(make_20_clinic_config(episode_horizon=4), seed=2)
        self.state = self.env.reset(seed=2)

    def test_all_heuristics_emit_valid_facility_net_actions(self) -> None:
        for policy_cls in (MyopicPolicy, IsolatedPolicy, MeanDemandLookahead1Policy, MeanDemandLookahead2Policy):
            with self.subTest(policy=policy_cls.__name__):
                policy = policy_cls()
                action = policy.select_action(self.state, env=self.env)
                self.assertEqual(action.shape, (self.env.action_size,))
                self.assertTrue(np.all(action >= -1.0))
                self.assertTrue(np.all(action <= 1.0))

    def test_iso_disables_transfer_actions(self) -> None:
        action = IsolatedPolicy().select_action(self.state, env=self.env)
        n = self.env.config.num_facilities
        np.testing.assert_allclose(action[: 3 * n], np.zeros(3 * n, dtype=np.float32))

    def test_state_based_helper_matches_live_policy(self) -> None:
        live_action = MeanDemandLookahead2Policy().select_action(self.state, env=self.env)
        state_action = facility_net_action_from_state(
            self.state,
            asdict(self.env.config),
            settings=heuristic_settings_for_policy("mdl2"),
        )

        np.testing.assert_allclose(state_action, live_action)

    def test_formal_evaluation_summarizes_heuristic_rows(self) -> None:
        policy = MyopicPolicy()
        rows = evaluate_agent(
            policy,
            self.env,
            algorithm="myo",
            seed=11,
            replications=2,
            max_steps=3,
        )
        summary = summarize_rows(rows)
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["algorithm"], "myo")
        self.assertEqual(summary["replications"], 2)
        self.assertIn("total_cost_mean", summary)


if __name__ == "__main__":
    unittest.main()
