"""Synthetic arithmetic only: no env, historical checkpoint, model or optimizer."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.rl.validated_returns import OneStepRecord, ReplaySemantics, bellman_targets, make_return


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/fixed_window_value_contract_20260929.json"


def semantics(**changes):
    args = dict(reward_kind="absolute_environment", reward_definition_id="synthetic-window52-v1",
                reward_scale=1e-9, gamma=1., state_schema_id="synthetic-cost-and-time-v1",
                action_schema_id="synthetic-action-v1", state_dim=2, action_dim=1,
                bootstrap_on_truncation=True)
    return ReplaySemantics(**(args | changes))


def records(costs, contract):
    assert len(costs) == 52
    return [OneStepRecord(contract, "synthetic-source", "trajectory", "episode-A", t,
                          f"state-{t}", f"state-{t+1}", (0., t / 52), (0.,), -cost,
                          (0., (t + 1) / 52), t == 51, False)
            for t, cost in enumerate(costs)]


class FixedWindowObjectiveTest(unittest.TestCase):
    def test_one_step_targets_match_full_undiscounted_return(self):
        costs = [float((t % 7) + 1) * 1e6 for t in range(52)]
        contract = semantics()
        path = records(costs, contract)
        # Independent suffix sums are the exact continuation values in this fixture.
        future = np.array([[-sum(costs[t + 1:]) * 1e-9] for t in range(52)])
        targets = bellman_targets([make_return([r], contract) for r in path], future, contract)
        expected = np.array([[-sum(costs[t:]) * 1e-9] for t in range(52)])
        np.testing.assert_allclose(targets, expected, rtol=1e-14, atol=1e-14)
        self.assertAlmostEqual(make_return(path, contract).reward, float(expected[0, 0]))

    def test_late_and_early_cost_have_same_objective_weight(self):
        c = semantics()
        early, late = [0.] * 52, [0.] * 52
        early[0], late[-1] = 1e6, 1e6
        self.assertEqual(make_return(records(early, c), c).reward,
                         make_return(records(late, c), c).reward)

    def test_step_51_masks_any_finite_continuation(self):
        c = semantics()
        last = make_return([records([1e6] * 52, c)[-1]], c)
        for next_q in (-1e12, 0., 1e12):
            self.assertEqual(float(bellman_targets([last], np.array([[next_q]]), c)[0, 0]), -.001)
        self.assertEqual(last.next_state[-1], 1.)

    def test_collection_cutoff_is_not_objective_terminal(self):
        c = semantics()
        cut = replace(records([1e6] * 52, c)[12], truncated=True)
        target = bellman_targets([make_return([cut], c)], np.array([[-.039]]), c)
        self.assertAlmostEqual(float(target[0, 0]), -.040)
        self.assertFalse(cut.terminated)

    def test_endpoint_return_cannot_join_a_reset(self):
        c = semantics()
        path = records([1.] * 52, c)
        with self.assertRaisesRegex(ValueError, "boundary"):
            make_return([path[-1], path[0]], c)

    def test_independent_counterfactuals_are_not_a_trajectory(self):
        c = semantics()
        r = replace(records([1.] * 52, c)[0], origin="counterfactual", trajectory_id=None, step_index=None)
        self.assertEqual(make_return([r], c).n_steps, 1)
        with self.assertRaisesRegex(ValueError, "counterfactual"):
            make_return([r, r], c)

    def test_different_continuations_can_reverse_ranking(self):
        # Invented two-step costs; same immediate costs, different later policies.
        first_cost = np.array([0., 0.])
        mdl2_future_cost = np.array([10., 1.])
        frozen_future_cost = np.array([2., 8.])
        self.assertEqual(int(np.argmin(first_cost + mdl2_future_cost)), 1)
        self.assertEqual(int(np.argmin(first_cost + frozen_future_cost)), 0)

    def test_definition_ids_prevent_silent_target_mixing(self):
        a, b = semantics(reward_definition_id="mc-under-mdl2"), semantics(reward_definition_id="mc-under-frozen")
        sample = make_return([records([1.] * 52, a)[0]], a)
        with self.assertRaisesRegex(ValueError, "semantics"):
            bellman_targets([sample], np.zeros((1, 1)), b)

    def test_proposal_budget_arithmetic(self):
        config = json.loads(CONFIG.read_text())
        p, horizon = config["proposal"], config["objective"]["horizon"]
        actors, steps = len(p["actor_seeds"]), p["decision_steps"]
        draws, actions = p["discovery_draws"] + p["validation_draws"], p["maximum_unique_actions"]
        self.assertEqual(len(steps), len(set(steps)))
        self.assertTrue(all(0 <= t < horizon for t in steps))
        self.assertEqual(actors * len(steps) * actions * draws, p["maximum_continuation_records"])
        self.assertEqual(actors * horizon + actors * actions * draws * sum(horizon - t for t in steps),
                         p["maximum_environment_transitions"])
        self.assertEqual(actors + actors * len(steps) * draws, p["maximum_distinct_seed_starts"])

    def test_draft_does_not_authorize_execution_or_change_costs(self):
        config = json.loads(CONFIG.read_text())
        self.assertIs(config["scientific_execution_authorized"], False)
        self.assertIs(config["proposal"]["training_escalation_authorized"], False)
        self.assertEqual(config["proposal"]["critic_or_actor_fit_updates"], 0)
        self.assertIs(config["objective"]["cost_weights_changed"], False)
        self.assertIs(config["objective"]["full_settlement_claimed"], False)
        self.assertTrue(all(v is None for v in config["required_before_execution"].values()))

    def test_reviewed_sources_remain_unchanged(self):
        config = json.loads(CONFIG.read_text())
        for path, digest in config["source_locks"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)


if __name__ == "__main__":
    unittest.main()
