"""Synthetic target-contract checks; no agents, environment or optimizer."""

from dataclasses import replace
import unittest

import numpy as np

from src.rl.validated_returns import ReplaySemantics, OneStepRecord, make_return, bellman_targets


def contract(**changes):
    values = dict(reward_kind="anchor_relative", reward_definition_id="synthetic-paired-v1",
                  reward_scale=0.1, gamma=0.5, state_schema_id="synthetic-state-v1",
                  action_schema_id="synthetic-request-v1", state_dim=2, action_dim=1,
                  bootstrap_on_truncation=True)
    return ReplaySemantics(**(values | changes))


def record(index=0, *, semantics=None, **changes):
    values = dict(semantics=semantics or contract(), source_id="synthetic-collector-v1",
                  origin="trajectory", trajectory_id="episode-A", step_index=index,
                  state_token=f"full-{index}", next_state_token=f"full-{index + 1}",
                  state=(index, 1), action=(0.1,), raw_reward=index + 1,
                  next_state=(index + 1, 1), terminated=False, truncated=False)
    return OneStepRecord(**(values | changes))


class ValidatedReturnTests(unittest.TestCase):
    def test_one_step_uses_raw_reward_scale_once(self):
        sample = make_return([record(raw_reward=2)], contract())
        self.assertEqual(sample.reward, 0.2)
        self.assertEqual(sample.bootstrap_discount, 0.5)
        np.testing.assert_allclose(bellman_targets([sample], np.array([[4]]), contract()), [[2.2]])

    def test_four_step_return_and_full_bootstrap_factor(self):
        sample = make_return([record(i) for i in range(4)], contract())
        self.assertEqual(sample.n_steps, 4)
        self.assertAlmostEqual(sample.reward, 0.325)
        self.assertAlmostEqual(sample.one_step_reward, 0.1)
        self.assertEqual(sample.bootstrap_discount, 0.0625)
        self.assertEqual(sample.state, (0, 1))
        self.assertEqual(sample.next_state, (4, 1))
        np.testing.assert_allclose(bellman_targets([sample], np.array([[8]]), contract()), [[0.825]])

    def test_terminal_masks_bootstrap_and_stops_window(self):
        sample = make_return([record(0), record(1, terminated=True)], contract())
        self.assertEqual(sample.bootstrap_discount, 0)
        np.testing.assert_allclose(bellman_targets([sample], np.array([[100]]), contract()), [[0.2]])
        with self.assertRaisesRegex(ValueError, "boundary"):
            make_return([record(0, terminated=True), record(1)], contract())

    def test_truncation_policy_is_explicit_and_does_not_join_reset(self):
        for bootstrap, expected in ((True, 0.25), (False, 0.0)):
            semantics = contract(bootstrap_on_truncation=bootstrap)
            sample = make_return([record(0, semantics=semantics), record(1, semantics=semantics, truncated=True)], semantics)
            self.assertEqual(sample.bootstrap_discount, expected)
        with self.assertRaisesRegex(ValueError, "boundary"):
            make_return([record(0, truncated=True), record(1)], contract())
        both = make_return([record(0, terminated=True, truncated=True)], contract())
        self.assertEqual(both.bootstrap_discount, 0)

    def test_short_tail_uses_its_actual_length(self):
        for length in (1, 2, 3):
            sample = make_return([record(i) for i in range(length)], contract())
            self.assertEqual(sample.bootstrap_discount, 0.5 ** length)

    def test_independent_counterfactuals_are_one_step_only(self):
        cf = record(origin="counterfactual", trajectory_id=None, step_index=None)
        sample = make_return([cf], contract())
        self.assertEqual(sample.n_steps, 1)
        for values in ([cf, cf], [record(), cf]):
            with self.assertRaisesRegex(ValueError, "counterfactual"):
                make_return(values, contract())
        with self.assertRaisesRegex(ValueError, "trajectory order"):
            record(origin="counterfactual")

    def test_mixed_semantics_rejected_even_with_equal_numeric_rewards(self):
        alternatives = [contract(reward_kind="absolute_environment"),
                        contract(reward_definition_id="another-anchor-or-objective"),
                        contract(reward_scale=1), contract(gamma=0.9),
                        contract(action_schema_id="executed-action-not-request"),
                        contract(state_schema_id="other-order"),
                        contract(bootstrap_on_truncation=False)]
        for different in alternatives:
            with self.subTest(different=different), self.assertRaisesRegex(ValueError, "mixed"):
                make_return([record(0), record(1, semantics=different)], contract())

    def test_observation_equality_does_not_replace_full_state_lineage(self):
        for changes in ({"state_token": "other-hidden-state"}, {"source_id": "another-source"},
                        {"trajectory_id": "episode-B"}, {"step_index": 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                make_return([record(0), record(1, **changes)], contract())

    def test_matching_tokens_do_not_override_broken_observations(self):
        with self.assertRaisesRegex(ValueError, "observed state"):
            make_return([record(0), record(1, state=(99, 1))], contract())

    def test_same_semantics_can_batch_separate_sources_but_not_join_them(self):
        first = make_return([record(source_id="collector-one")], contract())
        second = make_return([record(source_id="collector-two", trajectory_id="episode-B")], contract())
        targets = bellman_targets([first, second], np.ones((2, 1)), contract())
        np.testing.assert_allclose(targets, [[0.6], [0.6]])

    def test_invalid_records_are_not_silently_coerced(self):
        for changes in ({"raw_reward": float("nan")}, {"raw_reward": float("inf")},
                        {"raw_reward": True}, {"raw_reward": "1"},
                        {"state": (0,)}, {"action": (float("inf"),)},
                        {"terminated": 1}, {"truncated": None},
                        {"source_id": ""}, {"state_token": ""}, {"step_index": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                record(**changes)

    def test_input_arrays_are_copied_to_immutable_values(self):
        state = np.array([0, 1], dtype=np.float32)
        rec = record(state=state)
        state[:] = 99
        self.assertEqual(rec.state, (0, 1))

    def test_empty_and_legacy_records_require_explicit_metadata(self):
        with self.assertRaises(ValueError):
            make_return([], contract())
        with self.assertRaises(TypeError):
            make_return([{"state": [0, 1], "reward": 1}], contract())

    def test_invalid_semantic_settings_fail(self):
        for changes in ({"reward_kind": "guess"}, {"reward_definition_id": ""},
                        {"reward_scale": 0}, {"reward_scale": True}, {"gamma": 1.1},
                        {"gamma": float("nan")}, {"state_dim": 0}, {"action_dim": True},
                        {"bootstrap_on_truncation": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                contract(**changes)

    def test_gamma_extremes(self):
        for gamma, expected in ((0, 0.1), (1, 2.3)):
            semantics = contract(gamma=gamma)
            sample = make_return([record(0, semantics=semantics), record(1, semantics=semantics)], semantics)
            target = bellman_targets([sample], np.array([[2]]), semantics)
            np.testing.assert_allclose(target, [[expected]])

    def test_batch_rejects_broadcasting_nonfinite_values_and_mixed_targets(self):
        sample = make_return([record()], contract())
        for q in (np.array([1]), np.array([[1, 2]]), np.array([[float("nan")]]),
                  np.array([[True]]), np.array([["1"]])):
            with self.subTest(q=q), self.assertRaises(ValueError):
                bellman_targets([sample], q, contract())
        different = contract(reward_kind="absolute_environment")
        foreign = make_return([record(semantics=different)], different)
        with self.assertRaisesRegex(ValueError, "semantics"):
            bellman_targets([sample, foreign], np.ones((2, 1)), contract())

    def test_target_overflow_and_malformed_sample_fail(self):
        sample = make_return([record()], contract())
        for changed in (replace(sample, n_steps=0), replace(sample, terminated=1),
                        replace(sample, reward=float("inf")),
                        replace(sample, origin="unknown"),
                        replace(sample, origin="counterfactual", n_steps=2)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                bellman_targets([changed], np.ones((1, 1)), contract())
        semantics = contract(gamma=1, reward_scale=1)
        huge = make_return([record(semantics=semantics, raw_reward=1e308)], semantics)
        with self.assertRaisesRegex(ValueError, "finite"):
            bellman_targets([huge], np.array([[1e308]]), semantics)

    def test_window_accumulation_overflow_is_rejected(self):
        semantics = contract(gamma=1, reward_scale=1)
        with self.assertRaisesRegex(ValueError, "overflow"):
            make_return([record(i, semantics=semantics, raw_reward=1e308) for i in range(2)], semantics)

    def test_reference_agrees_with_independent_backward_recursion(self):
        for gamma in (0, 0.5, 0.99, 1):
            for length in range(1, 6):
                for terminal in (False, True):
                    semantics = contract(gamma=gamma)
                    records = [record(i, semantics=semantics, raw_reward=(-1) ** i * (i + 1),
                                      terminated=terminal and i == length - 1) for i in range(length)]
                    value = 0 if terminal else 7.0
                    for step in reversed(records):
                        value = step.raw_reward * semantics.reward_scale + gamma * value
                    sample = make_return(records, semantics)
                    actual = bellman_targets([sample], np.array([[7]]), semantics)[0, 0]
                    self.assertAlmostEqual(actual, value)


if __name__ == "__main__":
    unittest.main()
