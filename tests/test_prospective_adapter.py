"""No-training N1/N2 acceptance harness using declared synthetic records."""

from dataclasses import replace
import unittest

import numpy as np

from src.models.matched_inputs import InputSchema, ObservationBatch
from src.rl.networks import torch
from src.rl.prospective_adapter import (
    ReplayInputContract, pack_actor_state, completed_segment_windows, prepare_replay_batch,
)
from src.rl.validated_returns import ReplaySemantics, OneStepRecord, bellman_targets


def contract(**changes):
    inputs = InputSchema("synthetic-integration-v1", ("A", "B"),
                         ("pipeline_0", "pipeline_1"), ("time",), ("support_A", "support_B"))
    settings = dict(reward_kind="absolute_environment", reward_definition_id="synthetic-reward-v1",
                    reward_scale=.1, gamma=.5, state_schema_id=inputs.definition_id + "/actor-flat",
                    action_schema_id=inputs.definition_id + "/action", state_dim=11, action_dim=2,
                    bootstrap_on_truncation=True)
    return ReplayInputContract(inputs, ReplaySemantics(**(settings | changes)))


def observed_state(index, binding):
    nodes = torch.tensor([[[index, 2.], [3., index + 4]]], dtype=torch.float64)
    globals_ = torch.tensor([[float(index)]], dtype=torch.float64)
    links = torch.tensor([[[0., index + 1.], [index + 1., 0.]]], dtype=torch.float64)
    anchor = torch.tensor([[.25 * index, -.25 * index]], dtype=torch.float64)
    return ObservationBatch(binding.inputs, nodes, globals_, links), anchor


def record(index, binding, **changes):
    obs, anchor = observed_state(index, binding)
    next_obs, next_anchor = observed_state(index + 1, binding)
    fields = dict(semantics=binding.replay, source_id="synthetic-source-v1", origin="trajectory",
                  trajectory_id="episode-A", step_index=index, state_token=f"state-{index}",
                  next_state_token=f"state-{index + 1}",
                  state=tuple(pack_actor_state(obs, anchor, binding)[0].tolist()),
                  action=(.125 * index, -.125 * index), raw_reward=float(index + 1),
                  next_state=tuple(pack_actor_state(next_obs, next_anchor, binding)[0].tolist()),
                  terminated=False, truncated=False)
    return OneStepRecord(**(fields | changes))


@unittest.skipIf(torch is None, "torch not available")
class ProspectiveAdapterTests(unittest.TestCase):
    def setUp(self):
        self.contract = contract()

    def prepare(self, windows, binding=None, **changes):
        settings = dict(dtype=torch.float64, device="cpu", message_mode="physical")
        return prepare_replay_batch(windows, binding or self.contract, **(settings | changes))

    def test_current_and_endpoint_states_anchors_and_actions_round_trip(self):
        records = [record(i, self.contract, terminated=(i == 3)) for i in range(4)]
        windows = completed_segment_windows(records, self.contract.replay, max_steps=3)
        batch = self.prepare(windows)
        self.assertEqual([s.n_steps for s in batch.samples], [3, 3, 2, 1])
        for i, window in enumerate(windows):
            torch.testing.assert_close(batch.current_actor.flat[i], torch.tensor(window[0].state, dtype=torch.float64))
            torch.testing.assert_close(batch.next_actor.flat[i], torch.tensor(window[-1].next_state, dtype=torch.float64))
            torch.testing.assert_close(batch.current_critic.context[i, -2:], torch.tensor(window[0].action, dtype=torch.float64))
            torch.testing.assert_close(batch.current_critic.flat[i, :-2], batch.current_actor.flat[i])
        self.assertFalse(torch.equal(batch.current_actor.context[:, -2:], batch.next_actor.context[:, -2:]))
        self.assertEqual(batch.bootstrap_discounts[:, 0].tolist(), [.125, 0., 0., 0.])

    def test_every_short_tail_is_retained_with_exact_discount_and_scale(self):
        records = [record(i, self.contract, truncated=(i == 3)) for i in range(4)]
        windows = completed_segment_windows(records, self.contract.replay, max_steps=4)
        batch = self.prepare(windows)
        self.assertEqual([s.first_step_index for s in batch.samples], [0, 1, 2, 3])
        self.assertEqual([s.n_steps for s in batch.samples], [4, 3, 2, 1])
        torch.testing.assert_close(batch.rewards[:, 0], torch.tensor([.325, .45, .5, .4], dtype=torch.float64))
        torch.testing.assert_close(batch.bootstrap_discounts[:, 0], torch.tensor([.0625, .125, .25, .5], dtype=torch.float64))
        torch.testing.assert_close(batch.targets(torch.full((4, 1), 8., dtype=torch.float64))[0],
                                   torch.tensor([.825], dtype=torch.float64))

    def test_tensor_targets_match_numpy_and_independent_backward_recursion(self):
        for dtype in (torch.float32, torch.float64):
            for gamma in (0., .5, 1.):
                for terminal, bootstrap in ((True, True), (False, True), (False, False)):
                    binding = contract(gamma=gamma, bootstrap_on_truncation=bootstrap)
                    records = [record(i, binding, raw_reward=(-1.) ** i * (i + 1),
                                      terminated=terminal and i == 4, truncated=not terminal and i == 4)
                               for i in range(5)]
                    windows = completed_segment_windows(records, binding.replay, max_steps=3)
                    batch = self.prepare(windows, binding, dtype=dtype)
                    q = torch.arange(1, 6, dtype=dtype).reshape(-1, 1)
                    actual = batch.targets(q)
                    reference = bellman_targets(batch.samples, q.numpy(), binding.replay)
                    expected = []
                    for i, window in enumerate(windows):
                        last = window[-1]
                        value = 0. if last.terminated or (last.truncated and not bootstrap) else float(i + 1)
                        for step in reversed(window):
                            value = step.raw_reward * binding.replay.reward_scale + gamma * value
                        expected.append(value)
                    np.testing.assert_allclose(actual.numpy()[:, 0], expected, rtol=1e-6, atol=1e-7)
                    np.testing.assert_allclose(actual.numpy(), reference, rtol=1e-6, atol=1e-7)

    def test_target_is_detached_and_not_given_an_extra_gamma(self):
        batch = self.prepare([[record(0, self.contract), record(1, self.contract)]])
        q = torch.tensor([[8.]], dtype=torch.float64, requires_grad=True)
        target = batch.targets(q)
        self.assertFalse(target.requires_grad)
        self.assertAlmostEqual(target.item(), .2 + .25 * 8.)
        self.assertNotAlmostEqual(target.item(), .2 + .5 * .25 * 8.)

    def test_neural_ablation_does_not_change_replay_or_targets(self):
        windows = [[record(0, self.contract)], [record(1, self.contract)]]
        graph, ablated = self.prepare(windows), self.prepare(windows, message_mode="self_only")
        for name in ("current_actor", "current_critic", "next_actor"):
            self.assertTrue(torch.equal(getattr(graph, name).flat, getattr(ablated, name).flat))
        self.assertEqual(graph.samples, ablated.samples)
        q = torch.ones(2, 1, dtype=torch.float64)
        self.assertTrue(torch.equal(graph.targets(q), ablated.targets(q)))

    def test_counterfactual_can_enter_only_as_one_step_window(self):
        cf = record(0, self.contract, origin="counterfactual", trajectory_id=None, step_index=None)
        batch = self.prepare([[cf], [cf]])
        self.assertEqual([s.n_steps for s in batch.samples], [1, 1])
        with self.assertRaisesRegex(ValueError, "counterfactual"):
            self.prepare([[cf, cf]])
        with self.assertRaisesRegex(ValueError, "counterfactual"):
            completed_segment_windows([cf], self.contract.replay, max_steps=1)

    def test_completed_segments_cannot_silently_drop_open_tails(self):
        with self.assertRaisesRegex(ValueError, "termination or truncation"):
            completed_segment_windows([record(0, self.contract)], self.contract.replay, max_steps=4)
        one = record(0, self.contract, terminated=True)
        self.assertEqual(completed_segment_windows([one], self.contract.replay, max_steps=4), ((one,),))
        for size in (0, -1, True):
            with self.assertRaises(ValueError):
                completed_segment_windows([one], self.contract.replay, max_steps=size)

    def test_one_step_mode_still_checks_full_segment_lineage_and_boundaries(self):
        for changes in ({"step_index": 7}, {"source_id": "other"}, {"trajectory_id": "other"},
                        {"state_token": "different-hidden-state"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                completed_segment_windows([record(0, self.contract), record(1, self.contract, truncated=True, **changes)],
                                          self.contract.replay, max_steps=1)
        with self.assertRaisesRegex(ValueError, "boundary"):
            completed_segment_windows([record(0, self.contract, terminated=True), record(1, self.contract, truncated=True)],
                                      self.contract.replay, max_steps=1)

    def test_schema_binding_and_dimensions_are_explicit(self):
        for change in ({"state_schema_id": "legacy-state"}, {"action_schema_id": "executed-not-requested"},
                       {"state_dim": 10}, {"action_dim": 3}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                contract(**change)
        observation, anchor = observed_state(0, self.contract)
        changed = replace(self.contract.inputs, action_names=("support_B", "support_A"))
        with self.assertRaisesRegex(ValueError, "schema"):
            pack_actor_state(replace(observation, schema=changed), anchor, self.contract)

    def test_legacy_arrays_and_bare_return_samples_are_not_migrated(self):
        legacy = {"state": [0.] * 11, "reward": 1., "done": True}
        with self.assertRaisesRegex(TypeError, "legacy"):
            self.prepare([[legacy]])
        with self.assertRaises(TypeError):
            self.prepare([self.prepare([[record(0, self.contract)]]).samples[0]])
        with self.assertRaisesRegex(TypeError, "ReplayInputContract"):
            prepare_replay_batch([], None, dtype=torch.float64, device="cpu", message_mode="physical")

    def test_reward_semantics_cannot_be_mixed_at_adapter_boundary(self):
        for changes in ({"reward_scale": 1.}, {"reward_kind": "anchor_relative"},
                        {"reward_definition_id": "another-objective"}, {"gamma": .9}):
            foreign = record(0, contract(**changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "mixed"):
                self.prepare([[foreign]])

    def test_target_shape_type_and_device_are_checked(self):
        batch = self.prepare([[record(0, self.contract)]])
        for q in (torch.ones(1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float32),
                  torch.full((1, 1), float("nan"), dtype=torch.float64),
                  torch.ones(1, 1, dtype=torch.float64, device="meta"), [[1.]]):
            with self.subTest(q=q), self.assertRaises((TypeError, ValueError)):
                batch.targets(q)

    def test_invalid_physical_metadata_survives_no_codec_shortcut(self):
        rec = record(0, self.contract)
        state = list(rec.state)
        state[5] = 1.  # First physical-link diagonal; actor layout has 4 node + 1 global slots.
        with self.assertRaisesRegex(ValueError, "self-loops"):
            self.prepare([[replace(rec, state=tuple(state))]])

    def test_empty_batches_bad_dtype_and_numeric_overflow_fail(self):
        with self.assertRaises(ValueError):
            self.prepare([])
        with self.assertRaises(ValueError):
            completed_segment_windows([], self.contract.replay, max_steps=4)
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.prepare([[record(0, self.contract)]], dtype=torch.int64)
        binding = contract(reward_scale=1., gamma=1.)
        huge = record(0, binding, raw_reward=1e308)
        with self.assertRaisesRegex(ValueError, "overflow"):
            self.prepare([[huge]], binding, dtype=torch.float32)
        batch = self.prepare([[huge]], binding)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            batch.targets(torch.tensor([[1e308]], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
