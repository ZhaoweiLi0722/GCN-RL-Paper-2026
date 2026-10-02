"""Artificial tensor forward/gradient checks only; no models or scientific calls."""

from dataclasses import fields, replace
import math
import unittest

import torch

from src.rl.paired_cohort_objective import PairedCohortState, paired_cohort_objective


class PairedCohortObjectiveTests(unittest.TestCase):
    def state(self, logits=(0., 0.), costs=((10., 6.), (14., 8.)), *, reference=0,
              dtype=torch.float64, label_grad=False):
        return PairedCohortState(
            logits=torch.tensor(logits, dtype=dtype, requires_grad=True),
            raw_costs=torch.tensor(costs, dtype=dtype, requires_grad=label_grad),
            class_keys=tuple((k, 0.) for k in range(len(logits))),
            replication_seed_ids=tuple(tuple(range(len(costs))) for _ in logits),
            reference_index=reference,
        )

    def objective(self, *states, **settings):
        return paired_cohort_objective(list(states or (self.state(),)),
                                       **(dict(cost_scale=2.) | settings))

    def test_hand_computed_loss_diagnostics_and_gradient_sign(self):
        state = self.state(label_grad=True)
        result = self.objective(state)
        diagnostic = result.states[0]
        self.assertAlmostEqual(result.loss.item(), -1.25)
        self.assertEqual(diagnostic.raw_mean_costs.tolist(), [12., 7.])
        self.assertEqual(diagnostic.raw_cost_advantages.tolist(), [0., -5.])
        self.assertEqual(diagnostic.scaled_cost_advantages.tolist(), [0., -2.5])
        self.assertEqual(diagnostic.objective_costs.tolist(), [0., -2.5])
        self.assertEqual(diagnostic.probabilities.tolist(), [.5, .5])
        self.assertAlmostEqual(diagnostic.expected_raw_cost.item(), 9.5)
        self.assertAlmostEqual(diagnostic.entropy.item(), math.log(2.))
        result.loss.backward()
        self.assertTrue(torch.isfinite(state.logits.grad).all())
        self.assertEqual(state.logits.grad.tolist(), [.625, -.625])
        self.assertIsNone(state.raw_costs.grad)

    def test_all_candidates_contribute_with_no_entropy_bonus(self):
        state = self.state(logits=(math.log(.2), math.log(.3), math.log(.5)),
                           costs=((12., 4., 8.),), reference=2)
        result = self.objective(state)
        # p dot ([12, 4, 8] - 8)/2 = -.2, without any entropy term.
        self.assertAlmostEqual(result.loss.item(), -.2)
        gradient, = torch.autograd.grad(result.loss, state.logits)
        expected = torch.tensor([.44, -.54, .1], dtype=torch.float64)
        torch.testing.assert_close(gradient, expected)

    def test_float64_cost_differences_reach_float32_actor_logits(self):
        actor_logits = torch.zeros(2, dtype=torch.float32, requires_grad=True)
        state = self.state(costs=((1e9, 1e9 + 1.),), reference=0)
        state = replace(state, logits=actor_logits.to(torch.float64),
                        replication_seed_ids=((0,), (0,)))
        out = self.objective(state, cost_scale=1e9)
        out.loss.backward()
        self.assertEqual(out.states[0].raw_cost_advantages.tolist(), [0., 1.])
        self.assertEqual(actor_logits.grad.dtype, torch.float32)
        torch.testing.assert_close(actor_logits.grad,
                                   torch.tensor([-2.5e-10, 2.5e-10]), rtol=1e-6, atol=0.)

    def test_nonuniform_gradient_matches_finite_differences(self):
        state = self.state(logits=(-.4, .8, .1), costs=((8., 2., 5.), (6., 4., 9.)))
        result = self.objective(state)
        gradient, = torch.autograd.grad(result.loss, state.logits)
        for k in range(3):
            losses = []
            for delta in (-1e-6, 1e-6):
                logits = state.logits.detach().clone()
                logits[k] += delta
                losses.append(self.objective(replace(state, logits=logits)).loss.item())
            self.assertAlmostEqual(gradient[k].item(), (losses[1] - losses[0]) / 2e-6, places=8)

    def test_baseline_invariance_to_same_per_state_raw_cost_offset(self):
        states = [self.state(), self.state(logits=(.2, -.8, .6), costs=((2., 4., 8.),), reference=1)]
        offsets = [128., -64.]
        shifted = [replace(s, logits=s.logits.detach().clone().requires_grad_(), raw_costs=s.raw_costs + offset)
                   for s, offset in zip(states, offsets)]
        original, changed = self.objective(*states), self.objective(*shifted)
        torch.testing.assert_close(original.loss, changed.loss)
        original.loss.backward()
        changed.loss.backward()
        for before, after, old, new, offset in zip(states, shifted, original.states, changed.states, offsets):
            torch.testing.assert_close(before.logits.grad, after.logits.grad)
            torch.testing.assert_close(old.raw_cost_advantages, new.raw_cost_advantages)
            torch.testing.assert_close(new.raw_mean_costs, old.raw_mean_costs + offset)

    def test_optional_reference_baseline_changes_value_not_gradient(self):
        for baseline in (True, False):
            for reference in (0, 1, 2):
                state = self.state(logits=(-.1, .4, .8), costs=((10., 4., 8.),), reference=reference)
                out = self.objective(state, subtract_reference_baseline=baseline)
                gradient, = torch.autograd.grad(out.loss, state.logits)
                diagnostic = out.states[0]
                expected = diagnostic.expected_raw_cost / 2.
                if baseline:
                    expected -= diagnostic.raw_mean_costs[reference] / 2.
                torch.testing.assert_close(out.loss, expected)
                torch.testing.assert_close(diagnostic.raw_cost_advantages,
                                           diagnostic.raw_mean_costs - diagnostic.raw_mean_costs[reference])
                if baseline and reference == 0:
                    original_gradient = gradient
                torch.testing.assert_close(gradient, original_gradient)

    def test_candidate_permutation_equivariance_including_reference(self):
        state = self.state(logits=(.2, -.8, .6), costs=((2., 8., 4.), (6., 2., 10.)), reference=2)
        order = [2, 0, 1]
        permuted = replace(state, logits=state.logits.detach()[order].requires_grad_(),
                           raw_costs=state.raw_costs[:, order],
                           class_keys=tuple(state.class_keys[k] for k in order),
                           replication_seed_ids=tuple(state.replication_seed_ids[k] for k in order),
                           reference_index=order.index(state.reference_index))
        first, second = self.objective(state), self.objective(permuted)
        torch.testing.assert_close(first.loss, second.loss)
        first.loss.backward()
        second.loss.backward()
        torch.testing.assert_close(second.states[0].raw_mean_costs, first.states[0].raw_mean_costs[order])
        torch.testing.assert_close(second.states[0].scaled_cost_advantages,
                                   first.states[0].scaled_cost_advantages[order])
        torch.testing.assert_close(second.states[0].probabilities, first.states[0].probabilities[order])
        torch.testing.assert_close(second.states[0].entropy, first.states[0].entropy)
        torch.testing.assert_close(permuted.logits.grad, state.logits.grad[order])

    def test_ragged_states_are_uniform_not_candidate_or_replication_weighted(self):
        one = self.state(costs=((8., 4.),))
        two = self.state(logits=(0., 0., 0.), costs=((12., 6., 0.),) * 4)
        solo = [self.objective(one), self.objective(two)]
        batch = self.objective(one, two)
        self.assertAlmostEqual(batch.loss.item(), -2.)  # mean(-1, -3)
        for state, individual in zip((one, two), solo):
            single_gradient, = torch.autograd.grad(individual.loss, state.logits)
            batch_gradient, = torch.autograd.grad(batch.loss, state.logits, retain_graph=True)
            torch.testing.assert_close(batch_gradient, single_gradient / 2.)
        # Repeating identical artificial label rows with distinct IDs cannot upweight a state.
        repeated = replace(one, raw_costs=one.raw_costs.repeat(7, 1),
                           replication_seed_ids=(tuple(range(7)),) * 2)
        torch.testing.assert_close(self.objective(repeated, two).loss, batch.loss)

    def test_equal_costs_have_zero_gradient_even_at_nonuniform_logits(self):
        for dtype in (torch.float32, torch.float64):
            for baseline in (True, False):
                state = self.state(logits=(-2., .5, 3.), costs=((4., 4., 4.), (8., 8., 8.)), dtype=dtype)
                out = self.objective(state, subtract_reference_baseline=baseline)
                out.loss.backward()
                torch.testing.assert_close(state.logits.grad, torch.zeros_like(state.logits))
                torch.testing.assert_close(out.states[0].raw_cost_advantages, torch.zeros_like(state.logits))
                if baseline:
                    self.assertEqual(out.loss.item(), 0.)

    def test_single_candidate_and_single_replication_are_valid(self):
        for baseline in (True, False):
            state = self.state(logits=(1.,), costs=((7.,),))
            out = self.objective(state, subtract_reference_baseline=baseline)
            self.assertEqual(out.loss.item(), 0. if baseline else 3.5)
            self.assertEqual(out.states[0].entropy.item(), 0.)
            self.assertEqual(out.states[0].probabilities.tolist(), [1.])
            out.loss.backward()
            self.assertEqual(state.logits.grad.tolist(), [0.])

    def test_scale_is_fixed_and_only_rescales_objective_gradient(self):
        state = self.state()
        first, second = self.objective(state, cost_scale=2.), self.objective(state, cost_scale=10.)
        g1, = torch.autograd.grad(first.loss, state.logits)
        g2, = torch.autograd.grad(second.loss, state.logits)
        torch.testing.assert_close(first.loss / 5., second.loss)
        torch.testing.assert_close(g1 / 5., g2)
        torch.testing.assert_close(first.states[0].raw_mean_costs, second.states[0].raw_mean_costs)
        torch.testing.assert_close(first.states[0].raw_cost_advantages, second.states[0].raw_cost_advantages)

    def test_labels_and_diagnostics_detached_without_mutating_original_data(self):
        state = self.state(label_grad=True)
        logits_before, costs_before = state.logits.detach().clone(), state.raw_costs.detach().clone()
        keys_before, seeds_before = state.class_keys, state.replication_seed_ids
        out = self.objective(state)
        # Diagnostic mutation must not alter labels or tensors needed by backward.
        for field in fields(out.states[0]):
            value = getattr(out.states[0], field.name)
            if isinstance(value, torch.Tensor):
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
                value.fill_(999.)
        out.loss.backward()
        self.assertEqual(state.logits.grad.tolist(), [.625, -.625])
        self.assertIsNone(state.raw_costs.grad)
        self.assertTrue(state.raw_costs.requires_grad)
        self.assertTrue(torch.equal(state.logits, logits_before))
        self.assertTrue(torch.equal(state.raw_costs, costs_before))
        self.assertEqual(state.class_keys, keys_before)
        self.assertEqual(state.replication_seed_ids, seeds_before)

    def test_nonleaf_label_graph_is_not_traversed(self):
        source = torch.tensor([[5., 3.], [7., 4.]], dtype=torch.float64, requires_grad=True)
        state = replace(self.state(), raw_costs=source * 2.)
        self.objective(state).loss.backward()
        self.assertIsNone(source.grad)
        self.assertIsNotNone(state.logits.grad)

    def test_distinct_metadata_describes_method_not_return_quality(self):
        result = self.objective()
        self.assertEqual(result.metadata, {
            "version": "paired-cohort-objective-v1",
            "method": "model_assisted_one_step_policy_improvement",
            "target_objective": "all_candidate_expected_raw_cohort_cost",
            "cost_scale": 2., "baseline": "reference_mean", "state_weighting": "uniform",
            "pairing_validation": "declared_ordered_replication_seed_ids_only",
        })
        self.assertEqual(self.objective(subtract_reference_baseline=False).metadata["baseline"], "none")

    def test_pairing_identifiers_only_check_declaration_not_common_streams(self):
        state = self.state()
        # These arbitrary labels carry no stream provenance. Matching declarations
        # pass, even if a producer had actually used different exogenous streams.
        declared = replace(state, replication_seed_ids=(("world-a", "world-b"),) * 2)
        self.assertTrue(torch.isfinite(self.objective(declared).loss))
        with self.assertRaisesRegex(ValueError, "same paired replication seeds"):
            self.objective(replace(declared, replication_seed_ids=(("world-a", "world-b"), ("world-a", "world-c"))))
        with self.assertRaisesRegex(ValueError, "same paired replication seeds"):
            self.objective(replace(state, replication_seed_ids=((0, 1), ("0", "1"))))

    def test_reject_missing_duplicate_nonfinite_or_mutable_class_keys(self):
        state = self.state()
        invalid = [None, (), ("one",), ("same", "same"), ((0, 0.), (0., 0)),
                   ("", "b"), ("   ", "b"), (None, "b"), (True, "b"),
                   (float("nan"), "b"), (float("inf"), "b"), ((0, float("nan")), "b"),
                   ((), "b"), ([0], "b"), ({"key": 0}, "b"), (torch.tensor(0), "b")]
        for keys in invalid:
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                self.objective(replace(state, class_keys=keys))

    def test_reject_missing_duplicate_nonfinite_or_unpaired_seed_identifiers(self):
        state = self.state()
        invalid = [None, (), ((0, 1),), ((0, 1), (0,)), ((0, 1), (1, 0)),
                   ((0, 1), (0, 2)), ((0, 0),) * 2, (("", "b"),) * 2,
                   (("   ", "b"),) * 2, ((None, 1),) * 2, ((True, 1),) * 2,
                   ((-1, 1),) * 2, ((0., 1),) * 2, ((float("nan"), 1),) * 2,
                   ((float("inf"), 1),) * 2, (([0], 1),) * 2, (0, 1)]
        for seeds in invalid:
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                self.objective(replace(state, replication_seed_ids=seeds))

    def test_reject_malformed_shapes_nonfinite_tensors_and_dtype_mismatch(self):
        state = self.state()
        invalid = {
            "logits": [None, [0., 0.], torch.tensor(0.), torch.zeros(2, 1), torch.empty(0),
                       torch.tensor([0, 1]), torch.zeros(2, dtype=torch.float16),
                       torch.tensor([0., float("nan")]), torch.tensor([0., float("inf")]),
                       torch.zeros(3, dtype=torch.float64)],
            "raw_costs": [None, [[1., 2.]], torch.tensor(1.), torch.zeros(2), torch.zeros(1, 2, 2),
                          torch.empty(0, 2), torch.empty(2, 0), torch.zeros(2, 3),
                          torch.ones(2, 2, dtype=torch.int64), torch.ones(2, 2, dtype=torch.float32),
                          torch.tensor([[1., float("nan")]], dtype=torch.float64),
                          torch.tensor([[1., float("-inf")]], dtype=torch.float64)],
        }
        for name, replacements in invalid.items():
            for value in replacements:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.objective(replace(state, **{name: value}))
        with self.assertRaisesRegex(ValueError, "all states must share"):
            self.objective(state, self.state(dtype=torch.float32))
        with self.assertRaises(ValueError):
            self.objective(replace(state, raw_costs=torch.ones(2, 2, device="meta", dtype=torch.float64)))

    def test_reject_missing_or_invalid_reference_index(self):
        state = self.state()
        for index in (None, -1, 2, True, 0., "0", torch.tensor(0)):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.objective(replace(state, reference_index=index))
        with self.assertRaises(TypeError):
            PairedCohortState(logits=state.logits, raw_costs=state.raw_costs,
                              class_keys=state.class_keys, replication_seed_ids=state.replication_seed_ids)

    def test_reject_invalid_container_scale_or_baseline(self):
        for states in (None, [], (), {}, [self.state(), None], [dict(logits=[0.])]):
            with self.subTest(states=states), self.assertRaises(ValueError):
                paired_cohort_objective(states, cost_scale=2.)
        for scale in (None, 0., -1., float("nan"), float("inf"), True, "1", [1.],
                      torch.tensor(1.), torch.tensor(1., requires_grad=True)):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.objective(cost_scale=scale)
        for baseline in (None, 0, 1, "true"):
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                self.objective(subtract_reference_baseline=baseline)
        with self.assertRaises(TypeError):
            paired_cohort_objective([self.state()])

    def test_reject_nonfinite_intermediate_arithmetic_without_clipping(self):
        for state, scale in (
            (self.state(costs=((1e308, 1e308), (1e308, 1e308))), 2.),
            (self.state(costs=((-1e308, 1e308),)), 2.),
            (self.state(), 1e-308),
            (self.state(logits=(-1e308, 1e308)), 2.),
            (self.state(dtype=torch.float32), 1e-300),
            (self.state(dtype=torch.float32), 1e300),
        ):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.objective(state, cost_scale=scale)

    def test_large_but_finite_logit_gap_has_finite_entropy_and_gradient(self):
        state = self.state(logits=(1000., -1000.))
        result = self.objective(state)
        result.loss.backward()
        self.assertTrue(torch.isfinite(result.states[0].entropy))
        self.assertTrue(torch.isfinite(state.logits.grad).all())

    def test_revalidates_tensors_after_record_construction(self):
        state = self.state()
        state.raw_costs[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "raw_costs must be finite"):
            self.objective(state)


if __name__ == "__main__":
    unittest.main()
