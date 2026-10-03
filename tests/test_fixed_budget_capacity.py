"""Invented tensor/gradient checks only; no optimizer, environment or checkpoint."""

from dataclasses import replace
import itertools
import json
import math
import unittest
from unittest.mock import patch

from src.models.capacity_ddpg import CapacityActor, CapacityModelConfig, validate_hours
from src.models.fixed_budget_capacity import (
    FixedBudgetCapacityActor, NATIVE_SUM_TOLERANCE, native_fixed_budget_hours,
    project_fixed_budget_logits,
)
from src.rl.networks import torch


@unittest.skipIf(torch is None, "torch required")
class FixedBudgetCapacityTests(unittest.TestCase):
    def setUp(self):
        for name in ("torch.optim.Adam.step", "torch.optim.SGD.step", "torch.load"):
            barrier = patch(name, side_effect=AssertionError("no updates or checkpoint loads in these tests"))
            barrier.start()
            self.addCleanup(barrier.stop)
        self.config = CapacityModelConfig(3, (32, 32), (32, 1), (4.,) * 4, 8., 2., 4.)

    def actor(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            return FixedBudgetCapacityActor(self.config)

    def test_same_parameter_shapes_state_keys_and_explicit_new_semantics(self):
        actor = self.actor()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            old = CapacityActor(self.config)
        self.assertEqual(list(old.state_dict()), list(actor.state_dict()))
        self.assertEqual(sum(p.numel() for p in actor.parameters()), sum(p.numel() for p in old.parameters()))
        for key, value in actor.state_dict().items():
            self.assertTrue(torch.equal(value, old.state_dict()[key]))
        metadata = json.loads(json.dumps(actor.metadata()))
        self.assertEqual(metadata["format"], "fixed-budget-capacity-logits-v1")
        self.assertEqual(metadata["forward_output"], "unclipped_logits")
        self.assertFalse(metadata["legacy_raw_hours_state_semantics_compatible"])
        self.assertFalse(metadata["common_logit_shift_invariant"])
        self.assertFalse(metadata["full_capped_simplex"])
        self.assertEqual(metadata["site_hour_bounds"], [.5, 3.5])

    def test_constructor_performs_no_forward_and_zero_affine_is_exact_fixed_start(self):
        with patch.object(CapacityActor, "forward", side_effect=AssertionError("initialization forward forbidden")):
            actor = self.actor()
        with torch.no_grad():
            actor.head[-1].weight.zero_()
            actor.head[-1].bias.zero_()
        for features in (torch.zeros(2, 4, 3), torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)):
            logits = actor(features)
            self.assertTrue(torch.equal(logits, torch.zeros(2, 4)))
            self.assertTrue(torch.equal(actor.project(logits), torch.full((2, 4), 2.)))
            self.assertEqual(actor.project_native(logits[0]), (2.,) * 4)

    def test_forward_is_unclipped_logits_without_offset_and_has_finite_gradients(self):
        actor = self.actor()
        features = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3) / 12
        for bias in (-9., 9.):
            with torch.no_grad():
                actor.head[-1].weight.zero_()
                actor.head[-1].bias.fill_(bias)
            self.assertTrue(torch.equal(actor(features), torch.full((1, 4), bias)))
        actor = self.actor()
        logits = actor(features)
        weights = torch.tensor([[1., -1., 2., -2.]])
        (actor.project(logits) * weights).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in actor.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in actor.parameters()), 0.)

    def test_map_matches_double_formula_and_existing_float32_admissibility(self):
        logits = torch.tensor([[0., .4, -1., 2.], [-3., 7., .1, -.2]], dtype=torch.float32)
        u = torch.tanh(logits.double())
        expected = (2 + u - u.mean(-1, keepdim=True)).float()
        result = project_fixed_budget_logits(logits)
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(((result >= .5) & (result <= 3.5)).all().item())
        torch.testing.assert_close(result.sum(-1), torch.full((2,), 8.), atol=4 * torch.finfo(torch.float32).eps * 8, rtol=0)
        validate_hours(result, self.config, batch_size=2)

    def test_projection_is_permutation_equivariant_not_shift_invariant(self):
        logits = torch.tensor([-.9, -.2, .4, 1.6])
        hours = project_fixed_budget_logits(logits)
        native = native_fixed_budget_hours(logits)
        for order in itertools.permutations(range(4)):
            indices = list(order)
            torch.testing.assert_close(project_fixed_budget_logits(logits[indices]), hours[indices], atol=5e-7, rtol=0)
            permuted_native = native_fixed_budget_hours(logits[indices])
            for actual, expected in zip(permuted_native, (native[i] for i in indices)):
                self.assertAlmostEqual(actual, expected, delta=1e-12)
        self.assertFalse(torch.allclose(project_fixed_budget_logits(logits + 1), hours))

    def test_actor_respects_ring_site_rotation(self):
        actor = self.actor()
        features = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 8
        hours = actor.project(actor(features))
        for offset in (1, 2, 3):
            rotated = actor.project(actor(features.roll(offset, dims=1)))
            torch.testing.assert_close(rotated, hours.roll(offset, dims=1), atol=5e-7, rtol=0)

    def test_zero_sum_tangent_and_common_output_mode_gradients(self):
        for initial in ([0.] * 4, [-.8, -.2, .3, 1.1]):
            logits = torch.tensor(initial, requires_grad=True)
            jacobian = torch.autograd.functional.jacobian(project_fixed_budget_logits, logits)
            torch.testing.assert_close(jacobian.sum(0), torch.zeros(4), atol=6e-8, rtol=0)
            gradient, = torch.autograd.grad(project_fixed_budget_logits(logits).sum(), logits)
            self.assertTrue(torch.equal(gradient, torch.zeros(4)))
            _, tangent = torch.autograd.functional.jvp(project_fixed_budget_logits, logits, torch.tensor([.3, -.2, .7, 1.]))
            self.assertAlmostEqual(float(tangent.sum()), 0., delta=1e-7)
            self.assertGreater(float(jacobian.abs().sum()), 0.)
        zero = torch.zeros(4, requires_grad=True)
        objective = (project_fixed_budget_logits(zero) * torch.tensor([1., -1., 2., -2.])).sum()
        gradient, = torch.autograd.grad(objective, zero)
        self.assertTrue(torch.equal(gradient, torch.tensor([1., -1., 2., -2.])))
        self.assertEqual(float(gradient.sum()), 0.)

    def test_native_closure_exact_strict_budget_with_extreme_and_varied_logits(self):
        maximum = torch.finfo(torch.float32).max
        invented = [[0.] * 4, [maximum] * 4, [-maximum] * 4,
                    [-maximum, maximum, maximum, maximum], [maximum, -maximum, -maximum, -maximum]]
        invented.extend(itertools.product((-20., -1., -.00001, .3, 3., 20.), repeat=4))
        # Fixed arithmetic fixture, not sampled worlds or training data.
        invented.extend([[math.sin(i * 1.17 + j) * 8 for j in range(4)] for i in range(256)])
        for values in invented:
            logits = torch.tensor(values, dtype=torch.float32)
            hours = native_fixed_budget_hours(logits)
            self.assertTrue(all(type(x) is float and .5 <= x <= 3.5 for x in hours))
            self.assertEqual(hours[-1], 8 - math.fsum(hours[:3]))
            self.assertEqual(math.fsum(hours), 8.)
            self.assertLessEqual(sum(hours), 8.)
            self.assertLessEqual(8 - sum(hours), 1e-12)
            self.assertLessEqual(8 - math.fsum(hours), NATIVE_SUM_TOLERANCE)
            torch.testing.assert_close(torch.tensor(hours), project_fixed_budget_logits(logits), atol=3e-7, rtol=0)
        self.assertEqual(native_fixed_budget_hours(torch.tensor([-maximum, maximum, maximum, maximum])), (.5, 2.5, 2.5, 2.5))

    def test_noise_is_supplied_in_logit_space_and_no_forward_or_rng_is_used(self):
        actor = self.actor()
        logits = torch.tensor([-.5, .1, .8, 1.3])
        noise = torch.tensor([.25, -.125, -.25, .125])
        state = torch.get_rng_state().clone()
        values = torch.tanh(logits.double() + noise.double())
        expected = (2 + values - values.mean()).float()
        with patch.object(actor, "forward", side_effect=AssertionError("extra forward forbidden")):
            actual = actor.project_noisy(logits, noise)
            native = actor.project_native(logits, noise=noise)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.allclose(actual, actor.project(logits) + noise))
        torch.testing.assert_close(torch.tensor(native), actual, atol=3e-7, rtol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        huge = torch.full((4,), torch.finfo(torch.float32).max)
        self.assertTrue(torch.equal(actor.project_noisy(huge, huge), torch.full((4,), 2.)))

    def test_projection_does_not_mutate_logits_noise_or_model_parameters(self):
        actor = self.actor()
        parameters = {name: value.clone() for name, value in actor.state_dict().items()}
        logits = torch.tensor([.1, -.7, 1., .4], requires_grad=True)
        noise = torch.tensor([.25, -.25, .125, -.125])
        before = logits.detach().clone(), noise.clone()
        actor.project(logits).sum().backward()
        actor.project_noisy(logits, noise)
        actor.project_native(logits, noise=noise)
        self.assertTrue(torch.equal(logits, before[0]))
        self.assertTrue(torch.equal(noise, before[1]))
        self.assertTrue(all(torch.equal(value, parameters[name]) for name, value in actor.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in actor.parameters()))

    def test_invalid_logits_noise_features_and_configs_are_rejected(self):
        invalid = [torch.tensor([0., 1., 2., float("nan")]), torch.tensor([0., 1., 2., float("inf")]),
                   torch.zeros(3), torch.zeros(1, 2, 4), torch.zeros(0, 4), torch.zeros(4, dtype=torch.float64),
                   torch.zeros(4, dtype=torch.int64), [0.] * 4]
        for value in invalid:
            with self.subTest(value=value):
                for function in (project_fixed_budget_logits, native_fixed_budget_hours):
                    with self.assertRaises(ValueError):
                        function(value)
        with self.assertRaises(ValueError):
            native_fixed_budget_hours(torch.zeros(1, 4))
        for noise in (torch.zeros(1, 4), torch.full((4,), float("nan")), torch.zeros(4, dtype=torch.float64)):
            with self.assertRaises(ValueError):
                project_fixed_budget_logits(torch.zeros(4), noise=noise)
        actor = self.actor()
        for features in (torch.zeros(4, 3), torch.zeros(1, 3, 3), torch.zeros(1, 4, 3, dtype=torch.float64), torch.full((1, 4, 3), float("inf"))):
            with self.assertRaises(ValueError):
                actor(features)
        for config in (replace(self.config, shared_hour_budget=7.), replace(self.config, raw_hour_offset=1.),
                       replace(self.config, site_hour_caps=(4.,) * 3)):
            with self.assertRaises(ValueError):
                FixedBudgetCapacityActor(config)


if __name__ == "__main__":
    unittest.main()
