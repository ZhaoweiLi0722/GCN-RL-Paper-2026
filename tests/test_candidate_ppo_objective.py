"""Hand-computed objective tests; no environment or optimizer steps."""

import math
import unittest

import numpy as np

from src.rl.candidate_ppo_objective import candidate_ppo_loss, normalize_rollout_advantages
from src.rl.candidate_rollout import evaluate_candidate_policy, reevaluate_candidate_decision, sample_candidate
from src.rl.networks import torch
from tests.test_candidate_policy_rollout import candidates, contract, observation, policy


@unittest.skipIf(torch is None, "torch unavailable")
class CandidatePPOObjectiveTests(unittest.TestCase):
    def arrays(self, *, grad=False):
        def tensor(value):
            return torch.tensor(value, dtype=torch.float64, requires_grad=grad)
        return dict(log_probs=tensor([math.log(.3), math.log(.1)]),
                    old_log_probs=tensor([math.log(.2), math.log(.2)]),
                    values=tensor([1., 2.]), entropies=tensor([.2, .4]),
                    advantages=tensor([2., -2.]), returns=tensor([0., 4.]))

    def loss(self, inputs=None, **settings):
        return candidate_ppo_loss(**(self.arrays() if inputs is None else inputs),
                                  **(dict(clip_ratio=.2, value_loss_coef=.5, entropy_coef=.1) | settings))

    def test_hand_calculated_positive_and_negative_advantage_clipping(self):
        found = self.loss()
        # min(1.5*2,1.2*2)=2.4; min(.5*(-2),.8*(-2))=-1.6.
        self.assertAlmostEqual(found.policy.item(), -.4)
        self.assertAlmostEqual(found.value.item(), 2.5)
        self.assertAlmostEqual(found.entropy.item(), .3)
        self.assertAlmostEqual(found.total.item(), .82)
        self.assertEqual(found.clip_fraction.item(), 1.)
        torch.testing.assert_close(found.ratios, torch.tensor([1.5, .5], dtype=torch.float64))

    def test_clipped_policy_branches_have_zero_gradient(self):
        x = self.arrays(grad=True)
        self.loss(x).total.backward()
        torch.testing.assert_close(x["log_probs"].grad, torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(x["values"].grad, torch.tensor([.5, -1.], dtype=torch.float64))
        torch.testing.assert_close(x["entropies"].grad, torch.tensor([-.05, -.05], dtype=torch.float64))
        for key in ("old_log_probs", "advantages", "returns"):
            self.assertIsNone(x[key].grad)

    def test_unfavorable_directions_are_not_clipped_away(self):
        x = self.arrays(grad=True)
        x["advantages"] = torch.tensor([-2., 2.], dtype=torch.float64, requires_grad=True)
        loss = self.loss(x)
        loss.total.backward()
        self.assertAlmostEqual(loss.policy.item(), 1.)
        torch.testing.assert_close(x["log_probs"].grad, torch.tensor([1.5, -.5], dtype=torch.float64))

    def test_unit_ratios_and_unclipped_finite_difference(self):
        x = self.arrays(grad=True)
        x["log_probs"] = x["old_log_probs"].detach().clone().requires_grad_()
        result = self.loss(x)
        result.total.backward()
        self.assertEqual(result.clip_fraction.item(), 0.)
        self.assertAlmostEqual(result.policy.item(), 0.)
        analytic = x["log_probs"].grad[0].item()
        losses = []
        for delta in (-1e-6, 1e-6):
            shifted = {k: v.detach().clone() for k, v in x.items()}
            shifted["log_probs"][0] += delta
            losses.append(self.loss(shifted).total.item())
        self.assertAlmostEqual(analytic, (losses[1] - losses[0]) / 2e-6, places=8)

    def test_entropy_bonus_and_zero_coefficients(self):
        base = self.loss(value_loss_coef=0., entropy_coef=0.)
        entropy = self.loss(value_loss_coef=0., entropy_coef=.25)
        self.assertAlmostEqual(base.total.item() - entropy.total.item(), .25 * .3)
        self.assertAlmostEqual(base.total.item(), base.policy.item())

    def test_advantage_normalization_is_population_based_detached_and_immutable(self):
        source = torch.tensor([1., 2., 3.], dtype=torch.float64, requires_grad=True)
        before = source.detach().clone()
        expected = (np.array([1., 2., 3.]) - 2.) / (math.sqrt(2 / 3) + 1e-8)
        normalized = normalize_rollout_advantages(source, enabled=True)
        np.testing.assert_allclose(normalized.numpy(), expected)
        self.assertFalse(normalized.requires_grad)
        normalized[0] = 99
        self.assertTrue(torch.equal(source.detach(), before))
        unchanged = normalize_rollout_advantages(source, enabled=False)
        self.assertTrue(torch.equal(unchanged, before))
        self.assertNotEqual(unchanged.data_ptr(), source.data_ptr())

    def test_constant_and_singleton_advantages(self):
        torch.testing.assert_close(normalize_rollout_advantages(torch.ones(3), enabled=True), torch.zeros(3))
        torch.testing.assert_close(normalize_rollout_advantages(torch.tensor([2.]), enabled=True), torch.tensor([2.]))

    def test_coefficient_and_normalization_validation(self):
        for key, value in (("clip_ratio", 0.), ("clip_ratio", 1.), ("clip_ratio", True),
                           ("entropy_coef", -.1), ("value_loss_coef", float("nan")),
                           ("value_loss_coef", "1")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.loss(**{key: value})
        with self.assertRaises(ValueError):
            normalize_rollout_advantages(torch.ones(2), enabled=1)

    def test_no_implicit_broadcasting_dtype_conversion_or_nonfinite_input(self):
        for key in self.arrays():
            for replacement in (torch.ones(2, 1, dtype=torch.float64), torch.ones(3, dtype=torch.float64),
                                torch.ones(2, dtype=torch.float32), torch.tensor([0., float("nan")], dtype=torch.float64)):
                x = self.arrays()
                x[key] = replacement
                with self.subTest(key=key, shape=replacement.shape), self.assertRaises(ValueError):
                    self.loss(x)
        for bad in (torch.empty(0), torch.tensor([1, 2]), [1., 2.]):
            with self.assertRaises(ValueError):
                normalize_rollout_advantages(bad, enabled=True)

    def test_invalid_categorical_values_and_ratio_overflow(self):
        for key, value in (("log_probs", .1), ("old_log_probs", .1), ("entropies", -.1)):
            x = self.arrays()
            x[key][0] = value
            with self.assertRaisesRegex(ValueError, "categorical"):
                self.loss(x)
        x = self.arrays()
        x["old_log_probs"][0] = -1000.
        with self.assertRaisesRegex(ValueError, "importance ratios"):
            self.loss(x)

    def test_value_or_normalization_overflow_is_rejected(self):
        x = self.arrays()
        x["values"].fill_(1e300)
        with self.assertRaisesRegex(ValueError, "nonfinite PPO"):
            self.loss(x)
        with self.assertRaisesRegex(ValueError, "nonfinite normalized"):
            normalize_rollout_advantages(torch.full((3,), torch.finfo(torch.float32).max), enabled=True)

    def test_sealed_candidate_likelihood_connects_to_combined_loss(self):
        for architecture, mode in (("graph", "physical"), ("graph", "self_only"), ("flat", "self_only")):
            model = policy(architecture, mode).double()
            obs = observation(dtype=torch.float64)
            evaluation = evaluate_candidate_policy(model, obs, candidates(), contract())
            decision = sample_candidate(evaluation, generator=torch.Generator().manual_seed(3))
            before = model.snapshot_sha256()
            logp, value, entropy = reevaluate_candidate_decision(model, obs, decision)
            out = candidate_ppo_loss(logp.reshape(1), value.reshape(1), entropy.reshape(1),
                torch.tensor([decision.old_log_prob], dtype=torch.float64),
                torch.tensor([1.], dtype=torch.float64), torch.tensor([-2.], dtype=torch.float64),
                clip_ratio=.2, value_loss_coef=.5, entropy_coef=.01)
            self.assertAlmostEqual(out.ratios.item(), 1.)
            out.total.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertEqual(model.snapshot_sha256(), before)

    def test_diagnostics_are_detached_and_inputs_not_mutated(self):
        x = self.arrays(grad=True)
        saved = {k: v.detach().clone() for k, v in x.items()}
        out = self.loss(x)
        self.assertFalse(out.ratios.requires_grad)
        self.assertFalse(out.clip_fraction.requires_grad)
        self.assertTrue(out.total.requires_grad)
        for k, value in saved.items():
            self.assertTrue(torch.equal(value, x[k]))


if __name__ == "__main__":
    unittest.main()
