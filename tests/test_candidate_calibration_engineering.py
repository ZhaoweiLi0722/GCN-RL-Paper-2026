"""Zero optimizer-call contract tests, distinct from the nine recorded fits."""

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from src.models.calibrated_reference_candidate import CandidateCalibration, CalibratedReferenceCandidatePolicy
from src.models.reference_prior_candidate import ReferencePriorCandidatePolicy
from src.rl.candidate_calibration_engineering import (CONFIG, charge, enumerated_objective,
    fixture_schema, gradient_readback, invented_data, model, scores)
from src.rl.candidate_ppo_kernel import CandidatePPOKernel
from src.rl.candidate_rollout import evaluate_candidate_policy
from src.rl.networks import torch
from src.rl.routing_candidate_contract import build_request_candidates


ROOT = Path(__file__).resolve().parents[1]


class CalibrationContractTests(unittest.TestCase):
    def setUp(self):
        for target in ("torch.optim.Adam.step", "torch.optim.SGD.step"):
            guard = patch(target, side_effect=AssertionError("no optimizer calls in unit tests"))
            guard.start()
            self.addCleanup(guard.stop)
        self.cfg = json.loads((ROOT / CONFIG).read_text())
        self.data = invented_data(self.cfg, [.15, .55])

    def test_config_is_bounded_and_not_a_scientific_permit(self):
        self.assertFalse(self.cfg["scientific_execution_authorized"])
        self.assertEqual(self.cfg["patient_calls_authorized"], 0)
        self.assertEqual(len(self.cfg["representations"])*len(self.cfg["initialization_seeds"]), 9)
        self.assertEqual(9*self.cfg["optimizer"]["steps_per_fixture"], 1152)
        self.assertFalse(set(self.cfg["training_times"]) & set(self.cfg["heldout_times"]))

    def test_calibration_rejects_bad_shapes_and_constants(self):
        good = model(self.cfg, "graph", 9).calibration
        for value in (0, -1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                replace(good, logit_gain=value)
        with self.assertRaises(ValueError):
            replace(good, node_divisors=[1.])
        with self.assertRaises(ValueError):
            replace(good, node_divisors=(1., 1.)).validate(fixture_schema())
        with self.assertRaises(TypeError):
            CalibratedReferenceCandidatePolicy(fixture_schema(), calibration=None)

    def test_initial_probabilities_value_and_frozen_weights(self):
        for rep in self.cfg["representations"]:
            p = model(self.cfg, rep, 9)
            frozen = copy.deepcopy(p)
            logits, values = scores(p, self.data[0], self.data[1])
            probs = logits.softmax(1)
            self.assertTrue(torch.allclose(probs[:, self.data[1].reference_class], torch.full((4,), .9), atol=1e-6, rtol=0))
            self.assertTrue((values == 0).all())
            self.assertEqual(p.snapshot_sha256(), frozen.snapshot_sha256())

    def test_same_raw_information_and_normalized_context(self):
        obs, bank = self.data[0][0], self.data[1]
        contexts, raw_states = [], []
        for rep in self.cfg["representations"]:
            p = model(self.cfg, rep, 9)
            captured = []
            hook = p.score_head[0].register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
            out = p(obs, bank)
            hook.remove()
            contexts.append(captured[0][:, 32:])
            raw_states.append(out.actor_state)
            self.assertLessEqual(float(captured[0].abs().max()), 2.)
        self.assertTrue(all(torch.equal(x, contexts[0]) for x in contexts))
        self.assertEqual(raw_states, [raw_states[0]]*3)
        self.assertIn(100000., raw_states[0])
        self.assertIn(2000., raw_states[0])

    def test_unit_conversion_does_not_change_normalized_neural_result(self):
        p = model(self.cfg, "flat", 9)
        with torch.no_grad():
            p.score_head[-1].weight.fill_(.03)
            p.value_head[-1].weight.fill_(.02)
        other = copy.deepcopy(p)
        other.calibration = replace(p.calibration, node_divisors=(10000.,))
        obs, bank = self.data[0][0], self.data[1]
        changed = replace(obs, nodes=obs.nodes*10)
        a, b = p(obs, bank), other(changed, bank)
        torch.testing.assert_close(a.logits, b.logits)
        torch.testing.assert_close(a.value, b.value)
        self.assertNotEqual(a.actor_state, b.actor_state)
        self.assertNotEqual(p.definition_sha256(), other.definition_sha256())

    def test_message_operator_remains_unscaled(self):
        p = model(self.cfg, "graph", 9)
        found = []
        hook = p.encoder.register_forward_pre_hook(lambda _, args: found.append(args[1].clone()))
        p(self.data[0][0], self.data[1])
        hook.remove()
        expected = torch.tensor([[[1., 100000.], [100000., 1.]]])/100001
        torch.testing.assert_close(found[0], expected)

    def test_graph_ablation_counts_equal_and_flat_disclosed(self):
        g, s, f = (model(self.cfg, r, 9) for r in self.cfg["representations"])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(g.parameters(), s.parameters())))
        count = lambda p: sum(v.numel() for v in p.parameters())
        self.assertEqual(count(g), count(s))
        self.assertNotEqual(count(g), count(f))

    def test_alias_order_invariance_and_singleton(self):
        p, obs, bank = model(self.cfg, "graph", 9), self.data[0][0], self.data[1]
        reversed_bank = build_request_candidates({"state_token": "another-token",
            "reference_request": bank.requests[0], "anchor_request": bank.requests[1],
            "option_requests": list(reversed(bank.requests[2:]))}, bank.schema)
        torch.testing.assert_close(p(obs, bank).logits, p(obs, reversed_bank).logits)
        one = build_request_candidates({"state_token": "one", "reference_request": bank.requests[0],
            "anchor_request": bank.requests[0], "option_requests": []}, bank.schema)
        self.assertEqual(p(obs, one).logits.softmax(0).item(), 1.)

    def test_gradient_components_finite_without_updating(self):
        p = model(self.cfg, "graph", 9)
        before = p.snapshot_sha256()
        logits, values = scores(p, self.data[0], self.data[1])
        loss = enumerated_objective(logits, values, logits.detach().log_softmax(1), self.data[2], self.cfg["optimizer"])
        grad = gradient_readback(p, loss, self.cfg["optimizer"])
        self.assertGreater(grad["actor_norm"], 0)
        self.assertGreater(grad["value_norm"], 0)
        self.assertEqual(grad["shared_actor_norm"], 0)
        self.assertEqual(grad["shared_value_norm"], 0)
        self.assertEqual(p.snapshot_sha256(), before)

    def test_expected_surrogate_matches_independent_weighted_formula(self):
        logits = torch.tensor([[.2, -.1, .4], [.1, .5, -.2]], requires_grad=True)
        old = torch.tensor([[.4, .3, .3], [.5, .3, .2]])
        q = torch.tensor([[-3., -2., -1.], [-.5, -.8, -.6]])
        values = torch.tensor([-1., -.5], requires_grad=True)
        loss = enumerated_objective(logits, values, old.log(), q, self.cfg["optimizer"])
        probs = logits.softmax(1)
        ratios = probs/old
        v = (old*q).sum(1)
        adv = q-v[:, None]
        expected = -(old*torch.minimum(ratios*adv, ratios.clamp(.8, 1.2)*adv)).sum(1).mean()
        torch.testing.assert_close(loss.policy, expected)
        torch.testing.assert_close(loss.value, ((values-v)/4).square().mean())
        grads = torch.autograd.grad(loss.policy, logits, retain_graph=True)[0]
        torch.testing.assert_close(grads, torch.autograd.grad(expected, logits)[0])

    def test_unregistered_scientific_receipts_reject_candidate(self):
        p = model(self.cfg, "graph", 9)
        with self.assertRaises(TypeError):
            evaluate_candidate_policy(p, self.data[0][0], self.data[1], None)
        with self.assertRaises(ValueError):
            CandidatePPOKernel(p, None, None, enabled=True, mode="online", sampling_seed=1, shuffle_seed=2)

    def test_nonfinite_precision_schema_and_overflow_rejected(self):
        p, obs, bank = model(self.cfg, "graph", 9), self.data[0][0], self.data[1]
        for altered in (replace(obs, nodes=torch.full_like(obs.nodes, float("nan"))),
                        replace(obs, nodes=obs.nodes.double()),
                        replace(obs, schema=replace(obs.schema, definition_id="other"))):
            with self.assertRaises((ValueError, TypeError)):
                p(altered, bank)
        p.calibration = replace(p.calibration, node_divisors=(1e-300,))
        with self.assertRaises(ValueError):
            p(obs, bank)

    def test_manifest_gains_and_raw_input_immutability(self):
        p = model(self.cfg, "graph", 9)
        before = torch.get_rng_state().clone()
        original = self.data[0][0].nodes.clone()
        out = p(self.data[0][0], self.data[1])
        p.calibration = replace(p.calibration, logit_gain=16)
        self.assertEqual(p.manifest()["calibration"]["logit_gain"], 16)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(original, self.data[0][0].nodes))
        self.assertFalse(p.manifest()["scientific_runner_integrated"])

    def test_persistent_budget_arithmetic_never_refunds(self):
        total = 0
        for _ in range(9):
            used = 0
            for _ in range(128):
                total, used = charge(total, used, 1152, 128)
            with self.assertRaises(RuntimeError):
                charge(total, used, 1152, 128)
        self.assertEqual(total, 1152)
        with self.assertRaises(RuntimeError):
            charge(total, 0, 1152, 128)


if __name__ == "__main__":
    unittest.main()
