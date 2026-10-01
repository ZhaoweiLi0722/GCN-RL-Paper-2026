"""Artificial contracts and derivatives only; optimizer.step is forbidden."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.rl.actor_positive_control import CONFIG, actor_objective, checkpoint, model, remaining_time, rng_snapshot
from src.rl.actor_control_verification import scalar_metrics, softmax, weight_digest
from src.rl.candidate_calibration_engineering import charge, invented_data, scores
from src.rl.candidate_patient_session import load_envelope, save_envelope
from src.rl.candidate_ppo_kernel import CandidatePPOKernel
from src.rl.candidate_rollout import evaluate_candidate_policy
from src.rl.networks import torch
from src.rl.routing_candidate_contract import build_request_candidates


ROOT = Path(__file__).resolve().parents[1]


class ActorControlTests(unittest.TestCase):
    def setUp(self):
        for target in ("torch.optim.Adam.step", "torch.optim.SGD.step"):
            guard = patch(target, side_effect=AssertionError("no optimizer updates in unit tests"))
            guard.start()
            self.addCleanup(guard.stop)
        self.cfg = json.loads((ROOT / CONFIG).read_text())
        self.data = invented_data(self.cfg, [.15, .55])
        self.bank = self.data[1]

    def actor(self, rep="graph"):
        return model(self.cfg, self.bank, rep, 909)

    def test_fixed_scope_and_previous_task_unchanged(self):
        self.assertFalse(self.cfg["scientific_execution_authorized"])
        self.assertEqual(self.cfg["patient_calls_authorized"], 0)
        self.assertEqual(self.cfg["caps"]["fixtures"]*self.cfg["optimizer"]["steps_per_fixture"], 1152)
        previous = json.loads((ROOT / "experiments/configs/candidate_calibration_engineering_20261001.json").read_text())
        for key in ("representations", "initialization_seeds", "training_times", "heldout_times", "cues",
                    "request_values", "positive_cue_winner_request", "negative_cue_winner_request", "invented_return"):
            self.assertEqual(self.cfg[key], previous[key])
        self.assertFalse(set(self.cfg["training_times"]) & set(self.cfg["heldout_times"]))

    def test_same_start_nearly_uniform_reference_tie_break(self):
        for rep in self.cfg["representations"]:
            p = self.actor(rep)
            frozen = copy.deepcopy(p)
            logits, sentinel = scores(p, self.data[0], self.bank)
            expected = torch.zeros_like(logits)
            expected[:, self.bank.reference_class] = self.cfg["initial_reference_logit"]
            torch.testing.assert_close(logits, expected)
            self.assertTrue((logits.argmax(1) == self.bank.reference_class).all())
            self.assertTrue((sentinel == 0).all())
            self.assertLess(float(logits.detach().softmax(1).max()), .167)
            self.assertEqual(p.snapshot_sha256(), frozen.snapshot_sha256())
            self.assertFalse(any("critic" in n or "value" in n for n, _ in p.named_parameters()))

    def test_preference_is_trainable_bias_not_forward_prior(self):
        p = self.actor()
        self.assertTrue(p.actor_head.bias.requires_grad)
        with torch.no_grad():
            p.actor_head.bias.zero_()
        logits, _ = scores(p, self.data[0], self.bank)
        self.assertTrue((logits == 0).all())

    def test_same_public_features_raw_receipts_and_parameter_disclosure(self):
        ps = [self.actor(r) for r in self.cfg["representations"]]
        raw, context = [], []
        for p in ps:
            seen = []
            hook = p.actor_head.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach()))
            out = p(self.data[0][0], self.bank)
            hook.remove()
            context.append(seen[0][:, 32:])
            raw.append(out.actor_state)
            self.assertLessEqual(float(seen[0].abs().max()), 2.)
        self.assertEqual(raw, [raw[0]]*3)
        self.assertIn(100000., raw[0])
        self.assertTrue(all(torch.equal(x, context[0]) for x in context))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(ps[0].parameters(), ps[1].parameters())))
        counts = [sum(v.numel() for v in p.parameters()) for p in ps]
        self.assertEqual(counts[0], counts[1])
        self.assertNotEqual(counts[0], counts[2])

    def test_fixed_bank_rejects_changes_and_preserves_canonical_order(self):
        p = self.actor()
        b = build_request_candidates({"state_token": "reordered", "reference_request": self.bank.requests[0],
            "anchor_request": self.bank.requests[1], "option_requests": list(reversed(self.bank.requests[2:]))}, self.bank.schema)
        torch.testing.assert_close(p(self.data[0][0], b).logits, p(self.data[0][0], self.bank).logits)
        short = build_request_candidates({"state_token": "short", "reference_request": self.bank.requests[0],
            "anchor_request": self.bank.requests[1], "option_requests": []}, self.bank.schema)
        with self.assertRaises(ValueError):
            p(self.data[0][0], short)
        with self.assertRaises(TypeError):
            p(self.data[0][0], {"labels": [2]})

    def test_bad_precision_schema_and_constants(self):
        p, obs = self.actor(), self.data[0][0]
        for bad in (replace(obs, nodes=obs.nodes.double()),
                    replace(obs, globals=torch.full_like(obs.globals, float("nan"))),
                    replace(obs, schema=replace(obs.schema, definition_id="wrong"))):
            with self.assertRaises((ValueError, TypeError)):
                p(bad, self.bank)
        for val in (0, -1, True, float("nan"), 1.):
            cfg = copy.deepcopy(self.cfg)
            cfg["initial_reference_logit"] = val
            with self.assertRaises(ValueError):
                model(cfg, self.bank, "graph", 1)

    def test_expected_surrogate_and_gradient_match_independent_formula(self):
        logits = torch.tensor([[.2, -.1, .4], [.1, .5, -.2]], requires_grad=True)
        old = torch.tensor([[.4, .3, .3], [.5, .3, .2]], requires_grad=True)
        q = torch.tensor([[-3., -2., -1.], [-.5, -.8, -.6]], requires_grad=True)
        loss = actor_objective(logits, old.log(), q, self.cfg["optimizer"])
        probs = logits.softmax(1)
        r = probs/old.detach()
        adv = (q-(old*q).sum(1, keepdim=True)).detach()
        expected = -(old.detach()*torch.minimum(r*adv, r.clamp(.8, 1.2)*adv)).sum(1).mean()
        expected += .01*(probs*logits.log_softmax(1)).sum(1).mean()
        torch.testing.assert_close(loss.total, expected)
        torch.testing.assert_close(torch.autograd.grad(loss.total, logits, retain_graph=True)[0],
                                   torch.autograd.grad(expected, logits, retain_graph=True)[0])
        self.assertEqual(loss.value.item(), 0.)
        self.assertEqual(torch.autograd.grad(loss.total, (old, q), allow_unused=True), (None, None))

    def test_gradient_finite_and_output_rows_receive_state_specific_signal(self):
        p = self.actor()
        before = p.snapshot_sha256()
        logits, _ = scores(p, self.data[0], self.bank)
        loss = actor_objective(logits, logits.detach().log_softmax(1), self.data[2], self.cfg["optimizer"])
        loss.total.backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in p.parameters()))
        cue_column = 32+1
        winner = self.data[3][-1]
        self.assertLess(p.actor_head.weight.grad[winner, cue_column].item(), 0)
        self.assertGreater(p.actor_head.weight.grad[self.bank.reference_class, cue_column].item(), 0)
        self.assertEqual(before, p.snapshot_sha256())

    def test_serialization_and_rng_without_optimization(self):
        before = rng_snapshot()
        p = self.actor()
        self.assertTrue(torch.equal(before["torch_cpu"], rng_snapshot()["torch_cpu"]))
        opt = torch.optim.Adam(p.parameters(), lr=.0003)
        with patch("src.rl.actor_positive_control.time.monotonic", return_value=12.):
            state = checkpoint(p, opt, self.cfg, 0, 0, 10., None)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "artificial.pt"
            save_envelope(path, state)
            loaded = load_envelope(path)
        other = self.actor()
        other.load_state_dict(loaded["policy"])
        restored_opt = torch.optim.Adam(other.parameters(), lr=.01)
        restored_opt.load_state_dict(loaded["optimizer"])
        self.assertEqual(p.snapshot_sha256(), other.snapshot_sha256())
        self.assertEqual(p.snapshot_sha256(), weight_digest(loaded))
        self.assertEqual(restored_opt.param_groups[0]["lr"], .0003)
        self.assertEqual(loaded["elapsed_seconds"], 2.)
        self.assertFalse(loaded["resume_authorized"])
        self.assertEqual(loaded["rng"]["python"], before["python"])
        self.assertTrue(torch.equal(loaded["rng"]["torch_cpu"], before["torch_cpu"]))
        nr = loaded["rng"]["numpy"]
        restored_rng = np.random.RandomState()
        restored_rng.set_state((nr["algorithm"], np.array(nr["keys_uint32"], dtype=np.uint32),
                                nr["position"], nr["has_gaussian"], nr["cached_gaussian"]))
        self.assertEqual(restored_rng.get_state()[1].tolist(), before["numpy"]["keys_uint32"])
        self.assertEqual(restored_rng.get_state()[2:], np.random.get_state()[2:])

    def test_real_collectors_and_kernel_reject_actor(self):
        with self.assertRaises(TypeError):
            evaluate_candidate_policy(self.actor(), self.data[0][0], self.bank, None)
        with self.assertRaises(ValueError):
            CandidatePPOKernel(self.actor(), None, None, enabled=True, mode="online", sampling_seed=1, shuffle_seed=2)

    def test_budget_and_wall_caps_do_not_refund(self):
        total = 0
        for _ in range(9):
            used = 0
            for _ in range(128):
                total, used = charge(total, used, 1152, 128)
            with self.assertRaises(RuntimeError):
                charge(total, used, 1152, 128)
        self.assertEqual(total, 1152)
        with patch("src.rl.actor_positive_control.time.monotonic", return_value=1800.):
            with self.assertRaises(TimeoutError):
                remaining_time(0, 1800)

    def test_bad_old_distribution_cannot_silently_change_objective(self):
        logits = torch.zeros(2, 3, requires_grad=True)
        q = torch.zeros_like(logits)
        with self.assertRaises(ValueError):
            actor_objective(logits, torch.zeros_like(logits), q, self.cfg["optimizer"])
        with self.assertRaises(ValueError):
            actor_objective(logits, torch.zeros(3), q, self.cfg["optimizer"])

    def test_independent_scalar_readback_on_known_arithmetic(self):
        result = scalar_metrics([[2., 1., -1.], [0., 2., 1.]], [0, 1])
        self.assertEqual(result, {"accuracy": 1., "minimum_winner_margin": 1., "greedy_classes": [0, 1]})
        failure = scalar_metrics([[2., 1.], [2., 1.]], [0, 1])
        self.assertEqual(failure["accuracy"], .5)
        self.assertEqual(failure["minimum_winner_margin"], -1.)
        self.assertAlmostEqual(sum(softmax([1000., 1001.])), 1.)
        with self.assertRaises(ValueError):
            scalar_metrics([[float("nan"), 0.]], [0])


if __name__ == "__main__":
    unittest.main()
