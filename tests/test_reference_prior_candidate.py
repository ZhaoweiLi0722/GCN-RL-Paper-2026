"""Invented tensors/receipts only. Optimizer steps and imitation fits are blocked."""

import copy
from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.models.candidate_policy import CandidatePolicy, CandidateScores
from src.models.matched_inputs import InputSchema, ObservationBatch
from src.models.reference_prior_candidate import ReferencePriorCandidatePolicy
from src.rl.candidate_imitation import CandidateImitationKernel, ImitationSettings
from src.rl.candidate_ppo_kernel import CandidatePPOKernel
from src.rl.candidate_rollout import (evaluate_candidate_policy, reevaluate_candidate_decision,
                                     sample_candidate, verify_behavior_evaluation)
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.routing_candidate_contract import (RoutingRequestSchema, build_request_candidates,
                                              choose_candidate, class_distribution)
from tests.test_candidate_policy_rollout import candidates, contract, observation, policy, request
from tests.test_candidate_ppo_kernel import invented_segment, settings


def prior_policy(architecture="graph", mode="physical", **changes):
    args = dict(enabled=True, architecture=architecture, message_mode=mode,
                encoder_width=3, head_width=5, seed=271, nonreference_mass=.1)
    return ReferencePriorCandidatePolicy(contract().inputs, **(args | changes))


def bank_of_size(k, reference_position=0):
    values = [0., .25, -.25, .5, -.5, .75][:k]
    ref = values.pop(reference_position)
    anchor = values.pop(0) if values else ref
    return candidates(reference_request=request(ref), anchor_request=request(anchor),
                      option_requests=[request(v) for v in values])


def owner(prototype=None, *, mode="online", sampling_seed=11):
    return CandidatePPOKernel(prior_policy() if prototype is None else prototype,
        contract(), settings(), enabled=True, mode=mode, sampling_seed=sampling_seed, shuffle_seed=12)


@unittest.skipIf(torch is None, "torch unavailable")
class ReferencePriorCandidateTests(unittest.TestCase):
    def setUp(self):
        for target in ("torch.optim.Adam.step", "torch.optim.SGD.step",
                       "src.rl.candidate_imitation.CandidateImitationKernel.fit"):
            guard = patch(target, side_effect=AssertionError("no optimizer/fitting authorized in this acceptance"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_explicit_prior_and_enablement(self):
        for mass in (None, True, 0, -.1, .5, .9, float("nan"), float("inf"), 1e-40, .5 - 1e-10):
            with self.subTest(mass=mass), self.assertRaises(ValueError):
                prior_policy(nonreference_mass=mass)
        with self.assertRaises(ValueError):
            prior_policy(enabled=False)
        with self.assertRaises(TypeError):
            ReferencePriorCandidatePolicy(contract().inputs, enabled=True, architecture="graph",
                message_mode="physical", encoder_width=3, head_width=5, seed=1)

    def test_exact_zero_residual_and_value_with_resolved_prior_all_supports(self):
        for architecture, mode, dtype, k in itertools.product(
                ("graph", "flat"), ("physical", "self_only"), (torch.float32, torch.float64), range(1, 7)):
            if architecture == "flat" and mode == "physical":
                continue
            p = prior_policy(architecture, mode).to(dtype=dtype)
            for ref in range(k):
                bank, obs = bank_of_size(k, ref), observation(dtype=dtype)
                raw = CandidatePolicy.forward(p, obs, bank)
                self.assertTrue(torch.equal(raw.logits, torch.zeros(k, dtype=dtype)))
                self.assertEqual(raw.value.item(), 0.)
                dist = class_distribution(p(obs, bank).logits, bank)
                expected = torch.full((k,), .1 / max(k - 1, 1), dtype=dtype)
                expected[bank.reference_class] = .9 if k > 1 else 1.
                torch.testing.assert_close(dist.probs, expected, atol=8 * torch.finfo(dtype).eps, rtol=0)
                self.assertTrue((dist.probs > 0).all())
                picked = choose_candidate(bank, int(dist.probs.argmax()))
                self.assertEqual(picked.submitted_request, bank.requests[0])

    def test_alias_permutation_and_reference_anchor_merge_do_not_add_mass(self):
        p, obs = prior_policy(), observation()
        reference = p(obs, candidates()).logits
        for options in itertools.permutations(candidates().requests[2:]):
            result = p(obs, candidates(option_requests=options)).logits
            self.assertTrue(torch.equal(reference, result))
        small = candidates(option_requests=[request(.125), request(-.125)])
        self.assertTrue(torch.equal(reference, p(obs, small).logits))
        self.assertEqual(small.anchor_class, small.reference_class)
        self.assertEqual(choose_candidate(small, small.reference_class).submitted_request, tuple(request(.01)))

    def test_prior_is_initial_behavior_not_a_permanent_reference_mask(self):
        p, bank, obs = prior_policy(), bank_of_size(6), observation()
        selected = (bank.reference_class + 1) % 6
        residual = torch.zeros(6)
        residual[selected] = 8.
        raw = CandidatePolicy.forward(p, obs, bank)
        with patch.object(CandidatePolicy, "forward", return_value=CandidateScores(residual, raw.value, raw.actor_state)):
            out = p(obs, bank)
        self.assertEqual(int(out.logits.argmax()), selected)
        self.assertLess(class_distribution(out.logits, bank).probs[bank.reference_class].item(), .9)

    def test_same_seed_graph_and_self_only_have_identical_tensors_and_behavior(self):
        graph, ablated = prior_policy(), prior_policy(mode="self_only")
        self.assertNotEqual(graph.definition_sha256(), ablated.definition_sha256())
        for a, b in zip(graph.parameters(), ablated.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(graph(observation(), candidates()).logits,
                                    ablated(observation(), candidates()).logits))

    def test_invented_full_width_schema_counts_and_information_parity(self):
        schema = InputSchema("invented-full-width-v1", tuple(f"f{i}" for i in range(20)),
            tuple(f"x{i}" for i in range(28)), ("time",), tuple(f"a{i}" for i in range(80)))
        obs = ObservationBatch(schema, torch.arange(560, dtype=torch.float32).reshape(1, 20, 28) / 100,
                               torch.tensor([[.5]]), torch.ones(1, 20, 20) - torch.eye(20).unsqueeze(0))
        r4, mdl2 = [0.] * 80, [0.] * 80
        r4[0], r4[1] = .05, -.05
        bank = build_request_candidates(dict(state_token="invented-wide", reference_request=r4,
            anchor_request=mdl2, option_requests=[]), RoutingRequestSchema(schema.definition_id + "/action", 20, 120., 6))
        states, logits = [], []
        for architecture, mode, count in (("graph", "physical", 59602), ("graph", "self_only", 59602),
                                           ("flat", "self_only", 238658)):
            p = ReferencePriorCandidatePolicy(schema, enabled=True, architecture=architecture,
                message_mode=mode, encoder_width=16, head_width=32, seed=7, nonreference_mass=.1)
            self.assertEqual(sum(v.numel() for v in p.parameters()), count)
            scores = p(obs, bank)
            states.append(scores.actor_state)
            logits.append(scores.logits)
        self.assertEqual(len(states[0]), 1041)
        self.assertTrue(all(x == states[0] for x in states))
        self.assertTrue(all(torch.equal(x, logits[0]) for x in logits))

    def test_first_backward_has_live_output_gradient_but_zero_encoder_gradient(self):
        for architecture, mode in (("graph", "physical"), ("graph", "self_only"), ("flat", "self_only")):
            p = prior_policy(architecture, mode).double()
            before = p.snapshot_sha256()
            bank = bank_of_size(6)
            out = p(observation(dtype=torch.float64), bank)
            loss = -torch.log_softmax(out.logits, 0)[(bank.reference_class + 1) % 6] + (out.value - 1).square()
            loss.backward()
            self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in p.parameters()))
            self.assertGreater(p.score_head[-1].weight.grad.abs().sum().item(), 0.)
            self.assertGreater(p.value_head[-1].weight.grad.abs().sum().item(), 0.)
            self.assertTrue(all(v.grad.count_nonzero().item() == 0 for v in p.encoder.parameters()))
            self.assertEqual(before, p.snapshot_sha256())

    def test_definition_binds_mass_and_rejects_cross_policy_receipts(self):
        a, b, legacy = prior_policy(), prior_policy(nonreference_mass=.2), policy()
        self.assertEqual(len({p.definition_sha256() for p in (a, b, legacy)}), 3)
        evaluation = evaluate_candidate_policy(a, observation(), candidates(), contract())
        decision = sample_candidate(evaluation, generator=torch.Generator().manual_seed(10))
        for other in (b, legacy):
            with self.assertRaises(ValueError):
                reevaluate_candidate_decision(other, observation(), decision)
            with self.assertRaises(ValueError):
                verify_behavior_evaluation(other, observation(), evaluation)

    def test_unknown_subclass_is_not_accepted_by_receipts_or_kernel(self):
        class UnknownPolicy(ReferencePriorCandidatePolicy):
            pass
        prototype = UnknownPolicy(contract().inputs, enabled=True, architecture="graph", message_mode="physical",
                                  encoder_width=3, head_width=5, seed=1, nonreference_mass=.1)
        with self.assertRaises(TypeError):
            evaluate_candidate_policy(prototype, observation(), candidates(), contract())
        with self.assertRaises(ValueError):
            owner(prototype)

    def test_actual_prior_likelihood_is_stored_and_unmodified_ratio_is_one(self):
        p, bank = prior_policy(), bank_of_size(6)
        evaluation = evaluate_candidate_policy(p, observation(), bank, contract())
        generator = torch.Generator().manual_seed(319)
        for _ in range(80):
            decision = sample_candidate(evaluation, generator=generator)
            new_log_prob, value, entropy = reevaluate_candidate_decision(p, observation(), decision)
            self.assertAlmostEqual(float(new_log_prob.detach()), decision.old_log_prob, places=6)
            self.assertAlmostEqual(float(torch.exp(new_log_prob.detach() - decision.old_log_prob)), 1., places=6)
            expected = .9 if decision.choice.class_index == bank.reference_class else .02
            self.assertAlmostEqual(math.exp(decision.old_log_prob), expected, places=6)
            self.assertEqual(float(value.detach()), 0.)
            self.assertGreater(float(entropy.detach()), 0.)

    def test_probability_sampler_invented_draws_match_prior_and_cover_support(self):
        p, bank = prior_policy(), bank_of_size(6)
        e = evaluate_candidate_policy(p, observation(), bank, contract())
        weights = torch.tensor(e.log_probs).exp()
        sample = torch.multinomial(weights, 10000, replacement=True, generator=torch.Generator().manual_seed(81))
        counts = torch.bincount(sample, minlength=6)
        self.assertLess(abs(counts[bank.reference_class].item() / 10000 - .9), .02)
        self.assertTrue((counts > 0).all())

    def test_common_frozen_ppo_and_bc_initial_sampling_without_updates(self):
        p = prior_policy()
        ppo, frozen = owner(p), owner(p, mode="frozen")
        bc = CandidateImitationKernel(p, contract(), ImitationSettings(.001, .5, 2, 8, 2, "training"),
            enabled=True, sampling_seed=11, shuffle_seed=12)
        self.assertIsNone(frozen.optimizer)
        self.assertFalse(ppo.optimizer.state)
        self.assertFalse(bc.optimizer.state)
        for _ in range(25):
            self.assertEqual(ppo.decide(observation(), candidates()), frozen.decide(observation(), candidates()))
        mirror = owner(p)
        for _ in range(25):
            self.assertEqual(mirror.decide(observation(), candidates()), bc.decide(observation(), candidates()))
        self.assertEqual(ppo.total_optimizer_steps, 0)
        self.assertEqual(bc.steps, 0)
        self.assertEqual({x.policy.snapshot_sha256() for x in (ppo, frozen, bc)}, {p.snapshot_sha256()})

    def test_pending_receipts_private_rng_and_empty_optimizer_restore(self):
        a = owner()
        a.add_segment(invented_segment(a))
        before = state_digest(a.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invented.pt"
            a.save(path)
            b = owner()
            b.load(path)
        self.assertEqual(before, state_digest(b.state_dict()))
        for _ in range(16):
            self.assertEqual(a.decide(observation(), candidates()), b.decide(observation(), candidates()))
        self.assertEqual(a.total_optimizer_steps, 0)
        self.assertEqual(len(a.pending), 1)

    def test_changed_prior_checkpoint_is_rejected_atomically(self):
        destination, source = owner(), owner(prior_policy(nonreference_mass=.2))
        before = state_digest(destination.state_dict())
        with self.assertRaises(ValueError):
            destination.load_state_dict(source.state_dict())
        self.assertEqual(before, state_digest(destination.state_dict()))
        bad = copy.deepcopy(destination.state_dict())
        bad["policy"]["score_head.2.bias"].add_(1.)
        with self.assertRaises(ValueError):
            destination.load_state_dict(bad)
        self.assertEqual(before, state_digest(destination.state_dict()))

    def test_bc_empty_state_restore_and_wrong_prior_rejection(self):
        def make(mass):
            return CandidateImitationKernel(prior_policy(nonreference_mass=mass), contract(),
                ImitationSettings(.001, .5, 2, 8, 2, "training"), enabled=True, sampling_seed=11, shuffle_seed=12)
        a, b = make(.1), make(.1)
        a.decide(observation(), candidates())
        b.load_state_dict(a.state_dict())
        self.assertEqual(a.decide(observation(), candidates()), b.decide(observation(), candidates()))
        before = state_digest(b.state_dict())
        with self.assertRaises(ValueError):
            b.load_state_dict(make(.2).state_dict())
        self.assertEqual(before, state_digest(b.state_dict()))

    def test_global_rng_inputs_and_legacy_policy_are_unchanged(self):
        before = (torch.get_rng_state().clone(), copy.deepcopy(np.random.get_state()), random.getstate())
        legacy, obs, bank = policy(), observation(), candidates()
        legacy_hash = legacy.snapshot_sha256()
        original = obs.nodes.clone()
        prior_policy()(obs, bank)
        self.assertEqual(legacy_hash, legacy.snapshot_sha256())
        self.assertEqual(legacy.manifest()["format"], "candidate-policy-v1")
        self.assertNotIn("reference_prior", legacy.manifest())
        self.assertTrue(torch.equal(before[0], torch.get_rng_state()))
        for a, b in zip(before[1], np.random.get_state()):
            np.testing.assert_equal(a, b)
        self.assertEqual(before[2], random.getstate())
        self.assertTrue(torch.equal(original, obs.nodes))

    def test_nonfinite_inputs_and_schema_mismatch_still_fail(self):
        p = prior_policy()
        for obs in (replace(observation(), nodes=torch.full((1, 2, 1), float("nan"))),
                    observation(dtype=torch.float64)):
            with self.assertRaises(ValueError):
                p(obs, candidates())
        with torch.no_grad():
            p.score_head[-1].weight.fill_(float("inf"))
        with self.assertRaises(ValueError):
            p(observation(), candidates())

    def test_design_config_is_not_an_execution_permit_and_caps_add_up(self):
        path = Path(__file__).resolve().parents[1] / "experiments/configs/candidate_reference_prior_design_20261001.json"
        cfg = json.loads(path.read_text())
        self.assertIs(cfg["scientific_execution_authorized"], False)
        self.assertTrue(cfg["engineering_only"])
        self.assertFalse(cfg["runner_integrated"])
        for key in ("new_patient_trajectories_authorized", "optimizer_steps_authorized", "new_scientific_attempts_authorized"):
            self.assertEqual(cfg[key], 0)
        caps = cfg["proposal_not_execution_caps"]
        self.assertEqual(sum(caps[k] for k in ("preflight_environment_calls", "prior_qualification_environment_calls",
            "ppo_environment_calls", "bc_continue_environment_calls", "evaluation_environment_calls")), caps["global_environment_calls"])
        self.assertEqual(caps["ppo_optimizer_steps"] + caps["bc_continue_optimizer_steps"], caps["global_optimizer_steps"])
        self.assertEqual(cfg["policy"]["nonreference_mass"], .1)


if __name__ == "__main__":
    unittest.main()
