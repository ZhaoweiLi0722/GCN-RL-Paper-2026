"""Invented inputs/trajectory receipts only; no environment or scientific fit."""

import copy
from dataclasses import FrozenInstanceError, replace
import itertools
import math
import random
import unittest
from unittest.mock import patch

import numpy as np

from src.models.candidate_policy import CandidatePolicy
from src.models.matched_inputs import InputSchema, ObservationBatch
from src.rl.candidate_rollout import (
    CandidateDecision, PolicyEvaluation, evaluate_candidate_policy, prepare_candidate_segment,
    reevaluate_candidate_decision, sample_candidate, verify_behavior_evaluation,
)
from src.rl.networks import torch
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.routing_candidate_contract import RoutingRequestSchema, build_request_candidates, class_distribution
from src.rl.validated_returns import OneStepRecord, ReplaySemantics


def contract(bootstrap=True):
    inputs = InputSchema("candidate-fixture-v1", ("A", "B"), ("queue",), ("time",),
                         tuple(f"request_{i}" for i in range(8)))
    replay = ReplaySemantics("absolute_environment", "invented-cost-v1", .1, .5,
                             inputs.definition_id + "/actor-flat", inputs.definition_id + "/action",
                             15, 8, bootstrap)
    return ReplayInputContract(inputs, replay)


def request(x):
    return [x, -x, 0., 0., 0., 0., -.5, -.5]


def candidates(index=0, **changes):
    data = dict(state_token=f"invented-state-{index}", reference_request=request(.01),
                anchor_request=request(0.), option_requests=[request(.02), request(.125), request(-.125), request(.14)])
    schema = RoutingRequestSchema(contract().inputs.definition_id + "/action", 2, 4., 6)
    return build_request_candidates(data | changes, schema)


def policy(architecture="graph", mode="physical", **changes):
    settings = dict(enabled=True, architecture=architecture, message_mode=mode,
                    encoder_width=3, head_width=5, seed=271)
    return CandidatePolicy(contract().inputs, **(settings | changes))


def observation(index=0, dtype=None):
    dtype = torch.float32 if dtype is None else dtype
    return ObservationBatch(contract().inputs, torch.tensor([[[2.], [3.]]], dtype=dtype),
                            torch.tensor([[index / 10]], dtype=dtype),
                            torch.tensor([[[0., 1.], [1., 0.]]], dtype=dtype))


@unittest.skipIf(torch is None, "torch unavailable")
class CandidatePolicyTests(unittest.TestCase):
    def test_explicit_settings_required(self):
        for changes in ({"enabled": False}, {"architecture": "guess"}, {"message_mode": "mask"},
                        {"encoder_width": 0}, {"head_width": True}, {"seed": -1}, {"seed": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                policy(**changes)
        with self.assertRaisesRegex(ValueError, "flat encoder"):
            policy("flat", "physical")
        with self.assertRaisesRegex(ValueError, "four-group"):
            CandidatePolicy(replace(contract().inputs, action_names=("route",)), enabled=True,
                            architecture="graph", message_mode="physical", encoder_width=3, head_width=5, seed=1)

    def test_common_raw_information_and_candidate_features(self):
        p = policy()
        flat = policy("flat", "self_only")
        ablated = policy(mode="self_only")
        observed = {}
        for name, model in (("graph", p), ("flat", flat), ("self", ablated)):
            hook = model.score_head[0].register_forward_pre_hook(
                lambda module, args, label=name: observed.update({label: args[0].detach().clone()}))
            result = model(observation(), candidates())
            hook.remove()
            self.assertEqual(result.logits.shape, (3,))
            self.assertEqual(result.value.shape, ())
            self.assertEqual(result.actor_state, p(observation(), candidates()).actor_state)
        # Only the six encoded node entries differ; every context/action/role is shared.
        for other in ("flat", "self"):
            torch.testing.assert_close(observed["graph"][:, 6:], observed[other][:, 6:])
        nparams = lambda model: sum(v.numel() for v in model.parameters())
        self.assertEqual(nparams(p), nparams(ablated))
        self.assertNotEqual(nparams(p), nparams(flat))

    def test_self_only_uses_same_weights_but_keeps_physical_metadata(self):
        p, ablated = policy(), policy(mode="self_only")
        for a, b in zip(p.parameters(), ablated.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(p(observation(), candidates()).logits,
                                     ablated(observation(), candidates()).logits))
        no_links = replace(observation(), physical_links=torch.zeros(1, 2, 2))
        self.assertFalse(torch.equal(ablated(observation(), candidates()).logits,
                                     ablated(no_links, candidates()).logits))

    def test_shared_scorer_is_option_order_and_alias_count_independent(self):
        p, obs, bank = policy(), observation(), candidates()
        expected = p(obs, bank)
        for options in itertools.permutations(bank.requests[2:]):
            found = p(obs, candidates(option_requests=options))
            self.assertTrue(torch.equal(expected.logits, found.logits))
            self.assertTrue(torch.equal(expected.value, found.value))
        smaller = candidates(option_requests=[request(.125), request(-.125)])
        self.assertTrue(torch.equal(expected.logits, p(obs, smaller).logits))

    def test_single_class_and_empty_globals(self):
        s = replace(contract().inputs, global_feature_names=())
        p = CandidatePolicy(s, enabled=True, architecture="flat", message_mode="self_only",
                            encoder_width=2, head_width=3, seed=0)
        obs = replace(observation(), schema=s, globals=torch.empty(1, 0))
        bank = candidates(option_requests=[])
        out = p(obs, bank)
        self.assertEqual(out.logits.shape, (1,))
        self.assertEqual(class_distribution(out.logits, bank).probs.item(), 1.)

    def test_tokens_and_hashes_are_not_network_features(self):
        p = policy()
        a, b = candidates(0), candidates(999)
        self.assertNotEqual(a.sha256, b.sha256)
        self.assertTrue(torch.equal(p(observation(), a).logits, p(observation(), b).logits))
        self.assertTrue(torch.equal(p(observation(), a).value, p(observation(), b).value))

    def test_inputs_cannot_silently_change_schema_shape_or_precision(self):
        p = policy()
        for obs in (replace(observation(), schema=replace(contract().inputs, node_ids=("B", "A"))),
                    observation(dtype=torch.float64), replace(observation(), globals=torch.tensor([[float("nan")]])),
                    replace(observation(), nodes=observation().nodes.expand(2, -1, -1),
                            globals=observation().globals.expand(2, -1),
                            physical_links=observation().physical_links.expand(2, -1, -1))):
            with self.assertRaises((ValueError, TypeError)):
                p(obs, candidates())
        bank = replace(candidates(), schema=replace(candidates().schema, action_schema_id="wrong/action"))
        with self.assertRaisesRegex(ValueError, "schemas"):
            p(observation(), bank)

    def test_all_floating_heads_have_gradients_without_optimizer(self):
        for architecture, mode in (("graph", "physical"), ("graph", "self_only"), ("flat", "self_only")):
            p = policy(architecture, mode).double()
            before = p.snapshot_sha256()
            out = p(observation(dtype=torch.float64), candidates())
            loss = -class_distribution(out.logits, candidates()).log_prob(torch.tensor(0)) + out.value.square()
            loss.backward()
            self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in p.parameters()))
            self.assertGreater(sum(v.grad.abs().sum().item() for v in p.parameters()), 0.)
            self.assertEqual(before, p.snapshot_sha256())

    def test_snapshot_binds_operator_dtype_and_weights(self):
        p = policy()
        digests = {p.snapshot_sha256(), policy(mode="self_only").snapshot_sha256(),
                   policy("flat", "self_only").snapshot_sha256(), copy.deepcopy(p).double().snapshot_sha256()}
        with torch.no_grad():
            p.score_head[-1].bias.add_(.25)
        digests.add(p.snapshot_sha256())
        self.assertEqual(len(digests), 5)
        clone = policy()
        clone.load_state_dict(p.state_dict())
        self.assertEqual(clone.snapshot_sha256(), p.snapshot_sha256())

    def test_initialization_and_forward_preserve_global_rng_and_input(self):
        before_torch = torch.get_rng_state().clone()
        before_numpy, before_python = np.random.get_state(), random.getstate()
        obs = observation()
        original = obs.nodes.clone()
        p = policy()
        p(observation(), candidates())
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        for a, b in zip(before_numpy, np.random.get_state()):
            np.testing.assert_equal(a, b)
        self.assertTrue(torch.equal(obs.nodes, original))

    def test_nonfinite_parameters_fail_before_sampling(self):
        p = policy()
        with torch.no_grad():
            p.score_head[0].weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            p(observation(), candidates())


@unittest.skipIf(torch is None, "torch unavailable")
class CandidateReceiptTests(unittest.TestCase):
    def setUp(self):
        self.binding = contract()
        self.policy = policy()
        self.evaluation = evaluate_candidate_policy(self.policy, observation(), candidates(), self.binding)

    def test_log_probabilities_agree_with_independent_logsumexp(self):
        logits = self.policy(observation(), candidates()).logits.detach().double().tolist()
        maximum = max(logits)
        total = math.log(math.fsum(math.exp(x - maximum) for x in logits)) + maximum
        np.testing.assert_allclose(self.evaluation.log_probs, [v - total for v in logits], rtol=0, atol=2e-7)
        verify_behavior_evaluation(self.policy, observation(), self.evaluation)

    def test_receipt_seals_copy_inputs_and_probability_identity(self):
        e = self.evaluation
        state, probs = list(e.actor_state), list(e.log_probs)
        copied = replace(e, actor_state=state, log_probs=probs)
        digest = copied.sha256
        state[0], probs[0] = 123., -100.
        self.assertEqual(digest, copied.sha256)
        self.assertEqual(e.sha256, copied.sha256)
        with self.assertRaises(FrozenInstanceError):
            copied.value = 1
        decision = sample_candidate(e, generator=torch.Generator().manual_seed(2))
        self.assertEqual(decision.old_log_prob, e.log_probs[decision.choice.class_index])
        self.assertEqual(decision.choice.submitted_request,
                         candidates().requests[candidates().representatives[decision.choice.class_index]])

    def test_behavior_recheck_detects_tampering_or_changed_model(self):
        for bad in (replace(self.evaluation, value=self.evaluation.value + .1),
                    replace(self.evaluation, log_probs=tuple(reversed(self.evaluation.log_probs)))):
            with self.assertRaisesRegex(ValueError, "reproduce"):
                verify_behavior_evaluation(self.policy, observation(), bad)
        with torch.no_grad():
            self.policy.value_head[-1].bias.add_(1.)
        with self.assertRaisesRegex(ValueError, "reproduce"):
            verify_behavior_evaluation(self.policy, observation(), self.evaluation)

    def test_policy_mutation_during_evaluation_rejected(self):
        def mutate(module, inputs, output):
            with torch.no_grad():
                module.value_head[-1].bias.add_(1.)
        hook = self.policy.register_forward_hook(mutate)
        try:
            with self.assertRaisesRegex(ValueError, "changed during"):
                evaluate_candidate_policy(self.policy, observation(), candidates(), self.binding)
        finally:
            hook.remove()

    def test_invalid_receipt_contracts(self):
        for changes in ({"behavior_sha256": "policy-v1"}, {"log_probs": (0., 0., 0.)},
                        {"log_probs": (float("nan"),) * 3}, {"actor_state": (0.,)},
                        {"value": True}, {"inference_dtype": "float16"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.evaluation, **changes)
        changed = list(self.evaluation.actor_state)
        changed[-1] = .2
        with self.assertRaisesRegex(ValueError, "anchor"):
            replace(self.evaluation, actor_state=changed)
        foreign = replace(self.binding, inputs=replace(self.binding.inputs, node_ids=("B", "A")))
        with self.assertRaisesRegex(ValueError, "matching"):
            evaluate_candidate_policy(self.policy, observation(), candidates(), foreign)

    def test_sampling_uses_explicit_rng_and_resumes_exactly(self):
        rng = torch.Generator().manual_seed(101)
        sample_candidate(self.evaluation, generator=rng)
        saved = rng.get_state().clone()
        global_before = torch.get_rng_state().clone()
        expected = [sample_candidate(self.evaluation, generator=rng) for _ in range(12)]
        fresh = torch.Generator()
        fresh.set_state(saved)
        actual = [sample_candidate(self.evaluation, generator=fresh) for _ in range(12)]
        self.assertEqual(expected, actual)
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))
        self.assertTrue(torch.equal(rng.get_state(), fresh.get_state()))
        with self.assertRaises(ValueError):
            sample_candidate(self.evaluation, generator=None)

    def test_sampling_failure_restores_rng(self):
        rng = torch.Generator().manual_seed(101)
        before = rng.get_state().clone()
        with patch("src.rl.candidate_rollout.CandidateDecision", side_effect=ValueError("fixture failure")):
            with self.assertRaisesRegex(ValueError, "fixture failure"):
                sample_candidate(self.evaluation, generator=rng)
        self.assertTrue(torch.equal(before, rng.get_state()))

    def test_choice_cannot_be_reassigned_to_another_candidate_bank(self):
        d = sample_candidate(self.evaluation, generator=torch.Generator().manual_seed(0))
        other = evaluate_candidate_policy(self.policy, observation(), candidates(1), self.binding)
        with self.assertRaisesRegex(ValueError, "support"):
            CandidateDecision(other, d.choice)

    def test_precision_boundary_rejected_before_returning_action_and_rng_restored(self):
        bank = candidates(reference_request=request(.125 - 1e-9), option_requests=[])
        e = evaluate_candidate_policy(self.policy, observation(), bank, self.binding)
        self.assertEqual(len(e.log_probs), 1)
        rng = torch.Generator().manual_seed(42)
        before = rng.get_state().clone()
        with self.assertRaisesRegex(ValueError, "precision conversion"):
            sample_candidate(e, generator=rng)
        self.assertTrue(torch.equal(before, rng.get_state()))

    def test_new_policy_likelihood_uses_old_support_and_matches_unit_ratio(self):
        d = sample_candidate(self.evaluation, generator=torch.Generator().manual_seed(11))
        logp, value, entropy = reevaluate_candidate_decision(self.policy, observation(), d)
        self.assertAlmostEqual(torch.exp(logp - d.old_log_prob).item(), 1.)
        self.assertAlmostEqual(value.item(), self.evaluation.value)
        self.assertGreater(entropy.item(), 0.)
        with torch.no_grad():
            self.policy.score_head[-1].weight.zero_()
        updated_logp, _, _ = reevaluate_candidate_decision(self.policy, observation(), d)
        self.assertAlmostEqual(updated_logp.item(), -math.log(3), places=6)
        self.assertNotEqual(updated_logp.item(), d.old_log_prob)
        (-updated_logp).backward()
        self.assertTrue(torch.isfinite(self.policy.score_head[-1].weight.grad).all())
        self.assertEqual(d.evaluation, self.evaluation)
        with self.assertRaisesRegex(ValueError, "observation differs"):
            reevaluate_candidate_decision(self.policy, observation(1), d)
        for other in (policy(mode="self_only"), policy("flat", "self_only"), policy(head_width=6)):
            with self.assertRaisesRegex(ValueError, "definition differs"):
                reevaluate_candidate_decision(other, observation(), d)

    def test_selected_log_probability_gradient_matches_finite_difference(self):
        p = policy().double()
        obs = observation(dtype=torch.float64)
        e = evaluate_candidate_policy(p, obs, candidates(), self.binding)
        d = sample_candidate(e, generator=torch.Generator().manual_seed(11))
        logp, _, _ = reevaluate_candidate_decision(p, obs, d)
        logp.backward()
        parameter = p.score_head[0].weight
        analytic = parameter.grad[0, -3].item()
        original = parameter[0, -3].item()
        values = []
        for delta in (-1e-5, 1e-5):
            with torch.no_grad():
                parameter[0, -3] = original + delta
            values.append(reevaluate_candidate_decision(p, obs, d)[0].item())
        with torch.no_grad():
            parameter[0, -3] = original
        self.assertAlmostEqual(analytic, (values[1] - values[0]) / 2e-5, places=8)


@unittest.skipIf(torch is None, "torch unavailable")
class CandidateSegmentTests(unittest.TestCase):
    def fixture(self, *, terminated=True, truncated=False, bootstrap_on_truncation=True):
        binding = contract(bootstrap_on_truncation)
        model = policy()
        with torch.no_grad():
            model.value_head[-1].weight.zero_()
            model.value_head[-1].bias.fill_(1.)
        evaluations = [evaluate_candidate_policy(model, observation(i), candidates(i), binding) for i in range(3)]
        rng = torch.Generator().manual_seed(12)
        decisions = [sample_candidate(e, generator=rng) for e in evaluations[:2]]
        records = [OneStepRecord(binding.replay, "invented-unit-fixture", "trajectory", "trajectory-0", i,
                                 f"invented-state-{i}", f"invented-state-{i+1}", evaluations[i].actor_state,
                                 decisions[i].choice.submitted_request, -2. * (i + 1), evaluations[i+1].actor_state,
                                 terminated and i == 1, truncated and i == 1) for i in range(2)]
        return binding, decisions, records, evaluations[-1]

    def prepare(self, fixture, **changes):
        binding, decisions, records, final = fixture
        settings = dict(behavior_sha256=decisions[0].evaluation.behavior_sha256,
                        bootstrap=None, gae_lambda=.8, max_steps=2)
        return prepare_candidate_segment(decisions, records, binding, **(settings | changes))

    def test_terminal_gae_matches_independent_two_step_arithmetic(self):
        result = self.prepare(self.fixture())
        np.testing.assert_allclose(result.advantages, [-1.26, -1.4], atol=2e-7, rtol=0)
        np.testing.assert_allclose(result.returns, [-.26, -.4], atol=2e-7, rtol=0)

    def test_truncation_bootstraps_without_connecting_another_episode(self):
        f = self.fixture(terminated=False, truncated=True)
        result = self.prepare(f, bootstrap=f[-1])
        np.testing.assert_allclose(result.advantages, [-1.06, -.9], atol=2e-7, rtol=0)
        np.testing.assert_allclose(result.returns, [-.06, .1], atol=2e-7, rtol=0)
        with self.assertRaisesRegex(ValueError, "sealed next-state"):
            self.prepare(f)
        logs = np.asarray(f[-1].log_probs)
        double_logs = tuple(logs - math.log(math.fsum(math.exp(v) for v in logs)))
        for bad in (replace(f[-1], behavior_sha256="a" * 64),
                    replace(f[-1], candidates=candidates(99)),
                    replace(f[-1], inference_dtype="float64", log_probs=double_logs)):
            with self.assertRaisesRegex(ValueError, "bootstrap"):
                self.prepare(f, bootstrap=bad)

    def test_terminal_precedence_and_non_bootstrapping_truncation(self):
        for f in (self.fixture(terminated=True, truncated=True),
                  self.fixture(terminated=False, truncated=True, bootstrap_on_truncation=False)):
            result = self.prepare(f)
            np.testing.assert_allclose(result.returns, [-.26, -.4], atol=2e-7, rtol=0)
            with self.assertRaisesRegex(ValueError, "must not supply"):
                self.prepare(f, bootstrap=f[-1])

    def test_unclosed_segment_or_stale_policy_rejected(self):
        with self.assertRaisesRegex(ValueError, "end in termination"):
            self.prepare(self.fixture(terminated=False))
        with self.assertRaisesRegex(ValueError, "versions"):
            self.prepare(self.fixture(), behavior_sha256="a" * 64)
        f = self.fixture()
        f[1][1] = replace(f[1][1], evaluation=replace(f[1][1].evaluation, behavior_sha256="b" * 64))
        with self.assertRaisesRegex(ValueError, "versions"):
            self.prepare(f)
        f = self.fixture()
        f[1][1] = replace(f[1][1], evaluation=replace(f[1][1].evaluation, policy_definition_sha256="c" * 64))
        with self.assertRaisesRegex(ValueError, "definitions"):
            self.prepare(f)

    def test_record_action_observation_and_lineage_are_not_silently_repaired(self):
        for change, expected in (({"action": tuple(request(.5))}, "original submitted"),
                                 ({"source_id": "another"}, "lineage"),
                                 ({"step_index": 9}, "adjacency"),
                                 ({"state_token": "another"}, "token"),
                                 ({"state": (0.,) * 15}, "adjacency")):
            f = self.fixture()
            f[2][1] = replace(f[2][1], **change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, expected):
                self.prepare(f)

    def test_counterfactual_or_boundary_crossing_rejected(self):
        f = self.fixture()
        f[2][0] = replace(f[2][0], origin="counterfactual", trajectory_id=None, step_index=None)
        with self.assertRaisesRegex(ValueError, "trajectory records"):
            self.prepare(f)
        f = self.fixture()
        f[2][0] = replace(f[2][0], truncated=True)
        with self.assertRaisesRegex(ValueError, "boundary"):
            self.prepare(f)

    def test_reward_semantics_are_bound_into_receipts(self):
        binding, decisions, records, final = self.fixture()
        for changed in (replace(binding.replay, reward_scale=.2), replace(binding.replay, gamma=.9),
                        replace(binding.replay, reward_kind="anchor_relative")):
            other = replace(binding, replay=changed)
            rows = [replace(r, semantics=changed) for r in records]
            with self.assertRaisesRegex(ValueError, "input/reward"):
                self.prepare((other, decisions, rows, final))

    def test_segment_caps_types_and_gae_limits(self):
        for changes in ({"max_steps": 1}, {"max_steps": True}, {"gae_lambda": -.1},
                        {"gae_lambda": True}, {"gae_lambda": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.prepare(self.fixture(), **changes)
        f = self.fixture()
        with self.assertRaisesRegex(ValueError, "count"):
            self.prepare((f[0], f[1][:1], f[2], f[3]))

    def test_gae_zero_one_lambda_and_zero_gamma(self):
        f = self.fixture()
        np.testing.assert_allclose(self.prepare(f, gae_lambda=0).advantages, [-.7, -1.4], atol=2e-7)
        np.testing.assert_allclose(self.prepare(f, gae_lambda=1).returns, [-.4, -.4], atol=2e-7)
        binding = replace(f[0], replay=replace(f[0].replay, gamma=0.))
        decisions = [replace(d, evaluation=replace(d.evaluation, contract=binding)) for d in f[1]]
        rows = [replace(r, semantics=binding.replay) for r in f[2]]
        np.testing.assert_allclose(self.prepare((binding, decisions, rows, f[-1])).returns, [-.2, -.4], atol=2e-7)

    def test_nonfinite_gae_overflow_fails_without_modifying_receipts(self):
        f = self.fixture()
        seals = [d.evaluation.sha256 for d in f[1]]
        f[2][0] = replace(f[2][0], raw_reward=1e100)
        with self.assertRaisesRegex(ValueError, "overflow"):
            self.prepare(f)
        self.assertEqual(seals, [d.evaluation.sha256 for d in f[1]])


if __name__ == "__main__":
    unittest.main()
