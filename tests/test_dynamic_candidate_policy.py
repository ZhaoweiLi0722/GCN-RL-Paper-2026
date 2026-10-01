"""Invented tensors only: no optimizer calls, fitting, environment or artifacts."""

import copy
from dataclasses import replace
import itertools
import json
import random
import unittest
from unittest.mock import patch

import numpy as np

from src.models.candidate_policy import CandidatePolicy, CandidateScores
from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.models.matched_inputs import InputSchema, ObservationBatch
from src.rl.networks import torch
from src.rl.routing_candidate_contract import (
    RoutingRequestSchema, build_request_candidates, choose_candidate,
    class_distribution, validate_choice_precision,
)


MODES = (("graph", "physical"), ("graph", "self_only"), ("flat", "self_only"))


def schema():
    return InputSchema("dynamic-invented-v1", ("A", "B"), ("queue",), ("time",),
                       tuple(f"request_{i}" for i in range(8)))


def request(x):
    return [x, -x, 0., 0., 0., 0., -.5, -.5]


def candidates(**changes):
    payload = dict(state_token="invented-state", reference_request=request(.01),
                   anchor_request=request(0.),
                   option_requests=[request(.02), request(.125), request(-.125), request(.14)])
    return build_request_candidates(payload | changes,
                                    RoutingRequestSchema(schema().definition_id + "/action", 2, 4., 6))


def observation(dtype=None):
    dtype = torch.float32 if dtype is None else dtype
    return ObservationBatch(schema(), torch.tensor([[[2.], [3.]]], dtype=dtype),
                            torch.tensor([[.2]], dtype=dtype),
                            torch.tensor([[[0., 1.], [1., 0.]]], dtype=dtype))


def policy(architecture="graph", message_mode="physical", **changes):
    settings = dict(enabled=True, architecture=architecture, message_mode=message_mode,
                    encoder_width=3, head_width=5, actor_seed=271, critic_seed=811,
                    initial_reference_bias=.1)
    return DynamicCandidatePolicy(schema(), **(settings | changes))


@unittest.skipIf(torch is None, "torch unavailable")
class DynamicCandidatePolicyTests(unittest.TestCase):
    def setUp(self):
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            sentinel = patch.object(optimizer, "step", side_effect=AssertionError("optimizer step forbidden"))
            mocked = sentinel.start()
            self.addCleanup(sentinel.stop)
            self.addCleanup(mocked.assert_not_called)

    def assert_same_state(self, left, right):
        self.assertEqual(tuple(left.state_dict()), tuple(right.state_dict()))
        for name, tensor in left.state_dict().items():
            self.assertTrue(torch.equal(tensor, right.state_dict()[name]), name)

    def test_explicit_opt_in_and_constructor_validation(self):
        bad = ({"enabled": False}, {"enabled": 1}, {"architecture": "unknown"},
               {"message_mode": "mask"}, {"architecture": "flat"},
               {"encoder_width": 0}, {"head_width": True}, {"encoder_width": 1.5},
               {"actor_seed": -1}, {"actor_seed": True}, {"critic_seed": 2**63},
               {"critic_seed": 1.5}, {"initial_reference_bias": True},
               {"initial_reference_bias": "0.1"}, {"initial_reference_bias": float("nan")},
               {"initial_reference_bias": float("inf")}, {"initial_reference_bias": 1e100})
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                policy(**changes)
        settings = dict(architecture="graph", message_mode="physical", encoder_width=3,
                        head_width=5, actor_seed=1, critic_seed=2, initial_reference_bias=0.)
        with self.assertRaisesRegex(ValueError, "explicitly enabled"):
            DynamicCandidatePolicy(schema(), **settings)
        with self.assertRaises(TypeError):
            DynamicCandidatePolicy({}, enabled=True, **settings)
        with self.assertRaisesRegex(ValueError, "four-group"):
            DynamicCandidatePolicy(replace(schema(), action_names=("route",)), enabled=True, **settings)

    def test_registered_modules_parameter_ownership_and_storage(self):
        for architecture, mode in MODES:
            p = policy(architecture, mode)
            self.assertIsInstance(p, torch.nn.Module)
            self.assertEqual(set(dict(p.named_children())), {"actor", "critic"})
            actor, critic = tuple(p.actor_parameters()), tuple(p.critic_parameters())
            self.assertEqual({id(v) for v in actor}, {id(v) for v in p.actor.parameters()})
            self.assertEqual({id(v) for v in critic}, {id(v) for v in p.critic.parameters()})
            self.assertFalse({id(v) for v in actor} & {id(v) for v in critic})
            self.assertEqual({id(v) for v in p.parameters()}, {id(v) for v in actor + critic})
            self.assertFalse({v.untyped_storage().data_ptr() for v in actor}
                             & {v.untyped_storage().data_ptr() for v in critic})
            self.assertEqual(p.actor.reference_bias.shape, ())
            self.assertTrue(p.actor.reference_bias.requires_grad)

    def test_actor_and_critic_backward_are_isolated_without_updates(self):
        for architecture, mode in MODES:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(architecture=architecture, mode=mode, dtype=dtype):
                    p, bank = policy(architecture, mode).to(dtype=dtype), candidates()
                    before = p.snapshot_sha256()
                    out = p(observation(dtype), bank)
                    self.assertIsInstance(out, CandidateScores)
                    actor_loss = -class_distribution(out.logits, bank).log_prob(torch.tensor(bank.reference_class))
                    actor_loss.backward()
                    self.assertTrue(all(v.grad is None for v in p.critic_parameters()))
                    self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all()
                                        for v in p.actor_parameters()))
                    self.assertGreater(sum(v.grad.abs().sum().item() for v in p.actor_parameters()), 0.)
                    self.assertNotEqual(p.actor.reference_bias.grad.item(), 0.)
                    p.zero_grad(set_to_none=True)
                    # V starts at zero, so a nonzero invented target exercises its gradient.
                    (out.value - 1.).square().backward()
                    self.assertTrue(all(v.grad is None for v in p.actor_parameters()))
                    self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all()
                                        for v in p.critic_parameters()))
                    self.assertGreater(sum(v.grad.abs().sum().item() for v in p.critic_parameters()), 0.)
                    self.assertEqual(before, p.snapshot_sha256())

    def test_seeds_are_independent_and_actor_matches_old_random_scorer(self):
        for architecture, mode in MODES:
            p = policy(architecture, mode, initial_reference_bias=0.)
            other_critic = policy(architecture, mode, critic_seed=17, initial_reference_bias=0.)
            other_actor = policy(architecture, mode, actor_seed=17, initial_reference_bias=0.)
            self.assert_same_state(p.actor, other_critic.actor)
            self.assert_same_state(p.critic, other_actor.critic)
            self.assertFalse(torch.equal(p.critic.value_head[0].weight, other_critic.critic.value_head[0].weight))
            self.assertFalse(torch.equal(p.actor.score_head[0].weight, other_actor.actor.score_head[0].weight))
            old = CandidatePolicy(schema(), enabled=True, architecture=architecture, message_mode=mode,
                                  encoder_width=3, head_width=5, seed=271)
            self.assert_same_state(p.actor.encoder.projection, old.encoder)
            self.assert_same_state(p.actor.score_head, old.score_head)
            self.assertTrue(torch.equal(p(observation(), candidates()).logits,
                                        old(observation(), candidates()).logits))
            self.assertEqual(p(observation(), candidates()).value.item(), 0.)
            self.assertEqual(torch.count_nonzero(p.critic.value_head[-1].weight).item(), 0)

    def test_global_rng_and_default_dtype_are_preserved(self):
        original_dtype = torch.get_default_dtype()
        before_torch = torch.get_rng_state().clone()
        before_numpy, before_python = np.random.get_state(), random.getstate()
        expected = policy()
        try:
            torch.set_default_dtype(torch.float64)
            actual = policy()
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            self.assert_same_state(actual, expected)
            actual(observation(), candidates()).logits.sum().backward()
            actual.manifest(), actual.definition_sha256(), actual.snapshot_sha256()
        finally:
            torch.set_default_dtype(original_dtype)
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        for left, right in zip(before_numpy, np.random.get_state()):
            np.testing.assert_equal(left, right)

    def test_permutations_and_aliases_do_not_change_logits_or_probability(self):
        for architecture, mode in MODES:
            p, bank = policy(architecture, mode), candidates()
            expected = p(observation(), bank)
            for options in itertools.permutations(bank.requests[2:]):
                other = candidates(option_requests=options)
                self.assertEqual(bank.class_keys, other.class_keys)
                actual = p(observation(), other)
                self.assertTrue(torch.equal(expected.logits, actual.logits))
                self.assertTrue(torch.equal(expected.value, actual.value))
            smaller = candidates(option_requests=[request(.125), request(-.125)])
            actual = p(observation(), smaller)
            self.assertTrue(torch.equal(expected.logits, actual.logits))
            self.assertTrue(torch.equal(class_distribution(expected.logits, bank).probs,
                                        class_distribution(actual.logits, smaller).probs))

    def test_dynamic_support_and_singleton_keep_parameter_meanings(self):
        for architecture, mode in MODES:
            p = policy(architecture, mode)
            before = p.snapshot_sha256()
            bank = candidates(option_requests=[request(.25)])
            expanded = candidates(option_requests=[request(.25), request(-.5), request(.75)])
            a, b = p(observation(), bank), p(observation(), expanded)
            self.assertEqual(a.logits.shape, (2,))
            self.assertEqual(b.logits.shape, (4,))
            for i, key in enumerate(bank.class_keys):
                torch.testing.assert_close(a.logits[i], b.logits[expanded.class_keys.index(key)])
            one = candidates(option_requests=[])
            out = p(observation(), one)
            self.assertEqual(out.logits.shape, (1,))
            self.assertEqual(out.value.shape, ())
            self.assertEqual(class_distribution(out.logits, one).probs.item(), 1.)
            self.assertEqual(before, p.snapshot_sha256())

    def test_exact_double_requests_are_not_replaced_by_network_features(self):
        x = .125 - 1e-9
        bank = candidates(reference_request=request(x),
                          option_requests=[request(.125), request(np.nextafter(.375, 1.))])
        sealed = bank.sha256
        for dtype in (torch.float32, torch.float64):
            p = policy().to(dtype=dtype)
            out = p(observation(dtype), bank)
            self.assertEqual(out.logits.dtype, dtype)
            self.assertEqual(out.logits.numel(), len(bank.class_keys))
            for i in range(len(bank.class_keys)):
                choice = choose_candidate(bank, i)
                self.assertEqual(choice.submitted_request, bank.requests[bank.representatives[i]])
                validate_choice_precision(choice, bank, replay_dtype=np.float64)
            self.assertEqual(bank.sha256, sealed)
            choice = choose_candidate(bank, bank.reference_class)
            self.assertEqual(choice.submitted_request[0], x)
            self.assertNotEqual(choice.submitted_request, bank.class_features[bank.reference_class])
            with self.assertRaisesRegex(ValueError, "precision conversion"):
                validate_choice_precision(choice, bank, replay_dtype=np.float32)

    def test_reference_bias_is_trainable_not_a_fixed_prior_or_clamp(self):
        p, bank = policy(initial_reference_bias=.1), candidates()
        expected_manifest = p.manifest()
        with torch.no_grad():
            for parameter in p.actor.parameters():
                parameter.zero_()
            p.actor.reference_bias.fill_(-100.)
            p.actor.score_head[-1].bias.fill_(40.)
        out = p(observation(), bank)
        expected = torch.full_like(out.logits, 40.)
        expected[bank.reference_class] = -60.
        self.assertTrue(torch.equal(out.logits, expected))
        out.logits.sum().backward()
        self.assertEqual(p.actor.reference_bias.grad.item(), 1.)
        self.assertEqual(expected_manifest, p.manifest())

    def test_common_public_inputs_and_role_indicators(self):
        bank = candidates(anchor_request=request(-.25))
        raw, states = {}, []
        for architecture, mode in MODES:
            p = policy(architecture, mode)
            captured = {}
            hooks = [p.actor.score_head[0].register_forward_pre_hook(
                         lambda module, args: captured.update(actor=args[0].detach().clone())),
                     p.critic.value_head[0].register_forward_pre_hook(
                         lambda module, args: captured.update(critic=args[0].detach().clone()))]
            try:
                states.append(p(observation(), bank).actor_state)
            finally:
                for hook in hooks:
                    hook.remove()
            base_width = captured["critic"].shape[1]
            torch.testing.assert_close(captured["actor"][:, 6:base_width],
                                       captured["critic"][:, 6:].expand(len(bank.class_keys), -1))
            torch.testing.assert_close(captured["actor"][:, base_width:base_width + 8],
                                       torch.tensor(bank.class_features))
            roles = [[i == bank.reference_class, i == bank.anchor_class] for i in range(len(bank.class_keys))]
            self.assertTrue(torch.equal(captured["actor"][:, -2:], torch.tensor(roles).float()))
            raw[(architecture, mode)] = captured["actor"][:, 6:]
        self.assertEqual(states, [states[0]] * len(MODES))
        for key in MODES[1:]:
            self.assertTrue(torch.equal(raw[MODES[0]], raw[key]))

    def test_critic_is_independent_of_support_with_nonzero_value_head(self):
        p = policy()
        with torch.no_grad():
            p.critic.value_head[-1].weight.fill_(.125)
        baseline = p(observation(), candidates())
        for bank in (candidates(option_requests=[]), candidates(option_requests=[request(.9)]),
                     candidates(state_token="different-seal")):
            self.assertTrue(torch.equal(baseline.value, p(observation(), bank).value))
        changed = replace(observation(), globals=torch.tensor([[.9]]))
        self.assertFalse(torch.equal(baseline.value, p(changed, candidates()).value))
        for bank in (candidates(reference_request=request(.75)), candidates(anchor_request=request(.75))):
            self.assertFalse(torch.equal(baseline.value, p(observation(), bank).value))

    def test_self_only_keeps_physical_metadata_and_empty_globals_work(self):
        p, self_only = policy(), policy(message_mode="self_only")
        self.assert_same_state(p, self_only)
        bank, obs = candidates(), observation()
        self.assertFalse(torch.equal(p(obs, bank).logits, self_only(obs, bank).logits))
        no_links = replace(obs, physical_links=torch.zeros_like(obs.physical_links))
        self.assertFalse(torch.equal(self_only(obs, bank).logits, self_only(no_links, bank).logits))
        empty = replace(schema(), global_feature_names=())
        p = DynamicCandidatePolicy(empty, enabled=True, architecture="flat", message_mode="self_only",
                                   encoder_width=2, head_width=3, actor_seed=0, critic_seed=0,
                                   initial_reference_bias=0.)
        out = p(replace(obs, schema=empty, globals=torch.empty(1, 0)), bank)
        self.assertEqual(len(out.actor_state), 14)

    def test_manifest_exact_initialization_counts_and_freeze_stability(self):
        p = policy(initial_reference_bias=.123456789)
        manifest = p.manifest()
        init = manifest["initialization"]
        self.assertEqual(init["actor_seed"], 271)
        self.assertEqual(init["critic_seed"], 811)
        self.assertEqual(init["initial_reference_bias"], .123456789)
        self.assertEqual(init["initial_reference_bias_float32"], float(np.float32(.123456789)))
        self.assertEqual(manifest["parameter_counts"], {"actor": 203, "critic": 152, "total": 355, "shared": 0})
        self.assertEqual(manifest["parameter_counts"]["total"], sum(v.numel() for v in p.parameters()))
        flat = policy("flat", "self_only").manifest()
        self.assertNotEqual(manifest["parameter_counts"], flat["parameter_counts"])
        self.assertFalse(manifest["graph_flat_parameter_counts_matched"])
        json.dumps(manifest, allow_nan=False)
        definition, snapshot = p.definition_sha256(), p.snapshot_sha256()
        p.requires_grad_(False).eval()
        self.assertEqual(manifest, p.manifest())
        self.assertEqual(definition, p.definition_sha256())
        self.assertEqual(snapshot, p.snapshot_sha256())

    def test_copy_restore_and_hashes_bind_definition_and_state(self):
        p = policy().double()
        definition, before = p.definition_sha256(), p.snapshot_sha256()
        state = copy.deepcopy(p.state_dict())
        original = p(observation(torch.float64), candidates())
        clone = copy.deepcopy(p).requires_grad_(False)
        self.assertEqual(clone.snapshot_sha256(), before)
        self.assertFalse({v.untyped_storage().data_ptr() for v in p.parameters()}
                         & {v.untyped_storage().data_ptr() for v in clone.parameters()})
        with torch.no_grad():
            p.actor.reference_bias.add_(1.)
            p.critic.value_head[-1].bias.add_(.5)
        self.assertEqual(definition, p.definition_sha256())
        self.assertNotEqual(before, p.snapshot_sha256())
        p.load_state_dict(state)
        self.assertEqual(before, p.snapshot_sha256())
        fresh = policy().double()
        fresh.load_state_dict(state)
        self.assertEqual(before, fresh.snapshot_sha256())
        for restored in (p, clone, fresh):
            actual = restored(observation(torch.float64), candidates())
            self.assertTrue(torch.equal(original.logits, actual.logits))
            self.assertTrue(torch.equal(original.value, actual.value))
            self.assertEqual(original.actor_state, actual.actor_state)
        definitions = {policy().definition_sha256(), policy(message_mode="self_only").definition_sha256(),
                       policy(actor_seed=1).definition_sha256(), policy(critic_seed=1).definition_sha256(),
                       policy(initial_reference_bias=.2).definition_sha256(), policy().double().definition_sha256()}
        self.assertEqual(len(definitions), 6)

    def test_schema_shapes_and_precision_are_explicit(self):
        p, obs, bank = policy(), observation(), candidates()
        for invalid in (None, {"nodes": obs.nodes}):
            with self.assertRaises(TypeError):
                p(invalid, bank)
        for invalid in (None, bank.requests):
            with self.assertRaises(TypeError):
                p(obs, invalid)
        with self.assertRaisesRegex(ValueError, "schemas"):
            p(obs, replace(bank, schema=replace(bank.schema, action_schema_id="other/action")))
        bad_observations = (
            replace(obs, schema=replace(schema(), node_ids=("B", "A"))),
            observation(torch.float64), replace(obs, globals=obs.globals.double()),
            replace(obs, nodes=obs.nodes.squeeze(0)), replace(obs, nodes=obs.nodes.long()),
            replace(obs, nodes=obs.nodes.expand(2, -1, -1), globals=obs.globals.expand(2, -1),
                    physical_links=obs.physical_links.expand(2, -1, -1)),
            replace(obs, physical_links=torch.tensor([[[0., 1.], [0., 0.]]])),
            replace(obs, physical_links=torch.eye(2).unsqueeze(0)),
        )
        for invalid in bad_observations:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                p(invalid, bank)
        for invalid in (policy().half(), policy().to("meta"), policy()):
            if next(invalid.parameters()).dtype == torch.float32 and next(invalid.parameters()).device.type == "cpu":
                invalid.critic.double()
            with self.assertRaisesRegex(ValueError, "uniform CPU float32/float64"):
                invalid(obs, bank)

    def test_nonfinite_input_parameters_outputs_and_overflow_fail(self):
        for field in ("nodes", "globals", "physical_links"):
            tensor = getattr(observation(), field).clone()
            tensor.flatten()[0] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                policy()(replace(observation(), **{field: tensor}), candidates())
        for target in ("actor", "critic"):
            for number in (float("nan"), float("inf")):
                p = policy()
                with torch.no_grad():
                    next(getattr(p, target).parameters()).fill_(number)
                for action in (lambda: p(observation(), candidates()), p.snapshot_sha256, p.manifest):
                    with self.assertRaisesRegex(ValueError, "nonfinite"):
                        action()
        p = policy()
        with torch.no_grad():
            p.actor.encoder.projection.linear.weight.fill_(torch.finfo(torch.float32).max)
        with self.assertRaisesRegex(ValueError, "nonfinite public-state encoding"):
            p(observation(), candidates())
        for target in ("actor", "critic"):
            p = policy()
            module = getattr(p, target)
            hook = module.register_forward_hook(lambda module, args, output: torch.full_like(output, float("inf")))
            try:
                with self.assertRaisesRegex(ValueError, "nonfinite candidate score or value"):
                    p(observation(), candidates())
            finally:
                hook.remove()

    def test_forward_does_not_mutate_inputs_or_bank(self):
        p, obs, bank = policy(), observation(), candidates()
        before = tuple(getattr(obs, field).clone() for field in ("nodes", "globals", "physical_links"))
        seal = bank.sha256
        p(obs, bank)
        self.assertEqual(seal, bank.sha256)
        for field, tensor in zip(("nodes", "globals", "physical_links"), before):
            self.assertTrue(torch.equal(tensor, getattr(obs, field)))

    def test_hidden_head_overflow_is_not_hidden_by_tanh(self):
        for component, name in (("actor", "score_head"), ("critic", "value_head")):
            p = policy()
            head = getattr(getattr(p, component), name)
            with torch.no_grad():
                head[0].weight.zero_()
                # Global time starts after the six encoded node features.
                head[0].weight[:, 6].fill_(torch.finfo(torch.float32).max)
            obs = replace(observation(), globals=torch.tensor([[2.]]))
            with self.assertRaisesRegex(ValueError, "nonfinite scalar head activation"):
                p(obs, candidates())


if __name__ == "__main__":
    unittest.main()
