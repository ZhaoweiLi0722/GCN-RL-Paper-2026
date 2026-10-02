"""Invented public tensors only; all optimizer steps are guards or fake steps.

No scientific artifact loads, environment execution, or scientific fitting.
Fake Adam populates zero moments and counters without applying any gradient.
"""

from dataclasses import asdict, replace
import io
import itertools
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.conservative_cohort_actor import (
    ConservativeCohortActor, ConservativeCohortActorSettings, _conservative_loss,
)
from src.rl.paired_cohort_actor import PairedCohortActor, PairedCohortExample, _actor_logits
from src.rl.paired_cohort_objective import PairedCohortState
from src.rl.prospective_ddpg_kernel import state_digest
from tests.test_candidate_policy_rollout import candidates, contract, observation, request


def prototype():
    return DynamicCandidatePolicy(contract().inputs, enabled=True, architecture="graph",
        message_mode="physical", encoder_width=3, head_width=5, actor_seed=17,
        critic_seed=19, initial_reference_bias=0.)


def settings(**changes):
    return ConservativeCohortActorSettings(**(dict(max_optimizer_steps=2, context_count=2) | changes))


def example(index=0, *, bank=None):
    bank = candidates(index) if bank is None else bank
    costs = torch.tensor([[1e9 + 2e8 * k + r + index for k in range(len(bank.class_keys))]
                          for r in range(4)], dtype=torch.float64)
    seeds = tuple(str(100 + 4 * index + r) for r in range(4))
    return PairedCohortExample(observation(index), bank, costs,
        tuple(seeds for _ in bank.class_keys), bank.reference_class, bank.class_keys)


def references(model, examples):
    with torch.no_grad():
        return tuple(_actor_logits(model, e).double().detach() for e in examples)


def learner(arm="paired_cost", *, examples=None, model=None, options=None,
            reference_logits=None, dataset="a" * 64, round_index=1, enabled=True):
    model = prototype() if model is None else model
    examples = (example(), example(1)) if examples is None else examples
    reference_logits = references(model, examples) if reference_logits is None else reference_logits
    return ConservativeCohortActor(model, contract(), settings() if options is None else options,
        examples, dataset_sha256=dataset, arm=arm, reference_logits=reference_logits,
        enabled=enabled, round_index=round_index)


def fake_adam(optimizer):
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise AssertionError("finite actor gradient required by fake step")
            moments = optimizer.state[parameter]
            if not moments:
                moments.update(step=torch.tensor(0., dtype=torch.float32),
                               exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
            moments["step"].add_(1)


def draw_rng():
    random.random()
    np.random.random()
    torch.rand(2)


def objective_state(logits, costs, ref=1):
    width = logits.numel()
    return PairedCohortState(logits, costs, tuple(f"class-{i}" for i in range(width)),
                            (("100", "101", "102", "103"),) * width, ref)


class ConservativeCohortActorTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(restore_global_rng, global_rng_state())
        for optimizer in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("real optimizer forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def mock_update(self, kernel, **kwargs):
        kwargs.setdefault("before_optimizer_step", lambda _: None)
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            return kernel.update(**kwargs)

    def assert_rollback(self, kernel, before, attempts):
        after = kernel.state_dict()
        for key in ("policy", "optimizer", "global_rng", "history", "manifest", "reference_logits"):
            self.assertEqual(state_digest(after[key]), state_digest(before[key]), key)
        self.assertEqual(after["steps"], before["steps"])
        self.assertEqual(after["attempted_steps"], attempts)
        self.assertIsNotNone(after["failure"])
        for call in (lambda: kernel.update(before_optimizer_step=lambda _: None),
                     lambda: kernel.loss(), lambda: kernel.load_state_dict(before),
                     lambda: learner(kernel.arm).load_state_dict(after)):
            with self.assertRaisesRegex(ValueError, "failed"):
                call()

    def test_fixed_settings_and_caps(self):
        defaults = ConservativeCohortActorSettings()
        self.assertEqual((defaults.max_optimizer_steps, defaults.context_count,
                          defaults.future_replications, defaults.kl_coefficient), (64, 6, 4, .05))
        bad = ({"learning_rate": .001}, {"betas": [.9, .999]}, {"betas": (.9, .99)},
               {"eps": 1e-7}, {"weight_decay": True}, {"max_grad_norm": 1.},
               {"cost_scale": 1.}, {"kl_coefficient": .1}, {"kl_coefficient": float("nan")},
               {"max_optimizer_steps": 65}, {"max_optimizer_steps": True},
               {"max_optimizer_steps": 0}, {"context_count": 7}, {"context_count": 0},
               {"future_replications": 2}, {"future_replications": 4.})
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                settings(**change)

    def test_constructor_has_no_forward_or_rng_draw_and_binds_all_sha(self):
        model, rows = prototype(), tuple(example(i) for i in range(6))
        q = references(model, rows)
        rng = state_digest(global_rng_state())
        with patch.object(DynamicCandidatePolicy, "forward", side_effect=AssertionError("no policy forward")), \
                patch.object(type(model.actor), "forward", side_effect=AssertionError("no actor forward")), \
                patch.object(type(model.critic), "forward", side_effect=AssertionError("no critic forward")):
            kernel = learner(model=model, examples=rows, reference_logits=q,
                             options=ConservativeCohortActorSettings())
            kernel.load_state_dict(kernel.state_dict())
        self.assertEqual(state_digest(global_rng_state()), rng)
        manifest = kernel.manifest
        self.assertEqual(manifest["settings_sha256"], state_digest(asdict(kernel.settings)))
        self.assertEqual(manifest["reference_logits_sha256"], state_digest(q))
        self.assertEqual(manifest["reference_policy_sha256"], model.snapshot_sha256())
        self.assertEqual(manifest["dataset_sha256"], "a" * 64)
        self.assertEqual(kernel.state_dict()["manifest"], manifest)
        self.assertEqual(state_digest(kernel.state_dict()["reference_logits"]), state_digest(q))
        self.assertEqual(kernel.optimizer.state_dict()["state"], {})
        group = kernel.optimizer.param_groups[0]
        self.assertEqual({k: group[k] for k in ("lr", "betas", "eps", "weight_decay", "foreach", "amsgrad")},
                         dict(lr=3e-4, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False, amsgrad=False))
        self.assertIs(ConservativeCohortActor.update, PairedCohortActor.update)
        self.assertIs(ConservativeCohortActor.load_state_dict, PairedCohortActor.load_state_dict)

    def test_reject_bad_enablement_round_dtype_arm_or_dataset(self):
        rows, q = (example(), example(1)), (torch.zeros(3, dtype=torch.float64),) * 2
        for changes in ({"enabled": False}, {"round_index": True}, {"round_index": 0},
                        {"round_index": 3}, {"arm": "ppo"}, {"dataset": "invalid"},
                        {"model": prototype().double()}):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                learner(examples=rows, reference_logits=q, **changes)

    def test_reference_logits_required_even_for_bc_and_reject_incomplete_support(self):
        rows = (example(), example(1))
        q = references(prototype(), rows)
        malformed = (list(q), q[:1], (q[0].float(), q[1]), (q[0][:1], q[1]),
                     (q[0].reshape(1, -1), q[1]), (q[0].to_sparse(), q[1]),
                     (torch.full_like(q[0], float("nan")), q[1]),
                     (torch.tensor([1.7e308, -1.7e308, 0.], dtype=torch.float64), q[1]))
        for arm in ("paired_cost", "bc_continue"):
            with self.assertRaises(TypeError):
                ConservativeCohortActor(prototype(), contract(), settings(), rows,
                    dataset_sha256="a" * 64, arm=arm, enabled=True)
            for value in malformed:
                with self.subTest(arm=arm, value=type(value)), self.assertRaises(ValueError):
                    learner(arm, examples=rows, reference_logits=value)

    def test_reject_incomplete_unpaired_nonfinite_or_filtered_examples(self):
        e, q = example(), (torch.zeros(3, dtype=torch.float64),) * 2
        bad = [replace(e, raw_costs=e.raw_costs.float()), replace(e, raw_costs=e.raw_costs[:2]),
               replace(e, raw_costs=e.raw_costs[:, :1]),
               replace(e, raw_costs=torch.full_like(e.raw_costs, float("inf"))),
               replace(e, raw_costs=torch.full_like(e.raw_costs, 1.7e308)),
               replace(e, reference_index=0), replace(e, class_keys=e.class_keys[::-1]),
               replace(e, replication_seed_ids=((100, 101, 102, 103), (100, 101, 102, 104),
                                               (100, 101, 102, 103))),
               replace(e, replication_seed_ids=((100, 100, 102, 103),) * 3),
               replace(e, observation=replace(e.observation, nodes=e.observation.nodes.double()))]
        for row in bad:
            with self.subTest(row=row.reference_index), self.assertRaises(ValueError):
                learner(examples=(row, example(1)), reference_logits=q)
        for rows in ((e, e), (e,), [e, example(1)]):
            with self.assertRaises(ValueError):
                learner(examples=rows, reference_logits=q)

    def test_float64_loss_and_analytic_gradient_detach_costs_and_q(self):
        logits = torch.tensor([.8, -.7, .2], dtype=torch.float64, requires_grad=True)
        costs = torch.tensor([[2e9, 1e9, -5e10], [3e9, 2e9, 9e10],
                              [4e9, 3e9, 2e9], [8e9, 4e9, -2e9]],
                             dtype=torch.float64, requires_grad=True)
        q = torch.tensor([-.1, .6, -.4], dtype=torch.float64, requires_grad=True)
        state = objective_state(logits, costs)
        p, log_p, log_q = logits.softmax(0), logits.log_softmax(0), q.detach().log_softmax(0)
        means = costs.detach().mean(0)
        a = (means - means[1]) / 1e9
        kl = (p * (log_p - log_q)).sum()
        expected = (p * a).sum() + .05 * kl
        expected_gradient = p * (a - (p * a).sum() + .05 * (log_p - log_q - kl))
        loss = _conservative_loss((state,), (q,), settings(context_count=1))
        self.assertEqual(loss.dtype, torch.float64)
        torch.testing.assert_close(loss, expected, rtol=0, atol=1e-14)
        loss.backward()
        torch.testing.assert_close(logits.grad, expected_gradient.detach(), rtol=1e-12, atol=1e-14)
        self.assertIsNone(costs.grad)
        self.assertIsNone(q.grad)

    def test_candidate_and_context_permutations_preserve_loss_and_gradient(self):
        z = torch.tensor([.8, -.7, .2], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([[1e9, 2e9, 4e9]] * 4, dtype=torch.float64)
        q = torch.tensor([.1, .3, .8], dtype=torch.float64)
        original = objective_state(z, c)
        expected = _conservative_loss((original,), (q,), settings(context_count=1))
        expected.backward()
        for permutation in itertools.permutations(range(3)):
            order = torch.tensor(permutation)
            permuted_z = z.detach()[order].clone().requires_grad_(True)
            state = replace(original, logits=permuted_z, raw_costs=c[:, order],
                reference_index=permutation.index(1),
                class_keys=tuple(original.class_keys[i] for i in permutation),
                replication_seed_ids=tuple(original.replication_seed_ids[i] for i in permutation))
            result = _conservative_loss((state,), (q[order],), settings(context_count=1))
            result.backward()
            torch.testing.assert_close(result, expected)
            torch.testing.assert_close(permuted_z.grad, z.grad[order])
        model, rows = prototype(), (example(), example(1))
        qs = references(model, rows)
        a = learner(model=model, examples=rows, reference_logits=qs)
        b = learner(model=model, examples=rows[::-1], reference_logits=qs[::-1])
        torch.testing.assert_close(a.loss(), b.loss(), rtol=0, atol=0)

    def test_uniform_ragged_mean_and_bc_targets_r4_not_argmin_or_q(self):
        banks = (candidates(0, reference_request=request(.125)),
                 candidates(1, option_requests=[request(.125)]),
                 candidates(2, option_requests=[]))
        rows = tuple(example(i, bank=bank) for i, bank in enumerate(banks))
        self.assertEqual([len(e.class_keys) for e in rows], [3, 2, 1])
        self.assertNotEqual(banks[0].reference_class, banks[0].anchor_class)
        model = prototype()
        q = references(model, rows)
        for arm in ("paired_cost", "bc_continue"):
            kernel = learner(arm, model=model, examples=rows, reference_logits=q,
                             options=settings(context_count=3))
            with torch.no_grad():
                kernel.policy.actor.reference_bias.add_(.8)
            expected = []
            for e, ref in zip(rows, q):
                z = _actor_logits(kernel.policy, e).double()
                p, means = z.softmax(0), e.raw_costs.mean(0)
                expected.append(((p * (means - means[e.bank.reference_class]) / 1e9).sum()
                                 + .05 * (p * (z.log_softmax(0) - ref.log_softmax(0))).sum())
                                if arm == "paired_cost" else -z.log_softmax(0)[e.bank.reference_class])
            with patch.object(kernel.policy.actor, "forward", wraps=kernel.policy.actor.forward) as forwards:
                found = kernel.loss()
            self.assertEqual(forwards.call_count, 3)
            torch.testing.assert_close(found, torch.stack(expected).mean(), rtol=0, atol=1e-15)
        bc = learner("bc_continue", model=model, examples=rows, reference_logits=q,
                     options=settings(context_count=3))
        other_q = tuple(ref + torch.arange(len(ref), dtype=torch.float64) for ref in q)
        other = learner("bc_continue", model=model, examples=rows, reference_logits=other_q,
                        options=settings(context_count=3))
        torch.testing.assert_close(bc.loss(), other.loss(), rtol=0, atol=0)
        self.assertNotEqual(bc.manifest_sha256, other.manifest_sha256)

    def test_equal_costs_leave_kl_only_and_stable_log_q_never_clips(self):
        z = torch.tensor([.5, -.1, .8], dtype=torch.float64, requires_grad=True)
        costs = torch.full((4, 3), 1e9, dtype=torch.float64)
        state = objective_state(z, costs)
        zero = _conservative_loss((state,), (z.detach().clone(),), settings(context_count=1))
        zero.backward()
        torch.testing.assert_close(zero, z.new_tensor(0.), rtol=0, atol=1e-15)
        torch.testing.assert_close(z.grad, torch.zeros_like(z), rtol=0, atol=1e-15)
        q = torch.tensor([0., -2000., -4000.], dtype=torch.float64)
        loss = _conservative_loss((state,), (q,), settings(context_count=1))
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 10.)

    def test_one_unit_cost_difference_survives_float64_cast_and_reaches_actor(self):
        rows = []
        for i in range(2):
            e = example(i)
            costs = torch.full_like(e.raw_costs, 1e9)
            costs[:, e.reference_index] += 1
            rows.append(replace(e, raw_costs=costs))
        kernel = learner(examples=tuple(rows))
        kernel.loss().backward()
        gradient = kernel.policy.actor.reference_bias.grad
        self.assertEqual(gradient.dtype, torch.float32)
        self.assertGreater(gradient.item(), 0.)
        self.assertLess(gradient.item(), 1e-9)

    def test_private_detached_labels_public_inputs_and_q_do_not_refresh(self):
        e, model = example(), prototype()
        e.raw_costs.requires_grad_(True)
        e.observation.nodes.requires_grad_(True)
        rows = (e, example(1))
        q = tuple(v.requires_grad_(True) for v in references(model, rows))
        kernel = learner(model=model, examples=rows, reference_logits=q)
        with patch.object(kernel.policy.critic, "forward", side_effect=AssertionError("critic forbidden")), \
                patch.object(DynamicCandidatePolicy, "forward", side_effect=AssertionError("full policy forbidden")):
            original = kernel.loss()
            original.backward()
        self.assertIsNone(e.raw_costs.grad)
        self.assertIsNone(e.observation.nodes.grad)
        self.assertTrue(all(v.grad is None for v in q))
        self.assertTrue(all(not v.requires_grad and v.grad_fn is None for v in kernel._reference_logits))
        self.assertTrue(all(p.grad is not None for p in kernel.policy.actor_parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in kernel.policy.critic_parameters()))
        with torch.no_grad():
            e.raw_costs.add_(1e9)
            e.observation.nodes.add_(10)
            q[0][0].add_(100)
        torch.testing.assert_close(original.detach(), kernel.loss().detach(), rtol=0, atol=0)
        manifest = kernel.manifest
        manifest["settings"]["kl_coefficient"] = 0.
        self.assertEqual(kernel.manifest["settings"]["kl_coefficient"], .05)

    def test_two_arm_round_lineage_and_one_explicit_adam_reset_per_round(self):
        model, rows = prototype(), (example(), example(1))
        q, initial = references(model, rows), model.snapshot_sha256()
        paired = learner(model=model, examples=rows, reference_logits=q)
        bc = learner("bc_continue", model=model, examples=rows, reference_logits=q)
        self.assertEqual(paired.policy.snapshot_sha256(), initial)
        self.assertEqual(bc.policy.snapshot_sha256(), initial)
        self.assertEqual(paired.manifest["examples_sha256"], bc.manifest["examples_sha256"])
        self.assertEqual(paired.manifest["reference_logits_sha256"], bc.manifest["reference_logits_sha256"])
        for index, kernel in enumerate((paired, bc), 1):
            def invented_step(opt):
                fake_adam(opt)
                with torch.no_grad():
                    opt.param_groups[0]["params"][0].add_(index * .25)

            with patch.object(torch.optim.Adam, "step", new=invented_step):
                kernel.update(before_optimizer_step=lambda _: None)
            next_rows = (example(2), example(3))
            next_round = learner(kernel.arm, model=kernel.policy, examples=next_rows, round_index=2)
            self.assertEqual(next_round.policy.snapshot_sha256(), kernel.policy.snapshot_sha256())
            self.assertNotEqual(next_round.policy.snapshot_sha256(), initial)
            self.assertEqual(next_round.optimizer.state_dict()["state"], {})
            self.assertTrue(kernel.optimizer.state_dict()["state"])
            self.assertEqual(next_round.steps, 0)
            reset = [{"round_index": 2, "reason": "round_start", "reset_count": 1, "carried_moments": False}]
            self.assertEqual(next_round.manifest["optimizer_reset_log"], reset)
            with patch.object(torch.optim.Adam, "__init__", side_effect=AssertionError("no additional reset")):
                self.mock_update(next_round)
                next_round.load_state_dict(next_round.state_dict())
            self.assertEqual(next_round.state_dict()["manifest"]["optimizer_reset_log"], reset)
        self.assertNotEqual(paired.policy.snapshot_sha256(), bc.policy.snapshot_sha256())
        self.assertEqual(model.snapshot_sha256(), initial)
        storages = lambda policy: {p.data_ptr() for p in policy.parameters()}
        self.assertFalse(storages(paired.policy) & storages(bc.policy))
        self.assertFalse(storages(paired.policy) & storages(model))

    def test_fake_updates_charge_before_step_clip_gradients_and_stop_at_cap(self):
        for arm in ("paired_cost", "bc_continue"):
            kernel, events, norms = learner(arm), [], []
            initial, rng = kernel.policy.snapshot_sha256(), state_digest(global_rng_state())
            q_sha = kernel.manifest["reference_logits_sha256"]

            def step(opt):
                events.append(("step", kernel.attempted_steps))
                norms.append(torch.linalg.vector_norm(torch.cat(
                    [p.grad.flatten() for group in opt.param_groups for p in group["params"]])).item())
                fake_adam(opt)

            def charge(owner):
                events.append((owner, kernel.attempted_steps))

            with patch.object(torch.optim.Adam, "step", new=step):
                for _ in range(2):
                    report = kernel.update(before_optimizer_step=charge)
            self.assertEqual(events, [("actor", 1), ("step", 1), ("actor", 2), ("step", 2)])
            self.assertLessEqual(max(norms), .5 + 1e-6)
            if arm == "bc_continue":
                self.assertGreater(report["actor_grad_norm_before_clip"], .5)
            self.assertEqual((report["optimizer_steps"], report["attempted_optimizer_steps"],
                              report["critic_optimizer_steps"]), (2, 2, 0))
            self.assertEqual(kernel.policy.snapshot_sha256(), initial)
            self.assertEqual(state_digest(global_rng_state()), rng)
            self.assertEqual(state_digest(kernel._reference_logits), q_sha)
            self.assertFalse(hasattr(kernel, "critic_optimizer"))
            owned = [id(p) for g in kernel.optimizer.param_groups for p in g["params"]]
            self.assertEqual(owned, [id(p) for p in kernel.policy.actor_parameters()])
            self.assertTrue(all(p.grad is None for p in kernel.policy.parameters()))
            before = state_digest(kernel.state_dict())
            with self.assertRaisesRegex(ValueError, "cap"):
                kernel.update(before_optimizer_step=lambda _: self.fail("unexpected charge"))
            self.assertEqual(state_digest(kernel.state_dict()), before)

    def test_hook_failure_or_interrupt_is_nonrefundable_with_rng_rollback(self):
        for error in (RuntimeError("debit then failure"), KeyboardInterrupt()):
            kernel, charges = learner(), []
            before = kernel.state_dict()

            def charge(owner):
                charges.append(owner)
                draw_rng()
                raise error

            with self.assertRaises(type(error)):
                kernel.update(before_optimizer_step=charge)
            self.assertEqual(charges, ["actor"])
            self.assert_rollback(kernel, before, 1)
            self.assertFalse(kernel.state_dict()["failure"]["optimizer_step_returned"])

    def test_partial_fake_step_failure_rolls_back_weights_optimizer_and_rng(self):
        kernel = learner()
        self.mock_update(kernel)
        before = kernel.state_dict()

        def explode(opt):
            fake_adam(opt)
            with torch.no_grad():
                opt.param_groups[0]["params"][0].add_(12.)
            draw_rng()
            raise RuntimeError("injected partial mutation")

        with patch.object(torch.optim.Adam, "step", new=explode), self.assertRaisesRegex(RuntimeError, "partial"):
            kernel.update(before_optimizer_step=lambda _: None)
        self.assert_rollback(kernel, before, 2)

    def test_post_step_failure_and_pre_step_compute_failure_are_terminal(self):
        for post_step in (False, True):
            kernel, called = learner(), []
            before = kernel.state_dict()

            def step(opt):
                fake_adam(opt)
                called.append(1)

            def check():
                if not post_step or called:
                    draw_rng()
                    raise TimeoutError("compute deadline")

            with patch.object(torch.optim.Adam, "step", new=step), self.assertRaises(TimeoutError):
                kernel.update(before_optimizer_step=lambda _: None, before_compute=check)
            self.assert_rollback(kernel, before, int(post_step))
            self.assertEqual(kernel.state_dict()["failure"]["optimizer_step_returned"], post_step)

    def test_internal_q_labels_critic_settings_or_round_tamper_stops_before_charge(self):
        for kind in ("q", "labels", "critic", "settings", "round", "q_gradient"):
            kernel = learner("bc_continue")
            if kind == "q":
                kernel._reference_logits[0][0] += 1
            elif kind == "labels":
                kernel._examples[0].raw_costs[0, 0] += 1
            elif kind == "critic":
                with torch.no_grad():
                    next(kernel.policy.critic_parameters()).add_(1)
            elif kind == "settings":
                kernel.settings = settings(max_optimizer_steps=1)
            elif kind == "round":
                kernel.round_index = 2
            else:
                kernel._reference_logits[0].requires_grad_(True)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                kernel.update(before_optimizer_step=lambda _: self.fail("unexpected charge"))
            self.assertEqual(kernel.attempted_steps, 0)
            self.assertIsNotNone(kernel.state_dict()["failure"])

    def test_nonfinite_gradient_is_terminal_before_charge(self):
        kernel = learner()
        before = kernel.state_dict()

        def invalid_loss(candidate, **_):
            return torch.sqrt((next(candidate.policy.actor_parameters()) * 0.).sum())

        with patch.object(ConservativeCohortActor, "loss", new=invalid_loss), self.assertRaisesRegex(ValueError, "gradient"):
            kernel.update(before_optimizer_step=lambda _: self.fail("unexpected charge"))
        self.assert_rollback(kernel, before, 0)

    def test_full_restore_roundtrip_rng_and_detached_checkpoint(self):
        for arm in ("paired_cost", "bc_continue"):
            kernel = learner(arm)
            self.mock_update(kernel)
            state = kernel.state_dict()
            clone = learner(arm)
            draw_rng()
            clone.load_state_dict(state)
            self.assertEqual(state_digest(clone.state_dict()), state_digest(state))
            with io.BytesIO() as buffer:
                torch.save(state, buffer)
                buffer.seek(0)
                decoded = torch.load(buffer, weights_only=True)
            clone.load_state_dict(decoded)
            state["reference_logits"][0][0] += 3
            state["policy"][next(iter(state["policy"]))].add_(2)
            state["optimizer"]["param_groups"][0]["lr"] = 1.
            self.assertEqual(state_digest(clone.state_dict()), state_digest(decoded))

    def test_restore_rejects_binding_changes_rewinds_and_tampered_q_atomically(self):
        kernel = learner()
        old = kernel.state_dict()
        self.mock_update(kernel)
        before = state_digest(kernel.state_dict())
        states = [old]
        for kind in ("q", "q_missing", "q_float", "dataset", "settings_sha", "q_sha", "reset_log",
                     "steps", "history", "same_counter_policy"):
            state = kernel.state_dict()
            if kind == "q":
                state["reference_logits"][0][0] += 1
            elif kind == "q_missing":
                del state["reference_logits"]
            elif kind == "q_float":
                state["reference_logits"] = tuple(v.float() for v in state["reference_logits"])
            elif kind in ("dataset", "settings_sha", "q_sha"):
                key = {"dataset": "dataset_sha256", "settings_sha": "settings_sha256",
                       "q_sha": "reference_logits_sha256"}[kind]
                state["manifest"][key] = "b" * 64
            elif kind == "reset_log":
                state["manifest"]["optimizer_reset_log"][0]["reset_count"] = 2
            elif kind == "steps":
                state["attempted_steps"] = 0
            elif kind == "history":
                state["history"][0]["loss"] += 1
            else:
                state["policy"][next(iter(state["policy"]))].add_(1)
            states.append(state)
        for state in states:
            with self.assertRaises(ValueError):
                kernel.load_state_dict(state)
            self.assertEqual(state_digest(kernel.state_dict()), before)
        for target in (learner("bc_continue"), learner(dataset="b" * 64), learner(round_index=2),
                       learner(reference_logits=tuple(q + 1 for q in old["reference_logits"]))):
            snapshot = state_digest(target.state_dict())
            with self.assertRaisesRegex(ValueError, "binding"):
                target.load_state_dict(kernel.state_dict())
            self.assertEqual(state_digest(target.state_dict()), snapshot)

    def test_restore_rejects_partial_loading_bad_adam_critic_or_rng_atomically(self):
        source = learner()
        self.mock_update(source)
        for kind in ("critic", "shape", "dtype", "adam_step", "adam_negative", "adam_nonfinite",
                     "groups", "rng_keys", "numpy_rng", "torch_rng", "history"):
            target, state = learner(), source.state_dict()
            before = state_digest(target.state_dict())
            moments = next(iter(state["optimizer"]["state"].values()))
            if kind == "critic":
                state["policy"][next(k for k in state["policy"] if k.startswith("critic."))].add_(1)
            elif kind in ("shape", "dtype"):
                key = next(iter(state["policy"]))
                state["policy"][key] = torch.ones(999) if kind == "shape" else state["policy"][key].double()
            elif kind == "adam_step":
                moments["step"].add_(1)
            elif kind == "adam_negative":
                moments["exp_avg_sq"].fill_(-1)
            elif kind == "adam_nonfinite":
                moments["exp_avg"].fill_(float("inf"))
            elif kind == "groups":
                state["optimizer"]["param_groups"][0]["lr"] = .001
            elif kind == "rng_keys":
                state["global_rng"]["extra"] = 1
            elif kind == "numpy_rng":
                state["global_rng"]["numpy"] = ("wrong",)
            elif kind == "torch_rng":
                state["global_rng"]["torch"] = torch.zeros(1, dtype=torch.uint8)
            else:
                state["history"][0]["actor_grad_norm_before_clip"] = -1
            with self.subTest(kind=kind), self.assertRaises((ValueError, TypeError, RuntimeError)):
                target.load_state_dict(state)
            self.assertEqual(state_digest(target.state_dict()), before)

    def test_invalid_callbacks_are_atomic_and_do_not_spend_attempts(self):
        kernel = learner()
        before = state_digest(kernel.state_dict())
        with self.assertRaises(TypeError):
            kernel.update()
        for kwargs in ({"before_optimizer_step": None},
                       {"before_optimizer_step": lambda _: None, "before_compute": None}):
            with self.assertRaises(ValueError):
                kernel.update(**kwargs)
        self.assertEqual(state_digest(kernel.state_dict()), before)


if __name__ == "__main__":
    unittest.main()
