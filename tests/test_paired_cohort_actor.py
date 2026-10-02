"""Invented public tensors/models only; every real optimizer step is forbidden.

The successful mock only creates zero Adam moments and increments counters.
It never applies gradients. No scientific artifact, patient engine or fit runs.
"""

import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.paired_cohort_actor import (
    PairedCohortActor, PairedCohortActorSettings, PairedCohortExample, _actor_logits,
)
from src.rl.prospective_ddpg_kernel import state_digest
from tests.test_candidate_policy_rollout import candidates, contract, observation, request


def prototype():
    return DynamicCandidatePolicy(contract().inputs, enabled=True, architecture="graph",
        message_mode="physical", encoder_width=3, head_width=5, actor_seed=17,
        critic_seed=19, initial_reference_bias=0.)


def settings(**changes):
    return PairedCohortActorSettings(**(dict(learning_rate=3e-4, betas=(.9, .999), eps=1e-8,
        weight_decay=0., max_grad_norm=.5, cost_scale=1e9, max_optimizer_steps=2,
        context_count=2, future_replications=2) | changes))


def example(index=0, *, bank=None, costs=None):
    bank = candidates(index) if bank is None else bank
    if costs is None:
        costs = torch.tensor([[1e9 + 2 * i + index for i in range(len(bank.class_keys))],
                              [1e9 + 2 * i + index + 1 for i in range(len(bank.class_keys))]],
                             dtype=torch.float64)
    return PairedCohortExample(observation(index), bank, costs,
        tuple((100 + index * 2, 101 + index * 2) for _ in bank.class_keys),
        bank.reference_class, bank.class_keys)


def learner(arm="paired_cost", *, examples=None, model=None, options=None, dataset="a" * 64):
    return PairedCohortActor(prototype() if model is None else model, contract(),
        settings() if options is None else options,
        (example(0), example(1)) if examples is None else examples,
        dataset_sha256=dataset, arm=arm, enabled=True)


def fake_adam(optimizer, *args, **kwargs):
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise AssertionError("finite actor gradient required by fake step")
            moments = optimizer.state[parameter]
            if not moments:
                moments.update(step=torch.tensor(0., dtype=torch.float32),
                               exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
            moments["step"].add_(1)


def draw_global_rng():
    random.random()
    np.random.random()
    torch.rand(2)


class PairedCohortActorTests(unittest.TestCase):
    def setUp(self):
        rng = global_rng_state()
        self.addCleanup(restore_global_rng, rng)
        for optimizer in (torch.optim.Adam, torch.optim.SGD, torch.optim.AdamW):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("real optimizer forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def mock_update(self, kernel, charges=None, **kwargs):
        charges = [] if charges is None else charges
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            return kernel.update(before_optimizer_step=charges.append, **kwargs)

    def assert_rollback(self, kernel, before, *, attempts):
        after = kernel.state_dict()
        for key in ("policy", "optimizer", "global_rng", "history", "manifest"):
            self.assertEqual(state_digest(after[key]), state_digest(before[key]), key)
        self.assertEqual(after["steps"], before["steps"])
        self.assertEqual(after["attempted_steps"], attempts)
        self.assertIsNotNone(after["failure"])
        for call in (lambda: kernel.update(before_optimizer_step=lambda _: None),
                     lambda: kernel.load_state_dict(before), lambda: kernel.loss(),
                     lambda: learner(kernel.arm).load_state_dict(after)):
            with self.assertRaisesRegex(ValueError, "failed"):
                call()

    def test_config_adapter_exact_protocol_and_no_constructor_forward(self):
        path = Path(__file__).resolve().parents[1] / "experiments/configs/paired_cohort_improvement_20261002.json"
        config = json.loads(path.read_text())
        configured = PairedCohortActorSettings.from_config(config)
        self.assertEqual((configured.max_optimizer_steps, configured.context_count), (128, 12))
        model = prototype()
        rng = state_digest(global_rng_state())
        with patch.object(DynamicCandidatePolicy, "forward", side_effect=AssertionError("no forward")), \
                patch.object(type(model.actor), "forward", side_effect=AssertionError("no actor forward")):
            kernel = learner(model=model, options=configured, examples=tuple(example(i) for i in range(12)))
        self.assertEqual(state_digest(global_rng_state()), rng)
        self.assertEqual(kernel.optimizer.state_dict()["state"], {})
        group = kernel.optimizer.param_groups[0]
        self.assertEqual({k: group[k] for k in ("lr", "betas", "eps", "weight_decay", "foreach", "amsgrad")},
                         dict(lr=3e-4, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False, amsgrad=False))
        bad = copy.deepcopy(config)
        bad["actor_updates_per_arm_block"] = 129
        with self.assertRaises(ValueError):
            PairedCohortActorSettings.from_config(bad)

    def test_settings_reject_searches_invalid_caps_and_precision(self):
        for change in ({"learning_rate": .001}, {"betas": (.9, .99)}, {"eps": 1e-7},
                       {"weight_decay": .1}, {"max_grad_norm": 1.}, {"cost_scale": 1.},
                       {"max_optimizer_steps": 129}, {"max_optimizer_steps": True},
                       {"context_count": 0}, {"future_replications": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                settings(**change)
        for kwargs in ({"model": prototype().double()}, {"dataset": "not-a-hash"}, {"arm": "ppo"}):
            with self.subTest(kwargs=tuple(kwargs)), self.assertRaises(ValueError):
                learner(**kwargs)
        with self.assertRaises(ValueError):
            PairedCohortActor(prototype(), contract(), settings(), (example(0), example(1)),
                              dataset_sha256="a" * 64, arm="paired_cost")

    def test_same_initializer_independent_forks_fresh_actor_only_optimizers(self):
        model = prototype()
        original = model.snapshot_sha256()
        paired, bc = learner(model=model), learner("bc_continue", model=model)
        self.assertEqual(paired.policy.snapshot_sha256(), original)
        self.assertEqual(bc.policy.snapshot_sha256(), original)
        self.assertEqual(paired.manifest["examples_sha256"], bc.manifest["examples_sha256"])
        self.assertEqual(paired.manifest["dataset_sha256"], "a" * 64)
        for kernel in (paired, bc):
            owned = {id(p) for g in kernel.optimizer.param_groups for p in g["params"]}
            self.assertEqual(owned, {id(p) for p in kernel.policy.actor_parameters()})
            self.assertFalse(owned & {id(p) for p in kernel.policy.critic_parameters()})
            self.assertTrue(kernel.policy.actor.reference_bias.requires_grad)
            self.assertFalse(any(p.requires_grad for p in kernel.policy.critic_parameters()))
            self.assertFalse(hasattr(kernel, "critic_optimizer"))
            self.assertEqual(kernel.optimizer.state_dict()["state"], {})
        storages = lambda m: {p.data_ptr() for p in m.parameters()}
        self.assertFalse(storages(paired.policy) & storages(bc.policy))
        self.assertFalse(storages(paired.policy) & storages(model))

    def test_reject_incomplete_nonfinite_misaligned_or_unpaired_examples(self):
        e = example()
        bad = [replace(e, raw_costs=e.raw_costs.float()),
               replace(e, raw_costs=e.raw_costs[:, :1]),
               replace(e, raw_costs=torch.full_like(e.raw_costs, float("nan"))),
               replace(e, raw_costs=torch.full_like(e.raw_costs, 1.7e308)),
               replace(e, reference_index=(e.reference_index + 1) % len(e.class_keys)),
               replace(e, class_keys=e.class_keys[::-1]),
               replace(e, replication_seed_ids=((100, 101), (100, 102), (100, 101))),
               replace(e, replication_seed_ids=((100, 100),) * len(e.class_keys)),
               replace(e, replication_seed_ids=((True, 101),) * len(e.class_keys)),
               replace(e, replication_seed_ids=[(100, 101)] * len(e.class_keys)),
               replace(e, observation=replace(e.observation, nodes=e.observation.nodes.double())),
               replace(e, observation=replace(e.observation, nodes=e.observation.nodes[:, :1])),
               replace(e, observation=replace(e.observation, physical_links=torch.ones(1, 2, 2))),
               replace(e, observation={"public": e.observation, "hidden": 1})]
        for row in bad:
            with self.subTest(row=type(row.raw_costs)), self.assertRaises((ValueError, TypeError)):
                learner(examples=(row, example(1)))
        for rows in ((e, e), (e,), [], iter((e, example(1)))):
            with self.subTest(rows=type(rows)), self.assertRaises(ValueError):
                learner(examples=rows)

    def test_sources_are_detached_private_copies_and_mutation_does_not_refresh_data(self):
        e = example()
        e.raw_costs.requires_grad_(True)
        e.observation.nodes.requires_grad_(True)
        kernel = learner(examples=(e, example(1)))
        loss = kernel.loss()
        loss.backward()
        self.assertIsNone(e.raw_costs.grad)
        self.assertIsNone(e.observation.nodes.grad)
        before = kernel.loss().detach().clone()
        with torch.no_grad():
            e.raw_costs.add_(1e8)
            e.observation.nodes.add_(100)
        torch.testing.assert_close(before, kernel.loss().detach(), rtol=0, atol=0)
        manifest = kernel.manifest
        manifest["dataset_sha256"] = "b" * 64
        self.assertEqual(kernel.manifest["dataset_sha256"], "a" * 64)

    def test_actor_path_matches_existing_policy_without_critic_forward(self):
        kernel = learner()
        e = example()
        expected = kernel.policy(e.observation, e.bank).logits
        with patch.object(kernel.policy.critic, "forward", side_effect=AssertionError("critic forbidden")), \
                patch.object(DynamicCandidatePolicy, "forward", side_effect=AssertionError("full policy forbidden")):
            torch.testing.assert_close(_actor_logits(kernel.policy, e), expected, rtol=0, atol=0)
            kernel.loss().backward()
        self.assertTrue(all(p.grad is not None for p in kernel.policy.actor_parameters()))
        self.assertTrue(all(p.grad is None for p in kernel.policy.critic_parameters()))

    def test_all_contexts_equal_weight_with_ragged_support_for_both_arms(self):
        bank = candidates(1, option_requests=[request(.125)])
        rows = (example(), example(1, bank=bank))
        self.assertNotEqual(len(rows[0].class_keys), len(rows[1].class_keys))
        for arm in ("paired_cost", "bc_continue"):
            kernel = learner(arm, examples=rows)
            expected = []
            for row in rows:
                logits = _actor_logits(kernel.policy, row).double()
                costs = row.raw_costs.mean(0)
                expected.append((logits.softmax(0) * (costs - costs[row.reference_index]) / 1e9).sum()
                                if arm == "paired_cost" else -logits.log_softmax(0)[row.reference_index])
            with patch.object(kernel.policy.actor, "forward", wraps=kernel.policy.actor.forward) as forward:
                found = kernel.loss()
            self.assertEqual(forward.call_count, 2)
            self.assertEqual(found.dtype, torch.float64)
            torch.testing.assert_close(found, torch.stack(expected).mean(), rtol=0, atol=1e-24)

    def test_one_unit_large_cost_difference_reaches_float32_reference_bias(self):
        e = example()
        costs = torch.full_like(e.raw_costs, 1e9)
        costs[:, e.reference_index] += 1
        kernel = learner(examples=(replace(e, raw_costs=costs), replace(example(1), raw_costs=costs)))
        kernel.loss().backward()
        gradient = kernel.policy.actor.reference_bias.grad
        self.assertEqual(gradient.dtype, torch.float32)
        self.assertGreater(gradient.item(), 0.)
        self.assertLess(gradient.item(), 1e-9)

    def test_equal_costs_zero_paired_gradient_but_bc_reference_gradient_remains(self):
        rows = tuple(replace(example(i), raw_costs=torch.full_like(example(i).raw_costs, 1e9)) for i in (0, 1))
        paired, bc = learner(examples=rows), learner("bc_continue", examples=rows)
        paired.loss().backward()
        self.assertTrue(all(torch.count_nonzero(p.grad).item() == 0 for p in paired.policy.actor_parameters()))
        bc.loss().backward()
        self.assertLess(bc.policy.actor.reference_bias.grad.item(), 0.)

    def test_mock_steps_charge_before_every_step_reach_cap_without_changing_weights(self):
        for arm in ("paired_cost", "bc_continue"):
            kernel = learner(arm)
            initial = kernel.policy.snapshot_sha256()
            events = []

            def step(opt):
                events.append(("step", kernel.attempted_steps))
                fake_adam(opt)

            def charge(owner):
                events.append((owner, kernel.attempted_steps))

            with patch.object(torch.optim.Adam, "step", new=step):
                for _ in range(2):
                    report = kernel.update(before_optimizer_step=charge)
            self.assertEqual(events, [("actor", 1), ("step", 1), ("actor", 2), ("step", 2)])
            self.assertEqual(report["optimizer_steps"], 2)
            self.assertEqual(report["attempted_optimizer_steps"], 2)
            self.assertEqual(report["critic_optimizer_steps"], 0)
            self.assertEqual(report["context_count"], 2)
            self.assertEqual(initial, kernel.policy.snapshot_sha256())
            self.assertTrue(all(p.grad is None for p in kernel.policy.parameters()))
            before = state_digest(kernel.state_dict())
            with self.assertRaisesRegex(ValueError, "cap"):
                kernel.update(before_optimizer_step=lambda _: self.fail("unexpected charge"))
            self.assertEqual(state_digest(kernel.state_dict()), before)

    def test_gradient_cap_applied_before_fake_step(self):
        kernel = learner("bc_continue")
        observed = []

        def step(opt):
            gradients = torch.cat([p.grad.flatten() for g in opt.param_groups for p in g["params"]])
            observed.append(torch.linalg.vector_norm(gradients).item())
            fake_adam(opt)

        with patch.object(torch.optim.Adam, "step", new=step):
            report = kernel.update(before_optimizer_step=lambda _: None)
        self.assertGreater(report["actor_grad_norm_before_clip"], .5)
        self.assertLessEqual(observed[0], .5 + 1e-6)

    def test_hook_failure_is_nonrefundable_even_when_it_records_then_raises(self):
        kernel = learner()
        before = kernel.state_dict()
        charges = []

        def charge(owner):
            charges.append(owner)
            draw_global_rng()
            raise RuntimeError("durable debit then failure")

        with self.assertRaisesRegex(RuntimeError, "durable debit"):
            kernel.update(before_optimizer_step=charge)
        self.assertEqual(charges, ["actor"])
        self.assert_rollback(kernel, before, attempts=1)
        self.assertFalse(kernel.state_dict()["failure"]["optimizer_step_returned"])

    def test_step_exception_rolls_back_actor_optimizer_and_all_cpu_global_rng(self):
        kernel = learner()
        self.mock_update(kernel)
        before = kernel.state_dict()

        def explode(opt):
            fake_adam(opt)
            with torch.no_grad():
                opt.param_groups[0]["params"][0].add_(12.)
            draw_global_rng()
            raise RuntimeError("injected partial optimizer mutation")

        with patch.object(torch.optim.Adam, "step", new=explode), self.assertRaisesRegex(RuntimeError, "partial"):
            kernel.update(before_optimizer_step=lambda _: None)
        self.assert_rollback(kernel, before, attempts=2)

    def test_post_step_failure_rolls_back_successful_mock_but_keeps_attempt(self):
        kernel = learner()
        before = kernel.state_dict()
        calls = []

        def step(opt):
            fake_adam(opt)
            draw_global_rng()
            calls.append(1)

        def check():
            if calls:
                raise TimeoutError("deadline after step")

        with patch.object(torch.optim.Adam, "step", new=step), self.assertRaises(TimeoutError):
            kernel.update(before_optimizer_step=lambda _: None, before_compute=check)
        self.assert_rollback(kernel, before, attempts=1)
        self.assertTrue(kernel.state_dict()["failure"]["optimizer_step_returned"])

    def test_compute_failure_before_step_is_terminal_without_optimizer_charge(self):
        kernel = learner()
        before = kernel.state_dict()

        def check():
            draw_global_rng()
            raise TimeoutError("compute budget")

        with self.assertRaises(TimeoutError):
            kernel.update(before_optimizer_step=lambda _: self.fail("unexpected charge"), before_compute=check)
        self.assert_rollback(kernel, before, attempts=0)

    def test_keyboard_interrupt_is_terminal_and_rolls_back_rng(self):
        kernel = learner()
        before = kernel.state_dict()

        def stop(_):
            draw_global_rng()
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            kernel.update(before_optimizer_step=stop)
        self.assert_rollback(kernel, before, attempts=1)

    def test_nonfinite_gradient_stops_before_charge(self):
        kernel = learner()
        before = kernel.state_dict()

        def invalid_loss(candidate, **_):
            p = next(candidate.policy.actor_parameters())
            return torch.sqrt((p * 0.).sum())

        with patch.object(PairedCohortActor, "loss", new=invalid_loss), self.assertRaisesRegex(ValueError, "gradient"):
            kernel.update(before_optimizer_step=lambda _: self.fail("unexpected charge"))
        self.assert_rollback(kernel, before, attempts=0)

    def test_internal_data_mutation_detected_without_charge(self):
        kernel = learner()
        kernel._examples[0].raw_costs[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "binding"):
            kernel.update(before_optimizer_step=lambda _: self.fail("unexpected charge"))
        self.assertEqual(kernel.attempted_steps, 0)
        self.assertIsNotNone(kernel.state_dict()["failure"])

    def test_full_restore_roundtrip_and_detached_snapshot(self):
        kernel = learner()
        self.mock_update(kernel)
        state = kernel.state_dict()
        clone = learner()
        draw_global_rng()
        clone.load_state_dict(state)
        self.assertEqual(state_digest(state), state_digest(clone.state_dict()))
        with io.BytesIO() as buffer:
            torch.save(state, buffer)
            buffer.seek(0)
            decoded = torch.load(buffer, weights_only=True)
        clone.load_state_dict(decoded)
        self.assertEqual(state_digest(decoded), state_digest(clone.state_dict()))
        state["policy"][next(iter(state["policy"]))].add_(5)
        state["optimizer"]["param_groups"][0]["lr"] = 1.
        self.assertEqual(state_digest(decoded), state_digest(clone.state_dict()))

    def test_rewind_same_counter_branch_and_binding_changes_are_atomic(self):
        kernel = learner()
        old = kernel.state_dict()
        self.mock_update(kernel)
        before = state_digest(kernel.state_dict())
        bad_states = [old]
        for kind in ("dataset", "content", "definition", "steps", "attempted", "history", "same_counter_policy"):
            state = kernel.state_dict()
            if kind in ("dataset", "content"):
                state["manifest"]["dataset_sha256" if kind == "dataset" else "examples_sha256"] = "b" * 64
            elif kind == "definition":
                state["manifest"]["settings"]["learning_rate"] = .1
            elif kind == "steps":
                state["steps"] = True
            elif kind == "attempted":
                state["attempted_steps"] = 0
            elif kind == "history":
                state["history"][0]["loss"] += 1
            else:
                state["policy"][next(iter(state["policy"]))].add_(1)
            bad_states.append(state)
        for state in bad_states:
            with self.assertRaises(ValueError):
                kernel.load_state_dict(state)
            self.assertEqual(state_digest(kernel.state_dict()), before)
        for clone in (learner(dataset="b" * 64), learner("bc_continue"),
                      learner(examples=(example(2), example(3)))):
            with self.assertRaisesRegex(ValueError, "binding"):
                clone.load_state_dict(kernel.state_dict())

    def test_restore_tamper_rejected_atomically_after_partial_candidate_loading(self):
        source = learner()
        self.mock_update(source)
        for kind in ("critic", "shape", "dtype", "adam_step", "adam_negative", "adam_nonfinite", "groups",
                     "rng_keys", "numpy_rng", "torch_rng", "rng_readback", "history"):
            target = learner()
            before = state_digest(target.state_dict())
            state = source.state_dict()
            moments = next(iter(state["optimizer"]["state"].values()))
            if kind == "critic":
                key = next(k for k in state["policy"] if k.startswith("critic."))
                state["policy"][key].add_(1)
            elif kind in ("shape", "dtype"):
                key = next(iter(state["policy"]))
                state["policy"][key] = (torch.ones(999) if kind == "shape" else state["policy"][key].double())
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
            elif kind == "rng_readback":
                state["global_rng"]["numpy"] = (*state["global_rng"]["numpy"][:-1], 1)
            else:
                state["history"][0]["actor_grad_norm_before_clip"] = -1
            with self.subTest(kind=kind), self.assertRaises((ValueError, TypeError, RuntimeError)):
                target.load_state_dict(state)
            self.assertEqual(state_digest(target.state_dict()), before, kind)

    def test_successful_mock_boundary_preserves_unconsumed_global_rng(self):
        kernel = learner()
        before = state_digest(global_rng_state())
        self.mock_update(kernel)
        self.assertEqual(state_digest(global_rng_state()), before)

    def test_callback_validation_does_not_consume_an_attempt(self):
        kernel = learner()
        before = state_digest(kernel.state_dict())
        with self.assertRaises(TypeError):
            kernel.update()
        for options in ({"before_optimizer_step": None},
                        {"before_optimizer_step": lambda _: None, "before_compute": None}):
            with self.assertRaises(ValueError):
                kernel.update(**options)
        self.assertEqual(state_digest(kernel.state_dict()), before)


if __name__ == "__main__":
    unittest.main()
