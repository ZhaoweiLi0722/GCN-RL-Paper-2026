"""Bounded invented-tensor fits only; no simulator or research training."""

import copy
import random
import unittest
from unittest.mock import patch

import numpy as np

from src.rl.candidate_imitation import CandidateImitationKernel, ImitationSettings, decode_example, public_example
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from tests.test_candidate_policy_rollout import candidates, contract, observation, policy
from tests.test_candidate_ppo_kernel import invented_segment, kernel


def learner(split="demonstration", **changes):
    settings = dict(learning_rate=.001, max_grad_norm=.5, batch_size=2,
                    max_optimizer_steps=6, max_updates=2, allowed_split=split)
    return CandidateImitationKernel(policy(), contract(), ImitationSettings(**(settings | changes)),
                                     enabled=True, shuffle_seed=41, sampling_seed=42 if split == "training" else None)


def examples(split="demonstration", start=0):
    return [public_example(observation(i), candidates(i), contract(), split=split, identity=f"invented/{i}")
            for i in range(start, start + 3)]


class CandidateImitationTests(unittest.TestCase):
    def test_opt_in_bounds_and_split(self):
        for changes in ({"learning_rate": 0}, {"max_grad_norm": True}, {"batch_size": 0},
                        {"max_optimizer_steps": 0}, {"max_updates": True}, {"allowed_split": "test"}):
            with self.assertRaises(ValueError):
                learner(**changes)
        with self.assertRaises(ValueError):
            CandidateImitationKernel(policy(), None, learner().settings, enabled=True, shuffle_seed=1)

    def test_public_example_excludes_hidden_labels_and_preserves_input(self):
        rows = examples()
        obs, bank = decode_example(rows[0], contract())
        self.assertEqual(bank, candidates())
        torch.testing.assert_close(obs.nodes, observation().nodes)
        with self.assertRaises(ValueError):
            decode_example(rows[0] | {"cost": 12}, contract())
        bad = copy.deepcopy(rows[0])
        bad["actor_state"] = bad["actor_state"][:-1] + (.12,)
        with self.assertRaises(ValueError):
            decode_example(bad, contract())

    def test_initial_fit_clones_prototype_changes_actor_not_value_and_counts(self):
        fit = learner()
        before = fit.policy.snapshot_sha256()
        value = state_digest(fit.policy.value_head.state_dict())
        debits = []
        result = fit.fit(examples(), replacement_steps=3, before_step=lambda: debits.append(1))
        self.assertEqual(result["optimizer_steps"], 3)
        self.assertEqual(len(debits), 3)
        self.assertNotEqual(fit.policy.snapshot_sha256(), before)
        self.assertEqual(state_digest(fit.policy.value_head.state_dict()), value)
        self.assertEqual([len(b["indices"]) for b in result["minibatches"]], [2, 2, 2])
        self.assertTrue(all(p.grad is None for p in fit.policy.parameters()))

    def test_first_adam_step_matches_independent_ce_and_closed_form(self):
        fit = learner(max_grad_norm=1e8, batch_size=3)
        rows = examples()
        model = copy.deepcopy(fit.policy)
        rng = torch.Generator().manual_seed(41)
        indices = torch.randint(3, (3,), generator=rng).tolist()
        losses = []
        for i in indices:
            obs, bank = decode_example(rows[i], contract())
            logits = model(obs, bank).logits
            losses.append(-torch.log_softmax(logits, dim=0)[bank.reference_class])
        loss = torch.stack(losses).mean()
        loss.backward()
        expected = [p.detach() - .001 * p.grad / (p.grad.abs() + 1e-8)
                    for name, p in model.named_parameters() if not name.startswith("value_head.")]
        result = fit.fit(rows, replacement_steps=1)
        self.assertAlmostEqual(result["minibatches"][0]["cross_entropy"], loss.item(), places=7)
        for actual, target in zip(fit.trainable(), expected):
            torch.testing.assert_close(actual, target, rtol=0, atol=1e-7)

    def test_continuation_whole_epochs_and_exact_next_fit_and_sample(self):
        fit = learner("training")
        result = fit.fit(examples("training"), epochs=2)
        self.assertEqual(result["optimizer_steps"], 4)
        batches = result["minibatches"]
        for offset in (0, 2):
            self.assertEqual(sorted(i for b in batches[offset:offset+2] for i in b["indices"]), [0, 1, 2])
        restored = learner("training")
        restored.load_state_dict(fit.state_dict())
        self.assertEqual(fit.fit(examples("training", 3), epochs=1),
                         restored.fit(examples("training", 3), epochs=1))
        self.assertEqual(fit.decide(observation(), candidates()), restored.decide(observation(), candidates()))
        self.assertEqual(state_digest(fit.state_dict()), state_digest(restored.state_dict()))

    def test_forbidden_splits_empty_duplicates_consumed_and_caps_are_atomic(self):
        fit = learner()
        fit.fit(examples(), replacement_steps=1)
        before = state_digest(fit.state_dict())
        for rows, count in (([], 1), (examples(), 1), (examples("test", 3), 1),
                            (examples("qualification", 3), 1), (examples(start=3) * 2, 1),
                            (examples(start=3), 6)):
            with self.assertRaises(ValueError):
                fit.fit(rows, replacement_steps=count)
            self.assertEqual(before, state_digest(fit.state_dict()))
        with self.assertRaises(ValueError):
            fit.fit(examples(start=3), epochs=1)

    def test_failure_rolls_back_kernel_not_external_debits(self):
        fit = learner()
        before = state_digest(fit.state_dict())
        debits = []
        original = torch.optim.Adam.step
        def failing(optimizer, *args, **kwargs):
            result = original(optimizer, *args, **kwargs)
            if len(debits) == 2:
                raise RuntimeError("invented failure")
            return result
        with patch.object(torch.optim.Adam, "step", failing), self.assertRaises(RuntimeError):
            fit.fit(examples(), replacement_steps=3, before_step=lambda: debits.append(1))
        self.assertEqual(len(debits), 2)
        self.assertEqual(before, state_digest(fit.state_dict()))

    def test_ppo_failure_callback_also_keeps_external_spent_budget(self):
        fit = kernel()
        fit.add_segment(invented_segment(fit))
        before, debits = state_digest(fit.state_dict()), []
        def debit():
            debits.append(1)
            if len(debits) == 2:
                raise RuntimeError("invented durable budget failure")
        with self.assertRaises(RuntimeError):
            fit.update(before_optimizer_step=debit)
        self.assertEqual(len(debits), 2)
        self.assertEqual(before, state_digest(fit.state_dict()))

    def test_no_global_rng_consumption(self):
        cpu, np_rng, py_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
        fit = learner("training")
        fit.fit(examples("training"), epochs=1)
        fit.decide(observation(), candidates())
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        actual = np.random.get_state()
        np.testing.assert_array_equal(np_rng[1], actual[1])
        self.assertEqual(np_rng[2:], actual[2:])
        self.assertEqual(py_rng, random.getstate())

    def test_corrupt_restore_is_atomic(self):
        fit = learner("training")
        fit.fit(examples("training"), epochs=1)
        saved, before = fit.state_dict(), state_digest(fit.state_dict())
        changes = [lambda s: s.update(history=[1]), lambda s: s.update(steps=True),
                   lambda s: s["history"][0].update(examples_sha256="z" * 64),
                   lambda s: s["history"][0]["identities"].append(s["history"][0]["identities"][0]),
                   lambda s: s["optimizer"]["param_groups"][0].update(lr=1),
                   lambda s: s["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1),
                   lambda s: s["optimizer"]["state"][0].update(step=torch.tensor([2.])),
                   lambda s: s["optimizer"]["state"][0]["exp_avg"].fill_(float("nan")),
                   lambda s: s["policy"]["value_head.0.weight"].add_(1),
                   lambda s: s.update(rng=s["rng"].float()),
                   lambda s: s.update(sampling_rng=torch.zeros(10, dtype=torch.uint8))]
        for mutate in changes:
            bad = copy.deepcopy(saved)
            mutate(bad)
            with self.assertRaises((ValueError, TypeError)):
                fit.load_state_dict(bad)
            self.assertEqual(before, state_digest(fit.state_dict()))


if __name__ == "__main__":
    unittest.main()
