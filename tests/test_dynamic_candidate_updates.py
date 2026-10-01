"""Invented receipts and fake Adam counters only; no fitting or environment.

All real Adam/SGD steps are forbidden. The fake only populates zero moments and
increments mock counters, never applies a gradient or changes a parameter.
"""

import copy
from dataclasses import replace
import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.rl.candidate_imitation import ImitationSettings, public_example
from src.rl.candidate_ppo_kernel import CandidatePPOSettings
from src.rl.candidate_rollout import evaluate_candidate_policy, prepare_candidate_segment
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.dynamic_candidate_rollout import evaluate_dynamic_policy, reevaluate_dynamic_decision
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.validated_returns import OneStepRecord
from tests.test_candidate_policy_rollout import candidates, contract, observation


def policy(**changes):
    args = dict(enabled=True, architecture="graph", message_mode="physical", encoder_width=3,
                head_width=5, actor_seed=17, critic_seed=19, initial_reference_bias=0.0)
    return DynamicCandidatePolicy(contract().inputs, **(args | changes))


def settings(**changes):
    args = dict(learning_rate=.0003, clip_ratio=.2, value_loss_coef=.5, entropy_coef=.01,
                max_grad_norm=.5, gae_lambda=1., normalize_advantages=True, epochs=2,
                batch_size=2, max_rollout_steps=8, max_updates=2, max_optimizer_steps=24)
    return CandidatePPOSettings(**(args | changes))


def learner(mode="online", **changes):
    return DynamicCandidatePPOKernel(policy(), contract(), settings(**changes), enabled=True,
                                      mode=mode, sampling_seed=101, shuffle_seed=103)


def imitation(split="training", **changes):
    args = dict(learning_rate=.0003, max_grad_norm=.5, batch_size=2, max_optimizer_steps=8,
                max_updates=2, allowed_split=split)
    return DynamicCandidateImitationKernel(policy(), contract(), ImitationSettings(**(args | changes)),
                                           enabled=True, shuffle_seed=107,
                                           sampling_seed=101 if split == "training" else None)


def segment(kernel, *, length=3, trajectory="invented", start=0, terminal=True):
    decisions = [kernel.decide(observation(i), candidates(i)) for i in range(start, start + length)]
    end = evaluate_dynamic_policy(kernel.policy, observation(start + length),
                                  candidates(start + length), kernel.contract)
    evaluations = [d.evaluation for d in decisions] + [end]
    records = tuple(OneStepRecord(
        kernel.contract.replay, "invented-source", "trajectory", trajectory, i,
        evaluations[i].candidates.state_token, evaluations[i + 1].candidates.state_token,
        evaluations[i].actor_state, decisions[i].choice.submitted_request, float(-10 - i),
        evaluations[i + 1].actor_state, terminal and i == length - 1,
        not terminal and i == length - 1) for i in range(length))
    return prepare_candidate_segment(decisions, records, kernel.contract,
                                     behavior_sha256=kernel.policy.snapshot_sha256(),
                                     bootstrap=None if terminal else end, gae_lambda=1., max_steps=8)


def examples(split="training", prefix="invented", length=3):
    return [public_example(observation(i), candidates(i), contract(),
                           split=split, identity=f"{prefix}-{i}") for i in range(length)]


def fake_adam(optimizer, *args, **kwargs):
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError("missing/nonfinite gradient in invented mock")
            state = optimizer.state[parameter]
            if not state:
                state.update(step=torch.tensor(0., dtype=torch.float32),
                             exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
            state["step"].add_(1)


@unittest.skipIf(torch is None, "torch unavailable")
class DynamicCandidateUpdatesTests(unittest.TestCase):
    def setUp(self):
        self.adam = patch.object(torch.optim.Adam, "step", side_effect=AssertionError("real Adam forbidden"))
        self.sgd = patch.object(torch.optim.SGD, "step", side_effect=AssertionError("real SGD forbidden"))
        self.adam.start()
        self.sgd.start()
        self.addCleanup(self.adam.stop)
        self.addCleanup(self.sgd.stop)

    def ready(self, **changes):
        kernel = learner(**changes)
        kernel.add_segment(segment(kernel))
        return kernel

    def mock_update(self, kernel, charges=None):
        charges = [] if charges is None else charges
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            return kernel.update(before_optimizer_step=charges.append, before_minibatch=lambda: None)

    def test_legacy_entrypoint_rejects_new_model_and_new_receipt_is_reproducible(self):
        p = policy()
        with self.assertRaises(TypeError):
            evaluate_candidate_policy(p, observation(), candidates(), contract())
        a = evaluate_dynamic_policy(p, observation(), candidates(), contract())
        self.assertEqual(a, evaluate_dynamic_policy(copy.deepcopy(p), observation(), candidates(), contract()))
        k = learner()
        decision = k.decide(observation(), candidates())
        logp, value, entropy = reevaluate_dynamic_decision(k.policy, observation(), decision)
        self.assertAlmostEqual(logp.item(), decision.old_log_prob)
        self.assertAlmostEqual(value.item(), decision.evaluation.value)
        self.assertGreater(entropy.item(), 0)
        with self.assertRaises(ValueError):
            reevaluate_dynamic_decision(k.policy, observation(1), decision)

    def test_explicit_opt_in_mode_precision_and_seeds(self):
        args = dict(enabled=True, mode="online", sampling_seed=1, shuffle_seed=2)
        for changes in ({"enabled": False}, {"mode": "guess"}, {"sampling_seed": True}, {"shuffle_seed": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                DynamicCandidatePPOKernel(policy(), contract(), settings(), **(args | changes))
        with self.assertRaises(ValueError):
            DynamicCandidatePPOKernel(policy().half(), contract(), settings(), **args)
        with self.assertRaises(ValueError):
            DynamicCandidateImitationKernel(policy().double(), contract(), ImitationSettings(.001,.5,2,4,1,"training"),
                                             enabled=True, shuffle_seed=1)

    def test_forks_start_identically_and_optimizers_own_disjoint_storage(self):
        ppo, frozen, bc = learner(), learner(mode="frozen"), imitation()
        self.assertEqual(ppo.policy.snapshot_sha256(), frozen.policy.snapshot_sha256())
        self.assertEqual(ppo.policy.snapshot_sha256(), bc.policy.snapshot_sha256())
        actor = {p.data_ptr() for group in ppo.actor_optimizer.param_groups for p in group["params"]}
        critic = {p.data_ptr() for group in ppo.critic_optimizer.param_groups for p in group["params"]}
        self.assertFalse(actor & critic)
        self.assertEqual(actor, {p.data_ptr() for p in ppo.policy.actor_parameters()})
        self.assertEqual(critic, {p.data_ptr() for p in ppo.policy.critic_parameters()})
        self.assertEqual(frozen.optimizers, {"actor": None, "critic": None})
        self.assertFalse(any(p.requires_grad for p in frozen.policy.parameters()))
        for _ in range(4):
            self.assertEqual(ppo.decide(observation(), candidates()), frozen.decide(observation(), candidates()))
            # BC uses the same sampler start even though its critic is frozen.
        initial = learner()
        self.assertEqual(initial.decide(observation(), candidates()), bc.decide(observation(), candidates()))

    def test_mock_transaction_counts_every_owner_without_changing_weights(self):
        k = self.ready()
        original = k.policy.snapshot_sha256()
        charges = []
        result = self.mock_update(k, charges)
        self.assertEqual(charges, ["actor", "critic"] * 4)
        self.assertEqual(result["optimizer_steps"], 8)
        self.assertEqual((result["actor_optimizer_steps"], result["critic_optimizer_steps"]), (4, 4))
        self.assertEqual(original, k.policy.snapshot_sha256())
        for epoch in (0, 1):
            rows = [i for row in result["minibatches"] if row["epoch"] == epoch for i in row["indices"]]
            self.assertEqual(sorted(rows), [0, 1, 2])
        clone = learner()
        clone.load_state_dict(k.state_dict())
        self.assertEqual(state_digest(k.state_dict()), state_digest(clone.state_dict()))
        self.assertFalse(k.pending)
        self.assertEqual(len(k.consumed), 3)

    def test_owner_cap_and_forged_pending_restore_count_twice(self):
        k = learner(max_optimizer_steps=6)
        s = segment(k)
        before = state_digest(k.state_dict())
        with self.assertRaisesRegex(ValueError, "two-owner"):
            k.add_segment(s)
        self.assertEqual(before, state_digest(k.state_dict()))
        from dataclasses import asdict
        state = k.state_dict()
        state["pending"] = [asdict(s)]
        with self.assertRaisesRegex(ValueError, "two-owner"):
            k.load_state_dict(state)
        self.assertEqual(before, state_digest(k.state_dict()))

    def test_failed_second_owner_preserves_partial_counters_but_is_not_retryable(self):
        k = self.ready()
        original = k.state_dict()
        durable = []

        def charge(owner):
            durable.append(owner)
            if owner == "critic":
                raise RuntimeError("injected debit failure after durable record")

        with patch.object(torch.optim.Adam, "step", new=fake_adam), self.assertRaisesRegex(RuntimeError, "debit"):
            k.update(before_optimizer_step=charge, before_minibatch=lambda: None)
        failed = k.state_dict()
        self.assertEqual(durable, ["actor", "critic"])
        self.assertEqual(failed["failure"]["acknowledged_charge_owners"], ["actor"])
        self.assertEqual(failed["failure"]["resource_authority"], "external_non_refundable_ledger")
        base = dict(failed, failure=None)
        self.assertEqual(state_digest(base), state_digest(original))
        partial = failed["failure"]["partial_transaction"]["optimizer"]
        self.assertEqual({s["step"].item() for s in partial["actor"]["state"].values()}, {1.})
        self.assertEqual(partial["critic"]["state"], {})
        for call in (lambda: k.load_state_dict(original),
                     lambda: k.update(before_optimizer_step=lambda o: None, before_minibatch=lambda: None),
                     lambda: k.decide(observation(), candidates()),
                     lambda: learner().load_state_dict(failed)):
            with self.assertRaisesRegex(ValueError, "failed"):
                call()

    def test_mock_imitation_never_changes_critic_and_has_exact_restore(self):
        for split in ("training", "demonstration"):
            k = imitation(split)
            before = k.policy.snapshot_sha256()
            charges = []
            options = {"epochs": 2} if split == "training" else {"replacement_steps": 3}
            with patch.object(torch.optim.Adam, "step", new=fake_adam):
                report = k.fit(examples(split), before_step=lambda: charges.append("actor"), **options)
            self.assertEqual(before, k.policy.snapshot_sha256())
            self.assertEqual(report["optimizer_steps"], 4 if split == "training" else 3)
            self.assertEqual(len(charges), report["optimizer_steps"])
            self.assertTrue(all(not p.requires_grad for p in k.policy.critic_parameters()))
            clone = imitation(split)
            clone.load_state_dict(k.state_dict())
            self.assertEqual(state_digest(k.state_dict()), state_digest(clone.state_dict()))

    def test_imitation_rejects_heldout_and_failed_debit_cannot_retry(self):
        k = imitation()
        with self.assertRaisesRegex(ValueError, "forbidden"):
            k.fit(examples("test"), epochs=1, before_step=lambda: self.fail("unreachable debit"))
        self.assertEqual(k.steps, 0)
        self.assertIsNotNone(k.state_dict()["failure"])
        k = imitation()
        before = k.state_dict()
        charges = []

        def fail():
            charges.append(1)
            raise RuntimeError("fake budget refuses")

        with self.assertRaisesRegex(RuntimeError, "budget"):
            k.fit(examples(), epochs=1, before_step=fail)
        self.assertEqual(charges, [1])
        self.assertEqual(state_digest(dict(k.state_dict(), failure=None)), state_digest(before))
        with self.assertRaisesRegex(ValueError, "failed"):
            k.load_state_dict(before)

    def test_receipt_forgery_and_duplicate_lineage_rejected_without_optimizer(self):
        k = learner()
        s = segment(k)
        with self.assertRaises(ValueError):
            k.add_segment(replace(s, returns=(123.,) * 3))
        k.add_segment(s)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            k.add_segment(s)
        self.mock_update(k)
        with self.assertRaisesRegex(ValueError, "consumed"):
            k.add_segment(segment(k))

    def test_owner_checkpoint_tamper_and_frozen_critic_changes_are_atomic(self):
        k = self.ready()
        self.mock_update(k)
        before = state_digest(k.state_dict())
        for tamper in ("step", "settings", "dtype", "nonfinite"):
            saved = k.state_dict()
            actor = saved["optimizer"]["actor"]
            first = next(iter(actor["state"].values()))
            if tamper == "step":
                first["step"].add_(1)
            elif tamper == "settings":
                actor["param_groups"][0]["lr"] *= 2
            elif tamper == "dtype":
                first["exp_avg"] = first["exp_avg"].double()
            else:
                first["exp_avg"].flatten()[0] = float("nan")
            with self.subTest(tamper=tamper), self.assertRaises(ValueError):
                k.load_state_dict(saved)
            self.assertEqual(before, state_digest(k.state_dict()))
        bc = imitation()
        saved = bc.state_dict()
        name = next(n for n in saved["policy"] if n.startswith("critic."))
        saved["policy"][name].add_(1)
        with self.assertRaisesRegex(ValueError, "critic"):
            bc.load_state_dict(saved)

    def test_sampler_pending_and_file_restore_are_exact(self):
        k = self.ready()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invented-state.pt"
            k.save(path)
            with self.assertRaises(FileExistsError):
                k.save(path)
            clone = learner()
            clone.load(path)
            self.assertEqual(state_digest(k.state_dict()), state_digest(clone.state_dict()))
            self.assertEqual(k.decide(observation(9), candidates(9)), clone.decide(observation(9), candidates(9)))

    def test_no_global_rng_consumed_and_normalization_occurs_once(self):
        import src.rl.dynamic_candidate_ppo as module
        cpu, python = torch.get_rng_state().clone(), random.getstate()
        numpy = copy.deepcopy(np.random.get_state())
        k = self.ready()
        k.add_segment(segment(k, start=7, length=2, trajectory="second", terminal=False))
        with patch.object(module, "normalize_rollout_advantages", wraps=module.normalize_rollout_advantages) as normalize:
            self.mock_update(k)
        self.assertEqual(normalize.call_count, 1)
        self.assertEqual(normalize.call_args.args[0].numel(), 5)
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertEqual(python, random.getstate())
        np.testing.assert_array_equal(numpy[1], np.random.get_state()[1])
        self.assertEqual(numpy[2:], np.random.get_state()[2:])

    def test_frozen_has_no_admission_or_update_and_charge_callbacks_are_mandatory(self):
        k = learner(mode="frozen")
        with self.assertRaises(ValueError):
            k.add_segment(None)
        with self.assertRaises(ValueError):
            k.update(before_optimizer_step=lambda o: None, before_minibatch=lambda: None)
        k = self.ready()
        with self.assertRaises(TypeError):
            k.update()
        with self.assertRaises(TypeError):
            imitation().fit(examples(), epochs=1)


if __name__ == "__main__":
    unittest.main()
