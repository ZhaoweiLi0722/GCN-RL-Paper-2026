"""Fake collection to mocked optimizer metadata, never numerical fitting."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl.dynamic_candidate_continuation import DynamicCandidateContinuation
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.dynamic_candidate_session import DynamicCandidateSession
from src.rl.prospective_ddpg_kernel import state_digest
from tests.test_dynamic_candidate_session import FakeEnv, FakeProducer, FakeReference, OPTIONS, session
from tests.test_dynamic_candidate_updates import fake_adam
from src.rl.networks import torch


class DynamicContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("real optimizer step forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)
        self.serial = 0

    def build(self, role="ppo", *, budget=None):
        kind = "ppo" if role == "ppo" else "bc"
        scope = "block60/" + role
        if budget is None:
            limits = {"trajectory": 8, "clone": 0, "actor": 4,
                      "critic": 4 if role == "ppo" else 0, "seconds": 100}
            plan = {"format": "dynamic-candidate-budget-plan-v1", "draft_sha256": "fake",
                    "limits": limits, "phases": {role: limits},
                    "sections": {scope: limits | {"phase": role}}}
            self.serial += 1
            budget = DynamicCandidateBudget(self.root / f"budget{self.serial}.jsonl", plan,
                                            enabled=True, clock=lambda: 0.)
            self.addCleanup(budget.close)
            budget.begin(scope)

        def factory(kernel, episode, seed):
            producer = FakeProducer()
            return DynamicCandidateSession(FakeEnv(), producer, FakeReference(producer), kernel,
                enabled=True, options=OPTIONS, trajectory_id=f"fake/{role}/{episode}/{seed}",
                split="training", selection="sample", source_id="fake-continuation-v1")

        learner = session(kind=kind).learner
        settings = {"episodes_per_model": 4, "episodes_per_rollout": 2,
                    "max_updates": 2, "epochs": 1, "gae_lambda": 1.}
        return DynamicCandidateContinuation(learner, budget, settings, enabled=True,
            role=role, scope=scope, seeds=(101, 102, 103, 104), horizon=2, session_factory=factory)

    def finish(self, run):
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            while not run.done:
                if run.update_due:
                    run.update()
                else:
                    run.step()

    def test_complete_fake_rollouts_charge_each_owner_and_do_not_fit(self):
        for role in ("ppo", "bc_continue"):
            with self.subTest(role=role):
                run = self.build(role)
                weights = state_digest(run.kernel.policy.state_dict())
                self.finish(run)
                self.assertEqual(run.updates, 2)
                self.assertEqual(len(run.receipts), 4)
                self.assertEqual(state_digest(run.kernel.policy.state_dict()), weights)
                checked = read_dynamic_ledger(run.budget.path)
                self.assertEqual(checked["counts"]["environment"], 8)
                self.assertEqual(checked["counts"]["optimizer"], 8 if role == "ppo" else 4)
                self.assertEqual(checked["owner_counts"].get("global:critic", 0), 4 if role == "ppo" else 0)
                with self.assertRaises(ValueError):
                    run.step()
                with self.assertRaises(ValueError):
                    run.update()

    def test_interrupted_collector_and_update_boundary_restore_match(self):
        for role in ("ppo", "bc_continue"):
            with self.subTest(role=role):
                uninterrupted, partial = self.build(role), self.build(role)
                self.finish(uninterrupted)
                partial.step()
                restored = self.build(role, budget=partial.budget)
                restored.load_state_dict(partial.state_dict())
                self.assertEqual(state_digest(restored.state_dict()), state_digest(partial.state_dict()))
                with patch.object(torch.optim.Adam, "step", new=fake_adam):
                    while not restored.update_due:
                        restored.step()
                    saved = restored.state_dict()
                    after_boundary = self.build(role, budget=restored.budget)
                    after_boundary.load_state_dict(saved)
                    after_boundary.update()
                self.finish(after_boundary)
                self.assertEqual(state_digest(after_boundary.kernel.state_dict()),
                                 state_digest(uninterrupted.kernel.state_dict()))
                self.assertEqual(after_boundary.receipts, uninterrupted.receipts)
                final = self.build(role, budget=after_boundary.budget)
                final.load_state_dict(after_boundary.state_dict())
                self.assertTrue(final.done)

    def test_debited_collection_failure_closes_budget_and_preserves_failure(self):
        run = self.build()
        before = run.state_dict()
        run.start_episode()
        with patch.object(run.active.env, "step", side_effect=RuntimeError("invented transition failure")):
            with self.assertRaises(RuntimeError):
                run.step()
        self.assertIsNotNone(run.failure)
        self.assertTrue(run.budget.failed)
        self.assertEqual(read_dynamic_ledger(run.budget.path)["counts"]["environment"], 1)
        self.assertEqual(run.active.index, 0)
        self.assertEqual(run.active.env.t, 0)
        self.assertIsNotNone(run.state_dict()["failure"])
        with self.assertRaises(ValueError):
            run.load_state_dict(before)
        with self.assertRaises(ValueError):
            run.step()

    def test_partial_optimizer_failure_never_refunds_or_publishes(self):
        run = self.build()
        while not run.update_due:
            run.step()
        before = state_digest(run.kernel.policy.state_dict())
        calls = []

        def fail_second(optimizer, *args, **kwargs):
            calls.append(optimizer)
            if len(calls) == 2:
                raise RuntimeError("invented critic optimizer failure")
            return fake_adam(optimizer, *args, **kwargs)

        with patch.object(torch.optim.Adam, "step", new=fail_second):
            with self.assertRaises(RuntimeError):
                run.update()
        self.assertEqual(state_digest(run.kernel.policy.state_dict()), before)
        self.assertEqual(read_dynamic_ledger(run.budget.path)["counts"]["optimizer"], 2)
        self.assertTrue(run.budget.failed)
        self.assertIsNotNone(run.kernel._failure)
        with self.assertRaises(ValueError):
            run.update()

    def test_forged_receipt_or_debit_restore_is_atomic(self):
        run = self.build()
        for _ in range(2):
            run.step()
        target = self.build(budget=run.budget)
        before = state_digest(target.state_dict())
        for key in ("seed", "steps"):
            saved = copy.deepcopy(run.state_dict())
            saved["receipts"][0][key] += 1
            with self.assertRaises(ValueError):
                target.load_state_dict(saved)
            self.assertEqual(state_digest(target.state_dict()), before)
        saved = copy.deepcopy(run.state_dict())
        saved["budget"]["owner_counts"]["global:trajectory"] = 0
        with self.assertRaises(ValueError):
            target.load_state_dict(saved)

    def test_interrupt_after_optimizer_debit_is_terminal_evidence(self):
        for role in ("ppo", "bc_continue"):
            with self.subTest(role=role):
                run = self.build(role)
                while not run.update_due:
                    run.step()
                with patch.object(torch.optim.Adam, "step", side_effect=KeyboardInterrupt("fixture interrupt")):
                    with self.assertRaises(KeyboardInterrupt):
                        run.update()
                self.assertTrue(run.budget.failed)
                self.assertEqual(read_dynamic_ledger(run.budget.path)["counts"]["optimizer"], 1)
                self.assertEqual(run.kernel._failure["error_type"], "KeyboardInterrupt")
                with self.assertRaises(ValueError):
                    run.update()


if __name__ == "__main__":
    unittest.main()
