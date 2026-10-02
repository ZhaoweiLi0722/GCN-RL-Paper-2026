"""Invented arrays and mock optimizer metadata only; no patient engine or fit."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.rl import cohort_public, patient_replay_collector
from src.rl.cohort_collection import CohortCollection
from src.rl.cohort_continuation import CohortContinuation
from src.rl.cohort_public import CohortPrefixSession
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from tests.test_cohort_public import ArrayLayout, MANIFEST, SOURCE, fixture, kernel
from tests.test_dynamic_candidate_session import FakeReference, OPTIONS, fake_info
from tests.test_dynamic_candidate_updates import fake_adam


def collect(learner, index, seed, *, restore_state=None):
    _, _, env, producer = fixture()
    env._episode_seed, env.cumulative_enrolled = seed, 0
    prefix = CohortPrefixSession(env, producer, FakeReference(producer), learner,
        enabled=True, options=OPTIONS, trajectory_id=MANIFEST[index]["trajectory_id"],
        split="training", selection="sample", source_id=SOURCE, environment_seed=seed)
    return CohortCollection(prefix, enabled=True, objective=getattr(learner, "objective", "none"),
        split="training", trajectory_id=prefix.trajectory_id,
        finish_prefix=lambda p: {"invented_prefix": p.index},
        followup_action=lambda e: np.array([0.] * 6 + [-1.] * 2))


@unittest.skipIf(torch is None, "torch unavailable")
class CohortContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root, self.serial = Path(self.temp.name), 0
        for owner, method in ((PatientConditionCapacityEnv, "__init__"),
                              (PatientConditionCapacityEnv, "step"),
                              (torch.optim.Adam, "step"), (torch.optim.SGD, "step")):
            guard = patch.object(owner, method, side_effect=AssertionError("real simulation/update forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)
        for module in (cohort_public, patient_replay_collector):
            guard = patch.object(module, "PatientConditionCapacityEnv", ArrayLayout)
            guard.start()
            self.addCleanup(guard.stop)
        original = ArrayLayout.step
        def tail(engine, action):
            if not getattr(engine, "_cohort_closed", False):
                return original(engine, action)
            engine.actions.append(action.copy())
            engine.t += 1
            engine.raw[-1] = engine.t / engine.config.episode_horizon
            info = fake_info(action)
            return engine.observation(), -info["cost"], engine.t == 5, info
        guard = patch.object(ArrayLayout, "step", new=tail)
        guard.start()
        self.addCleanup(guard.stop)

    def run_fixture(self, kind="cohort", budget=None, finish=None):
        scope = "invented/" + kind
        if budget is None:
            caps = dict(trajectory=20, clone=0, actor=4, critic=0 if kind == "bc" else 4, seconds=100)
            plan = dict(format="dynamic-candidate-budget-plan-v1", draft_sha256="invented",
                limits=caps, phases={kind: caps}, sections={scope: caps | {"phase": kind}})
            self.serial += 1
            budget = DynamicCandidateBudget(self.root / f"budget{self.serial}.jsonl", plan,
                                            enabled=True, clock=lambda: 0.)
            self.addCleanup(budget.close)
            budget.begin(scope)
        _, _, _, producer = fixture()
        return CohortContinuation(kernel(producer, kind), budget,
            dict(episodes_per_model=4, episodes_per_rollout=2, max_updates=2, epochs=1, gae_lambda=1.),
            enabled=True, role="bc_continue" if kind == "bc" else "ppo", scope=scope,
            seeds=(101, 102, 103, 104), prefix_steps=2, accounting_steps=3,
            session_factory=collect, finish_collection=finish or (lambda c: {"invented": c.index}))

    def finish(self, run):
        before = run.kernel.policy.snapshot_sha256()
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            while not run.done:
                run.update() if run.update_due else run.step()
        self.assertEqual(run.kernel.policy.snapshot_sha256(), before)

    def test_no_prefix_or_tail_only_update_and_all_calls_charged(self):
        run = self.run_fixture()
        for _ in range(2):
            run.step()
        self.assertEqual(run.episode, 0)
        self.assertTrue(run.active.prefix.closed)
        self.assertEqual(run.kernel.pending, [])
        with self.assertRaises(ValueError):
            run.update()
        for _ in range(3):
            run.step()
        self.assertEqual(len(run.kernel.pending), 1)
        self.assertEqual(len(run.kernel.pending[0].records), 2)
        self.assertEqual(run.receipts[0]["steps"], 5)
        self.assertEqual(run.receipts[0]["prefix_steps"], 2)
        self.finish(run)
        ledger = read_dynamic_ledger(run.budget.path)
        self.assertEqual(ledger["counts"]["environment"], 20)
        self.assertEqual(ledger["counts"]["optimizer"], 8)

    def test_both_objectives_and_bc_complete_restore_without_fitting(self):
        for kind in ("window", "cohort", "bc"):
            with self.subTest(kind=kind):
                run = self.run_fixture(kind)
                self.finish(run)
                restored = self.run_fixture(kind, budget=run.budget)
                restored.load_state_dict(run.state_dict())
                self.assertTrue(restored.done)
                self.assertEqual(state_digest(restored.state_dict()), state_digest(run.state_dict()))
                self.assertEqual(len(restored.receipts), 4)
                if kind == "bc":
                    self.assertEqual([len(r["identities"]) for r in restored.kernel.history], [4, 4])
                    self.assertEqual(read_dynamic_ledger(run.budget.path)["counts"]["optimizer"], 4)
                else:
                    self.assertEqual(len(restored.kernel.target_history), 2)

    def test_every_prefix_tail_and_update_boundary_restore_exact_without_steps(self):
        run = self.run_fixture()
        for _ in range(10):
            run.step()
            restored = self.run_fixture(budget=run.budget)
            with patch.object(ArrayLayout, "step", side_effect=AssertionError("replay forbidden")):
                restored.load_state_dict(run.state_dict())
            self.assertEqual(state_digest(run.state_dict()), state_digest(restored.state_dict()))
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            restored.update()
        after = self.run_fixture(budget=run.budget)
        after.load_state_dict(restored.state_dict())
        self.finish(after)
        direct = self.run_fixture()
        self.finish(direct)
        self.assertEqual(state_digest(after.kernel.state_dict()), state_digest(direct.kernel.state_dict()))

    def test_tail_record_failure_cannot_admit_or_retry_and_debits_stay(self):
        def fail(c):
            raise RuntimeError("invented persistence failure")
        run = self.run_fixture(finish=fail)
        saved = run.state_dict()
        for _ in range(4):
            run.step()
        with self.assertRaisesRegex(RuntimeError, "persistence"):
            run.step()
        self.assertEqual(run.episode, 0)
        self.assertEqual(run.kernel.pending, [])
        self.assertEqual(read_dynamic_ledger(run.budget.path)["counts"]["environment"], 5)
        self.assertTrue(run.budget.failed)
        with self.assertRaises(ValueError):
            run.load_state_dict(saved)
        with self.assertRaises(ValueError):
            run.step()

    def test_same_live_collection_cannot_rewind_or_replace_receipts(self):
        run = self.run_fixture()
        run.step()
        old = run.state_dict()
        run.step()
        before = state_digest(run.state_dict())
        with self.assertRaises(ValueError):
            run.load_state_dict(old)
        self.assertEqual(before, state_digest(run.state_dict()))
        self.finish(run)
        before = state_digest(run.state_dict())
        saved = copy.deepcopy(run.state_dict())
        saved["receipts"][0]["steps"] -= 1
        with self.assertRaises(ValueError):
            run.load_state_dict(saved)
        self.assertEqual(before, state_digest(run.state_dict()))


if __name__ == "__main__":
    unittest.main()
