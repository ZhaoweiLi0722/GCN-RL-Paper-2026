"""Invented two-step array data; all real optimizer.step calls forbidden."""

import copy
from dataclasses import asdict
import unittest
from unittest.mock import patch

import numpy as np

from src.rl.cohort_ppo import CohortPPOKernel
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from tests.test_dynamic_candidate_updates import fake_adam
from tests.test_time_baseline_integration import MANIFEST, collector, kernel as old_kernel


def kernel(objective="cohort"):
    base = old_kernel("collected_value")
    return CohortPPOKernel(base.policy, base.contract, base.settings,
        enabled=True, mode="online", sampling_seed=41, shuffle_seed=43,
        objective=objective, accounting_steps=3, training_manifest=MANIFEST,
        training_source_id="invented-training", episode_horizon=2, episodes_per_rollout=2)


def fixture_segments():
    base = old_kernel("collected_value")
    segments = []
    for i in range(2):
        session = collector(base, i, MANIFEST[i]["environment_seed"])
        while not session.closed:
            session.step(before_step=lambda: None)
        segments.append(session.segment(1.))
    return segments


def closure(index):
    return dict(MANIFEST[index], source_id="invented-training", split="training",
                tail_costs=(10., 20., 30.), terminal_active=0,
                prefix_state_sha256="a" * 64, final_state_sha256="b" * 64,
                tail_rows_sha256="c" * 64)


@unittest.skipIf(torch is None, "torch unavailable")
class CohortPPOTests(unittest.TestCase):
    def setUp(self):
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("real optimizer forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)
        self.segments = fixture_segments()

    def filled(self, objective="cohort"):
        k = kernel(objective)
        for i, segment in enumerate(self.segments):
            k.add_segment(segment, cohort=closure(i))
        return k

    def update(self, k):
        with patch.object(torch.optim.Adam, "step", new=fake_adam):
            return k.update(before_optimizer_step=lambda owner: None, before_minibatch=lambda: None)

    def test_tail_changes_targets_not_raw_records_or_number_of_actions(self):
        a, b = self.filled("window"), self.filled("cohort")
        raw = state_digest([asdict(s) for s in self.segments])
        ra, rb = a.target_receipt(), b.target_receipt()
        self.assertEqual(raw, state_digest([asdict(s) for s in b.pending]))
        self.assertEqual(ra["raw_rewards"], rb["raw_rewards"])
        self.assertEqual(ra["returns"], tuple(s.returns for s in self.segments))
        np.testing.assert_allclose(np.array(rb["returns"]) - ra["returns"], -.6, atol=1e-6)
        np.testing.assert_allclose(np.array(rb["advantages"]) - ra["advantages"], -.6, atol=1e-6)
        self.assertEqual(len(rb["returns"]) * len(rb["returns"][0]), 4)
        self.assertEqual(rb["cohort_receipts"][0]["added_terminal_charge"], 60.)

    def test_incomplete_wrong_seed_or_test_tail_cannot_admit_prefix(self):
        for change in (dict(split="test"), dict(environment_seed=999), dict(tail_costs=(1.,)),
                       dict(terminal_active=1), dict(tail_rows_sha256="missing")):
            k = kernel()
            before = state_digest(k.state_dict())
            with self.subTest(change=change), self.assertRaises(ValueError):
                k.add_segment(self.segments[0], cohort=closure(0) | change)
            self.assertEqual(before, state_digest(k.state_dict()))
        k = kernel()
        with self.assertRaises(ValueError):
            k.add_segment(self.segments[0])
        k.add_segment(self.segments[0], cohort=closure(0))
        with self.assertRaises(ValueError):
            k.target_receipt()

    def test_pending_objective_rng_and_raw_segments_restore_atomically(self):
        k = self.filled()
        restored = kernel()
        restored.load_state_dict(k.state_dict())
        self.assertEqual(state_digest(restored.state_dict()), state_digest(k.state_dict()))
        self.assertEqual(restored.target_receipt(), k.target_receipt())
        before = state_digest(restored.state_dict())
        for field in ("pending_cohorts", "manifest"):
            saved = k.state_dict()
            if field == "manifest":
                saved[field]["objective"] = "window"
            else:
                saved[field][0]["trajectory_id"] = "test/forged"
            with self.assertRaises(ValueError):
                restored.load_state_dict(saved)
            self.assertEqual(before, state_digest(restored.state_dict()))
        with self.assertRaises(ValueError):
            kernel("window").load_state_dict(k.state_dict())

    def test_mock_two_owner_dispatch_restore_and_hashes_without_fitting(self):
        k = self.filled()
        before = k.policy.snapshot_sha256()
        receipt = k.target_receipt()
        report = self.update(k)
        self.assertEqual(k.policy.snapshot_sha256(), before)
        self.assertEqual(report["target_receipt"], receipt)
        self.assertEqual(report["rollout_steps"], 4)
        self.assertEqual(k.pending_cohorts, [])
        restored = kernel()
        restored.load_state_dict(k.state_dict())
        self.assertEqual(state_digest(k.state_dict()), state_digest(restored.state_dict()))
        for key in ("returns", "raw_rewards", "closures", "consumed"):
            saved = k.state_dict()
            if key == "consumed":
                saved[key][0] = ("forged", "training", 0)
            elif key == "closures":
                saved["target_history"][0][key][0]["tail_costs"] = (1., 2., 3.)
            else:
                values = saved["target_history"][0][key]
                saved["target_history"][0][key] = ((999., values[0][1]), values[1])
            with self.subTest(key=key), self.assertRaises(ValueError):
                restored.load_state_dict(saved)

    def test_mock_critic_failure_retains_evidence_and_never_refunds(self):
        k = self.filled()
        before = state_digest(k.policy.state_dict())
        owners = []
        def fail(optimizer, *args, **kwargs):
            if owners[-1] == "critic":
                raise RuntimeError("invented critic failure")
            return fake_adam(optimizer)
        with patch.object(torch.optim.Adam, "step", new=fail), self.assertRaises(RuntimeError):
            k.update(before_optimizer_step=owners.append, before_minibatch=lambda: None)
        self.assertEqual(owners, ["actor", "critic"])
        self.assertEqual(before, state_digest(k.policy.state_dict()))
        self.assertEqual(k.target_history, [])
        self.assertEqual(len(k.pending_cohorts), 2)
        with self.assertRaises(ValueError):
            k.add_segment(self.segments[0], cohort=closure(0))
        with self.assertRaises(ValueError):
            kernel().load_state_dict(k.state_dict())


if __name__ == "__main__":
    unittest.main()
