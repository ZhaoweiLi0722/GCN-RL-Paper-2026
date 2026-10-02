"""Invented fake-engine continuations, never real patient construction or fitting."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.rl.conservative_cohort_collection import make_current_policy_branch, greedy_request
from src.rl.paired_cohort_collection import capture_context
from src.rl.patient_replay_collector import evidence_digest
from src.rl.prospective_ddpg_kernel import state_digest
from tests.test_paired_cohort_collection import (
    IDLE, Producer, Reference, fixture, make, public_context, example,
)


class Owner:
    mode, _failure = "frozen", None
    def __init__(self):
        self.value = 1
    def state_dict(self):
        return {"invented": self.value}


def branch(env, context, owner=None, **kwargs):
    return make_current_policy_branch(env, Producer, Reference(), context,
        continuation=owner or Owner(), round_index=0, before_forward=lambda: None,
        future_seed="18446744073709551617", candidate_index=1, replication=0,
        branch_id="invented-current-policy", record=kwargs.get("record", lambda *a: None),
        before_clone=lambda: None, enabled=True)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch("src.baselines.heuristics.facility_net_action_from_state", return_value=IDLE)
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def test_continuation_is_current_policy_not_R4(self):
        env = make()
        with patch("src.rl.paired_cohort_collection.dynamic_context_from_public", public_context), \
                patch("src.rl.paired_cohort_collection.public_example", example):
            context = capture_context(env, Producer(env), Reference(), [], context_id="invented/start",
                block=60, cohort=0, allowed_times=[0], source_id="invented-only")
        run = branch(env, context)
        request = IDLE.copy()
        request[0] = .1234567890123
        with patch("src.rl.conservative_cohort_collection.greedy_request", return_value=request) as infer:
            a = run.step(before_step=lambda: None)
            b = run.step(before_step=lambda: None)
        self.assertEqual(a["action_source"], "candidate")
        self.assertEqual(b["action_source"], "frozen_round_start_policy")
        np.testing.assert_array_equal(b["action"], request)
        infer.assert_called_once()
        self.assertFalse(run.manifest["event_aligned_crn_after_divergence_claimed"])

    def test_public_inference_uses_real_evaluation_interface(self):
        env, _ = fixture()
        owner = SimpleNamespace(policy="fake", contract="fake-contract")
        with patch("src.rl.conservative_cohort_collection.dynamic_context_from_public", public_context), \
                patch("src.rl.conservative_cohort_collection.evaluate_dynamic_policy",
                      return_value=SimpleNamespace(log_probs=[-3., -1.])) as evaluate:
            request = greedy_request(env, Producer(env), Reference(), [], owner)
        evaluate.assert_called_once()
        self.assertEqual(request.dtype, np.float64)
        self.assertEqual(request[0], .25000000001)

    def test_complete_restore_and_no_sunk_cost(self):
        for steps in (0, 1, 2, 4):
            env, context = fixture()
            first = branch(env, context)
            for _ in range(steps):
                first.step(before_step=lambda: None)
            second = branch(env, context)
            second.load_state_dict(first.state_dict())
            self.assertEqual(state_digest(first.state_dict()), state_digest(second.state_dict()))
            while not first.closed:
                a = first.step(before_step=lambda: None)
                b = second.step(before_step=lambda: None)
                self.assertEqual(evidence_digest(a), evidence_digest(b))
            self.assertEqual(first.receipt()["remaining_raw_cost"], 40.)

    def test_frozen_owner_change_and_failed_record_never_refund(self):
        env, context = fixture()
        owner, debits = Owner(), []
        run = branch(env, context, owner)
        owner.value = 2
        with self.assertRaisesRegex(ValueError, "continuation mutated"):
            run.step(before_step=lambda: debits.append(1))
        self.assertFalse(debits)
        def fail(*args):
            raise OSError("invented disk failure")
        run = branch(env, context, record=fail)
        old = copy.deepcopy(run.state_dict())
        with self.assertRaises(OSError):
            run.step(before_step=lambda: debits.append(1))
        self.assertEqual(debits, [1])
        self.assertFalse(run.failure["refunded"])
        with self.assertRaises(ValueError):
            run.load_state_dict(old)


if __name__ == "__main__":
    unittest.main()
