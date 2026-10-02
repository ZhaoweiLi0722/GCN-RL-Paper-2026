"""Invented clock/public fixtures, without scientific models or optimizer calls."""

import copy
from dataclasses import asdict
import unittest
from unittest.mock import patch

import numpy as np

from src.env.cohort_followup import same_payload
from src.rl.paired_cohort_collection import capture_context, make_branch, validate_context
from src.rl.patient_replay_collector import evidence_digest
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import decode_arrays, encode_arrays
from src.rl.routing_candidate_contract import RequestCandidates, RoutingRequestSchema
from tests.test_cohort_followup import make, IDLE, FakeEngine


class Reference:
    checkpoint_sha256 = "a" * 64

    def act(self, raw):
        return IDLE.copy()


class Producer:
    contract = None
    anchor_config = {"num_facilities": 1}

    def __init__(self, env):
        self.check_environment(env)

    def check_environment(self, env):
        if env._cohort_closed:
            raise ValueError("producer requires open prefix")


def public_context(producer, reference, raw, options, token):
    bank = RequestCandidates(RoutingRequestSchema("invented/action", 1, 4., 3), token,
        (tuple(IDLE), tuple(IDLE), (0.25000000001, 0., 0., -1.)))
    return None, bank


def example(obs, bank, contract, *, split, identity):
    return dict(split=split, identity=identity, actor_state=(), candidates=asdict(bank))


def fixture():
    env = make()
    env.step(IDLE)
    with patch("src.rl.paired_cohort_collection.dynamic_context_from_public", public_context), \
            patch("src.rl.paired_cohort_collection.public_example", example):
        context = capture_context(env, Producer(env), Reference(), [], context_id="invented/context",
                                  block=60, cohort=0, allowed_times=[1], source_id="invented-only")
    return env, context


def branch(env, context, *, record=lambda owner, event: None, before_clone=lambda: None,
           seed="18446744073709551617", candidate=1):
    return make_branch(env, Producer, Reference(), context, future_seed=seed, candidate_index=candidate,
                       replication=0, branch_id="invented/branch", record=record,
                       before_clone=before_clone, enabled=True)


class PairedCohortCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tail = patch("src.baselines.heuristics.facility_net_action_from_state", return_value=IDLE)
        self.tail.start()
        self.addCleanup(self.tail.stop)

    def test_context_hash_public_binding_and_exact_request_classes(self):
        env, raw = fixture()
        context, bank = validate_context(raw)
        self.assertEqual(len(bank.class_keys), 2)
        self.assertEqual(bank.members[0], (0, 1))
        self.assertTrue(same_payload(context["environment"], env.state_dict()))
        for key in ("environment_sha256", "context_sha256"):
            changed = copy.deepcopy(raw)
            changed[key] = "0" * 64
            with self.assertRaises(ValueError):
                validate_context(changed)
        changed = decode_arrays(raw)
        changed["public_example"]["candidates"]["representatives"] = (1, 2)
        body = dict(changed)
        body.pop("context_sha256")
        changed["context_sha256"] = evidence_digest(body)
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_context(encode_arrays(changed))

    def test_only_future_rng_changes_and_template_is_untouched(self):
        env, context = fixture()
        before = env.state_dict()
        debits = []
        a = branch(env, context, before_clone=lambda: debits.append(1))
        b = branch(env, context, seed=18446744073709551617, candidate=0)
        expected = copy.deepcopy(before)
        expected["rng_state"] = np.random.PCG64(18446744073709551617).state
        self.assertTrue(same_payload(a.env.state_dict(), expected))
        self.assertTrue(same_payload(a.env.state_dict(), b.env.state_dict()))
        self.assertTrue(same_payload(env.state_dict(), before))
        self.assertEqual(debits, [1])
        self.assertFalse(a.manifest["event_aligned_crn_after_divergence_claimed"])

    def test_first_original_request_once_full_tail_and_no_sunk_cost(self):
        env, context = fixture()
        events, debits = [], []
        run = branch(env, context, record=lambda owner, event: events.append(event))
        while not run.closed:
            run.step(before_step=lambda: debits.append(1))
        self.assertEqual([r["action_source"] for r in events], ["candidate"] + ["common_tail"] * 3)
        self.assertEqual(events[0]["action"].dtype, np.float64)
        self.assertEqual(events[0]["action"][0], .25000000001)
        self.assertEqual(len(debits), 4)
        receipt = run.receipt()
        self.assertEqual(receipt["remaining_raw_cost"], 40.)
        self.assertEqual(receipt["prefix_raw_cost"], 10.)
        self.assertEqual(receipt["tail_raw_cost"], 30.)
        self.assertFalse(receipt["included_sunk_cost"])
        self.assertEqual(receipt["terminal_active"], 0)

    def test_following_prefix_uses_fixed_r4_not_repeated_candidate(self):
        env, context = fixture()
        # A separate tiny complete-context fixture starts at t=0.
        env = make()
        with patch("src.rl.paired_cohort_collection.dynamic_context_from_public", public_context), \
                patch("src.rl.paired_cohort_collection.public_example", example):
            context = capture_context(env, Producer(env), Reference(), [], context_id="invented/start",
                                      block=60, cohort=0, allowed_times=[0], source_id="invented-only")
        run = branch(env, context)
        first = run.step(before_step=lambda: None)
        second = run.step(before_step=lambda: None)
        self.assertEqual(first["action_source"], "candidate")
        self.assertEqual(second["action_source"], "fixed_r4")
        np.testing.assert_array_equal(second["action"], IDLE)

    def test_complete_restore_at_prefix_boundary_midtail_and_final(self):
        for steps in (0, 1, 2, 4):
            with self.subTest(steps=steps):
                env, context = fixture()
                original = branch(env, context)
                for _ in range(steps):
                    original.step(before_step=lambda: None)
                checkpoint = original.state_dict()
                restored = branch(env, context)
                restored.load_state_dict(checkpoint)
                self.assertEqual(state_digest(restored.state_dict()), state_digest(checkpoint))
                while not original.closed:
                    a = original.step(before_step=lambda: None)
                    b = restored.step(before_step=lambda: None)
                    self.assertEqual(evidence_digest(a), evidence_digest(b))
                self.assertEqual(original.receipt(), restored.receipt())

    def test_rewind_or_wrong_source_does_not_mutate_live_branch(self):
        env, context = fixture()
        run = branch(env, context)
        old = run.state_dict()
        run.step(before_step=lambda: None)
        before = state_digest(run.state_dict())
        with self.assertRaises(ValueError):
            run.load_state_dict(old)
        self.assertEqual(before, state_digest(run.state_dict()))
        wrong = branch(env, context, seed=123).state_dict()
        with self.assertRaises(ValueError):
            run.load_state_dict(wrong)
        self.assertEqual(before, state_digest(run.state_dict()))
        changed = decode_arrays(run.state_dict())
        changed["rows"][0]["public_observation"][0] += 1
        with self.assertRaisesRegex(ValueError, "prefix"):
            run.load_state_dict(encode_arrays(changed))
        self.assertEqual(before, state_digest(run.state_dict()))

    def test_record_failure_retains_attempt_debit_and_latches_terminal(self):
        for at in (0, 1):
            env, context = fixture()
            run = branch(env, context)
            for _ in range(at):
                run.step(before_step=lambda: None)
            before, saved = run.env.state_dict(), run.state_dict()
            debits = []
            def fail(owner, event):
                raise OSError("invented disk failure")
            run.record = fail
            with self.assertRaisesRegex(OSError, "disk failure"):
                run.step(before_step=lambda: debits.append(1))
            self.assertTrue(same_payload(run.env.state_dict(), before))
            self.assertEqual(debits, [1])
            self.assertFalse(run.failure["refunded"])
            for action in (lambda: run.receipt(), lambda: run.load_state_dict(saved),
                           lambda: run.step(before_step=lambda: None)):
                with self.assertRaises(ValueError):
                    action()

    def test_debit_failure_never_steps_and_cannot_retry(self):
        env, context = fixture()
        run = branch(env, context)
        def fail():
            raise RuntimeError("invented budget exhausted")
        with patch.object(FakeEngine, "step", side_effect=AssertionError("must not step")) as step:
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                run.step(before_step=fail)
            step.assert_not_called()
        self.assertIsNotNone(run.failure)

    def test_nan_cost_restores_full_state_and_does_not_supply_label(self):
        env, context = fixture()
        run = branch(env, context)
        before = run.env.state_dict()
        raw_step = FakeEngine.step
        def bad(owner, request):
            obs, reward, done, info = raw_step(owner, request)
            return obs, reward, done, info | {"cost": float("nan")}
        with patch.object(FakeEngine, "step", bad), self.assertRaises(ValueError):
            run.step(before_step=lambda: None)
        self.assertTrue(same_payload(run.env.state_dict(), before))
        with self.assertRaises(ValueError):
            run.receipt()


if __name__ == "__main__":
    unittest.main()
