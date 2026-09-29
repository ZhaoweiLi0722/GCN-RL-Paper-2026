"""Tiny real-environment session tests, never aggregate policy performance."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluation.check_prospective_session import CONFIG, load_fixture, make_session, advance, summary
from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import ProspectiveDDPGKernel, state_digest
from src.rl.prospective_patient_session import (
    PatientLearningSession, encode_arrays, decode_arrays, restore_record, scheduled_windows,
)


class ProspectiveSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = json.loads(CONFIG.read_text())

    def session(self, index=0, mode="online"):
        return make_session(self.config, index, mode)

    def test_explicit_opt_in_and_session_bounds(self):
        session = self.session()
        options = dict(trajectory_id="unit", max_steps=6, return_steps=3, horizon_end="truncation",
                       noise_options=self.config["noise"])
        for changes in ({}, {"enabled": True, "max_steps": 7}, {"enabled": True, "return_steps": 4},
                        {"enabled": True, "noise_seed": -1},
                        {"enabled": True, "noise_options": {"mu": 0., "theta": .1, "sigma": float("nan")}}):
            with self.assertRaises(ValueError):
                PatientLearningSession(session.collector.env, session.collector.producer,
                                       session.learner, **(options | changes))

    def test_codec_preserves_array_types_and_rejects_mismatched_descriptor(self):
        value = {"arrays": [np.arange(4, dtype=dtype) for dtype in (np.float32, np.float64, np.int64, np.uint8)],
                 "bool": np.array([True, False]), "tuple": (np.float64(2.),)}
        encoded = encode_arrays(value)
        decoded = decode_arrays(encoded)
        for left, right in zip(value["arrays"], decoded["arrays"]):
            self.assertEqual(left.dtype, right.dtype)
            np.testing.assert_array_equal(left, right)
        self.assertEqual(state_digest(encoded), state_digest(encode_arrays(decoded)))
        bad = copy.deepcopy(encoded)
        bad["arrays"][0]["ndarray_dtype"] = "<f8"
        with self.assertRaisesRegex(ValueError, "dtype"):
            decode_arrays(bad)
        with self.assertRaises(ValueError):
            encode_arrays(np.array([object()], dtype=object))

    def test_zero_noise_matches_deterministic_deployment_and_noise_stays_specimen_only(self):
        for index in range(3):
            session = self.session(index)
            zero = np.zeros_like(session.noise.state)
            decision, expected = session.decision(zero)
            np.testing.assert_array_equal(decision.request, expected)
            noisy, _ = session.decision(np.full_like(zero, .2))
            n = zero.size
            np.testing.assert_array_equal(noisy.request[n:], decision.request[n:])
            self.assertTrue(np.any(noisy.request[:n] != decision.request[:n]))
            self.assertTrue(np.all(np.abs(noisy.request) <= 1))

    def test_pending_windows_and_update_schedule_are_exact(self):
        session = self.session()
        for step in range(1, 7):
            event = session.step()
            self.assertEqual(session.emitted_count, 6 if step == 6 else max(0, step - 2))
            self.assertEqual(len(session.pending), 0 if step == 6 else min(step, 2))
            self.assertEqual(session.learner.total_updates, max(0, step - 3))
            self.assertEqual(event["update"] is not None, step >= 4)
        self.assertEqual([w[0].step_index for w in session.learner.windows], [4, 5, 2, 3])
        self.assertEqual([len(w) for w in session.learner.windows], [2, 1, 3, 3])
        self.assertEqual(summary(session)["emitted_lengths"], [3, 3, 3, 3, 2, 1])

    def test_continuous_and_resumed_live_paths_all_modes_and_boundaries(self):
        for index in range(3):
            for mode in ("online", "frozen"):
                live = self.session(index, mode)
                initial_weights = live.learner.agent.weights_digest()
                advance(live, 4)
                checkpoint = live.state_dict()
                fresh = self.session(index, mode)
                fresh.load_state_dict(checkpoint)
                self.assertEqual(state_digest(checkpoint), state_digest(fresh.state_dict()))
                advance(live, 2)
                advance(fresh, 2)
                self.assertEqual(live.events, fresh.events)
                self.assertEqual(state_digest(live.state_dict()), state_digest(fresh.state_dict()))
                self.assertEqual(live.learner.total_updates, 3 if mode == "online" else 0)
                self.assertEqual(initial_weights == live.learner.agent.weights_digest(), mode == "frozen")
                last = live.events[-1]["receipt"]["record"]
                self.assertEqual(last["terminated"], index == 2)
                self.assertEqual(last["truncated"], index != 2)
                self.assertGreater(summary(live)["routing_count"], 0)

    def test_initial_early_and_closed_checkpoints_round_trip_without_new_steps(self):
        session = self.session()
        for index in range(7):
            before = session.state_dict()
            fresh = self.session()
            fresh.load_state_dict(before)
            self.assertEqual(state_digest(before), state_digest(fresh.state_dict()))
            if index < 6:
                session.step()
        before = state_digest(session.state_dict())
        with self.assertRaisesRegex(ValueError, "closed"):
            session.step()
        self.assertEqual(before, state_digest(session.state_dict()))

    def test_flushed_endpoint_targets_respect_all_three_boundary_mappings(self):
        for index in range(3):
            session = self.session(index)
            advance(session, 6)
            records = [restore_record(event["receipt"]["record"]) for event in session.events]
            windows, _ = scheduled_windows(records, session.learner.contract.replay, 3, True)
            batch = session.learner._prepare(windows)
            expected = [.9 ** 3] * 3 + ([0., 0., 0.] if index == 2 else [.9 ** 3, .9 ** 2, .9])
            np.testing.assert_allclose(batch.bootstrap_discounts[:, 0].numpy(), expected, rtol=1e-6)
            _, next_q, targets = session.learner.agent.replay_forward(batch)
            np.testing.assert_allclose(targets.numpy(), batch.rewards.numpy() + np.array(expected)[:, None] * next_q.numpy(),
                                       rtol=2e-6, atol=1e-7)

    def test_environment_rng_drift_is_rejected_before_noise_or_learning(self):
        session = self.session()
        session.collector.env.rng.random()
        before = state_digest(session.state_dict())
        with self.assertRaisesRegex(ValueError, "drift"):
            session.step()
        self.assertEqual(before, state_digest(session.state_dict()))

    def test_full_transaction_rolls_back_after_update_failure(self):
        session = self.session()
        advance(session, 3)
        before = session.state_dict()
        with patch.object(ProspectiveDDPGKernel, "update", side_effect=RuntimeError("injected kernel failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                session.step()
        self.assertEqual(state_digest(before), state_digest(session.state_dict()))
        reference = self.session()
        reference.load_state_dict(before)
        self.assertEqual(session.step(), reference.step())

    def test_full_transaction_rolls_back_after_environment_already_advanced(self):
        session = self.session()
        session.step()
        before = session.state_dict()
        original = PatientConditionCapacityEnv.step

        def fail_after_step(env, action):
            original(env, action)
            raise RuntimeError("injected post-env failure")

        with patch.object(PatientConditionCapacityEnv, "step", fail_after_step):
            with self.assertRaisesRegex(RuntimeError, "post-env"):
                session.step()
        self.assertEqual(state_digest(before), state_digest(session.state_dict()))

    def test_malformed_session_is_rejected_without_mutating_live_state(self):
        session = self.session()
        advance(session, 4)
        original = session.state_dict()
        mutations = []
        for path, value in ((["manifest_sha256"], "bad"), (["manifest", "max_steps"], 5),
                            (["collector", "index"], 3), (["collector", "closed"], True),
                            (["collector", "expected_token"], "bad"), (["emitted_count"], 3),
                            (["pending"], []), (["kernel", "position"], 3),
                            (["events", 0, "noise"], [0., 0., 0.]),
                            (["events", 0, "receipt", "record", "state_token"], "bad"),
                            (["events", 0, "receipt", "record", "raw_reward"], 0.),
                            (["events", 0, "receipt", "gate"], [2.] * 12),
                            (["events", 3, "update", "update"], 2),
                            (["noise", "rng_state", "state", "state"], 5)):
            bad = copy.deepcopy(original)
            cursor = bad
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            mutations.append(bad)
        for part in ("noise", "environment"):
            bad = copy.deepcopy(original)
            tensor = bad[part]["state"]["tensor"] if part == "noise" else bad[part]["arrays"]["reagents"]["tensor"]
            tensor.fill_(float("nan"))
            mutations.append(bad)
        bad = copy.deepcopy(original)
        del bad["environment"]["arrays"]["reagents"]
        mutations.append(bad)
        bad = copy.deepcopy(original)
        bad["kernel"]["windows"][0][0]["semantics"]["gamma"] = .5
        mutations.append(bad)
        for number, bad in enumerate(mutations):
            with self.subTest(number=number), self.assertRaises((ValueError, KeyError, TypeError, AssertionError)):
                session.load_state_dict(bad)
            self.assertEqual(state_digest(original), state_digest(session.state_dict()))

    def test_future_environment_snapshot_cannot_hide_behind_old_cursor(self):
        session = self.session()
        advance(session, 4)
        original = session.state_dict()
        session.step()
        future = session.state_dict()["environment"]
        session.load_state_dict(original)
        bad = copy.deepcopy(original)
        bad["environment"] = future
        with self.assertRaises(ValueError):
            session.load_state_dict(bad)
        self.assertEqual(state_digest(original), state_digest(session.state_dict()))

    def test_no_overwrite_and_safe_checkpoint_round_trip(self):
        session = self.session()
        advance(session, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.pt"
            session.save(path)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                session.save(path)
            self.assertEqual(path.read_bytes(), before)
            fresh = self.session()
            fresh.load(path)
            self.assertEqual(state_digest(session.state_dict()), state_digest(fresh.state_dict()))
            payload = torch.load(path, weights_only=True)
            payload["state_sha256"] = "bad"
            bad_path = Path(directory) / "bad.pt"
            torch.save(payload, bad_path)
            with self.assertRaisesRegex(ValueError, "checksum"):
                fresh.load(bad_path)

    def test_fixture_scope_and_hashes_cannot_be_overridden(self):
        for key, value in (("max_steps", 100), ("fixture_seed", 60), ("device", "mps"),
                           ("performance_evaluation", True), ("scientific_launch_authorized", True),
                           ("environment_source_sha256", "bad"), ("learner_evidence_sha256", "bad")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_fixture(self.config | {key: value})


if __name__ == "__main__":
    unittest.main()
