"""Raw preservation using handwritten accounting and a non-simulating facade."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.rl.candidate_pilot_recording import EpisodeRecorder, write_json_once
from tests.test_candidate_pilot_verification import raw_fixture


class CandidatePilotRecordingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.cfg, header, self.rows, self.final = raw_fixture()
        self.state = header["initial_state"]
        self.fake = SimpleNamespace(index=0, closed=False, trajectory_id=header["trajectory_id"],
            source_id=header["source_id"], split="test", selection="greedy", manifest={"invented": True},
            last_inference_seconds=.001, env=SimpleNamespace(t=0, state_dict=lambda: copy.deepcopy(self.state)),
            learner=SimpleNamespace(policy=SimpleNamespace(snapshot_sha256=lambda: "a" * 64)),
            save=lambda path: write_json_once(path, {"invented_not_a_real_checkpoint": True}))
        self.recorder = EpisodeRecorder(self.root, "fake/world0", self.fake, self.cfg, block=60,
            representation="graph", role="ppo", world_index=0, seed=123)
        self.addCleanup(self.recorder.handle.close)

    def advance(self, step):
        self.fake.index = step + 1
        self.fake.env.t = step + 1
        if step == 1:
            self.fake.closed = True
            self.state = self.final
        self.recorder.append(self.rows[step]["event"])

    def test_complete_recorder_is_raw_verified_and_cannot_overwrite(self):
        self.advance(0)
        prefix = self.recorder.snapshot()
        self.advance(1)
        self.recorder.assert_prefix(prefix)
        index = self.recorder.finish()
        self.assertEqual(set(index), {"header", "events", "final_state"})
        result = json.loads((self.root / "fake/world0/outcome.json").read_text())
        self.assertEqual(result["cost"], 22.)
        self.assertEqual(result["terminal_active"], 1)
        with self.assertRaises(ValueError):
            self.recorder.finish()
        with self.assertRaises(FileExistsError):
            EpisodeRecorder(self.root, "fake/world0", self.fake, self.cfg, block=60,
                representation="graph", role="ppo", world_index=0, seed=123)

    def test_partial_failure_preserves_bytes_without_outcome(self):
        self.advance(0)
        before = self.recorder.path.read_bytes()
        with self.assertRaises(ValueError):
            self.recorder.finish()
        self.recorder.close_partial()
        self.assertEqual(before, self.recorder.path.read_bytes())
        self.assertFalse((self.recorder.directory / "outcome.json").exists())
        with self.assertRaises(ValueError):
            self.recorder.append(self.rows[1]["event"])

    def test_duplicate_or_skipped_step_and_corrupt_prefix_rejected(self):
        self.advance(0)
        saved = self.recorder.snapshot()
        with self.assertRaises(ValueError):
            self.recorder.append(self.rows[0]["event"])
        bad = copy.deepcopy(saved)
        bad["file"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.recorder.assert_prefix(bad)

    def test_disk_write_failure_poisoned_recorder_keeps_existing_rows(self):
        self.fake.index = 1
        with patch("src.rl.candidate_pilot_recording.os.fsync", side_effect=OSError("invented disk failure")):
            with self.assertRaises(OSError):
                self.recorder.append(self.rows[0]["event"])
        self.assertTrue(self.recorder.failed)
        self.assertTrue(self.recorder.path.read_bytes())
        self.assertFalse((self.recorder.directory / "outcome.json").exists())

    def test_raw_audit_failure_keeps_final_snapshot_but_no_outcome(self):
        self.advance(0)
        self.advance(1)
        self.state["scalars"]["cumulative_lost"] = 99
        with self.assertRaises(ValueError):
            self.recorder.finish()
        self.assertTrue((self.recorder.directory / "final_state.json").exists())
        self.assertFalse((self.recorder.directory / "outcome.json").exists())
        self.assertTrue(self.recorder.failed)

    def test_actual_collector_receipt_shape_flows_through_independent_reader_on_fake_steps(self):
        from src.env.patient_capacity_planning import PatientConditionCapacityEnv
        from src.rl.candidate_pilot_verification import CAUSES
        from tests import test_candidate_patient_session as fixtures
        from tests.test_candidate_pilot_verification import patient

        fixtures.CandidatePatientSessionTests.setUp(self)
        def state(env):
            value = fixtures.fixture_state(env)
            value.update(scalars={"t": env.t, "_episode_seed": 123, "cumulative_enrolled": 12,
                                  "cumulative_lost": env.t, "cumulative_served": env.t},
                patients={f"p{i}": patient(f"p{i}", "lost" if i % 2 else "delivered")
                          if i < 2 * env.t else patient(f"p{i}") for i in range(12)},
                patient_queues=[[f"p{i}" for i in range(2 * env.t, 12)], []],
                in_production_patients=[[[], []], [[], []]], specimen_transits=[], product_return_transits=[])
            return value
        def step(env, action):
            raw, reward, done, info = fixtures.fixture_step(env, action)
            info.update({k: np.zeros(2) for k in CAUSES})
            info.update(demand=np.zeros(2), patients_lost_waiting_ineligible=np.array([1., 0.]),
                identity_active_count=12 - 2 * env.t, identity_terminal_count=2 * env.t,
                waiting_patients=np.array([12 - 2 * env.t, 0.]), in_production_patients=np.zeros(2),
                specimen_in_transit=np.zeros(2), completion_service_level=env.t / 12)
            return raw, reward, done, info
        with patch.object(PatientConditionCapacityEnv, "state_dict", state), patch.object(PatientConditionCapacityEnv, "step", step):
            run = fixtures.session()
            cfg = copy.deepcopy(self.cfg)
            cfg["objective"].update(horizon=4, reward_scale=.01, transfer_scale=run.env.config.max_specimen_transfer)
            recorder = EpisodeRecorder(self.root, "fake/real-receipt-schema", run, cfg, block=60,
                representation="graph", role="ppo", world_index=0, seed=123)
            self.addCleanup(recorder.handle.close)
            while not run.closed:
                recorder.append(run.step(before_step=lambda: None))
            recorder.finish()
            result = json.loads((recorder.directory / "outcome.json").read_text())
            self.assertEqual(result["losses"], 4)
            self.assertEqual(result["completions"], 4)
            self.assertEqual(result["terminal_active"], 4)
            self.assertEqual(result["cost"], 264.)


if __name__ == "__main__":
    unittest.main()
