"""Invented JSON artifacts only, no simulator or model loading/optimization."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src.env.cohort_followup import CohortTailSpec
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.cohort_recording import CohortRecorder
from tests.test_cohort_verification import fixture


class FakePrefixRecorder:
    def __init__(self, root, relative, session, config, **metadata):
        self.root, self.path, self.session = root, root / relative, session
        self.path.mkdir(parents=True, exist_ok=False)
        self.count, self.failed = 0, False

    def append(self, event):
        self.count += 1

    def finish(self):
        for name in ("header", "events", "final_state"):
            write_json_once(self.path / (name + ".json"),
                            self.session.env.state_dict() if name == "final_state" else {"invented": True})
        return {name: file_record(self.root, self.path / (name + ".json"))
                for name in ("header", "events", "final_state")}

    def snapshot(self):
        return {"invented_rows": self.count}

    def close_partial(self):
        self.failed = True


class CohortRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.start, self.end, self.rows = fixture()
        self.env = SimpleNamespace(_cohort_closed=False, cohort_spec=CohortTailSpec(5, 2, 3),
            config=SimpleNamespace(num_facilities=1), state_dict=lambda: copy.deepcopy(self.start))
        self.prefix = SimpleNamespace(env=self.env, trajectory_id="invented/test/0",
            source_id="invented", split="test", producer=SimpleNamespace(anchor_config={"num_facilities": 1}))
        self.writer = CohortRecorder(self.root, "episode0", self.prefix, {},
            prefix_recorder_type=FakePrefixRecorder, block=60, role="frozen", seed=101)
        self.addCleanup(self.writer.close_partial)

    def prepare(self):
        index = self.writer.finish_prefix(self.prefix)
        self.env._cohort_closed = True
        self.env.state_dict = lambda: copy.deepcopy(self.end)
        self.collection = SimpleNamespace(closed=True, failure=None, env=self.env,
            prefix_receipt=index, trajectory_id=self.prefix.trajectory_id,
            tail_events=copy.deepcopy(self.rows), objective="none", state_dict=lambda: {"invented": True})
        for row in self.rows:
            self.writer.record_tail(self.collection, row)
        return self.collection

    def test_two_part_evidence_independent_costs_and_no_target_for_test_data(self):
        c = self.prepare()
        old = self.writer.snapshot()
        self.writer.assert_prefix(old)
        index = self.writer.finish(c)
        report = json.loads((self.writer.tail_dir / "outcome.json").read_text())
        self.assertEqual(report["tail_cost"], 6.)
        self.assertEqual(report["tail_completions"], 3)
        self.assertTrue((self.writer.tail_dir / "collector.pt").is_file())
        self.assertFalse((self.writer.tail_dir / "training-target.json").exists())
        for part in index.values():
            for value in part.values():
                self.assertEqual(value, file_record(self.root, value["path"]))
        with self.assertRaises(ValueError):
            self.writer.assert_prefix(old)
        self.writer.assert_prefix(self.writer.snapshot())
        with self.assertRaises(ValueError):
            self.writer.finish(c)

    def test_nonoverwriting_paths_and_duplicate_tail_rows(self):
        with self.assertRaises(FileExistsError):
            CohortRecorder(self.root, "episode0", self.prefix, {}, prefix_recorder_type=FakePrefixRecorder)
        with self.assertRaises(ValueError):
            CohortRecorder(self.root, "../escape", self.prefix, {}, prefix_recorder_type=FakePrefixRecorder)
        c = self.prepare()
        with self.assertRaises(ValueError):
            self.writer.record_tail(c, self.rows[-1])
        self.assertEqual(self.writer.count, 3)

    def test_tampered_persisted_rows_fail_without_overwriting_partial_evidence(self):
        c = self.prepare()
        snapshot = self.writer.snapshot()
        path = self.writer.tail_dir / "events.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(self.rows[0]) + "\n")
        with self.assertRaises(ValueError):
            self.writer.assert_prefix(snapshot)
        with self.assertRaisesRegex(ValueError, "persisted tail"):
            self.writer.finish(c)
        self.assertTrue(self.writer.failed)
        self.assertEqual(len(path.read_text().splitlines()), 4)
        self.assertTrue((self.writer.tail_dir / "final_state.json").is_file())
        self.assertFalse((self.writer.tail_dir / "outcome.json").exists())
        with self.assertRaises(ValueError):
            self.writer.finish(c)

    def test_float64_tail_request_preserved_in_json(self):
        self.writer.finish_prefix(self.prefix)
        row = copy.deepcopy(self.rows[0])
        row["action"][0] = 1e-12
        c = SimpleNamespace(trajectory_id=self.prefix.trajectory_id)
        self.writer.record_tail(c, row)
        saved = json.loads((self.writer.tail_dir / "events.jsonl").read_text())
        self.assertEqual(saved["action"][0], 1e-12)


if __name__ == "__main__":
    unittest.main()
