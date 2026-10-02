"""Invented text artifacts only; no model files, loads, fitting or simulation."""

import json
from pathlib import Path
import tempfile
import unittest

from src.rl.time_baseline_sequence import TimeBaselineSequence


class TimeBaselineSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proposal = json.loads((Path(__file__).parents[1] /
            "specs/2026-10-02-time-baseline-comparison/proposal.json").read_text())
        self.sequence = TimeBaselineSequence(self.proposal, self.root, enabled=True)
        (self.root / "evidence.txt").write_text("invented engineering evidence only\n")

    def finish(self):
        self.sequence.finish(["evidence.txt"])

    def reach_seal(self):
        while self.sequence.next_job != "all_model_seal":
            self.sequence.begin(self.sequence.next_job)
            self.finish()
        self.sequence.begin("all_model_seal")

    def seal(self):
        self.reach_seal()
        paths = {}
        for index, key in enumerate(self.sequence.models):
            path = self.root / f"invented-owner{index}.txt"
            path.write_text(key)
            paths[key] = path.name
        self.sequence.seal_models(paths)
        self.finish()
        return paths

    def test_full_schedule_and_recovery_preserve_all_twelve_seals(self):
        self.seal()
        self.assertEqual(len(self.sequence.seals), 12)
        while self.sequence.next_job is not None:
            self.sequence.begin(self.sequence.next_job)
            self.finish()
        self.assertEqual(len(self.sequence.completed), 33)
        clone = TimeBaselineSequence(self.proposal, self.root, enabled=True)
        clone.load_state_dict(self.sequence.state_dict())
        self.assertEqual(clone.state_dict(), self.sequence.state_dict())

    def test_test_access_cannot_skip_training_or_partial_seal(self):
        with self.assertRaises(ValueError):
            self.sequence.begin("final_evaluation/block60/current_ppo")
        self.reach_seal()
        with self.assertRaises(ValueError):
            self.finish()
        with self.assertRaises(ValueError):
            self.sequence.seal_models({})

    def test_byte_change_blocks_test_and_restore(self):
        paths = self.seal()
        saved = self.sequence.state_dict()
        (self.root / next(iter(paths.values()))).write_text("changed invented owner")
        with self.assertRaises(ValueError):
            self.sequence.begin(self.sequence.next_job)
        clone = TimeBaselineSequence(self.proposal, self.root, enabled=True)
        with self.assertRaises(ValueError):
            clone.load_state_dict(saved)
        self.assertEqual(clone.completed, [])

    def test_failed_sequence_cannot_resume_or_reenter(self):
        saved = self.sequence.state_dict()
        self.sequence.begin(self.sequence.next_job)
        self.sequence.fail("invented terminal failure")
        with self.assertRaises(ValueError):
            self.sequence.load_state_dict(saved)
        with self.assertRaises(ValueError):
            self.sequence.begin("runtime_input_binding")


if __name__ == "__main__":
    unittest.main()
