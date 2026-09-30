"""Watchdog tests own only tiny artificial Python children, never research jobs."""

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from src.rl.candidate_pilot_resources import PilotBudget
from src.rl.candidate_pilot_watchdog import LedgerDeadline, supervise
from tests.test_candidate_pilot_resources import config


class CandidatePilotWatchdogTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def run_child(self, source, seconds=3):
        return supervise([sys.executable, "-c", source], cwd=self.root,
            stdout_path=self.root / "out.log", stderr_path=self.root / "err.log",
            ledger_path=self.root / "ledger.jsonl", report_path=self.root / "supervisor.json",
            maximum_seconds=seconds, poll_seconds=.01, termination_grace_seconds=.1)

    def test_success_is_reaped_and_evidence_is_no_overwrite(self):
        result = self.run_child("print('invented fixture only')")
        self.assertTrue(result["passed"])
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("invented", (self.root / "out.log").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(result["pid"], 0)
        with self.assertRaises(FileExistsError):
            self.run_child("pass")

    def test_global_deadline_stops_only_owned_child_and_reports_failure(self):
        result = self.run_child("import time; time.sleep(30)", seconds=.08)
        self.assertEqual(result["reason"], "wall_clock_deadline")
        self.assertFalse(result["passed"])
        self.assertLess(result["elapsed_seconds"], 2)
        with self.assertRaises(ProcessLookupError):
            os.kill(result["pid"], 0)

    def test_child_error_is_preserved_not_retried(self):
        result = self.run_child("raise RuntimeError('invented failure')")
        self.assertEqual(result["reason"], "child_failed")
        self.assertFalse(result["automatic_retry"])
        self.assertIn("invented failure", (self.root / "err.log").read_text())

    def test_section_deadline_and_partial_append_are_read_without_refund(self):
        cfg = config()
        cfg["caps"]["seconds"]["preflight_total"] = 1
        budget = PilotBudget(self.root / "ledger.jsonl", cfg)
        self.addCleanup(budget.close)
        watcher = LedgerDeadline(budget.path)
        watcher.poll()
        self.assertIsNone(watcher.active)
        budget.begin("preflight_including_clones")
        budget.debit("environment")
        watcher.poll()
        self.assertEqual(watcher.active, "preflight_including_clones")
        self.assertLess(watcher.deadline(time.monotonic() + 100), time.monotonic() + 1.01)
        budget.finish()
        watcher.poll()
        self.assertIsNone(watcher.active)
        # An incomplete line is temporarily buffered, not accepted as a debit.
        with budget.path.open("ab") as handle:
            handle.write(b'{"unfinished":')
        watcher.poll()
        self.assertEqual(watcher.partial, b'{"unfinished":')
        self.assertEqual(watcher.sequence, 4)

    def test_scope_deadline_terminates_hung_child_without_waiting_global_cap(self):
        cfg = config()
        cfg["caps"]["seconds"]["preflight_total"] = 0
        budget = PilotBudget(self.root / "ledger.jsonl", cfg)
        self.addCleanup(budget.close)
        budget.begin("preflight_including_clones")
        result = self.run_child("import time; time.sleep(30)", seconds=3)
        self.assertEqual(result["active_scope"], "preflight_including_clones")
        self.assertEqual(result["reason"], "wall_clock_deadline")
        self.assertLess(result["elapsed_seconds"], 2)

    def test_replaced_or_corrupted_ledger_fails_closed(self):
        budget = PilotBudget(self.root / "ledger.jsonl", config())
        self.addCleanup(budget.close)
        watcher = LedgerDeadline(budget.path)
        watcher.poll()
        budget.begin("preflight_including_clones")
        watcher.poll()
        with budget.path.open("ab") as handle:
            handle.write(b'{"sequence":2,"previous":"wrong","sha256":"wrong"}\n')
        with self.assertRaises(ValueError):
            watcher.poll()


if __name__ == "__main__":
    unittest.main()
