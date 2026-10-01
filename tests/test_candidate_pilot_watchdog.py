"""Watchdog tests own only tiny artificial Python children, never research jobs."""

import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest

from src.rl.candidate_pilot_resources import PilotBudget
from src.rl.candidate_pilot_watchdog import LedgerDeadline, supervise
from src.rl.candidate_pilot_resources import read_ledger
from src.utils.research_clock import CLOCK_ID, shared_monotonic
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
        self.assertLess(watcher.deadline(shared_monotonic() + 100), shared_monotonic() + 1.01)
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

    def test_fresh_child_clock_is_bracketed_by_parent_clock(self):
        before = shared_monotonic()
        child = float(subprocess.check_output([sys.executable, "-c",
            "from src.utils.research_clock import shared_monotonic; print(shared_monotonic())"],
            cwd=Path(__file__).resolve().parents[1], text=True))
        self.assertLessEqual(before, child)
        self.assertLessEqual(child, shared_monotonic())

    def child_ledger_source(self):
        cfg = config()
        cfg["caps"]["seconds"]["preflight_total"] = 1
        return (f"import sys, time; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r}); "
            "from src.rl.candidate_pilot_resources import PilotBudget; "
            f"budget=PilotBudget('ledger.jsonl', {cfg!r}); budget.begin('preflight_including_clones'); ")

    def test_older_parent_does_not_expire_fresh_child_scope_immediately(self):
        time.sleep(1.1)
        result = self.run_child(self.child_ledger_source() + "time.sleep(30)", seconds=5)
        self.assertEqual(result["reason"], "wall_clock_deadline")
        self.assertEqual(result["clock_id"], CLOCK_ID)
        self.assertEqual(result["active_scope"], "preflight_including_clones")
        self.assertGreaterEqual(result["elapsed_seconds"], .95)
        self.assertLess(result["elapsed_seconds"], 3)
        with self.assertRaises(ProcessLookupError):
            os.kill(result["pid"], 0)

    def test_fresh_child_finishes_scope_under_older_parent(self):
        time.sleep(1.1)
        result = self.run_child(self.child_ledger_source() + "time.sleep(.05); budget.finish(); budget.close()")
        self.assertTrue(result["passed"])
        self.assertEqual(result["last_ledger_sequence"], 3)

    def test_injected_clock_is_readable_historically_but_not_live(self):
        budget = PilotBudget(self.root / "ledger.jsonl", config(), clock=lambda: 100.)
        budget.begin("preflight_including_clones")
        budget.close()
        self.assertEqual(read_ledger(budget.path)["events"], 2)
        with self.assertRaisesRegex(ValueError, "cross-process"):
            LedgerDeadline(budget.path).poll()

    def test_legacy_clock_ledger_is_preserved_but_not_live_compatible(self):
        from src.rl.candidate_pilot_resources import digest
        budget = PilotBudget(self.root / "ledger.jsonl", config())
        budget.close()
        claim = json.loads(budget.path.read_text())
        del claim["clock_id"], claim["sha256"]
        claim["sha256"] = digest(claim)
        budget.path.write_text(json.dumps(claim) + "\n")
        self.assertEqual(read_ledger(budget.path)["events"], 1)
        result = self.run_child("import time; time.sleep(30)")
        self.assertEqual(result["reason"], "watchdog_or_launch_error")
        self.assertIn("cross-process", result["error"])


if __name__ == "__main__":
    unittest.main()
