"""Ledger and harmless subprocess fixtures; no learning or patient simulation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

from src.rl.dynamic_candidate_resources import DynamicCandidateBudget
from src.rl.dynamic_candidate_watchdog import DynamicLedgerDeadline, supervise_dynamic
from src.utils.research_clock import CLOCK_ID, shared_monotonic
from tests.test_dynamic_candidate_execution import tiny_plan


class DynamicWatchdogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_supervisor_setup_is_charged_to_initial_scope(self):
        plan = tiny_plan()
        now = [101.]
        path = self.root / "budget.jsonl"
        budget = DynamicCandidateBudget(path, plan, enabled=True, clock=lambda: now[0], started=100.)
        self.addCleanup(budget.close)
        budget.begin("runtime_input_binding")
        budget.finish()
        self.assertEqual(budget.phase_seconds["runtime_input_binding"], 1.)
        now[0] = 109.
        with self.assertRaises(TimeoutError):
            budget.check()

    def test_expired_setup_cannot_get_a_fresh_child_phase_budget(self):
        budget = DynamicCandidateBudget(self.root / "budget.jsonl", tiny_plan(), enabled=True,
            clock=lambda: 103., started=100.)
        self.addCleanup(budget.close)
        with self.assertRaises(TimeoutError):
            budget.begin("runtime_input_binding")

    def test_incremental_reader_keeps_terminal_deadline_after_scope_finish(self):
        plan = tiny_plan()
        path = self.root / "budget.jsonl"
        budget = DynamicCandidateBudget(path, plan, enabled=True)
        self.addCleanup(budget.close)
        watcher = DynamicLedgerDeadline(path, plan, budget.started)
        for section in plan["sections"]:
            budget.begin(section)
            watcher.poll()
            budget.finish()
            watcher.poll()
        self.assertIsNotNone(watcher.closure_deadline)
        self.assertEqual(set(watcher.closed), set(plan["sections"]))
        self.assertEqual(watcher.deadline(budget.started + 8), watcher.closure_deadline)
        self.assertIsNone(watcher.initial_deadline)

    def test_wrong_admitted_plan_or_origin_is_rejected(self):
        plan = tiny_plan()
        path = self.root / "budget.jsonl"
        budget = DynamicCandidateBudget(path, plan, enabled=True)
        self.addCleanup(budget.close)
        watcher = DynamicLedgerDeadline(path, plan, budget.started - 1)
        with self.assertRaisesRegex(ValueError, "origin"):
            watcher.poll()

    def test_zero_exit_without_ledger_is_not_completion(self):
        result = supervise_dynamic([sys.executable, "-c", "pass"], cwd=self.root, launcher=self.root,
            plan=tiny_plan(), started=shared_monotonic(), poll_seconds=.02)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "missing_budget_claim")
        self.assertEqual(result["exit_code"], 0)

    def test_owned_sleeping_dummy_is_reaped_at_deadline(self):
        result = supervise_dynamic([sys.executable, "-c", "import time; time.sleep(10)"], cwd=self.root,
            launcher=self.root, plan=tiny_plan(), started=shared_monotonic() - 1.7,
            poll_seconds=.02, termination_grace_seconds=.1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "wall_clock_deadline")
        self.assertIsNotNone(result["exit_code"])
        self.assertFalse(result["automatic_retry"])


if __name__ == "__main__":
    unittest.main()
