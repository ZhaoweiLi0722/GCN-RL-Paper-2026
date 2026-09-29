"""Independent row-audit and bounded-run checks, without training."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.check_completion_feedback_mechanics import (
    CONFIG, audit_case, noise_tape, quadrature, response_at, run, run_case,
)


class CompletionCheckTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(CONFIG.read_text())

    def test_private_response_families(self):
        f = self.fixture
        self.assertEqual(response_at(f, "persistent_left_slow", 7), (1, 1))
        self.assertEqual(response_at(f, "persistent_left_slow", 8), (0.5, 1.5))
        self.assertEqual(response_at(f, "persistent_right_slow", 8), (1.5, 0.5))
        self.assertEqual(response_at(f, "rapid_alternating", 8), (0.5, 1.5))
        self.assertEqual(response_at(f, "rapid_alternating", 9), (1.5, 0.5))
        self.assertEqual(response_at(f, "unchanged", 20), (1, 1))

    def test_tape_is_stable_and_does_not_use_global_rng(self):
        import numpy as np
        first = noise_tape(self.fixture, 0)
        np.random.random(100)
        self.assertTrue(np.array_equal(first, noise_tape(self.fixture, 0)))
        self.assertFalse(np.array_equal(first, noise_tape(self.fixture, 1)))
        self.assertEqual(first.shape, (288, 2))

    def test_complete_case_repeats_and_independently_audits(self):
        a = run_case(self.fixture, "persistent_left_slow", 1, 0)
        self.assertEqual(a, run_case(self.fixture, "persistent_left_slow", 1, 0))
        audit = audit_case(self.fixture, *a)
        self.assertEqual(audit["completion_events"], 8)
        self.assertGreater(audit["receipt_counts"]["right_censored"], 0)
        self.assertTrue(a[0]["ledger"]["settled"])

    def test_tampered_cost_event_posterior_and_terminal_are_rejected(self):
        case, rows = run_case(self.fixture, "unchanged", 1, 0)
        altered = deepcopy(rows)
        altered[0]["cost"]["total"] += 1
        with self.assertRaises(AssertionError):
            audit_case(self.fixture, case, altered)
        altered = deepcopy(rows)
        altered[1]["receipt"]["completed"] = (False, False) if any(altered[1]["receipt"]["completed"]) else (True, True)
        with self.assertRaises(AssertionError):
            audit_case(self.fixture, case, altered)
        altered = deepcopy(rows)
        altered[2]["filter_after"] = ((1, 0, 0), (1, 0, 0))
        with self.assertRaises(AssertionError):
            audit_case(self.fixture, case, altered)
        altered_case = deepcopy(case)
        altered_case["ledger"]["pending_prepaid_hours"] = (1, 0)
        with self.assertRaises(AssertionError):
            audit_case(self.fixture, altered_case, rows)

    def test_bounded_incomplete_closure_is_failure_not_truncation_success(self):
        fixture = deepcopy(self.fixture)
        fixture["horizon"], fixture["max_closure_steps"] = 1, 1
        case, rows = run_case(fixture, "unchanged", 1, 0)
        self.assertFalse(case["ledger"]["settled"])
        self.assertEqual(len(rows), 2)
        self.assertTrue(case["ledger"]["unfinished_jobs"])
        with self.assertRaises(AssertionError):
            audit_case(fixture, case, rows)

    def test_midpoint_quadrature_checks_expected_action_leverage(self):
        records = quadrature(self.fixture)
        self.assertEqual(len(records), 15)
        for rate in (0.5, 1, 1.5):
            frequencies = [r["midpoint_frequency"] for r in records if r["rate"] == rate]
            self.assertEqual(frequencies[0], 0)
            self.assertTrue(all(a < b for a, b in zip(frequencies, frequencies[1:])))

    def test_existing_output_refused_before_execution(self):
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaises(FileExistsError):
                run(Path(output))


if __name__ == "__main__":
    unittest.main()
