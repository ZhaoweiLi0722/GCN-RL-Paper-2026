"""Extra accounting tests after the bounded matrix needed no tail continuation.

The recorded sixteen cases remain unchanged; these are deliberate early-cutoff
unit fixtures, not extra scientific seeds or selected successful runs.
"""

import json
import unittest

from evaluation.check_completion_feedback_mechanics import CONFIG, audit_case, run_case


class CompletionClosureTests(unittest.TestCase):
    def test_early_cutoff_finishes_future_bookings_and_charges_tail(self):
        fixture = json.loads(CONFIG.read_text())
        fixture["horizon"] = 1
        case, rows = run_case(fixture, "persistent_right_slow", 1, 0)
        self.assertTrue(case["ledger"]["settled"])
        self.assertGreater(case["closure_intervals"], 24)
        self.assertEqual(rows[1]["action"], (0, 0))
        self.assertEqual(rows[1]["receipt"]["applied_hours"], rows[0]["action"])
        self.assertTrue(all(s == "done" for s in rows[-1]["after"]["stages"]))
        self.assertGreater(sum(row["cost"]["labor"] for row in rows[1:]), 0)
        self.assertGreater(sum(row["cost"]["holding"] for row in rows[1:]), 0)
        self.assertEqual(case["accounting_cost_not_policy_comparison"]["total"],
                         sum(row["cost"]["total"] for row in rows))
        self.assertEqual(audit_case(fixture, case, rows)["completion_events"], 8)

    def test_tail_never_becomes_free_when_decision_cost_is_reported(self):
        fixture = json.loads(CONFIG.read_text())
        fixture["horizon"] = 3
        case, rows = run_case(fixture, "unchanged", 1, 1)
        decision_cost = sum(row["cost"]["total"] for row in rows[:3])
        self.assertGreater(case["accounting_cost_not_policy_comparison"]["total"], decision_cost)
        case["accounting_cost_not_policy_comparison"]["total"] = decision_cost
        with self.assertRaises(AssertionError):
            audit_case(fixture, case, rows)


if __name__ == "__main__":
    unittest.main()
