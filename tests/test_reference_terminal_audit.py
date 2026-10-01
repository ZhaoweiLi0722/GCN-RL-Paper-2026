"""Artificial arithmetic/checkpoint fixtures, never patient episodes or fits."""

import importlib.util
import math
from pathlib import Path
import unittest

import torch


PATH = Path(__file__).resolve().parents[1] / "reports/2026-10-01-reference-prior-integration/terminal_audit.py"
SPEC = importlib.util.spec_from_file_location("p2_terminal_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def decision(probabilities, reference=0, chosen=0):
    return {"evaluation": {"log_probs": [math.log(p) for p in probabilities],
                           "candidates": {"request_to_class": [reference]}},
            "choice": {"class_index": chosen}}


class TerminalAuditTests(unittest.TestCase):
    def test_prior_margin(self):
        row = AUDIT.distribution_row(decision([.9, .05, .05]))
        self.assertAlmostEqual(row["margin"], math.log(18))
        self.assertAlmostEqual(row["learned_margin"], 0)
        self.assertAlmostEqual(row["reference_probability"], .9)

    def test_reference_is_mapped_not_fixed_index(self):
        row = AUDIT.distribution_row(decision([.1, .9], reference=1, chosen=0))
        self.assertFalse(row["chosen_reference"])
        self.assertAlmostEqual(row["margin"], math.log(9))

    def test_singleton_has_no_alternative_margin(self):
        summary = AUDIT.distribution_summary([AUDIT.distribution_row(decision([1.]))])
        self.assertIsNone(summary["minimum_reference_margin"])
        self.assertIsNone(summary["learned_margin_min_max"])
        self.assertEqual(summary["nonreference_choices"], 0)

    def test_summary_reports_changed_greedy_choice(self):
        rows = [AUDIT.distribution_row(decision([.9, .1])),
                AUDIT.distribution_row(decision([.4, .6], chosen=1))]
        result = AUDIT.distribution_summary(rows)
        self.assertEqual(result["decisions"], 2)
        self.assertEqual(result["nonreference_choices"], 1)
        self.assertLess(result["minimum_reference_margin"], 0)

    def test_bad_receipts_rejected(self):
        malformed = [decision([.5, .4]), decision([.5, .5], reference=3), decision([.5, .5], chosen=-1)]
        nonfinite = decision([.5, .5])
        nonfinite["evaluation"]["log_probs"][0] = float("nan")
        for row in malformed + [nonfinite]:
            with self.assertRaises(ValueError):
                AUDIT.distribution_row(row)
        with self.assertRaises(ValueError):
            AUDIT.distribution_summary([])

    def test_weight_difference_and_moment_counter(self):
        initial = {"policy": {"x": torch.tensor([0., 1.]), "y": torch.tensor([2.])}}
        final = {"manifest": {"format": "candidate-ppo-kernel-v1"},
                 "policy": {"x": torch.tensor([.5, 1.]), "y": torch.tensor([2.])},
                 "optimizer": {"state": {0: {"step": torch.tensor(128.)}}}, "history": [208] * 8, "pending": []}
        result = AUDIT.weight_readback(initial, final)
        self.assertEqual(result["changed_tensors"], 1)
        self.assertEqual(result["max_abs_delta_by_tensor"]["x"], .5)
        self.assertEqual(result["optimizer_steps_per_parameter"], [128.])
        final["policy"]["x"][0] = float("inf")
        with self.assertRaises(ValueError):
            AUDIT.weight_readback(initial, final)

    def test_bc_has_no_pending_field(self):
        initial = {"policy": {"x": torch.tensor([0.])}}
        final = {"manifest": {"format": "candidate-imitation-v1"}, "policy": {"x": torch.tensor([.1])},
                 "optimizer": {"state": {0: {"step": torch.tensor(128.)}}},
                 "steps": 128, "history": [{"steps": 16}] * 8}
        result = AUDIT.weight_readback(initial, final)
        self.assertIsNone(result["pending_segments"])
        self.assertEqual(result["completed_rollouts"], 8)
        final["steps"] = 129
        with self.assertRaises(ValueError):
            AUDIT.weight_readback(initial, final)


if __name__ == "__main__":
    unittest.main()
