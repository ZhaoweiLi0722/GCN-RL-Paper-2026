"""Arithmetic and information-boundary checks for existing-row repricing."""

from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest

from evaluation.audit_queue_cost_sensitivity import (
    components, dot, enumerate_bounds, path_components, reprice_edge, run,
    sum_components, unchanged_information, weights,
)
from evaluation.diagnose_service_queue_evidence import prefix_optimum


def step(holding=1, labor=2, switching=0.5):
    return {"before": {"epoch": 0}, "after": {"epoch": 1},
            "cost": {"holding": holding, "labor": labor, "switching": switching,
                     "total": holding + labor + switching},
            "receipt": {"labor_cost": labor, "change_cost": switching}}


def edge(closure=None):
    return {"step": step(), "cost": 3.5, "settlement_id": closure}


class QueueRepricingTests(unittest.TestCase):
    def test_multipliers_reject_invalid_values(self):
        for value in (0, -1, math.inf, math.nan, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                weights(1, value, 1)

    def test_nonfinite_or_negative_recorded_cost_rejected(self):
        for value in (math.nan, math.inf, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                components(step(labor=value))

    def test_local_cost_receipts_are_repriced_together(self):
        result = reprice_edge(edge(), {}, weights(1, 2, 0.5))
        self.assertEqual(result["cost"], 5.25)
        self.assertEqual(result["step"]["cost"]["total"], 5.25)
        self.assertEqual(result["step"]["receipt"], {"labor_cost": 4, "change_cost": 0.25})

    def test_closure_charged_once_and_hidden_from_current_observation(self):
        closures = {"tail": {"settlement": {"steps": [step(10, 4, 2), step(20, 6, 0)]}}}
        original = edge("tail")
        result = reprice_edge(original, closures, weights(1, 2, 0.5))
        self.assertEqual(result["cost"], 5.25 + 30 + 20 + 1)
        self.assertEqual(result["step"], reprice_edge(edge(), {}, weights(1, 2, 0.5))["step"])

    def test_missing_closure_is_not_a_free_terminal(self):
        with self.assertRaises(KeyError):
            reprice_edge(edge("absent"), {}, weights(1, 1, 1))

    def test_input_rows_are_immutable(self):
        original = edge("tail")
        closures = {"tail": {"settlement": {"steps": [step()]}}}
        expected = deepcopy((original, closures))
        reprice_edge(original, closures, weights(1, 2, 2))
        self.assertEqual((original, closures), expected)

    def test_complete_path_and_direct_component_accounting_agree(self):
        edges = {("w", (), "a"): edge(), ("w", ("a",), "b"): edge("tail")}
        closures = {"tail": {"settlement": {"steps": [step(10, 3, 2)]}}}
        parts = path_components(edges, closures, "w", ["a", "b"])
        self.assertEqual(parts, {"holding": 12, "labor": 7, "switching": 3})
        self.assertEqual(parts, sum_components([step(), step(), step(10, 3, 2)]))
        factors = weights(1, 0.5, 2)
        self.assertEqual(dot(parts, factors), sum(reprice_edge(e, closures, factors)["cost"] for e in edges.values()))

    def test_known_law_optimum_changes_action_when_cost_ratio_changes(self):
        original = {("w", (), "wait"): {"step": step(10, 0, 0), "cost": 10, "settlement_id": None},
                    ("w", (), "work"): {"step": step(0, 8, 0), "cost": 8, "settlement_id": None}}
        for labor, expected_action, expected_cost in ((0.5, "work", 4), (2, "wait", 10)):
            changed = {key: reprice_edge(e, {}, weights(1, labor, 1)) for key, e in original.items()}
            optimum = prefix_optimum(changed, (("w", 1),), ("wait", "work"), 1)
            self.assertEqual((optimum["root_action"], optimum["cost"]), (expected_action, expected_cost))
            self.assertEqual(enumerate_bounds(changed, (("w", 1),), ("wait", "work"), 1)["open_loop_cost"], expected_cost)

    def test_observation_partition_is_checked_separately_from_objective(self):
        original = {("left", (), "a"): edge(), ("right", (), "a"): edge("tail")}
        closures = {"tail": {"settlement": {"steps": [step(100, 0, 0)]}}}
        changed = {key: reprice_edge(e, closures, weights(1, 2, 2)) for key, e in original.items()}
        worlds = (("left", 0.5), ("right", 0.5))
        self.assertEqual(unchanged_information(original, changed, worlds, ("a",), 1), 1)
        changed["right", (), "a"]["step"]["leaked_future_cost"] = 100
        with self.assertRaises(AssertionError):
            unchanged_information(original, changed, worlds, ("a",), 1)

    def test_bounds_use_world_probabilities(self):
        edges = {("left", (), "a"): {"cost": 0}, ("right", (), "a"): {"cost": 4},
                 ("left", (), "b"): {"cost": 4}, ("right", (), "b"): {"cost": 0}}
        bounds = enumerate_bounds(edges, (("left", 0.25), ("right", 0.75)), ("a", "b"), 1)
        self.assertEqual(bounds, {"open_loop_cost": 1, "open_loop_actions": ("b",), "clairvoyant_cost": 0})

    def test_refuses_existing_output_before_doing_any_analysis(self):
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaises(FileExistsError):
                run(Path(output))


if __name__ == "__main__":
    unittest.main()
