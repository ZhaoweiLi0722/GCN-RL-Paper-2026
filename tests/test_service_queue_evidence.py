"""Hand-solvable information cases for the independent evidence recurrence."""

import unittest

from evaluation.diagnose_service_queue_evidence import prefix_optimum


def tiny_edges(reveal):
    edges = {}
    for world in ("left", "right"):
        for first in ("L", "R"):
            edges[world, (), first] = {"cost": 0, "step": {"before": {"signal": "hidden"},
                                                         "after": {"signal": world if reveal else "hidden"}}}
            for second in ("L", "R"):
                cost = 0 if (world == "left") == (second == "L") else 4
                edges[world, (first,), second] = {"cost": cost, "step": {"before": {}, "after": {}}}
    return edges


class EvidenceRecurrenceTests(unittest.TestCase):
    def test_public_revelation_permits_feedback(self):
        result = prefix_optimum(tiny_edges(True), (("left", 0.5), ("right", 0.5)), ("L", "R"), 2)
        self.assertEqual(result["cost"], 0)
        self.assertEqual(result["replayed_expected_cost"], 0)
        self.assertEqual([r["actions"][-1] for r in result["witness"]], ["L", "R"])
        self.assertFalse(result["online_parameter_updates"])

    def test_hidden_world_does_not_permit_clairvoyant_actions(self):
        result = prefix_optimum(tiny_edges(False), (("left", 0.5), ("right", 0.5)), ("L", "R"), 2)
        self.assertEqual(result["cost"], 2)
        self.assertEqual([r["actions"][-1] for r in result["witness"]], ["L", "L"])

    def test_probability_weighting_not_uniform_assumption(self):
        result = prefix_optimum(tiny_edges(False), (("left", 0.25), ("right", 0.75)), ("L", "R"), 2)
        self.assertEqual(result["cost"], 1)
        self.assertEqual([r["actions"][-1] for r in result["witness"]], ["R", "R"])

    def test_one_world_has_no_hidden_information(self):
        result = prefix_optimum(tiny_edges(False), (("right", 1.0),), ("L", "R"), 2)
        self.assertEqual(result["cost"], 0)

    def test_distinguishable_root_rejected_by_specialized_recurrence(self):
        edges = tiny_edges(True)
        edges["right", (), "L"]["step"]["before"] = {"signal": "right"}
        with self.assertRaises(ValueError):
            prefix_optimum(edges, (("left", 0.5), ("right", 0.5)), ("L", "R"), 2)


if __name__ == "__main__":
    unittest.main()
