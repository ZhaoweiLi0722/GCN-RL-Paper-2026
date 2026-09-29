"""Count-only comparator interface and persisted-edge replay checks."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from evaluation.audit_service_queue_boundary import CONFIG, read_rows
from evaluation.audit_service_queue_completion_rule import INPUT, REFERENCE, index_edges, replay, compute, run
from src.baselines.completion_count_control import CompletionCounts, completion_count_choice, count_observation


class CountControlTests(unittest.TestCase):
    def test_decisions_are_fixed_and_site_symmetric(self):
        for counts, expected in (((0, 0), "idle"), ((2, 2), "balanced"),
                                 ((2, 1), "left"), ((1, 2), "right")):
            with self.subTest(counts=counts):
                self.assertEqual(completion_count_choice(CompletionCounts(2, counts)), expected)

    def test_only_support_stage_is_counted(self):
        obs = count_observation(0, ("scheduled", "support", "waiting", "active", "done", "support"), (0, 0, 0, 1, 1, 1))
        self.assertEqual(obs, CompletionCounts(0, (1, 1)))

    def test_invalid_projection_rejected(self):
        for stages, sites in (((), ()), (("support",), ()), (("unknown",), (0,)),
                              (("support",), (2,)), (("support",), (True,))):
            with self.subTest(stages=stages, sites=sites), self.assertRaises(ValueError):
                count_observation(0, stages, sites)

    def test_invalid_counts_and_full_state_rejected(self):
        for counts in ((-1, 0), (0.2, 1), (True, 1), (1,), [1, 1]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                CompletionCounts(0, counts)
        with self.assertRaises(ValueError):
            CompletionCounts(-1, (1, 1))
        with self.assertRaises(TypeError):
            completion_count_choice({"remaining_work": [1, 2]})


class RecordedReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(CONFIG.read_text())
        cls.rows = read_rows(INPUT / "tree_transitions.jsonl.gz")
        cls.edges = index_edges(cls.rows)

    def test_replay_has_no_transition_or_planner_calls(self):
        with patch("src.env.service_queue_network.advance", side_effect=AssertionError("new rollout")), \
             patch("src.baselines.service_queue_control.plan_queue_mpc", side_effect=AssertionError("planner")):
            episode, rows = replay(self.fixture, "batch__shared_bottleneck__persistent_change", "left_slow", self.edges)
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(episode["cost"], episode["decision_cost"] + episode["closure_cost"])
        self.assertGreater(episode["closure_cost"], 0)
        self.assertTrue(all(set(r["public_input"]) == {"epoch", "support_by_site"} for r in rows))

    def test_poison_hidden_fields_does_not_change_decisions(self):
        cell, world = "booked_flow__shared_bottleneck__persistent_change", "left_slow"
        local = [copy.deepcopy(r) for r in self.rows if r["cell"] == cell and r["world"] == world]
        for row in local:
            for moment in ("before", "after"):
                row["step"][moment]["remaining_work"] = ["unavailable"] * 8
            for key in ("available_work", "delivered_work", "observation_kind"):
                row["step"]["receipt"][key] = "private"
        actual, _ = replay(self.fixture, cell, world, index_edges(local))
        expected, _ = replay(self.fixture, cell, world, self.edges)
        self.assertEqual(actual, expected)

    def test_future_outcomes_do_not_select_root_action(self):
        cell, world = "batch__nonbinding__no_change", "nominal"
        local = [copy.deepcopy(r) for r in self.rows if r["cell"] == cell and r["world"] == world]
        for row in local:
            row["cost"] += 1000
            row["step"]["cost"]["total"] += 1000
        changed, _ = replay(self.fixture, cell, world, index_edges(local))
        original, _ = replay(self.fixture, cell, world, self.edges)
        self.assertEqual(changed["actions"], original["actions"])
        self.assertAlmostEqual(changed["cost"] - original["cost"], 5000)

    def test_duplicate_and_missing_edges_fail(self):
        with self.assertRaises(ValueError):
            index_edges([self.rows[0], self.rows[0]])
        with self.assertRaises(KeyError):
            replay(self.fixture, "batch__nonbinding__no_change", "nominal", {})

    def test_compute_counts_and_information_order(self):
        summary = json.loads((INPUT / "summary.json").read_text())
        reference = json.loads(REFERENCE.read_text())
        result, episodes, rows = compute(self.fixture, summary, reference, self.rows)
        self.assertEqual((len(episodes), len(rows), len(result["comparisons"])), (12, 60, 8))
        self.assertEqual(result["new_simulator_queries"], 0)
        self.assertTrue(all(r["gap_to_completion_information_bound"] >= -1e-9
                            for r in result["comparisons"].values()))

    def test_existing_output_cannot_be_overwritten(self):
        with self.assertRaises(FileExistsError):
            run(INPUT)


if __name__ == "__main__":
    unittest.main()
