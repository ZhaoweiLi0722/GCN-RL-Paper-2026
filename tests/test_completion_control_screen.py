"""Bounded runner smoke fixtures, provenance and persisted-evidence checks."""

import copy
import itertools
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from evaluation.check_completion_feedback_mechanics import make_config
from evaluation.run_completion_control_screen import (
    CONFIG, adequacy_probe, audit_episode, case_id, choose, deadline_check,
    episode, paired_contrasts, planning_seed, run, world_tapes,
)
from src.baselines.completion_response_filter import CompletionResponseFilter
from src.baselines.completion_rollout_control import action_grid, named_seed
from src.env.completion_feedback_queue import initial_observation


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.screen = json.loads(CONFIG.read_text())
        self.fixture = json.loads(Path(self.screen["mechanics_config"]).read_text())
        self.screen["namespace"] = "S2-unit-fixture-not-scientific-stream"
        self.fixture.update(horizon=3, booked_releases=[0], change_epoch=1)

    def small_episode(self, method="booked_rule", **kwargs):
        return episode(self.screen, self.fixture, "persistent", 1, "discovery", 0, method, **kwargs)

    def test_all_six_methods_settle_and_replay_including_planner(self):
        for method in self.screen["methods"]:
            case, rows, _ = self.small_episode(method)
            case = json.loads(json.dumps(case, allow_nan=False))
            rows = json.loads(json.dumps(rows, allow_nan=False))
            self.assertTrue(case["ledger"]["settled"])
            self.assertEqual(audit_episode(self.screen, self.fixture, case, rows, replan=True)["audit"], "passed")

    def test_positive_tail_is_fully_billed_not_dropped(self):
        case, rows, _ = self.small_episode()
        self.assertGreater(case["closure_intervals"], 0)
        self.assertGreater(case["closure_cost"], 0)
        self.assertEqual(rows[self.fixture["horizon"]]["action"], (0,0))
        self.assertEqual(case["cost"]["total"], sum(r["cost"]["total"] for r in rows))

    def test_partial_rows_emitted_before_deadline_failure(self):
        saved = []
        def emit(row):
            saved.append(row)
            if len(saved) == 2:
                raise TimeoutError("injected fixture deadline")
        with self.assertRaises(TimeoutError):
            self.small_episode(emit=emit)
        self.assertEqual(len(saved), 2)

    def test_immediate_deadline_does_not_run_episode(self):
        with self.assertRaises(TimeoutError):
            self.small_episode(deadline=time.monotonic()-1)
        deadline_check(None)

    def test_unsettled_actual_world_fails_closed(self):
        self.fixture.update(horizon=1, max_closure_steps=1)
        with self.assertRaises(RuntimeError):
            self.small_episode()

    def test_tampered_cost_filter_action_or_noise_hash_rejected(self):
        case, rows, _ = self.small_episode()
        for target, field in (("cost", "total"), ("after", "epoch")):
            bad = copy.deepcopy(rows)
            bad[0][target][field] += 1
            with self.assertRaises(AssertionError):
                audit_episode(self.screen, self.fixture, case, bad)
        bad = copy.deepcopy(rows)
        bad[0]["filter_after"] = ((1,0,0), (1,0,0))
        with self.assertRaises(AssertionError):
            audit_episode(self.screen, self.fixture, case, bad)
        bad = copy.deepcopy(rows)
        bad[0]["action"] = (0.3,0.3)
        with self.assertRaises(AssertionError):
            audit_episode(self.screen, self.fixture, case, bad)
        with self.assertRaises(AssertionError):
            audit_episode(self.screen, self.fixture, dict(case, uniform_sha256="wrong"), rows)

    def test_duplicate_missing_or_reordered_intervals_rejected(self):
        case, rows, _ = self.small_episode()
        for bad in (rows[1:], rows+[rows[-1]], [rows[1],rows[0]]+rows[2:]):
            with self.assertRaises(AssertionError):
                audit_episode(self.screen, self.fixture, case, bad)

    def test_planner_response_and_argmin_are_audited(self):
        case, rows, _ = self.small_episode("id16")
        bad = copy.deepcopy(rows)
        bad[0]["estimate_used"] = (0.5,1.5)
        with self.assertRaises(AssertionError):
            audit_episode(self.screen, self.fixture, case, bad)
        bad = copy.deepcopy(rows)
        bad[0]["decision"]["selected_index"] = 999
        with self.assertRaises(AssertionError):
            audit_episode(self.screen, self.fixture, case, bad)

    def test_world_noise_pairing_split_separation_and_true_iid(self):
        a, p = world_tapes(self.screen, self.fixture, "persistent", "discovery", 0)
        b, f = world_tapes(self.screen, self.fixture, "fast_iid", "discovery", 0)
        np.testing.assert_array_equal(a,b)
        _, reverse = world_tapes(self.screen, self.fixture, "persistent", "discovery", 1)
        np.testing.assert_array_equal(p[1:], np.tile([0.5,1.5], (len(p)-1,1)))
        np.testing.assert_array_equal(reverse[1:], p[1:,::-1])
        orientations = f[1:,0]
        self.assertGreater(len(set(orientations)), 1)
        self.assertTrue(np.any(orientations[1:] == orientations[:-1]))
        c, _ = world_tapes(self.screen, self.fixture, "persistent", "replication", 0)
        self.assertFalse(np.array_equal(a,c))
        self.assertNotEqual(planning_seed(self.screen,"discovery",0,0), named_seed(self.screen["namespace"],"actual","discovery",0))

    def test_same_information_fixed_and_id_equal_before_receipts(self):
        cfg = make_config(self.fixture,1)
        state, mean = initial_observation(cfg), CompletionResponseFilter(**self.fixture["filter"]).mean
        args = (state, mean, cfg, self.fixture, action_grid(4), 17)
        self.assertEqual(choose("id16",*args),choose("fixed16",*args))

    def test_probe_retains_all_budgets_and_depths(self):
        cfg = make_config(self.fixture,1)
        state, filt = initial_observation(cfg), CompletionResponseFilter(**self.fixture["filter"])
        probe = adequacy_probe(self.screen,self.fixture,"persistent",1,(state,filt.mean,filt.posterior))
        self.assertEqual([(o["samples"],o["depth"]) for o in probe["options"]],[(16,1),(64,1),(64,2)])
        self.assertEqual([o["detail"]["candidates"] for o in probe["options"]],[15,15,225])
        self.assertFalse(probe["dominant_zero"])

    def test_contrasts_preserve_worlds_sign_and_independent_splits(self):
        costs = {"booked_rule": 10, "reservation_rule": 9, "fixed16": 8, "fixed64": 7, "id16": 9, "id64": 6}
        cases = [{"case":case_id(f,s,p,w,m), "cost":{"total":costs[m]+w}}
                 for f,s,p,w,m in itertools.product(self.screen["families"],self.screen["downstream_slots"],
                    self.screen["splits"],self.screen["world_indices"],self.screen["methods"])]
        contrasts = paired_contrasts(self.screen,cases)
        self.assertEqual(len(contrasts),60)
        for contrast in contrasts:
            self.assertEqual(len(contrast["pairs"]),4)
            self.assertEqual(contrast["mean"],costs[contrast["left"]]-costs[contrast["right"]])
            self.assertEqual(contrast["lower"],4)
        with self.assertRaises(ValueError):
            paired_contrasts(self.screen,cases+[cases[0]])
        with self.assertRaises(KeyError):
            paired_contrasts(self.screen,cases[1:])

    def test_output_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run(Path(directory))

    def test_recorded_failure_preserves_evidence_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"failed_fixture"
            with patch("evaluation.run_completion_control_screen.locked_files", return_value=[]), \
                    patch("evaluation.run_completion_control_screen.episode", side_effect=TimeoutError("fixture-only")) as mocked:
                with self.assertRaises(TimeoutError):
                    run(output)
                self.assertEqual(mocked.call_count,1)
            status = json.loads((output/"status.json").read_text())
            self.assertEqual(status["status"],"failed")
            self.assertEqual(status["completed"],0)
            self.assertTrue((output/"claim.json").exists())
            self.assertIn("fixture-only",(output/"error.txt").read_text())


if __name__ == "__main__":
    unittest.main()
