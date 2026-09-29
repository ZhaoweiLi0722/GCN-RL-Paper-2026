"""S3 numerical contracts and tiny invented-state recording/audit smoke."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from evaluation import check_completion_action_ranking as s3
from evaluation.check_completion_feedback_mechanics import CONFIG as MECHANICS, make_config
from src.baselines.completion_rollout_control import action_grid, named_seed
from src.env.completion_feedback_batch import forecast
from src.env.completion_feedback_queue import advance, initial_observation


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(MECHANICS.read_text())
        self.cfg = make_config(self.fixture, 1)
        self.state = initial_observation(self.cfg)
        self.grid = action_grid(4)
        self.probe = {"family": "invented_a", "slots": 1, "state": asdict(self.state),
                      "response_estimate": [1.0, 1.0], "options": [
                          {"depth": 1, "samples": 16, "action": [0, .75]},
                          {"depth": 1, "samples": 64, "action": [.25, .75]}]}

    def contexts(self, probes=None, count=1):
        probes = [self.probe] if probes is None else probes
        return s3.canonical_contexts(probes, self.fixture, self.grid, len(probes), count)

    def test_duplicate_public_contexts_ignore_family_and_private_fields(self):
        second = deepcopy(self.probe)
        second.update(family="invented_b", planning_seed=123, private_response=[9, 9])
        first = self.contexts()[0]
        merged = self.contexts([self.probe, second])[0]
        self.assertEqual(first["id"], merged["id"])
        self.assertEqual(len(merged["aliases"]), 2)
        self.assertEqual(s3.context_digest(merged["public"]), merged["id"])

    def test_public_capacity_response_state_and_mechanics_change_identity(self):
        first = self.contexts()[0]
        for key in ("downstream_slots", "response", "observation", "mechanics"):
            public = deepcopy(first["public"])
            if key == "downstream_slots":
                public[key] = 4
            elif key == "response":
                public[key][0] = .5
            elif key == "observation":
                public[key]["epoch"] = 1
            else:
                public[key]["holding_cost"] = 2
            self.assertNotEqual(first["id"], s3.context_digest(public))

    def test_missing_duplicate_or_incompatible_probes_fail(self):
        with self.assertRaises(ValueError):
            self.contexts([self.probe, self.probe])
        with self.assertRaises(ValueError):
            s3.canonical_contexts([], self.fixture, self.grid, 1, 1)
        second = deepcopy(self.probe)
        second["family"] = "invented_b"
        second["options"][0]["action"] = [0, 0]
        with self.assertRaises(ValueError):
            self.contexts([self.probe, second])
        with self.assertRaises(ValueError):
            self.contexts(count=2)

    def test_missing_duplicate_archived_action_fails(self):
        for options in ([self.probe["options"][0]], self.probe["options"]+[self.probe["options"][0]]):
            probe = dict(self.probe, options=options)
            with self.assertRaises(ValueError):
                self.contexts([probe])

    def test_prefix_and_chunk_boundaries_preserve_path_order(self):
        seed = named_seed("S3-unit-only", "invented-context", "selection")
        full = np.random.Generator(np.random.PCG64(seed)).random((512, 288, 2))
        chunked = np.concatenate([v for _, _, v in s3.noise_chunks(seed, 512, 288, 128)])
        small = np.concatenate([v for _, _, v in s3.noise_chunks(seed, 128, 288, 128)])
        np.testing.assert_array_equal(full, chunked)
        np.testing.assert_array_equal(small, full[:128])
        self.assertNotEqual(seed, named_seed("S3-unit-only", "invented-context", "validation_a"))
        self.assertNotEqual(seed, named_seed("S3-unit-only", "another-context", "selection"))

    def test_paired_se_uses_covariance_and_independent_arithmetic(self):
        left, right = [100, 200, 300, 400], [97, 197, 297, 397]
        expected = {"mean_left_minus_right": 3.0, "paired_mc_se": 0.0, "sign": 1}
        self.assertEqual(s3.paired_stats(left, right), expected)
        self.assertEqual(s3.paired_stats(left, right, independent=True), expected)
        self.assertAlmostEqual(s3.paired_stats([1, 3], [0, 0])["paired_mc_se"], 1)
        with self.assertRaises(ValueError):
            s3.paired_stats([1], [1, 2])
        with self.assertRaises(ValueError):
            s3.paired_stats([float("nan"), 0], [0, 0])

    def test_all_pairs_stable_ties_and_nested_budgets(self):
        costs = np.tile(np.arange(8, dtype=float), (15, 1))
        stats = s3.block_statistics(costs, self.grid, [2, 4, 8])
        self.assertEqual(stats, s3.block_statistics(costs, self.grid, [2, 4, 8], independent=True))
        for row in stats:
            self.assertEqual(row["argmin_index"], 0)
            self.assertEqual(row["exact_tied_indices"], list(range(15)))
            self.assertEqual(len(row["pairs"]), 105)
        self.assertEqual(s3.frozen_choice(costs, self.grid)["index"], 0)
        with self.assertRaises(ValueError):
            s3.block_statistics(costs[:, :-1], self.grid, [2, 4, 8])

    def test_frozen_selection_cannot_follow_validation_winner(self):
        context = self.contexts()[0]
        selection = np.full((15, 8), 4.0)
        selection[3] = 1
        chosen = s3.frozen_choice(selection, self.grid)
        validation = np.zeros((15, 8))
        validation[3] = 100
        comparison = s3.validation_comparison(validation, chosen["index"], context, [8])
        self.assertEqual(chosen["index"], 3)
        self.assertEqual(comparison[0]["comparators"]["booked_rule"]["mean_left_minus_right"], 100)
        self.assertEqual(chosen, s3.frozen_choice(selection, self.grid))

    def test_raw_batch_scalar_costs_and_queries(self):
        tape = np.random.Generator(np.random.PCG64(311)).random((3, 288, 2))
        result = forecast(self.state, np.asarray(self.grid)[:, None, :], (1, 1), tape, self.cfg, 32)
        total_queries = 0
        for a, action in enumerate(self.grid):
            for sample in range(3):
                cost, queries = s3.scalar_cost(self.state, action, (1, 1), tape[sample], self.cfg, 32)
                self.assertEqual(result["sample_costs"][a, sample], cost)
                total_queries += queries
        self.assertEqual(result["transition_queries"], total_queries)

    def test_first_request_delayed_and_full_tail_charged(self):
        no, _, _ = advance(self.state, (0, 0), (1, 1), (0, 0), self.cfg)
        yes, _, _ = advance(self.state, (1, 0), (1, 1), (0, 0), self.cfg)
        self.assertEqual(no.stages, yes.stages)
        self.assertNotEqual(no.pending_hours, yes.pending_hours)
        tape = np.random.Generator(np.random.PCG64(9)).random((257, 2))
        value, queries = s3.scalar_cost(self.state, (1, 0), (1, 1), tape, self.cfg, 1)
        self.assertGreater(queries, 24)
        self.assertGreater(value, 20)
        result = forecast(self.state, [[(1, 0)]], (1, 1), tape[None, :, :], self.cfg, 1)
        self.assertEqual(result["sample_costs"][0, 0], value)

    def test_closure_and_deadline_fail_closed(self):
        with self.assertRaises(RuntimeError):
            s3.scalar_cost(self.state, (0, 0), (1, 1), np.full((2, 2), .999), self.cfg, 32)
        with self.assertRaises(RuntimeError):
            forecast(self.state, [[(0, 0)]], (1, 1), np.full((1, 2, 2), .999), self.cfg, 32)
        with self.assertRaises(TimeoutError):
            s3.check_deadline(time.monotonic()-1)

    def test_raw_record_missing_and_duplicate_fail(self):
        config = {"blocks": ["selection"], "sample_budgets": [2, 4], "sample_chunk_size": 2}
        records = [{"context": "a", "block": "selection", "start": i, "stop": i+2, "path": str(i)} for i in (0, 2)]
        s3.check_records(records, [{"id": "a"}], config)
        for broken in (records[:1], records+[records[0]], [records[0], records[0]]):
            with self.assertRaises(ValueError):
                s3.check_records(broken, [{"id": "a"}], config)

    def test_argmin_or_improvement_sign_disagreement_is_unresolved(self):
        context = self.contexts()[0]
        records = []
        for block in ("selection", "validation_a", "validation_b", "validation_c"):
            records.append({"context": context["id"], "block": block,
                            "budgets": [{"argmin_index": 1}]*3,
                            "validation": [{"comparators": {k: {"sign": -1} for k in context["references"]}}]})
        frozen = {context["id"]: {"index": 1}}
        self.assertEqual(s3.summarize([context], records, frozen, 480, 1000)["unresolved_contexts"], 0)
        records[-1]["budgets"] = [{"argmin_index": 2}]*3
        self.assertEqual(s3.summarize([context], records, frozen, 480, 1000)["unresolved_contexts"], 1)
        records[-1]["budgets"] = [{"argmin_index": 1}]*3
        records[-1]["validation"][0]["comparators"]["booked_rule"]["sign"] = 1
        self.assertEqual(s3.summarize([context], records, frozen, 480, 1000)["unresolved_contexts"], 1)

    def tiny_recording(self, root, stack):
        inputs = root/"inputs"
        inputs.mkdir()
        s3.write_json(inputs/"mechanics.json", self.fixture)
        s3.write_json(inputs/"probes.json", [self.probe])
        config = s3.read_json(s3.CONFIG)
        config.update(s2_root=str(inputs), source_probe_count=1, expected_unique_contexts=1,
                      sample_budgets=[2, 4, 8], sample_chunk_size=2, max_forecast_paths=480,
                      namespace="invented-unit-smoke-not-S3-recorded-run")
        path = root/"config.json"
        s3.write_json(path, config)
        stack.enter_context(patch.object(s3, "CONFIG", path))
        stack.enter_context(patch.object(s3, "locked_files", return_value=[]))
        stack.enter_context(patch.object(s3, "input_locks", return_value=[]))
        stack.enter_context(redirect_stdout(io.StringIO()))
        output = root/"output"
        s3.run(output)
        return output

    def test_tiny_smoke_raw_audit_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            output = self.tiny_recording(Path(name), stack)
            audit = s3.audit_output(output)
            self.assertEqual(audit["forecast_paths"], 480)
            self.assertEqual(audit["scalar_replayed_paths"], 60)
            self.assertEqual(audit["paired_contrasts_recomputed"], 1260)
            self.assertEqual(s3.read_json(output/"status.json")["exit_code"], 0)
            with self.assertRaises(FileExistsError):
                s3.run(output)
            events = [json.loads(line) for line in (output/"events.jsonl").read_text().splitlines()]
            freeze = next(i for i, row in enumerate(events) if row["kind"] == "freeze")
            self.assertTrue(all(row.get("block") == "selection" for row in events[:freeze]))

    def test_audit_rejects_modified_raw_costs(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            output = self.tiny_recording(Path(name), stack)
            path = next((output/"raw").rglob("*.npy"))
            data = np.load(path, allow_pickle=False)
            data[0, 0] += 1
            np.save(path, data, allow_pickle=False)
            with self.assertRaises(ValueError):
                s3.audit_output(output)

    def test_audit_rejects_unindexed_raw_file_and_incomplete_status(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            output = self.tiny_recording(Path(name), stack)
            s3.write_json(output/"status.json", {"status": "running", "exit_code": 0})
            with self.assertRaises(ValueError):
                s3.audit_output(output)
            np.save(output/"raw/extra.npy", np.zeros((15, 2)), allow_pickle=False)
            with self.assertRaises(ValueError):
                s3.audit_output(output)

    def test_run_failure_retains_partial_chunks_and_never_retries(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            real_forecast = s3.forecast
            calls = []
            def fail_second(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("invented unit failure")
                return real_forecast(*args, **kwargs)
            stack.enter_context(patch.object(s3, "forecast", side_effect=fail_second))
            with self.assertRaises(RuntimeError):
                self.tiny_recording(Path(name), stack)
            output = Path(name)/"output"
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(list((output/"raw").rglob("*.npy"))), 1)
            self.assertEqual(s3.read_json(output/"status.json")["status"], "failed")
            self.assertTrue((output/"error.txt").exists())


if __name__ == "__main__":
    unittest.main()
