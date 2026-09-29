"""S4 public feedback, accounting, forecast parity and recording smoke tests."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluation import check_completion_action_ranking as s3
from evaluation import check_completion_continuation_comparison as s4
from evaluation.check_completion_feedback_mechanics import CONFIG as FIXTURE, make_config
from src.baselines.completion_feedback_tree import reachable_events, scalar_continuation, second_action_indices
from src.baselines.completion_rollout_control import action_grid, named_seed
from src.env.completion_continuations import forecast_continuations
from src.env.completion_feedback_batch import forecast
from src.env.completion_feedback_queue import advance, initial_observation


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = s3.read_json(FIXTURE)
        self.grid = action_grid(4)
        self.cfg = make_config(self.fixture, 1)
        self.initial = initial_observation(self.cfg)
        self.state, _, _ = advance(self.initial, (.5, .5), (1, 1), (.9, .9), self.cfg)
        self.noise = np.random.Generator(np.random.PCG64(921)).random((3, 287, 2))
        self.table = np.tile(np.arange(15), (4, 1))
        self.table[1] = 2
        self.table[2] = 11
        self.table[3] = 0

    def test_reachable_probabilities_and_previous_effort(self):
        self.assertEqual(len(reachable_events(self.initial, (1, 1), self.cfg)), 1)
        branches = reachable_events(self.state, (1, 1), self.cfg)
        self.assertEqual([b["code"] for b in branches], [0, 1, 2, 3])
        self.assertAlmostEqual(sum(b["probability"] for b in branches), 1)
        for branch in branches:
            observations = []
            for action in self.grid:
                after, receipt, _ = advance(self.state, action, (1, 1), branch["representative_uniforms"], self.cfg)
                self.assertEqual(list(receipt.completed), branch["events"])
                observations.append(after.stages)
            self.assertTrue(all(s == observations[0] for s in observations))

    def test_booked_forecast_matches_locked_old_implementation(self):
        for slots in (1, 4):
            cfg = make_config(self.fixture, slots)
            new = forecast_continuations(self.state, self.grid, (1, 1), self.noise, cfg, 32, "booked")
            old = forecast(self.state, np.asarray(self.grid)[:, None, :], (1, 1), self.noise, cfg, 32)
            np.testing.assert_array_equal(new["components"].sum(axis=-1), old["sample_costs"])
            self.assertEqual(new["transition_queries"], old["transition_queries"])

    def test_components_and_queries_match_scalar_all_methods(self):
        for method in ("booked", "reservation", "tree32", "tree128"):
            result = forecast_continuations(self.state, self.grid, (1, 1), self.noise, self.cfg, 32, method, self.table)
            count = 0
            for a in range(15):
                for sample, tape in enumerate(self.noise):
                    expected, trace = scalar_continuation(self.state, a, self.grid, (1, 1), tape, self.cfg, 32, method, self.table)
                    np.testing.assert_array_equal(result["components"][a, sample], expected)
                    count += len(trace)
            self.assertEqual(result["transition_queries"], count)

    def test_multiple_rates_capacities_and_full_closure(self):
        for slots, rates in ((1, (.5, 1.5)), (4, (1.5, .5))):
            cfg = make_config(self.fixture, slots)
            for method in ("reservation", "tree128"):
                result = forecast_continuations(self.state, self.grid, rates, self.noise, cfg, 2, method, self.table)
                for a in (0, 7, 11, 14):
                    expected, trace = scalar_continuation(self.state, a, self.grid, rates, self.noise[0], cfg, 2, method, self.table)
                    np.testing.assert_array_equal(result["components"][a, 0], expected)
                    self.assertGreater(len(trace), 24)
                    self.assertEqual(trace[1]["action"], [0, 0])

    def test_tree_second_request_uses_only_first_observed_events(self):
        first = np.full((287, 2), .2)
        second = np.full((287, 2), .3)
        first[0] = second[0] = (.99, .99)
        _, a = scalar_continuation(self.state, 7, self.grid, (1, 1), first, self.cfg, 32, "tree128", self.table)
        _, b = scalar_continuation(self.state, 7, self.grid, (1, 1), second, self.cfg, 32, "tree128", self.table)
        self.assertEqual(a[:2], b[:2])
        first[0] = (0, 0)
        _, changed = scalar_continuation(self.state, 7, self.grid, (1, 1), first, self.cfg, 32, "tree128", self.table)
        self.assertNotEqual(changed[1]["action"], a[1]["action"])
        self.assertEqual(changed[0]["action"], a[0]["action"])

    def test_invalid_unreachable_table_and_closure_fail(self):
        invalid = np.full((4, 15), -1, dtype=int)
        with self.assertRaises(ValueError):
            forecast_continuations(self.state, self.grid, (1, 1), self.noise, self.cfg, 32, "tree128", invalid)
        with self.assertRaises(ValueError):
            forecast_continuations(self.state, self.grid, (1, 1), self.noise, self.cfg, 32, "invented")
        with self.assertRaises(RuntimeError):
            forecast_continuations(self.state, self.grid, (1, 1), self.noise[:, :1], self.cfg, 32, "booked")
        with self.assertRaises(RuntimeError):
            scalar_continuation(self.state, 0, self.grid, (1, 1), self.noise[0, :1], self.cfg, 32, "booked")

    def test_inner_nested_choice_stable_ties_and_independent_arithmetic(self):
        costs = np.zeros((15, 128))
        costs[0, 32:] = 1
        choices = second_action_indices(costs, [32, 128])
        self.assertEqual(choices, {"32": 0, "128": 1})
        self.assertEqual(s4.inner_statistics(costs, [32, 128]), s4.inner_statistics(costs, [32, 128], True))
        with self.assertRaises(ValueError):
            second_action_indices(costs[:, :-1], [32, 128])

    def test_inner_outer_namespaces_and_prefix_are_separate(self):
        inner = named_seed("unit-S4", "context", "tree_fit", 0)
        outer = named_seed("unit-S4", "context", "outer", "selection")
        self.assertNotEqual(inner, outer)
        self.assertNotEqual(outer, named_seed("unit-S4", "context", "outer", "validation_a"))
        full = np.random.Generator(np.random.PCG64(inner)).random((128, 287, 2))
        small = np.random.Generator(np.random.PCG64(inner)).random((32, 287, 2))
        np.testing.assert_array_equal(full[:32], small)

    def test_duplicate_records_counts_and_arrays_fail_closed(self):
        row = {"kind": "inner", "context": "c", "label": "0", "index": 0}
        with self.assertRaises(ValueError):
            s4.unique_records([row, row], ("kind", "context", "label", "index"))
        config = s3.read_json(s4.CONFIG)
        with self.assertRaises(ValueError):
            s4.verify_counts(1, config["outer_paths"]-1, 1, config)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/"raw.npz"
            digest = s4.save_array(path, "costs", np.zeros((2, 2)))
            with self.assertRaises(ValueError):
                s4.load_array(path, "costs", (1, 2), digest)
            with self.assertRaises(ValueError):
                s4.load_array(path, "costs", (2, 2), "wrong hash")

    def test_validation_comparison_preserves_frozen_action_and_components(self):
        config = s3.read_json(s4.CONFIG)
        config["outer_budgets"] = [2, 4, 8]
        public = {"observation": asdict(self.initial), "response": [1., 1.],
                  "downstream_slots": 1, "mechanics": self.fixture}
        context = {"id": s3.context_digest(public), "public": public}
        components = np.zeros((4, 15, 8, 3))
        components[3, 7] = [1., 2., 3.]
        selected = {"booked": 2, "reservation": 3, "tree32": 4, "tree128": 7}
        actual = s4.comparisons(components, config, selected, context, self.grid)
        self.assertEqual(actual, s4.comparisons(components, config, selected, context, self.grid, True))
        row = actual[0]
        self.assertEqual(row["left_action_index"], 7)
        self.assertEqual(row["mean_left_minus_right"], 6)
        self.assertEqual(sum(r["mean_left_minus_right"] for r in row["component_differences"].values()), 6)
        self.assertEqual(row["paired_mc_se"], 0)

    def test_signal_gate_rejects_one_negative_result_reversal(self):
        config = s3.read_json(s4.CONFIG)
        context = {"id": "unit", "aliases": []}
        pairs = [f"tree{n}_minus_{m}_selected" for n in (32, 128) for m in ("booked", "reservation")]
        blocks = [{"context": "unit", "block": b, "ranks": [], "comparisons": [
            {"name": name, "samples": 2048, "mean_left_minus_right": -1} for name in pairs]}
            for b in ("validation_a", "validation_b", "validation_c")]
        self.assertEqual(s4.summary([context], [], blocks, config, 1, 1, 1)["contexts_with_signal"], 1)
        blocks[-1]["comparisons"][0]["mean_left_minus_right"] = 1
        self.assertEqual(s4.summary([context], [], blocks, config, 1, 1, 1)["contexts_with_signal"], 0)

    def tiny_recording(self, root, stack):
        config = s3.read_json(s4.CONFIG)
        config.update(expected_unique_contexts=1, source_probe_aliases=1,
                      outer_budgets=[2, 4, 8], chunk_size=2, outer_paths=1920,
                      namespace="S4-invented-unit-state-not-recorded-matrix")
        public = {"observation": asdict(self.initial), "response": [1., 1.],
                  "downstream_slots": 1, "mechanics": self.fixture}
        context = {"id": s3.context_digest(public), "public": public, "aliases": [{"family": "invented", "slots": 1}]}
        path = root/"config.json"
        write = s4.write_json
        write(path, config)
        stack.enter_context(patch.object(s4, "CONFIG", path))
        stack.enter_context(patch.object(s4, "locked_files", return_value=[]))
        stack.enter_context(patch.object(s4, "load_inputs", return_value=([context], [])))
        stack.enter_context(redirect_stdout(io.StringIO()))
        output = root/"run"
        s4.run(output)
        return output

    def test_recording_smoke_independent_audit_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            output = self.tiny_recording(Path(name), stack)
            audit = s4.audit_output(output)
            self.assertEqual(audit["inner_paths"], 28800)
            self.assertEqual(audit["outer_paths"], 1920)
            self.assertEqual(audit["scalar_replayed_paths"], 690)
            self.assertEqual(s3.read_json(output/"status.json")["exit_code"], 0)
            events = [json.loads(line) for line in (output/"events.jsonl").read_text().splitlines()]
            freeze = next(i for i, r in enumerate(events) if r["kind"] == "freeze_tree")
            self.assertTrue(all(e["kind"] == "inner" for e in events[:freeze]))
            with self.assertRaises(FileExistsError):
                s4.run(output)

    def test_raw_corruption_and_missing_rows_rejected(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            output = self.tiny_recording(Path(name), stack)
            path = output/"raw.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1])+"\n")
            with self.assertRaises((ValueError, KeyError)):
                s4.audit_output(output)

    def test_terminal_failure_keeps_partial_evidence_without_retry(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            real = s4.forecast
            calls = []
            def failing(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("invented failure")
                return real(*args, **kwargs)
            stack.enter_context(patch.object(s4, "forecast", side_effect=failing))
            with self.assertRaises(RuntimeError):
                self.tiny_recording(Path(name), stack)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(list((Path(name)/"run/raw").rglob("*.npz"))), 1)
            self.assertEqual(s3.read_json(Path(name)/"run/status.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
