"""Invented resource events and seed arithmetic; no simulation or model fit."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.scripts.audit_candidate_pilot_readiness import historical_files
from src.rl.candidate_pilot_resources import (
    PilotBudget, audit_stream_collisions, budget_sections, numeric_leaves, read_ledger, stream_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "experiments/configs/candidate_return_pilot_20260930.json").read_text())


class CandidatePilotResourcesTests(unittest.TestCase):
    def test_prespecified_scope_arithmetic_and_seed_counts(self):
        cfg = config()
        sections = budget_sections(cfg)
        self.assertEqual(sum(s["environment"] for s in sections.values()), 52728)
        self.assertEqual(sum(s["optimizer"] for s in sections.values()), 4608)
        self.assertEqual(len(sections), 63)
        manifest = stream_manifest(cfg)
        env = numeric_leaves(manifest["environment"])
        self.assertEqual(len(env), 174)
        self.assertEqual(len(set(env)), 174)
        self.assertEqual(len(manifest["neural"]), 37)
        self.assertEqual(len(set(numeric_leaves(manifest))), 214)
        self.assertEqual(manifest, stream_manifest(cfg))
        self.assertTrue(all(v > 2**64 for v in env))

    def test_collision_detects_numeric_seed_not_only_namespace(self):
        manifest = stream_manifest(config())
        seed = manifest["environment"]["test"]["60"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps({"different_namespace": "historical", "nested": [{"seed": seed}]}))
            found = audit_stream_collisions(manifest, [path])
            self.assertFalse(found["passed"])
            self.assertEqual(found["collisions"][0]["values"], [seed])
            path.write_text(json.dumps({"seed_as_text": str(seed)}))
            self.assertFalse(audit_stream_collisions(manifest, [path])["passed"])
            path.write_text('{"seed": 123}')
            self.assertTrue(audit_stream_collisions(manifest, [path])["passed"])
            path.write_text('{invalid')
            with self.assertRaises(ValueError):
                audit_stream_collisions(manifest, [path])
        with self.assertRaises(ValueError):
            audit_stream_collisions(manifest, [])

    def test_inventory_excludes_only_current_packet_and_rejects_unknown_seed_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "results/prior/seeds.json"
            old.parent.mkdir(parents=True)
            old.write_text('{"seed":1}')
            current = root / "reports/2026-09-30-candidate-pilot-integration/readiness.json"
            current.parent.mkdir(parents=True)
            current.write_text('{}')
            with patch("experiments.scripts.audit_candidate_pilot_readiness.subprocess.check_output",
                       return_value=b"experiments/configs/candidate_return_pilot_20260930.json\0"):
                self.assertEqual(historical_files(root), [old])
                old.with_suffix('.csv').write_text('seed\n1\n')
                with self.assertRaisesRegex(ValueError, "unhandled"):
                    historical_files(root)

    def test_rng_overlap_and_wrong_stream_range_rejected(self):
        for mutate in (lambda c: c["rng"]["ordinal_ranges"].update(preflight=[12, 23]),
                       lambda c: c["rng"]["ordinal_ranges"].update(test=[138, 174]),
                       lambda c: c["policy_init_seeds"].__setitem__(0, c["policy_init_seeds"][1])):
            cfg = config()
            mutate(cfg)
            with self.assertRaises(ValueError):
                stream_manifest(cfg)

    def test_durable_debit_matches_independent_ledger_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.jsonl"
            budget = PilotBudget(path, config())
            self.addCleanup(budget.close)
            budget.begin("preflight_including_clones")
            for _ in range(3):
                budget.debit("environment")
            self.assertEqual(read_ledger(path)["counts"], {"environment": 3})
            self.assertEqual(read_ledger(path)["last_sha256"], budget.snapshot()["ledger_sha256"])
            budget.finish()
            with self.assertRaises(FileExistsError):
                PilotBudget(path, config())
            with self.assertRaises(ValueError):
                budget.begin("preflight_including_clones")

    def test_scope_cannot_borrow_unspent_budget_or_refund_failure(self):
        cfg = config()
        cfg["caps"]["per_continuation_model_environment_steps"] = 2
        with tempfile.TemporaryDirectory() as directory:
            budget = PilotBudget(Path(directory) / "budget.jsonl", cfg)
            self.addCleanup(budget.close)
            budget.begin("block60/graph/ppo")
            budget.debit("environment")
            budget.debit("environment")
            with self.assertRaisesRegex(RuntimeError, "cap"):
                budget.debit("environment")
            self.assertEqual(budget.snapshot()["counts"]["environment"], 2)
            with self.assertRaises(RuntimeError):
                budget.begin("block61/graph/ppo")

    def test_no_optimizer_in_simulator_only_or_eval_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = PilotBudget(Path(directory) / "budget.jsonl", config())
            self.addCleanup(budget.close)
            budget.begin("block60/graph/frozen/evaluation")
            with self.assertRaises(RuntimeError):
                budget.debit("optimizer")
            self.assertEqual(budget.snapshot()["counts"]["optimizer"], 0)

    def test_phase_and_global_limits_are_independent_of_scope_allowance(self):
        for level in ("phase", "global"):
            cfg = config()
            if level == "phase":
                cfg["caps"]["environment_steps"]["ppo"] = 1
            else:
                cfg["caps"]["maximum_environment_steps"] = 1
            with tempfile.TemporaryDirectory() as directory:
                budget = PilotBudget(Path(directory) / "budget.jsonl", cfg)
                self.addCleanup(budget.close)
                budget.begin("block60/graph/ppo")
                budget.debit("environment")
                budget.finish()
                budget.begin("block61/graph/ppo")
                with self.assertRaises(RuntimeError):
                    budget.debit("environment")
                self.assertEqual(read_ledger(budget.path)["counts"], {"environment": 1})

    def test_time_cap_global_section_regression_and_after_last_operation(self):
        for end, section in ((601., "preflight_including_clones"),
                             (21601., None), (-1., None), (float("nan"), None)):
            with tempfile.TemporaryDirectory() as directory:
                clock = [0.]
                budget = PilotBudget(Path(directory) / "budget.jsonl", config(), clock=lambda: clock[0])
                self.addCleanup(budget.close)
                if section:
                    budget.begin(section)
                    budget.debit("environment")
                clock[0] = end
                with self.assertRaises(TimeoutError):
                    budget.finish() if section else budget.check()

    def test_failed_flush_poison_prevents_operation_and_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = PilotBudget(Path(directory) / "budget.jsonl", config())
            self.addCleanup(budget.close)
            budget.begin("preflight_including_clones")
            with patch("src.rl.candidate_pilot_resources.os.fsync", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    budget.debit("environment")
            with self.assertRaises(RuntimeError):
                budget.debit("environment")
            # Conservatively retained debit may have reached disk; never reused.
            self.assertEqual(read_ledger(budget.path)["counts"], {"environment": 1})

    def test_corrupt_ledger_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.jsonl"
            budget = PilotBudget(path, config())
            budget.begin("preflight_including_clones")
            budget.debit("environment")
            budget.close()
            rows = path.read_text().splitlines()
            row = json.loads(rows[-1])
            row["total"] = 0
            rows[-1] = json.dumps(row)
            path.write_text("\n".join(rows) + "\n")
            with self.assertRaises(ValueError):
                read_ledger(path)


if __name__ == "__main__":
    unittest.main()
