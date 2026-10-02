"""Pure metadata and fake-ledger tests; read JSON, never scientific checkpoints."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.paired_cohort_resources import budget_plan, stream_manifest, runtime_streams, inherited_metadata

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "experiments/configs/paired_cohort_improvement_20261002.json").read_text())


class PairedCohortResourcesTests(unittest.TestCase):
    def test_exact_complete_caps_and_per_owner_partition(self):
        plan = budget_plan(config())
        self.assertEqual(plan["limits"], dict(trajectory=14553, clone=18765, actor=768, critic=0, seconds=14400))
        self.assertEqual(plan["sections"]["paired_branches/block60"]["clone"], 6192)
        self.assertEqual(plan["sections"]["evaluation/block60/paired_cost"]["seconds"], 240)
        self.assertEqual(sum(p["seconds"] for p in plan["phases"].values()), 13740)
        changed = config()
        changed["maximum"]["environment_calls"] += 1
        with self.assertRaises(ValueError):
            budget_plan(changed)

    def test_actual_persisted_metadata_resolves_without_model_load(self):
        original = json.loads((ROOT / "specs/2026-10-02-terminal-obligation/frozen.json").read_text())
        recovery = json.loads((ROOT / "specs/2026-10-02-cohort-evaluation-recovery2/frozen.json").read_text())
        prepared = inherited_metadata(config(), original, recovery)
        self.assertEqual(len(prepared["model_inputs"]), 6)
        self.assertEqual(prepared["backend_config"]["totals"]["fresh_episode_builds"], 231)
        self.assertEqual(prepared["model_inputs"]["block60/initializer"], original["initializers"]["60"])
        streams = stream_manifest(config(), prepared["layout_seeds"])
        self.assertFalse(streams["seed_freshness_verified"])
        self.assertFalse(prepared["scientific_execution_authorized"])
        self.assertEqual(prepared["new_checkpoint_loads"], 0)
        self.assertEqual(len(streams["allocations"]), 130)
        runtime = runtime_streams(streams)
        for block in ("60", "61", "62"):
            self.assertIs(type(runtime["environment"][block]["layout"][0]), int)
            self.assertEqual(runtime["environment"][block]["training"], runtime["environment"][block]["context"])
            self.assertGreater(runtime["environment"][block]["test"][0], 2**53)

    def test_decimal_seed_transport_is_exact_and_rejects_float_or_flag(self):
        streams = stream_manifest(config(), {str(b): str(b) for b in (60, 61, 62)})
        before = copy.deepcopy(streams)
        runtime = runtime_streams(streams)
        self.assertEqual(streams, before)
        for k, values in streams["conditional_future"].items():
            self.assertEqual([str(v) for v in runtime["conditional_future"][k]], values)
        for bad in (True, 60., "060"):
            changed = copy.deepcopy(streams)
            changed["environment"]["60"]["test"][0] = bad
            with self.assertRaises(ValueError):
                runtime_streams(changed)

    def test_existing_ledger_rejects_critic_and_does_not_refund_actor(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ledger.jsonl"
            budget = DynamicCandidateBudget(path, budget_plan(config()), enabled=True, clock=lambda: 0.)
            budget.begin("paired_actor/block60")
            budget.debit_optimizer("actor")
            with self.assertRaises(RuntimeError):
                budget.debit_optimizer("critic")
            budget.close()
            saved = read_dynamic_ledger(path)
            self.assertEqual(saved["counts"]["optimizer"], 1)
            self.assertEqual(saved["owner_counts"]["global:actor"], 1)
            self.assertEqual(saved["owner_counts"].get("global:critic", 0), 0)


if __name__ == "__main__":
    unittest.main()
