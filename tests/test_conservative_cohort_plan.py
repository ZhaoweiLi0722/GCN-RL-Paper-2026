"""Metadata-only scope tests: no models, optimizers or patient episodes."""

import copy
import json
from pathlib import Path
import unittest

from src.rl.conservative_cohort_plan import budget_plan, cohort_ids, schedule, streams, validate_config

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "experiments/configs/conservative_cohort_improvement_20261002.json").read_text())


class PlanTests(unittest.TestCase):
    def test_all_calls_and_rounds_are_prospective_and_nontransferable(self):
        cfg = config()
        counts = validate_config(cfg)
        plan = budget_plan(cfg)
        self.assertEqual(plan["limits"], dict(trajectory=12285, clone=37341, actor=768, critic=0, seconds=28800))
        self.assertEqual(counts["environment_calls"], plan["limits"]["trajectory"] + plan["limits"]["clone"])
        self.assertEqual(37152, 3 * 2 * 2 * 4 * 6 * sum(63 - t for t in (4, 20, 36)))
        self.assertEqual(plan["sections"]["branches/round1/block60"]["clone"], 6192)
        jobs = schedule(cfg)
        seal = next(i for i, j in enumerate(jobs) if j["id"] == "seal")
        self.assertTrue(all(i > seal for i, j in enumerate(jobs) if j["phase"] == "evaluation"))
        for r in range(2):
            for b in cfg["blocks"]:
                names = [j["phase"] for j in jobs if j.get("round") == r and j.get("block") == b]
                self.assertEqual(names, ["contexts", "branches", "paired_actor", "bc_actor"])

    def test_exact_metadata_seeds_convert_without_float(self):
        old = json.loads((ROOT / "specs/2026-10-02-terminal-obligation/frozen.json").read_text())
        layouts = {b: s["layout"][0] for b, s in old["streams"]["environment"].items()}
        manifest = streams(config(), layouts)
        self.assertEqual(len(manifest["conditional_future"]), 36)
        self.assertEqual(len(set(manifest["allocations"].values())), 202)
        self.assertTrue(all(type(v) is str and v.isdecimal() for v in manifest["allocations"].values()))
        self.assertEqual(cohort_ids(0) + cohort_ids(1), (0, 1, 2, 3))
        training = {v for s in manifest["environment"].values() for v in s["context"]}
        tests = {v for s in manifest["environment"].values() for v in s["test"]}
        self.assertFalse(training & tests)

    def test_reject_changed_scope(self):
        for name, value in (("rounds", 3), ("future_replications", 8), ("critic_updates", 1),
                            ("scientific_execution_authorized", True)):
            cfg = config()
            cfg[name] = value
            with self.assertRaises(ValueError):
                budget_plan(cfg)


if __name__ == "__main__":
    unittest.main()
