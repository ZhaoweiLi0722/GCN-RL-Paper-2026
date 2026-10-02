"""Pure invented JSON budgets, no filesystem research data or scientific calls."""

import copy
import unittest

from src.rl.candidate_pilot_resources import digest
from src.rl.cohort_objective_plan import cohort_budget_plan, cohort_stream_manifest
from src.rl.dynamic_candidate_resources import validate_budget_plan
from tests.test_time_baseline_plan import artificial_proposal


def fixture():
    base = artificial_proposal()
    rename = lambda x: {"current_ppo": "window_ppo", "time_baseline_ppo": "cohort_ppo"}.get(x, x)
    rows = copy.deepcopy(base["phase_budgets"])
    for row in rows:
        row["id"] = rename(row["id"])
        row["trajectory"] += row["episodes"] * 11
        if row["id"] == "same_start_preflight":
            row["clone"] += 72 + 156
    totals = copy.deepcopy(base["totals"])
    totals.update(trajectory=32886, clone=300, environment=33186, environment_builds=528)
    owners = copy.deepcopy(base["per_owner"])
    for owner in owners.values():
        owner.update(prefix=owner["trajectory"], tail=owner["episodes"] * 11)
        owner["trajectory"] += owner["tail"]
    proposal = dict(schema="cohort-objective-proposal-v1", scientific_execution_authorized=False,
                    base_proposal_content_sha256=digest(base), blocks=[60, 61, 62],
                    training_arms=[rename(x) for x in base["training_arms"]],
                    evaluation_controllers=[rename(x) for x in base["evaluation_controllers"]],
                    enrollment_steps=52, patient_resolution_steps=8, accounting_steps=11,
                    tail_clone_steps_per_preflight_controller=4,
                    phase_budgets=rows, totals=totals, rng_namespace="invented-cohort-scope",
                    per_owner=owners, prefix_parity_environment_builds=3, prefix_parity_steps=156,
                    clone_instances=36, learned_artifacts_sealed_before_tests=12,
                    initialization_optimizer_calls=0, qualification_optimizer_calls=0,
                    episodes_per_training_arm_block=32, episodes_per_rollout=4,
                    rollouts_per_arm_block=8, epochs_per_rollout=4, minibatch_sizes=[64, 64, 64, 16],
                    test_worlds_per_block=12, preflight_worlds_per_block=1,
                    initial_historical_loads=6, prefix_clone_steps_per_preflight_controller=4)
    return base, proposal


class CohortPlanTests(unittest.TestCase):
    def test_complete_arithmetic_and_unchanged_optimizer_caps(self):
        base, p = fixture()
        plan = cohort_budget_plan(base, p)
        validate_budget_plan(plan)
        self.assertEqual(len(plan["sections"]), 33)
        self.assertEqual(plan["limits"]["trajectory"], 32886)
        self.assertEqual(plan["limits"]["actor"] + plan["limits"]["critic"], 1920)
        self.assertEqual(plan["sections"]["cohort_ppo/block60"]["trajectory"], 2016)
        self.assertEqual(plan["sections"]["same_start_preflight"]["clone"], 300)
        for phase, limit in plan["phases"].items():
            rows = [r for r in plan["sections"].values() if r["phase"] == phase]
            self.assertEqual({k: sum(r[k] for r in rows) for k in limit}, limit)

    def test_no_mutation_fresh_namespace_and_paired_owners(self):
        base, p = fixture()
        saved = copy.deepcopy((base, p))
        streams = cohort_stream_manifest(base, p)
        self.assertEqual((base, p), saved)
        self.assertEqual(streams["unique_worlds"], 138)
        self.assertFalse(streams["seed_freshness_verified"])
        self.assertFalse(streams["scientific_execution_authorized"])
        bindings = streams["neural_bindings"]["60"]["training"]
        self.assertEqual(bindings["window_ppo"], bindings["cohort_ppo"])
        self.assertEqual(streams["followup_new_rng_instances"], 0)

    def test_each_cap_and_input_binding_tamper_rejected(self):
        base, p = fixture()
        for key in p["totals"]:
            changed = copy.deepcopy(p)
            changed["totals"][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                cohort_budget_plan(base, changed)
        p["base_proposal_content_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            cohort_budget_plan(base, p)

    def test_followup_shortage_and_reused_namespace_rejected(self):
        base, p = fixture()
        p["accounting_steps"] = 7
        with self.assertRaises(ValueError):
            cohort_budget_plan(base, p)
        base, p = fixture()
        p["rng_namespace"] = base["rng"]["namespace"]
        with self.assertRaises(ValueError):
            cohort_stream_manifest(base, p)


if __name__ == "__main__":
    unittest.main()
