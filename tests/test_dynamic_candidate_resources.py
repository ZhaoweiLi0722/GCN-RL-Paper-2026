"""Budget/stream fixtures only: no model, optimizer, or patient environment."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import (
    DynamicCandidateBudget, assert_dynamic_budget_ancestor, dynamic_budget_plan, dynamic_stream_manifest,
    read_dynamic_ledger, validate_budget_plan,
)


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "specs/2026-10-01-adaptive-paper-delivery/pilot-budget-draft.json"


def miniature():
    scopes = {
        "collect": {"phase": "collect", "trajectory": 2, "clone": 1, "actor": 0, "critic": 0, "seconds": 8},
        "fit/a": {"phase": "fit", "trajectory": 0, "clone": 0, "actor": 1, "critic": 1, "seconds": 4},
        "fit/b": {"phase": "fit", "trajectory": 0, "clone": 0, "actor": 1, "critic": 1, "seconds": 4},
    }
    return {"format": "dynamic-candidate-budget-plan-v1", "draft_sha256": "fixture",
            "limits": {"trajectory": 2, "clone": 1, "actor": 2, "critic": 2, "seconds": 20},
            "phases": {"collect": {k: v for k, v in scopes["collect"].items() if k != "phase"},
                       "fit": {"trajectory": 0, "clone": 0, "actor": 2, "critic": 2, "seconds": 8}},
            "sections": scopes}


class DynamicResourcesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "ledger.jsonl"
        self.now = 10.0

    def budget(self, plan=None):
        value = DynamicCandidateBudget(self.path, miniature() if plan is None else plan,
                                       enabled=True, clock=lambda: self.now)
        self.addCleanup(value.close)
        return value

    def test_proposal_arithmetic_and_owner_partition(self):
        draft = json.loads(DRAFT.read_text())
        plan = dynamic_budget_plan(draft)
        self.assertEqual(plan["limits"], {"trajectory": 22152, "clone": 72,
                                         "actor": 1536, "critic": 384, "seconds": 21600})
        self.assertEqual(len(plan["sections"]), 33)
        self.assertEqual(sum(v["seconds"] for v in plan["phases"].values()), 21300)
        self.assertEqual(plan["sections"]["ppo_continuation/block60"]["actor"], 128)
        self.assertEqual(plan["sections"]["ppo_continuation/block60"]["critic"], 128)
        self.assertFalse(draft["scientific_execution_authorized"])
        self.assertFalse(draft["ready_to_launch"])

    def test_proposal_streams_are_distinct_but_not_claimed_fresh(self):
        draft = json.loads(DRAFT.read_text())
        result = dynamic_stream_manifest(draft)
        self.assertEqual(result["unique_worlds"], 168)
        self.assertEqual(result["episode_uses"], 426)
        self.assertEqual(len(result["neural"]), 19)
        self.assertFalse(result["seed_freshness_verified"])
        self.assertFalse(result["consumed"])
        self.assertFalse(result["scientific_execution_authorized"])
        self.assertEqual(result, dynamic_stream_manifest(draft))
        for ranges in result["environment"].values():
            self.assertEqual(len(ranges["training"]), 32)
            self.assertEqual(len(ranges["test"]), 12)

    def test_duplicate_missing_and_wrong_count_streams_rejected(self):
        original = json.loads(DRAFT.read_text())
        cases = []
        draft = copy.deepcopy(original)
        draft["rng_proposal"]["block_ordinal_ranges"]["61"]["test"] = [132, 143]
        cases.append(draft)
        draft = copy.deepcopy(original)
        draft["rng_proposal"]["block_ordinal_ranges"]["60"]["training"] = [36, 66]
        cases.append(draft)
        draft = copy.deepcopy(original)
        draft["rng_proposal"]["global_role_paths"] *= 2
        cases.append(draft)
        draft = copy.deepcopy(original)
        del draft["rng_proposal"]["block_ordinal_ranges"]["62"]
        cases.append(draft)
        for draft in cases:
            with self.subTest(draft=draft["rng_proposal"]), self.assertRaises(ValueError):
                dynamic_stream_manifest(draft)

    def test_plan_rejects_nonpartitioned_or_optional_allocations(self):
        for key in ("trajectory", "clone", "actor", "critic", "seconds"):
            plan = miniature()
            plan["sections"]["fit/a"][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_budget_plan(plan)
        plan = miniature()
        plan["limits"]["actor"] = True
        with self.assertRaises(ValueError):
            validate_budget_plan(plan)
        draft = json.loads(DRAFT.read_text())
        draft["optional_counterfactual"]["include_optional_counterfactual"] = True
        with self.assertRaises(ValueError):
            dynamic_budget_plan(draft)

    def test_opt_in_exclusive_file_and_explicit_owners(self):
        with self.assertRaises(ValueError):
            DynamicCandidateBudget(self.path, miniature())
        budget = self.budget()
        with self.assertRaises(FileExistsError):
            DynamicCandidateBudget(self.path, miniature(), enabled=True)
        for call in (lambda: budget.debit("optimizer"), lambda: budget.debit_optimizer("trajectory"),
                     lambda: budget.debit_environment("actor"), budget.check_minibatch):
            with self.assertRaises(ValueError):
                call()

    def test_raw_ledger_owner_and_clone_counts(self):
        budget = self.budget()
        budget.begin("collect")
        budget.debit_environment("trajectory")
        budget.debit_environment("clone")
        budget.debit_environment("trajectory")
        self.now += 3
        budget.finish()
        budget.begin("fit/a")
        budget.check_minibatch()
        budget.debit_optimizer("actor")
        budget.debit_optimizer("critic")
        self.now += 2
        budget.finish()
        budget.begin("fit/b")
        budget.check_minibatch()
        budget.debit_optimizer("actor")
        budget.debit_optimizer("critic")
        self.now += 2
        budget.finish()
        actual, snapshot = read_dynamic_ledger(self.path), budget.snapshot()
        self.assertEqual(actual["counts"], {"environment": 3, "optimizer": 4})
        self.assertEqual(actual["owner_counts"], snapshot["owner_counts"])
        self.assertEqual(actual["phase_seconds"], {"collect": 3, "fit": 4})
        self.assertEqual(actual["last_sha256"], snapshot["ledger_sha256"])
        self.assertEqual(actual["plan_sha256"], snapshot["plan_sha256"])

    def test_unused_clone_or_critic_slots_are_not_transferable(self):
        budget = self.budget()
        budget.begin("fit/a")
        budget.debit_optimizer("actor")
        with self.assertRaises(RuntimeError):
            budget.debit_optimizer("actor")
        self.assertTrue(budget.failed)
        self.assertEqual(read_dynamic_ledger(self.path)["counts"], {"optimizer": 1})
        with self.assertRaises(RuntimeError):
            budget.begin("fit/b")

    def test_minibatch_admission_does_not_debit_or_refund(self):
        budget = self.budget()
        budget.begin("fit/a")
        budget.check_minibatch()
        self.assertEqual(budget.counts["optimizer"], 0)
        budget.debit_optimizer("actor")
        with self.assertRaises(RuntimeError):
            budget.check_minibatch()
        self.assertEqual(read_dynamic_ledger(self.path)["counts"], {"optimizer": 1})

    def test_failure_after_debit_preserves_charge_without_model_restore_api(self):
        budget = self.budget()
        budget.begin("collect")
        budget.debit_environment("clone")
        model_snapshot = {"weights": "unchanged"}
        restored_model = copy.deepcopy(model_snapshot)
        self.assertEqual(restored_model, model_snapshot)
        self.assertEqual(read_dynamic_ledger(self.path)["owner_counts"]["global:clone"], 1)
        self.assertFalse(hasattr(budget, "load_state_dict"))
        with self.assertRaises(RuntimeError):
            budget.debit_environment("clone")

    def test_clock_section_global_and_regression_fail_closed(self):
        for elapsed in (9, 21, -1):
            with self.subTest(elapsed=elapsed):
                path = Path(self.directory.name) / f"clock{elapsed}.jsonl"
                self.now = 10.0
                budget = DynamicCandidateBudget(path, miniature(), enabled=True, clock=lambda: self.now)
                self.addCleanup(budget.close)
                budget.begin("collect")
                self.now += elapsed
                with self.assertRaises(TimeoutError):
                    budget.check()
                self.assertTrue(budget.failed)

    def test_partial_io_poison_and_raw_ledger_is_resource_authority(self):
        budget = self.budget()
        budget.begin("fit/a")
        with patch("src.rl.candidate_pilot_resources.os.fsync", side_effect=OSError("fixture fsync failure")):
            with self.assertRaises(OSError):
                budget.debit_optimizer("actor")
        self.assertTrue(budget.failed)
        self.assertEqual(budget.counts["optimizer"], 0)
        self.assertEqual(read_dynamic_ledger(self.path)["counts"], {"optimizer": 1})

    def test_resealed_owner_forgery_is_rejected(self):
        budget = self.budget()
        budget.begin("fit/a")
        budget.debit_optimizer("actor")
        budget.finish()
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        for row in rows:
            if row["event"] == "debit":
                row["operation_owner"] = "critic"
        previous = "0" * 64
        for index, row in enumerate(rows):
            row.pop("sha256")
            row["sequence"], row["previous"] = index, previous
            previous = row["sha256"] = digest(row)
        forged = Path(self.directory.name) / "forged.jsonl"
        forged.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        with self.assertRaisesRegex(ValueError, "owner ledger arithmetic"):
            read_dynamic_ledger(forged)

    def test_no_concurrent_repeated_or_unknown_scope(self):
        budget = self.budget()
        with self.assertRaises(ValueError):
            budget.begin("unknown")
        budget.begin("collect")
        with self.assertRaises(ValueError):
            budget.begin("fit/a")
        budget.finish()
        with self.assertRaises(ValueError):
            budget.begin("collect")

    def test_ancestor_restore_keeps_later_spend_and_rejects_owner_forgery(self):
        budget = self.budget()
        budget.begin("collect")
        budget.debit_environment("trajectory")
        old = budget.snapshot()
        budget.debit_environment("clone")
        live = budget.snapshot()
        assert_dynamic_budget_ancestor(old, budget)
        self.assertEqual(budget.snapshot(), live)
        forged = copy.deepcopy(old)
        forged["owner_counts"]["global:trajectory"] = 0
        with self.assertRaises(ValueError):
            assert_dynamic_budget_ancestor(forged, budget)
        budget.finish()
        with self.assertRaises(ValueError):
            assert_dynamic_budget_ancestor(old, budget)


if __name__ == "__main__":
    unittest.main()
