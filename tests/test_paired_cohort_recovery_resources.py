"""JSON and artificial-ledger tests only; no models, environments or optimizer."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.paired_cohort_recovery_resources import budget_plan, operation_limits
from tests.test_paired_cohort_resources import ROOT, config


def recovery():
    return json.loads((ROOT / "specs/2026-10-02-paired-cohort-improvement/recovery-proposal.json").read_text())


class RecoveryResourceTests(unittest.TestCase):
    def test_exact_remaining_owner_partition_without_original_mutation(self):
        cfg, proposal = config(), recovery()
        before = copy.deepcopy((cfg, proposal))
        plan = budget_plan(cfg, proposal)
        self.assertEqual((cfg, proposal), before)
        self.assertEqual(plan["limits"], dict(trajectory=13608, clone=12047, actor=768, critic=0, seconds=18000))
        self.assertEqual(sum(r["seconds"] for r in plan["phases"].values()), 16500)
        self.assertEqual(list(plan["sections"])[:4], ["binding", "paired_branches/block60",
            "paired_branches/block61", "paired_branches/block62"])
        for block, calls, seconds in ((60, 297, 600), (61, 6020, 2400), (62, 5730, 2400)):
            owner = plan["sections"][f"paired_branches/block{block}"]
            self.assertEqual((owner["clone"], owner["seconds"]), (calls, seconds))
        self.assertEqual(plan["sections"]["evaluation/block60/paired_cost"]["seconds"], 360)
        self.assertFalse(any("preflight" in name or "reference_context" in name for name in plan["sections"]))

    def test_scope_mutation_boolean_counts_and_refunds_rejected(self):
        for field, value in (("new_environment_cap", 25654), ("new_actor_updates", 769),
                ("old_interrupted_charge_preserved", 0), ("global_seconds", 14400),
                ("reuse_preflight_pairs", True), ("automatic_retry", True),
                ("scientific_execution_authorized", True), ("new_reference_cohorts_or_preflight_episodes", 1)):
            with self.subTest(field=field):
                proposal = recovery()
                proposal[field] = value
                with self.assertRaises(ValueError):
                    budget_plan(config(), proposal)
        proposal = recovery()
        proposal["remaining_by_block"]["60"]["environment_calls"] -= 1
        with self.assertRaises(ValueError):
            budget_plan(config(), proposal)

    def test_no_phase_transfer_or_critic_debit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ledger.jsonl"
            budget = DynamicCandidateBudget(path, budget_plan(config(), recovery()), enabled=True, clock=lambda: 0.)
            self.addCleanup(budget.close)
            budget.begin("paired_actor/block60")
            budget.debit_optimizer("actor")
            with self.assertRaises(RuntimeError):
                budget.debit_optimizer("critic")
            counts = read_dynamic_ledger(path)["counts"]
            self.assertEqual(counts["optimizer"], 1)
            self.assertEqual(counts.get("environment", 0), 0)

    def test_binding_has_no_environment_steps_and_all_operations_have_owners(self):
        plan = budget_plan(config(), recovery())
        ops = operation_limits(config(), recovery())
        self.assertEqual(list(ops), list(plan["sections"]))
        totals = {name: sum(row.get(name, 0) for row in ops.values()) for row in ops.values() for name in row}
        self.assertEqual(totals, dict(input_hash_verification=12, reference_and_layout_build=3,
            reference_checkpoint_load=3, layout_environment_build=3, checkpoint_load=6,
            template_build=3, context_load=36, branch_import=117, conditional_branch_clone=285,
            branch_step=12047, actor_fork=6, actor_update=768,
            episode_build=216, episode_dispatch=216, episode_step=13608))
        self.assertEqual(ops["paired_branches/block60"]["conditional_branch_clone"], 11)
        self.assertEqual(ops["seal"], {})
        self.assertEqual(ops["raw_verification"], {})


if __name__ == "__main__":
    unittest.main()
