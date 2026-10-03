"""Artificial saved JSONL evidence only; no simulation, model or fitting calls."""

import copy
import ast
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from src.rl.capacity_pilot_analysis import AnalysisIncomplete, COST_COMPONENTS, PHASES, run_analysis


PROPOSAL_PATH = Path(__file__).resolve().parents[1] / "specs/2026-10-03-dynamic-capacity-adaptation/pilot-proposal.json"


def artificial_trajectory(root, *, phase=PHASES[2], b=0, c=1, j=0,
                          role="online_matched_fork", compressed=True, cost=None, extra_loss=False):
    """Deliberately artificial constant costs and two synthetic patient records."""
    base, stride = {PHASES[0]: (62600000, 10), PHASES[1]: (62604000, 10), PHASES[2]: (62610000, 100)}[phase]
    world = dict(phase=phase, block=b, condition=c, replicate=j, seed=base + 1000*b + stride*c + j)
    costs = {"online_matched_fork": 90, "frozen_history": 100, "frozen_matched_exploration": 101,
             "adaptive_rule": 92, "id_mpc": 95, "fixed_allocation_reference": 110, "learner": 100, "teacher": 95}
    step_cost = float(costs[role] + 10*b + c if cost is None else cost)
    name = f"{phase}-b{b}-c{c}-j{j}-{role}"
    raw_path = root / "raw" / (name + (".jsonl.gz" if compressed else ".jsonl"))
    summary_path = root / "summaries" / (name + ".json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    patients = [f"artificial-{world['seed']}-{i}" for i in range(2)]
    rows = []
    for epoch in range(64):
        terminal = epoch >= 6
        lost = int(extra_loss and terminal)
        records = [{"patient_id": pid, "enrollment_epoch": 0, "collection_site": i,
                    "status": ("lost" if i == 0 and lost else "delivered") if terminal else ("in_production" if epoch >= 3 else "waiting"),
                    "age": min(epoch, 6), "survival": 0.9, "support_complete": epoch >= 2}
                   for i, pid in enumerate(patients)]
        components = dict.fromkeys(COST_COMPONENTS, 0.0)
        components["reagent_purchase_cost"] = step_cost
        rows.append({"epoch": epoch, "world": world, "role": role, "cost": step_cost,
                     "components": components, "reward": -step_cost,
                     "requested_hours": [2.0]*4 if epoch < 48 else [0.0]*4,
                     "executed_hours": [2.0]*4 if epoch < 48 else [0.0]*4,
                     "filter_resets": 0, "cumulative": {"enrolled": 2, "delivered": 2-lost if terminal else 0, "lost": lost},
                     "support_backlog": 2 if epoch < 2 else 0, "patient_records": records,
                     "service": {"epoch": epoch, "known_at": epoch+1, "ordinary_hours": [4.0]*4,
                                 "applied_hours": [2.0]*4 if 2 <= epoch < 50 else [0.0]*4,
                                 "completed_ids": [[patients[0]], [patients[1]], [], []] if epoch == 2 else [[], [], [], []]},
                     "info": {"cost": step_cost, "base_cost": step_cost, "native_cost": step_cost}})
    summary = {"world": world, "role": role, "raw_path": str(raw_path.relative_to(root)), "settled": True,
               "settlement": {"settled": True, "live_ids": [], "pending_obligations": 0, "resource_conservation": True,
                              "enrolled": 2, "delivered": 1 if extra_loss else 2, "lost": int(extra_loss)},
               "change_epoch": 20 if c == 1 else None, "tape_sha256": f"{world['seed']:064x}",
               "cost": step_cost*64, "model_seal_sha256": f"{b+10:064x}" if phase == PHASES[2] else None}
    write_rows(raw_path, rows)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return raw_path, summary_path, rows, summary


def write_rows(path, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


class CapacityPilotAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposal = json.loads(PROPOSAL_PATH.read_text())
        cls.complete_dir = tempfile.TemporaryDirectory(prefix="artificial-capacity-complete-")
        cls.complete_root = Path(cls.complete_dir.name)
        for phase in PHASES:
            roles = ([r["id"] for r in cls.proposal["controllers"]] if phase == PHASES[2]
                     else ["teacher" if phase == PHASES[0] else "learner"])
            for b in range(3):
                for c in range(3):
                    for j in range(4):
                        for role in roles:
                            artificial_trajectory(cls.complete_root, phase=phase, b=b, c=c, j=j, role=role)
        cls.report = run_analysis(cls.complete_root, cls.proposal)

    @classmethod
    def tearDownClass(cls):
        cls.complete_dir.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="artificial-capacity-analysis-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def partial(self):
        return run_analysis(self.root, self.proposal, require_complete=False)

    def full_copy(self):
        shutil.copytree(self.complete_root, self.root, dirs_exist_ok=True)

    def test_full_primary_pairs_patient_contrasts_and_signal(self):
        report = self.report
        self.assertTrue(report["primary_complete"])
        self.assertTrue(report["evaluation_complete"])
        self.assertEqual(report["observed_trajectories"], 288)
        self.assertEqual(report["observed_evaluation_trajectories"], 216)
        self.assertEqual(report["phase_counts"], {PHASES[2]: 216, PHASES[1]: 36, PHASES[0]: 36})
        self.assertEqual(len(report["paired_integrity"]), 36)
        for contrast in report["contrasts"].values():
            self.assertEqual(len(contrast["worlds"]), 36)
            self.assertTrue(contrast["overall"]["complete"])
            self.assertEqual(contrast["overall"]["mean_extra_lost"], 0)
        frozen = report["contrasts"]["online_vs_frozen_history"]
        self.assertEqual(frozen["overall"]["equal_condition_mean_savings"], 640)
        self.assertEqual(frozen["conditions"]["1"]["blocks"]["0"]["mean_savings"], 640)
        self.assertAlmostEqual(frozen["worlds"][4]["relative_savings_percent"], 100*10/101)
        self.assertTrue(report["signal"]["candidate_online_signal"])
        self.assertFalse(report["signal"]["clinical_success_claim"])
        self.assertEqual(report["signal"]["negative_findings"], [])
        json.dumps(report, allow_nan=False)

    def test_bootstrap_fixed_shared_hierarchical_draws(self):
        bootstrap = self.report["bootstrap"]
        self.assertEqual(bootstrap["seed"], 62690001)
        self.assertEqual(bootstrap["resamples"], 2000)
        self.assertTrue(bootstrap["same_draws_across_contrasts"])
        self.assertEqual(bootstrap["intervals"]["online_vs_frozen_history"]["conditions"]["1"]["savings"], [640, 640])
        # Relative savings vary by block, so a block-first CI is nondegenerate.
        rng = np.random.default_rng(62690001)
        blocks = rng.integers(3, size=(2000, 3))
        worlds = rng.integers(4, size=(2000, 3, 3, 4))
        self.assertEqual(worlds.shape, (2000, 3, 3, 4))
        relative = np.mean(1000/(101 + 10*blocks), axis=1)
        expected = np.percentile(relative, [2.5, 97.5]).tolist()
        actual = bootstrap["intervals"]["online_vs_frozen_history"]["conditions"]["1"]["relative_savings_percent"]
        np.testing.assert_allclose(actual, expected)

    def test_readonly_deterministic_and_proposal_path(self):
        def fingerprints():
            return {str(path.relative_to(self.complete_root)): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
                    for path in self.complete_root.rglob("*") if path.is_file()}
        before = fingerprints()
        again = run_analysis(self.complete_root, PROPOSAL_PATH)
        self.assertEqual(before, fingerprints())
        self.assertEqual(again["bootstrap"], self.report["bootstrap"])

    def test_bootstrap_resamples_worlds_inside_each_drawn_block_condition(self):
        self.full_copy()
        deltas = np.empty((3, 3, 4))
        for b in range(3):
            for c in range(3):
                for j in range(4):
                    difference = 1 + b + 2*c + 3*j
                    deltas[b, c, j] = 64*difference
                    artificial_trajectory(self.root, b=b, c=c, j=j, cost=100 + 10*b + c - difference)
        report = run_analysis(self.root, self.proposal)
        rng = np.random.default_rng(62690001)
        blocks = rng.integers(3, size=(2000, 3))
        worlds = rng.integers(4, size=(2000, 3, 3, 4))
        reference = np.empty((2000, 3))
        for draw in range(2000):
            for c in range(3):
                reference[draw, c] = sum(deltas[blocks[draw, position], c, world]
                                          for position in range(3)
                                          for world in worlds[draw, position, c]) / 12
        intervals = report["bootstrap"]["intervals"]["online_vs_frozen_history"]
        for c in range(3):
            np.testing.assert_allclose(intervals["conditions"][str(c)]["savings"], np.percentile(reference[:, c], [2.5, 97.5]))
        np.testing.assert_allclose(intervals["equal_condition_overall"]["savings"], np.percentile(reference.mean(axis=1), [2.5, 97.5]))

    def test_supported_secondaries_and_explicit_missing(self):
        row = self.report["trajectories"][0]
        self.assertTrue(row["cost_components"]["complete"])
        secondary = row["secondary"]
        self.assertEqual(secondary["loss_delivery_rates"]["enrolled"], 2)
        self.assertEqual(secondary["mean_p90_support_wait"]["completion_delay"]["mean"], 2)
        self.assertEqual(secondary["mean_p90_wait_turnaround"]["delivered_turnaround"]["p90"], 6)
        self.assertEqual(secondary["all_patient_time_to_loss_or_delivery"]["n"], 2)
        self.assertEqual(secondary["ordinary_flexible_booked_matured_productive_idle_hours"]["ordinary"]["observed_sum"], 1024)
        self.assertIn("productive/idle", row["missing_secondary"]["ordinary_flexible_booked_matured_productive_idle_hours"])
        self.assertIn("training_deployment_query_time_costs", row["missing_secondary"])
        self.assertEqual(secondary["filter_resets"]["total_resets"], 0)
        self.assertEqual(len(self.report["exploration_cost"]), 36)
        self.assertEqual(self.report["exploration_cost"][0]["incremental_cost"], 64)
        self.assertTrue(all(row["lag_epochs"] == 7 for row in self.report["recovery_lag"]))
        self.assertEqual(self.report["secondary_coverage"]["training_deployment_query_time_costs"]["status"], "missing")
        self.assertEqual(self.report["secondary_coverage"]["cost_components"]["status"], "available")

    def test_missing_components_do_not_impute_or_block_raw_cost_primary(self):
        self.full_copy()
        raw, summary, rows, _ = artificial_trajectory(self.root)
        for row in rows:
            del row["components"]["reagent_purchase_cost"]
        write_rows(raw, rows)
        report = run_analysis(self.root, self.proposal)
        trajectory = next(r for r in report["trajectories"] if r["raw_path"] == str(raw.relative_to(self.root)))
        self.assertTrue(report["primary_complete"])
        self.assertEqual(trajectory["cost"], 91*64)
        self.assertFalse(trajectory["cost_components"]["complete"])
        self.assertNotIn("reagent_purchase_cost", trajectory["cost_components"]["observed_totals"])
        self.assertEqual(trajectory["cost_components"]["missing_or_invalid_epochs"]["reagent_purchase_cost"], 64)
        self.assertEqual(len(trajectory["cost_components"]["reconciliation_errors"]), 64)

    def test_partial_unsummarized_gzip_prefix_and_strict_report(self):
        raw, summary, rows, _ = artificial_trajectory(self.root)
        summary.unlink()
        write_rows(raw, rows[:9])
        report = self.partial()
        row = report["trajectories"][0]
        self.assertEqual(row["cost"], 91*9)
        self.assertFalse(row["cost_complete"])
        self.assertEqual(row["patient_counts"], {"enrolled": 2, "delivered": 2, "lost": 0})
        self.assertEqual(report["signal"]["classification"], "engineering_failure_not_rl_null")
        self.assertIsNone(report["signal"]["candidate_online_signal"])
        self.assertIsNone(report["bootstrap"]["intervals"])
        with self.assertRaises(AnalysisIncomplete) as context:
            run_analysis(self.root, self.proposal)
        self.assertEqual(context.exception.report["trajectories"][0]["cost"], 91*9)

    def test_plain_jsonl_and_no_component_double_count_from_info(self):
        raw, _, rows, _ = artificial_trajectory(self.root, compressed=False)
        for row in rows:
            row["info"].update(transfer_cost=900000, base_cost=900000, native_cost=900000)
        write_rows(raw, rows)
        row = self.partial()["trajectories"][0]
        self.assertEqual(row["cost"], 91*64)
        self.assertEqual(row["cost_components"]["observed_totals"]["reagent_purchase_cost"], 91*64)
        self.assertTrue(row["cost_components"]["complete"])

    def test_summary_cost_and_registry_counters_checked_independently(self):
        raw, summary_path, rows, summary = artificial_trajectory(self.root)
        summary["cost"] += 100
        summary_path.write_text(json.dumps(summary))
        rows[10]["cumulative"]["delivered"] = 1
        write_rows(raw, rows)
        report = self.partial()
        row = report["trajectories"][0]
        codes = {item["code"] for item in row["issues"]}
        self.assertIn("summary_vs_raw_cost_mismatch", codes)
        self.assertIn("cumulative_vs_registry_mismatch", codes)
        self.assertEqual(row["cost"], 91*64)
        self.assertEqual(row["patient_counts"]["delivered"], 2)

    def test_duplicate_missing_epochs_and_nonfinite_rows_are_not_complete(self):
        raw, _, rows, _ = artificial_trajectory(self.root)
        rows[5]["epoch"] = 4
        rows[10]["cost"] = float("nan")
        write_rows(raw, rows)
        row = self.partial()["trajectories"][0]
        codes = {item["code"] for item in row["issues"]}
        self.assertIn("malformed_json_row", codes)
        self.assertIn("epochs_not_exactly_0_through_63", codes)
        self.assertFalse(row["cost_complete"])
        self.assertEqual(row["cost"], 91*63)

    def test_pair_digest_model_and_patient_cohort_mismatch(self):
        self.full_copy()
        raw, summary_path, rows, summary = artificial_trajectory(self.root)
        summary["tape_sha256"] = "f"*64
        summary["model_seal_sha256"] = "e"*64
        summary_path.write_text(json.dumps(summary))
        for row in rows:
            row["patient_records"][0]["patient_id"] = "different-artificial-patient"
        write_rows(raw, rows)
        report = self.partial()
        world = next(r for r in report["paired_integrity"] if (r["block"], r["condition"], r["replicate"]) == (0, 1, 0))
        self.assertIn("six_arm_tape_mismatch", world["issues"])
        self.assertIn("same_start_model_seal_mismatch", world["issues"])
        self.assertIn("paired_patient_cohort_mismatch", world["issues"])
        self.assertFalse(report["evaluation_complete"])

    def test_cost_down_losses_up_is_not_positive_signal(self):
        self.full_copy()
        for b in range(3):
            for j in range(4):
                artificial_trajectory(self.root, b=b, c=1, j=j, extra_loss=True)
        report = run_analysis(self.root, self.proposal)
        self.assertFalse(report["signal"]["candidate_online_signal"])
        self.assertEqual(report["signal"]["classification"], "no_reliable_online_benefit_at_declared_budget_not_equivalence")
        self.assertTrue(report["signal"]["cost_patient_tradeoffs"])
        cell = report["contrasts"]["online_vs_frozen_history"]["conditions"]["1"]
        self.assertEqual(cell["mean_extra_lost"], 1)
        self.assertEqual(cell["mean_extra_delivered"], -1)
        self.assertEqual(cell["mean_loss_rate_difference"], 0.5)

    def test_one_nonpositive_training_block_prevents_signal(self):
        self.full_copy()
        for j in range(4):
            artificial_trajectory(self.root, b=0, c=1, j=j, cost=101)
        report = run_analysis(self.root, self.proposal)
        self.assertFalse(report["signal"]["candidate_online_signal"])
        self.assertFalse(report["signal"]["checks"]["online_vs_frozen_history"]["all_three_block_savings_positive"])

    def test_zero_comparator_cost_is_explicit_not_divide_by_zero(self):
        self.full_copy()
        artificial_trajectory(self.root, role="frozen_history", cost=0)
        report = run_analysis(self.root, self.proposal)
        row = next(r for r in report["contrasts"]["online_vs_frozen_history"]["worlds"]
                   if (r["block"], r["condition"], r["replicate"]) == (0, 1, 0))
        self.assertIsNone(row["relative_savings_percent"])
        self.assertIsNone(report["bootstrap"]["intervals"]["online_vs_frozen_history"]["conditions"]["1"]["relative_savings_percent"])
        json.dumps(report, allow_nan=False)

    def test_truncated_gzip_preserves_decoded_rows(self):
        raw, _, _, _ = artificial_trajectory(self.root)
        raw.write_bytes(raw.read_bytes()[:-8])
        row = self.partial()["trajectories"][0]
        self.assertTrue(any(x["code"] == "unreadable_or_truncated_raw_file" for x in row["issues"]))
        self.assertFalse(row["primary_complete"])
        self.assertGreater(row["observed_rows"], 0)

    def test_duplicate_raw_and_missing_training_fail_completeness(self):
        self.full_copy()
        artificial_trajectory(self.root, compressed=False)
        path = next((self.root / "raw").glob(PHASES[0] + "-*.gz"))
        path.unlink()
        report = self.partial()
        codes = {item["code"] for item in report["issues"]}
        self.assertIn("duplicate_trajectory", codes)
        self.assertIn("summary_without_raw", codes)
        self.assertIn("missing_trajectories", codes)
        self.assertFalse(report["primary_complete"])

    def test_protocol_not_mutated_and_bootstrap_contract_not_overridden(self):
        before = copy.deepcopy(self.proposal)
        self.partial()
        self.assertEqual(self.proposal, before)
        modified = copy.deepcopy(before)
        modified["analysis"]["bootstrap"]["resamples"] = 2
        with self.assertRaisesRegex(ValueError, "numerical contract"):
            run_analysis(self.root, modified, require_complete=False)

    def test_actual_runner_constants_without_importing_scientific_modules(self):
        root = PROPOSAL_PATH.parents[2]
        def constant(path, name):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                    return ast.literal_eval(node.value)
            self.fail(f"missing source constant {name}")
        self.assertEqual(COST_COMPONENTS, constant(root / "src/rl/capacity_pilot_runner.py", "COST_KEYS"))
        self.assertEqual(PHASES, constant(root / "src/rl/capacity_adaptation_campaign.py", "PHASES"))

    def test_raw_reward_is_unscaled_and_reset_count_is_cumulative(self):
        raw, _, rows, _ = artificial_trajectory(self.root)
        for row in rows:
            row["filter_resets"] = row["epoch"] // 10
        write_rows(raw, rows)
        trajectory = self.partial()["trajectories"][0]
        self.assertEqual(trajectory["secondary"]["filter_resets"]["total_resets"], 6)
        rows[47]["reward"] /= 100000
        write_rows(raw, rows)
        trajectory = self.partial()["trajectories"][0]
        self.assertTrue(any(i["code"] == "raw_reward_must_equal_negative_unscaled_cost" for i in trajectory["issues"]))
        self.assertEqual(trajectory["cost"], 91*64)


if __name__ == "__main__":
    unittest.main()
