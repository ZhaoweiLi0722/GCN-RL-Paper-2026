"""Artificial JSON and temporary ledgers only: no saved models or scientific work."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_planner_tail_recovery1_resources import (
    PARTIAL, PARTIAL_REMAINING, RecoveryBudget, remainder, remaining_scope)


def inputs():
    study = json.loads('''{
      "schema": "capacity-planner-tail-study-v1",
      "scientific_execution_authorized": false,
      "design": {
        "blocks": 5, "conditions": 3, "reference_worlds_per_block": 24,
        "evaluation_worlds_per_condition_block": 4,
        "train_roles": ["observed_td", "planner_tail_td", "planner_tail_mc"],
        "eval_roles": ["plain_h8", "existing_frozen", "observed_td",
                       "planner_tail_td", "planner_tail_mc", "plain_h16"],
        "new_training_worlds": 120, "evaluation_trajectories": 360,
        "total_native_trajectories": 480
      },
      "planning": {"candidate_sequences": 16, "response_quantiles": [0.1, 0.5, 0.9],
                   "summary_weights": [0.25, 0.5, 0.25], "horizon": 8,
                   "long_reference_horizon": 16},
      "value": {"updates_per_world": 32, "batch_size": 64, "td_horizon": 8},
      "tails": {"per_world": 96, "final_epoch": 64},
      "budget": {
        "attempts": 1, "automatic_retry": false, "native_steps": 30720,
        "native_operations": 31680, "value_optimizer_steps": 11520,
        "actor_optimizer_steps": 0, "optimizer_examples": 737280,
        "forward_calls": 30000, "maximum_forward_batch": 64,
        "planning_decisions": 23040, "candidate_quantile_rollouts": 1105920,
        "planning_epochs": 9953280, "training_tail_epochs": 374400,
        "total_predictor_epochs": 10327680, "forecast_clones": 11520,
        "filter_transitions": 3072000, "new_final_seals": 15,
        "seconds": {"admission": 600, "reference_and_tails": 10800,
                    "value_fitting": 3600, "frozen_evaluation": 14400,
                    "analysis_archive": 1200, "failure_preservation": 1800,
                    "global": 32400},
        "owner_seconds": {"evaluation_h16_world": 240},
        "raw_bytes": 4294967296, "archive_bytes": 4294967296,
        "combined_disk_bytes": 8589934592, "file_cap": 6000,
        "rss_bytes": 4294967296, "compute_threads": 4
      }
    }''')
    counts = dict(
        trajectories=480, native_steps=30666, native_constructions=480,
        construction_triggered_resets=480, total_native_operations=31626,
        control_steps=23002, tail_steps=7664, value_optimizer_steps=11520,
        total_optimizer_steps=11520, optimizer_example_presentations=737280,
        neural_forward_module_calls=29040, planner_total_decisions=23003,
        planner_candidate_rollouts=1104144, planner_total_model_epochs=9924864,
        training_tail_model_epochs=374400, forecast_clones=11520,
        estimator_receipt_updates=30666, estimator_hypothesis_transitions=3066600,
        final_seals=15)
    reference, fitting, analysis = (dict.fromkeys(counts, 0) for _ in range(3))
    reference.update(trajectories=120, native_steps=7680, native_constructions=120,
        construction_triggered_resets=120, total_native_operations=7920,
        control_steps=5760, tail_steps=1920, planner_total_decisions=5760,
        planner_candidate_rollouts=276480, planner_total_model_epochs=2211840,
        training_tail_model_epochs=374400, forecast_clones=11520,
        estimator_receipt_updates=7680, estimator_hypothesis_transitions=768000)
    fitting.update(value_optimizer_steps=11520, total_optimizer_steps=11520,
        optimizer_example_presentations=737280, neural_forward_module_calls=17520,
        final_seals=15)
    evaluation = {k: v - reference[k] - fitting[k] for k, v in counts.items()}
    old = dict(format="capacity-planner-tail-budget-v1", counts=counts,
        phases=dict(reference_and_tails=reference, value_fitting=fitting,
                    frozen_evaluation=evaluation, analysis_archive=analysis),
        failed=True, owner="frozen_evaluation", phase="frozen_evaluation",
        job_id=PARTIAL, job_kind="evaluation_h16_world",
        job_times={PARTIAL: 13.66981799993664}, elapsed=19559.984044999816,
        times=dict(admission_lock_binding=5.039817999815568,
                   reference_and_tails=5083.981840000255, value_fitting=70.95946999988519,
                   frozen_evaluation=14400.00291699986, analysis_archive=0.,
                   failure_flush_shutdown_reserve=0., **{"global": 0.}),
        pending_chunks=dict(planner_total_model_epochs=dict(reserved=768, dispatched=411)),
        verified_completed_chunk_calls=dict(planner_total_model_epochs=9924096,
            training_tail_model_epochs=374400, estimator_hypothesis_transitions=3066600))
    terminal = dict(status="failed", automatic_retry=False, scientific_completion_verified=False,
                    child_exit_code=1, exit_code=1, elapsed=19560.15342499991,
                    packet_sha256="a" * 64)
    storage = dict(raw_bytes=100, archive_bytes=20, total_bytes=120, files=3)
    # Exercise only the persisted JSON value types, with no real artifact fixtures.
    return json.loads(json.dumps([study, old, terminal, storage]))


def proposal(study):
    return dict(planner_tail_study=copy.deepcopy(study),
        scope=dict(study="capacity-planner-tail-v1", automatic_retry=False, resume=False),
        proposed_budget=dict(time_seconds={}, raw_artifact_cap_bytes=4294967296,
            archive_cap_bytes=4294967296, combined_disk_cap_bytes=8589934592,
            file_cap=6000, rss_cap_bytes=4294967296, compute_threads=4))


class FakeClock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


class RemainingContractTests(unittest.TestCase):
    def test_exact_remaining_and_combined_arithmetic_no_authority(self):
        data = inputs()
        untouched = copy.deepcopy(data)
        r = remainder(*data)
        nonzero = {k: v for k, v in r["limits"].items() if v}
        self.assertEqual(nonzero, dict(native_steps=54, total_native_operations=54,
            control_steps=38, tail_steps=16, neural_forward_module_calls=960,
            planner_total_decisions=38, planner_candidate_rollouts=1824,
            planner_total_model_epochs=29184, estimator_receipt_updates=54,
            estimator_hypothesis_transitions=5400))
        self.assertEqual(r["seconds"], dict(admission_lock_binding=120,
            frozen_evaluation=240, analysis_archive=1200,
            failure_flush_shutdown_reserve=240, **{"global": 1800}))
        self.assertEqual(sum(v for k, v in r["seconds"].items() if k != "global"), 1800)
        self.assertEqual(r["prior_elapsed_seconds"], data[2]["elapsed"])
        self.assertEqual(r["status"], "DRAFT")
        self.assertFalse(r["scientific_execution_authorized"])
        self.assertFalse(r["training_permitted"])
        self.assertEqual(r["remaining_scope"], remaining_scope)
        self.assertEqual(r["boundary"], dict(job_id=PARTIAL, epoch=10, saved_rows=10,
            completed_reference=120, completed_evaluation=359))
        for k, v in r["limits"].items():
            self.assertEqual(sum(p[k] for p in r["phase_limits"].values()), v)
            replacement = dict(planner_total_decisions=1, planner_candidate_rollouts=48,
                               planner_total_model_epochs=768).get(k, 0)
            self.assertEqual(r["combined_limits"][k], r["original_limits"][k] + replacement)
        self.assertEqual(r["combined_limits"]["planner_total_model_epochs"], 9954048)
        self.assertEqual(r["combined_limits"]["planner_total_decisions"], 23041)
        self.assertEqual(r["combined_limits"]["planner_candidate_rollouts"], 1105968)
        self.assertEqual(r["retained_interrupted_reservation"], data[1]["pending_chunks"])
        self.assertEqual(data, untouched)
        r["old_counts"]["native_steps"] = -1
        r["original_budget"]["counts"]["native_steps"] = -2
        r["prior_storage"]["files"] = 1
        self.assertEqual(data, untouched)

    def test_phase_limits_all_other_counts_zero(self):
        r = remainder(*inputs())
        for phase in ("reference_and_tails", "value_fitting"):
            self.assertFalse(any(r["phase_limits"][phase].values()))
        self.assertEqual({k: v for k, v in r["phase_limits"]["analysis_archive"].items() if v},
                         dict(neural_forward_module_calls=960))
        self.assertEqual(r["phase_limits"]["frozen_evaluation"]["neural_forward_module_calls"], 0)
        for k in ("trajectories", "native_constructions", "construction_triggered_resets",
                  "value_optimizer_steps", "total_optimizer_steps", "optimizer_example_presentations",
                  "final_seals", "forecast_clones", "training_tail_model_epochs"):
            self.assertEqual(r["limits"][k], 0)

    def test_reject_every_changed_missing_extra_or_noninteger_counter(self):
        for key in inputs()[1]["counts"]:
            for value in (-1, 0, True, 1.5, "0", None):
                data = inputs()
                data[1]["counts"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    remainder(*data)
            data = inputs()
            del data[1]["counts"][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                remainder(*data)
        data = inputs()
        data[1]["counts"]["extra"] = 0
        with self.assertRaises(ValueError):
            remainder(*data)

    def test_reject_every_phase_counter_change_even_with_unchanged_totals(self):
        for phase, counts in inputs()[1]["phases"].items():
            for key in counts:
                data = inputs()
                data[1]["phases"][phase][key] += 1
                with self.subTest(phase=phase, counter=key), self.assertRaises(ValueError):
                    remainder(*data)
        for missing in (True, False):
            data = inputs()
            if missing:
                del data[1]["phases"]["value_fitting"]
            else:
                data[1]["phases"]["unexpected"] = {}
            with self.assertRaises(ValueError):
                remainder(*data)

    def test_reject_wrong_original_failure_identity_and_terminal(self):
        old_changes = dict(format="other", failed=False, owner="analysis_archive",
            phase="value_fitting", job_id=PARTIAL + "-other", job_kind="evaluation_h8_world")
        terminal_changes = dict(status="complete", automatic_retry=True,
            scientific_completion_verified=True, exit_code=0, child_exit_code=True)
        for index, changes in ((1, old_changes), (2, terminal_changes)):
            for key, value in changes.items():
                data = inputs()
                data[index][key] = value
                with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                    remainder(*data)
        data = inputs()
        data[1]["job_times"][PARTIAL] = 0
        with self.assertRaises(ValueError):
            remainder(*data)

    def test_reject_pending_reservation_refunds_progress_changes_and_completed_claims(self):
        for pending in ({}, {"other": dict(reserved=768, dispatched=411)},
                        {"planner_total_model_epochs": dict(reserved=411, dispatched=411)},
                        {"planner_total_model_epochs": dict(reserved=768, dispatched=410)},
                        {"planner_total_model_epochs": dict(reserved=768., dispatched=411)}):
            data = inputs()
            data[1]["pending_chunks"] = pending
            with self.subTest(pending=pending), self.assertRaises(ValueError):
                remainder(*data)
        for key in inputs()[1]["verified_completed_chunk_calls"]:
            data = inputs()
            data[1]["verified_completed_chunk_calls"][key] += 411
            with self.subTest(completed=key), self.assertRaises(ValueError):
                remainder(*data)

    def test_reject_nonfinite_inconsistent_global_or_nonphase_timeout(self):
        for field in inputs()[1]["times"]:
            for value in (float("nan"), float("inf"), -1, True, "0"):
                data = inputs()
                data[1]["times"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    remainder(*data)
        for index, key, value in ((1, "elapsed", 1), (2, "elapsed", 32400),
                                  (2, "elapsed", float("nan")), (2, "elapsed", 19559)):
            data = inputs()
            data[index][key] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                remainder(*data)
        data = inputs()
        data[1]["times"]["frozen_evaluation"] = 14400
        data[1]["elapsed"] = sum(data[1]["times"].values())
        with self.assertRaises(ValueError):
            remainder(*data)

    def test_reject_changed_original_design_time_or_storage_caps(self):
        for key, value in (("file_cap", 6001), ("combined_disk_bytes", 8589934593)):
            data = inputs()
            data[0]["budget"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                remainder(*data)
        data = inputs()
        data[0]["budget"]["seconds"]["global"] += 1
        data[0]["budget"]["seconds"]["frozen_evaluation"] += 1
        with self.assertRaises(ValueError):
            remainder(*data)
        data = inputs()
        data[0]["planning"]["long_reference_horizon"] = 8
        with self.assertRaises(ValueError):
            remainder(*data)

    def test_storage_subtracts_each_original_category_and_rejects_invalid_totals(self):
        r = remainder(*inputs())
        self.assertEqual(r["remaining_storage"], dict(raw_bytes=4294967196,
            archive_bytes=4294967276, total_bytes=8589934472, files=5997))
        for key in inputs()[3]:
            for value in (-1, True, 1.5, "100", None):
                data = inputs()
                data[3][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    remainder(*data)
        for storage in (dict(raw_bytes=100, archive_bytes=20, total_bytes=121, files=3),
                        dict(raw_bytes=100, archive_bytes=20, total_bytes=120, files=6000),
                        dict(raw_bytes=4294967296, archive_bytes=0, total_bytes=4294967296, files=3),
                        dict(raw_bytes=0, archive_bytes=4294967297, total_bytes=4294967297, files=3)):
            data = inputs()
            data[3] = storage
            with self.subTest(storage=storage), self.assertRaises(ValueError):
                remainder(*data)
        for i in range(4):
            data = inputs()
            data[i] = {}
            with self.subTest(missing=i), self.assertRaises(ValueError):
                remainder(*data)


class RecoveryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(dir="/private/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "launcher"
        self.clock = FakeClock()
        self.data = inputs()
        self.proposal = proposal(self.data[0])
        self.contract = remainder(*self.data)

    def budget(self):
        budget = RecoveryBudget(self.proposal, self.contract, root=self.root, clock=self.clock)
        self.addCleanup(budget.close)
        return budget

    def evaluation(self, budget):
        budget.enter("frozen_evaluation", "frozen_evaluation")
        budget.job(PARTIAL_REMAINING, "evaluation_h16_world")

    def test_zero_new_counts_old_reservation_retained_and_inputs_unchanged(self):
        self.proposal["_recovery_contract"] = copy.deepcopy(self.contract)
        self.proposal["current_scope"] = remaining_scope
        before = copy.deepcopy([self.proposal, self.contract])
        budget = self.budget()
        self.assertFalse(any(budget.counts.values()))
        self.assertEqual(budget.chunks, {})
        self.assertEqual(budget.job_caps, dict(evaluation_h16_world=240))
        self.assertEqual(budget.snapshot()["combined_counts"], self.data[1]["counts"])
        self.assertEqual(budget.snapshot()["retained_interrupted_reservation"],
                         self.data[1]["pending_chunks"])
        caps = budget.proposal["proposed_budget"]
        self.assertEqual(caps["time_seconds"], self.contract["seconds"])
        self.assertEqual(caps["raw_artifact_cap_bytes"], 4294967196)
        self.assertEqual(caps["archive_cap_bytes"], 4294967276)
        self.assertEqual(caps["combined_disk_cap_bytes"], 8589934472)
        self.assertEqual(caps["file_cap"], 5997)
        self.assertEqual(caps["planner_model_epochs_per_decision"], 768)
        budget.contract["limits"]["native_steps"] = 0
        self.assertEqual([self.proposal, self.contract], before)

    def test_complete_artificial_receipts_preserve_old_charges_and_hash_chain(self):
        budget = self.budget()
        self.evaluation(budget)
        evaluation = self.contract["phase_limits"]["frozen_evaluation"]
        budget.debit({k: v for k, v in evaluation.items()
                      if v and k not in budget.executed_chunks})
        for _ in range(38):
            budget.chunk_call("planner_total_model_epochs", 768, 411)
            budget.chunk_call("planner_total_model_epochs", 768, 357)
            budget.finish_chunk("planner_total_model_epochs")
        for _ in range(54):
            budget.chunk_call("estimator_hypothesis_transitions", 100, 100)
            budget.finish_chunk("estimator_hypothesis_transitions")
        budget.enter("analysis_archive", "analysis_archive")
        budget.debit(dict(neural_forward_module_calls=960))
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["counts"], self.contract["limits"])
        self.assertEqual(snapshot["combined_counts"], self.contract["combined_limits"])
        self.assertEqual(snapshot["pending_chunks"], {})
        self.assertEqual(snapshot["verified_completed_chunk_calls"]["planner_total_model_epochs"], 29184)
        self.assertFalse(snapshot["scientific_execution_authorized"])
        rows = [json.loads(line) for line in (self.root / "budget.jsonl").read_text().splitlines()]
        previous = "0" * 64
        for sequence, row in enumerate(rows):
            seal = row.pop("sha256")
            self.assertEqual(row["sequence"], sequence)
            self.assertEqual(row["previous"], previous)
            self.assertEqual(digest(row), seal)
            previous = seal
        self.assertEqual(previous, snapshot["ledger_sha256"])
        self.assertEqual(rows[0]["original_counts"], self.data[1]["counts"])
        self.assertEqual(rows[0]["retained_reservation"], self.data[1]["pending_chunks"])
        with self.assertRaises(RuntimeError):
            budget.debit(dict(neural_forward_module_calls=1))

    def test_reject_tampered_contract_before_creating_ledger(self):
        changes = (("limits", "native_steps", 55), ("limits", "total_optimizer_steps", 1),
                   ("seconds", "frozen_evaluation", 241),
                   ("remaining_storage", "files", 6000),
                   ("boundary", "epoch", 11))
        for section, key, value in changes:
            bad = copy.deepcopy(self.contract)
            bad[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                RecoveryBudget(self.proposal, bad, root=self.root, clock=self.clock)
            self.assertFalse(self.root.exists())
        for key, value in (("current_scope", "other"), ("_recovery_contract", {})):
            bad = copy.deepcopy(self.proposal)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                RecoveryBudget(bad, self.contract, root=self.root, clock=self.clock)
            self.assertFalse(self.root.exists())

    def test_training_phases_jobs_and_cross_owner_phase_borrowing_rejected(self):
        budget = self.budget()
        for owner, phase in (("reference_and_tails", "reference_and_tails"),
                             ("value_fitting", "value_fitting"),
                             ("admission_lock_binding", "frozen_evaluation"),
                             ("failure_flush_shutdown_reserve", "analysis_archive"),
                             ("analysis_archive", "frozen_evaluation"), ("global", None)):
            with self.subTest(owner=owner, phase=phase), self.assertRaises(ValueError):
                budget.enter(owner, phase)
        self.evaluation(budget)
        for name, kind in ((PARTIAL, "evaluation_h16_world"),
                           (PARTIAL_REMAINING + "-retry", "evaluation_h16_world"),
                           (PARTIAL_REMAINING, "fit"), (PARTIAL_REMAINING, "seal")):
            with self.subTest(name=name, kind=kind), self.assertRaises(ValueError):
                budget.job(name, kind)
        with self.assertRaises(RuntimeError):
            budget.debit(dict(neural_forward_module_calls=1))
        self.assertTrue(budget.failed)

    def test_all_forbidden_counters_fail_closed_without_partial_debits(self):
        for key, limit in self.contract["limits"].items():
            if limit:
                continue
            with self.subTest(counter=key), TemporaryDirectory(dir="/private/tmp") as path:
                budget = RecoveryBudget(self.proposal, self.contract,
                                        root=Path(path) / "launcher", clock=self.clock)
                try:
                    self.evaluation(budget)
                    with self.assertRaises(RuntimeError):
                        budget.debit({"native_steps": 1, key: 1})
                    self.assertFalse(any(budget.counts.values()))
                    self.assertTrue(budget.failed)
                finally:
                    budget.close()

    def test_native_work_cannot_borrow_analysis_allowance(self):
        budget = self.budget()
        budget.enter("analysis_archive", "analysis_archive")
        with self.assertRaises(RuntimeError):
            budget.debit(dict(native_steps=1))

    def test_evaluation_debit_requires_partial_owner(self):
        budget = self.budget()
        budget.enter("frozen_evaluation", "frozen_evaluation")
        with self.assertRaises(RuntimeError):
            budget.debit(dict(native_steps=1))

    def test_partial_owner_time_cannot_be_reset_by_job_or_phase_reentry(self):
        budget = self.budget()
        self.evaluation(budget)
        self.clock.now += 100
        budget.check()
        budget.job(PARTIAL_REMAINING, "evaluation_h16_world")
        self.assertAlmostEqual(budget.job_times[PARTIAL_REMAINING], 113.66981799993664)
        budget.enter("analysis_archive", "analysis_archive")
        self.clock.now += 5
        self.evaluation(budget)
        self.clock.now += 126.34
        with self.assertRaises(TimeoutError):
            budget.check()
        self.assertLess(budget.times["frozen_evaluation"], 240)
        self.assertGreater(budget.job_times[PARTIAL_REMAINING], 240)

    def test_exact_owner_cap_and_no_extra_fraction(self):
        budget = self.budget()
        self.evaluation(budget)
        self.clock.now += 240 - self.contract["partial_job_seconds"]
        budget.check()
        self.clock.now += .001
        with self.assertRaises(TimeoutError):
            budget.check()

    def test_new_phase_caps_are_not_original_remaining_time(self):
        for owner, phase, cap in (("admission_lock_binding", None, 120),
                                  ("frozen_evaluation", "frozen_evaluation", 240),
                                  ("analysis_archive", "analysis_archive", 1200),
                                  ("failure_flush_shutdown_reserve", None, 240)):
            with self.subTest(owner=owner), TemporaryDirectory(dir="/private/tmp") as path:
                clock = FakeClock()
                budget = RecoveryBudget(self.proposal, self.contract,
                                        root=Path(path) / "launcher", clock=clock)
                try:
                    budget.enter(owner, phase)
                    clock.now += cap
                    budget.check()
                    clock.now += .001
                    with self.assertRaises(TimeoutError):
                        budget.check()
                finally:
                    budget.close()

    def test_global_deadline_includes_prior_recovery_elapsed_time(self):
        self.clock.now = 2000.
        budget = RecoveryBudget(self.proposal, self.contract, root=self.root,
                                clock=self.clock, started=self.clock.now - 1790)
        self.addCleanup(budget.close)
        self.clock.now += 11
        with self.assertRaises(TimeoutError):
            budget.check()

    def test_budget_is_single_use_and_disk_cap_is_enforced(self):
        budget = self.budget()
        initial_bytes = (self.root / "budget.jsonl").read_bytes()
        with self.assertRaises(FileExistsError):
            RecoveryBudget(self.proposal, self.contract, root=self.root, clock=self.clock)
        self.assertEqual((self.root / "budget.jsonl").read_bytes(), initial_bytes)
        budget.proposal["proposed_budget"]["combined_disk_cap_bytes"] = len(initial_bytes) - 1
        with self.assertRaises(RuntimeError):
            budget.check(storage=True)

    def test_unfinished_new_chunk_is_retained_and_prevents_phase_transition(self):
        budget = self.budget()
        self.evaluation(budget)
        budget.chunk_call("planner_total_model_epochs", 768, 411)
        with self.assertRaises(ValueError):
            budget.finish_chunk("planner_total_model_epochs")
        with self.assertRaises(RuntimeError):
            budget.enter("analysis_archive", "analysis_archive")
        self.assertEqual(budget.job_id, PARTIAL_REMAINING)
        self.assertEqual(budget.counts["planner_total_model_epochs"], 768)
        self.assertEqual(budget.snapshot()["combined_counts"]["planner_total_model_epochs"], 9925632)
        self.assertEqual(budget.chunks, dict(planner_total_model_epochs=dict(reserved=768, dispatched=411)))


if __name__ == "__main__":
    unittest.main()
