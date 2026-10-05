"""DRAFT remaining-only resource accounting, not scientific authorization.

``remainder(study, old, terminal, storage)`` accepts JSON dictionaries: the
original planner-tail study, child-failure.json's *budget* snapshot, terminal.json,
and storage totals (raw_bytes, archive_bytes, total_bytes, files). It performs no
file reads, model loading, environment calls, fitting, or historical re-audit.

``RecoveryBudget(proposal, contract, root=..., clock=..., started=...)`` writes a
new single-use ledger under root; its parent must contain only recovery output.
The proposal contains planner_tail_study. Optional runtime _recovery_contract
and current_scope must match contract and remaining_scope, respectively. Input
objects are never mutated. The coordinator owns restoration/admission and must
verify the saved epoch-10 state and ten raw rows before any authorized dispatch.

Enter frozen_evaluation with the matching phase, then job(PARTIAL_REMAINING,
"evaluation_h16_world"). Only that owner may debit evaluation work. Its original
time is carried forward; changing names cannot restart the 240-second owner cap.
Enter analysis_archive with the matching phase for diagnostic forward receipts.
Admission and failure reserve have no scientific phase. Reference/fitting phases
have zero counters and cannot be entered. Ledger receipts are not execution.
"""

import copy
import math
import os
from pathlib import Path

from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_planner_tail_resources import PHASES, numeric_contract
from src.rl.patient_constrained_resources import PatientConstrainedBudget
from src.utils.research_clock import shared_monotonic


remaining_scope = "capacity-planner-tail-recovery1"
RECOVERY_SCOPE = remaining_scope
PARTIAL = "evaluation-b4-c2-j3-plain_h16"
PARTIAL_REMAINING = PARTIAL + "-remaining"
PARTIAL_SECONDS = 13.66981799993664
REPLACEMENT = dict(planner_total_decisions=1, planner_candidate_rollouts=48,
                   planner_total_model_epochs=768)
SECONDS = dict(admission_lock_binding=120, frozen_evaluation=240,
               analysis_archive=1200, failure_flush_shutdown_reserve=240,
               **{"global": 1800})
ORIGINAL_SECONDS = dict(admission=600, reference_and_tails=10800,
                        value_fitting=3600, frozen_evaluation=14400,
                        analysis_archive=1200, failure_preservation=1800,
                        **{"global": 32400})
STORAGE_CAPS = dict(raw_bytes=4 * 1024**3, archive_bytes=4 * 1024**3,
                    total_bytes=8 * 1024**3, files=6000)
STORAGE_FIELDS = dict(raw_bytes="raw_artifact_cap_bytes",
                      archive_bytes="archive_cap_bytes",
                      total_bytes="combined_disk_cap_bytes", files="file_cap")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _exact_counts(actual, expected, label):
    _require(isinstance(actual, dict) and actual == expected
             and all(type(v) is int and v >= 0 for v in actual.values()),
             "unexpected original " + label)


def _remaining_counts(original):
    phases = {p: dict.fromkeys(original["limits"], 0) for p in PHASES}
    phases["frozen_evaluation"].update(
        native_steps=54, total_native_operations=54, control_steps=38, tail_steps=16,
        planner_total_decisions=38, planner_candidate_rollouts=38 * 48,
        planner_total_model_epochs=38 * 48 * 16, estimator_receipt_updates=54,
        estimator_hypothesis_transitions=5400)
    phases["analysis_archive"]["neural_forward_module_calls"] = 960
    limits = {k: sum(p[k] for p in phases.values()) for k in original["limits"]}
    return limits, phases


def _remainder(study, old, terminal, storage):
    original = numeric_contract(study)
    b = study["budget"]
    _require(b["seconds"] == ORIGINAL_SECONDS
             and b["owner_seconds"]["evaluation_h16_world"] == 240
             and b["raw_bytes"] == STORAGE_CAPS["raw_bytes"]
             and b["archive_bytes"] == STORAGE_CAPS["archive_bytes"]
             and b["combined_disk_bytes"] == STORAGE_CAPS["total_bytes"]
             and b["file_cap"] == STORAGE_CAPS["files"],
             "unexpected original time/storage contract")
    limits, phases = _remaining_counts(original)
    expected_phases = copy.deepcopy(original["phase_limits"])
    for p, counts in expected_phases.items():
        for k in counts:
            counts[k] -= phases[p][k]
            if p == "frozen_evaluation":
                counts[k] += REPLACEMENT.get(k, 0)
    expected = {k: sum(p[k] for p in expected_phases.values()) for k in limits}
    _exact_counts(old["counts"], expected, "counters")
    _require(set(old["phases"]) == set(PHASES), "unexpected original phases")
    for p in PHASES:
        _exact_counts(old["phases"][p], expected_phases[p], "phase counters: " + p)
    _require(old["format"] == "capacity-planner-tail-budget-v1"
             and old["failed"] is True and old["owner"] == "frozen_evaluation"
             and old["phase"] == "frozen_evaluation" and old["job_id"] == PARTIAL
             and old["job_kind"] == "evaluation_h16_world",
             "unexpected original failure boundary")
    _require(set(old["pending_chunks"]) == {"planner_total_model_epochs"},
             "unexpected original pending chunks")
    _exact_counts(old["pending_chunks"]["planner_total_model_epochs"],
                  dict(reserved=768, dispatched=411), "interrupted reservation")
    _exact_counts(old["verified_completed_chunk_calls"], dict(
        planner_total_model_epochs=expected["planner_total_model_epochs"] - 768,
        training_tail_model_epochs=374400, estimator_hypothesis_transitions=3066600),
        "completed chunk calls")
    _require(terminal["status"] == "failed" and terminal["automatic_retry"] is False
             and terminal["scientific_completion_verified"] is False
             and type(terminal["exit_code"]) is int and terminal["exit_code"] == 1
             and type(terminal["child_exit_code"]) is int and terminal["child_exit_code"] == 1,
             "unexpected original terminal failure")
    old_seconds = copy.deepcopy(ORIGINAL_SECONDS)
    old_seconds["admission_lock_binding"] = old_seconds.pop("admission")
    old_seconds["failure_flush_shutdown_reserve"] = old_seconds.pop("failure_preservation")
    times = old["times"]
    _require(set(times) == set(old_seconds)
             and all(_finite_nonnegative(v) for v in times.values())
             and _finite_nonnegative(old["elapsed"])
             and _finite_nonnegative(terminal["elapsed"]), "invalid original times")
    _require(times["frozen_evaluation"] > old_seconds["frozen_evaluation"]
             and times["global"] == times["analysis_archive"]
             == times["failure_flush_shutdown_reserve"] == 0
             and all(times[k] <= cap for k, cap in old_seconds.items()
                     if k not in ("global", "frozen_evaluation"))
             and math.isclose(sum(times.values()), old["elapsed"], rel_tol=0, abs_tol=1e-6)
             and old["elapsed"] <= terminal["elapsed"] < old_seconds["global"],
             "original failure must exhaust evaluation phase, not global time")
    partial_seconds = old["job_times"][PARTIAL]
    _require(_finite_nonnegative(partial_seconds)
             and math.isclose(partial_seconds, PARTIAL_SECONDS, rel_tol=0, abs_tol=1e-9),
             "unexpected original partial-world time")
    _require(all(type(storage[k]) is int and storage[k] >= 0 for k in STORAGE_CAPS)
             and storage["total_bytes"] == storage["raw_bytes"] + storage["archive_bytes"],
             "invalid original storage totals")
    remaining_storage = {k: v - storage[k] for k, v in STORAGE_CAPS.items()}
    _require(all(v > 0 for v in remaining_storage.values()),
             "no remaining combined storage allowance")

    # Keep only the JSON evidence needed to revalidate this contract at admission.
    evidence_keys = ("format", "counts", "phases", "failed", "owner", "phase",
                     "job_id", "job_kind", "times", "elapsed", "pending_chunks",
                     "verified_completed_chunk_calls")
    evidence = {k: copy.deepcopy(old[k]) for k in evidence_keys}
    evidence["job_times"] = {PARTIAL: partial_seconds}
    return dict(
        format="capacity-planner-tail-recovery1-contract-v1", status="DRAFT",
        scientific_execution_authorized=False, training_permitted=False,
        remaining_scope=remaining_scope, limits=limits, phase_limits=phases,
        seconds=copy.deepcopy(SECONDS), old_counts=copy.deepcopy(old["counts"]),
        original_limits=copy.deepcopy(original["limits"]),
        combined_limits={k: old["counts"][k] + v for k, v in limits.items()},
        prior_elapsed_seconds=terminal["elapsed"], partial_job_seconds=partial_seconds,
        retained_interrupted_reservation=copy.deepcopy(old["pending_chunks"]),
        replacement_planning_charges=copy.deepcopy(REPLACEMENT),
        prior_storage={k: storage[k] for k in STORAGE_CAPS},
        remaining_storage=remaining_storage, original_budget=evidence,
        original_terminal=copy.deepcopy(terminal),
        boundary=dict(job_id=PARTIAL, epoch=10, saved_rows=10,
                      completed_reference=120, completed_evaluation=359))


def remainder(study, old, terminal, storage):
    """Validate the fixed terminal JSON boundary and return a detached draft."""
    try:
        return _remainder(study, old, terminal, storage)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("malformed original recovery evidence") from exc


class RecoveryBudget(PatientConstrainedBudget):
    """Durable *new* receipts only; old charges/reservations are never refunded."""

    format = "capacity-planner-tail-recovery1-budget-v1"

    def __init__(self, proposal, contract, *, root, clock=shared_monotonic, started=None):
        try:
            verified = remainder(proposal["planner_tail_study"], contract["original_budget"],
                                 contract["original_terminal"], contract["prior_storage"])
            _require(digest(contract) == digest(verified), "modified recovery contract")
            _require(proposal.get("current_scope", remaining_scope) == remaining_scope,
                     "unexpected runtime current_scope")
            _require(digest(proposal.get("_recovery_contract", contract)) == digest(contract),
                     "runtime _recovery_contract mismatch")
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("malformed recovery proposal/contract") from exc
        self.root, self.proposal, self.clock = Path(root), copy.deepcopy(proposal), clock
        self.contract, self.limits = verified, copy.deepcopy(verified["limits"])
        caps = self.proposal["proposed_budget"]
        caps["time_seconds"] = copy.deepcopy(verified["seconds"])
        for k, field in STORAGE_FIELDS.items():
            caps[field] = verified["remaining_storage"][k]
        b = proposal["planner_tail_study"]["budget"]
        caps.update(rss_cap_bytes=b["rss_bytes"], compute_threads=b["compute_threads"],
                    neural_forward_max_batch=64, planner_model_epochs_per_decision=768)
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {p: dict.fromkeys(self.limits, 0) for p in PHASES}
        self.started = self.last_clock = clock() if started is None else started
        _require(_finite_nonnegative(self.started), "invalid recovery start clock")
        self.times = dict.fromkeys(verified["seconds"], 0.)
        self.owner, self.phase, self.failed = "admission_lock_binding", None, False
        self.chunks, self.executed_chunks = {}, dict(
            planner_total_model_epochs=0, training_tail_model_epochs=0,
            estimator_hypothesis_transitions=0)
        self.previous, self.sequence = "0" * 64, 0
        self.job_id = self.job_kind = None
        self.job_last_clock = self.last_clock
        self.job_times = {PARTIAL_REMAINING: verified["partial_job_seconds"]}
        self.job_caps = {"evaluation_h16_world": 240}
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root / "budget.jsonl").open("x", encoding="utf8")
        try:
            self._append(dict(event="claim", pid=os.getpid(), started=self.started,
                              proposal_sha256=digest(proposal), contract_sha256=digest(verified),
                              remaining_scope=remaining_scope, status="DRAFT",
                              scientific_execution_authorized=False, limits=self.limits,
                              original_counts=verified["old_counts"],
                              retained_reservation=verified["retained_interrupted_reservation"],
                              partial_job_seconds=verified["partial_job_seconds"]))
        except BaseException:
            self.close()
            raise

    def enter(self, owner, phase=None):
        allowed = {("admission_lock_binding", None),
                   ("failure_flush_shutdown_reserve", None),
                   ("frozen_evaluation", "frozen_evaluation"),
                   ("analysis_archive", "analysis_archive")}
        _require((owner, phase) in allowed, "no training or cross-owner phase permitted")
        # The inherited enter clears job ownership before checking pending chunks.
        if self.chunks:
            raise RuntimeError("unfinished query reservation at phase boundary")
        super().enter(owner, phase)

    def job(self, identifier, kind):
        _require(self.phase == self.owner == "frozen_evaluation"
                 and identifier == PARTIAL_REMAINING and kind == "evaluation_h16_world",
                 "only the original partial-world owner is permitted")
        super().job(identifier, kind)

    def debit(self, charges):
        if self.phase == "frozen_evaluation" and self.job_id != PARTIAL_REMAINING:
            self.failed = True
            raise RuntimeError("partial-world owner required before evaluation receipts")
        super().debit(charges)

    def snapshot(self):
        return {**super().snapshot(), "remaining_scope": remaining_scope,
                "scientific_execution_authorized": False,
                "old_counts": copy.deepcopy(self.contract["old_counts"]),
                "combined_counts": {k: self.contract["old_counts"][k] + v
                                    for k, v in self.counts.items()},
                "retained_interrupted_reservation": copy.deepcopy(
                    self.contract["retained_interrupted_reservation"])}
