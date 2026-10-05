"""Separate real-world and nested-public-model counters on the durable ledger."""

import copy
import os
from pathlib import Path

from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_policy_tail_design import numeric_contract as design_contract
from src.rl.patient_constrained_resources import PatientConstrainedBudget
from src.utils.research_clock import shared_monotonic


SCOPE = "capacity-policy-tail-v1"
PHASES = ("reference_and_tails", "value_fitting", "frozen_evaluation", "analysis_archive")


def numeric_contract(study):
    d = design_contract(study)["limits"]
    limits = dict(trajectories=480, native_steps=30720, native_constructions=480,
        construction_triggered_resets=480, total_native_operations=31680,
        control_steps=23040, tail_steps=7680, value_optimizer_steps=11520,
        total_optimizer_steps=11520, optimizer_example_presentations=737280,
        neural_forward_module_calls=d["forward_calls"], planner_total_decisions=23040,
        planner_candidate_rollouts=1105920, planner_total_model_epochs=d["native_planning_epochs"],
        branch_planner_decisions=d["continuation_planning_decisions"],
        branch_candidate_rollouts=d["continuation_planning_decisions"] * 48,
        branch_planner_model_epochs=d["continuation_planning_epochs"],
        prefix_model_epochs=d["prefix_epochs"], training_tail_model_epochs=d["paired_tail_epochs"],
        forecast_root_constructions=720, forecast_tail_clones=1440,
        branch_filter_transitions=d["branch_filter_transitions"],
        estimator_receipt_updates=30720, estimator_hypothesis_transitions=3072000,
        final_seals=15)
    phases = {p: dict.fromkeys(limits, 0) for p in PHASES}
    for phase, n, predictions in ((PHASES[0], 120, 2211840), (PHASES[2], 360, 7741440)):
        phases[phase].update(trajectories=n, native_steps=n * 64, native_constructions=n,
            construction_triggered_resets=n, total_native_operations=n * 66,
            control_steps=n * 48, tail_steps=n * 16, planner_total_decisions=n * 48,
            planner_candidate_rollouts=n * 48 * 48, planner_total_model_epochs=predictions,
            estimator_receipt_updates=n * 64, estimator_hypothesis_transitions=n * 6400)
    for key in ("branch_planner_decisions", "branch_candidate_rollouts", "branch_planner_model_epochs",
                "prefix_model_epochs", "training_tail_model_epochs", "forecast_root_constructions",
                "forecast_tail_clones", "branch_filter_transitions"):
        phases[PHASES[0]][key] = limits[key]
    phases[PHASES[0]]["neural_forward_module_calls"] = d["continuation_value_forwards"]
    phases[PHASES[1]].update(value_optimizer_steps=11520, total_optimizer_steps=11520,
        optimizer_example_presentations=737280, final_seals=15,
        neural_forward_module_calls=11520 + d["training_bootstrap_forwards"])
    phases[PHASES[2]]["neural_forward_module_calls"] = 11520
    if any(sum(p[k] for p in phases.values()) != v for k, v in limits.items()):
        raise ValueError("nested-work phase counters do not reconcile")
    return dict(limits=limits, phase_limits=phases)


def merged_proposal(original, study):
    numeric_contract(study)
    p = copy.deepcopy(original)
    p["policy_tail_study"] = copy.deepcopy(study)
    p["design"].update(blocks=5, training_seeds=study["streams"]["sampler_seeds"],
        rng_namespace=study["streams"]["namespace"], substream_purposes=study["streams"]["purposes"])
    p["controllers"] = [dict(id=r) for r in study["design"]["eval_roles"]]
    p["scope"] = dict(study=SCOPE, automatic_retry=False, resume=False)
    b = study["budget"]
    seconds = copy.deepcopy(b["seconds"])
    seconds["admission_lock_binding"] = seconds.pop("admission")
    seconds["failure_flush_shutdown_reserve"] = seconds.pop("failure_preservation")
    p["proposed_budget"] = dict(neural_forward_max_batch=64, planner_model_epochs_per_decision=384,
        time_seconds=seconds, rss_cap_bytes=b["rss_bytes"], raw_artifact_cap_bytes=b["raw_bytes"],
        archive_cap_bytes=b["archive_bytes"], combined_disk_cap_bytes=b["combined_disk_bytes"],
        file_cap=b["file_cap"], compute_threads=b["compute_threads"])
    return p


class CapacityPolicyTailBudget(PatientConstrainedBudget):
    format = "capacity-policy-tail-budget-v1"

    def __init__(self, root, proposal, *, clock=shared_monotonic, started=None):
        self.root, self.proposal, self.clock = Path(root), copy.deepcopy(proposal), clock
        self.contract = numeric_contract(proposal["policy_tail_study"])
        self.limits = self.contract["limits"]
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {p: dict.fromkeys(self.limits, 0) for p in PHASES}
        self.started = self.last_clock = clock() if started is None else started
        self.times = dict.fromkeys(proposal["proposed_budget"]["time_seconds"], 0.)
        self.owner, self.phase, self.failed = "admission_lock_binding", None, False
        self.chunks = {}
        self.executed_chunks = dict.fromkeys(("planner_total_model_epochs", "branch_planner_model_epochs",
            "estimator_hypothesis_transitions", "branch_filter_transitions"), 0)
        self.previous, self.sequence = "0" * 64, 0
        self.job_id = self.job_kind = None
        self.job_last_clock, self.job_times = self.last_clock, {}
        self.job_caps = proposal["policy_tail_study"]["budget"]["owner_seconds"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root / "budget.jsonl").open("x", encoding="utf8")
        self._append(dict(event="claim", pid=os.getpid(), started=self.started,
                          proposal_sha256=digest(proposal), limits=self.limits))
