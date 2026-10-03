"""Exact one-arm allocation schedule on the existing durable capacity ledger."""

import copy
import os
from pathlib import Path

from src.rl.candidate_pilot_resources import digest
from src.rl.patient_constrained_resources import (
    PHASES, PatientConstrainedBudget, merged_proposal as inherited_proposal,
)
from src.utils.research_clock import shared_monotonic


def merged_proposal(original, inherited, study):
    p = inherited_proposal(original, inherited)
    p["fixed_budget_study"] = copy.deepcopy(study)
    d, b, streams = study["design"], study["budget"], study["streams"]
    p["design"].update(training_seeds=d["training_seeds"], rng_namespace=streams["namespace"],
                       substream_purposes=streams["purposes"])
    p["controllers"] = [{"id": role} for role in d["evaluation_arms"]]
    p["scope"] = dict(study="fixed-budget-capacity-training-v1", resume=False,
                      automatic_retry=False, historical_checkpoint_load=False)
    # The inherited numerical learner contract is unchanged; scheduling is new.
    p["patient_constrained"]["design"].update(evaluation_arms=d["evaluation_arms"],
        training_seeds=d["training_seeds"], namespace=streams["namespace"])
    p["analysis"] = copy.deepcopy(study["readout"])
    p["proposed_budget"] = dict(neural_forward_max_batch=b["forward_max_batch"],
        planner_model_epochs_per_decision=384, flush_every_epochs=8,
        time_seconds=copy.deepcopy(b["seconds"]), rss_cap_bytes=b["rss_bytes"],
        raw_artifact_cap_bytes=b["raw_bytes"], archive_cap_bytes=b["archive_bytes"],
        combined_disk_cap_bytes=b["combined_disk_bytes"], file_cap=b["file_cap"],
        compute_threads=b["compute_threads"])
    return p


def numeric_contract(study):
    b, d = study["budget"], study["design"]
    limits = dict(trajectories=d["total_worlds"], native_steps=b["native_steps"],
        native_constructions=b["native_constructions"], construction_triggered_resets=b["construction_resets"],
        total_native_operations=b["native_operations"], control_steps=d["total_worlds"]*48,
        tail_steps=d["total_worlds"]*16, actor_optimizer_steps=b["actor_steps"],
        critic_optimizer_steps=b["critic_steps"], total_optimizer_steps=b["optimizer_steps"],
        optimizer_example_presentations=b["optimizer_example_presentations"],
        neural_forward_module_calls=b["forwards"], scalar_multiplier_updates=b["scalar_multiplier_updates"],
        planner_total_decisions=b["planner_decisions"], planner_candidate_rollouts=b["planner_candidate_quantile_rollouts"],
        planner_total_model_epochs=b["planner_forecast_epochs"], estimator_receipt_updates=b["filter_receipts"],
        estimator_hypothesis_transitions=b["filter_transitions"], warmup_batches=3*256,
        training_updates=3*12*48, fresh_initializers=3, initial_seals=d["initial_seals"],
        final_seals=d["final_seals"], training_fork_restorations=d["training_forks"],
        learned_evaluation_arm_restorations=d["evaluation_model_restorations"])
    phases = {}
    for row in b["phases"]:
        phase = dict.fromkeys(limits, 0)
        n = row["worlds"]
        phase.update(trajectories=n, native_steps=row["native_steps"], native_constructions=n,
            construction_triggered_resets=n, total_native_operations=row["native_steps"]+2*n,
            control_steps=48*n, tail_steps=16*n, actor_optimizer_steps=row["actor_steps"],
            critic_optimizer_steps=row["critic_steps"], total_optimizer_steps=row["actor_steps"]+row["critic_steps"],
            optimizer_example_presentations=(row["actor_steps"]+row["critic_steps"])*64,
            neural_forward_module_calls=row["forwards"], estimator_receipt_updates=64*n,
            estimator_hypothesis_transitions=64*n*100)
        phases[row["id"]] = phase
    phases[PHASES[0]].update(warmup_batches=768, fresh_initializers=3, initial_seals=3)
    phases[PHASES[1]].update(training_updates=1728, scalar_multiplier_updates=36,
                            training_fork_restorations=3, final_seals=3)
    phases[PHASES[2]].update(planner_total_decisions=1728, planner_candidate_rollouts=82944,
                            planner_total_model_epochs=663552, learned_evaluation_arm_restorations=36)
    if (set(phases) != set(PHASES) or any(type(v) is not int or v < 0 for v in limits.values())
            or any(sum(p[k] for p in phases.values()) != v for k,v in limits.items())
            or (limits["trajectories"], limits["native_steps"], limits["total_optimizer_steps"],
                limits["neural_forward_module_calls"]) != (180, 11520, 6720, 21120)):
        raise ValueError("fixed-budget numerical package arithmetic mismatch")
    return dict(limits=limits, phase_limits=phases)


class FixedBudgetCapacityBudget(PatientConstrainedBudget):
    format = "fixed-budget-capacity-budget-v1"

    def __init__(self, root, proposal, *, clock=shared_monotonic, started=None):
        self.root, self.proposal, self.clock = Path(root), copy.deepcopy(proposal), clock
        self.contract = numeric_contract(proposal["fixed_budget_study"])
        self.limits = self.contract["limits"]
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {k:dict.fromkeys(self.limits, 0) for k in PHASES}
        self.started = self.last_clock = clock() if started is None else started
        self.times = dict.fromkeys(proposal["proposed_budget"]["time_seconds"], 0.)
        self.owner, self.phase, self.failed = "admission_lock_binding", None, False
        self.chunks, self.executed_chunks = {}, dict(planner_total_model_epochs=0, estimator_hypothesis_transitions=0)
        self.previous, self.sequence = "0"*64, 0
        self.job_id = self.job_kind = None
        self.job_last_clock, self.job_times = self.last_clock, {}
        self.job_caps = proposal["fixed_budget_study"]["budget"]["owner_seconds"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root/"budget.jsonl").open("x", encoding="utf8")
        self._append(dict(event="claim", pid=os.getpid(), started=self.started,
                          proposal_sha256=digest(proposal), limits=self.limits))
