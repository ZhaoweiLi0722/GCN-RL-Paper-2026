"""Prospective bounded counts; reuse the durable capacity ledger, not its scope."""

import copy
import os
from pathlib import Path

from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_pilot_resources import CapacityPilotBudget
from src.utils.research_clock import shared_monotonic


PHASES = ("reference_and_warmup", "two_arm_training", "four_arm_evaluation")
ARMS = ("constrained", "cost_only")


def merged_proposal(original, study):
    p = copy.deepcopy(original)
    p["patient_constrained"] = copy.deepcopy(study)
    p["design"].update(training_seeds=study["design"]["training_seeds"],
                       rng_namespace=study["design"]["namespace"],
                       substream_purposes=study["design"]["purposes"])
    p["controllers"] = [{"id": role} for role in study["design"]["evaluation_arms"]]
    p["scope"] = dict(study="patient-constrained-training-only-v1", resume=False,
                      automatic_retry=False, historical_checkpoint_load=False)
    p["analysis"] = copy.deepcopy(study["readout"])
    b = study["budget"]
    p["proposed_budget"] = dict(neural_forward_max_batch=64, planner_model_epochs_per_decision=384,
        flush_every_epochs=8, time_seconds=dict(admission_lock_binding=b["seconds"]["admission"],
            reference_and_warmup=b["seconds"]["reference_and_warmup"],
            two_arm_training=b["seconds"]["training"], four_arm_evaluation=b["seconds"]["evaluation"],
            analysis_archive_verification=b["seconds"]["readout_archive"],
            failure_flush_shutdown_reserve=b["seconds"]["failure_flush"], **{"global": b["seconds"]["global"]}),
        rss_cap_bytes=b["rss_bytes"], raw_artifact_cap_bytes=b["raw_bytes"],
        archive_cap_bytes=b["archive_bytes"], combined_disk_cap_bytes=b["combined_disk_bytes"],
        file_cap=b["raw_file_cap"], compute_threads=b["compute_threads"])
    return p


def numeric_contract(study):
    b, d = study["budget"], study["design"]
    limits = dict(trajectories=d["total_trajectories"], native_steps=b["native_steps"],
        native_constructions=b["native_constructions"], construction_triggered_resets=b["construction_resets"],
        total_native_operations=b["native_operations"], control_steps=d["total_trajectories"]*48,
        tail_steps=d["total_trajectories"]*16, actor_optimizer_steps=b["actor_optimizer_steps"],
        critic_optimizer_steps=b["critic_optimizer_steps"], total_optimizer_steps=b["total_optimizer_steps"],
        optimizer_example_presentations=b["optimizer_example_presentations"],
        neural_forward_module_calls=b["neural_forwards"], scalar_multiplier_updates=b["scalar_multiplier_updates"],
        planner_total_decisions=b["planner_decisions"], planner_candidate_rollouts=b["planner_candidate_quantile_rollouts"],
        planner_total_model_epochs=b["planner_forecast_epochs"], estimator_receipt_updates=b["filter_receipt_updates"],
        estimator_hypothesis_transitions=b["filter_hypothesis_transitions"], warmup_batches=3*256,
        training_updates=3*2*12*48, fresh_initializers=3, initial_seals=3, final_seals=6,
        training_fork_restorations=d["training_fork_restorations"],
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
    phases[PHASES[1]].update(training_updates=3456, scalar_multiplier_updates=36,
                            training_fork_restorations=6, final_seals=6)
    phases[PHASES[2]].update(planner_total_decisions=1728, planner_candidate_rollouts=82944,
                            planner_total_model_epochs=663552, learned_evaluation_arm_restorations=72)
    if (set(phases) != set(PHASES) or any(type(v) is not int or v < 0 for v in limits.values())
            or any(sum(p[k] for p in phases.values()) != v for k, v in limits.items())
            or (limits["native_steps"], limits["total_optimizer_steps"], limits["neural_forward_module_calls"])
            != (16128, 11904, 38400)):
        raise ValueError("patient-constrained approved arithmetic mismatch")
    return dict(limits=limits, phase_limits=phases)


class PatientConstrainedBudget(CapacityPilotBudget):
    format = "patient-constrained-budget-v1"

    def __init__(self, root, proposal, *, clock=shared_monotonic, started=None):
        self.root, self.proposal, self.clock = Path(root), copy.deepcopy(proposal), clock
        self.contract = numeric_contract(proposal["patient_constrained"])
        self.limits = self.contract["limits"]
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {k: dict.fromkeys(self.limits, 0) for k in PHASES}
        self.started = self.last_clock = clock() if started is None else started
        self.times = dict.fromkeys(proposal["proposed_budget"]["time_seconds"], 0.)
        self.owner, self.phase, self.failed = "admission_lock_binding", None, False
        self.chunks, self.executed_chunks = {}, dict(planner_total_model_epochs=0, estimator_hypothesis_transitions=0)
        self.previous, self.sequence = "0"*64, 0
        self.job_id = self.job_kind = None
        self.job_last_clock, self.job_times = self.last_clock, {}
        self.job_caps = proposal["patient_constrained"]["budget"]["owner_seconds"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root/"budget.jsonl").open("x", encoding="utf8")
        self._append(dict(event="claim", pid=os.getpid(), started=self.started,
                          proposal_sha256=digest(proposal), limits=self.limits))

    def check(self, *, storage=False):
        now = super().check(storage=storage)
        if self.job_id is not None:
            self.job_times[self.job_id] += now-self.job_last_clock
            if self.job_times[self.job_id] > self.job_caps[self.job_kind]:
                self.failed = True
                raise TimeoutError("declared per-owner wall cap exhausted: "+self.job_id)
        self.job_last_clock = now
        return now

    def enter(self, owner, phase=None):
        self.check(storage=True)
        self.job_id = self.job_kind = None
        super().enter(owner, phase)

    def job(self, identifier, kind):
        now = self.check()
        if kind not in self.job_caps or not identifier:
            raise ValueError("unknown bounded owner")
        self.job_id, self.job_kind = identifier, kind
        self.job_times.setdefault(identifier, 0.)
        self.job_last_clock = now
        self._append(dict(event="job_enter", job=identifier, kind=kind, clock=now,
                          consumed=self.job_times[identifier], cap=self.job_caps[kind]))

    def snapshot(self):
        return {**super().snapshot(), "job_id": self.job_id, "job_kind": self.job_kind,
                "job_times": copy.deepcopy(self.job_times)}
