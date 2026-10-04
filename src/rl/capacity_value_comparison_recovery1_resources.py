"""Remaining-only accounting; old charges and interrupted reservations persist."""

import copy
import math
import os
from pathlib import Path

from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_value_comparison_resources import numeric_contract, PHASES
from src.rl.patient_constrained_resources import PatientConstrainedBudget
from src.utils.research_clock import shared_monotonic

RECOVERY_SCOPE = "capacity-value-comparison-recovery1"
PARTIAL = "continuation-b0-c2-j7-flat_value_mpc"
REPLACEMENT = dict(planner_total_decisions=1, planner_candidate_rollouts=48,
                   planner_total_model_epochs=384)


def remainder(study, old, terminal, storage):
    original = numeric_contract(study)
    if (old["counts"]["native_steps"] != 4581 or old["counts"]["total_optimizer_steps"] != 3040
            or old["counts"]["trajectories"] != 72 or old["counts"]["initial_seals"] != 2
            or old["counts"]["final_seals"] != 0 or terminal["status"] != "failed"
            or old["pending_chunks"] != {"planner_total_model_epochs": dict(reserved=384, dispatched=13)}):
        raise ValueError("unexpected original failure boundary")
    phases = {p:{k:v-old["phases"][p][k] for k,v in caps.items()}
              for p,caps in original["phase_limits"].items()}
    for k,v in REPLACEMENT.items():
        phases[PHASES[1]][k] += v
    limits = {k:sum(p[k] for p in phases.values()) for k in original["limits"]}
    if any(v < 0 for p in phases.values() for v in p.values()):
        raise ValueError("historical consumption exceeds original contract")
    caps = study["budget"]["seconds"]
    seconds = {k:math.floor(v-(terminal["elapsed"] if k == "global" else old["times"][k]))
               for k,v in caps.items()}
    if min(seconds.values()) <= 0:
        raise ValueError("no remaining phase/time allowance")
    return dict(limits=limits, phase_limits=phases, seconds=seconds,
                prior_elapsed_seconds=terminal["elapsed"], old_counts=copy.deepcopy(old["counts"]),
                retained_interrupted_reservation=copy.deepcopy(old["pending_chunks"]),
                replacement_planning_charges=REPLACEMENT,
                partial_job_seconds=old["job_times"][PARTIAL], prior_storage=storage)


class CapacityValueComparisonRecovery1Budget(PatientConstrainedBudget):
    format = "capacity-value-comparison-recovery1-budget-v1"

    def __init__(self, root, proposal, contract, *, clock=shared_monotonic, started=None):
        self.root, self.proposal, self.clock = Path(root), copy.deepcopy(proposal), clock
        self.contract, self.limits = copy.deepcopy(contract), copy.deepcopy(contract["limits"])
        caps = self.proposal["proposed_budget"]
        caps["time_seconds"] = copy.deepcopy(contract["seconds"])
        prior = contract["prior_storage"]
        caps["raw_artifact_cap_bytes"] -= prior["raw_bytes"]
        caps["archive_cap_bytes"] -= prior["archive_bytes"]
        caps["combined_disk_cap_bytes"] -= prior["total_bytes"]
        caps["file_cap"] -= prior["files"]
        if min(caps[k] for k in ("raw_artifact_cap_bytes", "archive_cap_bytes", "combined_disk_cap_bytes", "file_cap")) <= 0:
            raise ValueError("no remaining combined storage allowance")
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {p:dict.fromkeys(self.limits, 0) for p in PHASES}
        self.started = self.last_clock = clock() if started is None else started
        self.times = dict.fromkeys(contract["seconds"], 0.)
        self.owner, self.phase, self.failed = "admission_lock_binding", None, False
        self.chunks, self.executed_chunks = {}, dict(planner_total_model_epochs=0, estimator_hypothesis_transitions=0)
        self.previous, self.sequence = "0"*64, 0
        self.job_id = self.job_kind = None
        self.job_last_clock = self.last_clock
        self.job_times = {PARTIAL+"-remaining": contract["partial_job_seconds"]}
        self.job_caps = proposal["value_comparison_study"]["budget"]["owner_seconds"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root/"budget.jsonl").open("x", encoding="utf8")
        self._append(dict(event="claim", pid=os.getpid(), started=self.started,
                          proposal_sha256=digest(proposal), limits=self.limits,
                          original_counts=contract["old_counts"], retained_reservation=contract["retained_interrupted_reservation"]))
