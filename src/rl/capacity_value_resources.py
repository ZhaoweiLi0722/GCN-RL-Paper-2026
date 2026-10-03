"""Value-only schedule on the existing nonrefundable capacity accounting ledger."""

import copy
import os
from pathlib import Path

from src.rl.candidate_pilot_resources import digest
from src.rl.patient_constrained_resources import PatientConstrainedBudget
from src.utils.research_clock import shared_monotonic

PHASES = ("initial_value_learning", "continuation_value_learning", "frozen_evaluation")


def worlds(study, phase, block):
    if phase not in ("initial", "continuation", "evaluation") or block not in range(3):
        raise ValueError("unknown value-MPC phase or block")
    n = 12 if phase == "evaluation" else 24
    for index in range(n):
        c, j = index % 3, index // 3
        yield dict(phase=phase, block=block, condition=c, replicate=j,
                   seed=study["streams"][phase+"_base"]+1000*block+100*c+j)


def merged_proposal(original, study):
    numeric_contract(study)
    p = copy.deepcopy(original)
    p["value_mpc_study"] = copy.deepcopy(study)
    p["design"].update(training_seeds=study["design"]["model_seeds"],
        rng_namespace=study["streams"]["namespace"], substream_purposes=study["streams"]["purposes"])
    p["controllers"] = [{"id": role} for role in study["design"]["roles"]]
    p["scope"] = dict(study="capacity-value-mpc-v1", automatic_retry=False, resume=False)
    p["analysis"] = copy.deepcopy(study["readout"])
    b = study["budget"]
    p["proposed_budget"] = dict(neural_forward_max_batch=64, planner_model_epochs_per_decision=384,
        time_seconds=b["seconds"], rss_cap_bytes=b["rss_bytes"], raw_artifact_cap_bytes=b["raw_bytes"],
        archive_cap_bytes=b["archive_bytes"], combined_disk_cap_bytes=b["combined_disk_bytes"],
        file_cap=b["file_cap"], compute_threads=b["compute_threads"])
    return p


def numeric_contract(study):
    d, v = study["design"], study["value"]
    expected = dict(blocks=3, initial_worlds_per_block=24, continuation_worlds_per_block=24,
                    evaluation_worlds_per_condition_block=4, conditions=3, total_worlds=288,
                    evaluation_worlds=144, all_final_seals_before_tests=True)
    if (any(d.get(k) != x for k, x in expected.items()) or len(d["model_seeds"]) != 3
            or d["roles"] != ["plain_mpc", "initial_value_mpc", "updated_value_mpc", "fixed_allocation_reference"]
            or any(v.get(k) != x for k, x in dict(batch_size=64, updates_per_world=32,
                                                  td_horizon=8, gamma=1., evaluation_updates=0).items())):
        raise ValueError("value study schedule differs from the prospective package")
    rows = ((PHASES[0], 72, 72, 0, 2304, 3, 0),
            (PHASES[1], 72, 72, 72, 2304, 0, 3),
            (PHASES[2], 144, 108, 72, 0, 0, 0))
    phases = {}
    for name, n, planning, scoring, updates, initial, final in rows:
        phases[name] = dict(trajectories=n, native_steps=n*64, native_constructions=n,
            construction_triggered_resets=n, total_native_operations=n*66,
            control_steps=n*48, tail_steps=n*16, value_optimizer_steps=updates,
            total_optimizer_steps=updates, optimizer_example_presentations=updates*64,
            neural_forward_module_calls=updates+(72 if updates else 0)+scoring*48,
            planner_total_decisions=planning*48, planner_candidate_rollouts=planning*48*48,
            planner_total_model_epochs=planning*48*384, estimator_receipt_updates=n*64,
            estimator_hypothesis_transitions=n*64*100, initial_seals=initial, final_seals=final)
    limits = {k:sum(p[k] for p in phases.values()) for k in phases[PHASES[0]]}
    b = study["budget"]
    for key, field in (("trajectories","worlds"),("native_steps","native_steps"),("total_native_operations","native_operations"),
                       ("total_optimizer_steps","value_optimizer_steps"),("neural_forward_module_calls","forwards"),
                       ("planner_total_decisions","planner_decisions"),("planner_candidate_rollouts","planner_rollouts"),
                       ("planner_total_model_epochs","planner_epochs"),("estimator_hypothesis_transitions","filter_transitions")):
        if limits[key] != b[field]:
            raise ValueError("value study budget arithmetic mismatch: "+key)
    if (b["actor_optimizer_steps"] != 0 or b["maximum_forward_batch"] != 64
            or sum(x for k,x in b["seconds"].items() if k != "global") != b["seconds"]["global"]):
        raise ValueError("invalid optimizer/forward/time bounds")
    return dict(limits=limits, phase_limits=phases)


class CapacityValueBudget(PatientConstrainedBudget):
    format = "capacity-value-mpc-budget-v1"

    def __init__(self, root, proposal, *, clock=shared_monotonic, started=None):
        self.root, self.proposal, self.clock = Path(root), copy.deepcopy(proposal), clock
        self.contract = numeric_contract(proposal["value_mpc_study"])
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
        self.job_caps = proposal["value_mpc_study"]["budget"]["owner_seconds"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root/"budget.jsonl").open("x", encoding="utf8")
        self._append(dict(event="claim", pid=os.getpid(), started=self.started,
                          proposal_sha256=digest(proposal), limits=self.limits))
