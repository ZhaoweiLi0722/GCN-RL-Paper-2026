"""Prospective graph/flat value comparison with shared initial data, not calls."""

import copy
import os
from pathlib import Path

from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_value_resources import PHASES
from src.rl.patient_constrained_resources import PatientConstrainedBudget
from src.utils.research_clock import shared_monotonic

ARCHITECTURES = ("graph", "flat")
ROLES = ("plain_mpc", "graph_value_mpc", "flat_value_mpc", "fixed_allocation_reference")
SCOPE = "capacity-value-comparison-v1"


def worlds(study, phase, block):
    if phase not in ("initial", "continuation", "evaluation") or type(block) is not int or block not in range(5):
        raise ValueError("unknown comparison phase or block")
    for index in range(12 if phase == "evaluation" else 24):
        c, j = index % 3, index // 3
        yield dict(phase=phase, block=block, condition=c, replicate=j,
                   seed=study["streams"][phase+"_base"]+1000*block+100*c+j)


def numeric_contract(study):
    d, v, b = study["design"], study["value"], study["budget"]
    expected = dict(blocks=5, initial_worlds_per_block=24, shared_initial_data=True,
        continuation_worlds_per_architecture_block=24, evaluation_worlds_per_condition_block=4,
        conditions=3, total_worlds=600, evaluation_worlds=240,
        all_final_seals_before_tests=True, selection="final_only")
    if (study.get("schema") != "capacity-value-comparison-study-v1"
            or any(d.get(k) != x for k, x in expected.items())
            or d["architectures"] != list(ARCHITECTURES) or d["roles"] != list(ROLES)
            or set(d["model_seeds"]) != set(ARCHITECTURES)
            or any(len(d["model_seeds"][a]) != 5 or any(type(s) is not int for s in d["model_seeds"][a]) for a in ARCHITECTURES)
            or any(v.get(k) != x for k,x in dict(width=32, flat_hidden_sizes=[21,23],
                max_parameter_gap_fraction=.01, lr=.0003, batch_size=64, updates_per_world=32,
                gradient_norm_cap=5., cost_scale=1000000., td_horizon=8, gamma=1.,
                zero_residual_head=True, evaluation_updates=0).items())):
        raise ValueError("comparison differs from prospective package")
    # Shared initialization makes ONE native trajectory and TWO fits per world.
    schedule = ((PHASES[0],120,120,0,240,10,0),
                (PHASES[1],240,240,240,240,0,10),
                (PHASES[2],240,180,120,0,0,0))
    phases = {}
    for name,n,planning,scoring,fits,initial,final in schedule:
        updates = fits*32
        phases[name] = dict(trajectories=n, native_steps=n*64, native_constructions=n,
            construction_triggered_resets=n, total_native_operations=n*66,
            control_steps=n*48, tail_steps=n*16, value_optimizer_steps=updates,
            total_optimizer_steps=updates, optimizer_example_presentations=updates*64,
            neural_forward_module_calls=updates+fits+scoring*48,
            planner_total_decisions=planning*48, planner_candidate_rollouts=planning*48*48,
            planner_total_model_epochs=planning*48*384, estimator_receipt_updates=n*64,
            estimator_hypothesis_transitions=n*64*100, initial_seals=initial, final_seals=final)
    limits = {k:sum(p[k] for p in phases.values()) for k in phases[PHASES[0]]}
    for key,field in (("trajectories","worlds"),("native_steps","native_steps"),
            ("total_native_operations","native_operations"),("total_optimizer_steps","value_optimizer_steps"),
            ("neural_forward_module_calls","forwards"),("planner_total_decisions","planner_decisions"),
            ("planner_candidate_rollouts","planner_rollouts"),("planner_total_model_epochs","planner_epochs"),
            ("estimator_hypothesis_transitions","filter_transitions")):
        if limits[key] != b[field]:
            raise ValueError("comparison budget arithmetic mismatch: "+key)
    if (b["actor_optimizer_steps"] != 0 or b["maximum_forward_batch"] != 64
            or b["attempts"] != 1 or b["automatic_retry"] is not False
            or b["workers"] != 1 or b["compute_threads"] != 4
            or sum(x for k,x in b["seconds"].items() if k != "global") != b["seconds"]["global"]):
        raise ValueError("invalid comparison execution bounds")
    return dict(limits=limits, phase_limits=phases)


def merged_proposal(original, study):
    numeric_contract(study)
    p = copy.deepcopy(original)
    p["value_comparison_study"] = copy.deepcopy(study)
    p["design"].update(blocks=5, training_seeds=study["design"]["model_seeds"]["graph"],
        rng_namespace=study["streams"]["namespace"], substream_purposes=study["streams"]["purposes"])
    p["controllers"] = [{"id": r} for r in ROLES]
    p["scope"] = dict(study=SCOPE, automatic_retry=False, resume=False)
    p["analysis"] = copy.deepcopy(study["readout"])
    b = study["budget"]
    p["proposed_budget"] = dict(neural_forward_max_batch=64, planner_model_epochs_per_decision=384,
        time_seconds=b["seconds"], rss_cap_bytes=b["rss_bytes"], raw_artifact_cap_bytes=b["raw_bytes"],
        archive_cap_bytes=b["archive_bytes"], combined_disk_cap_bytes=b["combined_disk_bytes"],
        file_cap=b["file_cap"], compute_threads=b["compute_threads"])
    return p


class CapacityValueComparisonBudget(PatientConstrainedBudget):
    format = "capacity-value-comparison-budget-v1"

    def __init__(self, root, proposal, *, clock=shared_monotonic, started=None):
        self.root, self.proposal, self.clock = Path(root), copy.deepcopy(proposal), clock
        self.contract = numeric_contract(proposal["value_comparison_study"])
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
        self.job_caps = proposal["value_comparison_study"]["budget"]["owner_seconds"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root/"budget.jsonl").open("x", encoding="utf8")
        self._append(dict(event="claim", pid=os.getpid(), started=self.started,
                          proposal_sha256=digest(proposal), limits=self.limits))
