"""Nonrefundable native-branch ledger, separate from forecast-model queries."""

import copy
import os
from pathlib import Path

from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_native_tail_design import counts, numeric_contract as design_contract, forecast_compatibility_study
from src.rl.capacity_policy_tail_resources import numeric_contract as forecast_contract, merged_proposal as forecast_proposal, PHASES
from src.rl.patient_constrained_resources import PatientConstrainedBudget
from src.utils.research_clock import shared_monotonic


SCOPE = "capacity-native-tail-v1"


def numeric_contract(study):
    design_contract(study)
    contract = forecast_contract(forecast_compatibility_study(study))
    limits, phases, d = contract["limits"], contract["phase_limits"], counts()
    additions = dict(native_branch_clones=240, native_branch_steps=9720,
        native_branch_control_steps=5880, native_branch_settlement_steps=3840,
        native_branch_planning_decisions=4100, native_branch_candidate_rollouts=196800,
        native_branch_model_epochs=1574400, native_branch_filter_transitions=972000)
    limits.update(additions)
    for phase in phases.values():
        phase.update(dict.fromkeys(additions, 0))
    phases[PHASES[0]].update(additions)
    limits["neural_forward_module_calls"] = d["forward_calls"]
    phases[PHASES[0]]["neural_forward_module_calls"] += 4100
    phases[PHASES[1]]["neural_forward_module_calls"] = 11520 + d["training_bootstrap_forwards"]
    if any(sum(p[k] for p in phases.values()) != v for k, v in limits.items()):
        raise ValueError("native phase counters do not reconcile")
    return contract


def merged_proposal(original, study):
    numeric_contract(study)
    p = forecast_proposal(original, forecast_compatibility_study(study))
    p.pop("policy_tail_study")
    p["native_tail_study"] = copy.deepcopy(study)
    p["controllers"] = [dict(id=r) for r in study["design"]["eval_roles"]]
    p["scope"]["study"] = SCOPE
    return p


class CapacityNativeTailBudget(PatientConstrainedBudget):
    format = "capacity-native-tail-budget-v1"

    def __init__(self, root, proposal, *, clock=shared_monotonic, started=None):
        self.root, self.proposal, self.clock = Path(root), copy.deepcopy(proposal), clock
        self.contract = numeric_contract(proposal["native_tail_study"])
        self.limits = self.contract["limits"]
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {p: dict.fromkeys(self.limits, 0) for p in PHASES}
        self.started = self.last_clock = clock() if started is None else started
        self.times = dict.fromkeys(proposal["proposed_budget"]["time_seconds"], 0.)
        self.owner, self.phase, self.failed = "admission_lock_binding", None, False
        self.chunks = {}
        self.executed_chunks = dict.fromkeys(("planner_total_model_epochs", "branch_planner_model_epochs",
            "estimator_hypothesis_transitions", "branch_filter_transitions",
            "native_branch_model_epochs", "native_branch_filter_transitions"), 0)
        self.previous, self.sequence = "0" * 64, 0
        self.job_id = self.job_kind = None
        self.job_last_clock, self.job_times = self.last_clock, {}
        self.job_caps = proposal["native_tail_study"]["budget"]["owner_seconds"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root / "budget.jsonl").open("x", encoding="utf8")
        self._append(dict(event="claim", pid=os.getpid(), started=self.started,
                          proposal_sha256=digest(proposal), limits=self.limits))
