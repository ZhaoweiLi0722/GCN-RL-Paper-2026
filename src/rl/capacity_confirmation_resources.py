"""Fresh-study ledger using the established nonrefundable watchdog contract."""

import copy
import os
from pathlib import Path

from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_confirmation_design import PHASES, numeric_contract
from src.rl.patient_constrained_resources import PatientConstrainedBudget
from src.utils.research_clock import shared_monotonic


class CapacityConfirmationBudget(PatientConstrainedBudget):
    format = "capacity-confirmation-budget-v1"

    def __init__(self, root, proposal, *, clock=shared_monotonic, started=None):
        self.root, self.proposal, self.clock = Path(root), copy.deepcopy(proposal), clock
        self.contract = numeric_contract(proposal["confirmation_study"])
        self.limits = self.contract["limits"]
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {p: dict.fromkeys(self.limits, 0) for p in PHASES}
        self.started = self.last_clock = clock() if started is None else started
        self.times = dict.fromkeys(proposal["proposed_budget"]["time_seconds"], 0.)
        self.owner, self.phase, self.failed = "admission_lock_binding", None, False
        self.chunks = {}
        self.executed_chunks = dict.fromkeys(("planner_total_model_epochs", "estimator_hypothesis_transitions",
            "native_branch_model_epochs", "native_branch_filter_transitions"), 0)
        self.previous, self.sequence = "0" * 64, 0
        self.job_id = self.job_kind = None
        self.job_last_clock, self.job_times = self.last_clock, {}
        self.job_caps = proposal["confirmation_study"]["budget"]["owner_seconds"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root / "budget.jsonl").open("x", encoding="utf8")
        self._append(dict(event="claim", pid=os.getpid(), started=self.started,
            proposal_sha256=digest(proposal), limits=self.limits))
