"""Recovery-only serial cursor with the original twelve-model seal barrier."""

import copy
from pathlib import Path

from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_resources import digest
from src.rl.paired_cohort_sequence import PairedCohortSequence
from src.rl.paired_cohort_recovery_resources import budget_plan


class PairedCohortRecoverySequence(PairedCohortSequence):
    def __init__(self, config, recovery_proposal, root, *, enabled=False):
        if enabled is not True:
            raise ValueError("explicit finite recovery sequence required")
        plan = budget_plan(config, recovery_proposal)
        self.config, self.proposal = copy.deepcopy((config, recovery_proposal))
        self.root, self.jobs = Path(root).resolve(), list(plan["sections"])
        self.models = [f"block{b}/{role}" for b in config["blocks"]
                       for role in config["evaluation_controllers"][:4]]
        self.completed, self.active, self.failure, self.seals = [], None, None, {}

    def state_dict(self):
        return super().state_dict() | dict(format="paired-cohort-recovery-sequence-v1",
                                           recovery_proposal_sha256=digest(self.proposal))

    def load_state_dict(self, state):
        if self.failure is not None:
            raise ValueError("failed recovery cannot restore to retry")
        candidate = type(self)(self.config, self.proposal, self.root, enabled=True)
        if (set(state) != set(candidate.state_dict()) or state["format"] != "paired-cohort-recovery-sequence-v1"
                or state["config_sha256"] != digest(self.config)
                or state["recovery_proposal_sha256"] != digest(self.proposal)
                or state["completed"][:len(self.completed)] != self.completed
                or (self.active is not None and (state["active"] != self.active
                                                or state["completed"] != self.completed))):
            raise ValueError("recovery binding or nonrefundable serial boundary differs")
        for row in state["completed"]:
            candidate.begin(row["job"])
            if candidate.active == "seal":
                candidate.seals = copy.deepcopy(state["seals"])
            if any(file_record(self.root, r["path"]) != r for r in row["evidence"]):
                raise ValueError("completed recovery evidence changed")
            candidate.finish([r["path"] for r in row["evidence"]])
        if state["active"] is not None:
            candidate.begin(state["active"])
            if candidate.active == "seal":
                candidate.seals = copy.deepcopy(state["seals"])
                if candidate.seals:
                    candidate.check_seals()
        if state["failure"] is not None:
            candidate.fail(state["failure"]["reason"])
        if candidate.state_dict() != state:
            raise ValueError("recovery serial boundary did not roundtrip")
        self.__dict__.update(candidate.__dict__)


def dispatch_serial(sequence, budget, perform, persist, *, admit):
    """Admit before work; a failed attempt cannot adopt or refund old ledgers."""
    if (type(sequence) is not PairedCohortRecoverySequence
            or budget.plan != budget_plan(sequence.config, sequence.proposal)
            or any(not callable(f) for f in (perform, persist, admit))
            or sequence.active is not None or budget.active is not None
            or sequence.failure is not None
            or list(budget.closed) != [r["job"] for r in sequence.completed]):
        raise ValueError("matching idle recovery sequence and budget required")
    try:
        while sequence.next_job is not None:
            job = sequence.next_job
            admit(job)
            sequence.begin(job)
            budget.begin(job)
            persist("begin", sequence.state_dict(), budget.snapshot())
            evidence = perform(job, sequence, budget)
            budget.check()
            sequence.finish(evidence)
            budget.finish()
            persist("complete", sequence.state_dict(), budget.snapshot())
    except BaseException as error:
        if sequence.failure is None:
            sequence.fail(repr(error))
        persist("failed", sequence.state_dict(), budget.snapshot())
        raise
    return sequence.state_dict()
