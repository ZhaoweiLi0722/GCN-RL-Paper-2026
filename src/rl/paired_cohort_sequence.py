"""Serial phase cursor and twelve-model test barrier, not execution permission."""

import copy
from pathlib import Path

from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_sequence import DynamicPilotSequence
from src.rl.paired_cohort_resources import budget_plan


class PairedCohortSequence(DynamicPilotSequence):
    def __init__(self, config, root, *, enabled=False):
        if enabled is not True:
            raise ValueError("explicit finite sequence required")
        plan = budget_plan(config)
        self.config, self.root = copy.deepcopy(config), Path(root).resolve()
        self.jobs = list(plan["sections"])
        roles = config["evaluation_controllers"]
        if roles != ["own_frozen", "paired_cost", "bc_continue", "saved_cohort_ppo", "r4", "full_mdl2"]:
            raise ValueError("prescribed six-controller inventory required")
        self.models = [f"block{b}/{role}" for b in config["blocks"] for role in roles[:4]]
        self.completed, self.active, self.failure, self.seals = [], None, None, {}

    def begin(self, job):
        if job is not None and job.startswith("evaluation/"):
            self.check_seals()
        super().begin(job)

    def seal_models(self, paths):
        if self.failure or self.active != "seal" or self.seals or set(paths) != set(self.models):
            raise ValueError("all twelve frozen/trained models must seal once before test")
        records = {name: file_record(self.root, path) for name, path in paths.items()}
        if len({r["path"] for r in records.values()}) != len(records):
            raise ValueError("separate named model artifacts required")
        self.seals = records

    def finish(self, evidence):
        if (self.active == "seal" or (self.active or "").startswith("evaluation/")
                or self.active in ("raw_verification", "archive", "closure")):
            self.check_seals()
        super().finish(evidence)

    def state_dict(self):
        return super().state_dict() | {"format": "paired-cohort-sequence-v1"}

    def load_state_dict(self, state):
        if self.failure is not None:
            raise ValueError("failed attempt cannot restore to retry")
        candidate = PairedCohortSequence(self.config, self.root, enabled=True)
        if (set(state) != set(candidate.state_dict()) or state["format"] != "paired-cohort-sequence-v1"
                or state["config_sha256"] != digest(self.config)
                or state["completed"][:len(self.completed)] != self.completed):
            raise ValueError("sequence binding or non-refundable completed prefix differs")
        for row in state["completed"]:
            candidate.begin(row["job"])
            if candidate.active == "seal":
                candidate.seals = copy.deepcopy(state["seals"])
            if any(file_record(self.root, r["path"]) != r for r in row["evidence"]):
                raise ValueError("completed evidence bytes changed")
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
            raise ValueError("serial boundary did not roundtrip")
        self.__dict__.update(candidate.__dict__)


def dispatch_serial(sequence, budget, perform, persist, *, admit):
    """Run the exact unblocked finite chain; exceptions remain terminal.

    The enclosing entry owns packet/claim admission, native phase operations,
    full process state, watchdog and archive supervision. This does not provide
    a scientific launch command. Recovery must reconcile the outer durable
    ledger before calling again, and cannot reopen a terminal exception.
    """
    if (type(sequence) is not PairedCohortSequence or budget.plan != budget_plan(sequence.config)
            or any(not callable(f) for f in (perform, persist, admit))
            or sequence.active is not None or budget.active is not None
            or sequence.failure is not None
            or list(budget.closed) != [r["job"] for r in sequence.completed]):
        raise ValueError("matching idle serial/budget boundary and callbacks required")
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
