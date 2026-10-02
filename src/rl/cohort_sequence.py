"""Explicit cohort controller order and twelve-file test-entry barrier."""

import copy
from pathlib import Path

from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_resources import digest
from src.rl.cohort_objective_plan import cohort_budget_plan
from src.rl.dynamic_candidate_sequence import DynamicPilotSequence


class CohortSequence(DynamicPilotSequence):
    def __init__(self, base, proposal, root, *, enabled=False):
        if enabled is not True:
            raise ValueError("explicit cohort sequence required")
        plan = cohort_budget_plan(base, proposal)
        self.base, self.config, self.root = copy.deepcopy(base), copy.deepcopy(proposal), Path(root).resolve()
        self.jobs = list(plan["sections"])
        self.models = [f"block{b}/graph/{role}" for b in proposal["blocks"]
                       for role in proposal["evaluation_controllers"][:4]]
        self.completed, self.active, self.failure, self.seals = [], None, None, {}

    def finish(self, evidence):
        if self.active in ("raw_verification", "payload_archive", "supervisor_closure"):
            self.check_seals()
        super().finish(evidence)

    def state_dict(self):
        return super().state_dict() | {"format": "cohort-sequence-v1", "base_sha256": digest(self.base)}

    def load_state_dict(self, state):
        if self.failure is not None:
            raise ValueError("failed sequence cannot retry")
        candidate = CohortSequence(self.base, self.config, self.root, enabled=True)
        if (set(state) != set(candidate.state_dict()) or state["format"] != "cohort-sequence-v1"
                or state["base_sha256"] != digest(self.base) or state["config_sha256"] != digest(self.config)
                or state["completed"][:len(self.completed)] != self.completed):
            raise ValueError("cohort sequence contract or completed evidence differs")
        for row in state["completed"]:
            candidate.begin(row["job"])
            if candidate.active == "all_model_seal":
                candidate.seals = copy.deepcopy(state["seals"])
            if any(file_record(self.root, r["path"]) != r for r in row["evidence"]):
                raise ValueError("completed evidence bytes changed")
            candidate.finish([r["path"] for r in row["evidence"]])
        if state["active"] is not None:
            candidate.begin(state["active"])
            if candidate.active == "all_model_seal":
                candidate.seals = copy.deepcopy(state["seals"])
                if candidate.seals:
                    candidate.check_seals()
        if state["failure"] is not None:
            candidate.fail(state["failure"]["reason"])
        if candidate.state_dict() != state:
            raise ValueError("cohort cursor or receipts differ")
        self.__dict__.update(candidate.__dict__)
