"""Serial dynamic-pilot cursor and all-model test barrier; no experiment entrypoint."""

import copy
from pathlib import Path

from src.rl.candidate_pilot_driver import PilotSequence, file_record
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import dynamic_budget_plan


def dynamic_jobs(draft):
    plan = dynamic_budget_plan(draft)
    jobs = [name for phase in draft["phase_budgets"] for name, scope in plan["sections"].items()
            if scope["phase"] == phase["id"]]
    if len(jobs) != len(set(jobs)) or set(jobs) != set(plan["sections"]):
        raise ValueError("serial jobs must exhaust the exact budget scopes")
    return jobs


def dynamic_required_models(draft):
    if draft["representations"] != ["graph"] or draft["final_controllers"] != [
            "own_frozen", "own_ppo", "own_bc_continue", "r4", "full_mdl2"]:
        raise ValueError("this adapter is restricted to the prospective single-GCN comparison")
    return [f"block{block}/graph/{role}" for block in draft["blocks"]
            for role in ("own_frozen", "own_ppo", "own_bc_continue")]


class DynamicPilotSequence(PilotSequence):
    """Byte-verified barriers with terminal failure, not permission to execute."""

    def __init__(self, draft, root, *, enabled=False):
        if enabled is not True:
            raise ValueError("dynamic pilot sequence requires explicit opt-in")
        self.config, self.root = copy.deepcopy(draft), Path(root).resolve()
        self.jobs = dynamic_jobs(draft)
        self.models = dynamic_required_models(draft)
        self.completed, self.active, self.failure, self.seals = [], None, None, {}

    def begin(self, job):
        if self.active is not None or job != self.next_job or job is None:
            raise ValueError("failed, repeated, concurrent or out-of-order dynamic job")
        if job.startswith("final_evaluation/"):
            self.check_seals()
        self.active = job

    def seal_models(self, paths):
        if self.failure or self.active != "all_model_seal" or self.seals or set(paths) != set(self.models):
            raise ValueError("all learned artifacts must be sealed once at the test barrier")
        records = {name: file_record(self.root, path) for name, path in paths.items()}
        if len({r["path"] for r in records.values()}) != len(records):
            raise ValueError("separate named model artifacts required")
        self.seals = records

    def check_seals(self):
        if set(self.seals) != set(self.models):
            raise ValueError("test data forbidden before every learned artifact is frozen")
        for record in self.seals.values():
            if file_record(self.root, record["path"]) != record:
                raise ValueError("sealed model bytes changed")

    def finish(self, evidence):
        if self.failure or self.active != self.next_job or self.active is None or not evidence:
            raise ValueError("active successful job and nonempty evidence required")
        records = [file_record(self.root, p) for p in evidence]
        if len({r["path"] for r in records}) != len(records):
            raise ValueError("duplicate evidence")
        if (self.active == "all_model_seal" or self.active.startswith("final_evaluation/")
                or self.active in ("saved_data_verification_analysis", "local_archive_verification",
                                   "supervisor_dispatch_terminal_closure")):
            self.check_seals()
        self.completed.append({"job": self.active, "evidence": records})
        self.active = None

    def state_dict(self):
        return copy.deepcopy({"format": "dynamic-pilot-sequence-v1", "config_sha256": digest(self.config),
                              "completed": self.completed, "active": self.active,
                              "failure": self.failure, "seals": self.seals})

    def load_state_dict(self, state):
        if self.failure is not None:
            raise ValueError("failed sequence cannot restore an earlier state to retry")
        candidate = DynamicPilotSequence(self.config, self.root, enabled=True)
        if (set(state) != set(candidate.state_dict()) or state["config_sha256"] != digest(self.config)
                or state["format"] != "dynamic-pilot-sequence-v1"):
            raise ValueError("dynamic sequence contract mismatch")
        for completed in state["completed"]:
            candidate.begin(completed["job"])
            if candidate.active == "all_model_seal":
                candidate.seals = copy.deepcopy(state["seals"])
            for record in completed["evidence"]:
                if file_record(self.root, record["path"]) != record:
                    raise ValueError("completed evidence changed")
            candidate.finish([r["path"] for r in completed["evidence"]])
        if state["active"] is not None:
            candidate.begin(state["active"])
            if candidate.active == "all_model_seal":
                candidate.seals = copy.deepcopy(state["seals"])
                if candidate.seals:
                    candidate.check_seals()
        if state["failure"] is not None:
            candidate.fail(state["failure"]["reason"])
        if candidate.state_dict() != state:
            raise ValueError("sequence history or failure cursor differs")
        self.__dict__.update(candidate.__dict__)
