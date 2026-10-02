"""Fixed six-controller schedule and twelve-model barrier; no launch permit."""

import copy
from pathlib import Path

from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_sequence import DynamicPilotSequence
from src.rl.time_baseline_plan import CONTROLLERS, time_baseline_serial_sections


def time_baseline_required_models(proposal):
    time_baseline_serial_sections(proposal)
    return [f"block{block}/graph/{role}" for block in proposal["blocks"] for role in CONTROLLERS[:4]]


class TimeBaselineSequence(DynamicPilotSequence):
    def __init__(self, proposal, root, *, enabled=False):
        if enabled is not True:
            raise ValueError("explicit time-baseline sequencing required")
        self.config, self.root = copy.deepcopy(proposal), Path(root).resolve()
        self.jobs = time_baseline_serial_sections(proposal)
        self.models = time_baseline_required_models(proposal)
        self.completed, self.active, self.failure, self.seals = [], None, None, {}

    def finish(self, evidence):
        if self.active in ("raw_verification", "payload_archive", "supervisor_closure"):
            self.check_seals()
        super().finish(evidence)

    def state_dict(self):
        return super().state_dict() | {"format": "time-baseline-sequence-v1"}

    def load_state_dict(self, state):
        if self.failure is not None:
            raise ValueError("failed sequence cannot rewind or retry")
        candidate = TimeBaselineSequence(self.config, self.root, enabled=True)
        if (set(state) != set(candidate.state_dict()) or state["config_sha256"] != digest(self.config)
                or state["format"] != "time-baseline-sequence-v1"):
            raise ValueError("time-baseline sequence contract differs")
        for row in state["completed"]:
            candidate.begin(row["job"])
            if candidate.active == "all_model_seal":
                candidate.seals = copy.deepcopy(state["seals"])
            if any(file_record(self.root, r["path"]) != r for r in row["evidence"]):
                raise ValueError("completed evidence changed")
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
            raise ValueError("sequence cursor or receipts differ")
        self.__dict__.update(candidate.__dict__)
