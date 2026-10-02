"""Additive continuation from qualified weights, not restoration of failed S1.

The frozen rollout/update, recording and seal machinery is reused. Only the
new-attempt bootstrap, schedule and in-owner restore adapter differ. Scientific
construction requires the separately source-bound recovery admission.
"""

import copy
from pathlib import Path

from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import EpisodeRecorder
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_campaign import DynamicCandidateCampaign
from src.rl.dynamic_candidate_factory import fork_dynamic_initializer
from src.rl.dynamic_candidate_recovery_plan import recovery_budget_plan, recovery_config
from src.rl.dynamic_candidate_resource_verification import read_dynamic_raw_episodes, verify_dynamic_raw_bundle
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, assert_dynamic_budget_ancestor
from src.rl.dynamic_candidate_sequence import DynamicPilotSequence, dynamic_required_models
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import decode_arrays, encode_arrays


class RecoverySequence(DynamicPilotSequence):
    def __init__(self, original, root):
        self.original = copy.deepcopy(original)
        self.config, self.root = recovery_config(original), Path(root).resolve()
        plan = recovery_budget_plan(original)
        self.jobs = [name for phase in self.config["phase_budgets"]
                     for name, row in plan["sections"].items() if row["phase"] == phase["id"]]
        self.models = dynamic_required_models(original)
        self.completed, self.active, self.failure, self.seals = [], None, None, {}

    def load_state_dict(self, state):
        if self.failure is not None:
            raise ValueError("failed sequence is terminal")
        candidate = RecoverySequence(self.original, self.root)
        if (set(state) != set(candidate.state_dict()) or state["config_sha256"] != digest(self.config)
                or state["format"] != candidate.state_dict()["format"]):
            raise ValueError("recovery sequence contract changed")
        for row in state["completed"]:
            candidate.begin(row["job"])
            if candidate.active == "all_model_seal":
                candidate.seals = copy.deepcopy(state["seals"])
            if any(file_record(self.root, r["path"]) != r for r in row["evidence"]):
                raise ValueError("completed recovery evidence changed")
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
            raise ValueError("recovery cursor differs")
        self.__dict__.update(candidate.__dict__)


class RecoveryCampaign(DynamicCandidateCampaign):
    def __init__(self, root, original, streams, budget, backend, *, initializer_loader,
                 qualifications, enabled=False, engineering_only=True,
                 recorder=EpisodeRecorder, reader=read_dynamic_raw_episodes,
                 verifier=verify_dynamic_raw_bundle, final_lock_check=lambda: None,
                 execution_admission=None):
        if enabled is not True or type(engineering_only) is not bool:
            raise ValueError("explicit recovery mode required")
        config = recovery_config(original)
        if engineering_only:
            if getattr(backend, "engineering_fixture", False) is not True or execution_admission is not None:
                raise ValueError("invented backend required in engineering mode")
        else:
            from src.rl.dynamic_candidate_recovery_execution import RecoveryAdmission
            if type(execution_admission) is not RecoveryAdmission:
                raise PermissionError("separate approved recovery admission required")
            execution_admission.bind_campaign(root, config, streams, budget, backend)
        if (type(budget) is not DynamicCandidateBudget or budget.plan != recovery_budget_plan(original)
                or not callable(initializer_loader)
                or set(qualifications) != {str(b) for b in config["blocks"]}
                or any(q.get("passed") is not True for q in qualifications.values())):
            raise ValueError("exact new budget, loader and completed qualifications required")
        self.original = copy.deepcopy(original)
        self.root, self.config, self.streams = Path(root).resolve(), config, copy.deepcopy(streams)
        self.budget, self.backend = budget, backend
        self.recorder_type, self.reader, self.verifier = recorder, reader, verifier
        self.final_lock_check, self.engineering_only = final_lock_check, engineering_only
        self.initializer_loader, self.saved_qualifications = initializer_loader, copy.deepcopy(qualifications)
        self.sequence = RecoverySequence(original, self.root)
        self.templates, self.initializers, self.models, self.model_paths = {}, {}, {}, {}
        self.demonstrations, self.qualification, self.qualified = {}, {}, {}
        self.test_index, self.raw_index = [], []
        self.work = self.continuation = self.live_session = self.clone = self.recorder = None
        self.closed_session = self.failure = self.payload_seal = None
        self.status_serial = self.boundary_serial = 0

    def _advance(self):
        self._check()
        if self.work is None or self.sequence.active != self.work["job"] or self.budget.active != self.work["job"]:
            raise ValueError("exact active recovery phase required")
        if self.work["job"] != "runtime_input_binding":
            return super()._advance()
        b = self.config["blocks"][self.work["cursor"]]
        self.budget.check()
        effective = self.backend.prepare(b, self._seeds(b, "prototype_preflight")[0])
        initializer = self.initializer_loader(b)
        self.budget.check()
        if initializer.contract != self.backend.producer(b).contract:
            raise ValueError("saved initializer and unchanged public input contract differ")
        qualification = self.saved_qualifications[str(b)]
        forks = fork_dynamic_initializer(initializer, qualification, self.config, self.streams, b)
        self.qualified[self._key(b)] = copy.deepcopy(qualification)
        for role, kernel in forks.items():
            key = self._key(b, role)
            self.models[key] = kernel
            path = self._state_file(f"payload/models/{key}/initial.pt", kernel.state_dict())
            self.work["evidence"].append(self._record_path(path))
            if role == "own_frozen":
                self.model_paths[key] = self._record_path(path)
        self._json(f"payload/binding/block{b}.json", effective | {
            "saved_initializer_sha256": state_digest(initializer.state_dict()),
            "qualification_reused": True, "qualification_rescored": False,
            "initializer_fit_calls": 0, "forked_policy_sha256": initializer.policy.snapshot_sha256()})
        self.work["cursor"] += 1
        self.save_boundary("saved-owner-bound")
        done = self.work["cursor"] == len(self.config["blocks"])
        if done:
            self._finish_phase()
        return done

    def restore_checkpoint_state(self, saved):
        """Same-owner recovery only; retain the live ledger, raw prefix and clock.

        This mirrors the frozen coordinator's atomic restore with the new
        sequence factory. It does not reconstruct/reload historical initializers.
        """
        self._check()
        state, current = decode_arrays(saved), decode_arrays(self.checkpoint_state())
        fixed = ("format", "config_sha256", "streams_sha256", "sequence", "model_paths", "failure", "payload_seal")
        if set(state) != set(current) or any(state[k] != current[k] for k in fixed):
            raise ValueError("restore changed recovery phase, seals or contract")
        if state["failure"] is not None:
            raise ValueError("terminal failure cannot be resumed")
        assert_dynamic_budget_ancestor(state["budget"], self.budget)
        sequence = RecoverySequence(self.original, self.root)
        sequence.load_state_dict(state["sequence"])
        self.backend.assert_restore_compatible(state["backend"])
        for name in ("test_index", "raw_index", "templates", "initializers", "demonstrations", "qualification", "qualified"):
            if state_digest(state[name]) != state_digest(current[name]):
                raise ValueError("restore cannot replace immutable imported evidence or raw rows")
        if self.models.keys() != state["models"].keys():
            raise ValueError("restore changed model ownership")
        models = copy.deepcopy(self.models)
        for key, kernel in models.items():
            self.budget.check()
            kernel.load_state_dict(state["models"][key])
        for name in ("live_session", "closed_session", "clone", "continuation"):
            if (getattr(self, name) is None) != (state[name] is None):
                raise ValueError("restore changed live collection ownership")
        if (self.recorder is None) != (state["recorder"] is None):
            raise ValueError("restore changed raw recorder ownership")
        if self.recorder is not None:
            self.recorder.assert_prefix(state["recorder"])
            if self.recorder.count != state["recorder"]["rows"]:
                raise ValueError("persisted raw rows cannot be rewound")
        continuation = copy.copy(self.continuation)
        live, clone, closed = (copy.deepcopy(getattr(self, name))
                               for name in ("live_session", "clone", "closed_session"))
        if closed is not None:
            closed.load_state_dict(encode_arrays(state["closed_session"]))
        if continuation is not None:
            factory = continuation.factory
            cached = {s.trajectory_id: s for s in (live, closed) if s is not None}
            def restored_factory(kernel, episode, seed):
                path = f"training/{state['work']['model_key']}/episode{episode:02d}"
                if path not in cached or seed != continuation.seeds[episode]:
                    raise ValueError("restore needs the already-owned matching collector")
                restored = copy.deepcopy(cached[path])
                restored.learner = kernel
                return restored
            continuation.factory = restored_factory
            try:
                continuation.load_state_dict(encode_arrays(state["continuation"]))
            finally:
                continuation.factory = factory
            models[state["work"]["model_key"]] = continuation.kernel
            live = continuation.active
        elif live is not None:
            live.load_state_dict(encode_arrays(state["live_session"]))
        if clone is not None:
            clone.load_state_dict(encode_arrays(state["clone"]))
        rng = global_rng_state()
        try:
            restore_global_rng(state["global_rng"])
        finally:
            restore_global_rng(rng)
        self.models, self.sequence, self.continuation = models, sequence, continuation
        self.live_session, self.clone, self.closed_session = live, clone, closed
        if self.recorder is not None:
            self.recorder.session = live
        self.work = copy.deepcopy(state["work"])
        self.status_serial = max(self.status_serial, state["status_serial"])
        self.boundary_serial = max(self.boundary_serial, state["boundary_serial"])
        restore_global_rng(state["global_rng"])
        self._check()
