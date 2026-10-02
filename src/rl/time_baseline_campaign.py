"""Six-controller serial coordinator with separate scientific admission.

Collection, logging, nonrefundable accounting and archives reuse frozen helpers.
The new code binds changed owners, target methods, streams and phase names.
Invented fixtures need no scientific permit. Real execution requires the exact
new scope-specific admission. Its presence is not a launch permit.
"""

import copy
from pathlib import Path

import numpy as np

from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.candidate_pilot_recording import EpisodeRecorder
from src.rl.dynamic_candidate_campaign import DynamicCandidateCampaign
from src.rl.dynamic_candidate_continuation import DynamicCandidateContinuation
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, assert_dynamic_budget_ancestor
from src.rl.dynamic_candidate_rollout import evaluate_dynamic_policy
from src.rl.dynamic_candidate_session import dynamic_context_from_public
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import decode_arrays, encode_arrays
from src.rl.time_baseline_collection import TimeBaselineContinuation
from src.rl.time_baseline_factory import time_baseline_config, fork_time_baseline_initializer
from src.rl.time_baseline_plan import CONTROLLERS, TRAINING_ARMS, time_baseline_budget_plan, time_baseline_stream_manifest
from src.rl.time_baseline_sequence import TimeBaselineSequence
from src.rl.time_baseline_target_verification import verify_time_baseline_targets
from src.utils.research_archive import inventory


class TimeBaselineCampaign(DynamicCandidateCampaign):
    def __init__(self, root, original, proposal, streams, budget, backend, *, initializer_loader,
                 qualifications, enabled=False, engineering_only=True, recorder=EpisodeRecorder,
                 reader=None, verifier=None, final_lock_check=lambda: None, execution_admission=None):
        if enabled is not True or type(engineering_only) is not bool:
            raise ValueError("explicit time-baseline mode required")
        config = time_baseline_config(original, proposal)
        if engineering_only:
            if getattr(backend, "engineering_fixture", False) is not True or execution_admission is not None:
                raise PermissionError("only invented backends may use engineering mode")
        else:
            from src.rl.time_baseline_execution import TimeBaselineAdmission
            if type(execution_admission) is not TimeBaselineAdmission:
                raise PermissionError("exact separate time-baseline admission required")
            execution_admission.bind_campaign(root, config, streams, budget, backend)
        if (type(budget) is not DynamicCandidateBudget or budget.plan != time_baseline_budget_plan(proposal)
                or streams != time_baseline_stream_manifest(proposal) or not callable(initializer_loader)
                or set(qualifications) != {str(b) for b in config["blocks"]}
                or any(q.get("passed") is not True for q in qualifications.values())):
            raise ValueError("exact budget, streams, loader and passed saved qualifications required")
        if reader is None or verifier is None:
            from src.rl.time_baseline_verification import read_time_baseline_raw_episodes, verify_time_baseline_raw_bundle
            reader = reader or read_time_baseline_raw_episodes
            verifier = verifier or verify_time_baseline_raw_bundle
        self.original, self.proposal = copy.deepcopy(original), copy.deepcopy(proposal)
        self.root, self.config, self.streams = Path(root).resolve(), config, copy.deepcopy(streams)
        self.budget, self.backend = budget, backend
        self.recorder_type, self.reader, self.verifier = recorder, reader, verifier
        self.final_lock_check, self.engineering_only = final_lock_check, engineering_only
        self.initializer_loader, self.saved_qualifications = initializer_loader, copy.deepcopy(qualifications)
        self.sequence = TimeBaselineSequence(proposal, self.root, enabled=True)
        self.templates, self.initializers, self.models, self.model_paths = {}, {}, {}, {}
        self.demonstrations, self.qualification, self.qualified = {}, {}, {}
        self.test_index, self.raw_index = [], []
        self.work = self.continuation = self.live_session = self.clone = self.recorder = None
        self.closed_session = self.failure = self.payload_seal = None
        self.status_serial = self.boundary_serial = 0

    def _seeds(self, block, split):
        return [int(s) for s in self.streams["environment"][str(block)][split]]

    def _session(self, descriptor):
        block, role, world, seed, split, selection, name = descriptor
        key = self._key(block, role if role in CONTROLLERS[:4] else "own_frozen")
        if split == "test":
            self.sequence.check_seals()
        kernel = copy.deepcopy((self.templates if split == "preflight" else self.models)[key])
        if split == "test":
            self.budget.check()
            if state_digest(kernel.state_dict()) != state_digest(load_envelope(self.root / self.model_paths[key])):
                raise ValueError("evaluation owner differs from sealed model")
        self.budget.check()
        self.live_session = self.backend.session(block, kernel, seed, trajectory=name, split=split, selection=selection)
        self.recorder = self.recorder_type(self.root, f"payload/episodes/{name}", self.live_session, self.config,
            block=block, representation="reference" if role in ("r4", "full_mdl2") else "graph",
            role=role, world_index=world, seed=seed)
        self.work["episode_initial_kernel"] = state_digest(kernel.state_dict())
        self.work["clone_compared"] = 0

    def _compare_forks(self, block, run):
        self.budget.check()
        observation, bank = dynamic_context_from_public(run.producer, run.reference,
            run.env.observation(), run.options, run.expected_token)
        choices = []
        for role in CONTROLLERS[:4]:
            self.budget.check()
            owner = self.models[self._key(block, role)]
            result = evaluate_dynamic_policy(owner.policy, observation, bank, owner.contract)
            choices.append((int(np.argmax(result.log_probs)), result.log_probs))
        if any(c != choices[0] for c in choices[1:]):
            raise ValueError("same-start four-fork probabilities differ")

    def _descriptors(self, job):
        if job.startswith("final_evaluation/"):
            return super()._descriptors(job)
        if job != "same_start_preflight":
            return []
        return [(b, role, 0, self._seeds(b, "preflight")[0], "preflight",
                 "reference" if role == "r4" else "anchor" if role == "full_mdl2"
                 else "greedy" if role == "own_frozen" else "sample",
                 f"preflight/fork/block{b}/{role}") for b in self.config["blocks"] for role in CONTROLLERS]

    def _bind_block(self):
        b = self.config["blocks"][self.work["cursor"]]
        self.budget.check()
        effective = self.backend.prepare(b, self._seeds(b, "layout")[0])
        initializer = self.initializer_loader(b)
        self.budget.check()
        if initializer.contract != self.backend.producer(b).contract:
            raise ValueError("saved initializer/public producer contract differs")
        qualification = self.saved_qualifications[str(b)]
        self.qualified[self._key(b)] = copy.deepcopy(qualification)
        for split in ("training", "preflight"):
            forks = fork_time_baseline_initializer(initializer, qualification, self.config, self.streams, b, split=split)
            for role, owner in forks.items():
                key = self._key(b, role)
                if split == "preflight":
                    self.templates[key] = owner
                    continue
                self.models[key] = owner
                path = self._state_file(f"payload/models/{key}/initial.pt", owner.state_dict())
                self.work["evidence"].append(self._record_path(path))
                if role == "own_frozen":
                    self.model_paths[key] = self._record_path(path)
        self._json(f"payload/binding/block{b}.json", effective | {
            "saved_initializer_sha256": state_digest(initializer.state_dict()),
            "qualification_reused": True, "qualification_rescored": False, "initializer_fit_calls": 0,
            "forked_policy_sha256": initializer.policy.snapshot_sha256(), "separate_preflight_rng": True})
        self.work["cursor"] += 1
        self.save_boundary("saved-owner-bound")
        return self.work["cursor"] == len(self.config["blocks"])

    def _train_step(self, job):
        role, block = job.split("/")
        b, cfg = int(block[5:]), self.config["continuation"]
        model_key = self._key(b, role)
        if self.continuation is None:
            self.work["model_key"] = model_key
            settings = dict(episodes_per_model=cfg["episodes_per_arm_per_block"],
                episodes_per_rollout=cfg["episodes_per_rollout"], max_updates=cfg["rollouts_per_arm_per_block"],
                epochs=cfg["epochs"], gae_lambda=self.config["objective"]["gae_lambda"])
            def factory(kernel, episode, seed):
                self.budget.check()
                return self.backend.session(b, kernel, seed, trajectory=f"training/{model_key}/episode{episode:02d}",
                                            split="training", selection="sample")
            cls = DynamicCandidateContinuation if role == "bc_continue" else TimeBaselineContinuation
            extra = {"role": "bc_continue"} if role == "bc_continue" else {}
            self.continuation = cls(self.models[model_key], self.budget, settings, enabled=True,
                scope=job, seeds=self._seeds(b, "training"), horizon=self.config["objective"]["horizon"],
                session_factory=factory, **extra)
            self.save_boundary("continuation-start")
        run = self.continuation
        if run.update_due:
            self.budget.check()
            self.work["updates"].append(run.update())
            self.models[model_key] = run.kernel
            self.save_boundary("update-complete")
        elif not run.done:
            if run.active is None:
                self.budget.check()
                self.live_session = run.start_episode()
                episode = run.episode
                self.recorder = self.recorder_type(self.root,
                    f"payload/episodes/training/{model_key}/episode{episode:02d}", self.live_session, self.config,
                    block=b, representation="graph", role=role, world_index=episode, seed=run.seeds[episode])
            self.budget.check()
            event = run.step()
            self.budget.check()
            self.recorder.append(event)
            if self.recorder.session.closed:
                self.closed_session = copy.deepcopy(self.recorder.session)
                index = self.recorder.finish()
                self.work["indexes"].append(index)
                self.raw_index.append(index)
                self.recorder, self.live_session = None, None
        if not run.done:
            return False
        self.models[model_key] = run.kernel
        final = self._state_file(f"payload/models/{model_key}/final.pt", run.kernel.state_dict())
        self.model_paths[model_key] = self._record_path(final)
        self.work["evidence"].append(self._record_path(final))
        self.continuation = self.closed_session = None
        return True

    def _advance(self):
        self._check()
        if self.work is None or self.sequence.active != self.work["job"] or self.budget.active != self.work["job"]:
            raise ValueError("exact active phase required")
        job, done = self.work["job"], False
        if job == "runtime_input_binding":
            done = self._bind_block()
        elif job.split("/")[0] in TRAINING_ARMS:
            done = self._train_step(job)
        elif job == "all_model_seal":
            self.sequence.seal_models(self.model_paths)
            self._json("payload/model-seals.json", self.sequence.seals)
            done = True
        elif job == "raw_verification":
            self._verify()
            done = True
        elif job == "payload_archive":
            self._archive()
            done = True
        elif job == "supervisor_closure":
            self.sequence.check_seals()
            self.final_lock_check()
            self._json("launcher/closure.json", dict(engineering_fixture=self.engineering_only, scientific_execution=not self.engineering_only,
                payload_unchanged=inventory(self.root / "payload") == self.payload_seal, automatic_followon=False))
            done = True
        else:
            descriptors = self._descriptors(job)
            if not descriptors:
                raise ValueError("unknown time-baseline phase")
            descriptor = descriptors[self.work["cursor"]]
            self._collect_step(descriptor, clone=job == "same_start_preflight",
                compare_forks=job == "same_start_preflight" and descriptor[1] == "own_frozen")
            done = self.work["cursor"] == len(descriptors)
        if done:
            self._finish_phase()
        return done

    def _verify(self):
        self.sequence.check_seals()
        self.budget.check()
        targets = verify_time_baseline_targets(self.root, self.raw_index, self.config, self.streams)
        self.budget.check()
        self._json("payload/independent-target-verification.json", targets)
        super()._verify()

    def _finish_phase(self):
        job = self.work["job"]
        location = "launcher/phase-evidence" if self.payload_seal is not None else "payload/phases"
        path = self._json(f"{location}/{job}.json", self.work)
        evidence = self.work["evidence"] + [self._record_path(path)]
        self.save_boundary("phase-work-complete")
        self.emit_status("phase_completed", completed_job=job)
        self.sequence.finish(evidence)
        if job == "supervisor_closure":
            self._json("launcher/completed.json", dict(status="completed", exit_code=0,
                engineering_fixture=self.engineering_only, scientific_execution=not self.engineering_only, sequence=self.sequence.state_dict(),
                budget=self.budget.snapshot(), automatic_followon=False, requires_successful_supervisor_receipt=True))
        self.budget.check()
        self.budget.finish()
        self.work = None

    def restore_checkpoint_state(self, saved):
        """In-owner only: no new loads, environments, raw-prefix rewind or refund."""
        self._check()
        state, current = decode_arrays(saved), decode_arrays(self.checkpoint_state())
        fixed = ("format", "config_sha256", "streams_sha256", "sequence", "model_paths", "failure", "payload_seal")
        if set(state) != set(current) or any(state[k] != current[k] for k in fixed) or state["failure"] is not None:
            raise ValueError("restore changed phase/seals/contract or cleared failure")
        assert_dynamic_budget_ancestor(state["budget"], self.budget)
        sequence = TimeBaselineSequence(self.proposal, self.root, enabled=True)
        sequence.load_state_dict(state["sequence"])
        self.backend.assert_restore_compatible(state["backend"])
        for name in ("test_index", "raw_index", "templates", "initializers", "demonstrations", "qualification", "qualified"):
            if state_digest(state[name]) != state_digest(current[name]):
                raise ValueError("immutable imported evidence/raw prefix changed")
        if self.models.keys() != state["models"].keys():
            raise ValueError("model ownership changed")
        models = copy.deepcopy(self.models)
        for key, kernel in models.items():
            self.budget.check()
            kernel.load_state_dict(state["models"][key])
        for name in ("live_session", "closed_session", "clone", "continuation"):
            if (getattr(self, name) is None) != (state[name] is None):
                raise ValueError("live collection ownership changed")
        if (self.recorder is None) != (state["recorder"] is None):
            raise ValueError("raw recorder ownership changed")
        if self.recorder is not None:
            self.recorder.assert_prefix(state["recorder"])
            if self.recorder.count != state["recorder"]["rows"]:
                raise ValueError("persisted raw rows cannot rewind")
        continuation = copy.copy(self.continuation)
        live, clone, closed = (copy.deepcopy(getattr(self, name)) for name in ("live_session", "clone", "closed_session"))
        if closed is not None:
            closed.load_state_dict(encode_arrays(state["closed_session"]))
        if continuation is not None:
            factory = continuation.factory
            cached = {s.trajectory_id: s for s in (live, closed) if s is not None}
            def restored_factory(kernel, episode, seed):
                path = f"training/{state['work']['model_key']}/episode{episode:02d}"
                if path not in cached or seed != continuation.seeds[episode]:
                    raise ValueError("matching already-owned collector required")
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
