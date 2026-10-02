"""Single serial, budget-owned cohort comparison using sealed saved initializers."""

import copy
from pathlib import Path

import numpy as np

from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.cohort_collection import CohortCollection
from src.rl.cohort_continuation import CohortContinuation
from src.rl.cohort_factory import cohort_config, fork_cohort_initializer
from src.rl.cohort_objective_plan import cohort_budget_plan, cohort_stream_manifest
from src.rl.cohort_recording import CohortRecorder
from src.rl.cohort_sequence import CohortSequence
from src.rl.dynamic_candidate_campaign import DynamicCandidateCampaign
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, assert_dynamic_budget_ancestor
from src.rl.dynamic_candidate_rollout import evaluate_dynamic_policy
from src.rl.dynamic_candidate_session import dynamic_context_from_public
from src.rl.patient_replay_collector import evidence_digest
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import decode_arrays, encode_arrays
from src.rl.time_baseline_campaign import TimeBaselineCampaign
from src.utils.research_archive import inventory


def private_collection(run):
    """Copy owned numerical state, never a recorder handle or bound campaign."""
    if run is None:
        return None
    other = copy.copy(run)
    other.prefix = copy.deepcopy(run.prefix)
    other.env = other.prefix.env
    for name in ("prefix_costs", "tail_events", "prefix_snapshot", "prefix_receipt", "failure"):
        setattr(other, name, copy.deepcopy(getattr(run, name)))
    other.record_prefix = other.record_tail = lambda *args: None
    other.finish_prefix = lambda prefix: copy.deepcopy(run.prefix_receipt)
    return other


class CohortCampaign(TimeBaselineCampaign):
    def __init__(self, root, original, base, proposal, streams, budget, backend, *, initializer_loader,
                 qualifications, enabled=False, engineering_only=True, recorder=CohortRecorder,
                 reader=None, verifier=None, target_verifier=None,
                 final_lock_check=lambda: None, execution_admission=None):
        if enabled is not True or type(engineering_only) is not bool:
            raise ValueError("explicit cohort campaign mode required")
        config = cohort_config(original, base, proposal)
        if engineering_only:
            if getattr(backend, "engineering_fixture", False) is not True or execution_admission is not None:
                raise PermissionError("only invented backends may use engineering mode")
        else:
            from src.rl.cohort_execution import CohortAdmission
            if type(execution_admission) is not CohortAdmission:
                raise PermissionError("separate exact cohort admission required")
            execution_admission.bind_campaign(root, config, streams, budget, backend)
        if (type(budget) is not DynamicCandidateBudget or budget.plan != cohort_budget_plan(base, proposal)
                or streams != cohort_stream_manifest(base, proposal) or not callable(initializer_loader)
                or set(qualifications) != {str(b) for b in config["blocks"]}
                or any(q.get("passed") is not True for q in qualifications.values())):
            raise ValueError("exact cohort budget, streams and saved qualifications required")
        if reader is None or verifier is None:
            from src.rl.cohort_bundle_verification import read_cohort_raw_episodes, verify_cohort_raw_bundle
            reader, verifier = reader or read_cohort_raw_episodes, verifier or verify_cohort_raw_bundle
        if target_verifier is None:
            from src.rl.cohort_target_verification import verify_cohort_targets
            target_verifier = verify_cohort_targets
        self.original, self.base, self.proposal = copy.deepcopy((original, base, proposal))
        self.root, self.config, self.streams = Path(root).resolve(), config, copy.deepcopy(streams)
        self.budget, self.backend = budget, backend
        self.recorder_type, self.reader, self.verifier = recorder, reader, verifier
        self.target_verifier = target_verifier
        self.final_lock_check, self.engineering_only = final_lock_check, engineering_only
        self.initializer_loader, self.saved_qualifications = initializer_loader, copy.deepcopy(qualifications)
        self.sequence = CohortSequence(base, proposal, self.root, enabled=True)
        self.templates, self.initializers, self.models, self.model_paths = {}, {}, {}, {}
        self.demonstrations, self.qualification, self.qualified = {}, {}, {}
        self.test_index, self.raw_index = [], []
        self.work = self.continuation = self.live_session = self.clone = self.recorder = None
        self.closed_session = self.failure = self.payload_seal = self.parity = None
        self.status_serial = self.boundary_serial = 0

    def _make_collection(self, block, role, world, seed, split, selection, name, kernel):
        self.budget.check()
        prefix = self.backend.session(block, kernel, seed, trajectory=name, split=split, selection=selection)
        self.recorder = self.recorder_type(self.root, f"payload/episodes/{name}", prefix, self.config,
            block=block, representation="reference" if role in ("r4", "full_mdl2") else "graph",
            role=role, world_index=world, seed=seed)
        run = CohortCollection(prefix, enabled=True, objective=getattr(kernel, "objective", "none")
            if split == "training" else "none", split=split, trajectory_id=name,
            followup_action=lambda env: env.common_followup_request(prefix.producer.anchor_config),
            finish_prefix=self.recorder.finish_prefix, record_prefix=self.recorder.record_prefix,
            record_tail=self.recorder.record_tail)
        self.live_session = run
        return run

    def _session(self, descriptor):
        b, role, world, seed, split, selection, name = descriptor
        key = self._key(b, role if role in self.config["final_controllers"][:4] else "own_frozen")
        if split == "test":
            self.sequence.check_seals()
        kernel = copy.deepcopy((self.templates if split == "preflight" else self.models)[key])
        if split == "test" and state_digest(kernel.state_dict()) != state_digest(load_envelope(self.root / self.model_paths[key])):
            raise ValueError("evaluation owner differs from sealed model")
        self._make_collection(*descriptor, kernel)
        self.work.update(episode_initial_kernel=state_digest(kernel.state_dict()), clone_compared=0,
                         clones_completed=0, parity_rows=[])
        if split == "preflight" and role == "own_frozen":
            self.parity = self.backend.parity_environment(b, seed)
            if evidence_digest(self.parity.state_dict()) != evidence_digest(self.live_session.env.state_dict()):
                raise ValueError("original engine and cohort prefix initialization differ")

    def _compare_forks(self, block, run):
        prefix = run.prefix
        self.budget.check()
        obs, bank = dynamic_context_from_public(prefix.producer, prefix.reference,
            prefix.env.observation(), prefix.options, prefix.expected_token)
        choices = []
        for role in self.config["final_controllers"][:4]:
            self.budget.check()
            owner = self.models[self._key(block, role)]
            result = evaluate_dynamic_policy(owner.policy, obs, bank, owner.contract)
            choices.append((int(np.argmax(result.log_probs)), result.log_probs))
        if any(c != choices[0] for c in choices[1:]):
            raise ValueError("same-start four-fork probabilities differ")

    def _descriptors(self, job):
        if job.startswith("final_evaluation/"):
            return DynamicCandidateCampaign._descriptors(self, job)
        if job != "same_start_preflight":
            return []
        return [(b, role, 0, self._seeds(b, "preflight")[0], "preflight",
                 "reference" if role == "r4" else "anchor" if role == "full_mdl2"
                 else "greedy" if role == "own_frozen" else "sample",
                 f"preflight/fork/block{b}/{role}")
                for b in self.config["blocks"] for role in self.config["final_controllers"]]

    def _bind_block(self):
        b = self.config["blocks"][self.work["cursor"]]
        self.budget.check()
        effective = self.backend.prepare(b, self._seeds(b, "layout")[0])
        initializer = self.initializer_loader(b)
        if initializer.contract != self.backend.producer(b).contract:
            raise ValueError("saved initializer and public input contract differ")
        qualification = self.saved_qualifications[str(b)]
        self.qualified[self._key(b)] = copy.deepcopy(qualification)
        for split in ("training", "preflight"):
            for role, owner in fork_cohort_initializer(initializer, qualification, self.config, self.streams, b, split=split).items():
                key = self._key(b, role)
                if split == "preflight":
                    self.templates[key] = owner
                    continue
                self.models[key] = owner
                path = self._state_file(f"payload/models/{key}/initial.pt", owner.state_dict())
                self.work["evidence"].append(self._record_path(path))
                if role == "own_frozen":
                    self.model_paths[key] = self._record_path(path)
        self._json(f"payload/binding/block{b}.json", effective | dict(
            saved_initializer_sha256=state_digest(initializer.state_dict()), qualification_reused=True,
            qualification_rescored=False, initializer_fit_calls=0, separate_preflight_rng=True))
        self.work["cursor"] += 1
        self.save_boundary("saved-owner-bound")
        return self.work["cursor"] == len(self.config["blocks"])

    def _collect_step(self, descriptor, *, clone=False, compare_forks=False):
        if self.live_session is None:
            self._session(descriptor)
        run = self.live_session
        b, role, world, seed, split, selection, name = descriptor
        prefix_steps, tail_steps = self.proposal["enrollment_steps"], self.proposal["accounting_steps"]
        midpoints = (prefix_steps // 2, prefix_steps + 1)
        clone_counts = (min(4, prefix_steps - midpoints[0]), min(4, tail_steps - 1))
        if clone and run.index in midpoints:
            which = midpoints.index(run.index)
            if self.clone is not None or self.work["clones_completed"] != which:
                raise ValueError("preflight clone ownership differs")
            path = self._state_file(f"payload/preflight/{name}/restore-{which}.pt", run.state_dict())
            self.clone = private_collection(run)
            self.clone.load_state_dict(load_envelope(path))
            self.work.update(clone_compared=0, clone_target=clone_counts[which])
            self.work["evidence"].append(self._record_path(path))
            self.save_boundary("restored-cohort-boundary")
        if compare_forks and not run.prefix.closed:
            self._compare_forks(b, run)
        self.budget.check()
        event = run.step(before_step=lambda: self.budget.debit_environment("trajectory"))
        if self.parity is not None and event["stage"] == "prefix":
            record = event["event"]["audit"]["record"]
            self.budget.debit_environment("clone")
            _, reward, done, info = self.parity.step(np.array(record["action"], dtype=np.float64))
            expected = run.env._cohort_prefix_state if run.prefix.closed else run.env.state_dict()
            if (reward != record["raw_reward"] or done != run.prefix.closed
                    or evidence_digest(info) != evidence_digest(event["event"]["info"])
                    or evidence_digest(self.parity.state_dict()) != evidence_digest(expected)):
                raise ValueError("original-engine prefix parity differs")
            self.work["parity_rows"].append(dict(step=run.prefix.index,
                action_sha256=evidence_digest(record["action"]), state_sha256=evidence_digest(expected),
                raw_reward=reward, info_sha256=evidence_digest(info), identical=True))
        if self.clone is not None:
            other = self.clone.step(before_step=lambda: self.budget.debit_environment("clone"))
            if (state_digest(encode_arrays(event)) != state_digest(encode_arrays(other))
                    or state_digest(run.state_dict()) != state_digest(self.clone.state_dict())):
                raise ValueError("restored cohort event/state/RNG differs")
            self.work["clone_compared"] += 1
            if self.work["clone_compared"] == self.work["clone_target"]:
                self.clone = None
                self.work["clones_completed"] += 1
        if not run.closed:
            return False
        if clone and self.work["clones_completed"] != 2:
            raise ValueError("missing prefix/tail restoration comparisons")
        if selection != "sample" and state_digest(run.learner.state_dict()) != self.work["episode_initial_kernel"]:
            raise ValueError("deterministic collection changed model or RNG")
        if self.parity is not None:
            if len(self.work["parity_rows"]) != prefix_steps:
                raise ValueError("incomplete original-engine parity trace")
            path = self._json(f"payload/preflight/{name}/prefix-parity.json", self.work["parity_rows"])
            self.work["evidence"].append(self._record_path(path))
        index = self.recorder.finish(run)
        self.raw_index.append(index)
        self.work["indexes"].append(index)
        if split == "test":
            self.test_index.append(index)
        self.live_session = self.clone = self.recorder = self.parity = None
        self.work["cursor"] += 1
        return True

    def _train_step(self, job):
        role, block = job.split("/")
        b, cfg = int(block[5:]), self.config["continuation"]
        model_key = self._key(b, role)
        if self.continuation is None:
            self.work["model_key"] = model_key
            settings = dict(episodes_per_model=cfg["episodes_per_arm_per_block"],
                episodes_per_rollout=cfg["episodes_per_rollout"], max_updates=cfg["rollouts_per_arm_per_block"],
                epochs=cfg["epochs"], gae_lambda=self.config["objective"]["gae_lambda"])
            def factory(kernel, episode, seed, *, restore_state=None):
                name = f"training/{model_key}/episode{episode:02d}"
                if restore_state is not None:
                    cached = next((s for s in (self.live_session, self.closed_session)
                                   if s is not None and s.trajectory_id == name), None)
                    if cached is None:
                        raise ValueError("restore requires an already-owned cohort")
                    restored = private_collection(cached)
                    restored.prefix.learner = kernel
                    return restored
                return self._make_collection(b, role, episode, seed, "training", "sample", name, kernel)
            def finish(collection):
                index = self.recorder.finish(collection)
                self.work["indexes"].append(index)
                self.raw_index.append(index)
                self.closed_session = private_collection(collection)
                return index
            self.continuation = CohortContinuation(self.models[model_key], self.budget, settings, enabled=True,
                role="bc_continue" if role == "bc_continue" else "ppo", scope=job,
                seeds=self._seeds(b, "training"), prefix_steps=self.proposal["enrollment_steps"],
                accounting_steps=self.proposal["accounting_steps"], session_factory=factory, finish_collection=finish)
            self.save_boundary("continuation-start")
        run = self.continuation
        if run.update_due:
            self.work["updates"].append(run.update())
            self.models[model_key] = run.kernel
            self.save_boundary("update-complete")
            self.emit_status("training_update", model=model_key, episodes=run.episode, updates=run.updates)
        elif not run.done:
            if run.active is None:
                run.start_episode()
            run.step()
            if run.active is None:
                self.recorder = self.live_session = None
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
            raise ValueError("exact active cohort phase required")
        job, done = self.work["job"], False
        if job == "runtime_input_binding":
            done = self._bind_block()
        elif job.split("/")[0] in self.proposal["training_arms"]:
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
            self._json("launcher/closure.json", dict(engineering_fixture=self.engineering_only,
                scientific_execution=not self.engineering_only,
                payload_unchanged=inventory(self.root / "payload") == self.payload_seal, automatic_followon=False))
            done = True
        else:
            descriptors = self._descriptors(job)
            if not descriptors:
                raise ValueError("unknown cohort phase")
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
        self._json("payload/independent-target-verification.json",
                   self.target_verifier(self.root, self.raw_index, self.config, self.streams))
        DynamicCandidateCampaign._verify(self)

    def checkpoint_state(self):
        state = decode_arrays(super().checkpoint_state())
        state.update(format="cohort-campaign-v1", parity=None if self.parity is None else self.parity.state_dict())
        return encode_arrays(state)

    def restore_checkpoint_state(self, saved):
        self._check()
        state, current = decode_arrays(saved), decode_arrays(self.checkpoint_state())
        fixed = ("format", "config_sha256", "streams_sha256", "sequence", "model_paths", "failure", "payload_seal")
        if set(state) != set(current) or any(state[k] != current[k] for k in fixed) or state["failure"] is not None:
            raise ValueError("restore changed phase/seals/contract or cleared failure")
        assert_dynamic_budget_ancestor(state["budget"], self.budget)
        sequence = CohortSequence(self.base, self.proposal, self.root, enabled=True)
        sequence.load_state_dict(state["sequence"])
        self.backend.assert_restore_compatible(state["backend"])
        for name in ("test_index", "raw_index", "templates", "initializers", "demonstrations", "qualification", "qualified"):
            if state_digest(state[name]) != state_digest(current[name]):
                raise ValueError("immutable imported/raw evidence changed")
        if self.models.keys() != state["models"].keys():
            raise ValueError("model ownership changed")
        models = copy.deepcopy(self.models)
        for key, kernel in models.items():
            kernel.load_state_dict(state["models"][key])
        for name in ("live_session", "closed_session", "clone", "continuation", "parity", "recorder"):
            if (getattr(self, name) is None) != (state[name] is None):
                raise ValueError("live cohort ownership changed")
        if self.recorder is not None:
            self.recorder.assert_prefix(state["recorder"])
        live, closed, clone = (private_collection(getattr(self, name))
                               for name in ("live_session", "closed_session", "clone"))
        if closed is not None:
            closed.load_state_dict(encode_arrays(state["closed_session"]))
        continuation = copy.copy(self.continuation)
        if continuation is not None:
            continuation.load_state_dict(encode_arrays(state["continuation"]))
            models[state["work"]["model_key"]] = continuation.kernel
            live = continuation.active
        elif live is not None:
            live.load_state_dict(encode_arrays(state["live_session"]))
        if clone is not None:
            clone.load_state_dict(encode_arrays(state["clone"]))
        parity = copy.deepcopy(self.parity)
        if parity is not None:
            parity.load_state_dict(state["parity"])
            if evidence_digest(parity.state_dict()) != evidence_digest(state["parity"]):
                raise ValueError("original engine parity restore differs")
        rng = global_rng_state()
        try:
            restore_global_rng(state["global_rng"])
        finally:
            restore_global_rng(rng)
        self.models, self.sequence, self.continuation = models, sequence, continuation
        self.live_session, self.closed_session, self.clone, self.parity = live, closed, clone, parity
        if live is not None and self.recorder is not None:
            live.record_prefix, live.record_tail = self.recorder.record_prefix, self.recorder.record_tail
            live.finish_prefix = self.recorder.finish_prefix
            self.recorder.prefix.session = live.prefix
        self.work = copy.deepcopy(state["work"])
        self.status_serial = max(self.status_serial, state["status_serial"])
        self.boundary_serial = max(self.boundary_serial, state["boundary_serial"])
        restore_global_rng(state["global_rng"])
        self._check()
