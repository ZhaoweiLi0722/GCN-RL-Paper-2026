"""Serial dynamic-policy engineering runner with explicit external ownership.

This is not an authorization source. Fixture execution needs an invented backend;
scientific execution additionally requires the source-bound admission object
from the exclusive execution wrapper. No approval is bundled with this module.
Every advance performs one recoverable operation, never retries a failed one.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import traceback

import numpy as np

from src.rl.candidate_patient_session import save_envelope, load_envelope
from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import EpisodeRecorder, write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_continuation import DynamicCandidateContinuation
from src.rl.dynamic_candidate_factory import (
    dynamic_template, dynamic_initializer, fork_dynamic_initializer,
    qualify_dynamic_path, qualify_dynamic_outcomes,
)
from src.rl.dynamic_candidate_resources import (
    DynamicCandidateBudget, dynamic_budget_plan, assert_dynamic_budget_ancestor,
    read_dynamic_ledger,
)
from src.rl.dynamic_candidate_rollout import evaluate_dynamic_policy
from src.rl.dynamic_candidate_sequence import DynamicPilotSequence
from src.rl.dynamic_candidate_session import dynamic_context_from_public
from src.rl.dynamic_candidate_verification import read_dynamic_raw_episodes, verify_dynamic_raw_bundle
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import encode_arrays, decode_arrays
from src.utils.research_archive import create_archive, copy_verified, inventory


ROLES = ("own_frozen", "own_ppo", "own_bc_continue", "r4", "full_mdl2")


class DynamicCandidateCampaign:
    """An incremental fixed-phase coordinator; fixture execution is opt-in.

    Recovery is an in-owner checkpoint interface, not a cold-relaunch permit:
    the original backend, recorder handles and irreversible budget remain owned
    by the calling process. Later raw rows cannot be truncated, a failed attempt
    cannot be restored, and a saved ledger never replaces the live ledger.
    """

    def __init__(self, root, config, streams, budget, backend, *, enabled=False,
                 engineering_only=True, recorder=EpisodeRecorder,
                 reader=read_dynamic_raw_episodes, verifier=verify_dynamic_raw_bundle,
                 final_lock_check=lambda: None, execution_admission=None):
        if enabled is not True or type(engineering_only) is not bool:
            raise ValueError("explicit execution mode required")
        if engineering_only:
            if getattr(backend, "engineering_fixture", False) is not True or execution_admission is not None:
                raise ValueError("patient or unclassified backends are forbidden in engineering mode")
        else:
            from src.rl.dynamic_candidate_execution import ScientificAdmission
            if type(execution_admission) is not ScientificAdmission:
                raise PermissionError("scientific execution requires the exclusive approved admission")
            execution_admission.bind_campaign(root, config, streams, budget, backend)
        if type(budget) is not DynamicCandidateBudget or budget.plan != dynamic_budget_plan(config):
            raise ValueError("exact external dynamic budget required")
        self.root, self.config, self.streams = Path(root).resolve(), copy.deepcopy(config), copy.deepcopy(streams)
        self.budget, self.backend = budget, backend
        self.recorder_type, self.reader, self.verifier = recorder, reader, verifier
        self.final_lock_check = final_lock_check
        self.engineering_only = engineering_only
        self.sequence = DynamicPilotSequence(config, self.root, enabled=True)
        self.templates, self.initializers, self.models, self.model_paths = {}, {}, {}, {}
        self.demonstrations, self.qualification, self.qualified = {}, {}, {}
        self.test_index, self.raw_index = [], []
        self.work, self.continuation, self.live_session, self.clone, self.recorder = None, None, None, None, None
        self.closed_session = None
        self.failure, self.payload_seal = None, None
        self.status_serial = self.boundary_serial = 0

    def _check(self):
        if self.failure is not None or self.sequence.failure is not None:
            raise ValueError("failed campaign is terminal; no retry or rewind")
        self.budget.check()
        if self.payload_seal is not None and inventory(self.root / "payload") != self.payload_seal:
            raise ValueError("archived payload changed")

    def _json(self, relative, value):
        self._check()
        path = self.root / relative
        write_json_once(path, value)
        self._check()
        return path

    def _state_file(self, relative, value):
        self._check()
        path = self.root / relative
        save_envelope(path, value)
        self._check()
        return path

    def _record_path(self, path):
        return Path(path).relative_to(self.root).as_posix()

    def emit_status(self, state, **extra):
        path = self._json(f"launcher/status/{self.status_serial:06d}.json", {
            "status": state, "engineering_fixture": self.engineering_only,
            "scientific_execution": not self.engineering_only, "job": self.sequence.active or self.sequence.next_job,
            "completed_jobs": len(self.sequence.completed), "budget": self.budget.snapshot(), **extra})
        self.status_serial += 1
        return path

    def checkpoint_state(self):
        return encode_arrays({"format": "dynamic-campaign-v1", "config_sha256": digest(self.config),
            "streams_sha256": digest(self.streams), "sequence": self.sequence.state_dict(),
            "budget": self.budget.snapshot(), "global_rng": global_rng_state(),
            "templates": {k: v.state_dict() for k, v in self.templates.items()},
            "initializers": {k: v.state_dict() for k, v in self.initializers.items()},
            "models": {k: v.state_dict() for k, v in self.models.items()},
            "model_paths": self.model_paths, "demonstrations": self.demonstrations,
            "qualification": self.qualification, "qualified": self.qualified,
            "test_index": self.test_index, "raw_index": self.raw_index, "work": self.work,
            "live_session": None if self.live_session is None else self.live_session.state_dict(),
            "closed_session": None if self.closed_session is None else self.closed_session.state_dict(),
            "clone": None if self.clone is None else self.clone.state_dict(),
            "continuation": None if self.continuation is None else self.continuation.state_dict(),
            "recorder": None if self.recorder is None or self.recorder.closed or self.recorder.failed
                        else self.recorder.snapshot(),
            "status_serial": self.status_serial, "boundary_serial": self.boundary_serial,
            "backend": self.backend.state_dict() if callable(getattr(self.backend, "state_dict", None)) else None,
            "failure": self.failure, "payload_seal": self.payload_seal})

    def save_boundary(self, label):
        self._check()
        serial = self.boundary_serial
        self.boundary_serial += 1
        return self._state_file(f"launcher/recovery/{serial:06d}-{label}.pt", self.checkpoint_state())

    def restore_checkpoint_state(self, saved):
        self._check()
        state, current = decode_arrays(saved), decode_arrays(self.checkpoint_state())
        fixed = ("format", "config_sha256", "streams_sha256", "sequence", "model_paths", "failure", "payload_seal")
        if set(state) != set(current) or any(state[k] != current[k] for k in fixed):
            raise ValueError("restore changed phase, seals or scientific contract")
        if state["failure"] is not None:
            raise ValueError("terminal failure evidence is not resumable")
        assert_dynamic_budget_ancestor(state["budget"], self.budget)
        sequence = DynamicPilotSequence(self.config, self.root, enabled=True)
        sequence.load_state_dict(state["sequence"])
        if state["backend"] is not None:
            if not callable(getattr(self.backend, "assert_restore_compatible", None)):
                raise ValueError("backend does not expose irreversible restore compatibility")
            self.backend.assert_restore_compatible(state["backend"])
        elif current["backend"] is not None:
            raise ValueError("backend checkpoint ownership changed")
        for name in ("test_index", "raw_index"):
            if state[name] != current[name]:
                raise ValueError("cannot rewind completed raw evidence")
        replacements = {}
        for group in ("templates", "initializers", "models"):
            owners = getattr(self, group)
            if owners.keys() != state[group].keys():
                raise ValueError("restore changed model ownership")
            replacements[group] = copy.deepcopy(owners)
            for key, kernel in replacements[group].items():
                self.budget.check()
                kernel.load_state_dict(state[group][key])
        for name in ("live_session", "closed_session", "clone", "continuation"):
            owner = getattr(self, name)
            if (owner is None) != (state[name] is None):
                raise ValueError("restore changed live collection ownership")
        if (self.recorder is None) != (state["recorder"] is None):
            raise ValueError("restore changed raw recorder ownership")
        if self.recorder is not None:
            self.recorder.assert_prefix(state["recorder"])
            if self.recorder.count != state["recorder"]["rows"]:
                raise ValueError("cannot rewind persisted raw rows")
        continuation = copy.copy(self.continuation)
        live, clone = copy.deepcopy(self.live_session), copy.deepcopy(self.clone)
        closed = copy.deepcopy(self.closed_session)
        if closed is not None:
            self.budget.check()
            closed.load_state_dict(encode_arrays(state["closed_session"]))
        if continuation is not None:
            self.budget.check()
            # The legacy continuation reconstructs sessions through a factory.
            # Reuse the two already-owned collectors here: recovery must not
            # construct/reset an additional patient environment or spend a seed.
            original_factory = continuation.factory
            cached = {s.trajectory_id: s for s in (live, closed) if s is not None}
            def restored_factory(kernel, episode, seed):
                path = f"training/{state['work']['model_key']}/episode{episode:02d}"
                if path not in cached or seed != continuation.seeds[episode]:
                    raise ValueError("restore requires the matching already-owned collector")
                restored = copy.deepcopy(cached[path])
                restored.learner = kernel
                return restored
            continuation.factory = restored_factory
            try:
                continuation.load_state_dict(encode_arrays(state["continuation"]))
            finally:
                continuation.factory = original_factory
            replacements["models"][state["work"]["model_key"]] = continuation.kernel
            live = continuation.active
        elif live is not None:
            self.budget.check()
            live.load_state_dict(encode_arrays(state["live_session"]))
        if clone is not None:
            self.budget.check()
            clone.load_state_dict(encode_arrays(state["clone"]))
        # Validate the global RNG privately before committing any owner.
        rng = global_rng_state()
        try:
            restore_global_rng(state["global_rng"])
        finally:
            restore_global_rng(rng)
        for group, owners in replacements.items():
            setattr(self, group, owners)
        self.continuation, self.live_session, self.clone = continuation, live, clone
        self.sequence = sequence
        self.closed_session = closed
        if self.recorder is not None:
            self.recorder.session = live
        for name in ("demonstrations", "qualification", "qualified", "work"):
            setattr(self, name, copy.deepcopy(state[name]))
        self.status_serial = max(self.status_serial, state["status_serial"])
        self.boundary_serial = max(self.boundary_serial, state["boundary_serial"])
        restore_global_rng(state["global_rng"])
        self._check()

    def begin_next(self):
        self._check()
        if self.work is not None or self.sequence.active is not None or self.budget.active is not None:
            raise ValueError("another phase is already active")
        job = self.sequence.next_job
        if job is None:
            return None
        self.budget.begin(job)
        try:
            self.sequence.begin(job)
            self.work = {"job": job, "cursor": 0, "indexes": [], "evidence": [], "updates": []}
            self.emit_status("running")
            self.save_boundary("phase-start")
        except BaseException as error:
            self.fail(error)
            raise
        return job

    def _seeds(self, block, split):
        return self.streams["environment"][str(block)][split]

    @staticmethod
    def _key(block, role=None):
        return f"block{block}/graph" + (f"/{role}" if role is not None else "")

    def _session(self, descriptor):
        block, role, world, seed, split, selection, name = descriptor
        if split == "test":
            self.sequence.check_seals()
        if role == "preflight" or split == "demonstration":
            kernel = copy.deepcopy(self.templates[self._key(block)])
        elif split == "qualification":
            kernel = copy.deepcopy(self.initializers[self._key(block)])
        else:
            key = self._key(block, role if role.startswith("own_") else "own_frozen")
            kernel = copy.deepcopy(self.models[key])
            if split == "test" and state_digest(kernel.state_dict()) != state_digest(load_envelope(self.root / self.model_paths[key])):
                raise ValueError("evaluation owner differs from sealed model")
        self.budget.check()
        run = self.backend.session(block, kernel, seed, trajectory=name, split=split, selection=selection)
        self.live_session = run
        self.recorder = self.recorder_type(self.root, f"payload/episodes/{name}", run, self.config,
            block=block, representation="reference" if role in ("r4", "full_mdl2") else "graph",
            role=role, world_index=world, seed=seed)
        self.work["episode_initial_kernel"] = state_digest(kernel.state_dict())
        self.work["clone_compared"] = 0

    def _compare_forks(self, block, run):
        self.budget.check()
        observation, bank = dynamic_context_from_public(run.producer, run.reference,
            run.env.observation(), run.options, run.expected_token)
        choices = []
        for role in ROLES[:3]:
            self.budget.check()
            kernel = self.models[self._key(block, role)]
            result = evaluate_dynamic_policy(kernel.policy, observation, bank, kernel.contract)
            choices.append((int(np.argmax(result.log_probs)), result.log_probs))
        if any(c != choices[0] for c in choices[1:]):
            raise ValueError("same-start greedy fork probabilities or choices differ")

    def _collect_step(self, descriptor, *, clone=False, compare_forks=False):
        if self.live_session is None:
            self._session(descriptor)
        run = self.live_session
        block, role, world, seed, split, selection, name = descriptor
        midpoint = self.config["objective"]["horizon"] // 2
        clone_count = min(4, self.config["objective"]["horizon"] - midpoint)
        if clone and run.index == midpoint and self.clone is None:
            self.budget.check()
            saved = run.state_dict()
            path = self._state_file(f"payload/preflight/{name}/midpoint.pt", saved)
            self.clone = copy.deepcopy(run)
            self.budget.check()
            self.clone.load_state_dict(load_envelope(path))
            self.work["evidence"].append(self._record_path(path))
            self.save_boundary("restored-midpoint")
        if compare_forks:
            self._compare_forks(block, run)
        self.budget.check()
        event = run.step(before_step=lambda: self.budget.debit_environment("trajectory"))
        self.budget.check()
        self.recorder.append(event)
        if clone and self.clone is not None and self.work["clone_compared"] < clone_count:
            self.budget.check()
            other = self.clone.step(before_step=lambda: self.budget.debit_environment("clone"))
            if (state_digest(encode_arrays(event)) != state_digest(encode_arrays(other))
                    or state_digest(run.state_dict()) != state_digest(self.clone.state_dict())):
                raise ValueError("restored preflight decision, reward, patient state or RNG differs")
            self.work["clone_compared"] += 1
        if not run.closed:
            return False
        if clone and self.work["clone_compared"] != clone_count:
            raise ValueError("missing mandatory restored preflight steps")
        if selection != "sample" and state_digest(run.learner.state_dict()) != self.work["episode_initial_kernel"]:
            raise ValueError("deterministic collection changed model or private RNG")
        self.budget.check()
        index = self.recorder.finish()
        self.raw_index.append(index)
        self.work["indexes"].append(index)
        if split == "demonstration":
            self.demonstrations.setdefault(block, []).extend(copy.deepcopy(run.examples))
        elif split == "qualification":
            self.qualification.setdefault(block, {}).setdefault(role, []).extend(copy.deepcopy(run.examples))
        elif split == "test":
            self.test_index.append(index)
        self.live_session, self.clone, self.recorder = None, None, None
        self.work["cursor"] += 1
        # Raw rows and each collector are already durable. Full owner snapshots
        # belong at phase/block/rollout boundaries, not twice per episode with
        # increasingly duplicated demonstrations and model optimizer payloads.
        if ((split == "demonstration" and world + 1 == len(self._seeds(block, "demonstration")))
                or (split == "qualification" and role == "initializer_greedy"
                    and world + 1 == len(self._seeds(block, "qualification")))):
            self.save_boundary("collection-block-complete")
        return True

    def _descriptors(self, job):
        rows = []
        blocks = self.config["blocks"]
        if job.startswith("final_evaluation/"):
            _, block, role = job.split("/")
            b = int(block[5:])
            selection = "reference" if role == "r4" else "anchor" if role == "full_mdl2" else "greedy"
            return [(b, role, w, seed, "test", selection, f"evaluation/block{b}/{role}/world{w:02d}")
                    for w, seed in enumerate(self._seeds(b, "test"))]
        for b in blocks:
            if job == "prototype_preflight":
                rows.append((b, "preflight", 0, self._seeds(b, "prototype_preflight")[0],
                             "preflight", "sample", f"preflight/prototype/block{b}"))
            elif job == "initialization_collection":
                rows.extend((b, "r4", w, seed, "demonstration", "reference", f"demonstration/block{b}/world{w:02d}")
                            for w, seed in enumerate(self._seeds(b, "demonstration")))
            elif job == "qualification":
                rows.extend((b, role, w, seed, "qualification", selection, f"qualification/block{b}/{role}/world{w:02d}")
                            for w, seed in enumerate(self._seeds(b, "qualification"))
                            for role, selection in (("r4", "reference"), ("initializer_greedy", "greedy")))
            elif job == "same_start_preflight":
                rows.extend((b, role, 0, self._seeds(b, "fork_preflight")[0], "preflight",
                             "reference" if role == "r4" else "anchor" if role == "full_mdl2"
                             else "greedy" if role == "own_frozen" else "sample",
                             f"preflight/fork/block{b}/{role}") for role in ROLES)
        return rows

    def _qualify(self):
        self.budget.check()
        _, outcomes = self.reader(self.root, self.work["indexes"], self.config)
        for b in self.config["blocks"]:
            key = self._key(b)
            owner = self.initializers[key]
            paths = {role: qualify_dynamic_path(owner, self.qualification[b][role], self.config,
                                               before_forward=self.budget.check)
                     for role in ("r4", "initializer_greedy")}
            result = qualify_dynamic_outcomes([r for r in outcomes if r["block"] == b], self.config, b)
            self.qualified[key] = {"passed": result["passed"] and all(r["passed"] for r in paths.values()),
                "paths": paths, "outcomes": result, "kernel_sha256": state_digest(owner.state_dict())}
        self._json("payload/qualification.json", self.qualified)
        if not all(row["passed"] for row in self.qualified.values()):
            raise ValueError("qualification failed; attempt closed without continuation or test")
        for b in self.config["blocks"]:
            key = self._key(b)
            self.budget.check()
            forks = fork_dynamic_initializer(self.initializers[key], self.qualified[key], self.config, self.streams, b)
            for role, owner in forks.items():
                name = self._key(b, role)
                self.models[name] = owner
                path = self._state_file(f"payload/models/{name}/initial.pt", owner.state_dict())
                self.work["evidence"].append(self._record_path(path))
                if role == "own_frozen":
                    self.model_paths[name] = self._record_path(path)

    def _train_step(self, job):
        phase, block = job.split("/")
        b = int(block[5:])
        role = "ppo" if phase == "ppo_continuation" else "bc_continue"
        model_key = self._key(b, "own_" + role)
        cfg = self.config["continuation"]
        if self.continuation is None:
            self.work["model_key"] = model_key
            settings = {"episodes_per_model": cfg["episodes_per_arm_per_block"],
                        "episodes_per_rollout": cfg["episodes_per_rollout"],
                        "max_updates": cfg["rollouts_per_arm_per_block"], "epochs": cfg["epochs"],
                        "gae_lambda": self.config["objective"]["gae_lambda"]}
            def factory(kernel, episode, seed):
                self.budget.check()
                return self.backend.session(b, kernel, seed, trajectory=f"training/{model_key}/episode{episode:02d}",
                                            split="training", selection="sample")
            self.continuation = DynamicCandidateContinuation(self.models[model_key], self.budget, settings,
                enabled=True, role=role, scope=job, seeds=self._seeds(b, "training"),
                horizon=self.config["objective"]["horizon"], session_factory=factory)
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
                    block=b, representation="graph", role="own_" + role, world_index=episode,
                    seed=run.seeds[episode])
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
        self.continuation = None
        self.closed_session = None
        return True

    def _verify(self):
        self.sequence.check_seals()
        self.final_lock_check()
        self.budget.check()
        if len(self.raw_index) != self.config["totals"]["main_full_episodes"]:
            raise ValueError("missing prescribed complete raw episodes")
        files, outcomes = self.reader(self.root, self.raw_index, self.config)
        result = self.verifier(self.root, self.test_index, self.config, self.streams)
        self.budget.check()
        ledger = read_dynamic_ledger(self.budget.path)
        if ledger["counts"] != self.budget.counts or ledger["owner_counts"] != self.budget.owner_counts:
            raise ValueError("durable budget does not match live counts")
        for phase, caps in self.budget.plan["phases"].items():
            for owner in ("trajectory", "clone", "actor", "critic"):
                if ledger["owner_counts"].get(f"phase/{phase}:{owner}", 0) != caps[owner]:
                    raise ValueError("incomplete prescribed phase/owner work")
        self._json("payload/independent-verification.json", result)
        self._json("payload/all-episode-verification.json", {"files": files, "outcomes": outcomes})
        self._json("payload/compute-accounting.json", {"owner_counts": ledger["owner_counts"],
                    "counts": ledger["counts"], "phase_seconds": ledger["phase_seconds"],
                    "engineering_fixture": self.engineering_only, "scientific_performance_claim": False})
        self.budget.check()
        copy_verified(self.budget.path, self.root / "payload/budget-through-verification.jsonl")

    def _archive(self):
        self.final_lock_check()
        self._json("payload/artifact-inventory.json", inventory(self.root / "payload"))
        self.budget.check()
        result = create_archive(self.root / "payload", self.root / "archives/completed-payload.tar.gz")
        self.payload_seal = result["files"]
        self._json("launcher/archive-receipt.json", {"archive": result, "local_archive_verified": True,
                   "dropbox_exported": False, "cloud_sync_verified": False, "howard_access_verified": False})

    def advance(self):
        """Perform one active-phase operation and latch any execution failure."""
        try:
            return self._advance()
        except BaseException as error:
            self.fail(error)
            raise

    def _advance(self):
        self._check()
        if self.work is None or self.sequence.active != self.work["job"] or self.budget.active != self.work["job"]:
            raise ValueError("begin the exact next phase before advancing")
        job = self.work["job"]
        done = False
        if job == "runtime_input_binding":
            b = self.config["blocks"][self.work["cursor"]]
            self.budget.check()
            effective = self.backend.prepare(b, self._seeds(b, "prototype_preflight")[0])
            self.templates[self._key(b)] = dynamic_template(self.backend.producer(b), self.config, self.streams, b)
            self._json(f"payload/binding/block{b}.json", effective)
            self.work["cursor"] += 1
            self.save_boundary("runtime-owner-bound")
            done = self.work["cursor"] == len(self.config["blocks"])
        elif job.startswith("initialization_fit/"):
            b = int(job.split("/")[1][5:])
            key = self._key(b)
            owner = dynamic_initializer(self.templates[key], self.config, self.streams, b)
            self.initializers[key] = owner
            self.save_boundary("initialization-start")
            self.budget.check()
            self.work["updates"].append(owner.fit(self.demonstrations[b],
                replacement_steps=self.config["initialization"]["actor_adam_calls_per_block"],
                before_step=lambda: self.budget.debit_optimizer("actor"), before_compute=self.budget.check))
            self._state_file(f"payload/initialization/{key}/final.pt", owner.state_dict())
            done = True
        elif job.startswith(("ppo_continuation/", "bc_continuation/")):
            done = self._train_step(job)
        elif job == "all_model_seal":
            self.sequence.seal_models(self.model_paths)
            self._json("payload/model-seals.json", self.sequence.seals)
            done = True
        elif job == "saved_data_verification_analysis":
            self._verify()
            done = True
        elif job == "local_archive_verification":
            self._archive()
            done = True
        elif job == "supervisor_dispatch_terminal_closure":
            self.sequence.check_seals()
            self.final_lock_check()
            self._json("launcher/closure.json", {"engineering_fixture": self.engineering_only,
                "payload_unchanged": inventory(self.root / "payload") == self.payload_seal,
                "automatic_followon": False, "scientific_execution": not self.engineering_only})
            done = True
        else:
            descriptors = self._descriptors(job)
            if not descriptors:
                raise ValueError("unknown fixed dynamic campaign phase")
            descriptor = descriptors[self.work["cursor"]]
            self._collect_step(descriptor, clone=job in ("prototype_preflight", "same_start_preflight"),
                compare_forks=job == "same_start_preflight" and descriptor[1] == "own_frozen")
            done = self.work["cursor"] == len(descriptors)
            if done and job == "qualification":
                self._qualify()
        if done:
            self._finish_phase()
        return done

    def _finish_phase(self):
        job = self.work["job"]
        location = "launcher/phase-evidence" if self.payload_seal is not None else "payload/phases"
        path = self._json(f"{location}/{job}.json", self.work)
        evidence = self.work["evidence"] + [self._record_path(path)]
        self.save_boundary("phase-work-complete")
        self.emit_status("phase_completed", completed_job=job)
        self.sequence.finish(evidence)
        if job == "supervisor_dispatch_terminal_closure":
            self._json("launcher/completed.json", {"status": "completed", "exit_code": 0,
                "engineering_fixture": self.engineering_only, "scientific_execution": not self.engineering_only,
                "sequence": self.sequence.state_dict(), "budget": self.budget.snapshot(),
                "automatic_followon": False, "requires_successful_supervisor_receipt": True})
        self.budget.check()
        self.budget.finish()
        self.work = None

    def dispatch(self, job):
        if self.sequence.active != job:
            raise ValueError("dispatch cannot open or skip phases")
        while self.work is not None:
            self.advance()

    def fail(self, error):
        if self.failure is not None:
            return
        self.failure = {"error_type": type(error).__name__, "message": str(error)}
        self.sequence.fail(repr(error))
        failure = {"status": "failed", "error": self.failure, "traceback": traceback.format_exc(),
                   "sequence": self.sequence.state_dict(), "budget": self.budget.snapshot(), "retry_permitted": False}
        try:
            save_envelope(self.root / "launcher/failure-state.pt", self.checkpoint_state())
        except BaseException as preservation_error:
            failure["checkpoint_preservation_error"] = repr(preservation_error)
        if self.recorder is not None:
            self.recorder.close_partial()
        write_json_once(self.root / "launcher/failure.json", failure)
        self.budget.close()

    def run(self):
        try:
            while self.sequence.next_job is not None:
                if self.work is None:
                    self.begin_next()
                self.dispatch(self.sequence.active)
            return 0
        except BaseException as error:
            self.fail(error)
            return 1
        finally:
            self.budget.close()
