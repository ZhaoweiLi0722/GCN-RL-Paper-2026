"""One bounded P1 campaign, bound only by the locked scientific entrypoint.

The backend is injectable for invented-fixture acceptance. Importing this module
does not create an environment. Recovery validates against the live, irreversible
ledger; it is not a failed-experiment retry or a process-restart command.
"""

from __future__ import annotations

import copy
from dataclasses import fields
import json
from pathlib import Path
import random
import traceback

import numpy as np
import torch

from src.rl.candidate_patient_session import CandidatePatientSession, context_from_public, save_envelope, load_envelope
from src.rl.candidate_pilot_driver import (CandidateContinuation, PilotSequence, assert_budget_ancestor,
    candidate_prototype, initialization_kernel, fork_initializer, qualify_initializer, file_record)
from src.rl.candidate_pilot_recording import EpisodeRecorder, write_json_once
from src.rl.candidate_pilot_resources import digest, read_ledger
from src.rl.candidate_pilot_compatibility import inspect_patient_layout, require_supported_layouts
from src.rl.candidate_pilot_verification import verify_raw_bundle
from src.rl.candidate_ppo_kernel import CandidatePPOKernel, CandidatePPOSettings
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import encode_arrays, decode_arrays
from src.utils.research_archive import create_archive, copy_verified, inventory


def global_rng_state():
    return encode_arrays({"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch": torch.get_rng_state()})


def restore_global_rng(state):
    state = decode_arrays(state)
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state(state["numpy"])
    torch.Generator().set_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


class PatientBackend:
    """Real factory. Called only after exclusive claim and budget scope entry."""

    def __init__(self, workspace, config, streams):
        self.workspace, self.config, self.streams = Path(workspace), config, streams
        self.contexts = {}

    def prepare(self, block, seed):
        from src.rl.experiment import build_env
        from src.rl.frozen_value_probe import assert_scenario
        from src.rl.patient_replay_collector import PatientObservationProducer
        from src.rl.strict_frozen_policy import StrictFrozenPolicy
        spec = self.config["reference"]
        path = self.workspace / spec["directory"]
        config_path = path / spec["config_template"].format(seed=block)
        runtime = json.loads(config_path.read_text())
        message_graph = self.config.get("candidate_message_graph", "shared_relations")
        require_supported_layouts(inspect_patient_layout(runtime, self.config["objective"], message_graph=message_graph))
        before = global_rng_state()
        try:
            reference = StrictFrozenPolicy(path / spec["policy_template"].format(seed=block), config_path,
                checkpoint_sha256=spec["locks"][str(block)]["policy"],
                config_sha256=spec["locks"][str(block)]["config"], device="cpu")
            env = build_env(runtime, seed)
            effective = assert_scenario(env, runtime, self.config["objective"]["scenario"])
            producer = PatientObservationProducer(env, enabled=True,
                gamma=self.config["objective"]["gamma"], reward_scale=self.config["objective"]["reward_scale"],
                message_graph=message_graph)
        finally:
            restore_global_rng(before)
        self.contexts[block] = (runtime, producer, reference)
        return effective

    def producer(self, block):
        return self.contexts[block][1]

    def session(self, block, kernel, seed, *, trajectory, split, selection):
        from src.rl.experiment import build_env
        runtime, producer, reference = self.contexts[block]
        env = build_env(runtime, seed)
        return CandidatePatientSession(env, producer, reference, kernel, enabled=True,
            options=self.config["candidate_support"]["options"], trajectory_id=trajectory,
            split=split, selection=selection, source_id=self.config["rng"]["namespace"])


class PilotCampaign:
    def __init__(self, root, config, streams, budget, backend, *, recorder=EpisodeRecorder,
                 verifier=verify_raw_bundle, final_lock_check=lambda: None):
        self.root, self.config, self.streams = Path(root).resolve(), copy.deepcopy(config), copy.deepcopy(streams)
        self.budget, self.backend = budget, backend
        self.recorder_type, self.verifier, self.final_lock_check = recorder, verifier, final_lock_check
        self.sequence = PilotSequence(config, self.root)
        self.templates, self.initializers, self.models = {}, {}, {}
        self.demonstrations, self.qualification, self.qualified, self.test_index = {}, {}, {}, []
        self.model_paths, self.recorder, self.continuation, self.live_session = {}, None, None, None
        self.status_serial = 0

    def emit_status(self, status, **extra):
        self.budget.check()
        path = self.root / "launcher" / "status" / f"{self.status_serial:06d}.json"
        write_json_once(path, {"status": status, "job": self.sequence.active or self.sequence.next_job,
            "completed_jobs": len(self.sequence.completed), "total_jobs": len(self.sequence.jobs),
            "budget": self.budget.snapshot(), **extra})
        self.status_serial += 1
        print(json.dumps({"status": status, "job": self.sequence.active or self.sequence.next_job,
                          "counts": self.budget.counts, **extra}, sort_keys=True), flush=True)

    def checkpoint_state(self):
        return encode_arrays({"format": "p1-campaign-v1", "config_sha256": digest(self.config),
            "streams_sha256": digest(self.streams), "sequence": self.sequence.state_dict(),
            "budget": self.budget.snapshot(), "global_rng": global_rng_state(),
            "templates": {k: v.state_dict() for k, v in self.templates.items()},
            "initializers": {k: v.state_dict() for k, v in self.initializers.items()},
            "models": {k: v.state_dict() for k, v in self.models.items()},
            "demonstrations": self.demonstrations, "qualification": self.qualification,
            "qualified": self.qualified, "test_index": self.test_index,
            "model_paths": {k: str(v.relative_to(self.root)) for k, v in self.model_paths.items()},
            "continuation": None if self.continuation is None else self.continuation.state_dict(),
            "live_session": None if self.live_session is None else self.live_session.state_dict(),
            "recorder": None if self.recorder is None or self.recorder.closed or self.recorder.failed else self.recorder.snapshot()})

    def restore_checkpoint_state(self, saved):
        """Restore into already bound owners, with no file truncation/budget reset.

        Preflight exercises collector restore separately. This interface requires
        identical phase/key ownership, so it cannot jump phases or open tests.
        """
        state = decode_arrays(saved)
        current = decode_arrays(self.checkpoint_state())
        if (set(state) != set(current) or any(state[k] != current[k] for k in
                ("format", "config_sha256", "streams_sha256", "sequence", "model_paths"))):
            raise ValueError("campaign recovery phase/contract changed")
        assert_budget_ancestor(state["budget"], self.budget)
        sequence = PilotSequence(self.config, self.root)
        sequence.load_state_dict(state["sequence"])
        replacements = {}
        for group in ("templates", "initializers", "models"):
            owners = getattr(self, group)
            if owners.keys() != state[group].keys():
                raise ValueError("campaign recovery model ownership changed")
            replacements[group] = copy.deepcopy(owners)
            for key, owner in replacements[group].items():
                owner.load_state_dict(state[group][key])
        if state["recorder"] is not None:
            if self.recorder is None:
                raise ValueError("missing live recorder owner")
            self.recorder.assert_prefix(state["recorder"])
            if state["recorder"]["rows"] != self.recorder.count:
                raise ValueError("cannot rewind past already persisted raw rows")
        continuation = copy.copy(self.continuation)
        if (continuation is None) != (state["continuation"] is None):
            raise ValueError("continuation ownership changed")
        if continuation is not None:
            continuation.load_state_dict(encode_arrays(state["continuation"]))
        session = copy.deepcopy(self.live_session)
        if (session is None) != (state["live_session"] is None):
            raise ValueError("collector ownership changed")
        if session is not None:
            session.load_state_dict(encode_arrays(state["live_session"]))
        rng = decode_arrays(state["global_rng"])
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
        torch.Generator().set_state(rng["torch"])
        # Validation above is private; publish all numerical owners together.
        for group, value in replacements.items():
            setattr(self, group, value)
        self.sequence, self.continuation, self.live_session = sequence, continuation, session
        if continuation is not None:
            self.models[continuation.scope] = continuation.kernel
            if continuation.active is not None:
                self.live_session = continuation.active
        if self.recorder is not None and self.live_session is not None:
            self.recorder.session = self.live_session
        for key in ("demonstrations", "qualification", "qualified", "test_index"):
            setattr(self, key, copy.deepcopy(state[key]))
        restore_global_rng(state["global_rng"])

    def save_boundary(self, name):
        path = self.root / "payload" / "boundaries" / f"{name}.pt"
        save_envelope(path, self.checkpoint_state())
        return path

    def make_session(self, block, kernel, seed, split, selection, name, *, representation, role, world):
        self.budget.check()
        if split == "test":
            self.sequence.check_seals()
        session = self.backend.session(block, kernel, seed, trajectory=name, split=split, selection=selection)
        self.live_session = session
        self.recorder = self.recorder_type(self.root, f"payload/episodes/{name}", session, self.config,
            block=block, representation=representation, role=role, world_index=world, seed=seed)
        return session

    def collect(self, block, kernel, seed, split, selection, name, *, representation="graph", role="r4", world=0):
        run = self.make_session(block, kernel, seed, split, selection, name,
                                representation=representation, role=role, world=world)
        while not run.closed:
            self.recorder.append(run.step(before_step=lambda: self.budget.debit("environment")))
        index = self.recorder.finish()
        self.recorder, self.live_session = None, None
        self.budget.check()
        return run.examples, index

    def preflight(self):
        evidence, ordinal, counts = [], 0, []
        settings = CandidatePPOSettings(**{f.name: self.config["ppo"][f.name] for f in fields(CandidatePPOSettings)})
        for block in self.config["blocks"]:
            runtime = self.backend.prepare(block, self.streams["environment"]["preflight"][ordinal])
            path = self.root / "payload" / "preflight" / f"block{block}-environment.json"
            write_json_once(path, runtime)
            evidence.append(path)
            tensors = {}
            for rep in self.config["representations"]:
                representation = rep["name"]
                prototype = candidate_prototype(self.backend.producer(block), self.config,
                                                block=block, representation=representation)
                tensors[representation] = state_digest(prototype.state_dict())
                key = f"block{block}/{representation}"
                template = CandidatePPOKernel(prototype, self.backend.producer(block).contract, settings,
                    enabled=True, mode="frozen", sampling_seed=self.streams["neural"][key + "/continuation/sample"],
                    shuffle_seed=self.streams["neural"][key + "/ppo/shuffle"])
                self.templates[key] = copy.deepcopy(template)
                seed = self.streams["environment"]["preflight"][ordinal]
                run = self.make_session(block, template, seed, "preflight", "sample", f"preflight/{key}",
                                        representation=representation, role="preflight", world=ordinal)
                # These copies have public producer metadata and a policy, no
                # environment/registry handle. Inference receives only raw floats.
                _, isolated_bank = context_from_public(copy.deepcopy(run.producer), copy.deepcopy(run.reference),
                    run.env.observation().copy(), run.options, run.expected_token)
                midpoint = self.config["objective"]["horizon"] // 2
                clone, compared = None, 0
                while not run.closed:
                    if run.index == midpoint:
                        saved = run.state_dict()
                        path = self.root / "payload" / "preflight" / f"block{block}-{representation}-midpoint.pt"
                        save_envelope(path, saved)
                        clone = copy.deepcopy(run)
                        clone.load_state_dict(load_envelope(path))
                        evidence.append(path)
                    event = run.step(before_step=lambda: self.budget.debit("environment"))
                    if run.index == 1 and event["audit"]["decision"]["evaluation"]["candidates"]["class_keys"] != isolated_bank.class_keys:
                        raise ValueError("isolated public-input candidate support differs")
                    self.recorder.append(event)
                    if clone is not None and compared < min(4, self.config["objective"]["horizon"] - midpoint):
                        other = clone.step(before_step=lambda: self.budget.debit("environment"))
                        if (state_digest(encode_arrays(event)) != state_digest(encode_arrays(other))
                                or state_digest(run.state_dict()) != state_digest(clone.state_dict())):
                            raise ValueError("preflight restored decision/reward/identity/RNG differs")
                        compared += 1
                self.recorder.finish()
                self.recorder, self.live_session = None, None
                counts.append({"block": block, "representation": representation, "seed": seed,
                               "full_episode_steps": run.index, "cloned_steps": compared, "passed": True,
                               "isolated_public_input_support_equal": True,
                               "sampler": "disposable copy of declared continuation generator; templates unchanged"})
                ordinal += 1
            if tensors["graph"] != tensors["self_only"]:
                raise ValueError("graph/self-only initialization tensors differ")
        path = self.root / "payload" / "preflight" / "summary.json"
        write_json_once(path, {"passed": True, "cases": counts, "optimizer_steps": 0,
                               "unused_preflight_seeds_not_consumed": self.streams["environment"]["preflight"][ordinal:]})
        return evidence + [path]

    def collect_shared(self, split):
        target = self.demonstrations if split == "demonstration" else self.qualification
        indexes = []
        for block in self.config["blocks"]:
            target[block] = []
            for world, seed in enumerate(self.streams["environment"][split][str(block)]):
                examples, index = self.collect(block, self.templates[f"block{block}/graph"], seed,
                    split, "reference", f"{split}/block{block}/world{world}", world=world)
                target[block].extend(examples)
                indexes.append(index)
        path = self.root / "payload" / f"{split}-index.json"
        write_json_once(path, indexes)
        return [path]

    def initialize(self, key):
        b, rep = key.split("/")
        block = int(b[5:])
        template = self.templates[key]
        kernel = initialization_kernel(template.policy, template.contract, self.config, self.streams,
                                       block=block, representation=rep)
        self.initializers[key] = kernel
        result = kernel.fit(self.demonstrations[block],
            replacement_steps=self.config["initialization"]["optimizer_steps_per_model"],
            before_step=lambda: self.budget.debit("optimizer"))
        path = self.root / "payload" / "initialization" / key / "fit.json"
        write_json_once(path, result)
        state = path.with_suffix(".pt")
        save_envelope(state, kernel.state_dict())
        return [path, state]

    def qualify_and_fork(self):
        evidence = self.collect_shared("qualification")
        expected = self.config["objective"]["horizon"] * self.config["initialization"]["qualification_episodes_per_block"]
        for block in self.config["blocks"]:
            for rep in self.config["representations"]:
                key = f"block{block}/{rep['name']}"
                result = qualify_initializer(self.initializers[key], self.qualification[block], expected_rows=expected,
                    minimum_agreement=self.config["initialization"]["minimum_greedy_reference_agreement_each_model"])
                self.qualified[key] = result
        path = self.root / "payload" / "qualification.json"
        write_json_once(path, self.qualified)
        if not all(r["passed"] for r in self.qualified.values()):
            raise ValueError("initialization qualification below locked agreement gate; attempt closed")
        for key, kernel in self.initializers.items():
            b, rep = key.split("/")
            forks = fork_initializer(kernel, self.qualified[key], self.config, self.streams,
                                     block=int(b[5:]), representation=rep)
            for role, owner in forks.items():
                name = f"{key}/{role}"
                self.models[name] = owner
                initial = self.root / "payload" / "models" / name / "initial.pt"
                save_envelope(initial, owner.state_dict())
                evidence.append(initial)
                if role == "frozen":
                    self.model_paths[name] = initial
        return evidence + [path]

    def continuation_factory(self, scope):
        b, rep, role = scope.split("/")
        block = int(b[5:])
        def factory(kernel, episode, seed):
            return self.backend.session(block, kernel, seed, split="training", selection="sample",
                trajectory=f"training/{scope}/episode{episode:02d}")
        return factory

    def train(self, scope):
        block = int(scope.split("/")[0][5:])
        role = scope.split("/")[-1]
        self.continuation = CandidateContinuation(self.models[scope], self.budget, self.config, role=role,
            scope=scope, seeds=self.streams["environment"]["training"][str(block)],
            session_factory=self.continuation_factory(scope))
        indexes, updates, evidence = [], [], []
        while not self.continuation.done:
            if self.continuation.update_due:
                updates.append(self.continuation.update())
                self.live_session = None
                evidence.append(self.save_boundary(f"{scope}/episode{self.continuation.episode:02d}"))
                self.emit_status("running", completed_episodes=self.continuation.episode,
                                 completed_updates=self.continuation.updates)
            else:
                if self.continuation.active is None:
                    self.live_session = self.continuation.start_episode()
                    episode = self.continuation.episode
                    self.recorder = self.recorder_type(self.root,
                        f"payload/episodes/training/{scope}/episode{episode:02d}", self.live_session, self.config,
                        block=block, representation=scope.split("/")[1], role=role, world_index=episode,
                        seed=self.continuation.seeds[episode])
                event = self.continuation.step()
                self.recorder.append(event)
                if self.recorder.session.closed:
                    indexes.append(self.recorder.finish())
                    self.recorder, self.live_session = None, None
        self.models[scope] = self.continuation.kernel
        final = self.root / "payload" / "models" / scope / "final.pt"
        save_envelope(final, self.models[scope].state_dict())
        self.model_paths[scope] = final
        path = final.parent / "training.json"
        write_json_once(path, {"episodes": self.continuation.episode, "updates": updates, "raw_episodes": indexes})
        self.continuation = None
        return evidence + [final, path]

    def seal(self):
        self.sequence.seal_models(self.model_paths)
        path = self.root / "payload" / "model-seals.json"
        write_json_once(path, self.sequence.seals)
        return [path]

    def evaluate(self, job):
        parts = job.split("/")
        block = int(parts[0][5:])
        rep, role = (parts[1], parts[2]) if len(parts) == 4 else ("reference", parts[1])
        key = "/".join(parts[:-1])
        kernel = self.models[key] if rep != "reference" else self.models[f"block{block}/graph/frozen"]
        before = state_digest(kernel.state_dict())
        if rep != "reference" and before != state_digest(load_envelope(self.model_paths[key])):
            raise ValueError("evaluation owner differs from sealed final model")
        indexes = []
        selection = "reference" if role == "r4" else "anchor" if role == "mdl2" else "greedy"
        for world, seed in enumerate(self.streams["environment"]["test"][str(block)]):
            _, index = self.collect(block, kernel, seed, "test", selection, f"evaluation/{key}/world{world:02d}",
                                     representation=rep, role=role, world=world)
            indexes.append(index)
        if state_digest(kernel.state_dict()) != before:
            raise ValueError("evaluation modified frozen policy/optimizer/RNG")
        self.test_index.extend(indexes)
        path = self.root / "payload" / "evaluation" / key / "index.json"
        write_json_once(path, indexes)
        return [path]

    def verify(self):
        self.sequence.check_seals()
        self.final_lock_check()
        result = self.verifier(self.root, self.test_index, self.config, self.streams)
        checked = read_ledger(self.budget.path)
        if checked["counts"] != self.budget.counts or checked["last_sha256"] != self.budget.previous:
            raise ValueError("raw budget differs from live counts")
        expected_env = {**self.config["caps"]["environment_steps"]}
        expected_env.pop("preflight_including_clones")
        for phase, count in expected_env.items():
            if self.budget.phases.get(phase + ":environment", 0) != count:
                raise ValueError("incomplete prescribed environment work")
        for phase, count in self.config["caps"]["optimizer_steps"].items():
            if self.budget.phases.get(phase + ":optimizer", 0) != count:
                raise ValueError("incomplete prescribed optimizer work")
        path = self.root / "payload" / "independent-verification.json"
        write_json_once(path, result)
        ledger = self.root / "payload" / "budget-through-verification.jsonl"
        copy_verified(self.budget.path, ledger)
        return [path, ledger]

    def archive(self):
        self.final_lock_check()
        write_json_once(self.root / "payload" / "artifact-inventory.json", inventory(self.root / "payload"))
        output = self.root / "archives" / "completed-payload.tar.gz"
        result = create_archive(self.root / "payload", output)
        destination = Path(self.config["dropbox_directory_proposed"])
        copies = [copy_verified(p, destination / p.name) for p in (output, output.with_name(output.name + ".manifest.json"))]
        self.budget.check()
        path = self.root / "launcher" / "archive-receipt.json"
        write_json_once(path, {"archive": result, "local_copies": copies, "cloud_sync_verified": False,
                               "howard_access_verified": False})
        return [path]

    def dispatch(self, job):
        if job == "preflight_including_clones":
            return self.preflight()
        if job == "demonstrations":
            return self.collect_shared("demonstration")
        if job.endswith("/initialization"):
            return self.initialize(job.rsplit("/", 1)[0])
        if job == "qualification":
            return self.qualify_and_fork()
        if job.endswith(("/ppo", "/bc_continue")):
            return self.train(job)
        if job == "seal_all_models":
            return self.seal()
        if job.endswith("/evaluation"):
            return self.evaluate(job)
        if job == "independent_verification":
            return self.verify()
        if job == "archive":
            return self.archive()
        raise ValueError("unknown fixed P1 job")

    def run(self):
        try:
            while self.sequence.next_job is not None:
                job = self.sequence.next_job
                self.sequence.begin(job)
                if job in self.budget.sections:
                    self.budget.begin(job)
                self.emit_status("running")
                evidence = self.dispatch(job)
                self.budget.check()
                if self.budget.active is not None:
                    self.budget.finish()
                self.sequence.finish(evidence)
                # Archive must not mutate the payload it has just sealed.
                if job != "archive":
                    self.save_boundary(f"phase{len(self.sequence.completed):02d}")
                self.emit_status("phase_completed", completed_job=job)
            write_json_once(self.root / "launcher" / "completed.json",
                {"status": "completed", "exit_code": 0, "sequence": self.sequence.state_dict(),
                 "budget": self.budget.snapshot(), "automatic_followon": False})
            return 0
        except BaseException as exc:
            self.sequence.fail(repr(exc))
            failures = {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
                        "sequence": self.sequence.state_dict(), "budget": self.budget.snapshot(), "retry_permitted": False}
            try:
                save_envelope(self.root / "launcher" / "failure-state.pt", self.checkpoint_state())
            except BaseException as preservation_error:
                failures["checkpoint_preservation_error"] = repr(preservation_error)
            if self.recorder is not None:
                self.recorder.close_partial()
            write_json_once(self.root / "launcher" / "failure.json", failures)
            print(failures["traceback"], flush=True)
            return 1
        finally:
            self.budget.close()
