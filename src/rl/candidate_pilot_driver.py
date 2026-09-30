"""Serial P1 sequencing and recoverable continuation transactions.

No patient factory or command-line scientific launch is provided here. The
locked runner must bind factories after source/input/runtime readiness checks.
Restoration is an engineering interface, not permission to retry a failed run.
"""

from __future__ import annotations

import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np

from src.models.candidate_policy import CandidatePolicy
from src.rl.candidate_imitation import CandidateImitationKernel, ImitationSettings, decode_example
from src.rl.candidate_patient_session import load_envelope, save_envelope
from src.rl.candidate_pilot_resources import digest, read_ledger
from src.rl.candidate_ppo_kernel import CandidatePPOKernel, CandidatePPOSettings
from src.rl.candidate_rollout import evaluate_candidate_policy
from src.rl.prospective_ddpg_kernel import state_digest


def candidate_prototype(producer, config, *, block, representation):
    if block not in config["blocks"]:
        raise ValueError("unknown initialization block")
    matches = [r for r in config["representations"] if r["name"] == representation]
    if len(matches) != 1:
        raise ValueError("unknown/duplicate representation")
    spec = matches[0]
    seed = config["policy_init_seeds"][config["blocks"].index(block)]
    model = CandidatePolicy(producer.contract.inputs, enabled=True,
        architecture=spec["architecture"], message_mode=spec["message_mode"],
        encoder_width=config["model"]["encoder_width"], head_width=config["model"]["head_width"], seed=seed)
    if sum(p.numel() for p in model.parameters()) != spec["expected_parameters"]:
        raise ValueError("declared model parameter count differs from runtime schema")
    return model


def initialization_kernel(prototype, contract, config, streams, *, block, representation):
    cfg = config["initialization"]
    return CandidateImitationKernel(prototype, contract,
        ImitationSettings(cfg["learning_rate"], cfg["gradient_clip"], cfg["batch_size"],
                          cfg["optimizer_steps_per_model"], 1, "demonstration"),
        enabled=True, shuffle_seed=streams["neural"][f"block{block}/{representation}/bc_init/shuffle"])


def fork_initializer(initializer, qualification, config, streams, *, block, representation):
    """Tensor-identical forks; neither trainable arm inherits imitation moments."""
    if (qualification.get("passed") is not True or qualification["kernel_sha256"] != state_digest(initializer.state_dict())
            or initializer.steps != config["initialization"]["optimizer_steps_per_model"]):
        raise ValueError("complete unchanged qualified initializer required")
    base = f"block{block}/{representation}"
    sample = streams["neural"][base + "/continuation/sample"]
    settings = CandidatePPOSettings(**{f.name: config["ppo"][f.name] for f in fields(CandidatePPOSettings)})
    result = {role: CandidatePPOKernel(initializer.policy, initializer.contract, settings, enabled=True,
               mode="frozen" if role == "frozen" else "online", sampling_seed=sample,
               shuffle_seed=streams["neural"][base + "/ppo/shuffle"]) for role in ("frozen", "ppo")}
    bc = config["bc_continue"]
    result["bc_continue"] = CandidateImitationKernel(initializer.policy, initializer.contract,
        ImitationSettings(bc["learning_rate"], bc["max_grad_norm"], bc["batch_size"],
                          bc["max_optimizer_steps"], bc["max_updates"], "training"),
        enabled=True, sampling_seed=sample, shuffle_seed=streams["neural"][base + "/bc_continue/shuffle"])
    for role, kernel in result.items():
        if (kernel.policy.snapshot_sha256() != initializer.policy.snapshot_sha256()
                or (kernel.optimizer is not None and kernel.optimizer.state_dict()["state"])):
            raise ValueError("fork policy mismatch or inherited optimizer moments")
    if state_digest(result["ppo"].sampling_rng.get_state()) != state_digest(result["bc_continue"].sampling_rng.get_state()):
        raise ValueError("continuation samplers not cloned")
    return result


def pilot_jobs(config):
    """One fixed serial order. Testing follows the all-model seal barrier."""
    jobs = ["preflight_including_clones", "demonstrations"]
    models = [f"block{b}/{r['name']}" for b in config["blocks"] for r in config["representations"]]
    jobs += [m + "/initialization" for m in models]
    jobs += ["qualification"]
    jobs += [m + "/" + role for m in models for role in ("ppo", "bc_continue")]
    jobs += ["seal_all_models"]
    for b in config["blocks"]:
        jobs += [f"block{b}/{r['name']}/{role}/evaluation" for r in config["representations"]
                 for role in config["candidate_roles"]]
        jobs += [f"block{b}/{role}/evaluation" for role in config["reference_roles"]]
    jobs += ["independent_verification", "archive"]
    if len(jobs) != len(set(jobs)):
        raise ValueError("duplicate P1 jobs")
    return jobs


def required_models(config):
    return [f"block{b}/{r['name']}/{role}" for b in config["blocks"]
            for r in config["representations"] for role in config["candidate_roles"]]


def file_record(root, path):
    root, path = Path(root).resolve(), Path(path)
    if not path.is_absolute():
        path = root / path
    # Reject symlinked evidence, including parent directory redirects.
    if path.is_symlink() or path.resolve() != path.absolute() or not path.is_file():
        raise ValueError("evidence must be a regular non-symlink file")
    relative = path.relative_to(root).as_posix()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": h.hexdigest()}


class PilotSequence:
    """Fail-closed phase cursor with a byte-verified test-entry barrier."""

    def __init__(self, config, root):
        self.config, self.root = copy.deepcopy(config), Path(root).resolve()
        self.jobs = pilot_jobs(config)
        self.completed, self.active, self.failure, self.seals = [], None, None, {}

    @property
    def next_job(self):
        return None if self.failure or len(self.completed) == len(self.jobs) else self.jobs[len(self.completed)]

    def begin(self, job):
        if self.active is not None or job != self.next_job or job is None:
            raise ValueError("failed, repeated, concurrent or out-of-order P1 job")
        if job.endswith("/evaluation"):
            self.check_seals()
        self.active = job

    def seal_models(self, paths):
        if self.active != "seal_all_models" or self.seals or set(paths) != set(required_models(self.config)):
            raise ValueError("all prescribed final/frozen policies must be sealed once before tests")
        records = {name: file_record(self.root, path) for name, path in paths.items()}
        if len({r["path"] for r in records.values()}) != len(records):
            raise ValueError("separate named model artifacts required")
        self.seals = records

    def check_seals(self):
        if set(self.seals) != set(required_models(self.config)):
            raise ValueError("test data forbidden before every model is frozen")
        for record in self.seals.values():
            if file_record(self.root, record["path"]) != record:
                raise ValueError("sealed model bytes changed")

    def finish(self, evidence):
        if self.failure or self.active != self.next_job or self.active is None or not evidence:
            raise ValueError("active successful job and nonempty evidence required")
        records = [file_record(self.root, p) for p in evidence]
        if len({r["path"] for r in records}) != len(records):
            raise ValueError("duplicate evidence")
        if self.active == "seal_all_models" or self.active.endswith("/evaluation"):
            self.check_seals()
        self.completed.append({"job": self.active, "evidence": records})
        self.active = None

    def fail(self, reason):
        if not isinstance(reason, str) or not reason or self.failure is not None:
            raise ValueError("one explicit terminal failure required")
        self.failure = {"job": self.active or self.next_job, "reason": reason}

    def state_dict(self):
        return copy.deepcopy({"format": "candidate-pilot-sequence-v1", "config_sha256": digest(self.config),
                              "completed": self.completed, "active": self.active,
                              "failure": self.failure, "seals": self.seals})

    def load_state_dict(self, state):
        candidate = PilotSequence(self.config, self.root)
        if set(state) != set(candidate.state_dict()) or state["config_sha256"] != digest(self.config) or state["format"] != "candidate-pilot-sequence-v1":
            raise ValueError("sequence contract mismatch")
        for completed in state["completed"]:
            candidate.begin(completed["job"])
            if candidate.active == "seal_all_models":
                candidate.seals = copy.deepcopy(state["seals"])
            for record in completed["evidence"]:
                if file_record(self.root, record["path"]) != record:
                    raise ValueError("completed evidence changed")
            candidate.finish([r["path"] for r in completed["evidence"]])
        if state["active"] is not None:
            candidate.begin(state["active"])
            if candidate.active == "seal_all_models":
                candidate.seals = copy.deepcopy(state["seals"])
                if candidate.seals:
                    candidate.check_seals()
        if state["failure"] is not None:
            candidate.fail(state["failure"]["reason"])
        if candidate.state_dict() != state:
            raise ValueError("sequence history or failure cursor differs")
        self.__dict__.update(candidate.__dict__)


def assert_budget_ancestor(snapshot, budget):
    """A checkpoint may not rewrite the external budget or erase later spend."""
    live = budget.snapshot()
    checked = read_ledger(budget.path)
    if checked["events"] != live["events"] or checked["last_sha256"] != live["ledger_sha256"]:
        raise ValueError("budget owner differs from durable ledger")
    for kind in ("environment", "optimizer"):
        if checked["counts"].get(kind, 0) != live["counts"][kind]:
            raise ValueError("budget owner count mismatch")
    rows = [json.loads(line) for line in Path(budget.path).read_text().splitlines()]
    count = snapshot.get("events")
    if type(count) is not int or not 1 <= count <= len(rows) or rows[count - 1]["sha256"] != snapshot["ledger_sha256"]:
        raise ValueError("checkpoint budget is not a prefix of the live ledger")
    counts, phases, scopes, closed, active = {"environment": 0, "optimizer": 0}, {}, {}, [], None
    for row in rows[:count]:
        if row["event"] == "begin":
            active = row["section"]
        elif row["event"] == "finish":
            closed.append(row["section"])
            active = None
        elif row["event"] == "debit":
            kind = row["resource"]
            counts[kind] += 1
            scopes[row["section"] + ":" + kind] = row["scope_total"]
            phases[row["phase"] + ":" + kind] = row["phase_total"]
    expected = {"counts": counts, "phases": phases, "scopes": scopes, "closed": closed,
                "active": active, "events": count, "ledger_sha256": rows[count - 1]["sha256"], "failed": False}
    if snapshot != expected or live["failed"] or live["active"] != active:
        raise ValueError("checkpoint scope/counts changed or budget owner closed")
    # Deliberately retain all later debits and elapsed wall time in `budget`.
    budget.check()


class CandidateContinuation:
    """One serial arm, including live collection and four-episode update state.

    session_factory(kernel, episode_index, seed) constructs the matching fresh
    episode without stepping it. An external runner records raw events, marks
    failures terminal, and never calls restore/retry after scientific failure.
    """

    def __init__(self, kernel, budget, config, *, role, scope, seeds, session_factory):
        if role not in ("ppo", "bc_continue") or budget.active != scope:
            raise ValueError("active declared continuation scope required")
        if (role == "ppo" and type(kernel) is not CandidatePPOKernel) or (role == "bc_continue" and type(kernel) is not CandidateImitationKernel):
            raise TypeError("matching continuation kernel required")
        settings = config[role]
        if len(seeds) != settings["episodes_per_model"] or len(set(seeds)) != len(seeds):
            raise ValueError("exact fresh ordered training starts required")
        if settings["episodes_per_model"] != settings["episodes_per_rollout"] * settings["max_updates"]:
            raise ValueError("whole fixed rollouts required")
        self.kernel, self.budget, self.factory = kernel, budget, session_factory
        self.settings, self.role, self.scope, self.seeds = copy.deepcopy(settings), role, scope, tuple(seeds)
        self.horizon = config["objective"]["horizon"]
        self.manifest = {"format": "candidate-continuation-v1", "role": role, "scope": scope,
                         "seeds": self.seeds, "settings": self.settings, "horizon": self.horizon,
                         "initial_kernel_sha256": state_digest(kernel.state_dict())}
        self.episode, self.active, self.examples, self.receipts, self.last_closed = 0, None, [], [], None

    @property
    def updates(self):
        return self.kernel.total_updates if self.role == "ppo" else len(self.kernel.history)

    @property
    def done(self):
        return self.episode == len(self.seeds) and self.updates == self.settings["max_updates"] and self.active is None

    @property
    def update_due(self):
        return self.episode == (self.updates + 1) * self.settings["episodes_per_rollout"] and self.active is None

    def start_episode(self):
        if self.done or self.update_due or self.episode >= len(self.seeds) or self.budget.active != self.scope:
            raise ValueError("collection forbidden at closed/update/wrong-scope boundary")
        self.budget.check()
        if self.active is None:
            self.active = self.factory(self.kernel, self.episode, self.seeds[self.episode])
            if (self.active.split != "training" or self.active.selection != "sample" or self.active.learner is not self.kernel
                    or self.active.boundary.episode_horizon != self.horizon):
                raise ValueError("training must sample this exact kernel")
        return self.active

    def step(self):
        self.start_episode()
        event = self.active.step(before_step=lambda: self.budget.debit("environment"))
        if self.active.closed:
            if self.role == "ppo":
                self.kernel.add_segment(self.active.segment(self.settings["gae_lambda"]))
            else:
                self.examples.extend(copy.deepcopy(self.active.examples))
            self.last_closed = self.active.state_dict()
            self.receipts.append({"episode": self.episode, "seed": self.seeds[self.episode],
                                  "trajectory_id": self.active.trajectory_id,
                                  "session_sha256": state_digest(self.last_closed),
                                  "steps": self.active.index})
            self.episode += 1
            self.active = None
        return event

    def update(self):
        if not self.update_due or self.budget.active != self.scope:
            raise ValueError("only a complete fixed rollout may update")
        self.budget.check()
        if self.role == "ppo":
            result = self.kernel.update(before_optimizer_step=lambda: self.budget.debit("optimizer"))
        else:
            result = self.kernel.fit(self.examples, epochs=self.settings["epochs"],
                                     before_step=lambda: self.budget.debit("optimizer"))
            self.examples = []
        self.budget.check()
        return result

    def state_dict(self):
        return copy.deepcopy({"manifest": self.manifest, "kernel": self.kernel.state_dict(),
                              "episode": self.episode, "examples": self.examples, "receipts": self.receipts,
                              "last_closed": self.last_closed,
                              "active": None if self.active is None else self.active.state_dict(),
                              "budget": self.budget.snapshot()})

    def _session_from_state(self, saved, episode, kernel):
        learner = copy.deepcopy(kernel)
        learner.load_state_dict(saved["kernel"])
        learner.sampling_rng.set_state(saved["initial_rng"])
        session = self.factory(learner, episode, self.seeds[episode])
        session.load_state_dict(saved)
        return session

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()) or state["manifest"] != self.manifest:
            raise ValueError("continuation checkpoint contract mismatch")
        assert_budget_ancestor(state["budget"], self.budget)
        kernel = copy.deepcopy(self.kernel)
        kernel.load_state_dict(state["kernel"])
        episode = state["episode"]
        if type(episode) is not int or not 0 <= episode <= len(self.seeds) or len(state["receipts"]) != episode:
            raise ValueError("continuation episode/receipt cursor differs")
        updates = kernel.total_updates if self.role == "ppo" else len(kernel.history)
        pending_episodes = episode - updates * self.settings["episodes_per_rollout"]
        if not 0 <= pending_episodes <= self.settings["episodes_per_rollout"]:
            raise ValueError("update and collection cursors disagree")
        for i, receipt in enumerate(state["receipts"]):
            if receipt["episode"] != i or receipt["seed"] != self.seeds[i] or receipt["steps"] != self.horizon:
                raise ValueError("receipt order/seed mismatch")
        if len({r["trajectory_id"] for r in state["receipts"]}) != episode:
            raise ValueError("duplicate trajectory")
        if self.role == "ppo":
            if state["examples"] or len(kernel.pending) != pending_episodes:
                raise ValueError("PPO pending rollout mismatch")
        else:
            expected = sum(r["steps"] for r in state["receipts"][-pending_episodes:]) if pending_episodes else 0
            examples = state["examples"]
            if len(examples) != expected or len({e["identity"] for e in examples}) != expected:
                raise ValueError("imitation pending rollout mismatch")
            for example in examples:
                decode_example(example, kernel.contract)
                if example["split"] != "training":
                    raise ValueError("non-training pending imitation data")
        if episode:
            if state["last_closed"] is None or state_digest(state["last_closed"]) != state["receipts"][-1]["session_sha256"]:
                raise ValueError("closed collector boundary differs")
            last = self._session_from_state(state["last_closed"], episode - 1, kernel)
            if not last.closed or last.trajectory_id != state["receipts"][-1]["trajectory_id"]:
                raise ValueError("last collector is not the declared closed episode")
        elif state["last_closed"] is not None:
            raise ValueError("unexpected closed collector")
        active = None
        if state["active"] is not None:
            if episode >= len(self.seeds) or pending_episodes == self.settings["episodes_per_rollout"]:
                raise ValueError("unexpected active episode at update boundary")
            active = self._session_from_state(state["active"], episode, kernel)
            if active.closed or state_digest(active.learner.state_dict()) != state_digest(kernel.state_dict()):
                raise ValueError("active collector/kernel mismatch")
            kernel = active.learner
        spent = state["budget"]["scopes"]
        optimizer_steps = kernel.total_optimizer_steps if self.role == "ppo" else kernel.steps
        if (spent.get(self.scope + ":environment", 0) < episode * self.horizon + (active.index if active else 0)
                or spent.get(self.scope + ":optimizer", 0) < optimizer_steps):
            raise ValueError("collector/optimizer work has no corresponding external debit")
        self.kernel, self.active, self.episode = kernel, active, episode
        self.examples, self.receipts = copy.deepcopy(state["examples"]), copy.deepcopy(state["receipts"])
        self.last_closed = copy.deepcopy(state["last_closed"])

    def save(self, path):
        save_envelope(path, self.state_dict())

    def load(self, path):
        self.load_state_dict(load_envelope(path))


def qualify_initializer(kernel, examples, *, expected_rows, minimum_agreement):
    if len(examples) != expected_rows or not examples or len({e["identity"] for e in examples}) != len(examples):
        raise ValueError("exact independent qualification rows required")
    before = state_digest(kernel.state_dict())
    counts, hits, multi, multi_hits = [], 0, 0, 0
    for example in examples:
        if example["split"] != "qualification":
            raise ValueError("qualification must not reuse demonstrations/training/test")
        obs, bank = decode_example(example, kernel.contract)
        value = evaluate_candidate_policy(kernel.policy, obs, bank, kernel.contract)
        hit = int(np.argmax(value.log_probs)) == bank.reference_class
        counts.append(len(bank.class_keys))
        hits += int(hit)
        if len(bank.class_keys) > 1:
            multi += 1
            multi_hits += int(hit)
    if state_digest(kernel.state_dict()) != before:
        raise ValueError("qualification changed model/optimizer/RNG")
    return {"rows": len(examples), "agreement": hits / len(examples), "support_counts": counts,
            "multi_class_rows": multi, "multi_class_agreement": multi_hits / multi if multi else None,
            "passed": hits / len(examples) >= minimum_agreement, "kernel_sha256": before}
