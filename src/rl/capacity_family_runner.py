"""One fixed two-stage comparison on the existing native support-hour task."""

from dataclasses import asdict
import copy
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle

import numpy as np

from src.baselines.capacity_planner_tail_mpc import CapacityPlannerTailMPC
from src.env.patient_support_capacity import PatientSupportAction
from src.rl.capacity_confirmation_learner import CapacityConfirmationTailLearner
from src.rl.capacity_family_analysis import select_finalists, summaries
from src.rl.capacity_family_design import METHODS, PHASES, SCOPE, expected_learning_counts, seeds, worlds
from src.rl.capacity_family_learner import FamilyLearner
from src.rl.capacity_family_value import CapacityFamilyValue
from src.rl.capacity_pilot_runner import CapacityPilotRunner, COST_KEYS, jsonable, trajectory_id
from src.rl.capacity_public_features import resource_adjacency
from src.rl.capacity_value_features import observed_features
from src.rl.capacity_value_learner import td_rows
from src.rl.networks import torch
from src.utils.research_clock import shared_monotonic


def actor_rows(data):
    """Settlement belongs to the final decision, without discount or omission."""
    tail = math.fsum(data["costs"][48:])
    return [dict(state=data["features"][t], next_state=data["features"][t+1],
        action=data["actions"][t], metadata=data["metadata"][t], done=t == 47,
        control_cost=data["costs"][t], settlement_cost=tail if t == 47 else None,
        full_cost=data["costs"][t]+(tail if t == 47 else 0),
        reward=-(data["costs"][t]+(tail if t == 47 else 0))/1e6) for t in range(48)]


class CapacityFamilyRunner(CapacityPilotRunner):
    def __init__(self, root, proposal, budget, *, admission, workspace, **backends):
        if admission != dict(verified=True, scope=SCOPE):
            raise PermissionError("exact family-selection admission required")
        backends.setdefault("controller_factory", CapacityPlannerTailMPC)
        super().__init__(root, proposal, budget,
                         admission=dict(verified=True, scope="dynamic-capacity-pilot-v1"), **backends)
        self.study, self.workspace = proposal["family_study"], Path(workspace)
        self.records, self.selection, self.episode_data = {}, None, None
        self.method, self.training, self.horizon = None, False, 8
        self.episode_start_counts = None
        self.replay_sources = []
        self.started = False

    def _new(self, method, architecture, phase, block, lr):
        model_seed, sampler_seed = seeds(self.study, phase, block)
        if method == "value_td":
            config = dict(self.study["value"], lr=lr, architecture=architecture, sampler_seed=sampler_seed)
            return CapacityFamilyValue(config, seed=model_seed, before_forward=self._forward,
                                       before_optimizer=self._optimizer)
        adjacency = resource_adjacency(self.p) if architecture == "graph" else np.eye(4, dtype=np.float32)
        return FamilyLearner(method, adjacency=adjacency, model_seed=model_seed,
            sampler_seed=sampler_seed, lr=lr, before_forward=self._forward, before_optimizer=self._optimizer)

    def _forward(self, name, batch):
        if self.episode_start_counts is not None:
            allowed = 400 if self.training else 48
            if self.budget.counts["neural_forward_module_calls"]-self.episode_start_counts["neural_forward_module_calls"] >= allowed:
                raise RuntimeError("per-world forward allowance exhausted")
        super()._forward(name, batch)

    def _optimizer(self, name, batch):
        if not self.training or not self.budget.phase.endswith("training") or not 1 <= batch <= 64:
            raise PermissionError("no optimizer outside bounded training")
        if self.budget.counts["total_optimizer_steps"]-self.episode_start_counts["total_optimizer_steps"] >= 96:
            raise RuntimeError("per-world optimizer allowance exhausted")
        super()._optimizer(name, batch)

    def _query(self, payload):
        self.budget.chunk_call("planner_total_model_epochs", 48*self.horizon, payload["model_epochs"])

    def _tape(self, world):
        path = self.root/"tapes"/(trajectory_id(world, "exogenous")+".json")
        tape = self.tape_factory(self.p, world)
        if not path.exists():
            self._json(path.relative_to(self.root), asdict(tape))
        elif json.loads(path.read_text()) != jsonable(asdict(tape)):
            raise ValueError("paired tape changed")
        return tape

    def state_dict(self):
        return dict(format=SCOPE, active=self.active, epoch=self.epoch,
            environment=None if self.env is None else self.env.state_dict(),
            controller=None if self.controller is None else self.controller.state_dict(),
            learner=None if self.learner is None else self.learner.state_dict(),
            episode_data=self.episode_data, completed=self.completed, selection=self.selection,
            replay_sources=self.replay_sources, budget=self.budget.snapshot(), resume_authorized=False)

    def _checkpoint(self, ident):
        state = self.state_dict()
        if state["learner"] is not None and "replay" in state["learner"]:
            state["learner"]["replay"] = dict(reconstruct_from=self.replay_sources,
                transform="capacity_family_runner.actor_rows", count=len(self.learner.replay))
        with gzip.open(self.root/"states"/(ident+"-after-fit.pkl.gz"), "xb") as stream:
            pickle.dump(state, stream, protocol=5)
        self.budget.check(storage=True)

    def _seal(self, name, *, method, architecture, phase, block, lr):
        self.budget.job(name, "seal")
        buffer = io.BytesIO()
        torch.save(self.learner.state_dict(), buffer)
        raw = buffer.getvalue()
        record = dict(name=name, method=method, architecture=architecture, phase=phase,
            block=block, lr=lr, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
            path="models/"+name+".pt", counts=self.learner.counts)
        with (self.root/record["path"]).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self._json("models/"+name+".json", record)
        self.records[name] = record
        self.status("model_sealed", model=record)
        return record

    def _restore(self, record):
        raw = (self.root/record["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("sealed checkpoint changed")
        learner = self._new(record["method"], record["architecture"], record["phase"], record["block"], record["lr"])
        learner.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False))
        return learner

    def _episode(self, world, role, tape, *, method=None, seal=None):
        self.active, self.epoch, self.method = dict(world, role=role), 0, method
        self.training = world["phase"].endswith("training")
        self.horizon = 16 if role == "plain_h16" else 8
        ident, started = trajectory_id(world, role), shared_monotonic()
        self.budget.job(ident, "h16_world" if self.horizon == 16 else "actor_world" if method in METHODS[:4] else "h8_world")
        self.episode_start_counts = dict(self.budget.counts)
        self.budget.debit(dict(trajectories=1))
        self.env = self.env_factory(self.p, tape, self._native)
        self.controller = self.controller_factory(self.p, base_control=self.base_control, scientific=True,
            value=self.learner if method in ("value_td", "legacy") else None, planning_horizon=self.horizon)
        public = self.capture(self.env)
        self._observe(public)
        x, h = observed_features(public, self.controller)
        data = dict(features=[x], heuristics=[h], costs=[], actions=[], metadata=[])
        self.episode_data = data
        lost_ids = set()
        self.status("trajectory_started")
        with gzip.open(self.root/"raw"/(ident+".jsonl.gz"), "xt") as stream:
            for epoch in range(64):
                self.epoch = epoch
                metadata = {}
                if epoch >= 48:
                    hours = np.zeros(4)
                elif method in METHODS[:4]:
                    hours, metadata = self.learner.act(data["features"][-1], explore=self.training)
                else:
                    self.budget.debit(dict(planner_total_decisions=1, planner_candidate_rollouts=48))
                    hours = self.controller.act(public, before_query=self._query, before_filter=self._filter)
                    self.budget.finish_chunk("planner_total_model_epochs")
                base = self.base_control(public, self.p, tail=epoch >= 48)
                self.controller.record_operation(public, base)
                _, reward, done, info = self.env.step(PatientSupportAction(tuple(map(float, base)), tuple(map(float, hours))))
                cost = float(info["cost"])
                components = {k: float(info[k]) for k in COST_KEYS}
                if (not math.isfinite(cost) or cost < 0
                        or not math.isclose(math.fsum(components.values()), cost, rel_tol=1e-12, abs_tol=1e-6)
                        or not math.isclose(-float(reward), cost, rel_tol=1e-12, abs_tol=1e-6)
                        or bool(done) != (epoch == 63)):
                    raise ValueError("native cost/settlement mismatch")
                following = self.capture(self.env)
                self.budget.debit(dict(estimator_receipt_updates=1))
                self._observe(following)
                x, h = observed_features(following, self.controller)
                records = following.operations.patients
                losses = {p.patient_id for p in records if p.status == "lost"}
                if not lost_ids.issubset(losses):
                    raise ValueError("lost patient reopened")
                executed = info["support_public_receipt"]["committed_hours"]
                if epoch < 48 and not np.allclose(hours, executed, rtol=0, atol=2e-6):
                    raise ValueError("policy action and native committed action differ")
                row = dict(epoch=epoch, world=world, role=role, cost=cost, reward=float(reward),
                    components=components, new_lost_patients=len(losses-lost_ids), patient_records=records,
                    requested_hours=hours, executed_hours=executed, public_input=public,
                    value_features=data["features"][-1], value_base=data["heuristics"][-1],
                    next_value_features=x, next_value_base=h, policy_metadata=metadata,
                    plan=self.controller.last_plan if epoch < 48 and method not in METHODS[:4] else None, info=info)
                stream.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False)+"\n")
                data["features"].append(x)
                data["heuristics"].append(h)
                data["costs"].append(cost)
                if epoch < 48:
                    data["actions"].append(np.asarray(executed, dtype=np.float32))
                    data["metadata"].append(metadata)
                lost_ids, public = losses, following
                if (epoch+1) % 8 == 0:
                    stream.flush()
                    os.fsync(stream.fileno())
                    self.budget.check(storage=True)
                    self.status("epoch_boundary")
        settlement = self.env.settlement()
        if not settlement["settled"] or settlement["lost"] != len(lost_ids):
            raise ValueError("unsettled complete cohort")
        summary = dict(world=world, role=role, raw_path="raw/"+ident+".jsonl.gz",
            cost=math.fsum(data["costs"]), lost=len(lost_ids), settled=True, settlement=settlement,
            tape_sha256=tape.digest(), model_seal_sha256=seal, optimizer_updates_during_trajectory=0,
            elapsed=shared_monotonic()-started,
            compute={k: self.budget.counts[k]-v for k, v in self.episode_start_counts.items()})
        self._json("summaries/"+ident+".json", summary)
        self.completed.append(ident)
        self.epoch = 64
        self.status("trajectory_completed", summary=summary)
        return data

    def _fit(self, data):
        ident = trajectory_id(self.active, self.active["role"])
        self.budget.job(ident+"-fit", "fit")
        self.budget.debit(dict(fit_batches=32))
        if self.method == "value_td":
            self.learner.admit_episode(td_rows(**{k: data[k] for k in ("features", "heuristics", "costs")}, horizon=8, scale=1e6))
            self._json("updates/"+ident+"-targets.json", dict(targets=self.learner.pending["targets"]))
            receipts = [self.learner.update() for _ in range(32)]
        else:
            self.learner.observe_episode(actor_rows(data))
            receipts = self.learner.fit_episode(updates=32, batch_size=64)
        self._json("updates/"+ident+".json", dict(receipts=receipts, counts=self.learner.counts))
        self.replay_sources.append("raw/"+ident+".jsonl.gz")
        self.episode_data = None
        self._checkpoint(ident)
        self.status("fit_completed", learner_counts=self.learner.counts)

    def _train_job(self, phase, block, method, architecture, li):
        lr = self.study["learning_rates"][li]
        role = f"{method}-lr{li}-{architecture}"
        name = f"{phase}-b{block}-{role}"
        self.learner = self._new(method, architecture, phase, block, lr)
        self.episode_start_counts, self.replay_sources = None, []
        metadata = dict(method=method, architecture=architecture, phase=phase, block=block, lr=lr)
        self._seal(name+"-initial", **metadata)
        for world in worlds(self.study, phase, block):
            self._fit(self._episode(world, role, self._tape(world), method=method))
        self.episode_start_counts = None
        self._seal(name+"-final", **metadata)

    def _evaluation(self, phase, block, entries):
        for world in worlds(self.study, phase, block):
            tape = self._tape(world)
            for role, record in entries:
                self.learner, method, seal = None, None, None
                if record is not None:
                    if record.get("legacy"):
                        path = self.workspace/self.study["legacy"]["root"]/self.study["legacy"]["names"][block]
                        raw = path.read_bytes()
                        seal = self.study["legacy"]["sha256"][block]
                        self.learner = CapacityConfirmationTailLearner.from_bytes(raw, expected_sha256=seal,
                            before_forward=self._forward, before_optimizer=self._optimizer)
                        method = "legacy"
                    else:
                        self.learner, method, seal = self._restore(record), record["method"], record["sha256"]
                self._episode(world, role, tape, method=method, seal=seal)

    def run(self):
        if self.started:
            raise RuntimeError("single attempt; no resume")
        self.started = True
        self.budget.enter(PHASES[0], PHASES[0])
        for block in range(3):
            for method in METHODS:
                for li in range(2):
                    self._train_job(PHASES[0], block, method, "graph", li)
        self.budget.enter(PHASES[1], PHASES[1])
        for block in range(3):
            entries = [("plain_h8", None), ("plain_h16", None)]
            for method in METHODS:
                for li in range(2):
                    role = f"{method}-lr{li}-graph"
                    entries.append((role, self.records[f"{PHASES[0]}-b{block}-{role}-final"]))
            self._evaluation(PHASES[1], block, entries)
        self.budget.enter("analysis_archive_verification")
        self.selection = select_finalists(summaries(self.root, PHASES[1]), self.study)
        self._json("selection.json", self.selection)
        self.status("finalists_selected", selection=self.selection)
        self.budget.enter(PHASES[2], PHASES[2])
        for block in range(5):
            for entry in self.selection["finalists"]:
                for architecture in ("graph", "self_only"):
                    self._train_job(PHASES[2], block, entry["method"], architecture, entry["lr_index"])
        self._json("models/all-finalists-sealed.json", self.records)
        self.budget.enter(PHASES[3], PHASES[3])
        for block in range(5):
            entries = [("plain_h8", None), ("plain_h16", None), ("legacy", dict(legacy=True))]
            for entry in self.selection["finalists"]:
                method, li = entry["method"], entry["lr_index"]
                for arch, version in (("graph", "final"), ("self_only", "final"), ("graph", "initial")):
                    record = self.records[f"{PHASES[2]}-b{block}-{method}-lr{li}-{arch}-{version}"]
                    entries.append((f"{method}-{arch}-{version}", record))
            self._evaluation(PHASES[3], block, entries)
        if len(self.completed) != 6744 or self.budget.counts["native_steps"] != 431616:
            raise ValueError("fixed native allocation incomplete")
        expected = expected_learning_counts(self.selection["finalists"])
        if any(self.budget.counts[k] != v for k, v in expected.items()):
            raise ValueError("actual optimizer/forward receipts disagree with selected recipe")
        self.finished = True
        self.budget.enter("analysis_archive_verification")
        self.status("all_trajectories_complete")
        return dict(trajectories=len(self.completed), selection=self.selection, budget=self.budget.snapshot())
