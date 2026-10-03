"""Single admitted value-learning/MPC comparison, reusing native IO boundaries."""

import gzip
import json
import math
import os
import pickle

import numpy as np

from src.baselines.capacity_value_mpc import CapacityValueMPC
from src.env.patient_support_capacity import PatientSupportAction
from src.rl.capacity_pilot_runner import CapacityPilotRunner, COST_KEYS, jsonable, trajectory_id
from src.rl.capacity_value_features import FEATURE_NAMES, observed_features
from src.rl.capacity_value_learner import CapacityValueLearner, td_rows
from src.rl.capacity_value_resources import PHASES, worlds


class CapacityValueRunner(CapacityPilotRunner):
    def __init__(self, root, proposal, budget, *, admission, value_factory=CapacityValueLearner,
                 feature_adapter=observed_features, **backends):
        if admission != dict(verified=True, scope="capacity-value-mpc-v1"):
            raise PermissionError("new exact value-MPC admission required")
        backends.setdefault("controller_factory", CapacityValueMPC)
        super().__init__(root, proposal, budget,
            admission=dict(verified=True, scope="dynamic-capacity-pilot-v1"), **backends)
        self.value_factory, self.feature_adapter = value_factory, feature_adapter
        self.study = proposal["value_mpc_study"]
        self.initial, self.final, self.episode_data = {}, {}, None
        self.started = False

    def _optimizer(self, name, batch):
        if name != "value" or self.budget.phase == PHASES[2]:
            raise PermissionError("only admitted training value updates")
        self.budget.debit(dict(value_optimizer_steps=1, total_optimizer_steps=1,
                               optimizer_example_presentations=batch))

    def state_dict(self):
        return dict(format="capacity-value-runner-v1", active=self.active, epoch=self.epoch,
            environment=None if self.env is None else self.env.state_dict(),
            controller=None if self.controller is None else self.controller.state_dict(),
            learner=None if self.learner is None else self.learner.state_dict(),
            episode_data=self.episode_data, completed=list(self.completed),
            initial={k:v[1] for k,v in self.initial.items()}, final={k:v[1] for k,v in self.final.items()},
            budget=self.budget.snapshot(), resume_authorized=False)

    def _save_state(self, name):
        self.budget.check(storage=True)
        with gzip.open(self.root/"states"/(name+".pkl.gz"), "xb") as handle:
            pickle.dump(self.state_dict(), handle, protocol=5)
        self.budget.check(storage=True)

    def _seal_value(self, block, final=False):
        name = f"block{block}-"+("final" if final else "initial")
        self.budget.job(name, "seal")
        raw, sha = self.learner.snapshot()
        self.budget.debit({"final_seals" if final else "initial_seals": 1})
        with (self.root/"models"/(name+".pt")).open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        self._json("models/"+name+".json", dict(sha256=sha, bytes=len(raw), updates=self.learner.updates))
        (self.final if final else self.initial)[block] = raw, sha
        self.status("final_sealed" if final else "initial_sealed", block=block, sha256=sha)

    def _episode(self, world, role, tape, *, seal=None):
        self.active, self.epoch = dict(world, role=role), 0
        ident = trajectory_id(world, role)
        self.budget.job(ident, "world")
        self.budget.debit(dict(trajectories=1))
        self.env = self.env_factory(self.p, tape, self._native)
        learned = role in ("continuation", "initial_value_mpc", "updated_value_mpc")
        self.controller = self.controller_factory(self.p, base_control=self.base_control,
                                                  scientific=True, value=self.learner if learned else None)
        public = self.capture(self.env)
        self._observe(public)
        states, heuristic = self.feature_adapter(public, self.controller)
        data = dict(features=[states], heuristics=[heuristic], costs=[])
        self.episode_data = data
        lost_ids, total = set(), 0.
        self.status("trajectory_started")
        with gzip.open(self.root/"raw"/(ident+".jsonl.gz"), "xt") as handle:
            for epoch in range(64):
                self.epoch = epoch
                if epoch >= 48:
                    hours = np.zeros(4)
                elif role == "fixed_allocation_reference":
                    hours = self.controller.act(public, role=role, before_filter=self._filter)
                else:
                    self.budget.debit(dict(planner_total_decisions=1, planner_candidate_rollouts=48))
                    hours = self.controller.act(public, role="id_mpc", before_query=self._query, before_filter=self._filter)
                    self.budget.finish_chunk("planner_total_model_epochs")
                base = self.base_control(public, self.p, tail=epoch >= 48)
                self.controller.record_operation(public, base)
                _, reward, done, info = self.env.step(PatientSupportAction(tuple(map(float, base)), tuple(map(float, hours))))
                cost = float(info["cost"])
                components = {k:float(info[k]) for k in COST_KEYS}
                if (not math.isfinite(cost) or cost < 0 or not all(math.isfinite(x) for x in components.values())
                        or not math.isclose(math.fsum(components.values()), cost, rel_tol=1e-12, abs_tol=1e-6)
                        or not math.isclose(-float(reward), cost, rel_tol=1e-12, abs_tol=1e-6)
                        or bool(done) != (epoch == 63)):
                    raise ValueError("native objective or terminal mismatch")
                following = self.capture(self.env)
                self.budget.debit(dict(estimator_receipt_updates=1))
                self._observe(following)
                x, h = self.feature_adapter(following, self.controller)
                data["features"].append(x)
                data["heuristics"].append(h)
                data["costs"].append(cost)
                records = following.operations.patients
                losses = {p.patient_id for p in records if p.status == "lost"}
                if not lost_ids.issubset(losses):
                    raise ValueError("lost patient reopened")
                raw = dict(epoch=epoch, world=world, role=role, cost=cost, reward=float(reward), components=components,
                    new_lost_patients=len(losses-lost_ids), patient_records=records,
                    cumulative=dict(enrolled=len(records), lost=len(losses), delivered=sum(p.status=="delivered" for p in records)),
                    requested_hours=hours, executed_hours=info["support_public_receipt"]["committed_hours"],
                    public_input=public, value_features=data["features"][-2], value_base=data["heuristics"][-2],
                    next_value_features=x, next_value_base=h, plan=self.controller.last_plan if epoch < 48 else None, info=info)
                handle.write(json.dumps(jsonable(raw), sort_keys=True, allow_nan=False)+"\n")
                lost_ids, public, total = losses, following, total+cost
                if (epoch+1) % 8 == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    self.budget.check(storage=True)
                    self.status("epoch_boundary")
        settlement = self.env.settlement()
        summary = dict(world=world, role=role, raw_path="raw/"+ident+".jsonl.gz", cost=total,
            settled=settlement["settled"], settlement=settlement, lost=len(lost_ids), tape_sha256=tape.digest(),
            change_epoch=tape.change_epoch, model_seal_sha256=seal if world["phase"] == "evaluation" else None,
            initial_ancestor_sha256=seal if world["phase"] == "continuation" else None,
            value_updates_before_trajectory=None if self.learner is None else self.learner.updates,
            optimizer_updates_during_trajectory=0)
        self._json("summaries/"+ident+".json", summary)
        if not settlement["settled"] or settlement["lost"] != len(lost_ids):
            raise ValueError("unsettled cohort; no horizon extension")
        self.epoch = 64
        self.completed.append(ident)
        self._save_state(ident)
        self.status("trajectory_completed", summary=summary)
        self.episode_data = None
        return data

    def _learn(self, data):
        ident = trajectory_id({k:v for k,v in self.active.items() if k != "role"}, self.active["role"])
        self.budget.job(ident+"-fit", "fit")
        c = self.study["value"]
        rows = td_rows(**data, horizon=c["td_horizon"], scale=c["cost_scale"])
        self.learner.admit_episode(rows)
        self._json("updates/"+ident+"-targets.json", dict(targets=self.learner.pending["targets"],
            td_horizon=c["td_horizon"], terminal_bootstrap_zero=True))
        with (self.root/"updates"/(ident+".jsonl")).open("x") as handle:
            for _ in range(c["updates_per_world"]):
                receipt = self.learner.update()
                handle.write(json.dumps(jsonable(receipt), allow_nan=False)+"\n")
                handle.flush()
                os.fsync(handle.fileno())
        self._save_state(ident+"-after-fit")
        self.status("value_fit_completed", updates=self.learner.updates)

    def run(self):
        if self.started:
            raise RuntimeError("single attempt; no resume/relaunch")
        self.started = True
        for block, seed in enumerate(self.study["design"]["model_seeds"]):
            self.budget.enter(PHASES[0], PHASES[0])
            self.learner = self.value_factory(self.study["value"], seed=seed, feature_dim=len(FEATURE_NAMES),
                                            before_forward=self._forward, before_optimizer=self._optimizer)
            for w in worlds(self.study, "initial", block):
                self._learn(self._episode(w, "plain_mpc", self._tape(w)))
            self._seal_value(block)
            self.budget.enter(PHASES[1], PHASES[1])
            self.learner = self.value_factory.from_bytes(self.initial[block][0], before_forward=self._forward,
                                                         before_optimizer=self._optimizer)
            for w in worlds(self.study, "continuation", block):
                self._learn(self._episode(w, "continuation", self._tape(w), seal=self.initial[block][1]))
            self._seal_value(block, final=True)
        if len(self.final) != 3:
            raise RuntimeError("all final models required before tests")
        self.status("all_models_sealed")
        self.budget.enter(PHASES[2], PHASES[2])
        for block in range(3):
            for w in worlds(self.study, "evaluation", block):
                tape = self._tape(w)
                for role in self.study["design"]["roles"]:
                    value = self.initial.get(block) if role == "initial_value_mpc" else self.final.get(block) if role == "updated_value_mpc" else None
                    self.learner = None if value is None else self.value_factory.from_bytes(value[0],
                        before_forward=self._forward, before_optimizer=self._optimizer)
                    self._episode(w, role, tape, seal=None if value is None else value[1])
        if self.budget.counts != self.budget.limits:
            raise RuntimeError("value study completion counts mismatch")
        self.finished = True
        self.status("all_trajectories_complete")
        return dict(trajectories=len(self.completed), initial_models={k:v[1] for k,v in self.initial.items()},
                    final_models={k:v[1] for k,v in self.final.items()}, budget=self.budget.snapshot())
