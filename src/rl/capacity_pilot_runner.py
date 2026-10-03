"""Real serial capacity comparison; scientific admission belongs to the launcher.

The same entry is injected with fake factories in integration tests. No retry,
hidden qualification, test-set selection, or additional rollout is available.
"""

from dataclasses import asdict, is_dataclass
from enum import Enum
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle

import numpy as np

from src.baselines.capacity_completion_control import CapacityCompletionControl
from src.env.patient_support_capacity import PatientSupportAction
from src.rl.capacity_adaptation_campaign import PHASES
from src.rl.capacity_ddpg_learner import CapacityDDPGLearner, CapacityTransition
from src.rl.capacity_native import CapacityPilotEnv, common_operation, keyed_seed, make_world_tape
from src.rl.capacity_public_features import public_features, resource_adjacency
from src.rl.public_support_input import PublicSupportControlInput
from src.utils.research_clock import shared_monotonic


COST_KEYS = ("reagent_purchase_cost", "reagent_holding_cost", "reagent_shortage_cost",
             "bioreactor_holding_cost", "bioreactor_shortage_cost", "specimen_transfer_cost",
             "capacity_transfer_cost", "reagent_transfer_cost", "patient_loss_cost", "expiry_cost",
             "urgency_cost", "support_ordinary_staff_cost", "support_flexible_labor_cost", "support_switching_cost")


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def worlds(proposal, phase, block):
    d = proposal["design"]
    base = {PHASES[0]: 62600000, PHASES[1]: 62604000, PHASES[2]: 62610000}[phase]
    for index in range(d["conditions"] * d["worlds_per_condition_per_phase_block"]):
        condition, replicate = index % d["conditions"], index // d["conditions"]
        yield dict(phase=phase, block=block, condition=condition, replicate=replicate,
                   seed=base + 1000 * block + (100 if phase == PHASES[2] else 10) * condition + replicate)


def trajectory_id(world, role):
    return f"{world['phase']}-b{world['block']}-c{world['condition']}-j{world['replicate']}-{role}"


class CapacityPilotRunner:
    def __init__(self, root, proposal, budget, *, admission, env_factory=CapacityPilotEnv,
                 learner_factory=CapacityDDPGLearner, controller_factory=CapacityCompletionControl,
                 tape_factory=make_world_tape, capture=PublicSupportControlInput.capture,
                 features=public_features, base_control=common_operation):
        if admission.get("verified") is not True or admission.get("scope") != "dynamic-capacity-pilot-v1":
            raise PermissionError("verified one-attempt launcher admission required")
        self.root, self.p, self.budget = Path(root), proposal, budget
        self.env_factory, self.learner_factory, self.controller_factory = env_factory, learner_factory, controller_factory
        self.tape_factory, self.capture, self.features, self.base_control = tape_factory, capture, features, base_control
        self.env = self.learner = self.controller = None
        self.epoch, self.active, self.finished = 0, None, False
        self.seals, self.completed = {}, []
        for directory in ("raw", "summaries", "models", "states", "tapes", "updates"):
            (self.root / directory).mkdir(exist_ok=False)

    def _json(self, relative, value):
        self.budget.check(storage=True)
        path = self.root / relative
        raw = (json.dumps(jsonable(value), sort_keys=True, allow_nan=False) + "\n").encode()
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        self.budget.check(storage=True)

    def status(self, event, **values):
        # Progress is append-only, unlike evidence snapshots it can share one file.
        with (self.root / "progress.jsonl").open("a", encoding="utf8") as handle:
            handle.write(json.dumps(jsonable(dict(event=event, active=self.active, epoch=self.epoch,
                                                 counts=self.budget.counts, **values)), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _native(self, kind):
        key = {"construction": "native_constructions", "construction_reset": "construction_triggered_resets",
               "step": "native_steps"}[kind]
        charges = {key: 1, "total_native_operations": 1}
        if kind == "step":
            charges["control_steps" if self.epoch < self.p["synthetic_system"]["control_epochs"] else "tail_steps"] = 1
        self.budget.debit(charges)

    def _forward(self, name, batch):
        if not 1 <= batch <= self.p["proposed_budget"]["neural_forward_max_batch"]:
            raise ValueError("module forward batch cap")
        self.budget.debit({"neural_forward_module_calls": 1})

    def _optimizer(self, name, batch):
        self.budget.debit({f"{name}_optimizer_steps": 1, "total_optimizer_steps": 1,
                           "optimizer_example_presentations": batch})

    def _filter(self, payload):
        total = len(self.p["synthetic_system"]["site_ids"]) * self.p["information"]["estimator"]["pairs_per_site_receipt"]
        self.budget.chunk_call("estimator_hypothesis_transitions", total, payload["hypothesis_transitions"])

    def _query(self, payload):
        self.budget.chunk_call("planner_total_model_epochs", self.p["proposed_budget"]["planner_model_epochs_per_decision"], payload["model_epochs"])

    def _observe(self, view):
        self.controller.observe(view, before_filter=self._filter)
        if view.common.epoch:
            self.budget.finish_chunk("estimator_hypothesis_transitions")

    def _fit(self, mode):
        key = {"bc": "bc_actor_steps", "warmup": "critic_only_warmup_steps",
               "offline": "ddpg_update_pairs", "online": "ddpg_update_pairs"}[mode]
        self.budget.debit({key: 1})
        method = {"bc": "bc_step", "warmup": "critic_warmup_step", "offline": "offline_ddpg_step", "online": "online_ddpg_step"}[mode]
        started = shared_monotonic()
        receipt = getattr(self.learner, method)()
        duration = shared_monotonic()-started
        with (self.root / "updates" / f"block{self.active['block']}.jsonl").open("a", encoding="utf8") as handle:
            handle.write(json.dumps(jsonable(dict(active=self.active, epoch=self.epoch, mode=mode,
                                                receipt=receipt, duration_seconds=duration, counts=self.learner.counts)), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _save_state(self, name, *, seal_reference=None):
        state = dict(environment=None if self.env is None else self.env.state_dict(),
                     controller=None if self.controller is None else self.controller.state_dict(),
                     learner=None if self.learner is None else self.learner.state_dict(),
                     active=self.active, epoch=self.epoch, budget=self.budget.snapshot(), resume_authorized=False)
        if seal_reference is not None and state["learner"] is not None:
            # Offline rows are immutable in evaluation; reference the verified seal.
            state["learner"]["offline_replay"] = {"sealed_reference": seal_reference}
        self.budget.check(storage=True)
        with gzip.open(self.root / "states" / f"{name}.pkl.gz", "xb") as handle:
            pickle.dump(state, handle, protocol=5)
        self.budget.check(storage=True)

    def _episode(self, world, role, tape, *, seal_reference=None):
        p, s = self.p, self.p["synthetic_system"]
        self.active, self.epoch = dict(world, role=role), 0
        identifier = trajectory_id(world, role)
        self.budget.debit({"trajectories": 1})
        self.env = self.env_factory(p, tape, self._native)
        self.controller = self.controller_factory(p, base_control=self.base_control, scientific=True)
        public = self.capture(self.env)
        self._observe(public)
        states = self.features(public, self.controller.filter, p)
        noise_rng = np.random.default_rng(keyed_seed(p, world, "exploration"))
        noises = noise_rng.normal(0, p["learner"]["exploration"]["sigma_hours"], (s["control_epochs"], len(s["site_ids"])))
        total, tail_total, terminal = 0., 0., None
        teacher_rows = []
        raw_path = f"raw/{identifier}.jsonl.gz"
        self.status("trajectory_started")
        with gzip.open(self.root / raw_path, "xt", encoding="utf8") as handle:
            for epoch in range(s["host_total_horizon"]):
                self.epoch = epoch
                tail = epoch >= s["control_epochs"]
                action_started = shared_monotonic()
                if tail:
                    raw_hours = np.zeros(len(s["site_ids"]), dtype=float)
                elif role in ("teacher", "id_mpc"):
                    kind = "planner_teacher_decisions" if role == "teacher" else "planner_evaluation_decisions"
                    self.budget.debit({kind: 1, "planner_total_decisions": 1,
                                      "planner_candidate_rollouts": p["id_mpc"]["candidate_sequences_per_decision"] * len(p["id_mpc"]["response_quantiles"])})
                    raw_hours = self.controller.act(public, role="id_mpc", before_query=self._query, before_filter=self._filter)
                    self.budget.finish_chunk("planner_total_model_epochs")
                elif role in ("adaptive_rule", "fixed_allocation_reference"):
                    raw_hours = self.controller.act(public, role=role, before_filter=self._filter)
                else:
                    explore = role != "frozen_history"
                    raw_hours, _ = self.learner.act(states, explore=explore, noise_hours=noises[epoch] if explore else None)
                base = self.base_control(public, p, tail=tail)
                self.controller.record_operation(public, base)
                action_seconds = shared_monotonic()-action_started
                native_started = shared_monotonic()
                _, reward, done, info = self.env.step(PatientSupportAction(tuple(float(v) for v in base), tuple(float(v) for v in raw_hours)))
                native_seconds = shared_monotonic()-native_started
                cost = float(info["cost"])
                components = {k: float(info[k]) for k in COST_KEYS}
                if (not math.isfinite(cost) or cost < 0 or not math.isclose(math.fsum(components.values()), cost, abs_tol=1e-6, rel_tol=1e-12)
                        or not math.isclose(-float(reward), cost, abs_tol=1e-6, rel_tol=1e-12)
                        or bool(done) != (epoch == s["host_total_horizon"] - 1)):
                    raise ValueError("raw cost/reward/terminal mismatch")
                next_public = self.capture(self.env)
                self.budget.debit({"estimator_receipt_updates": 1})
                filter_started = shared_monotonic()
                self._observe(next_public)
                filter_seconds = shared_monotonic()-filter_started
                next_states = self.features(next_public, self.controller.filter, p)
                receipt = info["support_public_receipt"]
                executed = receipt["committed_hours"]
                total += cost
                if tail:
                    tail_total += cost
                records = next_public.operations.patients
                cumulative = dict(enrolled=len(records), delivered=sum(x.status == "delivered" for x in records),
                                  lost=sum(x.status == "lost" for x in records))
                row = dict(epoch=epoch, world=world, role=role, cost=cost, components=components, reward=float(reward),
                           requested_hours=raw_hours, executed_hours=executed,
                           filter_resets=len(self.controller.filter.reset_events), cumulative=cumulative,
                           support_backlog=sum(not x.support_complete and x.status not in ("lost", "delivered") for x in records),
                           patient_records=records, service=next_public.operations.last_service, info=info,
                           timing_seconds=dict(controller=action_seconds,native=native_seconds,filter=filter_seconds),
                           public_input=public, filter_summary=self.controller.filter.node_summaries())
                handle.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + "\n")
                if (epoch + 1) % p["proposed_budget"]["flush_every_epochs"] == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    self.budget.check(storage=True)
                    self.status("epoch_boundary")
                if not tail:
                    transition = dict(state=states, executed_hours=executed, next_state=next_states,
                                      world_id=identifier, control_cost=cost, reward_divisor=100000.)
                    if epoch == s["control_epochs"] - 1:
                        terminal = transition
                    else:
                        self._transition(transition, role, teacher_rows, done=False, settlement_cost=None)
                public, states = next_public, next_states
        settlement = self.env.settlement()
        summary = dict(world=world, role=role, raw_path=raw_path, settled=settlement["settled"], settlement=settlement,
                       cost=total, change_epoch=tape.change_epoch, tape_sha256=tape.digest(), model_seal_sha256=seal_reference)
        self._json(f"summaries/{identifier}.json", summary)
        self.epoch = s["host_total_horizon"]
        if not settlement["settled"]:
            self._save_state(identifier + "-unsettled", seal_reference=seal_reference)
            raise ValueError("fixed tail not settled; no cost-complete claim or extension")
        terminal["next_state"] = states
        self._transition(terminal, role, teacher_rows, done=True, settlement_cost=tail_total)
        self._save_state(identifier, seal_reference=seal_reference)
        self.completed.append(identifier)
        self.status("trajectory_completed", summary=summary)
        return teacher_rows

    def _transition(self, arguments, role, teacher_rows, *, done, settlement_cost):
        row = CapacityTransition.from_cost(**arguments, done=done, settlement_cost=settlement_cost)
        if role == "teacher":
            teacher_rows.append(row)
        elif role == "learner":
            self.learner.add_offline(row)
            self._fit("offline")
        elif role == "online_matched_fork":
            self.learner.add_world(row)
            # Receipt 48 is deliberately deferred until its complete tail closes.
            if done or self.epoch + 1 >= self.p["learner"]["online_update_receipts_inclusive"][0]:
                self._fit("online")

    def _tape(self, world):
        tape = self.tape_factory(self.p, world)
        self._json(f"tapes/{trajectory_id(world, 'exogenous')}.json", asdict(tape))
        return tape

    def run(self):
        if self.finished or self.active is not None:
            raise RuntimeError("one serial invocation only")
        p = self.p
        for block, model_seed in enumerate(p["design"]["training_seeds"]):
            self.budget.enter("initialization_training_and_seals", PHASES[0])
            teacher_rows = []
            self.learner = None
            for world in worlds(p, PHASES[0], block):
                teacher_rows.extend(self._episode(world, "teacher", self._tape(world)))
            self.budget.debit({"fresh_actor_critic_pairs": 1})
            self.learner = self.learner_factory.from_proposal(p, node_input_dim=len(teacher_rows[0].state[0]),
                adjacency=resource_adjacency(p), model_seed=model_seed,
                replay_seed=keyed_seed(p, world, "replay"), before_forward=self._forward, before_optimizer=self._optimizer)
            for row in teacher_rows:
                self.learner.add_offline(row)
            for _ in range(p["learner"]["bc_updates_per_block"]):
                self._fit("bc")
            for _ in range(p["learner"]["critic_warmup_updates_per_block"]):
                self._fit("warmup")
            self.status("initializer_completed")
            self.budget.enter("initialization_training_and_seals", PHASES[1])
            for world in worlds(p, PHASES[1], block):
                self._episode(world, "learner", self._tape(world))
            if len(self.learner.offline_replay) != p["learner"]["offline_replay_rows_per_seal"]:
                raise ValueError("offline seal row count mismatch")
            seal = self.learner.snapshot()
            with (self.root / "models" / f"block{block}-seal.pt").open("xb") as handle:
                handle.write(seal.payload)
                handle.flush()
                os.fsync(handle.fileno())
            self.seals[block] = seal
            self._json(f"models/block{block}-seal.json", dict(sha256=seal.sha256, bytes=len(seal.payload),
                                                            block=block, counts=self.learner.counts))
            self.status("model_sealed", seal=seal.sha256)
        if len(self.seals) != p["design"]["blocks"]:
            raise RuntimeError("test worlds must remain unopened until every model is sealed")
        self.budget.enter("evaluation_and_settlement", PHASES[2])
        for block in range(p["design"]["blocks"]):
            for world in worlds(p, PHASES[2], block):
                tape = self._tape(world)
                for controller in p["controllers"]:
                    role = controller["id"]
                    self.learner = None
                    seal = self.seals[block]
                    if role in ("frozen_history", "frozen_matched_exploration", "online_matched_fork"):
                        self.budget.debit({"learned_evaluation_arm_restorations": 1})
                        self.learner = self.learner_factory.from_snapshot(seal, before_forward=self._forward, before_optimizer=self._optimizer)
                        self.learner.begin_world(trajectory_id(world, role), replay_seed=keyed_seed(p, world, "replay"),
                                                 exploration_seed=keyed_seed(p, world, "exploration"))
                    self._episode(world, role, tape, seal_reference=seal.sha256)
        if self.budget.counts != self.budget.limits:
            raise RuntimeError(f"final counters differ: {[(k,self.budget.counts[k],v) for k,v in self.budget.limits.items() if self.budget.counts[k]!=v]}")
        self.finished = True
        self.status("all_trajectories_complete")
        return dict(trajectories=len(self.completed), model_seals={k: v.sha256 for k,v in self.seals.items()}, budget=self.budget.snapshot())
