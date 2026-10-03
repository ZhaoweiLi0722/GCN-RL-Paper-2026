"""Versioned, in-memory serial rehearsal of the capacity pilot, FAKE ONLY.

No scientific factory, launch command, result root, tensor or optimizer lives
here. All counts describe artificial protocol dispatches, not fitted algorithms
or patient evidence. A trusted injected test double must declare the protocol;
that declaration is not a sandbox for arbitrary Python supplied by a caller.

Native binding remains separate work: the construction_reset event models a
reset *inside* construction, not permission for an extra native reset. Likewise
planner_epoch means one approximate model query, never a patient-env rollout.
Wall/RSS/disk enforcement, durable recording, real replay and CRNs are not
implemented by this in-memory interface. Restored snapshots are inspection-only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Protocol

from src.rl.candidate_pilot_resources import digest
from src.rl.public_support_collector import CAPACITY_CONTROL_ROLES


FORMAT = "capacity-adaptation-artificial-campaign-v1"
BACKEND_PROTOCOL = "capacity-adaptation-artificial-backend-v1"
STATE_KEYS = frozenset(("actor", "critic", "target_actor", "target_critic",
                        "actor_optimizer", "critic_optimizer", "replay"))
PHASES = ("teacher_bc_critic_warmup", "offline_ddpg", "six_arm_evaluation")


@dataclass(frozen=True)
class ArtificialCall:
    """One dispatch. Payloads are detached JSON data, never scientific objects.

    initialize/restore/remember/*_step/targets return a STATE_KEYS dictionary of
    artificial tokens. construction_reset and control return arbitrary JSON
    public/action tokens. control_step/tail_step return ArtificialReceipt.
    filter returns a JSON filter token; other operations return None.
    No operation may do scientific work.
    """

    kind: str
    phase: str
    block: int
    owner: str | None
    epoch: int | None
    payload: dict


@dataclass(frozen=True)
class ArtificialReceipt:
    epoch: int
    public: dict
    raw_cost: float
    settled: bool = False


class ArtificialBackend(Protocol):
    backend_protocol: str

    def dispatch(self, call: ArtificialCall) -> object:
        """Execute only an artificial record operation, never a real backend."""
        ...


def _json_data(value):
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_data(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():
            _json_data(item)
        return
    raise TypeError("artificial protocol accepts finite JSON data only")


def _state(value):
    _json_data(value)
    if type(value) is not dict or set(value) != STATE_KEYS:
        raise ValueError("complete artificial actor/critic/targets/optimizers/replay required")
    return copy.deepcopy(value)


def numeric_contract(proposal):
    """Derive and reconcile the existing proposal; does not authorize science."""
    if (proposal.get("schema") != "dynamic-capacity-pilot-proposal-v1"
            or proposal.get("scientific_execution_authorized") is not False
            or proposal.get("execution_packet_ready") is not False
            or not proposal.get("current_scientific_allowance")
            or any(type(v) is not int or v != 0
                   for v in proposal["current_scientific_allowance"].values())):
        raise PermissionError("fake-only interface; scientific execution is disabled")
    d, s, l, p = (proposal[k] for k in ("design", "synthetic_system", "learner", "proposed_budget"))
    if tuple(row["id"] for row in proposal["controllers"]) != CAPACITY_CONTROL_ROLES:
        raise ValueError("exact six-controller inventory required")
    values = [d[k] for k in ("blocks", "conditions", "worlds_per_condition_per_phase_block")]
    values += [s[k] for k in ("control_epochs", "settlement_epochs", "host_total_horizon")]
    values += [l[k] for k in ("bc_updates_per_block", "critic_warmup_updates_per_block", "batch_size")]
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("positive native integer dimensions required")
    blocks, conditions, worlds, control, tail, horizon, bc, warm, batch = values
    seeds = d["training_seeds"]
    if (len(seeds) != blocks or any(type(v) is not int for v in seeds)
            or len(set(seeds)) != blocks or len(proposal["conditions"]) != conditions
            or [r["index"] for r in proposal["conditions"]] != list(range(conditions))
            or control + tail != horizon):
        raise ValueError("seed/condition/horizon mismatch")
    first, last = l["online_update_receipts_inclusive"]
    if (type(first) is not int or type(last) is not int
            or not 1 <= first <= last == control):
        raise ValueError("online schedule must end at the last control receipt")
    per_block, trajectories = conditions * worlds, blocks * conditions * worlds
    online, pairs = last - first + 1, trajectories * (control + last - first + 1)
    planner = proposal["id_mpc"]
    candidates, summaries, prediction = (planner["candidate_sequences_per_decision"],
                                        len(planner["response_quantiles"]), planner["predictive_horizon"])
    if any(type(v) is not int or v <= 0 for v in (candidates, summaries, prediction)):
        raise ValueError("positive integer planner dimensions required")
    native = trajectories * 8 * horizon
    actor, critic = blocks * bc + pairs, blocks * warm + pairs
    decisions = 2 * trajectories * control
    limits = dict(
        trajectories=trajectories * 8, control_steps=trajectories * 8 * control,
        tail_steps=trajectories * 8 * tail, native_steps=native,
        native_constructions=trajectories * 8, construction_triggered_resets=trajectories * 8,
        extra_explicit_resets_or_clones=0, total_native_operations=trajectories * 8 * (horizon + 2),
        actor_optimizer_steps=actor, critic_optimizer_steps=critic,
        total_optimizer_steps=actor + critic, optimizer_example_presentations=(actor + critic) * batch,
        neural_forward_module_calls=4 * trajectories * control + blocks * bc + 3 * blocks * warm + 5 * pairs,
        bc_actor_steps=blocks * bc, critic_only_warmup_steps=blocks * warm, ddpg_update_pairs=pairs,
        fresh_actor_critic_pairs=blocks, historical_model_loads=0,
        learned_evaluation_arm_restorations=trajectories * 3,
        planner_teacher_decisions=trajectories * control, planner_evaluation_decisions=trajectories * control,
        planner_total_decisions=decisions, planner_candidate_rollouts=decisions * candidates * summaries,
        planner_total_model_epochs=decisions * candidates * summaries * prediction,
        estimator_receipt_updates=native,
        estimator_hypothesis_transitions=native * len(s["site_ids"]) * proposal["information"]["estimator"]["pairs_per_site_receipt"],
    )
    for key, expected in limits.items():
        actual = d["total_trajectories"] if key == "trajectories" else p[key]
        if type(actual) is not int or actual != expected:
            raise ValueError(f"proposal arithmetic mismatch: {key}")
    equalities = (
        (d["teacher_trajectories_per_block"], per_block),
        (d["learner_trajectories_per_block"], per_block),
        (d["evaluation_worlds_per_block"], per_block), (d["evaluation_arms"], 6),
        (d["evaluation_trajectories"], trajectories * 6),
        (l["offline_ddpg_pairs_per_block"], per_block * control),
        (l["offline_replay_rows_per_seal"], per_block * control * 2),
        (l["online_replay_max_rows"], per_block * control * 2 + control),
        (l["online_pairs_per_world"], online),
        (p["planner_model_epochs_per_decision"], candidates * summaries * prediction),
    )
    if any(type(a) is not int or a != b for a, b in equalities):
        raise ValueError("proposal stage/replay arithmetic mismatch")
    seed_formulas = {"teacher_world_seed": "62600000 + 1000*b + 10*c + j",
                     "learner_world_seed": "62604000 + 1000*b + 10*c + j",
                     "evaluation_world_seed": "62610000 + 1000*b + 100*c + j"}
    if (any(d[k] != v for k, v in seed_formulas.items())
            or d["phase_order"] != "episode_c_equals_index_mod_3_j_equals_floor_index_over_3"
            or p["neural_forward_max_batch"] != batch):
        raise ValueError("unsupported seed/order/forward contract")
    phase_limits = {}
    for name, multiplier, a, c in ((PHASES[0], 1, blocks * bc, blocks * warm),
                                  (PHASES[1], 1, trajectories * control, trajectories * control),
                                  (PHASES[2], 6, trajectories * online, trajectories * online)):
        n = trajectories * multiplier
        phase_limits[name] = dict(trajectories=n, control_steps=n * control,
                                 tail_steps=n * tail, native_steps=n * horizon,
                                 actor_optimizer_steps=a, critic_optimizer_steps=c)
    if p["phases"] != [dict(id=k, **v) for k, v in phase_limits.items()]:
        raise ValueError("proposal phase budgets differ")
    settlement = proposal["settlement"]
    if (settlement["tail_epochs_inclusive"] != [control, horizon - 1]
            or any(settlement[k] != 0 for k in ("tail_neural_actions", "tail_transition_optimizer_calls", "tail_planner_queries"))
            or settlement["deferred_last_control_update"] != "already_counted_in_fixed_pairs_runs_after_tail_cost_is_attached"
            or proposal["scope"]["resume"] is not False or p["unused_budget_reusable"] is not False):
        raise ValueError("tail schedule/no-resume contract differs")
    return {"limits": limits, "phase_limits": phase_limits}


class CapacityAdaptationCampaign:
    """One serial fake attempt; every invocation is charged before dispatch.

    run() completes the entire numeric packet or latches its first failure.
    State restoration never reopens an attempt, even from an unused snapshot.
    A backend receives copies: it cannot mutate another arm's seal/history by
    retaining or modifying a payload reference. No real backend is accepted.
    """

    def __init__(self, proposal, backend: ArtificialBackend):
        self._proposal = copy.deepcopy(proposal)
        self._contract = numeric_contract(self._proposal)
        self._backend = backend
        self._guard()
        self._counts = dict.fromkeys(self._contract["limits"], 0)
        self._phase_counts = {k: dict.fromkeys(self._counts, 0) for k in PHASES}
        self._started = self._restored = False
        self._status, self._failure = "new", None
        self._phase, self._block, self._epoch = PHASES[0], 0, None
        self._owner, self._session = None, None
        self._seals, self._completed = {}, []
        self._training_state, self._offline_rows = None, 0
        self._dispatches = 0

    def _guard(self):
        if (getattr(self._backend, "backend_protocol", None) != BACKEND_PROTOCOL
                or not callable(getattr(self._backend, "dispatch", None))):
            raise PermissionError("explicit artificial backend protocol required; no scientific backend")

    def _call(self, kind, charges=None, **payload):
        if self._status != "running" or self._failure is not None or self._restored:
            raise RuntimeError("attempt is not dispatchable")
        charges = charges or {}
        phase_caps = self._contract["phase_limits"][self._phase]
        for key, amount in charges.items():
            if (key not in self._counts or type(amount) is not int or amount <= 0
                    or self._counts[key] + amount > self._contract["limits"][key]
                    or (key in phase_caps and self._phase_counts[self._phase][key] + amount > phase_caps[key])):
                raise RuntimeError(f"nonrefundable artificial budget exhausted: {key}")
        # Debit even when dispatch raises or its returned record fails validation.
        for key, amount in charges.items():
            self._counts[key] += amount
            self._phase_counts[self._phase][key] += amount
        self._dispatches += 1
        return self._backend.dispatch(ArtificialCall(
            kind, self._phase, self._block, self._owner, self._epoch, copy.deepcopy(payload)))

    def _forward(self, module, state, *, batch):
        self._call("forward", {"neural_forward_module_calls": 1}, module=module, state=state, batch=batch)

    def _update(self, state, mode, *, transition=None):
        batch = self._proposal["learner"]["batch_size"]
        if mode == "bc":
            groups = (("actor", ("actor",)),)
        elif mode == "warmup":
            groups = (("critic", ("target_actor", "target_critic", "critic")),)
        else:
            groups = (("critic", ("target_actor", "target_critic", "critic")),
                      ("actor", ("actor", "actor_loss_critic")))
            self._call("ddpg_pair", {"ddpg_update_pairs": 1}, transition=transition)
        for component, modules in groups:
            for module in modules:
                self._forward(module, state, batch=batch)
            charges = {f"{component}_optimizer_steps": 1, "total_optimizer_steps": 1,
                       "optimizer_example_presentations": batch}
            if mode in ("bc", "warmup"):
                charges["bc_actor_steps" if mode == "bc" else "critic_only_warmup_steps"] = 1
            state = _state(self._call(f"{component}_step", charges, state=state,
                                      mode=mode, transition=transition, batch=batch))
            if self._session is not None:
                self._session["state"] = state
            if self._phase != PHASES[2]:
                self._training_state = state
        if mode != "bc":
            state = _state(self._call("targets", state=state, mode=mode))
            if self._session is not None:
                self._session["state"] = state
            if self._phase != PHASES[2]:
                self._training_state = state
        return state

    def _plan(self, public, history):
        counter = "planner_teacher_decisions" if self._phase == PHASES[0] else "planner_evaluation_decisions"
        self._call("planner_decision", {"planner_total_decisions": 1, counter: 1}, public=public, history=history)
        p = self._proposal["id_mpc"]
        for candidate in range(p["candidate_sequences_per_decision"]):
            for summary in range(len(p["response_quantiles"])):
                self._call("planner_rollout", {"planner_candidate_rollouts": 1}, candidate=candidate, summary=summary)
                for predicted_epoch in range(p["predictive_horizon"]):
                    self._call("planner_epoch", {"planner_total_model_epochs": 1},
                               candidate=candidate, summary=summary, predicted_epoch=predicted_epoch)

    def _worlds(self, phase):
        d = self._proposal["design"]
        base, condition_stride = {"teacher": (62600000, 10), "learner": (62604000, 10),
                                  "evaluation": (62610000, 100)}[phase]
        for episode in range(d["conditions"] * d["worlds_per_condition_per_phase_block"]):
            c, j = episode % d["conditions"], episode // d["conditions"]
            yield dict(phase=phase, block=self._block, condition=c, replicate=j,
                       seed=base + 1000 * self._block + condition_stride * c + j)

    def _episode(self, world, role, state):
        s, l = self._proposal["synthetic_system"], self._proposal["learner"]
        control, horizon = s["control_epochs"], s["host_total_horizon"]
        self._owner = f"{world['phase']}/{self._block}/{world['condition']}/{world['replicate']}/{role}"
        self._epoch = None
        state = None if state is None else _state(state)
        session = dict(state=state, history=[], public=None, transitions=0, total_cost=0.0,
                       filter=None, world=world, role=role, pending_transition=None)
        self._session = session
        self._call("construct", {"native_constructions": 1, "total_native_operations": 1, "trajectories": 1}, world=world)
        session["public"] = self._call("construction_reset", {"construction_triggered_resets": 1,
                                      "total_native_operations": 1}, world=world, history=[], filter=None)
        _json_data(session["public"])
        session["public"] = copy.deepcopy(session["public"])
        if world["phase"] == "evaluation" and role in CAPACITY_CONTROL_ROLES[:3]:
            restored = _state(self._call("restore", {"learned_evaluation_arm_restorations": 1}, state=state))
            if restored != state:
                raise ValueError("learned fork must start from the identical complete seal")
            state = restored
        pending = None
        for epoch in range(horizon):
            self._epoch = epoch
            before = copy.deepcopy(session["public"])
            if epoch < control:
                if role == "id_mpc":
                    self._plan(before, session["history"])
                if role in CAPACITY_CONTROL_ROLES[:3] or role == "offline_learner":
                    self._forward("behavior_actor", state, batch=1)
                noise_key = digest(dict(namespace=self._proposal["design"]["rng_namespace"],
                                        world=world, purpose="exploration", epoch=epoch))
                action = self._call("control", public=before, history=session["history"],
                                    filter=session["filter"], state=state, role=role,
                                    noise_key=noise_key if role in ("frozen_matched_exploration", "online_matched_fork", "offline_learner") else None)
                _json_data(action)
                action = copy.deepcopy(action)
            else:
                action = {"artificial_tail": True, "requested_hours": self._proposal["settlement"]["new_flexible_commitment"]}
            kind = "control_step" if epoch < control else "tail_step"
            receipt = self._call(kind, {"native_steps": 1, "total_native_operations": 1,
                                       "control_steps" if epoch < control else "tail_steps": 1}, action=action)
            if (type(receipt) is not ArtificialReceipt or type(receipt.epoch) is not int
                    or receipt.epoch != epoch or type(receipt.raw_cost) not in (float, int)
                    or not math.isfinite(receipt.raw_cost) or type(receipt.settled) is not bool
                    or (epoch < horizon - 1 and receipt.settled)):
                raise ValueError("invalid artificial receipt")
            _json_data(receipt.public)
            session["total_cost"] += receipt.raw_cost
            if not math.isfinite(session["total_cost"]):
                raise ValueError("nonfinite artificial accumulated cost")
            session["public"] = copy.deepcopy(receipt.public)
            session["history"].append(dict(epoch=epoch, action=copy.deepcopy(action),
                                           public=copy.deepcopy(receipt.public), raw_cost=receipt.raw_cost))
            session["filter"] = self._call("filter", {"estimator_receipt_updates": 1,
                "estimator_hypothesis_transitions": len(s["site_ids"]) * self._proposal["information"]["estimator"]["pairs_per_site_receipt"]},
                public=session["public"], history=session["history"], previous=session["filter"])
            _json_data(session["filter"])
            session["filter"] = copy.deepcopy(session["filter"])
            if epoch < control:
                pending = dict(receipt=epoch + 1, before=before, after=copy.deepcopy(receipt.public),
                               action=copy.deepcopy(action), raw_cost=receipt.raw_cost, done=False)
            else:
                pending["raw_cost"] += receipt.raw_cost
                pending["after"] = copy.deepcopy(receipt.public)
            session["pending_transition"] = pending
            if epoch == control - 1 or control <= epoch < horizon - 1:
                continue
            if epoch == horizon - 1:
                if not receipt.settled:
                    raise ValueError("unsettled artificial liability; no extension or full-cost claim")
                pending["done"] = True
            pending["available_at"] = epoch + 1
            pending["reward"] = -pending["raw_cost"] / 100000
            _json_data(pending)
            session["transitions"] += 1
            if world["phase"] != "evaluation" or role == "online_matched_fork":
                state = _state(self._call("remember", state=state, transition=pending))
                session["state"] = state
                if world["phase"] != "evaluation":
                    self._offline_rows += 1
                    self._training_state = state
                if role == "offline_learner" or (role == "online_matched_fork" and pending["receipt"] >= l["online_update_receipts_inclusive"][0]):
                    state = self._update(state, "ddpg", transition=pending)
            self._call("transition", transition=pending)
            session["state"] = state
            session["pending_transition"] = None
        self._call("episode_complete", transitions=session["transitions"], artificial_cost=session["total_cost"])
        self._completed.append(self._owner)
        self._session, self._owner, self._epoch = None, None, None
        return state

    def run(self):
        if self._started or self._restored or self._failure is not None:
            raise RuntimeError("no repeat, resume, replacement or refunded budget")
        self._guard()
        self._started, self._status = True, "running"
        try:
            d, l = self._proposal["design"], self._proposal["learner"]
            for block, seed in enumerate(d["training_seeds"]):
                self._block, self._phase = block, PHASES[0]
                state = _state(self._call("initialize", {"fresh_actor_critic_pairs": 1}, seed=seed))
                self._training_state, self._offline_rows = state, 0
                for world in self._worlds("teacher"):
                    state = self._episode(world, "id_mpc", state)
                for _ in range(l["bc_updates_per_block"]):
                    state = self._update(state, "bc")
                state = _state(self._call("targets", state=state, mode="after_bc_copy"))
                self._training_state = state
                for _ in range(l["critic_warmup_updates_per_block"]):
                    state = self._update(state, "warmup")
                self._phase = PHASES[1]
                for world in self._worlds("learner"):
                    state = self._episode(world, "offline_learner", state)
                if self._offline_rows != l["offline_replay_rows_per_seal"]:
                    raise ValueError("offline replay rows differ from the fixed seal")
                self._seals[str(block)] = dict(state=copy.deepcopy(state), sha256=digest(state),
                                              replay_rows=self._offline_rows)
                self._call("seal", seal=self._seals[str(block)])
            self._phase = PHASES[2]
            if set(self._seals) != set(map(str, range(d["blocks"]))):
                raise ValueError("all training blocks must seal before evaluation")
            self._call("seal_barrier", seals=self._seals)
            for block in range(d["blocks"]):
                self._block = block
                for world in self._worlds("evaluation"):
                    for role in CAPACITY_CONTROL_ROLES:
                        seal = self._seals[str(block)]
                        if digest(seal["state"]) != seal["sha256"]:
                            raise ValueError("artificial seal changed")
                        state = seal["state"] if role in CAPACITY_CONTROL_ROLES[:3] else None
                        self._episode(world, role, state)
            if self._counts != self._contract["limits"]:
                raise ValueError("incomplete artificial numeric packet")
            self._status = "complete"
        except BaseException as error:
            self._failure = dict(type=type(error).__name__, message=str(error),
                                 dispatches=self._dispatches, owner=self._owner, epoch=self._epoch)
            self._status = "failed"
            raise
        return self.state_dict()

    def state_dict(self):
        return copy.deepcopy(dict(format=FORMAT, artificial_only=True,
            proposal_sha256=digest(self._proposal), status=self._status, failure=self._failure,
            counts=self._counts, phase_counts=self._phase_counts, dispatches=self._dispatches,
            phase=self._phase, block=self._block, epoch=self._epoch, owner=self._owner,
            session=self._session, seals=self._seals, completed=self._completed,
            training_state=self._training_state, offline_rows=self._offline_rows))

    def load_state_dict(self, snapshot):
        """Restore for inspection only; reject stale snapshots without mutation."""
        _json_data(snapshot)
        current = self.state_dict()
        if (set(snapshot) != set(current) or snapshot["format"] != FORMAT
                or snapshot["artificial_only"] is not True
                or snapshot["proposal_sha256"] != current["proposal_sha256"]
                or snapshot["status"] not in ("new", "running", "failed", "complete")
                or set(snapshot["counts"]) != set(self._counts)
                or set(snapshot["phase_counts"]) != set(PHASES)
                or type(snapshot["dispatches"]) is not int or snapshot["dispatches"] < self._dispatches
                or (self._failure is not None and snapshot["failure"] != self._failure)
                or (snapshot["status"] == "failed") != (snapshot["failure"] is not None)):
            raise ValueError("invalid, stale or failure-clearing artificial snapshot")
        for key, limit in self._contract["limits"].items():
            value = snapshot["counts"][key]
            if type(value) is not int or not self._counts[key] <= value <= limit:
                raise ValueError("consumed counters cannot be refunded")
            for phase in PHASES:
                row = snapshot["phase_counts"][phase]
                cap = self._contract["phase_limits"][phase].get(key, limit)
                if (set(row) != set(self._counts) or type(row[key]) is not int
                        or not self._phase_counts[phase][key] <= row[key] <= cap):
                    raise ValueError("consumed phase counters cannot be refunded")
            if sum(snapshot["phase_counts"][phase][key] for phase in PHASES) != value:
                raise ValueError("phase/global counters differ")
        for seal in snapshot["seals"].values():
            if digest(_state(seal["state"])) != seal["sha256"]:
                raise ValueError("restored seal differs")
        if snapshot["completed"][:len(self._completed)] != self._completed:
            raise ValueError("completed prefix cannot be refunded")
        for name in ("counts", "phase_counts", "dispatches", "status", "failure", "phase", "block",
                     "epoch", "owner", "session", "seals", "completed", "training_state", "offline_rows"):
            setattr(self, "_" + name, copy.deepcopy(snapshot[name]))
        self._started = self._restored = True
