"""Remaining-only runner; imports saved work and never restarts block0."""

from dataclasses import asdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil

import numpy as np

from src.baselines.capacity_completion_control_recovery2 import CapacityCompletionControl
from src.env.patient_support_capacity import PatientSupportAction
from src.rl.capacity_adaptation_campaign import PHASES
from src.rl.capacity_ddpg_learner import CapacitySnapshot
from src.rl.capacity_native import CapacityWorldTape, keyed_seed
from src.rl.capacity_pilot_runner import CapacityPilotRunner, COST_KEYS, jsonable, trajectory_id, worlds
from src.rl.capacity_public_features import resource_adjacency
from src.rl.capacity_saved_teacher_replay import decode_public, reconstruct_teacher
from src.utils.research_clock import shared_monotonic


class CapacityRecovery2Runner(CapacityPilotRunner):
    def __init__(self, root, proposal, budget, *, prior_root, admission,
                 input_files, snapshot_factory=CapacitySnapshot, **backends):
        if admission.get("remaining_scope") != "dynamic-capacity-recovery2":
            raise PermissionError("separate remaining-work admission required")
        backends.setdefault("controller_factory", CapacityCompletionControl)
        super().__init__(root, proposal, budget, admission=admission, **backends)
        self.prior = Path(prior_root)
        self.input_files = dict(input_files)
        self.snapshot_factory = snapshot_factory

    def _prior_bytes(self, relative):
        path = self.prior / relative
        if relative not in self.input_files or path.is_symlink():
            raise PermissionError("unbound prior input")
        raw = path.read_bytes()
        record = self.input_files[relative]
        if hashlib.sha256(raw).hexdigest() != record["sha256"] or len(raw) != record["bytes"]:
            raise ValueError("immutable prior input changed: "+relative)
        self.budget.check(storage=True)
        return raw

    def _prior_state(self, relative):
        return pickle.loads(gzip.decompress(self._prior_bytes(relative)))

    def _rows(self, identifier):
        raw = self._prior_bytes(f"raw/{identifier}.jsonl.gz")
        return [json.loads(row) for row in gzip.decompress(raw).splitlines()]

    def _copy_prior(self, relative):
        raw = self._prior_bytes(relative)
        target = self.root / relative
        with target.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        self.budget.check(storage=True)

    def _reconstruct(self, rows, state):
        def before_receipt():
            self.budget.debit({"saved_receipt_reconstruction_updates": 1})
        def before_filter(payload):
            self.budget.chunk_call("saved_reconstruction_hypothesis_transitions", 100,
                                   payload["hypothesis_transitions"])
        def after_receipt():
            self.budget.finish_chunk("saved_reconstruction_hypothesis_transitions")
        return reconstruct_teacher(rows, state["controller"], self.p,
            before_receipt=before_receipt, before_filter=before_filter, after_receipt=after_receipt)

    def _import_completed(self):
        # Copy only raw evidence needed by the common reader, not the old archive.
        for name in sorted(self.input_files):
            if name.startswith(("raw/", "summaries/", "tapes/")) or name in (
                    "models/block0-seal.pt", "models/block0-seal.json"):
                self._copy_prior(name)
        self.completed = [Path(name).stem for name in sorted(self.input_files) if name.startswith("summaries/")]
        if len(self.completed) != 35:
            raise ValueError("expected35immutable completions")
        self.budget.debit({"imported_seal_artifacts": 1})
        payload = self._prior_bytes("models/block0-seal.pt")
        self.seals[0] = self.snapshot_factory(payload, hashlib.sha256(payload).hexdigest())
        self._json("reuse-provenance.json", dict(prior_root=str(self.prior), imported_completed=self.completed,
            imported_model_sha256=self.seals[0].sha256, prior_inputs=self.input_files,
            old_debits_refunded=False, homogeneous_repaired_initializer=False))
        self.status("saved_model_and_35_trajectories_imported")

    def _resume_teacher(self, world, tape, state, rows, teacher_rows):
        p, s = self.p, self.p["synthetic_system"]
        seal_reference = None
        self.active, self.epoch = dict(world, role="teacher"), 47
        role = "teacher"
        identifier = trajectory_id(world, role)
        self.budget.debit({"partial_restorations": 1})
        self.env = self.env_factory(p, tape, self._native)
        self.env.load_state_dict(state["environment"])
        self.controller = self.controller_factory(p, base_control=self.base_control, scientific=True)
        self.controller.load_recovery1_boundary(state["controller"])
        public = self.capture(self.env)
        if jsonable(public) != jsonable(decode_public(state["controller"]["filter"]["last_view"])):
            raise ValueError("restored public boundary differs; no recollection fallback")
        states = self.features(public, self.controller.filter, p)
        if self.env.t != 47 or len(rows) != 47 or len(teacher_rows) != 47:
            raise ValueError("exact epoch47 partial boundary required")
        # No receipt re-observation here: its filter state is already saved.
        total, tail_total, terminal = sum(float(r["cost"]) for r in rows), 0., None
        raw_path = f"raw/{identifier}.jsonl.gz"
        self.status("trajectory_started")
        with gzip.open(self.root / raw_path, "at", encoding="utf8") as handle:
            for epoch in range(47, s["host_total_horizon"]):
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
                    raise RuntimeError("partial continuation is teacher-only")
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


    def run(self):
        if self.finished or self.active is not None:
            raise RuntimeError("one serial invocation only")
        p = self.p
        self.budget.enter("initialization_training_and_seals", PHASES[0])
        self._import_completed()
        for block in (1, 2):
            model_seed = p["design"]["training_seeds"][block]
            self.budget.enter("initialization_training_and_seals", PHASES[0])
            teacher_rows = []
            self.learner = None
            for world in worlds(p, PHASES[0], block):
                identifier = trajectory_id(world, "teacher")
                if block == 1:
                    rows = self._rows(identifier)
                    if identifier in self.completed:
                        state = self._prior_state(f"states/{identifier}.pkl.gz")
                        teacher_rows.extend(self._reconstruct(rows, state))
                    else:
                        state = self._prior_state("failure-state.pkl.gz")
                        if state["active"] != dict(world, role="teacher") or state["epoch"] != 47 or state["learner"] is not None:
                            raise ValueError("unexpected saved failure boundary")
                        saved_rows = self._reconstruct(rows, state)
                        data = json.loads(self._prior_bytes(f"tapes/{trajectory_id(world, 'exogenous')}.json"))
                        tape = CapacityWorldTape(**data)
                        if tape.digest() != state["environment"]["world_tape_sha256"]:
                            raise ValueError("partial restore tape mismatch")
                        teacher_rows.extend(self._resume_teacher(world, tape, state, rows, saved_rows))
                else:
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
        if len(self.completed) != 288:
            raise ValueError("imported plus new completions must equal288")
        self.finished = True
        self.status("all_trajectories_complete")
        return dict(trajectories=len(self.completed), model_seals={k: v.sha256 for k,v in self.seals.items()}, budget=self.budget.snapshot())

