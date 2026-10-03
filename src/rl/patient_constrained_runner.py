"""Single reference-paired training comparison, with no deployment fitting."""

from dataclasses import asdict
import gzip
import json
import math
import os
import pickle

import numpy as np

from src.baselines.capacity_completion_control_recovery2 import CapacityCompletionControl
from src.env.patient_support_capacity import PatientSupportAction
from src.rl.capacity_native import keyed_seed, make_world_tape
from src.rl.capacity_pilot_runner import CapacityPilotRunner, COST_KEYS, jsonable, trajectory_id
from src.rl.capacity_public_features import resource_adjacency
from src.rl.patient_constrained_resources import PHASES, ARMS
from src.utils.research_clock import shared_monotonic


def worlds(proposal, phase, block):
    if phase not in ("training", "evaluation"):
        raise ValueError("unknown stream phase")
    base = 62700000 if phase == "training" else 62720000
    for index in range(12):
        c, j = index % 3, index // 3
        yield dict(phase=phase, block=block, condition=c, replicate=j,
                   seed=base+1000*block+100*c+j)


class PatientConstrainedRunner(CapacityPilotRunner):
    def __init__(self, root, proposal, budget, *, admission, learner_factory=None, **backends):
        if admission.get("verified") is not True or admission.get("scope") != "patient-constrained-capacity-v1":
            raise PermissionError("new exact patient-constrained admission required")
        if learner_factory is None:
            from src.rl.patient_constrained_learner import PatientConstrainedLearner
            learner_factory = PatientConstrainedLearner
        backends.setdefault("controller_factory", CapacityCompletionControl)
        # Reuse IO/public-input helpers; old scientific scheduling is never called.
        super().__init__(root, proposal, budget,
            admission=dict(verified=True, scope="dynamic-capacity-pilot-v1"),
            learner_factory=learner_factory, **backends)
        self.initial_seals, self.reference_losses, self.training_tapes = {}, {}, {}
        self.episode_context = None
        self.started = False

    def _optimizer(self, name, batch):
        if name not in ("actor", "critic", "reward_critic", "loss_critic"):
            raise ValueError("undeclared optimizer")
        component = "actor" if name == "actor" else "critic"
        self.budget.debit({f"{component}_optimizer_steps": 1, "total_optimizer_steps": 1,
                           "optimizer_example_presentations": batch})

    def _update_receipt(self, mode, receipt, duration):
        path = self.root/"updates"/f"block{self.active['block']}-{self.active['role']}.jsonl"
        with path.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(jsonable(dict(mode=mode, active=self.active, epoch=self.epoch,
                receipt=receipt, duration_seconds=duration, counts=self.learner.counts)), allow_nan=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _fit(self, mode, condition):
        self.budget.debit({"warmup_batches" if mode == "warmup" else "training_updates": 1})
        start = shared_monotonic()
        method = self.learner.warmup_step if mode == "warmup" else self.learner.training_step
        self._update_receipt(mode, method(condition), shared_monotonic()-start)

    def state_dict(self):
        return dict(environment=None if self.env is None else self.env.state_dict(),
            controller=None if self.controller is None else self.controller.state_dict(),
            learner=None if self.learner is None else self.learner.state_dict(), active=self.active,
            epoch=self.epoch, budget=self.budget.snapshot(), completed=list(self.completed),
            initial_seals={str(k): v.sha256 for k, v in self.initial_seals.items()},
            final_seals={f"{k[0]}:{k[1]}": v.sha256 for k, v in self.seals.items()},
            reference_losses={":".join(map(str,k)): v for k,v in self.reference_losses.items()},
            episode_context=self.episode_context, resume_authorized=False)

    def _save_state(self, name, **unused):
        self.budget.check(storage=True)
        with gzip.open(self.root/"states"/f"{name}.pkl.gz", "xb") as handle:
            pickle.dump(self.state_dict(), handle, protocol=5)
        self.budget.check(storage=True)

    def _seal(self, key, *, initial=False):
        seal = self.learner.snapshot()
        name = f"block{key}-initial" if initial else f"block{key[0]}-{key[1]}-final"
        self.budget.debit({"initial_seals" if initial else "final_seals": 1})
        self.budget.check(storage=True)
        with (self.root/"models"/f"{name}.pt").open("xb") as handle:
            handle.write(seal.payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._json(f"models/{name}.json", dict(sha256=seal.sha256, bytes=len(seal.payload),
            counts=self.learner.counts, role="initial" if initial else key[1],
            final_only=not initial, evaluation_weight_updates=False))
        (self.initial_seals if initial else self.seals)[key] = seal
        self.status("initial_sealed" if initial else "model_sealed", name=name, sha256=seal.sha256)
        return seal

    def _episode(self, world, role, tape, *, seal_reference=None):
        from src.rl.patient_constrained_learner import PatientLossTransition
        p, s = self.p, self.p["synthetic_system"]
        self.active, self.epoch = dict(world, role=role), 0
        ident = trajectory_id(world, role)
        self.budget.debit({"trajectories": 1})
        self.env = self.env_factory(p, tape, self._native)
        self.controller = self.controller_factory(p, base_control=self.base_control, scientific=True)
        public = self.capture(self.env)
        self._observe(public)
        states = self.features(public, self.controller.filter, p)
        noise = np.random.default_rng(keyed_seed(p, world, "exploration")).normal(0, .25, (48, 4))
        total, tail_total, tail_losses = 0., 0., 0
        lost_ids, rows, terminal = set(), [], None
        learned = role in ARMS or role.endswith("_greedy_frozen")
        training = role in ARMS
        self.episode_context = dict(world=world, role=role, noises=noise,
                                    transitions=rows, pending=None, last_public=public)
        self.status("trajectory_started")
        raw_path = f"raw/{ident}.jsonl.gz"
        with gzip.open(self.root/raw_path, "xt", encoding="utf8") as handle:
            for epoch in range(64):
                self.epoch, tail = epoch, epoch >= 48
                before = shared_monotonic()
                if tail:
                    hours = np.zeros(4)
                elif role == "id_mpc":
                    self.budget.debit({"planner_total_decisions": 1, "planner_candidate_rollouts": 48})
                    hours = self.controller.act(public, role=role, before_query=self._query, before_filter=self._filter)
                    self.budget.finish_chunk("planner_total_model_epochs")
                elif learned:
                    hours, _ = self.learner.act(states, explore=training, noise_hours=noise[epoch] if training else None)
                else:
                    hours = self.controller.act(public, role="fixed_allocation_reference", before_filter=self._filter)
                base = self.base_control(public, p, tail=tail)
                self.controller.record_operation(public, base)
                controller_seconds = shared_monotonic()-before
                before = shared_monotonic()
                _, reward, done, info = self.env.step(PatientSupportAction(tuple(map(float, base)), tuple(map(float, hours))))
                native_seconds = shared_monotonic()-before
                cost = float(info["cost"])
                components = {name: float(info[name]) for name in COST_KEYS}
                if (not math.isfinite(cost) or cost < 0 or not all(math.isfinite(v) for v in components.values())
                        or not math.isclose(math.fsum(components.values()), cost, rel_tol=1e-12, abs_tol=1e-6)
                        or not math.isclose(-float(reward), cost, rel_tol=1e-12, abs_tol=1e-6)
                        or bool(done) != (epoch == 63)):
                    raise ValueError("native cost/reward/terminal contract mismatch")
                next_public = self.capture(self.env)
                self.budget.debit({"estimator_receipt_updates": 1})
                before = shared_monotonic()
                self._observe(next_public)
                filter_seconds = shared_monotonic()-before
                next_states = self.features(next_public, self.controller.filter, p)
                records = next_public.operations.patients
                current_lost = {r.patient_id for r in records if r.status == "lost"}
                if not lost_ids.issubset(current_lost):
                    raise ValueError("lost patient reopened or missing")
                new_losses = len(current_lost-lost_ids)
                lost_ids = current_lost
                executed = info["support_public_receipt"]["committed_hours"]
                total += cost
                if tail:
                    tail_total += cost
                    tail_losses += new_losses
                else:
                    args = dict(state=states, executed_hours=executed, next_state=next_states,
                        world_id=ident, control_cost=cost, reward_divisor=100000.,
                        patient_losses=new_losses, condition=world["condition"])
                    if epoch == 47:
                        terminal = args
                    else:
                        rows.append(PatientLossTransition.from_cost(**args, done=False, settlement_cost=None, settlement_losses=None))
                raw = dict(epoch=epoch, world=world, role=role, cost=cost, reward=float(reward), components=components,
                    new_lost_patients=new_losses, patient_records=records,
                    cumulative=dict(enrolled=len(records), lost=len(lost_ids), delivered=sum(r.status == "delivered" for r in records)),
                    requested_hours=hours, executed_hours=executed, filter_resets=len(self.controller.filter.reset_events),
                    support_backlog=sum(not r.support_complete and r.status not in ("lost", "delivered") for r in records),
                    service=next_public.operations.last_service, public_input=public,
                    filter_summary=self.controller.filter.node_summaries(), info=info,
                    timing_seconds=dict(controller=controller_seconds, native=native_seconds, filter=filter_seconds))
                handle.write(json.dumps(jsonable(raw), sort_keys=True, allow_nan=False)+"\n")
                public, states = next_public, next_states
                self.episode_context.update(pending=terminal, total_cost=total, tail_cost=tail_total,
                    tail_losses=tail_losses, lost_ids=sorted(lost_ids), last_public=public, last_features=states)
                if (epoch+1) % 8 == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    self.budget.check(storage=True)
                    self.status("epoch_boundary")
        terminal["next_state"] = states
        rows.append(PatientLossTransition.from_cost(**terminal, done=True, settlement_cost=tail_total, settlement_losses=tail_losses))
        if len(rows) != 48 or sum(row.total_losses for row in rows) != len(lost_ids):
            raise ValueError("full cost/loss transition count mismatch")
        settlement = self.env.settlement()
        summary = dict(world=world, role=role, raw_path=raw_path, cost=total, settled=settlement["settled"],
            settlement=settlement, lost=len(lost_ids), tape_sha256=tape.digest(), change_epoch=tape.change_epoch,
            model_seal_sha256=seal_reference, optimizer_updates_during_trajectory=0)
        self._json(f"summaries/{ident}.json", summary)
        self.epoch = 64
        if not settlement["settled"] or settlement["lost"] != len(lost_ids):
            self._save_state(ident+"-unsettled")
            raise ValueError("unsettled cohort; no extension")
        self.completed.append(ident)
        self._save_state(ident)
        self.status("trajectory_completed", summary=summary)
        self.episode_context = None
        return rows, summary

    def run(self):
        if self.started or self.finished:
            raise RuntimeError("single serial attempt; no repeat or resume")
        self.started = True
        for block, seed in enumerate(self.p["design"]["training_seeds"]):
            self.budget.enter(PHASES[0], PHASES[0])
            self.budget.job(f"reference-b{block}", "reference_block")
            self.learner = None
            reference = []
            for world in worlds(self.p, "training", block):
                key = (block, world["condition"], world["replicate"])
                tape = self._tape(world)
                self.training_tapes[key] = tape
                rows, summary = self._episode(world, "fixed_allocation_reference", tape)
                reference.extend(rows)
                self.reference_losses[key] = summary["lost"]
            self.budget.debit({"fresh_initializers": 1})
            self.learner = self.learner_factory.from_proposal(self.p, node_input_dim=len(reference[0].state[0]),
                model_seed=seed, replay_seed=keyed_seed(self.p, world, "replay"), adjacency=resource_adjacency(self.p),
                before_forward=self._forward, before_optimizer=self._optimizer)
            for row in reference:
                self.learner.add_reference(row)
            self.active = dict(world, role="warmup")
            for update in range(256):
                self._fit("warmup", update % 3)
                if (update+1) % 64 == 0:
                    self.status("warmup_boundary", update=update+1)
            initial = self._seal(block, initial=True)
            self.budget.enter(PHASES[1], PHASES[1])
            for arm in ARMS:
                self.budget.job(f"training-b{block}-{arm}", "learned_arm_block")
                self.budget.debit({"training_fork_restorations": 1})
                self.learner = self.learner_factory.from_snapshot(initial,
                    before_forward=self._forward, before_optimizer=self._optimizer)
                self.learner.set_arm(arm)
                for world in worlds(self.p, "training", block):
                    key = (block, world["condition"], world["replicate"])
                    rows, summary = self._episode(world, arm, self.training_tapes[key], seal_reference=initial.sha256)
                    for row in rows:
                        self.learner.add_training(row)
                    if arm == "constrained":
                        self.budget.debit({"scalar_multiplier_updates": 1})
                        receipt = self.learner.update_multiplier(world["condition"], summary["lost"], self.reference_losses[key])
                        self._update_receipt("multiplier", receipt, 0.)
                    for _ in range(48):
                        self._fit("training", world["condition"])
                    self._save_state(trajectory_id(world, arm)+"-after-fit")
                    self.status("training_world_fit_completed", arm=arm)
                self._seal((block, arm))
        if len(self.seals) != 6:
            raise RuntimeError("all six final models required before test stream")
        self.status("all_models_sealed")
        self.budget.enter(PHASES[2], PHASES[2])
        for block in range(3):
            for world in worlds(self.p, "evaluation", block):
                tape = self._tape(world)
                for role in self.p["patient_constrained"]["design"]["evaluation_arms"]:
                    self.budget.job(f"eval-b{block}-c{world['condition']}-{role}", "evaluation_controller_block_condition")
                    self.learner, seal = None, None
                    if role.endswith("_greedy_frozen"):
                        arm = role.removesuffix("_greedy_frozen")
                        seal = self.seals[(block, arm)]
                        self.budget.debit({"learned_evaluation_arm_restorations": 1})
                        self.learner = self.learner_factory.from_snapshot(seal,
                            before_forward=self._forward, before_optimizer=self._optimizer)
                    self._episode(world, role, tape, seal_reference=None if seal is None else seal.sha256)
        if self.budget.counts != self.budget.limits:
            raise RuntimeError("final numerical counters differ: "+repr({k:(self.budget.counts[k],v)
                for k,v in self.budget.limits.items() if self.budget.counts[k]!=v}))
        self.finished = True
        self.status("all_trajectories_complete")
        return dict(trajectories=len(self.completed), final_models={f"{k[0]}:{k[1]}":v.sha256 for k,v in self.seals.items()},
                    initial_models={str(k):v.sha256 for k,v in self.initial_seals.items()}, budget=self.budget.snapshot())
