"""Additive recovery of the single block0/flat epoch37 comparison boundary.

The coordinator copies completed raw/summaries/updates/states/models/tapes and
the original progress.jsonl into root BEFORE construction. The incomplete raw
is an explicit read-only partial_raw_path, outside root/raw (or in the old root).
Its canonical path in root/raw must be absent: the new complete gzip is a
distinct file whose decompressed prefix is byte-for-byte the saved 37 rows.

Pass the ORIGINAL merged proposal unchanged, a fresh remaining-only budget,
and the already deserialized, input-bound failure_state. No locks, copying,
approval decisions, predictor changes, or scientific execution live here.
Budget counts/limits are new work only; progress counts remain cumulative for
the original analyzer, with new_counts and current_old_counts explicit. Copy
history before appending and the original analyzer needs no changes.
"""

import copy
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from src.baselines.capacity_value_mpc_recovery1 import CapacityValueMPCRecovery1
from src.env import capacity_planning as host_module
from src.env.aging_inventory import AgingInventory
from src.env.patient_condition import PatientConditionModel
from src.env.patient_support_capacity import PatientSupportAction, validate_support_host_contract
from src.env.patient_support_work import _canonical
from src.rl.capacity_native import CapacityPilotEnv, CapacityWorldTape, common_operation, make_world_tape, native_configs
from src.rl.capacity_pilot_runner import COST_KEYS, jsonable, trajectory_id
from src.rl.capacity_saved_teacher_replay import decode_public
from src.rl.capacity_value_comparison_learner import CapacityValueComparisonLearner
from src.rl.capacity_value_comparison_resources import ARCHITECTURES, PHASES, SCOPE, worlds
from src.rl.capacity_value_comparison_runner import CapacityValueComparisonRunner, parameter_count
from src.rl.capacity_value_features import FEATURE_NAMES, observed_features
from src.rl.public_support_input import PublicSupportControlInput


RECOVERY_SCOPE = "capacity-value-comparison-recovery1"


class CapacityValueRecoveryEnv(CapacityPilotEnv):
    """Native mechanics unchanged; initialize contracts without construction/reset."""

    @classmethod
    def from_state_dict(cls, proposal, tape, state, before_native):
        out = cls.__new__(cls)
        out._proposal, out._tape = copy.deepcopy(proposal), tape
        out._before_native, out._reset_consumed = before_native, True
        out._tape_digest, out._failed = tape.digest(), False
        out.env_config, out.support_config = native_configs(proposal)
        out.config = c = out.env_config.base
        out._response_tape = validate_support_host_contract(out.env_config, out.support_config, tape.responses)
        contracts = dict(format=out.support_format, host_contract=_canonical(asdict(out.env_config)),
                         support_contract=_canonical(asdict(out.support_config)), response_tape=out._response_tape,
                         world_tape_sha256=out._tape_digest)
        if any(_canonical(state[k]) != _canonical(v) for k, v in contracts.items()):
            raise ValueError("saved native contract/tape differs from original proposal")
        out._validate_config()
        out._validate_patient_routing_config()
        n = c.num_facilities
        out.rng = np.random.default_rng(0)  # Replaced by the exact saved host RNG below.
        for name in ("demand_rates", "demand_regime_initial_multipliers", "demand_regime_final_multipliers",
                     "initial_specimens", "initial_reagents", "initial_idle_bioreactors", "max_specimens",
                     "max_reagents", "max_idle_bioreactors", "max_reagent_replenishment", "supplier_disruption_rate"):
            setattr(out, name, host_module._as_vector(getattr(c, name), n, name))
        out.base_demand_rates = out.demand_rates.copy()
        out.base_supplier_disruption_rate = out.supplier_disruption_rate.copy()
        out.demand_rate_estimates = (out.base_demand_rates.copy() if c.demand_rate_estimates is None
                                    else host_module._as_vector(c.demand_rate_estimates, n, "demand_rate_estimates"))
        out.clinic_coordinates = host_module.normalize_coordinates(c.clinic_coordinates, n)
        out.clinic_distance_matrix = (np.asarray(host_module.geographic_distance_matrix(out.clinic_coordinates))
                                      if out.clinic_coordinates else None)
        out.clinic_transfer_time_hours_matrix = (np.asarray(host_module.geographic_transfer_time_matrix(
            out.clinic_coordinates, speed_mph=c.geographic_transfer_speed_mph,
            fixed_handling_hours=c.geographic_transfer_fixed_hours)) if out.clinic_coordinates else None)
        out.transfer_delay_thresholds = tuple(float(x) for x in (c.transfer_lead_time_distance_thresholds or ()))
        out._train_disruption_range = out._train_forecast_error_range = out._train_demand_rate_multiplier_range = None
        # native_configs always supplies the frozen explicit graph and facility-net action space.
        for name in ("specimen_edges", "capacity_edges", "resource_edges", "information_edges"):
            setattr(out, name, host_module._normalize_edges(getattr(c, name), (), n))
        for name, edges in (("specimen", out.specimen_edges), ("capacity", out.capacity_edges),
                            ("reagent", out.resource_edges)):
            setattr(out, name+"_transfer_priorities", out._facility_net_transfer_priorities(edges))
        out.hub_index = n if c.include_central_capacity_hub else None
        out.features_per_facility = (3 + c.production_lead_time + int(c.include_supplier_state)
            + int(c.include_demand_forecast_state) + 3*int(c.include_transfer_pipeline_state)
            + 3*int(c.include_demand_history_state)
            + 3*int(c.include_demand_sequence_state)*int(c.demand_sequence_length)
            + int(c.include_on_order_state)*(out._procurement_pipeline_depth()+1))
        out._construction_seed = int(tape.world["seed"])
        out.patient_model = PatientConditionModel(out.env_config.patient)
        out._summary_edges = np.asarray(out.env_config.survival_bucket_edges, dtype=float)
        out.routing_summary_width = 4 if out.env_config.include_specimen_routing_state else 0
        out.summary_width = 6 + len(out._summary_edges) + 1 + out.routing_summary_width
        out.base_observation_size = n*out.features_per_facility
        out.observation_size = out.base_observation_size + n*out.summary_width + int(c.include_time_state)
        out.action_size = 4*n
        for name in ("overtime_surge_headroom", "previous_overtime_fraction", "overtime_outstanding",
                     "overtime_fatigue", "overtime_active_capacity"):
            setattr(out, name, np.zeros(n))
        out.overtime_commitment_pipeline = np.zeros((0, n))
        out._pending_support_request = out._support_receipt = None
        viability = out._viability_fn if out.env_config.enable_viability_hook else None
        out.finished_product = [AgingInventory(out.env_config.finished_shelf_life, viability) for _ in range(n)]
        out._initial_reactor_total = float(sum(proposal["synthetic_system"]["initial_idle_bioreactors"]))
        out._restore(copy.deepcopy(state))
        return out


class CapacityValueComparisonRecovery1Runner(CapacityValueComparisonRunner):
    def __init__(self, root, proposal, budget, *, admission, failure_state, partial_raw_path,
                 env_factory=CapacityValueRecoveryEnv, controller_factory=CapacityValueMPCRecovery1,
                 value_factory=CapacityValueComparisonLearner, tape_factory=make_world_tape,
                 capture=PublicSupportControlInput.capture, feature_adapter=observed_features,
                 parameter_counter=parameter_count, base_control=common_operation):
        if admission != dict(verified=True, scope=SCOPE, remaining_scope=RECOVERY_SCOPE):
            raise PermissionError("verified remaining-comparison admission required")
        self.root, self.p, self.budget = Path(root), proposal, budget
        self.study = proposal["value_comparison_study"]
        self.failure_state = copy.deepcopy(failure_state)
        self.partial_raw_path = Path(partial_raw_path)
        self._validate_boundary()
        if any(budget.counts.values()):
            raise ValueError("remaining-work budget must start at zero")
        self.current_old_counts = copy.deepcopy(failure_state["budget"]["counts"])
        self.env_factory, self.recovery_controller_factory = env_factory, controller_factory
        self.value_factory, self.tape_factory = value_factory, tape_factory
        self.capture, self.feature_adapter = capture, feature_adapter
        self.parameter_counter, self.base_control = parameter_counter, base_control
        self.env = self.controller = self.learner = None
        self.epoch, self.active, self.finished, self.started = 0, None, False, False
        self._score_value = False
        self.initial, self.final, self.episode_data = {}, {}, None
        self.training_learners, self.parameter_counts = {}, {}
        self.completed = list(failure_state["completed"])

        def select_controller(*args, value=None, **kwargs):
            return controller_factory(*args, value=self.learner if self._score_value else None, **kwargs)

        self.controller_factory = select_controller
        for directory in ("raw", "summaries", "models", "states", "tapes", "updates"):
            (self.root/directory).mkdir(exist_ok=True)
        if not (self.root/"progress.jsonl").is_file():
            raise ValueError("coordinator must copy historical progress.jsonl first")

    def _validate_boundary(self):
        s = self.failure_state
        w = list(worlds(self.study, "continuation", 0))[-1]
        expected = [trajectory_id(x, "plain_mpc") for x in worlds(self.study, "initial", 0)]
        expected += [trajectory_id(x, a+"_value_mpc") for x in worlds(self.study, "continuation", 0)
                     for a in ARCHITECTURES][:-1]
        if (s["format"] != "capacity-value-comparison-runner-v1" or s["active"] != dict(w, role="flat_value_mpc")
                or s["epoch"] != 37 or s["completed"] != expected or s["final"]
                or set(s["initial"]) != {"block0-graph", "block0-flat"}
                or set(s["training_learners"]) != set(ARCHITECTURES)
                or {k:len(v) for k,v in s["episode_data"].items()} != dict(features=38, heuristics=38, costs=37)):
            raise ValueError("only the saved block0/flat epoch37 boundary is supported")
        for a, updates in (("graph", 1536), ("flat", 1504)):
            learner = s["training_learners"][a]
            if (learner["updates"] != updates or learner["pending"] is not None
                    or learner["config"] != dict(self.study["value"], architecture=a)
                    or learner["feature_dim"] != len(FEATURE_NAMES)):
                raise ValueError("saved training learner boundary/config mismatch")

    def status(self, event, **values):
        # The old analyzer checks cumulative optimizer counts at test opening.
        cumulative = {k:v+self.current_old_counts.get(k, 0) for k,v in self.budget.counts.items()}
        record = dict(event=event, active=self.active, epoch=self.epoch, counts=cumulative,
                      new_counts=self.budget.counts, current_old_counts=self.current_old_counts, **values)
        with (self.root/"progress.jsonl").open("a", encoding="utf8") as handle:
            handle.write(json.dumps(jsonable(record), allow_nan=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def state_dict(self):
        state = super().state_dict()
        state.update(format="capacity-value-comparison-recovery1-runner-v1",
                     current_old_counts=self.current_old_counts, partial_raw_path=str(self.partial_raw_path))
        return state

    def _restore_learners(self):
        for a in ARCHITECTURES:
            s = copy.deepcopy(self.failure_state["training_learners"][a])
            learner = self.value_factory(s["config"], seed=0, feature_dim=s["feature_dim"],
                                        before_forward=self._forward, before_optimizer=self._optimizer)
            learner.load_state_dict(s)
            self.training_learners[a] = learner
            name = f"block0-{a}"
            raw = (self.root/"models"/(name+"-initial.pt")).read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            if sha != self.failure_state["initial"][name]:
                raise ValueError("copied initial seal differs from failure state")
            self.initial[name] = raw, sha
        self.parameter_counts = copy.deepcopy(self.failure_state["parameter_counts"])
        self.learner = self.training_learners["flat"]

    def _resume_partial(self):
        s = self.failure_state
        self.active, self.epoch, self._score_value = copy.deepcopy(s["active"]), 37, True
        w = {k:v for k,v in self.active.items() if k != "role"}
        role, ident = self.active["role"], trajectory_id(w, self.active["role"])
        raw_path = "raw/"+ident+".jsonl.gz"
        if (self.root/raw_path).exists() or (self.root/raw_path).is_symlink():
            raise FileExistsError("completed partial output must be distinct; keep copied prefix outside raw/")
        prefix = gzip.decompress(self.partial_raw_path.read_bytes())
        rows = [json.loads(line) for line in prefix.splitlines()]
        data = copy.deepcopy(s["episode_data"])
        if (len(rows) != 37 or not prefix.endswith(b"\n")
                or any(r["epoch"] != i or r["world"] != w or r["role"] != role for i,r in enumerate(rows))):
            raise ValueError("partial raw must contain exactly the original epochs 0..36")
        saved_data = dict(features=[r["value_features"] for r in rows]+[rows[-1]["next_value_features"]],
                          heuristics=[r["value_base"] for r in rows]+[rows[-1]["next_value_base"]],
                          costs=[r["cost"] for r in rows])
        if jsonable(data) != saved_data:
            raise ValueError("partial raw and saved training data differ")
        tape_data = json.loads((self.root/"tapes"/(trajectory_id(w, "exogenous")+".json")).read_text())
        tape = CapacityWorldTape(**tape_data)
        if tape.world != w:
            raise ValueError("partial tape identity mismatch")
        self.budget.job(ident+"-remaining", "world")
        self.env = self.env_factory.from_state_dict(self.p, tape, s["environment"], self._native)
        public = self.capture(self.env)
        if self.env.t != 37 or public.common.epoch != 37:
            raise ValueError("restored environment must be at epoch37")
        self.controller = self.recovery_controller_factory(self.p, base_control=self.base_control,
                                                           scientific=True, value=self.learner)
        self.controller.load_state_dict(copy.deepcopy(s["controller"]))
        self.controller.value = self.training_learners["flat"]
        saved_view = decode_public(s["controller"]["filter"]["last_view"])
        if jsonable(public) != jsonable(saved_view):
            raise ValueError("environment and saved controller public boundary differ")
        # Receipt 37 and operation 36 are already recorded. Never observe/replay here.
        self.episode_data = data
        losses = {p["patient_id"] for p in rows[-1]["patient_records"] if p["status"] == "lost"}
        total = sum(data["costs"])
        self.status("partial_trajectory_restored", prefix_rows=37, remaining_steps=27,
                    prefix_sha256=hashlib.sha256(prefix).hexdigest(), source_raw=str(self.partial_raw_path))
        with gzip.open(self.root/raw_path, "xb") as handle:
            handle.write(prefix)
            for epoch in range(37, 64):
                self.epoch = epoch
                if epoch >= 48:
                    hours = np.zeros(4)
                else:
                    self.budget.debit(dict(planner_total_decisions=1, planner_candidate_rollouts=48))
                    hours = self.controller.act(public, role="id_mpc", before_query=self._query, before_filter=self._filter)
                    self.budget.finish_chunk("planner_total_model_epochs")
                base = self.base_control(public, self.p, tail=epoch >= 48)
                self.controller.record_operation(public, base)
                _, reward, done, info = self.env.step(PatientSupportAction(tuple(map(float, base)), tuple(map(float, hours))))
                cost = float(info["cost"])
                components = {k:float(info[k]) for k in COST_KEYS}
                if (not math.isfinite(cost) or cost < 0 or not all(math.isfinite(v) for v in components.values())
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
                next_losses = {p.patient_id for p in records if p.status == "lost"}
                if not losses.issubset(next_losses):
                    raise ValueError("lost patient reopened")
                raw = dict(epoch=epoch, world=w, role=role, cost=cost, reward=float(reward), components=components,
                    new_lost_patients=len(next_losses-losses), patient_records=records,
                    cumulative=dict(enrolled=len(records), lost=len(next_losses), delivered=sum(p.status=="delivered" for p in records)),
                    requested_hours=hours, executed_hours=info["support_public_receipt"]["committed_hours"],
                    public_input=public, value_features=data["features"][-2], value_base=data["heuristics"][-2],
                    next_value_features=x, next_value_base=h, plan=self.controller.last_plan if epoch < 48 else None, info=info)
                handle.write((json.dumps(jsonable(raw), sort_keys=True, allow_nan=False)+"\n").encode())
                losses, public, total = next_losses, following, total+cost
                if (epoch+1) % 8 == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    self.budget.check(storage=True)
                    self.status("epoch_boundary")
        settlement = self.env.settlement()
        summary = dict(world=w, role=role, raw_path=raw_path, cost=total, settled=settlement["settled"],
            settlement=settlement, lost=len(losses), tape_sha256=tape.digest(), change_epoch=tape.change_epoch,
            model_seal_sha256=None, initial_ancestor_sha256=self.initial["block0-flat"][1],
            value_updates_before_trajectory=self.learner.updates, optimizer_updates_during_trajectory=0)
        self._json("summaries/"+ident+".json", summary)
        if not settlement["settled"] or settlement["lost"] != len(losses):
            raise ValueError("unsettled cohort; no horizon extension")
        self.epoch = 64
        self.completed.append(ident)
        self._save_state(ident)
        self.status("trajectory_completed", summary=summary)
        self.episode_data = None
        self._fit_data(w, "flat", data, role)

    def _fresh_block(self, block):
        self.budget.enter(PHASES[0], PHASES[0])
        self.training_learners = {}
        for a in ARCHITECTURES:
            learner = self.value_factory(dict(copy.deepcopy(self.study["value"]), architecture=a),
                seed=self.study["design"]["model_seeds"][a][block], feature_dim=len(FEATURE_NAMES),
                before_forward=self._forward, before_optimizer=self._optimizer)
            self.training_learners[a] = learner
            self.parameter_counts[a] = self.parameter_counter(learner)
        gap = abs(self.parameter_counts["graph"]-self.parameter_counts["flat"])/self.parameter_counts["graph"]
        if gap > self.study["value"]["max_parameter_gap_fraction"]:
            raise ValueError("graph/flat parameter gap exceeds frozen bound")
        self._json(f"models/block{block}-architecture.json", dict(parameters=self.parameter_counts,
            gap_fraction=gap, public_features=list(FEATURE_NAMES), initial_data_shared=True))
        for w in worlds(self.study, "initial", block):
            self.learner = None
            data = self._episode(w, "plain_mpc", self._tape(w))
            for a in ARCHITECTURES:
                self.learner = self.training_learners[a]
                self._fit_data(w, a, data, "plain_mpc")
        for a in ARCHITECTURES:
            self.learner = self.training_learners[a]
            self._seal(block, a)
        self.budget.enter(PHASES[1], PHASES[1])
        for a in ARCHITECTURES:
            self.training_learners[a] = self.value_factory.from_bytes(self.initial[f"block{block}-{a}"][0],
                before_forward=self._forward, before_optimizer=self._optimizer)
        for w in worlds(self.study, "continuation", block):
            tape = self._tape(w)
            for a in ARCHITECTURES:
                self.learner = self.training_learners[a]
                role = a+"_value_mpc"
                data = self._episode(w, role, tape, seal=self.initial[f"block{block}-{a}"][1])
                self._fit_data(w, a, data, role)
        self._finalize_block(block)

    def _finalize_block(self, block):
        for a in ARCHITECTURES:
            self.learner = self.training_learners[a]
            self._seal(block, a, final=True)
        self.training_learners = {}

    def run(self):
        if self.started:
            raise RuntimeError("single attempt; no resume/relaunch")
        self.started = True
        self.budget.enter(PHASES[1], PHASES[1])
        self._restore_learners()
        self._resume_partial()
        self._finalize_block(0)
        for block in range(1, 5):
            self._fresh_block(block)
        if len(self.final) != 10 or len(self.initial) != 10:
            raise RuntimeError("all initial/final models required before tests")
        self.status("all_models_sealed")
        self.budget.enter(PHASES[2], PHASES[2])
        for block in range(5):
            for w in worlds(self.study, "evaluation", block):
                tape = self._tape(w)
                for role in self.study["design"]["roles"]:
                    value = self.final.get(f"block{block}-{role.split('_')[0]}")
                    self.learner = None if value is None else self.value_factory.from_bytes(value[0],
                        before_forward=self._forward, before_optimizer=self._optimizer)
                    self._episode(w, role, tape, seal=None if value is None else value[1])
        if self.budget.counts != self.budget.limits or len(set(self.completed)) != 600 or len(self.completed) != 600:
            raise RuntimeError("remaining comparison completion counts mismatch")
        self.finished = True
        self.status("all_trajectories_complete")
        return dict(trajectories=len(self.completed), new_completed_trajectories=529,
                    initial_models={k:v[1] for k,v in self.initial.items()},
                    final_models={k:v[1] for k,v in self.final.items()}, parameter_counts=self.parameter_counts,
                    current_old_counts=self.current_old_counts, budget=self.budget.snapshot())
