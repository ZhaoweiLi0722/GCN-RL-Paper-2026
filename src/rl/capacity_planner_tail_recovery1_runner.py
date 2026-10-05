"""Resume the final H16 world, then finish the original saved-root diagnostics.

Admission/copying is owned by the execution layer. No world construction,
training, changed predictor, or replay of the saved ten native steps is allowed.
"""

import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from src.baselines.capacity_planner_tail_mpc import CapacityPlannerTailMPC
from src.env.patient_support_capacity import PatientSupportAction
from src.rl.capacity_native import CapacityWorldTape, common_operation
from src.rl.capacity_pilot_runner import COST_KEYS, jsonable, trajectory_id
from src.rl.capacity_planner_tail_design import EVAL_ROLES, TRAIN_ROLES, worlds
from src.rl.capacity_planner_tail_learner import CapacityPlannerTailLearner
from src.rl.capacity_planner_tail_resources import SCOPE
from src.rl.capacity_planner_tail_runner import CapacityPlannerTailRunner
from src.rl.capacity_saved_teacher_replay import decode_public
from src.rl.capacity_value_comparison_learner import CapacityValueComparisonLearner
from src.rl.capacity_value_comparison_recovery1_runner import CapacityValueRecoveryEnv
from src.rl.capacity_value_features import observed_features
from src.rl.public_support_input import PublicSupportControlInput
from src.utils.research_clock import shared_monotonic


RECOVERY_SCOPE = "capacity-planner-tail-recovery1"
PARTIAL = "evaluation-b4-c2-j3-plain_h16"


class CapacityPlannerTailRecovery1Runner(CapacityPlannerTailRunner):
    def __init__(self, root, proposal, budget, *, admission, failure_state,
                 partial_raw_path, env_factory=CapacityValueRecoveryEnv,
                 controller_factory=CapacityPlannerTailMPC,
                 value_factory=CapacityPlannerTailLearner,
                 frozen_factory=CapacityValueComparisonLearner,
                 capture=PublicSupportControlInput.capture,
                 feature_adapter=observed_features, base_control=common_operation,
                 public_decoder=decode_public):
        if admission != dict(verified=True, scope=SCOPE, remaining_scope=RECOVERY_SCOPE):
            raise PermissionError("verified final-world recovery admission required")
        self.root, self.p, self.budget = Path(root), proposal, budget
        self.study = proposal["planner_tail_study"]
        self.failure_state = copy.deepcopy(failure_state)
        self.partial_raw_path = Path(partial_raw_path)
        self._validate_boundary()
        if any(budget.counts.values()):
            raise ValueError("remaining budget must start at zero")
        self.current_old_counts = copy.deepcopy(failure_state["budget"]["counts"])
        self.env_factory, self.controller_factory = env_factory, controller_factory
        self.value_factory, self.frozen_factory = value_factory, frozen_factory
        self.capture, self.feature_adapter = capture, feature_adapter
        self.base_control, self.public_decoder = base_control, public_decoder
        self.env = self.controller = self.learner = None
        self.epoch, self.active = 10, copy.deepcopy(failure_state["active"])
        self.role, self.finished, self.started = "plain_h16", False, False
        self.initial, self.final, self.ancestors, self.training_learners = {}, {}, {}, {}
        self.episode_data, self.pending_tail_length = None, None
        self.reference_tails = []
        self.reference_files = list(failure_state["reference_files"])
        self.completed = list(failure_state["completed"])
        for name in ("raw", "summaries", "models", "states", "tapes", "updates", "training-tails"):
            if not (self.root/name).is_dir():
                raise ValueError("verified historical payload must be copied before recovery")
        if not (self.root/"progress.jsonl").is_file():
            raise ValueError("historical progress is required for the original test barrier")

    def _validate_boundary(self):
        s = self.failure_state
        references = [trajectory_id(w, "plain_h8") for b in range(5)
                      for w in worlds(self.study, "reference", b)]
        evaluations = [trajectory_id(w, r) for b in range(5)
                       for w in worlds(self.study, "evaluation", b) for r in EVAL_ROLES]
        world = list(worlds(self.study, "evaluation", 4))[-1]
        if (s["format"] != "capacity-planner-tail-runner-v1"
                or s["active"] != dict(world, role="plain_h16") or s["epoch"] != 10
                or s["role"] != "plain_h16" or s["learner"] is not None
                or s["training_learners"] or s["reference_tails"]
                or s["pending_tail_length"] is not None
                or s["completed"] != references+evaluations[:-1]
                or s["reference_files"] != ["training-tails/"+n+".pkl.gz" for n in references]
                or set(s["ancestors"]) != set(range(5))
                or set(s["final"]) != {f"block{b}-{r}" for b in range(5) for r in TRAIN_ROLES}
                or {k:len(v) for k,v in s["episode_data"].items()}
                   != dict(features=11, heuristics=11, costs=10)):
            raise ValueError("only the saved final H16 epoch10 boundary is supported")

    def _optimizer(self, *args, **kwargs):
        raise PermissionError("recovery is evaluation/diagnostics only; no optimizer calls")

    def status(self, event, **values):
        cumulative = {k:v+self.current_old_counts[k] for k,v in self.budget.counts.items()}
        record = dict(event=event, active=self.active, epoch=self.epoch, counts=cumulative,
                      new_counts=self.budget.counts, current_old_counts=self.current_old_counts, **values)
        with (self.root/"progress.jsonl").open("a", encoding="utf8") as handle:
            handle.write(json.dumps(jsonable(record), allow_nan=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def state_dict(self):
        state = super().state_dict()
        state.update(format="capacity-planner-tail-recovery1-runner-v1",
                     current_old_counts=self.current_old_counts,
                     partial_raw_path=str(self.partial_raw_path), automatic_resume=False)
        return state

    def _bind_models(self):
        for b, expected in self.failure_state["ancestors"].items():
            raw = (self.root/"models"/f"block{b}-ancestor.pt").read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("ancestor bytes differ from saved state")
            self.ancestors[b] = raw, expected
        for key, expected in self.failure_state["final"].items():
            raw = (self.root/"models"/(key+"-final.pt")).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("final model bytes differ from saved state")
            self.final[key] = raw, expected
        barrier = json.loads((self.root/"models/all-sealed.json").read_text())
        if barrier != dict(final_models=self.failure_state["final"],
                           ancestors={f"block{k}":v for k,v in self.failure_state["ancestors"].items()}):
            raise ValueError("original model barrier mismatch")

    def _resume_partial(self):
        s = self.failure_state
        w = {k:v for k,v in self.active.items() if k != "role"}
        relative = "raw/"+PARTIAL+".jsonl.gz"
        if (self.root/relative).exists() or (self.root/relative).is_symlink():
            raise FileExistsError("new canonical raw must be absent; preserve old prefix separately")
        prefix = gzip.decompress(self.partial_raw_path.read_bytes())
        rows = [json.loads(line) for line in prefix.splitlines()]
        if (len(rows) != 10 or not prefix.endswith(b"\n")
                or any(r["epoch"] != i or r["world"] != w or r["role"] != self.role
                       for i,r in enumerate(rows))):
            raise ValueError("saved raw must contain exactly epochs0..9 of the final H16 world")
        data = copy.deepcopy(s["episode_data"])
        expected = dict(features=[r["value_features"] for r in rows]+[rows[-1]["next_value_features"]],
                        heuristics=[r["value_base"] for r in rows]+[rows[-1]["next_value_base"]],
                        costs=[r["cost"] for r in rows])
        if jsonable(data) != expected:
            raise ValueError("saved raw and saved episode data differ")
        tape = CapacityWorldTape(**json.loads((self.root/"tapes"/
                                   (trajectory_id(w, "exogenous")+".json")).read_text()))
        if tape.world != w:
            raise ValueError("saved tape identity mismatch")
        self.budget.job(PARTIAL+"-remaining", "evaluation_h16_world")
        self.env = self.env_factory.from_state_dict(self.p, tape, s["environment"], self._native)
        public = self.capture(self.env)
        if self.env.t != 10 or public.common.epoch != 10:
            raise ValueError("native restore must start at epoch10 without construction/reset")
        self.controller = self.controller_factory(self.p, base_control=self.base_control,
            scientific=True, value=None, planning_horizon=16, capture_epochs=(),
            before_tail_clone=self._tail_clone, before_tail_step=self._tail_step)
        self.controller.load_state_dict(copy.deepcopy(s["controller"]))
        if jsonable(public) != jsonable(self.public_decoder(s["controller"]["filter"]["last_view"])):
            raise ValueError("saved public/filter boundary mismatch")
        # The epoch10 receipt was consumed before interruption. Do not replay it.
        self.episode_data = data
        losses = {p["patient_id"] for p in rows[-1]["patient_records"] if p["status"] == "lost"}
        total = sum(data["costs"])
        self.status("partial_trajectory_restored", prefix_rows=10, remaining_steps=54,
                    prefix_sha256=hashlib.sha256(prefix).hexdigest())
        with gzip.open(self.root/relative, "xb") as handle:
            handle.write(prefix)
            for epoch in range(10, 64):
                self.epoch = epoch
                if epoch >= 48:
                    hours = np.zeros(4)
                else:
                    self.budget.debit(dict(planner_total_decisions=1, planner_candidate_rollouts=48))
                    start = shared_monotonic()
                    hours = self.controller.act(public, role="id_mpc", before_query=self._query,
                                                before_filter=self._filter)
                    self.controller.last_plan["decision_wall_seconds"] = shared_monotonic()-start
                    self.controller.last_plan["includes_training_tail_capture"] = False
                    self.budget.finish_chunk("planner_total_model_epochs")
                base = self.base_control(public, self.p, tail=epoch >= 48)
                self.controller.record_operation(public, base)
                _, reward, done, info = self.env.step(PatientSupportAction(
                    tuple(map(float, base)), tuple(map(float, hours))))
                cost = float(info["cost"])
                components = {k:float(info[k]) for k in COST_KEYS}
                if (not math.isfinite(cost) or cost < 0 or not all(math.isfinite(v) for v in components.values())
                        or not math.isclose(math.fsum(components.values()), cost, rel_tol=1e-12, abs_tol=1e-6)
                        or not math.isclose(-float(reward), cost, rel_tol=1e-12, abs_tol=1e-6)
                        or bool(done) != (epoch == 63)):
                    raise ValueError("native objective/terminal mismatch")
                following = self.capture(self.env)
                self.budget.debit(dict(estimator_receipt_updates=1))
                self._observe(following)
                x, h = self.feature_adapter(following, self.controller)
                data["features"].append(x)
                data["heuristics"].append(h)
                data["costs"].append(cost)
                patients = following.operations.patients
                next_losses = {p.patient_id for p in patients if p.status == "lost"}
                if not losses.issubset(next_losses):
                    raise ValueError("lost patient reopened")
                raw = dict(epoch=epoch, world=w, role=self.role, cost=cost, reward=float(reward),
                    components=components, new_lost_patients=len(next_losses-losses), patient_records=patients,
                    cumulative=dict(enrolled=len(patients), lost=len(next_losses),
                                    delivered=sum(p.status == "delivered" for p in patients)),
                    requested_hours=hours, executed_hours=info["support_public_receipt"]["committed_hours"],
                    public_input=public, value_features=data["features"][-2], value_base=data["heuristics"][-2],
                    next_value_features=x, next_value_base=h,
                    plan=self.controller.last_plan if epoch < 48 else None, info=info)
                handle.write((json.dumps(jsonable(raw), sort_keys=True, allow_nan=False)+"\n").encode())
                losses, public, total = next_losses, following, total+cost
                if (epoch+1) % 8 == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    self.budget.check(storage=True)
                    self.status("epoch_boundary")
        settlement = self.env.settlement()
        summary = dict(world=w, role=self.role, raw_path=relative, cost=total,
            settled=settlement["settled"], settlement=settlement, lost=len(losses),
            tape_sha256=tape.digest(), change_epoch=tape.change_epoch, model_seal_sha256=None,
            initial_ancestor_sha256=None, value_updates_before_trajectory=None,
            optimizer_updates_during_trajectory=0)
        if not settlement["settled"] or settlement["lost"] != len(losses):
            raise ValueError("unsettled cohort; no horizon extension")
        self._json("summaries/"+PARTIAL+".json", summary)
        self.epoch = 64
        self.completed.append(PARTIAL)
        self._save_state(PARTIAL)
        self.status("trajectory_completed", summary=summary)
        self.episode_data = None

    def run(self):
        if self.started:
            raise RuntimeError("single attempt; no recovery relaunch")
        self.started = True
        self._bind_models()
        self.budget.enter("frozen_evaluation", "frozen_evaluation")
        self._resume_partial()
        self._diagnostics()
        if (self.budget.counts != self.budget.limits or self.budget.chunks
                or len(self.completed) != 480 or len(set(self.completed)) != 480):
            raise RuntimeError("remaining-only completion counters mismatch")
        self.finished = True
        self.status("all_trajectories_complete")
        return dict(trajectories=480, new_completed_trajectories=1,
                    native_steps_reused=30666, native_steps_new=54,
                    current_old_counts=self.current_old_counts,
                    budget=self.budget.snapshot(), independent_confirmation=False)
