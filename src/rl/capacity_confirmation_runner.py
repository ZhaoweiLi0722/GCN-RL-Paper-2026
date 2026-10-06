"""Fresh shared-data initialization and one sealed seven-role comparison."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np

from src.baselines.capacity_planner_tail_mpc import CapacityPlannerTailMPC
from src.rl.capacity_confirmation_design import (SCOPE, WARM_ROLES, TRAIN_ROLES, EVAL_ROLES,
    capture_epochs, worlds, warm_config, tail_config)
from src.rl.capacity_confirmation_learner import CapacityConfirmationWarmLearner, CapacityConfirmationTailLearner
from src.rl.capacity_native_tail_collector import collect_native_tail
from src.rl.capacity_native_tail_runner import CapacityNativeTailRunner
from src.rl.capacity_pilot_runner import CapacityPilotRunner, trajectory_id, jsonable
from src.rl.capacity_value_runner import CapacityValueRunner
from src.rl.capacity_value_comparison_learner import CapacityValueComparisonLearner
from src.rl.capacity_value_features import observed_features
from src.utils.research_clock import shared_monotonic


class CapacityConfirmationRunner(CapacityNativeTailRunner):
    def __init__(self, root, proposal, budget, *, admission, workspace,
                 warm_factory=CapacityConfirmationWarmLearner, value_factory=CapacityConfirmationTailLearner,
                 frozen_factory=CapacityValueComparisonLearner, controller_factory=CapacityPlannerTailMPC,
                 native_collector=collect_native_tail, feature_adapter=observed_features, **backends):
        if admission != dict(verified=True, scope=SCOPE):
            raise PermissionError("fresh numerical package admission required")
        self.workspace, self.study = Path(workspace), proposal["confirmation_study"]
        self.warm_factory, self.value_factory, self.frozen_factory = warm_factory, value_factory, frozen_factory
        self.native_collector, self.unwrapped_controller_factory = native_collector, controller_factory
        self.feature_adapter = feature_adapter
        self.training_learners, self.warm, self.ancestors, self.legacy, self.final, self.initial = {}, {}, {}, {}, {}, {}
        self.episode_data, self.continuation, self.native_branch_boundary, self.branch_failure = None, None, None, None
        self.reference_tails, self.native_tails, self.reference_files = [], [], []
        self.role, self.started = None, False

        def controller(*args, **kwargs):
            kwargs.update(value=self.learner if self.role not in ("plain_h8", "plain_h16") else None,
                          planning_horizon=16 if self.role == "plain_h16" else 8, capture_epochs=())
            out = controller_factory(*args, **kwargs)
            act = out.act
            def timed(*a, **k):
                start = shared_monotonic()
                result = act(*a, **k)
                collecting = self.active["phase"] == "reference" and a[0].common.epoch in capture_epochs(self.active["index"])
                if collecting:
                    self.native_tails.append(self._collect_native(a[0], out))
                out.last_plan.update(decision_wall_seconds=shared_monotonic() - start,
                    includes_training_tail_capture=collecting)
                return result
            out.act = timed
            return out

        CapacityPilotRunner.__init__(self, root, proposal, budget,
            admission=dict(verified=True, scope="dynamic-capacity-pilot-v1"),
            controller_factory=controller, **backends)
        (self.root / "training-tails").mkdir(exist_ok=False)
        (self.root / "native-branches").mkdir(exist_ok=False)

    def state_dict(self):
        state = super().state_dict()
        state.update(format="capacity-confirmation-runner-v1",
            warm={k: v[1] for k, v in self.warm.items()}, legacy={k: v[1] for k, v in self.legacy.items()})
        return state

    def _optimizer(self, name, batch):
        if self.budget.phase != "value_fitting" or name != "value" or self.role not in (*WARM_ROLES, *TRAIN_ROLES):
            raise PermissionError("optimizer outside explicit training role/phase")
        self.budget.debit(dict(value_optimizer_steps=1, total_optimizer_steps=1,
                               optimizer_example_presentations=batch))

    def _episode(self, world, role, tape, *, seal=None):
        if role not in EVAL_ROLES or world["phase"] != "evaluation" and role != "plain_h8":
            raise ValueError("unknown role or nonshared training behavior")
        self.role, self.native_tails = role, []
        original = self.budget.job
        kind = {"warmup": "warmup_world", "reference": "reference_world_with_tails",
                "evaluation": "evaluation_h16_world" if role == "plain_h16" else "evaluation_h8_world"}[world["phase"]]
        self.budget.job = lambda identifier, tag: original(identifier, kind if tag == "world" else tag)
        try:
            return CapacityValueRunner._episode(self, world, role, tape, seal=seal)
        finally:
            self.budget.job = original

    def _write_tails(self, world):
        if len(self.native_tails) != 2:
            raise ValueError("exactly two complete native branches required")
        relative = "training-tails/" + trajectory_id(world, "plain_h8") + ".pkl.gz"
        with gzip.open(self.root / relative, "xb") as handle:
            pickle.dump(dict(native=self.native_tails), handle, protocol=5)
        self.budget.check(storage=True)
        self.reference_files.append(relative)
        return relative

    def _fit(self, world, role, data, tails_path=None):
        self.budget.enter("value_fitting", "value_fitting")
        self.role, self.active = role, dict(world, role=role)
        name = trajectory_id(world, role)
        self.budget.job(name, "fit")
        self.learner = self.training_learners[role]
        if world["phase"] == "warmup":
            self.learner.admit_episode({k: np.asarray(v, dtype=np.float32 if k == "features" else np.float64)
                                        for k, v in data.items()})
        else:
            fields = ("format", "root_epoch", "candidate", "start_epoch", "label_policy",
                "continuation_sha256", "source_state_sha256", "tape_sha256", "provenance",
                "epochs", "features", "heuristics", "costs")
            self.learner.admit_tails([{k: r[k] for k in fields} for r in self.native_tails],
                world_index=world["index"], tape_sha256=self.env.state_dict()["world_tape_sha256"])
        self._json("updates/" + name + "-targets.json", dict(targets=self.learner.pending["targets"],
            source_hashes=self.learner.pending.get("source_hashes", []),
            source_native=trajectory_id(world, "plain_h8"), source_tails=tails_path,
            own_ancestor_sha256=getattr(self.learner, "ancestor_sha256", None),
            continuation_sha256=self.ancestors[world["block"]][1] if tails_path else None, method=role))
        with (self.root / "updates" / (name + ".jsonl")).open("x") as handle:
            for _ in range(32):
                receipt = self.learner.update()
                handle.write(json.dumps(jsonable(receipt), allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        self._save_state(name + "-after-fit")
        self.status("value_fit_completed", updates=self.learner.updates, method=role)

    def _seal(self, block, role, *, warm=False):
        learner = self.training_learners[role]
        updates = 1536 if warm else 768
        if learner.updates != updates or learner.pending is not None:
            raise ValueError("incomplete model seal")
        key = f"block{block}-{role}"
        name = key + ("-warm" if warm else "-final")
        self.budget.job(name, "seal")
        raw, sha = learner.snapshot()
        self.budget.debit({"warm_seals" if warm else "final_seals": 1})
        with (self.root / "models" / (name + ".pt")).open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        self._json("models/" + name + ".json", dict(sha256=sha, bytes=len(raw), updates=updates,
            role=role, block=block, config=learner.config, parameters=3169,
            own_ancestor_sha256=getattr(learner, "ancestor_sha256", None),
            continuation_sha256=None if warm else self.ancestors[block][1]))
        (self.warm if warm else self.final)[key] = raw, sha
        if warm and role == "graph":
            self.ancestors[block] = raw, sha
        self.status("warm_sealed" if warm else "final_sealed", block=block, role=role, sha256=sha)

    def _load_value(self, block, role):
        if role in ("plain_h8", "plain_h16"):
            return None, None
        if role == "existing_frozen":
            pair, factory = self.legacy[block], self.frozen_factory
        elif role == "fresh_frozen":
            pair, factory = self.ancestors[block], self.warm_factory
        else:
            pair, factory = self.final[f"block{block}-{role}"], self.value_factory
        return factory.from_bytes(pair[0], before_forward=self._forward, before_optimizer=self._optimizer), pair[1]

    def run(self):
        if self.started:
            raise RuntimeError("single attempt; no automatic resume or relaunch")
        self.started = True
        for block in range(5):
            source = self.study["initial_models"]
            path = self.workspace / source["root"] / source["names"][block]
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            if sha != source["sha256"][block]:
                raise ValueError("legacy reference hash mismatch")
            self.legacy[block] = raw, sha
            self._json(f"models/block{block}-legacy.json", dict(sha256=sha, updates=1536,
                source_path=str(path.relative_to(self.workspace)), historical_reference_only=True))
            self.training_learners = {role: self.warm_factory(warm_config(self.study, role),
                seed=self.study["streams"]["model_seeds"][block], feature_dim=31,
                before_forward=self._forward, before_optimizer=self._optimizer) for role in WARM_ROLES}
            for world in worlds(self.study, "warmup", block):
                self.budget.enter("warmup_reference", "warmup_reference")
                self.learner = None
                data = self._episode(world, "plain_h8", self._tape(world))
                for role in WARM_ROLES:
                    self._fit(world, role, data)
            for role in WARM_ROLES:
                self._seal(block, role, warm=True)
            self.training_learners = {}
            parent_raw, parent_sha = self.ancestors[block]
            self.continuation = self.warm_factory.from_bytes(parent_raw,
                before_forward=self._forward, before_optimizer=self._optimizer)
            for role in TRAIN_ROLES:
                architecture = "self_only" if role == "self_only_td" else "graph"
                raw, sha = self.warm[f"block{block}-{architecture}"]
                self.training_learners[role] = self.value_factory.fork_weights(raw,
                    tail_config(self.study, role, parent_sha), seed=self.study["streams"]["sampler_seeds"][block],
                    expected_sha256=sha, before_forward=self._forward, before_optimizer=self._optimizer)
            for world in worlds(self.study, "reference", block):
                self.budget.enter("reference_and_tails", "reference_and_tails")
                self.learner = None
                data = self._episode(world, "plain_h8", self._tape(world))
                path = self._write_tails(world)
                for role in TRAIN_ROLES:
                    self._fit(world, role, data, path)
            for role in TRAIN_ROLES:
                self._seal(block, role)
            self.training_learners = {}
        if len(self.warm) != 10 or len(self.final) != 15 or len(self.legacy) != 5:
            raise RuntimeError("all fresh models must seal before tests")
        self._json("models/all-sealed.json", dict(warm={k: v[1] for k, v in self.warm.items()},
            final={k: v[1] for k, v in self.final.items()}, legacy={str(k): v[1] for k, v in self.legacy.items()}))
        self.status("all_models_sealed")
        self.budget.enter("frozen_evaluation", "frozen_evaluation")
        for block in range(5):
            for world in worlds(self.study, "evaluation", block):
                tape = self._tape(world)
                for role in EVAL_ROLES:
                    self.learner, sha = self._load_value(block, role)
                    self._episode(world, role, tape, seal=sha)
        self.budget.enter("analysis_archive", "analysis_archive")
        if self.budget.counts != self.budget.limits:
            raise ValueError("fresh-study counters do not reconcile")
        self.finished = True
        self.status("all_trajectories_complete")
        return dict(trajectories=len(self.completed), budget=self.budget.snapshot(),
                    final_models={k: v[1] for k, v in self.final.items()})
