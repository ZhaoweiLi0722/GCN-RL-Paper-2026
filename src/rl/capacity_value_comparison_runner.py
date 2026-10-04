"""Shared-data initialization, paired closed-loop training, then frozen tests."""

import copy
import os

from src.baselines.capacity_value_mpc import CapacityValueMPC
from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_pilot_runner import CapacityPilotRunner, jsonable, trajectory_id
from src.rl.capacity_value_comparison_learner import CapacityValueComparisonLearner
from src.rl.capacity_value_comparison_resources import ARCHITECTURES, PHASES, SCOPE, worlds
from src.rl.capacity_value_features import FEATURE_NAMES, observed_features
from src.rl.capacity_value_runner import CapacityValueRunner


def parameter_count(learner):
    return sum(p.numel() for p in learner.model.parameters() if p.requires_grad)


class CapacityValueComparisonRunner(CapacityValueRunner):
    def __init__(self, root, proposal, budget, *, admission,
                 value_factory=CapacityValueComparisonLearner, feature_adapter=observed_features,
                 parameter_counter=parameter_count, **backends):
        if admission != dict(verified=True, scope=SCOPE):
            raise PermissionError("new exact graph/flat comparison admission required")
        controller = backends.pop("controller_factory", CapacityValueMPC)
        self._score_value = False

        def select_controller(*args, value=None, **kwargs):
            # Old trajectory IO is unchanged; learned roles are explicitly bound here.
            return controller(*args, value=self.learner if self._score_value else None, **kwargs)

        CapacityPilotRunner.__init__(self, root, proposal, budget,
            admission=dict(verified=True, scope="dynamic-capacity-pilot-v1"),
            controller_factory=select_controller, **backends)
        self.value_factory, self.feature_adapter = value_factory, feature_adapter
        self.parameter_counter = parameter_counter
        self.study = proposal["value_comparison_study"]
        self.initial, self.final, self.episode_data = {}, {}, None
        self.training_learners, self.parameter_counts = {}, {}
        self.started = False

    def state_dict(self):
        state = super().state_dict()
        state.update(format="capacity-value-comparison-runner-v1",
            score_value=self._score_value,
            training_learners={a:l.state_dict() for a,l in self.training_learners.items()},
            parameter_counts=self.parameter_counts)
        return state

    def _episode(self, world, role, tape, *, seal=None):
        if role not in ("plain_mpc", "graph_value_mpc", "flat_value_mpc", "fixed_allocation_reference"):
            raise ValueError("unknown comparison controller")
        self._score_value = role in ("graph_value_mpc", "flat_value_mpc")
        if self._score_value and (self.learner is None or self.learner.config["architecture"] != role.split("_")[0]):
            raise ValueError("wrong architecture for controller")
        return super()._episode(world, role, tape, seal=seal)

    def _seal(self, block, architecture, *, final=False):
        key = f"block{block}-{architecture}"
        name = key+("-final" if final else "-initial")
        if self.learner.pending is not None or self.learner.updates != (1536 if final else 768):
            raise ValueError("incorrect complete-model seal boundary")
        self.budget.job(name, "seal")
        raw, sha = self.learner.snapshot()
        self.budget.debit({"final_seals" if final else "initial_seals": 1})
        with (self.root/"models"/(name+".pt")).open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        self._json("models/"+name+".json", dict(sha256=sha, bytes=len(raw),
            updates=self.learner.updates, architecture=architecture,
            parameters=self.parameter_counts[architecture], config=self.learner.config))
        (self.final if final else self.initial)[key] = raw, sha
        self.status("final_sealed" if final else "initial_sealed", block=block,
                    architecture=architecture, sha256=sha)

    def _fit_data(self, world, architecture, data, source_role):
        role = architecture+"_value_fit"
        self.active = dict(world, role=role)
        ident = trajectory_id(world, role)
        self._learn(data)
        self._json("updates/"+ident+"-input.json", dict(
            architecture=architecture, source_trajectory=trajectory_id(world, source_role),
            data_sha256=digest(jsonable(data)), updates_after=self.learner.updates))

    def run(self):
        if self.started:
            raise RuntimeError("single attempt; no resume/relaunch")
        self.started = True
        for block in range(5):
            self.budget.enter(PHASES[0], PHASES[0])
            self.training_learners = {}
            for architecture in ARCHITECTURES:
                config = dict(copy.deepcopy(self.study["value"]), architecture=architecture)
                learner = self.value_factory(config, seed=self.study["design"]["model_seeds"][architecture][block],
                    feature_dim=len(FEATURE_NAMES), before_forward=self._forward, before_optimizer=self._optimizer)
                self.training_learners[architecture] = learner
                self.parameter_counts[architecture] = self.parameter_counter(learner)
            gap = abs(self.parameter_counts["graph"]-self.parameter_counts["flat"])/self.parameter_counts["graph"]
            if gap > self.study["value"]["max_parameter_gap_fraction"]:
                raise ValueError("graph/flat parameter gap exceeds frozen bound")
            self._json(f"models/block{block}-architecture.json", dict(parameters=self.parameter_counts,
                gap_fraction=gap, public_features=list(FEATURE_NAMES), initial_data_shared=True))
            for w in worlds(self.study, "initial", block):
                self.learner = None
                data = self._episode(w, "plain_mpc", self._tape(w))
                for architecture in ARCHITECTURES:
                    self.learner = self.training_learners[architecture]
                    self._fit_data(w, architecture, data, "plain_mpc")
            for architecture in ARCHITECTURES:
                self.learner = self.training_learners[architecture]
                self._seal(block, architecture)
            self.budget.enter(PHASES[1], PHASES[1])
            for architecture in ARCHITECTURES:
                key = f"block{block}-{architecture}"
                self.training_learners[architecture] = self.value_factory.from_bytes(self.initial[key][0],
                    before_forward=self._forward, before_optimizer=self._optimizer)
            for w in worlds(self.study, "continuation", block):
                tape = self._tape(w)
                for architecture in ARCHITECTURES:
                    key, role = f"block{block}-{architecture}", architecture+"_value_mpc"
                    self.learner = self.training_learners[architecture]
                    data = self._episode(w, role, tape, seal=self.initial[key][1])
                    self._fit_data(w, architecture, data, role)
            for architecture in ARCHITECTURES:
                self.learner = self.training_learners[architecture]
                self._seal(block, architecture, final=True)
            self.training_learners = {}
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
        if self.budget.counts != self.budget.limits:
            raise RuntimeError("comparison completion counts mismatch")
        self.finished = True
        self.status("all_trajectories_complete")
        return dict(trajectories=len(self.completed), initial_models={k:v[1] for k,v in self.initial.items()},
            final_models={k:v[1] for k,v in self.final.items()}, parameter_counts=self.parameter_counts,
            budget=self.budget.snapshot())
