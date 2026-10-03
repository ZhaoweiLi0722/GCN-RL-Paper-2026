"""Budget-hooked, training-only dual-critic capacity learner.

The coordinator admits complete settled cohorts and owns scientific execution.
Snapshots preserve consumed work; neither loading nor a failure grants a retry.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import math

import numpy as np

from src.models.capacity_ddpg import CapacityCritic, CapacityModelConfig
from src.rl.capacity_ddpg_learner import (
    CapacityBatch, CapacityDDPGLearner, CapacitySnapshot, CapacityTransition, td_targets,
)
from src.rl.networks import torch


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"native integer {name} >= {minimum} required")
    return value


@dataclass(frozen=True)
class PatientLossTransition(CapacityTransition):
    patient_losses: int
    settlement_losses: int | None
    condition: int
    total_losses: int

    def __post_init__(self):
        super().__post_init__()
        _integer(self.patient_losses, "patient_losses")
        _integer(self.condition, "condition")
        _integer(self.total_losses, "total_losses")
        if self.condition >= 3:
            raise ValueError("unknown training condition")
        if self.done:
            _integer(self.settlement_losses, "settlement_losses")
        elif self.settlement_losses is not None:
            raise ValueError("tail losses belong only on a terminal transition")
        if self.total_losses != self.patient_losses + (self.settlement_losses or 0):
            raise ValueError("patient and settlement losses do not reconcile")
        if self.total_losses > np.finfo(np.float32).max:
            raise ValueError("loss target outside float32 support")

    @classmethod
    def from_cost(cls, *, patient_losses, settlement_losses=None, condition, **kwargs):
        base = CapacityTransition.from_cost(**kwargs)
        _integer(patient_losses, "patient_losses")
        if base.done:
            _integer(settlement_losses, "settlement_losses")
        return cls(**asdict(base), patient_losses=patient_losses,
                   settlement_losses=settlement_losses, condition=condition,
                   total_losses=patient_losses + (settlement_losses or 0))


@dataclass(frozen=True)
class PatientConstrainedConfig:
    model: CapacityModelConfig
    actor_lr: float
    critic_lr: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    gradient_norm_cap: float
    gamma: float
    target_polyak_tau: float
    batch_size: int
    exploration_sigma_hours: float
    warmup_batches: int
    training_worlds: int
    updates_per_world: int
    reference_capacity: int
    training_capacity: int
    reference_batch_rows: int
    training_batch_rows: int
    multiplier_initial: tuple[float, ...]
    multiplier_step_size: float
    multiplier_bounds: tuple[float, float]
    conditions: int = 3
    reward_divisor: float = 100000.0

    def __post_init__(self):
        for name in ("betas", "multiplier_initial", "multiplier_bounds"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("batch_size", "warmup_batches", "training_worlds", "updates_per_world",
                     "reference_capacity", "training_capacity", "reference_batch_rows", "training_batch_rows", "conditions"):
            _integer(getattr(self, name), name, 1)
        positive = (self.actor_lr, self.critic_lr, self.epsilon, self.gradient_norm_cap,
                    self.reward_divisor, self.multiplier_step_size)
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("finite positive optimizer/reward/dual settings required")
        if (len(self.betas) != 2 or any(not 0 <= x < 1 for x in self.betas)
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0
                or not math.isfinite(self.exploration_sigma_hours) or self.exploration_sigma_hours < 0
                or not 0 <= self.gamma <= 1 or not 0 < self.target_polyak_tau <= 1):
            raise ValueError("invalid optimizer/discount/noise configuration")
        if (self.conditions != 3 or self.batch_size != 64
                or (self.reference_batch_rows, self.training_batch_rows) != (32, 32)
                or self.updates_per_world != 48 or self.training_worlds != 12
                or self.warmup_batches != 256
                or (self.reference_capacity, self.training_capacity) != (576, 576)
                or self.reward_divisor != 100000.0):
            raise ValueError("unsupported bounded patient-constrained schedule")
        if (len(self.multiplier_bounds) != 2 or self.multiplier_bounds[0] != 0
                or not math.isfinite(self.multiplier_bounds[1]) or self.multiplier_bounds[1] <= 0
                or len(self.multiplier_initial) != self.conditions
                or any(not self.multiplier_bounds[0] <= x <= self.multiplier_bounds[1]
                       for x in self.multiplier_initial)):
            raise ValueError("invalid multiplier configuration")

    # The unchanged actor, gradient, and initialization helpers use these names.
    @property
    def offline_actor_lr(self):
        return self.actor_lr

    @property
    def offline_critic_lr(self):
        return self.critic_lr

    @classmethod
    def from_proposal(cls, proposal, *, node_input_dim):
        numeric = proposal["patient_constrained"]
        learner, objective = numeric["learner"], numeric["objective"]
        support = proposal["synthetic_system"]["support"]
        if (learner["bc_steps"] != 0 or not learner["zero_final_actor_affine"]
                or not learner["sample_current_training_condition_with_replacement"]
                or not learner["rollout_weights_fixed"] or learner["optimizer"] != "Adam"
                or learner["shared_encoder"] or learner["evaluation_optimizer_steps"] != 0
                or learner["independent_critics"] != ["reward", "patient_loss_count"]
                or not objective["cost_only_shadow_loss_critic"]
                or objective["cost_only_ablation_lambda"] != [0, 0, 0]
                or objective["reward"] != "negative_full_settled_cost_divided_by_100000"
                or learner["exact_initial_projected_hours"] != [2, 2, 2, 2]
                or support["site_hour_caps"] != [4, 4, 4, 4] or support["shared_hour_budget"] != 8):
            raise ValueError("unsupported patient-constrained numerical contract")
        model = CapacityModelConfig(node_input_dim, tuple(learner["graph_widths"]),
            tuple(learner["actor_head"]), tuple(support["site_hour_caps"]),
            support["shared_hour_budget"], 2.0, proposal["learner"]["feature_divisors"]["hours"])
        return cls(model, learner["actor_lr"], learner["critic_lr"], tuple(learner["betas"]),
            learner["epsilon"], learner["weight_decay"], learner["gradient_norm_cap"],
            learner["gamma"], learner["target_polyak_tau"], learner["batch_size"],
            learner["exploration_sigma_hours"], learner["shared_dual_critic_warmup_batches_per_block"],
            learner["training_worlds_per_arm_block"], learner["updates_after_each_settled_world"],
            learner["reference_replay_capacity"], learner["learned_replay_capacity_per_arm"],
            learner["update_batch_reference_rows"], learner["update_batch_learned_rows"],
            tuple(objective["multiplier_initial"]), objective["multiplier_step_size"],
            tuple(objective["multiplier_bounds"]), numeric["design"]["conditions"])


@dataclass(frozen=True)
class PatientLossBatch(CapacityBatch):
    losses: object
    condition: int


class PatientConstrainedLearner(CapacityDDPGLearner):
    module_names = ("actor", "critic", "loss_critic", "target_actor", "target_critic", "target_loss_critic")
    optimizer_names = ("actor", "reward_critic", "loss_critic")

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.torch_rng_state)
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                self.loss_critic = CapacityCritic(config.model, adjacency=kwargs.get("adjacency"))
            finally:
                torch.set_default_dtype(previous_dtype)
            self.torch_rng_state = torch.get_rng_state().clone()
        with torch.no_grad():
            self.actor.head[-1].weight.zero_()
            self.actor.head[-1].bias.zero_()
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_loss_critic = copy.deepcopy(self.loss_critic)
        self.target_loss_critic.requires_grad_(False)
        self.loss_critic_optimizer = torch.optim.Adam(self.loss_critic.parameters(), lr=config.critic_lr,
            betas=config.betas, eps=config.epsilon, weight_decay=config.weight_decay, foreach=False)
        self.reference_replay, self.training_replay = [], []
        self.arm = None
        self.multipliers = list(config.multiplier_initial)
        self.counts = dict.fromkeys(("neural_forward_module_calls", "actor_optimizer_steps",
            "critic_optimizer_steps", "reward_critic_optimizer_steps", "loss_critic_optimizer_steps",
            "total_optimizer_steps", "optimizer_example_presentations", "warmup_steps", "training_steps",
            "scalar_multiplier_updates", "reference_rows", "training_rows"), 0)
        self.completed = dict(warmup=0, training=0, worlds=0, multiplier=0)
        self.cohort = dict(world_id=None, condition=None, rows=0, losses=0,
                           pending_updates=0, multiplier_ready=False)
        self._fresh_fork = False

    @classmethod
    def from_proposal(cls, proposal, *, node_input_dim, **kwargs):
        return cls(PatientConstrainedConfig.from_proposal(proposal, node_input_dim=node_input_dim), **kwargs)

    @property
    def reward_critic(self):
        return self.critic

    @property
    def reward_critic_optimizer(self):
        return self.critic_optimizer

    def _condition(self, condition):
        _integer(condition, "condition")
        if condition >= self.config.conditions:
            raise ValueError("unknown training condition")

    def _validate_row(self, row):
        if type(row) is not PatientLossTransition:
            raise TypeError("PatientLossTransition required")
        PatientLossTransition(**asdict(row))
        self._condition(row.condition)
        base = CapacityTransition(**{f.name: getattr(row, f.name) for f in fields(CapacityTransition)})
        super()._validate_row(base)

    def add_reference(self, row):
        def add():
            if self.arm is not None or self.counts["warmup_steps"] or len(self.reference_replay) >= self.config.reference_capacity:
                raise ValueError("reference replay sealed/full")
            self._validate_row(row)
            self.reference_replay.append(row)
            self.counts["reference_rows"] += 1
        return self._run(add)

    def set_arm(self, arm):
        def select():
            if (not self._fresh_fork or self.arm is not None or arm not in ("constrained", "cost_only")
                    or self.completed["warmup"] != self.config.warmup_batches
                    or self.counts["training_steps"] or self.training_replay):
                raise ValueError("arm selection requires a fresh complete warmup fork")
            self.arm = arm
            self.multipliers = list(self.config.multiplier_initial) if arm == "constrained" else [0.0] * self.config.conditions
            self._fresh_fork = False
        return self._run(select)

    def add_training(self, row):
        def add():
            c, cohort = self.config, self.cohort
            if self.arm is None or cohort["pending_updates"] or len(self.training_replay) >= c.training_capacity:
                raise ValueError("training replay unavailable, pending updates or full")
            self._validate_row(row)
            if row.condition != self.completed["worlds"] % c.conditions:
                raise ValueError("training conditions must follow the fixed cycle")
            if cohort["world_id"] is None:
                if any(old.world_id == row.world_id for old in self.training_replay):
                    raise ValueError("training world already consumed")
                cohort.update(world_id=row.world_id, condition=row.condition)
            if row.world_id != cohort["world_id"] or row.condition != cohort["condition"]:
                raise ValueError("mixed training cohort")
            if row.done != (cohort["rows"] + 1 == c.updates_per_world):
                raise ValueError("exact 48-row settled control cohort required")
            self.training_replay.append(row)
            self.counts["training_rows"] += 1
            cohort["rows"] += 1
            cohort["losses"] += row.total_losses
            if row.done:
                cohort["pending_updates"] = c.updates_per_world
                cohort["multiplier_ready"] = self.arm == "cost_only"
        return self._run(add)

    def update_multiplier(self, condition, learned_losses, reference_losses):
        def update():
            self._condition(condition)
            _integer(learned_losses, "learned_losses")
            _integer(reference_losses, "reference_losses")
            cohort, c = self.cohort, self.config
            if (self.arm != "constrained" or cohort["condition"] != condition
                    or cohort["pending_updates"] != c.updates_per_world or cohort["multiplier_ready"]
                    or learned_losses != cohort["losses"]
                    or self.completed["multiplier"] >= c.training_worlds):
                raise ValueError("multiplier requires one unconsumed settled constrained cohort")
            self.counts["scalar_multiplier_updates"] += 1
            before = self.multipliers[condition]
            difference = learned_losses - reference_losses
            after = min(c.multiplier_bounds[1], max(c.multiplier_bounds[0], before + c.multiplier_step_size * difference))
            self.multipliers[condition] = after
            cohort["multiplier_ready"] = True
            self.completed["multiplier"] += 1
            return dict(condition=condition, learned_losses=learned_losses, reference_losses=reference_losses,
                        difference=difference, before=before, after=after,
                        saturated=after in c.multiplier_bounds, world_id=cohort["world_id"])
        return self._run(update)

    def sample_batch(self, condition, *, warmup=False):
        def sample():
            self._condition(condition)
            refs = [i for i, row in enumerate(self.reference_replay) if row.condition == condition]
            own = [i for i, row in enumerate(self.training_replay) if row.condition == condition]
            if not refs or (not warmup and not own):
                raise ValueError("missing current-condition replay stratum")
            nref = self.config.batch_size if warmup else self.config.reference_batch_rows
            reference_indices = self.replay_rng.choice(refs, nref, replace=True).tolist()
            training_indices = [] if warmup else self.replay_rng.choice(own, self.config.training_batch_rows, replace=True).tolist()
            rows = [self.reference_replay[i] for i in reference_indices] + [self.training_replay[i] for i in training_indices]
            def tensor(name):
                return torch.tensor([getattr(row, name) for row in rows], dtype=torch.float32)
            return PatientLossBatch(tensor("state"), tensor("executed_hours"), tensor("reward").unsqueeze(-1),
                tensor("next_state"), tensor("done").unsqueeze(-1),
                tuple(["reference"] * nref + ["training"] * len(training_indices)),
                tuple(reference_indices + training_indices), tensor("total_losses").unsqueeze(-1), condition)
        return self._run(sample)

    def _optimizer(self, name):
        if name not in self.optimizer_names:
            raise ValueError("unknown dual-critic optimizer")
        self.counts[f"{name}_optimizer_steps"] += 1
        if name != "actor":
            self.counts["critic_optimizer_steps"] += 1
        self.counts["total_optimizer_steps"] += 1
        self.counts["optimizer_example_presentations"] += self.config.batch_size
        self.before_optimizer(name, self.config.batch_size)
        getattr(self, name + "_optimizer").step()

    def _dual_critic_losses(self, batch):
        with torch.no_grad():
            raw = self._forward("target_actor", self.target_actor, batch.next_states)
            executed = self.target_actor.project(raw)
            reward_q = self._forward("target_reward_critic", self.target_critic, batch.next_states, executed)
            loss_q = self._forward("target_loss_critic", self.target_loss_critic, batch.next_states, executed)
            reward_target = td_targets(batch.rewards, batch.dones, reward_q, self.config.gamma)
            loss_target = td_targets(batch.losses, batch.dones, loss_q, self.config.gamma)
        reward_now = self._forward("reward_critic", self.critic, batch.states, batch.executed_hours)
        loss_now = self._forward("loss_critic", self.loss_critic, batch.states, batch.executed_hours)
        mse = torch.nn.functional.mse_loss
        return mse(reward_now, reward_target), mse(loss_now, loss_target), reward_target, loss_target

    def _constrained_actor_loss(self, batch):
        self.critic.requires_grad_(False)
        self.loss_critic.requires_grad_(False)
        try:
            raw = self._forward("actor", self.actor, batch.states)
            executed = self.actor.project(raw)
            reward_q = self._forward("actor_reward_critic", self.critic, batch.states, executed)
            loss_q = self._forward("actor_loss_critic", self.loss_critic, batch.states, executed)
            return (-reward_q + self.multipliers[batch.condition] * loss_q).mean()
        finally:
            self.critic.requires_grad_(True)
            self.loss_critic.requires_grad_(True)

    def _fit(self, condition, *, warmup):
        def update():
            self._condition(condition)
            c, cohort = self.config, self.cohort
            if warmup:
                if (self.arm is not None or self.completed["warmup"] >= c.warmup_batches
                        or len(self.reference_replay) != c.reference_capacity
                        or condition != self.completed["warmup"] % c.conditions):
                    raise ValueError("warmup requires full reference replay and the fixed condition cycle")
                key = "warmup"
            else:
                if (self.arm is None or self.completed["warmup"] != c.warmup_batches
                        or not cohort["pending_updates"] or condition != cohort["condition"]
                        or not cohort["multiplier_ready"]
                        or self.completed["training"] >= c.training_worlds * c.updates_per_world):
                    raise ValueError("training requires a settled cohort and its multiplier decision")
                key = "training"
            self.counts[key + "_steps"] += 1
            batch = self.sample_batch(condition, warmup=warmup)
            for name in self.optimizer_names:
                getattr(self, name + "_optimizer").zero_grad(set_to_none=True)
            reward_loss, loss_loss, reward_target, loss_target = self._dual_critic_losses(batch)
            result = dict(mode=key, condition=condition, sources=batch.sources, indices=batch.indices,
                multiplier=self.multipliers[condition], reward_targets=reward_target[:, 0].tolist(),
                loss_targets=loss_target[:, 0].tolist(), reward_critic_loss=float(reward_loss.detach()),
                loss_critic_loss=float(loss_loss.detach()))
            result["reward_critic_grad_norm"] = self._backward(reward_loss, self.critic)
            if any(p.grad is not None for module in (self.actor, self.loss_critic) for p in module.parameters()):
                raise RuntimeError("reward-critic gradients leaked")
            self._optimizer("reward_critic")
            self.critic_optimizer.zero_grad(set_to_none=True)
            result["loss_critic_grad_norm"] = self._backward(loss_loss, self.loss_critic)
            if any(p.grad is not None for module in (self.actor, self.critic) for p in module.parameters()):
                raise RuntimeError("loss-critic gradients leaked")
            self._optimizer("loss_critic")
            self.loss_critic_optimizer.zero_grad(set_to_none=True)
            if not warmup:
                actor_loss = self._constrained_actor_loss(batch)
                result["actor_loss"] = float(actor_loss.detach())
                result["actor_grad_norm"] = self._backward(actor_loss, self.actor)
                if any(p.grad is not None for module in (self.critic, self.loss_critic) for p in module.parameters()):
                    raise RuntimeError("actor gradients leaked into critics")
                self._optimizer("actor")
            self._polyak(actor=not warmup, critic=True)
            with torch.no_grad():
                for current, previous in zip(self.loss_critic.parameters(), self.target_loss_critic.parameters()):
                    previous.mul_(1 - c.target_polyak_tau).add_(current, alpha=c.target_polyak_tau)
            self._finite_state()
            self.completed[key] += 1
            if not warmup:
                cohort["pending_updates"] -= 1
                if not cohort["pending_updates"]:
                    self.completed["worlds"] += 1
                    self.cohort = dict(world_id=None, condition=None, rows=0, losses=0,
                                       pending_updates=0, multiplier_ready=False)
            return result
        return self._run(update)

    def warmup_step(self, condition):
        return self._fit(condition, warmup=True)

    def training_step(self, condition):
        return self._fit(condition, warmup=False)

    def _update(self, mode):
        def reject():
            raise RuntimeError("legacy BC/offline/online schedule is unavailable; use explicit conditioned steps")
        return self._run(reject)

    def _finite_state(self):
        super()._finite_state()
        for module in (self.loss_critic, self.target_loss_critic):
            for value in module.state_dict().values():
                if value.device.type != "cpu" or value.dtype != torch.float32 or not torch.isfinite(value).all().item():
                    raise ValueError("invalid loss-critic numerical state")
        for state in self.loss_critic_optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor) and (value.device.type != "cpu" or value.dtype != torch.float32
                                                        or not torch.isfinite(value).all().item()):
                    raise ValueError("invalid loss-critic optimizer state")

    def state_dict(self):
        return copy.deepcopy(dict(format="patient-constrained-learner-v1", config=asdict(self.config),
            modules={name: getattr(self, name).state_dict() for name in self.module_names},
            optimizers={name: getattr(self, name + "_optimizer").state_dict() for name in self.optimizer_names},
            reference_replay=[asdict(row) for row in self.reference_replay],
            training_replay=[asdict(row) for row in self.training_replay],
            arm=self.arm, multipliers=self.multipliers, counts=self.counts, completed=self.completed,
            cohort=self.cohort, failure=self.failure, replay_rng=self.replay_rng.bit_generator.state,
            exploration_rng=self.exploration_rng.bit_generator.state, python_rng=self.python_rng.getstate(),
            torch_rng=self.torch_rng_state))

    @classmethod
    def from_snapshot(cls, snapshot: CapacitySnapshot, *, before_forward, before_optimizer):
        state = snapshot.state_dict()
        config = dict(state["config"])
        config["model"] = CapacityModelConfig(**config["model"])
        learner = cls(PatientConstrainedConfig(**config), model_seed=0, replay_seed=0,
                      adjacency=state["modules"]["actor"]["encoder.adjacency"],
                      before_forward=before_forward, before_optimizer=before_optimizer)
        learner.load_state_dict(state)
        learner._fresh_fork = learner.arm is None and learner.completed["warmup"] == learner.config.warmup_batches
        return learner

    def _check_optimizer_state(self, name, saved):
        optimizer = getattr(self, name + "_optimizer")
        expected = optimizer.state_dict()
        if (set(saved) != set(expected) or saved["param_groups"] != expected["param_groups"]):
            raise ValueError("optimizer fields/settings/parameter mapping mismatch")
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        ids = [i for group in expected["param_groups"] for i in group["params"]]
        if not set(saved["state"]).issubset(ids):
            raise ValueError("unknown optimizer state parameter")
        for index, parameter in zip(ids, parameters):
            if index not in saved["state"]:
                continue
            moments = saved["state"][index]
            if set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("invalid Adam moment fields")
            for key, shape in (("step", ()), ("exp_avg", parameter.shape), ("exp_avg_sq", parameter.shape)):
                value = moments[key]
                if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                        or value.device.type != "cpu" or value.shape != shape or not torch.isfinite(value).all().item()):
                    raise ValueError("invalid Adam moment shape/dtype/value")
            if ((moments["exp_avg_sq"] < 0).any().item() or moments["step"].item() < 0
                    or not moments["step"].item().is_integer()):
                raise ValueError("invalid Adam step/second moment")

    def load_state_dict(self, state):
        """Atomic full restore; failed instances and counter refunds are rejected."""
        self._guard()
        state = copy.deepcopy(state)
        if (set(state) != set(self.state_dict()) or state["format"] != "patient-constrained-learner-v1"
                or state["config"] != asdict(self.config) or set(state["counts"]) != set(self.counts)
                or any(type(v) is not int or v < self.counts[k] for k, v in state["counts"].items())):
            raise ValueError("snapshot contract mismatch or consumed counters refunded")
        c, count = self.config, state["counts"]
        caps = dict(warmup=c.warmup_batches, training=c.training_worlds * c.updates_per_world,
                    worlds=c.training_worlds, multiplier=c.training_worlds)
        if (set(state["completed"]) != set(caps)
                or any(type(v) is not int or not self.completed[k] <= v <= caps[k] for k, v in state["completed"].items())
                or state["arm"] not in (None, "constrained", "cost_only")
                or (self.arm is not None and state["arm"] != self.arm)
                or (state["failure"] is not None and not isinstance(state["failure"], str))):
            raise ValueError("invalid/refunded stage, arm or failure state")
        if (count["total_optimizer_steps"] != count["actor_optimizer_steps"] + count["critic_optimizer_steps"]
                or count["critic_optimizer_steps"] != count["reward_critic_optimizer_steps"] + count["loss_critic_optimizer_steps"]
                or count["optimizer_example_presentations"] != c.batch_size * count["total_optimizer_steps"]
                or count["warmup_steps"] < state["completed"]["warmup"]
                or count["training_steps"] < state["completed"]["training"]
                or count["scalar_multiplier_updates"] != state["completed"]["multiplier"]):
            raise ValueError("snapshot counters do not reconcile")
        done = state["completed"]
        attempted = count["warmup_steps"] + count["training_steps"]
        completed = done["warmup"] + done["training"]
        failed = state["failure"] is not None
        if (count["warmup_steps"] > c.warmup_batches
                or count["training_steps"] > caps["training"]
                or attempted - completed > int(failed)
                or not done["training"] <= count["actor_optimizer_steps"] <= count["training_steps"]
                or any(not completed <= count[name + "_optimizer_steps"] <= attempted
                       for name in ("reward_critic", "loss_critic"))
                or count["neural_forward_module_calls"] < 5 * done["warmup"] + 8 * done["training"]
                or (state["arm"] is None and (count["training_steps"] or count["training_rows"] or done["multiplier"]))
                or (state["arm"] is not None and done["warmup"] != c.warmup_batches)):
            raise ValueError("snapshot stage/call bounds do not reconcile")
        multipliers = state["multipliers"]
        if (len(multipliers) != c.conditions
                or any(not math.isfinite(x) or not c.multiplier_bounds[0] <= x <= c.multiplier_bounds[1] for x in multipliers)
                or (state["arm"] == "cost_only" and (any(multipliers) or count["scalar_multiplier_updates"]))):
            raise ValueError("invalid saved multipliers")
        candidate = copy.deepcopy(self, {id(self.before_forward): self.before_forward, id(self.before_optimizer): self.before_optimizer})
        if set(state["modules"]) != set(self.module_names) or set(state["optimizers"]) != set(self.optimizer_names):
            raise ValueError("incomplete dual-critic snapshot")
        for name, saved in state["modules"].items():
            module = getattr(candidate, name)
            expected = module.state_dict()
            if set(saved) != set(expected) or any(not isinstance(saved[k], torch.Tensor)
                    or saved[k].shape != expected[k].shape or saved[k].dtype != expected[k].dtype
                    or saved[k].device.type != "cpu" or not torch.isfinite(saved[k]).all().item() for k in saved):
                raise ValueError("module shape/dtype/value mismatch")
            if not torch.equal(saved["encoder.adjacency"], expected["encoder.adjacency"]):
                raise ValueError("fixed graph changed")
            module.load_state_dict(saved, strict=True)
        for name, saved in state["optimizers"].items():
            candidate._check_optimizer_state(name, saved)
            getattr(candidate, name + "_optimizer").load_state_dict(saved)
        for name, capacity, counter in (("reference_replay", c.reference_capacity, "reference_rows"),
                                        ("training_replay", c.training_capacity, "training_rows")):
            rows = [PatientLossTransition(**row) for row in state[name]]
            current = getattr(self, name)
            if len(rows) > capacity or len(rows) != count[counter] or rows[:len(current)] != current:
                raise ValueError("invalid/refunded replay")
            for row in rows:
                candidate._validate_row(row)
            setattr(candidate, name, rows)
        cohort = state["cohort"]
        if set(cohort) != set(self.cohort) or type(cohort["multiplier_ready"]) is not bool:
            raise ValueError("invalid cohort fields")
        for key in ("rows", "losses", "pending_updates"):
            _integer(cohort[key], "cohort " + key)
        if (cohort["rows"] > c.updates_per_world or cohort["pending_updates"] > c.updates_per_world
                or (cohort["world_id"] is None and any(cohort[k] for k in ("rows", "losses", "pending_updates")))
                or (cohort["pending_updates"] and cohort["rows"] != c.updates_per_world)):
            raise ValueError("invalid settled cohort boundary")
        if cohort["world_id"] is not None:
            candidate._condition(cohort["condition"])
            if not isinstance(cohort["world_id"], str) or not cohort["world_id"]:
                raise ValueError("invalid cohort identity")
            rows = [row for row in candidate.training_replay if row.world_id == cohort["world_id"]]
            if (len(rows) != cohort["rows"] or any(row.condition != cohort["condition"] for row in rows)
                    or sum(row.total_losses for row in rows) != cohort["losses"]):
                raise ValueError("cohort/replay mismatch")
        for name in ("arm", "multipliers", "counts", "completed", "cohort", "failure"):
            setattr(candidate, name, state[name])
        candidate.replay_rng.bit_generator.state = state["replay_rng"]
        candidate.exploration_rng.bit_generator.state = state["exploration_rng"]
        candidate.python_rng.setstate(state["python_rng"])
        generator = torch.Generator(device="cpu")
        generator.set_state(state["torch_rng"])
        candidate.torch_rng_state = generator.get_state()
        candidate._fresh_fork = False
        candidate._finite_state()
        self.__dict__.update(candidate.__dict__)
