"""Budget-hooked individual capacity DDPG steps, with no environment access.

The coordinator owns admission, receipts, final-tail closure, stage scheduling
and durable global counters. This module never runs a fit loop or loads files.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import io
import math
import random
from typing import Callable

import numpy as np

from src.models.capacity_ddpg import (
    CapacityActor, CapacityCritic, CapacityModelConfig, validate_features, validate_hours,
)
from src.rl.networks import require_torch, torch


@dataclass(frozen=True)
class CapacityLearnerConfig:
    model: CapacityModelConfig
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    gradient_norm_cap: float
    offline_actor_lr: float
    offline_critic_lr: float
    online_actor_lr: float
    online_critic_lr: float
    batch_size: int
    gamma: float
    target_polyak_tau: float
    reward_divisor: float
    bc_updates: int
    critic_warmup_updates: int
    offline_ddpg_pairs: int
    offline_replay_capacity: int
    online_replay_capacity: int
    online_offline_rows: int
    online_world_rows: int
    online_pairs_per_world: int
    exploration_sigma_hours: float

    def __post_init__(self):
        object.__setattr__(self, "betas", tuple(self.betas))
        if len(self.betas) != 2 or any(not 0 <= x < 1 for x in self.betas):
            raise ValueError("invalid Adam betas")
        positive = (self.epsilon, self.gradient_norm_cap, self.offline_actor_lr,
                    self.offline_critic_lr, self.online_actor_lr, self.online_critic_lr, self.reward_divisor)
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("finite positive optimizer/reward settings required")
        if (not 0 <= self.gamma <= 1 or not 0 < self.target_polyak_tau <= 1
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0
                or not math.isfinite(self.exploration_sigma_hours) or self.exploration_sigma_hours < 0):
            raise ValueError("invalid discount/Polyak/regularization/noise settings")
        for name in ("batch_size", "bc_updates", "critic_warmup_updates", "offline_ddpg_pairs",
                     "offline_replay_capacity", "online_replay_capacity", "online_offline_rows",
                     "online_world_rows", "online_pairs_per_world"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"positive native integer required: {name}")
        if self.online_offline_rows + self.online_world_rows != self.batch_size:
            raise ValueError("online mixture must exactly fill batch")

    @classmethod
    def from_proposal(cls, proposal, *, node_input_dim):
        p, s = proposal["learner"], proposal["synthetic_system"]["support"]
        expected = {"activation": "ReLU", "optimizer": "Adam", "neural_dtype": "float32",
                    "graph": "resource_ring_with_self_loops_symmetric_normalization",
                    "actor_raw_hours": "clip(2_plus_z,0,4)",
                    "critic": "executed_hours_as_node_features_mean_pool_32_ReLU_1",
                    "reward": "negative_full_cost_divided_by_100000",
                    "terminal_transition": "last_control_step_plus_all_settlement_cost_done_true",
                    "shared_actor_critic_encoder": False, "reward_clipping": False,
                    "reward_shaping": False, "online_sampling_with_replacement": True}
        if any(p[key] != value for key, value in expected.items()):
            raise ValueError("unsupported capacity learner numerical contract")
        if (p["exploration"]["kind"] != "iid_Gaussian_hours_then_clip_and_project"
                or p["exploration"]["annealing"] or s["site_hour_caps"] != [4, 4, 4, 4]
                or s["projection"] != "site_cap_then_radial_shared_scaling"):
            raise ValueError("unsupported action/noise contract")
        # These constants decode the explicit symbolic expressions checked above.
        model = CapacityModelConfig(node_input_dim, tuple(p["graph_widths"]), tuple(p["actor_head"]),
                                    tuple(s["site_hour_caps"]), s["shared_hour_budget"],
                                    2.0, p["feature_divisors"]["hours"])
        return cls(model, tuple(p["betas"]), p["epsilon"], p["weight_decay"], p["gradient_norm_cap"],
                   p["offline_actor_lr"], p["offline_critic_lr"], p["online_actor_lr"], p["online_critic_lr"],
                   p["batch_size"], p["gamma"], p["target_polyak_tau"], 100000.0,
                   p["bc_updates_per_block"], p["critic_warmup_updates_per_block"],
                   p["offline_ddpg_pairs_per_block"], p["offline_replay_rows_per_seal"],
                   p["online_replay_max_rows"], p["online_batch_offline_rows"],
                   p["online_batch_current_world_rows"], p["online_pairs_per_world"],
                   p["exploration"]["sigma_hours"])


@dataclass(frozen=True)
class CapacityTransition:
    state: tuple[tuple[float, ...], ...]
    executed_hours: tuple[float, ...]
    reward: float
    next_state: tuple[tuple[float, ...], ...]
    done: bool
    world_id: str
    full_cost: float
    settlement_cost: float | None

    def __post_init__(self):
        for name in ("state", "next_state"):
            object.__setattr__(self, name, tuple(tuple(float(x) for x in row) for row in getattr(self, name)))
        object.__setattr__(self, "executed_hours", tuple(float(x) for x in self.executed_hours))
        if (type(self.done) is not bool or not isinstance(self.world_id, str) or not self.world_id
                or not math.isfinite(self.reward) or not math.isfinite(self.full_cost) or self.full_cost < 0):
            raise ValueError("invalid typed transition/cost")
        if self.done:
            if (self.settlement_cost is None or not math.isfinite(self.settlement_cost)
                    or not 0 <= self.settlement_cost <= self.full_cost):
                raise ValueError("terminal row requires caller-supplied complete settlement cost")
        elif self.settlement_cost is not None:
            raise ValueError("settlement belongs only on the final control transition")

    @classmethod
    def from_cost(cls, *, state, executed_hours, next_state, done, world_id,
                  control_cost, settlement_cost, reward_divisor):
        if not math.isfinite(control_cost) or control_cost < 0 or not math.isfinite(reward_divisor) or reward_divisor <= 0:
            raise ValueError("invalid cost/reward divisor")
        total = control_cost + (0.0 if settlement_cost is None else settlement_cost)
        return cls(state, executed_hours, -total / reward_divisor, next_state, done,
                   world_id, total, settlement_cost)


@dataclass(frozen=True)
class CapacityBatch:
    states: object
    executed_hours: object
    rewards: object
    next_states: object
    dones: object
    sources: tuple[str, ...]
    indices: tuple[int, ...]


@dataclass(frozen=True)
class CapacitySnapshot:
    """Immutable, in-memory, weights-only-compatible snapshot, not a file loader."""
    payload: bytes
    sha256: str

    def state_dict(self):
        if hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise ValueError("snapshot digest mismatch")
        return torch.load(io.BytesIO(self.payload), map_location="cpu", weights_only=True)


def td_targets(rewards, dones, next_q, gamma):
    """Terminal transitions never bootstrap, including already-lumped tail cost."""
    return (rewards + gamma * (1 - dones) * next_q).detach()


class CapacityDDPGLearner:
    """A serial learner. Hooks take (module_or_optimizer_name, batch_size).

    Hooks are mandatory admission/accounting boundaries and run before dispatch.
    Hook failures consume local counters too. A failure latches this instance;
    restore cannot erase counters or clear the latch. Fork only unused seals.
    """

    def __init__(self, config: CapacityLearnerConfig, *, model_seed: int, replay_seed: int, adjacency=None,
                 before_forward: Callable, before_optimizer: Callable):
        require_torch()
        if any(type(x) is not int or x < 0 for x in (model_seed, replay_seed)):
            raise ValueError("native nonnegative integer seeds required")
        if not callable(before_forward) or not callable(before_optimizer):
            raise TypeError("explicit accounting hooks required")
        self.config, self.before_forward, self.before_optimizer = config, before_forward, before_optimizer
        # No global Python/NumPy/Torch RNG mutation survives fresh initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model_seed)
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                self.actor = CapacityActor(config.model, adjacency=adjacency)
                self.critic = CapacityCritic(config.model, adjacency=adjacency)
            finally:
                torch.set_default_dtype(previous_dtype)
            self.torch_rng_state = torch.get_rng_state().clone()
        self.target_actor, self.target_critic = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.target_actor.requires_grad_(False)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.offline_actor_lr,
                                               betas=config.betas, eps=config.epsilon,
                                               weight_decay=config.weight_decay, foreach=False)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.offline_critic_lr,
                                                betas=config.betas, eps=config.epsilon,
                                                weight_decay=config.weight_decay, foreach=False)
        self.replay_rng = np.random.default_rng(replay_seed)
        self.exploration_rng = np.random.default_rng(model_seed)
        self.python_rng = random.Random(model_seed)
        self.offline_replay, self.world_replay = [], []
        self.world_position, self.world_id, self.world_updates = 0, None, 0
        self.counts = dict.fromkeys(("neural_forward_module_calls", "actor_optimizer_steps",
                                    "critic_optimizer_steps", "total_optimizer_steps", "optimizer_example_presentations",
                                    "bc_steps", "warmup_steps", "offline_pairs", "online_pairs"), 0)
        self.completed = dict.fromkeys(("bc", "warmup", "offline", "online"), 0)
        self.failure = None

    @classmethod
    def from_proposal(cls, proposal, *, node_input_dim, **kwargs):
        return cls(CapacityLearnerConfig.from_proposal(proposal, node_input_dim=node_input_dim), **kwargs)

    def _guard(self):
        if self.failure is not None:
            raise RuntimeError(f"learner failure latched; no retry: {self.failure}")

    def _run(self, operation):
        self._guard()
        try:
            return operation()
        except BaseException as error:
            self.failure = f"{type(error).__name__}: {error}"
            raise

    def _forward(self, name, module, *args):
        self._guard()
        batch = args[0].shape[0]
        if not 1 <= batch <= self.config.batch_size:
            raise ValueError("forward batch outside declared contract")
        self.counts["neural_forward_module_calls"] += 1
        self.before_forward(name, batch)
        return module(*args)

    def _optimizer(self, name):
        self.counts[f"{name}_optimizer_steps"] += 1
        self.counts["total_optimizer_steps"] += 1
        self.counts["optimizer_example_presentations"] += self.config.batch_size
        self.before_optimizer(name, self.config.batch_size)
        getattr(self, f"{name}_optimizer").step()

    def _validate_row(self, row):
        if type(row) is not CapacityTransition:
            raise TypeError("CapacityTransition required, never an environment/info dictionary")
        c = self.config
        for state in (row.state, row.next_state):
            validate_features(torch.tensor([state], dtype=torch.float32), c.model)
        validate_hours(torch.tensor([row.executed_hours], dtype=torch.float32), c.model, batch_size=1)
        if not math.isclose(row.reward, -row.full_cost / c.reward_divisor, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("reward must contain exactly the caller-supplied full cost")
        if not np.isfinite(np.float32(row.reward)):
            raise ValueError("reward outside float32 support")

    def add_offline(self, row: CapacityTransition):
        def add():
            if self.world_id is not None or len(self.offline_replay) >= self.config.offline_replay_capacity:
                raise ValueError("offline replay sealed/full")
            self._validate_row(row)
            self.offline_replay.append(row)
        return self._run(add)

    def begin_world(self, world_id: str, *, replay_seed: int, exploration_seed: int):
        def begin():
            if self.world_id is not None or self.world_replay:
                raise ValueError("each evaluation world requires a fresh same-start fork")
            if not isinstance(world_id, str) or not world_id:
                raise ValueError("world identity required")
            if any(type(x) is not int or x < 0 for x in (replay_seed, exploration_seed)):
                raise ValueError("native nonnegative integer seeds required")
            self.world_id = world_id
            self.replay_rng = np.random.default_rng(replay_seed)
            self.exploration_rng = np.random.default_rng(exploration_seed)
        return self._run(begin)

    def add_world(self, row: CapacityTransition):
        def add():
            self._validate_row(row)
            if self.world_id is None or row.world_id != self.world_id:
                raise ValueError("online replay must belong to the current world")
            if len(self.world_replay) < self.config.online_replay_capacity:
                self.world_replay.append(row)
            else:
                self.world_replay[self.world_position] = row
            self.world_position = (self.world_position + 1) % self.config.online_replay_capacity
        return self._run(add)

    def sample_batch(self, *, online=False):
        def sample():
            c = self.config
            if not self.offline_replay or (online and (not self.world_replay or self.world_id is None)):
                raise ValueError("required offline/current-world replay is empty")
            # Fixed mixture with replacement, never substitute one source for another.
            offline_count = c.online_offline_rows if online else c.batch_size
            indices = self.replay_rng.choice(len(self.offline_replay), offline_count, replace=online).tolist()
            rows = [self.offline_replay[i] for i in indices]
            sources = ["offline"] * offline_count
            if online:
                world_indices = self.replay_rng.choice(len(self.world_replay), c.online_world_rows, replace=True).tolist()
                rows += [self.world_replay[i] for i in world_indices]
                indices += world_indices
                sources += ["world"] * c.online_world_rows
            def tensor(name):
                return torch.tensor([getattr(row, name) for row in rows], dtype=torch.float32, device="cpu")
            return CapacityBatch(tensor("state"), tensor("executed_hours"), tensor("reward").unsqueeze(-1),
                                 tensor("next_state"), tensor("done").unsqueeze(-1), tuple(sources), tuple(indices))
        return self._run(sample)

    def act(self, features, *, explore=False, noise_hours=None):
        """Return (raw_after_optional_noise, executed); one actor call, no tail logic.

        np.float32 [sites, F] inputs return two np.float32 [sites] arrays.
        CPU float32 [batch, sites, F] tensor inputs return two batched tensors.

        Provide matched physical-hour innovations with noise_hours. Otherwise an
        independent snapshotted exploration stream draws configured Gaussian noise.
        """
        def act():
            numpy_input = isinstance(features, np.ndarray)
            if numpy_input:
                if features.dtype != np.float32 or features.shape != (self.config.model.num_nodes, self.config.model.node_input_dim):
                    raise ValueError("np.float32 [sites, node_input_dim] public features required")
                inputs = torch.from_numpy(features.copy()).unsqueeze(0)
            else:
                inputs = features
            validate_features(inputs, self.config.model)
            with torch.no_grad():
                raw = self._forward("behavior_actor", self.actor, inputs)
                if noise_hours is not None and not explore:
                    raise ValueError("noise requires explore=True")
                if explore:
                    noise = (self.exploration_rng.normal(0, self.config.exploration_sigma_hours, tuple(raw.shape))
                             if noise_hours is None else noise_hours)
                    noise = torch.as_tensor(noise, dtype=torch.float32, device="cpu")
                    if numpy_input and noise.ndim == 1:
                        noise = noise.unsqueeze(0)
                    validate_hours(noise, self.config.model, batch_size=raw.shape[0], legal=False)
                    raw = torch.minimum((raw + noise).clamp_min(0), raw.new_tensor(self.config.model.site_hour_caps))
                executed = self.actor.project(raw)
                if numpy_input:
                    return raw[0].numpy().copy(), executed[0].numpy().copy()
                return raw.clone(), executed
        return self._run(act)

    def _critic_loss(self, batch):
        with torch.no_grad():
            raw = self._forward("target_actor", self.target_actor, batch.next_states)
            next_q = self._forward("target_critic", self.target_critic, batch.next_states, self.target_actor.project(raw))
            targets = td_targets(batch.rewards, batch.dones, next_q, self.config.gamma)
        q = self._forward("critic", self.critic, batch.states, batch.executed_hours)
        return torch.nn.functional.mse_loss(q, targets), targets

    def _actor_loss(self, batch):
        # Frozen critic weights still differentiate with respect to executed hours.
        self.critic.requires_grad_(False)
        try:
            raw = self._forward("actor", self.actor, batch.states)
            q = self._forward("actor_loss_critic", self.critic, batch.states, self.actor.project(raw))
            return -q.mean()
        finally:
            self.critic.requires_grad_(True)

    def _backward(self, loss, module):
        if not torch.isfinite(loss).item():
            raise ValueError("nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), self.config.gradient_norm_cap, error_if_nonfinite=True)
        return float(norm)

    def _polyak(self, *, actor=True, critic=True):
        with torch.no_grad():
            for enabled, source, target in ((actor, self.actor, self.target_actor),
                                            (critic, self.critic, self.target_critic)):
                if enabled:
                    for current, previous in zip(source.parameters(), target.parameters()):
                        previous.mul_(1 - self.config.target_polyak_tau).add_(current, alpha=self.config.target_polyak_tau)

    def _finite_state(self):
        def check(value):
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu" or (value.is_floating_point() and
                        (value.dtype != torch.float32 or not torch.isfinite(value).all().item())):
                    raise ValueError("nonfinite/non-CPU-float32 model or optimizer state")
            elif isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    check(item)
        for module in (self.actor, self.critic, self.target_actor, self.target_critic):
            check(module.state_dict())
        check(self.actor_optimizer.state_dict())
        check(self.critic_optimizer.state_dict())

    def _validate_optimizer_snapshot(self, name, saved):
        optimizer = getattr(self, name)
        reference = optimizer.state_dict()
        if set(saved) != {"state", "param_groups"} or len(saved["param_groups"]) != len(reference["param_groups"]):
            raise ValueError("optimizer snapshot fields/groups differ")
        component = name.removesuffix("_optimizer")
        allowed_lr = (getattr(self.config, f"offline_{component}_lr"),
                      getattr(self.config, f"online_{component}_lr"))
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        ids = []
        for group, expected in zip(saved["param_groups"], reference["param_groups"]):
            if (set(group) != set(expected) or group["lr"] not in allowed_lr
                    or any(group[k] != expected[k] for k in expected if k != "lr")):
                raise ValueError("optimizer hyperparameters/parameter mapping differ")
            ids.extend(group["params"])
        if not set(saved["state"]).issubset(ids):
            raise ValueError("unknown optimizer parameter state")
        for index, parameter in zip(ids, parameters):
            if index not in saved["state"]:
                continue
            moments = saved["state"][index]
            if set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("invalid Adam moment fields")
            for key, shape in (("step", ()), ("exp_avg", tuple(parameter.shape)),
                               ("exp_avg_sq", tuple(parameter.shape))):
                value = moments[key]
                if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                        or value.device.type != "cpu" or tuple(value.shape) != shape
                        or not torch.isfinite(value).all().item()):
                    raise ValueError("invalid Adam moment shape/dtype/value")
            if ((moments["exp_avg_sq"] < 0).any().item() or moments["step"].item() < 0
                    or not moments["step"].item().is_integer()):
                raise ValueError("invalid Adam second moment/step")

    def _update(self, mode):
        def update():
            c = self.config
            cap = {"bc": c.bc_updates, "warmup": c.critic_warmup_updates,
                   "offline": c.offline_ddpg_pairs, "online": c.online_pairs_per_world}[mode]
            consumed = self.world_updates if mode == "online" else self.completed[mode]
            if consumed >= cap:
                raise ValueError("declared update cap reached")
            if mode != "bc" and self.completed["bc"] != c.bc_updates:
                raise ValueError("complete fixed BC stage first")
            if mode in ("offline", "online") and self.completed["warmup"] != c.critic_warmup_updates:
                raise ValueError("complete fixed critic warmup first")
            if mode == "online" and self.completed["offline"] != c.offline_ddpg_pairs:
                raise ValueError("online requires the fixed-final offline seal")
            if mode != "online" and self.world_id is not None:
                raise ValueError("cannot train offline within an evaluation world")
            batch = self.sample_batch(online=mode == "online")
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            key = {"bc": "bc_steps", "warmup": "warmup_steps", "offline": "offline_pairs", "online": "online_pairs"}[mode]
            self.counts[key] += 1
            result = {"mode": mode, "sources": batch.sources, "indices": batch.indices}
            if mode == "bc":
                raw = self._forward("actor", self.actor, batch.states)
                loss = torch.nn.functional.mse_loss(self.actor.project(raw), batch.executed_hours)
                result["actor_grad_norm"] = self._backward(loss, self.actor)
                result["actor_loss"] = float(loss.detach())
                self._optimizer("actor")
                self.target_actor.load_state_dict(self.actor.state_dict())
            else:
                online = mode == "online"
                for name in ("actor", "critic"):
                    lr = getattr(c, f"{'online' if online else 'offline'}_{name}_lr")
                    for group in getattr(self, f"{name}_optimizer").param_groups:
                        group["lr"] = lr
                loss, targets = self._critic_loss(batch)
                result["critic_grad_norm"] = self._backward(loss, self.critic)
                result.update(critic_loss=float(loss.detach()), targets=targets[:, 0].tolist())
                if any(p.grad is not None for p in self.actor.parameters()):
                    raise RuntimeError("critic gradients leaked into actor")
                self._optimizer("critic")
                self.critic_optimizer.zero_grad(set_to_none=True)
                if mode != "warmup":
                    loss = self._actor_loss(batch)
                    result["actor_grad_norm"] = self._backward(loss, self.actor)
                    result["actor_loss"] = float(loss.detach())
                    if any(p.grad is not None for p in self.critic.parameters()):
                        raise RuntimeError("actor gradients leaked into critic")
                    self._optimizer("actor")
                self._polyak(actor=mode != "warmup")
            self._finite_state()
            self.completed[mode] += 1
            if mode == "online":
                self.world_updates += 1
            return result
        return self._run(update)

    def bc_step(self):
        return self._update("bc")

    def critic_warmup_step(self):
        return self._update("warmup")

    def offline_ddpg_step(self):
        return self._update("offline")

    def online_ddpg_step(self):
        return self._update("online")

    def state_dict(self):
        modules = {name: getattr(self, name).state_dict()
                   for name in ("actor", "critic", "target_actor", "target_critic")}
        return copy.deepcopy(dict(format="capacity-ddpg-v1", config=asdict(self.config), modules=modules,
                                  actor_optimizer=self.actor_optimizer.state_dict(),
                                  critic_optimizer=self.critic_optimizer.state_dict(),
                                  offline_replay=[asdict(row) for row in self.offline_replay],
                                  world_replay=[asdict(row) for row in self.world_replay],
                                  world_position=self.world_position, world_id=self.world_id,
                                  world_updates=self.world_updates, counts=self.counts, completed=self.completed,
                                  failure=self.failure, replay_rng=self.replay_rng.bit_generator.state,
                                  exploration_rng=self.exploration_rng.bit_generator.state,
                                  python_rng=self.python_rng.getstate(), torch_rng=self.torch_rng_state))

    def snapshot(self):
        stream = io.BytesIO()
        torch.save(self.state_dict(), stream)
        payload = stream.getvalue()
        return CapacitySnapshot(payload, hashlib.sha256(payload).hexdigest())

    @classmethod
    def from_snapshot(cls, snapshot: CapacitySnapshot, *, before_forward, before_optimizer):
        state = snapshot.state_dict()
        config = dict(state["config"])
        config["model"] = CapacityModelConfig(**config["model"])
        learner = cls(CapacityLearnerConfig(**config), model_seed=0, replay_seed=0,
                      before_forward=before_forward, before_optimizer=before_optimizer)
        learner.load_state_dict(state)
        return learner

    def load_state_dict(self, state):
        """Atomic restoration; consumption and failure can never be rolled back."""
        self._guard()
        state = copy.deepcopy(state)
        if (set(state) != set(self.state_dict()) or state["format"] != "capacity-ddpg-v1"
                or state["config"] != asdict(self.config)
                or set(state["counts"]) != set(self.counts)
                or any(type(v) is not int or v < self.counts[k] for k, v in state["counts"].items())):
            raise ValueError("snapshot contract mismatch or consumed counters would be refunded")
        caps = {"bc": self.config.bc_updates, "warmup": self.config.critic_warmup_updates,
                "offline": self.config.offline_ddpg_pairs, "online": self.config.online_pairs_per_world}
        if (set(state["completed"]) != set(self.completed)
                or any(type(v) is not int or not self.completed[k] <= v <= caps[k]
                       for k, v in state["completed"].items())
                or type(state["world_updates"]) is not int
                or not self.world_updates <= state["world_updates"] <= self.config.online_pairs_per_world
                or state["world_updates"] != state["completed"]["online"]
                or (state["failure"] is not None and not isinstance(state["failure"], str))
                or (state["world_id"] is not None and
                    (not isinstance(state["world_id"], str) or not state["world_id"]))):
            raise ValueError("invalid/refunded stage or world snapshot")
        counts = state["counts"]
        if (counts["total_optimizer_steps"] != counts["actor_optimizer_steps"] + counts["critic_optimizer_steps"]
                or counts["optimizer_example_presentations"] != counts["total_optimizer_steps"] * self.config.batch_size):
            raise ValueError("optimizer counters do not reconcile")
        # A bound hook may own a non-copyable coordinator ledger; never clone it.
        candidate = copy.deepcopy(self, {id(self.before_forward): self.before_forward,
                                         id(self.before_optimizer): self.before_optimizer})
        for name, saved in state["modules"].items():
            if name not in ("actor", "critic", "target_actor", "target_critic"):
                raise ValueError("unknown module")
            reference = getattr(candidate, name).state_dict()
            if set(saved) != set(reference) or any(
                    saved[k].shape != reference[k].shape or saved[k].dtype != reference[k].dtype for k in saved):
                raise ValueError("module shape/dtype mismatch")
            if not torch.allclose(saved["encoder.adjacency"], reference["encoder.adjacency"], atol=1e-7, rtol=1e-6):
                raise ValueError("snapshot changed fixed ring adjacency")
            getattr(candidate, name).load_state_dict(saved, strict=True)
        if set(state["modules"]) != {"actor", "critic", "target_actor", "target_critic"}:
            raise ValueError("incomplete model snapshot")
        for name in ("actor_optimizer", "critic_optimizer"):
            candidate._validate_optimizer_snapshot(name, state[name])
            getattr(candidate, name).load_state_dict(state[name])
        for name, capacity in (("offline_replay", self.config.offline_replay_capacity),
                               ("world_replay", self.config.online_replay_capacity)):
            rows = [CapacityTransition(**row) for row in state[name]]
            if len(rows) > capacity:
                raise ValueError("oversized replay")
            for row in rows:
                candidate._validate_row(row)
                if name == "world_replay" and row.world_id != state["world_id"]:
                    raise ValueError("cross-world replay")
            setattr(candidate, name, rows)
        position, size = state["world_position"], len(candidate.world_replay)
        if (type(position) is not int or not 0 <= position < self.config.online_replay_capacity
                or (size < self.config.online_replay_capacity and position != size)):
            raise ValueError("invalid replay ring position")
        for name in ("counts", "completed", "failure", "world_id", "world_position", "world_updates"):
            setattr(candidate, name, state[name])
        candidate.replay_rng.bit_generator.state = state["replay_rng"]
        candidate.exploration_rng.bit_generator.state = state["exploration_rng"]
        candidate.python_rng.setstate(state["python_rng"])
        generator = torch.Generator(device="cpu")
        generator.set_state(state["torch_rng"])
        candidate.torch_rng_state = generator.get_state()
        candidate._finite_state()
        self.__dict__.update(candidate.__dict__)
