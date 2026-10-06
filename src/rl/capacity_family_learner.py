"""CPU support-hour family learners, with no environment or file access.

All actors share a two-layer width-32 mean network and the same box transform
``2 * (1 + tanh(z))`` followed by the existing radial sum-8 projection. Critics
see these feasible *requests*, not demand-dependent consumed hours. A row's
``executed_hours`` is accepted as an alias only for that same action meaning.

Rows must contain state, next_state, action (or executed_hours), reward, done,
and metadata. Supply full_cost on every row and settlement_cost on the terminal
row, either at top level or in metadata. Full cost includes settlement and reward
must equal -full_cost / 1e6. Exactly one complete, contiguous episode is admitted
at a time; no inferred four-step stitching or additional inference is performed.

PPO stores Gaussian *latent* log probabilities before tanh and radial projection;
its likelihood ratio is for the recorded latent policy, not the many-to-one
projected density. SAC uses the corrected density of the box proposal, also NOT
the projected density. Its fixed temperature is 0.02 in normalized reward units.

Hooks receive (actor|critic|value, batch_size), before each top-level neural call
or Adam step, including target and twin-critic calls. Counts are charged before
admission; completed steps and receipts are separate. Any failure latches the
instance. Restore is in-memory, into an unused learner with identical config;
the coordinator owns durable budgets and weight/state sealing.
"""

from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

from src.baselines.ppo import _compute_gae
from src.models.capacity_ddpg import (
    CapacityActor, CapacityCritic, CapacityModelConfig, validate_features,
)
from src.rl.networks import nn, require_torch, torch


METHODS = ("ddpg", "td3", "sac", "ppo")
NAMES = ("actor", "critic", "value")


def _integer(value, name, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a native integer in [{minimum}, {maximum}]")
    return value


def _number(value, name, minimum=-math.inf):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be finite numeric")
    value = float(value)
    if not math.isfinite(value) or value < minimum or abs(value) > np.finfo(np.float32).max:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _features(value, dim):
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != (4, dim) or not np.isfinite(array).all():
        raise ValueError("finite float32 [4, feature_dim] features required")
    return array.copy()


def _vector(value, name, *, feasible=False):
    array = np.asarray(value)
    if array.shape != (4,) or array.dtype.kind not in "fi" or not np.isfinite(array).all():
        raise ValueError(f"finite numeric {name} [4] required")
    if np.max(np.abs(array)) > np.finfo(np.float32).max:
        raise ValueError(f"{name} outside float32 support")
    if feasible and (np.any(array < 0) or np.any(array > 4 + 4e-6) or array.sum() > 8 + 4e-6):
        raise ValueError(f"{name} violates feasible hours")
    return array.astype(np.float32, copy=True)


def _state_hash(features):
    return hashlib.sha256(features.tobytes(order="C")).hexdigest()


def _normal_log_prob(latent, mean, log_std):
    return (-0.5 * ((latent - mean) * torch.exp(-log_std)).square()
            - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)


if torch is not None:
    class _MeanActor(CapacityActor):
        def __init__(self, config, stochastic):
            super().__init__(config)
            # Added after the common mean initialization, without drawing RNG.
            if stochastic:
                self.log_std = nn.Parameter(torch.full((4,), math.log(0.3), dtype=torch.float32))

        def forward(self, features):
            validate_features(features, self.config)
            return self.head(self.encoder(features)).squeeze(-1)


    class _Value(CapacityActor):
        def forward(self, features):
            validate_features(features, self.config)
            return self.head(self.encoder(features).mean(1))


class FamilyLearner:
    """Fresh scratch learners; no BC, no pretrained model or scientific I/O."""

    def __init__(self, method, *, feature_dim=31, adjacency, model_seed, sampler_seed,
                 lr=3e-4, before_forward, before_optimizer, replay_capacity=8192):
        require_torch()
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        _integer(feature_dim, "feature_dim", 1)
        _integer(model_seed, "model_seed")
        _integer(sampler_seed, "sampler_seed")
        _integer(replay_capacity, "replay_capacity", 64, 8192)
        lr = _number(lr, "lr", 0)
        if lr == 0 or not callable(before_forward) or not callable(before_optimizer):
            raise ValueError("positive lr and explicit accounting hooks required")
        adj = np.asarray(adjacency, dtype=np.float32)
        ring = np.asarray([[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]],
                          dtype=np.float32) / 3
        if adj.shape != (4, 4) or not np.isfinite(adj).all() or not (
                np.allclose(adj, ring, rtol=0, atol=1e-7) or np.array_equal(adj, np.eye(4))):
            raise ValueError("adjacency must be the normalized four-site ring or self-only identity")
        self.adjacency = torch.from_numpy(adj.copy())
        self.method, self.feature_dim = method, feature_dim
        self.before_forward, self.before_optimizer = before_forward, before_optimizer
        self.replay_capacity = replay_capacity
        self.config = dict(method=method, feature_dim=feature_dim, model_seed=model_seed,
                           sampler_seed=sampler_seed, lr=lr, replay_capacity=replay_capacity,
                           gamma=1.0, tau=0.005, gae_lambda=0.95, ppo_clip=0.2,
                           alpha=0.02, exploration_hours=0.3, target_noise_hours=0.2,
                           target_noise_clip_hours=0.5, policy_delay=2, gradient_norm_cap=5.0,
                           reward_divisor=1e6, critic_action="projected_request_hours",
                           device="cpu", dtype="float32", graph_widths=(32, 32), actor_head=(32, 1),
                           site_hour_caps=(4.,) * 4, shared_hour_budget=8., hours_divisor=4.,
                           actor_transform="2*(1+tanh(latent));radial_sum8",
                           initial_log_std=math.log(0.3), log_std_bounds=(-5., 1.),
                           ppo_value_loss_coefficient=0.5, ppo_entropy_coefficient=0.,
                           sampling_with_replacement=True)
        model_config = CapacityModelConfig(feature_dim, (32, 32), (32, 1), (4.,) * 4, 8., 2., 4.)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model_seed)
            dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                self.actor = _MeanActor(model_config, method in ("sac", "ppo"))
                self.modules = {"actor": self.actor}
                if method == "ppo":
                    self.modules["value"] = _Value(model_config)
                else:
                    self.modules["critic1"] = CapacityCritic(model_config)
                    if method in ("td3", "sac"):
                        self.modules["critic2"] = CapacityCritic(model_config)
            finally:
                torch.set_default_dtype(dtype)
        for module in self.modules.values():
            module.encoder.adjacency.copy_(self.adjacency)
        self.optimizers = {name: torch.optim.Adam(module.parameters(), lr=lr, foreach=False)
                           for name, module in self.modules.items()}
        for name, module in list(self.modules.items()):
            if name.startswith("critic") or (name == "actor" and method in ("ddpg", "td3")):
                self.modules["target_" + name] = copy.deepcopy(module).requires_grad_(False)
        self.sampler = np.random.default_rng(sampler_seed)
        self.noise = torch.Generator(device="cpu").manual_seed(sampler_seed)
        self.replay, self.pending = [], None
        self.counts = {key: dict.fromkeys(NAMES, 0) for key in (
            "forward_calls", "forward_examples", "optimizer_attempts", "optimizer_steps", "optimizer_examples")}
        self.counts.update(updates=0, episodes=0, act_calls=0)
        self.optimizer_module_steps = dict.fromkeys(self.optimizers, 0)
        self.failure = None

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

    def _forward(self, module_name, features, *args):
        self._guard()
        batch = features.shape[0]
        _integer(batch, "forward batch", 1, 64)
        category = "actor" if "actor" in module_name else "value" if module_name == "value" else "critic"
        self.counts["forward_calls"][category] += 1
        self.counts["forward_examples"][category] += batch
        self.before_forward(category, batch)
        output = self.modules[module_name](features, *args)
        if not torch.isfinite(output).all().item():
            raise ValueError("nonfinite neural output")
        return output

    @staticmethod
    def _proposal(mean):
        return 2.0 * (1.0 + torch.tanh(mean))

    def _policy(self, features, *, stochastic=False, target=False):
        mean = self._forward("target_actor" if target else "actor", features)
        log_std = self.actor.log_std.clamp(-5., 1.).expand_as(mean) if self.method in ("sac", "ppo") else None
        latent = mean
        if stochastic:
            latent = mean + log_std.exp() * torch.randn(mean.shape, generator=self.noise, dtype=torch.float32)
        proposal = self._proposal(latent)
        log_prob = None
        if log_std is not None:
            log_prob = _normal_log_prob(latent, mean, log_std)
            if self.method == "sac":
                # Stable log |d(2*(1+tanh(z)))/dz|, including the hours scale.
                log_jac = math.log(2.) + 2 * (math.log(2.) - latent - torch.nn.functional.softplus(-2 * latent))
                log_prob = log_prob - log_jac.sum(-1, keepdim=True)
            if not torch.isfinite(log_prob).all().item():
                raise ValueError("nonfinite stochastic log probability")
        return proposal, log_prob, latent, mean, log_std

    def act(self, features, explore=False):
        def operation():
            if type(explore) is not bool:
                raise ValueError("explore must be bool")
            array = _features(features, self.feature_dim)
            inputs = torch.from_numpy(array).unsqueeze(0)
            with torch.no_grad():
                proposal, log_prob, latent, mean, log_std = self._policy(
                    inputs, stochastic=explore and self.method in ("sac", "ppo"))
                if explore and self.method in ("ddpg", "td3"):
                    proposal = (proposal + self.config["exploration_hours"] * torch.randn(
                        proposal.shape, generator=self.noise, dtype=torch.float32)).clamp(0., 4.)
                hours = self.actor.project(proposal)
                metadata = dict(method=self.method, explore=explore, state_sha256=_state_hash(array),
                                policy_version=self.counts["optimizer_steps"]["actor"],
                                action_semantics="projected_request_hours",
                                proposal_hours=proposal[0].tolist(), requested_hours=hours[0].tolist(),
                                learner_config=dict(self.config), parameter_counts=self.parameter_counts())
                if log_prob is not None:
                    metadata.update(latent=latent[0].tolist(), mean=mean[0].tolist(),
                                    log_std=log_std[0].tolist(), log_prob=float(log_prob[0, 0]),
                                    likelihood="latent_gaussian" if self.method == "ppo" else "box_proposal")
                if self.method == "ppo" and explore:
                    metadata["value"] = float(self._forward("value", inputs)[0, 0])
            self.counts["act_calls"] += 1
            return hours[0].numpy().copy(), metadata
        return self._run(operation)

    def _row(self, row, *, on_policy=False):
        if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
            raise ValueError("transition dict and metadata required")
        metadata = copy.deepcopy(row["metadata"])
        if "method" in metadata and metadata["method"] != self.method:
            raise ValueError("transition metadata belongs to another learner method")
        state = _features(row["state"], self.feature_dim)
        next_state = _features(row["next_state"], self.feature_dim)
        done = row["done"]
        if type(done) is not bool:
            raise ValueError("done must be a bool")
        def cost_field(name):
            if name in row and name in metadata and row[name] != metadata[name]:
                raise ValueError(f"conflicting {name}")
            return row.get(name, metadata.get(name))
        full_cost = _number(cost_field("full_cost"), "full_cost", 0)
        reward = _number(row["reward"], "reward")
        if not math.isclose(reward, -full_cost / 1e6, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("reward must be negative full_cost / 1e6, including settlement")
        settlement = cost_field("settlement_cost")
        if done:
            settlement = _number(settlement, "terminal settlement_cost", 0)
            if settlement > full_cost:
                raise ValueError("terminal settlement_cost exceeds full_cost")
        elif settlement is not None:
            raise ValueError("settlement_cost belongs only on the terminal row")
        control = cost_field("control_cost")
        if control is not None and not math.isclose(
                full_cost, _number(control, "control_cost", 0) + (settlement or 0), rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("full_cost must include control plus settlement")
        action = _vector(row.get("action", row.get("executed_hours")), "action", feasible=True)
        if "action" in row and "executed_hours" in row and not np.allclose(
                action, _vector(row["executed_hours"], "executed_hours", feasible=True), rtol=0, atol=2e-6):
            raise ValueError("action and executed_hours must mean the same projected request")
        if "requested_hours" in metadata and not np.allclose(
                action, _vector(metadata["requested_hours"], "requested_hours", feasible=True), rtol=0, atol=2e-6):
            raise ValueError("replay action differs from policy projected request")
        if "state_sha256" in metadata and metadata["state_sha256"] != _state_hash(state):
            raise ValueError("metadata belongs to a different state")
        if "action_semantics" in metadata and metadata["action_semantics"] != "projected_request_hours":
            raise ValueError("critic requires projected request action semantics")
        if on_policy:
            if (metadata.get("method") != "ppo" or metadata.get("explore") is not True
                    or type(metadata.get("policy_version")) is not int
                    or metadata["policy_version"] != self.counts["optimizer_steps"]["actor"]
                    or metadata.get("likelihood") != "latent_gaussian"
                    or metadata.get("state_sha256") != _state_hash(state)):
                raise ValueError("PPO requires current-policy exploratory action metadata")
            latent = torch.from_numpy(_vector(metadata.get("latent"), "latent")).unsqueeze(0)
            mean = torch.from_numpy(_vector(metadata.get("mean"), "mean")).unsqueeze(0)
            log_std = torch.from_numpy(_vector(metadata.get("log_std"), "log_std")).unsqueeze(0)
            if not torch.equal(log_std[0], self.actor.log_std.detach().clamp(-5., 1.)):
                raise ValueError("PPO log_std differs from current policy")
            expected_lp = float(_normal_log_prob(latent, mean, log_std)[0, 0])
            if not math.isclose(_number(metadata.get("log_prob"), "log_prob"), expected_lp,
                                rel_tol=1e-6, abs_tol=1e-5):
                raise ValueError("PPO log_prob does not match recorded latent")
            projected = self.actor.project(self._proposal(latent))[0].numpy()
            if not np.allclose(projected, action, rtol=0, atol=2e-6):
                raise ValueError("PPO latent does not map to recorded action")
            _number(metadata.get("value"), "PPO value")
        return dict(state=state, next_state=next_state, action=action, reward=reward,
                    done=done, full_cost=full_cost, settlement_cost=settlement, metadata=metadata)

    def observe_episode(self, rows):
        def operation():
            if self.pending is not None:
                raise ValueError("fit the pending episode before admitting another")
            if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= 48:
                raise ValueError("one complete episode of 1..48 control rows required")
            admitted = [self._row(row, on_policy=self.method == "ppo") for row in rows]
            if not admitted[-1]["done"] or any(row["done"] for row in admitted[:-1]):
                raise ValueError("exactly the final row must be done with settlement")
            if any(not np.array_equal(left["next_state"], right["state"])
                   for left, right in zip(admitted, admitted[1:])):
                raise ValueError("episode transitions must be contiguous; no inferred stitching")
            self.pending = admitted
            if self.method != "ppo":
                self.replay = (self.replay + copy.deepcopy(admitted))[-self.replay_capacity:]
            self.counts["episodes"] += 1
        return self._run(operation)

    def _step(self, name, loss, batch):
        if loss.numel() != 1 or not torch.isfinite(loss).item():
            raise ValueError("nonfinite optimizer loss")
        module, optimizer = self.modules[name], self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), self.config["gradient_norm_cap"],
                                            error_if_nonfinite=True)
        category = name if name in ("actor", "value") else "critic"
        self.counts["optimizer_attempts"][category] += 1
        self.counts["optimizer_examples"][category] += batch
        self.before_optimizer(category, batch)
        optimizer.step()
        self.counts["optimizer_steps"][category] += 1
        self.optimizer_module_steps[name] += 1
        if any(not torch.isfinite(p).all().item() for p in module.parameters()):
            raise ValueError("nonfinite optimizer parameters")
        return dict(loss=float(loss.detach()), grad_norm=float(norm), optimizer_completed=True)

    def _polyak(self, names):
        with torch.no_grad():
            for name in names:
                for source, target in zip(self.modules[name].parameters(), self.modules["target_" + name].parameters()):
                    target.lerp_(source, self.config["tau"])

    def _off_policy_update(self, states, actions, rewards, next_states, dones, batch):
        critics = ["critic1"] + (["critic2"] if self.method in ("td3", "sac") else [])
        with torch.no_grad():
            proposal, log_prob, _, _, _ = self._policy(
                next_states, stochastic=self.method == "sac", target=self.method != "sac")
            if self.method == "td3":
                noise = (torch.randn(proposal.shape, generator=self.noise, dtype=torch.float32)
                         * self.config["target_noise_hours"]).clamp(
                             -self.config["target_noise_clip_hours"], self.config["target_noise_clip_hours"])
                proposal = (proposal + noise).clamp(0., 4.)
            hours = self.actor.project(proposal)
            values = [self._forward("target_" + name, next_states, hours) for name in critics]
            next_q = torch.stack(values).min(0).values
            if self.method == "sac":
                next_q -= self.config["alpha"] * log_prob
            targets = rewards + (1. - dones) * next_q
        receipt = dict(target_mean=float(targets.mean()), losses={})
        for name in critics:
            loss = torch.nn.functional.mse_loss(self._forward(name, states, actions), targets)
            receipt["losses"][name] = self._step(name, loss, batch)
        update_actor = self.method != "td3" or (self.counts["updates"] + 1) % 2 == 0
        if update_actor:
            for name in critics:
                self.modules[name].requires_grad_(False)
            try:
                proposal, log_prob, _, _, _ = self._policy(states, stochastic=self.method == "sac")
                hours = self.actor.project(proposal)
                q = self._forward("critic1", states, hours)
                if self.method == "sac":
                    q = torch.minimum(q, self._forward("critic2", states, hours))
                    loss = (self.config["alpha"] * log_prob - q).mean()
                else:
                    loss = -q.mean()
                receipt["losses"]["actor"] = self._step("actor", loss, batch)
            finally:
                for name in critics:
                    self.modules[name].requires_grad_(True)
        if self.method != "td3" or update_actor:
            self._polyak(critics + (["actor"] if self.method != "sac" else []))
        return receipt

    def fit_episode(self, updates=32, batch_size=64):
        def operation():
            _integer(updates, "updates", 1, 32)
            _integer(batch_size, "batch_size", 1, 64)
            if self.pending is None:
                raise ValueError("a newly observed complete episode is required")
            source = self.pending if self.method == "ppo" else self.replay
            if self.method == "ppo":
                advantages, returns = _compute_gae(
                    rewards=np.asarray([row["reward"] for row in source], dtype=np.float32),
                    dones=np.asarray([row["done"] for row in source], dtype=np.float32),
                    values=np.asarray([row["metadata"]["value"] for row in source], dtype=np.float32),
                    last_value=0., gamma=1., gae_lambda=0.95)
                if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
                    raise ValueError("nonfinite PPO GAE/returns")
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            receipts = []
            for _ in range(updates):
                before = copy.deepcopy(self.counts)
                indices = self.sampler.integers(0, len(source), size=batch_size)
                rows = [source[index] for index in indices]
                def tensor(key):
                    return torch.as_tensor(np.stack([row[key] for row in rows]), dtype=torch.float32)
                states = tensor("state")
                if self.method == "ppo":
                    latent = torch.tensor([row["metadata"]["latent"] for row in rows], dtype=torch.float32)
                    old_lp = torch.tensor([[row["metadata"]["log_prob"]] for row in rows], dtype=torch.float32)
                    mean = self._forward("actor", states)
                    log_std = self.actor.log_std.clamp(-5., 1.).expand_as(mean)
                    log_prob = _normal_log_prob(latent, mean, log_std)
                    ratio = torch.exp(log_prob - old_lp)
                    advantage = torch.from_numpy(advantages[indices]).unsqueeze(-1)
                    clipped = ratio.clamp(1. - self.config["ppo_clip"], 1. + self.config["ppo_clip"])
                    loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                    actor_receipt = self._step("actor", loss, batch_size)
                    prediction = self._forward("value", states)
                    value_loss = 0.5 * torch.nn.functional.mse_loss(
                        prediction, torch.from_numpy(returns[indices]).unsqueeze(-1))
                    receipt = dict(losses=dict(actor=actor_receipt, value=self._step("value", value_loss, batch_size)),
                                   ratio_mean=float(ratio.detach().mean()),
                                   target_mean=float(np.mean(returns[indices])))
                else:
                    receipt = self._off_policy_update(states, tensor("action"), tensor("reward").unsqueeze(-1),
                                                      tensor("next_state"), tensor("done").unsqueeze(-1), batch_size)
                self.counts["updates"] += 1
                receipt.update(method=self.method, update=self.counts["updates"], batch_size=batch_size,
                               indices=indices.tolist(), source="current_episode" if self.method == "ppo" else "replay",
                               source_rows=len(source), accounting={key: {
                                   name: self.counts[key][name] - before[key][name] for name in NAMES}
                                   for key in ("forward_calls", "forward_examples", "optimizer_attempts",
                                               "optimizer_steps", "optimizer_examples")})
                receipts.append(receipt)
            self.pending = None
            return receipts
        return self._run(operation)

    def parameter_counts(self):
        counts = {name: sum(p.numel() for p in module.parameters()) for name, module in self.modules.items()}
        return dict(modules=counts, trainable=sum(counts[name] for name in self.optimizers),
                    targets=sum(count for name, count in counts.items() if name.startswith("target_")),
                    actor_mean=counts["actor"] - (4 if self.method in ("sac", "ppo") else 0),
                    stochastic_log_std=4 if self.method in ("sac", "ppo") else 0)

    def accounting(self, updates=32, batch_size=64, *, act_calls=48, explore=True):
        """Deterministic successful-call forecast, NOT evidence of work executed.

        PPO training acts cost actor+value; evaluation acts cost actor only. Fit
        forecasts start at the present TD3 update index, including odd boundaries.
        """
        _integer(updates, "updates", 0, 32)
        _integer(batch_size, "batch_size", 1, 64)
        _integer(act_calls, "act_calls")
        if type(explore) is not bool:
            raise ValueError("explore must be bool")
        if self.method == "ddpg":
            forwards, steps = (2 * updates, 3 * updates, 0), (updates, updates, 0)
        elif self.method == "td3":
            actors = (self.counts["updates"] + updates) // 2 - self.counts["updates"] // 2
            forwards, steps = (updates + actors, 4 * updates + actors, 0), (actors, 2 * updates, 0)
        elif self.method == "sac":
            forwards, steps = (2 * updates, 6 * updates, 0), (updates, 2 * updates, 0)
        else:
            forwards, steps = (updates, 0, updates), (updates, 0, updates)
        fits, opts = dict(zip(NAMES, forwards)), dict(zip(NAMES, steps))
        acts = dict(actor=act_calls, critic=0, value=act_calls if self.method == "ppo" and explore else 0)
        return dict(fit_forward_calls=fits, act_forward_calls=acts,
                    total_forward_calls={name: fits[name] + acts[name] for name in NAMES},
                    forward_examples={name: fits[name] * batch_size + acts[name] for name in NAMES},
                    optimizer_steps=opts, optimizer_examples={name: opts[name] * batch_size for name in NAMES},
                    parameter_counts=self.parameter_counts())

    def state_dict(self):
        return copy.deepcopy(dict(schema=1, config=self.config, adjacency=self.adjacency,
                                  modules={name: module.state_dict() for name, module in self.modules.items()},
                                  optimizers={name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
                                  sampler_rng=self.sampler.bit_generator.state, noise_rng=self.noise.get_state(),
                                  counts=self.counts, optimizer_module_steps=self.optimizer_module_steps,
                                  replay=self.replay, pending=self.pending, failure=self.failure))

    def load_state_dict(self, state):
        """Validate into temporary objects; no forwards, RNG draws or file loads."""
        self._guard()
        if self.counts != {**{key: dict.fromkeys(NAMES, 0) for key in (
                "forward_calls", "forward_examples", "optimizer_attempts", "optimizer_steps", "optimizer_examples")},
                "updates": 0, "episodes": 0, "act_calls": 0} or self.pending is not None or self.replay:
            raise ValueError("restore requires an unused learner; counters cannot be refunded")
        saved = copy.deepcopy(state)
        if set(saved) != set(self.state_dict()) or saved["schema"] != 1 or saved["config"] != self.config:
            raise ValueError("checkpoint schema/config mismatch")
        if not isinstance(saved["adjacency"], torch.Tensor) or not torch.equal(saved["adjacency"], self.adjacency):
            raise ValueError("checkpoint adjacency mismatch; cannot change self-only")
        if set(saved["modules"]) != set(self.modules) or set(saved["optimizers"]) != set(self.optimizers):
            raise ValueError("checkpoint module/optimizer names mismatch")
        counts = saved["counts"]
        if set(counts) != set(self.counts):
            raise ValueError("checkpoint counter schema mismatch")
        for key, expected in self.counts.items():
            if isinstance(expected, dict):
                if not isinstance(counts[key], dict) or set(counts[key]) != set(NAMES):
                    raise ValueError("checkpoint category counts mismatch")
                for value in counts[key].values():
                    _integer(value, key)
            else:
                _integer(counts[key], key)
        for name in NAMES:
            if counts["optimizer_steps"][name] > counts["optimizer_attempts"][name]:
                raise ValueError("completed optimizers exceed attempts")
        if set(saved["optimizer_module_steps"]) != set(self.optimizers):
            raise ValueError("optimizer module counters mismatch")
        for value in saved["optimizer_module_steps"].values():
            _integer(value, "optimizer module steps")
        for category in NAMES:
            total = sum(value for name, value in saved["optimizer_module_steps"].items()
                        if (name if name in ("actor", "value") else "critic") == category)
            if total != counts["optimizer_steps"][category]:
                raise ValueError("optimizer category/module steps disagree")
        if saved["failure"] is not None and not isinstance(saved["failure"], str):
            raise ValueError("invalid failure latch")
        modules = copy.deepcopy(self.modules)
        for name, module in modules.items():
            model_state = saved["modules"][name]
            expected = module.state_dict()
            if set(model_state) != set(expected):
                raise ValueError("checkpoint model keys mismatch")
            for key, value in model_state.items():
                if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                        or value.dtype != expected[key].dtype or value.shape != expected[key].shape
                        or not torch.isfinite(value).all().item()):
                    raise ValueError("invalid model tensor dtype/shape/finite/device")
                if key.endswith("adjacency") and not torch.equal(value, self.adjacency):
                    raise ValueError("checkpoint model adjacency mismatch")
            module.load_state_dict(model_state)
        optimizers = {}
        for name in self.optimizers:
            optimizer = torch.optim.Adam(modules[name].parameters(), lr=self.config["lr"], foreach=False)
            data = saved["optimizers"][name]
            if set(data) != {"state", "param_groups"} or data["param_groups"] != optimizer.state_dict()["param_groups"]:
                raise ValueError("optimizer settings mismatch")
            ids = data["param_groups"][0]["params"]
            steps = saved["optimizer_module_steps"][name]
            if set(data["state"]) != (set(ids) if steps else set()):
                raise ValueError("missing/unexpected optimizer moments")
            for pid, parameter in zip(ids, modules[name].parameters()):
                if not steps:
                    continue
                moments = data["state"][pid]
                if set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("invalid Adam moment keys")
                for field, value in moments.items():
                    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                            or value.dtype != torch.float32 or not torch.isfinite(value).all().item()
                            or value.shape != (() if field == "step" else parameter.shape)):
                        raise ValueError("invalid optimizer tensor")
                if (moments["step"].item() != steps or (moments["exp_avg_sq"] < 0).any().item()):
                    raise ValueError("invalid optimizer step/moments")
            optimizer.load_state_dict(data)
            optimizers[name] = optimizer
        if not isinstance(saved["replay"], list) or len(saved["replay"]) > self.replay_capacity:
            raise ValueError("invalid replay capacity")
        if self.method == "ppo" and saved["replay"]:
            raise ValueError("PPO may not restore off-policy replay")
        replay = [self._row(row) for row in saved["replay"]]
        pending = saved["pending"]
        if pending is not None:
            if not isinstance(pending, list) or not 1 <= len(pending) <= 48:
                raise ValueError("invalid pending episode")
            validator = copy.copy(self)
            validator.actor, validator.counts = modules["actor"], counts
            pending = [validator._row(row, on_policy=self.method == "ppo" and saved["failure"] is None)
                       for row in pending]
            if not pending[-1]["done"] or any(row["done"] for row in pending[:-1]):
                raise ValueError("invalid pending terminal boundary")
            if any(not np.array_equal(a["next_state"], b["state"]) for a, b in zip(pending, pending[1:])):
                raise ValueError("invalid pending continuity")
        sampler, noise = np.random.default_rng(0), torch.Generator(device="cpu")
        sampler.bit_generator.state = saved["sampler_rng"]
        noise.set_state(saved["noise_rng"])
        self.modules, self.optimizers = modules, optimizers
        self.actor = modules["actor"]
        self.sampler, self.noise = sampler, noise
        self.replay, self.pending = replay, pending
        self.counts, self.failure = counts, saved["failure"]
        self.optimizer_module_steps = saved["optimizer_module_steps"]
