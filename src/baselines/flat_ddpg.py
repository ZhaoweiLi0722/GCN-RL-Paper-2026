"""Flat-state / MLP-DDPG baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.heuristics import facility_net_action_from_state, heuristic_settings_for_policy
from src.rl.action_projection import project_action
from src.rl.networks import MLPActor, MLPCritic, require_torch, resolve_torch_device, torch
from src.rl.noise import OUNoise
from src.rl.preprocessing import FixedObservationScaler, reward_scale_from_config
from src.rl.replay_buffer import ReplayBuffer


class FlatDDPGAgent:
    """DDPG agent that consumes a flattened environment observation."""

    algorithm = "flat_ddpg"

    def __init__(self, state_dim: int, action_dim: int, config: dict[str, Any]):
        require_torch()
        seed = int(config.get("seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.seed = seed
        self.env_config = dict(config.get("env", {}))
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.batch_size = int(config.get("batch_size", 128))
        self.reward_scale = reward_scale_from_config(config)
        self.observation_scaler = FixedObservationScaler.from_config(config, state_dim)
        residual_config = dict(config.get("residual_action", {}))
        self.residual_action_enabled = bool(residual_config.get("enabled", False))
        self.residual_scale = float(residual_config.get("scale", 0.25))
        self.residual_scale_vector = self._make_residual_scale_vector(residual_config)
        self.residual_center_slices = self._make_residual_center_slices(residual_config)
        self.residual_l2_weight = float(residual_config.get("l2_weight", 0.0))
        self.residual_base_policy = str(residual_config.get("base_policy", "mdl2"))
        self.residual_base_settings = heuristic_settings_for_policy(
            self.residual_base_policy,
            dict(residual_config.get("base_policy_config", {})),
        )
        if self.residual_action_enabled:
            if self.env_config.get("action_mode") != "facility_net":
                raise ValueError("residual_action requires env.action_mode='facility_net'")
            n = int(self.env_config.get("num_facilities", 0))
            if self.action_dim != 4 * n:
                raise ValueError("residual_action requires a facility-net action layout")
        imitation_config = dict(config.get("imitation_pretrain", {}))
        self.imitation_regularization_weight = float(
            imitation_config.get("regularization_weight", 0.0)
        )
        self.imitation_regularization_batch_size = int(
            imitation_config.get("regularization_batch_size", self.batch_size)
        )
        self.imitation_states = None
        self.imitation_actions = None
        self.imitation_rng = np.random.default_rng(seed + 300000)
        hidden_sizes = tuple(config.get("hidden_sizes", [256, 256]))
        self.device = resolve_torch_device(config.get("device"))

        self.actor = MLPActor(state_dim, action_dim, hidden_sizes).to(self.device)
        self.actor_target = MLPActor(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic = MLPCritic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic_target = MLPCritic(state_dim, action_dim, hidden_sizes).to(self.device)
        if self.residual_action_enabled and bool(residual_config.get("zero_init_actor", False)):
            self._zero_initialize_actor_output(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(config.get("actor_lr", 1e-4)))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(config.get("critic_lr", 1e-3)))
        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=int(config.get("replay_buffer_size", 1000000)),
            seed=seed,
        )
        exploration = config.get("exploration_noise", {})
        self.noise = OUNoise(
            action_dim=action_dim,
            seed=seed,
            theta=float(exploration.get("theta", 0.15)),
            sigma=float(exploration.get("sigma", 0.2)),
        )

    def reset(self) -> None:
        self.noise.reset()

    def select_action(self, state: np.ndarray, explore: bool = True, env=None) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            state_tensor = self.observation_scaler.normalize_tensor(state_tensor)
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        if explore:
            action = action + self.noise.sample()
        action = self._compose_action_np(state, action)
        return project_action(action, env_state=env, action_space_info=self.action_dim).action

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, float(reward) * self.reward_scale, next_state, done)

    def update(self) -> dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        raw_states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        raw_next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        states = self.observation_scaler.normalize_tensor(raw_states)
        next_states = self.observation_scaler.normalize_tensor(raw_next_states)

        with torch.no_grad():
            next_network_actions = self.actor_target(next_states)
            next_actions = self._compose_actions_tensor(raw_next_states, next_network_actions)
            target_q = self.critic_target(next_states, next_actions)
            q_targets = rewards + self.gamma * (1.0 - dones) * target_q

        q_expected = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        network_actions = self.actor(states)
        actor_actions = self._compose_actions_tensor(raw_states, network_actions)
        actor_loss = -self.critic(states, actor_actions).mean()
        residual_l2_loss_value = None
        if self.residual_action_enabled and self.residual_l2_weight > 0.0:
            residual_l2_loss = self._transform_network_residuals_tensor(network_actions).pow(2).mean()
            actor_loss = actor_loss + self.residual_l2_weight * residual_l2_loss
            residual_l2_loss_value = float(residual_l2_loss.item())
        imitation_loss_value = None
        if self.imitation_regularization_weight > 0.0 and self.imitation_states is not None:
            imitation_loss = self._actor_imitation_loss()
            actor_loss = actor_loss + self.imitation_regularization_weight * imitation_loss
            imitation_loss_value = float(imitation_loss.item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        metrics = {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item())}
        if residual_l2_loss_value is not None:
            metrics["residual_l2_loss"] = residual_l2_loss_value
        if imitation_loss_value is not None:
            metrics["imitation_loss"] = imitation_loss_value
        return metrics

    def _actor_imitation_loss(self):
        sample_count = int(self.imitation_states.shape[0])
        batch_size = min(max(self.imitation_regularization_batch_size, 1), sample_count)
        indices = self.imitation_rng.choice(sample_count, size=batch_size, replace=False)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        raw_states = self.imitation_states[index_tensor]
        states = self.observation_scaler.normalize_tensor(raw_states)
        return self._supervised_action_loss(
            raw_states,
            self.actor(states),
            self.imitation_actions[index_tensor],
        )

    def pretrain_with_heuristic(self, env, settings: dict[str, Any]) -> dict[str, Any]:
        """Warm-start the actor from deterministic heuristic demonstrations."""

        from src.baselines.heuristics import get_heuristic_class

        policy_name = str(settings.get("policy", "mdl2"))
        episodes = int(settings.get("episodes", 0))
        epochs = int(settings.get("epochs", 1))
        batch_size = int(settings.get("batch_size", self.batch_size))
        max_steps = int(settings.get("max_steps_per_episode", env.config.episode_horizon))
        seed = int(settings.get("seed", self.seed + 100000))
        populate_replay = bool(settings.get("populate_replay_buffer", True))
        if episodes <= 0 or epochs <= 0:
            return {"policy": policy_name, "samples": 0, "final_loss": 0.0}

        heuristic = get_heuristic_class(policy_name)(
            state_dim=env.observation_size,
            action_dim=env.action_size,
            config=dict(settings.get("policy_config", {})),
        )
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        for episode in range(episodes):
            state = env.reset(seed=seed + episode)
            heuristic.reset()
            for _step in range(max_steps):
                action = heuristic.select_action(state, explore=False, env=env)
                next_state, reward, done, _info = env.step(action)
                states.append(np.asarray(state, dtype=np.float32))
                actions.append(np.asarray(action, dtype=np.float32))
                if populate_replay:
                    self.replay_buffer.add(
                        state,
                        action,
                        float(reward) * self.reward_scale,
                        next_state,
                        done,
                    )
                state = next_state
                if done:
                    break

        if not states:
            return {"policy": policy_name, "samples": 0, "final_loss": 0.0}

        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        summary = self._fit_action_tensors(
            state_tensor,
            action_tensor,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            target_mode="residual" if self.residual_action_enabled else "action",
        )
        self.imitation_states = state_tensor.detach()
        self.imitation_actions = action_tensor.detach()
        self.imitation_rng = np.random.default_rng(seed + 300000)
        return {"policy": policy_name, **summary}

    def fit_action_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        settings: dict[str, Any] | None = None,
        weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Fit the actor to an externally supplied state-action batch."""

        settings = settings or {}
        if weights is None:
            weights = settings.get("weights")
        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        if state_tensor.ndim != 2 or state_tensor.shape[1] != self.state_dim:
            raise ValueError(f"Expected states shape (batch, {self.state_dim}), got {tuple(state_tensor.shape)}")
        if action_tensor.ndim != 2 or action_tensor.shape[1] != self.action_dim:
            raise ValueError(
                f"Expected actions shape (batch, {self.action_dim}), got {tuple(action_tensor.shape)}"
            )
        if state_tensor.shape[0] == 0:
            return {"samples": 0, "final_loss": 0.0}
        weight_tensor = self._fit_action_weights(weights, int(state_tensor.shape[0]))
        return self._fit_action_tensors(
            state_tensor,
            action_tensor,
            epochs=int(settings.get("epochs", 1)),
            batch_size=int(settings.get("batch_size", self.batch_size)),
            seed=int(settings.get("seed", self.seed + 400000)),
            target_mode=str(
                settings.get("target_mode", "residual" if self.residual_action_enabled else "action")
            ),
            weights=weight_tensor,
        )

    def _fit_action_tensors(
        self,
        state_tensor,
        action_tensor,
        *,
        epochs: int,
        batch_size: int,
        seed: int,
        target_mode: str,
        weights=None,
    ) -> dict[str, Any]:
        sample_count = int(state_tensor.shape[0])
        batch_size = min(max(int(batch_size), 1), sample_count)
        generator = torch.Generator().manual_seed(seed)
        final_loss = 0.0
        self.actor.train()
        for _epoch in range(max(int(epochs), 1)):
            permutation = torch.randperm(sample_count, generator=generator)
            for start in range(0, sample_count, batch_size):
                indices = permutation[start : start + batch_size].to(self.device)
                raw_states = state_tensor[indices]
                states = self.observation_scaler.normalize_tensor(raw_states)
                loss = self._supervised_action_loss(
                    raw_states,
                    self.actor(states),
                    action_tensor[indices],
                    None if weights is None else weights[indices],
                    target_mode=target_mode,
                )
                self.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
                self.actor_optimizer.step()
                final_loss = float(loss.item())
        self.actor_target.load_state_dict(self.actor.state_dict())
        return {"samples": sample_count, "final_loss": final_loss, "target_mode": target_mode}

    def _fit_action_weights(self, weights: np.ndarray | None, sample_count: int):
        if weights is None:
            return None
        weight_array = np.asarray(weights, dtype=np.float32)
        if weight_array.ndim != 1 or weight_array.shape[0] != sample_count:
            raise ValueError(f"Expected weights shape ({sample_count},), got {tuple(weight_array.shape)}")
        if not np.all(np.isfinite(weight_array)):
            raise ValueError("fit_action_batch weights must be finite")
        if np.any(weight_array < 0.0):
            raise ValueError("fit_action_batch weights must be non-negative")
        if float(weight_array.sum()) <= 0.0:
            return None
        weight_array = weight_array / float(weight_array.mean())
        return torch.as_tensor(weight_array, dtype=torch.float32, device=self.device)

    def _supervised_action_loss(
        self,
        states,
        network_actions,
        target_actions,
        weights=None,
        *,
        target_mode: str | None = None,
    ):
        target_mode = target_mode or ("residual" if self.residual_action_enabled else "action")
        if target_mode == "action":
            predicted_actions = self._compose_actions_tensor(states, network_actions)
            return self._weighted_action_mse(predicted_actions, target_actions, weights)
        if target_mode != "residual":
            raise ValueError(f"Unsupported supervised action target mode: {target_mode}")
        if not self.residual_action_enabled:
            raise ValueError("residual target mode requires residual_action.enabled")
        residual_targets = self._residual_targets_tensor(states, target_actions)
        residual_mask = self._residual_loss_mask(network_actions)
        predicted_residuals = self._transform_network_residuals_tensor(network_actions)
        return self._weighted_action_mse(predicted_residuals, residual_targets, weights, residual_mask)

    def _weighted_action_mse(self, predicted_actions, target_actions, weights, dim_mask=None):
        squared_error = (predicted_actions - target_actions).pow(2)
        if dim_mask is not None:
            mask = dim_mask.to(dtype=squared_error.dtype, device=squared_error.device)
            per_sample_loss = (squared_error * mask).sum(dim=1) / mask.sum().clamp_min(1.0)
        else:
            per_sample_loss = squared_error.mean(dim=1)
        if weights is None:
            return per_sample_loss.mean()
        return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)

    def _residual_targets_tensor(self, states, target_actions):
        base_actions = self._base_actions_from_states_tensor(states)
        scale = torch.as_tensor(
            self.residual_scale_vector,
            dtype=target_actions.dtype,
            device=target_actions.device,
        ).unsqueeze(0)
        active = scale.abs() > 1e-8
        safe_scale = torch.where(active, scale, torch.ones_like(scale))
        residual_targets = (target_actions - base_actions) / safe_scale
        residual_targets = torch.where(active, residual_targets, torch.zeros_like(residual_targets))
        residual_targets = torch.clamp(residual_targets, -1.0, 1.0)
        return self._transform_network_residuals_tensor(residual_targets)

    def _residual_loss_mask(self, network_actions):
        scale = torch.as_tensor(
            self.residual_scale_vector,
            dtype=network_actions.dtype,
            device=network_actions.device,
        )
        return (scale.abs() > 1e-8).to(dtype=network_actions.dtype).unsqueeze(0)

    def _compose_action_np(self, state: np.ndarray, network_action: np.ndarray) -> np.ndarray:
        if not self.residual_action_enabled:
            return np.asarray(network_action, dtype=np.float32)
        base_action = self._base_action_from_state_np(state)
        residual_action = self._transform_network_residual_np(network_action)
        return np.clip(
            base_action + self.residual_scale_vector * residual_action,
            -1.0,
            1.0,
        ).astype(np.float32)

    def _compose_actions_tensor(self, states, network_actions):
        if not self.residual_action_enabled:
            return network_actions
        base_actions = self._base_actions_from_states_tensor(states)
        residual_actions = self._transform_network_residuals_tensor(network_actions)
        scale = torch.as_tensor(
            self.residual_scale_vector,
            dtype=network_actions.dtype,
            device=network_actions.device,
        )
        return torch.clamp(base_actions + scale * residual_actions, -1.0, 1.0)

    def _transform_network_residual_np(self, network_action: np.ndarray) -> np.ndarray:
        residual = np.asarray(network_action, dtype=np.float32).copy()
        for group_slice in self.residual_center_slices:
            residual[group_slice] = residual[group_slice] - float(residual[group_slice].mean())
        return np.clip(residual, -1.0, 1.0).astype(np.float32)

    def _transform_network_residuals_tensor(self, network_actions):
        if not self.residual_center_slices:
            return torch.clamp(network_actions, -1.0, 1.0)
        residuals = network_actions.clone()
        for group_slice in self.residual_center_slices:
            residuals[:, group_slice] = residuals[:, group_slice] - residuals[:, group_slice].mean(
                dim=1,
                keepdim=True,
            )
        return torch.clamp(residuals, -1.0, 1.0)

    def _base_action_from_state_np(self, state: np.ndarray) -> np.ndarray:
        return facility_net_action_from_state(
            state,
            self.env_config,
            settings=self.residual_base_settings,
        )

    def _base_actions_from_states_tensor(self, states):
        states_np = states.detach().cpu().numpy()
        base_actions = np.stack(
            [self._base_action_from_state_np(state) for state in states_np],
            axis=0,
        )
        return torch.as_tensor(base_actions, dtype=torch.float32, device=self.device)

    def _make_residual_scale_vector(self, residual_config: dict[str, Any]) -> np.ndarray:
        scale_vector = np.full(self.action_dim, self.residual_scale, dtype=np.float32)
        group_scales = residual_config.get("group_scales")
        if not group_scales:
            return scale_vector
        n = int(self.env_config.get("num_facilities", 0))
        if self.action_dim != 4 * n:
            raise ValueError("residual_action.group_scales requires a facility-net action layout")
        group_slices = self._facility_net_group_slices(n)
        for group, value in dict(group_scales).items():
            if group not in group_slices:
                raise ValueError(f"Unsupported residual action group scale: {group}")
            scale_vector[group_slices[group]] = float(value)
        return scale_vector

    def _make_residual_center_slices(self, residual_config: dict[str, Any]) -> tuple[slice, ...]:
        center_groups = residual_config.get("center_groups", ())
        if isinstance(center_groups, str):
            center_groups = (center_groups,)
        center_groups = tuple(center_groups or ())
        if not center_groups:
            return ()
        n = int(self.env_config.get("num_facilities", 0))
        if self.action_dim != 4 * n:
            raise ValueError("residual_action.center_groups requires a facility-net action layout")
        group_slices = self._facility_net_group_slices(n)
        slices = []
        for group in center_groups:
            if group not in group_slices:
                raise ValueError(f"Unsupported residual action center group: {group}")
            slices.append(group_slices[group])
        return tuple(slices)

    def _facility_net_group_slices(self, n: int) -> dict[str, slice]:
        return {
            "specimen_transfer": slice(0, n),
            "specimen": slice(0, n),
            "reagent_transfer": slice(n, 2 * n),
            "reagent": slice(n, 2 * n),
            "capacity_transfer": slice(2 * n, 3 * n),
            "capacity": slice(2 * n, 3 * n),
            "replenishment": slice(3 * n, 4 * n),
            "purchase": slice(3 * n, 4 * n),
        }

    def _zero_initialize_actor_output(self, actor) -> None:
        """Make a residual actor start as the heuristic anchor (zero correction)."""

        last_linear = None
        for module in actor.modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module
        if last_linear is None:
            raise ValueError("Could not locate actor output layer for zero initialization")
        torch.nn.init.zeros_(last_linear.weight)
        torch.nn.init.zeros_(last_linear.bias)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.algorithm,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            output_path,
        )

    def load_actor(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])

    def _soft_update(self, local_model, target_model) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
