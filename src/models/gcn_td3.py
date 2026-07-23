"""GNN-TD3: TD3 backbone on a GCN encoder (graph method family, Phase 6).

Mirrors ``src/baselines/td3.py`` exactly — clipped double-Q, target-policy
smoothing, delayed actor update — with the only differences being (a) flat
states are converted to GCN node features via ``flat_state_to_node_features`` and
(b) the MLP actor/critic are replaced by ``GCNActor``/``GCNCritic``.

When ``residual_action.enabled`` is set, the actor output is interpreted as a
bounded residual around a deterministic heuristic anchor. This gives the
manuscript pipeline a TD3-backed version of the safety-gated graph residual
policy while keeping plain ``gcn_td3`` as the from-scratch baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.heuristics import facility_net_action_from_state, heuristic_settings_for_policy
from src.models.gcn import GCNActor, GCNCritic, transfer_matching_parameters
from src.models.graph_features import build_graph_spec, flat_state_to_node_features
from src.rl.action_projection import project_action
from src.rl.networks import require_torch, resolve_torch_device, torch
from src.rl.noise import GaussianNoise
from src.rl.preprocessing import reward_scale_from_config
from src.rl.replay_buffer import ReplayBuffer


class GCNTD3Agent:
    """TD3 agent that embeds the manufacturing network with a GCN."""

    algorithm = "gcn_td3"

    def __init__(self, state_dim: int, action_dim: int, config: dict[str, Any]):
        require_torch()
        seed = int(config.get("seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.algorithm = str(config.get("algorithm", self.algorithm))
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.seed = seed
        self.env_config = dict(config.get("env", {}))
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.batch_size = int(config.get("batch_size", 128))
        self.policy_noise = float(config.get("policy_noise", 0.2))
        self.noise_clip = float(config.get("noise_clip", 0.5))
        self.policy_delay = int(config.get("policy_delay", 2))
        self.freeze_actor_updates = bool(config.get("freeze_actor_updates", False))
        self.total_updates = 0
        self.reward_scale = reward_scale_from_config(config)
        self.graph_spec = build_graph_spec(config, state_dim)
        self.device = resolve_torch_device(config.get("device"))
        residual_config = dict(config.get("residual_action", {}))
        self.residual_action_enabled = bool(residual_config.get("enabled", False))
        self.residual_scale = float(residual_config.get("scale", 0.25))
        self.residual_scale_vector = self._make_residual_scale_vector(residual_config)
        self.residual_center_slices = self._make_residual_center_slices(residual_config)
        self.residual_positive_slices = self._make_residual_positive_slices(residual_config)
        self.residual_state_gate_config = dict(residual_config.get("state_gate", {}))
        self.residual_state_gate_groups = self._make_residual_state_gate_groups(residual_config)
        self.residual_pressure_projection_groups = self._make_pressure_projection_groups(
            residual_config
        )
        self.residual_l2_weight = float(residual_config.get("l2_weight", 0.0))
        self.residual_base_policy = str(residual_config.get("base_policy", "mdl2"))
        self.residual_base_settings = heuristic_settings_for_policy(
            self.residual_base_policy,
            dict(residual_config.get("base_policy_config", {})),
        )
        if self.residual_action_enabled:
            if self.env_config.get("action_mode") != "facility_net":
                raise ValueError("residual_action requires env.action_mode='facility_net'")
            if self.action_dim != 4 * self.graph_spec.num_facilities:
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
        self.imitation_node_features = None
        self.imitation_rng = np.random.default_rng(seed + 300000)
        advantage_config = dict(config.get("anchor_advantage_actor_loss", {}))
        self.anchor_advantage_actor_loss_enabled = bool(advantage_config.get("enabled", False))
        self.anchor_advantage_margin = float(advantage_config.get("margin", 0.0))
        self.anchor_advantage_temperature = max(float(advantage_config.get("temperature", 0.05)), 1e-6)
        self.anchor_advantage_use_twin_min = bool(advantage_config.get("use_twin_min", True))
        self.anchor_advantage_negative_penalty_weight = float(
            advantage_config.get("negative_penalty_weight", 0.0)
        )
        patient_proxy_config = dict(config.get("patient_service_proxy_actor_loss", {}))
        self.patient_service_proxy_actor_loss_enabled = bool(patient_proxy_config.get("enabled", False))
        self.patient_service_proxy_weight = float(patient_proxy_config.get("weight", 0.0))
        self.patient_service_proxy_cost_weight = float(patient_proxy_config.get("cost_weight", 0.0))
        self.patient_service_proxy_low_pressure_weight = float(
            patient_proxy_config.get("low_pressure_weight", 0.0)
        )
        self.patient_service_proxy_group = str(patient_proxy_config.get("group", "replenishment"))
        self.patient_service_proxy_positive_only = bool(patient_proxy_config.get("positive_only", True))

        gcn_hidden_sizes = tuple(config.get("gcn_hidden_sizes", [64, 64]))
        head_hidden_sizes = tuple(config.get("hidden_sizes", [256, 256]))
        include_global_context = bool(config.get("include_global_context", True))
        readout_mode = str(config.get("actor_readout_mode", "global_flat"))

        def make_actor():
            return GCNActor(
                self.graph_spec.node_feature_dim,
                self.graph_spec.num_facilities,
                self.graph_spec.num_nodes,
                action_dim,
                self.graph_spec.edge_index,
                gcn_hidden_sizes,
                head_hidden_sizes,
                include_global_context=include_global_context,
                readout_mode=readout_mode,
                edge_weights=self.graph_spec.edge_weights,
            ).to(self.device)

        def make_critic():
            return GCNCritic(
                self.graph_spec.node_feature_dim,
                self.graph_spec.num_facilities,
                self.graph_spec.num_nodes,
                action_dim,
                self.graph_spec.edge_index,
                gcn_hidden_sizes,
                head_hidden_sizes,
                include_global_context=include_global_context,
                edge_weights=self.graph_spec.edge_weights,
            ).to(self.device)

        self.actor = make_actor()
        self.actor_target = make_actor()
        self.critic1 = make_critic()
        self.critic2 = make_critic()
        self.critic1_target = make_critic()
        self.critic2_target = make_critic()
        if self.residual_action_enabled and bool(residual_config.get("zero_init_actor", False)):
            self._zero_initialize_actor_output(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(config.get("actor_lr", 1e-4)))
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=float(config.get("critic_lr", 1e-3)))
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=float(config.get("critic_lr", 1e-3)))
        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=int(config.get("replay_buffer_size", 1000000)),
            seed=seed,
        )
        self.exploration_noise = GaussianNoise(
            action_dim=action_dim,
            std=float(config.get("exploration_noise_std", 0.1)),
            seed=seed,
        )

    def _nodes(self, states):
        return flat_state_to_node_features(states, self.graph_spec)

    def reset(self) -> None:
        return None

    def select_action(self, state: np.ndarray, explore: bool = True, env=None) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            network_action = self.actor(self._nodes(state_tensor)).cpu().numpy()[0]
        self.actor.train()
        if explore:
            network_action = network_action + self.exploration_noise.sample()
        action = self._compose_action_np(state, network_action)
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

        self.total_updates += 1
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        node_features = self._nodes(states)
        next_node_features = self._nodes(next_states)

        with torch.no_grad():
            noise = torch.normal(
                mean=0.0,
                std=self.policy_noise,
                size=(self.batch_size, self.action_dim),
                device=self.device,
            ).clamp(-self.noise_clip, self.noise_clip)
            next_network_actions = (self.actor_target(next_node_features) + noise).clamp(-1.0, 1.0)
            next_actions = self._compose_actions_tensor(next_states, next_network_actions)
            target_q1 = self.critic1_target(next_node_features, next_actions)
            target_q2 = self.critic2_target(next_node_features, next_actions)
            target_q = torch.minimum(target_q1, target_q2)
            q_targets = rewards + self.gamma * (1.0 - dones) * target_q

        q1_expected = self.critic1(node_features, actions)
        q2_expected = self.critic2(node_features, actions)
        critic1_loss = torch.nn.functional.mse_loss(q1_expected, q_targets)
        critic2_loss = torch.nn.functional.mse_loss(q2_expected, q_targets)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        metrics = {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
        }

        if self.total_updates % self.policy_delay == 0:
            if self.freeze_actor_updates:
                self._soft_update(self.critic1, self.critic1_target)
                self._soft_update(self.critic2, self.critic2_target)
                metrics["actor_frozen"] = 1.0
                return metrics
            network_actions = self.actor(node_features)
            actor_actions = self._compose_actions_tensor(states, network_actions)
            actor_loss, advantage_metrics = self._actor_objective(node_features, states, actor_actions)
            residual_l2_loss_value = None
            if self.residual_action_enabled and self.residual_l2_weight > 0.0:
                policy_residuals = self._policy_residuals_tensor(states, network_actions)
                residual_l2_loss = policy_residuals.pow(2).mean()
                actor_loss = actor_loss + self.residual_l2_weight * residual_l2_loss
                residual_l2_loss_value = float(residual_l2_loss.item())
            elif self.patient_service_proxy_actor_loss_enabled:
                policy_residuals = self._policy_residuals_tensor(states, network_actions)
            else:
                policy_residuals = None
            proxy_metrics = {}
            if (
                self.patient_service_proxy_actor_loss_enabled
                and self.residual_action_enabled
                and policy_residuals is not None
            ):
                proxy_loss, proxy_metrics = self._patient_service_proxy_actor_loss(
                    states,
                    policy_residuals,
                )
                actor_loss = actor_loss + proxy_loss
            imitation_loss_value = None
            if self.imitation_regularization_weight > 0.0 and self.imitation_states is not None:
                imitation_loss = self._actor_imitation_loss()
                actor_loss = actor_loss + self.imitation_regularization_weight * imitation_loss
                imitation_loss_value = float(imitation_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            metrics["actor_loss"] = float(actor_loss.item())
            metrics.update(advantage_metrics)
            metrics.update(proxy_metrics)
            if residual_l2_loss_value is not None:
                metrics["residual_l2_loss"] = residual_l2_loss_value
            if imitation_loss_value is not None:
                metrics["imitation_loss"] = imitation_loss_value

        return metrics

    def _patient_service_proxy_actor_loss(self, states, residuals):
        n = self.graph_spec.num_facilities
        group_slices = self._facility_net_group_slices(n)
        group_slice = group_slices.get(self.patient_service_proxy_group)
        if group_slice is None:
            raise ValueError(
                "Unsupported patient_service_proxy_actor_loss group: "
                f"{self.patient_service_proxy_group}"
            )
        group_residual = residuals[:, group_slice]
        if self.patient_service_proxy_positive_only:
            group_residual = group_residual.clamp_min(0.0)
        pressure = self._resource_pressure_tensor(states)
        positive_pressure = pressure.clamp_min(0.0)
        denominator = positive_pressure.amax(dim=1, keepdim=True).clamp_min(1e-6)
        pressure_pattern = positive_pressure / denominator
        alignment = (group_residual * pressure_pattern).mean()
        loss = -self.patient_service_proxy_weight * alignment
        low_pressure_penalty = torch.zeros((), dtype=residuals.dtype, device=residuals.device)
        if self.patient_service_proxy_low_pressure_weight > 0.0:
            low_pressure = 1.0 - pressure_pattern
            low_pressure_penalty = (group_residual.pow(2) * low_pressure).mean()
            loss = loss + self.patient_service_proxy_low_pressure_weight * low_pressure_penalty
        cost_penalty = torch.zeros((), dtype=residuals.dtype, device=residuals.device)
        if self.patient_service_proxy_cost_weight > 0.0:
            cost_penalty = group_residual.clamp_min(0.0).mean()
            loss = loss + self.patient_service_proxy_cost_weight * cost_penalty
        return loss, {
            "actor_patient_service_proxy_alignment": float(alignment.item()),
            "actor_patient_service_proxy_low_pressure_penalty": float(
                low_pressure_penalty.item()
            ),
            "actor_patient_service_proxy_cost_penalty": float(cost_penalty.item()),
        }

    def _actor_objective(self, node_features, states, actor_actions):
        if not self.anchor_advantage_actor_loss_enabled or not self.residual_action_enabled:
            return -self.critic1(node_features, actor_actions).mean(), {}

        actor_q = self._policy_q_value(node_features, actor_actions)
        with torch.no_grad():
            anchor_actions = self._base_actions_from_states_tensor(states)
            anchor_q = self._policy_q_value(node_features, anchor_actions)
        advantage = actor_q - anchor_q
        shifted = (advantage - self.anchor_advantage_margin) / self.anchor_advantage_temperature
        actor_loss = -(
            self.anchor_advantage_temperature * torch.nn.functional.softplus(shifted)
        ).mean()
        negative_advantage_penalty = torch.zeros(
            (),
            dtype=advantage.dtype,
            device=advantage.device,
        )
        if self.anchor_advantage_negative_penalty_weight > 0.0:
            negative_shifted = (
                self.anchor_advantage_margin - advantage
            ) / self.anchor_advantage_temperature
            negative_advantage_penalty = (
                self.anchor_advantage_temperature
                * torch.nn.functional.softplus(negative_shifted)
            ).mean()
            actor_loss = actor_loss + (
                self.anchor_advantage_negative_penalty_weight * negative_advantage_penalty
            )
        return actor_loss, {
            "actor_anchor_advantage_mean": float(advantage.mean().item()),
            "actor_anchor_advantage_positive_fraction": float(
                (advantage > 0.0).to(dtype=torch.float32).mean().item()
            ),
            "actor_anchor_negative_advantage_penalty": float(
                negative_advantage_penalty.item()
            ),
        }

    def _policy_q_value(self, node_features, actions):
        q1 = self.critic1(node_features, actions)
        if not self.anchor_advantage_use_twin_min:
            return q1
        q2 = self.critic2(node_features, actions)
        return torch.minimum(q1, q2)

    def _actor_imitation_loss(self):
        sample_count = int(self.imitation_states.shape[0])
        batch_size = min(max(self.imitation_regularization_batch_size, 1), sample_count)
        indices = self.imitation_rng.choice(sample_count, size=batch_size, replace=False)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        if self.imitation_node_features is None:
            node_features = self._nodes(self.imitation_states[index_tensor])
        else:
            node_features = self.imitation_node_features[index_tensor]
        return self._supervised_action_loss(
            self.imitation_states[index_tensor],
            self.actor(node_features),
            self.imitation_actions[index_tensor],
        )

    def pretrain_with_heuristic(self, env, settings: dict[str, Any]) -> dict[str, Any]:
        """Warm-start the TD3 actor from deterministic heuristic demonstrations."""

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

        policy_config = dict(settings.get("policy_config", {}))
        heuristic = get_heuristic_class(policy_name)(
            state_dim=env.observation_size,
            action_dim=env.action_size,
            config=policy_config,
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

        summary = self.fit_action_batch(
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "target_mode": "residual" if self.residual_action_enabled else "action",
            },
        )
        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        self.imitation_states = state_tensor.detach()
        self.imitation_actions = action_tensor.detach()
        with torch.no_grad():
            self.imitation_node_features = self._nodes(self.imitation_states).detach()
        self.imitation_rng = np.random.default_rng(seed + 300000)
        return {"policy": policy_name, "samples": summary["samples"], "final_loss": summary["final_loss"]}

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
        sample_count = int(state_tensor.shape[0])
        if sample_count == 0:
            return {"samples": 0, "final_loss": 0.0}

        weight_tensor = self._fit_action_weights(weights, sample_count)
        epochs = int(settings.get("epochs", 1))
        batch_size = min(max(int(settings.get("batch_size", self.batch_size)), 1), sample_count)
        seed = int(settings.get("seed", self.seed + 400000))
        target_mode = str(
            settings.get("target_mode", "residual" if self.residual_action_enabled else "action")
        )
        with torch.no_grad():
            node_feature_tensor = self._nodes(state_tensor).detach()
            residual_target_tensor = None
            residual_mask_tensor = None
            if target_mode == "residual":
                if not self.residual_action_enabled:
                    raise ValueError("residual target mode requires residual_action.enabled")
                residual_target_tensor = self._residual_targets_tensor(
                    state_tensor,
                    action_tensor,
                ).detach()
                residual_mask_tensor = self._residual_loss_mask(
                    action_tensor,
                    state_tensor,
                ).detach()
        generator = torch.Generator().manual_seed(seed)
        final_loss = 0.0

        self.actor.train()
        for _epoch in range(max(epochs, 1)):
            permutation = torch.randperm(sample_count, generator=generator)
            for start in range(0, sample_count, batch_size):
                indices = permutation[start : start + batch_size].to(self.device)
                node_features = node_feature_tensor[indices]
                network_actions = self.actor(node_features)
                batch_weights = None if weight_tensor is None else weight_tensor[indices]
                if target_mode == "residual":
                    predicted_residuals = self._policy_residuals_tensor(
                        state_tensor[indices],
                        network_actions,
                    )
                    residual_mask = (
                        residual_mask_tensor
                        if residual_mask_tensor.shape[0] == 1
                        else residual_mask_tensor[indices]
                    )
                    loss = self._weighted_action_mse(
                        predicted_residuals,
                        residual_target_tensor[indices],
                        batch_weights,
                        residual_mask,
                    )
                else:
                    loss = self._supervised_action_loss(
                        state_tensor[indices],
                        network_actions,
                        action_tensor[indices],
                        batch_weights,
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
        weight_sum = float(weight_array.sum())
        if weight_sum <= 0.0:
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
        residual_mask = self._residual_loss_mask(network_actions, states)
        predicted_residuals = self._policy_residuals_tensor(states, network_actions)
        return self._weighted_action_mse(predicted_residuals, residual_targets, weights, residual_mask)

    def _weighted_action_mse(self, predicted_actions, target_actions, weights, dim_mask=None):
        squared_error = (predicted_actions - target_actions).pow(2)
        if dim_mask is not None:
            mask = dim_mask.to(dtype=squared_error.dtype, device=squared_error.device)
            denominator = mask.sum(dim=1).clamp_min(1.0) if mask.ndim == 2 else mask.sum().clamp_min(1.0)
            per_sample_loss = (squared_error * mask).sum(dim=1) / denominator
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
        residual_targets = self._transform_network_residuals_tensor(residual_targets)
        return self._apply_state_gate_residuals_tensor(states, residual_targets)

    def _residual_loss_mask(self, network_actions, states=None):
        scale = torch.as_tensor(
            self.residual_scale_vector,
            dtype=network_actions.dtype,
            device=network_actions.device,
        )
        mask = (scale.abs() > 1e-8).to(dtype=network_actions.dtype).unsqueeze(0)
        if states is not None and self.residual_state_gate_groups:
            mask = mask.expand(network_actions.shape[0], -1)
            mask = mask * self._state_gate_action_mask_tensor(states, dtype=network_actions.dtype)
        return mask

    def _compose_action_np(self, state: np.ndarray, network_action: np.ndarray) -> np.ndarray:
        if not self.residual_action_enabled:
            return np.asarray(network_action, dtype=np.float32)
        base_action = self._base_action_from_state_np(state)
        residual_action = self._policy_residual_np(state, network_action)
        return np.clip(
            base_action + self.residual_scale_vector * residual_action,
            -1.0,
            1.0,
        ).astype(np.float32)

    def _compose_actions_tensor(self, states, network_actions):
        if not self.residual_action_enabled:
            return network_actions
        base_actions = self._base_actions_from_states_tensor(states)
        residual_actions = self._policy_residuals_tensor(states, network_actions)
        scale = torch.as_tensor(
            self.residual_scale_vector,
            dtype=network_actions.dtype,
            device=network_actions.device,
        )
        return torch.clamp(base_actions + scale * residual_actions, -1.0, 1.0)

    def _policy_residual_np(self, state: np.ndarray, network_action: np.ndarray) -> np.ndarray:
        if not self.residual_pressure_projection_groups and not self.residual_state_gate_groups:
            return self._transform_network_residual_np(network_action)
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            action_tensor = torch.as_tensor(
                network_action,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            residual = self._policy_residuals_tensor(state_tensor, action_tensor)
        return residual.cpu().numpy()[0].astype(np.float32)

    def _policy_residuals_tensor(self, states, network_actions):
        residuals = self._transform_network_residuals_tensor(network_actions)
        if self.residual_pressure_projection_groups:
            residuals = self._project_residuals_to_pressure_patterns(states, residuals)
            residuals = self._apply_positive_residual_slices_tensor(residuals)
        residuals = self._apply_state_gate_residuals_tensor(states, residuals)
        return torch.clamp(residuals, -1.0, 1.0)

    def _transform_network_residual_np(self, network_action: np.ndarray) -> np.ndarray:
        residual = np.asarray(network_action, dtype=np.float32).copy()
        for group_slice in self.residual_center_slices:
            residual[group_slice] = residual[group_slice] - float(residual[group_slice].mean())
        for group_slice in self.residual_positive_slices:
            residual[group_slice] = np.maximum(residual[group_slice], 0.0)
        return np.clip(residual, -1.0, 1.0).astype(np.float32)

    def _transform_network_residuals_tensor(self, network_actions):
        residuals = network_actions
        for group_slice in self.residual_center_slices:
            centered = residuals[:, group_slice] - residuals[:, group_slice].mean(
                dim=1,
                keepdim=True,
            )
            residuals = self._replace_action_slice_tensor(residuals, group_slice, centered)
        residuals = self._apply_positive_residual_slices_tensor(residuals)
        return torch.clamp(residuals, -1.0, 1.0)

    def _apply_positive_residual_slices_tensor(self, residuals):
        if not self.residual_positive_slices:
            return residuals
        positive = residuals
        for group_slice in self.residual_positive_slices:
            positive = self._replace_action_slice_tensor(
                positive,
                group_slice,
                positive[:, group_slice].clamp_min(0.0),
            )
        return positive

    def _project_residuals_to_pressure_patterns(self, states, residuals):
        patterns = self._residual_pressure_patterns_tensor(states)
        projected = residuals
        n = self.graph_spec.num_facilities
        group_patterns = {
            "reagent_transfer": patterns["resource"],
            "reagent": patterns["resource"],
            "capacity_transfer": patterns["capacity"],
            "capacity": patterns["capacity"],
            "replenishment": patterns["resource"],
            "purchase": patterns["resource"],
        }
        group_slices = self._facility_net_group_slices(n)
        for group in self.residual_pressure_projection_groups:
            pattern = group_patterns.get(group)
            group_slice = group_slices.get(group)
            if pattern is None or group_slice is None:
                continue
            current = projected[:, group_slice]
            denominator = pattern.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-6)
            coefficient = (current * pattern).sum(dim=1, keepdim=True) / denominator
            projected = self._replace_action_slice_tensor(
                projected,
                group_slice,
                coefficient * pattern,
            )
        return torch.clamp(projected, -1.0, 1.0)

    def _apply_state_gate_residuals_tensor(self, states, residuals):
        if not self.residual_state_gate_groups:
            return residuals
        return residuals * self._state_gate_action_mask_tensor(states, dtype=residuals.dtype)

    def _state_gate_action_mask_tensor(self, states, *, dtype):
        n = self.graph_spec.num_facilities
        group_slices = self._facility_net_group_slices(n)
        pressure = self._resource_pressure_tensor(states)
        threshold = float(self.residual_state_gate_config.get("threshold", 0.0))
        gate = (pressure > threshold).to(dtype=dtype)
        mask = torch.ones((states.shape[0], self.action_dim), dtype=dtype, device=states.device)
        for group in self.residual_state_gate_groups:
            group_slice = group_slices[group]
            mask = self._replace_action_slice_tensor(mask, group_slice, gate)
        return mask

    def _replace_action_slice_tensor(self, actions, group_slice: slice, replacement):
        start = 0 if group_slice.start is None else int(group_slice.start)
        stop = actions.shape[1] if group_slice.stop is None else int(group_slice.stop)
        if group_slice.step not in (None, 1):
            raise ValueError("Residual action tensor slices must be contiguous")
        pieces = []
        if start > 0:
            pieces.append(actions[:, :start])
        pieces.append(replacement)
        if stop < actions.shape[1]:
            pieces.append(actions[:, stop:])
        return torch.cat(pieces, dim=1)

    def _residual_pressure_patterns_tensor(self, states):
        pressure_terms = self._resource_pressure_terms_tensor(states)
        resource_pressure = pressure_terms["resource_pressure"]
        capacity_pressure = pressure_terms["capacity_pressure"]
        return {
            "resource": self._centered_unit_pattern_tensor(resource_pressure),
            "capacity": self._centered_unit_pattern_tensor(capacity_pressure),
        }

    def _resource_pressure_tensor(self, states):
        return self._resource_pressure_terms_tensor(states)["resource_pressure"]

    def _resource_pressure_terms_tensor(self, states):
        n = self.graph_spec.num_facilities
        lead_time = int(self.env_config.get("production_lead_time", 3))
        include_supplier = int(bool(self.env_config.get("include_supplier_state", False)))
        include_forecast = int(bool(self.env_config.get("include_demand_forecast_state", False)))
        include_transfer_pipeline = int(
            bool(self.env_config.get("include_transfer_pipeline_state", False))
        )
        features_per_facility = 3 + lead_time + include_supplier + include_forecast
        features_per_facility += 3 * include_transfer_pipeline
        facility_state = states[:, : n * features_per_facility].reshape(
            states.shape[0],
            n,
            features_per_facility,
        )
        demand = facility_state[:, :, 0]
        specimens = facility_state[:, :, 1]
        reagents = facility_state[:, :, 2]
        idle_bioreactors = facility_state[:, :, 3]
        if include_forecast:
            forecast_col = 3 + lead_time + include_supplier
            forecast = facility_state[:, :, forecast_col]
        else:
            forecast = demand
        risk = self._patient_risk_signal_tensor(states, features_per_facility)
        resource_pressure = demand + 0.25 * forecast + specimens - reagents + 0.5 * risk
        capacity_pressure = demand + 0.25 * forecast + specimens - idle_bioreactors + 0.5 * risk
        return {
            "resource_pressure": resource_pressure,
            "capacity_pressure": capacity_pressure,
        }

    def _patient_risk_signal_tensor(self, states, features_per_facility: int):
        if self.env_config.get("env_type") != "patient_condition":
            return torch.zeros(
                (states.shape[0], self.graph_spec.num_facilities),
                dtype=states.dtype,
                device=states.device,
            )
        n = self.graph_spec.num_facilities
        summary_edges = tuple(self.env_config.get("survival_bucket_edges", (0.85, 0.90, 0.97)))
        summary_width = 3 + len(summary_edges) + 1
        base_width = n * int(features_per_facility)
        expected_width = base_width + n * summary_width
        if states.shape[1] < expected_width:
            return torch.zeros((states.shape[0], n), dtype=states.dtype, device=states.device)
        summary = states[:, base_width:expected_width].reshape(states.shape[0], n, summary_width)
        near_expiry = summary[:, :, 2]
        critical_survival = summary[:, :, 3] if summary_width > 3 else torch.zeros_like(near_expiry)
        return near_expiry + critical_survival

    def _centered_unit_pattern_tensor(self, values):
        centered = values - values.mean(dim=1, keepdim=True)
        denominator = centered.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        return centered / denominator

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
        n = self.graph_spec.num_facilities
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
        n = self.graph_spec.num_facilities
        if self.action_dim != 4 * n:
            raise ValueError("residual_action.center_groups requires a facility-net action layout")
        group_slices = self._facility_net_group_slices(n)
        slices = []
        for group in center_groups:
            if group not in group_slices:
                raise ValueError(f"Unsupported residual action center group: {group}")
            slices.append(group_slices[group])
        return tuple(slices)

    def _make_residual_positive_slices(self, residual_config: dict[str, Any]) -> tuple[slice, ...]:
        positive_groups = residual_config.get("positive_only_groups", ())
        if isinstance(positive_groups, str):
            positive_groups = (positive_groups,)
        positive_groups = tuple(positive_groups or ())
        if not positive_groups:
            return ()
        n = self.graph_spec.num_facilities
        if self.action_dim != 4 * n:
            raise ValueError("residual_action.positive_only_groups requires a facility-net action layout")
        group_slices = self._facility_net_group_slices(n)
        slices = []
        for group in positive_groups:
            if group not in group_slices:
                raise ValueError(f"Unsupported residual action positive-only group: {group}")
            slices.append(group_slices[group])
        return tuple(slices)

    def _make_residual_state_gate_groups(self, residual_config: dict[str, Any]) -> tuple[str, ...]:
        gate_config = dict(residual_config.get("state_gate", {}))
        if not bool(gate_config.get("enabled", False)):
            return ()
        groups = gate_config.get("groups", ())
        if isinstance(groups, str):
            groups = (groups,)
        groups = tuple(groups or ())
        if not groups:
            return ()
        n = self.graph_spec.num_facilities
        if self.action_dim != 4 * n:
            raise ValueError("residual_action.state_gate.groups requires a facility-net action layout")
        group_slices = self._facility_net_group_slices(n)
        normalized = []
        for group in groups:
            if group not in group_slices:
                raise ValueError(f"Unsupported residual action state-gated group: {group}")
            normalized.append(group)
        return tuple(normalized)

    def _make_pressure_projection_groups(self, residual_config: dict[str, Any]) -> tuple[str, ...]:
        if not bool(residual_config.get("enabled", False)):
            return ()
        projection = residual_config.get("pressure_projection", {})
        if isinstance(projection, bool):
            enabled = projection
            groups = ("reagent_transfer", "capacity_transfer", "replenishment")
        else:
            projection = dict(projection or {})
            enabled = bool(projection.get("enabled", False))
            groups = projection.get(
                "groups",
                ("reagent_transfer", "capacity_transfer", "replenishment"),
            )
        if not enabled:
            return ()
        if isinstance(groups, str):
            groups = (groups,)
        supported = set(self._facility_net_group_slices(self.graph_spec.num_facilities))
        unknown = [str(group) for group in tuple(groups or ()) if str(group) not in supported]
        if unknown:
            raise ValueError(f"Unsupported pressure_projection groups: {unknown}")
        return tuple(str(group) for group in tuple(groups or ()))

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
        """Make a residual actor start as the heuristic anchor."""

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
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
            },
            output_path,
        )

    def load_actor(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])

    def warm_start_actor(self, path: str | Path) -> dict[str, list[str]]:
        """Curriculum warm-start: seed this agent's actor from a checkpoint trained
        on a (possibly smaller) network. Transfers every shape-matching weight and
        syncs the target actor. With ``actor_readout_mode='facility_action'`` the
        actor is size-invariant, so the full policy transfers; ``global_flat`` transfers
        only the encoder. Returns the transferred/skipped key summary.
        """

        checkpoint = torch.load(path, map_location=self.device)
        summary = transfer_matching_parameters(checkpoint["actor"], self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        return summary

    def _soft_update(self, local_model, target_model) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
