"""GCN-DDPG agent for graph-aware PRM capacity planning."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.heuristics import facility_net_action_from_state, heuristic_settings_for_policy
from src.models.gcn import (
    GCNActor,
    GCNCorrectionGate,
    GCNCritic,
    transfer_matching_parameters,
)
from src.models.graph_features import (
    GraphStateSpec,
    build_graph_spec,
    flat_state_to_node_features,
)
from src.rl.action_projection import project_action, project_tensor_to_pattern_basis
from src.rl.networks import require_torch, resolve_torch_device, torch
from src.rl.noise import OUNoise
from src.rl.preprocessing import reward_scale_from_config
from src.rl.replay_buffer import ReplayBuffer

# Re-exported for backward compatibility (these used to live in this module).
__all__ = ["GCNDDPGAgent", "GraphStateSpec", "build_graph_spec", "flat_state_to_node_features"]


class GCNDDPGAgent:
    """DDPG agent that embeds the manufacturing network with a GCN."""

    algorithm = "gcn_ddpg"

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
        self.graph_spec = build_graph_spec(config, state_dim)
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.batch_size = int(config.get("batch_size", 128))
        self.reward_scale = reward_scale_from_config(config)
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
        pressure_projection_config = dict(
            residual_config.get("pressure_projection", {})
        )
        self.residual_replenishment_uniform_basis = bool(
            pressure_projection_config.get("replenishment_uniform_basis", False)
        )
        self.residual_l2_weight = float(residual_config.get("l2_weight", 0.0))
        correction_gate_config = dict(residual_config.get("correction_gate", {}))
        self.correction_gate_enabled = bool(
            self.residual_action_enabled
            and correction_gate_config.get("enabled", False)
        )
        self.correction_gate_threshold = float(
            correction_gate_config.get("threshold", 0.5)
        )
        self.correction_gate_mode = str(
            correction_gate_config.get("mode", "classification")
        )
        if self.correction_gate_mode not in ("classification", "advantage"):
            raise ValueError(
                "residual_action.correction_gate.mode must be "
                "'classification' or 'advantage'"
            )
        if (
            self.correction_gate_mode == "classification"
            and not 0.0 <= self.correction_gate_threshold <= 1.0
        ):
            raise ValueError(
                "classification correction-gate threshold must lie in [0, 1]"
            )
        self.correction_gate_advantage_scale = max(
            float(correction_gate_config.get("advantage_scale", 1_000_000.0)),
            1e-8,
        )
        self.correction_gate_advantage_clip = max(
            float(correction_gate_config.get("advantage_clip", 5.0)),
            1e-6,
        )
        self.correction_gate_target_delta = max(
            float(correction_gate_config.get("target_delta", 1e-5)),
            0.0,
        )
        self.correction_gate_loss_weight = max(
            float(correction_gate_config.get("loss_weight", 1.0)),
            0.0,
        )
        self.correction_gate_groups = tuple(
            str(group)
            for group in correction_gate_config.get("groups", ())
        )
        valid_gate_groups = self._facility_net_group_slices(
            self.graph_spec.num_facilities
        )
        unknown_gate_groups = tuple(
            group
            for group in self.correction_gate_groups
            if group not in valid_gate_groups
        )
        if unknown_gate_groups:
            raise ValueError(
                "Unsupported correction-gate groups: "
                + ", ".join(unknown_gate_groups)
            )
        self.correction_gate_output_dim = max(
            len(self.correction_gate_groups),
            1,
        )
        raw_group_thresholds = dict(
            correction_gate_config.get("group_thresholds", {})
        )
        self.correction_gate_group_thresholds = tuple(
            float(
                raw_group_thresholds.get(
                    group,
                    self.correction_gate_threshold,
                )
            )
            for group in self.correction_gate_groups
        )
        if (
            self.correction_gate_mode == "classification"
            and any(
                threshold < 0.0 or threshold > 1.0
                for threshold in self.correction_gate_group_thresholds
            )
        ):
            raise ValueError(
                "classification correction-gate group thresholds must lie in [0, 1]"
            )
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
        self.imitation_weights = None
        self.imitation_rng = np.random.default_rng(seed + 300000)
        advantage_config = dict(config.get("anchor_advantage_actor_loss", {}))
        self.anchor_advantage_actor_loss_enabled = bool(advantage_config.get("enabled", False))
        self.anchor_advantage_margin = float(advantage_config.get("margin", 0.0))
        self.anchor_advantage_temperature = max(
            float(advantage_config.get("temperature", 0.05)),
            1e-6,
        )
        self.anchor_advantage_negative_penalty_weight = float(
            advantage_config.get("negative_penalty_weight", 0.0)
        )
        patient_proxy_config = dict(config.get("patient_service_proxy_actor_loss", {}))
        self.patient_service_proxy_actor_loss_enabled = bool(
            patient_proxy_config.get("enabled", False)
        )
        self.patient_service_proxy_weight = float(patient_proxy_config.get("weight", 0.0))
        self.patient_service_proxy_cost_weight = float(
            patient_proxy_config.get("cost_weight", 0.0)
        )
        self.patient_service_proxy_low_pressure_weight = float(
            patient_proxy_config.get("low_pressure_weight", 0.0)
        )
        self.patient_service_proxy_group = str(
            patient_proxy_config.get("group", "replenishment")
        )
        self.patient_service_proxy_positive_only = bool(
            patient_proxy_config.get("positive_only", True)
        )
        gcn_hidden_sizes = tuple(config.get("gcn_hidden_sizes", [64, 64]))
        actor_hidden_sizes = tuple(config.get("actor_hidden_sizes", config.get("hidden_sizes", [256, 128])))
        critic_hidden_sizes = tuple(config.get("critic_hidden_sizes", config.get("hidden_sizes", [256, 128])))
        include_global_context = bool(config.get("include_global_context", True))
        actor_readout_mode = str(config.get("actor_readout_mode", "global_flat"))
        self.device = resolve_torch_device(config.get("device"))

        self.actor = GCNActor(
            self.graph_spec.node_feature_dim,
            self.graph_spec.num_facilities,
            self.graph_spec.num_nodes,
            action_dim,
            self.graph_spec.edge_index,
            gcn_hidden_sizes,
            actor_hidden_sizes,
            include_global_context=include_global_context,
            readout_mode=actor_readout_mode,
            edge_weights=self.graph_spec.edge_weights,
            resource_edges=self.graph_spec.resource_edge_index,
            capacity_edges=self.graph_spec.capacity_edge_index,
            resource_edge_features=self.graph_spec.resource_edge_features,
            capacity_edge_features=self.graph_spec.capacity_edge_features,
        ).to(self.device)
        self.actor_target = GCNActor(
            self.graph_spec.node_feature_dim,
            self.graph_spec.num_facilities,
            self.graph_spec.num_nodes,
            action_dim,
            self.graph_spec.edge_index,
            gcn_hidden_sizes,
            actor_hidden_sizes,
            include_global_context=include_global_context,
            readout_mode=actor_readout_mode,
            edge_weights=self.graph_spec.edge_weights,
            resource_edges=self.graph_spec.resource_edge_index,
            capacity_edges=self.graph_spec.capacity_edge_index,
            resource_edge_features=self.graph_spec.resource_edge_features,
            capacity_edge_features=self.graph_spec.capacity_edge_features,
        ).to(self.device)
        self.critic = GCNCritic(
            self.graph_spec.node_feature_dim,
            self.graph_spec.num_facilities,
            self.graph_spec.num_nodes,
            action_dim,
            self.graph_spec.edge_index,
            gcn_hidden_sizes,
            critic_hidden_sizes,
            include_global_context=include_global_context,
            edge_weights=self.graph_spec.edge_weights,
        ).to(self.device)
        self.critic_target = GCNCritic(
            self.graph_spec.node_feature_dim,
            self.graph_spec.num_facilities,
            self.graph_spec.num_nodes,
            action_dim,
            self.graph_spec.edge_index,
            gcn_hidden_sizes,
            critic_hidden_sizes,
            include_global_context=include_global_context,
            edge_weights=self.graph_spec.edge_weights,
        ).to(self.device)
        if self.residual_action_enabled and bool(residual_config.get("zero_init_actor", False)):
            self._zero_initialize_actor_output(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(config.get("actor_lr", 1e-4)))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(config.get("critic_lr", 1e-3)))
        if self.correction_gate_enabled:
            self.correction_gate = GCNCorrectionGate(
                self.graph_spec.node_feature_dim,
                self.graph_spec.num_facilities,
                self.graph_spec.num_nodes,
                self.graph_spec.edge_index,
                gcn_hidden_sizes,
                tuple(correction_gate_config.get("hidden_sizes", (64, 32))),
                include_global_context=include_global_context,
                edge_weights=self.graph_spec.edge_weights,
                output_dim=self.correction_gate_output_dim,
            ).to(self.device)
            self.correction_gate_optimizer = torch.optim.Adam(
                self.correction_gate.parameters(),
                lr=float(correction_gate_config.get("lr", 3e-4)),
            )
        else:
            self.correction_gate = None
            self.correction_gate_optimizer = None
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
            sigma=float(exploration.get("sigma", exploration.get("initial_sigma", 0.2))),
        )

    def reset(self) -> None:
        self.noise.reset()

    def select_action(self, state: np.ndarray, explore: bool = True, env=None) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            node_features = flat_state_to_node_features(state_tensor, self.graph_spec)
            network_action = self.actor(node_features).cpu().numpy()[0]
        self.actor.train()
        if explore:
            network_action = network_action + self.noise.sample()
        # The correction gate is a deployment safety layer. Exploration must
        # reach the environment so the residual actor and critic can learn.
        action = self._compose_action_np(
            state,
            network_action,
            apply_correction_gate=not explore,
        )
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
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        node_features = flat_state_to_node_features(states, self.graph_spec)
        next_node_features = flat_state_to_node_features(next_states, self.graph_spec)
        with torch.no_grad():
            next_network_actions = self.actor_target(next_node_features)
            next_actions = self._compose_actions_tensor(
                next_states,
                next_network_actions,
                apply_correction_gate=False,
            )
            target_q = self.critic_target(next_node_features, next_actions)
            q_targets = rewards + self.gamma * (1.0 - dones) * target_q

        q_expected = self.critic(node_features, actions)
        critic_loss = torch.nn.functional.mse_loss(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        network_actions = self.actor(node_features)
        actor_actions = self._compose_actions_tensor(
            states,
            network_actions,
            apply_correction_gate=False,
        )
        actor_loss, advantage_metrics = self._actor_objective(
            node_features,
            states,
            actor_actions,
        )
        residual_l2_loss_value = None
        if self.residual_action_enabled and self.residual_l2_weight > 0.0:
            policy_residuals = self._policy_residuals_tensor(
                states,
                network_actions,
                apply_correction_gate=False,
            )
            residual_l2_loss = policy_residuals.pow(2).mean()
            actor_loss = actor_loss + self.residual_l2_weight * residual_l2_loss
            residual_l2_loss_value = float(residual_l2_loss.item())
        elif self.patient_service_proxy_actor_loss_enabled:
            policy_residuals = self._policy_residuals_tensor(
                states,
                network_actions,
                apply_correction_gate=False,
            )
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
        self._soft_update(self.critic, self.critic_target)
        metrics = {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item())}
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

        low_pressure_penalty = torch.zeros(
            (),
            dtype=residuals.dtype,
            device=residuals.device,
        )
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
            return -self.critic(node_features, actor_actions).mean(), {}

        actor_q = self.critic(node_features, actor_actions)
        with torch.no_grad():
            anchor_actions = self._base_actions_from_states_tensor(states)
            anchor_q = self.critic(node_features, anchor_actions)
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

    def _actor_imitation_loss(self):
        sample_count = int(self.imitation_states.shape[0])
        batch_size = min(max(self.imitation_regularization_batch_size, 1), sample_count)
        indices = self.imitation_rng.choice(sample_count, size=batch_size, replace=False)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        if self.imitation_node_features is None:
            node_features = flat_state_to_node_features(
                self.imitation_states[index_tensor],
                self.graph_spec,
            )
        else:
            node_features = self.imitation_node_features[index_tensor]
        batch_weights = (
            None
            if self.imitation_weights is None
            else self.imitation_weights[index_tensor]
        )
        return self._supervised_action_loss(
            self.imitation_states[index_tensor],
            self.actor(node_features),
            self.imitation_actions[index_tensor],
            batch_weights,
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

        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        sample_count = int(state_tensor.shape[0])
        batch_size = min(max(batch_size, 1), sample_count)
        generator = torch.Generator().manual_seed(seed)
        final_loss = 0.0

        self.actor.train()
        for _epoch in range(epochs):
            permutation = torch.randperm(sample_count, generator=generator)
            for start in range(0, sample_count, batch_size):
                indices = permutation[start : start + batch_size].to(self.device)
                node_features = flat_state_to_node_features(state_tensor[indices], self.graph_spec)
                loss = self._supervised_action_loss(
                    state_tensor[indices],
                    self.actor(node_features),
                    action_tensor[indices],
                )
                self.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
                self.actor_optimizer.step()
                final_loss = float(loss.item())

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.imitation_states = state_tensor.detach()
        self.imitation_actions = action_tensor.detach()
        with torch.no_grad():
            self.imitation_node_features = flat_state_to_node_features(
                self.imitation_states,
                self.graph_spec,
            ).detach()
        self.imitation_weights = None
        self.imitation_rng = np.random.default_rng(seed + 300000)
        return {"policy": policy_name, "samples": sample_count, "final_loss": final_loss}

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
            node_feature_tensor = flat_state_to_node_features(state_tensor, self.graph_spec).detach()
            residual_target_tensor = None
            residual_mask_tensor = None
            correction_gate_labels = None
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
                if (
                    self.correction_gate_enabled
                    and self.correction_gate_mode == "classification"
                ):
                    gate_mask = residual_mask_tensor
                    if gate_mask.shape[0] == 1:
                        gate_mask = gate_mask.expand(sample_count, -1)
                    correction_gate_labels = self._correction_gate_labels_tensor(
                        residual_target_tensor,
                        gate_mask,
                    )
        generator = torch.Generator().manual_seed(seed)
        final_loss = 0.0
        final_gate_loss = 0.0

        self.actor.train()
        if self.correction_gate is not None:
            self.correction_gate.train()
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
                        apply_correction_gate=False,
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
                if correction_gate_labels is not None:
                    gate_logits = self.correction_gate(node_features)
                    gate_per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                        gate_logits,
                        correction_gate_labels[indices],
                        reduction="none",
                    )
                    if gate_per_sample.ndim > 1:
                        gate_per_sample = gate_per_sample.mean(dim=1)
                    if batch_weights is None:
                        gate_loss = gate_per_sample.mean()
                    else:
                        gate_loss = (
                            gate_per_sample * batch_weights
                        ).sum() / batch_weights.sum().clamp_min(1e-8)
                    self.correction_gate_optimizer.zero_grad()
                    (self.correction_gate_loss_weight * gate_loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.correction_gate.parameters(),
                        max_norm=5.0,
                    )
                    self.correction_gate_optimizer.step()
                    final_gate_loss = float(gate_loss.item())

        self.actor_target.load_state_dict(self.actor.state_dict())
        if bool(settings.get("retain_for_regularization", False)):
            self.imitation_states = state_tensor.detach()
            self.imitation_actions = action_tensor.detach()
            self.imitation_node_features = node_feature_tensor.detach()
            self.imitation_weights = (
                None if weight_tensor is None else weight_tensor.detach()
            )
            self.imitation_rng = np.random.default_rng(seed + 300000)
        summary = {
            "samples": sample_count,
            "final_loss": final_loss,
            "target_mode": target_mode,
        }
        if correction_gate_labels is not None:
            self.correction_gate.eval()
            with torch.no_grad():
                gate_probabilities = torch.sigmoid(
                    self.correction_gate(node_feature_tensor)
                )
                gate_predictions = (
                    gate_probabilities
                    >= self._correction_gate_threshold_tensor(
                        gate_probabilities,
                    )
                ).to(dtype=torch.float32)
                positives = correction_gate_labels > 0.5
                predicted_positives = gate_predictions > 0.5
                true_positives = positives & predicted_positives
                summary.update(
                    {
                        "correction_gate_loss": final_gate_loss,
                        "correction_gate_threshold": self.correction_gate_threshold,
                        "correction_gate_accuracy": float(
                            (gate_predictions == correction_gate_labels)
                            .to(dtype=torch.float32)
                            .mean()
                            .item()
                        ),
                        "correction_gate_recall": float(
                            true_positives.sum().item()
                            / max(int(positives.sum().item()), 1)
                        ),
                        "correction_gate_precision": float(
                            true_positives.sum().item()
                            / max(int(predicted_positives.sum().item()), 1)
                        ),
                        "correction_gate_label_rate": float(
                            positives.to(dtype=torch.float32).mean().item()
                        ),
                        "correction_gate_prediction_fraction": float(
                            predicted_positives.to(dtype=torch.float32).mean().item()
                        ),
                        "correction_gate_groups": "|".join(
                            self.correction_gate_groups
                        ),
                        "correction_gate_group_label_rates": self._pipe_group_rates(
                            positives,
                        ),
                        "correction_gate_group_prediction_rates": self._pipe_group_rates(
                            predicted_positives,
                        ),
                    }
                )
        return summary

    def _correction_gate_labels_tensor(self, residual_targets, residual_mask):
        magnitudes = residual_targets.abs() * residual_mask
        if not self.correction_gate_groups:
            return (
                magnitudes.amax(dim=1)
                > self.correction_gate_target_delta
            ).to(dtype=torch.float32)
        group_slices = self._facility_net_group_slices(
            self.graph_spec.num_facilities
        )
        return torch.stack(
            [
                (
                    magnitudes[:, group_slices[group]].amax(dim=1)
                    > self.correction_gate_target_delta
                ).to(dtype=torch.float32)
                for group in self.correction_gate_groups
            ],
            dim=1,
        )

    def _pipe_group_rates(self, values) -> str:
        if values.ndim == 1:
            return f"{float(values.to(dtype=torch.float32).mean().item()):.8g}"
        return "|".join(
            f"{float(values[:, index].to(dtype=torch.float32).mean().item()):.8g}"
            for index in range(values.shape[1])
        )

    def select_ungated_residual_action(
        self,
        state: np.ndarray,
        *,
        env=None,
        residual_scale: float = 1.0,
    ) -> np.ndarray:
        """Return the actor correction without the learned accept/reject gate."""

        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            node_features = flat_state_to_node_features(
                state_tensor,
                self.graph_spec,
            )
            network_action = self.actor(node_features)
            residual = self._policy_residuals_tensor(
                state_tensor,
                network_action,
                apply_correction_gate=False,
            )
            base_action = self._base_actions_from_states_tensor(state_tensor)
            scale = torch.as_tensor(
                self.residual_scale_vector,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            action = torch.clamp(
                base_action + float(residual_scale) * scale * residual,
                -1.0,
                1.0,
            )
        return project_action(
            action.cpu().numpy()[0],
            env_state=env,
            action_space_info=self.action_dim,
        ).action

    def fit_correction_gate_batch(
        self,
        states: np.ndarray,
        labels: np.ndarray,
        settings: dict[str, Any] | None = None,
        weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Fit the state-level gate to actor-specific CRN advantage labels."""

        if self.correction_gate is None or self.correction_gate_optimizer is None:
            raise ValueError("correction_gate must be enabled before fitting")
        settings = settings or {}
        state_array = np.asarray(states, dtype=np.float32)
        sample_count = int(state_array.shape[0])
        label_array = self._coerce_correction_gate_array(
            labels,
            sample_count,
            name="labels",
            dtype=np.float32,
        )
        if state_array.ndim != 2 or state_array.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected states shape (batch, {self.state_dim}), "
                f"got {state_array.shape}"
            )
        if not np.all(np.isfinite(label_array)) or np.any(
            (label_array != 0.0) & (label_array != 1.0)
        ):
            raise ValueError("Correction gate labels must be binary and finite")
        if sample_count == 0:
            return {"samples": 0, "final_loss": 0.0}
        state_tensor = torch.as_tensor(
            state_array,
            dtype=torch.float32,
            device=self.device,
        )
        label_tensor = torch.as_tensor(
            label_array,
            dtype=torch.float32,
            device=self.device,
        )
        mode = str(settings.get("mode", self.correction_gate_mode))
        if mode not in ("classification", "advantage"):
            raise ValueError("Correction gate fit mode must be classification or advantage")
        self.correction_gate_mode = mode
        target_tensor = label_tensor
        if mode == "advantage":
            if settings.get("advantages") is None:
                raise ValueError("Advantage gate fitting requires advantages")
            advantage_array = self._coerce_correction_gate_array(
                settings["advantages"],
                sample_count,
                name="advantages",
                dtype=np.float32,
            )
            if not np.all(np.isfinite(advantage_array)):
                raise ValueError("Gate advantages must be finite and align with states")
            scaled_targets = np.clip(
                advantage_array / self.correction_gate_advantage_scale,
                -self.correction_gate_advantage_clip,
                self.correction_gate_advantage_clip,
            )
            feasible = self._coerce_correction_gate_array(
                settings.get("feasible", np.ones(sample_count, dtype=bool)),
                sample_count,
                name="feasibility",
                dtype=bool,
            )
            scaled_targets[~feasible] = np.minimum(
                scaled_targets[~feasible],
                -abs(self.correction_gate_threshold),
            )
            target_tensor = torch.as_tensor(
                scaled_targets,
                dtype=torch.float32,
                device=self.device,
            )
        node_features = flat_state_to_node_features(
            state_tensor,
            self.graph_spec,
        ).detach()
        weight_tensor = self._fit_action_weights(weights, sample_count)
        epochs = max(int(settings.get("epochs", 1)), 1)
        batch_size = min(
            max(int(settings.get("batch_size", self.batch_size)), 1),
            sample_count,
        )
        generator = torch.Generator().manual_seed(
            int(settings.get("seed", self.seed + 450000))
        )
        final_loss = 0.0
        self.correction_gate.train()
        for _epoch in range(epochs):
            permutation = torch.randperm(sample_count, generator=generator)
            for start in range(0, sample_count, batch_size):
                indices = permutation[start : start + batch_size].to(self.device)
                logits = self.correction_gate(node_features[indices])
                if mode == "classification":
                    per_sample = (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            logits,
                            label_tensor[indices],
                            reduction="none",
                        )
                    )
                else:
                    per_sample = torch.nn.functional.smooth_l1_loss(
                        logits,
                        target_tensor[indices],
                        reduction="none",
                    )
                if per_sample.ndim > 1:
                    per_sample = per_sample.mean(dim=1)
                batch_weights = (
                    None if weight_tensor is None else weight_tensor[indices]
                )
                loss = (
                    per_sample.mean()
                    if batch_weights is None
                    else (per_sample * batch_weights).sum()
                    / batch_weights.sum().clamp_min(1e-8)
                )
                self.correction_gate_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.correction_gate.parameters(),
                    max_norm=5.0,
                )
                self.correction_gate_optimizer.step()
                final_loss = float(loss.item())
        self.correction_gate.eval()
        with torch.no_grad():
            raw_scores = self.correction_gate(node_features)
            probabilities = (
                torch.sigmoid(raw_scores)
                if mode == "classification"
                else raw_scores
            )
            predictions = (
                probabilities
                >= self._correction_gate_threshold_tensor(probabilities)
            ).to(dtype=torch.float32)
            positives = label_tensor > 0.5
            predicted_positives = predictions > 0.5
            true_positives = positives & predicted_positives
        return {
            "samples": sample_count,
            "final_loss": final_loss,
            "mode": mode,
            "threshold": self.correction_gate_threshold,
            "label_rate": float(positives.to(dtype=torch.float32).mean().item()),
            "prediction_rate": float(
                predicted_positives.to(dtype=torch.float32).mean().item()
            ),
            "accuracy": float(
                (predictions == label_tensor).to(dtype=torch.float32).mean().item()
            ),
            "precision": float(
                true_positives.sum().item()
                / max(int(predicted_positives.sum().item()), 1)
            ),
            "recall": float(
                true_positives.sum().item() / max(int(positives.sum().item()), 1)
            ),
            "groups": "|".join(self.correction_gate_groups),
            "group_label_rates": self._pipe_group_rates(positives),
            "group_prediction_rates": self._pipe_group_rates(
                predicted_positives
            ),
        }

    def _coerce_correction_gate_array(
        self,
        values,
        sample_count: int,
        *,
        name: str,
        dtype,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=dtype)
        if self.correction_gate_output_dim == 1:
            array = array.reshape(-1)
            if array.shape != (sample_count,):
                raise ValueError(
                    f"Correction gate {name} must align with states"
                )
            return array
        if array.shape == (sample_count,):
            return np.repeat(
                array[:, None],
                self.correction_gate_output_dim,
                axis=1,
            )
        expected = (sample_count, self.correction_gate_output_dim)
        if array.shape != expected:
            raise ValueError(
                f"Correction gate {name} must have shape {expected}"
            )
        return array

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
        predicted_residuals = self._policy_residuals_tensor(
            states,
            network_actions,
            apply_correction_gate=False,
        )
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

    def _compose_action_np(
        self,
        state: np.ndarray,
        network_action: np.ndarray,
        *,
        apply_correction_gate: bool = True,
    ) -> np.ndarray:
        if not self.residual_action_enabled:
            return np.asarray(network_action, dtype=np.float32)
        base_action = self._base_action_from_state_np(state)
        residual_action = self._policy_residual_np(
            state,
            network_action,
            apply_correction_gate=apply_correction_gate,
        )
        return np.clip(
            base_action + self.residual_scale_vector * residual_action,
            -1.0,
            1.0,
        ).astype(np.float32)

    def _compose_actions_tensor(
        self,
        states,
        network_actions,
        *,
        apply_correction_gate: bool = True,
    ):
        if not self.residual_action_enabled:
            return network_actions
        base_actions = self._base_actions_from_states_tensor(states)
        residual_actions = self._policy_residuals_tensor(
            states,
            network_actions,
            apply_correction_gate=apply_correction_gate,
        )
        scale = torch.as_tensor(
            self.residual_scale_vector,
            dtype=network_actions.dtype,
            device=network_actions.device,
        )
        return torch.clamp(base_actions + scale * residual_actions, -1.0, 1.0)

    def _policy_residual_np(
        self,
        state: np.ndarray,
        network_action: np.ndarray,
        *,
        apply_correction_gate: bool = True,
    ) -> np.ndarray:
        if (
            not self.residual_pressure_projection_groups
            and not self.residual_state_gate_groups
            and (self.correction_gate is None or not apply_correction_gate)
        ):
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
            residual = self._policy_residuals_tensor(
                state_tensor,
                action_tensor,
                apply_correction_gate=apply_correction_gate,
                hard_correction_gate=True,
            )
        return residual.cpu().numpy()[0].astype(np.float32)

    def _policy_residuals_tensor(
        self,
        states,
        network_actions,
        *,
        apply_correction_gate: bool = True,
        hard_correction_gate: bool = True,
    ):
        residuals = self._transform_network_residuals_tensor(network_actions)
        if self.residual_pressure_projection_groups:
            residuals = self._project_residuals_to_pressure_patterns(states, residuals)
            residuals = self._apply_positive_residual_slices_tensor(residuals)
        residuals = self._apply_state_gate_residuals_tensor(states, residuals)
        if apply_correction_gate and self.correction_gate is not None:
            gate_scores = self.correction_gate(
                flat_state_to_node_features(states, self.graph_spec)
            ).detach()
            gate_probabilities = (
                torch.sigmoid(gate_scores)
                if self.correction_gate_mode == "classification"
                else gate_scores
            )
            if hard_correction_gate:
                gate_probabilities = (
                    gate_probabilities
                    >= self._correction_gate_threshold_tensor(
                        gate_probabilities,
                    )
                ).to(dtype=residuals.dtype)
            elif self.correction_gate_mode == "advantage":
                gate_probabilities = torch.sigmoid(
                    gate_probabilities - self.correction_gate_threshold
                )
            if gate_probabilities.ndim == 1:
                residuals = residuals * gate_probabilities.unsqueeze(1)
            else:
                residuals = self._apply_group_gate_probabilities(
                    residuals,
                    gate_probabilities,
                )
        return torch.clamp(residuals, -1.0, 1.0)

    def _correction_gate_threshold_tensor(self, gate_values):
        if gate_values.ndim == 1 or not self.correction_gate_group_thresholds:
            return torch.as_tensor(
                self.correction_gate_threshold,
                dtype=gate_values.dtype,
                device=gate_values.device,
            )
        return torch.as_tensor(
            self.correction_gate_group_thresholds,
            dtype=gate_values.dtype,
            device=gate_values.device,
        ).reshape(1, -1)

    def _apply_group_gate_probabilities(self, residuals, probabilities):
        if probabilities.shape[1] != len(self.correction_gate_groups):
            raise ValueError(
                "Correction-gate outputs do not match configured action groups"
            )
        gated = residuals
        group_slices = self._facility_net_group_slices(
            self.graph_spec.num_facilities
        )
        for index, group in enumerate(self.correction_gate_groups):
            group_slice = group_slices[group]
            gated = self._replace_action_slice_tensor(
                gated,
                group_slice,
                gated[:, group_slice] * probabilities[:, index : index + 1],
            )
        return gated

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
                dim=1, keepdim=True
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
            projected = self._replace_action_slice_tensor(
                projected,
                group_slice,
                project_tensor_to_pattern_basis(
                    current,
                    pattern,
                    include_uniform=(
                        self.residual_replenishment_uniform_basis
                        and group in ("replenishment", "purchase")
                    ),
                ),
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
        include_history = int(
            bool(self.env_config.get("include_demand_history_state", False))
        )
        include_transfer_pipeline = int(
            bool(self.env_config.get("include_transfer_pipeline_state", False))
        )
        features_per_facility = 3 + lead_time + include_supplier + include_forecast
        features_per_facility += 3 * include_transfer_pipeline
        features_per_facility += 3 * include_history
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
        pending_reagents = torch.zeros_like(demand)
        pending_capacity = torch.zeros_like(demand)
        if include_transfer_pipeline:
            pending_start = 3 + lead_time + include_supplier + include_forecast
            pending_reagents = facility_state[:, :, pending_start + 1]
            pending_capacity = facility_state[:, :, pending_start + 2]
        risk = self._patient_risk_signal_tensor(states, features_per_facility)
        resource_pressure = (
            demand
            + 0.25 * forecast
            + specimens
            - reagents
            - pending_reagents
            + 0.5 * risk
        )
        capacity_pressure = (
            demand
            + 0.25 * forecast
            + specimens
            - idle_bioreactors
            - pending_capacity
            + 0.5 * risk
        )
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
        summary_width = 6 + len(summary_edges) + 1
        base_width = n * int(features_per_facility)
        expected_width = base_width + n * summary_width
        if states.shape[1] < expected_width:
            return torch.zeros((states.shape[0], n), dtype=states.dtype, device=states.device)
        summary = states[:, base_width:expected_width].reshape(states.shape[0], n, summary_width)
        near_expiry = summary[:, :, 2]
        patient_config = dict(self.env_config.get("patient", {}))
        risk_threshold = float(patient_config.get("eligibility_threshold", 0.80)) + float(
            self.env_config.get("urgency_margin", 0.10)
        )
        at_risk_buckets = sum(
            float(edge) <= risk_threshold + 1e-8 for edge in summary_edges
        )
        waiting_histogram = summary[:, :, 6:]
        waiting_at_risk = (
            waiting_histogram[:, :, :at_risk_buckets].sum(dim=2)
            if at_risk_buckets > 0
            else torch.zeros_like(near_expiry)
        )
        return near_expiry + waiting_at_risk

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
        """Make a residual actor start as the heuristic anchor (zero correction)."""

        if hasattr(actor, "zero_initialize_output_heads"):
            actor.zero_initialize_output_heads()
            return
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
                "graph_spec": asdict(self.graph_spec),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "correction_gate": (
                    None
                    if self.correction_gate is None
                    else self.correction_gate.state_dict()
                ),
                "correction_gate_mode": self.correction_gate_mode,
                "correction_gate_threshold": self.correction_gate_threshold,
                "correction_gate_group_thresholds": (
                    self.correction_gate_group_thresholds
                ),
            },
            output_path,
        )

    def load_actor(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        if (
            self.correction_gate is not None
            and checkpoint.get("correction_gate") is not None
        ):
            self.correction_gate.load_state_dict(checkpoint["correction_gate"])
            self.correction_gate.eval()
            self.correction_gate_mode = str(
                checkpoint.get(
                    "correction_gate_mode",
                    self.correction_gate_mode,
                )
            )
            self.correction_gate_threshold = float(
                checkpoint.get(
                    "correction_gate_threshold",
                    self.correction_gate_threshold,
                )
            )
            saved_group_thresholds = checkpoint.get(
                "correction_gate_group_thresholds"
            )
            if saved_group_thresholds is not None:
                saved_group_thresholds = tuple(
                    float(value) for value in saved_group_thresholds
                )
                if len(saved_group_thresholds) == len(
                    self.correction_gate_groups
                ):
                    self.correction_gate_group_thresholds = (
                        saved_group_thresholds
                    )

    def warm_start_actor(self, path: str | Path) -> dict[str, list[str]]:
        """Curriculum warm-start from a (possibly smaller-network) checkpoint. With
        ``actor_readout_mode='facility_action'`` the full policy transfers; otherwise
        only the size-invariant encoder does. Returns the transferred/skipped summary."""

        checkpoint = torch.load(path, map_location=self.device)
        summary = transfer_matching_parameters(checkpoint["actor"], self.actor)
        if (
            self.correction_gate is not None
            and checkpoint.get("correction_gate") is not None
        ):
            gate_summary = transfer_matching_parameters(
                checkpoint["correction_gate"],
                self.correction_gate,
            )
            summary = {
                "transferred": [
                    *summary["transferred"],
                    *(f"correction_gate.{key}" for key in gate_summary["transferred"]),
                ],
                "skipped": [
                    *summary["skipped"],
                    *(f"correction_gate.{key}" for key in gate_summary["skipped"]),
                ],
            }
        self.actor_target.load_state_dict(self.actor.state_dict())
        return summary

    def _soft_update(self, local_model, target_model) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
