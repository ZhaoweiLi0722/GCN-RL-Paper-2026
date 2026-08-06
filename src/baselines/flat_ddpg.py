"""Flat-state / MLP-DDPG baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.heuristics import facility_net_action_from_state, heuristic_settings_for_policy
from src.models.graph_features import build_graph_spec
from src.models.temporal import (
    TemporalFlatActor,
    TemporalFlatCorrectionGate,
    TemporalFlatCritic,
)
from src.rl.action_projection import project_action, project_tensor_to_pattern_basis
from src.rl.networks import (
    MLPActor,
    MLPCorrectionGate,
    MLPCritic,
    require_torch,
    resolve_torch_device,
    torch,
)
from src.rl.noise import OUNoise
from src.rl.preprocessing import (
    FixedObservationScaler,
    facility_state_width,
    reward_scale_from_config,
)
from src.rl.replay_buffer import ReplayBuffer
from src.rl.residual_endpoint_projection import ResidualEndpointProjection
from src.rl.residual_temporal_guard import ResidualTemporalGuard


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
        self.actor_update_frequency = max(
            int(config.get("actor_update_frequency", 1)),
            1,
        )
        self.critic_warmup_updates = max(
            int(config.get("critic_warmup_updates", 0)),
            0,
        )
        self.total_updates = 0
        self.reward_scale = reward_scale_from_config(config)
        self.observation_scaler = FixedObservationScaler.from_config(config, state_dim)
        residual_config = dict(config.get("residual_action", {}))
        self.residual_action_enabled = bool(residual_config.get("enabled", False))
        self.include_base_action_features = bool(
            residual_config.get("include_base_action_features", False)
        )
        self.residual_scale = float(residual_config.get("scale", 0.25))
        self.residual_scale_vector = self._make_residual_scale_vector(residual_config)
        self.residual_center_slices = self._make_residual_center_slices(residual_config)
        self.residual_positive_slices = self._make_residual_positive_slices(residual_config)
        self.residual_endpoint_projection = ResidualEndpointProjection(
            num_facilities=int(self.env_config.get("num_facilities", 0)),
            action_dim=self.action_dim,
            settings=residual_config.get("endpoint_projection", {}),
        )
        self.residual_pressure_projection_groups = (
            self._make_pressure_projection_groups(residual_config)
        )
        pressure_projection_config = dict(
            residual_config.get("pressure_projection", {})
        )
        self.residual_replenishment_uniform_basis = bool(
            pressure_projection_config.get("replenishment_uniform_basis", False)
        )
        self.residual_pressure_projection_subtract_pipeline = bool(
            pressure_projection_config.get("subtract_pipeline", True)
        )
        self.residual_pressure_projection_coefficient_mode = str(
            pressure_projection_config.get(
                "coefficient_mode",
                "least_squares",
            )
        )
        if self.residual_pressure_projection_coefficient_mode not in (
            "least_squares",
            "mean",
        ):
            raise ValueError(
                "residual_action.pressure_projection.coefficient_mode must be "
                "'least_squares' or 'mean'"
            )
        self.residual_pressure_projection_positive_coefficient = bool(
            pressure_projection_config.get(
                "positive_coefficient",
                False,
            )
        )
        if (
            self.residual_pressure_projection_coefficient_mode == "mean"
            and self.residual_replenishment_uniform_basis
        ):
            raise ValueError(
                "mean pressure coefficients cannot be combined with "
                "replenishment_uniform_basis"
            )
        self.residual_state_gate_config = dict(residual_config.get("state_gate", {}))
        self.residual_state_gate_groups = self._make_residual_state_gate_groups(residual_config)
        self.residual_l2_weight = float(residual_config.get("l2_weight", 0.0))
        sparse_target_config = dict(
            residual_config.get("supervised_sparse_target", {})
        )
        self.supervised_sparse_target_enabled = bool(
            sparse_target_config.get("enabled", False)
        )
        self.supervised_sparse_target_threshold = max(
            float(sparse_target_config.get("change_threshold", 1e-6)),
            0.0,
        )
        self.supervised_sparse_target_changed_weight = max(
            float(sparse_target_config.get("changed_weight", 1.0)),
            0.0,
        )
        self.supervised_sparse_target_zero_weight = max(
            float(sparse_target_config.get("zero_weight", 0.05)),
            0.0,
        )
        self.supervised_support_ranking_weight = max(
            float(
                sparse_target_config.get(
                    "support_ranking_weight",
                    0.0,
                )
            ),
            0.0,
        )
        self.supervised_support_ranking_margin = max(
            float(
                sparse_target_config.get(
                    "support_ranking_margin",
                    0.02,
                )
            ),
            0.0,
        )
        if (
            self.supervised_sparse_target_enabled
            and self.supervised_sparse_target_changed_weight <= 0.0
            and self.supervised_sparse_target_zero_weight <= 0.0
        ):
            raise ValueError(
                "supervised_sparse_target requires a positive dimension weight"
            )
        correction_gate_config = dict(residual_config.get("correction_gate", {}))
        self.correction_gate_enabled = bool(
            self.residual_action_enabled
            and correction_gate_config.get("enabled", False)
        )
        self.correction_gate_during_exploration = bool(
            correction_gate_config.get(
                "apply_during_exploration",
                False,
            )
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
        self.correction_gate_positive_class_weight_power = max(
            float(
                correction_gate_config.get(
                    "positive_class_weight_power",
                    0.0,
                )
            ),
            0.0,
        )
        self.correction_gate_positive_class_weight_max = max(
            float(
                correction_gate_config.get(
                    "positive_class_weight_max",
                    10.0,
                )
            ),
            1.0,
        )
        self.correction_gate_negative_class_weight = max(
            float(
                correction_gate_config.get(
                    "negative_class_weight",
                    1.0,
                )
            ),
            0.0,
        )
        self.correction_gate_groups = tuple(
            str(group)
            for group in correction_gate_config.get("groups", ())
        )
        valid_gate_groups = self._facility_net_group_slices(
            int(self.env_config.get("num_facilities", 0))
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
        temporal_guard_config = dict(
            residual_config.get("temporal_guard", {})
        )
        if (
            temporal_guard_config.get("enabled", False)
            and not self.residual_action_enabled
        ):
            raise ValueError(
                "residual_action.temporal_guard requires residual_action.enabled"
            )
        self.residual_temporal_guard = ResidualTemporalGuard(
            num_facilities=int(
                self.env_config.get("num_facilities", 0)
            ),
            action_dim=self.action_dim,
            env_config=self.env_config,
            settings=temporal_guard_config,
        )
        self.last_residual_guard_info = dict(
            self.residual_temporal_guard.last_info
        )
        if self.include_base_action_features and not self.residual_action_enabled:
            raise ValueError(
                "residual_action.include_base_action_features requires residual_action.enabled"
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
        self.imitation_weights = None
        self.imitation_rng = np.random.default_rng(seed + 300000)
        advantage_config = dict(config.get("anchor_advantage_actor_loss", {}))
        self.anchor_advantage_actor_loss_enabled = bool(
            advantage_config.get("enabled", False)
        )
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
        hidden_sizes = tuple(config.get("hidden_sizes", [256, 256]))
        include_global_context = bool(
            config.get("include_global_context", True)
        )
        actor_readout_mode = str(
            config.get("actor_readout_mode", "global_flat")
        )
        self.device = resolve_torch_device(config.get("device"))
        temporal_encoder_config = dict(
            config.get("temporal_demand_encoder", {})
        )
        self.temporal_demand_encoder_enabled = bool(
            temporal_encoder_config.get("enabled", False)
        )
        self.temporal_demand_hidden_size = (
            int(temporal_encoder_config.get("hidden_size", 16))
            if self.temporal_demand_encoder_enabled
            else 0
        )
        self.temporal_graph_spec = (
            build_graph_spec(config, state_dim)
            if self.temporal_demand_encoder_enabled
            else None
        )
        if (
            self.temporal_demand_encoder_enabled
            and not self.temporal_graph_spec.include_demand_sequence_state
        ):
            raise ValueError(
                "temporal_demand_encoder requires "
                "env.include_demand_sequence_state=true"
            )
        if (
            self.temporal_demand_encoder_enabled
            and actor_readout_mode == "pressure_intensity"
            and self.residual_pressure_projection_coefficient_mode != "mean"
        ):
            raise ValueError(
                "pressure_intensity actor readout requires "
                "pressure_projection.coefficient_mode='mean'"
            )
        if (
            self.temporal_demand_encoder_enabled
            and actor_readout_mode == "pressure_intensity"
            and self.residual_center_slices
        ):
            raise ValueError(
                "pressure_intensity actor readout requires "
                "residual_action.center_groups=[]"
            )

        actor_input_dim = state_dim + action_dim if self.include_base_action_features else state_dim
        if self.temporal_demand_encoder_enabled:
            self.actor = TemporalFlatActor(
                self.temporal_graph_spec,
                action_dim,
                self.temporal_demand_hidden_size,
                hidden_sizes,
                include_global_context=include_global_context,
                readout_mode=actor_readout_mode,
            ).to(self.device)
            self.actor_target = TemporalFlatActor(
                self.temporal_graph_spec,
                action_dim,
                self.temporal_demand_hidden_size,
                hidden_sizes,
                include_global_context=include_global_context,
                readout_mode=actor_readout_mode,
            ).to(self.device)
            self.critic = TemporalFlatCritic(
                self.temporal_graph_spec,
                action_dim,
                self.temporal_demand_hidden_size,
                hidden_sizes,
                include_global_context=include_global_context,
            ).to(self.device)
            self.critic_target = TemporalFlatCritic(
                self.temporal_graph_spec,
                action_dim,
                self.temporal_demand_hidden_size,
                hidden_sizes,
                include_global_context=include_global_context,
            ).to(self.device)
        else:
            self.actor = MLPActor(actor_input_dim, action_dim, hidden_sizes).to(self.device)
            self.actor_target = MLPActor(actor_input_dim, action_dim, hidden_sizes).to(self.device)
            self.critic = MLPCritic(state_dim, action_dim, hidden_sizes).to(self.device)
            self.critic_target = MLPCritic(state_dim, action_dim, hidden_sizes).to(self.device)
        if self.residual_action_enabled and bool(residual_config.get("zero_init_actor", False)):
            self._zero_initialize_actor_output(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(config.get("actor_lr", 1e-4)))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(config.get("critic_lr", 1e-3)))
        if self.correction_gate_enabled:
            if self.temporal_demand_encoder_enabled:
                self.correction_gate = TemporalFlatCorrectionGate(
                    self.temporal_graph_spec,
                    self.temporal_demand_hidden_size,
                    tuple(
                        correction_gate_config.get(
                            "hidden_sizes",
                            (64, 32),
                        )
                    ),
                    self.correction_gate_output_dim,
                    include_global_context=include_global_context,
                ).to(self.device)
            else:
                self.correction_gate = MLPCorrectionGate(
                    actor_input_dim,
                    tuple(correction_gate_config.get("hidden_sizes", (64, 32))),
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
            sigma=float(exploration.get("sigma", 0.2)),
        )

    def reset(self) -> None:
        self.noise.reset()
        self.residual_temporal_guard.reset()
        self.last_residual_action = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )
        self.last_residual_guard_info = dict(
            self.residual_temporal_guard.last_info
        )

    def select_action(self, state: np.ndarray, explore: bool = True, env=None) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            actor_inputs = self._actor_input_tensor(state_tensor)
            action = self.actor(actor_inputs).cpu().numpy()[0]
        self.actor.train()
        if explore:
            action = action + self.noise.sample()
        action = self._compose_action_np(
            state,
            action,
            apply_correction_gate=(
                not explore
                or self.correction_gate_during_exploration
            ),
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

        self.total_updates += 1
        batch = self.replay_buffer.sample(self.batch_size)
        raw_states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        raw_next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        states = (
            raw_states
            if self.temporal_demand_encoder_enabled
            else self.observation_scaler.normalize_tensor(raw_states)
        )
        next_states = (
            raw_next_states
            if self.temporal_demand_encoder_enabled
            else self.observation_scaler.normalize_tensor(raw_next_states)
        )
        actor_inputs = self._actor_input_tensor(raw_states, normalized_states=states)
        next_actor_inputs = self._actor_input_tensor(
            raw_next_states,
            normalized_states=next_states,
        )

        with torch.no_grad():
            next_network_actions = self.actor_target(next_actor_inputs)
            next_actions = self._compose_actions_tensor(raw_next_states, next_network_actions)
            target_q = self.critic_target(next_states, next_actions)
            q_targets = rewards + self.gamma * (1.0 - dones) * target_q

        q_expected = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_update_due = (
            self.total_updates > self.critic_warmup_updates
            and (
                self.total_updates - self.critic_warmup_updates
            )
            % self.actor_update_frequency
            == 0
        )
        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_updated": float(actor_update_due),
        }
        if not actor_update_due:
            self._soft_update(self.critic, self.critic_target)
            return metrics

        network_actions = self.actor(actor_inputs)
        actor_actions = self._compose_actions_tensor(raw_states, network_actions)
        actor_loss, advantage_metrics = self._actor_objective(
            states,
            raw_states,
            actor_actions,
        )
        residual_l2_loss_value = None
        if self.residual_action_enabled and self.residual_l2_weight > 0.0:
            policy_residuals = self._policy_residuals_tensor(raw_states, network_actions)
            residual_l2_loss = policy_residuals.pow(2).mean()
            actor_loss = actor_loss + self.residual_l2_weight * residual_l2_loss
            residual_l2_loss_value = float(residual_l2_loss.item())
        elif self.patient_service_proxy_actor_loss_enabled:
            policy_residuals = self._policy_residuals_tensor(raw_states, network_actions)
        else:
            policy_residuals = None
        proxy_metrics = {}
        if (
            self.patient_service_proxy_actor_loss_enabled
            and self.residual_action_enabled
            and policy_residuals is not None
        ):
            proxy_loss, proxy_metrics = self._patient_service_proxy_actor_loss(
                raw_states,
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

        metrics["actor_loss"] = float(actor_loss.item())
        metrics.update(advantage_metrics)
        metrics.update(proxy_metrics)
        if residual_l2_loss_value is not None:
            metrics["residual_l2_loss"] = residual_l2_loss_value
        if imitation_loss_value is not None:
            metrics["imitation_loss"] = imitation_loss_value
        return metrics

    def _actor_input_tensor(self, raw_states, *, normalized_states=None):
        if self.temporal_demand_encoder_enabled:
            return raw_states
        normalized_states = (
            self.observation_scaler.normalize_tensor(raw_states)
            if normalized_states is None
            else normalized_states
        )
        if not self.include_base_action_features:
            return normalized_states
        base_actions = self._base_actions_from_states_tensor(raw_states)
        return torch.cat((normalized_states, base_actions), dim=1)

    def _actor_objective(self, states, raw_states, actor_actions):
        if not self.anchor_advantage_actor_loss_enabled or not self.residual_action_enabled:
            return -self.critic(states, actor_actions).mean(), {}

        actor_q = self.critic(states, actor_actions)
        with torch.no_grad():
            anchor_actions = self._base_actions_from_states_tensor(raw_states)
            anchor_q = self.critic(states, anchor_actions)
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
                self.anchor_advantage_negative_penalty_weight
                * negative_advantage_penalty
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

    def _patient_service_proxy_actor_loss(self, states, residuals):
        n = int(self.env_config.get("num_facilities", 0))
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
            loss = loss + (
                self.patient_service_proxy_low_pressure_weight
                * low_pressure_penalty
            )

        cost_penalty = torch.zeros(
            (),
            dtype=residuals.dtype,
            device=residuals.device,
        )
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

    def _actor_imitation_loss(self):
        sample_count = int(self.imitation_states.shape[0])
        batch_size = min(max(self.imitation_regularization_batch_size, 1), sample_count)
        indices = self.imitation_rng.choice(sample_count, size=batch_size, replace=False)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        raw_states = self.imitation_states[index_tensor]
        actor_inputs = self._actor_input_tensor(raw_states)
        batch_weights = (
            None
            if self.imitation_weights is None
            else self.imitation_weights[index_tensor]
        )
        return self._supervised_action_loss(
            raw_states,
            self.actor(actor_inputs),
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
        self.imitation_weights = None
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
        epochs = int(settings.get("epochs", 1))
        if epochs < 0:
            raise ValueError(
                "fit_action_batch epochs must be non-negative"
            )
        target_mode = str(
            settings.get(
                "target_mode",
                "residual" if self.residual_action_enabled else "action",
            )
        )
        if epochs == 0:
            summary = {
                "samples": int(state_tensor.shape[0]),
                "final_loss": 0.0,
                "target_mode": target_mode,
            }
        else:
            summary = self._fit_action_tensors(
                state_tensor,
                action_tensor,
                epochs=epochs,
                batch_size=int(
                    settings.get("batch_size", self.batch_size)
                ),
                seed=int(settings.get("seed", self.seed + 400000)),
                target_mode=target_mode,
                weights=weight_tensor,
            )
        if bool(settings.get("retain_for_regularization", False)):
            self.imitation_states = state_tensor.detach()
            self.imitation_actions = action_tensor.detach()
            self.imitation_weights = (
                None if weight_tensor is None else weight_tensor.detach()
            )
            self.imitation_rng = np.random.default_rng(
                int(settings.get("seed", self.seed + 400000)) + 300000
            )
        return summary

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
        with torch.no_grad():
            actor_input_tensor = self._actor_input_tensor(
                state_tensor,
            ).detach()
            residual_target_tensor = None
            residual_mask_tensor = None
            correction_gate_labels = None
            correction_gate_positive_weights = None
            if target_mode == "residual":
                residual_target_tensor = self._residual_targets_tensor(
                    state_tensor,
                    action_tensor,
                ).detach()
                residual_mask_tensor = self._residual_loss_mask(
                    action_tensor,
                    state_tensor,
                ).detach()
                residual_mask_tensor = (
                    self._supervised_residual_dimension_mask(
                        residual_mask_tensor,
                        residual_target_tensor,
                    ).detach()
                )
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
                    correction_gate_positive_weights = (
                        self._correction_gate_positive_weights(
                            correction_gate_labels
                        )
                    )
        generator = torch.Generator().manual_seed(seed)
        final_loss = 0.0
        final_gate_loss = 0.0
        self.actor.train()
        if self.correction_gate is not None:
            self.correction_gate.train()
        for _epoch in range(max(int(epochs), 1)):
            permutation = torch.randperm(sample_count, generator=generator)
            for start in range(0, sample_count, batch_size):
                indices = permutation[start : start + batch_size].to(self.device)
                raw_states = state_tensor[indices]
                actor_inputs = actor_input_tensor[indices]
                network_actions = self.actor(actor_inputs)
                batch_weights = None if weights is None else weights[indices]
                if target_mode == "residual":
                    predicted_residuals = self._policy_residuals_tensor(
                        raw_states,
                        network_actions,
                        apply_correction_gate=False,
                    )
                    residual_mask = (
                        residual_mask_tensor
                        if residual_mask_tensor.shape[0] == 1
                        else residual_mask_tensor[indices]
                    )
                    loss = self._supervised_residual_loss(
                        predicted_residuals,
                        residual_target_tensor[indices],
                        batch_weights,
                        residual_mask,
                    )
                else:
                    loss = self._supervised_action_loss(
                        raw_states,
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
                    gate_logits = self.correction_gate(actor_inputs.detach())
                    gate_per_sample = self._correction_gate_bce(
                        gate_logits,
                        correction_gate_labels[indices],
                        pos_weight=correction_gate_positive_weights,
                    )
                    if gate_per_sample.ndim > 1:
                        gate_per_sample = gate_per_sample.mean(dim=1)
                    gate_loss = (
                        gate_per_sample.mean()
                        if batch_weights is None
                        else (gate_per_sample * batch_weights).sum()
                        / batch_weights.sum().clamp_min(1e-8)
                    )
                    self.correction_gate_optimizer.zero_grad()
                    (self.correction_gate_loss_weight * gate_loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.correction_gate.parameters(),
                        max_norm=5.0,
                    )
                    self.correction_gate_optimizer.step()
                    final_gate_loss = float(gate_loss.item())
        self.actor_target.load_state_dict(self.actor.state_dict())
        summary = {
            "samples": sample_count,
            "final_loss": final_loss,
            "target_mode": target_mode,
        }
        if correction_gate_labels is not None:
            self.correction_gate.eval()
            with torch.no_grad():
                gate_probabilities = torch.sigmoid(
                    self.correction_gate(actor_input_tensor)
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
                    "correction_gate_group_positive_weights": (
                        self._pipe_group_rates(
                            correction_gate_positive_weights.unsqueeze(0)
                        )
                    ),
                }
            )
        return summary

    def evaluate_action_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        target_mode: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate supervised actor/gate loss without updating parameters."""

        state_tensor = torch.as_tensor(
            np.asarray(states),
            dtype=torch.float32,
            device=self.device,
        )
        action_tensor = torch.as_tensor(
            np.asarray(actions),
            dtype=torch.float32,
            device=self.device,
        )
        if (
            state_tensor.ndim != 2
            or state_tensor.shape[1] != self.state_dim
        ):
            raise ValueError(
                f"Expected states shape (batch, {self.state_dim}), "
                f"got {tuple(state_tensor.shape)}"
            )
        if (
            action_tensor.ndim != 2
            or action_tensor.shape[1] != self.action_dim
            or action_tensor.shape[0] != state_tensor.shape[0]
        ):
            raise ValueError(
                f"Expected actions shape (batch, {self.action_dim}), "
                f"got {tuple(action_tensor.shape)}"
            )
        sample_count = int(state_tensor.shape[0])
        if sample_count == 0:
            return {
                "samples": 0,
                "loss": 0.0,
                "actor_loss": 0.0,
            }
        weight_tensor = self._fit_action_weights(
            weights,
            sample_count,
        )
        resolved_target_mode = str(
            target_mode
            or (
                "residual"
                if self.residual_action_enabled
                else "action"
            )
        )
        actor_was_training = self.actor.training
        gate_was_training = (
            self.correction_gate.training
            if self.correction_gate is not None
            else False
        )
        self.actor.eval()
        if self.correction_gate is not None:
            self.correction_gate.eval()
        try:
            with torch.no_grad():
                actor_inputs = self._actor_input_tensor(
                    state_tensor
                )
                network_actions = self.actor(actor_inputs)
                gate_labels = None
                if resolved_target_mode == "residual":
                    if not self.residual_action_enabled:
                        raise ValueError(
                            "residual target mode requires "
                            "residual_action.enabled"
                        )
                    residual_targets = self._residual_targets_tensor(
                        state_tensor,
                        action_tensor,
                    )
                    residual_mask = self._residual_loss_mask(
                        action_tensor,
                        state_tensor,
                    )
                    residual_mask = self._supervised_residual_dimension_mask(
                        residual_mask,
                        residual_targets,
                    )
                    predicted_residuals = self._policy_residuals_tensor(
                        state_tensor,
                        network_actions,
                        apply_correction_gate=False,
                    )
                    actor_loss = self._supervised_residual_loss(
                        predicted_residuals,
                        residual_targets,
                        weight_tensor,
                        residual_mask,
                    )
                    if (
                        self.correction_gate_enabled
                        and self.correction_gate_mode
                        == "classification"
                    ):
                        gate_mask = residual_mask
                        if gate_mask.shape[0] == 1:
                            gate_mask = gate_mask.expand(
                                sample_count,
                                -1,
                            )
                        gate_labels = (
                            self._correction_gate_labels_tensor(
                                residual_targets,
                                gate_mask,
                            )
                        )
                else:
                    actor_loss = self._supervised_action_loss(
                        state_tensor,
                        network_actions,
                        action_tensor,
                        weight_tensor,
                        target_mode=resolved_target_mode,
                    )
                gate_loss = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=self.device,
                )
                diagnostics: dict[str, Any] = {}
                if gate_labels is not None:
                    gate_logits = self.correction_gate(actor_inputs)
                    gate_per_sample = self._correction_gate_bce(
                        gate_logits,
                        gate_labels,
                    )
                    if gate_per_sample.ndim > 1:
                        gate_per_sample = gate_per_sample.mean(dim=1)
                    gate_loss = (
                        gate_per_sample.mean()
                        if weight_tensor is None
                        else (gate_per_sample * weight_tensor).sum()
                        / weight_tensor.sum().clamp_min(1e-8)
                    )
                    probabilities = torch.sigmoid(gate_logits)
                    predictions = (
                        probabilities
                        >= self._correction_gate_threshold_tensor(
                            probabilities
                        )
                    )
                    positives = gate_labels > 0.5
                    predicted_positives = predictions > 0.5
                    true_positives = (
                        positives & predicted_positives
                    )
                    diagnostics.update(
                        {
                            "correction_gate_loss": float(
                                gate_loss.item()
                            ),
                            "correction_gate_accuracy": float(
                                (
                                    predictions
                                    == positives
                                )
                                .to(dtype=torch.float32)
                                .mean()
                                .item()
                            ),
                            "correction_gate_recall": float(
                                true_positives.sum().item()
                                / max(
                                    int(positives.sum().item()),
                                    1,
                                )
                            ),
                            "correction_gate_precision": float(
                                true_positives.sum().item()
                                / max(
                                    int(
                                        predicted_positives.sum().item()
                                    ),
                                    1,
                                )
                            ),
                            "correction_gate_label_rate": float(
                                positives.to(
                                    dtype=torch.float32
                                )
                                .mean()
                                .item()
                            ),
                            "correction_gate_prediction_fraction": float(
                                predicted_positives.to(
                                    dtype=torch.float32
                                )
                                .mean()
                                .item()
                            ),
                        }
                    )
                combined_loss = (
                    actor_loss
                    + self.correction_gate_loss_weight * gate_loss
                )
                diagnostics.update(
                    {
                        "samples": sample_count,
                        "loss": float(combined_loss.item()),
                        "actor_loss": float(actor_loss.item()),
                        "target_mode": resolved_target_mode,
                    }
                )
                return diagnostics
        finally:
            self.actor.train(actor_was_training)
            if self.correction_gate is not None:
                self.correction_gate.train(gate_was_training)

    def _correction_gate_bce(
        self,
        logits,
        labels,
        *,
        pos_weight=None,
    ):
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=pos_weight,
        )
        if self.correction_gate_negative_class_weight != 1.0:
            losses = losses * torch.where(
                labels > 0.5,
                torch.ones_like(losses),
                torch.full_like(
                    losses,
                    self.correction_gate_negative_class_weight,
                ),
            )
        return losses

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

    def _correction_gate_positive_weights(self, labels):
        label_matrix = labels.unsqueeze(1) if labels.ndim == 1 else labels
        positives = label_matrix.sum(dim=0)
        negatives = float(label_matrix.shape[0]) - positives
        ratios = (
            negatives.clamp_min(1.0)
            / positives.clamp_min(1.0)
        ).pow(self.correction_gate_positive_class_weight_power)
        weights = ratios.clamp(
            min=1.0,
            max=self.correction_gate_positive_class_weight_max,
        )
        return torch.where(
            positives > 0.0,
            weights,
            torch.ones_like(weights),
        )

    def _correction_gate_labels_tensor(self, residual_targets, residual_mask):
        magnitudes = residual_targets.abs() * residual_mask
        if not self.correction_gate_groups:
            return (
                magnitudes.amax(dim=1)
                > self.correction_gate_target_delta
            ).to(dtype=torch.float32)
        group_slices = self._facility_net_group_slices(
            int(self.env_config.get("num_facilities", 0))
        )
        labels = torch.stack(
            [
                (
                    magnitudes[:, group_slices[group]].amax(dim=1)
                    > self.correction_gate_target_delta
                ).to(dtype=torch.float32)
                for group in self.correction_gate_groups
            ],
            dim=1,
        )
        return labels[:, 0] if labels.shape[1] == 1 else labels

    def _pipe_group_rates(self, values) -> str:
        if values.ndim == 1:
            return f"{float(values.to(dtype=torch.float32).mean().item()):.8g}"
        return "|".join(
            f"{float(values[:, index].to(dtype=torch.float32).mean().item()):.8g}"
            for index in range(values.shape[1])
        )

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
        residual_mask = self._supervised_residual_dimension_mask(
            residual_mask,
            residual_targets,
        )
        predicted_residuals = self._policy_residuals_tensor(states, network_actions)
        return self._supervised_residual_loss(
            predicted_residuals,
            residual_targets,
            weights,
            residual_mask,
        )

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

    def _supervised_residual_loss(
        self,
        predicted_residuals,
        residual_targets,
        weights,
        dim_mask,
    ):
        loss = self._weighted_action_mse(
            predicted_residuals,
            residual_targets,
            weights,
            dim_mask,
        )
        if self.supervised_support_ranking_weight <= 0.0:
            return loss
        ranking = self._support_ranking_loss(
            predicted_residuals,
            residual_targets,
            weights,
            dim_mask,
        )
        return loss + self.supervised_support_ranking_weight * ranking

    def _support_ranking_loss(
        self,
        predicted_residuals,
        residual_targets,
        weights,
        dim_mask,
    ):
        active = dim_mask.to(
            dtype=torch.bool,
            device=predicted_residuals.device,
        )
        if active.shape[0] == 1:
            active = active.expand_as(residual_targets)
        changed = (
            residual_targets.abs()
            > self.supervised_sparse_target_threshold
        ) & active
        unchanged = (~changed) & active
        changed_count = changed.sum(dim=1)
        unchanged_count = unchanged.sum(dim=1)
        valid = (changed_count > 0) & (unchanged_count > 0)
        changed_score = (
            predicted_residuals
            * residual_targets.sign()
            * changed
        ).sum(dim=1) / changed_count.clamp_min(1)
        unchanged_score = torch.where(
            unchanged,
            predicted_residuals.abs(),
            torch.full_like(predicted_residuals, -1.0),
        ).amax(dim=1)
        per_sample = torch.relu(
            self.supervised_support_ranking_margin
            + unchanged_score
            - changed_score
        )
        valid_weights = valid.to(dtype=per_sample.dtype)
        if weights is not None:
            valid_weights = valid_weights * weights
        denominator = valid_weights.sum()
        if float(denominator.detach().item()) <= 0.0:
            return predicted_residuals.sum() * 0.0
        return (per_sample * valid_weights).sum() / denominator

    def _supervised_residual_dimension_mask(
        self,
        base_mask,
        residual_targets,
    ):
        if not self.supervised_sparse_target_enabled:
            return base_mask
        mask = base_mask.to(
            dtype=residual_targets.dtype,
            device=residual_targets.device,
        ).expand_as(residual_targets)
        changed = (
            residual_targets.abs()
            > self.supervised_sparse_target_threshold
        )
        target_weights = torch.where(
            changed,
            torch.full_like(
                residual_targets,
                self.supervised_sparse_target_changed_weight,
            ),
            torch.full_like(
                residual_targets,
                self.supervised_sparse_target_zero_weight,
            ),
        )
        return mask * target_weights

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
            self.last_residual_action = np.zeros(
                self.action_dim,
                dtype=np.float32,
            )
            return np.asarray(network_action, dtype=np.float32)
        base_action = self._base_action_from_state_np(state)
        residual_action = self._policy_residual_np(
            state,
            network_action,
            apply_correction_gate=apply_correction_gate,
        )
        scaled_residual = (
            self.residual_scale_vector * residual_action
        ).astype(np.float32)
        self.last_residual_action = self.residual_temporal_guard.apply(
            scaled_residual,
            state,
        )
        self.last_residual_guard_info = dict(
            self.residual_temporal_guard.last_info
        )
        return np.clip(
            base_action + self.last_residual_action,
            -1.0,
            1.0,
        ).astype(np.float32)

    def configure_residual_temporal_guard(
        self,
        settings: dict[str, Any] | None,
    ) -> None:
        """Replace deployment-time temporal constraints and reset their state."""

        self.residual_temporal_guard = ResidualTemporalGuard(
            num_facilities=int(
                self.env_config.get("num_facilities", 0)
            ),
            action_dim=self.action_dim,
            env_config=self.env_config,
            settings=settings,
        )
        self.residual_temporal_guard.reset()
        self.last_residual_guard_info = dict(
            self.residual_temporal_guard.last_info
        )

    def configure_residual_endpoint_projection(
        self,
        settings: dict[str, Any] | None,
    ) -> None:
        """Replace deployment-time endpoint sparsity constraints."""

        self.residual_endpoint_projection = ResidualEndpointProjection(
            num_facilities=int(
                self.env_config.get("num_facilities", 0)
            ),
            action_dim=self.action_dim,
            settings=settings,
        )

    def _compose_actions_tensor(
        self,
        states,
        network_actions,
        *,
        apply_correction_gate: bool = False,
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
            and not self.residual_endpoint_projection.enabled
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
            )
        return residual.cpu().numpy()[0].astype(np.float32)

    def _policy_residuals_tensor(
        self,
        states,
        network_actions,
        *,
        apply_correction_gate: bool = False,
        hard_correction_gate: bool = True,
    ):
        residuals = self._transform_network_residuals_tensor(network_actions)
        if self.residual_pressure_projection_groups:
            residuals = self._project_residuals_to_pressure_patterns(states, residuals)
            residuals = self._apply_positive_residual_slices_tensor(residuals)
        residuals = self._apply_state_gate_residuals_tensor(states, residuals)
        residuals = self.residual_endpoint_projection.apply_tensor(residuals)
        if apply_correction_gate and self.correction_gate is not None:
            gate_scores = self.correction_gate(
                self._actor_input_tensor(states)
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
        if gate_values.ndim == 1:
            threshold = (
                self.correction_gate_group_thresholds[0]
                if len(self.correction_gate_group_thresholds) == 1
                else self.correction_gate_threshold
            )
            return torch.as_tensor(
                threshold,
                dtype=gate_values.dtype,
                device=gate_values.device,
            )
        if not self.correction_gate_group_thresholds:
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
            int(self.env_config.get("num_facilities", 0))
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
        group_patterns = {
            "reagent_transfer": patterns["resource"],
            "reagent": patterns["resource"],
            "capacity_transfer": patterns["capacity"],
            "capacity": patterns["capacity"],
            "replenishment": patterns["resource"],
            "purchase": patterns["resource"],
        }
        group_slices = self._facility_net_group_slices(
            int(self.env_config.get("num_facilities", 0))
        )
        projected = residuals
        for group in self.residual_pressure_projection_groups:
            pattern = group_patterns.get(group)
            group_slice = group_slices.get(group)
            if pattern is None or group_slice is None:
                continue
            current = projected[:, group_slice]
            if self.residual_pressure_projection_coefficient_mode == "mean":
                coefficient = current.mean(dim=1, keepdim=True)
                if self.residual_pressure_projection_positive_coefficient:
                    coefficient = coefficient.clamp_min(0.0)
                replacement = coefficient * pattern
            else:
                replacement = project_tensor_to_pattern_basis(
                    current,
                    pattern,
                    include_uniform=(
                        self.residual_replenishment_uniform_basis
                        and group in ("replenishment", "purchase")
                    ),
                )
            projected = self._replace_action_slice_tensor(
                projected,
                group_slice,
                replacement,
            )
        return torch.clamp(projected, -1.0, 1.0)

    def _apply_state_gate_residuals_tensor(self, states, residuals):
        if not self.residual_state_gate_groups:
            return residuals
        return residuals * self._state_gate_action_mask_tensor(states, dtype=residuals.dtype)

    def _state_gate_action_mask_tensor(self, states, *, dtype):
        n = int(self.env_config.get("num_facilities", 0))
        group_slices = self._facility_net_group_slices(n)
        pressure = self._resource_pressure_tensor(states)
        threshold = float(self.residual_state_gate_config.get("threshold", 0.0))
        gate = (pressure > threshold).to(dtype=dtype)
        mask = torch.ones((states.shape[0], self.action_dim), dtype=dtype, device=states.device)
        for group in self.residual_state_gate_groups:
            group_slice = group_slices[group]
            mask = self._replace_action_slice_tensor(mask, group_slice, gate)
        return mask

    def _resource_pressure_tensor(self, states):
        return self._resource_pressure_terms_tensor(states)["resource_pressure"]

    def _residual_pressure_patterns_tensor(self, states):
        pressure_terms = self._resource_pressure_terms_tensor(
            states,
            subtract_pipeline=(
                self.residual_pressure_projection_subtract_pipeline
            ),
        )
        return {
            "resource": self._centered_unit_pattern_tensor(
                pressure_terms["resource_pressure"]
            ),
            "capacity": self._centered_unit_pattern_tensor(
                pressure_terms["capacity_pressure"]
            ),
        }

    def _resource_pressure_terms_tensor(
        self,
        states,
        *,
        subtract_pipeline: bool = True,
    ):
        n = int(self.env_config.get("num_facilities", 0))
        lead_time = int(self.env_config.get("production_lead_time", 3))
        include_supplier = int(bool(self.env_config.get("include_supplier_state", False)))
        include_forecast = int(bool(self.env_config.get("include_demand_forecast_state", False)))
        include_history = int(
            bool(self.env_config.get("include_demand_history_state", False))
        )
        include_transfer_pipeline = int(
            bool(self.env_config.get("include_transfer_pipeline_state", False))
        )
        features_per_facility = facility_state_width(self.env_config)
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
        if include_transfer_pipeline and subtract_pipeline:
            pending_start = 3 + lead_time + include_supplier + include_forecast
            pending_reagents = facility_state[:, :, pending_start + 1]
            pending_capacity = facility_state[:, :, pending_start + 2]
        risk = self._patient_risk_signal_tensor(states, features_per_facility)
        return {
            "resource_pressure": (
                demand
                + 0.25 * forecast
                + specimens
                - reagents
                - pending_reagents
                + 0.5 * risk
            ),
            "capacity_pressure": (
                demand
                + 0.25 * forecast
                + specimens
                - idle_bioreactors
                - pending_capacity
                + 0.5 * risk
            ),
        }

    def _patient_risk_signal_tensor(self, states, features_per_facility: int):
        if self.env_config.get("env_type") != "patient_condition":
            n = int(self.env_config.get("num_facilities", 0))
            return torch.zeros((states.shape[0], n), dtype=states.dtype, device=states.device)
        n = int(self.env_config.get("num_facilities", 0))
        summary_edges = tuple(self.env_config.get("survival_bucket_edges", (0.85, 0.90, 0.97)))
        summary_width = (
            6
            + len(summary_edges)
            + 1
            + (
                4
                if self.env_config.get("include_specimen_routing_state", False)
                else 0
            )
        )
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

    def _make_residual_positive_slices(self, residual_config: dict[str, Any]) -> tuple[slice, ...]:
        positive_groups = residual_config.get("positive_only_groups", ())
        if isinstance(positive_groups, str):
            positive_groups = (positive_groups,)
        positive_groups = tuple(positive_groups or ())
        if not positive_groups:
            return ()
        n = int(self.env_config.get("num_facilities", 0))
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
        n = int(self.env_config.get("num_facilities", 0))
        if self.action_dim != 4 * n:
            raise ValueError("residual_action.state_gate.groups requires a facility-net action layout")
        group_slices = self._facility_net_group_slices(n)
        normalized = []
        for group in groups:
            if group not in group_slices:
                raise ValueError(f"Unsupported residual action state-gated group: {group}")
            normalized.append(group)
        return tuple(normalized)

    def _make_pressure_projection_groups(
        self,
        residual_config: dict[str, Any],
    ) -> tuple[str, ...]:
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
        supported = set(
            self._facility_net_group_slices(
                int(self.env_config.get("num_facilities", 0))
            )
        )
        normalized = tuple(str(group) for group in tuple(groups or ()))
        unknown = [group for group in normalized if group not in supported]
        if unknown:
            raise ValueError(f"Unsupported pressure_projection groups: {unknown}")
        return normalized

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

    def _soft_update(self, local_model, target_model) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
