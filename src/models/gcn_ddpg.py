"""GCN-DDPG agent for graph-aware PRM capacity planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.baselines.heuristics import facility_net_action_from_state, heuristic_settings_for_policy
from src.graph.edges import Edge, complete_undirected_edges, k_nearest_ring_edges, ring_edges
from src.models.gcn import GCNActor, GCNCritic
from src.rl.action_projection import project_action
from src.rl.networks import require_torch, resolve_torch_device, torch
from src.rl.noise import OUNoise
from src.rl.preprocessing import graph_node_feature_scale, reward_scale_from_config
from src.rl.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class GraphStateSpec:
    """Shape and edge metadata needed to rebuild graph observations."""

    num_facilities: int
    production_lead_time: int
    include_supplier_state: bool
    include_central_capacity_hub: bool
    include_transfer_pipeline_state: bool
    features_per_facility: int
    node_feature_dim: int
    num_nodes: int
    edge_index: tuple[Edge, ...]
    normalize_node_features: bool = False
    node_feature_scale: tuple[float, ...] = ()


def build_graph_spec(config: dict[str, Any], state_dim: int) -> GraphStateSpec:
    """Build fixed graph metadata from an experiment config."""

    env_config = dict(config.get("env", {}))
    if "num_facilities" in env_config:
        num_facilities = int(env_config["num_facilities"])
    else:
        num_facilities = _infer_num_facilities(state_dim)
    production_lead_time = int(env_config.get("production_lead_time", 3))
    include_supplier_state = bool(env_config.get("include_supplier_state", False))
    include_hub = bool(env_config.get("include_central_capacity_hub", False))
    include_transfer_pipeline = bool(env_config.get("include_transfer_pipeline_state", False))
    features_per_facility = 3 + production_lead_time + int(include_supplier_state)
    if include_transfer_pipeline:
        features_per_facility += 3
    expected_state_dim = num_facilities * features_per_facility
    if state_dim != expected_state_dim:
        raise ValueError(
            f"GCN-DDPG expected state_dim={expected_state_dim} from config, got {state_dim}"
        )

    default_dense_edges = complete_undirected_edges(num_facilities)
    action_mode = str(env_config.get("action_mode", "edge_transfer"))
    if action_mode == "facility_net":
        default_specimen_edges = ring_edges(num_facilities)
        default_resource_edges = ring_edges(num_facilities)
        default_capacity_edges = default_dense_edges
    else:
        default_specimen_edges = default_dense_edges
        default_resource_edges = default_dense_edges
        default_capacity_edges = default_dense_edges

    information_edges = _configured_edges(
        env_config.get("information_edges"),
        k_nearest_ring_edges(num_facilities, k=2),
        num_facilities,
    )
    specimen_edges = _configured_edges(
        env_config.get("specimen_edges"), default_specimen_edges, num_facilities
    )
    capacity_edges = _configured_edges(
        env_config.get("capacity_edges"), default_capacity_edges, num_facilities
    )
    resource_edges = _configured_edges(
        env_config.get("resource_edges"), default_resource_edges, num_facilities
    )

    ablation = env_config.get("graph_ablation", config.get("graph_ablation", "full_graph"))
    if ablation == "no_capacity_sharing_edges":
        capacity_edges = ()
    elif ablation == "no_resource_sharing_edges":
        resource_edges = ()
    elif ablation == "no_interfacility_edges":
        specimen_edges = ()
        capacity_edges = ()
        resource_edges = ()
    elif ablation == "flat_state_no_graph":
        information_edges = ()
        specimen_edges = ()
        capacity_edges = ()
        resource_edges = ()

    num_nodes = num_facilities + int(include_hub)
    capacity_graph_edges = capacity_edges
    if include_hub:
        capacity_graph_edges = tuple((i, num_facilities) for i in range(num_facilities)) if capacity_edges else ()

    edge_sets = {
        "information_edges": information_edges,
        "specimen_edges": specimen_edges,
        "capacity_edges": capacity_graph_edges,
        "resource_edges": resource_edges,
    }
    selected_edge_types = tuple(
        config.get(
            "gcn_edge_types",
            ("information_edges", "specimen_edges", "capacity_edges", "resource_edges"),
        )
    )
    graph_edges: list[Edge] = []
    for edge_type in selected_edge_types:
        if edge_type not in edge_sets:
            raise ValueError(f"Unsupported GCN edge type: {edge_type}")
        graph_edges.extend(edge_sets[edge_type])
    graph_edges = list(_dedupe_edges(graph_edges, num_nodes))

    node_feature_dim = 5 + int(include_supplier_state) + 3 * int(include_transfer_pipeline) + int(include_hub)
    normalize_node_features = bool(config.get("normalize_observations", False))
    node_feature_scale = graph_node_feature_scale(config, node_feature_dim)
    return GraphStateSpec(
        num_facilities=num_facilities,
        production_lead_time=production_lead_time,
        include_supplier_state=include_supplier_state,
        include_central_capacity_hub=include_hub,
        include_transfer_pipeline_state=include_transfer_pipeline,
        features_per_facility=features_per_facility,
        node_feature_dim=node_feature_dim,
        num_nodes=num_nodes,
        edge_index=tuple(graph_edges),
        normalize_node_features=normalize_node_features,
        node_feature_scale=node_feature_scale,
    )


def flat_state_to_node_features(state, graph_spec: GraphStateSpec):
    """Convert replay-buffer flat states into GCN node features."""

    require_torch()
    if state.dim() == 1:
        state = state.unsqueeze(0)
    batch_size = state.shape[0]
    n = graph_spec.num_facilities
    lead_time = graph_spec.production_lead_time
    facility_state = state.reshape(batch_size, n, graph_spec.features_per_facility)

    demand = facility_state[:, :, 0:1]
    specimens = facility_state[:, :, 1:2]
    reagents = facility_state[:, :, 2:3]
    idle_bioreactors = facility_state[:, :, 3:4]
    total_bioreactors = facility_state[:, :, 3 : 3 + lead_time].sum(dim=-1, keepdim=True)
    feature_parts = [demand, specimens, reagents, idle_bioreactors, total_bioreactors]
    if graph_spec.include_supplier_state:
        feature_parts.append(facility_state[:, :, 3 + lead_time : 4 + lead_time])
    if graph_spec.include_transfer_pipeline_state:
        pending_start = 3 + lead_time + int(graph_spec.include_supplier_state)
        feature_parts.append(facility_state[:, :, pending_start : pending_start + 3])
    if graph_spec.include_central_capacity_hub:
        feature_parts.append(torch.zeros_like(demand))

    node_features = torch.cat(feature_parts, dim=-1)
    if not graph_spec.include_central_capacity_hub:
        return _normalize_node_features(node_features, graph_spec)

    hub_features = torch.zeros(
        batch_size,
        1,
        graph_spec.node_feature_dim,
        dtype=node_features.dtype,
        device=node_features.device,
    )
    hub_features[:, 0, 3] = idle_bioreactors.sum(dim=1).squeeze(-1)
    hub_features[:, 0, 4] = total_bioreactors.sum(dim=1).squeeze(-1)
    hub_features[:, 0, -1] = 1.0
    node_features = torch.cat((node_features, hub_features), dim=1)
    return _normalize_node_features(node_features, graph_spec)


class GCNDDPGAgent:
    """DDPG agent that embeds the manufacturing network with a GCN."""

    algorithm = "gcn_ddpg"

    def __init__(self, state_dim: int, action_dim: int, config: dict[str, Any]):
        require_torch()
        seed = int(config.get("seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)

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
        self.imitation_rng = np.random.default_rng(seed + 300000)
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
        ).to(self.device)
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
            next_actions = self._compose_actions_tensor(next_states, next_network_actions)
            target_q = self.critic_target(next_node_features, next_actions)
            q_targets = rewards + self.gamma * (1.0 - dones) * target_q

        q_expected = self.critic(node_features, actions)
        critic_loss = torch.nn.functional.mse_loss(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        network_actions = self.actor(node_features)
        actor_actions = self._compose_actions_tensor(states, network_actions)
        actor_loss = -self.critic(node_features, actor_actions).mean()
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
        node_features = flat_state_to_node_features(
            self.imitation_states[index_tensor],
            self.graph_spec,
        )
        return self._supervised_action_loss(
            self.imitation_states[index_tensor],
            self.actor(node_features),
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
        generator = torch.Generator().manual_seed(seed)
        final_loss = 0.0

        self.actor.train()
        for _epoch in range(max(epochs, 1)):
            permutation = torch.randperm(sample_count, generator=generator)
            for start in range(0, sample_count, batch_size):
                indices = permutation[start : start + batch_size].to(self.device)
                node_features = flat_state_to_node_features(state_tensor[indices], self.graph_spec)
                loss = self._supervised_action_loss(
                    state_tensor[indices],
                    self.actor(node_features),
                    action_tensor[indices],
                    None if weight_tensor is None else weight_tensor[indices],
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
            },
            output_path,
        )

    def load_actor(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])

    def _soft_update(self, local_model, target_model) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


def _normalize_node_features(node_features, graph_spec: GraphStateSpec):
    if not graph_spec.normalize_node_features:
        return node_features
    scale = torch.as_tensor(
        graph_spec.node_feature_scale,
        dtype=node_features.dtype,
        device=node_features.device,
    )
    return (node_features / scale).clamp(-10.0, 10.0)


def _infer_num_facilities(state_dim: int) -> int:
    if state_dim % 6 == 0:
        return state_dim // 6
    if state_dim % 8 == 0:
        return state_dim // 8
    raise ValueError("num_facilities must be provided in config['env'] for GCN-DDPG")


def _configured_edges(
    configured_edges: Sequence[Sequence[int]] | None,
    default_edges: tuple[Edge, ...],
    num_nodes: int,
) -> tuple[Edge, ...]:
    edges = default_edges if configured_edges is None else tuple(configured_edges)
    return _dedupe_edges(edges, num_nodes)


def _dedupe_edges(edges: Sequence[Sequence[int]], num_nodes: int) -> tuple[Edge, ...]:
    normalized = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"Edge must have two endpoints, got {edge}")
        i, j = int(edge[0]), int(edge[1])
        if i == j:
            continue
        if i < 0 or j < 0 or i >= num_nodes or j >= num_nodes:
            raise ValueError(f"Edge {(i, j)} is outside the {num_nodes}-node graph")
        normalized.append((min(i, j), max(i, j)))
    return tuple(dict.fromkeys(normalized))
