"""GCN-DDPG agent for graph-aware PRM capacity planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.graph.edges import Edge, complete_undirected_edges, k_nearest_ring_edges, ring_edges
from src.models.gcn import GCNActor, GCNCritic
from src.rl.action_projection import project_action
from src.rl.networks import require_torch, torch
from src.rl.noise import OUNoise
from src.rl.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class GraphStateSpec:
    """Shape and edge metadata needed to rebuild graph observations."""

    num_facilities: int
    production_lead_time: int
    include_supplier_state: bool
    include_central_capacity_hub: bool
    features_per_facility: int
    node_feature_dim: int
    num_nodes: int
    edge_index: tuple[Edge, ...]


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
    features_per_facility = 3 + production_lead_time + int(include_supplier_state)
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

    node_feature_dim = 5 + int(include_supplier_state) + int(include_hub)
    return GraphStateSpec(
        num_facilities=num_facilities,
        production_lead_time=production_lead_time,
        include_supplier_state=include_supplier_state,
        include_central_capacity_hub=include_hub,
        features_per_facility=features_per_facility,
        node_feature_dim=node_feature_dim,
        num_nodes=num_nodes,
        edge_index=tuple(graph_edges),
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
    if graph_spec.include_central_capacity_hub:
        feature_parts.append(torch.zeros_like(demand))

    node_features = torch.cat(feature_parts, dim=-1)
    if not graph_spec.include_central_capacity_hub:
        return node_features

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
    return torch.cat((node_features, hub_features), dim=1)


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
        self.graph_spec = build_graph_spec(config, state_dim)
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.batch_size = int(config.get("batch_size", 128))
        gcn_hidden_sizes = tuple(config.get("gcn_hidden_sizes", [64, 64]))
        actor_hidden_sizes = tuple(config.get("actor_hidden_sizes", config.get("hidden_sizes", [256, 128])))
        critic_hidden_sizes = tuple(config.get("critic_hidden_sizes", config.get("hidden_sizes", [256, 128])))
        include_global_context = bool(config.get("include_global_context", True))
        device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)

        self.actor = GCNActor(
            self.graph_spec.node_feature_dim,
            self.graph_spec.num_facilities,
            self.graph_spec.num_nodes,
            action_dim,
            self.graph_spec.edge_index,
            gcn_hidden_sizes,
            actor_hidden_sizes,
            include_global_context=include_global_context,
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
            action = self.actor(node_features).cpu().numpy()[0]
        self.actor.train()
        if explore:
            action = action + self.noise.sample()
        return project_action(action, env_state=env, action_space_info=self.action_dim).action

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

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
            next_actions = self.actor_target(next_node_features)
            target_q = self.critic_target(next_node_features, next_actions)
            q_targets = rewards + self.gamma * (1.0 - dones) * target_q

        q_expected = self.critic(node_features, actions)
        critic_loss = torch.nn.functional.mse_loss(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(node_features, self.actor(node_features)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item())}

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
