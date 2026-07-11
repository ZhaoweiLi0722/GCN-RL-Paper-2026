"""Shared graph-observation plumbing for the GCN method family.

`GraphStateSpec` + `build_graph_spec` + `flat_state_to_node_features` are used by
every graph agent (GCN-DDPG/TD3/SAC/PPO) to (a) derive fixed graph metadata from
an experiment config and (b) reconstruct per-node GCN features from the flat
replay-buffer states. Extracted from ``gcn_ddpg.py`` so the newer agents can
depend on the plumbing without pulling in the DDPG agent's residual/heuristic
machinery. ``gcn_ddpg.py`` re-exports these names for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.graph.edges import Edge, complete_undirected_edges, k_nearest_ring_edges, ring_edges
from src.rl.networks import require_torch, torch
from src.rl.preprocessing import graph_node_feature_scale


@dataclass(frozen=True)
class GraphStateSpec:
    """Shape and edge metadata needed to rebuild graph observations."""

    num_facilities: int
    production_lead_time: int
    include_supplier_state: bool
    include_demand_forecast_state: bool
    include_central_capacity_hub: bool
    include_transfer_pipeline_state: bool
    features_per_facility: int
    node_feature_dim: int
    num_nodes: int
    edge_index: tuple[Edge, ...]
    normalize_node_features: bool = False
    node_feature_scale: tuple[float, ...] = ()
    patient_summary_width: int = 0


def _patient_summary_width(env_config: dict[str, Any]) -> int:
    """Per-clinic patient-summary width, matching PatientConditionCapacityEnv.

    Layout is ``[count, mean_survival, near_expiry, histogram...]`` where the
    histogram has ``len(edges) + 1`` buckets, i.e. ``3 + len(edges) + 1``.
    """

    if env_config.get("env_type") != "patient_condition":
        return 0
    edges = env_config.get("survival_bucket_edges", (0.85, 0.90, 0.97))
    return 3 + len(tuple(edges)) + 1


def build_graph_spec(config: dict[str, Any], state_dim: int) -> GraphStateSpec:
    """Build fixed graph metadata from an experiment config."""

    env_config = dict(config.get("env", {}))
    if "num_facilities" in env_config:
        num_facilities = int(env_config["num_facilities"])
    else:
        num_facilities = _infer_num_facilities(state_dim)
    production_lead_time = int(env_config.get("production_lead_time", 3))
    include_supplier_state = bool(env_config.get("include_supplier_state", False))
    include_forecast = bool(env_config.get("include_demand_forecast_state", False))
    include_hub = bool(env_config.get("include_central_capacity_hub", False))
    include_transfer_pipeline = bool(env_config.get("include_transfer_pipeline_state", False))
    features_per_facility = 3 + production_lead_time + int(include_supplier_state) + int(include_forecast)
    if include_transfer_pipeline:
        features_per_facility += 3
    summary_width = _patient_summary_width(env_config)
    expected_state_dim = num_facilities * (features_per_facility + summary_width)
    if state_dim != expected_state_dim:
        raise ValueError(
            f"GCN agent expected state_dim={expected_state_dim} from config, got {state_dim}"
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

    node_feature_dim = (
        5
        + int(include_supplier_state)
        + int(include_forecast)
        + 3 * int(include_transfer_pipeline)
        + int(include_hub)
        + summary_width
    )
    normalize_node_features = bool(config.get("normalize_observations", False))
    node_feature_scale = graph_node_feature_scale(config, node_feature_dim)
    return GraphStateSpec(
        num_facilities=num_facilities,
        production_lead_time=production_lead_time,
        include_supplier_state=include_supplier_state,
        include_demand_forecast_state=include_forecast,
        include_central_capacity_hub=include_hub,
        include_transfer_pipeline_state=include_transfer_pipeline,
        features_per_facility=features_per_facility,
        node_feature_dim=node_feature_dim,
        num_nodes=num_nodes,
        edge_index=tuple(graph_edges),
        normalize_node_features=normalize_node_features,
        node_feature_scale=node_feature_scale,
        patient_summary_width=summary_width,
    )


def flat_state_to_node_features(state, graph_spec: GraphStateSpec):
    """Convert replay-buffer flat states into GCN node features.

    For the patient env the flat state is ``[base | per-clinic summary]``; the
    summary block is reshaped and hstacked onto the clinic nodes (hub row
    zero-padded), reproducing ``PatientConditionCapacityEnv.graph_observation``.
    """

    require_torch()
    if state.dim() == 1:
        state = state.unsqueeze(0)
    batch_size = state.shape[0]
    n = graph_spec.num_facilities
    lead_time = graph_spec.production_lead_time
    summary_width = graph_spec.patient_summary_width

    base_width = n * graph_spec.features_per_facility
    if summary_width > 0:
        base_state = state[:, :base_width]
        summary_state = state[:, base_width:].reshape(batch_size, n, summary_width)
    else:
        base_state = state
        summary_state = None
    facility_state = base_state.reshape(batch_size, n, graph_spec.features_per_facility)

    demand = facility_state[:, :, 0:1]
    specimens = facility_state[:, :, 1:2]
    reagents = facility_state[:, :, 2:3]
    idle_bioreactors = facility_state[:, :, 3:4]
    total_bioreactors = facility_state[:, :, 3 : 3 + lead_time].sum(dim=-1, keepdim=True)
    feature_parts = [demand, specimens, reagents, idle_bioreactors, total_bioreactors]
    if graph_spec.include_supplier_state:
        feature_parts.append(facility_state[:, :, 3 + lead_time : 4 + lead_time])
    if graph_spec.include_demand_forecast_state:
        forecast_start = 3 + lead_time + int(graph_spec.include_supplier_state)
        feature_parts.append(facility_state[:, :, forecast_start : forecast_start + 1])
    if graph_spec.include_transfer_pipeline_state:
        pending_start = (
            3
            + lead_time
            + int(graph_spec.include_supplier_state)
            + int(graph_spec.include_demand_forecast_state)
        )
        feature_parts.append(facility_state[:, :, pending_start : pending_start + 3])
    if graph_spec.include_central_capacity_hub:
        feature_parts.append(torch.zeros_like(demand))
    if summary_state is not None:
        feature_parts.append(summary_state)

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
    # Hub aggregates capacity; columns 3/4 are idle/total bioreactors, last flags the hub.
    hub_features[:, 0, 3] = idle_bioreactors.sum(dim=1).squeeze(-1)
    hub_features[:, 0, 4] = total_bioreactors.sum(dim=1).squeeze(-1)
    hub_flag_col = graph_spec.node_feature_dim - 1 - summary_width
    hub_features[:, 0, hub_flag_col] = 1.0
    node_features = torch.cat((node_features, hub_features), dim=1)
    return _normalize_node_features(node_features, graph_spec)


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
    raise ValueError("num_facilities must be provided in config['env'] for GCN agents")


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
