"""Static P1 layout veto; never constructs an environment or loads a policy.

Passing is necessary, not sufficient for real preflight. Topology defaults mirror
CapacityPlanningEnv's constructor using its existing pure graph helpers. The
producer's live checks remain in place; this audit does not adapt its contract.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from src.env.capacity_planning import _normalize_edges
from src.env.patient_capacity_planning import patient_env_config_from_dict
from src.graph.edges import complete_undirected_edges, k_nearest_ring_edges, ring_edges
from src.graph.geography import geographic_knn_edges, normalize_coordinates
from src.rl.experiment import apply_graph_ablation
from src.rl.candidate_pilot_resources import digest


RELATIONS = ("specimen_edges", "resource_edges", "capacity_edges", "information_edges")


def inspect_patient_layout(runtime, objective=None, *, message_graph="shared_relations"):
    if message_graph not in ("shared_relations", "specimen_routes"):
        raise ValueError("unknown declared message graph")
    raw = dict(runtime.get("env", {}))
    ablation = raw.pop("graph_ablation", runtime.get("graph_ablation", "full_graph"))
    scenario = raw.pop("scenario_name", runtime.get("scenario", "default"))
    if raw.get("env_type") != "patient_condition":
        raise ValueError("P1 requires the patient_condition environment")
    patient = patient_env_config_from_dict(raw)
    patient = replace(patient, base=apply_graph_ablation(patient.base, ablation))
    base, reasons = patient.base, []
    n = base.num_facilities
    coordinates = normalize_coordinates(base.clinic_coordinates, n)
    geographic = geographic_knn_edges(coordinates, k=int(base.geographic_neighbor_k)) if coordinates else ()
    complete = complete_undirected_edges(n)
    local = geographic or ring_edges(n)
    facility_net = base.action_mode == "facility_net"
    defaults = (local if facility_net else complete, local if facility_net else complete,
                complete, geographic or k_nearest_ring_edges(n, k=2))
    edges = {name: tuple(sorted(_normalize_edges(getattr(base, name), default, n)))
             for name, default in zip(RELATIONS, defaults)}
    equal = all(value == edges[RELATIONS[0]] for value in edges.values())
    if not facility_net:
        reasons.append("producer_requires_facility_net")
    unsupported = {name: getattr(base, name) for name in (
        "enable_overtime_control", "include_central_capacity_hub", "include_on_order_state",
        "enable_stochastic_procurement", "reagent_purchase_lead_time") if getattr(base, name)}
    if message_graph == "specimen_routes":
        unsupported.pop("include_central_capacity_hub", None)
    reasons += ["unsupported_producer_layout:" + name for name in unsupported]
    if not equal and message_graph == "shared_relations":
        reasons.append("heterogeneous_relations_cannot_use_single_adjacency")
    # Widths are reported only for the existing basic raw facility-net layout.
    widths = None
    if facility_net and not (base.include_on_order_state or base.enable_overtime_control):
        node_width = (3 + base.production_lead_time + int(base.include_supplier_state)
            + int(base.include_demand_forecast_state) + 3 * int(base.include_transfer_pipeline_state)
            + 3 * int(base.include_demand_history_state)
            + 3 * base.demand_sequence_length * int(base.include_demand_sequence_state))
        summary_width = 6 + len(patient.survival_bucket_edges) + 1 + 4 * int(patient.include_specimen_routing_state)
        widths = {"num_facilities": n, "horizon": base.episode_horizon,
                  "raw_state_width": n * (node_width + summary_width) + int(base.include_time_state),
                  "action_width": 4 * n, "scenario": scenario}
        if objective is not None:
            reasons += ["objective_mismatch:" + key for key, value in widths.items() if objective[key] != value]
    elif objective is not None:
        reasons.append("objective_widths_not_supported")
    capacity_graph_edges = (tuple((i, n) for i in range(n)) if edges["capacity_edges"] else ()) \
        if base.include_central_capacity_hub else edges["capacity_edges"]
    return {"kind": "static_patient_producer_layout_v1", "passed": not reasons, "reasons": reasons,
            "effective_environment_sha256": digest(asdict(patient)), "graph_ablation": ablation,
            "unsupported_layout_fields": unsupported, "objective_layout": widths,
            "physical_edges": {key: [list(edge) for edge in value] for key, value in edges.items()},
            "physical_edge_counts": {key: len(value) for key, value in edges.items()},
            "all_relations_identical": equal, "raw_facility_nodes": n,
            "reference_graph_nodes": n + int(base.include_central_capacity_hub),
            "reference_capacity_graph_edges": [list(edge) for edge in capacity_graph_edges],
            "candidate_contract": ("single_adjacency_no_hub_no_relation_collapse" if message_graph == "shared_relations"
                                   else "specimen_routes_only_facility_nodes_environment_and_reference_unchanged"),
            "new_environment_constructions": 0, "new_environment_steps": 0,
            "new_policy_loads": 0, "realized_topology_verified": False}


def audit_reference_layouts(root, config):
    reference, reports, environments = config["reference"], [], []
    for block in config["blocks"]:
        name = str(Path(reference["directory"]) / reference["config_template"].format(seed=block))
        raw = (Path(root) / name).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != reference["locks"][str(block)]["config"]:
            raise ValueError("static layout audit requires the locked R4 config")
        runtime = json.loads(raw)
        environments.append(runtime["env"])
        reports.append({"block": block, "path": name, "sha256": sha,
                        "layout": inspect_patient_layout(runtime, config["objective"],
                            message_graph=config.get("candidate_message_graph", "shared_relations"))})
    equal = bool(reports) and all(env == environments[0] for env in environments[1:])
    return {"kind": "p1_static_reference_compatibility_v1", "effective_environments_equal": equal,
            "blocks": reports, "passed": equal and all(row["layout"]["passed"] for row in reports),
            "new_environment_constructions": 0, "new_environment_steps": 0, "new_policy_loads": 0}


def require_supported_layouts(report):
    if report.get("passed") is not True:
        raise ValueError("static producer compatibility failed; no scientific claim or environment construction permitted")
