"""Audit historical graph/flat interfaces without agents, rollouts or training."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

from evaluation.audit_formal_method_contract import (
    ALGORITHMS, MANIFEST_HASH, REPO, TRAIN_REF, git_bytes, historical_source,
    sha, validate_config,
)


MODULES = (
    "src/rl/networks.py", "src/graph/edges.py", "src/graph/geography.py",
    "src/rl/preprocessing.py", "src/rl/tensor_conversion.py",
    "src/models/graph_features.py", "src/models/gcn.py",
)
ALLOWED_DIFFERENCES = {
    "algorithm", "checkpoint_dir", "config_snapshot_path", "result_csv_path",
    "training_state_checkpoint_path", "env.graph_ablation",
    "evaluation_replications", "gcn_hidden_sizes", "hidden_sizes",
    "history_screen.gcn_hidden_sizes", "history_screen.hidden_sizes",
    "include_global_context",
}


def config_differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = {}
        for key in sorted(left.keys() | right.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result[path] = {"gcn": left.get(key), "flat": right.get(key),
                                "gcn_present": key in left, "flat_present": key in right}
            else:
                result.update(config_differences(left[key], right[key], path))
        return result
    return {} if left == right else {prefix: {"gcn": left, "flat": right}}


def check_pair(left, right):
    differences = config_differences(left, right)
    unexpected = set(differences) - ALLOWED_DIFFERENCES
    if unexpected:
        raise ValueError(f"unreviewed arm differences: {sorted(unexpected)}")
    return differences


def load_historical_modules():
    # Only these reviewed definition-only modules are loaded. Resolve their
    # src imports from the pinned modules, never from the current simulator.
    loaded = {}
    for path in MODULES:
        key = path[:-3].replace("/", ".")
        module = ModuleType("_formal_graph_audit_" + key.replace(".", "_"))
        tree = ast.parse(git_bytes(TRAIN_REF, path), filename=path)
        body = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
                dependency = loaded[node.module]
                for alias in node.names:
                    module.__dict__[alias.asname or alias.name] = getattr(dependency, alias.name)
            else:
                body.append(node)
        tree.body = body
        sys.modules[module.__name__] = module
        exec(compile(tree, f"{TRAIN_REF}:{path}", "exec"), module.__dict__)
        loaded[key] = module
    return loaded


def historical_function(path, name, namespace):
    tree = ast.parse(git_bytes(TRAIN_REF, path))
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"ambiguous function: {path}:{name}")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), matches[0]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


def reconstruct(config, loaded):
    features = loaded["src.models.graph_features"]
    nets = loaded["src.rl.networks"]
    gcn = loaded["src.models.gcn"]
    pre = loaded["src.rl.preprocessing"]
    n = config["env"]["num_facilities"]
    state_dim = n * (pre.facility_state_width(config["env"]) + features._patient_summary_width(config["env"])) + 1
    action_dim = 4 * n
    gate_config = config["residual_action"]["correction_gate"]
    if config.get("temporal_demand_encoder", {}).get("enabled", False):
        raise ValueError("audit reconstruction is limited to non-temporal formal models")
    if not gate_config["enabled"] or gate_config["groups"] != ["specimen_transfer"]:
        raise ValueError("unexpected gate contract")
    if not config["residual_action"]["include_base_action_features"]:
        raise ValueError("expected anchor features")
    if not gate_config.get("include_proposed_residual_features"):
        raise ValueError("expected proposed-residual gate configuration")
    if config.get("residual_action", {}).get("edge_selector", {}).get("enabled", False):
        raise ValueError("unexpected edge selector")
    if config.get("actor_readout_mode") != "network_residual":
        raise ValueError("unexpected actor readout")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        if config["algorithm"] == ALGORITHMS[0]:
            spec = features.build_graph_spec(config, state_dim)
            actor = gcn.GCNActor(
                spec.node_feature_dim, n, spec.num_nodes, action_dim, spec.edge_index,
                config["gcn_hidden_sizes"], config["actor_hidden_sizes"],
                include_global_context=config["include_global_context"],
                readout_mode=config["actor_readout_mode"], edge_weights=spec.edge_weights,
                specimen_routing_enabled=config["specimen_routing_head_enabled"],
                specimen_edges=spec.specimen_edge_index, resource_edges=spec.resource_edge_index,
                capacity_edges=spec.capacity_edge_index,
                specimen_edge_features=spec.specimen_edge_features,
                resource_edge_features=spec.resource_edge_features,
                capacity_edge_features=spec.capacity_edge_features,
            )
            critic = gcn.GCNCritic(spec.node_feature_dim, n, spec.num_nodes, action_dim,
                                  spec.edge_index, config["gcn_hidden_sizes"], config["critic_hidden_sizes"],
                                  include_global_context=config["include_global_context"], edge_weights=spec.edge_weights)
            gate = gcn.GCNCorrectionGate(spec.node_feature_dim + 4, n, spec.num_nodes,
                                         spec.edge_index, config["gcn_hidden_sizes"], gate_config.get("hidden_sizes", (64, 32)),
                                         include_global_context=config["include_global_context"], edge_weights=spec.edge_weights)
            dimensions = {"raw_state": state_dim, "action": action_dim,
                          "nodes": spec.num_nodes, "actor_critic_node_features": spec.node_feature_dim,
                          "gate_node_features": spec.node_feature_dim + 4,
                          "critic_gate_ordered_readout": gcn.graph_readout_dim(n, config["gcn_hidden_sizes"][-1], config["include_global_context"]),
                          "message_edges": len(spec.edge_index),
                          "specimen_head_edges": len(spec.specimen_edge_index),
                          "reagent_head_edges": len(spec.resource_edge_index),
                          "capacity_head_edges": len(spec.capacity_edge_index),
                          "edge_features": spec.edge_feature_dim,
                          "demand_sequence_length": spec.demand_sequence_length}
            probe = torch.ones(2, spec.num_nodes, spec.node_feature_dim)
            gate_probe = torch.ones(2, spec.num_nodes, spec.node_feature_dim + 4)
        else:
            actor = nets.MLPActor(state_dim + action_dim, action_dim, config["hidden_sizes"])
            critic = nets.MLPCritic(state_dim, action_dim, config["hidden_sizes"])
            gate = nets.MLPCorrectionGate(state_dim + action_dim, gate_config.get("hidden_sizes", (64, 32)))
            dimensions = {"raw_state": state_dim, "action": action_dim,
                          "actor_gate_input": state_dim + action_dim,
                          "critic_state_input": state_dim,
                          "critic_state_action_input": state_dim + action_dim}
            probe = torch.ones(2, state_dim + action_dim)
            gate_probe = probe
        modules = {"actor": actor, "critic": critic, "correction_gate": gate}
        with torch.no_grad():
            action = actor(probe)
            critic_probe = probe if config["algorithm"] == ALGORITHMS[0] else probe[:, :state_dim]
            outputs = [action, critic(critic_probe, action), gate(gate_probe)]
        if not all(torch.isfinite(output).all() for output in outputs):
            raise ValueError("nonfinite structural forward probe")
        counts = {k: sum(p.numel() for p in m.parameters()) for k, m in modules.items()}
        counter = historical_function("evaluation/train_multiscenario_network_residual.py",
                                      "agent_parameter_count", {})
        count = counter(SimpleNamespace(**modules))
        if count != sum(counts.values()):
            raise ValueError("parameter counter mismatch")
        return {"dimensions": dimensions, "parameters": counts, "total_parameters": count,
                "forward_shapes": [list(o.shape) for o in outputs],
                "finite_untrained_cpu_forward": True}


def interface_probes(config, loaded):
    features = loaded["src.models.graph_features"]
    spec = features.build_graph_spec(config, 561)
    # Disable only the anchor-derived block in this layout probe to avoid
    # importing or executing a heuristic/environment; it is not a policy test.
    raw_spec = replace(spec, include_base_action_features=False, node_feature_dim=spec.node_feature_dim - 4)
    state = torch.ones(1, 561)
    rearranged = state.clone()
    rearranged[0, 4] = 0
    rearranged[0, 5] = 2
    original = features.flat_state_to_node_features(state, raw_spec)
    alternative = features.flat_state_to_node_features(rearranged, raw_spec)
    identity = historical_function("src/rl/experiment.py", "apply_graph_ablation", {})
    sentinel = object()
    if identity(sentinel, "flat_state_no_graph") is not sentinel:
        raise ValueError("flat ablation changed physical environment")
    namespace = {"torch": torch, "flat_state_to_node_features": lambda states, spec: torch.zeros(len(states), 21, 36)}
    gate_features = historical_function("src/models/gcn_ddpg.py", "correction_gate_node_features", namespace)
    agent = SimpleNamespace(graph_spec=spec, correction_gate_include_proposed_residual_features=True,
                            action_dim=80, residual_scale_vector=[.1] * 20 + [0.] * 60,
                            correction_gate_proposed_residual_feature_mode="clipped_action_delta",
                            correction_gate_proposed_residual_feature_dim=4,
                            _base_actions_from_states_tensor=lambda states: torch.zeros(len(states), 80))
    residual = torch.zeros(1, 80)
    before = gate_features(agent, state, residuals=residual)
    residual[0, 0] = .5
    after = gate_features(agent, state, residuals=residual)
    flat_input = historical_function("src/baselines/flat_ddpg.py", "_actor_input_tensor", {"torch": torch})
    flat_agent = SimpleNamespace(temporal_demand_encoder_enabled=False, include_base_action_features=True,
                                 _base_actions_from_states_tensor=lambda states: torch.zeros(len(states), 80))
    flat_features = flat_input(flat_agent, state, normalized_states=state)
    flat_source = git_bytes(TRAIN_REF, "src/baselines/flat_ddpg.py").decode()
    return {"scope": "synthetic layout and gate-input probes, not trained-policy outcomes",
            "capacity_pipeline_redistribution_preserves_explicit_graph_raw_block": bool(torch.equal(original, alternative)),
            "raw_flat_states_differ": not bool(torch.equal(state, rearranged)),
            "anchor_block_excluded_from_pipeline_probe": True,
            "flat_ablation_preserves_physical_config": True,
            "gcn_proposal_changes_gate_input": not bool(torch.equal(before, after)),
            "gcn_nonzero_proposal_features": int(torch.count_nonzero(after - before)),
            "flat_actor_gate_feature_width": flat_features.shape[1],
            "flat_actor_gate_raw_state_preserved": bool(torch.equal(flat_features[:, :561], state)),
            "flat_source_references_proposed_residual_config": "include_proposed_residual_features" in flat_source,
            "flat_source_references_adaptive_feature_helper": "flat_state_to_adaptive_demand_features" in flat_source}


def audit(root):
    manifest_path = root / "training_manifest.json"
    snapshots = {manifest_path: sha(manifest_path.read_bytes())}
    if snapshots[manifest_path] != MANIFEST_HASH:
        raise ValueError("historical manifest hash mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    runs = {(r["algorithm"], r["seed"]): r for r in manifest["runs"]}
    if len(manifest["runs"]) != 10 or set(runs) != {(a, s) for a in ALGORITHMS for s in range(10, 15)}:
        raise ValueError("manifest run inventory mismatch")
    loaded = load_historical_modules()
    records, pairs = [], []
    for seed in range(10, 15):
        configs = []
        for algorithm in ALGORITHMS:
            run = root / algorithm / f"seed{seed}"
            for name in ("config.json", "summary.json"):
                path = run / name
                snapshots[path] = sha(path.read_bytes())
            config = json.loads((run / "config.json").read_bytes())
            summary = json.loads((run / "summary.json").read_bytes())
            validate_config(config, algorithm, seed)
            reconstruction = reconstruct(config, loaded)
            count = reconstruction["total_parameters"]
            if count != summary["parameter_count"] or count != runs[algorithm, seed]["parameter_count"]:
                raise ValueError(f"parameter mismatch: {algorithm}, {seed}")
            configs.append(config)
            records.append({"algorithm": algorithm, "seed": seed, **reconstruction})
        pairs.append({"seed": seed, "config_differences": check_pair(*configs)})
    probes = interface_probes(configs[0], loaded)
    evidence_path = REPO / "experiments/evidence/patient_indexed_specimen_routing_publication_evidence_map.json"
    snapshots[evidence_path] = sha(evidence_path.read_bytes())
    evidence = json.loads(evidence_path.read_bytes())
    source_symbols = {
        "src/models/gcn.py": ["GCNActor", "GCNCritic", "GCNCorrectionGate", "graph_readout"],
        "src/models/graph_features.py": ["build_graph_spec", "flat_state_to_node_features"],
        "src/models/gcn_ddpg.py": ["correction_gate_node_features"],
        "src/baselines/flat_ddpg.py": ["_actor_input_tensor", "_policy_residuals_tensor"],
        "src/rl/experiment.py": ["apply_graph_ablation"],
        "evaluation/train_multiscenario_network_residual.py": ["agent_parameter_count"],
    }
    sources = [historical_source(path, names) for path, names in source_symbols.items()]
    gap = (records[0]["total_parameters"] - records[1]["total_parameters"]) / records[1]["total_parameters"]
    if not 0 <= gap <= .01:
        raise ValueError("parameter matching outside one-percent budget")
    for path, digest in snapshots.items():
        if sha(path.read_bytes()) != digest:
            raise ValueError(f"input changed during audit: {path}")
    return {"status": "audit_complete_with_methodological_differences", "training_commit": TRAIN_REF,
            "historical_manifest_sha256_verified": MANIFEST_HASH,
            "provenance_limit": "current config/summary hashes; counts linked to locked historical manifest, not independent historical config hashes",
            "inputs": [{"path": str(p), "sha256": h} for p, h in snapshots.items()],
            "loaded_source_hashes": {p: sha(git_bytes(TRAIN_REF, p)) for p in MODULES},
            "runs": records, "paired_config_checks": pairs, "interface_probes": probes,
            "parameter_gap_relative_to_flat": gap,
            "parameter_count_scope": "actor + critic + correction gate; excludes target/reference copies, buffers and optimizer state; not compute matching",
            "historical_sources": sources,
            "publication_evidence_claim_ids_reviewed": [c["id"] for c in evidence["claims"]],
            "topology_evidence_limit": "the frozen publication map supplies no isolated feature/head/gate-matched message-edge ablation or held-out-topology result",
            "interpretation": "formal package advantage remains; message passing alone is not isolated",
            "new_training_runs": 0, "new_evaluation_runs": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite audit evidence")
    report = audit(args.training_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]),
                      "parameter_gap_relative_to_flat": report["parameter_gap_relative_to_flat"]}))


if __name__ == "__main__":
    main()
