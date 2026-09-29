"""Count-only development preflight; no environment or optimizer execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from src.models.matched_inputs import InputSchema, ObservationBatch, build_matched_inputs
from src.models.prospective_forward_agent import ProspectiveForwardAgent
from src.rl.networks import torch
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.validated_returns import ReplaySemantics

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("actor", "critic", "gate")
PERMISSIONS = ("environment_steps", "optimizer_updates", "performance_evaluation", "remote_actions", "formal_holdout")
LAUNCH_REQUIREMENTS = {
    "operational_task_measurement_and_calibration", "complete_objective_units_and_terminal_accounting",
    "replicated_safe_action_headroom_and_learnability", "competent_frozen_and_adaptive_comparator_acceptance",
    "optimizer_replay_and_exact_resume_acceptance", "new_task_input_action_model_and_gate_lock",
    "fresh_stream_budget_effect_margin_and_analysis_lock", "explicit_scientific_launch_authorization",
}


def parameter_counts(contract, architecture, width):
    """Exact count of the current common-head family; confirmed against modules."""
    if not isinstance(contract, ReplayInputContract):
        raise TypeError("explicit replay/input contract required")
    if architecture not in ("graph", "flat") or type(width) is not int or width < 1:
        raise ValueError("invalid architecture or width")
    schema = contract.inputs
    f, a, state = len(schema.node_feature_names), len(schema.action_names), contract.replay.state_dim
    extra = f * (f + 1) if architecture == "graph" else 0
    counts = {
        "actor": (state + 1) * width + (width + 1) * a + extra,
        "critic": (state + a + 1) * width + width + 1 + extra,
        "gate": (state + a + 1) * width + (width + 1) * a + extra,
    }
    return counts | {"total": sum(counts.values())}


def match_flat_width(contract, *, graph_width, flat_range, component_limit, total_limit):
    if (not isinstance(flat_range, (tuple, list)) or len(flat_range) != 2
            or any(type(x) is not int for x in flat_range)
            or not 1 <= flat_range[0] <= flat_range[1] <= 4096):
        raise ValueError("finite positive flat-width range required")
    for limit in (component_limit, total_limit):
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not 0 <= limit <= .01:
            raise ValueError("gap fractions must be finite and <= 1 percent")
    reference = parameter_counts(contract, "graph", graph_width)
    candidates = []
    for width in range(flat_range[0], flat_range[1] + 1):
        counts = parameter_counts(contract, "flat", width)
        gaps = {name: abs(counts[name] - reference[name]) / reference[name] for name in reference}
        candidates.append({"width": width, "counts": counts, "gaps_fraction": gaps,
                           "max_component_gap_fraction": max(gaps[name] for name in COMPONENTS),
                           "eligible": (max(gaps[name] for name in COMPONENTS) <= component_limit
                                        and gaps["total"] <= total_limit)})
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = min(eligible, key=lambda c: (c["max_component_gap_fraction"],
                                           c["gaps_fraction"]["total"], c["width"])) if eligible else None
    return {"reference_counts": reference, "selected": selected, "candidates": candidates}


def _agent(contract, architecture, width, mode="physical"):
    return ProspectiveForwardAgent(contract, enabled=True, architecture=architecture, message_mode=mode,
                                   hidden_width=width, residual_scale=.1, seed=0)


def _views(contract, mode):
    """Invented numeric tensor fixture, not scenario observations or outcomes."""
    schema = contract.inputs
    n, f, g, a = (len(schema.node_ids), len(schema.node_feature_names),
                  len(schema.global_feature_names), len(schema.action_names))
    nodes = torch.arange(1, n * f + 1, dtype=torch.float32).reshape(1, n, f) / (n * f)
    links = torch.zeros(1, n, n)
    for i in range(n - 1):
        links[0, i, i + 1] = links[0, i + 1, i] = 1
    observation = ObservationBatch(schema, nodes, torch.zeros(1, g), links)
    anchor, proposal = torch.zeros(1, a), torch.full((1, a), .1)
    return {role: build_matched_inputs(observation, schema, anchor, role=role, message_mode=mode,
                                       **({} if role == "actor" else {"action": proposal}),
                                       **({"detach_proposal": True} if role == "gate" else {}))
            for role in COMPONENTS}


def verify_modules(contract, graph_width, flat_width):
    graph = _agent(contract, "graph", graph_width)
    no_messages = _agent(contract, "graph", graph_width, "self_only")
    flat = _agent(contract, "flat", flat_width)
    if graph.weights_digest() != no_messages.weights_digest():
        raise AssertionError("message ablation changed model parameters or buffers")
    physical, self_only = _views(contract, "physical"), _views(contract, "self_only")
    for name in COMPONENTS:
        if not torch.equal(physical[name].flat, self_only[name].flat):
            raise AssertionError("message ablation changed available numeric information")
    if len(contract.inputs.node_ids) < 2 or torch.equal(
            physical["actor"].message_adjacency, self_only["actor"].message_adjacency):
        raise AssertionError("operator check needs at least one physical link")
    actual, participation = {}, {}
    for name, agent, width in (("graph", graph, graph_width), ("flat", flat, flat_width)):
        before = agent.weights_digest()
        inventory = agent.inventory()
        expected = parameter_counts(contract, name, width)
        if (inventory["components"] != {key: expected[key] for key in COMPONENTS}
                or inventory["total_unique"] != expected["total"]
                or inventory["shared_duplicate_numel"] != 0):
            raise AssertionError("analytical count differs from real module inventory")
        actual[name] = inventory
        participation[name] = {}
        for component in COMPONENTS:
            head = getattr(agent, component)
            head.requires_grad_(True)
            output = head(physical[component])
            if not torch.isfinite(output).all().item():
                raise AssertionError("nonfinite diagnostic forward")
            output.sum().backward()
            parameters = list(head.parameters())
            if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in parameters):
                raise AssertionError("unused padding or disconnected parameter")
            participation[name][component] = sum(p.numel() for p in parameters)
            head.zero_grad(set_to_none=True)
            head.requires_grad_(False)
        if before != agent.weights_digest() or any(p.grad is not None or p.requires_grad for p in agent.parameters()):
            raise AssertionError("differentiation check mutated weights or left trainable state")
    with torch.no_grad():
        for component in COMPONENTS:
            head = getattr(flat, component)
            if not torch.equal(head(physical[component]), head(self_only[component])):
                raise AssertionError("flat network unexpectedly consumed neural adjacency")
    return {"actual_inventory": actual, "backward_participation_numel": participation,
            "physical_self_only_weights_identical": True, "operator_only_change": True,
            "flat_outputs_invariant_to_neural_adjacency": True, "weights_unchanged": True,
            "nonzero_or_useful_gradient_claimed": False, "target_copies_in_count": False}


def validate_config(config):
    if config.get("kind") != "prospective_design_preflight_not_training":
        raise ValueError("preflight-only config required")
    permissions = config.get("permissions", {})
    if set(permissions) != set(PERMISSIONS) or any(permissions[key] is not False for key in PERMISSIONS):
        raise ValueError("all execution permissions must explicitly be false")
    if config.get("selection_rule") != "min_max_component_gap_then_total_gap_then_width":
        raise ValueError("unexpected count-selection rule")
    objective = config.get("proposed_training_objective", {})
    expected = {"reward_kind": "absolute_environment", "reward_definition": "negative_complete_step_cost",
                "reward_scale_rule": "single_positive_constant_frozen_from_pretraining_units",
                "gamma": 1.0, "return_steps": 1, "task": "finite_fully_settled_episode",
                "terminal_bootstrap": False, "collector_cut_bootstrap": True,
                "legacy_cache_import": False, "online_gate_parameter_updates": False}
    if any(objective.get(key) != value or (isinstance(value, bool) and objective.get(key) is not value)
           for key, value in expected.items()):
        raise ValueError("proposed objective differs from the finite settled-task design")
    if (isinstance(objective.get("gamma"), bool)
            or type(objective.get("return_steps")) is not int):
        raise ValueError("numeric objective fields cannot be flags or implicit conversions")
    comparators = config.get("required_deployable_comparators", [])
    if len(comparators) != 4 or set(comparators) != {
            "competent_frozen_history_policy", "online_ddpg", "adaptive_rule", "online_identification_mpc"}:
        raise ValueError("required comparator omitted or changed")
    if config.get("primary_contrast") != "same_initial_graph_policy_online_minus_frozen":
        raise ValueError("online attribution contrast changed")
    if config.get("secondary_contrasts") != [
            "physical_messages_minus_self_only_same_architecture",
            "common_information_parameter_matched_graph_minus_flat"]:
        raise ValueError("secondary attribution contrasts changed")
    evidence = config.get("scientific_launch_evidence")
    if not isinstance(evidence, dict) or set(evidence) != LAUNCH_REQUIREMENTS:
        raise ValueError("explicit unresolved launch evidence required")
    return [{"requirement": key, "status": "missing" if value is None else "unreviewed_reference"}
            for key, value in evidence.items()]


def load_contract(path, expected_sha256):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("immutable N4 schema-source hash mismatch")
    saved = json.loads(content)["cases"][0]["contract"]
    schema = InputSchema(**{key: value if key == "definition_id" else tuple(value)
                            for key, value in saved["inputs"].items()})
    return ReplayInputContract(schema, ReplaySemantics(**saved["replay"]))


def audit(config, root=ROOT):
    blockers = validate_config(config)
    contract = load_contract(root / config["schema_source"], config["schema_source_sha256"])
    matching = match_flat_width(contract, graph_width=config["graph_hidden_width"],
                                flat_range=config["flat_hidden_width_range"],
                                component_limit=config["max_component_gap_fraction"],
                                total_limit=config["max_total_gap_fraction"])
    selected = matching["selected"]
    verification = verify_modules(contract, config["graph_hidden_width"], selected["width"]) if selected else None
    return {"kind": "count_and_operator_preflight_not_performance", "input_schema": asdict(contract.inputs),
            "schema_source_sha256": config["schema_source_sha256"], "matching": matching,
            "module_verification": verification, "parameter_preflight_passed": selected is not None,
            "launch_authorized": False, "environment_steps": 0, "optimizer_updates": 0,
            "scientific_launch_blockers": blockers, "current_engineering_schema_only": True,
            "online_gain_claimed": False, "state": "preparation_only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; never overwrite evidence")
    config_path = ROOT / "experiments/configs/prospective_development_preflight_20260929.json"
    if args.config.resolve() != config_path:
        parser.error("use the committed N5 preflight config")
    dirty = subprocess.check_output([
        "git", "status", "--porcelain", "--untracked-files=all", "--", "src",
        "evaluation/audit_prospective_design.py", str(config_path.relative_to(ROOT)),
        "specs/2026-09-29-prospective-development-design/protocol.md"], cwd=ROOT, text=True)
    if dirty.strip():
        parser.error("commit source/config/protocol before recorded preflight")
    result = audit(json.loads(args.config.read_text()))
    result["execution_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    modules = ("evaluation/audit_prospective_design.py", "src/models/matched_inputs.py",
               "src/models/prospective_forward_agent.py", "src/models/gcn.py", "src/rl/networks.py",
               "src/rl/prospective_adapter.py", "src/rl/validated_returns.py")
    result["audited_source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in modules}
    result["runtime"] = {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"parameter_preflight_passed": result["parameter_preflight_passed"],
                      "launch_authorized": False, "blockers": len(result["scientific_launch_blockers"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
