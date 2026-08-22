"""Read-only Stage G0 audit of DDPG critic-to-action transfer.

The audit uses immutable Stage F1 checkpoints and fresh diagnostic CRNs. It
does not update parameters. For fixed frozen-pretrain trajectory states, it
compares finite-horizon legal-action values with each checkpoint critic, its
straight-through action gradient, and the action that actually survives
projection and integer patient-lot quantization.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.audit_ddpg_online_identifiability import (
    collect_frozen_trajectory_snapshots,
    critic_legal_action_diagnostics,
    sha256_file,
)
from evaluation.run_gcn_residual_sweep import mean_rollout_metrics_after_action
from src.baselines.heuristics import get_heuristic_class
from src.models.graph_features import flat_state_to_node_features
from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env, write_rows
from src.rl.networks import torch


DEFAULT_CONFIG = Path(
    "experiments/configs/"
    "patient_indexed_specimen_routing_ddpg_actor_projection_transfer_g0.json"
)
FORMAL_HOLDOUT_SEED = 91_100_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.smoke:
        config = smoke_config(config)
    if args.output_root:
        config["output_root"] = str(args.output_root)
    result = run_audit(config, config_path=config_path, smoke=bool(args.smoke))
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    first_seed = int(config["training_seeds"][0])
    smoke["training_seeds"] = [first_seed]
    smoke["scenario_by_training_seed"] = {
        str(first_seed): str(
            config["scenario_by_training_seed"][str(first_seed)]
        )
    }
    smoke["audit"]["checkpoint_variants"] = ["pretrain", "final"]
    smoke["audit"]["decision_steps"] = [
        int(config["audit"]["decision_steps"][0])
    ]
    smoke["audit"]["horizons"] = [4]
    smoke["audit"]["rollout_replications"] = 1
    smoke["stage_g0_gate"]["primary_horizon"] = "4"
    smoke["output_root"] = str(config["smoke_output_root"])
    smoke["name"] = f"{config['name']}_smoke"
    return smoke


def validate_config(config: dict[str, Any]) -> None:
    if str(config["primary_algorithm"]) != "gcn_residual_mdl2_network_ddpg_afd":
        raise ValueError("Stage G0 is locked to the primary GCN-DDPG algorithm")
    roles = dict(config["roles"])
    if set(roles) != {"control", "candidate"}:
        raise ValueError("Stage G0 requires control and candidate manifests")
    seeds = tuple(int(value) for value in config["training_seeds"])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Stage G0 training seeds must be unique and non-empty")
    assignment = {
        int(seed): str(scenario)
        for seed, scenario in config["scenario_by_training_seed"].items()
    }
    if set(assignment) != set(seeds):
        raise ValueError("Stage G0 scenario assignment must match training seeds")

    audit = dict(config["audit"])
    variants = tuple(str(value) for value in audit["checkpoint_variants"])
    if "pretrain" not in variants or "final" not in variants:
        raise ValueError("Stage G0 requires pretrain and final checkpoints")
    suffixes = dict(audit["checkpoint_suffixes"])
    if any(variant not in suffixes for variant in variants):
        raise ValueError("Every Stage G0 checkpoint variant needs a suffix")
    steps = tuple(int(value) for value in audit["decision_steps"])
    max_steps = int(audit["max_steps_per_episode"])
    if not steps or len(steps) != len(set(steps)):
        raise ValueError("Stage G0 decision steps must be unique and non-empty")
    if any(step < 0 or step >= max_steps for step in steps):
        raise ValueError("Stage G0 decision step is outside the episode")
    if int(audit["rollout_replications"]) <= 0:
        raise ValueError("Stage G0 rollout replications must be positive")
    horizons = tuple(audit["horizons"])
    if not horizons:
        raise ValueError("Stage G0 requires at least one horizon")
    if any(value != "remaining" and int(value) <= 0 for value in horizons):
        raise ValueError("Stage G0 horizons must be positive")

    forbidden = {int(value) for value in config["forbidden_crn_seeds"]}
    if FORMAL_HOLDOUT_SEED not in forbidden:
        raise ValueError("Stage G0 must explicitly forbid the formal holdout")
    used = diagnostic_crn_seed_sequence(config)
    if len(used) != len(set(used)):
        raise ValueError("Stage G0 diagnostic CRN streams overlap")
    overlap = sorted(set(used) & forbidden)
    if overlap:
        raise ValueError(f"Stage G0 uses forbidden CRN seeds: {overlap}")

    gates = dict(config["stage_g0_gate"])
    fraction_keys = (
        "minimum_headroom_state_fraction",
        "minimum_candidate_critic_top1_accuracy",
        "minimum_candidate_critic_gain",
        "minimum_critic_gradient_accuracy_gap",
        "maximum_incremental_executed_action_difference_fraction",
        "minimum_incremental_quantization_collapse_fraction",
    )
    for key in fraction_keys:
        value = float(gates[key])
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"Stage G0 gate {key} must be in [0, 1]")


def diagnostic_crn_seed_sequence(config: dict[str, Any]) -> list[int]:
    audit = dict(config["audit"])
    seeds = tuple(int(value) for value in config["training_seeds"])
    steps = tuple(int(value) for value in audit["decision_steps"])
    horizons = tuple(audit["horizons"])
    replications = int(audit["rollout_replications"])
    result = [
        int(audit["live_seed"]) + index * 100_000
        for index, _seed in enumerate(seeds)
    ]
    for scenario_index, _seed in enumerate(seeds):
        for step in steps:
            for horizon_index, _horizon in enumerate(horizons):
                start = (
                    int(audit["rollout_seed"])
                    + scenario_index * 1_000_000
                    + step * 100
                    + horizon_index * 10
                )
                result.extend(start + rep for rep in range(replications))
    return result


def immutable_tree_snapshot(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    artifacts = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"), key=lambda value: str(value))
        if path.is_file()
    }
    payload = json.dumps(
        artifacts,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "root": str(root),
        "artifact_count": len(artifacts),
        "artifact_tree_sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_locked_inputs(config: dict[str, Any]) -> dict[str, Any]:
    files = {}
    for entry in config["locked_files"]:
        name = str(entry["name"])
        path = Path(str(entry["path"]))
        observed = sha256_file(path)
        if observed != str(entry["sha256"]):
            raise ValueError(f"Stage G0 locked file changed: {name}")
        files[name] = {"path": str(path), "sha256": observed}
    trees = {}
    for entry in config["locked_trees"]:
        name = str(entry["name"])
        snapshot = immutable_tree_snapshot(Path(str(entry["root"])))
        if snapshot["artifact_tree_sha256"] != str(entry["sha256"]):
            raise ValueError(f"Stage G0 locked tree changed: {name}")
        if int(snapshot["artifact_count"]) != int(entry["artifact_count"]):
            raise ValueError(f"Stage G0 locked tree count changed: {name}")
        trees[name] = snapshot
    return {"files": files, "trees": trees}


def run_audit(
    config: dict[str, Any],
    *,
    config_path: Path | None,
    smoke: bool = False,
) -> dict[str, Any]:
    validate_config(config)
    output_root = Path(str(config["output_root"]))
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    before = verify_locked_inputs(config)
    manifests = {
        role: _load_json(Path(str(entry["training_manifest"])))
        for role, entry in config["roles"].items()
    }
    runs = {
        role: manifest_runs(
            manifest,
            algorithm=str(config["primary_algorithm"]),
            seeds=tuple(int(value) for value in config["training_seeds"]),
        )
        for role, manifest in manifests.items()
    }
    comparison = _load_json(Path(str(config["comparison_summary"])))
    if bool(
        comparison["decision"]["candidate_final_vs_frozen_gate_passed"]
    ):
        raise ValueError("Stage G0 requires the failed Stage F1 primary gate")

    rollout_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    paired_state_records: dict[
        tuple[int, int, str, str], dict[str, Any]
    ] = {}
    audit = dict(config["audit"])
    algorithm = str(config["primary_algorithm"])
    for scenario_index, seed in enumerate(config["training_seeds"]):
        seed = int(seed)
        control_run = runs["control"][seed]
        control_runtime = load_config(Path(control_run["config"]))
        control_runtime["device"] = str(audit.get("device", "cpu"))
        control_runtime["replay_buffer_size"] = 1
        env = build_env(
            control_runtime,
            seed=int(audit["live_seed"]) + scenario_index * 100_000,
        )
        control_paths = checkpoint_paths(
            control_run,
            variants=tuple(str(v) for v in audit["checkpoint_variants"]),
            suffixes=dict(audit["checkpoint_suffixes"]),
        )
        frozen = get_agent_class(algorithm)(
            env.observation_size,
            env.action_size,
            control_runtime,
        )
        frozen.load_actor(control_paths["pretrain"])
        anchor_name = str(
            control_runtime["residual_action"].get("base_policy", "mdl2")
        )
        anchor = get_heuristic_class(anchor_name)(
            env.observation_size,
            env.action_size,
            dict(
                control_runtime["residual_action"].get(
                    "base_policy_config", {}
                )
            ),
        )
        snapshot_config = {
            "replay": {"max_steps_per_episode": int(audit["max_steps_per_episode"])},
            "manifold": {
                "decision_steps": list(audit["decision_steps"]),
                "explicit_options": list(audit["explicit_options"]),
                "live_seed": int(audit["live_seed"]),
            },
        }
        snapshots = collect_frozen_trajectory_snapshots(
            snapshot_config,
            env=env,
            frozen=frozen,
            anchor=anchor,
            scenario_index=scenario_index,
        )
        true_values = rollout_legal_actions(
            config,
            snapshots=snapshots,
            anchor=anchor,
            training_seed=seed,
            scenario_index=scenario_index,
        )
        rollout_rows.extend(true_values["rows"])

        for role in ("control", "candidate"):
            run = runs[role][seed]
            runtime = load_config(Path(run["config"]))
            runtime["device"] = str(audit.get("device", "cpu"))
            runtime["replay_buffer_size"] = 1
            paths = checkpoint_paths(
                run,
                variants=tuple(str(v) for v in audit["checkpoint_variants"]),
                suffixes=dict(audit["checkpoint_suffixes"]),
            )
            for variant, checkpoint_path in paths.items():
                agent = get_agent_class(algorithm)(
                    env.observation_size,
                    env.action_size,
                    runtime,
                )
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=agent.device,
                    weights_only=False,
                )
                agent.load_actor(checkpoint_path)
                agent.critic.load_state_dict(checkpoint["critic"])
                agent.actor.eval()
                agent.critic.eval()
                for snapshot in snapshots:
                    records = audit_checkpoint_snapshot(
                        config,
                        agent=agent,
                        snapshot=snapshot,
                        true_values=true_values["by_step"],
                        role=role,
                        checkpoint_variant=variant,
                        training_seed=seed,
                    )
                    diagnostic_rows.extend(records["rows"])
                    paired_state_records[
                        (seed, int(snapshot["step"]), role, variant)
                    ] = records["state_record"]
                del checkpoint

    after = verify_locked_inputs(config)
    if before != after:
        raise RuntimeError("Stage G0 changed an immutable Stage F1 input")

    summary_metrics = summarize_diagnostic_rows(
        diagnostic_rows,
        material_improvement=float(audit["material_improvement"]),
    )
    cross_arm = summarize_cross_arm(
        paired_state_records,
        seeds=tuple(int(v) for v in config["training_seeds"]),
        steps=tuple(int(v) for v in audit["decision_steps"]),
        variants=tuple(str(v) for v in audit["checkpoint_variants"]),
    )
    training_scale = training_scale_diagnostics(
        runs,
        algorithm=algorithm,
    )
    decision = classify_actor_transfer(
        summary_metrics,
        cross_arm=cross_arm,
        gates=dict(config["stage_g0_gate"]),
    )

    rollout_path = output_root / "rollout_rows.csv"
    diagnostic_path = output_root / "transfer_rows.csv"
    write_rows(rollout_rows, rollout_path)
    write_rows(diagnostic_rows, diagnostic_path)
    result = {
        "name": str(config["name"]),
        "experimental_role": str(config["experimental_role"]),
        "smoke": bool(smoke),
        "config": None if config_path is None else str(config_path),
        "config_sha256": (
            None
            if config_path is None or not config_path.is_file()
            else sha256_file(config_path)
        ),
        "formal_holdout_reused": False,
        "stage_f1_decision": comparison["decision"],
        "locked_inputs": before,
        "fresh_crn_seeds": diagnostic_crn_seed_sequence(config),
        "rollout_rows": str(rollout_path),
        "rollout_rows_sha256": sha256_file(rollout_path),
        "transfer_rows": str(diagnostic_path),
        "transfer_rows_sha256": sha256_file(diagnostic_path),
        "metrics": summary_metrics,
        "cross_arm": cross_arm,
        "training_scale": training_scale,
        "decision": decision,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary"] = str(summary_path)
    result["summary_sha256"] = sha256_file(summary_path)
    return result


def manifest_runs(
    manifest: dict[str, Any],
    *,
    algorithm: str,
    seeds: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    result = {
        int(run["seed"]): dict(run)
        for run in manifest["runs"]
        if str(run["algorithm"]) == algorithm and int(run["seed"]) in seeds
    }
    if set(result) != set(seeds):
        raise ValueError("Stage G0 training manifest is incomplete")
    return result


def checkpoint_paths(
    run: dict[str, Any],
    *,
    variants: tuple[str, ...],
    suffixes: dict[str, str],
) -> dict[str, Path]:
    summary = _load_json(Path(run["config"]).parent / "summary.json")
    checkpoint_dir = Path(run["checkpoint"]).parent
    algorithm = str(run["algorithm"])
    seed = int(run["seed"])
    result = {}
    for variant in variants:
        if variant == "pretrain":
            path = Path(summary["pretrain_checkpoint"])
        elif variant == "final":
            path = Path(run["checkpoint"])
        else:
            path = checkpoint_dir / (
                f"{algorithm}_seed{seed}_{suffixes[variant]}.pt"
            )
        if not path.is_file():
            raise FileNotFoundError(path)
        result[variant] = path
    return result


def rollout_legal_actions(
    config: dict[str, Any],
    *,
    snapshots: list[dict[str, Any]],
    anchor: Any,
    training_seed: int,
    scenario_index: int,
) -> dict[str, Any]:
    audit = dict(config["audit"])
    rows = []
    by_step: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for snapshot in snapshots:
        for horizon_index, configured in enumerate(audit["horizons"]):
            horizon = (
                int(snapshot["remaining_horizon"])
                if configured == "remaining"
                else min(int(configured), int(snapshot["remaining_horizon"]))
            )
            seed_start = (
                int(audit["rollout_seed"])
                + scenario_index * 1_000_000
                + int(snapshot["step"]) * 100
                + horizon_index * 10
            )
            rollout_seeds = tuple(
                seed_start + index
                for index in range(int(audit["rollout_replications"]))
            )
            metrics = []
            for action in snapshot["actions"]:
                if hasattr(anchor, "reset"):
                    anchor.reset()
                metrics.append(
                    mean_rollout_metrics_after_action(
                        snapshot["env"],
                        anchor,
                        action,
                        horizon=horizon,
                        rollout_seeds=rollout_seeds,
                    )
                )
            anchor_cost = float(metrics[0]["total_cost"])
            group = []
            for index, (label, values) in enumerate(
                zip(snapshot["labels"], metrics)
            ):
                row = {
                    "training_seed": int(training_seed),
                    "scenario": str(
                        config["scenario_by_training_seed"][str(training_seed)]
                    ),
                    "step": int(snapshot["step"]),
                    "configured_horizon": str(configured),
                    "horizon": int(horizon),
                    "rollout_seed_start": int(seed_start),
                    "rollout_replications": len(rollout_seeds),
                    "candidate_index": int(index),
                    "candidate_label": str(label),
                    "total_cost": float(values["total_cost"]),
                    "true_cost_advantage": (
                        anchor_cost - float(values["total_cost"])
                    ),
                    "completion_service_level": float(
                        values["completion_service_level"]
                    ),
                    "patients_lost": float(values["patients_lost"]),
                    "patient_ineligibility_during_manufacturing_rate": float(
                        values["patient_ineligibility_during_manufacturing_rate"]
                    ),
                }
                rows.append(row)
                group.append(row)
            by_step[(int(snapshot["step"]), str(configured))] = group
    return {"rows": rows, "by_step": by_step}


def audit_checkpoint_snapshot(
    config: dict[str, Any],
    *,
    agent: Any,
    snapshot: dict[str, Any],
    true_values: dict[tuple[int, str], list[dict[str, Any]]],
    role: str,
    checkpoint_variant: str,
    training_seed: int,
) -> dict[str, Any]:
    state = np.asarray(snapshot["state"], dtype=np.float32)
    actions = np.asarray(snapshot["actions"], dtype=np.float32)
    policy_action = np.asarray(
        agent.select_action(state, explore=False, env=snapshot["env"]),
        dtype=np.float32,
    )
    network_action = actor_network_action(agent, state)
    critic = critic_legal_action_diagnostics(
        agent,
        state=state,
        actions=actions,
        policy_action=policy_action,
    )
    policy_id = executed_action_identifier(agent, policy_action)
    distinct = [
        index
        for index, first in enumerate(critic["first_indices"])
        if index == int(first)
    ]
    critic_best = max(distinct, key=lambda i: float(critic["q_values"][i]))
    gradient_best = max(
        distinct,
        key=lambda i: float(critic["gradient_scores"][i]),
    )
    policy_match = next(
        (
            index
            for index in distinct
            if critic["executed_action_ids"][index] == policy_id
        ),
        -1,
    )
    rows = []
    for configured in config["audit"]["horizons"]:
        group = true_values[(int(snapshot["step"]), str(configured))]
        true_by_index = {
            int(row["candidate_index"]): float(row["true_cost_advantage"])
            for row in group
        }
        true_best = max(distinct, key=lambda i: true_by_index[i])
        for index in distinct:
            rows.append(
                {
                    "role": str(role),
                    "checkpoint_variant": str(checkpoint_variant),
                    "training_seed": int(training_seed),
                    "scenario": str(
                        config["scenario_by_training_seed"][str(training_seed)]
                    ),
                    "step": int(snapshot["step"]),
                    "configured_horizon": str(configured),
                    "candidate_index": int(index),
                    "candidate_label": str(snapshot["labels"][index]),
                    "executed_action_id": str(
                        critic["executed_action_ids"][index]
                    ),
                    "policy_executed_action_id": str(policy_id),
                    "policy_candidate_index": int(policy_match),
                    "true_cost_advantage": float(true_by_index[index]),
                    "true_best_index": int(true_best),
                    "critic_q": float(critic["q_values"][index]),
                    "critic_best_index": int(critic_best),
                    "gradient_score": float(critic["gradient_scores"][index]),
                    "gradient_best_index": int(gradient_best),
                    "policy_linf_to_candidate": float(
                        np.max(np.abs(policy_action - actions[index]))
                    ),
                }
            )
    return {
        "rows": rows,
        "state_record": {
            "network_action": network_action,
            "policy_action": policy_action,
            "policy_executed_action_id": policy_id,
            "critic_best_index": int(critic_best),
            "gradient_best_index": int(gradient_best),
            "policy_candidate_index": int(policy_match),
        },
    }


def actor_network_action(agent: Any, state: np.ndarray) -> np.ndarray:
    raw = torch.as_tensor(
        np.asarray(state, dtype=np.float32).reshape(1, -1),
        dtype=torch.float32,
        device=agent.device,
    )
    with torch.no_grad():
        if hasattr(agent, "graph_spec"):
            actor_input = flat_state_to_node_features(raw, agent.graph_spec)
        else:
            normalized = agent.observation_scaler.normalize_tensor(raw)
            actor_input = agent._actor_input_tensor(
                raw,
                normalized_states=normalized,
            )
        value = agent.actor(actor_input)
    return value.detach().cpu().numpy().reshape(-1)


def executed_action_identifier(agent: Any, action: np.ndarray) -> str:
    tensor = torch.as_tensor(
        np.asarray(action, dtype=np.float32).reshape(1, -1),
        dtype=torch.float32,
        device=agent.device,
    )
    with torch.no_grad():
        executed = agent._critic_actions_tensor(tensor)
    value = executed.detach().cpu().numpy().reshape(-1)
    return hashlib.sha256(
        np.round(value.astype(np.float64), decimals=7).tobytes()
    ).hexdigest()[:16]


def summarize_diagnostic_rows(
    rows: list[dict[str, Any]],
    *,
    material_improvement: float,
) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["role"]),
                str(row["checkpoint_variant"]),
                str(row["configured_horizon"]),
            )
        ].append(row)
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (role, checkpoint, horizon), group in sorted(groups.items()):
        result[role][checkpoint][horizon] = transfer_metrics(
            group,
            material_improvement=material_improvement,
        )
    return {
        role: {checkpoint: dict(values) for checkpoint, values in checkpoints.items()}
        for role, checkpoints in result.items()
    }


def transfer_metrics(
    rows: list[dict[str, Any]],
    *,
    material_improvement: float,
) -> dict[str, Any]:
    states: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        states[(int(row["training_seed"]), int(row["step"]))].append(row)
    headroom = []
    critic_correct = []
    gradient_correct = []
    policy_correct = []
    policy_legal = []
    critic_gradient = []
    for group in states.values():
        first = group[0]
        true_best = int(first["true_best_index"])
        critic_best = int(first["critic_best_index"])
        gradient_best = int(first["gradient_best_index"])
        policy_index = int(first["policy_candidate_index"])
        maximum = max(float(row["true_cost_advantage"]) for row in group)
        headroom.append(maximum >= float(material_improvement))
        critic_correct.append(critic_best == true_best)
        gradient_correct.append(gradient_best == true_best)
        policy_correct.append(policy_index == true_best)
        policy_legal.append(policy_index >= 0)
        critic_gradient.append(critic_best == gradient_best)
    return {
        "states": len(states),
        "material_headroom_fraction": float(np.mean(headroom)),
        "critic_top1_true_accuracy": float(np.mean(critic_correct)),
        "gradient_top1_true_accuracy": float(np.mean(gradient_correct)),
        "policy_top1_true_accuracy": float(np.mean(policy_correct)),
        "policy_legal_candidate_fraction": float(np.mean(policy_legal)),
        "critic_gradient_top1_agreement": float(np.mean(critic_gradient)),
    }


def summarize_cross_arm(
    records: dict[tuple[int, int, str, str], dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    steps: tuple[int, ...],
    variants: tuple[str, ...],
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    by_variant = {}
    for variant in variants:
        network_deltas = []
        policy_deltas = []
        executed_different = []
        network_different = []
        collapse = []
        critic_different = []
        gradient_different = []
        for seed in seeds:
            for step in steps:
                control = records[(seed, step, "control", variant)]
                candidate = records[(seed, step, "candidate", variant)]
                network_delta = float(
                    np.max(
                        np.abs(
                            candidate["network_action"]
                            - control["network_action"]
                        )
                    )
                )
                policy_delta = float(
                    np.max(
                        np.abs(
                            candidate["policy_action"]
                            - control["policy_action"]
                        )
                    )
                )
                different = (
                    candidate["policy_executed_action_id"]
                    != control["policy_executed_action_id"]
                )
                network_deltas.append(network_delta)
                policy_deltas.append(policy_delta)
                executed_different.append(different)
                changed = network_delta > tolerance
                network_different.append(changed)
                collapse.append(changed and not different)
                critic_different.append(
                    candidate["critic_best_index"]
                    != control["critic_best_index"]
                )
                gradient_different.append(
                    candidate["gradient_best_index"]
                    != control["gradient_best_index"]
                )
        by_variant[variant] = {
            "states": len(network_deltas),
            "network_action_linf_mean": float(np.mean(network_deltas)),
            "network_action_linf_max": float(np.max(network_deltas)),
            "projected_policy_linf_mean": float(np.mean(policy_deltas)),
            "projected_policy_linf_max": float(np.max(policy_deltas)),
            "network_action_difference_fraction": float(
                np.mean(network_different)
            ),
            "executed_action_difference_fraction": float(
                np.mean(executed_different)
            ),
            "incremental_quantization_collapse_fraction": float(
                sum(collapse) / max(sum(network_different), 1)
            ),
            "critic_top1_difference_fraction": float(
                np.mean(critic_different)
            ),
            "gradient_top1_difference_fraction": float(
                np.mean(gradient_different)
            ),
        }
    pretrain = by_variant["pretrain"]
    if pretrain["network_action_linf_max"] > tolerance:
        raise ValueError("Stage F1 control/candidate pretrain actors are not identical")
    if pretrain["executed_action_difference_fraction"] != 0.0:
        raise ValueError("Stage F1 control/candidate pretrain actions are not identical")
    return by_variant


def training_scale_diagnostics(
    runs: dict[str, dict[int, dict[str, Any]]],
    *,
    algorithm: str,
) -> dict[str, Any]:
    del algorithm
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    fields = (
        "online_rl_critic_online_paired_advantage_sign_accuracy_mean",
        "online_rl_critic_online_paired_advantage_weighted_loss_mean",
        "online_rl_critic_loss_mean",
        "online_rl_pretrain_reference_action_mse_mean",
        "online_rl_actor_loss_mean",
    )
    result = {}
    for role, role_runs in runs.items():
        per_seed = {}
        for seed, run in role_runs.items():
            training_path = Path(run["config"]).parent / "training.csv"
            with training_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            values = {}
            for field in fields:
                observed = [
                    float(row[field])
                    for row in rows
                    if row.get(field) not in (None, "")
                ]
                values[field] = (
                    None
                    if not observed
                    else float(np.mean(observed))
                )
            paired = values[
                "online_rl_critic_online_paired_advantage_weighted_loss_mean"
            ]
            critic = values["online_rl_critic_loss_mean"]
            values["paired_to_total_critic_loss_ratio"] = (
                None
                if paired is None or critic in (None, 0.0)
                else float(paired / critic)
            )
            values["actor_drift_from_pretrain"] = dict(
                run["actor_drift_from_pretrain"]
            )
            per_seed[str(seed)] = values
        result[role] = per_seed
    return result


def classify_actor_transfer(
    metrics: dict[str, Any],
    *,
    cross_arm: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    horizon = str(gates["primary_horizon"])
    control = metrics["control"]["final"][horizon]
    candidate = metrics["candidate"]["final"][horizon]
    final_cross = cross_arm["final"]
    headroom = float(candidate["material_headroom_fraction"])
    critic_accuracy = float(candidate["critic_top1_true_accuracy"])
    critic_gain = critic_accuracy - float(control["critic_top1_true_accuracy"])
    gradient_accuracy = float(candidate["gradient_top1_true_accuracy"])
    gradient_gap = critic_accuracy - gradient_accuracy
    execution_difference = float(
        final_cross["executed_action_difference_fraction"]
    )
    collapse = float(
        final_cross["incremental_quantization_collapse_fraction"]
    )

    if headroom < float(gates["minimum_headroom_state_fraction"]):
        classification = "insufficient_legal_action_headroom"
        intervention = "close_action_alignment_extension"
    elif critic_accuracy < float(
        gates["minimum_candidate_critic_top1_accuracy"]
    ) or critic_gain < float(gates["minimum_candidate_critic_gain"]):
        classification = "paired_critic_did_not_generalize"
        intervention = "normalize_and_margin_rank_legal_actions_before_actor_change"
    elif gradient_gap >= float(
        gates["minimum_critic_gradient_accuracy_gap"]
    ):
        classification = "straight_through_gradient_bottleneck"
        intervention = "critic_ranked_legal_action_selection"
    elif execution_difference <= float(
        gates["maximum_incremental_executed_action_difference_fraction"]
    ) and collapse >= float(
        gates["minimum_incremental_quantization_collapse_fraction"]
    ):
        classification = "actor_projection_transfer_bottleneck"
        intervention = "quantization_aware_advantage_weighted_actor_update"
    else:
        classification = "actor_transfer_inconclusive"
        intervention = "do_not_authorize_training"

    justified = classification in {
        "straight_through_gradient_bottleneck",
        "actor_projection_transfer_bottleneck",
    }
    return {
        "classification": classification,
        "recommended_intervention": intervention,
        "action_aligned_ddpg_design_justified": justified,
        "training_authorized": False,
        "formal_confirmation_authorized": False,
        "primary_horizon": horizon,
        "observed": {
            "material_headroom_state_fraction": headroom,
            "candidate_critic_top1_true_accuracy": critic_accuracy,
            "candidate_minus_control_critic_top1_accuracy": critic_gain,
            "candidate_gradient_top1_true_accuracy": gradient_accuracy,
            "critic_minus_gradient_accuracy": gradient_gap,
            "candidate_control_executed_action_difference_fraction": execution_difference,
            "incremental_quantization_collapse_fraction": collapse,
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


if __name__ == "__main__":
    main()
