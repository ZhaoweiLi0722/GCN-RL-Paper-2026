"""Audit the learnability contract for routing-primary residual DDPG.

This diagnostic never trains a learned policy. It measures whether the frozen
teacher actions fit candidate residual gains and whether the exact look-ahead
teacher has independent four-scenario headroom over the MDL-2 anchor.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evaluation.aggregate_stats import paired_two_level_summary
from evaluation.network_residual_headroom import run_online_teacher
from src.baselines.heuristics import (
    facility_net_action_from_state,
    heuristic_settings_for_policy,
)
from src.rl.config import load_config
from src.rl.residual_endpoint_projection import ResidualEndpointProjection


DEFAULT_CONFIG = (
    "experiments/configs/"
    "patient_indexed_specimen_routing_mac_mps_ddpg_teacher_audit.json"
)
METRICS = (
    "total_cost",
    "completion_service_level",
    "patients_lost",
    "patient_ineligibility_during_manufacturing_rate",
)


def facility_net_group_slices(num_facilities: int) -> dict[str, slice]:
    n = int(num_facilities)
    if n <= 0:
        raise ValueError("num_facilities must be positive")
    return {
        "specimen_transfer": slice(0, n),
        "reagent_transfer": slice(n, 2 * n),
        "capacity_transfer": slice(2 * n, 3 * n),
        "replenishment": slice(3 * n, 4 * n),
    }


def residual_scale_vector(
    action_dim: int,
    num_facilities: int,
    residual_config: dict[str, Any],
) -> np.ndarray:
    action_dim = int(action_dim)
    scale = float(residual_config.get("scale", 0.25))
    result = np.full(action_dim, scale, dtype=np.float64)
    group_scales = dict(residual_config.get("group_scales", {}))
    if not group_scales:
        return result
    if action_dim != 4 * int(num_facilities):
        raise ValueError("group scales require a four-group facility-net action")
    group_slices = facility_net_group_slices(num_facilities)
    for group, value in group_scales.items():
        if group not in group_slices:
            raise ValueError(f"unsupported residual group scale: {group}")
        result[group_slices[group]] = float(value)
    return result


def saturation_summary(
    base_actions: np.ndarray,
    teacher_actions: np.ndarray,
    *,
    num_facilities: int,
    base_scale_vector: np.ndarray,
    multipliers: Sequence[float],
    changed_tolerance: float = 1e-6,
) -> dict[str, Any]:
    base_actions = np.asarray(base_actions, dtype=np.float64)
    teacher_actions = np.asarray(teacher_actions, dtype=np.float64)
    scale_vector = np.asarray(base_scale_vector, dtype=np.float64)
    if base_actions.shape != teacher_actions.shape:
        raise ValueError("base and teacher actions must have matching shapes")
    if base_actions.ndim != 2 or base_actions.shape[1] != scale_vector.size:
        raise ValueError("actions and residual scale vector do not align")

    deltas = teacher_actions - base_actions
    group_slices = facility_net_group_slices(num_facilities)
    result: dict[str, Any] = {
        "samples": int(base_actions.shape[0]),
        "action_dim": int(base_actions.shape[1]),
        "changed_values": int(np.sum(np.abs(deltas) > changed_tolerance)),
        "multipliers": {},
    }
    for multiplier_value in multipliers:
        multiplier = float(multiplier_value)
        if multiplier <= 0.0:
            raise ValueError("residual scale multipliers must be positive")
        effective_scale = scale_vector * multiplier
        active = np.abs(effective_scale) > 1e-12
        safe_scale = np.where(active, effective_scale, 1.0)
        raw_targets = deltas / safe_scale.reshape(1, -1)
        changed = (np.abs(deltas) > changed_tolerance) & active.reshape(1, -1)
        saturated = np.abs(raw_targets) > 1.0 + 1e-7
        reconstructed = np.clip(raw_targets, -1.0, 1.0) * effective_scale.reshape(1, -1)
        reconstruction_error = np.abs(deltas - reconstructed)

        changed_count = int(changed.sum())
        active_mask = np.broadcast_to(active.reshape(1, -1), deltas.shape)
        active_count = int(active_mask.sum())
        group_results = {}
        for group, group_slice in group_slices.items():
            group_changed = changed[:, group_slice]
            group_saturated = saturated[:, group_slice]
            group_error = reconstruction_error[:, group_slice]
            group_count = int(group_changed.sum())
            group_results[group] = {
                "changed_values": group_count,
                "saturated_changed_values": int(
                    np.logical_and(group_changed, group_saturated).sum()
                ),
                "saturated_changed_fraction": (
                    float(group_saturated[group_changed].mean())
                    if group_count
                    else 0.0
                ),
                "mean_unrepresentable_abs_delta": (
                    float(group_error[group_changed].mean())
                    if group_count
                    else 0.0
                ),
            }
        result["multipliers"][f"{multiplier:g}"] = {
            "effective_active_scale_min": float(np.min(np.abs(effective_scale[active]))),
            "effective_active_scale_max": float(np.max(np.abs(effective_scale[active]))),
            "active_values": active_count,
            "changed_values": changed_count,
            "saturated_active_fraction": (
                float(saturated[active_mask].mean()) if active_count else 0.0
            ),
            "saturated_changed_values": int(
                np.logical_and(changed, saturated).sum()
            ),
            "saturated_changed_fraction": (
                float(saturated[changed].mean()) if changed_count else 0.0
            ),
            "mean_abs_raw_target_when_changed": (
                float(np.abs(raw_targets[changed]).mean())
                if changed_count
                else 0.0
            ),
            "p95_abs_raw_target_when_changed": (
                float(np.percentile(np.abs(raw_targets[changed]), 95.0))
                if changed_count
                else 0.0
            ),
            "mean_unrepresentable_abs_delta": (
                float(reconstruction_error[changed].mean())
                if changed_count
                else 0.0
            ),
            "by_group": group_results,
        }
    return result


def endpoint_support_summary(
    base_actions: np.ndarray,
    teacher_actions: np.ndarray,
    *,
    num_facilities: int,
    settings: dict[str, Any],
    changed_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Measure whether teacher residual support fits endpoint projection."""

    base = np.asarray(base_actions, dtype=np.float32)
    teacher = np.asarray(teacher_actions, dtype=np.float32)
    if base.shape != teacher.shape or base.ndim != 2:
        raise ValueError("base and teacher actions must be aligned matrices")
    projection = ResidualEndpointProjection(
        num_facilities=int(num_facilities),
        action_dim=int(base.shape[1]),
        settings=settings,
    )
    deltas = teacher - base
    projected = projection.apply_numpy(deltas)
    error = np.abs(deltas - projected)
    group_results = {}
    incompatible_rows = np.zeros(base.shape[0], dtype=bool)
    for group, group_slice in facility_net_group_slices(
        num_facilities
    ).items():
        values = deltas[:, group_slice]
        changed = np.abs(values) > changed_tolerance
        changed_rows = changed.any(axis=1)
        positive = (values > changed_tolerance).sum(axis=1)
        negative = (values < -changed_tolerance).sum(axis=1)
        if projection.enabled and group in projection.groups:
            incompatible = changed_rows & (
                (positive > projection.max_endpoints_per_side)
                | (negative > projection.max_endpoints_per_side)
            )
        else:
            incompatible = np.zeros_like(changed_rows)
        incompatible_rows |= incompatible
        group_results[group] = {
            "changed_rows": int(changed_rows.sum()),
            "incompatible_rows": int(incompatible.sum()),
            "incompatible_changed_fraction": (
                float(incompatible[changed_rows].mean())
                if np.any(changed_rows)
                else 0.0
            ),
            "mean_changed_endpoints": (
                float(changed.sum(axis=1)[changed_rows].mean())
                if np.any(changed_rows)
                else 0.0
            ),
            "mean_projection_abs_error_when_changed": (
                float(error[:, group_slice][changed].mean())
                if np.any(changed)
                else 0.0
            ),
        }
    any_changed = np.abs(deltas).max(axis=1) > changed_tolerance
    return {
        "projection_enabled": projection.enabled,
        "projection_groups": list(projection.groups),
        "max_endpoints_per_side": projection.max_endpoints_per_side,
        "min_abs": projection.min_abs,
        "changed_rows": int(any_changed.sum()),
        "incompatible_rows": int(incompatible_rows.sum()),
        "incompatible_changed_fraction": (
            float(incompatible_rows[any_changed].mean())
            if np.any(any_changed)
            else 0.0
        ),
        "mean_projection_abs_error": float(error.mean()),
        "by_group": group_results,
    }


def teacher_target_saturation(config: dict[str, Any]) -> dict[str, Any]:
    teacher_template = load_config(config["teacher_template_config"])
    benchmark = load_config(config["benchmark_config"])
    algorithm = str(config.get("algorithm", "gcn_residual_mdl2_network_ddpg_afd"))
    residual_config = dict(
        benchmark["algorithm_settings"][algorithm]["config_overrides"][
            "residual_action"
        ]
    )
    env_config = load_config(teacher_template["env_config"])
    env_config.update(dict(teacher_template.get("env_overrides", {})))
    settings = heuristic_settings_for_policy(
        str(residual_config.get("base_policy", "mdl2")),
        dict(residual_config.get("base_policy_config", {})),
    )
    with np.load(config["teacher_cache"], allow_pickle=False) as cache:
        states = np.asarray(cache["states"], dtype=np.float32)
        teacher_actions = np.asarray(cache["actions"], dtype=np.float32)
    base_actions = np.stack(
        [
            facility_net_action_from_state(
                state,
                env_config,
                settings=settings,
            )
            for state in states
        ],
        axis=0,
    )
    scales = residual_scale_vector(
        teacher_actions.shape[1],
        int(env_config["num_facilities"]),
        residual_config,
    )
    result = saturation_summary(
        base_actions,
        teacher_actions,
        num_facilities=int(env_config["num_facilities"]),
        base_scale_vector=scales,
        multipliers=config["residual_scale_multipliers"],
    )
    result.update(
        {
            "teacher_cache": str(config["teacher_cache"]),
            "algorithm": algorithm,
            "base_policy": str(residual_config.get("base_policy", "mdl2")),
            "base_group_scales": {
                group: float(scales[group_slice][0])
                for group, group_slice in facility_net_group_slices(
                    int(env_config["num_facilities"])
                ).items()
            },
        }
    )
    result["endpoint_support"] = endpoint_support_summary(
        base_actions,
        teacher_actions,
        num_facilities=int(env_config["num_facilities"]),
        settings=dict(residual_config.get("endpoint_projection", {})),
    )
    return result


def teacher_holdout_target_saturation(
    config: dict[str, Any],
    holdout: dict[str, Any],
) -> dict[str, Any]:
    """Measure generated scenario caches against their matching MDL-2 states."""

    benchmark = load_config(config["benchmark_config"])
    algorithm = str(
        config.get("algorithm", "gcn_residual_mdl2_network_ddpg_afd")
    )
    residual_config = dict(
        benchmark["algorithm_settings"][algorithm]["config_overrides"][
            "residual_action"
        ]
    )
    settings = heuristic_settings_for_policy(
        str(residual_config.get("base_policy", "mdl2")),
        dict(residual_config.get("base_policy_config", {})),
    )
    base_parts = []
    teacher_parts = []
    cache_paths = []
    num_facilities: int | None = None
    for scenario_name in holdout["scenarios"]:
        scenario_config = holdout["by_scenario"][scenario_name]["config"]
        env_config = load_config(scenario_config["env_config"])
        env_config.update(dict(scenario_config.get("env_overrides", {})))
        scenario_facilities = int(env_config["num_facilities"])
        if num_facilities is None:
            num_facilities = scenario_facilities
        elif scenario_facilities != num_facilities:
            raise ValueError("Holdout scenarios must share the facility count")
        cache_path = Path(scenario_config["demonstration_path"])
        with np.load(cache_path, allow_pickle=False) as cache:
            states = np.asarray(cache["states"], dtype=np.float32)
            actions = np.asarray(cache["actions"], dtype=np.float32)
        base_parts.append(
            np.stack(
                [
                    facility_net_action_from_state(
                        state,
                        env_config,
                        settings=settings,
                    )
                    for state in states
                ],
                axis=0,
            )
        )
        teacher_parts.append(actions)
        cache_paths.append(str(cache_path))
    if num_facilities is None:
        raise ValueError("Holdout saturation requires at least one scenario")
    base_actions = np.concatenate(base_parts, axis=0)
    teacher_actions = np.concatenate(teacher_parts, axis=0)
    scales = residual_scale_vector(
        teacher_actions.shape[1],
        num_facilities,
        residual_config,
    )
    result = saturation_summary(
        base_actions,
        teacher_actions,
        num_facilities=num_facilities,
        base_scale_vector=scales,
        multipliers=config["residual_scale_multipliers"],
    )
    result.update(
        {
            "teacher_caches": cache_paths,
            "saturation_source": "generated_holdout_caches",
            "algorithm": algorithm,
            "base_policy": str(residual_config.get("base_policy", "mdl2")),
            "base_group_scales": {
                group: float(scales[group_slice][0])
                for group, group_slice in facility_net_group_slices(
                    num_facilities
                ).items()
            },
        }
    )
    result["endpoint_support"] = endpoint_support_summary(
        base_actions,
        teacher_actions,
        num_facilities=num_facilities,
        settings=dict(residual_config.get("endpoint_projection", {})),
    )
    return result


def scenario_teacher_config(
    audit_config: dict[str, Any],
    teacher_template: dict[str, Any],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(teacher_template)
    scenario_name = str(scenario["name"])
    output_root = Path(audit_config["output_root"]) / "teacher_holdout" / scenario_name
    result.update(
        {
            "name": f"{audit_config['name']}_{scenario_name}",
            "env_config": str(scenario["env_config"]),
            "env_overrides": copy.deepcopy(scenario.get("env_overrides", {})),
            "seed": int(audit_config["teacher_seed"]),
            "lookahead_seed": int(audit_config["lookahead_seed"]),
            "teacher_replications": int(
                audit_config["teacher_replications_per_scenario"]
            ),
            "teacher_replication_start": 0,
            "lookahead_decision_offset": 0,
            "output_root": str(output_root),
            "demonstration_path": str(output_root / "teacher_cache.npz"),
        }
    )
    return result


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def completed_scenario_result(
    scenario_root: Path,
    scenario_config: dict[str, Any],
) -> dict[str, Any] | None:
    required = (
        scenario_root / "anchor.csv",
        scenario_root / "teacher.csv",
        scenario_root / "teacher_cache.npz",
        scenario_root / "summary.json",
    )
    existing = tuple(path.exists() for path in required)
    if not any(existing):
        return None
    if not all(existing):
        missing = [str(path) for path, present in zip(required, existing) if not present]
        raise RuntimeError(f"partial scenario audit cannot be resumed: {missing}")
    result = json.loads((scenario_root / "summary.json").read_text())
    expected_scenario = str(scenario_config["env_overrides"]["scenario_name"])
    expected_replications = int(scenario_config["teacher_replications"])
    expected_decisions = expected_replications * int(scenario_config["max_steps"])
    for summary_name in ("anchor_summary", "teacher_summary"):
        summary = result[summary_name]
        if str(summary["scenario"]) != expected_scenario:
            raise RuntimeError("completed scenario name does not match the audit config")
        if int(summary["replications"]) != expected_replications:
            raise RuntimeError("completed replication count does not match the audit config")
    if int(result["teacher_total_decisions"]) != expected_decisions:
        raise RuntimeError("completed teacher decisions do not match the audit config")
    if int(result["teacher_demonstration_samples"]) != expected_decisions:
        raise RuntimeError("completed teacher cache size does not match the audit config")
    return result


def teacher_holdout_audit(
    config: dict[str, Any],
    *,
    resume: bool = False,
) -> dict[str, Any]:
    benchmark = load_config(config["benchmark_config"])
    template = load_config(config["teacher_template_config"])
    scenario_names = tuple(str(name) for name in config["scenarios"])
    scenario_lookup = {
        str(scenario["name"]): scenario for scenario in benchmark["scenarios"]
    }
    missing = [name for name in scenario_names if name not in scenario_lookup]
    if missing:
        raise ValueError(f"audit scenarios are absent from benchmark: {missing}")

    by_scenario = {}
    teacher_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    resumed_scenarios: list[str] = []
    for scenario_name in scenario_names:
        scenario_config = scenario_teacher_config(
            config,
            template,
            scenario_lookup[scenario_name],
        )
        scenario_root = Path(scenario_config["output_root"])
        result = (
            completed_scenario_result(scenario_root, scenario_config)
            if resume
            else None
        )
        if result is None:
            scenario_root.mkdir(parents=True, exist_ok=False)
            (scenario_root / "scenario_config.json").write_text(
                json.dumps(scenario_config, indent=2, sort_keys=True) + "\n"
            )
            result = run_online_teacher(scenario_config)
            (scenario_root / "summary.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        else:
            resumed_scenarios.append(scenario_name)
        by_scenario[scenario_name] = {
            "config": scenario_config,
            "result": result,
        }
        teacher_rows.extend(read_csv_rows(scenario_root / "teacher.csv"))
        anchor_rows.extend(read_csv_rows(scenario_root / "anchor.csv"))

    pooled = {
        metric: paired_two_level_summary(
            teacher_rows,
            anchor_rows,
            metric=metric,
            pairing_keys=(
                "training_seed",
                "scenario",
                "evaluation_seed",
                "replication",
            ),
            resamples=int(config.get("bootstrap_resamples", 20000)),
            seed=int(config["bootstrap_seed"]),
        )
        for metric in METRICS
    }
    all_scenario_cost_better = all(
        float(value["result"]["paired"]["total_cost"]["mean_difference"]) < 0.0
        for value in by_scenario.values()
    )
    pooled_clinical_noninferiority = bool(
        float(pooled["completion_service_level"]["mean_difference"]) >= 0.0
        and float(pooled["patients_lost"]["mean_difference"]) <= 0.0
        and float(
            pooled["patient_ineligibility_during_manufacturing_rate"][
                "mean_difference"
            ]
        )
        <= 0.0
    )
    return {
        "screening_only": True,
        "scenarios": list(scenario_names),
        "replications_per_scenario": int(
            config["teacher_replications_per_scenario"]
        ),
        "resumed_complete_scenarios": resumed_scenarios,
        "by_scenario": by_scenario,
        "pooled": pooled,
        "decision": {
            "all_scenario_mean_cost_better_than_mdl2": all_scenario_cost_better,
            "pooled_cost_ci_high_below_zero": bool(
                float(pooled["total_cost"]["ci_high"]) < 0.0
            ),
            "pooled_clinical_noninferiority": pooled_clinical_noninferiority,
            "teacher_has_cross_scenario_headroom": bool(
                all_scenario_cost_better
                and float(pooled["total_cost"]["ci_high"]) < 0.0
                and pooled_clinical_noninferiority
            ),
        },
    }


def recommendation(
    saturation: dict[str, Any],
    holdout: dict[str, Any],
    *,
    target_multiplier: float,
) -> dict[str, Any]:
    multiplier_key = f"{float(target_multiplier):g}"
    target = saturation["multipliers"][multiplier_key]
    teacher_pass = bool(
        holdout["decision"]["teacher_has_cross_scenario_headroom"]
    )
    saturation_fraction = float(target["saturated_changed_fraction"])
    if not teacher_pass:
        next_step = "revise_or_expand_teacher_before_ddpg_training"
    elif saturation_fraction > 0.25:
        next_step = "retarget_teacher_to_bounded_gain_before_ddpg_training"
    else:
        next_step = "run_fresh_scale_aligned_ddpg_smoke"
    return {
        "target_multiplier": float(target_multiplier),
        "target_saturated_changed_fraction": saturation_fraction,
        "teacher_cross_scenario_headroom_pass": teacher_pass,
        "next_step": next_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(config["output_root"])
    if output_root.exists() and not args.resume:
        raise FileExistsError(
            f"audit output already exists; use --resume only for verified complete scenarios: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    holdout = teacher_holdout_audit(config, resume=args.resume)
    saturation_source = str(
        config.get("saturation_source", "configured_teacher_cache")
    )
    if saturation_source == "configured_teacher_cache":
        saturation = teacher_target_saturation(config)
    elif saturation_source == "generated_holdout_caches":
        saturation = teacher_holdout_target_saturation(config, holdout)
    else:
        raise ValueError(f"unsupported saturation_source: {saturation_source}")
    result = {
        "name": str(config["name"]),
        "experimental_role": str(config["experimental_role"]),
        "config": config,
        "target_saturation": saturation,
        "teacher_holdout": holdout,
        "recommendation": recommendation(
            saturation,
            holdout,
            target_multiplier=float(config["target_residual_scale_multiplier"]),
        ),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["recommendation"], indent=2, sort_keys=True))
    print(f"wrote DDPG teacher audit to {summary_path}")


if __name__ == "__main__":
    main()
