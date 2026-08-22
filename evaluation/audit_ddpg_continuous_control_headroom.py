"""Stage H0 frozen audit of DDPG-compatible continuous control surfaces.

The current patient-indexed specimen action is discretized into patient lots.
This development-only audit asks whether reagent and joint reagent/capacity
residuals provide a smoother and more reproducible control surface. It never
updates a model, selects a checkpoint, or reads a formal holdout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.audit_ddpg_actor_projection_transfer import (
    manifest_runs,
    verify_locked_inputs,
)
from evaluation.audit_ddpg_legal_action_ranker_feasibility import (
    load_pretrain_agent,
)
from evaluation.audit_ddpg_online_identifiability import (
    collect_frozen_trajectory_snapshots,
    sha256_file,
)
from evaluation.run_gcn_residual_sweep import mean_rollout_metrics_after_action
from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    make_scenario_env_config,
    select_scenarios,
)
from src.baselines.heuristics import get_heuristic_class
from src.rl.config import load_config
from src.rl.experiment import build_env, write_rows
from src.rl.residual_options import make_explicit_residual_option_specs


DEFAULT_CONFIG = Path(
    "experiments/configs/"
    "patient_indexed_specimen_routing_ddpg_continuous_control_headroom_h0.json"
)
FORMAL_HOLDOUT_SEED = 91_100_000
CONTINUOUS_GROUPS = ("reagent_transfer", "combined_transfer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_audit_config(config_path)
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
        str(first_seed): str(config["scenario_by_training_seed"][str(first_seed)])
    }
    smoke["audit"]["decision_steps"] = [
        int(config["audit"]["decision_steps"][0])
    ]
    smoke["audit"]["horizon"] = 4
    smoke["audit"]["discovery_replications"] = 1
    smoke["audit"]["validation_replications"] = 1
    smoke["name"] = f"{config['name']}_smoke"
    smoke["output_root"] = str(config["smoke_output_root"])
    return smoke


def validate_config(config: dict[str, Any]) -> None:
    if str(config["primary_algorithm"]) != "gcn_residual_mdl2_network_ddpg_afd":
        raise ValueError("Stage H0 is locked to the primary GCN-DDPG algorithm")
    seeds = tuple(int(value) for value in config["training_seeds"])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Stage H0 training seeds must be unique and non-empty")
    assignment = {
        int(seed): str(scenario)
        for seed, scenario in config["scenario_by_training_seed"].items()
    }
    if set(assignment) != set(seeds):
        raise ValueError("Stage H0 scenario assignment must match training seeds")

    audit = dict(config["audit"])
    steps = tuple(int(value) for value in audit["decision_steps"])
    max_steps = int(audit["max_steps_per_episode"])
    if not steps or len(steps) != len(set(steps)):
        raise ValueError("Stage H0 decision steps must be unique and non-empty")
    if any(step < 0 or step >= max_steps for step in steps):
        raise ValueError("Stage H0 decision step is outside the episode")
    configured_horizon = audit["horizon"]
    if configured_horizon != "remaining" and int(configured_horizon) <= 0:
        raise ValueError("Stage H0 horizon must be positive")
    if int(audit["discovery_replications"]) <= 0:
        raise ValueError("Stage H0 discovery replications must be positive")
    if int(audit["validation_replications"]) <= 0:
        raise ValueError("Stage H0 validation replications must be positive")
    for key in (
        "material_improvement",
        "minimum_pairwise_cost_gap",
        "execution_tolerance",
    ):
        value = float(audit[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Stage H0 {key} must be positive and finite")

    specs = make_explicit_residual_option_specs(audit["explicit_options"])
    observed: dict[str, set[tuple[float, float]]] = defaultdict(set)
    for spec in specs:
        if spec.is_anchor:
            continue
        observed[str(spec.group)].add((float(spec.epsilon), float(spec.sign)))
    expected_groups = {
        "specimen_transfer",
        "reagent_transfer",
        "combined_transfer",
    }
    expected_options = {
        (0.05, -1.0),
        (0.05, 1.0),
        (0.10, -1.0),
        (0.10, 1.0),
    }
    if set(observed) != expected_groups or any(
        options != expected_options for options in observed.values()
    ):
        raise ValueError("Stage H0 requires four locked options for each group")

    for key, raw in config["stage_h0_gate"].items():
        value = float(raw)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"Stage H0 gate {key} must be in [0, 1]")

    ranges = tuple(config["forbidden_crn_ranges"])
    if not any(
        int(entry["start"]) <= FORMAL_HOLDOUT_SEED <= int(entry["end"])
        for entry in ranges
    ):
        raise ValueError("Stage H0 must explicitly forbid the formal holdout")
    used = diagnostic_crn_seed_sequence(config)
    if len(used) != len(set(used)):
        raise ValueError("Stage H0 diagnostic CRN streams overlap")
    overlap = [
        seed
        for seed in used
        if any(
            int(entry["start"]) <= seed <= int(entry["end"])
            for entry in ranges
        )
    ]
    if overlap:
        raise ValueError(f"Stage H0 uses forbidden CRN seeds: {overlap[:10]}")


def diagnostic_crn_seed_sequence(config: dict[str, Any]) -> list[int]:
    audit = dict(config["audit"])
    seeds = tuple(int(value) for value in config["training_seeds"])
    steps = tuple(int(value) for value in audit["decision_steps"])
    result = [
        int(audit["live_seed"]) + index * 100_000
        for index, _seed in enumerate(seeds)
    ]
    for scenario_index, _seed in enumerate(seeds):
        for step in steps:
            result.append(
                int(audit["execution_seed"])
                + scenario_index * 1_000_000
                + step * 100
            )
            for base_key, replication_key in (
                ("discovery_rollout_seed", "discovery_replications"),
                ("validation_rollout_seed", "validation_replications"),
            ):
                start = (
                    int(audit[base_key])
                    + scenario_index * 1_000_000
                    + step * 100
                )
                result.extend(
                    start + replication
                    for replication in range(int(audit[replication_key]))
                )
    return result


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

    locked_before = verify_locked_inputs(config)
    manifest = _load_json(Path(str(config["training_manifest"])))
    seeds = tuple(int(value) for value in config["training_seeds"])
    algorithm = str(config["primary_algorithm"])
    runs = manifest_runs(manifest, algorithm=algorithm, seeds=seeds)
    rows: list[dict[str, Any]] = []
    for scenario_index, seed in enumerate(seeds):
        rows.extend(
            collect_seed_rows(
                config,
                run=runs[seed],
                training_seed=seed,
                scenario_index=scenario_index,
            )
        )
    validate_rows(config, rows)

    locked_after = verify_locked_inputs(config)
    if locked_before != locked_after:
        raise RuntimeError("Stage H0 changed an immutable input")

    metrics = {
        group: summarize_group(
            rows,
            group=group,
            material_improvement=float(config["audit"]["material_improvement"]),
            minimum_pairwise_cost_gap=float(
                config["audit"]["minimum_pairwise_cost_gap"]
            ),
        )
        for group in (
            "specimen_transfer",
            "reagent_transfer",
            "combined_transfer",
        )
    }
    decision = stage_h0_decision(
        metrics,
        gates=dict(config["stage_h0_gate"]),
        smoke=smoke,
    )
    rows_path = output_root / "continuous_control_rows.csv"
    write_rows(rows, rows_path)
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
        "effective_config_sha256": hashlib.sha256(
            json.dumps(
                config,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "formal_holdout_reused": False,
        "training_performed": False,
        "actor_updated": False,
        "checkpoint_selected": False,
        "scenario_reconstruction": "benchmark_plan_explicit_assignment",
        "locked_inputs": locked_before,
        "fresh_crn_seed_count": len(diagnostic_crn_seed_sequence(config)),
        "rows": {
            "count": len(rows),
            "path": str(rows_path),
            "sha256": sha256_file(rows_path),
        },
        "metrics": metrics,
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


def collect_seed_rows(
    config: dict[str, Any],
    *,
    run: dict[str, Any],
    training_seed: int,
    scenario_index: int,
) -> list[dict[str, Any]]:
    audit = dict(config["audit"])
    algorithm = str(config["primary_algorithm"])
    runtime = load_config(Path(str(run["config"])))
    scenario = str(config["scenario_by_training_seed"][str(training_seed)])
    runtime = runtime_for_scenario(
        config,
        runtime=runtime,
        algorithm=algorithm,
        scenario=scenario,
    )
    runtime["device"] = str(audit["device"])
    runtime["replay_buffer_size"] = 1
    env = build_env(
        runtime,
        seed=int(audit["live_seed"]) + scenario_index * 100_000,
    )
    if str(getattr(env, "scenario_name", "")) != scenario:
        raise RuntimeError(
            f"Stage H0 built {env.scenario_name!r}, expected {scenario!r}"
        )
    frozen, _checkpoint = load_pretrain_agent(
        algorithm,
        run=run,
        runtime=runtime,
        state_dim=env.observation_size,
        action_dim=env.action_size,
    )
    anchor_name = str(runtime["residual_action"].get("base_policy", "mdl2"))
    anchor = get_heuristic_class(anchor_name)(
        env.observation_size,
        env.action_size,
        dict(runtime["residual_action"].get("base_policy_config", {})),
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
    specs = make_explicit_residual_option_specs(audit["explicit_options"])
    guardrails = dict(audit["guardrails"])
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        step = int(snapshot["step"])
        actions = np.asarray(snapshot["actions"], dtype=np.float32)
        execution = probe_one_step_execution(
            snapshot["env"],
            actions=actions,
            specs=specs,
            seed=(
                int(audit["execution_seed"])
                + scenario_index * 1_000_000
                + step * 100
            ),
            tolerance=float(audit["execution_tolerance"]),
        )
        horizon = (
            int(snapshot["remaining_horizon"])
            if audit["horizon"] == "remaining"
            else min(int(audit["horizon"]), int(snapshot["remaining_horizon"]))
        )
        stream_metrics: dict[str, list[dict[str, float]]] = {}
        stream_advantages: dict[str, np.ndarray] = {}
        for stream, base_key, replication_key in (
            ("discovery", "discovery_rollout_seed", "discovery_replications"),
            ("validation", "validation_rollout_seed", "validation_replications"),
        ):
            seed_start = (
                int(audit[base_key])
                + scenario_index * 1_000_000
                + step * 100
            )
            rollout_seeds = tuple(
                seed_start + index
                for index in range(int(audit[replication_key]))
            )
            values = []
            for action in actions:
                anchor.reset()
                values.append(
                    mean_rollout_metrics_after_action(
                        snapshot["env"],
                        anchor,
                        action,
                        horizon=horizon,
                        rollout_seeds=rollout_seeds,
                    )
                )
            costs = np.asarray(
                [float(value["total_cost"]) for value in values],
                dtype=np.float64,
            )
            stream_metrics[stream] = values
            stream_advantages[stream] = costs[0] - costs

        for index, (label, spec) in enumerate(zip(snapshot["labels"], specs)):
            row: dict[str, Any] = {
                "training_seed": int(training_seed),
                "scenario": str(
                    config["scenario_by_training_seed"][str(training_seed)]
                ),
                "built_scenario": str(getattr(env, "scenario_name", "")),
                "step": step,
                "configured_horizon": str(audit["horizon"]),
                "horizon": int(horizon),
                "candidate_index": int(index),
                "candidate_label": str(label),
                "candidate_group": str(spec.group),
                "candidate_epsilon": float(spec.epsilon),
                "candidate_sign": float(spec.sign),
                **execution[index],
            }
            for stream in ("discovery", "validation"):
                values = stream_metrics[stream][index]
                anchor_values = stream_metrics[stream][0]
                row[f"{stream}_total_cost"] = float(values["total_cost"])
                row[f"{stream}_cost_advantage"] = float(
                    stream_advantages[stream][index]
                )
                for key in (
                    "completion_service_level",
                    "patients_lost",
                    "patient_ineligibility_during_manufacturing_rate",
                ):
                    row[f"{stream}_{key}"] = float(values[key])
                row[f"{stream}_clinical_noninferior"] = clinical_noninferior(
                    values,
                    anchor_values,
                    guardrails=guardrails,
                )
            rows.append(row)
    return rows


def probe_one_step_execution(
    env: Any,
    *,
    actions: np.ndarray,
    specs: tuple[Any, ...],
    seed: int,
    tolerance: float,
) -> list[dict[str, Any]]:
    action_array = np.asarray(actions, dtype=np.float32)
    physical = []
    for action in action_array:
        clone = copy.deepcopy(env)
        clone.rng = np.random.default_rng(int(seed))
        _state, _reward, _done, info = clone.step(action)
        physical.append(
            {
                "specimen_transfer": np.asarray(
                    info["specimen_transfers"], dtype=np.float64
                ),
                "reagent_transfer": np.asarray(
                    info["reagent_transfers"], dtype=np.float64
                ),
                "capacity_transfer": np.asarray(
                    info["capacity_transfers"], dtype=np.float64
                ),
            }
        )
    n = int(action_array.shape[1] // 4)
    result = []
    for index, spec in enumerate(specs):
        group = str(spec.group)
        if bool(spec.is_anchor):
            result.append(
                {
                    "input_action_delta_l2": 0.0,
                    "execution_delta_l2": 0.0,
                    "execution_survived": True,
                    "anchor_execution_id": "",
                    "execution_id": _execution_id(
                        np.concatenate(
                            (
                                physical[index]["specimen_transfer"],
                                physical[index]["reagent_transfer"],
                                physical[index]["capacity_transfer"],
                            )
                        )
                    ),
                }
            )
            continue
        action_slice = _action_group_slice(group, n)
        candidate_vector = _physical_group_vector(physical[index], group)
        anchor_vector = _physical_group_vector(physical[0], group)
        execution_delta = float(np.linalg.norm(candidate_vector - anchor_vector))
        result.append(
            {
                "input_action_delta_l2": float(
                    np.linalg.norm(
                        action_array[index, action_slice]
                        - action_array[0, action_slice]
                    )
                ),
                "execution_delta_l2": execution_delta,
                "execution_survived": bool(execution_delta > tolerance),
                "anchor_execution_id": _execution_id(anchor_vector),
                "execution_id": _execution_id(candidate_vector),
            }
        )
    return result


def runtime_for_scenario(
    config: dict[str, Any],
    *,
    runtime: dict[str, Any],
    algorithm: str,
    scenario: str,
) -> dict[str, Any]:
    """Rebuild the actual online scenario rather than the reference env.

    Multiscenario training snapshots intentionally keep the pretraining
    reference environment under ``env`` and record online assignments under
    ``multi_scenario_training``. Post-hoc audits must resolve the assignment
    through the benchmark plan explicitly.
    """

    rebuilt = copy.deepcopy(runtime)
    plan = load_benchmark_plan(Path(str(config["plan"])))
    selected = select_scenarios(plan, (str(scenario),))
    if len(selected) != 1:
        raise ValueError(f"Stage H0 could not resolve scenario {scenario!r}")
    env_config = make_scenario_env_config(plan, algorithm, selected[0])
    multiscenario = dict(rebuilt.get("multi_scenario_training", {}))
    declared = tuple(str(value) for value in multiscenario.get("scenarios", ()))
    expected_declared = str(config["scenario_by_training_seed"].get(
        str(rebuilt.get("seed", "")),
        scenario,
    ))
    allow_out_of_distribution = bool(
        config["audit"].get("allow_out_of_distribution_scenario_probe", False)
    )
    if (
        declared
        and declared != (expected_declared,)
        and not allow_out_of_distribution
    ):
        raise ValueError(
            "Stage H0 run scenario metadata does not match the locked assignment"
        )
    env_config = _deep_update(
        env_config,
        dict(multiscenario.get("env_overrides", {})),
    )
    rebuilt["env"] = env_config
    return rebuilt


def clinical_noninferior(
    values: dict[str, float],
    anchor: dict[str, float],
    *,
    guardrails: dict[str, Any],
) -> bool:
    tolerance = 1e-12
    completion_delta = float(values["completion_service_level"]) - float(
        anchor["completion_service_level"]
    )
    lost_delta = float(values["patients_lost"]) - float(anchor["patients_lost"])
    ineligibility_delta = float(
        values["patient_ineligibility_during_manufacturing_rate"]
    ) - float(anchor["patient_ineligibility_during_manufacturing_rate"])
    return bool(
        completion_delta + tolerance
        >= float(guardrails["min_completion_service_level_delta"])
        and lost_delta
        <= float(guardrails["max_patients_lost_delta"]) + tolerance
        and ineligibility_delta
        <= float(
            guardrails[
                "max_patient_ineligibility_during_manufacturing_rate_delta"
            ]
        )
        + tolerance
    )


def summarize_group(
    rows: list[dict[str, Any]],
    *,
    group: str,
    material_improvement: float,
    minimum_pairwise_cost_gap: float,
    include_per_seed: bool = True,
) -> dict[str, Any]:
    anchors = {
        (int(row["training_seed"]), int(row["step"])): row
        for row in rows
        if str(row["candidate_group"]) == "anchor"
    }
    candidates: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["candidate_group"]) == group:
            candidates[(int(row["training_seed"]), int(row["step"]))].append(row)
    if set(candidates) != set(anchors):
        raise ValueError(f"Stage H0 {group} state coverage is incomplete")

    best_agreement = []
    pairwise_agreement = []
    execution_survival = []
    any_distinct = []
    fully_distinct = []
    validation_headroom = []
    discovery_selected_validation = []
    stable_material = []
    for key in sorted(anchors):
        group_rows = [anchors[key]] + sorted(
            candidates[key],
            key=lambda row: int(row["candidate_index"]),
        )
        discovery = np.asarray(
            [float(row["discovery_cost_advantage"]) for row in group_rows]
        )
        validation = np.asarray(
            [float(row["validation_cost_advantage"]) for row in group_rows]
        )
        discovery_best = set(np.flatnonzero(discovery == discovery.max()))
        validation_best = set(np.flatnonzero(validation == validation.max()))
        best_agreement.append(float(bool(discovery_best & validation_best)))
        for left, right in combinations(range(len(group_rows)), 2):
            discovery_delta = float(discovery[left] - discovery[right])
            validation_delta = float(validation[left] - validation[right])
            if min(abs(discovery_delta), abs(validation_delta)) < float(
                minimum_pairwise_cost_gap
            ):
                continue
            pairwise_agreement.append(
                float(discovery_delta * validation_delta > 0.0)
            )

        nonanchor = group_rows[1:]
        survived = [bool(row["execution_survived"]) for row in nonanchor]
        execution_survival.extend(float(value) for value in survived)
        any_distinct.append(float(any(survived)))
        signatures = {str(row["execution_id"]) for row in nonanchor}
        signatures.add(str(nonanchor[0]["anchor_execution_id"]))
        fully_distinct.append(float(len(signatures) == len(group_rows)))

        validation_valid = [
            index
            for index, row in enumerate(group_rows[1:], start=1)
            if bool(row["validation_clinical_noninferior"])
            and float(row["validation_cost_advantage"]) >= material_improvement
        ]
        validation_headroom.append(float(bool(validation_valid)))

        discovery_valid = [
            index
            for index, row in enumerate(group_rows[1:], start=1)
            if bool(row["discovery_clinical_noninferior"])
            and float(row["discovery_cost_advantage"]) > 0.0
        ]
        selected = (
            0
            if not discovery_valid
            else max(discovery_valid, key=lambda index: float(discovery[index]))
        )
        discovery_selected_validation.append(
            float(
                selected > 0
                and bool(group_rows[selected]["validation_clinical_noninferior"])
                and float(validation[selected]) >= material_improvement
            )
        )
        common_best = (discovery_best & validation_best) - {0}
        stable_material.append(
            float(
                bool(common_best)
                and float(discovery.max()) >= material_improvement
                and float(validation.max()) >= material_improvement
                and any(
                    bool(group_rows[index]["discovery_clinical_noninferior"])
                    and bool(group_rows[index]["validation_clinical_noninferior"])
                    for index in common_best
                )
            )
        )

    per_seed = {}
    if include_per_seed:
        for seed in sorted({int(row["training_seed"]) for row in rows}):
            per_seed[str(seed)] = summarize_group(
                [row for row in rows if int(row["training_seed"]) == seed],
                group=group,
                material_improvement=material_improvement,
                minimum_pairwise_cost_gap=minimum_pairwise_cost_gap,
                include_per_seed=False,
            )
    return {
        "states": len(anchors),
        "nonanchor_options": len(execution_survival),
        "nonanchor_execution_survival_fraction": float(
            np.mean(execution_survival)
        ),
        "states_with_any_distinct_execution_fraction": float(
            np.mean(any_distinct)
        ),
        "fully_distinct_execution_state_fraction": float(
            np.mean(fully_distinct)
        ),
        "discovery_validation_best_action_agreement": float(
            np.mean(best_agreement)
        ),
        "discovery_validation_pairwise_sign_agreement": (
            None
            if not pairwise_agreement
            else float(np.mean(pairwise_agreement))
        ),
        "pairwise_comparisons": len(pairwise_agreement),
        "validation_material_clinically_noninferior_opportunity_fraction": float(
            np.mean(validation_headroom)
        ),
        "discovery_selected_validation_material_success_fraction": float(
            np.mean(discovery_selected_validation)
        ),
        "stable_material_best_action_fraction": float(
            np.mean(stable_material)
        ),
        "per_seed": per_seed,
    }


def stage_h0_decision(
    metrics: dict[str, dict[str, Any]],
    *,
    gates: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    checks: dict[str, dict[str, bool]] = {}
    eligible = []
    for group in CONTINUOUS_GROUPS:
        values = metrics[group]
        group_checks = {
            "execution_survival": float(
                values["nonanchor_execution_survival_fraction"]
            )
            >= float(gates["minimum_execution_survival_fraction"]),
            "best_action_stability": float(
                values["discovery_validation_best_action_agreement"]
            )
            >= float(gates["minimum_best_action_agreement"]),
            "pairwise_stability": (
                values["discovery_validation_pairwise_sign_agreement"] is not None
                and float(
                    values["discovery_validation_pairwise_sign_agreement"]
                )
                >= float(gates["minimum_pairwise_sign_agreement"])
            ),
            "validated_headroom": float(
                values[
                    "validation_material_clinically_noninferior_opportunity_fraction"
                ]
            )
            >= float(gates["minimum_validated_material_opportunity_fraction"]),
            "prospective_success": float(
                values[
                    "discovery_selected_validation_material_success_fraction"
                ]
            )
            >= float(gates["minimum_prospective_validation_success_fraction"]),
        }
        checks[group] = group_checks
        if all(group_checks.values()):
            eligible.append(group)
    selected = None
    if eligible:
        selected = max(
            eligible,
            key=lambda group: (
                float(
                    metrics[group][
                        "discovery_selected_validation_material_success_fraction"
                    ]
                ),
                float(
                    metrics[group][
                        "validation_material_clinically_noninferior_opportunity_fraction"
                    ]
                ),
                float(
                    metrics[group][
                        "discovery_validation_pairwise_sign_agreement"
                    ]
                    or 0.0
                ),
            ),
        )
    if smoke:
        classification = "smoke_only_no_scientific_decision"
    elif selected is None:
        classification = "continuous_control_geometry_not_supported"
    else:
        classification = "continuous_control_full_state_audit_warranted"
    return {
        "classification": classification,
        "group_checks": checks,
        "eligible_continuous_groups": eligible,
        "selected_group": selected,
        "full_156_state_frozen_audit_authorized": bool(selected and not smoke),
        "online_training_authorized": False,
        "formal_confirmation_authorized": False,
    }


def validate_rows(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    expected_states = len(config["training_seeds"]) * len(
        config["audit"]["decision_steps"]
    )
    expected_candidates = len(config["audit"]["explicit_options"]) + 1
    expected_rows = expected_states * expected_candidates
    if len(rows) != expected_rows:
        raise ValueError("Stage H0 row count mismatch")
    keys = {
        (
            int(row["training_seed"]),
            int(row["step"]),
            int(row["candidate_index"]),
        )
        for row in rows
    }
    if len(keys) != expected_rows:
        raise ValueError("Stage H0 rows contain duplicates")
    for row in rows:
        expected_scenario = str(
            config["scenario_by_training_seed"][str(row["training_seed"])]
        )
        if (
            str(row["scenario"]) != expected_scenario
            or str(row["built_scenario"]) != expected_scenario
        ):
            raise ValueError("Stage H0 row scenario reconstruction mismatch")
        for key in (
            "input_action_delta_l2",
            "execution_delta_l2",
            "discovery_total_cost",
            "discovery_cost_advantage",
            "validation_total_cost",
            "validation_cost_advantage",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Stage H0 has non-finite {key}")


def _action_group_slice(group: str, n: int) -> slice:
    if group == "specimen_transfer":
        return slice(0, n)
    if group == "reagent_transfer":
        return slice(n, 2 * n)
    if group == "combined_transfer":
        return slice(n, 3 * n)
    raise ValueError(f"Unsupported Stage H0 action group: {group}")


def _physical_group_vector(values: dict[str, np.ndarray], group: str) -> np.ndarray:
    if group == "specimen_transfer":
        return values["specimen_transfer"]
    if group == "reagent_transfer":
        return values["reagent_transfer"]
    if group == "combined_transfer":
        return np.concatenate(
            (values["reagent_transfer"], values["capacity_transfer"])
        )
    raise ValueError(f"Unsupported Stage H0 physical group: {group}")


def _execution_id(values: np.ndarray) -> str:
    rounded = np.round(np.asarray(values, dtype=np.float64), decimals=7)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_audit_config(path: Path) -> dict[str, Any]:
    raw = load_config(path)
    base_value = raw.pop("base_config", None)
    append_files = list(raw.pop("locked_files_append", ()))
    append_trees = list(raw.pop("locked_trees_append", ()))
    if base_value is None:
        config = raw
    else:
        base_path = Path(str(base_value))
        if not base_path.is_file():
            base_path = path.parent / base_path
        config = _deep_update(load_audit_config(base_path), raw)
    if append_files:
        config["locked_files"] = list(config.get("locked_files", ())) + append_files
    if append_trees:
        config["locked_trees"] = list(config.get("locked_trees", ())) + append_trees
    return config


if __name__ == "__main__":
    main()
