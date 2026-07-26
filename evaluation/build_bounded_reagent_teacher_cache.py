"""Convert dense teacher advantages into bounded reagent-transfer targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    make_scenario_env_config,
    select_scenarios,
)
from evaluation.run_gcn_residual_sweep import (
    load_local_search_demonstrations,
    save_local_search_demonstrations,
)
from src.baselines.heuristics import (
    facility_net_action_from_state,
    heuristic_settings_for_policy,
)
from src.rl.config import load_config
from src.rl.residual_options import residual_pressure_patterns_from_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    result = build_bounded_reagent_teacher_cache(load_config(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))


def build_bounded_reagent_teacher_cache(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Write conservative targets selected from causal dense CRN advantages."""

    source_path = Path(config["source_cache"])
    output_path = Path(config["output_cache"])
    source = load_local_search_demonstrations(source_path)
    scenario_names = tuple(
        str(value) for value in source.get("scenario_names", ())
    )
    if not scenario_names:
        raise ValueError("Bounded reagent targets require scenario provenance")

    plan = load_benchmark_plan(
        config.get(
            "plan",
            "experiments/configs/residual_policy_benchmark.json",
        )
    )
    algorithm = str(
        config.get(
            "algorithm",
            "gcn_residual_mdl2_network_ddpg_afd",
        )
    )
    selected_scenarios = select_scenarios(plan, scenario_names)
    scenarios_by_name = {
        str(scenario["name"]): scenario
        for scenario in selected_scenarios
    }
    env_configs = tuple(
        make_scenario_env_config(
            plan,
            algorithm,
            scenarios_by_name[name],
        )
        for name in scenario_names
    )
    base_policy = str(config.get("base_policy", "mdl2"))
    anchor_settings = heuristic_settings_for_policy(
        base_policy,
        dict(config.get("base_policy_config", {})),
    )
    bounded, summary = make_bounded_reagent_demonstrations(
        source,
        env_configs=env_configs,
        anchor_settings=anchor_settings,
        min_advantage=float(config.get("min_advantage", 500_000.0)),
        max_residual_scale=float(config.get("max_residual_scale", 0.1)),
        coefficient_advantage_scale=float(
            config.get("coefficient_advantage_scale", 2_000_000.0)
        ),
        min_positive_coefficient=float(
            config.get("min_positive_coefficient", 0.25)
        ),
        weight_advantage_scale=float(
            config.get("weight_advantage_scale", 1_000_000.0)
        ),
        weight_cap=float(config.get("weight_cap", 3.0)),
        coefficient_mode=str(
            config.get("coefficient_mode", "advantage")
        ),
    )
    save_local_search_demonstrations(output_path, bounded)

    result = {
        "name": str(
            config.get("name", "bounded_reagent_teacher_cache")
        ),
        "source_cache": str(source_path),
        "output_cache": str(output_path),
        "algorithm": algorithm,
        "base_policy": base_policy,
        "scenario_names": list(scenario_names),
        **summary,
    }
    manifest_path = Path(
        config.get(
            "manifest",
            output_path.with_suffix(".manifest.json"),
        )
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["manifest"] = str(manifest_path)
    return result


def make_bounded_reagent_demonstrations(
    source: dict[str, Any],
    *,
    env_configs: tuple[dict[str, Any], ...],
    anchor_settings,
    min_advantage: float,
    max_residual_scale: float,
    coefficient_advantage_scale: float,
    min_positive_coefficient: float,
    weight_advantage_scale: float,
    weight_cap: float,
    coefficient_mode: str = "advantage",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build state-action labels for one bounded graph-pattern coefficient."""

    if min_advantage < 0.0:
        raise ValueError("min_advantage must be non-negative")
    if not 0.0 < max_residual_scale <= 1.0:
        raise ValueError("max_residual_scale must lie in (0, 1]")
    if coefficient_advantage_scale <= 0.0:
        raise ValueError("coefficient_advantage_scale must be positive")
    if not 0.0 <= min_positive_coefficient <= 1.0:
        raise ValueError("min_positive_coefficient must lie in [0, 1]")
    if weight_advantage_scale <= 0.0:
        raise ValueError("weight_advantage_scale must be positive")
    if weight_cap < 1.0:
        raise ValueError("weight_cap must be at least 1")
    if coefficient_mode not in ("advantage", "selected_option"):
        raise ValueError(
            "coefficient_mode must be advantage or selected_option"
        )

    states = np.asarray(source["states"], dtype=np.float32)
    scenario_ids = np.asarray(source.get("scenario_ids", ()), dtype=np.int64)
    row_count = int(states.shape[0])
    if scenario_ids.shape != (row_count,):
        raise ValueError("Bounded reagent targets require row-aligned scenario_ids")
    if not env_configs or np.any(
        (scenario_ids < 0) | (scenario_ids >= len(env_configs))
    ):
        raise ValueError("Scenario ids do not align with environment configs")

    advantages = np.asarray(
        source.get("option_advantages", ()),
        dtype=np.float32,
    )
    feasible = np.asarray(source.get("option_feasible", ()), dtype=bool)
    groups = np.asarray(source.get("option_groups", ()), dtype="U32")
    epsilons = np.asarray(
        source.get("option_epsilons", ()),
        dtype=np.float32,
    )
    signs = np.asarray(source.get("option_signs", ()), dtype=np.float32)
    if (
        advantages.ndim != 2
        or advantages.shape != feasible.shape
        or advantages.shape[0] != row_count
        or groups.shape != (advantages.shape[1],)
        or epsilons.shape != groups.shape
        or signs.shape != groups.shape
    ):
        raise ValueError(
            "Bounded reagent targets require aligned dense option advantages"
        )
    candidate_indices = np.flatnonzero(
        (groups == "anchor") | (groups == "reagent_transfer")
    )
    anchor_positions = np.flatnonzero(groups[candidate_indices] == "anchor")
    if anchor_positions.size != 1 or candidate_indices.size < 3:
        raise ValueError(
            "Dense options must contain one anchor and signed reagent transfers"
        )

    candidate_advantages = advantages[:, candidate_indices]
    candidate_feasible = feasible[:, candidate_indices]
    masked_advantages = np.where(
        candidate_feasible,
        candidate_advantages,
        -np.inf,
    )
    best_positions = np.argmax(masked_advantages, axis=1)
    best_indices = candidate_indices[best_positions]
    best_advantages = masked_advantages[
        np.arange(row_count),
        best_positions,
    ]
    best_signs = signs[best_indices]
    improved = (
        (groups[best_indices] == "reagent_transfer")
        & (best_advantages >= float(min_advantage))
    )

    excess_advantage = np.maximum(
        best_advantages - float(min_advantage),
        0.0,
    )
    if coefficient_mode == "selected_option":
        positive_coefficients = np.clip(
            epsilons[best_indices] / float(max_residual_scale),
            0.0,
            1.0,
        )
    else:
        positive_coefficients = (
            float(min_positive_coefficient)
            + (1.0 - float(min_positive_coefficient))
            * np.clip(
                excess_advantage / float(coefficient_advantage_scale),
                0.0,
                1.0,
            )
        )
    coefficients = np.where(
        improved,
        best_signs * positive_coefficients,
        0.0,
    ).astype(np.float32)

    target_actions = np.zeros_like(source["actions"], dtype=np.float32)
    for row_index, state in enumerate(states):
        env_config = env_configs[int(scenario_ids[row_index])]
        n = int(env_config["num_facilities"])
        anchor_action = facility_net_action_from_state(
            state,
            env_config,
            settings=anchor_settings,
        )
        resource_pattern, _capacity_pattern = (
            residual_pressure_patterns_from_state(
                state,
                env_config,
            )
        )
        target = np.asarray(anchor_action, dtype=np.float32).copy()
        target[n : 2 * n] += (
            float(max_residual_scale)
            * float(coefficients[row_index])
            * resource_pattern
        )
        target_actions[row_index] = np.clip(
            target,
            -1.0,
            1.0,
        )

    weights = scenario_balanced_weights(scenario_ids)
    advantage_factors = np.ones(row_count, dtype=np.float32)
    advantage_factors[improved] = np.minimum(
        1.0
        + best_advantages[improved] / float(weight_advantage_scale),
        float(weight_cap),
    )
    weights *= advantage_factors
    weights = equalize_scenario_weight_mass(
        scenario_ids,
        weights,
    )

    excluded = {
        "actions",
        "weights",
        "improved_mask",
        "transition_states",
        "transition_actions",
        "transition_rewards",
        "transition_next_states",
        "transition_dones",
    }
    bounded = {
        key: np.asarray(value).copy()
        for key, value in source.items()
        if key not in excluded
        and isinstance(value, (np.ndarray, np.generic))
    }
    bounded.update(
        {
            "states": states.copy(),
            "actions": target_actions,
            "weights": weights.astype(np.float32),
            "improved_mask": improved.astype(bool),
            "improved_steps": int(improved.sum()),
            "anchor_keep_steps": int((~improved).sum()),
            "service_rejected_steps": 0,
            "mean_step_improvement": (
                float(best_advantages[improved].mean())
                if np.any(improved)
                else 0.0
            ),
            "improved_weight_fraction": (
                float(weights[improved].sum()) / float(weights.sum())
                if weights.size
                else 0.0
            ),
            "scenario_cache_version": max(
                int(source.get("scenario_cache_version", 1)),
                3,
            ),
        }
    )
    if "demand_history_window" in source:
        bounded["demand_history_window"] = int(
            source["demand_history_window"]
        )

    scenario_summary = {}
    scenario_names = tuple(
        str(value) for value in source.get("scenario_names", ())
    )
    for scenario_id, name in enumerate(scenario_names):
        mask = scenario_ids == scenario_id
        positive = improved & mask
        scenario_summary[name] = {
            "rows": int(mask.sum()),
            "corrections": int(positive.sum()),
            "correction_rate": float(positive.sum() / max(int(mask.sum()), 1)),
        }
    summary = {
        "rows": row_count,
        "min_advantage": float(min_advantage),
        "max_residual_scale": float(max_residual_scale),
        "coefficient_advantage_scale": float(coefficient_advantage_scale),
        "coefficient_mode": coefficient_mode,
        "min_positive_coefficient": float(min_positive_coefficient),
        "weight_advantage_scale": float(weight_advantage_scale),
        "weight_cap": float(weight_cap),
        "corrections": int(improved.sum()),
        "correction_rate": float(improved.mean()) if row_count else 0.0,
        "positive_direction_corrections": int(
            np.sum(improved & (best_signs > 0.0))
        ),
        "negative_direction_corrections": int(
            np.sum(improved & (best_signs < 0.0))
        ),
        "mean_positive_advantage": (
            float(best_advantages[improved].mean())
            if np.any(improved)
            else 0.0
        ),
        "coefficient_quantiles": (
            np.quantile(
                np.abs(coefficients[improved]),
                (0.1, 0.5, 0.9),
            ).tolist()
            if np.any(improved)
            else [0.0, 0.0, 0.0]
        ),
        "scenario_summary": scenario_summary,
    }
    return bounded, summary


def scenario_balanced_weights(scenario_ids: np.ndarray) -> np.ndarray:
    """Give each scenario equal total mass without balancing action labels."""

    values = np.asarray(scenario_ids, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scenario_ids must be a non-empty vector")
    unique = np.unique(values)
    weights = np.zeros(values.shape, dtype=np.float32)
    for scenario_id in unique:
        mask = values == scenario_id
        weights[mask] = float(values.size) / (
            float(unique.size) * float(mask.sum())
        )
    return weights


def equalize_scenario_weight_mass(
    scenario_ids: np.ndarray,
    source_weights: np.ndarray,
) -> np.ndarray:
    """Retain advantage weighting while restoring equal scenario mass."""

    values = np.asarray(scenario_ids, dtype=np.int64)
    weights = np.asarray(source_weights, dtype=np.float64).copy()
    if values.shape != weights.shape or np.any(weights <= 0.0):
        raise ValueError("Scenario ids and positive weights must align")
    unique = np.unique(values)
    target_mass = float(values.size) / float(unique.size)
    for scenario_id in unique:
        mask = values == scenario_id
        weights[mask] *= target_mass / float(weights[mask].sum())
    return weights.astype(np.float32)


if __name__ == "__main__":
    main()
