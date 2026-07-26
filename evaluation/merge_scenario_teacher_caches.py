"""Build leakage-safe, trajectory-split teacher caches across scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.restrict_option_teacher_cache import (
    restrict_teacher_demonstrations,
)
from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    make_scenario_env_config,
    select_scenarios,
)
from evaluation.run_gcn_residual_sweep import (
    load_local_search_demonstrations,
    save_local_search_demonstrations,
)
from src.baselines.heuristics import heuristic_settings_for_policy
from src.rl.config import load_config
from src.rl.experiment import build_env


PER_ROW_KEYS = (
    "states",
    "actions",
    "weights",
    "improved_mask",
    "transition_states",
    "transition_actions",
    "transition_rewards",
    "transition_next_states",
    "transition_dones",
    "option_advantages",
    "option_feasible",
)
OPTION_METADATA_KEYS = (
    "option_groups",
    "option_epsilons",
    "option_signs",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="JSON config listing scenario names and teacher caches.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = build_multiscenario_teacher_caches(config)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_multiscenario_teacher_caches(
    config: dict[str, Any],
) -> dict[str, Any]:
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
    base_policy = str(config.get("base_policy", "mdl2"))
    allowed_groups = tuple(
        str(value)
        for value in config.get(
            "allowed_groups",
            (
                "reagent_transfer",
                "combined_transfer",
                "combined_network",
            ),
        )
    )
    min_advantage = float(config.get("min_advantage", 500_000.0))
    validation_fraction = float(config.get("validation_fraction", 0.2))
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0, 1)")
    split_seed = int(config.get("split_seed", 740_000))
    restrict_options = bool(config.get("restrict_options", True))
    balance_scenarios = bool(config.get("balance_scenarios", True))
    sources = tuple(config.get("sources", ()))
    if len(sources) < 2:
        raise ValueError("At least two scenario teacher sources are required")

    scenario_names: list[str] = []
    train_parts: list[dict[str, Any]] = []
    validation_parts: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    trajectory_offset = 0

    for scenario_id, source_value in enumerate(sources):
        source = dict(source_value)
        scenario_name = str(source["scenario"])
        scenario = select_scenarios(plan, (scenario_name,))[0]
        env_config = make_scenario_env_config(
            plan,
            algorithm,
            scenario,
        )
        env = build_env({"env": env_config}, seed=split_seed + scenario_id)
        cache_path = Path(source["cache"])
        demonstrations = load_local_search_demonstrations(cache_path)
        if bool(env_config.get("include_demand_history_state", False)):
            environment_window = int(
                env_config.get("demand_history_window", 4)
            )
            cache_window = demonstrations.get("demand_history_window")
            if (
                cache_window is not None
                and int(cache_window) != environment_window
            ):
                raise ValueError(
                    f"{scenario_name} teacher cache history window "
                    f"{cache_window} does not match environment window "
                    f"{environment_window}"
                )
            demonstrations["demand_history_window"] = environment_window
        if restrict_options:
            demonstrations, restriction_summary = (
                restrict_teacher_demonstrations(
                    demonstrations,
                    env_config=env_config,
                    anchor_settings=heuristic_settings_for_policy(
                        base_policy
                    ),
                    allowed_groups=allowed_groups,
                    min_advantage=min_advantage,
                )
            )
        else:
            restriction_summary = {}

        row_count = int(demonstrations["states"].shape[0])
        trajectory_length = int(
            source.get(
                "trajectory_length",
                env.config.episode_horizon,
            )
        )
        if trajectory_length <= 0 or row_count % trajectory_length:
            raise ValueError(
                f"{scenario_name} cache has {row_count} rows, which cannot "
                f"be split into trajectories of length {trajectory_length}"
            )
        if demonstrations["states"].shape[1] != env.observation_size:
            raise ValueError(
                f"{scenario_name} cache observation width "
                f"{demonstrations['states'].shape[1]} does not match "
                f"environment width {env.observation_size}"
            )
        if demonstrations["actions"].shape[1] != env.action_size:
            raise ValueError(
                f"{scenario_name} cache action width does not match "
                "the environment"
            )

        local_trajectory_count = row_count // trajectory_length
        local_ids = np.arange(local_trajectory_count, dtype=np.int64)
        rng = np.random.default_rng(split_seed + scenario_id * 1009)
        rng.shuffle(local_ids)
        validation_count = validation_trajectory_count(
            local_trajectory_count,
            validation_fraction,
        )
        validation_ids = np.sort(local_ids[:validation_count])
        train_ids = np.sort(local_ids[validation_count:])
        row_trajectory_ids = np.repeat(
            np.arange(local_trajectory_count, dtype=np.int64),
            trajectory_length,
        )
        trajectory_steps = np.tile(
            np.arange(trajectory_length, dtype=np.int64),
            local_trajectory_count,
        )
        global_trajectory_ids = row_trajectory_ids + trajectory_offset
        provenance = {
            "scenario_ids": np.full(
                row_count,
                scenario_id,
                dtype=np.int64,
            ),
            "trajectory_ids": global_trajectory_ids,
            "trajectory_steps": trajectory_steps,
        }
        train_mask = np.isin(row_trajectory_ids, train_ids)
        validation_mask = np.isin(row_trajectory_ids, validation_ids)
        train_parts.append(
            subset_demonstrations(
                demonstrations,
                train_mask,
                provenance=provenance,
            )
        )
        if validation_count:
            validation_parts.append(
                subset_demonstrations(
                    demonstrations,
                    validation_mask,
                    provenance=provenance,
                )
            )
        scenario_names.append(scenario_name)
        source_summaries.append(
            {
                "scenario": scenario_name,
                "cache": str(cache_path),
                "rows": row_count,
                "trajectories": local_trajectory_count,
                "train_trajectories": int(train_ids.size),
                "validation_trajectories": int(validation_ids.size),
                "train_trajectory_ids": train_ids.tolist(),
                "validation_trajectory_ids": validation_ids.tolist(),
                "restriction": restriction_summary,
            }
        )
        trajectory_offset += local_trajectory_count

    train_cache = merge_scenario_parts(
        train_parts,
        scenario_names=tuple(scenario_names),
        balance_scenarios=balance_scenarios,
    )
    validation_cache = (
        merge_scenario_parts(
            validation_parts,
            scenario_names=tuple(scenario_names),
            balance_scenarios=balance_scenarios,
        )
        if validation_parts
        else None
    )

    output_root = Path(
        config.get(
            "output_root",
            "results/multiscenario_teacher_cache",
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "teacher_train.npz"
    validation_path = output_root / "teacher_validation.npz"
    save_local_search_demonstrations(train_path, train_cache)
    if validation_cache is not None:
        save_local_search_demonstrations(
            validation_path,
            validation_cache,
        )

    result = {
        "name": str(config.get("name", "multiscenario_teacher_cache")),
        "algorithm": algorithm,
        "base_policy": base_policy,
        "scenario_names": scenario_names,
        "split_seed": split_seed,
        "validation_fraction": validation_fraction,
        "restrict_options": restrict_options,
        "allowed_groups": list(allowed_groups),
        "min_advantage": min_advantage,
        "balance_scenarios": balance_scenarios,
        "train_cache": str(train_path),
        "validation_cache": (
            str(validation_path)
            if validation_cache is not None
            else ""
        ),
        "train_rows": int(train_cache["states"].shape[0]),
        "validation_rows": (
            int(validation_cache["states"].shape[0])
            if validation_cache is not None
            else 0
        ),
        "sources": source_summaries,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def validation_trajectory_count(
    trajectory_count: int,
    fraction: float,
) -> int:
    if trajectory_count < 1:
        raise ValueError("Each scenario cache needs at least one trajectory")
    if fraction <= 0.0 or trajectory_count == 1:
        return 0
    count = max(1, int(round(trajectory_count * fraction)))
    return min(count, trajectory_count - 1)


def subset_demonstrations(
    demonstrations: dict[str, Any],
    mask: np.ndarray,
    *,
    provenance: dict[str, np.ndarray],
) -> dict[str, Any]:
    row_count = int(demonstrations["states"].shape[0])
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (row_count,):
        raise ValueError("Teacher subset mask must align with cache rows")
    result: dict[str, Any] = {}
    for key in PER_ROW_KEYS:
        if key in demonstrations:
            values = np.asarray(demonstrations[key])
            if values.shape[0] != row_count:
                raise ValueError(
                    f"Teacher array {key} does not align with state rows"
                )
            result[key] = values[selected]
    for key in OPTION_METADATA_KEYS:
        if key in demonstrations:
            result[key] = np.asarray(demonstrations[key]).copy()
    for key, values in provenance.items():
        result[key] = np.asarray(values)[selected]
    if "demand_history_window" in demonstrations:
        result["demand_history_window"] = int(
            demonstrations["demand_history_window"]
        )
    return recompute_cache_summary(result)


def merge_scenario_parts(
    parts: list[dict[str, Any]],
    *,
    scenario_names: tuple[str, ...],
    balance_scenarios: bool,
) -> dict[str, Any]:
    if not parts:
        raise ValueError("At least one scenario cache part is required")
    validate_matching_metadata(parts)
    common_row_keys = tuple(
        key
        for key in PER_ROW_KEYS
        if all(key in part for part in parts)
    )
    merged = {
        key: np.concatenate(
            [np.asarray(part[key]) for part in parts],
            axis=0,
        )
        for key in common_row_keys
    }
    for key in ("scenario_ids", "trajectory_ids", "trajectory_steps"):
        merged[key] = np.concatenate(
            [np.asarray(part[key], dtype=np.int64) for part in parts],
            axis=0,
        )
    for key in OPTION_METADATA_KEYS:
        if key in parts[0]:
            merged[key] = np.asarray(parts[0][key]).copy()
    history_windows = {
        int(part["demand_history_window"])
        for part in parts
        if "demand_history_window" in part
    }
    if len(history_windows) > 1:
        raise ValueError("Scenario caches use different demand-history windows")
    if history_windows:
        merged["demand_history_window"] = history_windows.pop()
    merged["scenario_cache_version"] = 1
    merged["scenario_names"] = np.asarray(scenario_names, dtype="U96")
    merged = recompute_cache_summary(merged)
    if balance_scenarios:
        merged["weights"] = scenario_balanced_label_weights(
            merged["scenario_ids"],
            merged["improved_mask"],
            merged["weights"],
        )
        merged = recompute_cache_summary(merged)
    return merged


def validate_matching_metadata(parts: list[dict[str, Any]]) -> None:
    reference = parts[0]
    for part in parts[1:]:
        if (
            part["states"].shape[1] != reference["states"].shape[1]
            or part["actions"].shape[1] != reference["actions"].shape[1]
        ):
            raise ValueError("Scenario teacher caches have incompatible shapes")
        for key in OPTION_METADATA_KEYS:
            if key not in reference and key not in part:
                continue
            if key not in reference or key not in part:
                raise ValueError(
                    f"Scenario teacher cache metadata differs for {key}"
                )
            if key == "option_groups":
                matches = np.array_equal(reference[key], part[key])
            else:
                matches = np.allclose(
                    reference[key],
                    part[key],
                    rtol=0.0,
                    atol=1e-6,
                )
            if not matches:
                raise ValueError(
                    f"Scenario teacher cache metadata differs for {key}"
                )


def recompute_cache_summary(cache: dict[str, Any]) -> dict[str, Any]:
    result = dict(cache)
    weights = np.asarray(result["weights"], dtype=np.float32)
    improved = np.asarray(
        result.get("improved_mask", weights > 1.0 + 1e-6),
        dtype=bool,
    )
    if weights.shape != improved.shape:
        raise ValueError("Teacher weights and labels must align")
    result["weights"] = weights
    result["improved_mask"] = improved
    result["improved_steps"] = int(improved.sum())
    result["anchor_keep_steps"] = int((~improved).sum())
    result["service_rejected_steps"] = 0
    if "option_advantages" in result and np.any(improved):
        advantages = np.asarray(
            result["option_advantages"],
            dtype=np.float32,
        )
        selected = np.max(advantages, axis=1)
        result["mean_step_improvement"] = float(
            selected[improved].mean()
        )
    else:
        result["mean_step_improvement"] = 0.0
    total_weight = float(weights.sum())
    result["improved_weight_fraction"] = (
        float(weights[improved].sum()) / total_weight
        if total_weight > 0.0
        else 0.0
    )
    return result


def scenario_balanced_label_weights(
    scenario_ids: np.ndarray,
    improved_mask: np.ndarray,
    source_weights: np.ndarray,
) -> np.ndarray:
    scenario_values = np.asarray(scenario_ids, dtype=np.int64)
    improved = np.asarray(improved_mask, dtype=bool)
    source = np.asarray(source_weights, dtype=np.float64)
    if (
        scenario_values.shape != improved.shape
        or source.shape != improved.shape
    ):
        raise ValueError("Scenario balancing arrays must align")
    if np.any(source <= 0.0) or not np.all(np.isfinite(source)):
        raise ValueError("Teacher weights must be finite and positive")

    balanced = np.zeros_like(source)
    unique_scenarios = np.unique(scenario_values)
    for scenario_id in unique_scenarios:
        scenario_mask = scenario_values == scenario_id
        labels = tuple(
            label for label in (False, True)
            if np.any(scenario_mask & (improved == label))
        )
        for label in labels:
            group_mask = scenario_mask & (improved == label)
            group_total = float(source[group_mask].sum())
            target_mass = 1.0 / (
                float(unique_scenarios.size) * float(len(labels))
            )
            balanced[group_mask] = (
                source[group_mask] * target_mass / group_total
            )
    balanced *= float(balanced.size) / float(balanced.sum())
    return balanced.astype(np.float32)


if __name__ == "__main__":
    main()
