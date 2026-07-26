"""Collect leakage-safe DAgger labels on states visited by a residual policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.evaluate_multiscenario_network_residual import (
    configure_deployment,
    normalized_deployment_candidate,
)
from evaluation.merge_scenario_teacher_caches import (
    merge_scenario_parts,
    recompute_cache_summary,
    subset_demonstrations,
)
from evaluation.network_residual_headroom import ClinicalLookaheadTeacher
from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    make_scenario_env_config,
    select_scenarios,
)
from evaluation.run_gcn_residual_sweep import (
    load_local_search_demonstrations,
    save_local_search_demonstrations,
)
from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    result = collect_multiscenario_dagger(load_config(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))


def collect_multiscenario_dagger(
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """Query a look-ahead teacher on causal states visited by learned policies.

    The source validation cache is never read or modified. For non-stationary
    scenarios, labels begin only after the configured regime-change epoch plus
    a causal detection delay, so the student is not supervised with actions
    that require knowing an unobserved future scenario.
    """

    plan = load_benchmark_plan(
        run_config.get(
            "plan",
            "experiments/configs/residual_policy_benchmark.json",
        )
    )
    training_manifest_path = Path(run_config["training_manifest"])
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8")
    )
    selection_summary_path = Path(run_config["selection_summary"])
    selection_summary = json.loads(
        selection_summary_path.read_text(encoding="utf-8")
    )
    algorithm = str(
        run_config.get(
            "algorithm",
            "gcn_residual_mdl2_network_ddpg_afd",
        )
    )
    training_seeds = tuple(
        int(seed)
        for seed in run_config.get("training_seeds", (0, 1, 2))
    )
    if not training_seeds:
        raise ValueError("At least one behavior-policy seed is required")
    checkpoint_variant = str(
        run_config.get("checkpoint_variant", "pretrain")
    )
    if checkpoint_variant not in ("pretrain", "final"):
        raise ValueError("checkpoint_variant must be pretrain or final")

    scenario_entries = tuple(run_config.get("scenarios", ()))
    if len(scenario_entries) < 2:
        raise ValueError("DAgger collection requires at least two scenarios")
    scenario_names = tuple(
        str(entry["name"] if isinstance(entry, dict) else entry)
        for entry in scenario_entries
    )
    selected_scenarios = select_scenarios(plan, scenario_names)
    scenarios_by_name = {
        str(scenario["name"]): scenario
        for scenario in selected_scenarios
    }
    scenarios = [scenarios_by_name[name] for name in scenario_names]

    base_cache_path = Path(run_config["base_teacher_cache"])
    base_cache = load_local_search_demonstrations(base_cache_path)
    raw_base_rows = int(base_cache["states"].shape[0])
    base_scenario_names = tuple(
        str(value) for value in base_cache.get("scenario_names", ())
    )
    if base_scenario_names != scenario_names:
        raise ValueError(
            "DAgger scenarios must exactly match the base-cache scenario order"
        )
    require_cache_provenance(base_cache)

    runs_by_key = {
        (str(run["algorithm"]), int(run["seed"])): run
        for run in training_manifest["runs"]
    }
    deployments = selected_deployments(
        selection_summary,
        algorithm=algorithm,
    )
    teacher_configs = {
        str(name): Path(path)
        for name, path in dict(run_config["teacher_configs"]).items()
    }
    missing_teacher_configs = set(scenario_names) - set(teacher_configs)
    if missing_teacher_configs:
        raise ValueError(
            "Missing teacher configs for: "
            + ", ".join(sorted(missing_teacher_configs))
        )

    output_root = Path(
        run_config.get(
            "output_root",
            "results/multiscenario_network_afr_dagger",
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rollouts_per_policy = int(run_config.get("rollouts_per_policy", 1))
    if rollouts_per_policy < 1:
        raise ValueError("rollouts_per_policy must be positive")
    max_steps = int(run_config.get("max_steps", 52))
    collection_seed = int(run_config.get("collection_seed", 8_500_000))
    detection_delay = int(run_config.get("detection_delay", 4))
    if detection_delay < 0:
        raise ValueError("detection_delay cannot be negative")
    teacher_behavior_probability = float(
        run_config.get("teacher_behavior_probability", 0.0)
    )
    if not 0.0 <= teacher_behavior_probability <= 1.0:
        raise ValueError(
            "teacher_behavior_probability must lie in [0, 1]"
        )
    teacher_overrides = dict(run_config.get("teacher_overrides", {}))
    balance_scenarios = bool(run_config.get("balance_scenarios", True))
    include_base_cache = bool(
        run_config.get("include_base_cache", True)
    )
    filter_base_cache_causally = bool(
        run_config.get("filter_base_cache_causally", False)
    )
    if filter_base_cache_causally and include_base_cache:
        scenario_env_configs = [
            make_scenario_env_config(
                plan,
                algorithm,
                scenario,
            )
            for scenario in scenarios
        ]
        base_cache = causally_filter_base_cache(
            base_cache,
            scenario_env_configs=scenario_env_configs,
            detection_delay=detection_delay,
        )

    next_trajectory_id = (
        int(np.max(base_cache["trajectory_ids"])) + 1
        if include_base_cache and base_cache["trajectory_ids"].size
        else 0
    )
    rng = np.random.default_rng(collection_seed + 991)
    collected_parts: list[dict[str, Any]] = []
    collection_rows: list[dict[str, Any]] = []

    for policy_index, training_seed in enumerate(training_seeds):
        run_key = (algorithm, training_seed)
        if run_key not in runs_by_key:
            raise ValueError(f"Training manifest is missing run {run_key}")
        if training_seed not in deployments:
            raise ValueError(
                f"Selection summary is missing {algorithm} seed {training_seed}"
            )
        run = runs_by_key[run_key]
        checkpoint_key = (
            "pretrain_checkpoint"
            if checkpoint_variant == "pretrain"
            else "checkpoint"
        )
        checkpoint = Path(run[checkpoint_key])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        config_snapshot = load_config(run["config"])

        for scenario_index, (scenario_name, scenario) in enumerate(
            zip(scenario_names, scenarios)
        ):
            config = dict(config_snapshot)
            env_config = make_scenario_env_config(
                plan,
                algorithm,
                scenario,
            )
            config["env"] = env_config
            env = build_env(
                config,
                seed=collection_seed
                + policy_index * 1_000_000
                + scenario_index * 100_000,
            )
            agent = get_agent_class(algorithm)(
                env.observation_size,
                env.action_size,
                config,
            )
            agent.load_actor(checkpoint)
            deployment = deployments[training_seed]
            configure_deployment(agent, deployment)

            teacher_config = deep_update(
                load_config(teacher_configs[scenario_name]),
                teacher_overrides,
            )
            teacher_config["seed"] = (
                collection_seed
                + policy_index * 1_000_000
                + scenario_index * 100_000
            )
            teacher_config["max_steps"] = max_steps
            teacher_config["lookahead_decision_offset"] = (
                policy_index * len(scenarios) * rollouts_per_policy * max_steps
                + scenario_index * rollouts_per_policy * max_steps
            )
            query_start = causal_query_start_step(
                env_config,
                detection_delay=detection_delay,
                override=query_start_override(
                    scenario_entries[scenario_index]
                ),
            )
            if query_start > 0 and not (
                bool(env_config.get("include_demand_history_state", False))
                and bool(env_config.get("include_time_state", False))
            ):
                raise ValueError(
                    f"{scenario_name} requires causal demand history and time "
                    "features before post-change teacher labels are allowed"
                )
            teacher = ClinicalLookaheadTeacher(teacher_config, env)
            provenance_scenario_ids: list[int] = []
            provenance_trajectory_ids: list[int] = []
            provenance_trajectory_steps: list[int] = []
            teacher_behavior_steps = 0
            total_steps = 0

            for rollout in range(rollouts_per_policy):
                trajectory_id = next_trajectory_id
                next_trajectory_id += 1
                rollout_seed = (
                    collection_seed
                    + policy_index * 1_000_000
                    + scenario_index * 100_000
                    + rollout
                )
                state = env.reset(seed=rollout_seed)
                agent.reset()
                teacher.reset()
                done = False
                step = 0
                while not done and step < max_steps:
                    student_action = agent.select_action(
                        state,
                        explore=False,
                        env=env,
                    )
                    teacher_action = None
                    if step >= query_start:
                        teacher_action = teacher.select_action(
                            state,
                            explore=False,
                            env=env,
                        )
                        provenance_scenario_ids.append(scenario_index)
                        provenance_trajectory_ids.append(trajectory_id)
                        provenance_trajectory_steps.append(step)
                    behavior_action = student_action
                    if (
                        teacher_action is not None
                        and rng.random() < teacher_behavior_probability
                    ):
                        behavior_action = teacher_action
                        teacher_behavior_steps += 1
                    state, _reward, done, _info = env.step(
                        behavior_action
                    )
                    step += 1
                    total_steps += 1

            demonstrations = teacher.demonstrations()
            sample_count = int(demonstrations["states"].shape[0])
            if sample_count != len(provenance_scenario_ids):
                raise RuntimeError(
                    "Teacher labels and DAgger provenance are misaligned"
                )
            demonstrations.update(
                {
                    "scenario_ids": np.asarray(
                        provenance_scenario_ids,
                        dtype=np.int64,
                    ),
                    "trajectory_ids": np.asarray(
                        provenance_trajectory_ids,
                        dtype=np.int64,
                    ),
                    "trajectory_steps": np.asarray(
                        provenance_trajectory_steps,
                        dtype=np.int64,
                    ),
                    "scenario_names": np.asarray(
                        scenario_names,
                        dtype="U96",
                    ),
                    "scenario_cache_version": 2,
                    "demand_history_window": int(
                        env_config.get("demand_history_window", 1)
                    ),
                }
            )
            demonstrations = recompute_cache_summary(demonstrations)
            collected_parts.append(demonstrations)
            collection_rows.append(
                {
                    "algorithm": algorithm,
                    "training_seed": training_seed,
                    "scenario": scenario_name,
                    "checkpoint": str(checkpoint),
                    "deployment": deployment,
                    "rollouts": rollouts_per_policy,
                    "total_steps": total_steps,
                    "query_start_step": query_start,
                    "teacher_labels": sample_count,
                    "teacher_corrections": int(
                        demonstrations["improved_steps"]
                    ),
                    "teacher_correction_rate": (
                        float(demonstrations["improved_steps"])
                        / max(sample_count, 1)
                    ),
                    "teacher_behavior_steps": teacher_behavior_steps,
                }
            )
            print(
                "multiscenario_dagger "
                f"seed={training_seed} scenario={scenario_name} "
                f"labels={sample_count} "
                f"correction_rate="
                f"{float(demonstrations['improved_steps']) / max(sample_count, 1):.3f}",
                flush=True,
            )

    dagger_cache = merge_scenario_parts(
        collected_parts,
        scenario_names=scenario_names,
        balance_scenarios=balance_scenarios,
    )
    aggregate_cache = (
        merge_scenario_parts(
            [base_cache, *collected_parts],
            scenario_names=scenario_names,
            balance_scenarios=balance_scenarios,
        )
        if include_base_cache
        else dagger_cache
    )
    dagger_cache_path = output_root / "teacher_dagger_only.npz"
    aggregate_cache_path = output_root / "teacher_train_dagger.npz"
    save_local_search_demonstrations(
        dagger_cache_path,
        dagger_cache,
    )
    save_local_search_demonstrations(
        aggregate_cache_path,
        aggregate_cache,
    )
    result = {
        "name": str(
            run_config.get(
                "name",
                "multiscenario_network_afr_dagger",
            )
        ),
        "training_manifest": str(training_manifest_path),
        "selection_summary": str(selection_summary_path),
        "base_teacher_cache": str(base_cache_path),
        "algorithm": algorithm,
        "training_seeds": list(training_seeds),
        "checkpoint_variant": checkpoint_variant,
        "scenario_names": list(scenario_names),
        "detection_delay": detection_delay,
        "teacher_behavior_probability": teacher_behavior_probability,
        "include_base_cache": include_base_cache,
        "filter_base_cache_causally": filter_base_cache_causally,
        "rollouts_per_policy": rollouts_per_policy,
        "collection_seed": collection_seed,
        "teacher_overrides": teacher_overrides,
        "dagger_cache": str(dagger_cache_path),
        "aggregate_train_cache": str(aggregate_cache_path),
        "raw_base_rows": raw_base_rows,
        "base_rows": int(base_cache["states"].shape[0]),
        "base_rows_in_aggregate": (
            int(base_cache["states"].shape[0])
            if include_base_cache
            else 0
        ),
        "dagger_rows": int(dagger_cache["states"].shape[0]),
        "aggregate_rows": int(aggregate_cache["states"].shape[0]),
        "collections": collection_rows,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def selected_deployments(
    summary: dict[str, Any],
    *,
    algorithm: str,
) -> dict[int, dict[str, Any]]:
    deployments: dict[int, dict[str, Any]] = {}
    for run in summary.get("runs", ()):
        if str(run["algorithm"]) != str(algorithm):
            continue
        selected = run["selected_deployment"]
        if str(selected.get("selection_source")) != "validation_only":
            raise ValueError(
                "DAgger behavior deployments must be selected on validation only"
            )
        deployments[int(run["training_seed"])] = (
            normalized_deployment_candidate(selected["candidate"])
        )
    return deployments


def causal_query_start_step(
    env_config: dict[str, Any],
    *,
    detection_delay: int,
    override: int | None = None,
) -> int:
    """Delay supervision until a deterministic regime change is observable."""

    horizon = int(env_config.get("episode_horizon", 52))
    if override is not None:
        start = int(override)
    else:
        initial = np.asarray(
            env_config.get("demand_regime_initial_multipliers", 1.0),
            dtype=float,
        )
        final = np.asarray(
            env_config.get("demand_regime_final_multipliers", 1.0),
            dtype=float,
        )
        has_regime_change = not np.allclose(
            initial,
            final,
            rtol=0.0,
            atol=1e-12,
        )
        start = (
            int(env_config.get("demand_regime_change_step", 0))
            + int(detection_delay)
            if has_regime_change
            else 0
        )
    if not 0 <= start < horizon:
        raise ValueError(
            f"Teacher query start {start} must lie within horizon {horizon}"
        )
    return start


def query_start_override(entry: Any) -> int | None:
    if not isinstance(entry, dict) or "query_start_step" not in entry:
        return None
    return int(entry["query_start_step"])


def require_cache_provenance(cache: dict[str, Any]) -> None:
    row_count = int(cache["states"].shape[0])
    for key in ("scenario_ids", "trajectory_ids", "trajectory_steps"):
        values = np.asarray(cache.get(key, ()), dtype=np.int64)
        if values.shape != (row_count,):
            raise ValueError(
                f"Base teacher cache requires row-aligned {key}"
            )
    if "scenario_names" not in cache:
        raise ValueError("Base teacher cache requires scenario_names")


def causally_filter_base_cache(
    cache: dict[str, Any],
    *,
    scenario_env_configs: list[dict[str, Any]],
    detection_delay: int,
) -> dict[str, Any]:
    """Remove labels produced before a non-stationary regime is observable."""

    require_cache_provenance(cache)
    scenario_names = tuple(
        str(value) for value in cache["scenario_names"]
    )
    if len(scenario_env_configs) != len(scenario_names):
        raise ValueError(
            "Scenario environment configs must align with cache names"
        )
    scenario_ids = np.asarray(cache["scenario_ids"], dtype=np.int64)
    trajectory_steps = np.asarray(
        cache["trajectory_steps"],
        dtype=np.int64,
    )
    keep = np.ones(scenario_ids.shape, dtype=bool)
    for scenario_id, env_config in enumerate(scenario_env_configs):
        expected_name = str(env_config.get("scenario_name", "default"))
        if expected_name != scenario_names[scenario_id]:
            raise ValueError(
                "Scenario environment order does not match cache metadata: "
                f"{expected_name} != {scenario_names[scenario_id]}"
            )
        query_start = causal_query_start_step(
            env_config,
            detection_delay=detection_delay,
        )
        scenario_rows = scenario_ids == scenario_id
        keep[scenario_rows] = (
            trajectory_steps[scenario_rows] >= query_start
        )
    filtered = subset_demonstrations(
        cache,
        keep,
        provenance={
            "scenario_ids": scenario_ids,
            "trajectory_ids": np.asarray(
                cache["trajectory_ids"],
                dtype=np.int64,
            ),
            "trajectory_steps": trajectory_steps,
        },
    )
    filtered["scenario_names"] = np.asarray(
        scenario_names,
        dtype="U96",
    )
    filtered["scenario_cache_version"] = max(
        int(cache.get("scenario_cache_version", 1)),
        2,
    )
    return filtered


def deep_update(
    base: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    main()
