"""Reproduce the exploratory stochastic-procurement variability diagnostic.

This audit deliberately does not implement the unsigned draft protocol's
optimality-gap gate. It evaluates one narrower estimand: the paired cost
difference for the same tuned lead-aware MDL-2 policy under stochastic lead
times and under a lead fixed at the same mean. The seed families were frozen
after the original reported result was known, so this is a reproducibility
check rather than an independent confirmation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluation.evaluate_formal import evaluate_agent
from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    make_scenario_env_config,
    select_scenarios,
)
from src.baselines.heuristics import get_heuristic_class
from src.rl.experiment import build_env


DEFAULT_CONFIG = Path(
    "experiments/configs/stochastic_procurement_variability_reproduction.json"
)
FORMAL_HOLDOUT_SEED = 91_100_000
PROTECTED_SEEDS = {
    FORMAL_HOLDOUT_SEED,
    94_000_000,
    94_100_000,
    95_200_000,
    95_300_000,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.smoke:
        config = smoke_config(config)
    if args.output_root:
        config["output_root"] = args.output_root
    validate_config(config)
    run_reproduction(config, config_path=config_path)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_values(block: dict[str, Any]) -> tuple[int, ...]:
    start = int(block["start"])
    count = int(block["count"])
    return tuple(range(start, start + count))


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    smoke["name"] = f"{config['name']}_smoke"
    smoke["scenarios"] = list(config["scenarios"][:1])
    smoke["safety_multiplier_grid"] = [1.0]
    smoke["development_seeds"]["count"] = 1
    smoke["evaluation_seeds"]["count"] = 2
    smoke["output_root"] = f"{config['output_root']}_smoke"
    smoke["experimental_role"] = f"SMOKE ONLY. {config['experimental_role']}"
    return smoke


def validate_config(config: dict[str, Any]) -> None:
    required = (
        "name",
        "experimental_role",
        "plan",
        "scenarios",
        "algorithm",
        "lead_time_env_overrides",
        "safety_multiplier_grid",
        "development_seeds",
        "evaluation_seeds",
        "reported_reference",
        "output_root",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"reproduction config missing fields: {missing}")
    if config["algorithm"] != "mdl2_lt":
        raise ValueError("the reproduction must use the lead-aware mdl2_lt policy")

    overrides = config["lead_time_env_overrides"]
    if not bool(overrides.get("enable_stochastic_procurement", False)):
        raise ValueError("stochastic procurement must be enabled")
    probabilities = np.asarray(
        overrides.get("reagent_lead_time_probabilities", ()), dtype=float
    )
    if probabilities.size < 2 or np.any(probabilities < 0.0):
        raise ValueError("lead-time probabilities must be nonnegative and nondegenerate")
    if not np.isclose(float(probabilities.sum()), 1.0):
        raise ValueError("lead-time probabilities must sum to one")
    expected_lead = float(np.dot(np.arange(probabilities.size), probabilities))
    if not expected_lead.is_integer():
        raise ValueError("fixed-mean pairing requires an integer expected lead")

    grid = [float(value) for value in config["safety_multiplier_grid"]]
    if not grid or sorted(set(grid)) != grid or grid[0] < 0.0:
        raise ValueError("safety_multiplier_grid must be unique, sorted, and nonnegative")

    development = seed_values(config["development_seeds"])
    evaluation = seed_values(config["evaluation_seeds"])
    if not development or not evaluation:
        raise ValueError("development and evaluation seed sets must be nonempty")
    if set(development) & set(evaluation):
        raise ValueError("development and evaluation seeds must be disjoint")
    collision = sorted((set(development) | set(evaluation)) & PROTECTED_SEEDS)
    if collision:
        raise ValueError(f"reproduction seeds collide with protected streams: {collision}")
    if len(set(config["scenarios"])) != len(config["scenarios"]):
        raise ValueError("scenarios must be unique")
    output_root = str(config["output_root"])
    smoke_output = str(config["name"]).endswith("_smoke") and output_root.startswith(
        "/private/tmp/"
    )
    if not output_root.startswith("results/") and not smoke_output:
        raise ValueError("output_root must be under results/ (or /private/tmp for smoke)")


def scenario_env_dict(
    plan: dict[str, Any], config: dict[str, Any], scenario: dict[str, Any]
) -> dict[str, Any]:
    env = make_scenario_env_config(plan, str(config["algorithm"]), scenario)
    env = dict(env)
    env.update(config["lead_time_env_overrides"])
    env["scenario_name"] = str(scenario["name"])
    return env


def fixed_mean_lead(config: dict[str, Any]) -> int:
    probabilities = np.asarray(
        config["lead_time_env_overrides"]["reagent_lead_time_probabilities"],
        dtype=float,
    )
    return int(np.dot(np.arange(probabilities.size), probabilities))


def run_episode(
    env_dict: dict[str, Any],
    *,
    seed: int,
    safety_multiplier: float,
    fixed_lead: int | None,
) -> tuple[dict[str, Any], str]:
    paired_env = copy.deepcopy(env_dict)
    paired_env["procurement_lead_override"] = fixed_lead
    env = build_env({"env": paired_env}, seed=seed)
    policy = get_heuristic_class("mdl2_lt")(
        config={"safety_multiplier": float(safety_multiplier)}
    )
    row = evaluate_agent(
        policy,
        env,
        algorithm="mdl2_lt",
        seed=seed,
        replications=1,
        max_steps=int(env.config.episode_horizon),
    )[0]
    rng_digest = hashlib.sha256(
        json.dumps(env.rng.bit_generator.state, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return row, rng_digest


def tune_multiplier(
    scenario_envs: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for multiplier in config["safety_multiplier_grid"]:
        costs: list[float] = []
        for scenario_name, env_dict in scenario_envs.items():
            for seed in seed_values(config["development_seeds"]):
                row, _ = run_episode(
                    env_dict,
                    seed=seed,
                    safety_multiplier=float(multiplier),
                    fixed_lead=None,
                )
                costs.append(float(row["total_cost"]))
        rows.append(
            {
                "safety_multiplier": float(multiplier),
                "mean_stochastic_cost": float(np.mean(costs)),
                "episodes": len(costs),
            }
        )
    selected = min(rows, key=lambda row: row["mean_stochastic_cost"])
    return float(selected["safety_multiplier"]), rows


def paired_rows(
    scenario_envs: dict[str, dict[str, Any]],
    config: dict[str, Any],
    selected_multiplier: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mean_lead = fixed_mean_lead(config)
    for scenario_name, env_dict in scenario_envs.items():
        for seed in seed_values(config["evaluation_seeds"]):
            stochastic, stochastic_rng = run_episode(
                env_dict,
                seed=seed,
                safety_multiplier=selected_multiplier,
                fixed_lead=None,
            )
            fixed, fixed_rng = run_episode(
                env_dict,
                seed=seed,
                safety_multiplier=selected_multiplier,
                fixed_lead=mean_lead,
            )
            fixed_cost = float(fixed["total_cost"])
            stochastic_cost = float(stochastic["total_cost"])
            rows.append(
                {
                    "scenario": scenario_name,
                    "seed": int(seed),
                    "safety_multiplier": float(selected_multiplier),
                    "fixed_mean_lead": int(mean_lead),
                    "stochastic_cost": stochastic_cost,
                    "fixed_mean_cost": fixed_cost,
                    "cost_difference": stochastic_cost - fixed_cost,
                    "gap_fraction": (stochastic_cost - fixed_cost) / fixed_cost,
                    "stochastic_completion_service_level": stochastic.get(
                        "completion_service_level"
                    ),
                    "fixed_completion_service_level": fixed.get(
                        "completion_service_level"
                    ),
                    "stochastic_patients_lost": stochastic.get("patients_lost"),
                    "fixed_patients_lost": fixed.get("patients_lost"),
                    "rng_end_state_match": stochastic_rng == fixed_rng,
                }
            )
    return rows


def metric_summary(values: Iterable[float]) -> dict[str, float | int]:
    data = np.asarray(list(values), dtype=float)
    mean = float(data.mean())
    sem = float(data.std(ddof=1) / np.sqrt(data.size)) if data.size > 1 else 0.0
    return {
        "count": int(data.size),
        "mean": mean,
        "sem": sem,
        "normal_95_half_width": 1.96 * sem,
    }


def pooled_seed_cluster_summary(
    rows: list[dict[str, Any]], *, expected_scenarios: int
) -> dict[str, float | int | str]:
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(float(row["gap_fraction"]))
    incomplete = {
        seed: len(values)
        for seed, values in by_seed.items()
        if len(values) != expected_scenarios
    }
    if incomplete:
        raise ValueError(f"pooled seed clusters are incomplete: {incomplete}")
    cluster_stats = metric_summary(np.mean(values) for values in by_seed.values())
    return {
        "count": len(rows),
        "mean": cluster_stats["mean"],
        "sem": cluster_stats["sem"],
        "normal_95_half_width": cluster_stats["normal_95_half_width"],
        "independent_seed_clusters": cluster_stats["count"],
        "interval_unit": "evaluation_seed_mean_across_scenarios",
    }


def summarize(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    selected_multiplier: float,
    tuning_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_scenario: dict[str, Any] = {}
    for scenario in config["scenarios"]:
        selected = [row for row in rows if row["scenario"] == scenario]
        stats = metric_summary(row["gap_fraction"] for row in selected)
        stats["seeds_stochastic_worse"] = sum(
            float(row["cost_difference"]) > 0.0 for row in selected
        )
        stats["rng_end_state_matches"] = sum(
            bool(row["rng_end_state_match"]) for row in selected
        )
        by_scenario[scenario] = stats
    pooled = pooled_seed_cluster_summary(
        rows, expected_scenarios=len(config["scenarios"])
    )
    pooled["seeds_stochastic_worse"] = sum(
        float(row["cost_difference"]) > 0.0 for row in rows
    )
    pooled["rng_end_state_matches"] = sum(
        bool(row["rng_end_state_match"]) for row in rows
    )
    return {
        "name": config["name"],
        "scientific_status": (
            "post_hoc_reproduction_of_exploratory_variability_diagnostic; "
            "not_independent_confirmation; not_optimality_gap_gate"
        ),
        "experimental_role": config["experimental_role"],
        "selected_safety_multiplier": selected_multiplier,
        "tuning": tuning_rows,
        "by_scenario": by_scenario,
        "pooled": pooled,
        "reported_reference": config["reported_reference"],
        "formal_gate_decision": None,
        "policy_training_performed": False,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run_reproduction(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    plan = load_benchmark_plan(config["plan"])
    scenarios = select_scenarios(plan, config["scenarios"])
    scenario_envs = {
        str(scenario["name"]): scenario_env_dict(plan, config, scenario)
        for scenario in scenarios
    }
    selected_multiplier, tuning_rows = tune_multiplier(scenario_envs, config)
    rows = paired_rows(scenario_envs, config, selected_multiplier)
    if not rows or not all(bool(row["rng_end_state_match"]) for row in rows):
        raise RuntimeError("paired runs did not preserve exact RNG end-state matching")

    output_root = Path(config["output_root"])
    rows_path = output_root / "variability_rows.csv"
    summary_path = output_root / "summary.json"
    inventory_path = output_root / "artifact_inventory.json"
    write_rows(rows, rows_path)
    summary = summarize(
        rows,
        config=config,
        selected_multiplier=selected_multiplier,
        tuning_rows=tuning_rows,
    )
    source_paths = (
        Path("evaluation/reproduce_stochastic_procurement_variability.py"),
        Path("src/env/capacity_planning.py"),
        Path("src/env/patient_capacity_planning.py"),
        Path("src/baselines/heuristics.py"),
    )
    summary["provenance"] = {
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "plan_path": str(config["plan"]),
        "plan_sha256": sha256(Path(config["plan"])),
        "rows_sha256": sha256(rows_path),
        "source_files": [
            {"path": str(path), "sha256": sha256(path)} for path in source_paths
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inventory = {
        "files": [
            {"path": str(config_path), "sha256": sha256(config_path)},
            {
                "path": str(config["plan"]),
                "sha256": sha256(Path(config["plan"])),
            },
            *[
                {"path": str(path), "sha256": sha256(path)}
                for path in source_paths
            ],
            {"path": str(rows_path), "sha256": sha256(rows_path)},
            {"path": str(summary_path), "sha256": sha256(summary_path)},
        ]
    }
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    main()
