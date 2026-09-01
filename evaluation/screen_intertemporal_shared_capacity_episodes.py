"""Prospective episode-level pre-screen for intertemporal shared capacity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.heuristics import get_heuristic_class
from src.rl.experiment import build_env


DEFAULT_CONFIG = Path(
    "experiments/configs/intertemporal_shared_capacity_episode_prescreen.json"
)
FORBIDDEN_SEEDS = {
    91_100_000,
    94_000_000,
    94_100_000,
    95_200_000,
    95_300_000,
    96_700_000,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rng_digest(env) -> str:
    payload = json.dumps(
        env.rng.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_values(block: dict[str, Any]) -> tuple[int, ...]:
    start = int(block["start"])
    count = int(block["count"])
    return tuple(range(start, start + count))


def validate_config(config: dict[str, Any]) -> None:
    discovery = set(seed_values(config["discovery_seeds"]))
    validation = set(seed_values(config["validation_seeds"]))
    if not discovery or not validation or discovery & validation:
        raise ValueError("discovery and validation seeds must be nonempty and disjoint")
    if (discovery | validation) & FORBIDDEN_SEEDS:
        raise ValueError("configuration uses a forbidden prior or formal seed")
    if len(config["scenarios"]) < 3:
        raise ValueError("J2a requires at least three schedule variants")
    required_families = {"static_ot", "forecast_iot", "graph_forecast_iot"}
    if set(config["tuning_grids"]) != required_families:
        raise ValueError("tuning_grids must contain exactly the three frozen families")
    overrides = dict(config["common_env_overrides"])
    for key in (
        "enable_scheduled_referral_waves",
        "enable_overtime_control",
        "enable_intertemporal_overtime_commitment",
    ):
        if not bool(overrides.get(key)):
            raise ValueError(f"J2a requires {key}")


def scenario_env(config: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    env = load_json(Path(config["env_config"]))
    env.update(dict(config["common_env_overrides"]))
    env.update(dict(scenario["overrides"]))
    return env


def policy_config(family: str, value: float) -> dict[str, float]:
    if family == "static_ot":
        return {"static_overtime_fraction": float(value)}
    return {"forecast_overtime_budget_fraction": float(value)}


def run_episode(
    *,
    env_config: dict[str, Any],
    scenario: str,
    seed: int,
    family: str,
    tuning_value: float,
    phase: str,
) -> dict[str, Any]:
    env = build_env({"env": env_config}, seed=seed)
    policy = get_heuristic_class(family)(
        config=policy_config(family, tuning_value)
    )
    total_cost = 0.0
    total_lost = 0.0
    max_requested = 0.0
    max_active = 0.0
    final_info: dict[str, Any] = {}
    done = False
    while not done:
        action = policy.select_action(env.observation(), env=env)
        _, reward, done, info = env.step(action)
        if not math.isfinite(float(reward)) or not math.isfinite(float(info["cost"])):
            raise RuntimeError("non-finite episode metric")
        total_cost += float(info["cost"])
        total_lost += float(np.asarray(info["patients_lost"], dtype=float).sum())
        max_requested = max(
            max_requested,
            float(np.asarray(info["overtime_requested_surge"], dtype=float).sum()),
        )
        max_active = max(
            max_active,
            float(np.asarray(info["overtime_active_capacity"], dtype=float).sum()),
        )
        final_info = info
    budget = float(env._shared_overtime_budget())
    if max_requested > budget + 1e-8:
        raise RuntimeError("shared overtime budget exceeded")
    return {
        "phase": phase,
        "scenario": scenario,
        "seed": int(seed),
        "family": family,
        "tuning_value": float(tuning_value),
        "total_cost": total_cost,
        "patients_lost": total_lost,
        "completion_service_level": float(final_info["completion_service_level"]),
        "max_requested_capacity": max_requested,
        "max_active_capacity": max_active,
        "shared_budget": budget,
        "rng_sha256": rng_digest(env),
    }


def select_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), float(row["tuning_value"]))].append(
            float(row["total_cost"])
        )
    selected: dict[str, float] = {}
    for family in ("static_ot", "forecast_iot", "graph_forecast_iot"):
        candidates = [
            (float(np.mean(costs)), value)
            for (candidate_family, value), costs in grouped.items()
            if candidate_family == family
        ]
        selected[family] = float(min(candidates)[1])
    return selected


def assert_rng_alignment(rows: list[dict[str, Any]]) -> int:
    grouped: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in rows:
        grouped[
            (str(row["phase"]), str(row["scenario"]), int(row["seed"]))
        ].add(str(row["rng_sha256"]))
    mismatches = [key for key, digests in grouped.items() if len(digests) != 1]
    if mismatches:
        raise RuntimeError(f"policy arms did not preserve exact RNG use: {mismatches}")
    return len(grouped)


def normal_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    sem = float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "sem": sem,
        "normal_95_half_width": 1.96 * sem,
    }


def candidate_summary(
    rows: list[dict[str, Any]],
    *,
    candidate: str,
    scenarios: tuple[str, ...],
    validation_seeds: tuple[int, ...],
) -> dict[str, Any]:
    lookup = {
        (str(row["family"]), str(row["scenario"]), int(row["seed"])): row
        for row in rows
    }
    differences: dict[tuple[str, int], float] = {}
    by_scenario: dict[str, Any] = {}
    for scenario in scenarios:
        scenario_values = []
        for seed in validation_seeds:
            static = lookup[("static_ot", scenario, seed)]
            treatment = lookup[(candidate, scenario, seed)]
            saving = (
                float(static["total_cost"]) - float(treatment["total_cost"])
            ) / float(static["total_cost"])
            differences[(scenario, seed)] = saving
            scenario_values.append(saving)
        by_scenario[scenario] = {
            **normal_summary(scenario_values),
            "positive_pairs": int(sum(value > 0.0 for value in scenario_values)),
        }

    seed_cluster_values = [
        float(np.mean([differences[(scenario, seed)] for scenario in scenarios]))
        for seed in validation_seeds
    ]
    static_rows = [row for row in rows if row["family"] == "static_ot"]
    candidate_rows = [row for row in rows if row["family"] == candidate]
    static_lost = float(sum(float(row["patients_lost"]) for row in static_rows))
    candidate_lost = float(
        sum(float(row["patients_lost"]) for row in candidate_rows)
    )
    static_service = float(
        np.mean([float(row["completion_service_level"]) for row in static_rows])
    )
    candidate_service = float(
        np.mean([float(row["completion_service_level"]) for row in candidate_rows])
    )
    return {
        "candidate": candidate,
        "pooled_seed_clustered": normal_summary(seed_cluster_values),
        "by_scenario": by_scenario,
        "positive_pairs": int(sum(value > 0.0 for value in differences.values())),
        "pair_count": len(differences),
        "clinical": {
            "static_patients_lost": static_lost,
            "candidate_patients_lost": candidate_lost,
            "static_mean_completion_service_level": static_service,
            "candidate_mean_completion_service_level": candidate_service,
            "noninferior": bool(
                candidate_lost <= static_lost
                and candidate_service + 1e-12 >= static_service
            ),
        },
    }


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


def run(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    validate_config(config)
    scenarios = tuple(str(item["name"]) for item in config["scenarios"])
    discovery_seeds = seed_values(config["discovery_seeds"])
    validation_seeds = seed_values(config["validation_seeds"])
    discovery_rows: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        env_config = scenario_env(config, scenario)
        for seed in discovery_seeds:
            for family, values in config["tuning_grids"].items():
                for value in values:
                    discovery_rows.append(
                        run_episode(
                            env_config=env_config,
                            scenario=str(scenario["name"]),
                            seed=seed,
                            family=str(family),
                            tuning_value=float(value),
                            phase="discovery",
                        )
                    )
    selected = select_values(discovery_rows)
    discovery_rng_groups = assert_rng_alignment(discovery_rows)

    validation_rows: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        env_config = scenario_env(config, scenario)
        for seed in validation_seeds:
            for family, value in selected.items():
                validation_rows.append(
                    run_episode(
                        env_config=env_config,
                        scenario=str(scenario["name"]),
                        seed=seed,
                        family=family,
                        tuning_value=value,
                        phase="validation",
                    )
                )
    validation_rng_groups = assert_rng_alignment(validation_rows)

    candidates = {
        family: candidate_summary(
            validation_rows,
            candidate=family,
            scenarios=scenarios,
            validation_seeds=validation_seeds,
        )
        for family in ("forecast_iot", "graph_forecast_iot")
    }
    gate = config["gate"]
    for result in candidates.values():
        result["passes"] = bool(
            float(result["pooled_seed_clustered"]["mean"])
            >= float(gate["minimum_relative_saving"])
            and (
                not bool(gate["require_positive_every_scenario"])
                or all(
                    float(item["mean"]) > 0.0
                    for item in result["by_scenario"].values()
                )
            )
            and (
                not bool(gate["require_clinical_noninferiority"])
                or bool(result["clinical"]["noninferior"])
            )
        )
    eligible = [result for result in candidates.values() if result["passes"]]
    winner = (
        max(
            eligible,
            key=lambda item: float(item["pooled_seed_clustered"]["mean"]),
        )["candidate"]
        if eligible
        else None
    )

    output_root = Path(config["output_root"])
    rows_path = output_root / "episode_rows.csv"
    summary_path = output_root / "summary.json"
    inventory_path = output_root / "artifact_inventory.json"
    write_rows(discovery_rows + validation_rows, rows_path)
    summary = {
        "name": config["name"],
        "experimental_role": config["experimental_role"],
        "policy_training_performed": False,
        "formal_confirmation_performed": False,
        "selected_tuning_values": selected,
        "discovery_episode_count": len(discovery_rows),
        "validation_episode_count": len(validation_rows),
        "paired_rng_audit": {
            "discovery_groups": discovery_rng_groups,
            "validation_groups": validation_rng_groups,
            "passes": True,
        },
        "candidates": candidates,
        "decision": (
            "episode_level_state_dependent_signal_detected"
            if winner is not None
            else "episode_level_prescreen_failed"
        ),
        "selected_candidate": winner,
        "j2_authorized": bool(winner is not None),
        "training_authorized": False,
        "provenance": {
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "rows_sha256": sha256(rows_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_paths = (
        Path("evaluation/screen_intertemporal_shared_capacity_episodes.py"),
        Path("src/env/capacity_planning.py"),
        Path("src/env/patient_capacity_planning.py"),
        Path("src/baselines/heuristics.py"),
    )
    inventory = {
        "files": [
            {"path": str(config_path), "sha256": sha256(config_path)},
            *[
                {"path": str(path), "sha256": sha256(path)}
                for path in source_paths
            ],
            {"path": str(rows_path), "sha256": sha256(rows_path)},
            {"path": str(summary_path), "sha256": sha256(summary_path)},
        ]
    }
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_json(config_path)
    if args.output_root:
        config["output_root"] = args.output_root
    run(config, config_path=config_path)


if __name__ == "__main__":
    main()
