"""Fixed-state J2 screen for intertemporal shared capacity."""

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

from evaluation.screen_intertemporal_shared_capacity_episodes import (
    FORBIDDEN_SEEDS,
    load_json,
    normal_summary,
    scenario_env,
    seed_values,
    sha256,
)
from src.baselines.heuristics import GraphSmoothedIntertemporalForecastPolicy
from src.rl.experiment import build_env


DEFAULT_CONFIG = Path(
    "experiments/configs/intertemporal_shared_capacity_fixed_state_j2.json"
)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(config["source_scenario_config"])
    if sha256(source_path) != str(config["source_scenario_config_sha256"]):
        raise ValueError("source scenario config hash mismatch")
    source = load_json(source_path)
    generation = set(seed_values(config["generation_seeds"]))
    discovery_start = int(config["discovery_worlds"]["start"])
    discovery_count = int(config["discovery_worlds"]["count_per_state"])
    validation_start = int(config["validation_worlds"]["start"])
    validation_count = int(config["validation_worlds"]["count_per_state"])
    if not generation or discovery_count < 1 or validation_count < 1:
        raise ValueError("state and world seed counts must be positive")
    expected_states = (
        len(source["scenarios"])
        * len(generation)
        * len(config["decision_epochs"])
    )
    discovery = set(
        range(discovery_start, discovery_start + expected_states * discovery_count)
    )
    validation = set(
        range(validation_start, validation_start + expected_states * validation_count)
    )
    if generation & discovery or generation & validation or discovery & validation:
        raise ValueError("state generation and world seed families must be disjoint")
    reserved = generation | discovery | validation
    if reserved & FORBIDDEN_SEEDS:
        raise ValueError("fixed-state screen uses a forbidden seed")
    epochs = tuple(int(value) for value in config["decision_epochs"])
    if (
        not epochs
        or any(value < 0 for value in epochs)
        or len(set(epochs)) != len(epochs)
        or tuple(sorted(epochs)) != epochs
    ):
        raise ValueError(
            "decision_epochs must be unique, increasing, and nonnegative"
        )
    if int(config["lookahead"]) < int(
        source["common_env_overrides"]["overtime_commitment_lead_time"]
    ) + 1:
        raise ValueError("lookahead must extend beyond commitment maturity")
    return source


def candidate_specs(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    specs = []
    for budget_fraction in config["action_library"][
        "forecast_overtime_budget_fractions"
    ]:
        for smoothing in config["action_library"]["graph_forecast_smoothing"]:
            specs.append(
                {
                    "candidate": f"b{float(budget_fraction):.2f}_a{float(smoothing):.2f}",
                    "budget_fraction": float(budget_fraction),
                    "smoothing": float(smoothing),
                }
            )
    if len({item["candidate"] for item in specs}) != len(specs):
        raise ValueError("action library contains duplicate candidates")
    if len(specs) < int(config["gate"]["minimum_distinct_selected_actions"]):
        raise ValueError("action library is smaller than the distinct-action gate")
    if any(
        not 0.0 <= item[key] <= 1.0
        for item in specs
        for key in ("budget_fraction", "smoothing")
    ):
        raise ValueError("candidate parameters must be within [0, 1]")
    return tuple(specs)


def make_policy(spec: dict[str, Any]) -> GraphSmoothedIntertemporalForecastPolicy:
    return GraphSmoothedIntertemporalForecastPolicy(
        config={
            "forecast_overtime_budget_fraction": float(spec["budget_fraction"]),
            "graph_forecast_smoothing": float(spec["smoothing"]),
        }
    )


def observation_sha256(env) -> str:
    return hashlib.sha256(np.asarray(env.observation()).tobytes()).hexdigest()


def generate_states(
    config: dict[str, Any], source: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    behavior = make_policy(
        {
            "budget_fraction": config["behavior_policy"][
                "forecast_overtime_budget_fraction"
            ],
            "smoothing": config["behavior_policy"]["graph_forecast_smoothing"],
        }
    )
    epochs = set(int(value) for value in config["decision_epochs"])
    max_epoch = max(epochs)
    states: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    index = 0
    for scenario in source["scenarios"]:
        env_config = scenario_env(source, scenario)
        for generation_seed in seed_values(config["generation_seeds"]):
            env = build_env({"env": env_config}, seed=generation_seed)
            while env.t <= max_epoch:
                if env.t in epochs:
                    state_id = (
                        f"{scenario['name']}__g{generation_seed}__t{int(env.t):03d}"
                    )
                    snapshot = env.state_dict()
                    states.append(
                        {
                            "state_index": index,
                            "state_id": state_id,
                            "scenario": str(scenario["name"]),
                            "generation_seed": int(generation_seed),
                            "epoch": int(env.t),
                            "env_config": env_config,
                            "snapshot": snapshot,
                        }
                    )
                    manifest.append(
                        {
                            "state_index": index,
                            "state_id": state_id,
                            "scenario": str(scenario["name"]),
                            "generation_seed": int(generation_seed),
                            "epoch": int(env.t),
                            "observation_sha256": observation_sha256(env),
                        }
                    )
                    index += 1
                if env.t >= max_epoch:
                    break
                action = behavior.select_action(env.observation(), env=env)
                env.step(action)
    return states, manifest


def rollout(
    *,
    state: dict[str, Any],
    candidate: dict[str, Any],
    continuation_spec: dict[str, Any],
    world_seed: int,
    phase: str,
    lookahead: int,
) -> dict[str, Any]:
    env = build_env({"env": state["env_config"]}, seed=world_seed)
    env.load_state_dict(state["snapshot"])
    env.rng = np.random.default_rng(world_seed)
    first_policy = make_policy(candidate)
    continuation = make_policy(
        {
            "budget_fraction": continuation_spec[
                "forecast_overtime_budget_fraction"
            ],
            "smoothing": continuation_spec["graph_forecast_smoothing"],
        }
    )
    first_action = first_policy.select_action(env.observation(), env=env)
    fraction, requested = env._decode_overtime(first_action)
    projected_fraction, projected = env._project_overtime_commitment(
        fraction, requested
    )
    budget = float(env._shared_overtime_budget())
    allocation_scale = budget if budget > 0.0 else 1.0
    allocation_signature = ",".join(
        f"{value:.3f}" for value in projected / allocation_scale
    )
    total_cost = 0.0
    total_lost = 0.0
    final_info: dict[str, Any] = {}
    for step in range(lookahead):
        action = (
            first_action
            if step == 0
            else continuation.select_action(env.observation(), env=env)
        )
        _, reward, done, info = env.step(action)
        if not math.isfinite(float(reward)) or not math.isfinite(float(info["cost"])):
            raise RuntimeError("non-finite fixed-state rollout")
        total_cost += float(info["cost"])
        total_lost += float(np.asarray(info["patients_lost"], dtype=float).sum())
        final_info = info
        if done:
            break
    return {
        "phase": phase,
        "state_index": int(state["state_index"]),
        "state_id": str(state["state_id"]),
        "scenario": str(state["scenario"]),
        "generation_seed": int(state["generation_seed"]),
        "epoch": int(state["epoch"]),
        "world_seed": int(world_seed),
        "candidate": str(candidate["candidate"]),
        "budget_fraction": float(candidate["budget_fraction"]),
        "smoothing": float(candidate["smoothing"]),
        "total_cost": total_cost,
        "patients_lost": total_lost,
        "completion_service_level": float(final_info["completion_service_level"]),
        "request_total": float(projected.sum()),
        "allocation_signature": allocation_signature,
        "shared_budget": budget,
        "local_interior_fraction": float(
            np.mean((projected_fraction > 1e-8) & (projected_fraction < 1.0 - 1e-8))
        ),
        "shared_budget_interior": bool(float(projected.sum()) < budget - 1e-8),
        "rng_sha256": hashlib.sha256(
            json.dumps(
                env.rng.bit_generator.state,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def sensitivity_fraction(
    states: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[int, int, float]:
    delta = float(config["local_raw_perturbation"])
    behavior_spec = {
        "budget_fraction": config["behavior_policy"][
            "forecast_overtime_budget_fraction"
        ],
        "smoothing": config["behavior_policy"]["graph_forecast_smoothing"],
    }
    policy = make_policy(behavior_spec)
    changed = 0
    total = 0
    for state in states:
        env = build_env({"env": state["env_config"]}, seed=state["generation_seed"])
        env.load_state_dict(state["snapshot"])
        action = policy.select_action(env.observation(), env=env)
        n = env.config.num_facilities
        base_fraction, base_surge = env._decode_overtime(action)
        _, base_projected = env._project_overtime_commitment(
            base_fraction, base_surge
        )
        for facility in range(n):
            perturbed = np.asarray(action, dtype=float).copy()
            index = 4 * n + facility
            direction = delta if perturbed[index] <= 1.0 - delta else -delta
            perturbed[index] = np.clip(perturbed[index] + direction, -1.0, 1.0)
            fraction, surge = env._decode_overtime(perturbed)
            _, projected = env._project_overtime_commitment(fraction, surge)
            changed += int(float(np.max(np.abs(projected - base_projected))) > 1e-8)
            total += 1
    return changed, total, changed / total if total else 0.0


def select_candidates(
    rows: list[dict[str, Any]], candidates: tuple[dict[str, Any], ...]
) -> tuple[str, dict[int, str]]:
    costs: dict[tuple[int, str], list[float]] = defaultdict(list)
    pooled: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = (int(row["state_index"]), str(row["candidate"]))
        costs[key].append(float(row["total_cost"]))
        pooled[str(row["candidate"])].append(float(row["total_cost"]))
    candidate_ids = tuple(str(item["candidate"]) for item in candidates)
    global_candidate = min(
        candidate_ids,
        key=lambda candidate: (float(np.mean(pooled[candidate])), candidate),
    )
    state_indices = sorted({int(row["state_index"]) for row in rows})
    selected = {
        state_index: min(
            candidate_ids,
            key=lambda candidate: (
                float(np.mean(costs[(state_index, candidate)])),
                candidate,
            ),
        )
        for state_index in state_indices
    }
    return global_candidate, selected


def validation_summary(
    rows: list[dict[str, Any]],
    *,
    global_candidate: str,
    selected: dict[int, str],
    candidates: tuple[dict[str, Any], ...],
    gate: dict[str, Any],
    behavioral_sensitivity_fraction: float,
) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["state_index"]), str(row["candidate"]))].append(row)
    state_results = []
    for state_index, selected_candidate in sorted(selected.items()):
        global_rows = grouped[(state_index, global_candidate)]
        selected_rows = grouped[(state_index, selected_candidate)]
        global_cost = float(np.mean([float(row["total_cost"]) for row in global_rows]))
        selected_cost = float(
            np.mean([float(row["total_cost"]) for row in selected_rows])
        )
        global_lost = float(
            np.mean([float(row["patients_lost"]) for row in global_rows])
        )
        selected_lost = float(
            np.mean([float(row["patients_lost"]) for row in selected_rows])
        )
        global_service = float(
            np.mean([float(row["completion_service_level"]) for row in global_rows])
        )
        selected_service = float(
            np.mean([float(row["completion_service_level"]) for row in selected_rows])
        )
        saving = global_cost - selected_cost
        clinical = bool(
            selected_lost <= global_lost
            and selected_service + 1e-12 >= global_service
        )
        first = selected_rows[0]
        relative_saving = saving / global_cost
        state_results.append(
            {
                "state_index": state_index,
                "state_id": str(first["state_id"]),
                "scenario": str(first["scenario"]),
                "generation_seed": int(first["generation_seed"]),
                "selected_candidate": selected_candidate,
                "global_cost": global_cost,
                "selected_cost": selected_cost,
                "saving": saving,
                "relative_saving": relative_saving,
                "clinical_noninferior": clinical,
                "interior": bool(
                    bool(first["shared_budget_interior"])
                    and float(first["local_interior_fraction"]) >= 0.5
                ),
                "material": bool(
                    relative_saving
                    >= float(gate["material_relative_saving"])
                    and clinical
                ),
                "allocation_signature": str(first["allocation_signature"]),
                "global_lost": global_lost,
                "selected_lost": selected_lost,
                "global_service": global_service,
                "selected_service": selected_service,
            }
        )

    global_total = float(sum(item["global_cost"] for item in state_results))
    selected_total = float(sum(item["selected_cost"] for item in state_results))
    relative_saving = (global_total - selected_total) / global_total
    cluster_values = []
    clusters = defaultdict(list)
    for item in state_results:
        clusters[(item["scenario"], item["generation_seed"])].append(item)
    for items in clusters.values():
        global_cluster = float(sum(item["global_cost"] for item in items))
        selected_cluster = float(sum(item["selected_cost"] for item in items))
        cluster_values.append((global_cluster - selected_cluster) / global_cluster)
    material_fraction = float(np.mean([item["material"] for item in state_results]))
    interior_fraction = float(np.mean([item["interior"] for item in state_results]))
    distinct = len({item["allocation_signature"] for item in state_results})
    clinical_noninferior = bool(
        sum(item["selected_lost"] for item in state_results)
        <= sum(item["global_lost"] for item in state_results)
        and np.mean([item["selected_service"] for item in state_results]) + 1e-12
        >= np.mean([item["global_service"] for item in state_results])
    )
    passes = bool(
        behavioral_sensitivity_fraction
        >= float(gate["minimum_behavioral_sensitivity_fraction"])
        and material_fraction >= float(gate["minimum_material_state_fraction"])
        and relative_saving >= float(gate["minimum_relative_saving"])
        and interior_fraction >= float(gate["minimum_interior_selected_fraction"])
        and distinct >= int(gate["minimum_distinct_selected_actions"])
        and (
            not bool(gate["require_clinical_noninferiority"])
            or clinical_noninferior
        )
    )
    return {
        "global_candidate": global_candidate,
        "state_count": len(state_results),
        "relative_saving": relative_saving,
        "seed_scenario_clustered": normal_summary(cluster_values),
        "material_state_fraction": material_fraction,
        "interior_selected_fraction": interior_fraction,
        "distinct_selected_actions": distinct,
        "clinical_noninferior": clinical_noninferior,
        "behavioral_sensitivity_fraction": behavioral_sensitivity_fraction,
        "passes": passes,
        "decision": (
            "fixed_state_headroom_established"
            if passes
            else "fixed_state_j2_failed"
        ),
        "state_results": state_results,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
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
    source = validate_config(config)
    candidates = candidate_specs(config)
    states, manifest = generate_states(config, source)
    expected_states = (
        len(source["scenarios"])
        * int(config["generation_seeds"]["count"])
        * len(config["decision_epochs"])
    )
    if len(states) != expected_states:
        raise RuntimeError("state generation count mismatch")
    print(json.dumps({"phase": "states", "completed": len(states)}), flush=True)

    changed, sensitivity_total, sensitivity = sensitivity_fraction(states, config)
    rows: list[dict[str, Any]] = []
    for phase, block in (
        ("discovery", config["discovery_worlds"]),
        ("validation", config["validation_worlds"]),
    ):
        count = int(block["count_per_state"])
        start = int(block["start"])
        for state in states:
            for replication in range(count):
                world_seed = start + int(state["state_index"]) * count + replication
                world_rows = []
                for candidate in candidates:
                    row = rollout(
                        state=state,
                        candidate=candidate,
                        continuation_spec=config["continuation_policy"],
                        world_seed=world_seed,
                        phase=phase,
                        lookahead=int(config["lookahead"]),
                    )
                    rows.append(row)
                    world_rows.append(row)
                if len({row["rng_sha256"] for row in world_rows}) != 1:
                    raise RuntimeError("candidate arms did not preserve exact RNG use")
        print(
            json.dumps(
                {
                    "phase": phase,
                    "completed_rows": sum(row["phase"] == phase for row in rows),
                }
            ),
            flush=True,
        )

    discovery_rows = [row for row in rows if row["phase"] == "discovery"]
    validation_rows = [row for row in rows if row["phase"] == "validation"]
    global_candidate, selected = select_candidates(discovery_rows, candidates)
    result = validation_summary(
        validation_rows,
        global_candidate=global_candidate,
        selected=selected,
        candidates=candidates,
        gate=config["gate"],
        behavioral_sensitivity_fraction=sensitivity,
    )

    output_root = Path(config["output_root"])
    rows_path = output_root / "headroom_rows.csv"
    manifest_path = output_root / "state_manifest.json"
    summary_path = output_root / "summary.json"
    inventory_path = output_root / "artifact_inventory.json"
    write_csv(rows, rows_path)
    manifest_path.write_text(
        json.dumps({"states": manifest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "name": config["name"],
        "experimental_role": config["experimental_role"],
        "policy_training_performed": False,
        "formal_confirmation_performed": False,
        "candidate_count": len(candidates),
        "state_count": len(states),
        "discovery_row_count": len(discovery_rows),
        "validation_row_count": len(validation_rows),
        "behavioral_sensitivity": {
            "changed": changed,
            "total": sensitivity_total,
            "fraction": sensitivity,
        },
        **result,
        "j3_authorized": bool(result["passes"]),
        "training_authorized": False,
        "provenance": {
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "source_scenario_config": str(config["source_scenario_config"]),
            "source_scenario_config_sha256": sha256(
                Path(config["source_scenario_config"])
            ),
            "rows_sha256": sha256(rows_path),
            "state_manifest_sha256": sha256(manifest_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_paths = (
        Path("evaluation/screen_intertemporal_shared_capacity_states.py"),
        Path("evaluation/screen_intertemporal_shared_capacity_episodes.py"),
        Path("src/env/capacity_planning.py"),
        Path("src/env/patient_capacity_planning.py"),
        Path("src/baselines/heuristics.py"),
    )
    inventory = {
        "files": [
            {"path": str(config_path), "sha256": sha256(config_path)},
            {
                "path": str(config["source_scenario_config"]),
                "sha256": sha256(Path(config["source_scenario_config"])),
            },
            *[
                {"path": str(path), "sha256": sha256(path)}
                for path in source_paths
            ],
            {"path": str(rows_path), "sha256": sha256(rows_path)},
            {"path": str(manifest_path), "sha256": sha256(manifest_path)},
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
