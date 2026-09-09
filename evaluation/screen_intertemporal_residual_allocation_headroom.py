"""Prospective screen for budget-neutral per-facility overtime residuals.

This module deliberately contains no learned policy.  It asks whether local
continuous reallocations around the strongest frozen graph-forecast policy
have enough prospective value and label stability to justify a later learner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.screen_intertemporal_shared_capacity_episodes import (
    FORBIDDEN_SEEDS,
    load_json,
    normal_summary,
    seed_values,
    sha256,
)
from evaluation.screen_intertemporal_shared_capacity_states import (
    generate_states,
    make_policy,
)
from src.rl.experiment import build_env


DEFAULT_CONFIG = Path(
    "experiments/configs/intertemporal_residual_allocation_headroom.json"
)
BASELINE_CANDIDATE = "baseline_graph_forecast"
USER_EXCEPTION = {
    "mode": "zhaowei_directed_development_screen_20260908",
    "record": "specs/2026-09-08-user-authorized-residual-screen/authorization.md",
    "record_sha256": "6103f6eb1719ee13c596addb8d58de6f863d33221487b97803297a0442f50676",
    "original_config": str(DEFAULT_CONFIG),
    "original_config_sha256": "df9ed23407dbe7cbf47050a550755027743fbf30fb4688848a4f413d80a73ca2",
}
USER_OUTPUT_ROOT = (
    "results/intertemporal_residual_allocation_headroom_development/"
    "r0_zhaowei_20260908"
)


def execution_approvals(config: dict[str, Any]) -> dict[str, bool]:
    block = config.get("execution_authorization", {})
    return {
        role: details.get("approved") is True
        for role, details in block.items()
    }


def require_execution_authorization(config: dict[str, Any]) -> None:
    approvals = execution_approvals(config)
    required = {"zhaowei", "howard"}
    if set(approvals) != required:
        raise PermissionError(
            "execution requires explicit zhaowei and howard authorization records"
        )
    if "execution_exception" in config:
        _require_user_directed_exception(config, approvals)
        return
    pending = sorted(role for role, approved in approvals.items() if not approved)
    if pending:
        raise PermissionError(
            "scientific execution is not authorized; pending approval: "
            + ", ".join(pending)
        )


def _require_user_directed_exception(
    config: dict[str, Any], approvals: dict[str, bool]
) -> None:
    # This dated exception cannot authorize a different scientific contract.
    if config["execution_exception"] != USER_EXCEPTION:
        raise PermissionError("unrecognized execution exception record")
    if approvals != {"zhaowei": True, "howard": False}:
        raise PermissionError("user exception must retain Howard as unapproved")
    if config["execution_authorization"]["zhaowei"].get("date") != "2026-09-08":
        raise PermissionError("user exception requires the actual authorization date")
    for path_key, hash_key in (
        ("record", "record_sha256"),
        ("original_config", "original_config_sha256"),
    ):
        path = Path(USER_EXCEPTION[path_key])
        if not path.is_file() or sha256(path) != USER_EXCEPTION[hash_key]:
            raise PermissionError(f"user exception evidence hash mismatch: {path}")
    original = load_json(Path(USER_EXCEPTION["original_config"]))
    metadata = {
        "name", "experimental_role", "execution_authorization",
        "execution_exception", "output_root",
    }
    scientific = {key: value for key, value in config.items() if key not in metadata}
    frozen = {key: value for key, value in original.items() if key not in metadata}
    if scientific != frozen:
        raise PermissionError("user exception may not change scientific settings")
    if config["output_root"] != USER_OUTPUT_ROOT:
        raise PermissionError("user exception requires its separate single-use root")


def _seed_set(start: int, count: int) -> set[int]:
    return set(range(int(start), int(start) + int(count)))


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    protocol_path = Path(config["protocol"])
    if sha256(protocol_path) != str(config["protocol_sha256"]):
        raise ValueError("frozen protocol hash mismatch")
    source_path = Path(config["source_scenario_config"])
    if sha256(source_path) != str(config["source_scenario_config_sha256"]):
        raise ValueError("source scenario config hash mismatch")
    parent_path = Path(config["parent_j2_summary"])
    if sha256(parent_path) != str(config["parent_j2_summary_sha256"]):
        raise ValueError("parent J2 summary hash mismatch")
    parent = load_json(parent_path)
    if parent.get("decision") != config["required_parent_decision"]:
        raise ValueError("parent J2 decision does not match the frozen contract")
    if parent.get("global_candidate") != config["required_parent_global_candidate"]:
        raise ValueError("parent J2 comparator does not match the frozen contract")

    source = load_json(source_path)
    expected_states = (
        len(source["scenarios"])
        * int(config["generation_seeds"]["count"])
        * len(config["decision_epochs"])
    )
    generation = set(seed_values(config["generation_seeds"]))
    discovery = _seed_set(
        config["discovery_worlds"]["start"],
        expected_states * int(config["discovery_worlds"]["count_per_state"]),
    )
    validation = _seed_set(
        config["validation_worlds"]["start"],
        expected_states * int(config["validation_worlds"]["count_per_state"]),
    )
    if not generation or not discovery or not validation:
        raise ValueError("all seed families must be nonempty")
    if generation & discovery or generation & validation or discovery & validation:
        raise ValueError("state and world seed families must be disjoint")
    reserved = generation | discovery | validation
    if reserved & FORBIDDEN_SEEDS:
        raise ValueError("configuration uses a forbidden prior or formal seed")
    for lower, upper in config["forbidden_seed_ranges"]:
        if any(int(lower) <= value <= int(upper) for value in reserved):
            raise ValueError("configuration overlaps a frozen prior seed range")

    epochs = tuple(int(value) for value in config["decision_epochs"])
    if not epochs or tuple(sorted(set(epochs))) != epochs or min(epochs) < 0:
        raise ValueError("decision_epochs must be unique, increasing, and nonnegative")
    if int(config["lookahead"]) < int(
        source["common_env_overrides"]["overtime_commitment_lead_time"]
    ) + 1:
        raise ValueError("lookahead must extend beyond commitment maturity")

    expected_policy = {
        "forecast_overtime_budget_fraction": 1.0,
        "graph_forecast_smoothing": 0.25,
    }
    for key in ("behavior_policy", "baseline_policy", "continuation_policy"):
        if config[key] != expected_policy:
            raise ValueError(f"{key} must equal the frozen strongest J2 comparator")

    library = config["action_library"]
    fractions = tuple(float(value) for value in library["transfer_budget_fractions"])
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError("transfer fractions must be nonempty and unique")
    if any(not 0.0 < value <= 0.5 for value in fractions):
        raise ValueError("transfer fractions must lie in (0, 0.5]")
    if tuple(library["directions"]) != ("decrease", "increase"):
        raise ValueError("directions must be frozen as decrease then increase")

    gate = config["gate"]
    for key in (
        "minimum_optimistic_relative_saving",
        "minimum_prospective_relative_saving",
        "material_relative_saving",
    ):
        if float(gate[key]) < 0.005:
            raise ValueError(f"{key} may not be weaker than the prior 0.5% gate")
    if float(gate["minimum_material_state_fraction"]) < 0.30:
        raise ValueError("material-state gate may not be weaker than 30%")
    if float(gate["minimum_selected_validation_win_fraction"]) < 0.70:
        raise ValueError("selected-action stability may not be weaker than 70%")
    if float(gate["minimum_pairwise_material_sign_agreement"]) < 0.80:
        raise ValueError("pairwise sign stability may not be weaker than 80%")
    return source


def candidate_specs(
    config: dict[str, Any], num_facilities: int
) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = [
        {
            "candidate": BASELINE_CANDIDATE,
            "facility": -1,
            "direction": "baseline",
            "transfer_budget_fraction": 0.0,
        }
    ]
    for facility in range(int(num_facilities)):
        for direction in config["action_library"]["directions"]:
            for fraction in config["action_library"]["transfer_budget_fractions"]:
                specs.append(
                    {
                        "candidate": (
                            f"f{facility:02d}_{direction}_{float(fraction):.2f}"
                        ),
                        "facility": facility,
                        "direction": str(direction),
                        "transfer_budget_fraction": float(fraction),
                    }
                )
    if len({spec["candidate"] for spec in specs}) != len(specs):
        raise ValueError("residual action library contains duplicate candidates")
    return tuple(specs)


def budget_neutral_reallocation(
    base_allocation: np.ndarray,
    caps: np.ndarray,
    *,
    target: int,
    signed_amount: float,
) -> tuple[np.ndarray, float]:
    """Move capacity to or from one facility while preserving total capacity."""

    base = np.asarray(base_allocation, dtype=float)
    limits = np.asarray(caps, dtype=float)
    if base.ndim != 1 or limits.shape != base.shape:
        raise ValueError("base allocation and caps must be matching vectors")
    if not 0 <= int(target) < base.size:
        raise ValueError("target facility is out of range")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(limits)):
        raise ValueError("allocation inputs must be finite")
    if np.any(base < -1e-12) or np.any(base > limits + 1e-12):
        raise ValueError("base allocation violates local caps")

    result = base.copy()
    amount = abs(float(signed_amount))
    moved = 0.0
    if signed_amount > 0.0:
        donors = np.maximum(base, 0.0)
        donors[int(target)] = 0.0
        moved = min(amount, float(limits[target] - base[target]), float(donors.sum()))
        if moved > 0.0:
            result[target] += moved
            result -= moved * donors / float(donors.sum())
    elif signed_amount < 0.0:
        receiver_room = np.maximum(limits - base, 0.0)
        receiver_room[int(target)] = 0.0
        moved = min(amount, float(base[target]), float(receiver_room.sum()))
        if moved > 0.0:
            result[target] -= moved
            result += moved * receiver_room / float(receiver_room.sum())

    result[np.abs(result) < 1e-14] = 0.0
    if np.any(result < -1e-10) or np.any(result > limits + 1e-10):
        raise RuntimeError("budget-neutral residual violated a local cap")
    if abs(float(result.sum() - base.sum())) > 1e-9:
        raise RuntimeError("budget-neutral residual changed total capacity")
    return result, moved


def _overtime_block(allocation: np.ndarray, caps: np.ndarray) -> np.ndarray:
    fractions = np.divide(
        np.asarray(allocation, dtype=float),
        np.asarray(caps, dtype=float),
        out=np.zeros_like(np.asarray(allocation, dtype=float)),
        where=np.asarray(caps, dtype=float) > 0.0,
    )
    return np.clip(2.0 * fractions - 1.0, -1.0, 1.0)


def residual_first_action(
    env,
    base_action: np.ndarray,
    candidate: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replace only the overtime slice with one frozen residual candidate."""

    action = np.asarray(base_action, dtype=float).copy()
    n = int(env.config.num_facilities)
    if action.shape != (5 * n,):
        raise ValueError("residual screen requires the five-block overtime action")
    base_fraction, base_requested = env._decode_overtime(action)
    _, base_allocation = env._project_overtime_commitment(
        base_fraction, base_requested
    )
    allocation = np.asarray(base_allocation, dtype=float).copy()
    moved = 0.0
    if candidate["candidate"] != BASELINE_CANDIDATE:
        if candidate["direction"] not in {"decrease", "increase"}:
            raise ValueError("residual candidate has an invalid direction")
        sign = 1.0 if candidate["direction"] == "increase" else -1.0
        requested_amount = (
            float(candidate["transfer_budget_fraction"])
            * float(env._shared_overtime_budget())
        )
        allocation, moved = budget_neutral_reallocation(
            allocation,
            np.asarray(env.overtime_surge_headroom, dtype=float),
            target=int(candidate["facility"]),
            signed_amount=sign * requested_amount,
        )
        action[4 * n : 5 * n] = _overtime_block(
            allocation, np.asarray(env.overtime_surge_headroom, dtype=float)
        )

    decoded_fraction, decoded_requested = env._decode_overtime(action)
    _, realized = env._project_overtime_commitment(
        decoded_fraction, decoded_requested
    )
    budget_error = float(realized.sum() - base_allocation.sum())
    if abs(budget_error) > 1e-8:
        raise RuntimeError("encoded residual action is not budget neutral")
    if not np.array_equal(action[: 4 * n], np.asarray(base_action)[: 4 * n]):
        raise RuntimeError("residual candidate changed a non-overtime action slice")
    residual = realized - base_allocation
    realized_transfer = float(np.abs(residual).sum() / 2.0)
    if abs(realized_transfer - moved) > 1e-8:
        raise RuntimeError("encoded action changed the feasible residual magnitude")
    return action, {
        "realized_transfer_units": realized_transfer,
        "residual_linf": float(np.max(np.abs(residual))),
        "changed_facilities": int(np.sum(np.abs(residual) > 1e-8)),
        "budget_error": budget_error,
        "request_total": float(realized.sum()),
        "shared_budget": float(env._shared_overtime_budget()),
        "allocation_signature": ",".join(f"{value:.6f}" for value in realized),
        "requested_transfer_units": float(
            abs(float(candidate["transfer_budget_fraction"]))
            * float(env._shared_overtime_budget())
        ),
        "reported_moved_units": float(moved),
    }


def rollout(
    *,
    state: dict[str, Any],
    candidate: dict[str, Any],
    baseline_spec: dict[str, Any],
    continuation_spec: dict[str, Any],
    world_seed: int,
    phase: str,
    lookahead: int,
) -> dict[str, Any]:
    env = build_env({"env": state["env_config"]}, seed=world_seed)
    env.load_state_dict(state["snapshot"])
    env.rng = np.random.default_rng(world_seed)
    baseline = make_policy(
        {
            "budget_fraction": baseline_spec["forecast_overtime_budget_fraction"],
            "smoothing": baseline_spec["graph_forecast_smoothing"],
        }
    )
    continuation = make_policy(
        {
            "budget_fraction": continuation_spec[
                "forecast_overtime_budget_fraction"
            ],
            "smoothing": continuation_spec["graph_forecast_smoothing"],
        }
    )
    base_action = baseline.select_action(env.observation(), env=env)
    first_action, mechanics = residual_first_action(env, base_action, candidate)

    total_cost = 0.0
    total_lost = 0.0
    final_info: dict[str, Any] = {}
    for step in range(int(lookahead)):
        action = (
            first_action
            if step == 0
            else continuation.select_action(env.observation(), env=env)
        )
        _, reward, done, info = env.step(action)
        if not math.isfinite(float(reward)) or not math.isfinite(float(info["cost"])):
            raise RuntimeError("non-finite residual-screen rollout")
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
        "facility": int(candidate["facility"]),
        "direction": str(candidate["direction"]),
        "transfer_budget_fraction": float(candidate["transfer_budget_fraction"]),
        "total_cost": total_cost,
        "patients_lost": total_lost,
        "completion_service_level": float(final_info["completion_service_level"]),
        **mechanics,
        "rng_sha256": hashlib.sha256(
            json.dumps(
                env.rng.bit_generator.state,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _mean_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[tuple[int, str], dict[str, float]], dict[int, dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[int, dict[str, Any]] = {}
    for row in rows:
        state_index = int(row["state_index"])
        grouped[(state_index, str(row["candidate"]))].append(row)
        metadata[state_index] = row
    means: dict[tuple[int, str], dict[str, float]] = {}
    for key, items in grouped.items():
        means[key] = {
            field: float(np.mean([float(item[field]) for item in items]))
            for field in ("total_cost", "patients_lost", "completion_service_level")
        }
    return means, metadata


def select_by_state(
    rows: list[dict[str, Any]], candidates: tuple[dict[str, Any], ...]
) -> dict[int, str]:
    means, metadata = _mean_rows(rows)
    candidate_ids = tuple(str(item["candidate"]) for item in candidates)
    return {
        state_index: min(
            candidate_ids,
            key=lambda candidate: (
                means[(state_index, candidate)]["total_cost"],
                candidate,
            ),
        )
        for state_index in sorted(metadata)
    }


def comparison_summary(
    rows: list[dict[str, Any]],
    choices: dict[int, str],
    *,
    material_threshold: float,
) -> dict[str, Any]:
    means, metadata = _mean_rows(rows)
    state_results: list[dict[str, Any]] = []
    for state_index, candidate in sorted(choices.items()):
        baseline = means[(state_index, BASELINE_CANDIDATE)]
        selected = means[(state_index, candidate)]
        relative_saving = (
            baseline["total_cost"] - selected["total_cost"]
        ) / baseline["total_cost"]
        clinical = bool(
            selected["patients_lost"] <= baseline["patients_lost"]
            and selected["completion_service_level"] + 1e-12
            >= baseline["completion_service_level"]
        )
        row = metadata[state_index]
        state_results.append(
            {
                "state_index": state_index,
                "state_id": str(row["state_id"]),
                "scenario": str(row["scenario"]),
                "generation_seed": int(row["generation_seed"]),
                "epoch": int(row["epoch"]),
                "candidate": candidate,
                "baseline_cost": baseline["total_cost"],
                "selected_cost": selected["total_cost"],
                "relative_saving": relative_saving,
                "clinical_noninferior": clinical,
                "material": bool(relative_saving >= material_threshold and clinical),
                "baseline_patients_lost": baseline["patients_lost"],
                "selected_patients_lost": selected["patients_lost"],
                "baseline_service": baseline["completion_service_level"],
                "selected_service": selected["completion_service_level"],
            }
        )

    baseline_total = float(sum(item["baseline_cost"] for item in state_results))
    selected_total = float(sum(item["selected_cost"] for item in state_results))
    scenario_results: dict[str, Any] = {}
    for scenario in sorted({item["scenario"] for item in state_results}):
        items = [item for item in state_results if item["scenario"] == scenario]
        base = float(sum(item["baseline_cost"] for item in items))
        chosen = float(sum(item["selected_cost"] for item in items))
        scenario_results[scenario] = {
            "state_count": len(items),
            "relative_saving": (base - chosen) / base,
            "material_state_fraction": float(np.mean([item["material"] for item in items])),
        }
    clusters: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in state_results:
        clusters[(item["scenario"], item["generation_seed"])].append(item)
    cluster_values = []
    for items in clusters.values():
        base = float(sum(item["baseline_cost"] for item in items))
        chosen = float(sum(item["selected_cost"] for item in items))
        cluster_values.append((base - chosen) / base)
    clinical = bool(
        sum(item["selected_patients_lost"] for item in state_results)
        <= sum(item["baseline_patients_lost"] for item in state_results)
        and np.mean([item["selected_service"] for item in state_results]) + 1e-12
        >= np.mean([item["baseline_service"] for item in state_results])
    )
    return {
        "state_count": len(state_results),
        "relative_saving": (baseline_total - selected_total) / baseline_total,
        "material_state_fraction": float(
            np.mean([item["material"] for item in state_results])
        ),
        "clinical_noninferior": clinical,
        "positive_every_scenario": all(
            float(item["relative_saving"]) > 0.0
            for item in scenario_results.values()
        ),
        "seed_scenario_clustered": normal_summary(cluster_values),
        "by_scenario": scenario_results,
        "state_results": state_results,
    }


def label_stability(
    discovery_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    discovery_choices: dict[int, str],
    validation_oracle: dict[int, str],
    *,
    material_threshold: float,
) -> dict[str, Any]:
    discovery, metadata = _mean_rows(discovery_rows)
    validation, _ = _mean_rows(validation_rows)
    selected_material = 0
    selected_validation_wins = 0
    material_pairs = 0
    sign_matches = 0
    candidate_ids = sorted(
        {
            candidate
            for _, candidate in discovery
            if candidate != BASELINE_CANDIDATE
        }
    )
    for state_index in sorted(metadata):
        discovery_base = discovery[(state_index, BASELINE_CANDIDATE)]["total_cost"]
        validation_base = validation[(state_index, BASELINE_CANDIDATE)]["total_cost"]
        selected = discovery_choices[state_index]
        discovery_selected = discovery[(state_index, selected)]["total_cost"]
        discovery_saving = (discovery_base - discovery_selected) / discovery_base
        if discovery_saving >= material_threshold:
            selected_material += 1
            validation_selected = validation[(state_index, selected)]["total_cost"]
            selected_validation_wins += int(validation_selected < validation_base)
        for candidate in candidate_ids:
            discovery_candidate = discovery[(state_index, candidate)]["total_cost"]
            discovery_delta = (discovery_base - discovery_candidate) / discovery_base
            if abs(discovery_delta) < material_threshold:
                continue
            validation_candidate = validation[(state_index, candidate)]["total_cost"]
            validation_delta = (validation_base - validation_candidate) / validation_base
            material_pairs += 1
            sign_matches += int(discovery_delta * validation_delta > 0.0)
    state_count = len(metadata)
    return {
        "exact_best_action_agreement": float(
            np.mean(
                [
                    discovery_choices[index] == validation_oracle[index]
                    for index in sorted(metadata)
                ]
            )
        ),
        "discovery_material_selected_states": selected_material,
        "discovery_material_selected_state_fraction": (
            selected_material / state_count if state_count else 0.0
        ),
        "selected_validation_wins": selected_validation_wins,
        "selected_validation_win_fraction": (
            selected_validation_wins / selected_material
            if selected_material
            else 0.0
        ),
        "discovery_material_candidate_pairs": material_pairs,
        "pairwise_material_sign_matches": sign_matches,
        "pairwise_material_sign_agreement": (
            sign_matches / material_pairs if material_pairs else 0.0
        ),
    }


def mechanics_summary(
    rows: list[dict[str, Any]], config: dict[str, Any], num_facilities: int
) -> dict[str, Any]:
    unique: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        unique.setdefault((int(row["state_index"]), str(row["candidate"])), row)
    residual_rows = [
        row for row in unique.values() if row["candidate"] != BASELINE_CANDIDATE
    ]
    epsilon = float(config["mechanics"]["minimum_realized_transfer_units"])
    noncollapsed = [
        row for row in residual_rows if float(row["realized_transfer_units"]) > epsilon
    ]
    covered = {
        (int(row["facility"]), str(row["direction"])) for row in noncollapsed
    }
    required = {
        (facility, direction)
        for facility in range(int(num_facilities))
        for direction in ("decrease", "increase")
    }
    fraction = len(noncollapsed) / len(residual_rows) if residual_rows else 0.0
    maximum_budget_error = max(
        (abs(float(row["budget_error"])) for row in residual_rows), default=0.0
    )
    passes = bool(
        fraction
        >= float(config["mechanics"]["minimum_noncollapsed_candidate_fraction"])
        and covered == required
        and maximum_budget_error
        <= float(config["mechanics"]["maximum_budget_error"])
    )
    return {
        "candidate_state_count": len(residual_rows),
        "noncollapsed_candidate_state_count": len(noncollapsed),
        "noncollapsed_candidate_fraction": fraction,
        "covered_facility_directions": len(covered),
        "required_facility_directions": len(required),
        "maximum_budget_error": maximum_budget_error,
        "passes": passes,
    }


def evaluate_rows(
    discovery_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    candidates: tuple[dict[str, Any], ...],
    config: dict[str, Any],
    *,
    num_facilities: int,
) -> dict[str, Any]:
    gate = config["gate"]
    material = float(gate["material_relative_saving"])
    discovery_choices = select_by_state(discovery_rows, candidates)
    validation_oracle = select_by_state(validation_rows, candidates)
    optimistic = comparison_summary(
        validation_rows, validation_oracle, material_threshold=material
    )
    prospective = comparison_summary(
        validation_rows, discovery_choices, material_threshold=material
    )
    stability = label_stability(
        discovery_rows,
        validation_rows,
        discovery_choices,
        validation_oracle,
        material_threshold=material,
    )
    mechanics = mechanics_summary(discovery_rows, config, num_facilities)
    optimistic_pass = bool(
        optimistic["relative_saving"]
        >= float(gate["minimum_optimistic_relative_saving"])
        and optimistic["material_state_fraction"]
        >= float(gate["minimum_material_state_fraction"])
        and (
            not bool(gate["require_clinical_noninferiority"])
            or optimistic["clinical_noninferior"]
        )
    )
    prospective_pass = bool(
        prospective["relative_saving"]
        >= float(gate["minimum_prospective_relative_saving"])
        and prospective["material_state_fraction"]
        >= float(gate["minimum_material_state_fraction"])
        and (
            not bool(gate["require_positive_every_scenario"])
            or prospective["positive_every_scenario"]
        )
        and (
            not bool(gate["require_clinical_noninferiority"])
            or prospective["clinical_noninferior"]
        )
        and stability["selected_validation_win_fraction"]
        >= float(gate["minimum_selected_validation_win_fraction"])
        and stability["pairwise_material_sign_agreement"]
        >= float(gate["minimum_pairwise_material_sign_agreement"])
    )
    passes = bool(mechanics["passes"] and optimistic_pass and prospective_pass)
    return {
        "mechanics": mechanics,
        "optimistic_validation_oracle": optimistic,
        "prospective_validation": prospective,
        "label_stability": stability,
        "optimistic_gate_passes": optimistic_pass,
        "prospective_gate_passes": prospective_pass,
        "passes": passes,
        "decision": (
            "budget_neutral_residual_support_established"
            if passes
            else "budget_neutral_residual_support_failed"
        ),
        "observable_ranking_screen_authorized": passes,
        "ddpg_training_authorized": False,
    }


def write_csv(
    rows: list[dict[str, Any]], path: Path, *, append: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        if not append:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def update_status(
    output_root: Path, record: dict[str, Any], **changes: Any
) -> None:
    record.update(changes)
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(output_root / "status.json", record)


def describe(config: dict[str, Any]) -> dict[str, Any]:
    source = validate_config(config)
    base_env = load_json(Path(source["env_config"]))
    num_facilities = int(base_env["num_facilities"])
    candidates = candidate_specs(config, num_facilities)
    state_count = (
        len(source["scenarios"])
        * int(config["generation_seeds"]["count"])
        * len(config["decision_epochs"])
    )
    approvals = execution_approvals(config)
    try:
        require_execution_authorization(config)
        authorized = True
    except PermissionError:
        authorized = False
    return {
        "name": config["name"],
        "execution_approvals": approvals,
        "execution_authorized": authorized,
        "authorization_mode": config.get("execution_exception", {}).get(
            "mode", "dual_named_approval"
        ),
        "candidate_count": len(candidates),
        "state_count": state_count,
        "expected_discovery_rows": (
            state_count
            * len(candidates)
            * int(config["discovery_worlds"]["count_per_state"])
        ),
        "expected_validation_rows": (
            state_count
            * len(candidates)
            * int(config["validation_worlds"]["count_per_state"])
        ),
        "policy_training_performed": False,
    }


def run(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    source = validate_config(config)
    require_execution_authorization(config)
    output_root = Path(config["output_root"])
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite residual-screen output root: {output_root}"
        ) from None
    status = {
        "status": "running",
        "phase": "preparing_states",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "exit_code": None,
        "persisted_rows": 0,
        "execution_approvals": execution_approvals(config),
        "authorization_mode": config.get("execution_exception", {}).get(
            "mode", "dual_named_approval"
        ),
        "policy_training_performed": False,
        "formal_confirmation_performed": False,
    }
    started = time.monotonic()
    update_status(output_root, status)
    try:
        claim = {
            **status,
            "cwd": str(Path.cwd()),
            "executable": sys.executable,
            "argv": sys.argv,
            "execution_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, timeout=10
            ).strip(),
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "execution_exception": config.get("execution_exception"),
        }
        write_json_atomic(output_root / "claim.json", claim)
        result = _execute_screen(
            config, config_path=config_path, source=source,
            output_root=output_root, status=status,
        )
    except BaseException as exc:
        update_status(
            output_root, status, status="failed", exit_code=1,
            error_type=type(exc).__name__, error=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )
        raise
    update_status(
        output_root, status, status="completed", phase="completed", exit_code=0,
        decision=result["decision"], elapsed_seconds=time.monotonic() - started,
    )
    return result


def _execute_screen(
    config: dict[str, Any], *, config_path: Path, source: dict[str, Any],
    output_root: Path, status: dict[str, Any],
) -> dict[str, Any]:
    base_env = load_json(Path(source["env_config"]))
    num_facilities = int(base_env["num_facilities"])
    candidates = candidate_specs(config, num_facilities)
    states, manifest = generate_states(config, source)
    expected = describe(config)
    if len(states) != int(expected["state_count"]):
        raise RuntimeError("state generation count mismatch")
    rows_path = output_root / "residual_headroom_rows.csv"
    manifest_path = output_root / "state_manifest.json"
    write_json_atomic(manifest_path, {"states": manifest})
    update_status(output_root, status, state_count=len(states))
    print(json.dumps({"phase": "states", "completed": len(states)}), flush=True)

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
                        baseline_spec=config["baseline_policy"],
                        continuation_spec=config["continuation_policy"],
                        world_seed=world_seed,
                        phase=phase,
                        lookahead=int(config["lookahead"]),
                    )
                    if any(
                        not math.isfinite(value)
                        for value in row.values()
                        if isinstance(value, (int, float))
                    ):
                        raise RuntimeError("non-finite persisted screen metric")
                    rows.append(row)
                    world_rows.append(row)
                if len({row["rng_sha256"] for row in world_rows}) != 1:
                    raise RuntimeError("candidate arms did not preserve exact RNG use")
                write_csv(world_rows, rows_path, append=rows_path.exists())
                update_status(
                    output_root, status, phase=phase,
                    current_state=int(state["state_index"]),
                    current_scenario=str(state["scenario"]),
                    replication=replication + 1, persisted_rows=len(rows),
                )
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "completed_states": int(state["state_index"]) + 1,
                    }
                ),
                flush=True,
            )

    discovery_rows = [row for row in rows if row["phase"] == "discovery"]
    validation_rows = [row for row in rows if row["phase"] == "validation"]
    if len(discovery_rows) != int(expected["expected_discovery_rows"]):
        raise RuntimeError("discovery row count mismatch")
    if len(validation_rows) != int(expected["expected_validation_rows"]):
        raise RuntimeError("validation row count mismatch")
    result = evaluate_rows(
        discovery_rows,
        validation_rows,
        candidates,
        config,
        num_facilities=num_facilities,
    )

    summary_path = output_root / "summary.json"
    inventory_path = output_root / "artifact_inventory.json"
    summary = {
        "name": config["name"],
        "experimental_role": config["experimental_role"],
        "policy_training_performed": False,
        "formal_confirmation_performed": False,
        "candidate_count": len(candidates),
        "state_count": len(states),
        "discovery_row_count": len(discovery_rows),
        "validation_row_count": len(validation_rows),
        **result,
        "observable_ranking_screen_eligible": result["passes"],
        "observable_ranking_screen_authorized": False,
        "execution_approvals": execution_approvals(config),
        "execution_exception": config.get("execution_exception"),
        "collaborator_review_pending": not execution_approvals(config)["howard"],
        "provenance": {
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "protocol_path": str(config["protocol"]),
            "protocol_sha256": sha256(Path(config["protocol"])),
            "parent_j2_summary": str(config["parent_j2_summary"]),
            "parent_j2_summary_sha256": sha256(Path(config["parent_j2_summary"])),
            "rows_sha256": sha256(rows_path),
            "state_manifest_sha256": sha256(manifest_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_paths = (
        Path("evaluation/screen_intertemporal_residual_allocation_headroom.py"),
        Path("evaluation/screen_intertemporal_shared_capacity_states.py"),
        Path("evaluation/screen_intertemporal_shared_capacity_episodes.py"),
        Path("src/env/capacity_planning.py"),
        Path("src/env/patient_capacity_planning.py"),
        Path("src/baselines/heuristics.py"),
    )
    inventory = {
        "files": [
            {"path": str(config_path), "sha256": sha256(config_path)},
            {"path": str(config["protocol"]), "sha256": sha256(Path(config["protocol"]))},
            {"path": str(config["source_scenario_config"]), "sha256": sha256(Path(config["source_scenario_config"]))},
            {"path": str(source["env_config"]), "sha256": sha256(Path(source["env_config"]))},
            {"path": str(config["parent_j2_summary"]), "sha256": sha256(Path(config["parent_j2_summary"]))},
            *[
                {"path": str(path), "sha256": sha256(path)}
                for path in source_paths
            ],
            {"path": str(rows_path), "sha256": sha256(rows_path)},
            {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            {"path": str(summary_path), "sha256": sha256(summary_path)},
            {"path": str(output_root / "claim.json"), "sha256": sha256(output_root / "claim.json")},
        ]
    }
    if "execution_exception" in config:
        for key in ("record", "original_config"):
            path = Path(config["execution_exception"][key])
            inventory["files"].append({"path": str(path), "sha256": sha256(path)})
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--describe",
        action="store_true",
        help="validate and print the frozen contract without consuming seeds",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_json(config_path)
    if args.describe:
        print(json.dumps(describe(config), indent=2, sort_keys=True))
        return
    run(config, config_path=config_path)


if __name__ == "__main__":
    main()
