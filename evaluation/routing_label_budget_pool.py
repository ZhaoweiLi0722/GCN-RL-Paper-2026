"""Routing label budget study: collect one complete paired-CRN outcome pool.

Stage G1 closed the online-DDPG extension on the finding that routing's
counterfactual best-action labels do not replicate -- 54.5% agreement against
a 70% gate -- but it estimated those labels from only 3 discovery and 5
validation worlds per action. Best-action agreement in this simulator is
budget-sensitive, and CRN pairing here is exact, so disagreement between
streams is genuine world-to-world variation that more worlds legitimately
average away. Was 54.5% a property of the channel, or of an eight-world budget?

This module collects the raw material for that question and nothing else: a
uniform, complete pool of every legal specimen action evaluated on every world,
for every surviving decision state. Because the pool is complete, every budget
and allocation question is answered afterwards by replaying policies against it
-- no adaptive sampling, no contamination, and uniform and sequential-halving
allocation scored on identical data.

Trains nothing, selects nothing, and touches no protected stream.
See specs/2026-08-29-routing-label-budget-study/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    make_scenario_env_config,
    select_scenarios,
)
from src.baselines.heuristics import get_heuristic_class
from src.rl.experiment import build_env

DEFAULT_CONFIG = Path("experiments/configs/routing_label_budget_study.json")

# Streams this study must never touch.
FORMAL_HOLDOUT_SEED = 91_100_000
DEV_CRN_SEEDS = (94_000_000, 94_100_000)
G1_SEED_FAMILIES = (101_000_000, 103_000_000, 107_000_000)
RESERVED_SCENARIO = "routing_regional_drift"
RESERVED_SEED_RANGES = ((97_100_000, 97_100_009), (97_200_000, 97_200_004), (97_300_000, 97_300_009))


def validate_config(config: dict[str, Any]) -> None:
    for key in (
        "name", "experimental_role", "plan", "anchor_algorithm", "scenarios",
        "env_overrides", "state_generation", "action_family", "worlds",
        "analysis", "reading_rule", "output_root",
    ):
        if key not in config:
            raise ValueError(f"config missing required key: {key}")

    overrides = config["env_overrides"]
    if not overrides.get("enable_specimen_routing", False):
        raise ValueError("this is a routing study; enable_specimen_routing must be true")
    if overrides.get("enable_overtime_control", False):
        raise ValueError("overtime must be off in the routing study")
    if overrides.get("enable_production_throttle", False):
        raise ValueError("the production throttle is dormant and may not be enabled")

    if RESERVED_SCENARIO in config["scenarios"]:
        raise ValueError(
            f"scenario {RESERVED_SCENARIO!r} is reserved for the overtime confirmatory run"
        )

    generation = config["state_generation"]
    worlds = config["worlds"]
    seeds = (
        [int(s) for s in generation["seeds"]]
        + [int(s) for s in generation.get("fallback_seeds", [])]
        + [int(worlds["seed_start"]) + i for i in range(int(worlds["count"]))]
        + [int(config["analysis"]["split_seed"])]
    )
    if len(set(seeds)) != len(seeds):
        raise ValueError("state, world, and analysis seeds must be disjoint")
    for seed in seeds:
        if seed == FORMAL_HOLDOUT_SEED or seed in DEV_CRN_SEEDS:
            raise ValueError(f"seed {seed} collides with a protected stream")
        if any(low <= seed <= high for low, high in RESERVED_SEED_RANGES):
            raise ValueError(f"seed {seed} is reserved for the overtime confirmatory run")
        if any(family <= seed < family + 1_000_000 for family in G1_SEED_FAMILIES):
            raise ValueError(f"seed {seed} collides with a Stage G1 seed family")

    shifts = [float(s) for s in config["action_family"]["shifts"]]
    if 0.0 not in shifts:
        raise ValueError("the anchor (shift 0.0) must be one of the arms")
    if len(set(shifts)) != len(shifts):
        raise ValueError("action shifts must be distinct")


def scenario_env_dict(plan, config, scenario) -> dict[str, Any]:
    env = dict(make_scenario_env_config(plan, str(config["anchor_algorithm"]), scenario))
    env.update(config["env_overrides"])
    env["scenario_name"] = scenario["name"]
    return env


def build_scenario_env(env_dict: dict[str, Any], seed: int):
    env = build_env({"env": dict(env_dict)}, seed)
    if getattr(env, "scenario_name", None) != env_dict["scenario_name"]:
        raise RuntimeError("scenario provenance mismatch at construction")
    if not env.env_config.enable_specimen_routing:
        raise RuntimeError("built environment lost specimen routing")
    return env


def centred_pressure(env) -> np.ndarray:
    """Who should give up specimens and who should receive them.

    A uniform shift cannot route anything: specimen flow is conserved, so
    moving every clinic the same way leaves no counterparties. Centring the
    pattern creates both sides of the trade.
    """

    waiting = np.asarray(env.waiting_counts(), dtype=float)
    capacity = np.minimum(
        np.asarray(env.bioreactors[:, 0], dtype=float),
        np.asarray(env.reagents, dtype=float),
    )
    pattern = waiting - capacity
    pattern = pattern - pattern.mean()
    scale = np.abs(pattern).max()
    return pattern / scale if scale > 0 else pattern


def anchor_action(policy, env) -> np.ndarray:
    return np.asarray(policy.select_action(env.observation(), env=env), dtype=np.float32)


def arm_action(policy, env, shift: float, pattern: np.ndarray) -> np.ndarray:
    action = anchor_action(policy, env).copy()
    if shift != 0.0:
        n = env.config.num_facilities
        action[:n] = np.clip(action[:n] + shift * pattern, -1.0, 1.0)
    return action


def executed_signature(info) -> tuple:
    """Executed flows, never requested ones.

    The probe that motivated the spec amendment showed requested integer net
    can differ across arms while nothing actually moves; filtering on requests
    would admit degenerate states.
    """

    return (
        tuple(np.asarray(info["specimen_transfers"], dtype=int)),
        int(round(float(info["specimen_route_count"]))),
    )


def collect_states(env_dict, policy, state_seed: int, epochs: list[int], shifts: list[float]):
    """Snapshot decision states and keep only those with distinct executions."""

    env = build_scenario_env(env_dict, state_seed)
    env.reset(seed=state_seed)
    kept, dropped = [], []
    for epoch in sorted(epochs):
        while env.t < epoch:
            env.step(anchor_action(policy, env))
        snapshot = env.state_dict()
        pattern = centred_pressure(env)
        signatures, actions = [], {}
        for shift in shifts:
            env.load_state_dict(snapshot)
            env.rng = np.random.default_rng(int(state_seed) + 1)
            action = arm_action(policy, env, shift, pattern)
            actions[shift] = action
            _, _, _, info = env.step(action)
            signatures.append(executed_signature(info))
        env.load_state_dict(snapshot)
        state_id = f"{env_dict['scenario_name']}:seed{state_seed}:epoch{epoch}"
        record = {
            "state_id": state_id,
            "state_seed": int(state_seed),
            "epoch": int(epoch),
            "snapshot": snapshot,
            "actions": actions,
            "distinct_executions": len(set(signatures)),
        }
        (kept if len(set(signatures)) == len(shifts) else dropped).append(record)
    return kept, dropped


def counters(env) -> dict[str, float]:
    return {
        "enrolled": float(env.cumulative_enrolled),
        "served": float(env.cumulative_served),
        "lost": float(env.cumulative_lost),
        "started": float(env.cumulative_started),
        "manufacturing_lost": float(env.cumulative_manufacturing_lost),
    }


def rollout(env, policy, action) -> dict[str, float]:
    start = counters(env)
    total = 0.0
    _, reward, done, _ = env.step(action)
    total += -float(reward)
    while not done:
        _, reward, done, _ = env.step(anchor_action(policy, env))
        total += -float(reward)
    end = counters(env)
    enrolled = max(end["enrolled"] - start["enrolled"], 1.0)
    started = max(end["started"] - start["started"], 1.0)
    return {
        "remaining_cost": total,
        "completion_service_level": (end["served"] - start["served"]) / enrolled,
        "patients_lost": end["lost"] - start["lost"],
        "manufacturing_ineligibility": (end["manufacturing_lost"] - start["manufacturing_lost"]) / started,
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--limit-worlds", type=int, default=None,
                        help="smoke only: truncate the world pool")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    validate_config(config)
    if args.output_root:
        config["output_root"] = args.output_root

    plan = load_benchmark_plan(Path(config["plan"]))
    scenarios = select_scenarios(plan, config["scenarios"])
    policy = get_heuristic_class(str(config["anchor_algorithm"]))()
    shifts = [float(s) for s in config["action_family"]["shifts"]]
    generation = config["state_generation"]
    epochs = [int(e) for e in generation["decision_epochs"]]
    world_count = int(config["worlds"]["count"])
    if args.limit_worlds:
        world_count = min(world_count, args.limit_worlds)
    world_seeds = [int(config["worlds"]["seed_start"]) + i for i in range(world_count)]

    seed_list = [int(s) for s in generation["seeds"]]
    kept_all, dropped_all = [], []
    for scenario in scenarios:
        env_dict = scenario_env_dict(plan, config, scenario)
        for state_seed in seed_list:
            policy.reset()
            kept, dropped = collect_states(env_dict, policy, state_seed, epochs, shifts)
            for record in kept:
                record["env_dict"] = env_dict
            kept_all.extend(kept)
            dropped_all.extend(dropped)

    minimum = int(generation.get("minimum_surviving_states", 0))
    if len(kept_all) < minimum:
        raise SystemExit(
            f"only {len(kept_all)} states survived the distinct-execution filter, "
            f"below the prespecified minimum of {minimum}; the spec requires adding "
            f"fallback generation seeds {generation.get('fallback_seeds')} before any "
            "outcome is examined"
        )

    print(f"states kept {len(kept_all)}, dropped {len(dropped_all)} "
          f"(distinct-execution filter), worlds {world_count}")

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(kept_all, 1):
        env = build_scenario_env(record["env_dict"], record["state_seed"])
        for shift in shifts:
            action = record["actions"][shift]
            for world in world_seeds:
                env.load_state_dict(record["snapshot"])
                env.rng = np.random.default_rng(int(world))
                if getattr(env, "scenario_name", None) != record["env_dict"]["scenario_name"]:
                    raise RuntimeError("scenario provenance mismatch during rollout")
                metrics = rollout(env, policy, action)
                rows.append({
                    "scenario": record["env_dict"]["scenario_name"],
                    "state_id": record["state_id"],
                    "state_seed": record["state_seed"],
                    "epoch": record["epoch"],
                    "arm": f"shift_{shift:+.2f}",
                    "shift": shift,
                    "world_seed": world,
                    **metrics,
                })
        print(f"  [{index}/{len(kept_all)}] {record['state_id']}", flush=True)

    seen = set()
    for row in rows:
        key = (row["state_id"], row["arm"], row["world_seed"])
        if key in seen:
            raise RuntimeError(f"duplicate pool row {key}")
        seen.add(key)
        for metric in ("remaining_cost", "completion_service_level",
                       "patients_lost", "manufacturing_ineligibility"):
            if not np.isfinite(row[metric]):
                raise RuntimeError(f"non-finite {metric} at {key}")
    expected = len(kept_all) * len(shifts) * world_count
    if len(rows) != expected:
        raise RuntimeError(f"pool incomplete: {len(rows)} rows, expected {expected}")

    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    columns = ["scenario", "state_id", "state_seed", "epoch", "arm", "shift",
               "world_seed", "remaining_cost", "completion_service_level",
               "patients_lost", "manufacturing_ineligibility"]
    rows_path = root / "pool_rows.csv"
    with open(rows_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: row[c] for c in columns} for row in rows)

    summary = {
        "name": config["name"],
        "experimental_role": config["experimental_role"],
        "states_kept": len(kept_all),
        "states_dropped": len(dropped_all),
        "dropped_state_ids": [r["state_id"] for r in dropped_all],
        "arms": [f"shift_{s:+.2f}" for s in shifts],
        "world_count": world_count,
        "row_count": len(rows),
        "config_sha256": sha256_path(config_path),
        "rows_sha256": sha256_path(rows_path),
    }
    (root / "pool_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"pool rows: {rows_path} ({len(rows)} rows)")
    print(f"pool summary: {root / 'pool_summary.json'}")


if __name__ == "__main__":
    main()
