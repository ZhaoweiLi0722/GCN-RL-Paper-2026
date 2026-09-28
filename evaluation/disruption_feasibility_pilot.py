"""Small baseline-only synthetic pilot, separate from all locked campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.run_full_benchmark import load_benchmark_plan, make_scenario_env_config
from src.baselines.heuristics import get_heuristic_class
from src.env.development_disruption import DevelopmentDisruptionEnv, DisruptionProfile
from src.rl.experiment import EpisodeMetrics, build_env


CELLS = ("capacity_nochange", "capacity_outage", "lead_nochange", "lead_shift")
METRICS = ("total_cost", "completion_service_level", "patient_ineligibility_during_manufacturing_rate", "patients_lost")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_spec(spec):
    discovery, validation = spec["discovery_seeds"], spec["validation_seeds"]
    if not discovery or not validation or len(set(discovery + validation)) != len(discovery + validation):
        raise ValueError("nonempty, unique, disjoint discovery and validation seeds required")
    names = [p["name"] for p in spec["policies"]]
    if len(set(names)) != len(names) or "mdl2" not in names:
        raise ValueError("unique policy names and mdl2 baseline required")
    if any(p["algorithm"] not in ("mdl2", "mdl2_lt") for p in spec["policies"]):
        raise ValueError("only declared heuristic policies are permitted")


def compose(spec):
    plan = load_benchmark_plan(spec["benchmark_plan"])
    scenario = next(s for s in plan["scenarios"] if s["name"] == spec["source_scenario"])
    config = make_scenario_env_config(plan, "mdl2", scenario)
    config.update(spec["env_overrides"])
    return config, scenario["env_config"]


def make_env(config, spec, cell, seed):
    base = build_env({"env": config}, seed=seed)
    profile = DisruptionProfile(
        change_step=spec["change_step"],
        blocked_capacity_fraction=tuple(spec["capacity_fraction"] if cell == "capacity_outage" else [0] * base.config.num_facilities),
        shifted_lead_probabilities=tuple(spec["shifted_lead_probabilities"] if cell == "lead_shift" else base.config.reagent_lead_time_probabilities),
    )
    return DevelopmentDisruptionEnv(base.env_config, profile, seed=seed)


def run_episode(env, policy_config, key, log):
    policy = get_heuristic_class(policy_config["algorithm"])(config=policy_config["settings"])
    state = env.reset(seed=key["world_seed"])
    policy.reset()
    metrics = EpisodeMetrics()
    arrival_hash = hashlib.sha256()
    total_capacity = float(env.initial_idle_bioreactors.sum())
    max_capacity_error = 0.0
    max_cost_error = 0.0
    for step in range(env.config.episode_horizon):
        demand = env.demand.copy()
        arrival_hash.update(demand.astype("<f8").tobytes())
        action = policy.select_action(state, explore=False, env=env)
        state, reward, done, info = env.step(action)
        env.assert_identity_conservation()
        stock = float(env.bioreactors.sum() + env.blocked_idle.sum() + env.capacity_transfer_pipeline.sum())
        capacity_error = abs(stock - total_capacity)
        max_capacity_error = max(max_capacity_error, capacity_error)
        if capacity_error > 1e-7:
            raise RuntimeError(f"nonconserved reactor stock: {key}, {step}, {capacity_error}")
        cost_error = abs(float(info["cost"]) + float(reward))
        max_cost_error = max(max_cost_error, cost_error)
        if not np.isfinite(state).all() or not math.isfinite(reward) or cost_error > 1e-7:
            raise RuntimeError("invalid state or reward/cost accounting")
        metrics.update(info)
        record = dict(key, step=step, cost=float(info["cost"]), demand=demand.tolist(),
                      action=np.asarray(action).tolist(),
                      procurement_leads=np.asarray(info["procurement_leads"]).tolist(),
                      replenishment=np.asarray(info["replenishment"]).tolist(),
                      blocked_capacity=env.blocked_idle.tolist(), reactor_stock=stock,
                      waiting=float(env.specimens.sum()),
                      on_order=float(env.reagent_purchase_pipeline.sum()))
        log.write(json.dumps(record, allow_nan=False) + "\n")
        if done:
            break
    row = dict(key, steps=step + 1, total_cost=metrics.total_cost,
               **metrics.patient_row(), **metrics.routing_row(), **metrics.cost_components)
    row.update(
        terminal_waiting=float(env.specimens.sum()),
        terminal_in_production=sum(len(stage) for facility in env.in_production_patients for stage in facility),
        terminal_specimen_in_transit=len(env.specimen_transits),
        terminal_product_in_transit=len(env.product_return_transits),
        terminal_procurement_on_order=float(env.reagent_purchase_pipeline.sum()),
        terminal_reagents=float(env.reagents.sum()),
        terminal_blocked_capacity=float(env.blocked_idle.sum()),
        arrival_sha256=arrival_hash.hexdigest(),
        rng_sha256=hashlib.sha256(json.dumps(env.rng.bit_generator.state, sort_keys=True).encode()).hexdigest(),
        maximum_capacity_error=max_capacity_error, maximum_reward_cost_error=max_cost_error,
    )
    if any(not math.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
        raise RuntimeError("nonfinite persisted episode metric")
    return row


def summarize(rows, spec):
    by_key = {(r["cell"], r["split"], r["world_seed"], r["policy"]): r for r in rows}
    expected = {(c, phase, s, p["name"]) for c in CELLS for phase in ("discovery", "validation")
                for s in spec[phase + "_seeds"] for p in spec["policies"]}
    if len(by_key) != len(rows) or set(by_key) != expected:
        raise ValueError("missing or duplicate episode rows")
    fingerprints = defaultdict(set)
    for r in rows:
        fingerprints[r["world_seed"]].add((r["arrival_sha256"], r["rng_sha256"]))
    if any(len(v) != 1 for v in fingerprints.values()):
        raise ValueError("world pairing failed across cells or policies")
    for phase in ("discovery", "validation"):
        for seed in spec[phase + "_seeds"]:
            for p in spec["policies"]:
                a, b = (by_key[(c, phase, seed, p["name"])] for c in ("capacity_nochange", "lead_nochange"))
                if any(a[k] != b[k] for k in a if k != "cell"):
                    raise ValueError("identical no-change controls disagree")
    cells = {}
    for cell in CELLS:
        discovery_means = {p["name"]: float(np.mean([by_key[(cell, "discovery", s, p["name"])]["total_cost"]
                                                   for s in spec["discovery_seeds"]])) for p in spec["policies"]}
        selected = min(discovery_means, key=discovery_means.get)
        comparisons = {}
        for p in spec["policies"]:
            pairs = [(by_key[(cell, "validation", s, p["name"])], by_key[(cell, "validation", s, "mdl2")])
                     for s in spec["validation_seeds"]]
            differences = {m: float(np.mean([a[m] - b[m] for a, b in pairs])) for m in METRICS}
            differences["cost_change_pct"] = 100 * differences["total_cost"] / float(np.mean([b["total_cost"] for _, b in pairs]))
            differences["lower_cost_worlds"] = sum(a["total_cost"] < b["total_cost"] for a, b in pairs)
            comparisons[p["name"]] = differences
        cells[cell] = {"discovery_selected": selected, "discovery_mean_cost": discovery_means,
                       "validation_differences_vs_mdl2": comparisons}
    return {"status": "completed", "scope": "synthetic_baseline_feasibility_only", "online_rl_evidence": False,
            "clinical_noninferiority_established": False, "training_authorized": False,
            "episodes": len(rows), "steps": sum(r["steps"] for r in rows),
            "paired_arrivals_and_rng_verified": True, "identical_controls_verified": True,
            "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/disruption_feasibility_20260928.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-output", default="reports/2026-09-28-disruption-feasibility/smoke")
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    validate_spec(spec)
    config, source = compose(spec)
    if args.smoke:
        spec.update(discovery_seeds=[123], validation_seeds=[124], change_step=2)
        config["episode_horizon"] = 6
    output = Path(spec["output_root"] if not args.smoke else args.smoke_output)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if dirty and not args.smoke:
        raise RuntimeError("commit source/config before the prespecified pilot")
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "execution_commit": revision, "smoke_only": args.smoke,
              "start_unix": time.time(), "python": platform.python_version(), "numpy": np.__version__}
    write_json(output / "status.json", status)
    source_paths = [args.config, spec["benchmark_plan"], source, "evaluation/disruption_feasibility_pilot.py",
                    "src/env/development_disruption.py", "src/env/patient_capacity_planning.py",
                    "src/env/capacity_planning.py", "src/baselines/heuristics.py", "src/rl/experiment.py"]
    source_hashes = {p: digest(p) for p in source_paths}
    write_json(output / "execution.json", {"spec": spec, "composed_env": config, "source_sha256": source_hashes})
    rows = []
    try:
        with (output / "steps.jsonl").open("x") as log, (output / "episodes.jsonl").open("x") as episodes:
            for phase in ("discovery", "validation"):
                for cell in CELLS:
                    for seed in spec[phase + "_seeds"]:
                        for p in spec["policies"]:
                            key = {"cell": cell, "split": phase, "world_seed": seed, "policy": p["name"]}
                            row = run_episode(make_env(config, spec, cell, seed), p, key, log)
                            rows.append(row)
                            episodes.write(json.dumps(row, allow_nan=False) + "\n")
                            episodes.flush()
                    print(f"{phase} {cell}: {len(rows)} episodes saved", flush=True)
        summary = summarize(rows, spec)
        if source_hashes != {p: digest(p) for p in source_paths}:
            raise RuntimeError("source changed during run")
        write_json(output / "summary.json", summary)
        status.update(status="completed", exit_code=0, end_unix=time.time())
    except Exception as error:
        status.update(status="failed", exit_code=1, error=repr(error), end_unix=time.time())
        write_json(output / "status.json", status)
        raise
    write_json(output / "status.json", status)
    write_json(output / "inventory.json", {"source_sha256": source_hashes,
               "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
