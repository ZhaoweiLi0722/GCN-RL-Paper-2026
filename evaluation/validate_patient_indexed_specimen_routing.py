"""Deterministic mechanics/headroom gate for patient-indexed routing.

This gate does not train a policy. It proves that a concrete patient lot can be
routed legally and that the shared MDL-2 executor can exploit routing headroom
in a preregistered regional bottleneck.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.heuristics import get_heuristic_class
from src.rl.config import load_config
from src.rl.experiment import EpisodeMetrics, build_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = validate_mechanics(load_config(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))


def validate_mechanics(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 7_100_000))
    max_steps = int(config.get("max_steps", 6))
    routing_env_config = deepcopy(dict(config["env"]))
    if not bool(routing_env_config.get("enable_specimen_routing", False)):
        raise ValueError("Mechanics gate requires routing to be enabled")
    control_env_config = deepcopy(routing_env_config)
    control_env_config["enable_specimen_routing"] = False
    requirements = dict(config.get("requirements", {}))

    forced = build_env({"env": routing_env_config}, seed=seed)
    forced.reset(seed=seed)
    donor = int(config.get("forced_route", {}).get("donor", 0))
    receiver = int(config.get("forced_route", {}).get("receiver", 1))
    lots = int(config.get("forced_route", {}).get("lots", 1))
    action = forced.noop_action()
    normalized_lots = float(lots) / float(forced.config.max_specimen_transfer)
    action[donor] = -normalized_lots
    action[receiver] = normalized_lots
    _state, _reward, _done, forced_info = forced.step(action)
    forced_events = [dict(event) for event in forced_info["specimen_route_events"]]
    forced.assert_identity_conservation()

    routing_metrics = _run_mdl2(
        routing_env_config,
        seed=seed,
        max_steps=max_steps,
    )
    control_metrics = _run_mdl2(
        control_env_config,
        seed=seed,
        max_steps=max_steps,
    )
    minimum_routes = float(requirements.get("minimum_routes", 1.0))
    minimum_completion_gain = float(
        requirements.get("minimum_completion_gain", 1.0)
    )
    checks = {
        "forced_route_nonzero": len(forced_events) >= lots,
        "forced_route_qualified": all(
            event["origin_facility"] == donor
            and event["destination_facility"] == receiver
            for event in forced_events
        ),
        "forced_route_identity_unique": len(
            {event["patient_id"] for event in forced_events}
        )
        == len(forced_events),
        "mdl2_route_nonzero": (
            routing_metrics["specimen_route_count"] >= minimum_routes
        ),
        "mdl2_completion_headroom": (
            routing_metrics["patients_completed"]
            - control_metrics["patients_completed"]
            >= minimum_completion_gain
        ),
        "paired_demand_equal": (
            routing_metrics["demand_trace_sha256"]
            == control_metrics["demand_trace_sha256"]
        ),
    }
    payload = {
        "name": str(config.get("name", "patient_indexed_specimen_routing_mechanics")),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "seed": seed,
        "max_steps": max_steps,
        "checks": checks,
        "forced_route_events": forced_events,
        "routing_mdl2": routing_metrics,
        "no_routing_mdl2": control_metrics,
        "return_assumption": routing_env_config.get(
            "finished_product_return_assumption",
            "return_to_collection_facility",
        ),
        "clinical_calibration_claim": False,
    }
    if payload["status"] != "PASS":
        raise RuntimeError(json.dumps(payload, sort_keys=True))

    output = Path(config["output_path"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite mechanics report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def _run_mdl2(
    env_config: dict[str, Any],
    *,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    env = build_env({"env": deepcopy(env_config)}, seed=seed)
    policy = get_heuristic_class("mdl2")(
        state_dim=env.observation_size,
        action_dim=env.action_size,
        config={},
    )
    state = env.reset(seed=seed)
    policy.reset()
    metrics = EpisodeMetrics()
    demand_trace: list[list[float]] = []
    for _step in range(max_steps):
        action = policy.select_action(state, explore=False, env=env)
        state, _reward, done, info = env.step(action)
        metrics.update(info)
        demand_trace.append(np.asarray(info["demand"], dtype=float).tolist())
        env.assert_identity_conservation()
        if done:
            break
    demand_payload = json.dumps(
        demand_trace,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "total_cost": float(metrics.total_cost),
        "patients_completed": float(metrics.patients_completed),
        "patients_lost": float(metrics.patients_lost),
        "specimen_route_count": float(metrics.specimen_route_count),
        "specimen_route_distance_miles": float(
            metrics.specimen_route_distance_miles
        ),
        "specimen_route_time_hours": float(metrics.specimen_route_time_hours),
        "specimen_route_cost": float(metrics.specimen_route_cost),
        "blocked_specimen_requests": float(metrics.blocked_specimen_requests),
        "transferred_patient_ids": list(metrics.transferred_patient_ids),
        "demand_trace_sha256": hashlib.sha256(demand_payload).hexdigest(),
    }


if __name__ == "__main__":
    main()
