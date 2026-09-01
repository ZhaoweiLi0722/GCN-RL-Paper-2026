"""Run the engineering smoke for intertemporal shared capacity.

This command validates full-scale environment mechanics only. It trains no
policy and makes no scientific gate decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.heuristics import get_heuristic_class
from src.rl.experiment import build_env


DEFAULT_CONFIG = Path(
    "experiments/configs/intertemporal_shared_capacity_mechanics_smoke.json"
)
FORBIDDEN_SEED = 91_100_000


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


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merged_env_config(config: dict[str, Any]) -> dict[str, Any]:
    env = load_config(Path(config["env_config"]))
    env.update(dict(config["env_overrides"]))
    return env


def validate_config(config: dict[str, Any]) -> None:
    if int(config["seed"]) == FORBIDDEN_SEED:
        raise ValueError("formal holdout seed is forbidden")
    if int(config["steps"]) < 3:
        raise ValueError("mechanics smoke requires at least three steps")
    overrides = dict(config["env_overrides"])
    for key in (
        "enable_scheduled_referral_waves",
        "enable_overtime_control",
        "enable_intertemporal_overtime_commitment",
    ):
        if not bool(overrides.get(key)):
            raise ValueError(f"mechanics smoke requires {key}")


def run_policy(
    *,
    config: dict[str, Any],
    env_config: dict[str, Any],
    policy_spec: dict[str, Any],
) -> dict[str, Any]:
    env = build_env({"env": env_config}, seed=int(config["seed"]))
    algorithm = str(policy_spec["algorithm"])
    policy = get_heuristic_class(algorithm)(config=dict(policy_spec.get("config", {})))
    total_cost = 0.0
    max_requested = 0.0
    max_active = 0.0
    active_steps = 0
    schedule_blocks: set[tuple[float, ...]] = set()
    budget = float(env._shared_overtime_budget())

    for _ in range(int(config["steps"])):
        state = env.observation()
        graph = env.graph_observation()
        if not np.isfinite(state).all() or not np.isfinite(graph["node_features"]).all():
            raise RuntimeError("non-finite observation in mechanics smoke")
        action = policy.select_action(state, env=env)
        _, reward, done, info = env.step(action)
        if not np.isfinite(float(reward)) or not np.isfinite(float(info["cost"])):
            raise RuntimeError("non-finite reward or cost in mechanics smoke")
        requested = np.asarray(info["overtime_requested_surge"], dtype=float)
        active = np.asarray(info["overtime_active_capacity"], dtype=float)
        if float(requested.sum()) > budget + 1e-9:
            raise RuntimeError("shared overtime budget exceeded")
        max_requested = max(max_requested, float(requested.sum()))
        max_active = max(max_active, float(active.sum()))
        active_steps += int(float(active.sum()) > 0.0)
        total_cost += float(info["cost"])
        schedule_blocks.add(
            tuple(np.asarray(info["scheduled_referral_multiplier"], dtype=float))
        )
        if done:
            break

    return {
        "algorithm": algorithm,
        "steps": int(env.t),
        "total_cost": total_cost,
        "max_requested_capacity": max_requested,
        "max_active_capacity": max_active,
        "shared_budget": budget,
        "active_steps": active_steps,
        "distinct_schedule_blocks": len(schedule_blocks),
        "final_rng_sha256": rng_digest(env),
    }


def run(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    validate_config(config)
    env_config = merged_env_config(config)
    policy_rows = [
        run_policy(config=config, env_config=env_config, policy_spec=policy_spec)
        for policy_spec in config["policies"]
    ]
    rng_digests = {row["final_rng_sha256"] for row in policy_rows}
    if len(rng_digests) != 1:
        raise RuntimeError("policy arms did not preserve action-independent RNG use")
    summary = {
        "name": config["name"],
        "experimental_role": config["experimental_role"],
        "policy_training_performed": False,
        "scientific_gate_decision": None,
        "seed": int(config["seed"]),
        "policies": policy_rows,
        "rng_end_state_match": True,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "source_sha256": sha256(
            Path("evaluation/smoke_intertemporal_shared_capacity.py")
        ),
    }
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.output:
        config["output"] = args.output
    run(config, config_path=config_path)


if __name__ == "__main__":
    main()
