"""Stage E2 overtime headroom screen (spec 2026-08-29-continuous-overtime-control).

Evaluation-only development screen. At fixed anchor-policy decision states
reconstructed from the routing benchmark plan, it compares the MDL-2-OT anchor
against a fixed ladder of one-epoch overtime actions applied at the
capacity-bound clinic set, under disjoint discovery and validation paired-CRN
replication streams rolled to episode end.

Prospective gate (frozen in the config before the run): the screen passes when,
in at least one non-nominal scenario, at least ``state_fraction`` of states
show a validation-stream improvement of at least ``material_threshold`` modeled
units from some ladder action while remaining clinically noninferior.

It never trains, updates, or selects a model, never touches a formal holdout
stream, and asserts per-row that the built scenario matches its declaration
(the F0/G0/G1 scenario-reconstruction defect guard).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
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

DEFAULT_CONFIG = Path("experiments/configs/continuous_overtime_headroom_e2.json")
FORMAL_HOLDOUT_SEED = 91_100_000
DEV_CRN_SEEDS = (94_000_000, 94_100_000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root")
    args = parser.parse_args()

    config = load_screen_config(Path(args.config))
    if args.smoke:
        config = smoke_config(config)
    if args.output_root:
        config["output_root"] = args.output_root
    validate_config(config)
    run_screen(config)


def load_screen_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    """Shrink the screen for wiring tests; never a scientific result."""

    smoke = copy.deepcopy(config)
    smoke["name"] = f"{config['name']}_smoke"
    smoke["scenarios"] = list(config["non_nominal_scenarios"][:1])
    smoke["non_nominal_scenarios"] = list(smoke["scenarios"])
    smoke["state_generation"] = {
        "seeds": list(config["state_generation"]["seeds"][:1]),
        "decision_epochs": list(config["state_generation"]["decision_epochs"][:2]),
    }
    smoke["action_ladder"] = [config["action_ladder"][0], config["action_ladder"][-1]]
    smoke["replications"] = {
        "discovery_seeds": list(config["replications"]["discovery_seeds"][:1]),
        "validation_seeds": list(config["replications"]["validation_seeds"][:1]),
    }
    smoke["output_root"] = f"{config['output_root']}_smoke"
    smoke["smoke"] = True
    return smoke


def validate_config(config: dict[str, Any]) -> None:
    required = (
        "name",
        "experimental_role",
        "plan",
        "anchor_algorithm",
        "scenarios",
        "non_nominal_scenarios",
        "overtime_env_overrides",
        "state_generation",
        "action_ladder",
        "replications",
        "gates",
        "output_root",
    )
    for key in required:
        if key not in config:
            raise ValueError(f"E2 config missing required key: {key}")
    overrides = config["overtime_env_overrides"]
    if not bool(overrides.get("enable_overtime_control", False)):
        raise ValueError("overtime_env_overrides must enable_overtime_control")
    if bool(overrides.get("enable_production_throttle", False)):
        raise ValueError("Decision B is dormant; the screen may not enable it")
    unknown = set(config["non_nominal_scenarios"]) - set(config["scenarios"])
    if unknown:
        raise ValueError(f"non_nominal_scenarios not in scenarios: {sorted(unknown)}")
    if not config["non_nominal_scenarios"]:
        raise ValueError("at least one non-nominal scenario is required")
    ladder = [float(value) for value in config["action_ladder"]]
    if sorted(set(ladder)) != ladder or not all(0.0 <= u <= 1.0 for u in ladder):
        raise ValueError("action_ladder must be strictly increasing within [0, 1]")
    replications = config["replications"]
    discovery = [int(seed) for seed in replications["discovery_seeds"]]
    validation = [int(seed) for seed in replications["validation_seeds"]]
    state_seeds = [int(seed) for seed in config["state_generation"]["seeds"]]
    all_seeds = discovery + validation + state_seeds
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("state, discovery, and validation seeds must be disjoint")
    for seed in all_seeds:
        if seed == FORMAL_HOLDOUT_SEED or seed in DEV_CRN_SEEDS:
            raise ValueError(f"seed {seed} collides with a protected CRN stream")
    gates = config["gates"]
    if float(gates["material_threshold"]) <= 0.0:
        raise ValueError("material_threshold must be positive")
    if not 0.0 < float(gates["state_fraction"]) <= 1.0:
        raise ValueError("state_fraction must be within (0, 1]")
    epochs = [int(epoch) for epoch in config["state_generation"]["decision_epochs"]]
    if sorted(set(epochs)) != epochs or epochs[0] < 1:
        raise ValueError("decision_epochs must be strictly increasing and >= 1")


def scenario_env_dict(
    plan: dict[str, Any], config: dict[str, Any], scenario: dict[str, Any]
) -> dict[str, Any]:
    """Benchmark-plan scenario env plus the frozen overtime overrides."""

    env = make_scenario_env_config(plan, str(config["anchor_algorithm"]), scenario)
    env = dict(env)
    env.update(config["overtime_env_overrides"])
    env["scenario_name"] = scenario["name"]
    return env


def scenario_declaration_digest(env_dict: dict[str, Any]) -> str:
    canonical = json.dumps(env_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_scenario_env(env_dict: dict[str, Any], seed: int):
    env = build_env({"env": dict(env_dict)}, seed)
    if getattr(env, "scenario_name", None) != env_dict["scenario_name"]:
        raise RuntimeError(
            "scenario declaration mismatch: built "
            f"{getattr(env, 'scenario_name', None)!r}, declared "
            f"{env_dict['scenario_name']!r}"
        )
    if not env.config.enable_overtime_control:
        raise RuntimeError("built environment lost enable_overtime_control")
    return env


def capacity_bound_clinics(env) -> np.ndarray:
    """Clinics where capacity binds and reagents could feed a surge."""

    waiting = np.asarray(env.waiting_counts(), dtype=float)
    idle = np.asarray(env.bioreactors[:, 0], dtype=float)
    reagents = np.asarray(env.reagents, dtype=float)
    return (np.minimum(waiting, reagents) > idle).astype(bool)


def anchor_action(policy, env) -> np.ndarray:
    return np.asarray(policy.select_action(env.observation(), env=env), dtype=np.float32)


def ladder_action(
    policy, env, rung: float, bound_mask: np.ndarray
) -> np.ndarray:
    """Anchor action with the overtime block overridden at bound clinics."""

    action = anchor_action(policy, env).copy()
    n = env.config.num_facilities
    raw = np.float32(2.0 * rung - 1.0)
    block = action[4 * n : 5 * n]
    block[bound_mask] = raw
    action[4 * n : 5 * n] = block
    return action


def cumulative_counters(env) -> dict[str, float]:
    return {
        "enrolled": float(env.cumulative_enrolled),
        "served": float(env.cumulative_served),
        "lost": float(env.cumulative_lost),
        "started": float(env.cumulative_started),
        "manufacturing_lost": float(env.cumulative_manufacturing_lost),
    }


def rollout_remaining(env, policy, first_action: np.ndarray) -> dict[str, float]:
    """Apply one action, then roll the anchor to episode end."""

    start = cumulative_counters(env)
    total_cost = 0.0
    _, reward, done, _ = env.step(first_action)
    total_cost += -float(reward)
    while not done:
        _, reward, done, _ = env.step(anchor_action(policy, env))
        total_cost += -float(reward)
    end = cumulative_counters(env)
    enrolled = max(end["enrolled"] - start["enrolled"], 1.0)
    started = max(end["started"] - start["started"], 1.0)
    return {
        "remaining_cost": total_cost,
        "completion_service_level": (end["served"] - start["served"]) / enrolled,
        "patients_lost": end["lost"] - start["lost"],
        "manufacturing_ineligibility": (
            end["manufacturing_lost"] - start["manufacturing_lost"]
        )
        / started,
    }


def generate_states(
    env_dict: dict[str, Any],
    policy,
    state_seed: int,
    decision_epochs: list[int],
) -> list[dict[str, Any]]:
    env = build_scenario_env(env_dict, state_seed)
    env.reset(seed=state_seed)
    states: list[dict[str, Any]] = []
    targets = set(decision_epochs)
    for epoch in range(max(decision_epochs) + 1):
        if epoch in targets:
            snapshot = env.state_dict()
            bound = capacity_bound_clinics(env)
            states.append(
                {
                    "state_id": f"{env_dict['scenario_name']}"
                    f":seed{state_seed}:epoch{epoch}",
                    "state_seed": int(state_seed),
                    "decision_epoch": int(epoch),
                    "snapshot": snapshot,
                    "bound_mask": bound,
                    "bound_count": int(bound.sum()),
                }
            )
        env.step(anchor_action(policy, env))
    return states


def evaluate_state(
    env,
    policy,
    state: dict[str, Any],
    arms: list[tuple[str, float | None]],
    stream_seeds: dict[str, list[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stream, seeds in stream_seeds.items():
        for crn_seed in seeds:
            for arm_name, rung in arms:
                env.load_state_dict(state["snapshot"])
                env.rng = np.random.default_rng(int(crn_seed))
                if rung is None:
                    action = anchor_action(policy, env)
                else:
                    action = ladder_action(policy, env, rung, state["bound_mask"])
                metrics = rollout_remaining(env, policy, action)
                rows.append(
                    {
                        "state_id": state["state_id"],
                        "decision_epoch": state["decision_epoch"],
                        "bound_count": state["bound_count"],
                        "stream": stream,
                        "crn_seed": int(crn_seed),
                        "arm": arm_name,
                        "overtime_rung": -1.0 if rung is None else float(rung),
                        **metrics,
                    }
                )
    return rows


def clinically_noninferior(
    anchor: dict[str, float], candidate: dict[str, float]
) -> bool:
    return (
        candidate["completion_service_level"] >= anchor["completion_service_level"]
        and candidate["patients_lost"] <= anchor["patients_lost"]
        and candidate["manufacturing_ineligibility"]
        <= anchor["manufacturing_ineligibility"]
    )


def stream_means(
    rows: list[dict[str, Any]], state_id: str, stream: str
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["state_id"] == state_id and row["stream"] == stream:
            grouped[row["arm"]].append(row)
    metrics = (
        "remaining_cost",
        "completion_service_level",
        "patients_lost",
        "manufacturing_ineligibility",
    )
    return {
        arm: {metric: float(np.mean([r[metric] for r in entries])) for metric in metrics}
        for arm, entries in grouped.items()
    }


def summarize_scenario(
    scenario_name: str,
    rows: list[dict[str, Any]],
    state_ids: list[str],
    material_threshold: float,
) -> dict[str, Any]:
    per_state: list[dict[str, Any]] = []
    for state_id in state_ids:
        discovery = stream_means(rows, state_id, "discovery")
        validation = stream_means(rows, state_id, "validation")
        anchor_val = validation["anchor"]
        material_arms = []
        for arm, means in validation.items():
            if arm == "anchor":
                continue
            saving = anchor_val["remaining_cost"] - means["remaining_cost"]
            if saving >= material_threshold and clinically_noninferior(
                anchor_val, means
            ):
                material_arms.append(
                    {"arm": arm, "validated_saving": saving}
                )
        best_discovery = min(
            (arm for arm in discovery if arm != "anchor"),
            key=lambda arm: discovery[arm]["remaining_cost"],
            default=None,
        )
        best_validation = min(
            (arm for arm in validation if arm != "anchor"),
            key=lambda arm: validation[arm]["remaining_cost"],
            default=None,
        )
        prospective = None
        if best_discovery is not None:
            means = validation[best_discovery]
            saving = anchor_val["remaining_cost"] - means["remaining_cost"]
            prospective = {
                "arm": best_discovery,
                "validated_saving": saving,
                "material_and_noninferior": bool(
                    saving >= material_threshold
                    and clinically_noninferior(anchor_val, means)
                ),
            }
        per_state.append(
            {
                "state_id": state_id,
                "material_validated_arms": material_arms,
                "has_material_validated_opportunity": bool(material_arms),
                "best_discovery_arm": best_discovery,
                "best_validation_arm": best_validation,
                "best_arm_agreement": bool(
                    best_discovery is not None and best_discovery == best_validation
                ),
                "prospective_selection": prospective,
            }
        )
    total = len(per_state)
    material = sum(1 for s in per_state if s["has_material_validated_opportunity"])
    agreement = sum(1 for s in per_state if s["best_arm_agreement"])
    prospective_success = sum(
        1
        for s in per_state
        if s["prospective_selection"]
        and s["prospective_selection"]["material_and_noninferior"]
    )
    return {
        "scenario": scenario_name,
        "states": per_state,
        "state_count": total,
        "material_validated_states": material,
        "material_validated_fraction": material / max(total, 1),
        "best_arm_agreement_fraction": agreement / max(total, 1),
        "prospective_success_fraction": prospective_success / max(total, 1),
    }


def run_screen(config: dict[str, Any]) -> dict[str, Any]:
    plan = load_benchmark_plan(Path(config["plan"]))
    scenarios = select_scenarios(plan, config["scenarios"])
    policy = get_heuristic_class(str(config["anchor_algorithm"]))()
    ladder = [float(u) for u in config["action_ladder"]]
    arms: list[tuple[str, float | None]] = [("anchor", None)] + [
        (f"u_{rung:.2f}", rung) for rung in ladder
    ]
    stream_seeds = {
        "discovery": [int(s) for s in config["replications"]["discovery_seeds"]],
        "validation": [int(s) for s in config["replications"]["validation_seeds"]],
    }
    epochs = [int(e) for e in config["state_generation"]["decision_epochs"]]

    all_rows: list[dict[str, Any]] = []
    scenario_summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        env_dict = scenario_env_dict(plan, config, scenario)
        declared_digest = scenario_declaration_digest(env_dict)
        state_ids: list[str] = []
        scenario_rows: list[dict[str, Any]] = []
        for state_seed in config["state_generation"]["seeds"]:
            policy.reset()
            states = generate_states(env_dict, policy, int(state_seed), epochs)
            env = build_scenario_env(env_dict, int(state_seed))
            for state in states:
                policy.reset()
                rows = evaluate_state(env, policy, state, arms, stream_seeds)
                for row in rows:
                    row["scenario"] = scenario["name"]
                    row["scenario_declaration_sha256"] = declared_digest
                    if scenario_declaration_digest(env_dict) != declared_digest:
                        raise RuntimeError(
                            f"scenario declaration drifted for {scenario['name']}"
                        )
                scenario_rows.extend(rows)
                state_ids.append(state["state_id"])
        all_rows.extend(scenario_rows)
        scenario_summaries.append(
            summarize_scenario(
                scenario["name"],
                scenario_rows,
                state_ids,
                float(config["gates"]["material_threshold"]),
            )
        )

    validate_rows(config, all_rows, len(arms), stream_seeds)
    decision = gate_decision(config, scenario_summaries)
    summary = {
        "name": config["name"],
        "experimental_role": config["experimental_role"],
        "config_echo": {
            key: config[key]
            for key in (
                "anchor_algorithm",
                "scenarios",
                "non_nominal_scenarios",
                "overtime_env_overrides",
                "action_ladder",
                "gates",
            )
        },
        "scenario_summaries": scenario_summaries,
        "decision": decision,
        "row_count": len(all_rows),
    }
    write_outputs(config, all_rows, summary)
    return summary


def gate_decision(
    config: dict[str, Any], scenario_summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    fraction_gate = float(config["gates"]["state_fraction"])
    non_nominal = set(config["non_nominal_scenarios"])
    passing = [
        s["scenario"]
        for s in scenario_summaries
        if s["scenario"] in non_nominal
        and s["material_validated_fraction"] >= fraction_gate
    ]
    passed = bool(passing)
    return {
        "gate": "e2_headroom",
        "state_fraction_required": fraction_gate,
        "non_nominal_scenarios_passing": passing,
        "headroom_gate_passed": passed,
        "classification": (
            "overtime_headroom_established"
            if passed
            else "overtime_headroom_not_established"
        ),
        "e3_authorized": passed,
    }


def validate_rows(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    arm_count: int,
    stream_seeds: dict[str, list[int]],
) -> None:
    expected_per_state = arm_count * sum(len(s) for s in stream_seeds.values())
    per_state: dict[str, int] = defaultdict(int)
    keys = set()
    for row in rows:
        for metric in (
            "remaining_cost",
            "completion_service_level",
            "patients_lost",
            "manufacturing_ineligibility",
        ):
            if not np.isfinite(row[metric]):
                raise RuntimeError(f"non-finite {metric} in row {row['state_id']}")
        key = (row["state_id"], row["stream"], row["crn_seed"], row["arm"])
        if key in keys:
            raise RuntimeError(f"duplicate row {key}")
        keys.add(key)
        per_state[row["state_id"]] += 1
    for state_id, count in per_state.items():
        if count != expected_per_state:
            raise RuntimeError(
                f"state {state_id} has {count} rows, expected {expected_per_state}"
            )


def write_outputs(
    config: dict[str, Any], rows: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "headroom_rows.csv"
    columns = [
        "scenario",
        "state_id",
        "decision_epoch",
        "bound_count",
        "stream",
        "crn_seed",
        "arm",
        "overtime_rung",
        "remaining_cost",
        "completion_service_level",
        "patients_lost",
        "manufacturing_ineligibility",
        "scenario_declaration_sha256",
    ]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in columns))
    rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = dict(summary)
    summary["rows_sha256"] = sha256_path(rows_path)
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"E2 rows: {rows_path} ({summary['row_count']} rows)")
    print(f"E2 summary: {summary_path}")
    print(f"E2 decision: {summary['decision']['classification']}")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
