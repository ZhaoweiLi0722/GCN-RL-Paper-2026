"""Audit route-only action headroom around a frozen residual policy."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluation.evaluate_multiscenario_network_residual import (
    apply_multiscenario_env_contract,
)
from evaluation.network_residual_headroom import (
    evaluate_candidate_specs,
    select_clinical_candidate,
)
from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    make_scenario_env_config,
    select_scenarios,
)
from src.baselines.heuristics import get_heuristic_class
from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env, write_rows
from src.rl.residual_options import (
    make_explicit_residual_option_specs,
    residual_option_actions_from_env,
)


DEFAULT_CONFIG = Path(
    "experiments/configs/"
    "patient_indexed_specimen_routing_ddpg_frozen_route_headroom_audit.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.smoke:
        config = smoke_config(config)
    result = run_audit(config, config_path=config_path)
    print(json.dumps(result, indent=2, sort_keys=True))


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    smoke["training_seeds"] = [int(config["training_seeds"][0])]
    first_seed = str(smoke["training_seeds"][0])
    smoke["scenario_by_training_seed"] = {
        first_seed: str(config["scenario_by_training_seed"][first_seed])
    }
    smoke["max_steps"] = 1
    smoke["discovery_replications"] = 1
    smoke["validation_replications"] = 1
    smoke["output_root"] = str(
        config.get(
            "smoke_output_root",
            "/private/tmp/ddpg_frozen_route_headroom_audit_smoke",
        )
    )
    smoke["name"] = f"{config['name']}_smoke"
    return smoke


def validate_config(config: dict[str, Any]) -> None:
    algorithm = str(config["algorithm"])
    if not algorithm:
        raise ValueError("Frozen-route audit requires one algorithm")
    seeds = tuple(int(value) for value in config["training_seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Frozen-route audit training seeds must be unique")
    assignment = {
        int(seed): str(scenario)
        for seed, scenario in config[
            "scenario_by_training_seed"
        ].items()
    }
    if set(assignment) != set(seeds):
        raise ValueError(
            "Frozen-route audit scenario assignment must match training seeds"
        )
    max_steps = int(config["max_steps"])
    if max_steps <= 0:
        raise ValueError("Frozen-route audit max_steps must be positive")
    for key in ("discovery_replications", "validation_replications"):
        if int(config[key]) <= 0:
            raise ValueError(f"Frozen-route audit {key} must be positive")
    evidence_seeds = {
        int(config["discovery_seed"]),
        int(config["validation_seed"]),
    }
    if len(evidence_seeds) != 2:
        raise ValueError(
            "Discovery and validation CRN streams must be disjoint"
        )
    forbidden = {int(value) for value in config.get("forbidden_crn_seeds", ())}
    if evidence_seeds & forbidden:
        raise ValueError("Frozen-route audit uses a forbidden CRN stream")
    options = make_explicit_residual_option_specs(config["explicit_options"])
    if any(
        not option.is_anchor and option.group != "specimen_transfer"
        for option in options
    ):
        raise ValueError(
            "Frozen-route audit options must be specimen-transfer only"
        )


def verify_locked_file(path: Path, expected_sha256: str | None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected_sha256 not in (None, "") and actual != str(
        expected_sha256
    ).lower():
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected_sha256}, "
            f"got {actual}"
        )
    return actual


def run_audit(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    validate_config(config)
    output_root = Path(config["output_root"])
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    manifest_path = Path(config["training_manifest"])
    plan_path = Path(config["plan"])
    manifest_sha256 = verify_locked_file(
        manifest_path,
        config.get("training_manifest_sha256"),
    )
    plan_sha256 = verify_locked_file(
        plan_path,
        config.get("plan_sha256"),
    )
    manifest = load_config(manifest_path)
    plan = load_benchmark_plan(plan_path)
    algorithm = str(config["algorithm"])
    seeds = tuple(int(value) for value in config["training_seeds"])
    assignment = {
        int(seed): str(scenario)
        for seed, scenario in config[
            "scenario_by_training_seed"
        ].items()
    }
    scenarios = {
        str(scenario["name"]): scenario
        for scenario in select_scenarios(
            plan,
            tuple(assignment[seed] for seed in seeds),
        )
    }
    runs = {
        (str(run["algorithm"]), int(run["seed"])): dict(run)
        for run in manifest["runs"]
    }
    expected = {(algorithm, seed) for seed in seeds}
    if not expected.issubset(runs):
        missing = sorted(expected - set(runs))
        raise ValueError(f"Frozen-route audit manifest is missing {missing}")

    all_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for scenario_index, seed in enumerate(seeds):
        scenario_name = assignment[seed]
        rows, source = run_seed(
            config,
            plan=plan,
            run=runs[(algorithm, seed)],
            seed=seed,
            scenario=scenarios[scenario_name],
            scenario_index=scenario_index,
        )
        all_rows.extend(rows)
        provenance.append(source)

    if not all_rows:
        raise RuntimeError("Frozen-route audit produced no states")
    rows_path = output_root / "headroom_rows.csv"
    write_rows(all_rows, rows_path)
    result = summarize_headroom(
        all_rows,
        config=config,
        provenance=provenance,
    )
    result["name"] = str(config["name"])
    result["config"] = None if config_path is None else str(config_path)
    result["config_sha256"] = (
        None
        if config_path is None or not config_path.is_file()
        else sha256_file(config_path)
    )
    result["training_manifest"] = str(manifest_path)
    result["training_manifest_sha256"] = manifest_sha256
    result["plan"] = str(plan_path)
    result["plan_sha256"] = plan_sha256
    result["rows"] = str(rows_path)
    result["rows_sha256"] = sha256_file(rows_path)
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def run_seed(
    config: dict[str, Any],
    *,
    plan: dict[str, Any],
    run: dict[str, Any],
    seed: int,
    scenario: dict[str, Any],
    scenario_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    algorithm = str(config["algorithm"])
    config_path = Path(str(run["config"]))
    run_config = load_config(config_path)
    run_config["device"] = str(config.get("device", "cpu"))
    scenario_env = make_scenario_env_config(plan, algorithm, scenario)
    run_config["env"] = apply_multiscenario_env_contract(
        scenario_env,
        run_config,
    )
    live_seed = int(config["live_seed"]) + scenario_index * 100_000
    env = build_env(run_config, seed=live_seed)
    frozen = get_agent_class(algorithm)(
        env.observation_size,
        env.action_size,
        run_config,
    )
    checkpoint = _checkpoint_for_variant(run, config["checkpoint_variant"])
    frozen.load_actor(checkpoint)
    if frozen.residual_temporal_guard.enabled:
        raise ValueError(
            "Frozen-route audit requires a stateless residual temporal guard"
        )
    anchor_name = str(
        run_config["residual_action"].get("base_policy", "mdl2")
    )
    anchor = get_heuristic_class(anchor_name)(
        env.observation_size,
        env.action_size,
        dict(
            run_config["residual_action"].get(
                "base_policy_config",
                {},
            )
        ),
    )
    option_specs = make_explicit_residual_option_specs(
        config["explicit_options"]
    )
    state = env.reset(seed=live_seed)
    frozen.reset()
    anchor.reset()
    rows: list[dict[str, Any]] = []
    done = False
    step = 0
    max_steps = int(config["max_steps"])
    while not done and step < max_steps:
        specs = candidate_specs(
            state,
            env,
            frozen,
            anchor,
            option_specs,
        )
        remaining_horizon = max_steps - step
        discovery_seeds = _rollout_seeds(
            int(config["discovery_seed"]),
            scenario_index=scenario_index,
            step=step,
            replications=int(config["discovery_replications"]),
        )
        evaluated = evaluate_candidate_specs(
            specs,
            env,
            frozen,
            lookahead=remaining_horizon,
            rollout_seeds=discovery_seeds,
        )
        best_index, scores, feasible = select_clinical_candidate(
            evaluated,
            score_weights=dict(config["score_weights"]),
            guardrails=dict(config["guardrails"]),
        )
        baseline = evaluated[0]
        selected = evaluated[best_index]
        validation = validation_metrics(
            config,
            env=env,
            frozen=frozen,
            baseline=specs[0],
            selected=specs[best_index],
            selected_index=best_index,
            remaining_horizon=remaining_horizon,
            scenario_index=scenario_index,
            step=step,
        )
        baseline_metrics = baseline["metrics"]
        selected_metrics = selected["metrics"]
        row = {
            "training_seed": seed,
            "scenario": str(scenario["name"]),
            "step": step,
            "remaining_horizon": remaining_horizon,
            "candidate_count": len(evaluated),
            "feasible_candidate_count": int(sum(feasible)),
            "selected_group": str(selected["group"]),
            "selected_epsilon": float(selected["epsilon"]),
            "selected_sign": float(selected.get("sign", 0.0)),
            "discovery_score_improvement": float(
                scores[0] - scores[best_index]
            ),
            "discovery_cost_improvement": float(
                baseline_metrics["total_cost"]
                - selected_metrics["total_cost"]
            ),
            "discovery_completion_service_level_delta": float(
                selected_metrics["completion_service_level"]
                - baseline_metrics["completion_service_level"]
            ),
            "discovery_patients_lost_delta": float(
                selected_metrics["patients_lost"]
                - baseline_metrics["patients_lost"]
            ),
            "discovery_manufacturing_ineligibility_delta": float(
                selected_metrics[
                    "patient_ineligibility_during_manufacturing_rate"
                ]
                - baseline_metrics[
                    "patient_ineligibility_during_manufacturing_rate"
                ]
            ),
            **validation,
        }
        row["validated_opportunity"] = int(
            is_validated_opportunity(row)
        )
        rows.append(row)
        state, _reward, done, _info = env.step(baseline["action"])
        step += 1
        progress_interval = max(int(config.get("progress_interval", 5)), 0)
        if progress_interval and (step % progress_interval == 0 or done):
            print(
                "frozen_route_headroom "
                f"seed={seed} step={step}/{max_steps} "
                f"discovery={sum(row['selected_group'] != 'frozen' for row in rows)} "
                f"validated={sum(bool(row['validated_opportunity']) for row in rows)}",
                flush=True,
            )
    return rows, {
        "training_seed": seed,
        "scenario": str(scenario["name"]),
        "live_seed": live_seed,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def candidate_specs(
    state: np.ndarray,
    env: Any,
    frozen: Any,
    anchor: Any,
    option_specs: Iterable[Any],
) -> list[dict[str, Any]]:
    frozen_action = np.asarray(
        frozen.select_action(state, explore=False, env=env),
        dtype=np.float32,
    )
    anchor_action = np.asarray(
        anchor.select_action(state, explore=False, env=env),
        dtype=np.float32,
    )
    option_specs = tuple(option_specs)
    option_actions = residual_option_actions_from_env(
        anchor_action,
        env,
        option_specs,
    )
    specs = [
        {
            "group": "frozen",
            "epsilon": 0.0,
            "sign": 0.0,
            "variant": 0,
            "action": frozen_action,
        }
    ]
    seen = {frozen_action.tobytes()}
    for index, (option, action) in enumerate(
        zip(option_specs, option_actions),
        start=1,
    ):
        normalized = np.asarray(action, dtype=np.float32)
        if normalized.tobytes() in seen:
            continue
        seen.add(normalized.tobytes())
        specs.append(
            {
                "group": "mdl2" if option.is_anchor else option.group,
                "epsilon": float(option.epsilon),
                "sign": float(option.sign),
                "variant": index,
                "action": normalized,
            }
        )
    return specs


def validation_metrics(
    config: dict[str, Any],
    *,
    env: Any,
    frozen: Any,
    baseline: dict[str, Any],
    selected: dict[str, Any],
    selected_index: int,
    remaining_horizon: int,
    scenario_index: int,
    step: int,
) -> dict[str, float]:
    empty = {
        "validation_cost_improvement": 0.0,
        "validation_completion_service_level_delta": 0.0,
        "validation_patients_lost_delta": 0.0,
        "validation_manufacturing_ineligibility_delta": 0.0,
    }
    if selected_index == 0:
        return empty
    evaluated = evaluate_candidate_specs(
        [baseline, selected],
        env,
        frozen,
        lookahead=remaining_horizon,
        rollout_seeds=_rollout_seeds(
            int(config["validation_seed"]),
            scenario_index=scenario_index,
            step=step,
            replications=int(config["validation_replications"]),
        ),
    )
    baseline_metrics = evaluated[0]["metrics"]
    selected_metrics = evaluated[1]["metrics"]
    return {
        "validation_cost_improvement": float(
            baseline_metrics["total_cost"]
            - selected_metrics["total_cost"]
        ),
        "validation_completion_service_level_delta": float(
            selected_metrics["completion_service_level"]
            - baseline_metrics["completion_service_level"]
        ),
        "validation_patients_lost_delta": float(
            selected_metrics["patients_lost"]
            - baseline_metrics["patients_lost"]
        ),
        "validation_manufacturing_ineligibility_delta": float(
            selected_metrics[
                "patient_ineligibility_during_manufacturing_rate"
            ]
            - baseline_metrics[
                "patient_ineligibility_during_manufacturing_rate"
            ]
        ),
    }


def is_validated_opportunity(row: dict[str, Any]) -> bool:
    return bool(
        str(row["selected_group"]) != "frozen"
        and float(row["validation_cost_improvement"]) > 0.0
        and float(row["validation_completion_service_level_delta"]) >= 0.0
        and float(row["validation_patients_lost_delta"]) <= 0.0
        and float(row["validation_manufacturing_ineligibility_delta"])
        <= 0.0
    )


def summarize_headroom(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    discovery = [row for row in rows if row["selected_group"] != "frozen"]
    validated = [row for row in rows if is_validated_opportunity(row)]
    validated_specimen = [
        row for row in validated if row["selected_group"] == "specimen_transfer"
    ]
    per_seed = []
    for seed in sorted({int(row["training_seed"]) for row in rows}):
        subset = [row for row in rows if int(row["training_seed"]) == seed]
        seed_discovery = [
            row for row in subset if row["selected_group"] != "frozen"
        ]
        seed_validated = [
            row for row in subset if is_validated_opportunity(row)
        ]
        per_seed.append(
            {
                "training_seed": seed,
                "scenario": str(subset[0]["scenario"]),
                "states": len(subset),
                "discovery_opportunities": len(seed_discovery),
                "discovery_opportunity_rate": len(seed_discovery)
                / len(subset),
                "validated_opportunities": len(seed_validated),
                "validated_opportunity_rate": len(seed_validated)
                / len(subset),
                "validated_selected_group_counts": dict(
                    Counter(
                        str(row["selected_group"])
                        for row in seed_validated
                    )
                ),
            }
        )
    improvements = [
        float(row["validation_cost_improvement"])
        for row in validated
    ]
    specimen_improvements = [
        float(row["validation_cost_improvement"])
        for row in validated_specimen
    ]
    return {
        "status": "completed",
        "scope": "single-decision route-only oracle around a frozen policy",
        "checkpoint_variant": str(config["checkpoint_variant"]),
        "live_seed": int(config["live_seed"]),
        "discovery_seed": int(config["discovery_seed"]),
        "discovery_replications": int(config["discovery_replications"]),
        "validation_seed": int(config["validation_seed"]),
        "validation_replications": int(config["validation_replications"]),
        "max_steps": int(config["max_steps"]),
        "states": len(rows),
        "discovery_opportunities": len(discovery),
        "discovery_opportunity_rate": len(discovery) / len(rows),
        "validated_opportunities": len(validated),
        "validated_opportunity_rate": len(validated) / len(rows),
        "validated_specimen_opportunities": len(validated_specimen),
        "validated_specimen_option_counts": dict(
            Counter(
                f"{float(row['selected_sign']):+g}x{float(row['selected_epsilon']):g}"
                for row in validated_specimen
            )
        ),
        "mean_validated_cost_improvement": _mean_or_zero(improvements),
        "median_validated_cost_improvement": _median_or_zero(improvements),
        "mean_validated_specimen_cost_improvement": _mean_or_zero(
            specimen_improvements
        ),
        "median_validated_specimen_cost_improvement": _median_or_zero(
            specimen_improvements
        ),
        "validated_specimen_above_1m": sum(
            value > 1_000_000.0 for value in specimen_improvements
        ),
        "validated_specimen_above_2m": sum(
            value > 2_000_000.0 for value in specimen_improvements
        ),
        "validated_specimen_above_5m": sum(
            value > 5_000_000.0 for value in specimen_improvements
        ),
        "per_seed": per_seed,
        "provenance": provenance,
        "interpretation_limit": (
            "This development audit estimates single-decision opportunities "
            "on frozen-policy state distributions. Per-state advantages are "
            "dependent and must not be summed into an episode-level gain or "
            "treated as publication evidence."
        ),
    }


def _checkpoint_for_variant(
    run: dict[str, Any],
    variant: str,
) -> Path:
    normalized = str(variant).lower()
    if normalized == "pretrain":
        path = Path(str(run["pretrain_checkpoint"]))
    elif normalized == "final":
        path = Path(str(run["checkpoint"]))
    elif normalized.startswith("episode"):
        episode = int(normalized.removeprefix("episode"))
        final = Path(str(run["checkpoint"]))
        algorithm = str(run["algorithm"])
        seed = int(run["seed"])
        path = final.parent / f"{algorithm}_seed{seed}_episode{episode}.pt"
    else:
        raise ValueError(f"Unsupported checkpoint variant: {variant}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _rollout_seeds(
    base: int,
    *,
    scenario_index: int,
    step: int,
    replications: int,
) -> tuple[int, ...]:
    start = int(base) + int(scenario_index) * 1_000_000 + int(step) * 100
    return tuple(start + index for index in range(int(replications)))


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _median_or_zero(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


if __name__ == "__main__":
    main()
