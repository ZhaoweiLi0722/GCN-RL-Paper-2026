"""CRN screen for ungated continuous residual-policy checkpoints.

The deployment gate is deliberately disabled here. This probe answers whether
the residual actor itself contains useful episode-level signal before an
actor-specific gate is calibrated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluation.evaluate_formal import evaluate_agent, summarize_rows
from evaluation.run_full_benchmark import (
    checkpoint_dir_path,
    load_benchmark_plan,
    make_evaluation_config,
    resolve_budget,
    select_scenarios,
)
from src.baselines.heuristics import get_heuristic_class
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env, write_rows


DEFAULT_PLAN = "experiments/configs/residual_policy_benchmark.json"
DEFAULT_ALGORITHM = "gcn_residual_mdl2_network_ddpg_afd"
DEFAULT_SCENARIO = "patient_condition_geo_demand_drift"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--budget", default="targeted_100")
    parser.add_argument("--algorithm", default=DEFAULT_ALGORITHM)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--base-policy", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--demand-history-window",
        type=int,
        default=None,
        help="Override the scenario demand-history window for evaluation.",
    )
    parser.add_argument("--gcn-hidden-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--actor-hidden-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--gate-hidden-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=(0.01, 0.025, 0.05, 0.1, 0.25),
    )
    parser.add_argument(
        "--gate-mode",
        choices=("disabled", "checkpoint"),
        default="disabled",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=(0.8,),
    )
    parser.add_argument(
        "--group-thresholds",
        nargs="+",
        default=None,
        help=(
            "Comma-separated per-group thresholds, for example "
            "0.3,1.0,0.4. Overrides --thresholds."
        ),
    )
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument(
        "--output-root",
        default="results/continuous_residual_checkpoint_probe",
    )
    args = parser.parse_args()

    plan = load_benchmark_plan(args.plan)
    budget = resolve_budget(plan, args.budget)
    scenario = select_scenarios(plan, (args.scenario,))[0]
    config = make_evaluation_config(
        plan,
        args.budget,
        budget,
        args.algorithm,
        scenario,
        args.seed,
    )
    if args.demand_history_window is not None:
        config.setdefault("env", {})["demand_history_window"] = int(
            args.demand_history_window
        )
    if args.gcn_hidden_sizes:
        config["gcn_hidden_sizes"] = [
            int(value) for value in args.gcn_hidden_sizes
        ]
    if args.actor_hidden_sizes:
        sizes = [int(value) for value in args.actor_hidden_sizes]
        if str(args.algorithm).startswith("flat_"):
            config["hidden_sizes"] = sizes
        else:
            config["actor_hidden_sizes"] = sizes
    if args.gate_hidden_sizes:
        config.setdefault("residual_action", {}).setdefault(
            "correction_gate",
            {},
        )["hidden_sizes"] = [
            int(value) for value in args.gate_hidden_sizes
        ]
    if args.base_policy:
        config.setdefault("residual_action", {})["base_policy"] = str(
            args.base_policy
        )
    evaluation_seed = int(
        args.evaluation_seed
        if args.evaluation_seed is not None
        else budget.get("evaluation_seed", 70_000)
    )
    checkpoint_paths = resolve_checkpoints(
        plan,
        args.budget,
        args.algorithm,
        scenario,
        args.seed,
        requested=args.checkpoints,
    )
    output_root = (
        Path(args.output_root)
        / args.budget
        / args.scenario
        / f"{args.algorithm}_seed{args.seed}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    anchor_name = str(
        config.get("residual_action", {}).get("base_policy", "mdl2")
    )
    anchor_env = build_env(config, seed=evaluation_seed)
    anchor = get_heuristic_class(anchor_name)(
        anchor_env.observation_size,
        anchor_env.action_size,
        dict(config.get("residual_action", {}).get("base_policy_config", {})),
    )
    anchor_rows = evaluate_agent(
        anchor,
        anchor_env,
        algorithm=anchor_name,
        seed=evaluation_seed,
        replications=args.replications,
        max_steps=int(budget["max_steps_per_episode"]),
    )
    add_training_seed(anchor_rows, int(args.seed))
    write_rows(anchor_rows, output_root / "anchor_rows.csv")
    anchor_summary = summarize_rows(anchor_rows)

    results: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        thresholds: tuple[float | tuple[float, ...] | None, ...]
        if args.gate_mode == "disabled":
            thresholds = (None,)
        elif args.group_thresholds:
            thresholds = tuple(
                tuple(float(item) for item in value.split(","))
                for value in args.group_thresholds
            )
        else:
            thresholds = tuple(float(value) for value in args.thresholds)
        for threshold in thresholds:
            for scale in args.scales:
                result = evaluate_checkpoint_candidate(
                    checkpoint_path,
                    scale=float(scale),
                    threshold=threshold,
                    gate_mode=args.gate_mode,
                    config=config,
                    budget=budget,
                    algorithm=args.algorithm,
                    anchor_rows=anchor_rows,
                    evaluation_seed=evaluation_seed,
                    replications=args.replications,
                    output_root=output_root,
                )
                results.append(result)
                print(
                    f"checkpoint={result['checkpoint_label']} "
                    f"scale={float(scale):g} "
                    f"threshold={result['gate_threshold']} "
                    f"cost_gap={result['cost_gap_pct']:.4f}% "
                    f"ci=({result['cost_difference_ci_low']:.1f},"
                    f"{result['cost_difference_ci_high']:.1f}) "
                    f"completion_delta={result['completion_service_level_delta']:.6f} "
                    f"patients_lost_delta={result['patients_lost_delta']:.3f}",
                    flush=True,
                )

    payload = {
        "algorithm": args.algorithm,
        "scenario": args.scenario,
        "training_seed": int(args.seed),
        "evaluation_seed": evaluation_seed,
        "replications": int(args.replications),
        "gate_mode": args.gate_mode,
        "anchor": anchor_summary,
        "results": results,
        "best_clinically_noninferior": select_best_candidate(results),
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["best_clinically_noninferior"], indent=2))


def resolve_checkpoints(
    plan: dict[str, Any],
    budget_name: str,
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
    *,
    requested: Iterable[str] | None,
) -> tuple[Path, ...]:
    if requested:
        paths = tuple(Path(value) for value in requested)
    else:
        directory = checkpoint_dir_path(
            plan,
            budget_name,
            algorithm,
            scenario,
            seed,
        )
        paths = tuple(sorted(directory.glob(f"{algorithm}_seed{seed}_*.pt")))
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(missing[0])
    if not paths:
        raise FileNotFoundError("No residual checkpoints found")
    return paths


def disable_deployment_gate(agent) -> None:
    if hasattr(agent, "correction_gate"):
        agent.correction_gate = None
    if hasattr(agent, "correction_gate_optimizer"):
        agent.correction_gate_optimizer = None


def evaluate_checkpoint_candidate(
    checkpoint_path: Path,
    *,
    scale: float,
    threshold: float | tuple[float, ...] | None,
    gate_mode: str,
    config: dict[str, Any],
    budget: dict[str, Any],
    algorithm: str,
    anchor_rows: list[dict[str, Any]],
    evaluation_seed: int,
    replications: int,
    output_root: Path,
) -> dict[str, Any]:
    env = build_env(config, seed=evaluation_seed)
    agent = get_agent_class(algorithm)(
        env.observation_size,
        env.action_size,
        config,
    )
    agent.load_actor(checkpoint_path)
    if gate_mode == "disabled":
        disable_deployment_gate(agent)
    elif isinstance(threshold, tuple):
        if len(threshold) != len(agent.correction_gate_groups):
            raise ValueError(
                "Group thresholds must match configured correction-gate groups"
            )
        agent.correction_gate_group_thresholds = tuple(
            float(value) for value in threshold
        )
    elif threshold is not None:
        agent.correction_gate_threshold = float(threshold)
        if getattr(agent, "correction_gate_groups", ()):
            agent.correction_gate_group_thresholds = tuple(
                float(threshold)
                for _group in agent.correction_gate_groups
            )
    original_scale = np.asarray(
        agent.residual_scale_vector,
        dtype=np.float32,
    ).copy()
    agent.residual_scale_vector = original_scale * float(scale)
    candidate_rows = evaluate_agent(
        agent,
        env,
        algorithm=algorithm,
        seed=evaluation_seed,
        replications=replications,
        max_steps=int(budget["max_steps_per_episode"]),
    )
    add_training_seed(candidate_rows, int(config.get("seed", 0)))
    label = checkpoint_label(checkpoint_path)
    threshold_label = (
        "disabled"
        if threshold is None
        else (
            ",".join(f"{float(value):g}" for value in threshold)
            if isinstance(threshold, tuple)
            else f"{float(threshold):g}"
        )
    )
    write_rows(
        candidate_rows,
        output_root
        / f"{label}_scale{float(scale):g}_threshold{threshold_label}_rows.csv",
    )
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_label": label,
        "scale": float(scale),
        "gate_threshold": threshold_label,
        **paired_candidate_summary(candidate_rows, anchor_rows),
    }


def checkpoint_label(path: Path) -> str:
    stem = path.stem
    return stem.rsplit("_", 1)[-1]


def add_training_seed(
    rows: list[dict[str, Any]],
    training_seed: int,
) -> None:
    """Keep policy-training and Monte Carlo seeds distinct in saved rows."""

    for row in rows:
        row["training_seed"] = int(training_seed)
        row["evaluation_seed"] = int(row.get("seed", 0))


def paired_candidate_summary(
    candidate_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(candidate_rows) != len(anchor_rows) or not candidate_rows:
        raise ValueError("Candidate and anchor rows must form non-empty CRN pairs")
    candidate_cost = np.asarray(
        [float(row["total_cost"]) for row in candidate_rows],
        dtype=float,
    )
    anchor_cost = np.asarray(
        [float(row["total_cost"]) for row in anchor_rows],
        dtype=float,
    )
    cost_difference = candidate_cost - anchor_cost
    sem = (
        float(cost_difference.std(ddof=1) / np.sqrt(cost_difference.size))
        if cost_difference.size > 1
        else 0.0
    )
    mean_difference = float(cost_difference.mean())
    ci_radius = 1.96 * sem

    completion_delta = _paired_metric_delta(
        candidate_rows,
        anchor_rows,
        "completion_service_level",
    )
    patients_lost_delta = _paired_metric_delta(
        candidate_rows,
        anchor_rows,
        "patients_lost",
    )
    manufacturing_delta = _paired_metric_delta(
        candidate_rows,
        anchor_rows,
        "patient_ineligibility_during_manufacturing_rate",
    )
    clinically_noninferior = bool(
        completion_delta >= -1e-12
        and patients_lost_delta <= 1e-12
        and manufacturing_delta <= 1e-12
    )
    return {
        "candidate_cost_mean": float(candidate_cost.mean()),
        "anchor_cost_mean": float(anchor_cost.mean()),
        "cost_difference_mean": mean_difference,
        "cost_gap_pct": 100.0 * mean_difference / float(anchor_cost.mean()),
        "cost_difference_ci_low": mean_difference - ci_radius,
        "cost_difference_ci_high": mean_difference + ci_radius,
        "paired_win_rate": float((cost_difference < 0.0).mean()),
        "completion_service_level_delta": completion_delta,
        "patients_lost_delta": patients_lost_delta,
        "manufacturing_ineligibility_rate_delta": manufacturing_delta,
        "clinically_noninferior": clinically_noninferior,
    }


def _paired_metric_delta(
    candidate_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    metric: str,
) -> float:
    return float(
        np.mean(
            [
                float(candidate[metric]) - float(anchor[metric])
                for candidate, anchor in zip(candidate_rows, anchor_rows)
            ]
        )
    )


def select_best_candidate(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        row
        for row in results
        if bool(row["clinically_noninferior"])
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda row: float(row["candidate_cost_mean"]))


if __name__ == "__main__":
    main()
