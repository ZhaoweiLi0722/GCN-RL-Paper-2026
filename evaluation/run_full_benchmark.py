"""Run the manuscript-facing benchmark matrix.

The default action is a dry run. Use ``--phase all`` with the ``smoke`` budget
first, then ``pilot`` for stability checks, and only then ``full``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluation.aggregate_results import (
    aggregate_rows,
    read_rows as read_aggregate_inputs,
    write_rows as write_aggregate_rows,
)
from evaluation.evaluate_formal import evaluate_agent, summarize_rows, write_summary
from evaluation.plot_results import _plot_grouped_bars, _read_rows as read_plot_rows
from evaluation.run_gcn_residual_sweep import local_search_metric_score, run_local_search_distillation
from src.baselines.heuristics import available_heuristics
from src.rl.agents import get_agent_class
from src.rl.config import load_config, save_config_snapshot
from src.rl.experiment import build_env, train_off_policy_agent, write_rows


DEFAULT_PLAN = "experiments/configs/full_benchmark.json"
PHASES = ("dry-run", "train", "evaluate", "aggregate", "plot", "all")
LEARNED_EVALUATION_SEED_STRIDE = 10000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--budget", default="smoke", help="Budget key from the plan: smoke, pilot, or full.")
    parser.add_argument("--phase", choices=PHASES, default="dry-run")
    parser.add_argument("--algorithms", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run jobs even when expected outputs exist.")
    parser.add_argument("--max-jobs", type=int, default=None, help="Run at most this many pending jobs per phase.")
    args = parser.parse_args()

    plan = load_benchmark_plan(args.plan)
    budget = resolve_budget(plan, args.budget)
    algorithms = select_algorithms(plan, args.algorithms, primary_only=args.primary_only)
    scenarios = select_scenarios(plan, args.scenarios)
    seeds = tuple(args.seeds if args.seeds is not None else budget["seeds"])

    if args.phase == "dry-run":
        print_dry_run(plan, args.budget, budget, algorithms, scenarios, seeds)
        return

    if args.phase in ("train", "all"):
        train_benchmark(
            plan,
            args.budget,
            budget,
            algorithms,
            scenarios,
            seeds,
            force=args.force,
            max_jobs=args.max_jobs,
        )
    if args.phase in ("evaluate", "all"):
        evaluate_benchmark(
            plan,
            args.budget,
            budget,
            algorithms,
            scenarios,
            seeds,
            force=args.force,
            max_jobs=args.max_jobs,
        )

    summary_path = benchmark_root(plan, args.budget) / "aggregate_summary.csv"
    if args.phase in ("aggregate", "all"):
        summary_path = aggregate_benchmark(plan, args.budget, algorithms, scenarios, seeds)
    if args.phase in ("plot", "all"):
        plot_benchmark(plan, args.budget, summary_path)


def load_benchmark_plan(path: str | Path = DEFAULT_PLAN) -> dict[str, Any]:
    plan = load_config(path)
    required = ("algorithms", "learned_config_paths", "scenarios", "budgets")
    missing = [key for key in required if key not in plan]
    if missing:
        raise ValueError(f"Benchmark plan is missing required keys: {missing}")
    return plan


def resolve_budget(plan: dict[str, Any], budget_name: str) -> dict[str, Any]:
    budgets = plan.get("budgets", {})
    if budget_name not in budgets:
        raise ValueError(f"Unknown budget {budget_name!r}. Available: {sorted(budgets)}")
    budget = dict(budgets[budget_name])
    budget.setdefault("seeds", [0])
    budget.setdefault("num_episodes", 1)
    budget.setdefault("max_steps_per_episode", 52)
    budget.setdefault("evaluation_replications", 500)
    budget.setdefault("evaluation_seed", 10000)
    return budget


def select_algorithms(
    plan: dict[str, Any],
    requested: Iterable[str] | None,
    *,
    primary_only: bool = False,
) -> tuple[str, ...]:
    if requested is not None:
        algorithms = tuple(requested)
    elif primary_only:
        algorithms = tuple(plan.get("primary_algorithms", plan["algorithms"]))
    else:
        algorithms = tuple(plan["algorithms"])

    known = set(plan["algorithms"])
    unknown = [algorithm for algorithm in algorithms if algorithm not in known]
    if unknown:
        raise ValueError(f"Unknown algorithms in benchmark selection: {unknown}")
    return algorithms


def select_scenarios(plan: dict[str, Any], requested: Iterable[str] | None) -> tuple[dict[str, Any], ...]:
    scenarios = tuple(plan["scenarios"])
    if requested is None:
        return scenarios
    requested_names = set(requested)
    selected = tuple(scenario for scenario in scenarios if scenario["name"] in requested_names)
    found_names = {scenario["name"] for scenario in selected}
    missing = sorted(requested_names - found_names)
    if missing:
        raise ValueError(f"Unknown scenarios in benchmark selection: {missing}")
    return selected


def train_benchmark(
    plan: dict[str, Any],
    budget_name: str,
    budget: dict[str, Any],
    algorithms: Iterable[str],
    scenarios: Iterable[dict[str, Any]],
    seeds: Iterable[int],
    *,
    force: bool = False,
    max_jobs: int | None = None,
) -> None:
    learned = set(plan["learned_config_paths"])
    jobs_run = 0
    for algorithm in algorithms:
        if algorithm not in learned:
            print(f"skip training heuristic {algorithm}", flush=True)
            continue
        for scenario in scenarios:
            for seed in seeds:
                config = make_training_config(plan, budget_name, budget, algorithm, scenario, seed)
                if not force and training_outputs_complete(plan, budget_name, budget, algorithm, scenario, seed):
                    print(f"skip existing training {algorithm} scenario={scenario['name']} seed={seed}", flush=True)
                    continue
                if max_jobs is not None and jobs_run >= max_jobs:
                    print(f"reached max training jobs: {max_jobs}", flush=True)
                    return
                print(f"train {algorithm} scenario={scenario['name']} seed={seed}", flush=True)
                env = build_env(config, seed=seed)
                agent = get_agent_class(algorithm)(env.observation_size, env.action_size, config)
                advantage_settings = advantage_distillation_settings(
                    config,
                    budget,
                    algorithm,
                )

                def post_imitation_pretrain(current_agent, current_env):
                    return run_advantage_distillation_pretrain(
                        advantage_settings,
                        algorithm=algorithm,
                        seed=seed,
                        agent=current_agent,
                        env=current_env,
                        config=config,
                        budget=budget,
                    )

                rows = train_off_policy_agent(
                    agent,
                    env,
                    config,
                    post_imitation_pretrain=(
                        post_imitation_pretrain
                        if bool(advantage_settings.get("enabled", False))
                        else None
                    ),
                )
                post_training_summary = run_post_training_steps(
                    plan,
                    budget_name,
                    budget,
                    algorithm,
                    scenario,
                    seed,
                    agent,
                    env,
                    config,
                )
                write_rows(rows, config["result_csv_path"])
                save_config_snapshot(config, config["config_snapshot_path"])
                if post_training_summary:
                    write_summary(
                        post_training_summary,
                        post_training_summary_path(plan, budget_name, algorithm, scenario, seed),
                    )
                jobs_run += 1


def evaluate_benchmark(
    plan: dict[str, Any],
    budget_name: str,
    budget: dict[str, Any],
    algorithms: Iterable[str],
    scenarios: Iterable[dict[str, Any]],
    seeds: Iterable[int],
    *,
    force: bool = False,
    max_jobs: int | None = None,
) -> None:
    jobs_run = 0
    for algorithm in algorithms:
        for scenario in scenarios:
            for seed in seeds:
                if not force and evaluation_outputs_complete(plan, budget_name, algorithm, scenario, seed):
                    print(f"skip existing evaluation {algorithm} scenario={scenario['name']} seed={seed}", flush=True)
                    continue
                if max_jobs is not None and jobs_run >= max_jobs:
                    print(f"reached max evaluation jobs: {max_jobs}", flush=True)
                    return
                config = make_evaluation_config(plan, budget_name, budget, algorithm, scenario, seed)
                env = build_env(config, seed=seed)
                agent = get_agent_class(algorithm)(env.observation_size, env.action_size, config)
                selection_metadata: dict[str, Any] = {"checkpoint_selection_enabled": False}
                if algorithm not in available_heuristics():
                    checkpoint = final_checkpoint_path(plan, budget_name, budget, algorithm, scenario, seed)
                    if not checkpoint.exists():
                        raise SystemExit(
                            f"Missing checkpoint for {algorithm} scenario={scenario['name']} seed={seed}: "
                            f"{checkpoint}"
                        )
                    evaluation_seed = int(budget["evaluation_seed"]) + int(seed) * LEARNED_EVALUATION_SEED_STRIDE
                    checkpoint, selection_metadata = select_validation_checkpoint(
                        plan,
                        budget_name,
                        budget,
                        algorithm,
                        scenario,
                        seed,
                        config,
                        checkpoint,
                        evaluation_seed,
                    )
                    agent.load_actor(checkpoint)

                evaluation_seed = int(budget["evaluation_seed"]) + int(seed) * LEARNED_EVALUATION_SEED_STRIDE
                agent, fallback_metadata = maybe_apply_anchor_fallback(
                    config,
                    budget,
                    algorithm,
                    seed,
                    evaluation_seed,
                    agent,
                )
                print(f"evaluate {algorithm} scenario={scenario['name']} seed={seed}", flush=True)
                rows = evaluate_agent(
                    agent,
                    env,
                    algorithm=algorithm,
                    seed=evaluation_seed,
                    replications=int(budget["evaluation_replications"]),
                    max_steps=int(budget["max_steps_per_episode"]),
                )
                for row in rows:
                    row["benchmark"] = plan.get("name", "benchmark")
                    row["budget"] = budget_name
                    row["training_seed"] = seed
                    row["evaluation_seed"] = evaluation_seed
                    row.update(selection_metadata)
                    row.update(fallback_metadata)
                output_path = evaluation_csv_path(plan, budget_name, algorithm, scenario, seed)
                write_rows(rows, output_path)

                summary = summarize_rows(rows)
                if summary:
                    summary["benchmark"] = plan.get("name", "benchmark")
                    summary["budget"] = budget_name
                    summary["training_seed"] = seed
                    summary["evaluation_seed"] = evaluation_seed
                    summary.update(selection_metadata)
                    summary.update(fallback_metadata)
                    write_summary(summary, evaluation_summary_path(plan, budget_name, algorithm, scenario, seed))
                jobs_run += 1


def aggregate_benchmark(
    plan: dict[str, Any],
    budget_name: str,
    algorithms: Iterable[str],
    scenarios: Iterable[dict[str, Any]],
    seeds: Iterable[int],
) -> Path:
    paths = [
        evaluation_csv_path(plan, budget_name, algorithm, scenario, seed)
        for algorithm in algorithms
        for scenario in scenarios
        for seed in seeds
    ]
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        raise SystemExit(f"No evaluation CSVs found under {benchmark_root(plan, budget_name)}")

    rows = read_aggregate_inputs(existing_paths)
    summary = aggregate_rows(rows)
    output_path = benchmark_root(plan, budget_name) / "aggregate_summary.csv"
    write_aggregate_rows(summary, output_path)
    print(f"wrote {len(summary)} aggregate rows to {output_path}", flush=True)
    return output_path


def plot_benchmark(plan: dict[str, Any], budget_name: str, summary_path: str | Path) -> None:
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise SystemExit(f"Missing aggregate summary: {summary_path}")

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/gcn_rl_matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_plot_rows(summary_path)
    for metric in plan.get("plot_metrics", ["total_cost_mean"]):
        output = figure_root(plan, budget_name) / f"{metric}.png"
        _plot_grouped_bars(
            rows,
            metric=metric,
            category_key="scenario",
            series_key="algorithm",
            output=output,
            title=metric.replace("_", " ").title(),
            plt=plt,
        )
        print(f"wrote figure to {output}", flush=True)


def make_training_config(
    plan: dict[str, Any],
    budget_name: str,
    budget: dict[str, Any],
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    if algorithm not in plan["learned_config_paths"]:
        raise ValueError(f"{algorithm} is not a learned algorithm")

    config = load_config(plan["learned_config_paths"][algorithm])
    config = _deep_update(config, algorithm_config_overrides(plan, algorithm))
    env = make_scenario_env_config(plan, algorithm, scenario)

    config["algorithm"] = algorithm
    config["seed"] = int(seed)
    config["num_episodes"] = int(budget["num_episodes"])
    config["max_steps_per_episode"] = int(budget["max_steps_per_episode"])
    if "batch_size" in config or "batch_size" in budget:
        config["batch_size"] = int(budget.get("batch_size", config.get("batch_size", 64)))
    config["checkpoint_interval"] = int(budget.get("checkpoint_interval", budget["num_episodes"]))
    config["progress_interval"] = int(budget.get("progress_interval", 0))
    config["checkpoint_dir"] = str(checkpoint_dir_path(plan, budget_name, algorithm, scenario, seed))
    config["result_csv_path"] = str(training_csv_path(plan, budget_name, algorithm, scenario, seed))
    config["config_snapshot_path"] = str(config_snapshot_path(plan, budget_name, algorithm, scenario, seed))
    config["env"] = env

    if algorithm in {"ppo", "gcn_ppo"}:
        config["rollout_length"] = max(1, int(budget["max_steps_per_episode"]))
        config["minibatch_size"] = min(
            int(config.get("minibatch_size", budget.get("batch_size", 64))),
            int(config["rollout_length"]),
        )
    return config


def run_post_training_steps(
    plan: dict[str, Any],
    budget_name: str,
    budget: dict[str, Any],
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
    agent,
    env,
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = local_search_settings(plan, budget, algorithm)
    if not bool(settings.get("enabled", False)):
        return {}
    if not hasattr(agent, "fit_action_batch"):
        raise ValueError(f"{algorithm} does not support local_search post-training")

    baseline_policy = str(
        settings.get(
            "baseline_policy",
            config.get("residual_action", {}).get("base_policy", "myo"),
        )
    )
    min_service_level_delta = settings.get(
        "min_service_level_delta",
        anchor_fallback_settings(config, budget).get("min_service_level_delta"),
    )
    summary = run_local_search_distillation(
        agent,
        env,
        seed=int(settings.get("seed", 90000)) + int(seed) * LEARNED_EVALUATION_SEED_STRIDE,
        rollouts=int(settings.get("rollouts", 0)),
        lookahead=int(settings.get("lookahead", 6)),
        epsilons=tuple(float(value) for value in settings.get("epsilons", [0.02, 0.05, 0.08])),
        epochs=int(settings.get("epochs", 16)),
        batch_size=int(settings.get("batch_size", config.get("batch_size", 64))),
        max_steps=int(settings.get("max_steps", budget["max_steps_per_episode"])),
        baseline_policy=baseline_policy,
        min_improvement=float(settings.get("min_improvement", 0.0)),
        anchor_keep_probability=float(settings.get("anchor_keep_probability", 0.0)),
        anchor_keep_weight=float(settings.get("anchor_keep_weight", 1.0)),
        anchor_keep_on_improved=bool(settings.get("anchor_keep_on_improved", True)),
        balance_label_weights=bool(settings.get("balance_label_weights", False)),
        retain_for_regularization=bool(settings.get("retain_for_regularization", False)),
        min_service_level_delta=(
            float(min_service_level_delta)
            if min_service_level_delta is not None
            else None
        ),
        service_level_weight=float(settings.get("service_level_weight", 0.0)),
        eligibility_rate_weight=float(settings.get("eligibility_rate_weight", 0.0)),
        at_risk_unserved_weight=float(settings.get("at_risk_unserved_weight", 0.0)),
        patients_lost_weight=float(settings.get("patients_lost_weight", 0.0)),
        candidate_groups=local_search_candidate_groups(settings),
        candidate_signs=local_search_candidate_signs(settings),
    )
    checkpoint_path = local_search_checkpoint_path(plan, budget_name, algorithm, scenario, seed)
    agent.save(checkpoint_path)
    summary.update(
        {
            "algorithm": algorithm,
            "scenario": scenario["name"],
            "graph_ablation": getattr(env, "graph_ablation", config.get("env", {}).get("graph_ablation", "")),
            "training_seed": int(seed),
            "selection_stage": "local_search",
            "checkpoint_path": str(checkpoint_path),
            "local_search_baseline_policy": baseline_policy,
        }
    )
    return summary


def advantage_distillation_settings(
    config: dict[str, Any],
    budget: dict[str, Any],
    algorithm: str,
) -> dict[str, Any]:
    settings = dict(config.get("advantage_distillation_pretrain", {}))
    budget_settings = dict(
        budget.get("advantage_distillation_pretrain", {}).get(algorithm, {})
    )
    return _deep_update(settings, budget_settings)


def run_advantage_distillation_pretrain(
    settings: dict[str, Any],
    *,
    algorithm: str,
    seed: int,
    agent,
    env,
    config: dict[str, Any],
    budget: dict[str, Any],
) -> dict[str, Any]:
    if not bool(settings.get("enabled", False)):
        return {}
    if not hasattr(agent, "fit_action_batch"):
        raise ValueError(f"{algorithm} does not support advantage distillation")

    baseline_policy = str(
        settings.get(
            "baseline_policy",
            config.get("residual_action", {}).get("base_policy", "mdl2"),
        )
    )
    summary = run_local_search_distillation(
        agent,
        env,
        seed=int(settings.get("seed", 98000))
        + int(seed) * LEARNED_EVALUATION_SEED_STRIDE,
        rollouts=int(settings.get("rollouts", 0)),
        lookahead=int(settings.get("lookahead", 6)),
        epsilons=tuple(float(value) for value in settings.get("epsilons", [0.005, 0.01])),
        epochs=int(settings.get("epochs", 16)),
        batch_size=int(settings.get("batch_size", config.get("batch_size", 64))),
        max_steps=int(settings.get("max_steps", budget["max_steps_per_episode"])),
        baseline_policy=baseline_policy,
        min_improvement=float(settings.get("min_improvement", 0.0)),
        anchor_keep_probability=float(settings.get("anchor_keep_probability", 1.0)),
        anchor_keep_weight=float(settings.get("anchor_keep_weight", 1.0)),
        anchor_keep_on_improved=bool(settings.get("anchor_keep_on_improved", False)),
        balance_label_weights=bool(settings.get("balance_label_weights", True)),
        retain_for_regularization=bool(settings.get("retain_for_regularization", True)),
        min_service_level_delta=(
            float(settings["min_service_level_delta"])
            if settings.get("min_service_level_delta") is not None
            else None
        ),
        service_level_weight=float(settings.get("service_level_weight", 0.0)),
        eligibility_rate_weight=float(settings.get("eligibility_rate_weight", 0.0)),
        at_risk_unserved_weight=float(settings.get("at_risk_unserved_weight", 0.0)),
        patients_lost_weight=float(settings.get("patients_lost_weight", 0.0)),
        candidate_groups=local_search_candidate_groups(settings),
        candidate_signs=local_search_candidate_signs(settings),
    )
    renamed = {
        key.replace("local_search_", "advantage_distillation_", 1): value
        for key, value in summary.items()
    }
    print(
        "advantage_distillation "
        f"algorithm={algorithm} samples={renamed.get('advantage_distillation_samples', 0)} "
        f"improved_steps={renamed.get('advantage_distillation_improved_steps', 0)} "
        f"anchor_steps={renamed.get('advantage_distillation_anchor_keep_steps', 0)} "
        f"improved_weight_fraction="
        f"{renamed.get('advantage_distillation_improved_weight_fraction', 0):.3f}",
        flush=True,
    )
    return renamed


def algorithm_config_overrides(plan: dict[str, Any], algorithm: str) -> dict[str, Any]:
    return _resolved_algorithm_config_overrides(plan, algorithm, seen=())


def _resolved_algorithm_config_overrides(
    plan: dict[str, Any],
    algorithm: str,
    *,
    seen: tuple[str, ...],
) -> dict[str, Any]:
    if algorithm in seen:
        chain = " -> ".join((*seen, algorithm))
        raise ValueError(f"Cyclic algorithm inheritance: {chain}")
    settings = dict(plan.get("algorithm_settings", {}).get(algorithm, {}))
    parent = settings.get("inherits")
    overrides: dict[str, Any] = {}
    if parent:
        overrides = _resolved_algorithm_config_overrides(
            plan,
            str(parent),
            seen=(*seen, algorithm),
        )
    return _deep_update(overrides, dict(settings.get("config_overrides", {})))


def local_search_settings(plan: dict[str, Any], budget: dict[str, Any], algorithm: str) -> dict[str, Any]:
    settings = dict(plan.get("algorithm_settings", {}).get(algorithm, {}).get("local_search", {}))
    budget_settings = dict(budget.get("local_search", {}).get(algorithm, {}))
    settings.update(budget_settings)
    return settings


def local_search_candidate_groups(settings: dict[str, Any]) -> tuple[str, ...] | None:
    raw = settings.get("candidate_groups")
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        raw = raw.replace("|", ",").split(",")
    groups = tuple(str(group).strip() for group in raw if str(group).strip())
    return groups or None


def local_search_candidate_signs(settings: dict[str, Any]) -> tuple[float, ...] | None:
    raw = settings.get("candidate_signs")
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        raw = raw.replace("|", ",").split(",")
    signs = tuple(float(sign) for sign in raw)
    return signs or None


def checkpoint_selection_settings(config: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config.get("checkpoint_selection", {}))
    settings.update(dict(budget.get("checkpoint_selection", {})))
    return settings


def select_validation_checkpoint(
    plan: dict[str, Any],
    budget_name: str,
    budget: dict[str, Any],
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    default_checkpoint: Path,
    evaluation_seed: int,
) -> tuple[Path, dict[str, Any]]:
    settings = checkpoint_selection_settings(config, budget)
    if not bool(settings.get("enabled", False)):
        return default_checkpoint, {"checkpoint_selection_enabled": False}

    candidates = learned_checkpoint_candidates(
        plan,
        budget_name,
        algorithm,
        scenario,
        seed,
        default_checkpoint,
    )
    if len(candidates) <= 1:
        return default_checkpoint, {
            "checkpoint_selection_enabled": True,
            "checkpoint_selection_selected_path": str(default_checkpoint),
            "checkpoint_selection_selected_label": checkpoint_label(default_checkpoint),
            "checkpoint_selection_candidate_count": len(candidates),
        }

    validation_replications = int(
        settings.get(
            "validation_replications",
            anchor_fallback_settings(config, budget).get("validation_replications", 5),
        )
    )
    validation_seed = (
        int(evaluation_seed)
        + int(settings.get("validation_seed_offset", 200000))
        + int(seed) * int(settings.get("validation_seed_stride", 1000))
    )
    max_steps = int(config.get("max_steps_per_episode", budget["max_steps_per_episode"]))
    score_weights = {
        "service_level": float(settings.get("service_level_weight", 0.0)),
        "eligibility_rate": float(settings.get("eligibility_rate_weight", 0.0)),
        "at_risk_unserved": float(settings.get("at_risk_unserved_weight", 0.0)),
        "patients_lost": float(settings.get("patients_lost_weight", 0.0)),
    }
    scored: list[tuple[float, dict[str, float], Path]] = []
    for path in candidates:
        validation_env = build_env(config, seed=validation_seed)
        validation_agent = get_agent_class(algorithm)(
            validation_env.observation_size,
            validation_env.action_size,
            config,
        )
        validation_agent.load_actor(path)
        rows = evaluate_agent(
            validation_agent,
            validation_env,
            algorithm=algorithm,
            seed=validation_seed,
            replications=validation_replications,
            max_steps=max_steps,
        )
        summary = summarize_rows(rows)
        metrics = checkpoint_score_metrics(summary)
        score = local_search_metric_score(metrics, score_weights)
        scored.append((score, metrics, path))

    selected_score, selected_metrics, selected_path = min(
        scored,
        key=lambda item: item[0],
    )
    selected_cost = selected_metrics["total_cost"]
    selected_service_level = selected_metrics["service_level"]
    print(
        "checkpoint_selection "
        f"algorithm={algorithm} selected={checkpoint_label(selected_path)} "
        f"cost={selected_cost:.3f} service={selected_service_level:.6f} "
        f"score={selected_score:.3f} candidates={len(candidates)}",
        flush=True,
    )
    return selected_path, {
        "checkpoint_selection_enabled": True,
        "checkpoint_selection_selected_path": str(selected_path),
        "checkpoint_selection_selected_label": checkpoint_label(selected_path),
        "checkpoint_selection_validation_cost": selected_cost,
        "checkpoint_selection_validation_service_level": selected_service_level,
        "checkpoint_selection_validation_eligibility_rate": selected_metrics["eligibility_rate"],
        "checkpoint_selection_validation_at_risk_unserved": selected_metrics["at_risk_unserved"],
        "checkpoint_selection_validation_patients_lost": selected_metrics["patients_lost"],
        "checkpoint_selection_validation_score": selected_score,
        "checkpoint_selection_service_level_weight": score_weights["service_level"],
        "checkpoint_selection_eligibility_rate_weight": score_weights["eligibility_rate"],
        "checkpoint_selection_at_risk_unserved_weight": score_weights["at_risk_unserved"],
        "checkpoint_selection_patients_lost_weight": score_weights["patients_lost"],
        "checkpoint_selection_candidate_count": len(candidates),
        "checkpoint_selection_validation_replications": validation_replications,
        "checkpoint_selection_validation_seed": validation_seed,
    }


def checkpoint_score_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """Normalize evaluation summary fields into the local-search score schema."""

    return {
        "total_cost": float(summary.get("total_cost_mean", float("inf"))),
        "service_level": float(summary.get("service_level_mean", 0.0)),
        "eligibility_rate": float(
            summary.get(
                "eligibility_rate_mean_mean",
                summary.get("eligibility_rate_mean", 0.0),
            )
        ),
        "at_risk_unserved": float(summary.get("at_risk_unserved_mean", 0.0)),
        "patients_lost": float(summary.get("patients_lost_mean", 0.0)),
    }


def learned_checkpoint_candidates(
    plan: dict[str, Any],
    budget_name: str,
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
    default_checkpoint: Path,
) -> tuple[Path, ...]:
    checkpoint_dir = checkpoint_dir_path(plan, budget_name, algorithm, scenario, seed)
    pattern = f"{algorithm}_seed{int(seed)}_episode*.pt"
    candidates = sorted(checkpoint_dir.glob(pattern), key=checkpoint_sort_key)
    local_search_path = local_search_checkpoint_path(plan, budget_name, algorithm, scenario, seed)
    if local_search_path.exists():
        candidates.append(local_search_path)
    if default_checkpoint.exists() and default_checkpoint not in candidates:
        candidates.append(default_checkpoint)
    return tuple(dict.fromkeys(candidates))


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if "_episode" in stem:
        try:
            return (int(stem.rsplit("_episode", 1)[1]), stem)
        except ValueError:
            return (10**9, stem)
    return (10**9, stem)


def checkpoint_label(path: Path) -> str:
    stem = path.stem
    if "_episode" in stem:
        return f"episode{stem.rsplit('_episode', 1)[1]}"
    if stem.endswith("_local_search"):
        return "local_search"
    return stem


def maybe_apply_anchor_fallback(
    config: dict[str, Any],
    budget: dict[str, Any],
    algorithm: str,
    seed: int,
    evaluation_seed: int,
    learned_agent,
) -> tuple[Any, dict[str, Any]]:
    """Select the learned residual policy only when it beats its heuristic anchor.

    The validation split is disjoint from the formal evaluation seed stream. The
    returned metadata is written to every evaluation row so manuscript-facing
    summaries can distinguish a genuine learned residual deployment from a
    conservative fallback to the anchor.
    """

    settings = anchor_fallback_settings(config, budget)
    if not bool(settings.get("enabled", False)):
        return learned_agent, {"anchor_fallback_enabled": False}
    if algorithm in available_heuristics():
        return learned_agent, {"anchor_fallback_enabled": False}

    residual_config = dict(config.get("residual_action", {}))
    if not bool(residual_config.get("enabled", False)):
        return learned_agent, {"anchor_fallback_enabled": False}
    anchor_policy = str(settings.get("anchor_policy", residual_config.get("base_policy", "")))
    if anchor_policy not in available_heuristics():
        raise ValueError(f"anchor_fallback requires a heuristic anchor policy, got {anchor_policy!r}")

    validation_replications = int(settings.get("validation_replications", 0))
    if validation_replications <= 0:
        return learned_agent, {
            "anchor_fallback_enabled": True,
            "anchor_fallback_selected_policy": "learned",
            "anchor_fallback_anchor_policy": anchor_policy,
            "anchor_fallback_validation_replications": 0,
        }

    validation_seed = (
        int(evaluation_seed)
        + int(settings.get("validation_seed_offset", 250000))
        + int(seed) * int(settings.get("validation_seed_stride", 1000))
    )
    max_steps = int(config.get("max_steps_per_episode", budget["max_steps_per_episode"]))

    anchor_config = dict(config)
    anchor_config["algorithm"] = anchor_policy
    anchor_policy_config = dict(
        settings.get(
            "anchor_policy_config",
            residual_config.get("base_policy_config", {}),
        )
    )
    anchor_env = build_env(anchor_config, seed=validation_seed)
    anchor_agent = get_agent_class(anchor_policy)(
        anchor_env.observation_size,
        anchor_env.action_size,
        anchor_policy_config,
    )
    anchor_rows = evaluate_agent(
        anchor_agent,
        anchor_env,
        algorithm=anchor_policy,
        seed=validation_seed,
        replications=validation_replications,
        max_steps=max_steps,
    )
    anchor_summary = summarize_rows(anchor_rows)

    original_residual_scale = copy_residual_scale_vector(learned_agent)
    scale_candidates = residual_deployment_scale_candidates(
        settings,
        has_residual_scale=original_residual_scale is not None,
    )
    learned_candidates: list[dict[str, Any]] = []
    try:
        for residual_scale in scale_candidates:
            if original_residual_scale is not None:
                set_residual_scale_vector(learned_agent, original_residual_scale * residual_scale)
            learned_env = build_env(config, seed=validation_seed)
            learned_rows = evaluate_agent(
                learned_agent,
                learned_env,
                algorithm=algorithm,
                seed=validation_seed,
                replications=validation_replications,
                max_steps=max_steps,
            )
            learned_candidates.append(
                {
                    "residual_scale": float(residual_scale),
                    "summary": summarize_rows(learned_rows),
                }
            )
    finally:
        if original_residual_scale is not None:
            set_residual_scale_vector(learned_agent, original_residual_scale)

    learned_candidate, decision = select_residual_deployment_candidate(
        learned_candidates,
        anchor_summary,
        settings,
    )
    selected_residual_scale = float(learned_candidate["residual_scale"])
    deployed_residual_scale = selected_residual_scale if decision == "learned" else 0.0
    learned_summary = learned_candidate["summary"]
    if decision == "learned" and original_residual_scale is not None:
        set_residual_scale_vector(learned_agent, original_residual_scale * selected_residual_scale)
    selected_agent = learned_agent if decision == "learned" else anchor_agent
    metadata = {
        "anchor_fallback_enabled": True,
        "anchor_fallback_selected_policy": decision,
        "anchor_fallback_anchor_policy": anchor_policy,
        "anchor_fallback_selected_residual_scale": deployed_residual_scale,
        "anchor_fallback_validation_selected_candidate_residual_scale": selected_residual_scale,
        "anchor_fallback_residual_scale_candidates": "|".join(
            f"{scale:g}" for scale in scale_candidates
        ),
        "anchor_fallback_residual_scale_candidate_count": len(scale_candidates),
        "anchor_fallback_validation_seed": validation_seed,
        "anchor_fallback_validation_replications": validation_replications,
        "anchor_fallback_validation_learned_cost_mean": learned_summary["total_cost_mean"],
        "anchor_fallback_validation_anchor_cost_mean": anchor_summary["total_cost_mean"],
        "anchor_fallback_validation_learned_service_level_mean": learned_summary.get(
            "service_level_mean",
            "",
        ),
        "anchor_fallback_validation_anchor_service_level_mean": anchor_summary.get(
            "service_level_mean",
            "",
        ),
        "anchor_fallback_min_improvement": float(settings.get("min_improvement", 0.0)),
        "anchor_fallback_min_service_level_delta": settings.get(
            "min_service_level_delta",
            "",
        ),
    }
    metadata.update(
        anchor_fallback_candidate_diagnostics(
            learned_candidates,
            anchor_summary,
            settings,
        )
    )
    print(
        "anchor_fallback "
        f"algorithm={algorithm} selected={decision} anchor={anchor_policy} "
        f"residual_scale={deployed_residual_scale:g} "
        f"candidate_scale={selected_residual_scale:g} "
        f"learned_cost={float(learned_summary['total_cost_mean']):.3f} "
        f"anchor_cost={float(anchor_summary['total_cost_mean']):.3f}",
        flush=True,
    )
    return selected_agent, metadata


def anchor_fallback_settings(config: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config.get("anchor_fallback", {}))
    settings.update(dict(budget.get("anchor_fallback", {})))
    return settings


def residual_deployment_scale_candidates(
    settings: dict[str, Any],
    *,
    has_residual_scale: bool,
) -> tuple[float, ...]:
    if not has_residual_scale:
        return (1.0,)
    raw = settings.get(
        "deployment_scale_candidates",
        settings.get("residual_scale_candidates", (1.0,)),
    )
    if isinstance(raw, str):
        raw = raw.replace("|", ",").split(",")
    try:
        candidates = [float(value) for value in raw]
    except TypeError:
        candidates = [float(raw)]
    cleaned: list[float] = []
    for value in candidates:
        if not np.isfinite(value):
            continue
        clipped = max(float(value), 0.0)
        if clipped not in cleaned:
            cleaned.append(clipped)
    return tuple(cleaned or [1.0])


def select_residual_deployment_candidate(
    candidates: list[dict[str, Any]],
    anchor_summary: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if not candidates:
        raise ValueError("At least one residual deployment candidate is required")
    min_service_level_delta = (
        float(settings["min_service_level_delta"])
        if "min_service_level_delta" in settings
        else None
    )
    feasible = []
    for candidate in candidates:
        summary = candidate["summary"]
        decision = select_anchor_fallback_policy(
            float(summary["total_cost_mean"]),
            float(anchor_summary["total_cost_mean"]),
            min_improvement=float(settings.get("min_improvement", 0.0)),
            learned_service_level=float(summary.get("service_level_mean", "nan")),
            anchor_service_level=float(anchor_summary.get("service_level_mean", "nan")),
            min_service_level_delta=min_service_level_delta,
        )
        if decision == "learned" and float(candidate["residual_scale"]) > 1e-12:
            feasible.append(candidate)
    if feasible:
        return min(feasible, key=lambda item: float(item["summary"]["total_cost_mean"])), "learned"
    return min(candidates, key=lambda item: float(item["summary"]["total_cost_mean"])), "anchor"


def anchor_fallback_candidate_diagnostics(
    candidates: list[dict[str, Any]],
    anchor_summary: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    anchor_cost = float(anchor_summary.get("total_cost_mean", float("nan")))
    anchor_service = float(anchor_summary.get("service_level_mean", float("nan")))
    min_service_level_delta = (
        float(settings["min_service_level_delta"])
        if "min_service_level_delta" in settings
        else None
    )
    diagnostics: list[dict[str, Any]] = []
    for candidate in candidates:
        scale = float(candidate["residual_scale"])
        summary = candidate["summary"]
        candidate_cost = float(summary.get("total_cost_mean", float("nan")))
        candidate_service = float(summary.get("service_level_mean", float("nan")))
        gate_decision = select_anchor_fallback_policy(
            candidate_cost,
            anchor_cost,
            min_improvement=float(settings.get("min_improvement", 0.0)),
            learned_service_level=candidate_service,
            anchor_service_level=anchor_service,
            min_service_level_delta=min_service_level_delta,
        )
        diagnostics.append(
            {
                "scale": scale,
                "decision": "anchor_equivalent" if scale <= 1e-12 else gate_decision,
                "cost": candidate_cost,
                "service": candidate_service,
                "eligibility": summary_float(
                    summary,
                    "eligibility_rate_mean_mean",
                    "eligibility_rate_mean",
                ),
                "patients_lost": summary_float(summary, "patients_lost_mean"),
                "at_risk_unserved": summary_float(summary, "at_risk_unserved_mean"),
                "cost_gap_pct": percentage_gap(candidate_cost, anchor_cost),
                "service_gap": candidate_service - anchor_service,
            }
        )
    best_nonzero = min(
        (row for row in diagnostics if float(row["scale"]) > 1e-12),
        key=lambda row: float(row["cost"]),
        default=None,
    )
    metadata = {
        "anchor_fallback_validation_candidate_scales": pipe_join(
            row["scale"] for row in diagnostics
        ),
        "anchor_fallback_validation_candidate_decisions": pipe_join(
            row["decision"] for row in diagnostics
        ),
        "anchor_fallback_validation_candidate_cost_means": pipe_join(
            row["cost"] for row in diagnostics
        ),
        "anchor_fallback_validation_candidate_service_level_means": pipe_join(
            row["service"] for row in diagnostics
        ),
        "anchor_fallback_validation_candidate_eligibility_rate_means": pipe_join(
            row["eligibility"] for row in diagnostics
        ),
        "anchor_fallback_validation_candidate_patients_lost_means": pipe_join(
            row["patients_lost"] for row in diagnostics
        ),
        "anchor_fallback_validation_candidate_at_risk_unserved_means": pipe_join(
            row["at_risk_unserved"] for row in diagnostics
        ),
        "anchor_fallback_validation_candidate_cost_gap_pct": pipe_join(
            row["cost_gap_pct"] for row in diagnostics
        ),
        "anchor_fallback_validation_candidate_service_gap": pipe_join(
            row["service_gap"] for row in diagnostics
        ),
    }
    if best_nonzero is None:
        metadata.update(
            {
                "anchor_fallback_validation_best_nonzero_scale": "",
                "anchor_fallback_validation_best_nonzero_decision": "",
                "anchor_fallback_validation_best_nonzero_cost_mean": "",
                "anchor_fallback_validation_best_nonzero_service_level_mean": "",
                "anchor_fallback_validation_best_nonzero_cost_gap_pct": "",
                "anchor_fallback_validation_best_nonzero_service_gap": "",
            }
        )
    else:
        metadata.update(
            {
                "anchor_fallback_validation_best_nonzero_scale": best_nonzero["scale"],
                "anchor_fallback_validation_best_nonzero_decision": best_nonzero["decision"],
                "anchor_fallback_validation_best_nonzero_cost_mean": best_nonzero["cost"],
                "anchor_fallback_validation_best_nonzero_service_level_mean": best_nonzero[
                    "service"
                ],
                "anchor_fallback_validation_best_nonzero_cost_gap_pct": best_nonzero[
                    "cost_gap_pct"
                ],
                "anchor_fallback_validation_best_nonzero_service_gap": best_nonzero[
                    "service_gap"
                ],
            }
        )
    return metadata


def summary_float(summary: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in summary and summary[key] != "":
            return float(summary[key])
    return float("nan")


def percentage_gap(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) <= 1e-12:
        return float("nan")
    return 100.0 * (float(value) - float(reference)) / abs(float(reference))


def pipe_join(values: Iterable[Any]) -> str:
    return "|".join(format_metadata_value(value) for value in values)


def format_metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.12g}"


def copy_residual_scale_vector(agent) -> np.ndarray | None:
    scale_vector = getattr(agent, "residual_scale_vector", None)
    if scale_vector is None:
        return None
    return np.asarray(scale_vector, dtype=np.float32).copy()


def set_residual_scale_vector(agent, scale_vector: np.ndarray) -> None:
    agent.residual_scale_vector = np.asarray(scale_vector, dtype=np.float32).copy()


def select_anchor_fallback_policy(
    learned_cost: float,
    anchor_cost: float,
    *,
    min_improvement: float = 0.0,
    learned_service_level: float | None = None,
    anchor_service_level: float | None = None,
    min_service_level_delta: float | None = None,
) -> str:
    """Return ``learned`` only if it clears cost and patient-facing safeguards."""

    if not np.isfinite(learned_cost):
        return "anchor"
    if not np.isfinite(anchor_cost):
        return "learned"
    threshold = float(anchor_cost) * (1.0 - float(min_improvement))
    if float(learned_cost) > threshold:
        return "anchor"
    if min_service_level_delta is not None:
        learned_service = float(
            learned_service_level if learned_service_level is not None else float("nan")
        )
        anchor_service = float(
            anchor_service_level if anchor_service_level is not None else float("nan")
        )
        if np.isfinite(learned_service) and np.isfinite(anchor_service):
            service_threshold = anchor_service + float(min_service_level_delta)
            if learned_service < service_threshold:
                return "anchor"
    return "learned"


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def make_evaluation_config(
    plan: dict[str, Any],
    budget_name: str,
    budget: dict[str, Any],
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    if algorithm in plan["learned_config_paths"]:
        return make_training_config(plan, budget_name, budget, algorithm, scenario, seed)
    config = {
        "algorithm": algorithm,
        "seed": int(seed),
        "max_steps_per_episode": int(budget["max_steps_per_episode"]),
        "env": make_scenario_env_config(plan, algorithm, scenario),
    }
    return _deep_update(config, algorithm_config_overrides(plan, algorithm))


def make_scenario_env_config(
    plan: dict[str, Any],
    algorithm: str,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Merge benchmark-wide environment defaults with a scenario override.

    Learned configs often carry common 20-clinic dynamics that scenario files
    only partially override. Heuristics must see the same merged environment,
    otherwise the benchmark compares learned policies against baselines running
    on an easier scenario definition.
    """

    reference_algorithm = algorithm
    if algorithm not in plan["learned_config_paths"]:
        reference_algorithm = str(plan.get("heuristic_env_reference_algorithm", ""))
        if not reference_algorithm:
            reference_algorithm = next(iter(plan["learned_config_paths"]))
    if reference_algorithm not in plan["learned_config_paths"]:
        raise ValueError(
            "heuristic_env_reference_algorithm must name a learned config, "
            f"got {reference_algorithm!r}"
        )

    reference_config = load_config(plan["learned_config_paths"][reference_algorithm])
    scenario_env = load_config(scenario["env_config"])
    base_env = dict(reference_config.get("env", {}))
    graph_ablation = base_env.get("graph_ablation", scenario_env.get("graph_ablation", "full_graph"))
    env = dict(base_env)
    env.update(scenario_env)
    env["graph_ablation"] = graph_ablation
    return env


def final_checkpoint_path(
    plan: dict[str, Any],
    budget_name: str,
    budget: dict[str, Any],
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
) -> Path:
    if bool(local_search_settings(plan, budget, algorithm).get("enabled", False)):
        return local_search_checkpoint_path(plan, budget_name, algorithm, scenario, seed)
    episode = int(budget["num_episodes"])
    return checkpoint_dir_path(plan, budget_name, algorithm, scenario, seed) / (
        f"{algorithm}_seed{int(seed)}_episode{episode}.pt"
    )


def local_search_checkpoint_path(
    plan: dict[str, Any],
    budget_name: str,
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
) -> Path:
    return checkpoint_dir_path(plan, budget_name, algorithm, scenario, seed) / (
        f"{algorithm}_seed{int(seed)}_local_search.pt"
    )


def post_training_summary_path(
    plan: dict[str, Any],
    budget_name: str,
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
) -> Path:
    return (
        benchmark_root(plan, budget_name)
        / "post_training"
        / scenario["name"]
        / f"{algorithm}_seed{int(seed)}_local_search_summary.csv"
    )


def training_outputs_complete(
    plan: dict[str, Any],
    budget_name: str,
    budget: dict[str, Any],
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
) -> bool:
    return (
        training_csv_path(plan, budget_name, algorithm, scenario, seed).exists()
        and config_snapshot_path(plan, budget_name, algorithm, scenario, seed).exists()
        and final_checkpoint_path(plan, budget_name, budget, algorithm, scenario, seed).exists()
    )


def evaluation_outputs_complete(
    plan: dict[str, Any],
    budget_name: str,
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
) -> bool:
    return (
        evaluation_csv_path(plan, budget_name, algorithm, scenario, seed).exists()
        and evaluation_summary_path(plan, budget_name, algorithm, scenario, seed).exists()
    )


def benchmark_root(plan: dict[str, Any], budget_name: str) -> Path:
    return Path(plan.get("output_root", "results/full_benchmark")) / budget_name


def figure_root(plan: dict[str, Any], budget_name: str) -> Path:
    return Path(plan.get("figure_root", "figures/full_benchmark")) / budget_name


def training_csv_path(plan: dict[str, Any], budget_name: str, algorithm: str, scenario: dict[str, Any], seed: int) -> Path:
    return benchmark_root(plan, budget_name) / "training" / scenario["name"] / f"{algorithm}_seed{int(seed)}.csv"


def evaluation_csv_path(
    plan: dict[str, Any],
    budget_name: str,
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
) -> Path:
    return benchmark_root(plan, budget_name) / "evaluation" / scenario["name"] / f"{algorithm}_seed{int(seed)}.csv"


def evaluation_summary_path(
    plan: dict[str, Any],
    budget_name: str,
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
) -> Path:
    return (
        benchmark_root(plan, budget_name)
        / "evaluation"
        / scenario["name"]
        / f"{algorithm}_seed{int(seed)}_summary.csv"
    )


def checkpoint_dir_path(
    plan: dict[str, Any],
    budget_name: str,
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
) -> Path:
    return benchmark_root(plan, budget_name) / "checkpoints" / scenario["name"] / f"{algorithm}_seed{int(seed)}"


def config_snapshot_path(
    plan: dict[str, Any],
    budget_name: str,
    algorithm: str,
    scenario: dict[str, Any],
    seed: int,
) -> Path:
    return benchmark_root(plan, budget_name) / "configs" / scenario["name"] / f"{algorithm}_seed{int(seed)}.json"


def print_dry_run(
    plan: dict[str, Any],
    budget_name: str,
    budget: dict[str, Any],
    algorithms: Iterable[str],
    scenarios: Iterable[dict[str, Any]],
    seeds: Iterable[int],
) -> None:
    algorithms = tuple(algorithms)
    scenarios = tuple(scenarios)
    seeds = tuple(seeds)
    learned = [algorithm for algorithm in algorithms if algorithm in plan["learned_config_paths"]]
    heuristics = [algorithm for algorithm in algorithms if algorithm in available_heuristics()]
    train_jobs = len(learned) * len(scenarios) * len(seeds)
    eval_jobs = len(algorithms) * len(scenarios) * len(seeds)
    print(f"plan={plan.get('name', 'benchmark')} budget={budget_name}", flush=True)
    print(f"algorithms={', '.join(algorithms)}", flush=True)
    print(f"scenarios={', '.join(scenario['name'] for scenario in scenarios)}", flush=True)
    print(f"seeds={', '.join(str(seed) for seed in seeds)}", flush=True)
    print(f"learned={', '.join(learned)}", flush=True)
    print(f"heuristics={', '.join(heuristics)}", flush=True)
    print(f"train_jobs={train_jobs}", flush=True)
    print(f"evaluation_jobs={eval_jobs}", flush=True)
    print(f"episodes_per_learned_job={budget['num_episodes']}", flush=True)
    print(f"max_steps_per_episode={budget['max_steps_per_episode']}", flush=True)
    print(f"mc_replications_per_eval_job={budget['evaluation_replications']}", flush=True)
    print(f"output_root={benchmark_root(plan, budget_name)}", flush=True)


if __name__ == "__main__":
    main()
