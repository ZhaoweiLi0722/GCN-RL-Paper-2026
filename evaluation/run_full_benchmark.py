"""Run the manuscript-facing benchmark matrix.

The default action is a dry run. Use ``--phase all`` with the ``smoke`` budget
first, then ``pilot`` for stability checks, and only then ``full``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

from evaluation.aggregate_results import (
    aggregate_rows,
    read_rows as read_aggregate_inputs,
    write_rows as write_aggregate_rows,
)
from evaluation.evaluate_formal import evaluate_agent, summarize_rows, write_summary
from evaluation.plot_results import _plot_grouped_bars, _read_rows as read_plot_rows
from evaluation.run_gcn_residual_sweep import run_local_search_distillation
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
                rows = train_off_policy_agent(agent, env, config)
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
                if algorithm not in available_heuristics():
                    checkpoint = final_checkpoint_path(plan, budget_name, budget, algorithm, scenario, seed)
                    if not checkpoint.exists():
                        raise SystemExit(
                            f"Missing checkpoint for {algorithm} scenario={scenario['name']} seed={seed}: "
                            f"{checkpoint}"
                        )
                    agent.load_actor(checkpoint)

                evaluation_seed = int(budget["evaluation_seed"]) + int(seed) * LEARNED_EVALUATION_SEED_STRIDE
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
                output_path = evaluation_csv_path(plan, budget_name, algorithm, scenario, seed)
                write_rows(rows, output_path)

                summary = summarize_rows(rows)
                if summary:
                    summary["benchmark"] = plan.get("name", "benchmark")
                    summary["budget"] = budget_name
                    summary["training_seed"] = seed
                    summary["evaluation_seed"] = evaluation_seed
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
    scenario_env = load_config(scenario["env_config"])
    base_env = dict(config.get("env", {}))
    graph_ablation = base_env.get("graph_ablation", scenario_env.get("graph_ablation", "full_graph"))
    env = dict(base_env)
    env.update(scenario_env)
    env["graph_ablation"] = graph_ablation

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

    if algorithm == "ppo":
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


def algorithm_config_overrides(plan: dict[str, Any], algorithm: str) -> dict[str, Any]:
    settings = dict(plan.get("algorithm_settings", {}).get(algorithm, {}))
    return dict(settings.get("config_overrides", {}))


def local_search_settings(plan: dict[str, Any], budget: dict[str, Any], algorithm: str) -> dict[str, Any]:
    settings = dict(plan.get("algorithm_settings", {}).get(algorithm, {}).get("local_search", {}))
    budget_settings = dict(budget.get("local_search", {}).get(algorithm, {}))
    settings.update(budget_settings)
    return settings


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
    return {
        "algorithm": algorithm,
        "seed": int(seed),
        "max_steps_per_episode": int(budget["max_steps_per_episode"]),
        "env": load_config(scenario["env_config"]),
    }


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
