"""Probe calibration knobs for the graph shield selector.

The probe trains one selector per class-weight setting, then evaluates the same
trained selector under several confidence fallback thresholds. It is intended
for quick method development before promoting a setting to the formal benchmark
plan.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from evaluation.aggregate_results import write_rows as write_summary_rows
from evaluation.evaluate_formal import evaluate_agent, summarize_rows, write_summary
from evaluation.run_full_benchmark import (
    make_training_config,
    load_benchmark_plan,
    resolve_budget,
    select_scenarios,
)
from src.baselines.heuristics import get_heuristic_class
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env, train_off_policy_agent, write_rows


DEFAULT_OUTPUT_ROOT = "results/selector_calibration_probe"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="experiments/configs/residual_policy_benchmark.json")
    parser.add_argument("--budget", default="mini_pilot")
    parser.add_argument("--scenario", default="patient_condition_geo")
    parser.add_argument("--algorithm", default="gcn_mdl2_shield_selector")
    parser.add_argument("--anchor", default="mdl2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--class-weight-powers", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--pretrain-episodes", type=int, default=20)
    parser.add_argument("--pretrain-max-steps", type=int, default=12)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--eval-replications", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=910000)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--include-anchor", action="store_true")
    args = parser.parse_args()

    plan = load_benchmark_plan(args.plan)
    budget = resolve_budget(plan, args.budget)
    budget["num_episodes"] = int(args.train_episodes)
    budget["evaluation_replications"] = int(args.eval_replications)
    scenario = select_scenarios(plan, [args.scenario])[0]
    output_root = Path(args.output_root)
    summary_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        if args.include_anchor:
            summary_rows.append(_evaluate_anchor(args, plan, budget, scenario, seed, output_root))
        for class_weight_power in args.class_weight_powers:
            agent, training_summary = _train_selector(
                args,
                plan,
                budget,
                scenario,
                seed,
                class_weight_power,
                output_root,
            )
            for threshold in args.thresholds:
                rows = _evaluate_selector(
                    args,
                    budget,
                    scenario,
                    seed,
                    class_weight_power,
                    threshold,
                    agent,
                    output_root,
                )
                summary = summarize_rows(rows)
                summary.update(training_summary)
                summary["class_weight_power"] = class_weight_power
                summary["confidence_threshold"] = threshold
                summary["training_seed"] = seed
                write_summary(
                    summary,
                    output_root
                    / args.scenario
                    / f"{args.algorithm}_seed{seed}_cw{_tag(class_weight_power)}_th{_tag(threshold)}_summary.csv",
                )
                summary_rows.append(summary)

    write_summary_rows(summary_rows, output_root / args.scenario / "calibration_summary.csv")
    print(f"wrote {len(summary_rows)} calibration rows to {output_root / args.scenario / 'calibration_summary.csv'}")


def _train_selector(
    args: argparse.Namespace,
    plan: dict[str, Any],
    budget: dict[str, Any],
    scenario: dict[str, Any],
    seed: int,
    class_weight_power: float,
    output_root: Path,
):
    config = make_training_config(plan, args.budget, budget, args.algorithm, scenario, seed)
    variant = f"{args.algorithm}_cw{_tag(class_weight_power)}"
    config["algorithm"] = variant
    config["checkpoint_dir"] = str(output_root / args.scenario / "checkpoints" / f"{variant}_seed{seed}")
    config["result_csv_path"] = str(output_root / args.scenario / "training" / f"{variant}_seed{seed}.csv")
    config["config_snapshot_path"] = str(output_root / args.scenario / "configs" / f"{variant}_seed{seed}.json")
    pretrain = dict(config.get("imitation_pretrain", {}))
    pretrain["enabled"] = True
    pretrain["episodes"] = int(args.pretrain_episodes)
    pretrain["max_steps_per_episode"] = int(args.pretrain_max_steps)
    pretrain["epochs"] = int(args.pretrain_epochs)
    pretrain["class_weighting"] = class_weight_power > 0.0
    pretrain["class_weight_power"] = float(class_weight_power)
    config["imitation_pretrain"] = pretrain

    env = build_env(config, seed=seed)
    agent = get_agent_class(args.algorithm)(env.observation_size, env.action_size, config)
    training_rows = train_off_policy_agent(agent, env, config)
    write_rows(training_rows, config["result_csv_path"])
    first_row = training_rows[0] if training_rows else {}
    return agent, {
        "pretrain_policy": first_row.get("pretrain_policy", ""),
        "pretrain_samples": first_row.get("pretrain_samples", ""),
        "pretrain_final_loss": first_row.get("pretrain_final_loss", ""),
        "pretrain_train_accuracy": first_row.get("pretrain_train_accuracy", ""),
        "pretrain_changed_fraction": first_row.get("pretrain_changed_fraction", ""),
        "pretrain_anchor_label_fraction": first_row.get("pretrain_anchor_label_fraction", ""),
        "pretrain_non_anchor_prediction_fraction": first_row.get(
            "pretrain_non_anchor_prediction_fraction",
            "",
        ),
        "pretrain_candidate_count": first_row.get("pretrain_candidate_count", ""),
    }


def _evaluate_selector(
    args: argparse.Namespace,
    budget: dict[str, Any],
    scenario: dict[str, Any],
    seed: int,
    class_weight_power: float,
    threshold: float,
    agent,
    output_root: Path,
) -> list[dict[str, Any]]:
    eval_config = copy.deepcopy(agent.env_config)
    config = {
        "algorithm": f"{args.algorithm}_cw{_tag(class_weight_power)}_th{_tag(threshold)}",
        "env": eval_config,
    }
    env = build_env(config, seed=seed)
    agent.confidence_threshold = float(threshold)
    rows = evaluate_agent(
        agent,
        env,
        algorithm=config["algorithm"],
        seed=int(args.eval_seed) + int(seed) * 10000,
        replications=int(budget["evaluation_replications"]),
        max_steps=int(budget["max_steps_per_episode"]),
    )
    output_path = (
        output_root
        / scenario["name"]
        / "evaluation"
        / f"{config['algorithm']}_seed{seed}.csv"
    )
    write_rows(rows, output_path)
    return rows


def _evaluate_anchor(
    args: argparse.Namespace,
    plan: dict[str, Any],
    budget: dict[str, Any],
    scenario: dict[str, Any],
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    config = make_training_config(plan, args.budget, budget, args.algorithm, scenario, seed)
    config["algorithm"] = args.anchor
    env = build_env(config, seed=seed)
    policy_class = get_heuristic_class(args.anchor)
    agent = policy_class(env.observation_size, env.action_size, config)
    rows = evaluate_agent(
        agent,
        env,
        algorithm=args.anchor,
        seed=int(args.eval_seed) + int(seed) * 10000,
        replications=int(budget["evaluation_replications"]),
        max_steps=int(budget["max_steps_per_episode"]),
    )
    output_path = output_root / scenario["name"] / "evaluation" / f"{args.anchor}_seed{seed}.csv"
    write_rows(rows, output_path)
    summary = summarize_rows(rows)
    summary["class_weight_power"] = ""
    summary["confidence_threshold"] = ""
    summary["training_seed"] = seed
    return summary


def _tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
