"""Formal Monte Carlo evaluation for learned and heuristic policies."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.heuristics import available_heuristics
from src.rl.agents import available_algorithms, get_agent_class
from src.rl.config import load_config
from src.rl.experiment import EpisodeMetrics, build_env, write_rows


SUMMARY_METRICS = (
    "total_reward",
    "total_cost",
    "service_level",
    "average_waiting_time",
    "reagent_shortage_frequency",
    "bioreactor_shortage_frequency",
    "bioreactor_utilization",
    "transshipment_count",
    "transshipment_cost",
    "average_inference_ms",
)

# Summarized only when present in the rows (patient-condition env).
PATIENT_METRICS = (
    "eligibility_rate",
    "eligibility_rate_mean",
    "patients_lost",
    "patients_lost_ineligible",
    "patients_lost_expired",
    "material_wasted",
    "at_risk_unserved",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=available_algorithms(), required=True)
    parser.add_argument("--config", default=None, help="Algorithm config containing an env block.")
    parser.add_argument("--env-config", default=None, help="Pure environment scenario config.")
    parser.add_argument("--checkpoint", default=None, help="Required for learned policies.")
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    args = parser.parse_args()

    config = load_evaluation_config(args)
    env = build_env(config, seed=args.seed)
    agent_cls = get_agent_class(args.algorithm)
    agent = agent_cls(env.observation_size, env.action_size, config)
    if args.algorithm not in available_heuristics():
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for learned-policy formal evaluation")
        agent.load_actor(args.checkpoint)

    rows = evaluate_agent(
        agent,
        env,
        algorithm=args.algorithm,
        seed=args.seed,
        replications=args.replications,
        max_steps=args.max_steps,
    )
    output = Path(args.output or f"results/formal_{args.algorithm}.csv")
    write_rows(rows, output)
    summary = summarize_rows(rows)
    summary_output = Path(args.summary_output or f"results/formal_{args.algorithm}_summary.csv")
    write_summary(summary, summary_output)
    print(f"wrote {len(rows)} replications to {output}")
    print(f"wrote summary to {summary_output}")


def load_evaluation_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.config and args.env_config:
        raise SystemExit("Use either --config or --env-config, not both")
    if args.config:
        config = load_config(args.config)
    elif args.env_config:
        env_config = load_config(args.env_config)
        config = {"algorithm": args.algorithm, "env": env_config}
    else:
        raise SystemExit("Provide --config or --env-config")
    config = dict(config)
    config["algorithm"] = args.algorithm
    config.setdefault("seed", args.seed)
    return config


def evaluate_agent(
    agent,
    env,
    *,
    algorithm: str,
    seed: int,
    replications: int,
    max_steps: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_steps = int(max_steps or env.config.episode_horizon)
    for replication in range(int(replications)):
        state = env.reset(seed=seed + replication)
        agent.reset()
        metrics = EpisodeMetrics()
        total_reward = 0.0
        inference_seconds = 0.0
        done = False
        step = 0
        while not done and step < max_steps:
            started = time.perf_counter()
            action = agent.select_action(state, explore=False, env=env)
            inference_seconds += time.perf_counter() - started
            state, reward, done, info = env.step(action)
            metrics.update(info)
            total_reward += float(reward)
            step += 1

        row = {
            "algorithm": algorithm,
            "seed": seed,
            "replication": replication,
            "scenario": getattr(env, "scenario_name", "default"),
            "graph_ablation": getattr(env, "graph_ablation", "full_graph"),
            "steps": step,
            "total_reward": total_reward,
            "total_cost": metrics.total_cost,
            "service_level": metrics.service_level,
            "average_waiting_time": metrics.average_waiting_time,
            "reagent_shortage_frequency": metrics.reagent_shortage_frequency,
            "bioreactor_shortage_frequency": metrics.bioreactor_shortage_frequency,
            "bioreactor_utilization": metrics.bioreactor_utilization,
            "transshipment_count": metrics.transshipment_count,
            "transshipment_cost": metrics.transshipment_cost,
            "average_inference_ms": 1000.0 * inference_seconds / max(step, 1),
        }
        if metrics.has_patient_metrics:
            row.update(
                {
                    "eligibility_rate": metrics.eligibility_rate_last,
                    "eligibility_rate_mean": metrics.eligibility_rate_mean,
                    "patients_lost": metrics.patients_lost,
                    "patients_lost_ineligible": metrics.patients_lost_ineligible,
                    "patients_lost_expired": metrics.patients_lost_expired,
                    "material_wasted": metrics.material_wasted,
                    "at_risk_unserved": metrics.at_risk_unserved,
                }
            )
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    summary: dict[str, Any] = {
        "algorithm": rows[0]["algorithm"],
        "scenario": rows[0]["scenario"],
        "graph_ablation": rows[0]["graph_ablation"],
        "replications": len(rows),
    }
    metrics_to_summarize = list(SUMMARY_METRICS)
    metrics_to_summarize += [m for m in PATIENT_METRICS if m in rows[0]]
    for metric in metrics_to_summarize:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary[f"{metric}_sem"] = (
            float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        )
    return summary


def write_summary(summary: dict[str, Any], path: str | Path) -> None:
    if not summary:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()

