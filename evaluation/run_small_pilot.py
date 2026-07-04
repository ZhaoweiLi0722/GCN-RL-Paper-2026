"""Run a small multi-seed pilot across learned agents and heuristics.

The pilot is for pipeline validation and coarse stability checks only. It is
not a substitute for full training budgets or 500-replication evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evaluation.evaluate_formal import evaluate_agent, summarize_rows, write_summary
from evaluation.run_smoke_comparison import _smoke_config
from src.baselines.heuristics import available_heuristics
from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env, train_off_policy_agent


DEFAULT_ALGORITHMS = ("gcn_ddpg", "flat_ddpg", "td3", "myo", "iso", "mdl1", "mdl2")
DEFAULT_CONFIGS = {
    "flat_ddpg": "configs/flat_ddpg_20_clinic.yaml",
    "gcn_ddpg": "configs/gcn_ddpg_20_clinic.yaml",
    "td3": "configs/td3_20_clinic.yaml",
    "myo": "experiments/configs/20_clinic_disruption_0_3.json",
    "iso": "experiments/configs/20_clinic_disruption_0_3.json",
    "mdl1": "experiments/configs/20_clinic_disruption_0_3.json",
    "mdl2": "experiments/configs/20_clinic_disruption_0_3.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heuristic-replications", type=int, default=2)
    parser.add_argument("--output", default="results/small_pilot.csv")
    parser.add_argument("--summary-output", default="results/small_pilot_summary.csv")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for algorithm in args.algorithms:
        for seed in args.seeds:
            config = _load_pilot_config(
                algorithm,
                seed=seed,
                episodes=args.episodes,
                steps=args.steps,
                batch_size=args.batch_size,
            )
            env = build_env(config, seed=seed)
            agent = get_agent_class(algorithm)(env.observation_size, env.action_size, config)
            if algorithm in available_heuristics():
                eval_rows = evaluate_agent(
                    agent,
                    env,
                    algorithm=algorithm,
                    seed=seed,
                    replications=args.heuristic_replications,
                    max_steps=args.steps,
                )
                rows.extend(eval_rows)
                summary_rows.append(summarize_rows(eval_rows))
            else:
                train_rows = train_off_policy_agent(agent, env, config)
                rows.extend(train_rows)

    _write_union_rows(rows, args.output)
    if summary_rows:
        _write_summary_rows(summary_rows, args.summary_output)
    print(f"wrote pilot rows to {args.output}")
    if summary_rows:
        print(f"wrote heuristic pilot summaries to {args.summary_output}")


def _load_pilot_config(
    algorithm: str,
    *,
    seed: int,
    episodes: int,
    steps: int,
    batch_size: int,
) -> dict[str, Any]:
    config = load_config(DEFAULT_CONFIGS[algorithm])
    if "env" not in config:
        config = {"algorithm": algorithm, "env": config}
    config = _smoke_config(
        config,
        algorithm=algorithm,
        seed=seed,
        episodes=episodes,
        steps=steps,
        batch_size=batch_size,
    )
    config["checkpoint_dir"] = f"checkpoints/{algorithm}_pilot_seed{seed}"
    config["result_csv_path"] = f"results/{algorithm}_pilot_seed{seed}.csv"
    return config


def _write_summary_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    clean_rows = [row for row in rows if row]
    if not clean_rows:
        return
    _write_union_rows(clean_rows, path)


def _write_union_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    import csv

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
