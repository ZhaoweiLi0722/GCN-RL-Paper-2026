"""Run a tiny GCN-DDPG vs flat DDPG smoke comparison.

This script is intentionally for pipeline validation only. Its output should
not be interpreted as an experimental result for the manuscript.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env, train_off_policy_agent, write_rows


DEFAULT_RUNS = (
    ("flat_ddpg", "configs/flat_ddpg_20_clinic.yaml"),
    ("gcn_ddpg", "configs/gcn_ddpg_20_clinic.yaml"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", default="results/smoke_gcn_vs_flat.csv")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for algorithm, config_path in DEFAULT_RUNS:
        config = _smoke_config(
            load_config(config_path),
            algorithm=algorithm,
            seed=args.seed,
            episodes=args.episodes,
            steps=args.steps,
            batch_size=args.batch_size,
        )
        env = build_env(config, seed=args.seed)
        try:
            agent = get_agent_class(algorithm)(env.observation_size, env.action_size, config)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        rows.extend(train_off_policy_agent(agent, env, config))

    output = Path(args.output)
    write_rows(rows, output)
    print(f"wrote {len(rows)} rows to {output}")


def _smoke_config(
    config: dict[str, Any],
    *,
    algorithm: str,
    seed: int,
    episodes: int,
    steps: int,
    batch_size: int,
) -> dict[str, Any]:
    smoke = dict(config)
    smoke["algorithm"] = algorithm
    smoke["seed"] = seed
    smoke["num_episodes"] = episodes
    smoke["max_steps_per_episode"] = steps
    smoke["batch_size"] = batch_size
    smoke["replay_buffer_size"] = max(100, batch_size * max(episodes, 1) * max(steps, 1))
    smoke["checkpoint_interval"] = max(episodes + 1, 2)
    smoke["result_csv_path"] = f"results/{algorithm}_smoke.csv"
    smoke["config_snapshot_path"] = f"results/{algorithm}_smoke_config_snapshot.json"
    env_config = dict(smoke.get("env", {}))
    env_config["episode_horizon"] = max(steps, 1)
    smoke["env"] = env_config
    return smoke


if __name__ == "__main__":
    main()
