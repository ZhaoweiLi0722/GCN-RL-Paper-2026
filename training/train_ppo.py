"""Train the PPO baseline."""

from __future__ import annotations

import argparse

from src.baselines.ppo import PPOAgent
from src.rl.config import load_config, save_config_snapshot
from src.rl.experiment import build_env, train_off_policy_agent, write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_20_clinic.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    env = build_env(config, seed=int(config.get("seed", 0)))
    try:
        agent = PPOAgent(env.observation_size, env.action_size, config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    rows = train_off_policy_agent(agent, env, config)
    write_rows(rows, config.get("result_csv_path", "results/ppo_training.csv"))
    save_config_snapshot(config, config.get("config_snapshot_path", "results/ppo_config_snapshot.json"))


if __name__ == "__main__":
    main()
