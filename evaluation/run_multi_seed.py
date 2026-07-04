"""Run a configured baseline for multiple random seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.evaluate_formal import evaluate_agent
from src.baselines.heuristics import available_heuristics
from src.rl.agents import available_algorithms, get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env, train_off_policy_agent, write_rows


DEFAULT_CONFIGS = {
    "flat_ddpg": "configs/flat_ddpg.yaml",
    "gcn_ddpg": "configs/gcn_ddpg_20_clinic.yaml",
    "iso": "experiments/configs/20_clinic_disruption_0_3.json",
    "mdl1": "experiments/configs/20_clinic_disruption_0_3.json",
    "mdl2": "experiments/configs/20_clinic_disruption_0_3.json",
    "myo": "experiments/configs/20_clinic_disruption_0_3.json",
    "td3": "configs/td3_20_clinic.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=available_algorithms(), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--heuristic-replications", type=int, default=1)
    args = parser.parse_args()

    config_path = args.config or DEFAULT_CONFIGS[args.algorithm]
    base_config = load_config(config_path)
    if "env" not in base_config:
        base_config = {"algorithm": args.algorithm, "env": base_config}
    rows = []

    for seed in args.seeds:
        config = dict(base_config)
        config["seed"] = seed
        config["result_csv_path"] = f"results/{args.algorithm}_seed{seed}.csv"
        env = build_env(config, seed=seed)
        try:
            agent = get_agent_class(args.algorithm)(env.observation_size, env.action_size, config)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if args.algorithm in available_heuristics():
            rows.extend(
                evaluate_agent(
                    agent,
                    env,
                    algorithm=args.algorithm,
                    seed=seed,
                    replications=args.heuristic_replications,
                    max_steps=int(config.get("max_steps_per_episode", env.config.episode_horizon)),
                )
            )
        else:
            rows.extend(train_off_policy_agent(agent, env, config))

    output = Path(args.output or f"results/{args.algorithm}_multi_seed.csv")
    write_rows(rows, output)


if __name__ == "__main__":
    main()
