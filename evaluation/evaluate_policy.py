"""Evaluate a trained deterministic actor checkpoint."""

from __future__ import annotations

import argparse

from src.baselines.flat_ddpg import FlatDDPGAgent
from src.baselines.td3 import TD3Agent
from src.rl.config import load_config
from src.rl.experiment import EpisodeMetrics, build_env, write_rows


AGENTS = {
    "flat_ddpg": FlatDDPGAgent,
    "td3": TD3Agent,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=sorted(AGENTS), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", default="results/evaluation.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    env = build_env(config, seed=int(config.get("seed", 0)))
    try:
        agent = AGENTS[args.algorithm](env.observation_size, env.action_size, config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    agent.load_actor(args.checkpoint)

    rows = []
    max_steps = int(config.get("max_steps_per_episode", env.config.episode_horizon))
    for episode in range(args.episodes):
        state = env.reset(seed=int(config.get("seed", 0)) + episode)
        metrics = EpisodeMetrics()
        total_reward = 0.0
        done = False
        step = 0
        while not done and step < max_steps:
            action = agent.select_action(state, explore=False, env=env)
            state, reward, done, info = env.step(action)
            metrics.update(info)
            total_reward += float(reward)
            step += 1
        rows.append(
            {
                "algorithm": args.algorithm,
                "seed": int(config.get("seed", 0)),
                "scenario": getattr(env, "scenario_name", "default"),
                "graph_ablation": getattr(env, "graph_ablation", "full_graph"),
                "episode": episode,
                "total_reward": total_reward,
                "total_cost": metrics.total_cost,
                "service_level": metrics.service_level,
                "average_waiting_time": metrics.average_waiting_time,
                "reagent_shortage_frequency": metrics.reagent_shortage_frequency,
                "bioreactor_shortage_frequency": metrics.bioreactor_shortage_frequency,
                "bioreactor_utilization": metrics.bioreactor_utilization,
                "transshipment_count": metrics.transshipment_count,
                "transshipment_cost": metrics.transshipment_cost,
            }
        )
    write_rows(rows, args.output)


if __name__ == "__main__":
    main()
