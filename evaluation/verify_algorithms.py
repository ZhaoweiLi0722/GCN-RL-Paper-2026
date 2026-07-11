"""Verification harness: do the repo's RL agents actually learn?

Trains each learned agent on a small LQR control task whose optimal policy is
known analytically (Riccati), then scores the trained agent between a random
policy (score 0) and the analytic LQR optimum (score 1):

    normalized_score = (random_cost - learned_cost) / (random_cost - lqr_cost)

An agent that learns lands near 1.0; a broken or mis-wired agent stays near 0
(no better than random) or goes negative. This is the V2/V3 gate from
``specs/tech-stack.md`` — cheap, analytic, and run *before* any algorithm is
trusted on the capacity-planning environment.

Usage:
    python -m evaluation.verify_algorithms
    python -m evaluation.verify_algorithms --algorithms td3 sac --train-steps 20000
    python -m evaluation.verify_algorithms --quick        # fast, for CI / smoke
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np

from src.rl.agents import get_agent_class
from src.verification.lqr_env import make_double_integrator

LEARNED_DEFAULT = ("flat_ddpg", "td3", "sac", "ppo")
PASS_THRESHOLD = 0.7


def _agent_config(algorithm: str, seed: int) -> dict:
    """Fair, algorithm-appropriate config (no capacity-planning assumptions).

    A verification harness must give each algorithm sensible hyperparameters —
    reusing DDPG's learning rate for PPO, for instance, would test the config not
    the implementation. Off-policy actor-critics share one setup; PPO gets its
    own (lower policy LR, an entropy bonus, a longer rollout). ``normalize_
    observations`` stays off and ``reward_scale`` stays 1.0 on purpose.
    """

    base = {"seed": seed, "hidden_sizes": [64, 64], "gamma": 0.99}
    if algorithm == "ppo":
        return {
            **base,
            "actor_lr": 3e-4,          # PPO's tuned default; NOT the off-policy 1e-3
            "critic_lr": 1e-3,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "entropy_coef": 0.01,      # small exploration bonus for a Gaussian policy
            "train_epochs": 10,
            "minibatch_size": 64,
            "rollout_length": 2048,
        }
    return {
        **base,
        "batch_size": 128,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "tau": 0.005,
        "exploration_noise_std": 0.1,
        "replay_buffer_size": 100_000,
    }


def _rollout_cost(env, action_fn: Callable[[np.ndarray], np.ndarray], seeds) -> float:
    """Mean total episode cost of a policy over a fixed set of episode seeds."""

    totals = []
    for seed in seeds:
        state = env.reset(seed=int(seed))
        done = False
        total = 0.0
        while not done:
            state, _reward, done, info = env.step(action_fn(state))
            total += float(info["cost"])
        totals.append(total)
    return float(np.mean(totals))


def _train_agent(agent, env, train_steps: int, train_seed: int) -> None:
    """Drive the agent through the same loop the real training pipeline uses."""

    state = env.reset(seed=train_seed)
    agent.reset()
    episode = 0
    for _ in range(train_steps):
        action = agent.select_action(state, explore=True, env=env)
        next_state, reward, done, _info = env.step(action)
        agent.observe(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        if done:
            episode += 1
            state = env.reset(seed=train_seed + episode)
            agent.reset()


def verify_algorithm(
    algorithm: str,
    *,
    train_steps: int,
    eval_episodes: int,
    seed: int,
    random_cost: float,
    lqr_cost: float,
) -> dict:
    """Train one agent on the LQR task and score it. Never raises."""

    env = make_double_integrator()
    eval_seeds = range(10_000, 10_000 + eval_episodes)
    result = {
        "algorithm": algorithm,
        "random_cost": round(random_cost, 4),
        "lqr_cost": round(lqr_cost, 4),
    }
    try:
        agent_cls = get_agent_class(algorithm)
        agent = agent_cls(state_dim=env.state_dim, action_dim=env.action_dim, config=_agent_config(algorithm, seed))
        _train_agent(agent, env, train_steps, train_seed=seed)
        learned_cost = _rollout_cost(
            env, lambda s: agent.select_action(s, explore=False, env=env), eval_seeds
        )
        denom = random_cost - lqr_cost
        score = (random_cost - learned_cost) / denom if abs(denom) > 1e-9 else float("nan")
        result.update(
            learned_cost=round(learned_cost, 4),
            normalized_score=round(score, 4),
            status="PASS" if score >= PASS_THRESHOLD else "FAIL",
        )
    except Exception as exc:  # noqa: BLE001 - report, don't crash the whole run
        result.update(learned_cost=float("nan"), normalized_score=float("nan"), status=f"ERROR: {exc}")
    return result


def run(algorithms, *, train_steps: int, eval_episodes: int, seed: int, output: Path | None) -> list[dict]:
    env = make_double_integrator()
    eval_seeds = range(10_000, 10_000 + eval_episodes)
    gain = env.optimal_gain()

    rng = np.random.default_rng(seed)
    random_cost = _rollout_cost(env, lambda s: rng.uniform(-1.0, 1.0, env.action_dim), eval_seeds)
    lqr_cost = _rollout_cost(env, lambda s: env.optimal_normalized_action(s, gain), eval_seeds)

    print(f"LQR verification task: double integrator, horizon={env.horizon}")
    print(f"reference costs  random={random_cost:.2f}  lqr_optimal={lqr_cost:.2f}")
    print(f"train_steps={train_steps} eval_episodes={eval_episodes} pass_threshold={PASS_THRESHOLD}\n")
    print(f'{"algorithm":<11}{"learned_cost":>14}{"score":>9}   status')

    results = []
    for algorithm in algorithms:
        res = verify_algorithm(
            algorithm,
            train_steps=train_steps,
            eval_episodes=eval_episodes,
            seed=seed,
            random_cost=random_cost,
            lqr_cost=lqr_cost,
        )
        results.append(res)
        lc = res["learned_cost"]
        sc = res["normalized_score"]
        lc_s = f"{lc:.2f}" if lc == lc else "nan"  # nan-safe
        sc_s = f"{sc:.3f}" if sc == sc else "nan"
        print(f'{res["algorithm"]:<11}{lc_s:>14}{sc_s:>9}   {res["status"]}')

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nwrote {len(results)} rows to {output}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", default=list(LEARNED_DEFAULT))
    parser.add_argument("--train-steps", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="fast preset (4k steps) for CI/smoke")
    parser.add_argument("--output", type=Path, default=Path("results/verification/lqr_verification.csv"))
    args = parser.parse_args()

    train_steps = 4_000 if args.quick else args.train_steps
    run(
        args.algorithms,
        train_steps=train_steps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
