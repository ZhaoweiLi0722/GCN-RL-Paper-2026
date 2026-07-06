"""Run targeted residual-anchor sweeps for GCN-DDPG.

This is a tuning utility for learning whether GCN-DDPG should correct around
MYO, MDL-1, or MDL-2 heuristic anchors before committing to a full benchmark.
Outputs are written under ``results/gcn_residual_sweep`` by default and should
be treated as pilot diagnostics, not manuscript-ready results.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.aggregate_results import write_rows as write_summary_rows
from evaluation.evaluate_formal import evaluate_agent, summarize_rows, write_summary
from src.baselines.heuristics import available_heuristics, get_heuristic_class
from src.rl.agents import get_agent_class
from src.rl.config import load_config, save_config_snapshot
from src.rl.experiment import EpisodeMetrics, build_env, train_off_policy_agent, write_rows as write_training_rows


DEFAULT_BASE_CONFIG = "configs/gcn_ddpg_20_clinic.yaml"
DEFAULT_ENV_CONFIG = "experiments/configs/20_clinic_graph_dynamic_transfer_delay.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--env-config", default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--base-policies", nargs="+", default=["myo", "mdl2"])
    parser.add_argument("--scales", nargs="+", type=float, default=[0.2, 0.35])
    parser.add_argument("--transfer-scale", type=float, default=None)
    parser.add_argument("--replenishment-scale", type=float, default=None)
    parser.add_argument("--l2-weights", nargs="+", type=float, default=[0.02])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=52)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--evaluate-checkpoints", action="store_true")
    parser.add_argument("--elite-epochs", type=int, default=None)
    parser.add_argument("--offline-elite-rollouts", type=int, default=0)
    parser.add_argument("--offline-elite-top-k", type=int, default=5)
    parser.add_argument("--offline-elite-cycles", type=int, default=1)
    parser.add_argument("--offline-elite-epochs", type=int, default=8)
    parser.add_argument("--offline-elite-seed", type=int, default=70000)
    parser.add_argument("--offline-elite-advantage-filter", action="store_true")
    parser.add_argument("--eval-replications", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=50000)
    parser.add_argument("--output-root", default="results/gcn_residual_sweep")
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--summary-output", default=None)
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    env_config = load_config(args.env_config)
    scenario_name = str(env_config.get("scenario_name", Path(args.env_config).stem))
    output_root = Path(args.output_root)
    summary_rows: list[dict[str, Any]] = []

    for base_policy in args.base_policies:
        if base_policy not in available_heuristics():
            raise SystemExit(f"Unsupported residual base policy: {base_policy}")
        for scale in args.scales:
            for l2_weight in args.l2_weights:
                for seed in args.seeds:
                    variant = residual_variant_name(
                        base_policy,
                        scale,
                        l2_weight,
                        transfer_scale=args.transfer_scale,
                        replenishment_scale=args.replenishment_scale,
                    )
                    config = make_residual_sweep_config(
                        base_config,
                        env_config,
                        base_policy=base_policy,
                        scale=scale,
                        transfer_scale=args.transfer_scale,
                        replenishment_scale=args.replenishment_scale,
                        l2_weight=l2_weight,
                        seed=seed,
                        episodes=args.episodes,
                        steps=args.steps,
                        batch_size=args.batch_size,
                        checkpoint_interval=args.checkpoint_interval,
                        elite_epochs=args.elite_epochs,
                        output_root=output_root,
                        scenario_name=scenario_name,
                        variant=variant,
                        progress_interval=args.progress_interval,
                    )
                    env = build_env(config, seed=seed)
                    agent = get_agent_class("gcn_ddpg")(env.observation_size, env.action_size, config)
                    train_rows = train_off_policy_agent(agent, env, config)
                    write_training_rows(train_rows, config["result_csv_path"])
                    save_config_snapshot(config, config["config_snapshot_path"])

                    eval_seed = int(args.evaluation_seed) + int(seed) * 10000
                    checkpoint_paths = [None]
                    if args.evaluate_checkpoints:
                        checkpoint_paths = list_checkpoint_paths(
                            Path(config["checkpoint_dir"]),
                            seed=int(seed),
                        )
                        if not checkpoint_paths:
                            checkpoint_paths = [None]
                    for checkpoint_path in checkpoint_paths:
                        summary = evaluate_policy_candidate(
                            agent,
                            env,
                            checkpoint_path=checkpoint_path,
                            eval_seed=eval_seed,
                            replications=int(args.eval_replications),
                            max_steps=int(args.steps),
                        )
                        summary.update(
                            summary_metadata(
                                variant=variant,
                                base_policy=base_policy,
                                scale=scale,
                                transfer_scale=args.transfer_scale,
                                replenishment_scale=args.replenishment_scale,
                                l2_weight=l2_weight,
                                elite_epochs=args.elite_epochs,
                                seed=seed,
                                eval_seed=eval_seed,
                                checkpoint_path=checkpoint_path,
                                checkpoint_episode=(
                                    int(args.episodes)
                                    if checkpoint_path is None
                                    else checkpoint_episode_from_path(checkpoint_path)
                                ),
                                selection_stage="checkpoint",
                            )
                        )
                        summary_path = (
                            output_root
                            / scenario_name
                            / variant
                            / f"seed{seed}_{summary['selection_stage']}_{summary['checkpoint_episode']}_summary.csv"
                        )
                        write_summary(summary, summary_path)
                        summary_rows.append(summary)

                    best_summary = best_variant_summary(summary_rows, variant, int(seed))
                    best_checkpoint_path = str(best_summary.get("checkpoint_path", ""))
                    if best_checkpoint_path:
                        agent.load_actor(Path(best_checkpoint_path))

                    if args.offline_elite_rollouts > 0:
                        offline_summary = run_offline_elite_distillation(
                            agent,
                            env,
                            seed=int(args.offline_elite_seed) + int(seed) * 10000,
                            rollouts=int(args.offline_elite_rollouts),
                            top_k=int(args.offline_elite_top_k),
                            cycles=int(args.offline_elite_cycles),
                            epochs=int(args.offline_elite_epochs),
                            batch_size=int(args.batch_size),
                            max_steps=int(args.steps),
                            baseline_policy=base_policy,
                            advantage_filter=bool(args.offline_elite_advantage_filter),
                        )
                        offline_checkpoint_path = (
                            Path(config["checkpoint_dir"]) / f"gcn_ddpg_seed{seed}_offline_elite.pt"
                        )
                        agent.save(offline_checkpoint_path)
                        summary = evaluate_policy_candidate(
                            agent,
                            env,
                            checkpoint_path=None,
                            eval_seed=eval_seed,
                            replications=int(args.eval_replications),
                            max_steps=int(args.steps),
                        )
                        summary.update(
                            summary_metadata(
                                variant=variant,
                                base_policy=base_policy,
                                scale=scale,
                                transfer_scale=args.transfer_scale,
                                replenishment_scale=args.replenishment_scale,
                                l2_weight=l2_weight,
                                elite_epochs=args.elite_epochs,
                                seed=seed,
                                eval_seed=eval_seed,
                                checkpoint_path=offline_checkpoint_path,
                                checkpoint_episode="offline_elite",
                                selection_stage="offline_elite",
                            )
                        )
                        summary.update(offline_summary)
                        summary_path = (
                            output_root
                            / scenario_name
                            / variant
                            / f"seed{seed}_offline_elite_summary.csv"
                        )
                        write_summary(summary, summary_path)
                        summary_rows.append(summary)

                    best_summary = min(
                        (row for row in summary_rows if row["variant"] == variant and int(row["training_seed"]) == seed),
                        key=lambda row: float(row["total_cost_mean"]),
                    )
                    print(
                        "best residual checkpoint "
                        f"variant={variant} seed={seed} episode={best_summary['checkpoint_episode']} "
                        f"cost={float(best_summary['total_cost_mean']):.3f}",
                        flush=True,
                    )

    summary_output = Path(args.summary_output or output_root / scenario_name / "sweep_summary.csv")
    write_summary_rows(summary_rows, summary_output)
    print(f"wrote {len(summary_rows)} residual sweep rows to {summary_output}")


def evaluate_policy_candidate(
    agent,
    env,
    *,
    checkpoint_path: Path | None,
    eval_seed: int,
    replications: int,
    max_steps: int,
) -> dict[str, Any]:
    if checkpoint_path is not None:
        agent.load_actor(checkpoint_path)
    eval_rows = evaluate_agent(
        agent,
        env,
        algorithm="gcn_ddpg",
        seed=eval_seed,
        replications=replications,
        max_steps=max_steps,
    )
    return summarize_rows(eval_rows)


def summary_metadata(
    *,
    variant: str,
    base_policy: str,
    scale: float,
    transfer_scale: float | None,
    replenishment_scale: float | None,
    l2_weight: float,
    elite_epochs: int | None,
    seed: int,
    eval_seed: int,
    checkpoint_path: Path | None,
    checkpoint_episode: int | str,
    selection_stage: str,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "residual_base_policy": base_policy,
        "residual_scale": float(scale),
        "residual_transfer_scale": "" if transfer_scale is None else float(transfer_scale),
        "residual_replenishment_scale": "" if replenishment_scale is None else float(replenishment_scale),
        "residual_l2_weight": float(l2_weight),
        "elite_epochs": "" if elite_epochs is None else int(elite_epochs),
        "training_seed": int(seed),
        "evaluation_seed": int(eval_seed),
        "checkpoint_episode": checkpoint_episode,
        "checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
        "selection_stage": selection_stage,
    }


def best_variant_summary(rows: list[dict[str, Any]], variant: str, seed: int) -> dict[str, Any]:
    return min(
        (row for row in rows if row["variant"] == variant and int(row["training_seed"]) == seed),
        key=lambda row: float(row["total_cost_mean"]),
    )


def run_offline_elite_distillation(
    agent,
    env,
    *,
    seed: int,
    rollouts: int,
    top_k: int,
    cycles: int,
    epochs: int,
    batch_size: int,
    max_steps: int,
    baseline_policy: str,
    advantage_filter: bool,
) -> dict[str, Any]:
    all_elites: list[tuple[float, float, float, float, np.ndarray, np.ndarray]] = []
    final_fit: dict[str, Any] = {}
    cycles = max(cycles, 1)
    for cycle in range(cycles):
        elites = collect_elite_rollouts(
            agent,
            env,
            seed=seed + cycle * max(rollouts, 1),
            rollouts=rollouts,
            top_k=top_k,
            max_steps=max_steps,
            baseline_policy=baseline_policy,
            advantage_filter=advantage_filter,
        )
        all_elites.extend(elites)
        all_elites.sort(key=lambda item: item[0])
        del all_elites[max(top_k, 1):]
        if not all_elites:
            print(
                "offline_elite "
                f"cycle={cycle + 1}/{cycles} no positive-advantage rollouts",
                flush=True,
            )
            continue
        states = np.concatenate([item[4] for item in all_elites], axis=0)
        actions = np.concatenate([item[5] for item in all_elites], axis=0)
        final_fit = agent.fit_action_batch(
            states,
            actions,
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed + 900000 + cycle,
            },
        )
        print(
            "offline_elite "
            f"cycle={cycle + 1}/{cycles} best_cost={all_elites[0][1]:.3f} "
            f"best_improvement={all_elites[0][3]:.3f} "
            f"episodes={len(all_elites)} samples={final_fit.get('samples')} "
            f"loss={final_fit.get('final_loss'):.6f}",
            flush=True,
        )
    return {
        "offline_elite_rollouts": int(rollouts),
        "offline_elite_top_k": int(top_k),
        "offline_elite_cycles": int(cycles),
        "offline_elite_epochs": int(epochs),
        "offline_elite_samples": final_fit.get("samples", 0),
        "offline_elite_loss": final_fit.get("final_loss", ""),
        "offline_elite_advantage_filter": bool(advantage_filter),
        "offline_elite_best_rollout_cost": all_elites[0][1] if all_elites else "",
        "offline_elite_best_baseline_cost": all_elites[0][2] if all_elites else "",
        "offline_elite_best_improvement": all_elites[0][3] if all_elites else "",
    }


def collect_elite_rollouts(
    agent,
    env,
    *,
    seed: int,
    rollouts: int,
    top_k: int,
    max_steps: int,
    baseline_policy: str,
    advantage_filter: bool,
) -> list[tuple[float, float, float, float, np.ndarray, np.ndarray]]:
    elites: list[tuple[float, float, float, float, np.ndarray, np.ndarray]] = []
    baseline = get_heuristic_class(baseline_policy)(
        state_dim=env.observation_size,
        action_dim=env.action_size,
        config={},
    )
    for rollout in range(max(rollouts, 0)):
        rollout_seed = seed + rollout
        baseline_cost = run_policy_episode_cost(
            baseline,
            env,
            seed=rollout_seed,
            max_steps=max_steps,
            explore=False,
        )
        state = env.reset(seed=rollout_seed)
        agent.reset()
        metrics = EpisodeMetrics()
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        for _step in range(max_steps):
            action = agent.select_action(state, explore=True, env=env)
            next_state, _reward, done, info = env.step(action)
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
            metrics.update(info)
            state = next_state
            if done:
                break
        if states:
            improvement = baseline_cost - metrics.total_cost
            if advantage_filter and improvement <= 0.0:
                continue
            score = -improvement if advantage_filter else metrics.total_cost
            elites.append(
                (
                    score,
                    metrics.total_cost,
                    baseline_cost,
                    improvement,
                    np.asarray(states, dtype=np.float32),
                    np.asarray(actions, dtype=np.float32),
                )
            )
    elites.sort(key=lambda item: item[0])
    return elites[: max(top_k, 1)]


def run_policy_episode_cost(
    policy,
    env,
    *,
    seed: int,
    max_steps: int,
    explore: bool,
) -> float:
    state = env.reset(seed=seed)
    policy.reset()
    metrics = EpisodeMetrics()
    for _step in range(max_steps):
        action = policy.select_action(state, explore=explore, env=env)
        state, _reward, done, info = env.step(action)
        metrics.update(info)
        if done:
            break
    return metrics.total_cost


def make_residual_sweep_config(
    base_config: dict[str, Any],
    env_config: dict[str, Any],
    *,
    base_policy: str,
    scale: float,
    transfer_scale: float | None,
    replenishment_scale: float | None,
    l2_weight: float,
    seed: int,
    episodes: int,
    steps: int,
    batch_size: int,
    checkpoint_interval: int | None,
    elite_epochs: int | None,
    output_root: Path,
    scenario_name: str,
    variant: str,
    progress_interval: int,
) -> dict[str, Any]:
    config = dict(base_config)
    config["algorithm"] = "gcn_ddpg"
    config["seed"] = int(seed)
    config["num_episodes"] = int(episodes)
    config["max_steps_per_episode"] = int(steps)
    config["batch_size"] = int(batch_size)
    config["checkpoint_interval"] = int(checkpoint_interval or episodes)
    config["progress_interval"] = int(progress_interval)

    merged_env = dict(config.get("env", {}))
    graph_ablation = merged_env.get("graph_ablation", env_config.get("graph_ablation", "full_graph"))
    merged_env.update(env_config)
    merged_env["graph_ablation"] = graph_ablation
    config["env"] = merged_env

    residual_action = dict(config.get("residual_action", {}))
    residual_action.update(
        {
            "enabled": True,
            "base_policy": base_policy,
            "scale": float(scale),
            "l2_weight": float(l2_weight),
        }
    )
    if transfer_scale is not None or replenishment_scale is not None:
        transfer = float(scale if transfer_scale is None else transfer_scale)
        replenishment = float(scale if replenishment_scale is None else replenishment_scale)
        residual_action["group_scales"] = {
            "specimen_transfer": transfer,
            "reagent_transfer": transfer,
            "capacity_transfer": transfer,
            "replenishment": replenishment,
        }
    config["residual_action"] = residual_action

    imitation_pretrain = dict(config.get("imitation_pretrain", {}))
    imitation_pretrain["enabled"] = True
    imitation_pretrain["policy"] = base_policy
    config["imitation_pretrain"] = imitation_pretrain

    if elite_epochs is not None:
        elite_imitation = dict(config.get("elite_imitation", {}))
        elite_imitation["enabled"] = True
        elite_imitation["epochs"] = int(elite_epochs)
        config["elite_imitation"] = elite_imitation

    run_root = output_root / scenario_name / variant / f"seed{seed}"
    config["checkpoint_dir"] = str(run_root / "checkpoints")
    config["result_csv_path"] = str(run_root / "training.csv")
    config["config_snapshot_path"] = str(run_root / "config_snapshot.json")
    return config


def residual_variant_name(
    base_policy: str,
    scale: float,
    l2_weight: float,
    *,
    transfer_scale: float | None = None,
    replenishment_scale: float | None = None,
) -> str:
    scale_token = f"{scale:.3g}".replace(".", "p")
    l2_token = f"{l2_weight:.3g}".replace(".", "p")
    variant = f"{base_policy}_scale{scale_token}_l2{l2_token}"
    if transfer_scale is not None or replenishment_scale is not None:
        transfer = scale if transfer_scale is None else transfer_scale
        replenishment = scale if replenishment_scale is None else replenishment_scale
        transfer_token = f"{transfer:.3g}".replace(".", "p")
        replenishment_token = f"{replenishment:.3g}".replace(".", "p")
        variant = f"{variant}_tr{transfer_token}_rep{replenishment_token}"
    return variant


def list_checkpoint_paths(checkpoint_dir: Path, *, seed: int) -> list[Path]:
    return sorted(
        checkpoint_dir.glob(f"gcn_ddpg_seed{seed}_episode*.pt"),
        key=checkpoint_episode_from_path,
    )


def checkpoint_episode_from_path(path: Path) -> int:
    token = path.stem.rsplit("episode", maxsplit=1)[-1]
    return int(token)


if __name__ == "__main__":
    main()
