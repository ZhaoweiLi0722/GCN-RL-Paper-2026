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

from evaluation.aggregate_results import write_rows as write_summary_rows
from evaluation.evaluate_formal import evaluate_agent, summarize_rows, write_summary
from src.baselines.heuristics import available_heuristics
from src.rl.agents import get_agent_class
from src.rl.config import load_config, save_config_snapshot
from src.rl.experiment import build_env, train_off_policy_agent, write_rows as write_training_rows


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
                    for checkpoint_path in checkpoint_paths:
                        if checkpoint_path is not None:
                            agent.load_actor(checkpoint_path)
                        eval_rows = evaluate_agent(
                            agent,
                            env,
                            algorithm="gcn_ddpg",
                            seed=eval_seed,
                            replications=int(args.eval_replications),
                            max_steps=int(args.steps),
                        )
                        summary = summarize_rows(eval_rows)
                        checkpoint_episode = (
                            int(args.episodes)
                            if checkpoint_path is None
                            else checkpoint_episode_from_path(checkpoint_path)
                        )
                        summary.update(
                            {
                                "variant": variant,
                                "residual_base_policy": base_policy,
                                "residual_scale": float(scale),
                                "residual_transfer_scale": (
                                    "" if args.transfer_scale is None else float(args.transfer_scale)
                                ),
                                "residual_replenishment_scale": (
                                    "" if args.replenishment_scale is None else float(args.replenishment_scale)
                                ),
                                "residual_l2_weight": float(l2_weight),
                                "elite_epochs": "" if args.elite_epochs is None else int(args.elite_epochs),
                                "training_seed": int(seed),
                                "evaluation_seed": eval_seed,
                                "checkpoint_episode": checkpoint_episode,
                                "checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
                            }
                        )
                        summary_path = (
                            output_root
                            / scenario_name
                            / variant
                            / f"seed{seed}_episode{checkpoint_episode}_summary.csv"
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
