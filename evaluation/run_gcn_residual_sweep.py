"""Run targeted residual-anchor sweeps for GCN-DDPG.

This is a tuning utility for learning whether GCN-DDPG should correct around
MYO, MDL-1, or MDL-2 heuristic anchors before committing to a full benchmark.
Outputs are written under ``results/gcn_residual_sweep`` by default and should
be treated as pilot diagnostics, not manuscript-ready results.
"""

from __future__ import annotations

import argparse
import copy
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
    parser.add_argument("--center-residual-groups", nargs="*", default=[])
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
    parser.add_argument(
        "--offline-elite-weighting",
        choices=["none", "improvement", "rank"],
        default="improvement",
    )
    parser.add_argument("--offline-elite-weight-power", type=float, default=1.0)
    parser.add_argument("--offline-elite-weight-floor", type=float, default=0.10)
    parser.add_argument("--local-search-rollouts", type=int, default=0)
    parser.add_argument("--local-search-seed", type=int, default=90000)
    parser.add_argument("--local-search-lookahead", type=int, default=6)
    parser.add_argument("--local-search-epsilons", nargs="+", type=float, default=[0.02, 0.05, 0.08])
    parser.add_argument("--local-search-epochs", type=int, default=12)
    parser.add_argument("--local-search-min-improvement", type=float, default=0.0)
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
                        center_residual_groups=tuple(args.center_residual_groups or ()),
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
                        center_residual_groups=tuple(args.center_residual_groups or ()),
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
                                center_residual_groups=tuple(args.center_residual_groups or ()),
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
                            weighting=str(args.offline_elite_weighting),
                            weight_power=float(args.offline_elite_weight_power),
                            weight_floor=float(args.offline_elite_weight_floor),
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
                                center_residual_groups=tuple(args.center_residual_groups or ()),
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

                    if args.local_search_rollouts > 0:
                        local_summary = run_local_search_distillation(
                            agent,
                            env,
                            seed=int(args.local_search_seed) + int(seed) * 10000,
                            rollouts=int(args.local_search_rollouts),
                            lookahead=int(args.local_search_lookahead),
                            epsilons=tuple(float(value) for value in args.local_search_epsilons),
                            epochs=int(args.local_search_epochs),
                            batch_size=int(args.batch_size),
                            max_steps=int(args.steps),
                            baseline_policy=base_policy,
                            min_improvement=float(args.local_search_min_improvement),
                        )
                        local_checkpoint_path = (
                            Path(config["checkpoint_dir"]) / f"gcn_ddpg_seed{seed}_local_search.pt"
                        )
                        agent.save(local_checkpoint_path)
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
                                center_residual_groups=tuple(args.center_residual_groups or ()),
                                elite_epochs=args.elite_epochs,
                                seed=seed,
                                eval_seed=eval_seed,
                                checkpoint_path=local_checkpoint_path,
                                checkpoint_episode="local_search",
                                selection_stage="local_search",
                            )
                        )
                        summary.update(local_summary)
                        summary_path = (
                            output_root
                            / scenario_name
                            / variant
                            / f"seed{seed}_local_search_summary.csv"
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
    center_residual_groups: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "variant": variant,
        "residual_base_policy": base_policy,
        "residual_scale": float(scale),
        "residual_transfer_scale": "" if transfer_scale is None else float(transfer_scale),
        "residual_replenishment_scale": "" if replenishment_scale is None else float(replenishment_scale),
        "residual_center_groups": "|".join(center_residual_groups),
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
    weighting: str,
    weight_power: float,
    weight_floor: float,
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
        weights = elite_sample_weights(
            all_elites,
            weighting=weighting,
            power=weight_power,
            floor=weight_floor,
        )
        final_fit = agent.fit_action_batch(
            states,
            actions,
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed + 900000 + cycle,
            },
            weights=weights,
        )
        weight_mean = "" if weights is None else f"{float(weights.mean()):.3f}"
        weight_max = "" if weights is None else f"{float(weights.max()):.3f}"
        print(
            "offline_elite "
            f"cycle={cycle + 1}/{cycles} best_cost={all_elites[0][1]:.3f} "
            f"best_improvement={all_elites[0][3]:.3f} "
            f"episodes={len(all_elites)} samples={final_fit.get('samples')} "
            f"weight_mean={weight_mean} weight_max={weight_max} "
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
        "offline_elite_weighting": weighting,
        "offline_elite_weight_power": float(weight_power),
        "offline_elite_weight_floor": float(weight_floor),
        "offline_elite_best_rollout_cost": all_elites[0][1] if all_elites else "",
        "offline_elite_best_baseline_cost": all_elites[0][2] if all_elites else "",
        "offline_elite_best_improvement": all_elites[0][3] if all_elites else "",
    }


def run_local_search_distillation(
    agent,
    env,
    *,
    seed: int,
    rollouts: int,
    lookahead: int,
    epsilons: tuple[float, ...],
    epochs: int,
    batch_size: int,
    max_steps: int,
    baseline_policy: str,
    min_improvement: float,
    anchor_keep_probability: float = 0.0,
    anchor_keep_weight: float = 1.0,
    anchor_keep_on_improved: bool = True,
    balance_label_weights: bool = False,
    retain_for_regularization: bool = False,
    min_service_level_delta: float | None = None,
    min_completion_service_level_delta: float | None = None,
    max_patients_lost_delta: float | None = None,
    max_patient_ineligibility_during_manufacturing_rate_delta: float | None = None,
    service_level_weight: float = 0.0,
    completion_service_level_weight: float = 0.0,
    eligibility_rate_weight: float = 0.0,
    patient_ineligibility_during_manufacturing_rate_weight: float = 0.0,
    at_risk_unserved_weight: float = 0.0,
    patients_lost_weight: float = 0.0,
    candidate_groups: tuple[str, ...] | None = None,
    candidate_signs: tuple[float, ...] | None = None,
    demonstration_path: str | Path | None = None,
    validation_demonstration_path: str | Path | None = None,
    populate_replay_buffer: bool = False,
    lookahead_replications: int = 1,
    lookahead_seed: int | None = None,
    dense_advantage_weight_scale: float | None = None,
    dense_advantage_weight_cap: float = 10.0,
    trajectory_validation_fraction: float = 0.0,
    trajectory_validation_min_per_scenario: int = 1,
    early_stopping_patience: int = 0,
    early_stopping_check_interval: int = 1,
    early_stopping_min_delta: float = 0.0,
    early_stopping_gate_loss_weight: float = 1.0,
) -> dict[str, Any]:
    demonstration_file = Path(demonstration_path) if demonstration_path else None
    if demonstration_file is not None and demonstration_file.exists():
        demos = load_local_search_demonstrations(demonstration_file)
        demonstration_source = "cache"
    else:
        demos = collect_local_search_demonstrations(
            env,
            seed=seed,
            rollouts=rollouts,
            lookahead=lookahead,
            epsilons=epsilons,
            max_steps=max_steps,
            baseline_policy=baseline_policy,
            min_improvement=min_improvement,
            anchor_keep_probability=anchor_keep_probability,
            anchor_keep_weight=anchor_keep_weight,
            anchor_keep_on_improved=anchor_keep_on_improved,
            balance_label_weights=balance_label_weights,
            min_service_level_delta=min_service_level_delta,
            min_completion_service_level_delta=min_completion_service_level_delta,
            max_patients_lost_delta=max_patients_lost_delta,
            max_patient_ineligibility_during_manufacturing_rate_delta=(
                max_patient_ineligibility_during_manufacturing_rate_delta
            ),
            service_level_weight=service_level_weight,
            completion_service_level_weight=completion_service_level_weight,
            eligibility_rate_weight=eligibility_rate_weight,
            patient_ineligibility_during_manufacturing_rate_weight=(
                patient_ineligibility_during_manufacturing_rate_weight
            ),
            at_risk_unserved_weight=at_risk_unserved_weight,
            patients_lost_weight=patients_lost_weight,
            candidate_groups=candidate_groups,
            candidate_signs=candidate_signs,
            lookahead_replications=lookahead_replications,
            lookahead_seed=lookahead_seed,
        )
        demonstration_source = "collected"
        if demonstration_file is not None:
            save_local_search_demonstrations(demonstration_file, demos)
    validation_file = (
        Path(validation_demonstration_path)
        if validation_demonstration_path
        else None
    )
    validation_demos = None
    if validation_file is not None:
        if not validation_file.is_file():
            raise FileNotFoundError(validation_file)
        if float(trajectory_validation_fraction) > 0.0:
            raise ValueError(
                "Choose either validation_demonstration_path or "
                "trajectory_validation_fraction"
            )
        validation_demos = load_local_search_demonstrations(
            validation_file
        )
    if dense_advantage_weight_scale is not None:
        demos = calibrate_dense_option_advantage_weights(
            demos,
            advantage_scale=dense_advantage_weight_scale,
            weight_cap=dense_advantage_weight_cap,
        )
        if validation_demos is not None:
            validation_demos = calibrate_dense_option_advantage_weights(
                validation_demos,
                advantage_scale=dense_advantage_weight_scale,
                weight_cap=dense_advantage_weight_cap,
            )
    if balance_label_weights:
        demos = balance_demonstration_label_weights(demos)
        if validation_demos is not None:
            validation_demos = balance_demonstration_label_weights(
                validation_demos
            )
    replay_transitions = 0
    if populate_replay_buffer:
        replay_transitions = populate_agent_replay_from_demonstrations(agent, demos)
    if demos["states"].size == 0:
        print("local_search no state-action demonstrations", flush=True)
        return {
            "local_search_rollouts": int(rollouts),
            "local_search_samples": 0,
            "local_search_improved_steps": 0,
            "local_search_anchor_keep_steps": 0,
            "local_search_anchor_keep_on_improved": bool(anchor_keep_on_improved),
            "local_search_balance_label_weights": bool(balance_label_weights),
            "local_search_dense_advantage_weight_scale": (
                ""
                if dense_advantage_weight_scale is None
                else float(dense_advantage_weight_scale)
            ),
            "local_search_dense_advantage_weight_cap": float(
                dense_advantage_weight_cap
            ),
            "local_search_retain_for_regularization": bool(
                retain_for_regularization
            ),
            "local_search_min_service_level_delta": (
                "" if min_service_level_delta is None else float(min_service_level_delta)
            ),
            "local_search_min_completion_service_level_delta": (
                ""
                if min_completion_service_level_delta is None
                else float(min_completion_service_level_delta)
            ),
            "local_search_max_patients_lost_delta": (
                "" if max_patients_lost_delta is None else float(max_patients_lost_delta)
            ),
            "local_search_max_patient_ineligibility_during_manufacturing_rate_delta": (
                ""
                if max_patient_ineligibility_during_manufacturing_rate_delta is None
                else float(max_patient_ineligibility_during_manufacturing_rate_delta)
            ),
            "local_search_service_level_weight": float(service_level_weight),
            "local_search_completion_service_level_weight": float(
                completion_service_level_weight
            ),
            "local_search_eligibility_rate_weight": float(eligibility_rate_weight),
            "local_search_patient_ineligibility_during_manufacturing_rate_weight": float(
                patient_ineligibility_during_manufacturing_rate_weight
            ),
            "local_search_at_risk_unserved_weight": float(at_risk_unserved_weight),
            "local_search_patients_lost_weight": float(patients_lost_weight),
            "local_search_candidate_groups": "|".join(candidate_groups or ()),
            "local_search_candidate_signs": "|".join(
                f"{float(sign):g}" for sign in tuple(candidate_signs or ())
            ),
            "local_search_service_rejected_steps": int(demos["service_rejected_steps"]),
            "local_search_demonstration_source": demonstration_source,
            "local_search_demonstration_path": (
                "" if demonstration_file is None else str(demonstration_file)
            ),
            "local_search_replay_transitions": replay_transitions,
            "local_search_loss": "",
        }
    split_summary: dict[str, Any] = {
        "trajectory_validation_enabled": False,
        "train_samples": int(demos["states"].shape[0]),
        "validation_samples": 0,
        "train_trajectory_ids": "",
        "validation_trajectory_ids": "",
    }
    if validation_demos is not None:
        split_summary = external_validation_summary(
            demos,
            validation_demos,
        )
        final_fit = fit_action_batch_with_early_stopping(
            agent,
            demos,
            validation_demos,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + 1200000,
            retain_for_regularization=bool(
                retain_for_regularization
            ),
            patience=int(early_stopping_patience),
            check_interval=int(early_stopping_check_interval),
            min_delta=float(early_stopping_min_delta),
            gate_loss_weight=float(
                early_stopping_gate_loss_weight
            ),
        )
    elif float(trajectory_validation_fraction) > 0.0:
        train_demos, validation_demos, split_summary = (
            split_demonstrations_by_trajectory(
                demos,
                validation_fraction=float(
                    trajectory_validation_fraction
                ),
                min_per_scenario=int(
                    trajectory_validation_min_per_scenario
                ),
                seed=seed + 1190000,
            )
        )
        final_fit = fit_action_batch_with_early_stopping(
            agent,
            train_demos,
            validation_demos,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + 1200000,
            retain_for_regularization=bool(
                retain_for_regularization
            ),
            patience=int(early_stopping_patience),
            check_interval=int(early_stopping_check_interval),
            min_delta=float(early_stopping_min_delta),
            gate_loss_weight=float(
                early_stopping_gate_loss_weight
            ),
        )
    else:
        final_fit = agent.fit_action_batch(
            demos["states"],
            demos["actions"],
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed + 1200000,
                "retain_for_regularization": bool(
                    retain_for_regularization
                ),
                "demonstrations": demos,
            },
            weights=demos["weights"],
        )
    print(
        "local_search "
        f"rollouts={rollouts} samples={demos['states'].shape[0]} "
        f"improved_steps={demos['improved_steps']} "
        f"anchor_keep_steps={demos['anchor_keep_steps']} "
        f"mean_step_improvement={demos['mean_step_improvement']:.3f} "
        f"loss={final_fit.get('final_loss'):.6f}"
        + (
            f" train_accuracy={float(final_fit['train_accuracy']):.3f}"
            if "train_accuracy" in final_fit
            else ""
        ),
        flush=True,
    )
    summary = {
        "local_search_rollouts": int(rollouts),
        "local_search_lookahead": int(lookahead),
        "local_search_lookahead_replications": int(lookahead_replications),
        "local_search_epsilons": "|".join(f"{value:.4g}" for value in epsilons),
        "local_search_epochs": int(epochs),
        "local_search_samples": int(demos["states"].shape[0]),
        "local_search_improved_steps": int(demos["improved_steps"]),
        "local_search_anchor_keep_steps": int(demos["anchor_keep_steps"]),
        "local_search_anchor_keep_probability": float(anchor_keep_probability),
        "local_search_anchor_keep_weight": float(anchor_keep_weight),
        "local_search_anchor_keep_on_improved": bool(anchor_keep_on_improved),
        "local_search_balance_label_weights": bool(balance_label_weights),
        "local_search_dense_advantage_weight_scale": (
            ""
            if dense_advantage_weight_scale is None
            else float(dense_advantage_weight_scale)
        ),
        "local_search_dense_advantage_weight_cap": float(
            dense_advantage_weight_cap
        ),
        "local_search_retain_for_regularization": bool(retain_for_regularization),
        "local_search_improved_weight_fraction": float(
            demos["improved_weight_fraction"]
        ),
        "local_search_min_service_level_delta": (
            "" if min_service_level_delta is None else float(min_service_level_delta)
        ),
        "local_search_min_completion_service_level_delta": (
            ""
            if min_completion_service_level_delta is None
            else float(min_completion_service_level_delta)
        ),
        "local_search_max_patients_lost_delta": (
            "" if max_patients_lost_delta is None else float(max_patients_lost_delta)
        ),
        "local_search_max_patient_ineligibility_during_manufacturing_rate_delta": (
            ""
            if max_patient_ineligibility_during_manufacturing_rate_delta is None
            else float(max_patient_ineligibility_during_manufacturing_rate_delta)
        ),
        "local_search_service_level_weight": float(service_level_weight),
        "local_search_completion_service_level_weight": float(
            completion_service_level_weight
        ),
        "local_search_eligibility_rate_weight": float(eligibility_rate_weight),
        "local_search_patient_ineligibility_during_manufacturing_rate_weight": float(
            patient_ineligibility_during_manufacturing_rate_weight
        ),
        "local_search_at_risk_unserved_weight": float(at_risk_unserved_weight),
        "local_search_patients_lost_weight": float(patients_lost_weight),
        "local_search_candidate_groups": "|".join(candidate_groups or ()),
        "local_search_candidate_signs": "|".join(
            f"{float(sign):g}" for sign in tuple(candidate_signs or ())
        ),
        "local_search_service_rejected_steps": int(demos["service_rejected_steps"]),
        "local_search_demonstration_source": demonstration_source,
        "local_search_demonstration_path": (
            "" if demonstration_file is None else str(demonstration_file)
        ),
        "local_search_validation_demonstration_path": (
            "" if validation_file is None else str(validation_file)
        ),
        "local_search_replay_transitions": replay_transitions,
        "local_search_mean_step_improvement": float(demos["mean_step_improvement"]),
        "local_search_loss": final_fit.get("final_loss", ""),
        "local_search_train_accuracy": final_fit.get("train_accuracy", ""),
        "local_search_correction_recall": final_fit.get("correction_recall", ""),
        "local_search_correction_accuracy": final_fit.get("correction_accuracy", ""),
        "local_search_anchor_prediction_fraction": final_fit.get(
            "anchor_prediction_fraction",
            "",
        ),
        "local_search_changed_fraction": final_fit.get("changed_fraction", ""),
        "local_search_option_count": final_fit.get("option_count", ""),
        "local_search_observed_option_count": final_fit.get(
            "observed_option_count",
            "",
        ),
        "local_search_target_mode": final_fit.get("target_mode", ""),
        "local_search_mean_best_advantage": final_fit.get(
            "mean_best_advantage",
            "",
        ),
        "local_search_mean_predicted_advantage": final_fit.get(
            "mean_predicted_advantage",
            "",
        ),
        "local_search_mean_advantage_regret": final_fit.get(
            "mean_advantage_regret",
            "",
        ),
        "local_search_predicted_feasible_fraction": final_fit.get(
            "predicted_feasible_fraction",
            "",
        ),
        "local_search_predicted_material_improvement_fraction": final_fit.get(
            "predicted_material_improvement_fraction",
            "",
        ),
        "local_search_best_epoch": final_fit.get("best_epoch", ""),
        "local_search_epochs_completed": final_fit.get("epochs_completed", ""),
        "local_search_correction_gate_threshold": final_fit.get(
            "correction_gate_threshold",
            "",
        ),
        "local_search_correction_gate_accuracy": final_fit.get(
            "correction_gate_accuracy",
            "",
        ),
        "local_search_correction_gate_recall": final_fit.get(
            "correction_gate_recall",
            "",
        ),
        "local_search_correction_gate_precision": final_fit.get(
            "correction_gate_precision",
            "",
        ),
        "local_search_correction_gate_prediction_fraction": final_fit.get(
            "correction_gate_prediction_fraction",
            "",
        ),
        "local_search_correction_gate_groups": final_fit.get(
            "correction_gate_groups",
            "",
        ),
        "local_search_correction_gate_group_label_rates": final_fit.get(
            "correction_gate_group_label_rates",
            "",
        ),
        "local_search_correction_gate_group_prediction_rates": final_fit.get(
            "correction_gate_group_prediction_rates",
            "",
        ),
        "local_search_correction_gate_group_positive_weights": final_fit.get(
            "correction_gate_group_positive_weights",
            "",
        ),
        "local_search_trajectory_validation_enabled": bool(
            split_summary["trajectory_validation_enabled"]
        ),
        "local_search_train_samples": int(split_summary["train_samples"]),
        "local_search_validation_samples": int(
            split_summary["validation_samples"]
        ),
        "local_search_train_trajectory_ids": str(
            split_summary["train_trajectory_ids"]
        ),
        "local_search_validation_trajectory_ids": str(
            split_summary["validation_trajectory_ids"]
        ),
        "local_search_validation_loss": final_fit.get(
            "validation_loss",
            "",
        ),
        "local_search_validation_actor_loss": final_fit.get(
            "validation_actor_loss",
            "",
        ),
        "local_search_validation_correction_gate_loss": final_fit.get(
            "validation_correction_gate_loss",
            "",
        ),
        "local_search_validation_correction_gate_precision": final_fit.get(
            "validation_correction_gate_precision",
            "",
        ),
        "local_search_validation_correction_gate_recall": final_fit.get(
            "validation_correction_gate_recall",
            "",
        ),
        "local_search_early_stopping_gate_loss_weight": final_fit.get(
            "early_stopping_gate_loss_weight",
            "",
        ),
    }
    return summary


def external_validation_summary(
    train_demos: dict[str, Any],
    validation_demos: dict[str, Any],
) -> dict[str, Any]:
    """Describe a pre-split trajectory validation cache."""

    train_ids = np.unique(
        np.asarray(
            train_demos.get("trajectory_ids", ()),
            dtype=np.int64,
        )
    )
    validation_ids = np.unique(
        np.asarray(
            validation_demos.get("trajectory_ids", ()),
            dtype=np.int64,
        )
    )
    if train_ids.size and validation_ids.size:
        overlap = np.intersect1d(train_ids, validation_ids)
        if overlap.size:
            raise ValueError(
                "Training and validation demonstration caches share "
                f"trajectory ids: {overlap.tolist()}"
            )
    return {
        "trajectory_validation_enabled": True,
        "train_samples": int(train_demos["states"].shape[0]),
        "validation_samples": int(
            validation_demos["states"].shape[0]
        ),
        "train_trajectory_ids": "|".join(
            str(int(value)) for value in train_ids
        ),
        "validation_trajectory_ids": "|".join(
            str(int(value)) for value in validation_ids
        ),
    }


_NON_ROW_DEMONSTRATION_KEYS = frozenset(
    {
        "scenario_names",
        "option_groups",
        "option_epsilons",
        "option_signs",
    }
)


def split_demonstrations_by_trajectory(
    demos: dict[str, Any],
    *,
    validation_fraction: float,
    min_per_scenario: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Hold out complete trajectories within every represented scenario."""

    validation_fraction = float(validation_fraction)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "trajectory validation_fraction must be between zero and one"
        )
    min_per_scenario = max(int(min_per_scenario), 1)
    row_count = int(np.asarray(demos["states"]).shape[0])
    for key in ("scenario_ids", "trajectory_ids"):
        if key not in demos:
            raise ValueError(
                "Trajectory validation requires scenario_ids and "
                "trajectory_ids in the teacher cache"
            )
        if np.asarray(demos[key]).shape != (row_count,):
            raise ValueError(f"{key} must align with demonstration rows")
    scenario_ids = np.asarray(demos["scenario_ids"], dtype=np.int64)
    trajectory_ids = np.asarray(
        demos["trajectory_ids"],
        dtype=np.int64,
    )
    validation_trajectory_ids: list[int] = []
    rng = np.random.default_rng(int(seed))
    for scenario_id in np.unique(scenario_ids):
        scenario_rows = scenario_ids == scenario_id
        scenario_trajectories = np.unique(
            trajectory_ids[scenario_rows]
        )
        if scenario_trajectories.size < 2:
            raise ValueError(
                "Trajectory validation requires at least two trajectories "
                f"for scenario {int(scenario_id)}"
            )
        for trajectory_id in scenario_trajectories:
            rows = trajectory_ids == trajectory_id
            if np.any(scenario_ids[rows] != scenario_id):
                raise ValueError(
                    "Each trajectory_id must belong to exactly one scenario"
                )
        validation_count = max(
            min_per_scenario,
            int(np.ceil(
                validation_fraction
                * int(scenario_trajectories.size)
            )),
        )
        validation_count = min(
            validation_count,
            int(scenario_trajectories.size) - 1,
        )
        chosen = rng.choice(
            scenario_trajectories,
            size=validation_count,
            replace=False,
        )
        validation_trajectory_ids.extend(
            int(value) for value in chosen
        )
    validation_mask = np.isin(
        trajectory_ids,
        np.asarray(validation_trajectory_ids, dtype=np.int64),
    )
    if not np.any(validation_mask) or not np.any(~validation_mask):
        raise ValueError(
            "Trajectory validation split must retain train and validation rows"
        )
    train_demos = subset_demonstrations(demos, ~validation_mask)
    validation_demos = subset_demonstrations(
        demos,
        validation_mask,
    )
    train_ids = sorted(
        int(value)
        for value in np.unique(train_demos["trajectory_ids"])
    )
    validation_ids = sorted(
        int(value)
        for value in np.unique(validation_demos["trajectory_ids"])
    )
    summary = {
        "trajectory_validation_enabled": True,
        "train_samples": int(train_demos["states"].shape[0]),
        "validation_samples": int(
            validation_demos["states"].shape[0]
        ),
        "train_trajectory_ids": "|".join(map(str, train_ids)),
        "validation_trajectory_ids": "|".join(
            map(str, validation_ids)
        ),
    }
    return train_demos, validation_demos, summary


def subset_demonstrations(
    demos: dict[str, Any],
    row_mask: np.ndarray,
) -> dict[str, Any]:
    """Slice row-aligned teacher arrays while retaining cache metadata."""

    mask = np.asarray(row_mask, dtype=bool)
    row_count = int(np.asarray(demos["states"]).shape[0])
    if mask.shape != (row_count,):
        raise ValueError("Demonstration subset mask must align with rows")
    subset: dict[str, Any] = {}
    for key, value in demos.items():
        if (
            key not in _NON_ROW_DEMONSTRATION_KEYS
            and isinstance(value, np.ndarray)
            and value.ndim > 0
            and value.shape[0] == row_count
        ):
            subset[key] = value[mask]
        else:
            subset[key] = value
    improved_mask = np.asarray(
        subset.get(
            "improved_mask",
            np.zeros(int(mask.sum()), dtype=bool),
        ),
        dtype=bool,
    )
    weights = np.asarray(subset["weights"], dtype=np.float32)
    subset["improved_steps"] = int(improved_mask.sum())
    subset["anchor_keep_steps"] = int((~improved_mask).sum())
    weight_total = float(weights.sum())
    subset["improved_weight_fraction"] = (
        float(weights[improved_mask].sum()) / weight_total
        if weight_total > 0.0
        else 0.0
    )
    return subset


def fit_action_batch_with_early_stopping(
    agent,
    train_demos: dict[str, Any],
    validation_demos: dict[str, Any],
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    retain_for_regularization: bool,
    patience: int,
    check_interval: int,
    min_delta: float,
    gate_loss_weight: float = 1.0,
) -> dict[str, Any]:
    """Select supervised actor/gate weights using held-out trajectories."""

    if not hasattr(agent, "evaluate_action_batch"):
        raise ValueError(
            "Trajectory early stopping requires evaluate_action_batch"
        )
    total_epochs = max(int(epochs), 1)
    check_interval = max(int(check_interval), 1)
    patience = max(int(patience), 0)
    min_delta = max(float(min_delta), 0.0)
    gate_loss_weight = max(float(gate_loss_weight), 0.0)
    best_metric = float("inf")
    best_epoch = 0
    epochs_completed = 0
    checks_without_improvement = 0
    best_state = snapshot_supervised_agent(agent)
    last_fit: dict[str, Any] = {}
    while epochs_completed < total_epochs:
        chunk_epochs = min(
            check_interval,
            total_epochs - epochs_completed,
        )
        last_fit = agent.fit_action_batch(
            train_demos["states"],
            train_demos["actions"],
            {
                "epochs": chunk_epochs,
                "batch_size": batch_size,
                "seed": seed + epochs_completed,
                "retain_for_regularization": bool(
                    retain_for_regularization
                ),
                "demonstrations": train_demos,
            },
            weights=train_demos["weights"],
        )
        epochs_completed += chunk_epochs
        validation = evaluate_supervised_action_batch(
            agent,
            validation_demos,
        )
        metric = float(validation["actor_loss"]) + gate_loss_weight * float(
            validation.get("correction_gate_loss", 0.0)
        )
        if metric < best_metric - min_delta:
            best_metric = metric
            best_epoch = epochs_completed
            checks_without_improvement = 0
            best_state = snapshot_supervised_agent(agent)
        else:
            checks_without_improvement += 1
        print(
            "supervised_early_stopping "
            f"epochs={epochs_completed}/{total_epochs} "
            f"validation_metric={metric:.6f} "
            f"best_metric={best_metric:.6f} "
            f"best_epoch={best_epoch} "
            f"stale_checks={checks_without_improvement}",
            flush=True,
        )
        if (
            patience > 0
            and checks_without_improvement >= patience
        ):
            break
    restore_supervised_agent(agent, best_state)
    train_metrics = evaluate_supervised_action_batch(
        agent,
        train_demos,
    )
    validation_metrics = evaluate_supervised_action_batch(
        agent,
        validation_demos,
    )
    final_fit = dict(last_fit)
    final_fit.update(train_metrics)
    final_fit.update(
        {
            f"validation_{key}": value
            for key, value in validation_metrics.items()
        }
    )
    final_fit.update(
        {
            "samples": int(train_demos["states"].shape[0]),
            "final_loss": float(train_metrics["actor_loss"]),
            "correction_gate_loss": train_metrics.get(
                "correction_gate_loss",
                "",
            ),
            "best_epoch": int(best_epoch),
            "epochs_completed": int(epochs_completed),
            "early_stopped": bool(
                epochs_completed < total_epochs
            ),
            "early_stopping_gate_loss_weight": gate_loss_weight,
        }
    )
    return final_fit


def evaluate_supervised_action_batch(
    agent,
    demonstrations: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "weights": demonstrations["weights"],
    }
    if (
        bool(getattr(agent, "edge_supervision_enabled", False))
        or bool(
            getattr(
                agent,
                "requires_demonstrations_for_action_evaluation",
                False,
            )
        )
    ):
        kwargs["demonstrations"] = demonstrations
    return agent.evaluate_action_batch(
        demonstrations["states"],
        demonstrations["actions"],
        **kwargs,
    )


def snapshot_supervised_agent(agent) -> dict[str, Any]:
    """Copy actor/gate modules and optimizers used by supervised fitting."""

    state: dict[str, Any] = {}
    for name in (
        "actor",
        "actor_target",
        "correction_gate",
        "q_network",
        "target_q_network",
    ):
        module = getattr(agent, name, None)
        if module is not None:
            state[name] = copy.deepcopy(module.state_dict())
    for name in (
        "actor_optimizer",
        "correction_gate_optimizer",
        "optimizer",
    ):
        optimizer = getattr(agent, name, None)
        if optimizer is not None:
            state[name] = copy.deepcopy(optimizer.state_dict())
    return state


def restore_supervised_agent(
    agent,
    state: dict[str, Any],
) -> None:
    """Restore a supervised actor/gate snapshot selected on validation."""

    for name in (
        "actor",
        "actor_target",
        "correction_gate",
        "q_network",
        "target_q_network",
    ):
        module = getattr(agent, name, None)
        if module is not None and name in state:
            module.load_state_dict(state[name])
    for name in (
        "actor_optimizer",
        "correction_gate_optimizer",
        "optimizer",
    ):
        optimizer = getattr(agent, name, None)
        if optimizer is not None and name in state:
            optimizer.load_state_dict(state[name])


def collect_local_search_demonstrations(
    env,
    *,
    seed: int,
    rollouts: int,
    lookahead: int,
    epsilons: tuple[float, ...],
    max_steps: int,
    baseline_policy: str,
    min_improvement: float,
    anchor_keep_probability: float = 0.0,
    anchor_keep_weight: float = 1.0,
    anchor_keep_on_improved: bool = True,
    balance_label_weights: bool = False,
    min_service_level_delta: float | None = None,
    min_completion_service_level_delta: float | None = None,
    max_patients_lost_delta: float | None = None,
    max_patient_ineligibility_during_manufacturing_rate_delta: float | None = None,
    service_level_weight: float = 0.0,
    completion_service_level_weight: float = 0.0,
    eligibility_rate_weight: float = 0.0,
    patient_ineligibility_during_manufacturing_rate_weight: float = 0.0,
    at_risk_unserved_weight: float = 0.0,
    patients_lost_weight: float = 0.0,
    candidate_groups: tuple[str, ...] | None = None,
    candidate_signs: tuple[float, ...] | None = None,
    lookahead_replications: int = 1,
    lookahead_seed: int | None = None,
) -> dict[str, Any]:
    baseline = get_heuristic_class(baseline_policy)(
        state_dim=env.observation_size,
        action_dim=env.action_size,
        config={},
    )
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    weights: list[float] = []
    improved_labels: list[bool] = []
    transition_states: list[np.ndarray] = []
    transition_actions: list[np.ndarray] = []
    transition_rewards: list[float] = []
    transition_next_states: list[np.ndarray] = []
    transition_dones: list[bool] = []
    improved_steps = 0
    anchor_keep_steps = 0
    service_rejected_steps = 0
    step_improvements: list[float] = []
    rng = np.random.default_rng(seed + 770000)
    lookahead_replications = max(int(lookahead_replications), 1)
    lookahead_seed = int(seed + 1770000 if lookahead_seed is None else lookahead_seed)
    anchor_keep_probability = float(np.clip(anchor_keep_probability, 0.0, 1.0))
    anchor_keep_weight = max(float(anchor_keep_weight), 0.0)
    score_weights = {
        "service_level": float(service_level_weight),
        "completion_service_level": float(completion_service_level_weight),
        "eligibility_rate": float(eligibility_rate_weight),
        "patient_ineligibility_during_manufacturing_rate": float(
            patient_ineligibility_during_manufacturing_rate_weight
        ),
        "at_risk_unserved": float(at_risk_unserved_weight),
        "patients_lost": float(patients_lost_weight),
    }
    for rollout in range(max(rollouts, 0)):
        state = env.reset(seed=seed + rollout)
        baseline.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            candidate_actions = local_search_candidate_actions(
                state,
                env,
                baseline,
                epsilons=epsilons,
                candidate_groups=candidate_groups,
                candidate_signs=candidate_signs,
            )
            decision_index = rollout * max(int(max_steps), 1) + step
            rollout_seeds = tuple(
                lookahead_seed + decision_index * lookahead_replications + replication
                for replication in range(lookahead_replications)
            )
            candidate_metrics = [
                mean_rollout_metrics_after_action(
                    env,
                    baseline,
                    action,
                    horizon=lookahead,
                    rollout_seeds=rollout_seeds,
                )
                for action in candidate_actions
            ]
            candidate_costs = [float(metrics["total_cost"]) for metrics in candidate_metrics]
            candidate_scores = [
                local_search_metric_score(metrics, score_weights)
                for metrics in candidate_metrics
            ]
            score_best_index = int(np.argmin(candidate_scores))
            best_index = score_best_index
            guardrails_enabled = any(
                value is not None
                for value in (
                    min_service_level_delta,
                    min_completion_service_level_delta,
                    max_patients_lost_delta,
                    max_patient_ineligibility_during_manufacturing_rate_delta,
                )
            )
            if guardrails_enabled:
                baseline_metrics = candidate_metrics[0]
                feasible_indices = []
                for index, metrics in enumerate(candidate_metrics):
                    service_ok = _minimum_metric_delta_satisfied(
                        metrics,
                        baseline_metrics,
                        "service_level",
                        min_service_level_delta,
                    )
                    completion_ok = _minimum_metric_delta_satisfied(
                        metrics,
                        baseline_metrics,
                        "completion_service_level",
                        min_completion_service_level_delta,
                    )
                    patient_loss_ok = _maximum_metric_delta_satisfied(
                        metrics,
                        baseline_metrics,
                        "patients_lost",
                        max_patients_lost_delta,
                    )
                    manufacturing_ineligibility_ok = _maximum_metric_delta_satisfied(
                        metrics,
                        baseline_metrics,
                        "patient_ineligibility_during_manufacturing_rate",
                        max_patient_ineligibility_during_manufacturing_rate_delta,
                    )
                    if (
                        service_ok
                        and completion_ok
                        and patient_loss_ok
                        and manufacturing_ineligibility_ok
                    ):
                        feasible_indices.append(index)
                if feasible_indices:
                    best_index = min(feasible_indices, key=lambda index: candidate_scores[index])
                else:
                    best_index = 0
                if score_best_index != best_index and score_best_index != 0:
                    service_rejected_steps += 1
            baseline_cost = float(candidate_costs[0])
            best_cost = float(candidate_costs[best_index])
            baseline_score = float(candidate_scores[0])
            best_score = float(candidate_scores[best_index])
            improvement = baseline_score - best_score
            selected_action = candidate_actions[best_index]
            improved = best_index != 0 and improvement > min_improvement
            if improved:
                states.append(np.asarray(state, dtype=np.float32))
                actions.append(np.asarray(selected_action, dtype=np.float32))
                weights.append(max(improvement, 1.0))
                improved_labels.append(True)
                step_improvements.append(improvement)
                improved_steps += 1
            keep_anchor = anchor_keep_on_improved or not improved
            if (
                keep_anchor
                and anchor_keep_probability > 0.0
                and rng.random() < anchor_keep_probability
            ):
                states.append(np.asarray(state, dtype=np.float32))
                actions.append(np.asarray(candidate_actions[0], dtype=np.float32))
                weights.append(max(anchor_keep_weight, 1.0))
                improved_labels.append(False)
                anchor_keep_steps += 1
            next_state, reward, done, _info = env.step(selected_action)
            transition_states.append(np.asarray(state, dtype=np.float32))
            transition_actions.append(np.asarray(selected_action, dtype=np.float32))
            transition_rewards.append(float(reward))
            transition_next_states.append(np.asarray(next_state, dtype=np.float32))
            transition_dones.append(bool(done))
            state = next_state
            step += 1

    if not states:
        return {
            "states": np.empty((0, env.observation_size), dtype=np.float32),
            "actions": np.empty((0, env.action_size), dtype=np.float32),
            "weights": np.empty((0,), dtype=np.float32),
            "improved_mask": np.empty((0,), dtype=bool),
            "transition_states": np.asarray(transition_states, dtype=np.float32).reshape(
                -1, env.observation_size
            ),
            "transition_actions": np.asarray(transition_actions, dtype=np.float32).reshape(
                -1, env.action_size
            ),
            "transition_rewards": np.asarray(transition_rewards, dtype=np.float32),
            "transition_next_states": np.asarray(
                transition_next_states,
                dtype=np.float32,
            ).reshape(-1, env.observation_size),
            "transition_dones": np.asarray(transition_dones, dtype=bool),
            "improved_steps": 0,
            "anchor_keep_steps": 0,
            "service_rejected_steps": service_rejected_steps,
            "mean_step_improvement": 0.0,
            "improved_weight_fraction": 0.0,
        }
    weight_array = np.asarray(weights, dtype=np.float32)
    improved_mask = np.asarray(improved_labels, dtype=bool)
    if balance_label_weights and np.any(improved_mask) and np.any(~improved_mask):
        improved_total = float(weight_array[improved_mask].sum())
        anchor_total = float(weight_array[~improved_mask].sum())
        if improved_total > 0.0 and anchor_total > 0.0:
            weight_array[improved_mask] *= 0.5 / improved_total
            weight_array[~improved_mask] *= 0.5 / anchor_total
            weight_array *= float(weight_array.size) / float(weight_array.sum())
    weight_total = float(weight_array.sum())
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "weights": weight_array,
        "improved_mask": improved_mask,
        "transition_states": np.asarray(transition_states, dtype=np.float32).reshape(
            -1, env.observation_size
        ),
        "transition_actions": np.asarray(transition_actions, dtype=np.float32).reshape(
            -1, env.action_size
        ),
        "transition_rewards": np.asarray(transition_rewards, dtype=np.float32),
        "transition_next_states": np.asarray(
            transition_next_states,
            dtype=np.float32,
        ).reshape(-1, env.observation_size),
        "transition_dones": np.asarray(transition_dones, dtype=bool),
        "improved_steps": improved_steps,
        "anchor_keep_steps": anchor_keep_steps,
        "service_rejected_steps": service_rejected_steps,
        "mean_step_improvement": float(np.mean(step_improvements)) if step_improvements else 0.0,
        "improved_weight_fraction": (
            float(weight_array[improved_mask].sum()) / weight_total
            if weight_total > 0.0
            else 0.0
        ),
    }


def save_local_search_demonstrations(path: str | Path, demos: dict[str, Any]) -> None:
    """Persist a teacher batch so matched graph/flat agents use identical labels."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "states": np.asarray(demos["states"], dtype=np.float32),
        "actions": np.asarray(demos["actions"], dtype=np.float32),
        "weights": np.asarray(demos["weights"], dtype=np.float32),
        "improved_steps": np.asarray(int(demos["improved_steps"]), dtype=np.int64),
        "anchor_keep_steps": np.asarray(int(demos["anchor_keep_steps"]), dtype=np.int64),
        "service_rejected_steps": np.asarray(
            int(demos["service_rejected_steps"]),
            dtype=np.int64,
        ),
        "mean_step_improvement": np.asarray(
            float(demos["mean_step_improvement"]),
            dtype=np.float64,
        ),
        "improved_weight_fraction": np.asarray(
            float(demos["improved_weight_fraction"]),
            dtype=np.float64,
        ),
    }
    for key, dtype in (
        ("improved_mask", bool),
        ("demand_history_window", np.int64),
        ("scenario_cache_version", np.int64),
        ("scenario_ids", np.int64),
        ("trajectory_ids", np.int64),
        ("trajectory_steps", np.int64),
        ("scenario_names", "U96"),
        ("transition_states", np.float32),
        ("transition_actions", np.float32),
        ("transition_rewards", np.float32),
        ("transition_next_states", np.float32),
        ("transition_dones", bool),
        ("option_advantages", np.float32),
        ("option_feasible", bool),
        ("option_groups", "U32"),
        ("option_epsilons", np.float32),
        ("option_signs", np.float32),
        ("edge_groups", "U32"),
        ("edge_sources", np.int64),
        ("edge_targets", np.int64),
        ("edge_flows", np.float32),
    ):
        if key in demos:
            payload[key] = np.asarray(demos[key], dtype=dtype)
    np.savez_compressed(output_path, **payload)


def load_local_search_demonstrations(path: str | Path) -> dict[str, Any]:
    """Load a fixed teacher batch without allowing pickled object payloads."""

    with np.load(Path(path), allow_pickle=False) as payload:
        states = np.asarray(payload["states"], dtype=np.float32)
        actions = np.asarray(payload["actions"], dtype=np.float32)
        weights = np.asarray(payload["weights"], dtype=np.float32)
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError("Cached demonstrations require rank-2 states and actions")
        if states.shape[0] != actions.shape[0] or weights.shape != (states.shape[0],):
            raise ValueError("Cached demonstration arrays have inconsistent sample counts")
        result = {
            "states": states,
            "actions": actions,
            "weights": weights,
            "improved_steps": int(payload["improved_steps"]),
            "anchor_keep_steps": int(payload["anchor_keep_steps"]),
            "service_rejected_steps": int(payload["service_rejected_steps"]),
            "mean_step_improvement": float(payload["mean_step_improvement"]),
            "improved_weight_fraction": float(payload["improved_weight_fraction"]),
        }
        if "improved_mask" in payload.files:
            improved_mask = np.asarray(payload["improved_mask"], dtype=bool)
            if improved_mask.shape != (states.shape[0],):
                raise ValueError(
                    "Cached demonstration improved_mask has inconsistent shape"
                )
            result["improved_mask"] = improved_mask
        option_keys = (
            "option_advantages",
            "option_feasible",
            "option_groups",
            "option_epsilons",
            "option_signs",
        )
        present_option_keys = tuple(
            key for key in option_keys if key in payload.files
        )
        if present_option_keys and len(present_option_keys) != len(option_keys):
            raise ValueError(
                "Cached option advantages require values, feasibility, and metadata"
            )
        if present_option_keys:
            option_advantages = np.asarray(
                payload["option_advantages"],
                dtype=np.float32,
            )
            option_feasible = np.asarray(payload["option_feasible"], dtype=bool)
            option_groups = np.asarray(payload["option_groups"], dtype="U32")
            option_epsilons = np.asarray(
                payload["option_epsilons"],
                dtype=np.float32,
            )
            option_signs = np.asarray(payload["option_signs"], dtype=np.float32)
            option_count = int(option_groups.size)
            if (
                option_advantages.shape != (states.shape[0], option_count)
                or option_feasible.shape != option_advantages.shape
                or option_epsilons.shape != (option_count,)
                or option_signs.shape != (option_count,)
                or not np.all(np.isfinite(option_advantages))
                or not np.all(np.isfinite(option_epsilons))
                or not np.all(np.isfinite(option_signs))
            ):
                raise ValueError(
                    "Cached option advantage arrays have inconsistent shapes or values"
                )
            result.update(
                {
                    "option_advantages": option_advantages,
                    "option_feasible": option_feasible,
                    "option_groups": option_groups,
                    "option_epsilons": option_epsilons,
                    "option_signs": option_signs,
                }
            )
        edge_keys = (
            "edge_groups",
            "edge_sources",
            "edge_targets",
            "edge_flows",
        )
        present_edge_keys = tuple(
            key for key in edge_keys if key in payload.files
        )
        if present_edge_keys and len(present_edge_keys) != len(edge_keys):
            raise ValueError(
                "Cached edge supervision requires groups, endpoints, and flows"
            )
        if present_edge_keys:
            edge_groups = np.asarray(payload["edge_groups"], dtype="U32")
            edge_sources = np.asarray(payload["edge_sources"], dtype=np.int64)
            edge_targets = np.asarray(payload["edge_targets"], dtype=np.int64)
            edge_flows = np.asarray(payload["edge_flows"], dtype=np.float32)
            expected_rows = states.shape[0]
            if (
                edge_groups.shape != (expected_rows,)
                or edge_sources.ndim != 2
                or edge_sources.shape[0] != expected_rows
                or edge_sources.shape[1] < 1
                or edge_targets.shape != edge_sources.shape
                or edge_flows.shape != edge_sources.shape
                or not np.all(np.isfinite(edge_flows))
                or np.any(edge_flows < 0.0)
            ):
                raise ValueError(
                    "Cached edge supervision arrays have inconsistent shapes or values"
                )
            valid_sources = edge_sources >= 0
            valid_targets = edge_targets >= 0
            if (
                not np.array_equal(valid_sources, valid_targets)
                or np.any(edge_sources[valid_sources] == edge_targets[valid_targets])
                or np.any(edge_flows[~valid_sources] != 0.0)
            ):
                raise ValueError(
                    "Cached edge supervision contains invalid padded endpoints"
                )
            result.update(
                {
                    "edge_groups": edge_groups,
                    "edge_sources": edge_sources,
                    "edge_targets": edge_targets,
                    "edge_flows": edge_flows,
                }
            )
        transition_keys = (
            "transition_states",
            "transition_actions",
            "transition_rewards",
            "transition_next_states",
            "transition_dones",
        )
        if all(key in payload.files for key in transition_keys):
            transition_states = np.asarray(payload["transition_states"], dtype=np.float32)
            transition_actions = np.asarray(payload["transition_actions"], dtype=np.float32)
            transition_rewards = np.asarray(payload["transition_rewards"], dtype=np.float32)
            transition_next_states = np.asarray(
                payload["transition_next_states"],
                dtype=np.float32,
            )
            transition_dones = np.asarray(payload["transition_dones"], dtype=bool)
            transition_count = transition_states.shape[0]
            if (
                transition_states.ndim != 2
                or transition_actions.ndim != 2
                or transition_next_states.ndim != 2
                or transition_actions.shape[0] != transition_count
                or transition_rewards.shape != (transition_count,)
                or transition_next_states.shape[0] != transition_count
                or transition_dones.shape != (transition_count,)
            ):
                raise ValueError("Cached demonstration transitions have inconsistent shapes")
            result.update(
                {
                    "transition_states": transition_states,
                    "transition_actions": transition_actions,
                    "transition_rewards": transition_rewards,
                    "transition_next_states": transition_next_states,
                    "transition_dones": transition_dones,
                }
            )
        if "demand_history_window" in payload.files:
            result["demand_history_window"] = int(
                payload["demand_history_window"]
            )
        if "scenario_cache_version" in payload.files:
            result["scenario_cache_version"] = int(
                payload["scenario_cache_version"]
            )
        if "scenario_names" in payload.files:
            scenario_names = np.asarray(
                payload["scenario_names"],
                dtype="U96",
            )
            if scenario_names.ndim != 1 or scenario_names.size == 0:
                raise ValueError(
                    "Cached scenario_names must be a non-empty vector"
                )
            result["scenario_names"] = scenario_names
        provenance_keys = (
            "scenario_ids",
            "trajectory_ids",
            "trajectory_steps",
        )
        present_provenance = tuple(
            key for key in provenance_keys if key in payload.files
        )
        if present_provenance and len(present_provenance) != len(
            provenance_keys
        ):
            raise ValueError(
                "Cached scenario provenance requires scenario, trajectory, "
                "and within-trajectory indices"
            )
        if present_provenance:
            for key in provenance_keys:
                values = np.asarray(payload[key], dtype=np.int64)
                if values.shape != (states.shape[0],):
                    raise ValueError(
                        f"Cached {key} must align with demonstration rows"
                    )
                if np.any(values < 0):
                    raise ValueError(f"Cached {key} cannot contain negatives")
                result[key] = values
            scenario_names = result.get("scenario_names")
            if scenario_names is None:
                raise ValueError(
                    "Cached scenario provenance requires scenario_names"
                )
            if (
                result["scenario_ids"].size
                and int(result["scenario_ids"].max())
                >= int(scenario_names.size)
            ):
                raise ValueError(
                    "Cached scenario_ids exceed the scenario_names table"
                )
        return result


def balance_demonstration_label_weights(
    demos: dict[str, Any],
) -> dict[str, Any]:
    """Give correction and anchor labels equal aggregate supervised weight."""

    balanced = dict(demos)
    weights = np.asarray(demos["weights"], dtype=np.float32).copy()
    improved = np.asarray(
        demos.get("improved_mask", weights > 1.0 + 1e-6),
        dtype=bool,
    )
    if improved.shape != weights.shape:
        raise ValueError("Demonstration improved_mask must match weights")
    if not np.any(improved) or not np.any(~improved):
        return balanced
    improved_total = float(weights[improved].sum())
    anchor_total = float(weights[~improved].sum())
    if improved_total <= 0.0 or anchor_total <= 0.0:
        return balanced
    weights[improved] *= 0.5 / improved_total
    weights[~improved] *= 0.5 / anchor_total
    weights *= float(weights.size) / float(weights.sum())
    balanced["weights"] = weights
    balanced["improved_mask"] = improved
    balanced["improved_weight_fraction"] = 0.5
    return balanced


def populate_agent_replay_from_demonstrations(agent, demos: dict[str, Any]) -> int:
    """Seed an off-policy critic with the same teacher trajectory used for actor fitting."""

    required = (
        "transition_states",
        "transition_actions",
        "transition_rewards",
        "transition_next_states",
        "transition_dones",
    )
    if not all(key in demos for key in required):
        return 0
    if not hasattr(agent, "observe"):
        raise ValueError("Replay population requires an agent.observe method")
    count = int(np.asarray(demos["transition_states"]).shape[0])
    explicit_labels = None
    label_builder = getattr(agent, "demonstration_option_labels", None)
    transition_adder = getattr(agent, "add_option_transition", None)
    if (
        callable(label_builder)
        and callable(transition_adder)
        and "option_advantages" in demos
        and "option_feasible" in demos
    ):
        explicit_labels = np.asarray(label_builder(demos), dtype=np.int64)
        if explicit_labels.shape != (count,):
            raise ValueError(
                "Dense option labels must align with demonstration transitions"
            )
    for index in range(count):
        if explicit_labels is not None:
            transition_adder(
                demos["transition_states"][index],
                int(explicit_labels[index]),
                float(demos["transition_rewards"][index]),
                demos["transition_next_states"][index],
                bool(demos["transition_dones"][index]),
            )
        else:
            agent.observe(
                demos["transition_states"][index],
                demos["transition_actions"][index],
                float(demos["transition_rewards"][index]),
                demos["transition_next_states"][index],
                bool(demos["transition_dones"][index]),
            )
    return count


def calibrate_demonstration_advantage_weights(
    demos: dict[str, Any],
    *,
    advantage_scale: float,
    weight_cap: float,
) -> dict[str, Any]:
    """Bound raw rollout advantages so correction labels do not erase anchors."""

    scale = float(advantage_scale)
    cap = float(weight_cap)
    if scale <= 0.0:
        raise ValueError("advantage_scale must be positive")
    if cap < 1.0:
        raise ValueError("weight_cap must be at least 1")
    calibrated = dict(demos)
    raw_weights = np.asarray(demos["weights"], dtype=np.float32)
    improved = np.asarray(
        demos.get("improved_mask", raw_weights > 1.0 + 1e-6),
        dtype=bool,
    )
    weights = np.ones_like(raw_weights, dtype=np.float32)
    weights[improved] = np.minimum(
        1.0 + raw_weights[improved] / scale,
        cap,
    )
    total = float(weights.sum())
    calibrated["weights"] = weights
    calibrated["improved_mask"] = improved
    calibrated["improved_weight_fraction"] = (
        float(weights[improved].sum()) / total if total > 0.0 else 0.0
    )
    return calibrated


def calibrate_dense_option_advantage_weights(
    demos: dict[str, Any],
    *,
    advantage_scale: float,
    weight_cap: float,
) -> dict[str, Any]:
    """Reweight corrections from dense feasible option advantages."""

    scale = float(advantage_scale)
    cap = float(weight_cap)
    if scale <= 0.0:
        raise ValueError("advantage_scale must be positive")
    if cap < 1.0:
        raise ValueError("weight_cap must be at least 1")
    if "option_advantages" not in demos or "option_feasible" not in demos:
        raise ValueError(
            "Dense advantage weighting requires option advantages and feasibility"
        )
    advantages = np.asarray(demos["option_advantages"], dtype=np.float32)
    feasible = np.asarray(demos["option_feasible"], dtype=bool)
    if (
        advantages.ndim != 2
        or advantages.shape != feasible.shape
        or advantages.shape[0] != np.asarray(demos["weights"]).shape[0]
    ):
        raise ValueError("Dense option advantage arrays must align with weights")
    if not np.all(np.isfinite(advantages)):
        raise ValueError("Dense option advantages must be finite")
    best_advantage = np.max(
        np.where(feasible, advantages, -np.inf),
        axis=1,
    )
    improved = np.asarray(
        demos.get("improved_mask", best_advantage > 0.0),
        dtype=bool,
    )
    if improved.shape != best_advantage.shape:
        raise ValueError("Demonstration improved_mask must align with advantages")
    weights = np.ones(best_advantage.shape, dtype=np.float32)
    weights[improved] = np.minimum(
        1.0 + np.maximum(best_advantage[improved], 0.0) / scale,
        cap,
    ).astype(np.float32)
    calibrated = dict(demos)
    calibrated["weights"] = weights
    calibrated["improved_mask"] = improved
    total = float(weights.sum())
    calibrated["improved_weight_fraction"] = (
        float(weights[improved].sum()) / total if total > 0.0 else 0.0
    )
    return calibrated


def local_search_candidate_actions(
    state: np.ndarray,
    env,
    baseline,
    *,
    epsilons: tuple[float, ...],
    candidate_groups: tuple[str, ...] | None = None,
    candidate_signs: tuple[float, ...] | None = None,
) -> list[np.ndarray]:
    baseline_action = baseline.select_action(state, explore=False, env=env)
    actions = [baseline_action]
    groups = set(
        candidate_groups
        or (
            "replenishment",
            "reagent_transfer",
            "capacity_transfer",
            "combined_transfer",
        )
    )
    signs = tuple(float(sign) for sign in (candidate_signs or (-1.0, 1.0)))
    if not signs:
        signs = (-1.0, 1.0)
    n = int(env.config.num_facilities)
    _pending_specimens, pending_reagents, pending_capacity = _pending_transfer_vectors(env, n)
    resource_pressure = (
        np.asarray(env.demand, dtype=float)
        + 0.25 * np.asarray(getattr(env, "demand_forecast", env.demand), dtype=float)
        + np.asarray(env.specimens, dtype=float)
        - np.asarray(env.reagents, dtype=float)
        - pending_reagents
    )
    if hasattr(env, "at_risk_counts"):
        resource_pressure = (
            resource_pressure
            + 0.5 * _env_vector(env, "at_risk_counts", n)
            + 0.5 * _env_vector(env, "near_expiry_counts", n)
        )
    capacity_pressure = (
        np.asarray(env.demand, dtype=float)
        + 0.25 * np.asarray(getattr(env, "demand_forecast", env.demand), dtype=float)
        + np.asarray(env.specimens, dtype=float)
        - np.asarray(env.bioreactors[:, 0], dtype=float)
        - pending_capacity
    )
    if hasattr(env, "at_risk_counts"):
        capacity_pressure = (
            capacity_pressure
            + 0.5 * _env_vector(env, "at_risk_counts", n)
            + 0.5 * _env_vector(env, "near_expiry_counts", n)
        )
    patient_risk_pressure = (
        _env_vector(env, "at_risk_counts", n)
        + _env_vector(env, "near_expiry_counts", n)
    )
    resource_pattern = _centered_unit_pattern(resource_pressure)
    capacity_pattern = _centered_unit_pattern(capacity_pressure)
    patient_risk_pattern = _positive_unit_pattern(patient_risk_pressure)
    patient_risk_resource_pattern = _positive_unit_pattern(
        np.maximum(patient_risk_pressure, 0.0)
        * (1.0 + np.maximum(resource_pressure, 0.0))
    )
    for epsilon in epsilons:
        epsilon = float(epsilon)
        for sign in signs:
            if "replenishment" in groups or "replenishment_uniform" in groups:
                uniform = baseline_action.copy()
                uniform[3 * n : 4 * n] = np.clip(
                    uniform[3 * n : 4 * n] + sign * epsilon,
                    -1.0,
                    1.0,
                )
                actions.append(uniform.astype(np.float32))

            if "replenishment" in groups or "replenishment_pressure" in groups:
                pressure_action = baseline_action.copy()
                pressure_action[3 * n : 4 * n] = np.clip(
                    pressure_action[3 * n : 4 * n] + sign * epsilon * resource_pattern,
                    -1.0,
                    1.0,
                )
                actions.append(pressure_action.astype(np.float32))

            if "replenishment_positive_pressure" in groups:
                pressure_action = baseline_action.copy()
                positive_resource_pattern = np.maximum(resource_pattern, 0.0)
                pressure_action[3 * n : 4 * n] = np.clip(
                    pressure_action[3 * n : 4 * n]
                    + max(sign, 0.0) * epsilon * positive_resource_pattern,
                    -1.0,
                    1.0,
                )
                actions.append(pressure_action.astype(np.float32))

            if "replenishment_patient_risk" in groups:
                risk_action = baseline_action.copy()
                risk_action[3 * n : 4 * n] = np.clip(
                    risk_action[3 * n : 4 * n]
                    + max(sign, 0.0) * epsilon * patient_risk_pattern,
                    -1.0,
                    1.0,
                )
                actions.append(risk_action.astype(np.float32))

            if "replenishment_patient_risk_pressure" in groups:
                risk_pressure_action = baseline_action.copy()
                risk_pressure_action[3 * n : 4 * n] = np.clip(
                    risk_pressure_action[3 * n : 4 * n]
                    + max(sign, 0.0) * epsilon * patient_risk_resource_pattern,
                    -1.0,
                    1.0,
                )
                actions.append(risk_pressure_action.astype(np.float32))

            if "reagent_transfer" in groups:
                reagent_transfer = baseline_action.copy()
                reagent_transfer[n : 2 * n] = np.clip(
                    reagent_transfer[n : 2 * n] + sign * epsilon * resource_pattern,
                    -1.0,
                    1.0,
                )
                actions.append(reagent_transfer.astype(np.float32))

            if "capacity_transfer" in groups:
                capacity_transfer = baseline_action.copy()
                capacity_transfer[2 * n : 3 * n] = np.clip(
                    capacity_transfer[2 * n : 3 * n] + sign * epsilon * capacity_pattern,
                    -1.0,
                    1.0,
                )
                actions.append(capacity_transfer.astype(np.float32))

            if "combined_transfer" in groups:
                combined_transfer = baseline_action.copy()
                combined_transfer[n : 2 * n] = np.clip(
                    combined_transfer[n : 2 * n] + sign * epsilon * resource_pattern,
                    -1.0,
                    1.0,
                )
                combined_transfer[2 * n : 3 * n] = np.clip(
                    combined_transfer[2 * n : 3 * n] + sign * epsilon * capacity_pattern,
                    -1.0,
                    1.0,
                )
                actions.append(combined_transfer.astype(np.float32))

            if "reagent_replenishment" in groups:
                resource_action = baseline_action.copy()
                resource_action[n : 2 * n] = np.clip(
                    resource_action[n : 2 * n] + sign * epsilon * resource_pattern,
                    -1.0,
                    1.0,
                )
                resource_action[3 * n : 4 * n] = np.clip(
                    resource_action[3 * n : 4 * n]
                    + sign * epsilon * resource_pattern,
                    -1.0,
                    1.0,
                )
                actions.append(resource_action.astype(np.float32))

            if "combined_network" in groups:
                network_action = baseline_action.copy()
                network_action[n : 2 * n] = np.clip(
                    network_action[n : 2 * n] + sign * epsilon * resource_pattern,
                    -1.0,
                    1.0,
                )
                network_action[2 * n : 3 * n] = np.clip(
                    network_action[2 * n : 3 * n] + sign * epsilon * capacity_pattern,
                    -1.0,
                    1.0,
                )
                network_action[3 * n : 4 * n] = np.clip(
                    network_action[3 * n : 4 * n]
                    + sign * epsilon * resource_pattern,
                    -1.0,
                    1.0,
                )
                actions.append(network_action.astype(np.float32))

            if sign > 0.0 and "reagent_transfer_shrink" in groups:
                reagent_shrink = baseline_action.copy()
                reagent_shrink[n : 2 * n] *= max(1.0 - epsilon, 0.0)
                actions.append(reagent_shrink.astype(np.float32))

            if sign > 0.0 and "capacity_transfer_shrink" in groups:
                capacity_shrink = baseline_action.copy()
                capacity_shrink[2 * n : 3 * n] *= max(1.0 - epsilon, 0.0)
                actions.append(capacity_shrink.astype(np.float32))

            if sign > 0.0 and "combined_transfer_shrink" in groups:
                combined_shrink = baseline_action.copy()
                combined_shrink[n : 3 * n] *= max(1.0 - epsilon, 0.0)
                actions.append(combined_shrink.astype(np.float32))

            if sign > 0.0 and "all_transfer_shrink" in groups:
                all_transfer_shrink = baseline_action.copy()
                all_transfer_shrink[: 3 * n] *= max(1.0 - epsilon, 0.0)
                actions.append(all_transfer_shrink.astype(np.float32))
    return actions


def _centered_unit_pattern(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    denominator = max(float(np.max(np.abs(centered))), 1e-6)
    return centered / denominator


def _positive_unit_pattern(values: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(values, dtype=float), 0.0)
    denominator = max(float(np.max(positive)), 1e-6)
    return positive / denominator


def _env_vector(env, name: str, length: int) -> np.ndarray:
    value = getattr(env, name, None)
    if value is None:
        return np.zeros(int(length), dtype=float)
    if callable(value):
        value = value()
    vector = np.asarray(value, dtype=float)
    if vector.shape != (int(length),):
        raise ValueError(f"Expected env.{name} shape {(int(length),)}, got {vector.shape}")
    return vector


def _pending_transfer_vectors(env, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pending = getattr(env, "_pending_transfer_arrivals", None)
    if callable(pending):
        vectors = pending()
        if len(vectors) == 3:
            return tuple(np.asarray(vector, dtype=float) for vector in vectors)  # type: ignore[return-value]
    zeros = np.zeros(int(length), dtype=float)
    return (
        _pipeline_pending_vector(env, "specimen_transfer_pipeline", length, zeros),
        _pipeline_pending_vector(env, "reagent_transfer_pipeline", length, zeros),
        _pipeline_pending_vector(env, "capacity_transfer_pipeline", length, zeros),
    )


def _pipeline_pending_vector(env, name: str, length: int, default: np.ndarray) -> np.ndarray:
    pipeline = getattr(env, name, None)
    if pipeline is None:
        return default.copy()
    array = np.asarray(pipeline, dtype=float)
    if array.ndim != 2 or array.shape[1] != int(length):
        return default.copy()
    return array.sum(axis=0)


def rollout_cost_after_action(env, baseline, action: np.ndarray, *, horizon: int) -> float:
    return float(rollout_metrics_after_action(env, baseline, action, horizon=horizon)["total_cost"])


def local_search_metric_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    """Cost-style score where lower is better and patient metrics can be valued."""

    return (
        float(metrics["total_cost"])
        - float(weights.get("service_level", 0.0)) * float(metrics.get("service_level", 0.0))
        - float(weights.get("completion_service_level", 0.0))
        * float(metrics.get("completion_service_level", 0.0))
        - float(weights.get("eligibility_rate", 0.0)) * float(metrics.get("eligibility_rate", 0.0))
        + float(weights.get("patient_ineligibility_during_manufacturing_rate", 0.0))
        * float(metrics.get("patient_ineligibility_during_manufacturing_rate", 0.0))
        + float(weights.get("at_risk_unserved", 0.0))
        * float(metrics.get("at_risk_unserved", 0.0))
        + float(weights.get("patients_lost", 0.0)) * float(metrics.get("patients_lost", 0.0))
    )


def mean_rollout_metrics_after_action(
    env,
    baseline,
    action: np.ndarray,
    *,
    horizon: int,
    rollout_seeds: tuple[int, ...],
) -> dict[str, float]:
    """Estimate candidate value with CRN draws independent of the live episode."""

    seeds = tuple(int(seed) for seed in rollout_seeds)
    if not seeds:
        raise ValueError("rollout_seeds must contain at least one seed")
    rows = [
        rollout_metrics_after_action(
            copy.deepcopy(env),
            baseline,
            action,
            horizon=horizon,
            rollout_seed=seed,
        )
        for seed in seeds
    ]
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0]
    }


def rollout_metrics_after_action(
    env,
    baseline,
    action: np.ndarray,
    *,
    horizon: int,
    rollout_seed: int | None = None,
) -> dict[str, float]:
    if rollout_seed is not None:
        env.rng = np.random.default_rng(int(rollout_seed))
    state, _reward, done, info = env.step(action)
    metrics = EpisodeMetrics()
    metrics.update(info)
    steps = 1
    while not done and steps < max(int(horizon), 1):
        followup_action = baseline.select_action(state, explore=False, env=env)
        state, _reward, done, info = env.step(followup_action)
        metrics.update(info)
        steps += 1
    return {
        "total_cost": float(metrics.total_cost),
        "service_level": float(metrics.service_level),
        "eligibility_rate": float(metrics.eligibility_rate_mean)
        if getattr(metrics, "has_patient_metrics", False)
        else float("nan"),
        "completion_service_level": float(metrics.completion_service_level_last),
        "patient_ineligibility_during_manufacturing_rate": float(
            metrics.patient_ineligibility_during_manufacturing_rate_last
        ),
        "at_risk_unserved": float(metrics.at_risk_unserved),
        "patients_lost": float(metrics.patients_lost),
    }


def _minimum_metric_delta_satisfied(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    key: str,
    minimum_delta: float | None,
) -> bool:
    if minimum_delta is None:
        return True
    value = float(metrics.get(key, float("nan")))
    baseline = float(baseline_metrics.get(key, float("nan")))
    return bool(
        np.isfinite(value)
        and np.isfinite(baseline)
        and value >= baseline + float(minimum_delta)
    )


def _maximum_metric_delta_satisfied(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    key: str,
    maximum_delta: float | None,
) -> bool:
    if maximum_delta is None:
        return True
    value = float(metrics.get(key, float("nan")))
    baseline = float(baseline_metrics.get(key, float("nan")))
    return bool(
        np.isfinite(value)
        and np.isfinite(baseline)
        and value <= baseline + float(maximum_delta)
    )


def elite_sample_weights(
    elites: list[tuple[float, float, float, float, np.ndarray, np.ndarray]],
    *,
    weighting: str,
    power: float,
    floor: float,
) -> np.ndarray | None:
    if weighting == "none" or not elites:
        return None
    if floor < 0.0:
        raise ValueError("offline elite weight floor must be non-negative")
    if power <= 0.0:
        raise ValueError("offline elite weight power must be positive")

    if weighting == "improvement":
        rollout_weights = np.asarray([max(float(item[3]), 0.0) for item in elites], dtype=np.float32)
        if float(rollout_weights.sum()) <= 0.0:
            return None
        rollout_weights = rollout_weights / float(rollout_weights.mean())
    elif weighting == "rank":
        rollout_weights = np.asarray(
            [len(elites) - rank for rank, _item in enumerate(elites)],
            dtype=np.float32,
        )
        rollout_weights = rollout_weights / float(rollout_weights.mean())
    else:
        raise ValueError(f"Unsupported offline elite weighting: {weighting}")

    rollout_weights = np.power(rollout_weights, float(power)).astype(np.float32)
    if floor > 0.0:
        rollout_weights = np.maximum(rollout_weights, float(floor)).astype(np.float32)
    rollout_weights = rollout_weights / float(rollout_weights.mean())
    sample_weights = [
        np.full(item[4].shape[0], rollout_weights[index], dtype=np.float32)
        for index, item in enumerate(elites)
    ]
    weights = np.concatenate(sample_weights, axis=0)
    return (weights / float(weights.mean())).astype(np.float32)


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
    center_residual_groups: tuple[str, ...] = (),
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
    if center_residual_groups:
        residual_action["center_groups"] = list(center_residual_groups)
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
    center_residual_groups: tuple[str, ...] = (),
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
    if center_residual_groups:
        center_token = "-".join(str(group).replace("_", "") for group in center_residual_groups)
        variant = f"{variant}_center{center_token}"
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
