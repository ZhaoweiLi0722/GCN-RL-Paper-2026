"""Diagnose weak online DDPG attribution from immutable campaign artifacts.

The diagnostic is read-only. It combines the locked Stage B checkpoint curve,
training logs, frozen teacher support, checkpoint critics, and final replay
buffers. It never trains, evaluates a simulator policy, or selects a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluation.audit_ddpg_critic_advantage_ranking import (
    critic_advantage_metrics,
)
from evaluation.run_gcn_residual_sweep import (
    load_local_search_demonstrations,
)
from src.models.graph_features import flat_state_to_node_features
from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.critic_advantage import teacher_advantage_support_mask
from src.rl.experiment import build_env
from src.rl.networks import torch


DEFAULT_STAGE_SPEC = (
    "experiments/configs/"
    "patient_indexed_specimen_routing_mac_mps_ddpg_confirmation_100_"
    "online_attribution_stage_b.json"
)
LOCKED_CURVE_SHA256 = (
    "0016e3bb8fa779984de1a21015bcade6f377d72a231b7581adc14ffc5a652ecf"
)
FORMAL_HOLDOUT_SEED = 91100000
CHECKPOINT_EPISODES = {
    "pretrain": 0,
    "episode25": 25,
    "episode50": 50,
    "episode75": 75,
    "episode100": 100,
}
PAIRING_KEYS = (
    "training_seed",
    "scenario",
    "evaluation_seed",
    "replication",
)
HIGH_LEVEL_COST_FIELDS = (
    "base_cost",
    "specimen_transfer_cost",
    "capacity_transfer_cost",
    "reagent_transfer_cost",
    "patient_loss_cost",
    "expiry_cost",
    "urgency_cost",
)
BASE_COST_FIELDS = (
    "reagent_purchase_cost",
    "reagent_holding_cost",
    "reagent_shortage_cost",
    "bioreactor_holding_cost",
    "bioreactor_shortage_cost",
)
TRAINING_METRICS = (
    "online_rl_updates",
    "online_rl_actor_updated_mean",
    "online_rl_actor_loss_mean",
    "online_rl_critic_bellman_loss_mean",
    "online_rl_critic_teacher_advantage_weighted_ranking_loss_mean",
    "online_rl_anchor_relative_reward_mean_abs_mean",
    "online_rl_anchor_relative_reward_positive_rate_mean",
    "online_rl_online_advantage_self_imitation_active_fraction_mean",
    "online_rl_online_advantage_self_imitation_eligible_fraction_mean",
    "online_rl_online_advantage_self_imitation_weighted_loss_mean",
    "online_rl_pretrain_reference_weighted_loss_mean",
    "online_rl_pretrain_reference_parameter_drift_rms_mean",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--stage-spec", default=DEFAULT_STAGE_SPEC)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairwise-samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    result = run_diagnosis(
        source_root=Path(args.source_root),
        artifact_root=Path(args.artifact_root),
        stage_spec_path=Path(args.stage_spec),
        pairwise_samples=int(args.pairwise_samples),
        batch_size=int(args.batch_size),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["diagnostic_summary"], indent=2, sort_keys=True))


def run_diagnosis(
    *,
    source_root: Path,
    artifact_root: Path,
    stage_spec_path: Path,
    pairwise_samples: int,
    batch_size: int,
) -> dict[str, Any]:
    if pairwise_samples <= 0:
        raise ValueError("pairwise_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source_root = source_root.resolve()
    artifact_root = artifact_root.resolve()
    spec_path = _resolve(source_root, stage_spec_path)
    spec = _load_json(spec_path)
    if int(spec["development_evaluation_seed"]) == FORMAL_HOLDOUT_SEED:
        raise ValueError("Stage B diagnostic cannot reuse the formal holdout")

    curve_path = _resolve(artifact_root, Path(spec["curve_summary"]))
    manifest_path = _resolve(artifact_root, Path(spec["training_manifest"]))
    teacher_path = _resolve(artifact_root, Path(spec["teacher"]))
    locked_hashes = {
        "checkpoint_curve": LOCKED_CURVE_SHA256,
        "training_manifest": str(spec["training_manifest_sha256"]),
        "teacher": str(spec["teacher_sha256"]),
    }
    observed_hashes = {
        "checkpoint_curve": sha256_file(curve_path),
        "training_manifest": sha256_file(manifest_path),
        "teacher": sha256_file(teacher_path),
    }
    if observed_hashes != locked_hashes:
        raise ValueError(
            "Locked Stage B input hash mismatch: "
            + json.dumps(
                {"expected": locked_hashes, "observed": observed_hashes},
                sort_keys=True,
            )
        )

    curve = _load_json(curve_path)
    if not bool(curve.get("crn_audit", {}).get("passed", False)):
        raise ValueError("Stage B checkpoint-curve CRN audit did not pass")
    manifest = _load_json(manifest_path)
    algorithms = tuple(str(value) for value in spec["algorithms"])
    seeds = tuple(int(value) for value in spec["training_seeds"])
    variants = tuple(str(value) for value in spec["checkpoint_variants"])
    if tuple(CHECKPOINT_EPISODES) != variants:
        raise ValueError("Unexpected Stage B checkpoint variants")

    phase_runs = _load_phase_runs(
        source_root=source_root,
        artifact_root=artifact_root,
        spec=spec,
        variants=variants,
    )
    manifest_runs = _manifest_runs(manifest, algorithms, seeds)
    teacher = load_local_search_demonstrations(teacher_path)
    support = teacher_support_summary(
        teacher,
        allowed_option_groups=("specimen_transfer",),
    )
    behavioral = checkpoint_curve_behavior(curve, algorithms, variants)
    paired = paired_final_pretrain_decomposition(
        phase_runs=phase_runs,
        algorithms=algorithms,
    )
    training = training_dynamics(
        artifact_root=artifact_root,
        manifest_runs=manifest_runs,
        algorithms=algorithms,
        seeds=seeds,
    )
    critic = checkpoint_critic_diagnosis(
        artifact_root=artifact_root,
        phase_runs=phase_runs,
        manifest_runs=manifest_runs,
        teacher=teacher,
        algorithms=algorithms,
        seeds=seeds,
        variants=variants,
        pairwise_samples=pairwise_samples,
        batch_size=batch_size,
    )

    return {
        "role": (
            "read-only development diagnosis; not training, checkpoint "
            "selection, formal evidence, or deployment tuning"
        ),
        "source_root": str(source_root),
        "artifact_root": str(artifact_root),
        "stage_spec": str(spec_path),
        "input_hashes": observed_hashes,
        "formal_holdout_reused": False,
        "development_evaluation_seed": int(
            spec["development_evaluation_seed"]
        ),
        "teacher_policy_support": support,
        "checkpoint_curve": behavioral,
        "final_vs_pretrain_decomposition": paired,
        "training_dynamics": training,
        "critic_diagnosis": critic,
        "diagnostic_summary": diagnostic_summary(
            support=support,
            behavioral=behavioral,
            paired=paired,
            training=training,
            critic=critic,
        ),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_support_summary(
    demonstrations: dict[str, Any],
    *,
    allowed_option_groups: Iterable[str],
) -> dict[str, Any]:
    advantages = np.asarray(
        demonstrations["option_advantages"], dtype=np.float64
    )
    feasible = np.asarray(demonstrations["option_feasible"], dtype=bool)
    groups = np.asarray(demonstrations["option_groups"], dtype="U64")
    best = np.argmax(np.where(feasible, advantages, -np.inf), axis=1)
    best_groups = groups[best]
    best_advantages = advantages[np.arange(advantages.shape[0]), best]
    supported = teacher_advantage_support_mask(
        advantages,
        feasible,
        groups,
        allowed_option_groups=tuple(allowed_option_groups),
    )
    counts = Counter(str(value) for value in best_groups)
    positive_counts = Counter(
        str(group)
        for group, value in zip(best_groups, best_advantages)
        if float(value) > 0.0
    )
    return {
        "samples": int(advantages.shape[0]),
        "option_count": int(advantages.shape[1]),
        "allowed_option_groups": [str(value) for value in allowed_option_groups],
        "globally_best_group_counts": dict(sorted(counts.items())),
        "positive_best_group_counts": dict(sorted(positive_counts.items())),
        "policy_supported_samples": int(supported.sum()),
        "policy_supported_fraction": float(supported.mean()),
        "policy_unsupported_samples": int((~supported).sum()),
        "policy_unsupported_fraction": float((~supported).mean()),
    }


def checkpoint_curve_behavior(
    curve: dict[str, Any],
    algorithms: tuple[str, ...],
    variants: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "locked_decision": dict(curve["decision"]),
        "algorithms": {},
    }
    for algorithm in algorithms:
        source = curve["algorithms"][algorithm]
        checkpoints = {}
        for variant in variants:
            entry = source["checkpoints"][variant]
            checkpoints[variant] = {
                "total_cost_vs_pretrain": dict(
                    entry["vs_pretrain"]["total_cost"]
                ),
                "residual_usage": dict(entry["residual_usage"]),
                "actor_drift_from_pretrain": dict(
                    entry["actor_drift_from_pretrain"]
                ),
            }
        segments = {
            name: {
                "total_cost": dict(value["metrics"]["total_cost"]),
                "clinical_noninferiority": dict(
                    value["clinical_noninferiority"]
                ),
            }
            for name, value in source["segments"].items()
        }
        result["algorithms"][algorithm] = {
            "checkpoints": checkpoints,
            "segments": segments,
        }
    return result


def paired_final_pretrain_decomposition(
    *,
    phase_runs: dict[str, dict[tuple[str, int], dict[str, Any]]],
    algorithms: tuple[str, ...],
) -> dict[str, Any]:
    result = {}
    for algorithm in algorithms:
        pretrain_rows = _combined_rows(phase_runs["pretrain"], algorithm)
        final_rows = _combined_rows(phase_runs["episode100"], algorithm)
        pretrain = _rows_by_key(pretrain_rows)
        final = _rows_by_key(final_rows)
        if pretrain.keys() != final.keys():
            raise ValueError(f"Final/pretrain CRN mismatch for {algorithm}")
        differences = {
            field: np.asarray(
                [float(final[key][field]) - float(pretrain[key][field]) for key in final],
                dtype=np.float64,
            )
            for field in (
                "total_cost",
                *HIGH_LEVEL_COST_FIELDS,
                *BASE_COST_FIELDS,
                "patients_lost",
                "patients_completed",
                "completion_service_level",
                "specimen_route_count",
                "specimen_route_distance_miles",
                "specimen_route_time_hours",
            )
        }
        total_by_seed: dict[str, list[float]] = defaultdict(list)
        total_by_scenario: dict[str, list[float]] = defaultdict(list)
        total_by_seed_scenario: dict[str, list[float]] = defaultdict(list)
        changed = 0
        route_changed = 0
        for key in final:
            left = pretrain[key]
            right = final[key]
            difference = float(right["total_cost"]) - float(left["total_cost"])
            total_by_seed[str(right["training_seed"])].append(difference)
            total_by_scenario[str(right["scenario"])].append(difference)
            total_by_seed_scenario[
                f"seed{right['training_seed']}:{right['scenario']}"
            ].append(difference)
            changed += int(
                float(right["total_cost"]) != float(left["total_cost"])
            )
            route_changed += int(
                float(right["specimen_route_count"])
                != float(left["specimen_route_count"])
                or float(right["specimen_route_distance_miles"])
                != float(left["specimen_route_distance_miles"])
            )
        result[algorithm] = {
            "pairs": len(final),
            "mean_differences": {
                field: float(values.mean())
                for field, values in differences.items()
            },
            "high_level_cost_sum_check": float(
                sum(
                    differences[field].mean()
                    for field in HIGH_LEVEL_COST_FIELDS
                )
                - differences["total_cost"].mean()
            ),
            "per_seed_total_cost_mean_difference": {
                key: float(np.mean(values))
                for key, values in sorted(total_by_seed.items())
            },
            "per_scenario_total_cost_mean_difference": {
                key: float(np.mean(values))
                for key, values in sorted(total_by_scenario.items())
            },
            "per_seed_scenario_total_cost_mean_difference": {
                key: float(np.mean(values))
                for key, values in sorted(total_by_seed_scenario.items())
            },
            "changed_total_cost_fraction": float(changed / len(final)),
            "changed_routing_fraction": float(route_changed / len(final)),
        }
    return result


def training_dynamics(
    *,
    artifact_root: Path,
    manifest_runs: dict[tuple[str, int], dict[str, Any]],
    algorithms: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for algorithm in algorithms:
        windows: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        per_seed_rows = {}
        for seed in seeds:
            run = manifest_runs[(algorithm, seed)]
            config = _load_json(_resolve(artifact_root, Path(run["config"])))
            csv_path = _resolve(artifact_root, Path(config["result_csv_path"]))
            rows = _read_csv(csv_path)
            if len(rows) != 100:
                raise ValueError(
                    f"Training CSV for {algorithm} seed {seed} has {len(rows)} rows"
                )
            episodes = [int(row["episode"]) for row in rows]
            if episodes != list(range(100)):
                raise ValueError(
                    f"Training episodes are not 0..99 for {algorithm} seed {seed}"
                )
            per_seed_rows[str(seed)] = len(rows)
            for row in rows:
                episode = int(row["episode"]) + 1
                start = ((episode - 1) // 25) * 25 + 1
                label = f"episodes{start}-{start + 24}"
                for metric in TRAINING_METRICS:
                    value = float(row[metric])
                    if not math.isfinite(value):
                        raise ValueError(
                            f"Nonfinite {metric} for {algorithm} seed {seed}"
                        )
                    windows[label][metric].append(value)
                if float(row["online_rl_updates"]) <= 0.0:
                    raise ValueError(
                        f"Zero online updates for {algorithm} seed {seed}"
                    )
        result[algorithm] = {
            "rows_per_seed": per_seed_rows,
            "windows": {
                label: {
                    metric: float(np.mean(values))
                    for metric, values in metrics.items()
                }
                for label, metrics in windows.items()
            },
        }
    return result


def checkpoint_critic_diagnosis(
    *,
    artifact_root: Path,
    phase_runs: dict[str, dict[tuple[str, int], dict[str, Any]]],
    manifest_runs: dict[tuple[str, int], dict[str, Any]],
    teacher: dict[str, Any],
    algorithms: tuple[str, ...],
    seeds: tuple[int, ...],
    variants: tuple[str, ...],
    pairwise_samples: int,
    batch_size: int,
) -> dict[str, Any]:
    supported_mask = teacher_advantage_support_mask(
        teacher["option_advantages"],
        teacher["option_feasible"],
        teacher["option_groups"],
        allowed_option_groups=("specimen_transfer",),
    )
    pooled: dict[
        str,
        dict[str, dict[str, dict[str, list[np.ndarray]]]],
    ] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    per_seed: dict[str, dict[str, Any]] = defaultdict(dict)
    checkpoint_hashes: dict[str, str] = {}
    training_state_hashes: dict[str, str] = {}
    configured_option_groups: dict[str, list[str]] = {}
    replay_composition: dict[str, dict[str, Any]] = {}

    for algorithm in algorithms:
        for seed in seeds:
            key = (algorithm, seed)
            run = manifest_runs[key]
            config_path = _resolve(artifact_root, Path(run["config"]))
            config = load_config(config_path)
            config["device"] = "cpu"
            config["replay_buffer_size"] = 1
            configured_option_groups[f"{algorithm}:seed{seed}"] = [
                str(value)
                for value in config["critic_teacher_advantage_calibration"].get(
                    "allowed_option_groups", ()
                )
            ]
            env = build_env(config, seed=seed)
            agent = get_agent_class(algorithm)(
                env.observation_size,
                env.action_size,
                config,
            )
            target_scale = float(
                config["critic_teacher_advantage_calibration"].get(
                    "target_scale", 1.0
                )
            )
            teacher_targets = _teacher_targets(
                teacher,
                reward_scale=float(agent.reward_scale),
                target_scale=target_scale,
            )
            teacher_states = np.asarray(teacher["states"], dtype=np.float32)
            teacher_actions = np.asarray(teacher["actions"], dtype=np.float32)
            teacher_base = _base_actions(agent, teacher_states, batch_size=batch_size)

            training_state_path = _resolve(
                artifact_root, Path(run["training_state_checkpoint"])
            )
            training_state_hashes[f"{algorithm}:seed{seed}"] = sha256_file(
                training_state_path
            )
            replay = torch.load(
                training_state_path,
                map_location="cpu",
                weights_only=False,
            )["agent"]["replay_buffer"]
            online_mask = np.asarray(replay["online_mask"], dtype=bool)
            online_states = np.asarray(replay["states"], dtype=np.float32)[
                online_mask
            ]
            online_actions = np.asarray(replay["actions"], dtype=np.float32)[
                online_mask
            ]
            online_targets = np.asarray(replay["rewards"], dtype=np.float32)[
                online_mask
            ].reshape(-1)
            replay_composition[f"{algorithm}:seed{seed}"] = {
                "total_transitions": int(online_mask.size),
                "online_transitions": int(online_mask.sum()),
                "offline_transitions": int((~online_mask).sum()),
                "online_fraction": float(online_mask.mean()),
                "configured_online_replay_fraction": config.get(
                    "online_replay_fraction"
                ),
            }
            expected_online = int(config["num_episodes"]) * int(
                config["max_steps_per_episode"]
            )
            if online_states.shape[0] != expected_online:
                raise ValueError(
                    f"Online replay count mismatch for {algorithm} seed {seed}: "
                    f"{online_states.shape[0]} != {expected_online}"
                )
            online_base = _base_actions(agent, online_states, batch_size=batch_size)
            seed_result: dict[str, Any] = {}
            for variant_index, variant in enumerate(variants):
                checkpoint_path = _resolve(
                    artifact_root,
                    Path(phase_runs[variant][key]["checkpoint"]),
                )
                checkpoint_hashes[
                    f"{algorithm}:seed{seed}:{variant}"
                ] = sha256_file(checkpoint_path)
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if str(checkpoint["algorithm"]) != str(agent.algorithm):
                    raise ValueError(
                        "Checkpoint algorithm mismatch for "
                        f"{algorithm} seed {seed} {variant}: "
                        f"{checkpoint['algorithm']} != {agent.algorithm}"
                    )
                if int(checkpoint["state_dim"]) != env.observation_size:
                    raise ValueError("Checkpoint state dimension mismatch")
                if int(checkpoint["action_dim"]) != env.action_size:
                    raise ValueError("Checkpoint action dimension mismatch")
                agent.critic.load_state_dict(checkpoint["critic"])
                agent.critic.eval()

                teacher_predictions = _critic_predictions(
                    agent,
                    teacher_states,
                    teacher_actions,
                    teacher_base,
                    batch_size=batch_size,
                )
                replay_predictions = _critic_predictions(
                    agent,
                    online_states,
                    online_actions,
                    online_base,
                    batch_size=batch_size,
                )
                pair_seed = 7_000_000 + seed * 1_000 + variant_index
                full_replay = sampled_advantage_metrics(
                    online_targets,
                    replay_predictions,
                    pairwise_samples=pairwise_samples,
                    seed=pair_seed,
                )
                episode = CHECKPOINT_EPISODES[variant]
                prefix_count = episode * int(config["max_steps_per_episode"])
                prefix = None
                if prefix_count:
                    prefix = sampled_advantage_metrics(
                        online_targets[:prefix_count],
                        replay_predictions[:prefix_count],
                        pairwise_samples=pairwise_samples,
                        seed=pair_seed + 500_000,
                    )
                seed_result[variant] = {
                    "teacher_all_options": critic_advantage_metrics(
                        teacher_targets, teacher_predictions
                    ),
                    "teacher_policy_supported": critic_advantage_metrics(
                        teacher_targets[supported_mask],
                        teacher_predictions[supported_mask],
                    ),
                    "fixed_full_online_replay": full_replay,
                    "observed_replay_prefix": prefix,
                    "observed_replay_prefix_samples": prefix_count,
                }
                _collect_pair(
                    pooled[algorithm][variant],
                    "teacher_all_options",
                    teacher_targets,
                    teacher_predictions,
                )
                _collect_pair(
                    pooled[algorithm][variant],
                    "teacher_policy_supported",
                    teacher_targets[supported_mask],
                    teacher_predictions[supported_mask],
                )
                _collect_pair(
                    pooled[algorithm][variant],
                    "fixed_full_online_replay",
                    online_targets,
                    replay_predictions,
                )
                if prefix_count:
                    _collect_pair(
                        pooled[algorithm][variant],
                        "observed_replay_prefix",
                        online_targets[:prefix_count],
                        replay_predictions[:prefix_count],
                    )
            per_seed[algorithm][str(seed)] = seed_result

    pooled_metrics: dict[str, Any] = {}
    for algorithm in algorithms:
        pooled_metrics[algorithm] = {}
        for variant_index, variant in enumerate(variants):
            pooled_metrics[algorithm][variant] = {}
            for label, arrays in pooled[algorithm][variant].items():
                targets = np.concatenate(arrays["targets"])
                predictions = np.concatenate(arrays["predictions"])
                if label.startswith("teacher"):
                    metrics = critic_advantage_metrics(targets, predictions)
                else:
                    metrics = sampled_advantage_metrics(
                        targets,
                        predictions,
                        pairwise_samples=pairwise_samples,
                        seed=8_000_000 + variant_index,
                    )
                pooled_metrics[algorithm][variant][label] = metrics
    return {
        "per_seed": dict(per_seed),
        "pooled": pooled_metrics,
        "checkpoint_sha256": checkpoint_hashes,
        "training_state_sha256": training_state_hashes,
        "teacher_policy_supported_mask_samples": int(supported_mask.sum()),
        "configured_teacher_allowed_option_groups": configured_option_groups,
        "final_replay_composition": replay_composition,
    }


def sampled_advantage_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    pairwise_samples: int,
    seed: int,
) -> dict[str, Any]:
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if target.shape != predicted.shape or target.size == 0:
        raise ValueError("Targets and predictions must be aligned and non-empty")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Targets and predictions must be finite")
    rng = np.random.default_rng(seed)
    left = rng.integers(0, target.size, size=pairwise_samples)
    right = rng.integers(0, target.size, size=pairwise_samples)
    distinct = left != right
    left = left[distinct]
    right = right[distinct]
    target_difference = target[left] - target[right]
    comparable = np.abs(target_difference) > 1e-15
    prediction_difference = predicted[left] - predicted[right]
    products = target_difference[comparable] * prediction_difference[comparable]
    scores = np.where(products > 0.0, 1.0, np.where(products == 0.0, 0.5, 0.0))
    target_positive = target > 0.0
    predicted_positive = predicted > 0.0
    target_std = float(target.std())
    prediction_std = float(predicted.std())
    covariance = float(np.mean((target - target.mean()) * (predicted - predicted.mean())))
    return {
        "samples": int(target.size),
        "target_mean": float(target.mean()),
        "target_std": target_std,
        "target_positive_fraction": float(target_positive.mean()),
        "prediction_mean": float(predicted.mean()),
        "prediction_std": prediction_std,
        "prediction_positive_fraction": float(predicted_positive.mean()),
        "mse": float(np.mean((predicted - target) ** 2)),
        "mae": float(np.mean(np.abs(predicted - target))),
        "pearson": _correlation(target, predicted),
        "spearman": _correlation(_average_ranks(target), _average_ranks(predicted)),
        "calibration_slope": (
            float(covariance / (prediction_std**2))
            if prediction_std > 1e-15
            else None
        ),
        "pairwise_sampled_pairs": int(scores.size),
        "pairwise_order_accuracy": float(scores.mean()) if scores.size else None,
        "positive_sign_accuracy": float(
            np.mean(target_positive == predicted_positive)
        ),
        "positive_recall": (
            float(np.mean(predicted_positive[target_positive]))
            if np.any(target_positive)
            else None
        ),
        "nonpositive_specificity": (
            float(np.mean(~predicted_positive[~target_positive]))
            if np.any(~target_positive)
            else None
        ),
    }


def diagnostic_summary(
    *,
    support: dict[str, Any],
    behavioral: dict[str, Any],
    paired: dict[str, Any],
    training: dict[str, Any],
    critic: dict[str, Any],
) -> dict[str, Any]:
    graph = "gcn_residual_mdl2_network_ddpg_afd"
    flat = "flat_residual_mdl2_network_ddpg_afd"
    graph_curve = behavioral["algorithms"][graph]
    variants = tuple(CHECKPOINT_EPISODES)

    def ranking_association(algorithm: str) -> dict[str, float]:
        costs = np.asarray(
            [
                behavioral["algorithms"][algorithm]["checkpoints"][variant][
                    "total_cost_vs_pretrain"
                ]["mean_difference"]
                for variant in variants
            ],
            dtype=np.float64,
        )
        replay_spearman = np.asarray(
            [
                critic["pooled"][algorithm][variant][
                    "fixed_full_online_replay"
                ]["spearman"]
                for variant in variants
            ],
            dtype=np.float64,
        )
        replay_pairwise = np.asarray(
            [
                critic["pooled"][algorithm][variant][
                    "fixed_full_online_replay"
                ]["pairwise_order_accuracy"]
                for variant in variants
            ],
            dtype=np.float64,
        )
        return {
            "cost_vs_replay_spearman_correlation": float(
                np.corrcoef(costs, replay_spearman)[0, 1]
            ),
            "cost_vs_replay_pairwise_accuracy_correlation": float(
                np.corrcoef(costs, replay_pairwise)[0, 1]
            ),
        }

    return {
        "locked_stage_b_classification": behavioral["locked_decision"][
            "classification"
        ],
        "gcn_episode50_vs_pretrain_total_cost": graph_curve["checkpoints"][
            "episode50"
        ]["total_cost_vs_pretrain"]["mean_difference"],
        "gcn_episode100_vs_pretrain_total_cost": graph_curve["checkpoints"][
            "episode100"
        ]["total_cost_vs_pretrain"]["mean_difference"],
        "gcn_episode75_to_100_total_cost": graph_curve["segments"][
            "episode75_to_episode100"
        ]["total_cost"]["mean_difference"],
        "teacher_policy_unsupported_fraction": support[
            "policy_unsupported_fraction"
        ],
        "gcn_final_changed_routing_fraction": paired[graph][
            "changed_routing_fraction"
        ],
        "flat_final_changed_routing_fraction": paired[flat][
            "changed_routing_fraction"
        ],
        "gcn_first_window_updates_per_episode": training[graph]["windows"][
            "episodes1-25"
        ]["online_rl_updates"],
        "gcn_teacher_supported_spearman_pretrain": critic["pooled"][graph][
            "pretrain"
        ]["teacher_policy_supported"]["spearman"],
        "gcn_teacher_supported_spearman_episode100": critic["pooled"][graph][
            "episode100"
        ]["teacher_policy_supported"]["spearman"],
        "gcn_online_replay_spearman_pretrain": critic["pooled"][graph][
            "pretrain"
        ]["fixed_full_online_replay"]["spearman"],
        "gcn_online_replay_spearman_episode100": critic["pooled"][graph][
            "episode100"
        ]["fixed_full_online_replay"]["spearman"],
        "descriptive_checkpoint_association": {
            graph: ranking_association(graph),
            flat: ranking_association(flat),
        },
    }


def _load_phase_runs(
    *,
    source_root: Path,
    artifact_root: Path,
    spec: dict[str, Any],
    variants: tuple[str, ...],
) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    result = {}
    for variant in variants:
        config_path = _resolve(
            source_root, Path(spec["evaluation_configs"][variant])
        )
        config = _load_json(config_path)
        if int(config["holdout_seed"]) == FORMAL_HOLDOUT_SEED:
            raise ValueError("Phase config reused the formal holdout")
        summary_path = _resolve(
            artifact_root, Path(config["output_root"]) / "summary.json"
        )
        summary = _load_json(summary_path)
        runs = {}
        for run in summary["runs"]:
            key = (str(run["algorithm"]), int(run["training_seed"]))
            if key in runs:
                raise ValueError(f"Duplicate phase run {variant} {key}")
            run_root = summary_path.parent / key[0] / f"seed{key[1]}"
            runs[key] = {
                "checkpoint": str(run["checkpoint"]),
                "rows": _read_csv(run_root / "holdout_rows.csv"),
            }
        result[variant] = runs
    return result


def _manifest_runs(
    manifest: dict[str, Any],
    algorithms: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for run in manifest["runs"]:
        key = (str(run["algorithm"]), int(run["seed"]))
        if key in result:
            raise ValueError(f"Duplicate training run {key}")
        result[key] = dict(run)
    expected = {(algorithm, seed) for algorithm in algorithms for seed in seeds}
    if result.keys() != expected:
        raise ValueError("Training manifest run identities do not match Stage B")
    return result


def _teacher_targets(
    teacher: dict[str, Any],
    *,
    reward_scale: float,
    target_scale: float,
) -> np.ndarray:
    advantages = np.asarray(teacher["option_advantages"], dtype=np.float32)
    feasible = np.asarray(teacher["option_feasible"], dtype=bool)
    best = np.max(np.where(feasible, advantages, -np.inf), axis=1)
    return (
        np.maximum(best, 0.0) * float(reward_scale) * float(target_scale)
    ).astype(np.float32)


def _base_actions(agent: Any, states: np.ndarray, *, batch_size: int) -> np.ndarray:
    result = []
    for start in range(0, states.shape[0], batch_size):
        tensor = torch.as_tensor(
            states[start : start + batch_size],
            dtype=torch.float32,
            device=agent.device,
        )
        result.append(
            agent._base_actions_from_states_tensor(tensor).detach().cpu().numpy()
        )
    return np.concatenate(result, axis=0)


def _critic_predictions(
    agent: Any,
    states: np.ndarray,
    actions: np.ndarray,
    base_actions: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    agent.critic.eval()
    with torch.no_grad():
        for start in range(0, states.shape[0], batch_size):
            raw_states = torch.as_tensor(
                states[start : start + batch_size],
                dtype=torch.float32,
                device=agent.device,
            )
            action_tensor = torch.as_tensor(
                actions[start : start + batch_size],
                dtype=torch.float32,
                device=agent.device,
            )
            anchor_tensor = torch.as_tensor(
                base_actions[start : start + batch_size],
                dtype=torch.float32,
                device=agent.device,
            )
            if hasattr(agent, "graph_spec"):
                critic_states = flat_state_to_node_features(
                    raw_states, agent.graph_spec
                )
            else:
                critic_states = (
                    raw_states
                    if agent.temporal_demand_encoder_enabled
                    else agent.observation_scaler.normalize_tensor(raw_states)
                )
            predicted = agent.critic(
                critic_states,
                agent._critic_actions_tensor(action_tensor),
            ) - agent.critic(
                critic_states,
                agent._critic_actions_tensor(anchor_tensor),
            )
            predictions.append(predicted.detach().cpu().numpy().reshape(-1))
    return np.concatenate(predictions)


def _collect_pair(
    collector: dict[str, dict[str, list[np.ndarray]]],
    label: str,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> None:
    collector[label]["targets"].append(np.asarray(targets))
    collector[label]["predictions"].append(np.asarray(predictions))


def _combined_rows(
    runs: dict[tuple[str, int], dict[str, Any]], algorithm: str
) -> list[dict[str, Any]]:
    return [
        row
        for (name, _seed), run in sorted(runs.items())
        if name == algorithm
        for row in run["rows"]
    ]


def _rows_by_key(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(str(row[field]) for field in PAIRING_KEYS)
        if key in result:
            raise ValueError(f"Duplicate paired row {key}")
        result[key] = row
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_array.size < 2:
        return None
    if float(left_array.std()) <= 1e-15 or float(right_array.std()) <= 1e-15:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


if __name__ == "__main__":
    main()
