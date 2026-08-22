"""Cross-fitted audit of DDPG critic ranking on frozen teacher trajectories."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluation.run_gcn_residual_sweep import (
    load_local_search_demonstrations,
)
from src.models.graph_features import flat_state_to_node_features
from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.critic_advantage import teacher_advantage_support_mask
from src.rl.experiment import build_env
from src.rl.networks import torch
from src.rl.training_state import load_off_policy_training_state


SAMPLE_ALIGNED_KEYS = frozenset(
    {
        "states",
        "actions",
        "weights",
        "improved_mask",
        "scenario_ids",
        "trajectory_ids",
        "trajectory_steps",
        "transition_states",
        "transition_actions",
        "transition_rewards",
        "transition_next_states",
        "transition_dones",
        "option_advantages",
        "option_feasible",
        "edge_groups",
        "edge_sources",
        "edge_targets",
        "edge_flows",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subset_demonstrations(
    demonstrations: dict[str, Any],
    mask: np.ndarray,
) -> dict[str, Any]:
    row_mask = np.asarray(mask, dtype=bool)
    sample_count = int(np.asarray(demonstrations["states"]).shape[0])
    if row_mask.shape != (sample_count,):
        raise ValueError("Demonstration subset mask does not match sample count")
    if not np.any(row_mask):
        raise ValueError("Demonstration subset cannot be empty")
    result: dict[str, Any] = {}
    for key, value in demonstrations.items():
        if key in SAMPLE_ALIGNED_KEYS:
            array = np.asarray(value)
            if array.shape[0] != sample_count:
                raise ValueError(f"Demonstration field {key} is not row aligned")
            result[key] = array[row_mask].copy()
        else:
            result[key] = copy.deepcopy(value)
    return result


def policy_supported_demonstrations(
    demonstrations: dict[str, Any],
    allowed_option_groups: Iterable[str],
) -> dict[str, Any]:
    allowed = tuple(str(group) for group in allowed_option_groups)
    if not allowed:
        return demonstrations
    mask = teacher_advantage_support_mask(
        demonstrations["option_advantages"],
        demonstrations["option_feasible"],
        demonstrations["option_groups"],
        allowed_option_groups=allowed,
    )
    return subset_demonstrations(demonstrations, mask)


def trajectory_group_folds(
    scenario_ids: np.ndarray,
    trajectory_ids: np.ndarray,
    *,
    fold_count: int = 2,
) -> list[dict[str, Any]]:
    scenarios = np.asarray(scenario_ids, dtype=np.int64)
    trajectories = np.asarray(trajectory_ids, dtype=np.int64)
    if scenarios.shape != trajectories.shape or scenarios.ndim != 1:
        raise ValueError("Scenario and trajectory ids must be aligned vectors")
    if scenarios.size == 0:
        raise ValueError("Trajectory folds require at least one sample")
    if int(fold_count) < 2:
        raise ValueError("Cross-fitting requires at least two folds")

    groups_by_scenario: dict[int, tuple[int, ...]] = {}
    for scenario in sorted(int(value) for value in np.unique(scenarios)):
        groups = tuple(
            sorted(
                int(value)
                for value in np.unique(trajectories[scenarios == scenario])
            )
        )
        if len(groups) < int(fold_count):
            raise ValueError(
                "Each scenario requires at least one trajectory per fold"
            )
        groups_by_scenario[scenario] = groups

    folds = []
    heldout_counts = np.zeros(scenarios.shape[0], dtype=np.int64)
    for fold_index in range(int(fold_count)):
        holdout_mask = np.zeros(scenarios.shape[0], dtype=bool)
        heldout_groups: dict[str, list[int]] = {}
        for scenario, groups in groups_by_scenario.items():
            selected = tuple(
                trajectory
                for index, trajectory in enumerate(groups)
                if index % int(fold_count) == fold_index
            )
            if not selected:
                raise ValueError("Every fold requires a trajectory per scenario")
            scenario_mask = scenarios == scenario
            group_mask = np.isin(trajectories, selected)
            holdout_mask |= scenario_mask & group_mask
            heldout_groups[str(scenario)] = list(selected)
        train_mask = ~holdout_mask
        if not np.any(train_mask) or not np.any(holdout_mask):
            raise ValueError("Every cross-fit fold requires train and holdout rows")
        heldout_counts += holdout_mask.astype(np.int64)
        folds.append(
            {
                "index": fold_index,
                "train_mask": train_mask,
                "holdout_mask": holdout_mask,
                "heldout_trajectory_ids_by_scenario": heldout_groups,
            }
        )
    if not np.all(heldout_counts == 1):
        raise ValueError(
            "The requested fold count does not assign every trajectory once"
        )
    return folds


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


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_array.size < 2:
        return None
    if float(left_array.std()) <= 1e-15 or float(right_array.std()) <= 1e-15:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def critic_advantage_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if target.shape != predicted.shape or target.size == 0:
        raise ValueError("Critic targets and predictions must be aligned and non-empty")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Critic targets and predictions must be finite")

    target_positive = target > 0.0
    predicted_positive = predicted > 0.0
    pair_left, pair_right = np.triu_indices(target.size, k=1)
    target_differences = target[pair_left] - target[pair_right]
    comparable = np.abs(target_differences) > 1e-15
    prediction_differences = predicted[pair_left] - predicted[pair_right]
    pair_products = target_differences[comparable] * prediction_differences[comparable]
    pair_scores = np.where(pair_products > 0.0, 1.0, np.where(pair_products == 0.0, 0.5, 0.0))

    true_positive = int(np.sum(target_positive & predicted_positive))
    true_negative = int(np.sum(~target_positive & ~predicted_positive))
    positive_count = int(np.sum(target_positive))
    zero_count = int(np.sum(~target_positive))
    return {
        "samples": int(target.size),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "target_positive_fraction": float(target_positive.mean()),
        "prediction_mean": float(predicted.mean()),
        "prediction_std": float(predicted.std()),
        "prediction_positive_fraction": float(predicted_positive.mean()),
        "mse": float(np.mean((predicted - target) ** 2)),
        "mae": float(np.mean(np.abs(predicted - target))),
        "pearson": _correlation(target, predicted),
        "spearman": _correlation(
            _average_ranks(target),
            _average_ranks(predicted),
        ),
        "pairwise_comparable_pairs": int(np.sum(comparable)),
        "pairwise_order_accuracy": (
            float(pair_scores.mean()) if pair_scores.size else None
        ),
        "positive_sign_accuracy": float(
            np.mean(target_positive == predicted_positive)
        ),
        "positive_recall": (
            float(true_positive / positive_count) if positive_count else None
        ),
        "zero_specificity": (
            float(true_negative / zero_count) if zero_count else None
        ),
    }


def exact_row_overlap(left: np.ndarray, right: np.ndarray) -> int:
    left_rows = {
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()
        for row in np.asarray(left)
    }
    right_rows = {
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()
        for row in np.asarray(right)
    }
    return len(left_rows & right_rows)


def _teacher_advantage_arrays(
    agent: Any,
    demonstrations: dict[str, Any],
    *,
    target_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(demonstrations["states"], dtype=np.float32)
    actions = np.asarray(demonstrations["actions"], dtype=np.float32)
    advantages = np.asarray(
        demonstrations["option_advantages"],
        dtype=np.float32,
    )
    feasible = np.asarray(demonstrations["option_feasible"], dtype=bool)
    best_advantages = np.max(
        np.where(feasible, advantages, -np.inf),
        axis=1,
    )
    targets = (
        np.maximum(best_advantages, 0.0)
        * float(agent.reward_scale)
        * float(target_scale)
    ).astype(np.float32)

    state_tensor = torch.as_tensor(
        states,
        dtype=torch.float32,
        device=agent.device,
    )
    action_tensor = torch.as_tensor(
        actions,
        dtype=torch.float32,
        device=agent.device,
    )
    agent.critic.eval()
    with torch.no_grad():
        node_features = flat_state_to_node_features(
            state_tensor,
            agent.graph_spec,
        )
        anchor_actions = agent._base_actions_from_states_tensor(state_tensor)
        predictions = (
            agent.critic(
                node_features,
                agent._critic_actions_tensor(action_tensor),
            )
            - agent.critic(
                node_features,
                agent._critic_actions_tensor(anchor_actions),
            )
        )
    return targets, predictions.detach().cpu().numpy().reshape(-1)


def _metrics_with_scenarios(
    targets: np.ndarray,
    predictions: np.ndarray,
    scenario_ids: np.ndarray,
    scenario_names: Iterable[str],
) -> dict[str, Any]:
    scenarios = np.asarray(scenario_ids, dtype=np.int64)
    names = tuple(str(value) for value in scenario_names)
    by_scenario = {}
    for scenario in sorted(int(value) for value in np.unique(scenarios)):
        name = names[scenario] if scenario < len(names) else str(scenario)
        mask = scenarios == scenario
        by_scenario[name] = critic_advantage_metrics(
            np.asarray(targets)[mask],
            np.asarray(predictions)[mask],
        )
    return {
        "pooled": critic_advantage_metrics(targets, predictions),
        "by_scenario": by_scenario,
    }


def _build_source_agent(
    runtime_config: dict[str, Any],
    checkpoint_contract: dict[str, Any],
    training_state: Path,
) -> tuple[Any, dict[str, Any]]:
    config = copy.deepcopy(runtime_config)
    config["device"] = "cpu"
    seed = int(config["seed"])
    env = build_env(config, seed=seed)
    algorithm = str(config["algorithm"])
    agent = get_agent_class(algorithm)(
        env.observation_size,
        env.action_size,
        config,
    )
    metadata = load_off_policy_training_state(
        agent,
        training_state,
        config=checkpoint_contract,
        restore_environment=False,
    )
    if int(metadata.get("next_episode", -1)) != 0:
        raise ValueError("Critic cross-fit requires an episode-0 training state")
    return agent, metadata


def run_crossfit_audit(
    *,
    runtime_config_path: Path,
    training_state_path: Path,
    cache_path: Path,
    positive_margin: float | None = None,
    positive_margin_weight: float | None = None,
    pairwise_difference_weight: float | None = None,
    calibration_option_groups: tuple[str, ...] = (),
    evaluation_option_groups: tuple[str, ...] = (),
) -> dict[str, Any]:
    input_paths = (runtime_config_path, training_state_path, cache_path)
    input_hashes_before = {str(path): sha256_file(path) for path in input_paths}
    runtime_config = load_config(runtime_config_path)
    runtime_config["device"] = "cpu"
    calibration = dict(
        runtime_config.get("critic_teacher_advantage_calibration", {})
    )
    if positive_margin is not None:
        calibration["positive_margin"] = float(positive_margin)
    if positive_margin_weight is not None:
        calibration["positive_margin_weight"] = float(
            positive_margin_weight
        )
    if pairwise_difference_weight is not None:
        calibration["pairwise_difference_weight"] = float(
            pairwise_difference_weight
        )
    if calibration_option_groups:
        calibration["allowed_option_groups"] = list(
            calibration_option_groups
        )
    runtime_config["critic_teacher_advantage_calibration"] = calibration
    if not bool(calibration.get("enabled", False)):
        raise ValueError("Runtime config must enable critic advantage calibration")
    if int(calibration.get("updates", 0)) <= 0:
        raise ValueError("Critic advantage calibration requires positive updates")

    checkpoint = torch.load(
        training_state_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_contract = dict(checkpoint["training_contract"])
    if str(checkpoint["algorithm"]) != str(runtime_config["algorithm"]):
        raise ValueError("Runtime config and training state algorithms differ")
    if int(checkpoint["seed"]) != int(runtime_config["seed"]):
        raise ValueError("Runtime config and training state seeds differ")

    demonstrations = load_local_search_demonstrations(cache_path)
    required = (
        "scenario_ids",
        "trajectory_ids",
        "trajectory_steps",
        "scenario_names",
        "option_advantages",
        "option_feasible",
    )
    missing = [key for key in required if key not in demonstrations]
    if missing:
        raise ValueError(
            "Critic cross-fit cache is missing: " + ", ".join(missing)
        )
    folds = trajectory_group_folds(
        demonstrations["scenario_ids"],
        demonstrations["trajectory_ids"],
        fold_count=2,
    )

    fold_results = []
    pooled_targets = []
    pooled_before = []
    pooled_after = []
    pooled_scenarios = []
    for fold in folds:
        train = subset_demonstrations(
            demonstrations,
            fold["train_mask"],
        )
        holdout = subset_demonstrations(
            demonstrations,
            fold["holdout_mask"],
        )
        train_evaluation = policy_supported_demonstrations(
            train,
            evaluation_option_groups,
        )
        holdout_evaluation = policy_supported_demonstrations(
            holdout,
            evaluation_option_groups,
        )
        agent, metadata = _build_source_agent(
            runtime_config,
            checkpoint_contract,
            training_state_path,
        )
        target_scale = float(calibration.get("target_scale", 1.0))
        train_targets, train_before = _teacher_advantage_arrays(
            agent,
            train_evaluation,
            target_scale=target_scale,
        )
        holdout_targets, holdout_before = _teacher_advantage_arrays(
            agent,
            holdout_evaluation,
            target_scale=target_scale,
        )
        configured = agent.configure_critic_teacher_advantage_calibration(train)
        calibration_summary = agent._calibrate_critic_teacher_advantage()
        _, train_after = _teacher_advantage_arrays(
            agent,
            train_evaluation,
            target_scale=target_scale,
        )
        _, holdout_after = _teacher_advantage_arrays(
            agent,
            holdout_evaluation,
            target_scale=target_scale,
        )
        scenario_names = demonstrations["scenario_names"]
        fold_results.append(
            {
                "fold": int(fold["index"]),
                "heldout_trajectory_ids_by_scenario": fold[
                    "heldout_trajectory_ids_by_scenario"
                ],
                "train_samples": int(train_targets.size),
                "holdout_samples": int(holdout_targets.size),
                "calibration_source_samples": int(train["states"].shape[0]),
                "exact_state_overlap": exact_row_overlap(
                    train_evaluation["states"],
                    holdout_evaluation["states"],
                ),
                "exact_action_overlap": exact_row_overlap(
                    train_evaluation["actions"],
                    holdout_evaluation["actions"],
                ),
                "source_next_episode": int(metadata["next_episode"]),
                "configured": configured,
                "calibration": calibration_summary,
                "train_before": _metrics_with_scenarios(
                    train_targets,
                    train_before,
                    train_evaluation["scenario_ids"],
                    scenario_names,
                ),
                "train_after": _metrics_with_scenarios(
                    train_targets,
                    train_after,
                    train_evaluation["scenario_ids"],
                    scenario_names,
                ),
                "holdout_before": _metrics_with_scenarios(
                    holdout_targets,
                    holdout_before,
                    holdout_evaluation["scenario_ids"],
                    scenario_names,
                ),
                "holdout_after": _metrics_with_scenarios(
                    holdout_targets,
                    holdout_after,
                    holdout_evaluation["scenario_ids"],
                    scenario_names,
                ),
            }
        )
        pooled_targets.append(holdout_targets)
        pooled_before.append(holdout_before)
        pooled_after.append(holdout_after)
        pooled_scenarios.append(
            np.asarray(holdout_evaluation["scenario_ids"])
        )

    targets = np.concatenate(pooled_targets)
    before = np.concatenate(pooled_before)
    after = np.concatenate(pooled_after)
    scenarios = np.concatenate(pooled_scenarios)
    pooled_before_metrics = _metrics_with_scenarios(
        targets,
        before,
        scenarios,
        demonstrations["scenario_names"],
    )
    pooled_after_metrics = _metrics_with_scenarios(
        targets,
        after,
        scenarios,
        demonstrations["scenario_names"],
    )
    before_pooled = pooled_before_metrics["pooled"]
    after_pooled = pooled_after_metrics["pooled"]
    input_hashes_after = {str(path): sha256_file(path) for path in input_paths}
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("Critic audit modified a locked input")
    return {
        "diagnostic_only": True,
        "actor_updates": 0,
        "online_environment_steps": 0,
        "algorithm": str(runtime_config["algorithm"]),
        "seed": int(runtime_config["seed"]),
        "runtime_config": str(runtime_config_path),
        "training_state": str(training_state_path),
        "teacher_cache": str(cache_path),
        "input_sha256": input_hashes_before,
        "fold_count": len(folds),
        "samples": int(targets.size),
        "trajectory_groups": int(
            np.unique(
                np.stack(
                    (
                        demonstrations["scenario_ids"],
                        demonstrations["trajectory_ids"],
                    ),
                    axis=1,
                ),
                axis=0,
            ).shape[0]
        ),
        "calibration_settings": {
            "updates": int(calibration["updates"]),
            "target_scale": float(calibration.get("target_scale", 1.0)),
            "ranking_weight": float(calibration.get("ranking_weight", 1.0)),
            "online_ranking_weight": float(
                calibration.get("online_ranking_weight", 0.0)
            ),
            "positive_margin": float(
                calibration.get("positive_margin", 0.0)
            ),
            "positive_margin_weight": float(
                calibration.get("positive_margin_weight", 0.0)
            ),
            "pairwise_difference_weight": float(
                calibration.get("pairwise_difference_weight", 0.0)
            ),
            "calibration_option_groups": list(calibration_option_groups),
            "evaluation_option_groups": list(evaluation_option_groups),
        },
        "folds": fold_results,
        "heldout_before": pooled_before_metrics,
        "heldout_after": pooled_after_metrics,
        "decision": {
            "heldout_mse_improved": bool(
                float(after_pooled["mse"]) < float(before_pooled["mse"])
            ),
            "heldout_spearman_improved": bool(
                after_pooled["spearman"] is not None
                and before_pooled["spearman"] is not None
                and float(after_pooled["spearman"])
                > float(before_pooled["spearman"])
            ),
            "heldout_pairwise_order_improved": bool(
                after_pooled["pairwise_order_accuracy"] is not None
                and before_pooled["pairwise_order_accuracy"] is not None
                and float(after_pooled["pairwise_order_accuracy"])
                > float(before_pooled["pairwise_order_accuracy"])
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--training-state", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive-margin", type=float, default=None)
    parser.add_argument(
        "--positive-margin-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--pairwise-difference-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--calibration-option-group",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--evaluation-option-group",
        action="append",
        default=[],
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = run_crossfit_audit(
        runtime_config_path=Path(args.runtime_config),
        training_state_path=Path(args.training_state),
        cache_path=Path(args.cache),
        positive_margin=args.positive_margin,
        positive_margin_weight=args.positive_margin_weight,
        pairwise_difference_weight=args.pairwise_difference_weight,
        calibration_option_groups=tuple(args.calibration_option_group),
        evaluation_option_groups=tuple(args.evaluation_option_group),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote critic advantage ranking audit to {output}", flush=True)


if __name__ == "__main__":
    main()
