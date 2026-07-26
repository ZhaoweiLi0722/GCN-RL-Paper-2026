"""Grouped cross-validation for the AFR graph correction gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    make_evaluation_config,
    resolve_budget,
    select_scenarios,
)
from evaluation.run_gcn_residual_sweep import load_local_search_demonstrations
from src.models.graph_features import flat_state_to_node_features
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env
from src.rl.networks import torch


DEFAULT_PLAN = "experiments/configs/residual_policy_benchmark.json"
DEFAULT_ALGORITHM = "gcn_residual_mdl2_network_ddpg_afd"
DEFAULT_SCENARIO = "patient_condition_geo_demand_drift"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--budget", default="diagnostic_pretrain")
    parser.add_argument("--algorithm", default=DEFAULT_ALGORITHM)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--base-policy", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument(
        "--demand-history-window",
        type=int,
        default=None,
        help=(
            "Override the scenario history window and verify it against "
            "teacher-cache metadata."
        ),
    )
    parser.add_argument("--trajectory-length", type=int, default=52)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--positive-weight-mass", type=float, default=0.25)
    parser.add_argument("--gcn-hidden-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--gate-hidden-sizes", nargs="+", type=int, default=None)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=(0.5, 0.7, 0.8, 0.9, 0.95),
    )
    parser.add_argument(
        "--output",
        default="results/correction_gate_cross_validation/summary.json",
    )
    args = parser.parse_args()

    demos = load_local_search_demonstrations(args.teacher_cache)
    states = np.asarray(demos["states"], dtype=np.float32)
    actions = np.asarray(demos["actions"], dtype=np.float32)
    if states.shape[0] % int(args.trajectory_length) != 0:
        raise ValueError("Teacher rows must divide into complete trajectories")
    trajectory_count = states.shape[0] // int(args.trajectory_length)
    if int(args.folds) > trajectory_count:
        raise ValueError("folds cannot exceed the number of trajectories")

    plan = load_benchmark_plan(args.plan)
    budget = resolve_budget(plan, args.budget)
    scenario = select_scenarios(plan, (args.scenario,))[0]
    config = make_evaluation_config(
        plan,
        args.budget,
        budget,
        args.algorithm,
        scenario,
        args.seed,
    )
    if args.demand_history_window is not None:
        config.setdefault("env", {})["demand_history_window"] = int(
            args.demand_history_window
        )
    if args.gcn_hidden_sizes:
        config["gcn_hidden_sizes"] = [
            int(value) for value in args.gcn_hidden_sizes
        ]
    if args.gate_hidden_sizes:
        config.setdefault("residual_action", {}).setdefault(
            "correction_gate",
            {},
        )["hidden_sizes"] = [
            int(value) for value in args.gate_hidden_sizes
        ]
    if args.base_policy:
        config.setdefault("residual_action", {})["base_policy"] = str(
            args.base_policy
        )
    cache_history_window = demos.get("demand_history_window")
    environment_history_window = int(
        config.get("env", {}).get("demand_history_window", 4)
    )
    if (
        cache_history_window is not None
        and environment_history_window != int(cache_history_window)
    ):
        raise ValueError(
            "Teacher cache demand_history_window="
            f"{cache_history_window} does not match environment "
            f"demand_history_window={environment_history_window}"
        )
    env = build_env(config, seed=args.seed)
    if states.shape[1] != env.observation_size:
        raise ValueError(
            f"Teacher state width {states.shape[1]} does not match "
            f"environment width {env.observation_size}"
        )
    prototype = get_agent_class(args.algorithm)(
        env.observation_size,
        env.action_size,
        config,
    )
    labels = correction_gate_labels(
        prototype,
        states,
        actions,
    )
    group_names = (
        tuple(prototype.correction_gate_groups)
        if labels.ndim > 1
        else ("all_actions",)
    )

    trajectory_ids = np.repeat(
        np.arange(trajectory_count),
        int(args.trajectory_length),
    )
    fold_ids = np.arange(trajectory_count) % int(args.folds)
    probabilities = np.full(labels.shape, np.nan, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(int(args.folds)):
        test_trajectories = np.where(fold_ids == fold)[0]
        test_mask = np.isin(trajectory_ids, test_trajectories)
        train_mask = ~test_mask
        agent = get_agent_class(args.algorithm)(
            env.observation_size,
            env.action_size,
            {**config, "seed": int(args.seed) + fold},
        )
        sample_labels = (
            labels[train_mask].max(axis=1)
            if labels.ndim > 1
            else labels[train_mask]
        )
        weights = binary_class_weights(
            sample_labels,
            positive_mass=float(args.positive_weight_mass),
        )
        fit = agent.fit_correction_gate_batch(
            states[train_mask],
            labels[train_mask],
            {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed) + 700_000 + fold,
                "mode": "classification",
            },
            weights=weights,
        )
        test_tensor = torch.as_tensor(
            states[test_mask],
            dtype=torch.float32,
            device=agent.device,
        )
        with torch.no_grad():
            node_features = flat_state_to_node_features(
                test_tensor,
                agent.graph_spec,
            )
            fold_probabilities = torch.sigmoid(
                agent.correction_gate(node_features)
            ).cpu().numpy()
        probabilities[test_mask] = fold_probabilities
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "test_positive_rate": _rates(labels[test_mask]),
                "fit": fit,
            }
        )

    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Cross-validation did not produce every prediction")
    group_summaries = {}
    for index, group in enumerate(group_names):
        group_labels = labels if labels.ndim == 1 else labels[:, index]
        group_probabilities = (
            probabilities
            if probabilities.ndim == 1
            else probabilities[:, index]
        )
        group_summaries[group] = {
            "positive_rate": float(group_labels.mean()),
            "roc_auc": binary_roc_auc(group_labels, group_probabilities),
            "thresholds": [
                threshold_metrics(
                    group_labels,
                    group_probabilities,
                    float(threshold),
                )
                for threshold in args.thresholds
            ],
        }
    summary = {
        "teacher_cache": str(args.teacher_cache),
        "rows": int(states.shape[0]),
        "trajectories": trajectory_count,
        "folds": int(args.folds),
        "positive_rate": _rates(labels),
        "positive_weight_mass": float(args.positive_weight_mass),
        "base_policy": str(
            config.get("residual_action", {}).get("base_policy", "")
        ),
        "demand_history_window": environment_history_window,
        "gcn_hidden_sizes": list(config.get("gcn_hidden_sizes", ())),
        "gate_hidden_sizes": list(
            config.get("residual_action", {})
            .get("correction_gate", {})
            .get("hidden_sizes", ())
        ),
        "group_summaries": group_summaries,
        "fold_summaries": fold_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def binary_class_weights(
    labels: np.ndarray,
    *,
    positive_mass: float,
) -> np.ndarray:
    label_array = np.asarray(labels, dtype=np.float32).reshape(-1)
    if not 0.0 < positive_mass < 1.0:
        raise ValueError("positive_mass must lie strictly between zero and one")
    positive = label_array > 0.5
    if not np.any(positive) or not np.any(~positive):
        return np.ones_like(label_array)
    weights = np.empty_like(label_array)
    weights[positive] = positive_mass / float(positive.sum())
    weights[~positive] = (1.0 - positive_mass) / float((~positive).sum())
    return weights * float(label_array.size)


def correction_gate_labels(
    agent,
    states: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
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
    with torch.no_grad():
        residual_targets = agent._residual_targets_tensor(
            state_tensor,
            action_tensor,
        )
        residual_mask = agent._residual_loss_mask(
            action_tensor,
            state_tensor,
        )
        if residual_mask.shape[0] == 1:
            residual_mask = residual_mask.expand(states.shape[0], -1)
        labels = agent._correction_gate_labels_tensor(
            residual_targets,
            residual_mask,
        )
    return labels.cpu().numpy().astype(np.float32)


def _rates(values: np.ndarray) -> float | list[float]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return float(array.mean())
    return [float(array[:, index].mean()) for index in range(array.shape[1])]


def threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.float32).reshape(-1) > 0.5
    predicted = np.asarray(probabilities, dtype=np.float32).reshape(-1) >= threshold
    true_positive = int(np.sum(truth & predicted))
    false_positive = int(np.sum(~truth & predicted))
    false_negative = int(np.sum(truth & ~predicted))
    true_negative = int(np.sum(~truth & ~predicted))
    return {
        "threshold": float(threshold),
        "prediction_rate": float(predicted.mean()),
        "precision": true_positive / max(true_positive + false_positive, 1),
        "recall": true_positive / max(true_positive + false_negative, 1),
        "specificity": true_negative / max(true_negative + false_positive, 1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.float32).reshape(-1) > 0.5
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive_count = int(truth.sum())
    negative_count = int((~truth).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    ranks = np.empty(score_array.size, dtype=np.float64)
    start = 0
    while start < score_array.size:
        stop = start + 1
        while stop < score_array.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_rank_sum = float(ranks[truth].sum())
    return (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / float(positive_count * negative_count)


if __name__ == "__main__":
    main()
