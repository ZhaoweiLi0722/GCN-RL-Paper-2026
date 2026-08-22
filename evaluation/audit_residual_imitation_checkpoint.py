"""Audit residual-policy checkpoint fit on a frozen teacher cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.run_gcn_residual_sweep import (
    load_local_search_demonstrations,
)
from src.models.graph_features import flat_state_to_node_features
from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env
from src.rl.networks import torch


GROUPS = (
    "specimen_transfer",
    "reagent_transfer",
    "capacity_transfer",
    "replenishment",
)
GATE_THRESHOLD_GRID = tuple(value / 10.0 for value in range(1, 10))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--cache", default=None)
    parser.add_argument(
        "--checkpoint-variant",
        choices=("pretrain", "final"),
        required=True,
    )
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--activity-threshold", type=float, default=0.04)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = audit_residual_imitation_checkpoint(
        training_manifest=Path(args.training_manifest),
        cache_path=None if args.cache is None else Path(args.cache),
        checkpoint_variant=str(args.checkpoint_variant),
        algorithm=args.algorithm,
        seed=int(args.seed),
        activity_threshold=float(args.activity_threshold),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["fit"], indent=2, sort_keys=True))
    print(f"wrote residual imitation audit to {output}", flush=True)


def audit_residual_imitation_checkpoint(
    *,
    training_manifest: Path,
    cache_path: Path | None,
    checkpoint_variant: str,
    algorithm: str | None,
    seed: int,
    activity_threshold: float,
) -> dict[str, Any]:
    manifest = json.loads(training_manifest.read_text(encoding="utf-8"))
    matching = [
        run
        for run in manifest["runs"]
        if int(run["seed"]) == int(seed)
        and (algorithm is None or str(run["algorithm"]) == str(algorithm))
    ]
    if len(matching) != 1:
        raise ValueError(
            "Expected exactly one manifest run for the requested algorithm/seed"
        )
    run = matching[0]
    checkpoint_key = (
        "pretrain_checkpoint"
        if checkpoint_variant == "pretrain"
        else "checkpoint"
    )
    checkpoint = Path(run[checkpoint_key])
    config_path = Path(run["config"])
    resolved_cache = Path(cache_path or manifest["teacher_cache"])
    for path in (checkpoint, config_path, resolved_cache):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = load_config(config_path)
    config["device"] = "cpu"
    env = build_env(config, seed=int(seed))
    agent = get_agent_class(str(run["algorithm"]))(
        env.observation_size,
        env.action_size,
        config,
    )
    agent.load_actor(checkpoint)

    demonstrations = load_local_search_demonstrations(resolved_cache)
    states = np.asarray(demonstrations["states"], dtype=np.float32)
    actions = np.asarray(demonstrations["actions"], dtype=np.float32)
    weights = np.asarray(demonstrations["weights"], dtype=np.float32)
    if states.shape != (actions.shape[0], env.observation_size):
        raise ValueError("Teacher states do not match the checkpoint environment")
    if actions.shape[1] != env.action_size:
        raise ValueError("Teacher actions do not match the checkpoint environment")

    state_tensor = torch.as_tensor(states, dtype=torch.float32, device=agent.device)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        node_features = flat_state_to_node_features(
            state_tensor,
            agent.graph_spec,
        )
        network_actions = agent.actor(node_features)
        predictions = agent._policy_residuals_tensor(
            state_tensor,
            network_actions,
            apply_correction_gate=False,
        )
        targets = agent._residual_targets_tensor(
            state_tensor,
            action_tensor,
        )
        base_actions = agent._base_actions_from_states_tensor(state_tensor)
        scale = torch.as_tensor(
            agent.residual_scale_vector,
            dtype=torch.float32,
            device=agent.device,
        ).reshape(1, -1)
        composed_actions = torch.clamp(
            base_actions + scale * predictions,
            -1.0,
            1.0,
        )
    predicted = predictions.cpu().numpy()
    target = targets.cpu().numpy()
    composed = composed_actions.cpu().numpy()

    fit = residual_fit_summary(
        target,
        predicted,
        num_facilities=int(agent.graph_spec.num_facilities),
        activity_threshold=activity_threshold,
    )
    fit["composed_action_mse"] = float(np.mean((composed - actions) ** 2))
    fit["unweighted_checkpoint_loss"] = agent.evaluate_action_batch(
        states,
        actions,
        target_mode="residual",
        demonstrations=demonstrations,
    )
    fit["weighted_checkpoint_loss"] = agent.evaluate_action_batch(
        states,
        actions,
        weights=weights,
        target_mode="residual",
        demonstrations=demonstrations,
    )
    if (
        getattr(agent, "correction_gate", None) is not None
        and getattr(agent, "correction_gate_mode", "") == "classification"
    ):
        with torch.no_grad():
            residual_mask = agent._residual_loss_mask(
                action_tensor,
                state_tensor,
            )
            residual_mask = agent._supervised_residual_dimension_mask(
                residual_mask,
                targets,
            )
            if residual_mask.shape[0] == 1:
                residual_mask = residual_mask.expand(states.shape[0], -1)
            gate_labels = agent._correction_gate_labels_tensor(
                targets,
                residual_mask,
            )
            gate_features = agent.correction_gate_node_features(
                state_tensor,
                residuals=predictions,
            )
            gate_probabilities = torch.sigmoid(
                agent.correction_gate(gate_features)
            )
        fit["correction_gate_calibration"] = gate_calibration_summary(
            gate_labels.cpu().numpy(),
            gate_probabilities.cpu().numpy(),
            groups=tuple(agent.correction_gate_groups),
            thresholds=GATE_THRESHOLD_GRID,
        )
    return {
        "training_manifest": str(training_manifest),
        "training_manifest_sha256": sha256_file(training_manifest),
        "algorithm": str(run["algorithm"]),
        "seed": int(seed),
        "checkpoint_variant": checkpoint_variant,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "cache": str(resolved_cache),
        "cache_sha256": sha256_file(resolved_cache),
        "samples": int(states.shape[0]),
        "activity_threshold": float(activity_threshold),
        "fit": fit,
    }


def gate_calibration_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    groups: tuple[str, ...],
    thresholds: tuple[float, ...] = GATE_THRESHOLD_GRID,
) -> dict[str, Any]:
    label_array = np.asarray(labels, dtype=bool)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if label_array.shape != probability_array.shape or label_array.ndim != 2:
        raise ValueError("Gate labels and probabilities require equal rank-2 shapes")
    if label_array.shape[1] != len(groups):
        raise ValueError("Gate groups must align with the probability columns")
    threshold_values = tuple(float(value) for value in thresholds)
    if not threshold_values or any(
        not 0.0 <= value <= 1.0 for value in threshold_values
    ):
        raise ValueError("Gate calibration thresholds must lie in [0, 1]")
    if not np.all(np.isfinite(probability_array)) or np.any(
        (probability_array < 0.0) | (probability_array > 1.0)
    ):
        raise ValueError("Gate probabilities must be finite and lie in [0, 1]")

    by_group = {}
    for index, group in enumerate(groups):
        group_labels = label_array[:, index]
        group_probabilities = probability_array[:, index]
        sweep = [
            gate_binary_metrics(
                group_labels,
                group_probabilities,
                threshold=threshold,
            )
            for threshold in threshold_values
        ]
        best_f1 = min(
            sweep,
            key=lambda item: (-float(item["f1"]), -float(item["threshold"])),
        )
        prevalence_threshold = prevalence_matched_gate_threshold(
            group_labels,
            group_probabilities,
        )
        by_group[str(group)] = {
            "label_rate": float(np.mean(group_labels)),
            "probability_quantiles": {
                f"q{int(quantile * 100):02d}": float(
                    np.quantile(group_probabilities, quantile)
                )
                for quantile in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
            "prevalence_matched": gate_binary_metrics(
                group_labels,
                group_probabilities,
                threshold=prevalence_threshold,
            ),
            "best_grid_f1": best_f1,
            "threshold_sweep": sweep,
        }
    return {
        "samples": int(label_array.shape[0]),
        "threshold_grid": list(threshold_values),
        "by_group": by_group,
    }


def prevalence_matched_gate_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    label_array = np.asarray(labels, dtype=bool).reshape(-1)
    probability_array = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    candidates = np.unique(
        np.concatenate(
            (
                np.asarray([0.0, 1.0], dtype=np.float64),
                probability_array,
            )
        )
    )
    label_rate = float(np.mean(label_array))
    return float(
        min(
            candidates,
            key=lambda threshold: (
                abs(
                    float(np.mean(probability_array >= threshold))
                    - label_rate
                ),
                -float(threshold),
            ),
        )
    )


def gate_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    label_array = np.asarray(labels, dtype=bool).reshape(-1)
    predictions = (
        np.asarray(probabilities, dtype=np.float64).reshape(-1)
        >= float(threshold)
    )
    true_positive = int(np.sum(label_array & predictions))
    false_positive = int(np.sum(~label_array & predictions))
    false_negative = int(np.sum(label_array & ~predictions))
    true_negative = int(np.sum(~label_array & ~predictions))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "threshold": float(threshold),
        "prediction_rate": float(np.mean(predictions)),
        "accuracy": (true_positive + true_negative) / max(label_array.size, 1),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(
            2.0 * precision * recall / max(precision + recall, 1e-12)
        ),
    }


def residual_fit_summary(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    num_facilities: int,
    activity_threshold: float,
) -> dict[str, Any]:
    target = np.asarray(targets, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if target.shape != predicted.shape or target.ndim != 2:
        raise ValueError("Residual targets and predictions require equal rank-2 shapes")
    if target.shape[1] != 4 * int(num_facilities):
        raise ValueError("Facility-net residuals require four facility-sized groups")
    if not 0.0 <= float(activity_threshold) <= 1.0:
        raise ValueError("activity_threshold must lie in [0, 1]")

    n = int(num_facilities)
    slices = {
        group: slice(index * n, (index + 1) * n)
        for index, group in enumerate(GROUPS)
    }
    return {
        "overall": residual_group_fit(
            target,
            predicted,
            activity_threshold=activity_threshold,
        ),
        "by_group": {
            group: residual_group_fit(
                target[:, group_slice],
                predicted[:, group_slice],
                activity_threshold=activity_threshold,
            )
            for group, group_slice in slices.items()
        },
    }


def residual_group_fit(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    activity_threshold: float,
) -> dict[str, Any]:
    target = np.asarray(targets, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    target_active = np.abs(target) > float(activity_threshold)
    predicted_active = np.abs(predicted) > float(activity_threshold)
    target_rows = target_active.any(axis=1)
    predicted_rows = predicted_active.any(axis=1)
    changed = target_active
    unchanged = ~changed
    target_norm = np.linalg.norm(target, axis=1)
    predicted_norm = np.linalg.norm(predicted, axis=1)
    cosine_rows = target_norm > 1e-12
    cosine = np.zeros(target.shape[0], dtype=np.float64)
    cosine[cosine_rows] = (
        (target[cosine_rows] * predicted[cosine_rows]).sum(axis=1)
        / np.maximum(
            target_norm[cosine_rows] * predicted_norm[cosine_rows],
            1e-12,
        )
    )
    return {
        "rows": int(target.shape[0]),
        "dimensions": int(target.shape[1]),
        "mse": float(np.mean((predicted - target) ** 2)),
        "mae": float(np.mean(np.abs(predicted - target))),
        "target_abs_mean": float(np.mean(np.abs(target))),
        "prediction_abs_mean": float(np.mean(np.abs(predicted))),
        "target_active_row_fraction": float(np.mean(target_rows)),
        "prediction_active_row_fraction": float(np.mean(predicted_rows)),
        "false_positive_row_fraction": float(
            np.mean(predicted_rows[~target_rows])
            if np.any(~target_rows)
            else 0.0
        ),
        "false_negative_row_fraction": float(
            np.mean(~predicted_rows[target_rows])
            if np.any(target_rows)
            else 0.0
        ),
        "changed_dimension_recall": float(
            np.mean(predicted_active[changed]) if np.any(changed) else 0.0
        ),
        "unchanged_dimension_false_positive_rate": float(
            np.mean(predicted_active[unchanged]) if np.any(unchanged) else 0.0
        ),
        "changed_dimension_sign_accuracy": float(
            np.mean(
                np.sign(predicted[changed]) == np.sign(target[changed])
            )
            if np.any(changed)
            else 0.0
        ),
        "changed_dimension_abs_ratio": float(
            np.mean(np.abs(predicted[changed]))
            / max(float(np.mean(np.abs(target[changed]))), 1e-12)
            if np.any(changed)
            else 0.0
        ),
        "mean_cosine_on_changed_rows": float(
            np.mean(cosine[target_rows]) if np.any(target_rows) else 0.0
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
