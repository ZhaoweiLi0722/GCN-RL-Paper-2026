"""Shared losses for teacher-calibrated off-policy critics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.rl.networks import torch


def teacher_advantage_support_mask(
    option_advantages: np.ndarray,
    option_feasible: np.ndarray,
    option_groups: np.ndarray,
    *,
    allowed_option_groups: Sequence[str],
) -> np.ndarray:
    """Select rows whose globally best feasible option is policy-supported."""

    advantages = np.asarray(option_advantages, dtype=np.float32)
    feasible = np.asarray(option_feasible, dtype=bool)
    groups = np.asarray(option_groups, dtype="U64")
    if advantages.ndim != 2 or feasible.shape != advantages.shape:
        raise ValueError("Teacher option advantages and feasibility must align")
    if groups.shape != (advantages.shape[1],):
        raise ValueError("Teacher option groups must align with option columns")
    if np.any(~np.any(feasible, axis=1)):
        raise ValueError("Every teacher state requires a feasible option")
    allowed = {str(group) for group in allowed_option_groups}
    if not allowed or any(not group for group in allowed):
        raise ValueError("Allowed teacher option groups cannot be empty")
    allowed.add("anchor")
    best_indices = np.argmax(
        np.where(feasible, advantages, -np.inf),
        axis=1,
    )
    return np.isin(groups[best_indices], sorted(allowed))


def teacher_advantage_loss(
    predictions,
    targets,
    *,
    positive_margin: float = 0.0,
    positive_margin_weight: float = 0.0,
    pairwise_difference_weight: float = 0.0,
):
    """Return pointwise MSE plus optional sign and pairwise structure."""

    regression_loss = torch.nn.functional.mse_loss(predictions, targets)
    margin_loss = predictions.sum() * 0.0
    if float(positive_margin_weight) > 0.0:
        material_positive = targets >= float(positive_margin)
        if bool(material_positive.any().item()):
            shortfall = torch.relu(
                float(positive_margin) - predictions[material_positive]
            )
            margin_loss = torch.mean(shortfall.square())
    pairwise_loss = predictions.sum() * 0.0
    if float(pairwise_difference_weight) > 0.0:
        predicted_values = predictions.reshape(-1)
        target_values = targets.reshape(-1)
        if int(predicted_values.numel()) > 1:
            left, right = torch.triu_indices(
                int(predicted_values.numel()),
                int(predicted_values.numel()),
                offset=1,
                device=predictions.device,
            )
            target_differences = target_values[left] - target_values[right]
            comparable = target_differences.abs() > 1e-15
            if bool(comparable.any().item()):
                predicted_differences = (
                    predicted_values[left] - predicted_values[right]
                )
                pairwise_loss = torch.nn.functional.mse_loss(
                    predicted_differences[comparable],
                    target_differences[comparable],
                )
    total_loss = (
        regression_loss
        + float(positive_margin_weight) * margin_loss
        + float(pairwise_difference_weight) * pairwise_loss
    )
    return total_loss, regression_loss, margin_loss, pairwise_loss
