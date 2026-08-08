"""Sparse, flow-conserving projection for facility-net residual actions."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.rl.networks import torch


class ResidualEndpointProjection:
    """Keep only the most confident transfer endpoints in each action group.

    Facility-net transfer actions encode source clinics as negative entries and
    destination clinics as positive entries. The projection retains at most
    ``max_endpoints_per_side`` entries on each side and balances the two totals
    without increasing either side's proposed flow.
    """

    _GROUP_INDEX = {
        "specimen_transfer": 0,
        "reagent_transfer": 1,
        "capacity_transfer": 2,
        "replenishment": 3,
    }

    def __init__(
        self,
        *,
        num_facilities: int,
        action_dim: int,
        settings: dict[str, Any] | None = None,
    ) -> None:
        config = dict(settings or {})
        self.num_facilities = int(num_facilities)
        self.action_dim = int(action_dim)
        self.enabled = bool(config.get("enabled", False))
        self.max_endpoints_per_side = max(
            int(config.get("max_endpoints_per_side", 1)),
            1,
        )
        self.min_abs = max(float(config.get("min_abs", 0.0)), 0.0)
        self.straight_through_gradient = bool(
            config.get("straight_through_gradient", False)
        )
        self.groups = tuple(
            str(group)
            for group in config.get(
                "groups",
                ("reagent_transfer", "capacity_transfer"),
            )
        )
        unknown = tuple(
            group for group in self.groups if group not in self._GROUP_INDEX
        )
        if unknown:
            raise ValueError(
                "Unsupported endpoint-projection groups: "
                + ", ".join(unknown)
            )
        if self.enabled and self.action_dim != 4 * self.num_facilities:
            raise ValueError(
                "Endpoint projection requires a facility-net action layout"
            )

    def apply_numpy(self, actions: np.ndarray) -> np.ndarray:
        array = np.asarray(actions, dtype=np.float32)
        if not self.enabled:
            return array.copy()
        single = array.ndim == 1
        batch = array.reshape(1, -1) if single else array.copy()
        if batch.ndim != 2 or batch.shape[1] != self.action_dim:
            raise ValueError(
                "Endpoint projection expected actions with shape "
                f"(batch, {self.action_dim}), got {tuple(batch.shape)}"
            )
        result = batch.copy()
        for group in self.groups:
            group_slice = self._group_slice(group)
            result[:, group_slice] = self._project_numpy_group(
                result[:, group_slice]
            )
        return result[0] if single else result

    def apply_tensor(self, actions):
        if not self.enabled:
            return actions
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(
                "Endpoint projection expected rank-2 actions with width "
                f"{self.action_dim}, got {tuple(actions.shape)}"
            )
        projected = actions
        for group in self.groups:
            group_slice = self._group_slice(group)
            values = projected[:, group_slice]
            replacement = self._project_tensor_group(values)
            if self.straight_through_gradient:
                # Preserve the exact hard projection in the forward pass while
                # keeping a zero-initialized residual actor trainable.
                replacement = values + (replacement - values).detach()
            projected = torch.cat(
                (
                    projected[:, : group_slice.start],
                    replacement,
                    projected[:, group_slice.stop :],
                ),
                dim=1,
            )
        return projected

    def _group_slice(self, group: str) -> slice:
        start = self._GROUP_INDEX[group] * self.num_facilities
        return slice(start, start + self.num_facilities)

    def _project_numpy_group(self, values: np.ndarray) -> np.ndarray:
        projected = np.zeros_like(values)
        count = min(self.max_endpoints_per_side, values.shape[1])
        for row_index, row in enumerate(values):
            source_scores = np.where(row < -self.min_abs, -row, 0.0)
            target_scores = np.where(row > self.min_abs, row, 0.0)
            source_indices = _positive_topk_indices(source_scores, count)
            target_indices = _positive_topk_indices(target_scores, count)
            if not source_indices.size or not target_indices.size:
                continue
            source_total = float(source_scores[source_indices].sum())
            target_total = float(target_scores[target_indices].sum())
            balanced_total = min(source_total, target_total)
            if balanced_total <= 0.0:
                continue
            projected[row_index, source_indices] = (
                row[source_indices] * balanced_total / source_total
            )
            projected[row_index, target_indices] = (
                row[target_indices] * balanced_total / target_total
            )
        return projected

    def _project_tensor_group(self, values):
        count = min(self.max_endpoints_per_side, values.shape[1])
        source_scores = torch.where(
            values < -self.min_abs,
            -values,
            torch.zeros_like(values),
        )
        target_scores = torch.where(
            values > self.min_abs,
            values,
            torch.zeros_like(values),
        )
        source_values, source_indices = torch.topk(
            source_scores,
            k=count,
            dim=1,
        )
        target_values, target_indices = torch.topk(
            target_scores,
            k=count,
            dim=1,
        )
        source_mask = torch.zeros_like(values).scatter(
            1,
            source_indices,
            (source_values > 0.0).to(dtype=values.dtype),
        )
        target_mask = torch.zeros_like(values).scatter(
            1,
            target_indices,
            (target_values > 0.0).to(dtype=values.dtype),
        )
        selected_sources = values.clamp_max(0.0) * source_mask
        selected_targets = values.clamp_min(0.0) * target_mask
        source_total = -selected_sources.sum(dim=1, keepdim=True)
        target_total = selected_targets.sum(dim=1, keepdim=True)
        balanced_total = torch.minimum(source_total, target_total)
        source_scale = torch.where(
            source_total > 0.0,
            balanced_total / source_total.clamp_min(1e-8),
            torch.zeros_like(source_total),
        )
        target_scale = torch.where(
            target_total > 0.0,
            balanced_total / target_total.clamp_min(1e-8),
            torch.zeros_like(target_total),
        )
        return (
            selected_sources * source_scale
            + selected_targets * target_scale
        )


def _positive_topk_indices(scores: np.ndarray, count: int) -> np.ndarray:
    positive = np.flatnonzero(scores > 0.0)
    if positive.size <= count:
        return positive
    local = np.argpartition(scores[positive], -count)[-count:]
    return positive[local]
