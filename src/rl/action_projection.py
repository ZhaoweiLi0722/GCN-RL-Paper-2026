"""Shared normalized-action projection utilities.

Policy networks emit normalized continuous actions in ``[-1, 1]``. The current
environment decodes those normalized actions into operational replenishment and
transfer decisions, then clips transfers against available inventory/capacity.
This module centralizes the policy-side projection used by all algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ProjectedAction:
    """Projected action and lightweight diagnostics."""

    action: np.ndarray
    clipped: bool
    repair_magnitude: float = 0.0


def projection_repair_magnitude(raw_action: Sequence[float]) -> float:
    """L2 norm of the feasibility repair (‖clip(a) − a‖); 0 if already in bounds.

    A per-step measure of how hard the policy pushes outside the normalized action
    box — the projection *load*. Always non-negative; used for the Phase 7
    projection-load report (the on/off ablation is deferred to Phase 9).
    """

    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    projected = np.clip(action, -1.0, 1.0)
    return float(np.linalg.norm(projected - action))


def project_action(
    raw_action: Sequence[float],
    env_state: Any | None = None,
    action_space_info: Any | None = None,
    graph_info: Any | None = None,
) -> ProjectedAction:
    """Project raw continuous policy output to the normalized action domain.

    Parameters are intentionally broad so future environment-specific feasibility
    repair can use state, graph, and action metadata without changing callers.
    The current environment handles operational feasibility during ``step``; this
    shared layer guarantees common shape and normalized bounds before execution.
    """

    del graph_info
    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    expected_size = _infer_action_size(env_state, action_space_info)
    if expected_size is not None and action.shape != (expected_size,):
        raise ValueError(f"Expected action shape {(expected_size,)}, got {action.shape}")

    projected = np.clip(action, -1.0, 1.0)
    return ProjectedAction(
        action=projected.astype(np.float32),
        clipped=not np.allclose(action, projected),
        repair_magnitude=float(np.linalg.norm(projected - action)),
    )


def _infer_action_size(env_state: Any | None, action_space_info: Any | None) -> int | None:
    for candidate in (action_space_info, env_state):
        if candidate is None:
            continue
        if isinstance(candidate, int):
            return candidate
        if hasattr(candidate, "action_size"):
            return int(candidate.action_size)
        if isinstance(candidate, dict) and "action_size" in candidate:
            return int(candidate["action_size"])
    return None
