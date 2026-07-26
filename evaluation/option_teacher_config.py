"""Configuration helpers for residual-option teacher caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.rl.residual_options import SUPPORTED_RESIDUAL_OPTION_GROUPS


def explicit_options_from_teacher_cache(
    path: str | Path,
) -> list[dict[str, float | str]]:
    """Return the cache's ordered correction options, excluding the anchor."""

    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as payload:
        required = ("option_groups", "option_epsilons", "option_signs")
        missing = [name for name in required if name not in payload.files]
        if missing:
            raise ValueError(
                f"Teacher cache is missing option metadata: {missing}"
            )
        groups = np.asarray(payload["option_groups"], dtype="U32")
        epsilons = np.asarray(payload["option_epsilons"], dtype=float)
        signs = np.asarray(payload["option_signs"], dtype=float)

    if groups.ndim != 1 or groups.size < 2:
        raise ValueError(
            "Teacher cache must contain one anchor and at least one correction option"
        )
    if epsilons.shape != groups.shape or signs.shape != groups.shape:
        raise ValueError("Teacher option metadata arrays must have identical shapes")
    if (
        groups[0] != "anchor"
        or not np.isclose(epsilons[0], 0.0)
        or not np.isclose(signs[0], 0.0)
    ):
        raise ValueError("Teacher option zero must be anchor:0:0")

    explicit_options: list[dict[str, float | str]] = []
    seen: set[tuple[str, float, float]] = set()
    supported = set(SUPPORTED_RESIDUAL_OPTION_GROUPS)
    for group, epsilon, sign in zip(groups[1:], epsilons[1:], signs[1:]):
        group_name = str(group)
        epsilon_value = float(epsilon)
        sign_value = float(sign)
        if group_name not in supported:
            raise ValueError(f"Unsupported teacher option group: {group_name}")
        if (
            not np.isfinite(epsilon_value)
            or epsilon_value <= 0.0
            or epsilon_value > 1.0
        ):
            raise ValueError("Teacher option epsilons must lie in (0, 1]")
        if sign_value not in (-1.0, 1.0):
            raise ValueError("Teacher option signs must be -1 or 1")
        key = (group_name, epsilon_value, sign_value)
        if key in seen:
            raise ValueError(
                f"Teacher cache contains duplicate option: "
                f"{group_name}:{sign_value:g}:{epsilon_value:g}"
            )
        seen.add(key)
        explicit_options.append(
            {
                "group": group_name,
                "epsilon": epsilon_value,
                "sign": sign_value,
            }
        )
    return explicit_options


def configure_options_from_teacher_cache(
    config: dict[str, Any],
    path: str | Path,
) -> list[dict[str, float | str]]:
    """Configure an option agent to use every ordered option in a cache."""

    explicit_options = explicit_options_from_teacher_cache(path)
    config.setdefault("residual_option", {})[
        "explicit_options"
    ] = explicit_options
    return explicit_options


def select_restricted_teacher_options(
    option_groups: np.ndarray,
    option_advantages: np.ndarray,
    option_feasible: np.ndarray,
    *,
    allowed_groups: tuple[str, ...],
    min_advantage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose the best feasible option from an anchor plus allowed groups."""

    groups = np.asarray(option_groups, dtype="U32")
    advantages = np.asarray(option_advantages, dtype=np.float32)
    feasible = np.asarray(option_feasible, dtype=bool)
    if groups.ndim != 1 or groups.size < 1 or groups[0] != "anchor":
        raise ValueError("Teacher options must start with the anchor")
    if advantages.ndim != 2 or advantages.shape[1] != groups.size:
        raise ValueError("Teacher advantages must align with option groups")
    if feasible.shape != advantages.shape:
        raise ValueError("Teacher feasibility must align with advantages")
    if not np.all(np.isfinite(advantages)):
        raise ValueError("Teacher advantages must be finite")
    if not np.all(feasible[:, 0]):
        raise ValueError("The anchor option must always be feasible")
    threshold = float(min_advantage)
    if threshold < 0.0:
        raise ValueError("min_advantage must be non-negative")

    allowed = tuple(str(group) for group in allowed_groups)
    unknown = sorted(
        set(allowed) - set(SUPPORTED_RESIDUAL_OPTION_GROUPS)
    )
    if unknown:
        raise ValueError(f"Unsupported restricted option groups: {unknown}")
    selected_indices = np.asarray(
        [
            index
            for index, group in enumerate(groups)
            if index == 0 or str(group) in allowed
        ],
        dtype=np.int64,
    )
    if selected_indices.size == 1:
        raise ValueError("Restricted teacher must retain a correction option")

    restricted_advantages = advantages[:, selected_indices].copy()
    restricted_feasible = feasible[:, selected_indices]
    if restricted_advantages.shape[1] > 1 and threshold > 0.0:
        restricted_advantages[:, 1:] -= threshold
    restricted_advantages[~restricted_feasible] = -np.inf
    local_labels = np.argmax(restricted_advantages, axis=1).astype(np.int64)
    original_labels = selected_indices[local_labels]
    selected_advantages = advantages[
        np.arange(advantages.shape[0]),
        original_labels,
    ]
    return selected_indices, original_labels, selected_advantages
