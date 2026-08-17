"""Validation helpers for seed-locked persistent scenario assignments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def normalize_scenario_by_seed(
    raw_assignment: Any,
    *,
    available_scenarios: Sequence[str],
    required_seeds: Sequence[int] = (),
    reject_extra_seeds: bool = False,
) -> dict[int, str]:
    """Normalize and validate an optional training-seed scenario contract."""

    if raw_assignment in (None, {}):
        return {}
    if not isinstance(raw_assignment, Mapping):
        raise TypeError("scenario_by_seed must be a mapping")

    available = {str(name) for name in available_scenarios}
    normalized: dict[int, str] = {}
    for raw_seed, raw_scenario in raw_assignment.items():
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid scenario_by_seed key: {raw_seed!r}"
            ) from error
        scenario = str(raw_scenario)
        if seed in normalized:
            raise ValueError(f"Duplicate scenario assignment for seed {seed}")
        if scenario not in available:
            raise ValueError(
                f"Scenario assignment for seed {seed} references unknown "
                f"scenario {scenario!r}"
            )
        normalized[seed] = scenario

    required = {int(seed) for seed in required_seeds}
    missing = sorted(required - set(normalized))
    if missing:
        raise ValueError(
            "scenario_by_seed is missing required seeds: "
            + ", ".join(str(seed) for seed in missing)
        )
    if reject_extra_seeds:
        extra = sorted(set(normalized) - required)
        if extra:
            raise ValueError(
                "scenario_by_seed contains unconfigured seeds: "
                + ", ".join(str(seed) for seed in extra)
            )
    return normalized


def serialized_scenario_by_seed(
    assignment: Mapping[int, str],
) -> dict[str, str]:
    return {
        str(seed): str(assignment[seed])
        for seed in sorted(assignment)
    }
