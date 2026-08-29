"""Value of state-dependence: the ceiling on what any learned policy can add.

A screen that asks "is there headroom over the anchor?" cannot tell whether a
learned policy is needed to capture it. Overtime Stage E2 passed that gate
while a single hard-coded rung captured essentially the whole gain, which is
the same failure mode the routing campaign documented: a simple baseline banks
the value and online learning adds nothing measurable.

This module measures the right quantity directly, from screen rows that have
already been collected:

- ``anchor_total``   -- the closed-form anchor policy's cost.
- ``constant_total`` -- the best single rung applied in every state (the tuned
  scalar reference a learned policy must beat).
- ``oracle_total``   -- the per-state best rung, an upper bound no policy can
  exceed, since it is chosen with knowledge of the realized outcome.

``value_of_state_dependence = constant_total - oracle_total`` is therefore the
entire budget available to any state-dependent policy, learned or otherwise.
When it is small relative to the attribution noise floor, no algorithm,
architecture, or training budget can produce a defensible win, and the channel
should be rejected before any agent is trained on it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

ANCHOR_ARM = "anchor"


def validation_means(
    rows: Iterable[Mapping[str, Any]],
    *,
    stream: str = "validation",
    metric: str = "remaining_cost",
) -> dict[tuple[str, str], float]:
    """Mean metric per (state, arm) over one replication stream."""

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if str(row["stream"]) != stream:
            continue
        grouped[(str(row["state_id"]), str(row["arm"]))].append(float(row[metric]))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def state_dependence_value(
    means: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    """Compare anchor, best-constant, and per-state-oracle totals.

    ``means`` maps (state_id, arm) to a mean cost, as returned by
    :func:`validation_means`. Lower cost is better.
    """

    states = sorted({state for state, _ in means})
    if not states:
        raise ValueError("no states in the supplied means")
    ladder = sorted({arm for _, arm in means if arm != ANCHOR_ARM})
    if not ladder:
        raise ValueError("no non-anchor arms in the supplied means")
    for state in states:
        missing = [arm for arm in ladder if (state, arm) not in means]
        if missing:
            raise ValueError(f"state {state} missing arms {missing}")

    anchor_total = sum(means[(state, ANCHOR_ARM)] for state in states)
    constant_totals = {
        arm: sum(means[(state, arm)] for state in states) for arm in ladder
    }
    best_constant_arm = min(constant_totals, key=constant_totals.get)
    constant_total = constant_totals[best_constant_arm]
    oracle_total = sum(min(means[(state, arm)] for arm in ladder) for state in states)

    per_state_best = {
        state: min(ladder, key=lambda arm: means[(state, arm)]) for state in states
    }
    distinct_best_arms = sorted(set(per_state_best.values()))
    interior_states = sum(
        1
        for state in states
        if per_state_best[state] not in (ladder[0], ladder[-1])
    )

    value = constant_total - oracle_total
    return {
        "state_count": len(states),
        "anchor_total": anchor_total,
        "best_constant_arm": best_constant_arm,
        "constant_total": constant_total,
        "oracle_total": oracle_total,
        "value_of_overtime_over_anchor": anchor_total - constant_total,
        "value_of_state_dependence": value,
        "value_of_state_dependence_fraction": value / anchor_total,
        "distinct_best_arms": distinct_best_arms,
        "distinct_best_arm_count": len(distinct_best_arms),
        "interior_best_arm_states": interior_states,
        "interior_best_arm_fraction": interior_states / len(states),
    }


def state_dependence_gate(
    report: Mapping[str, Any],
    *,
    minimum_fraction: float,
    minimum_interior_fraction: float = 0.0,
) -> dict[str, Any]:
    """Apply a prospective threshold to a state-dependence report.

    ``minimum_fraction`` should sit well above the attribution noise floor
    measured on the same simulator; a channel whose entire state-dependent
    budget is comparable to that floor cannot support a defensible learned
    result no matter how it is trained.
    """

    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must be within (0, 1]")
    if not 0.0 <= minimum_interior_fraction <= 1.0:
        raise ValueError("minimum_interior_fraction must be within [0, 1]")
    value_ok = report["value_of_state_dependence_fraction"] >= minimum_fraction
    interior_ok = report["interior_best_arm_fraction"] >= minimum_interior_fraction
    passed = bool(value_ok and interior_ok)
    return {
        "gate": "state_dependence",
        "minimum_fraction": minimum_fraction,
        "minimum_interior_fraction": minimum_interior_fraction,
        "measured_fraction": report["value_of_state_dependence_fraction"],
        "measured_interior_fraction": report["interior_best_arm_fraction"],
        "value_criterion_passed": bool(value_ok),
        "interior_criterion_passed": bool(interior_ok),
        "state_dependence_gate_passed": passed,
        "classification": (
            "state_dependent_headroom_established"
            if passed
            else "channel_captured_by_constant_policy"
        ),
    }
