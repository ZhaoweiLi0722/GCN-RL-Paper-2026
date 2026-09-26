"""Stage E3 label stability: does the per-state optimum replicate on fresh data?

Stage E2b established that choosing an overtime rung per state beats the best
constant. That measurement selected on the discovery stream and scored on the
validation stream, so its 96.3% best-arm agreement is a property of the two
streams used to make the selection. Re-reporting it as the E3 gate would be
circular: the same rows cannot both fit the selection and test whether it
replicates.

Stage E3 therefore re-runs the identical states and ladder under CRN streams
disjoint from every seed E2b used, and asks whether the same per-state optimum
comes back. Two criteria, mirroring the Stage G1 design that the routing
channel failed:

- ``best_action_agreement`` -- the fraction of states where two independent
  streams pick the same best rung. G1 required 70% and the routing channel
  managed 54.5%.
- ``pairwise_sign_agreement`` -- over pairs of rungs whose cost difference is
  material, the fraction where both streams agree which rung is cheaper. This
  is the weaker but broader signal; G1 found it can stay high while the
  top-action choice is unstable, so it is reported separately and never
  substitutes for the argmax criterion.

The module also reports agreement against the ORIGINAL E2b streams, which is
the strictest reading: a rung that replicates across four independent streams
is not a property of one seed family.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping

ANCHOR_ARM = "anchor"


def best_action_agreement(
    stream_a: Mapping[tuple[str, str], float],
    stream_b: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    """Fraction of states where two streams choose the same best rung."""

    states = sorted({state for state, _ in stream_a})
    ladder = sorted({arm for _, arm in stream_a if arm != ANCHOR_ARM})
    if not states or not ladder:
        raise ValueError("both states and non-anchor arms are required")
    per_state = {}
    for state in states:
        missing = [
            arm
            for arm in ladder
            if (state, arm) not in stream_a or (state, arm) not in stream_b
        ]
        if missing:
            raise ValueError(f"state {state} missing arms {missing}")
        choice_a = min(ladder, key=lambda arm: stream_a[(state, arm)])
        choice_b = min(ladder, key=lambda arm: stream_b[(state, arm)])
        per_state[state] = {
            "stream_a_best": choice_a,
            "stream_b_best": choice_b,
            "agree": choice_a == choice_b,
        }
    agreed = sum(1 for entry in per_state.values() if entry["agree"])
    return {
        "state_count": len(states),
        "agreeing_states": agreed,
        "best_action_agreement": agreed / len(states),
        "per_state": per_state,
    }


def pairwise_sign_agreement(
    stream_a: Mapping[tuple[str, str], float],
    stream_b: Mapping[tuple[str, str], float],
    *,
    material_threshold: float,
) -> dict[str, Any]:
    """Agreement on which rung is cheaper, over materially separated pairs.

    A pair counts as material when BOTH streams separate it by at least
    ``material_threshold``; pairs that are a coin flip in either stream carry
    no information about stability and are excluded rather than counted as
    disagreements.
    """

    if material_threshold <= 0.0:
        raise ValueError("material_threshold must be positive")
    states = sorted({state for state, _ in stream_a})
    ladder = sorted({arm for _, arm in stream_a if arm != ANCHOR_ARM})
    material = 0
    agreed = 0
    for state in states:
        for left, right in combinations(ladder, 2):
            delta_a = stream_a[(state, left)] - stream_a[(state, right)]
            delta_b = stream_b[(state, left)] - stream_b[(state, right)]
            if abs(delta_a) < material_threshold or abs(delta_b) < material_threshold:
                continue
            material += 1
            if (delta_a > 0) == (delta_b > 0):
                agreed += 1
    return {
        "material_pairs": material,
        "agreeing_pairs": agreed,
        "pairwise_sign_agreement": (agreed / material) if material else 0.0,
    }


def label_stability_gate(
    agreement: Mapping[str, Any],
    pairwise: Mapping[str, Any],
    *,
    minimum_best_action_agreement: float,
    minimum_pairwise_agreement: float,
) -> dict[str, Any]:
    """Apply both prospective E3 criteria.

    The argmax criterion is primary. Policy improvement needs a reliable choice
    of best action, not merely a weak average ordering -- the distinction that
    Stage G1 made explicit when the routing channel held 82.5% pairwise
    agreement while its top-action choice replicated only 54.5% of the time.
    """

    for name, value in (
        ("minimum_best_action_agreement", minimum_best_action_agreement),
        ("minimum_pairwise_agreement", minimum_pairwise_agreement),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be within (0, 1]")
    best_ok = agreement["best_action_agreement"] >= minimum_best_action_agreement
    pair_ok = pairwise["pairwise_sign_agreement"] >= minimum_pairwise_agreement
    passed = bool(best_ok and pair_ok)
    return {
        "gate": "e3_label_stability",
        "minimum_best_action_agreement": minimum_best_action_agreement,
        "minimum_pairwise_agreement": minimum_pairwise_agreement,
        "measured_best_action_agreement": agreement["best_action_agreement"],
        "measured_pairwise_sign_agreement": pairwise["pairwise_sign_agreement"],
        "material_pairs": pairwise["material_pairs"],
        "best_action_criterion_passed": bool(best_ok),
        "pairwise_criterion_passed": bool(pair_ok),
        "label_stability_gate_passed": passed,
        "classification": (
            "labels_replicate_on_fresh_streams"
            if passed
            else "unstable_counterfactual_labels"
        ),
        "e4_authorized": passed,
    }
