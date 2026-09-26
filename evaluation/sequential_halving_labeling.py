"""Budget-efficient counterfactual labelling: CRN plus sequential halving.

Stage G1 concluded that the routing channel's best-action labels do not
replicate (54.5% agreement against a 70% gate) and closed the online-DDPG
extension on that basis. The operations-research literature reports that
combining common random numbers with sequential halving *inside the labelling
step* reduces optimality gaps by more than an order of magnitude at a fixed
simulation budget. If that holds here, G1's negative was underpowered rather
than fundamental, and the conclusion drawn from it needs revisiting.

Two facts about this simulator shape the design, both verified rather than
assumed:

1. **CRN pairing is already exact.** Patient attributes are drawn once at
   enrollment, and enrollment count depends on demand, not on the action. Two
   arms replayed from one state under one seed consume the identical random
   stream: same final RNG state, same enrollment counts, same patient ids.
   So within a single world an arm-vs-arm comparison carries no Monte Carlo
   noise at all, and there is no stream-partitioning work left to do.

2. **The remaining variance is across worlds.** The quantity a policy needs is
   the argmax of EXPECTED cost. Estimating it from few worlds makes the argmax
   noisy, which is what low agreement measures. More worlds fix it — but
   uniform allocation spends most of its budget separating arms that were never
   contenders.

Sequential halving spends the budget where the decision actually is: evaluate
all surviving arms on a slice of worlds, discard the worst half, double down on
the rest. For the same total simulator calls it resolves the top of the ladder
far more sharply than uniform allocation, at the cost of loose estimates for
arms that were eliminated early — which is the right trade when the output is
an argmax rather than a full ranking.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# evaluate(arm, world_seed) -> cost, with lower being better.
Evaluator = Callable[[Any, int], float]


def uniform_allocation_best_arm(
    arms: Sequence[Any],
    evaluate: Evaluator,
    world_seeds: Sequence[int],
) -> dict[str, Any]:
    """Baseline: every arm on every world. Cost = len(arms) * len(world_seeds)."""

    means = {}
    for arm in arms:
        means[arm] = float(np.mean([evaluate(arm, seed) for seed in world_seeds]))
    best = min(means, key=means.get)
    return {
        "best_arm": best,
        "means": means,
        "simulator_calls": len(arms) * len(world_seeds),
        "worlds_on_best": len(world_seeds),
    }


def sequential_halving_best_arm(
    arms: Sequence[Any],
    evaluate: Evaluator,
    world_seeds: Sequence[int],
    *,
    budget: int | None = None,
) -> dict[str, Any]:
    """Sequential halving over worlds, reusing CRN seeds across surviving arms.

    ``budget`` is the total number of simulator calls; it defaults to the
    uniform-allocation budget so the two are directly comparable. Each round
    draws its worlds from ``world_seeds`` cyclically, so every arm in a round
    is scored on the same worlds and the comparison stays paired.
    """

    arms = list(arms)
    if len(arms) < 2:
        raise ValueError("at least two arms are required")
    if not world_seeds:
        raise ValueError("at least one world seed is required")
    if budget is None:
        budget = len(arms) * len(world_seeds)

    rounds = max(1, math.ceil(math.log2(len(arms))))
    surviving = list(arms)
    calls = 0
    cursor = 0
    history: list[dict[str, Any]] = []
    totals: dict[Any, list[float]] = {arm: [] for arm in arms}

    for round_index in range(rounds):
        if len(surviving) == 1:
            break
        per_arm = max(1, int(budget // (rounds * len(surviving))))
        seeds = [
            world_seeds[(cursor + offset) % len(world_seeds)]
            for offset in range(per_arm)
        ]
        cursor += per_arm
        round_means: dict[Any, float] = {}
        for arm in surviving:
            values = [evaluate(arm, seed) for seed in seeds]
            calls += len(values)
            totals[arm].extend(values)
            round_means[arm] = float(np.mean(totals[arm]))
        keep = max(1, len(surviving) // 2)
        ordered = sorted(surviving, key=lambda arm: round_means[arm])
        history.append(
            {
                "round": round_index,
                "arms": len(surviving),
                "worlds_per_arm": per_arm,
                "eliminated": [str(a) for a in ordered[keep:]],
            }
        )
        surviving = ordered[:keep]

    best = min(surviving, key=lambda arm: float(np.mean(totals[arm])))
    return {
        "best_arm": best,
        "means": {arm: float(np.mean(v)) for arm, v in totals.items() if v},
        "simulator_calls": calls,
        "worlds_on_best": len(totals[best]),
        "rounds": history,
    }


def agreement_between(
    labeller: Callable[[Sequence[int]], Any],
    seed_groups: Iterable[Sequence[int]],
) -> dict[str, Any]:
    """Run a labeller on disjoint world groups and report argmax agreement."""

    picks = [labeller(group) for group in seed_groups]
    if len(picks) < 2:
        raise ValueError("at least two seed groups are required")
    reference = picks[0]
    agree = sum(1 for pick in picks[1:] if pick == reference)
    return {
        "picks": [str(p) for p in picks],
        "agreement": agree / (len(picks) - 1),
        "unanimous": len(set(map(str, picks))) == 1,
    }
