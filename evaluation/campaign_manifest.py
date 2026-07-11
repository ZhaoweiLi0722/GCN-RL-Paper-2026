"""Full 20-clinic campaign configuration, scaled to avoid pilot undertraining.

The lean two-stage pilot (30k steps/seed) undertrained every learned policy at
20-clinic scale: the look-ahead heuristic ``mdl2`` beat the best graph backbone
by ~3x on cost and ~2x on eligibility, with zero divergent seeds -- i.e. the
learned policies were *stably underfit*, not unstable. Two levers address this:

  1. Budget: scale training ~10x. A 20-clinic action is ~80-dimensional; off-policy
     continuous control at this scale typically needs 1e5-1e6 steps, not 3e4.
  2. Representation: switch the graph actor from the default ``global_flat`` readout
     (head input = num_facilities x encoder_dim, ~1280-dim at 20 clinics, not
     size-invariant) to ``facility_action`` (a shared per-facility head: far fewer
     parameters, permutation-equivariant, and transferable across clinic counts,
     which also enables a 2->20 curriculum warm-start).

Lever (1) is verified-safe and encoded here. Lever (2) is a strong recommendation
but is TEST-GATED: ``facility_action`` reorders the action vector by facility, and
the env decodes actions grouped by *type* (w,e,q,p blocks), so the layout must be
verified before use. See specs/2026-07-11-pilot-experiments/campaign-scale-plan.md.
"""
from __future__ import annotations

from typing import Any

from evaluation.pilot_manifest import (
    HEURISTICS,
    SEEDS,
    STAGE_B_ENV,
    build_agent_config,
)

# ---- environment ---------------------------------------------------------
CAMPAIGN_ENV = STAGE_B_ENV  # 20-clinic patient-condition confirm env

# ---- training budget (the primary undertraining fix) ---------------------
# 10x the 30k pilot floor. Treat 300k as the campaign floor and 500k as the
# flagship-confirm budget; raise further if the flagship still trails mdl2.
CAMPAIGN_TRAIN_STEPS = 300_000
FLAGSHIP_TRAIN_STEPS = 500_000

# ---- roster --------------------------------------------------------------
# Full family: 3 flagship candidates + 2 ablations (graph-DDPG, flat-DDPG).
CAMPAIGN_LEARNED = ("gcn_td3", "gcn_sac", "gcn_ppo", "gcn_ddpg", "flat_ddpg")
CAMPAIGN_ROSTER = CAMPAIGN_LEARNED + HEURISTICS
CAMPAIGN_SEEDS = SEEDS  # 5 seeds; widen to range(10) for final tables if budget allows

# ---- wall-clock model ----------------------------------------------------
# Measured on the pilot Stage B (20-clinic, single CPU): the graph backbones ran
# ~4.5 min/seed at 30k steps => ~9 s per 1k steps per seed. flat_ddpg was faster
# (~2.8 min); 9.0 is a conservative per-learned-run rate.
SECONDS_PER_1K_STEPS_PER_SEED = 9.0


def campaign_config(algorithm: str, seed: int, *, steps: int = CAMPAIGN_TRAIN_STEPS) -> dict[str, Any]:
    """Training config for one learned algorithm on the 20-clinic campaign env.

    Reuses the pilot's verified builder (reward scale, no obs normalization,
    shared encoder/head sizes, 1M replay buffer defaults) and only raises the
    step budget. Heuristics need no training budget and are handled by the runner.
    """

    return build_agent_config(algorithm, CAMPAIGN_ENV, seed=seed, target_steps=steps)


def wallclock_estimate(
    steps: int = CAMPAIGN_TRAIN_STEPS,
    learned: tuple[str, ...] = CAMPAIGN_LEARNED,
    seeds: tuple[int, ...] = CAMPAIGN_SEEDS,
) -> dict[str, float]:
    """Serial wall-clock estimate for a campaign at a given step budget."""

    minutes_per_seed = SECONDS_PER_1K_STEPS_PER_SEED * (steps / 1000.0) / 60.0
    runs = len(learned) * len(seeds)
    return {
        "steps": steps,
        "runs": runs,
        "minutes_per_seed": round(minutes_per_seed, 1),
        "total_hours_serial": round(minutes_per_seed * runs / 60.0, 1),
    }


if __name__ == "__main__":
    for s in (150_000, CAMPAIGN_TRAIN_STEPS, FLAGSHIP_TRAIN_STEPS, 1_000_000):
        e = wallclock_estimate(s)
        print(
            f"{s:>9,} steps | {e['minutes_per_seed']:>5} min/seed | "
            f"{e['runs']} runs | ~{e['total_hours_serial']} h serial"
        )
