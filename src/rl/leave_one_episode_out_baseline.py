"""Unregistered scalar target transform; no model, optimizer or environment.

The caller must supply closed, equal-horizon, independently seeded training
episodes collected under one fixed behavior policy. Return targets and raw
rewards are not changed. Only the policy-gradient baseline is replaced.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TimeBaselineTargets:
    method: str
    trajectory_ids: tuple
    baselines: tuple
    advantages: tuple
    returns: tuple


def leave_one_episode_out_targets(episode_returns, *, trajectory_ids,
                                 environment_seeds, behavior_sha256s,
                                 terminal_flags):
    rows = tuple(tuple(values) for values in episode_returns)
    count = len(rows)
    if count < 2 or not rows[0] or len({len(row) for row in rows}) != 1:
        raise ValueError("at least two nonempty equal-horizon episodes required")
    ids, seeds, behavior, terminals = map(
        tuple, (trajectory_ids, environment_seeds, behavior_sha256s, terminal_flags))
    if any(len(values) != count for values in (ids, seeds, behavior, terminals)):
        raise ValueError("one provenance entry per episode required")
    if (any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != count
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != count):
        raise ValueError("distinct trajectory identifiers and environment seeds required")
    if (len(set(behavior)) != 1 or not isinstance(behavior[0], str)
            or len(behavior[0]) != 64 or any(c not in "0123456789abcdef" for c in behavior[0])):
        raise ValueError("one matching lowercase behavior SHA256 required")
    if any(flag is not True for flag in terminals):
        raise ValueError("only complete terminal episodes supported")
    if any(type(x) not in (int, float) or not math.isfinite(x) for row in rows for x in row):
        raise ValueError("finite scalar return targets required")
    # Exclude the owned episode before summation, not via total-minus-owned.
    baseline = tuple(tuple(math.fsum(rows[j][t] for j in range(count) if j != i)/(count-1)
                           for t in range(len(rows[i]))) for i in range(count))
    advantages = tuple(tuple(g-b for g, b in zip(row, base)) for row, base in zip(rows, baseline))
    if any(not math.isfinite(x) for field in (baseline, advantages) for row in field for x in row):
        raise ValueError("nonfinite transformed targets")
    return TimeBaselineTargets("leave-one-episode-out-time-v1", ids, baseline, advantages, rows)
