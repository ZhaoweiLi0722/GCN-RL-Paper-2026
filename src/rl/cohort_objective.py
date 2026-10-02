"""Detached raw-cost target receipt; no models, optimizers or environment calls."""

import math


def cohort_reward_receipt(prefix_costs, tail_costs, *, objective, prefix_steps,
                          accounting_steps, active_at_end, trajectory_id, split):
    """Append follow-up once to the last prefix reward, never rewrite raw costs.

    Both objectives acquire the complete same accounting window. The caller
    binds the episode/hash provenance and then applies the existing reward scale
    and PPO return calculation. No additional scaling/discounting occurs here.
    """
    if objective not in ("window", "cohort") or split != "training":
        raise ValueError("explicit training-only objective required")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory provenance required")
    if (type(prefix_steps) is not int or prefix_steps < 1
            or type(accounting_steps) is not int or accounting_steps < 1
            or type(active_at_end) is not int or active_at_end != 0):
        raise ValueError("fixed positive windows and fully resolved cohort required")
    prefix, tail = tuple(prefix_costs), tuple(tail_costs)
    if len(prefix) != prefix_steps or len(tail) != accounting_steps:
        raise ValueError("complete prefix and accounting tail required")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in prefix + tail):
        raise ValueError("finite nonnegative primitive costs required")
    prefix_total, tail_total = math.fsum(prefix), math.fsum(tail)
    cohort_total = math.fsum((prefix_total, tail_total))
    if not math.isfinite(cohort_total):
        raise ValueError("aggregate objective overflow")
    raw = tuple(-float(x) for x in prefix)
    target = list(raw)
    if objective == "cohort":
        target[-1] -= tail_total
    return dict(format="cohort-objective-target-v1", objective=objective,
                trajectory_id=trajectory_id, split=split,
                raw_prefix_rewards=raw, training_raw_rewards=tuple(target),
                window_cost=prefix_total, tail_cost=tail_total, cohort_cost=cohort_total,
                added_terminal_charge=tail_total if objective == "cohort" else 0.,
                environment_reward_unchanged=True, reward_scale_applied=False,
                bootstrap=0., tail_has_learned_actions=False)
