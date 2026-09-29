"""Batched forecast of S1 physics; actual episodes still use the scalar model."""

import numpy as np

from src.env.completion_feedback_queue import completion_probability, validate_action, validate_observation


STAGES = ("scheduled", "support", "waiting", "active", "done")


def forecast(observation, prefixes, response, uniforms, config, decision_end, *, trace=False):
    """Score fixed request prefixes then the declared public continuation to closure.

    uniforms has shape (samples, remaining_intervals, 2), shared across prefixes.
    No future actual-world tape or latent response enters this model forecast.
    """
    validate_observation(observation, config)
    prefixes, uniforms = np.asarray(prefixes, dtype=float), np.asarray(uniforms, dtype=float)
    if prefixes.ndim != 3 or prefixes.shape[2] != 2 or not all(prefixes.shape):
        raise ValueError("nonempty candidate/prefix/site array required")
    for action in prefixes.reshape(-1, 2):
        validate_action(action, config)
    if uniforms.ndim != 3 or uniforms.shape[2] != 2 or not all(uniforms.shape):
        raise ValueError("nonempty sample/time/site tape required")
    if not np.isfinite(uniforms).all() or np.any((uniforms < 0) | (uniforms >= 1)):
        raise ValueError("invalid model uniforms")
    if len(response) != 2:
        raise ValueError("two assumed responses required")
    for rate in response:
        completion_probability(rate, 1)
    if type(decision_end) is not int or decision_end < 0:
        raise ValueError("invalid decision boundary")
    candidates, depth, _ = prefixes.shape
    samples, intervals, _ = uniforms.shape
    batch, jobs = candidates*samples, len(config.jobs)
    stages = np.tile([STAGES.index(s) for s in observation.stages], (batch, 1))
    ready = np.tile(observation.ready_epochs, (batch, 1))
    remaining = np.tile(observation.downstream_remaining, (batch, 1))
    pending = np.tile(observation.pending_hours, (batch, 1)).astype(float)
    previous = np.tile(observation.previous_request, (batch, 1)).astype(float)
    release = np.array([j.release for j in config.jobs])
    durations = np.array([j.downstream_steps for j in config.jobs])
    site = np.array([j.site for j in config.jobs])
    order = sorted(range(jobs), key=lambda i: (config.jobs[i].release, config.jobs[i].name))
    rank = np.empty(jobs, dtype=int)
    rank[order] = np.arange(jobs)
    index = np.arange(batch)
    effort = config.effort
    fraction = min(effort.shared_hour_budget/2, *effort.site_hour_caps)
    costs, queries, history = np.zeros(batch), 0, []
    for offset in range(intervals):
        t = observation.epoch+offset
        running = ~((stages == 4).all(axis=1) & (pending == 0).all(axis=1) & (previous == 0).all(axis=1))
        if not running.any():
            break
        queries += int(running.sum())
        if t == decision_end:
            action = np.zeros((batch, 2))
        elif offset < depth and t < decision_end:
            action = np.repeat(prefixes[:, offset, :], samples, axis=0)
        else:
            action = np.column_stack([fraction*np.any((site == s) & ((stages == 1) | (release == t+1)), axis=1)
                                      for s in (0, 1)])
        action[~running] = 0
        active = stages == 3
        free = config.downstream_slots-active.sum(axis=1)
        for slot in range(min(config.downstream_slots, jobs)):
            priority = np.where(stages == 2, ready*(jobs+1)+rank, np.iinfo(np.int64).max)
            choice = priority.argmin(axis=1)
            selected = (free > slot) & (stages[index, choice] == 2)
            r, j = index[selected], choice[selected]
            stages[r, j], remaining[r, j], active[r, j] = 3, durations[j], True
        remaining[active] -= 1
        stages[active & (remaining == 0)] = 4
        draw = np.tile(uniforms[:, offset, :], (candidates, 1))
        for s in (0, 1):
            eligible = (stages == 1) & (site == s)
            priority = np.where(eligible, rank, jobs+1)
            head = priority.argmin(axis=1)
            exposed = eligible.any(axis=1) & (pending[:, s] > 0)
            probability = -np.expm1(-response[s]*pending[:, s])
            event = exposed & (draw[:, s] < probability)
            r, j = index[event], head[event]
            stages[r, j], ready[r, j] = 2, t+1
        holding = config.holding_cost*np.sum((release <= t) & (stages != 4), axis=1)
        labor = np.sum(effort.hourly_cost*action + effort.quadratic_cost*action*action, axis=1)
        switch = effort.switching_cost*np.sum(np.abs(action-previous), axis=1)
        increment = holding+labor+switch
        costs += increment
        stages[(stages == 0) & (release == t+1)] = 1
        pending, previous = action.copy(), action.copy()
        if trace:
            history.append({"epoch": t+1, "stages": stages.copy(), "ready_epochs": ready.copy(),
                            "downstream_remaining": remaining.copy(), "pending_hours": pending.copy(),
                            "previous_request": previous.copy(), "cost": increment.copy()})
    if not ((stages == 4).all() and (pending == 0).all() and (previous == 0).all()):
        raise RuntimeError("forecast closure cap reached; unpaid tails cannot be scored")
    return {"sample_costs": costs.reshape(candidates, samples), "transition_queries": queries, "trace": history}
