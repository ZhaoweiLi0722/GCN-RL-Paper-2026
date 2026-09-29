"""Additive S4 forecast with S1 physics and independently audited cost components.

The locked S1 batch implementation remains untouched. This evaluator adds public
reservation feedback and a frozen second-request tree, not new dynamics.
"""

import numpy as np

from src.env.completion_feedback_batch import STAGES
from src.env.completion_feedback_queue import completion_probability, validate_action, validate_observation


def forecast_continuations(state, grid, response, uniforms, cfg, boundary, method, table=None):
    validate_observation(state, cfg)
    if method not in ("booked", "reservation", "tree32", "tree128"):
        raise ValueError("unknown continuation")
    grid, noise = np.asarray(grid, dtype=float), np.asarray(uniforms, dtype=float)
    if grid.ndim != 2 or grid.shape[1] != 2 or not len(grid):
        raise ValueError("nonempty two-site action grid required")
    for action in grid:
        validate_action(action, cfg)
    if noise.ndim != 3 or noise.shape[2] != 2 or not all(noise.shape):
        raise ValueError("sample/time/site uniforms required")
    if not np.isfinite(noise).all() or np.any((noise < 0) | (noise >= 1)):
        raise ValueError("invalid uniforms")
    if len(response) != 2:
        raise ValueError("two rates required")
    for rate in response:
        completion_probability(rate, 1)
    if type(boundary) is not int or not state.epoch < boundary:
        raise ValueError("first request outside decision window")
    if method.startswith("tree"):
        table = np.asarray(table)
        if table.shape != (4, len(grid)) or not np.issubdtype(table.dtype, np.integer):
            raise ValueError("four-branch integer action table required")
        if np.any((table < -1) | (table >= len(grid))):
            raise ValueError("invalid tree action index")
    actions, samples, jobs = len(grid), len(noise), len(cfg.jobs)
    batch = actions*samples
    stages = np.tile([STAGES.index(s) for s in state.stages], (batch, 1))
    ready = np.tile(state.ready_epochs, (batch, 1))
    remaining = np.tile(state.downstream_remaining, (batch, 1))
    pending = np.tile(state.pending_hours, (batch, 1)).astype(float)
    previous = np.tile(state.previous_request, (batch, 1)).astype(float)
    release = np.array([j.release for j in cfg.jobs])
    durations = np.array([j.downstream_steps for j in cfg.jobs])
    site = np.array([j.site for j in cfg.jobs])
    order = sorted(range(jobs), key=lambda i: (cfg.jobs[i].release, cfg.jobs[i].name))
    rank = np.empty(jobs, dtype=int)
    rank[order] = np.arange(jobs)
    index, first_indices = np.arange(batch), np.repeat(np.arange(actions), samples)
    fraction = min(cfg.effort.shared_hour_budget/2, *cfg.effort.site_hour_caps)
    costs, queries, branches = np.zeros((batch, 3)), 0, np.zeros(batch, dtype=int)
    for offset in range(noise.shape[1]):
        t = state.epoch+offset
        running = ~((stages == 4).all(axis=1) & (pending == 0).all(axis=1) & (previous == 0).all(axis=1))
        if not running.any():
            break
        queries += int(running.sum())
        if t == boundary:
            action = np.zeros((batch, 2))
        elif offset == 0:
            action = grid[first_indices].copy()
        elif method.startswith("tree") and offset == 1 and t < boundary:
            choices = table[branches, first_indices]
            if np.any(choices[running] < 0):
                raise ValueError("unreachable tree branch encountered")
            action = grid[choices].copy()
        elif method == "reservation" and t < boundary:
            targets = []
            for s in (0, 1):
                count = np.sum((site == s) & (stages == 1), axis=1)
                new = np.any((site == s) & (release == t+1))
                need = np.where(new | (count >= 2), 1., np.where(count == 1, np.exp(-response[s]*pending[:, s]), 0.))
                targets.append(fraction*need)
            target = np.column_stack(targets)
            distances = np.sum((target[:, None, :]-grid[None, :, :])**2, axis=2)
            action = grid[np.argmin(distances, axis=1)].copy()
        else:
            action = np.column_stack([fraction*np.any((site == s) & ((stages == 1) | (release == t+1)), axis=1)
                                      for s in (0, 1)])
        action[~running] = 0
        active = stages == 3
        free = cfg.downstream_slots-active.sum(axis=1)
        for slot in range(min(cfg.downstream_slots, jobs)):
            priority = np.where(stages == 2, ready*(jobs+1)+rank, np.iinfo(np.int64).max)
            choice = priority.argmin(axis=1)
            selected = (free > slot) & (stages[index, choice] == 2)
            r, j = index[selected], choice[selected]
            stages[r, j], remaining[r, j], active[r, j] = 3, durations[j], True
        remaining[active] -= 1
        stages[active & (remaining == 0)] = 4
        draws = np.tile(noise[:, offset, :], (actions, 1))
        for s in (0, 1):
            eligible = (stages == 1) & (site == s)
            head = np.where(eligible, rank, jobs+1).argmin(axis=1)
            exposed = eligible.any(axis=1) & (pending[:, s] > 0)
            event = exposed & (draws[:, s] < -np.expm1(-response[s]*pending[:, s]))
            r, j = index[event], head[event]
            stages[r, j], ready[r, j] = 2, t+1
            if offset == 0:
                branches += (2**s)*event.astype(int)
        holding = cfg.holding_cost*np.sum((release <= t) & (stages != 4), axis=1)
        labor = np.sum(cfg.effort.hourly_cost*action + cfg.effort.quadratic_cost*action*action, axis=1)
        switch = cfg.effort.switching_cost*np.sum(np.abs(action-previous), axis=1)
        costs += np.column_stack((holding, labor, switch))
        stages[(stages == 0) & (release == t+1)] = 1
        pending, previous = action.copy(), action.copy()
    if not ((stages == 4).all() and (pending == 0).all() and (previous == 0).all()):
        raise RuntimeError("continuation closure cap reached")
    return {"components": costs.reshape(actions, samples, 3), "transition_queries": queries}
