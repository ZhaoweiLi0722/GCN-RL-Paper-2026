"""Public-event, two-request feedback tables; no online learning or oracle state."""

import itertools
import math

import numpy as np

from src.baselines.completion_rollout_control import reservation_action
from src.env.completion_feedback_queue import (
    advance, booked_backlog_action, closed, completion_probability, validate_observation,
)


def reachable_events(state, response, cfg):
    validate_observation(state, cfg)
    probabilities = []
    for site in (0, 1):
        eligible = any(j.site == site and s == "support" for j, s in zip(cfg.jobs, state.stages))
        probabilities.append(completion_probability(response[site], state.pending_hours[site]) if eligible else 0.0)
    branches = []
    for bits in itertools.product((False, True), repeat=2):
        probability = math.prod(p if bit else 1-p for p, bit in zip(probabilities, bits))
        if probability == 0:
            continue
        representative = [p/2 if bit else p+(1-p)/2 for p, bit in zip(probabilities, bits)]
        branches.append({"code": int(bits[0])+2*int(bits[1]), "events": list(bits),
                         "probability": probability, "representative_uniforms": representative})
    return sorted(branches, key=lambda b: b["code"])


def second_action_indices(costs, budgets):
    costs = np.asarray(costs)
    if costs.ndim != 2 or costs.shape[1] != max(budgets) or not np.isfinite(costs).all():
        raise ValueError("incomplete inner cost matrix")
    return {str(n): int(np.argmin(costs[:, :n].mean(axis=1))) for n in budgets}


def scalar_continuation(state, first_index, grid, response, noise, cfg, boundary, method, table=None):
    if method not in ("booked", "reservation", "tree32", "tree128"):
        raise ValueError("unknown continuation")
    costs, trace, branch = np.zeros(3), [], None
    for offset, draw in enumerate(noise):
        if closed(state):
            break
        if state.epoch == boundary:
            action = (0, 0)
        elif offset == 0:
            action = grid[first_index]
        elif state.epoch > boundary:
            action = booked_backlog_action(state, cfg)
        elif method == "reservation":
            action = reservation_action(state, response, cfg, grid)
        elif method.startswith("tree") and offset == 1:
            choice = int(table[branch, first_index])
            if not 0 <= choice < len(grid):
                raise ValueError("unreachable or invalid tree lookup")
            action = grid[choice]
        else:
            action = booked_backlog_action(state, cfg)
        state, receipt, charges = advance(state, action, response, draw, cfg)
        if offset == 0:
            branch = int(receipt.completed[0])+2*int(receipt.completed[1])
        costs += [charges[k] for k in ("holding", "labor", "switching")]
        trace.append({"epoch": receipt.epoch, "action": list(action), "branch": branch})
    if not closed(state):
        raise RuntimeError("unsettled scalar continuation tail")
    return costs, trace
