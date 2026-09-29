"""Matched fixed/identified full-settlement rollout controllers, not optimal MPC."""

import hashlib
import itertools
import math

import numpy as np

from src.env.completion_feedback_batch import forecast
from src.env.completion_feedback_queue import validate_observation


def named_seed(namespace, *parts):
    value = "|".join((namespace, *(str(p) for p in parts)))
    return int.from_bytes(hashlib.sha256(value.encode("ascii")).digest()[:8], "big")


def action_grid(denominator):
    if type(denominator) is not int or denominator < 1:
        raise ValueError("positive integer grid denominator required")
    return tuple((i/denominator, j/denominator) for i in range(denominator+1) for j in range(denominator+1-i))


def reservation_action(observation, response, config, grid):
    validate_observation(observation, config)
    if len(response) != 2 or any(not math.isfinite(r) or r <= 0 for r in response):
        raise ValueError("two positive response estimates required")
    target = []
    for site, rate in enumerate(response):
        count = sum(j.site == site and stage == "support" for j, stage in zip(config.jobs, observation.stages))
        new = any(j.site == site and j.release == observation.epoch+1 for j in config.jobs)
        need = 1.0 if new or count >= 2 else (math.exp(-rate*observation.pending_hours[site]) if count == 1 else 0.0)
        target.append(min(config.effort.shared_hour_budget/2, *config.effort.site_hour_caps)*need)
    return min(grid, key=lambda a: sum((x-y)**2 for x, y in zip(a, target)))


def zero_is_dominant(observation, config):
    return (config.effort.hourly_cost >= 2*config.effort.switching_cost and
            "support" not in observation.stages and
            not any(j.release == observation.epoch+1 for j in config.jobs))


def plan_rollout(observation, response, config, grid, samples, depth, seed, decision_end):
    validate_observation(observation, config)
    if len(response) != 2 or any(not math.isfinite(r) or r <= 0 for r in response):
        raise ValueError("two positive response estimates required")
    if type(samples) is not int or samples < 1 or depth not in (1, 2):
        raise ValueError("positive sample count and depth 1 or 2 required")
    if observation.epoch >= decision_end:
        raise ValueError("optimized request outside decision window")
    if zero_is_dominant(observation, config):
        return (0.0, 0.0), {"dominant_zero": True, "transition_queries": 0, "candidates": 0}
    prefixes = tuple(itertools.product(grid, repeat=depth))
    intervals = decision_end+config.max_closure_steps-observation.epoch
    uniforms = np.random.Generator(np.random.PCG64(seed)).random((samples, intervals, 2))
    result = forecast(observation, prefixes, tuple(response), uniforms, config, decision_end)
    costs = result["sample_costs"]
    means = costs.mean(axis=1)
    best = int(np.argmin(means))
    return prefixes[best][0], {"dominant_zero": False, "transition_queries": result["transition_queries"],
                               "candidates": len(prefixes), "samples": samples, "depth": depth,
                               "selected_prefix": prefixes[best], "selected_index": best,
                               "candidate_means": means.tolist(),
                               "candidate_mc_se": (costs.std(axis=1, ddof=1)/math.sqrt(samples)).tolist() if samples > 1 else None}
