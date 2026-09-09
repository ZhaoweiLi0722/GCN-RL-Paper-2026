"""Exact finite-action, finite-world planning for small mechanics fixtures.

The model exposes immutable hashable states, information_key(state) containing
the FULL permitted observation/action/reward history, and deterministic
transition(state, action) -> (cost, next_state). World identity is internal,
never automatically part of the information key. This is a known-model
diagnostic, not a production MPC or an online reinforcement-learning method.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from functools import lru_cache
from typing import Any


def solve_fixture(model: Any, *, worlds: tuple, actions: tuple[str, ...], horizon: int) -> dict[str, Any]:
    if type(horizon) is not int or not 1 <= horizon <= 6:
        raise ValueError("fixture horizon must be between one and six")
    if not worlds or len(worlds) > 2 or not actions or len(actions) > 4 or len(set(actions)) != len(actions):
        raise ValueError("fixture supports at most two worlds and four unique actions")
    if len({state for _, state in worlds}) != len(worlds):
        raise ValueError("world states must be distinct")
    if any(not math.isfinite(p) or p <= 0 for p, _ in worlds) or not math.isclose(sum(p for p, _ in worlds), 1.0, abs_tol=1e-12, rel_tol=0):
        raise ValueError("strictly positive world probabilities must sum to one")

    @lru_cache(maxsize=None)
    def transition(state, action):
        cost, successor = model.transition(state, action)
        if not math.isfinite(float(cost)):
            raise ValueError("nonfinite transition cost")
        return float(cost), successor

    def solve(group, remaining):
        if not remaining:
            return 0.0, {}
        partitions = defaultdict(list)
        for probability, state in group:
            partitions[model.information_key(state)].append((probability, state))
        if len(partitions) > 1:
            total, policy = 0.0, {}
            for subset in partitions.values():
                value, subpolicy = solve(tuple(subset), remaining)
                total += value
                policy.update(subpolicy)
            return total, policy
        key = (remaining, next(iter(partitions)))
        alternatives = []
        for action in actions:
            immediate, successors = 0.0, []
            for probability, state in group:
                cost, successor = transition(state, action)
                immediate += probability * cost
                successors.append((probability, successor))
            future, policy = solve(tuple(successors), remaining - 1)
            alternatives.append((immediate + future, action, policy))
        value, action, policy = min(alternatives, key=lambda item: (item[0], item[1]))
        return value, {**policy, key: action}

    nonanticipative, policy = solve(worlds, horizon)
    if not math.isfinite(nonanticipative):
        raise ValueError("nonfinite aggregate fixture cost")
    replay_value = 0.0
    for probability, state in worlds:
        for remaining in range(horizon, 0, -1):
            action = policy[(remaining, model.information_key(state))]
            cost, state = transition(state, action)
            replay_value += probability * cost
    if not math.isclose(nonanticipative, replay_value, rel_tol=1e-12, abs_tol=1e-8):
        raise RuntimeError("nonanticipative policy replay failed")

    best_open_loop = (math.inf, ())
    clairvoyant = {state: math.inf for _, state in worlds}
    for sequence in itertools.product(actions, repeat=horizon):
        total = 0.0
        for probability, initial in worlds:
            cost, state = 0.0, initial
            for action in sequence:
                increment, state = transition(state, action)
                cost += increment
            clairvoyant[initial] = min(clairvoyant[initial], cost)
            total += probability * cost
        best_open_loop = min(best_open_loop, (total, sequence))
    clairvoyant_value = sum(p * clairvoyant[state] for p, state in worlds)
    tolerance = 1e-8 + abs(nonanticipative) * 1e-12
    if clairvoyant_value > nonanticipative + tolerance or nonanticipative > best_open_loop[0] + tolerance:
        raise RuntimeError("information/feasibility bound ordering failed")
    root_keys = {model.information_key(state) for _, state in worlds}
    return {
        "nonanticipative_cost": nonanticipative, "policy_replay_cost": replay_value,
        "clairvoyant_cost": clairvoyant_value, "best_open_loop_cost": best_open_loop[0],
        "best_open_loop_actions": list(best_open_loop[1]),
        "root_information_sets": len(root_keys),
        "root_actions": sorted({policy[(horizon, key)] for key in root_keys}),
        "policy_information_sets": len(policy), "unique_transition_queries": transition.cache_info().currsize,
        "action_count": len(actions), "world_count": len(worlds), "horizon": horizon,
        "scope": "exact only for the declared finite action grid, known finite-world distribution and horizon",
    }
