"""Equal-public-information comparators for the synthetic queue mechanism."""

from functools import lru_cache
import math

from src.env.service_queue_network import QueueObservation, advance, available_work, close_episode


def backlog_choice(observation, response, config):
    work = tuple(w / r for w, r in zip(available_work(observation, config), response))
    if max(work) <= 1e-12:
        return "idle"
    if math.isclose(work[0], work[1], abs_tol=1e-12):
        return "balanced"
    return "left" if work[0] > work[1] else "right"


def plan_queue_mpc(observation, response, config, actions, decisions):
    """Exact finite-grid planning in a constant-response public model.

    Both MPC arms use the same horizon, task schedule, dynamics and continuation.
    Only the response estimate differs. The true change time/world is not input.
    """
    if not isinstance(observation, QueueObservation):
        raise TypeError("public observation required")
    if type(decisions) is not int or not 1 <= decisions <= 5:
        raise ValueError("bounded to one through five decisions")
    response = tuple(response)
    queries = 0

    def prediction_step(state, action):
        nonlocal queries
        queries += 1
        after, _, cost = advance(state, action, response, config)
        return after, cost["total"]

    @lru_cache(maxsize=None)
    def continuation(state):
        nonlocal queries
        # Closure executes this many deterministic predicted transitions.
        value, final, _ = close_episode(state, config, lambda epoch: response)
        queries += final.epoch - state.epoch
        return value

    @lru_cache(maxsize=None)
    def solve(state, left):
        if left == 0:
            return continuation(state), ""
        choices = []
        for name in sorted(actions):
            after, immediate = prediction_step(state, tuple(actions[name]))
            value, _ = solve(after, left - 1)
            choices.append((immediate + value, name))
        return min(choices)

    value, name = solve(observation, decisions)
    return name, value, queries
