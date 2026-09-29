"""Public-state baseline planners for a finite, synthetic service-work fixture."""

from __future__ import annotations

from dataclasses import replace
import itertools
import math

from src.env.service_effort_mechanics import PublicServiceObservation, ServiceEffortConfig


def predict_step(observation, action, response, config: ServiceEffortConfig):
    """Predict homogeneous unit-work orders using public aggregates only.

    This is not a general clone of patient manufacturing. The fixture explicitly
    declares each initial work order to require one work unit.
    """
    if not isinstance(observation, PublicServiceObservation):
        raise TypeError("public observation required")
    n = len(config.site_hour_caps)
    action, response = tuple(action), tuple(response)
    if len(action) != n or len(response) != n:
        raise ValueError("site count mismatch")
    if any(not math.isfinite(x) or x < 0 or x > cap for x, cap in zip(action, config.site_hour_caps)) or sum(action) > config.shared_hour_budget + 1e-12:
        raise ValueError("invalid effort allocation")
    if any(not math.isfinite(x) or x <= 0 for x in response):
        raise ValueError("invalid estimated response")
    work = tuple(max(0.0, w - h * e) for w, h, e in zip(observation.remaining_work, observation.pending_hours[0], response))
    unfinished = tuple(max(0, math.ceil(w - 1e-12)) for w in work)
    completed = tuple(done + old - new for done, old, new in zip(observation.completed_jobs, observation.unfinished_jobs, unfinished))
    after = replace(observation, epoch=observation.epoch + 1, remaining_work=work,
                    unfinished_jobs=unfinished, completed_jobs=completed,
                    pending_hours=observation.pending_hours[1:] + (action,), previous_commitment=action)
    effort_cost = sum(config.hourly_cost * x + config.quadratic_cost * x * x for x in action)
    effort_cost += config.switching_cost * sum(abs(x - p) for x, p in zip(action, observation.previous_commitment))
    return after, effort_cost


def backlog_action(observation, actions, response):
    """Allocate towards the largest remaining workload in estimated hours."""
    workloads = tuple(w / e for w, e in zip(observation.remaining_work, response))
    if max(workloads) <= 1e-12:
        return "idle"
    if math.isclose(workloads[0], workloads[1], abs_tol=1e-12):
        return "balanced"
    return "left" if workloads[0] > workloads[1] else "right"


def plan_public_mpc(observation, response, config, actions, remaining_decisions,
                    holding_weights, terminal_work_cost):
    """Exhaustive certainty-equivalent MPC, bounded to five remaining decisions.

    Both fixed-model and online-identification variants use this exact routine.
    Only their estimated response differs; neither gets a latent scenario tape.
    """
    if type(remaining_decisions) is not int or not 1 <= remaining_decisions <= 5:
        raise ValueError("bounded MPC supports one to five decisions")
    best = (math.inf, ())
    queries = 0
    for sequence in itertools.product(sorted(actions), repeat=remaining_decisions):
        current, total = observation, 0.0
        for name in sequence:
            current, cost = predict_step(current, actions[name], response, config)
            queries += 1
            total += cost + sum(w * c for w, c in zip(current.remaining_work, holding_weights))
        # Pay for and execute every matured commitment before terminal service.
        for _ in range(config.commitment_lead_steps):
            current, cost = predict_step(current, (0.0,) * len(response), response, config)
            queries += 1
            total += cost + sum(w * c for w, c in zip(current.remaining_work, holding_weights))
        total += terminal_work_cost * sum(current.remaining_work)
        if not math.isfinite(total):
            raise ValueError("nonfinite planning objective")
        best = min(best, (total, sequence))
    return best[1][0], best[0], queries
