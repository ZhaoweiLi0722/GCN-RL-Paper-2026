"""Synthetic support-work queues with indivisible, fixed-duration downstream jobs.

This module is not connected to patient manufacturing or release decisions.
All booked releases are public; response coefficients are supplied privately by
the evaluator. Controllers may predict with their own response estimates.
"""

from dataclasses import dataclass, replace
import math

from src.env.service_effort_mechanics import PublicServiceReceipt, ServiceEffortConfig


@dataclass(frozen=True)
class BookedJob:
    name: str
    site: int
    release: int
    work: float
    downstream_steps: int


@dataclass(frozen=True)
class QueueConfig:
    effort: ServiceEffortConfig
    jobs: tuple[BookedJob, ...]
    downstream_slots: int
    holding_cost: float
    max_settlement_steps: int

    def __post_init__(self):
        if len(self.effort.site_hour_caps) != 2 or self.effort.commitment_lead_steps != 1:
            raise ValueError("bounded queue fixture requires two sites and a one-step lead")
        if not self.jobs or len({j.name for j in self.jobs}) != len(self.jobs):
            raise ValueError("nonempty uniquely named jobs required")
        for job in self.jobs:
            if type(job.site) is not int or job.site not in (0, 1):
                raise ValueError("invalid job site")
            if type(job.release) is not int or job.release < 0:
                raise ValueError("invalid public release")
            if not math.isfinite(job.work) or job.work <= 0:
                raise ValueError("invalid support work")
            if type(job.downstream_steps) is not int or job.downstream_steps < 1:
                raise ValueError("invalid mandatory downstream duration")
        if type(self.downstream_slots) is not int or self.downstream_slots < 1:
            raise ValueError("positive downstream slots required")
        if not math.isfinite(self.holding_cost) or self.holding_cost < 0:
            raise ValueError("invalid holding charge")
        if type(self.max_settlement_steps) is not int or self.max_settlement_steps < 1:
            raise ValueError("invalid closure bound")


@dataclass(frozen=True)
class QueueObservation:
    epoch: int
    stages: tuple[str, ...]
    remaining_work: tuple[float, ...]
    ready_epochs: tuple[int, ...]
    downstream_remaining: tuple[int, ...]
    pending_hours: tuple[tuple[float, ...], ...]
    previous_commitment: tuple[float, ...]


def initial_observation(config):
    return QueueObservation(
        0, tuple("support" if j.release == 0 else "scheduled" for j in config.jobs),
        tuple(j.work for j in config.jobs), (-1,) * len(config.jobs),
        (0,) * len(config.jobs), ((0.0, 0.0),), (0.0, 0.0),
    )


def available_work(observation, config):
    return tuple(sum(w for j, stage, w in zip(config.jobs, observation.stages, observation.remaining_work)
                     if j.site == site and stage == "support") for site in (0, 1))


def advance(observation, action, response, config):
    """Immutable public-state transition given an explicit assumed/actual response.

    At interval start, ready jobs enter vacant downstream slots. That interval
    processes them while matured effort serves FIFO support work. Newly ready
    jobs can start downstream only next interval. Next booked releases become
    visible in the next observation, and are not charged for time not elapsed.
    """
    if not isinstance(observation, QueueObservation):
        raise TypeError("public QueueObservation required")
    action, response = tuple(action), tuple(response)
    effort = config.effort
    if len(action) != 2 or any(not math.isfinite(x) or not 0 <= x <= cap
                              for x, cap in zip(action, effort.site_hour_caps)):
        raise ValueError("invalid effort")
    if sum(action) > effort.shared_hour_budget + 1e-12:
        raise ValueError("shared budget exceeded")
    if len(response) != 2 or any(not math.isfinite(x) or x <= 0 for x in response):
        raise ValueError("invalid response")
    t = observation.epoch
    stages = list(observation.stages)
    work = list(observation.remaining_work)
    ready = list(observation.ready_epochs)
    downstream = list(observation.downstream_remaining)
    active = [i for i, stage in enumerate(stages) if stage == "active"]
    waiting = sorted((i for i, stage in enumerate(stages) if stage == "waiting"),
                     key=lambda i: (ready[i], config.jobs[i].release, config.jobs[i].name))
    started = waiting[:config.downstream_slots - len(active)]
    for i in started:
        stages[i], downstream[i] = "active", config.jobs[i].downstream_steps
    finished = []
    for i in active + started:
        downstream[i] -= 1
        if downstream[i] == 0:
            stages[i] = "done"
            finished.append(i)
    applied = observation.pending_hours[0]
    available = available_work(observation, config)
    delivered, completed, kinds = [], [], []
    for site in (0, 1):
        amount = min(available[site], applied[site] * response[site])
        left, count = amount, 0
        eligible = sorted((i for i, j in enumerate(config.jobs) if j.site == site and stages[i] == "support"),
                          key=lambda i: (config.jobs[i].release, config.jobs[i].name))
        for i in eligible:
            used = min(work[i], left)
            work[i] -= used
            left -= used
            if work[i] <= 1e-12:
                work[i], stages[i], ready[i] = 0.0, "waiting", t + 1
                count += 1
        delivered.append(amount)
        completed.append(count)
        kinds.append("no_effort" if applied[site] == 0 else (
            "backlog_limited" if amount >= available[site] - 1e-12 else "uncensored"))
    holding = config.holding_cost * sum(j.release <= t and stage != "done"
                                       for j, stage in zip(config.jobs, stages))
    released = []
    for i, job in enumerate(config.jobs):
        if stages[i] == "scheduled" and job.release == t + 1:
            stages[i] = "support"
            released.append(i)
    labor = sum(effort.hourly_cost * x + effort.quadratic_cost * x * x for x in action)
    change = effort.switching_cost * sum(abs(x - old) for x, old in zip(action, observation.previous_commitment))
    after = replace(observation, epoch=t + 1, stages=tuple(stages), remaining_work=tuple(work),
                    ready_epochs=tuple(ready), downstream_remaining=tuple(downstream),
                    pending_hours=(action,), previous_commitment=action)
    receipt = PublicServiceReceipt(t, action, applied, available, tuple(delivered), tuple(completed),
                                   tuple(kinds), labor, change)
    return after, receipt, {"holding": holding, "labor": labor, "switching": change,
                            "total": holding + labor + change,
                            "downstream_started": [config.jobs[i].name for i in started],
                            "downstream_finished": [config.jobs[i].name for i in finished],
                            "released": [config.jobs[i].name for i in released]}


def closed(observation):
    return (all(stage == "done" for stage in observation.stages)
            and not any(sum(row) for row in observation.pending_hours)
            and not any(observation.previous_commitment))


def fallback_action(observation, config):
    """Predeclared common continuation, not an optimized rescue policy."""
    return (0.5, 0.5) if any(available_work(observation, config)) else (0.0, 0.0)


def close_episode(observation, config, response_at, *, record=False):
    """First honor outstanding commitments, then process every booked job.

    All labor, switching, and holding costs remain charged. There is no salvage,
    free outsourcing, deletion, or bypass of the downstream workstation.
    """
    current, total, rows = observation, 0.0, []
    for offset in range(config.max_settlement_steps):
        if closed(current):
            return total, current, rows
        action = (0.0, 0.0) if offset < config.effort.commitment_lead_steps else fallback_action(current, config)
        after, receipt, cost = advance(current, action, response_at(current.epoch), config)
        total += cost["total"]
        if record:
            from dataclasses import asdict
            rows.append({"before": asdict(current), "after": asdict(after),
                         "receipt": asdict(receipt), "cost": cost})
        current = after
    if closed(current):
        return total, current, rows
    raise RuntimeError("bounded closure failed; no unpaid terminal truncation permitted")
