"""Hypothetical interval-completion support service, isolated from patient models."""

from dataclasses import dataclass, replace
import math

from src.env.service_effort_mechanics import ServiceEffortConfig


def completion_probability(rate, hours):
    if (isinstance(rate, bool) or isinstance(hours, bool) or
            not math.isfinite(rate) or rate <= 0 or not math.isfinite(hours) or hours < 0 or
            not math.isfinite(rate * hours)):
        raise ValueError("finite positive rate and nonnegative exposure required")
    return -math.expm1(-rate * hours)


@dataclass(frozen=True)
class CompletionJob:
    name: str
    site: int
    release: int
    downstream_steps: int


@dataclass(frozen=True)
class CompletionConfig:
    effort: ServiceEffortConfig
    jobs: tuple[CompletionJob, ...]
    downstream_slots: int
    holding_cost: float
    max_closure_steps: int

    def __post_init__(self):
        if len(self.effort.site_hour_caps) != 2 or self.effort.commitment_lead_steps != 1:
            raise ValueError("two sites and a one-interval request lead required")
        if not self.jobs or len({j.name for j in self.jobs}) != len(self.jobs):
            raise ValueError("nonempty unique job identities required")
        for job in self.jobs:
            if not isinstance(job.name, str) or not job.name or type(job.site) is not int or job.site not in (0, 1):
                raise ValueError("invalid job identity/site")
            if type(job.release) is not int or job.release < 0:
                raise ValueError("nonnegative integer release required")
            if type(job.downstream_steps) is not int or job.downstream_steps < 1:
                raise ValueError("positive fixed downstream duration required")
        if type(self.downstream_slots) is not int or self.downstream_slots < 1:
            raise ValueError("positive downstream slot count required")
        if type(self.max_closure_steps) is not int or self.max_closure_steps < 1:
            raise ValueError("positive bounded closure limit required")
        if not math.isfinite(self.holding_cost) or self.holding_cost < 0:
            raise ValueError("finite nonnegative holding charge required")


@dataclass(frozen=True)
class CompletionObservation:
    epoch: int
    stages: tuple[str, ...]
    ready_epochs: tuple[int, ...]
    downstream_remaining: tuple[int, ...]
    pending_hours: tuple[float, float]
    previous_request: tuple[float, float]


@dataclass(frozen=True)
class CompletionReceipt:
    epoch: int
    known_at: int
    requested_hours: tuple[float, float]
    applied_hours: tuple[float, float]
    eligible_tasks: tuple
    exposure_hours: tuple[float, float]
    completed: tuple[bool, bool]
    observation_kind: tuple[str, str]
    labor_cost: float
    switching_cost: float


def initial_observation(config):
    return CompletionObservation(
        0, tuple("support" if j.release == 0 else "scheduled" for j in config.jobs),
        (-1,) * len(config.jobs), (0,) * len(config.jobs), (0.0, 0.0), (0.0, 0.0))


def validate_action(action, config):
    action = tuple(action)
    if len(action) != 2 or any(isinstance(x, bool) or not math.isfinite(x) or not 0 <= x <= cap
                               for x, cap in zip(action, config.effort.site_hour_caps)):
        raise ValueError("invalid site effort request")
    if sum(action) > config.effort.shared_hour_budget + 1e-12:
        raise ValueError("purchasing budget exceeded")
    return action


def validate_observation(observation, config):
    if not isinstance(observation, CompletionObservation):
        raise TypeError("completion-only observation required")
    if type(observation.epoch) is not int or observation.epoch < 0:
        raise ValueError("invalid epoch")
    fields = (observation.stages, observation.ready_epochs, observation.downstream_remaining)
    if any(len(values) != len(config.jobs) for values in fields):
        raise ValueError("job state shape mismatch")
    for job, stage, ready, remaining in zip(config.jobs, *fields):
        if stage not in ("scheduled", "support", "waiting", "active", "done"):
            raise ValueError("invalid job stage")
        if (job.release > observation.epoch) != (stage == "scheduled"):
            raise ValueError("release/stage mismatch")
        if type(ready) is not int or ((ready == -1) != (stage in ("support", "scheduled"))):
            raise ValueError("invalid ready epoch")
        if stage not in ("support", "scheduled") and not job.release < ready <= observation.epoch:
            raise ValueError("future or invalid readiness event")
        if (type(remaining) is not int or not 0 <= remaining <= job.downstream_steps or
                (remaining > 0) != (stage == "active")):
            raise ValueError("downstream state mismatch")
    if observation.stages.count("active") > config.downstream_slots:
        raise ValueError("downstream occupancy exceeds capacity")
    validate_action(observation.pending_hours, config)
    validate_action(observation.previous_request, config)
    if observation.pending_hours != observation.previous_request:
        raise ValueError("one-interval commitment state mismatch")


def advance(observation, action, response, uniforms, config):
    """Evaluator-side transition; response and uniforms never enter public outputs."""
    validate_observation(observation, config)
    action = validate_action(action, config)
    if len(response) != 2 or len(uniforms) != 2:
        raise ValueError("private tape dimensions must match sites")
    for rate, u in zip(response, uniforms):
        completion_probability(rate, 1)
        if isinstance(u, bool) or not math.isfinite(u) or not 0 <= u < 1:
            raise ValueError("uniform outside [0,1)")
    t, effort = observation.epoch, config.effort
    stages, ready = list(observation.stages), list(observation.ready_epochs)
    remaining = list(observation.downstream_remaining)
    active = [i for i, stage in enumerate(stages) if stage == "active"]
    waiting = sorted((i for i, stage in enumerate(stages) if stage == "waiting"),
                     key=lambda i: (ready[i], config.jobs[i].release, config.jobs[i].name))
    started = waiting[:config.downstream_slots - len(active)]
    finished = []
    for i in started:
        stages[i], remaining[i] = "active", config.jobs[i].downstream_steps
    for i in active + started:
        remaining[i] -= 1
        if remaining[i] == 0:
            stages[i] = "done"
            finished.append(config.jobs[i].name)
    tasks, exposed, events, kinds = [], [], [], []
    for site, hours in enumerate(observation.pending_hours):
        eligible = sorted((i for i, job in enumerate(config.jobs) if job.site == site and stages[i] == "support"),
                          key=lambda i: (config.jobs[i].release, config.jobs[i].name))
        head = eligible[0] if eligible else None
        exposure = hours if head is not None else 0.0
        event = exposure > 0 and uniforms[site] < completion_probability(response[site], exposure)
        if event:
            stages[head], ready[head] = "waiting", t + 1
        tasks.append(config.jobs[head].name if head is not None else None)
        exposed.append(exposure)
        events.append(bool(event))
        kinds.append("idle" if head is None else ("no_effort" if hours == 0 else
                                                  ("completed" if event else "right_censored")))
    # Billing uses purchased availability, including idle or subsequently unused time.
    labor = sum(effort.hourly_cost * x + effort.quadratic_cost * x * x for x in action)
    switching = effort.switching_cost * sum(abs(x - old) for x, old in zip(action, observation.previous_request))
    holding = config.holding_cost * sum(j.release <= t and stage != "done" for j, stage in zip(config.jobs, stages))
    if not math.isfinite(labor + switching + holding):
        raise ValueError("nonfinite accounting cost")
    released = []
    for i, job in enumerate(config.jobs):
        if stages[i] == "scheduled" and job.release == t + 1:
            stages[i] = "support"
            released.append(job.name)
    after = replace(observation, epoch=t + 1, stages=tuple(stages), ready_epochs=tuple(ready),
                    downstream_remaining=tuple(remaining), pending_hours=action, previous_request=action)
    validate_observation(after, config)
    receipt = CompletionReceipt(t, t + 1, action, observation.pending_hours, tuple(tasks),
                                tuple(exposed), tuple(events), tuple(kinds), labor, switching)
    return after, receipt, {"holding": holding, "labor": labor, "switching": switching,
                            "total": holding + labor + switching,
                            "downstream_started": [config.jobs[i].name for i in started],
                            "downstream_finished": finished, "released": released}


def closed(observation):
    return all(stage == "done" for stage in observation.stages) and not any(observation.pending_hours) and not any(observation.previous_request)


def booked_backlog_action(observation, config):
    """Fixed public availability rule, not an optimized or adaptive-rate policy."""
    fraction = min(config.effort.shared_hour_budget / 2, *config.effort.site_hour_caps)
    return tuple(fraction if any(j.site == site and (stage == "support" or j.release == observation.epoch + 1)
                                for j, stage in zip(config.jobs, observation.stages)) else 0.0
                 for site in (0, 1))


def unresolved_ledger(observation, config):
    return {"settled": closed(observation),
            "unfinished_jobs": [j.name for j, stage in zip(config.jobs, observation.stages) if stage != "done"],
            "pending_prepaid_hours": observation.pending_hours, "previous_request": observation.previous_request}
