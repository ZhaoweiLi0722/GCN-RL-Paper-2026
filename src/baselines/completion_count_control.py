"""Fixed event-count rule; no remaining work, response model or fitting."""

from dataclasses import dataclass


STAGES = frozenset(("scheduled", "support", "waiting", "active", "done"))


@dataclass(frozen=True)
class CompletionCounts:
    epoch: int
    support_by_site: tuple[int, int]

    def __post_init__(self):
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError("nonnegative integer epoch required")
        if (type(self.support_by_site) is not tuple or len(self.support_by_site) != 2
                or any(type(n) is not int or n < 0 for n in self.support_by_site)):
            raise ValueError("two nonnegative integer support counts required")


def count_observation(epoch, stages, job_sites):
    """Project only recorded stages and the public task-to-site mapping."""
    if len(stages) != len(job_sites) or not stages:
        raise ValueError("matching nonempty stage and site sequences required")
    if any(stage not in STAGES for stage in stages):
        raise ValueError("unknown stage")
    if any(type(site) is not int or site not in (0, 1) for site in job_sites):
        raise ValueError("two-site fixture required")
    return CompletionCounts(epoch, tuple(sum(stage == "support" and site == i
                                           for stage, site in zip(stages, job_sites))
                                         for i in (0, 1)))


def completion_count_choice(observation):
    if not isinstance(observation, CompletionCounts):
        raise TypeError("CompletionCounts required, not full simulator state")
    left, right = observation.support_by_site
    if left == right:
        return "idle" if left == 0 else "balanced"
    return "left" if left > right else "right"
