"""Pure finite rollout accounting; does not authorize or execute any work."""

from dataclasses import dataclass
from itertools import product


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Branch:
    block: int
    cohort: int
    after_prefix_steps: int
    replication: int
    candidate: int
    prefix_calls: int
    tail_calls: int

    @property
    def environment_calls(self):
        return self.prefix_calls + self.tail_calls


def context_keys(config):
    blocks = config["blocks"]
    times = config["context_after_prefix_steps"]
    prefix = _positive(config["enrollment_steps"], "enrollment_steps")
    if not isinstance(blocks, list) or not blocks or any(type(b) is not int for b in blocks):
        raise ValueError("blocks must be explicit integers")
    if len(set(blocks)) != len(blocks):
        raise ValueError("duplicate block")
    if (not isinstance(times, list) or not times or times != sorted(set(times))
            or any(type(t) is not int or not 0 <= t < prefix for t in times)):
        raise ValueError("context times must be unique ordered prefix boundaries")
    cohorts = _positive(config["context_cohorts_per_block"], "context_cohorts_per_block")
    return tuple(product(blocks, range(cohorts), times))


def branch_plan(config, candidate_counts):
    """One action then fixed reference continuation, including the full tail.

    Candidate counts must come from canonical request classes, not six padded
    aliases. Lower support consumes fewer calls; the remainder is never reused.
    All candidates at a context/replication share the same future-RNG seed ID.
    That seed assignment belongs to the executor, not this arithmetic module.
    """
    keys = context_keys(config)
    if set(candidate_counts) != set(keys):
        raise ValueError("candidate counts must cover each prespecified context exactly")
    cap = _positive(config["max_original_requests"], "max_original_requests")
    reps = _positive(config["future_replications"], "future_replications")
    prefix = config["enrollment_steps"]
    endpoint = _positive(config["economic_endpoint"], "economic_endpoint")
    if endpoint <= prefix:
        raise ValueError("economic endpoint must include a positive fixed tail")
    result = []
    for block, cohort, t in keys:
        count = _positive(candidate_counts[block, cohort, t], "candidate count")
        if count > cap:
            raise ValueError("candidate support exceeds the fixed cap")
        for replication, candidate in product(range(reps), range(count)):
            result.append(Branch(block, cohort, t, replication, candidate,
                                 prefix - t, endpoint - prefix))
    return tuple(result)


def maximum_accounting(config):
    keys = context_keys(config)
    branches = branch_plan(config, {key: config["max_original_requests"] for key in keys})
    blocks = len(config["blocks"])
    endpoint = config["economic_endpoint"]
    contexts = blocks * config["context_cohorts_per_block"]
    roles = config["evaluation_controllers"]
    if not isinstance(roles, list) or not roles or any(type(r) is not str or not r for r in roles):
        raise ValueError("evaluation controllers must be explicit names")
    if len(roles) != len(set(roles)):
        raise ValueError("duplicate evaluation controller")
    evaluations = blocks * len(roles) * _positive(config["test_worlds_per_block"], "test_worlds_per_block")
    updates = blocks * 2 * _positive(config["actor_updates_per_arm_block"], "actor_updates_per_arm_block")
    branch_calls = sum(branch.environment_calls for branch in branches)
    phase_seconds = sum(_positive(p["seconds"], "phase seconds") for p in config["phase_budgets"])
    global_seconds = _positive(config["global_seconds"], "global_seconds")
    if phase_seconds > global_seconds:
        raise ValueError("phase time caps exceed the global cap")
    return dict(contexts=len(keys), context_cohorts=contexts,
                branch_instances=len(branches), branch_calls=branch_calls,
                preflight_original_cohorts=blocks, preflight_clone_instances=blocks,
                preflight_calls=blocks * 2 * endpoint,
                context_calls=contexts * endpoint,
                evaluation_cohorts=evaluations, evaluation_calls=evaluations * endpoint,
                environment_calls=branch_calls + (contexts + evaluations + blocks * 2) * endpoint,
                optimizer_calls=updates, actor_calls=updates, critic_calls=0,
                clone_instances=len(branches) + blocks,
                fresh_environment_builds=blocks * 2 + contexts + evaluations,
                phase_seconds=phase_seconds, global_seconds=global_seconds)
