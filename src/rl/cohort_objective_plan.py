"""Pure arithmetic/stream adapter for a prospective cohort-objective package.

Reuse the prior six-controller dimensions and nonrefundable budget interface,
not its scientific permission, streams or consumed attempt. No runtime is built.
"""

import copy

from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import validate_budget_plan
from src.rl.time_baseline_plan import time_baseline_budget_plan, time_baseline_stream_manifest


ROLE_MAP = {"current_ppo": "window_ppo", "time_baseline_ppo": "cohort_ppo"}


def _name(value):
    parts = value.split("/")
    return "/".join(ROLE_MAP.get(part, part) for part in parts)


def cohort_budget_plan(base, proposal):
    """Account every prefix/tail/clone call without permitting an experiment."""
    if (proposal["schema"] != "cohort-objective-proposal-v1"
            or proposal["scientific_execution_authorized"] is not False
            or proposal["base_proposal_content_sha256"] != digest(base)
            or proposal["blocks"] != base["blocks"]
            or proposal["training_arms"] != [_name(x) for x in base["training_arms"]]
            or proposal["evaluation_controllers"] != [_name(x) for x in base["evaluation_controllers"]]):
        raise ValueError("explicit unapproved cohort scope and unchanged base dimensions required")
    prefix = proposal["enrollment_steps"]
    patient, tail = proposal["patient_resolution_steps"], proposal["accounting_steps"]
    clone = proposal["tail_clone_steps_per_preflight_controller"]
    if (type(prefix) is not int or prefix != 52
            or any(type(x) is not int or x < 1 for x in (patient, tail, clone))
            or tail < patient or clone > tail):
        raise ValueError("bounded fixed positive follow-up and clone windows required")
    old = time_baseline_budget_plan(base)
    result = copy.deepcopy(old)
    result["draft_sha256"] = digest(proposal)
    result["phases"], result["sections"] = {}, {}
    expected_rows = []
    b, c = len(base["blocks"]), len(base["evaluation_controllers"])
    dimensions = {
        "episodes_per_training_arm_block": "episodes_per_training_arm_block",
        "episodes_per_rollout": "episodes_per_rollout",
        "rollouts_per_arm_block": "rollouts_per_training_arm_block",
        "epochs_per_rollout": "epochs_per_rollout",
        "minibatch_sizes": "minibatch_sizes",
        "test_worlds_per_block": "evaluation_worlds_per_block",
        "preflight_worlds_per_block": "preflight_worlds_per_block",
        "initial_historical_loads": "initial_historical_loads",
        "prefix_clone_steps_per_preflight_controller": "clone_steps_per_preflight_controller",
    }
    if any(proposal[key] != base[other] for key, other in dimensions.items()):
        raise ValueError("training, sampling or load dimensions changed")
    if (proposal["prefix_parity_environment_builds"] != b
            or proposal["prefix_parity_steps"] != b * prefix
            or proposal["clone_instances"] != 2 * b * c
            or proposal["learned_artifacts_sealed_before_tests"] != b * (c - 2)
            or proposal["initialization_optimizer_calls"] != 0
            or proposal["qualification_optimizer_calls"] != 0):
        raise ValueError("complete preflight, initialization and model inventory required")
    for row in base["phase_budgets"]:
        target = copy.deepcopy(row)
        target["id"] = _name(row["id"])
        target["trajectory"] += row["episodes"] * tail
        if row["id"] == "same_start_preflight":
            target["clone"] += b * c * clone + b * prefix
        expected_rows.append(target)
        result["phases"][target["id"]] = {k: target[k] for k in ("trajectory", "clone", "actor", "critic", "seconds")}
    if proposal["phase_budgets"] != expected_rows:
        raise ValueError("phase arithmetic differs from identical prefix acquisition plus fixed tails")
    for name, original in old["sections"].items():
        row = copy.deepcopy(original)
        row["phase"] = _name(row["phase"])
        episodes, rem = divmod(row["trajectory"], prefix)
        if rem:
            raise ValueError("nonintegral prefix episode inventory")
        row["trajectory"] += episodes * tail
        if name == "same_start_preflight":
            row["clone"] += b * c * clone + b * prefix
        result["sections"][_name(name)] = row
    totals = {k: sum(r[k] for r in expected_rows) for k in
              ("episodes", "trajectory", "clone", "actor", "critic", "seconds")}
    totals["phase_seconds"] = totals.pop("seconds")
    totals.update(environment=totals["trajectory"] + totals["clone"],
                  optimizer=totals["actor"] + totals["critic"],
                  environment_builds=base["totals"]["environment_builds"] + b,
                  global_seconds=base["totals"]["global_seconds"])
    if proposal["totals"] != totals:
        raise ValueError("complete cohort budget totals differ")
    for owner, original in base["per_owner"].items():
        expected = copy.deepcopy(original)
        expected.update(prefix=original["trajectory"], tail=original["episodes"] * tail)
        expected["trajectory"] += expected["tail"]
        if proposal["per_owner"][owner] != expected:
            raise ValueError("nontransferable per-owner caps differ")
    result["limits"] = {k: totals[k] for k in ("trajectory", "clone", "actor", "critic")}
    result["limits"]["seconds"] = totals["global_seconds"]
    validate_budget_plan(result)
    return result


def cohort_stream_manifest(base, proposal):
    """Fresh prospective purpose namespace; not a historical freshness claim."""
    cohort_budget_plan(base, proposal)
    namespace = proposal["rng_namespace"]
    if not isinstance(namespace, str) or not namespace.strip() or namespace == base["rng"]["namespace"]:
        raise ValueError("new explicit prospective namespace required")
    source = copy.deepcopy(base)
    source["rng"]["namespace"] = namespace
    manifest = time_baseline_stream_manifest(source)
    manifest.update(format="cohort-objective-streams-v1", proposal_sha256=digest(proposal),
                    followup_new_rng_instances=0, followup_uses_inherited_episode_state=True)
    for block in manifest["neural_bindings"].values():
        for split in tuple(block):
            block[split] = {_name(role): row for role, row in block[split].items()}
    return manifest
