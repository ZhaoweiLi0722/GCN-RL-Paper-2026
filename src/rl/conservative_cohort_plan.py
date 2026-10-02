"""Pure accounting and serial schedule for the prospective two-round package."""

import copy

from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import validate_budget_plan
from src.rl.paired_cohort_resources import stream_manifest as original_streams


ROLES = ("own_frozen", "paired_cost", "bc_continue", "r4", "full_mdl2")


def validate_config(cfg):
    fixed = dict(schema="conservative-cohort-improvement-proposal-v1",
        scientific_execution_authorized=False, blocks=[60, 61, 62], rounds=2,
        context_cohorts_per_round_block=2, context_after_prefix_steps=[4, 20, 36],
        enrollment_steps=52, economic_endpoint=63, max_original_requests=6,
        future_replications=4, actor_updates_per_round_arm_block=64,
        training_arms=["paired_cost", "bc_continue"], evaluation_controllers=list(ROLES),
        test_worlds_per_block=12, cost_scale=1e9, kl_coefficient=.05,
        critic_updates=0, entropy_coefficient=0, attempts=1, automatic_retry=False,
        automatic_followon=False, global_seconds=28800)
    for key, expected in fixed.items():
        if type(cfg.get(key)) is not type(expected) or cfg[key] != expected:
            raise ValueError("fixed conservative scope differs: " + key)
    expected = dict(context_states=36, context_cohorts=12, branch_instances=864,
        branch_calls=37152, context_calls=756, preflight_calls=378,
        evaluation_cohorts=180, evaluation_calls=11340, environment_calls=49626,
        optimizer_calls=768, critic_calls=0, clone_instances=867, fresh_environment_builds=198)
    if cfg["maximum"] != expected:
        raise ValueError("complete resource arithmetic differs")
    if sum(cfg["phase_seconds"].values()) != cfg["global_seconds"]:
        raise ValueError("complete phase times must sum to the global cap")
    return expected


def schedule(cfg):
    validate_config(cfg)
    jobs = [dict(id="binding", phase="binding")]
    jobs += [dict(id=f"preflight/block{b}", phase="preflight", block=b) for b in cfg["blocks"]]
    for rnd in range(2):
        for block in cfg["blocks"]:
            for phase in ("contexts", "branches", "paired_actor", "bc_actor"):
                jobs.append(dict(id=f"{phase}/round{rnd}/block{block}", phase=phase, block=block, round=rnd))
    jobs.append(dict(id="seal", phase="seal"))
    for block in cfg["blocks"]:
        for role in ROLES:
            jobs.append(dict(id=f"evaluation/block{block}/{role}", phase="evaluation", block=block, role=role))
    jobs += [dict(id=p, phase=p) for p in ("raw_verification", "archive", "closure")]
    return jobs


def budget_plan(cfg):
    jobs = schedule(cfg)
    sections, phases = {}, {}
    counts = dict(binding=(0, 0, 0), preflight=(63, 63, 0), contexts=(126, 0, 0),
        branches=(0, 6192, 0), paired_actor=(0, 0, 64), bc_actor=(0, 0, 64),
        seal=(0, 0, 0), evaluation=(756, 0, 0), raw_verification=(0, 0, 0),
        archive=(0, 0, 0), closure=(0, 0, 0))
    for job in jobs:
        phase = job["phase"]
        trajectory, clone, actor = counts[phase]
        limits = dict(trajectory=trajectory, clone=clone, actor=actor, critic=0)
        sections[job["id"]] = dict(limits, phase=phase,
            seconds=cfg["per_owner_seconds"].get(phase, cfg["phase_seconds"][phase]))
        phases.setdefault(phase, dict(trajectory=0, clone=0, actor=0, critic=0, seconds=cfg["phase_seconds"][phase]))
        for k, v in limits.items():
            phases[phase][k] += v
    limits = {k: sum(v[k] for v in phases.values()) for k in ("trajectory", "clone", "actor", "critic")}
    limits["seconds"] = cfg["global_seconds"]
    result = dict(format="dynamic-candidate-budget-plan-v1", draft_sha256=digest(cfg),
                  limits=limits, phases=phases, sections=sections)
    validate_budget_plan(result)
    return result


def streams(cfg, layout_seeds):
    """Reuse typed, disjoint seed allocation; cohort IDs 0,1 then 2,3 encode rounds."""
    validate_config(cfg)
    bridge = copy.deepcopy(cfg)
    bridge["context_cohorts_per_block"] = 4
    bridge["streams"] = dict(preflight=3, context=12, conditional_future=144, test=36, neural=6, bootstrap=1)
    result = original_streams(bridge, layout_seeds)
    result["format"] = "conservative-cohort-streams-v1"
    return result


def cohort_ids(round_index):
    if type(round_index) is not int or round_index not in (0, 1):
        raise ValueError("only two prospective rounds permitted")
    return (2 * round_index, 2 * round_index + 1)
