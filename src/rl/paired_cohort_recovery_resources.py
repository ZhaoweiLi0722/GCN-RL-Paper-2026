"""Exact remaining-work budgets; existing science and consumed debits stay fixed."""

from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import validate_budget_plan
from src.rl.paired_cohort_execution import _scope


ROLES = ["own_frozen", "paired_cost", "bc_continue", "saved_cohort_ppo", "r4", "full_mdl2"]
REMAINING = {"60": dict(branches=11, environment_calls=297, seconds=600),
             "61": dict(branches=140, environment_calls=6020, seconds=2400),
             "62": dict(branches=134, environment_calls=5730, seconds=2400)}
PHASE_SECONDS = dict(binding_import=300, remaining_branches=5400, paired_actor=1200,
    bc_actor=600, seal=120, evaluation=6480, raw_verification=600, archive=1200, closure=600)


def validate_proposal(config, proposal):
    _scope(config)
    if config["evaluation_controllers"] != ROLES or config["training_arms"] != ["paired_cost", "bc_continue"]:
        raise ValueError("original six roles and two training arms required")
    expected = dict(format="paired-cohort-remaining-work-proposal-v1",
        scientific_execution_authorized=False, preserved_failed_execution="3da1a09",
        reuse_preflight_pairs=3, reuse_reference_cohorts=12, reuse_context_states=36,
        reuse_complete_branches=117, new_reference_cohorts_or_preflight_episodes=0,
        remaining_branches=285, interrupted_branch_recomputed_with_same_seed=1,
        interrupted_branch_new_calls=27, old_interrupted_charge_preserved=1,
        remaining_by_block=REMAINING, new_branch_environment_calls=12047,
        new_actor_fit_jobs=6, actor_updates_per_fit=128, new_actor_updates=768,
        new_critic_updates=0, evaluation_cohorts=216, evaluation_environment_calls=13608,
        new_environment_cap=25655, cumulative_charged_environment_calls_including_failure=31901,
        phase_seconds=PHASE_SECONDS, evaluation_controller_block_seconds=360,
        global_seconds=18000, attempts=1, automatic_retry=False, automatic_followon=False,
        reward_seed_sample_architecture_support_change=False, external_or_remote_actions=False)
    if digest({key: proposal.get(key) for key in expected}) != digest(expected):
        raise ValueError("exact approved remaining-work package required; no refunded or expanded work")


def budget_plan(config, recovery_proposal):
    """Translate original config plus the additive proposal, never mutate either."""
    validate_proposal(config, recovery_proposal)
    phases, sections = {}, {}
    names = {"binding_import": "binding", "remaining_branches": "paired_branches"}
    for label, seconds in PHASE_SECONDS.items():
        phase = names.get(label, label)
        row = dict(trajectory=0, clone=0, actor=0, critic=0, seconds=seconds)
        phases[phase] = row
        if phase == "paired_branches":
            row["clone"] = 12047
            for block in config["blocks"]:
                cap = recovery_proposal["remaining_by_block"][str(block)]
                sections[f"{phase}/block{block}"] = dict(phase=phase, trajectory=0,
                    clone=cap["environment_calls"], actor=0, critic=0, seconds=cap["seconds"])
        elif phase in ("paired_actor", "bc_actor"):
            row["actor"] = 384
            for block in config["blocks"]:
                sections[f"{phase}/block{block}"] = dict(phase=phase, trajectory=0, clone=0,
                    actor=128, critic=0, seconds=seconds // 3)
        elif phase == "evaluation":
            row["trajectory"] = 13608
            for block in config["blocks"]:
                for role in ROLES:
                    sections[f"evaluation/block{block}/{role}"] = dict(phase=phase,
                        trajectory=756, clone=0, actor=0, critic=0, seconds=360)
        else:
            sections[phase] = dict(row, phase=phase)
    result = dict(format="dynamic-candidate-budget-plan-v1",
        draft_sha256=digest(dict(config=config, recovery_proposal=recovery_proposal)),
        limits=dict(trajectory=13608, clone=12047, actor=768, critic=0, seconds=18000),
        phases=phases, sections=sections)
    validate_budget_plan(result)
    return result


def operation_limits(config, recovery_proposal):
    """Write-ahead native operation contract, including non-stepped imports."""
    result = {}
    for job, row in budget_plan(config, recovery_proposal)["sections"].items():
        phase = row["phase"]
        if phase == "binding":
            caps = dict(input_hash_verification=12, reference_and_layout_build=3,
                reference_checkpoint_load=3, layout_environment_build=3, checkpoint_load=6,
                template_build=3, context_load=36, branch_import=117)
        elif phase == "paired_branches":
            block = job.split("block")[1]
            cap = recovery_proposal["remaining_by_block"][block]
            caps = dict(conditional_branch_clone=cap["branches"], branch_step=cap["environment_calls"])
        elif phase in ("paired_actor", "bc_actor"):
            caps = dict(actor_fork=1, actor_update=128)
        elif phase == "evaluation":
            caps = dict(episode_build=12, episode_dispatch=12, episode_step=756)
        else:
            caps = {}
        result[job] = caps
    return result
