"""Continuation-only accounting derived from the unchanged S1 design.

Completed work is imported as evidence, never represented as refunded spend.
No model, environment or numerical fitting is invoked by this module.
"""

import copy

from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import dynamic_budget_plan, validate_budget_plan


REMOVED = {"prototype_preflight", "initialization_collection", "initialization_fit", "qualification"}


def recovery_config(original):
    config = copy.deepcopy(original)
    config["phase_budgets"] = [row for row in config["phase_budgets"] if row["id"] not in REMOVED]
    totals = config["totals"]
    rows = config["phase_budgets"]
    for destination, source in (("main_trajectory_steps", "trajectory_steps"),
                                ("mandatory_restore_clone_steps", "mandatory_clone_steps"),
                                ("main_actor_adam_calls", "actor_adam_calls"),
                                ("main_critic_adam_calls", "critic_adam_calls"),
                                ("main_phase_seconds", "seconds")):
        totals[destination] = sum(row[source] for row in rows)
    totals["main_environment_steps"] = totals["main_trajectory_steps"] + totals["mandatory_restore_clone_steps"]
    totals["main_optimizer_calls"] = totals["main_actor_adam_calls"] + totals["main_critic_adam_calls"]
    horizon = original["objective"]["horizon"]
    if totals["main_trajectory_steps"] % horizon:
        raise ValueError("whole prescribed episodes required")
    totals["main_full_episodes"] = totals["main_trajectory_steps"] // horizon
    totals["fresh_episode_builds"] = totals["main_full_episodes"]
    totals["backend_layout_builds"] = len(config["blocks"])
    totals["build_env_calls_max"] = totals["fresh_episode_builds"] + totals["backend_layout_builds"]
    totals["combined_environment_steps_max"] = totals["main_environment_steps"]
    totals["combined_optimizer_calls_max"] = totals["main_optimizer_calls"]
    totals["mandatory_restored_session_clones"] = len(config["blocks"]) * len(config["final_controllers"])
    for name in ("optional_environment_clones_max", "optional_environment_steps_max", "optional_seconds_max"):
        totals[name] = 0
    totals.pop("old_incomplete_continuation_plus_evaluation_steps", None)
    totals["global_elapsed_seconds"] = (totals["main_phase_seconds"]
        + original["totals"]["global_elapsed_seconds"] - original["totals"]["main_phase_seconds"])
    totals["unique_world_start_allocations"] = len(config["blocks"]) * (
        1 + config["continuation"]["episodes_per_arm_per_block"]
        + config["evaluation"]["fresh_paired_worlds_per_block"])
    config["recovery"] = {"format": "dynamic-continuation-only-v1",
        "original_config_sha256": digest(original), "excluded_phases": sorted(REMOVED),
        "reuse_saved_initializers": True, "reuse_completed_qualification": True,
        "old_attempt_remains_terminal": True, "automatic_retry": False}
    return config


def recovery_budget_plan(original):
    full = dynamic_budget_plan(original)
    config = recovery_config(original)
    plan = copy.deepcopy(full)
    plan["phases"] = {k: v for k, v in full["phases"].items() if k not in REMOVED}
    plan["sections"] = {k: v for k, v in full["sections"].items() if v["phase"] not in REMOVED}
    plan["limits"] = {key: sum(row[key] for row in plan["phases"].values())
                      for key in ("trajectory", "clone", "actor", "critic", "seconds")}
    plan["limits"]["seconds"] = config["totals"]["global_elapsed_seconds"]
    plan["draft_sha256"] = digest(original)
    validate_budget_plan(plan)
    return plan
