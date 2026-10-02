"""Pure prospective config binding and explicit equal-weight model forks."""

import copy

from src.rl.candidate_imitation import ImitationSettings
from src.rl.cohort_objective_plan import cohort_budget_plan
from src.rl.cohort_ppo import CohortPPOKernel
from src.rl.dynamic_candidate_factory import ppo_settings
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.time_baseline_factory import time_baseline_config


def cohort_config(original, base, proposal):
    cohort_budget_plan(base, proposal)
    cfg = time_baseline_config(original, base)
    cfg.pop("time_baseline_proposal")
    cfg["cohort_proposal"] = copy.deepcopy(proposal)
    cfg["scientific_execution_authorized"] = False
    cfg["final_controllers"] = list(proposal["evaluation_controllers"])
    cfg["phase_budgets"] = copy.deepcopy(proposal["phase_budgets"])
    cfg["owner_budgets"] = copy.deepcopy(proposal["per_owner"])
    cfg["continuation"].update(environment_steps_total=288 * (proposal["enrollment_steps"] + proposal["accounting_steps"]))
    cfg["evaluation"].update(environment_steps=216 * (proposal["enrollment_steps"] + proposal["accounting_steps"]),
        primary="paired_raw_cohort_cost_cohort_ppo_minus_own_frozen",
        secondary_contrasts=["cohort_ppo_minus_window_ppo", "cohort_ppo_minus_bc_continue",
                            "cohort_ppo_minus_r4", "cohort_ppo_minus_full_mdl2", "own_frozen_minus_r4"])
    t = proposal["totals"]
    cfg["totals"].update(main_trajectory_steps=t["trajectory"], main_environment_steps=t["environment"],
        mandatory_restore_clone_steps=t["clone"] - proposal["prefix_parity_steps"],
        prefix_parity_steps=proposal["prefix_parity_steps"],
        prefix_parity_environment_builds=proposal["prefix_parity_environment_builds"],
        build_env_calls_max=t["environment_builds"],
        mandatory_restored_session_clones=proposal["clone_instances"])
    # Existing objective still describes raw prefix records. The sole changed
    # training estimand is explicit in each CohortPPOKernel's objective binding.
    cfg["cohort_objectives"] = dict(window_ppo="prefix_cost_only", cohort_ppo="prefix_plus_common_tail_cost",
        prefix_steps=proposal["enrollment_steps"], patient_resolution_steps=proposal["patient_resolution_steps"],
        accounting_steps=proposal["accounting_steps"], learned_tail_actions=False, new_cost_weights=False)
    return cfg


def fork_cohort_initializer(initializer, qualification, config, streams, block, *, split="training"):
    if (split not in ("training", "preflight") or block not in config["blocks"]
            or qualification.get("passed") is not True
            or qualification.get("kernel_sha256") != state_digest(initializer.state_dict())
            or initializer.steps != config["initialization"]["actor_adam_calls_per_block"]):
        raise ValueError("exact completed qualified initializer required; no refit/rescore")
    cont, opt, proposal = config["continuation"], config["optimizer"], config["cohort_proposal"]
    roles = proposal["evaluation_controllers"][:4]
    if roles != ["own_frozen", "window_ppo", "cohort_ppo", "bc_continue"]:
        raise ValueError("four prescribed learned owners required")
    seeds = [int(s) for s in streams["environment"][str(block)]["training"]]
    bindings = streams["neural_bindings"][str(block)][split]
    result = {}
    for role in roles:
        bound = bindings["window_ppo" if role == "own_frozen" else role]
        sample, shuffle = (int(streams["neural"][bound[k]]) for k in ("sampling_seed", "shuffle_seed"))
        common = dict(enabled=True, sampling_seed=sample, shuffle_seed=shuffle)
        if role == "own_frozen":
            owner = DynamicCandidatePPOKernel(initializer.policy, initializer.contract, ppo_settings(config),
                                             mode="frozen", **common)
        elif role == "bc_continue":
            owner = DynamicCandidateImitationKernel(initializer.policy, initializer.contract,
                ImitationSettings(opt["learning_rate"], opt["gradient_norm_cap_each_owner"], cont["batch_size"],
                    cont["bc_continue"]["actor_adam_calls_per_block"], cont["rollouts_per_arm_per_block"], "training"),
                **common)
        else:
            manifest = [dict(trajectory_id=f"training/block{block}/graph/{role}/episode{i:02d}",
                             environment_seed=seed) for i, seed in enumerate(seeds)]
            owner = CohortPPOKernel(initializer.policy, initializer.contract, ppo_settings(config),
                mode="online", objective="window" if role == "window_ppo" else "cohort",
                accounting_steps=proposal["accounting_steps"], training_manifest=manifest,
                training_source_id=streams["namespace"], episode_horizon=proposal["enrollment_steps"],
                episodes_per_rollout=cont["episodes_per_rollout"], **common)
        optimizers = [owner.optimizer] if role == "bc_continue" else owner.optimizers.values()
        if (owner.policy.snapshot_sha256() != initializer.policy.snapshot_sha256()
                or any(o is not None and o.state_dict()["state"] for o in optimizers)):
            raise ValueError("fork changed weights or inherited optimizer moments")
        result[role] = owner
    if len({state_digest(result[role].sampling_rng.get_state()) for role in roles}) != 1:
        raise ValueError("paired owner sampler starts differ")
    return result
