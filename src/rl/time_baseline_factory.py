"""Pure configuration and explicit saved-weight forks for one baseline contrast."""

import copy

from src.rl.candidate_imitation import ImitationSettings
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_factory import ppo_settings
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.time_baseline_plan import CONTROLLERS, time_baseline_budget_plan
from src.rl.time_baseline_ppo import TimeBaselinePPOKernel


def time_baseline_config(original, proposal):
    """Bind unchanged mechanisms to new counts; neither input is modified."""
    time_baseline_budget_plan(proposal)
    if (original["blocks"] != proposal["blocks"] or original["representations"] != ["graph"]
            or original["objective"]["horizon"] != 52
            or original["objective"]["num_facilities"] != 20
            or original["objective"]["reward_scale"] != 1e-9
            or original["objective"]["gamma"] != 1. or original["objective"]["gae_lambda"] != 1.
            or original["objective"]["terminal_cost_added"] != 0.
            or original["objective"]["terminal_bootstrap"] != 0.
            or original["objective"]["reward"] != "negative_absolute_environment_step_cost"
            or original["candidate_message_graph"] != "specimen_routes"):
        raise ValueError("unchanged qualified design required")
    cfg = copy.deepcopy(original)
    cont = cfg["continuation"]
    if (cont["episodes_per_arm_per_block"] != proposal["episodes_per_training_arm_block"]
            or cont["episodes_per_rollout"] != proposal["episodes_per_rollout"]
            or cont["rollouts_per_arm_per_block"] != proposal["rollouts_per_training_arm_block"]
            or cont["epochs"] != proposal["epochs_per_rollout"]
            or cont["minibatch_sizes"] != proposal["minibatch_sizes"]):
        raise ValueError("unchanged continuation arithmetic required")
    cfg["time_baseline_proposal"] = copy.deepcopy(proposal)
    cfg["original_config_sha256"] = digest(original)
    cfg["scientific_execution_authorized"] = False
    cfg["final_controllers"] = list(CONTROLLERS)
    cfg["phase_budgets"] = copy.deepcopy(proposal["phase_budgets"])
    cfg["owner_budgets"] = copy.deepcopy(proposal["per_owner"])
    cfg.pop("rng_proposal", None)
    cont.update(full_episodes_total=288, environment_steps_total=14976,
                ppo_totals_scope="each_of_two_ppo_arms")
    evaluation = cfg["evaluation"]
    evaluation.pop("all_nine_learned_artifacts_sealed_before_any_test", None)
    evaluation.update(controllers_per_block=6, total_episodes=216, environment_steps=11232,
                      all_twelve_learned_artifacts_sealed_before_any_test=True,
                      primary="paired_raw_total_cost_time_baseline_ppo_minus_own_frozen",
                      secondary_contrasts=["time_baseline_ppo_minus_current_ppo",
                          "time_baseline_ppo_minus_bc_continue", "time_baseline_ppo_minus_r4",
                          "time_baseline_ppo_minus_full_mdl2", "own_frozen_minus_r4"])
    t = proposal["totals"]
    cfg["totals"] = dict(main_full_episodes=t["episodes"], main_trajectory_steps=t["trajectory"],
        mandatory_restore_clone_steps=t["clone"], main_environment_steps=t["environment"],
        main_actor_adam_calls=t["actor"], main_critic_adam_calls=t["critic"],
        main_optimizer_calls=t["optimizer"], main_phase_seconds=t["phase_seconds"],
        global_elapsed_seconds=t["global_seconds"], fresh_episode_builds=proposal["episode_builds"],
        backend_layout_builds=proposal["layout_builds"], build_env_calls_max=t["environment_builds"],
        mandatory_restored_session_clones=proposal["preflight_clone_instances"],
        scheduled_restore_envelope_reads=proposal["scheduled_restore_envelope_reads"],
        optional_environment_clones_max=0, optional_environment_steps_max=0, optional_seconds_max=0)
    return cfg


def fork_time_baseline_initializer(initializer, qualification, config, streams, block, *, split="training"):
    """Four equal-weight, fresh-owner forks. Preflight has separate private RNGs."""
    if (split not in ("training", "preflight") or block not in config["blocks"]
            or qualification.get("passed") is not True
            or qualification.get("kernel_sha256") != state_digest(initializer.state_dict())
            or initializer.steps != config["initialization"]["actor_adam_calls_per_block"]):
        raise ValueError("exact completed qualified saved initializer required")
    cont, opt = config["continuation"], config["optimizer"]
    seeds = [int(s) for s in streams["environment"][str(block)]["training"]]
    bindings = streams["neural_bindings"][str(block)][split]
    result = {}
    for role in CONTROLLERS[:4]:
        bound = bindings["current_ppo" if role == "own_frozen" else role]
        sample, shuffle = (int(streams["neural"][bound[k]]) for k in ("sampling_seed", "shuffle_seed"))
        if role == "own_frozen":
            kernel = DynamicCandidatePPOKernel(initializer.policy, initializer.contract, ppo_settings(config),
                enabled=True, mode="frozen", sampling_seed=sample, shuffle_seed=shuffle)
        elif role == "bc_continue":
            kernel = DynamicCandidateImitationKernel(initializer.policy, initializer.contract,
                ImitationSettings(opt["learning_rate"], opt["gradient_norm_cap_each_owner"], cont["batch_size"],
                    cont["bc_continue"]["actor_adam_calls_per_block"], cont["rollouts_per_arm_per_block"], "training"),
                enabled=True, sampling_seed=sample, shuffle_seed=shuffle)
        else:
            manifest = [dict(trajectory_id=f"training/block{block}/graph/{role}/episode{i:02d}",
                             environment_seed=seed) for i, seed in enumerate(seeds)]
            kernel = TimeBaselinePPOKernel(initializer.policy, initializer.contract, ppo_settings(config),
                enabled=True, mode="online", sampling_seed=sample, shuffle_seed=shuffle,
                target_method="collected_value" if role == "current_ppo" else "leave_one_episode_out_time",
                training_manifest=manifest, training_source_id=streams["namespace"],
                episode_horizon=config["objective"]["horizon"], episodes_per_rollout=cont["episodes_per_rollout"])
        if kernel.policy.snapshot_sha256() != initializer.policy.snapshot_sha256():
            raise ValueError("saved model fork weights differ")
        optimizers = ([kernel.optimizer] if role == "bc_continue" else kernel.optimizers.values())
        if any(o is not None and o.state_dict()["state"] for o in optimizers):
            raise ValueError("fresh continuation optimizers required")
        result[role] = kernel
    if len({state_digest(result[r].sampling_rng.get_state()) for r in CONTROLLERS[1:4]}) != 1:
        raise ValueError("paired training sampler starts differ")
    return result
