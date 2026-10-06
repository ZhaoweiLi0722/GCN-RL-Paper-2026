"""Prospective fresh-seed study; arithmetic only, no scientific execution."""

import copy
import math

from src.rl.capacity_planner_tail_design import capture_epochs
from src.rl.capacity_policy_tail_design import candidate_index

SCOPE = "capacity-conditional-confirmation-v1"
WARM_ROLES = ("graph", "self_only")
TRAIN_ROLES = ("graph_td", "self_only_td", "graph_mc")
EVAL_ROLES = ("plain_h8", "fresh_frozen", "existing_frozen", *TRAIN_ROLES, "plain_h16")
PHASES = ("warmup_reference", "reference_and_tails", "value_fitting", "frozen_evaluation", "analysis_archive")


def worlds(study, phase, block):
    n = {"warmup": 48, "reference": 24, "evaluation": 12}[phase]
    base = study["streams"][phase + "_base"]
    for i in range(n):
        yield dict(phase=phase, block=block, condition=i % 3, replicate=i // 3,
                   index=i, seed=base + block * 1000 + (i % 3) * 100 + i // 3)


def counts():
    rows = [sum(56 - e for e in capture_epochs(i)) for i in range(24)]
    branch_steps = 5 * sum(sum(64 - e for e in capture_epochs(i)) for i in range(24))
    parent_decisions = 5 * sum(sum(max(0, 40 - e) for e in capture_epochs(i)) for i in range(24))
    updates = 5 * (2 * 1536 + 3 * 768)
    bootstrap = 240 * 2 + 5 * 2 * sum(math.ceil(n / 64) for n in rows)
    main = 240 + 120 + 420
    return dict(main_trajectories=main, warmup_worlds=240, reference_worlds=120,
                evaluation_trajectories=420, native_branch_clones=240,
                main_native_steps=main * 64, native_branch_steps=branch_steps,
                total_native_steps=main * 64 + branch_steps,
                total_native_operations=main * 66 + branch_steps,
                value_optimizer_steps=updates, actor_optimizer_steps=0,
                optimizer_examples=updates * 64, training_bootstrap_forwards=bootstrap,
                forward_calls=updates + bootstrap + parent_decisions + 5 * 60 * 48,
                main_planning_epochs=(main + 60) * 48 * 384,
                native_branch_planning_decisions=parent_decisions,
                native_branch_planning_epochs=parent_decisions * 384,
                total_predictor_epochs=(main + 60) * 48 * 384 + parent_decisions * 384,
                total_filter_transitions=(main * 64 + branch_steps) * 100,
                warm_seals=10, final_seals=15, legacy_bindings=5,
                native_suffix_rows=5 * sum(rows))


def validate(study):
    if (study["schema"] != SCOPE or study["scientific_execution_authorized"] is not False
            or study["design"] != dict(blocks=5, conditions=3, warmup_worlds_per_block=48,
                reference_worlds_per_block=24, evaluation_worlds_per_condition_block=4,
                warm_roles=list(WARM_ROLES), train_roles=list(TRAIN_ROLES), eval_roles=list(EVAL_ROLES),
                reference_behavior="plain_h8", all_final_seals_before_tests=True, selection="final_only")
            or study["planning"] != dict(candidate_sequences=16, response_quantiles=[.1, .5, .9],
                summary_weights=[.25, .5, .25], horizon=8, long_reference_horizon=16)
            or any(study["value"].get(k) != v for k, v in dict(width=32, lr=.0003,
                batch_size=64, updates_per_world=32, gradient_norm_cap=5., cost_scale=1000000.,
                td_horizon=8, gamma=1., evaluation_updates=0).items())
            or study["budget"]["counts"] != counts()):
        raise ValueError("fixed fresh confirmation design mismatch")
    b = study["budget"]
    if (b["attempts"] != 1 or b["automatic_retry"] is not False or b["workers"] != 1
            or b["compute_threads"] != 4
            or b["seconds"]["global"] != sum(v for k, v in b["seconds"].items() if k != "global")
            or b["raw_bytes"] + b["archive_bytes"] != b["combined_disk_bytes"]):
        raise ValueError("nonrefundable time/disk/single-worker contract mismatch")
    return counts()


def warm_config(study, architecture):
    if architecture not in WARM_ROLES:
        raise ValueError("unknown architecture")
    return dict(study["value"], architecture=architecture, method="observed_td", max_new_updates=1536)


def tail_config(study, role, continuation_sha256):
    if role not in TRAIN_ROLES:
        raise ValueError("unknown training role")
    return dict(study["value"], architecture="self_only" if role == "self_only_td" else "graph",
                max_new_updates=768, method="planner_tail_mc" if role == "graph_mc" else "planner_tail_td",
                tail_policy="frozen_mpc", tails_per_world=2,
                tail_schema="capacity-native-tail-record-v1", continuation_sha256=continuation_sha256)


def numeric_contract(study):
    d = validate(study)
    limits = dict(trajectories=780, native_steps=49920, native_constructions=780,
        construction_triggered_resets=780, total_native_operations=51480,
        control_steps=37440, tail_steps=12480, value_optimizer_steps=26880,
        total_optimizer_steps=26880, optimizer_example_presentations=1720320,
        neural_forward_module_calls=d["forward_calls"], planner_total_decisions=37440,
        planner_candidate_rollouts=37440 * 48, planner_total_model_epochs=d["main_planning_epochs"],
        estimator_receipt_updates=49920, estimator_hypothesis_transitions=4992000,
        native_branch_clones=240, native_branch_steps=9720, native_branch_control_steps=5880,
        native_branch_settlement_steps=3840, native_branch_planning_decisions=4100,
        native_branch_candidate_rollouts=196800, native_branch_model_epochs=1574400,
        native_branch_filter_transitions=972000, warm_seals=10, final_seals=15)
    phases = {p: dict.fromkeys(limits, 0) for p in PHASES}
    for phase, n, extra in ((PHASES[0], 240, 0), (PHASES[1], 120, 0), (PHASES[3], 420, 60)):
        phases[phase].update(trajectories=n, native_steps=n * 64, native_constructions=n,
            construction_triggered_resets=n, total_native_operations=n * 66, control_steps=n * 48,
            tail_steps=n * 16, planner_total_decisions=n * 48, planner_candidate_rollouts=n * 2304,
            planner_total_model_epochs=(n + extra) * 18432, estimator_receipt_updates=n * 64,
            estimator_hypothesis_transitions=n * 6400)
    phases[PHASES[1]].update({k: v for k, v in limits.items() if k.startswith("native_branch_")})
    phases[PHASES[1]]["neural_forward_module_calls"] = 4100
    phases[PHASES[2]].update(value_optimizer_steps=26880, total_optimizer_steps=26880,
        optimizer_example_presentations=1720320, neural_forward_module_calls=26880 + d["training_bootstrap_forwards"],
        warm_seals=10, final_seals=15)
    phases[PHASES[3]]["neural_forward_module_calls"] = 14400
    if any(sum(p[k] for p in phases.values()) != v for k, v in limits.items()):
        raise ValueError("phase counter mismatch")
    return dict(limits=limits, phase_limits=phases)


def merged_proposal(original, study):
    numeric_contract(study)
    p = copy.deepcopy(original)
    p["confirmation_study"] = copy.deepcopy(study)
    p["design"].update(blocks=5, training_seeds=study["streams"]["model_seeds"],
        rng_namespace=study["streams"]["namespace"], substream_purposes=study["streams"]["purposes"])
    p["controllers"] = [dict(id=r) for r in EVAL_ROLES]
    p["scope"] = dict(study=SCOPE, automatic_retry=False, resume=False)
    b = study["budget"]
    seconds = copy.deepcopy(b["seconds"])
    seconds["admission_lock_binding"] = seconds.pop("admission")
    seconds["failure_flush_shutdown_reserve"] = seconds.pop("failure_preservation")
    p["proposed_budget"] = dict(neural_forward_max_batch=64, planner_model_epochs_per_decision=384,
        time_seconds=seconds, rss_cap_bytes=b["rss_bytes"], raw_artifact_cap_bytes=b["raw_bytes"],
        archive_cap_bytes=b["archive_bytes"], combined_disk_cap_bytes=b["combined_disk_bytes"],
        file_cap=b["file_cap"], compute_threads=4)
    return p
