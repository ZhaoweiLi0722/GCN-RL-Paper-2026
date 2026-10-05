"""Prospective fixed-parent continuation comparison; pure bounded schedule."""

import math

from src.rl.capacity_planner_tail_design import capture_epochs, worlds


TRAIN_ROLES = ("adaptive_tail_td", "policy_tail_td", "policy_tail_mc")
EVAL_ROLES = ("plain_h8", "existing_frozen", *TRAIN_ROLES, "plain_h16")


def candidate_index(world_index, epoch):
    captures = capture_epochs(world_index)
    if type(epoch) is not int or epoch not in captures:
        raise ValueError("only the two prespecified roots may collect")
    return (world_index + 8 * captures.index(epoch)) % 16


def learner_config(study, role, ancestor_sha256):
    if role not in TRAIN_ROLES:
        raise ValueError("unknown continuation learner")
    policy = "adaptive" if role == "adaptive_tail_td" else "frozen_mpc"
    return dict(study["value"], architecture="graph", max_new_updates=768,
                method="planner_tail_mc" if role == "policy_tail_mc" else "planner_tail_td",
                tail_policy=policy, tails_per_world=6,
                tail_schema="capacity-policy-tail-record-v1",
                continuation_sha256=ancestor_sha256 if policy == "frozen_mpc" else None)


def counts():
    rows = [3 * sum(64 - (e + 8) for e in capture_epochs(i)) for i in range(24)]
    control = 5 * sum(3 * sum(max(0, 48 - e - 8) for e in capture_epochs(i)) for i in range(24))
    bootstrap = 2 * 5 * sum(math.ceil(n / 64) for n in rows)
    native_planning = 120 * 48 * 384 + (300 * 384 + 60 * 768) * 48
    tails = 5 * sum(rows)
    return dict(native_trajectories=480, native_steps=30720, native_operations=31680,
        value_optimizer_steps=11520, actor_optimizer_steps=0, optimizer_examples=737280,
        forward_calls=11520 + bootstrap + 11520 + control,
        training_bootstrap_forwards=bootstrap, continuation_value_forwards=control,
        native_planning_decisions=23040, native_planning_epochs=native_planning,
        continuation_planning_decisions=control, continuation_planning_epochs=control * 384,
        prefix_epochs=720 * 8, paired_tail_epochs=2 * tails,
        total_predictor_epochs=native_planning + control * 384 + 720 * 8 + 2 * tails,
        forecast_root_constructions=720, paired_tail_clones=1440,
        native_filter_transitions=3072000,
        branch_filter_transitions=(720 * 8 + tails) * 100,
        total_filter_transitions=3072000 + (720 * 8 + tails) * 100,
        new_final_seals=15, paired_endpoint_starts=720,
        tail_rows_per_policy=tails, tail_rows_per_world_index=rows)


def numeric_contract(study):
    expected = counts()
    if (study["schema"] != "capacity-policy-tail-study-v1"
            or study["design"]["train_roles"] != list(TRAIN_ROLES)
            or study["design"]["eval_roles"] != list(EVAL_ROLES)
            or study["design"]["blocks"] != 5
            or study["design"]["reference_worlds_per_block"] != 24
            or study["design"]["evaluation_worlds_per_condition_block"] != 4
            or study["tails"]["per_policy_per_world"] != 6
            or study["tails"]["candidate_selection"] != "(world_index + 8 * root_slot) % 16"
            or study["value"]["updates_per_world"] != 32
            or study["value"]["batch_size"] != 64
            or study["planning"] != dict(candidate_sequences=16, response_quantiles=[.1, .5, .9],
                summary_weights=[.25, .5, .25], horizon=8, long_reference_horizon=16)
            or any(study["value"].get(k) != v for k, v in dict(width=32, lr=.0003,
                gradient_norm_cap=5., cost_scale=1000000., td_horizon=8, gamma=1., evaluation_updates=0).items())
            or study["design"]["reference_behavior"] != "plain_h8"
            or study["design"]["total_native_trajectories"] != 480
            or study["tails"]["final_epoch"] != 64):
        raise ValueError("fixed-parent six-arm design mismatch")
    if any(study["budget"].get(k) != v for k, v in expected.items()):
        raise ValueError("nested continuation budget mismatch")
    b = study["budget"]
    if (b["attempts"] != 1 or b["automatic_retry"] is not False
            or b["seconds"]["global"] != sum(v for k, v in b["seconds"].items() if k != "global")
            or b["combined_disk_bytes"] != b["raw_bytes"] + b["archive_bytes"]):
        raise ValueError("single attempt/time/IO bounds mismatch")
    return dict(limits=expected, scientific_execution_authorized=False)


def schedule(study):
    numeric_contract(study)
    for block in range(5):
        yield dict(event="fork_ancestor", block=block, roles=list(TRAIN_ROLES))
        for world in worlds(study, "reference", block):
            yield dict(event="collect_reference", world=world,
                roots=[dict(epoch=e, candidate=candidate_index(world["index"], e))
                       for e in capture_epochs(world["index"])])
            for role in TRAIN_ROLES:
                yield dict(event="fit", world=world, role=role, updates=32)
        for role in TRAIN_ROLES:
            yield dict(event="seal", block=block, role=role, new_updates=768)
    yield dict(event="all_models_sealed", required_final_seals=15, unchanged_ancestors=5)
    for block in range(5):
        for world in worlds(study, "evaluation", block):
            for role in EVAL_ROLES:
                yield dict(event="evaluate", world=world, role=role, updates=0)
    yield dict(event="raw_readout_and_archive", new_scientific_calls=0)
