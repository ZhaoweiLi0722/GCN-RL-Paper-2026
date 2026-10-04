"""Pure schedule/accounting for a proposed comparison; no execution authority."""

import math


TRAIN_ROLES = ("observed_td", "planner_tail_td", "planner_tail_mc")
EVAL_ROLES = ("plain_h8", "existing_frozen", *TRAIN_ROLES, "plain_h16")


def capture_epochs(world_index):
    if type(world_index) is not int or not 0 <= world_index < 24:
        raise ValueError("reference world index must be an integer in 0..23")
    return world_index, world_index + 24


def worlds(study, phase, block):
    if phase not in ("reference", "evaluation") or type(block) is not int or block not in range(5):
        raise ValueError("unknown phase or block")
    for index in range(24 if phase == "reference" else 12):
        condition, replicate = index % 3, index // 3
        yield dict(phase=phase, block=block, index=index, condition=condition,
                   replicate=replicate,
                   seed=study["streams"][phase + "_base"] + 1000 * block + 100 * condition + replicate)


def schedule(study):
    """Yield the complete serial work order without executing any backend."""
    numeric_contract(study)
    for block in range(5):
        yield dict(event="fork_ancestor", block=block, roles=list(TRAIN_ROLES))
        for world in worlds(study, "reference", block):
            yield dict(event="collect_reference", world=world,
                       capture_epochs=capture_epochs(world["index"]))
            for role in TRAIN_ROLES:
                yield dict(event="fit", world=world, role=role, updates=32)
        for role in TRAIN_ROLES:
            yield dict(event="seal", block=block, role=role, new_updates=768)
    yield dict(event="all_models_sealed", required_final_seals=15, unchanged_ancestors=5)
    for block in range(5):
        for world in worlds(study, "evaluation", block):
            for role in EVAL_ROLES:
                yield dict(event="evaluate", world=world, role=role,
                           horizon=16 if role == "plain_h16" else 8, updates=0)
    yield dict(event="saved_training_diagnostics", forward_cap=960, native_steps=0)
    yield dict(event="raw_readout_and_archive", new_scientific_calls=0)


def numeric_contract(study):
    """Reconcile exact counts without constructing a model or environment."""
    d, v, p, b = (study[k] for k in ("design", "value", "planning", "budget"))
    if (study["schema"] != "capacity-planner-tail-study-v1"
            or d["blocks"] != 5 or d["conditions"] != 3
            or d["reference_worlds_per_block"] != 24
            or d["evaluation_worlds_per_condition_block"] != 4
            or d["train_roles"] != list(TRAIN_ROLES) or d["eval_roles"] != list(EVAL_ROLES)
            or p != dict(candidate_sequences=16, response_quantiles=[.1, .5, .9],
                         summary_weights=[.25, .5, .25], horizon=8, long_reference_horizon=16)
            or v["updates_per_world"] != 32 or v["batch_size"] != 64 or v["td_horizon"] != 8
            or study["tails"]["per_world"] != 96 or study["tails"]["final_epoch"] != 64):
        raise ValueError("schedule differs from proposed fixed design")
    references, tests, batch, fits = 120, 360, 64, 360
    tail_rows = [48 * sum(64 - (t + 8) for t in capture_epochs(i)) for i in range(24)]
    tail_epochs = 5 * sum(tail_rows)
    tail_bootstraps = 5 * sum(math.ceil(n / batch) for n in tail_rows)
    ref_plan, test_plan = references * 48 * 384, (300 * 384 + 60 * 768) * 48
    updates = fits * 32
    diagnostics = references * 2 * 4
    counts = dict(
        native_steps=(references + tests) * 64,
        native_operations=(references + tests) * 66,
        value_optimizer_steps=updates, actor_optimizer_steps=0,
        optimizer_examples=updates * batch,
        forward_calls=updates + references + tail_bootstraps + 240 * 48 + diagnostics,
        maximum_forward_batch=batch, planning_decisions=(references + tests) * 48,
        candidate_quantile_rollouts=(references + tests) * 48 * 48,
        planning_epochs=ref_plan + test_plan, training_tail_epochs=tail_epochs,
        total_predictor_epochs=ref_plan + test_plan + tail_epochs,
        forecast_clones=references * 96, filter_transitions=(references + tests) * 64 * 100,
        new_final_seals=15)
    if any(b.get(k) != value for k, value in counts.items()):
        raise ValueError("budget arithmetic mismatch")
    if (d["new_training_worlds"] != references or d["evaluation_trajectories"] != tests
            or d["total_native_trajectories"] != references + tests
            or b["attempts"] != 1 or b["automatic_retry"] is not False
            or b["seconds"]["global"] != sum(x for k, x in b["seconds"].items() if k != "global")
            or b["raw_bytes"] + b["archive_bytes"] != b["combined_disk_bytes"]):
        raise ValueError("invalid attempt/time/storage contract")
    return dict(limits=counts, reference_worlds=references, evaluation_trajectories=tests,
                tail_rows_per_world_index=tail_rows,
                phase_counts={
                    "reference_and_tails": dict(native_steps=7680, native_operations=7920,
                        planning_epochs=ref_plan, training_tail_epochs=tail_epochs,
                        forecast_clones=11520, filter_transitions=768000, optimizer_steps=0,
                        forward_calls=0),
                    "value_fitting": dict(optimizer_steps=updates,
                        observed_td_updates=3840, planner_tail_td_updates=3840,
                        planner_tail_mc_updates=3840, forward_calls=updates + references + tail_bootstraps),
                    "frozen_evaluation": dict(native_steps=23040, native_operations=23760,
                        planning_epochs=test_plan, filter_transitions=2304000,
                        optimizer_steps=0, forward_calls=11520),
                    "analysis_archive": dict(forward_calls=diagnostics, optimizer_steps=0,
                                             native_steps=0, predictor_epochs=0)})
