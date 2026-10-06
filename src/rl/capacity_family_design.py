"""Prospective finite five-family selection; no environment/model execution."""

import copy

SCOPE = "capacity-family-selection-v1"
METHODS = ("ddpg", "td3", "sac", "ppo", "value_td")
PHASES = ("development_training", "development_evaluation", "confirmation_training", "confirmation_evaluation")
TRAIN_COUNTS = {
    "ddpg": (208, 32, 32, 0), "td3": (240, 16, 64, 0),
    "sac": (304, 32, 64, 0), "ppo": (160, 32, 0, 32), "value_td": (81, 0, 0, 32),
}


def expected_learning_counts(finalists):
    fields = ("neural_forward_module_calls", "actor_optimizer_steps", "critic_optimizer_steps", "value_optimizer_steps")
    counts = dict.fromkeys(fields, 0)
    for method, repeats in [(m, 576) for m in METHODS] + [(r["method"], 960) for r in finalists]:
        for field, amount in zip(fields, TRAIN_COUNTS[method]):
            counts[field] += repeats*amount
    counts["neural_forward_module_calls"] += (720+840)*48
    counts["total_optimizer_steps"] = sum(counts[k] for k in fields[1:])
    counts["optimizer_example_presentations"] = 64*counts["total_optimizer_steps"]
    counts["fit_batches"] = 4800*32
    return counts


def worlds(study, phase, block):
    n = study["train_worlds"] if phase.endswith("training") else study["eval_worlds"]
    base = study["streams"]["world_base"] + PHASES.index(phase) * 100000
    return [dict(phase=phase, block=block, condition=j % 3, replicate=j // 3,
                 seed=base + block * 1000 + j) for j in range(n)]


def seeds(study, phase, block):
    offset = (0 if phase.startswith("development") else 1000) + block * 10
    return study["streams"]["model_base"] + offset, study["streams"]["sampler_base"] + offset


def numeric_contract(study):
    if (study["methods"] != list(METHODS) or study["development_blocks"] != 3
            or study["confirmation_blocks"] != 5 or study["max_finalists"] != 2
            or study["train_worlds"] != 96 or study["eval_worlds"] != 24
            or len(study["learning_rates"]) != 2):
        raise ValueError("finite five-family allocation changed")
    # Per-counter upper bounds: finalists are selected only after development.
    # Unused counters cannot be exchanged for trials, epochs or more worlds.
    populations = (2880, 864, 1920, 1080)
    planner_equivalents = (576, 360, 960, 840)
    phases = {}
    for phase, n, equiv in zip(PHASES, populations, planner_equivalents):
        training = phase.endswith("training")
        phases[phase] = dict(trajectories=n, native_steps=n*64, native_constructions=n,
            construction_triggered_resets=n, total_native_operations=n*66,
            control_steps=n*48, tail_steps=n*16, estimator_receipt_updates=n*64,
            estimator_hypothesis_transitions=n*6400, planner_total_decisions=equiv*48,
            planner_candidate_rollouts=equiv*48*48, planner_total_model_epochs=equiv*18432,
            neural_forward_module_calls=n*(400 if training else 48),
            actor_optimizer_steps=n*32 if training else 0,
            critic_optimizer_steps=n*64 if training else 0,
            value_optimizer_steps=n*32 if training else 0,
            total_optimizer_steps=n*96 if training else 0,
            optimizer_example_presentations=n*96*64 if training else 0,
            fit_batches=n*32 if training else 0)
    return dict(limits={k: sum(p[k] for p in phases.values()) for k in phases[PHASES[0]]},
                phase_limits=phases, counts_are_upper_bounds=True,
                native_clones=0, no_budget_transfer=True)


def merged_proposal(original, study):
    p = copy.deepcopy(original)
    p["family_study"] = copy.deepcopy(study)
    p["design"]["rng_namespace"] = study["streams"]["namespace"]
    p["design"]["training_seeds"] = [seeds(study, PHASES[0], b)[0] for b in range(3)]
    b = study["budget"]
    p["proposed_budget"] = dict(neural_forward_max_batch=64,
        planner_model_epochs_per_decision=384, flush_every_epochs=8,
        time_seconds=b["seconds"], rss_cap_bytes=b["rss_bytes"],
        raw_artifact_cap_bytes=b["raw_bytes"], archive_cap_bytes=b["archive_bytes"],
        combined_disk_cap_bytes=b["raw_bytes"]+b["archive_bytes"], file_cap=b["file_cap"],
        compute_threads=4)
    return p
