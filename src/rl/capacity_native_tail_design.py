"""Finite native-grounded versus forecast-grounded training comparison."""

import copy
import math

from src.rl.capacity_planner_tail_design import capture_epochs, worlds
from src.rl.capacity_policy_tail_design import candidate_index, counts as forecast_counts


TRAIN_ROLES = ("forecast_tail_td", "native_tail_td", "native_tail_mc")
EVAL_ROLES = ("plain_h8", "existing_frozen", *TRAIN_ROLES, "plain_h16")


def counts():
    d = forecast_counts()
    rows = [sum(56 - e for e in capture_epochs(i)) for i in range(24)]
    native_steps = 5 * sum(sum(64 - e for e in capture_epochs(i)) for i in range(24))
    decisions = d["continuation_planning_decisions"] // 3
    bootstrap = 5 * sum(math.ceil(n / 64) + math.ceil(3 * n / 64) for n in rows)
    d.update(training_bootstrap_forwards=bootstrap,
        forward_calls=11520 + bootstrap + 11520 + d["continuation_value_forwards"] + decisions,
        native_branch_clones=240, native_branch_steps=native_steps,
        native_branch_control_steps=native_steps - 240 * 16,
        native_branch_settlement_steps=240 * 16,
        native_branch_planning_decisions=decisions,
        native_branch_planning_epochs=decisions * 384,
        native_branch_filter_transitions=native_steps * 100,
        native_tail_rows=5 * sum(rows), native_rows_per_world_index=rows,
        total_native_steps=30720 + native_steps, total_native_operations=31680 + native_steps,
        total_predictor_epochs=d["total_predictor_epochs"] + decisions * 384,
        total_filter_transitions=d["total_filter_transitions"] + native_steps * 100)
    return d


def learner_config(study, role, ancestor_sha256):
    if role not in TRAIN_ROLES:
        raise ValueError("unknown native-grounding arm")
    return dict(study["value"], architecture="graph", max_new_updates=768,
        method="planner_tail_mc" if role == "native_tail_mc" else "planner_tail_td",
        tail_policy="frozen_mpc", continuation_sha256=ancestor_sha256,
        tails_per_world=6 if role == "forecast_tail_td" else 2,
        tail_schema="capacity-policy-tail-record-v1" if role == "forecast_tail_td" else "capacity-native-tail-record-v1")


def numeric_contract(study):
    if (study["schema"] != "capacity-native-tail-study-v1"
            or study["scientific_execution_authorized"] is not False
            or study["design"]["train_roles"] != list(TRAIN_ROLES)
            or study["design"]["eval_roles"] != list(EVAL_ROLES)
            or study["design"]["blocks"] != 5
            or study["design"]["reference_worlds_per_block"] != 24
            or study["design"]["evaluation_worlds_per_condition_block"] != 4
            or study["design"]["reference_behavior"] != "plain_h8"
            or study["tails"]["native_per_world"] != 2
            or study["tails"]["final_epoch"] != 64
            or study["tails"]["candidate_selection"] != "(world_index + 8 * root_slot) % 16"
            or study["planning"] != dict(candidate_sequences=16, response_quantiles=[.1, .5, .9],
                summary_weights=[.25, .5, .25], horizon=8, long_reference_horizon=16)
            or any(study["value"].get(k) != v for k, v in dict(width=32, lr=.0003, batch_size=64,
                updates_per_world=32, gradient_norm_cap=5., cost_scale=1000000., td_horizon=8,
                gamma=1., evaluation_updates=0).items())):
        raise ValueError("fixed native-grounding design mismatch")
    expected, budget = counts(), study["budget"]
    if any(budget.get(k) != v for k, v in expected.items()):
        raise ValueError("native and forecast work must be charged separately")
    if (budget["attempts"] != 1 or budget["automatic_retry"] is not False
            or budget["seconds"]["global"] != sum(v for k, v in budget["seconds"].items() if k != "global")
            or budget["combined_disk_bytes"] != budget["raw_bytes"] + budget["archive_bytes"]):
        raise ValueError("single attempt/time/disk mismatch")
    return dict(limits=expected, scientific_execution_authorized=False)


def forecast_compatibility_study(study):
    """Reuse the frozen counter arithmetic, never its execution authority."""
    from src.rl.capacity_policy_tail_design import TRAIN_ROLES as OLD_TRAIN, EVAL_ROLES as OLD_EVAL
    s = copy.deepcopy(study)
    s["schema"] = "capacity-policy-tail-study-v1"
    s["design"].update(train_roles=list(OLD_TRAIN), eval_roles=list(OLD_EVAL))
    s["tails"]["per_policy_per_world"] = 6
    s["budget"].update(forecast_counts())
    return s
