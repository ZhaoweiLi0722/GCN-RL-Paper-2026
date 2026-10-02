"""Independent JSON-only tail accounting; no simulator or learner imports."""

import math

from src.rl.candidate_pilot_verification import (
    OPERATING, PATIENT, close, count, finite, identity_counts, json_hash, vector,
)


IMMUTABLE_PATIENT = ("patient_id", "specimen_id", "enrollment_epoch", "collection_facility",
                     "health_index", "deterioration_epoch", "risk_type", "risk_multiplier")


def verify_cohort_tail(prefix_final, final, rows, receipt, *, enrollment_steps,
                       patient_resolution_steps, accounting_steps, num_facilities):
    """Reconstruct counts and costs from raw primitives, not summary deltas.

    The enclosing bundle verifier must also bind these receipts to declared
    controller/world/file hashes and verify the original prefix independently.
    This helper intentionally does not claim policy compliance from cost alone.
    """
    if (any(type(x) is not int or x < 1 for x in
            (enrollment_steps, patient_resolution_steps, accounting_steps, num_facilities))
            or patient_resolution_steps > accounting_steps):
        raise ValueError("fixed positive contract dimensions required")
    if (prefix_final["scalars"]["t"] != enrollment_steps
            or final["scalars"]["t"] != enrollment_steps + accounting_steps
            or len(rows) != accounting_steps
            or receipt["prefix_final_sha256"] != json_hash(prefix_final)
            or receipt["final_sha256"] != json_hash(final)):
        raise ValueError("exact hashed prefix/fixed tail boundary required")
    start, end = identity_counts(prefix_final), identity_counts(final)
    if set(prefix_final["patients"]) != set(final["patients"]) or end["active"]:
        raise ValueError("changed cohort identities or unresolved final patients")
    for pid, before in prefix_final["patients"].items():
        after = final["patients"][pid]
        if any(before[key] != after[key] for key in IMMUTABLE_PATIENT):
            raise ValueError("immutable patient attributes changed")
        if before["status"] in ("delivered", "lost") and before["status"] != after["status"]:
            raise ValueError("previously resolved patient changed outcome")
    active, losses, completions = start["active"], 0, 0
    resolved = 0 if active == 0 else None
    components = {key: [] for key in OPERATING + PATIENT}
    costs, waiting = [], 0
    idle = [0.] * (3 * num_facilities) + [-1.] * num_facilities
    for i, row in enumerate(rows, 1):
        info = row["info"]
        if (type(row["index"]) is not int or row["index"] != i
                or type(row["accounting_done"]) is not bool
                or row["accounting_done"] != (i == accounting_steps)):
            raise ValueError("tail rows are missing, reordered or prematurely terminated")
        request = row["action"]
        if len(request) != 4 * num_facilities:
            raise ValueError("full raw request width required")
        for value in request:
            finite(value, signed=True)
        if active == 0 and request != idle:
            raise ValueError("new commitment requested after patient resolution")
        if any(vector(info["demand"], num_facilities)):
            raise ValueError("nonzero post-window enrollment")
        for key in components:
            components[key].append(finite(info[key]))
        cost = finite(info["cost"])
        close(finite(info["base_cost"]), math.fsum(info[k] for k in OPERATING))
        close(cost, math.fsum(info[k] for k in OPERATING + PATIENT))
        close(finite(row["raw_reward"], signed=True), -cost)
        close(finite(row["cost"]), cost)
        lost = sum(vector(info["patients_lost"], num_facilities))
        served = sum(vector(info["patients_completed"], num_facilities))
        losses += lost
        completions += served
        active -= lost + served
        if active < 0 or count(row["active"]) != active or count(info["identity_active_count"]) != active:
            raise ValueError("patient flow and active receipts disagree")
        if i >= patient_resolution_steps and active:
            raise ValueError("patient-resolution bound exceeded")
        if active == 0 and resolved is None:
            resolved = i
        if row["resolution_step"] != resolved:
            raise ValueError("wrong first-resolution receipt")
        waiting += sum(vector(info["waiting_patients"], num_facilities))
        costs.append(cost)
    if losses != end["lost"] - start["lost"] or completions != end["delivered"] - start["delivered"]:
        raise ValueError("raw losses/infusions disagree with patient registry outcomes")
    arrays = final["arrays"]
    for field in ("specimen_transfer_pipeline", "reagent_transfer_pipeline", "capacity_transfer_pipeline"):
        for row in arrays[field]:
            if any(finite(x) != 0 for x in row):
                raise ValueError("resource transfer remains at fixed endpoint")
    if "reagent_purchase_pipeline" in arrays and any(
            finite(x) != 0 for row in arrays["reagent_purchase_pipeline"] for x in row):
        raise ValueError("unexpected outstanding procurement")
    for field in ("reagents", "bioreactors"):
        values = arrays[field] if field == "reagents" else [x for row in arrays[field] for x in row]
        for x in values:
            finite(x)
    return dict(format="independent-cohort-tail-v1", patient_resolution_step=resolved,
                fixed_economic_end=enrollment_steps + accounting_steps,
                tail_cost=math.fsum(costs), tail_patient_cost=math.fsum(costs[:resolved]),
                tail_post_resolution_cost=math.fsum(costs[resolved:]),
                tail_losses=losses, tail_completions=completions, tail_waiting_patient_steps=waiting,
                components={key: math.fsum(values) for key, values in components.items()},
                initial_compartments=start, final_compartments=end,
                retained_reagents=arrays["reagents"], retained_bioreactors=arrays["bioreactors"],
                terminal_stock_valued=False, primitive_costs_counted_once=True)
