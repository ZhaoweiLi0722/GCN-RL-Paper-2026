"""Versioned raw-only S1 reader with channel-specific resource contracts.

The source-frozen failed reader remains unchanged. Scalar/patient accounting,
file lineage and paired analysis reuse its independent helpers. Only executed
resource typing and floating-point flow conservation differ. No simulator,
checkpoint, model or optimizer is loaded, and no execution is authorized.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

from src.rl.candidate_pilot_verification import (
    finite, json_hash, read_raw_episodes, vector, verify_episode,
)
from src.rl.dynamic_candidate_verification import (
    EXECUTED, SPLIT_ROLES, _paired_analysis, _reread, _sha, _validate_plan,
)


def continuous_vector(values, n, *, signed=False):
    """Preserve finite raw quantities exactly, including legitimate fractions."""
    if not isinstance(values, (list, tuple)) or len(values) != n:
        raise ValueError("raw vector width mismatch")
    return [finite(value, signed=signed) for value in values]


def executed_resources(info, n):
    result = {"specimen_transfers": vector(info["specimen_transfers"], n, signed=True)}
    for key in EXECUTED[1:]:
        result[key] = continuous_vector(info[key], n, signed=key != "replenishment")
    if sum(result["specimen_transfers"]) != 0:
        raise ValueError("executed specimen transfer does not conserve flow")
    for key in ("capacity_transfers", "reagent_transfers"):
        try:
            net = math.fsum(result[key])
            throughput = math.fsum(abs(value) for value in result[key])
        except OverflowError as error:
            raise ValueError("nonfinite executed resource accounting") from error
        # Same precision scale as the historical scalar-cost reconciler, now
        # relative to total flow rather than a zero-valued expected net flow.
        tolerance = 1e-8 + 1e-12 * throughput
        if not math.isfinite(net) or not math.isfinite(throughput) or abs(net) > tolerance:
            raise ValueError("executed facility-net transfer does not conserve flow")
    return result


def _enrich(outcome, header, rows, final_state, config):
    # Preserve the frozen role/lineage checks explicitly in this new version;
    # do not patch its module globals or sanitize a copy of the raw evidence.
    role, split, n = header["role"], header["split"], config["objective"]["num_facilities"]
    if split not in SPLIT_ROLES or role not in SPLIT_ROLES[split]:
        raise ValueError("unexpected dynamic controller/split")
    expected_representation = "reference" if role in ("r4", "full_mdl2") else "graph"
    if header["representation"] != expected_representation:
        raise ValueError("controller representation mismatch")
    expected_selection = ("reference" if role == "r4" else "anchor" if role == "full_mdl2" else
                          "greedy" if role in ("own_frozen", "initializer_greedy") or split == "test" else "sample")
    if header["selection"] != expected_selection:
        raise ValueError("recorded controller selection mode differs")
    if any(not isinstance(header[k], str) or not header[k] for k in ("source_id", "trajectory_id")):
        raise ValueError("nonempty episode lineage required")
    _sha(header["policy_sha256"])
    if json_hash([header, rows, final_state]) != outcome["raw_episode_sha256"]:
        raise ValueError("raw episode changed between scalar and action verification")
    actions, waiting, corrections = [], 0, []
    for step, row in enumerate(rows):
        event = row["event"]
        record, decision = event["audit"]["record"], event["audit"]["decision"]
        evaluation, choice = decision["evaluation"], decision["choice"]
        bank, info = evaluation["candidates"], event["info"]
        request, reference, anchor = record["action"], bank["requests"][0], bank["requests"][1]
        greedy = ((split == "test" and role.startswith("own_"))
                  or role in ("own_frozen", "initializer_greedy"))
        if greedy and choice["class_index"] != int(np.argmax(evaluation["log_probs"])):
            raise ValueError("dynamic learned test did not use canonical greedy class")
        if role == "r4" and request != reference:
            raise ValueError("R4 original request changed")
        if role == "full_mdl2" and (request != anchor or reference != anchor):
            raise ValueError("full MDL-2 requires the full-anchor collector, not a restricted anchor")
        waiting += sum(vector(info["waiting_patients"], n))
        executed = executed_resources(info, n)
        correction = [float(a) - float(b) for a, b in zip(request, reference)]
        corrections.append({"step_index": step, "request_delta": correction,
                            "request_l1": sum(abs(v) for v in correction),
                            "different_class": choice["class_index"] != bank["request_to_class"][0]})
        actions.append({"step_index": step, "requested": copy.deepcopy(request),
                        "specimen_requested_integer_net": copy.deepcopy(info["specimen_requested_integer_net"]),
                        "executed": executed, "reference_request": copy.deepcopy(reference),
                        "selected_class_key": copy.deepcopy(bank["class_keys"][choice["class_index"]]),
                        "inference_seconds": finite(row["inference_seconds"])})
    terminal = outcome["terminal_compartments"]
    for field, compartment in (("waiting_patients", "waiting"), ("in_production_patients", "production"),
                               ("specimen_in_transit", "specimen_transit")):
        if sum(vector(rows[-1]["event"]["info"][field], n)) != terminal[compartment]:
            raise ValueError("final reported queue differs from raw patient registry")
    return outcome | {
        "source_id": header["source_id"], "waiting_patient_steps": waiting,
        "waiting_definition": "sum_of_end_of_epoch_waiting_patients",
        "expiry_losses": sum(outcome["loss_causes"][k] for k in
                             ("patients_lost_waiting_expired", "patients_lost_transit_expired", "finished_expired")),
        **{f"terminal_{k}": terminal[k] for k in ("waiting", "production", "specimen_transit", "finished_return")},
        "actions": actions, "reference_corrections": corrections,
        "correction_reference": "full_mdl2" if role == "full_mdl2" else "contemporaneous_r4",
        "raw_header_sha256": json_hash(header),
        "compute_cost": {"environment_steps": len(rows), "evaluation_env_steps": len(rows) if split == "test" else 0,
                         "decision_path_seconds": outcome["inference_seconds"],
                         "training_compute": "not_in_episode_receipts_use_campaign_budget_ledger"},
    }


def verify_dynamic_episode(header, rows, final_state, config):
    return _enrich(verify_episode(header, rows, final_state, config), header, rows, final_state, config)


def read_dynamic_raw_episodes(root, index, config):
    root = Path(root).resolve()
    files, scalar = read_raw_episodes(root, index, config)
    outcomes = []
    for entry, original in zip(index, scalar):
        header = _reread(root, entry["header"])
        rows = _reread(root, entry["events"], jsonl=True)
        final = _reread(root, entry["final_state"])
        outcomes.append(_enrich(original, header, rows, final, config) | {"raw_files": copy.deepcopy(entry)})
    return copy.deepcopy(files), outcomes


def verify_dynamic_raw_bundle(root, index, config, streams):
    worlds = _validate_plan(config, streams)
    if len(index) != config["evaluation"]["total_episodes"]:
        raise ValueError("missing or extra prescribed raw evaluation entries")
    files, outcomes = read_dynamic_raw_episodes(root, index, config)
    return {"format": "dynamic-candidate-resource-verification-v2", "files": copy.deepcopy(files),
            "index_sha256": json_hash(index), "config_sha256": json_hash(config), "streams_sha256": json_hash(streams),
            "outcomes": outcomes, "analysis": _paired_analysis(outcomes, config, streams, worlds),
            "verification": "raw hashes, scalar accounting, channel-specific resource typing and paired schedule verified",
            "grants_scientific_execution_authorization": False}
