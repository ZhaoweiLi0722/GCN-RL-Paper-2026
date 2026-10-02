"""Raw-only current-policy branch costs and prospective finite dataset assembly."""

import copy
import math

import numpy as np

from src.rl.candidate_pilot_resources import digest
from src.rl.candidate_pilot_verification import (
    CAUSES, OPERATING, PATIENT, close, finite, identity_counts, json_hash, vector,
)
from src.rl.cohort_verification import verify_cohort_tail
from src.rl.conservative_cohort_plan import cohort_ids, validate_config
from src.rl.paired_cohort_verification import (
    _context, _identities, _request, _routing, _resources, _snapshot_resources, raw_json,
)


def branch_reader(config, streams, block, round_index, continuation_sha256):
    """Expected source, round, continuation and seeds come from the coordinator."""
    validate_config(config)
    cfg = copy.deepcopy(config) | {"context_cohorts_per_block": 4}
    def verify(*, header, rows, states, receipt):
        header, rows, states, receipt = raw_json((header, rows, states, receipt))
        context, manifest = header["context"], header["manifest"]
        c, t = context["cohort"], context["after_prefix_steps"]
        if context["block"] != block or c not in cohort_ids(round_index):
            raise ValueError("wrong independently expected context block/round")
        expected_context = streams["environment"][str(block)]["context"][c]
        bank, n, scale = _context(context, cfg, expected_context)
        k, r = manifest["candidate_index"], manifest["replication"]
        if (type(k) is not int or not 0 <= k < len(bank["class_keys"])
                or type(r) is not int or not 0 <= r < 4
                or manifest["format"] != "conservative-cohort-branch-v1"
                or manifest["round"] != round_index or manifest["continuation_sha256"] != continuation_sha256
                or manifest["continuation"] != "frozen_round_start_greedy_then_common_tail"
                or receipt["manifest"] != manifest or manifest["context_sha256"] != context["context_sha256"]
                or manifest["reference_sha256"] != context["reference_sha256"]
                or any(manifest[name] != context[name] for name in
                       ("block", "cohort", "after_prefix_steps", "context_id", "source_id"))
                or manifest["endpoint"] != 63
                or manifest["future_rng_only"] is not True
                or manifest["event_aligned_crn_after_divergence_claimed"] is not False):
            raise ValueError("current-policy branch declaration differs")
        seed = streams["conditional_future"][f"block{block}/cohort{c}/after{t}"][r]
        if manifest["future_seed"] != str(seed):
            raise ValueError("conditional future seed differs")
        first = bank["requests"][bank["representatives"][k]]
        if manifest["first_request"] != first:
            raise ValueError("first request changed canonical full precision")
        initial, prefix, final = header["initial_state"], states["prefix_final"], states["final_state"]
        expected_initial = copy.deepcopy(context["environment"])
        expected_initial["rng_state"] = np.random.PCG64(int(seed)).state
        if (json_hash(initial) != json_hash(expected_initial)
                or json_hash(initial) != manifest["initial_state_sha256"]
                or len(rows) != 63 - t or receipt["steps"] != len(rows)
                or json_hash(rows) != receipt["raw_rows_sha256"]
                or json_hash(final) != receipt["final_state_sha256"]
                or receipt["included_sunk_cost"] is not False or receipt["terminal_active"] != 0
                or prefix["scalars"]["t"] != 52 or final["scalars"]["t"] != 63):
            raise ValueError("complete future-only branch hashes/counts required")
        _identities(initial, prefix)
        _identities(prefix, final)
        for state in (initial, prefix, final):
            if str(state["scalars"]["_episode_seed"]) != str(expected_context):
                raise ValueError("context world seed changed inside branch")
        initial_counts, final_counts = identity_counts(initial), identity_counts(final)
        costs, token = [], json_hash(initial)
        lead = context["anchor_config"]["production_lead_time"]
        resources = _snapshot_resources(initial, rows[0]["resources_before"], n, lead)
        losses = completed = waiting = 0
        for i, row in enumerate(rows):
            at = t + i
            if at == 52:
                if token != json_hash(prefix):
                    raise ValueError("prefix snapshot chain differs")
                closed = copy.deepcopy(prefix)
                for field in ("demand", "demand_forecast"):
                    closed["arrays"][field] = [0.] * n
                closed["scalars"]["demand_forecast_error"] = 0.
                token = json_hash(closed)
            expected_source = "candidate" if i == 0 else "frozen_round_start_policy" if at < 52 else "common_tail"
            if (row["absolute_step"] != at or row["branch_step"] != i
                    or row["stage"] != ("prefix" if at < 52 else "tail")
                    or row["action_source"] != expected_source or row["action_dtype"] != "float64"
                    or row["before_state_sha256"] != token):
                raise ValueError("branch raw step/source/state chain differs")
            request = _request(row["action"], n)
            if i == 0 and request != first:
                raise ValueError("first action raw bytes differ")
            info = row["info"]
            components = {key: finite(info[key]) for key in OPERATING + PATIENT}
            if any(key.endswith("_cost") and key not in (*components, "base_cost", "specimen_route_cost", "transshipment_cost")
                   for key in info):
                raise ValueError("unknown cost primitive")
            cost = math.fsum(components.values())
            close(info["base_cost"], math.fsum(components[k] for k in OPERATING))
            close(finite(info["cost"]), cost)
            close(finite(row["cost"]), cost)
            close(finite(row["raw_reward"], signed=True), -cost)
            close(finite(info["specimen_route_cost"]), components["specimen_transfer_cost"])
            close(finite(info["transshipment_cost"]), math.fsum(components[k] for k in OPERATING[-3:]))
            before = _resources(row["resources_before"], n, lead)
            after = _resources(row["resources_after"], n, lead)
            if before != resources or set(before) != set(after):
                raise ValueError("continuous resource chain differs")
            _routing(info, request, n, scale)
            loss = sum(vector(info["patients_lost"], n))
            if sum(sum(vector(info[key], n)) for key in CAUSES) != loss:
                raise ValueError("patient loss causes disagree")
            losses += loss
            completed += sum(vector(info["patients_completed"], n))
            waiting += sum(vector(info["waiting_patients"], n))
            costs.append(cost)
            token, resources = row["after_state_sha256"], after
        if (token != json_hash(final) or final_counts["active"] != 0
                or losses != final_counts["lost"] - initial_counts["lost"]
                or completed != final_counts["delivered"] - initial_counts["delivered"]
                or resources != _snapshot_resources(final, resources, n, lead)):
            raise ValueError("final registry/resource totals disagree with raw rows")
        verify_cohort_tail(prefix, final, rows[52 - t:],
            dict(prefix_final_sha256=json_hash(prefix), final_sha256=json_hash(final)),
            num_facilities=n, enrollment_steps=52, patient_resolution_steps=8, accounting_steps=11)
        for name, values in (("remaining_raw_cost", costs), ("prefix_raw_cost", costs[:52 - t]),
                             ("tail_raw_cost", costs[52 - t:])):
            close(finite(receipt[name]), math.fsum(values))
        return dict(block=block, round=round_index, cohort=c, after_prefix_steps=t,
            candidate_index=k, replication=r, future_seed=str(seed),
            environment_calls=63 - t,
            remaining_raw_cost=math.fsum(costs), remaining_losses=losses,
            remaining_completions=completed, remaining_waiting_patient_steps=waiting,
            continuation_sha256=continuation_sha256, raw_rows_sha256=json_hash(rows),
            context_sha256=context["context_sha256"], label_scope="current_policy_remaining_cost",
            independent_policy_replay_performed=False)
    return verify


def assemble_round(contexts, indexes, config, streams, block, round_index, continuation_sha256):
    """Seal exactly six states with every independently read canonical branch."""
    validate_config(config)
    expected = {(c, t) for c in cohort_ids(round_index) for t in config["context_after_prefix_steps"]}
    if set(contexts) != expected:
        raise ValueError("incomplete round context inventory")
    lookup = {}
    for entry in indexes:
        row = entry["result"]
        if (row["block"] != block or row["round"] != round_index
                or row["continuation_sha256"] != continuation_sha256):
            raise ValueError("foreign continuation result")
        slot = tuple(row[k] for k in ("cohort", "after_prefix_steps", "replication", "candidate_index"))
        if slot in lookup:
            raise ValueError("duplicate branch outcome")
        lookup[slot] = row
    states, wanted = [], set()
    for c, t in sorted(expected):
        context = raw_json(contexts[c, t])
        # Context envelopes are plain data after decode_arrays in the caller.
        bank = context["public_example"]["candidates"]
        k = len(bank["class_keys"])
        seeds = [str(s) for s in streams["conditional_future"][f"block{block}/cohort{c}/after{t}"]]
        costs = []
        for r in range(4):
            row_costs = []
            for candidate in range(k):
                slot = (c, t, r, candidate)
                wanted.add(slot)
                row = lookup[slot]
                if row["future_seed"] != seeds[r] or row["context_sha256"] != context["context_sha256"]:
                    raise ValueError("future/context differs from declared matrix")
                row_costs.append(finite(row["remaining_raw_cost"]))
            costs.append(row_costs)
        states.append(dict(block=block, round=round_index, cohort=c, after_prefix_steps=t,
            public_example=context["public_example"], raw_costs=costs,
            replication_seed_ids=[list(seeds) for _ in range(k)], class_keys=bank["class_keys"]))
    if set(lookup) != wanted:
        raise ValueError("extra or missing canonical branches")
    result = dict(format="paired-cohort-dataset-v1", block=block, round=round_index,
        states=states, continuation_sha256=continuation_sha256,
        branch_index_sha256=digest(indexes), cost_objective="unchanged_remaining_absolute_cost")
    result["dataset_sha256"] = digest(result)
    return result
