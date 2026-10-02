"""Independent raw-only conditional-branch accounting and finite label assembly.

No simulator, model, actor, optimizer or collector imports. Inputs are decoded
NumPy/Python records or their lossless JSON equivalents. Source locks, file-byte
hashes, controller execution and durable budgets remain the enclosing runner's
responsibility; an action-source declaration is not independent policy replay.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np

from src.rl.candidate_pilot_verification import (
    CAUSES, OPERATING, PATIENT, close, count, finite, identity_counts, json_hash, vector,
)
from src.rl.cohort_verification import IMMUTABLE_PATIENT, verify_cohort_tail
from src.rl.dynamic_candidate_resource_verification import continuous_vector, executed_resources


RESOURCE_FIELDS = ("reagents", "bioreactors", "reagent_transfer_pipeline",
                   "capacity_transfer_pipeline", "reagent_purchase_pipeline")
LABEL_SCOPE = "remaining_cost_conditional_on_captured_latent_state_and_fixed_reference_continuation"


def raw_json(value):
    """Match the collector's evidence hash normalization, without importing it."""
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError("decoded raw Python/NumPy data required")
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("explicit lowercase SHA256 required")
    return value


def _seed(value):
    if type(value) is int and value >= 0:
        return str(value)
    if (type(value) is str and value.isascii() and value.isdecimal()
            and str(int(value)) == value and int(value) >= 0):
        return value
    raise ValueError("canonical nonnegative int/decimal-string seed required")


def _text(value):
    if type(value) is not str or not value.strip():
        raise ValueError("nonempty lineage string required")
    return value


def _config(config):
    expected = dict(blocks=[60, 61, 62], context_cohorts_per_block=4,
                    context_after_prefix_steps=[4, 20, 36], enrollment_steps=52,
                    economic_endpoint=63, future_replications=2, max_original_requests=6)
    if any(json_hash(config[k]) != json_hash(v) for k, v in expected.items()):
        raise ValueError("fixed paired-cohort label schedule differs")
    _text(config["rng_namespace"])


def _request(values, n):
    values = continuous_vector(values, 4 * n, signed=True)
    if any(abs(v) > 1 for v in values):
        raise ValueError("original request outside normalized coordinates")
    return values


def _lots(values, scale):
    scaled = np.asarray(values, dtype=np.float64) * scale
    return [int(v) for v in np.sign(scaled) * np.floor(np.abs(scaled) + .5)]


def _support(context, config):
    example = context["public_example"]
    if (set(example) != {"split", "identity", "actor_state", "candidates"}
            or example["split"] != "training" or example["identity"] != context["context_id"]):
        raise ValueError("only the bound public training example is allowed")
    for value in example["actor_state"]:
        finite(value, signed=True)
    bank = example["candidates"]
    if set(bank) != {"schema", "state_token", "requests", "class_keys", "members", "representatives", "request_to_class"}:
        raise ValueError("complete canonical candidate bank required")
    schema = bank["schema"]
    n, scale = schema["num_facilities"], finite(schema["max_specimen_transfer"])
    if (type(n) is not int or n < 1 or not 0 < scale <= 2**31 - 1
            or type(schema["max_candidates"]) is not int
            or not 2 <= len(bank["requests"]) <= schema["max_candidates"] <= config["max_original_requests"]
            or bank["state_token"] != context["environment_sha256"]):
        raise ValueError("candidate schema/count/source state differs")
    _text(schema["action_schema_id"])
    requests = [_request(row, n) for row in bank["requests"]]
    if any(row[n:] != requests[0][n:] for row in requests):
        raise ValueError("candidate support changes non-specimen request groups")
    keys = [tuple(_lots(row[:n], scale) + row[n:]) for row in requests]
    classes = sorted(set(keys))
    members = [[i for i, key in enumerate(keys) if key == cls] for cls in classes]
    representatives = [0 if 0 in group else 1 if 1 in group else min(group, key=lambda i: requests[i])
                       for group in members]
    mapping = [classes.index(key) for key in keys]
    for key in bank["class_keys"]:
        if len(key) != 4 * n:
            raise ValueError("canonical class key width differs")
        vector(key[:n], n, signed=True)
        continuous_vector(key[n:], 3 * n, signed=True)
    if (bank["class_keys"] != [list(k) for k in classes] or bank["members"] != members
            or bank["representatives"] != representatives or bank["request_to_class"] != mapping
            or any(type(i) is not int for i in bank["representatives"] + bank["request_to_class"]
                   + [i for group in bank["members"] for i in group])):
        raise ValueError("support canonicalization/representatives differ")
    actor_state = example["actor_state"]
    if (len(actor_state) <= 4 * n or not np.array_equal(
            np.asarray(actor_state[-4 * n:], dtype=np.float32), np.asarray(requests[1], dtype=np.float32))):
        raise ValueError("public actor anchor differs from candidate bank")
    return bank, n, scale


def _context(context, config, expected_context_seed):
    body = dict(context)
    seal = body.pop("context_sha256")
    if (context["format"] != "paired-cohort-context-v1" or json_hash(body) != _sha(seal)
            or json_hash(context["environment"]) != _sha(context["environment_sha256"])
            or type(context["block"]) is not int or context["block"] not in config["blocks"]
            or type(context["cohort"]) is not int or not 0 <= context["cohort"] < config["context_cohorts_per_block"]
            or type(context["after_prefix_steps"]) is not int
            or context["after_prefix_steps"] not in config["context_after_prefix_steps"]
            or context["source_id"] != config["rng_namespace"]):
        raise ValueError("context hash, identity or prescribed training cohort differs")
    _text(context["context_id"])
    _sha(context["reference_sha256"])
    spec = dict(enrollment_steps=52, patient_resolution_steps=8, accounting_steps=11,
                followup_rule="full_mdl2_until_resolution_then_no_new_commitments")
    if json_hash(context["cohort_spec"]) != json_hash(spec):
        raise ValueError("unchanged common tail contract required")
    scalars = context["environment"]["scalars"]
    if (type(scalars["t"]) is not int or scalars["t"] != context["after_prefix_steps"]
            or _seed(scalars["_episode_seed"]) != _seed(expected_context_seed)):
        raise ValueError("captured clock/context-world seed differs")
    bank, n, scale = _support(context, config)
    anchor = context["anchor_config"]
    if (anchor["num_facilities"] != n or anchor["episode_horizon"] != 52
            or anchor["max_specimen_transfer"] != scale):
        raise ValueError("public candidate/anchor contract differs")
    return bank, n, scale


def _identities(before, after):
    if not set(before["patients"]) <= set(after["patients"]):
        raise ValueError("captured patient identities disappeared")
    for pid, patient in before["patients"].items():
        other = after["patients"][pid]
        if any(patient[key] != other[key] for key in IMMUTABLE_PATIENT):
            raise ValueError("captured immutable patient attributes changed")
        if patient["status"] in ("delivered", "lost") and patient != other:
            raise ValueError("previously resolved patient changed after branch start")


def _resources(values, n, lead):
    if not set(RESOURCE_FIELDS[:4]) <= set(values) or set(values) - set(RESOURCE_FIELDS):
        raise ValueError("complete raw resource inventory required")
    result = {}
    for key, value in values.items():
        if key == "reagents":
            result[key] = continuous_vector(value, n)
        else:
            if not isinstance(value, list) or (key == "bioreactors" and len(value) != n):
                raise ValueError("resource stock/pipeline layout differs")
            result[key] = [continuous_vector(row, lead if key == "bioreactors" else n) for row in value]
    return result


def _snapshot_resources(state, keys, n, lead):
    values = {k: state["arrays"].get(k, [[0.] * n]) for k in keys}
    if any(k not in state["arrays"] and k != "reagent_purchase_pipeline" for k in keys):
        raise ValueError("snapshot omits required resource stock")
    for row in state["arrays"]["specimen_transfer_pipeline"]:
        vector(row, n)
    return _resources(values, n, lead)


def _routing(info, request, n, scale):
    requested = vector(info["specimen_requested_integer_net"], n, signed=True)
    executed = executed_resources(info, n)
    actual = executed["specimen_transfers"]
    routes = count(info["specimen_route_count"])
    blocked = count(info["blocked_specimen_requests"])
    if (requested != _lots(request[:n], scale) or sum(max(v, 0) for v in actual) != routes
            or any((r >= 0 and not 0 <= a <= r) or (r < 0 and not r <= a <= 0)
                   for r, a in zip(requested, actual))):
        raise ValueError("original requested/realized specimen arithmetic differs")
    inbound = sum(max(v, 0) for v in requested) - routes
    outbound = sum(max(-v, 0) for v in requested) - routes
    if (blocked != max(inbound, outbound) or count(info["blocked_specimen_inbound_requests"]) != inbound
            or count(info["blocked_specimen_outbound_requests"]) != outbound):
        raise ValueError("blocked specimen counts differ")
    return executed, routes, blocked


def verify_paired_cohort_branch(context, manifest, initial_state, prefix_final, final_state, rows, receipt,
                               *, config, expected_context_seed, expected_future_seed):
    """Verify one complete branch, returning remaining (not new-episode) outcomes.

    For JSON rows require ``action_dtype='float64'``. For decoded NumPy rows
    also inspect actual dtype. No rounding, float32 conversion or sunk-cost
    addition is performed. Expected seeds must come from the locked schedule,
    not be read back from this branch's own manifest.
    """
    if not isinstance(rows, (list, tuple)):
        raise ValueError("finite complete raw row sequence required")
    for row in rows:
        action = row["action"]
        if isinstance(action, np.ndarray):
            if action.dtype != np.float64 or action.ndim != 1:
                raise ValueError("recorded original action must be float64")
        elif row.get("action_dtype") != "float64":
            raise ValueError("JSON actions require explicit action_dtype=float64 provenance")
        if row.get("action_dtype", "float64") != "float64":
            raise ValueError("original action dtype declaration differs")
    context, manifest, initial_state, prefix_final, final_state, rows, receipt, config = raw_json(
        [context, manifest, initial_state, prefix_final, final_state, rows, receipt, config])
    _config(config)
    bank, n, scale = _context(context, config, expected_context_seed)
    start, end = context["after_prefix_steps"], config["economic_endpoint"]
    k, r = manifest["candidate_index"], manifest["replication"]
    if (manifest["format"] != "paired-cohort-branch-v1" or receipt["manifest"] != manifest
            or any(json_hash(manifest[key]) != json_hash(context[key]) for key in
                   ("context_sha256", "context_id", "source_id", "block", "cohort", "after_prefix_steps", "reference_sha256"))
            or type(k) is not int or not 0 <= k < len(bank["class_keys"])
            or type(r) is not int or not 0 <= r < config["future_replications"]
            or type(manifest["future_seed"]) is not str or manifest["future_seed"] != _seed(expected_future_seed)
            or type(manifest["endpoint"]) is not int or manifest["endpoint"] != end
            or manifest["future_rng_only"] is not True
            or manifest["event_aligned_crn_after_divergence_claimed"] is not False):
        raise ValueError("branch manifest/receipt/slot/future seed differs")
    _text(manifest["branch_id"])
    if manifest["first_request"] != bank["requests"][bank["representatives"][k]]:
        raise ValueError("first action is not original canonical representative")
    expected_initial = copy.deepcopy(context["environment"])
    expected_initial["rng_state"] = np.random.PCG64(int(_seed(expected_future_seed))).state
    if json_hash(initial_state) != json_hash(expected_initial):
        raise ValueError("conditional branch changed non-RNG context state or future generator")
    if (json_hash(initial_state) != _sha(manifest["initial_state_sha256"])
            or json_hash(final_state) != _sha(receipt["final_state_sha256"])
            or json_hash(rows) != _sha(receipt["raw_rows_sha256"])
            or len(rows) != end - start or type(receipt["steps"]) is not int or receipt["steps"] != len(rows)
            or receipt["included_sunk_cost"] is not False or type(receipt["terminal_active"]) is not int
            or receipt["terminal_active"] != 0
            or prefix_final["scalars"]["t"] != 52 or final_state["scalars"]["t"] != end):
        raise ValueError("raw hashes, complete row counts or endpoint receipt differ")
    for state in (initial_state, prefix_final, final_state):
        if type(state["scalars"]["t"]) is not int or _seed(state["scalars"]["_episode_seed"]) != _seed(expected_context_seed):
            raise ValueError("branch must retain original context-world seed and clock")
    initial, prefix, final = (identity_counts(s) for s in (initial_state, prefix_final, final_state))
    _identities(initial_state, prefix_final)
    _identities(prefix_final, final_state)
    enrolled_by_step = {t: [0] * n for t in range(start, 52)}
    for pid in set(prefix_final["patients"]) - set(initial_state["patients"]):
        patient = prefix_final["patients"][pid]
        epoch, facility = patient["enrollment_epoch"], patient["collection_facility"]
        if (type(epoch) is not int or not start <= epoch < 52
                or type(facility) is not int or not 0 <= facility < n):
            raise ValueError("new identity enrolled outside remaining prefix")
        enrolled_by_step[epoch][facility] += 1
    lead = context["anchor_config"]["production_lead_time"]
    if type(lead) is not int or lead < 1:
        raise ValueError("positive production resource width required")
    resources = _snapshot_resources(initial_state, rows[0]["resources_before"], n, lead)
    token = json_hash(initial_state)
    components = {key: [] for key in OPERATING + PATIENT}
    causes = dict.fromkeys(CAUSES, 0)
    enrolled, lost, completed = initial["enrolled"], initial["lost"], initial["delivered"]
    costs, actions, arrivals = [], [], []
    waiting = routes = blocked = 0
    for i, row in enumerate(rows):
        t = start + i
        if t == 52:
            if token != json_hash(prefix_final):
                raise ValueError("prefix-final snapshot not bound to last prefix row")
            closed = copy.deepcopy(prefix_final)
            for field in ("demand", "demand_forecast"):
                closed["arrays"][field] = [0.] * n
            closed["scalars"]["demand_forecast_error"] = 0.
            token = json_hash(closed)
        if (type(row["absolute_step"]) is not int or row["absolute_step"] != t
                or type(row["branch_step"]) is not int or row["branch_step"] != i
                or row["stage"] != ("prefix" if t < 52 else "tail")
                or row["action_source"] != ("candidate" if i == 0 else "fixed_r4" if t < 52 else "common_tail")
                or _sha(row["before_state_sha256"]) != token
                or _sha(row["after_state_sha256"]) == token):
            raise ValueError("raw branch clock, controller sequence or state chain differs")
        request = _request(row["action"], n)
        if i == 0 and request != manifest["first_request"]:
            raise ValueError("recorded first action changed original float64 representative")
        if not row["public_observation"]:
            raise ValueError("recorded public observation required")
        for v in row["public_observation"]:
            finite(v, signed=True)
        before = _resources(row["resources_before"], n, lead)
        after = _resources(row["resources_after"], n, lead)
        if before != resources or set(before) != set(after):
            raise ValueError("resource stock/pipeline chain differs")
        info = row["info"]
        primitive = {key: finite(info[key]) for key in components}
        if any(key.endswith("_cost") and key not in (*components, "base_cost", "specimen_route_cost", "transshipment_cost")
               for key in info):
            raise ValueError("unknown raw cost component")
        cost = finite(math.fsum(primitive.values()))
        close(finite(info["base_cost"]), math.fsum(primitive[k] for k in OPERATING))
        close(finite(info["cost"]), cost)
        close(finite(info["specimen_route_cost"]), primitive["specimen_transfer_cost"])
        close(finite(info["transshipment_cost"]), math.fsum(primitive[k] for k in OPERATING[-3:]))
        if finite(row["cost"]) != info["cost"] or finite(row["raw_reward"], signed=True) != -info["cost"]:
            raise ValueError("negative raw reward/cost receipt differs")
        for key, value in primitive.items():
            components[key].append(value)
        demand = vector(info["demand"], n)
        if t < 52 and demand != enrolled_by_step[t]:
            raise ValueError("raw arrivals differ from new patient enrollment epochs/facilities")
        if t >= 52 and any(demand):
            raise ValueError("post-window branch enrollment")
        arrivals.append(demand)
        enrolled += sum(demand)
        loss = sum(vector(info["patients_lost"], n))
        served = sum(vector(info["patients_completed"], n))
        per_cause = {key: sum(vector(info[key], n)) for key in CAUSES}
        if sum(per_cause.values()) != loss:
            raise ValueError("disjoint patient loss causes differ")
        for key, value in per_cause.items():
            causes[key] += value
        lost, completed = lost + loss, completed + served
        active = count(info["identity_active_count"])
        if active != enrolled - lost - completed or count(info["identity_terminal_count"]) != lost + completed:
            raise ValueError("remaining-prefix patient conservation differs from captured counts")
        located = {key: sum(vector(info[key], n)) for key in
                   ("waiting_patients", "in_production_patients", "specimen_in_transit")}
        if sum(located.values()) > active:
            raise ValueError("located patient counts exceed active registry")
        waiting += located["waiting_patients"]
        vector(info["production"], n)
        executed, nroutes, nblocked = _routing(info, request, n, scale)
        routes, blocked = routes + nroutes, blocked + nblocked
        actions.append(dict(absolute_step=t, requested=request, executed=executed,
                            specimen_requested_integer_net=info["specimen_requested_integer_net"]))
        costs.append(cost)
        token, resources = row["after_state_sha256"], after
        if t in (51, 62):
            boundary, counts = (prefix_final, prefix) if t == 51 else (final_state, final)
            if (token != json_hash(boundary) or resources != _snapshot_resources(boundary, resources, n, lead)
                    or (enrolled, lost, completed, active) !=
                       (counts["enrolled"], counts["lost"], counts["delivered"], counts["active"])
                    or any(located[k] != counts[c] for k, c in
                           (("waiting_patients", "waiting"), ("in_production_patients", "production"),
                            ("specimen_in_transit", "specimen_transit")))):
                raise ValueError("raw boundary identities, resources or snapshot hash differ")
    tail = verify_cohort_tail(prefix_final, final_state, rows[52 - start:],
        dict(prefix_final_sha256=json_hash(prefix_final), final_sha256=json_hash(final_state)),
        num_facilities=n, enrollment_steps=52, patient_resolution_steps=8, accounting_steps=11)
    prefix_cost, tail_cost = math.fsum(costs[:52 - start]), math.fsum(costs[52 - start:])
    remaining = finite(math.fsum(costs))
    for key, value in (("remaining_raw_cost", remaining), ("prefix_raw_cost", prefix_cost), ("tail_raw_cost", tail_cost)):
        close(finite(receipt[key]), value)
    turnarounds = []
    for state in (initial_state, final_state):
        total = math.fsum(finite(p["age"]) for p in state["patients"].values() if p["status"] == "delivered")
        close(finite(state["scalars"]["cumulative_turnaround_time"]), total)
        turnarounds.append(total)
    close(finite(rows[-1]["info"]["average_turnaround_time"]), turnarounds[1] / max(1, final["delivered"]))
    close(finite(rows[-1]["info"]["completion_service_level"]), final["delivered"] / max(1, final["enrolled"]))
    return dict(format="independent-paired-cohort-branch-v1", **{key: manifest[key] for key in
        ("branch_id", "context_id", "block", "cohort", "after_prefix_steps", "candidate_index", "replication", "future_seed")},
        context_sha256=context["context_sha256"], manifest_sha256=json_hash(manifest),
        raw_rows_sha256=json_hash(rows), initial_state_sha256=json_hash(initial_state),
        prefix_final_sha256=json_hash(prefix_final), final_state_sha256=json_hash(final_state),
        receipt_sha256=json_hash(receipt), steps=len(rows), remaining_raw_cost=remaining,
        prefix_raw_cost=prefix_cost, tail_raw_cost=tail_cost,
        components={key: finite(math.fsum(values)) for key, values in components.items()},
        initial_compartments=initial, final_compartments=final, remaining_enrolled=final["enrolled"] - initial["enrolled"],
        remaining_losses=final["lost"] - initial["lost"], remaining_completions=final["delivered"] - initial["delivered"],
        remaining_turnaround_sum=turnarounds[1] - turnarounds[0], waiting_patient_steps=waiting,
        loss_causes=causes, arrivals=arrivals, route_count=routes, blocked_requests=blocked, actions=actions,
        tail=tail, label_scope=LABEL_SCOPE, included_sunk_cost=False, full_new_episode_outcome=False,
        event_aligned_crn_claimed=False, controller_requests_independently_replayed=False,
        resource_check_scope="continuous stocks/pipeline chain and executed net conservation; no simulator replay")


def assemble_paired_cohort_labels(contexts, branches, *, config, context_seeds, future_seeds):
    """Verify the exact 36-context x all-classes x two-replications raw matrix.

    ``context_seeds`` maps decimal block strings to four locked context seeds;
    ``future_seeds`` maps every context_id to its two locked future seeds.
    Each branch is a dict with manifest, initial_state, prefix_final, final_state,
    rows and receipt (the seven-argument verifier's context is joined here).
    Returned read-only NumPy float64 [R,K] matrices can be copied into the actor;
    public examples remain separate from hidden-state branch outcomes.
    """
    config = raw_json(config)
    _config(config)
    if not isinstance(contexts, (list, tuple)) or not isinstance(branches, (list, tuple)):
        raise ValueError("finite explicit context/branch sequences required")
    contexts = raw_json(contexts)
    expected = {(b, c, t) for b in config["blocks"] for c in range(config["context_cohorts_per_block"])
                for t in config["context_after_prefix_steps"]}
    if (set(context_seeds) != {str(b) for b in config["blocks"]}
            or any(not isinstance(v, (list, tuple)) or len(v) != 4 for v in context_seeds.values())):
        raise ValueError("exact locked context-world seed inventory required")
    context_seeds = {b: [_seed(s) for s in seeds] for b, seeds in context_seeds.items()}
    indexed, slots, banks = {}, set(), {}
    for context in contexts:
        slot = tuple(context[k] for k in ("block", "cohort", "after_prefix_steps"))
        cid = context["context_id"]
        if slot not in expected or slot in slots or cid in indexed:
            raise ValueError("duplicate or forbidden training context/cohort")
        bank, _, _ = _context(context, config, context_seeds[str(slot[0])][slot[1]])
        indexed[cid], banks[cid] = context, bank
        slots.add(slot)
    if slots != expected or set(future_seeds) != set(indexed):
        raise ValueError("missing prescribed contexts or future-seed declarations")
    for block in config["blocks"]:
        if len({c["reference_sha256"] for c in contexts if c["block"] == block}) != 1:
            raise ValueError("reference changed within a training block")
    futures = {}
    for cid, values in future_seeds.items():
        if not isinstance(values, (list, tuple)) or len(values) != config["future_replications"]:
            raise ValueError("exact two future replications required")
        futures[cid] = tuple(_seed(s) for s in values)
    all_seeds = [s for values in context_seeds.values() for s in values] + [s for values in futures.values() for s in values]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("reused context/future seed in declared fresh matrix")
    expected_branches = {(cid, k, r) for cid, bank in banks.items() for k in range(len(bank["class_keys"]))
                         for r in range(config["future_replications"])}
    verified, branch_ids = {}, set()
    fields = {"manifest", "initial_state", "prefix_final", "final_state", "rows", "receipt"}
    for branch in branches:
        if not isinstance(branch, dict) or set(branch) != fields:
            raise ValueError("complete raw branch payload required")
        manifest = branch["manifest"]
        slot = manifest["context_id"], manifest["candidate_index"], manifest["replication"]
        if slot not in expected_branches or slot in verified or manifest["branch_id"] in branch_ids:
            raise ValueError("duplicate or unexpected branch/class/replication")
        context = indexed[slot[0]]
        verified[slot] = verify_paired_cohort_branch(context, **branch, config=config,
            expected_context_seed=context_seeds[str(context["block"])][context["cohort"]],
            expected_future_seed=futures[slot[0]][slot[2]])
        branch_ids.add(manifest["branch_id"])
    if set(verified) != expected_branches:
        raise ValueError("missing class/replication branch; incomplete labels forbidden")
    labels, outcomes = [], []
    for context in sorted(contexts, key=lambda c: (c["block"], c["cohort"], c["after_prefix_steps"])):
        cid, bank = context["context_id"], banks[context["context_id"]]
        matrix = np.array([[verified[cid, k, r]["remaining_raw_cost"] for k in range(len(bank["class_keys"]))]
                           for r in range(config["future_replications"])], dtype=np.float64)
        matrix.setflags(write=False)
        labels.append(dict(context_id=cid, block=context["block"], cohort=context["cohort"],
            after_prefix_steps=context["after_prefix_steps"], context_sha256=context["context_sha256"],
            public_example=copy.deepcopy(context["public_example"]), raw_costs=matrix,
            class_keys=tuple(tuple(k) for k in bank["class_keys"]), reference_index=bank["request_to_class"][0],
            replication_seed_ids=tuple(futures[cid] for _ in bank["class_keys"])))
        outcomes.extend(verified[cid, k, r] for k in range(len(bank["class_keys"]))
                        for r in range(config["future_replications"]))
    binding = dict(format="paired-cohort-label-dataset-v1", config_sha256=json_hash(config),
        context_seeds=context_seeds, future_seeds=futures, labels=labels, branch_outcomes=outcomes,
        label_scope=LABEL_SCOPE, full_new_episode_outcome=False, event_aligned_crn_claimed=False)
    return binding | {"dataset_sha256": json_hash(raw_json(binding))}
