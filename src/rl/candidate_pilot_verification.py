"""Independent P1 raw-cost/outcome arithmetic. No environment or learner imports.

Consumes JSON-normalized raw receipts and initial/final simulator snapshots,
not a learner-written episode summary. Does not simulate, select or fit policies.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


# Deliberately independent of the collector's cost constants/summary code.
OPERATING = ("reagent_purchase_cost", "reagent_holding_cost", "reagent_shortage_cost",
             "bioreactor_holding_cost", "bioreactor_shortage_cost", "specimen_transfer_cost",
             "capacity_transfer_cost", "reagent_transfer_cost")
PATIENT = ("patient_loss_cost", "expiry_cost", "urgency_cost")
CAUSES = ("patients_lost_waiting_ineligible", "patients_lost_manufacturing",
          "patients_lost_waiting_expired", "patients_lost_transit_ineligible",
          "patients_lost_transit_expired", "patients_lost_transit_transport",
          "patients_lost_return_ineligible", "finished_expired")


def json_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value, *, signed=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (not signed and value < 0):
        raise ValueError("nonfinite/negative/non-numeric raw outcome")
    return value


def count(value):
    value = finite(value)
    if value != int(value):
        raise ValueError("non-integer patient/route count")
    return int(value)


def vector(values, n, *, signed=False):
    if not isinstance(values, (list, tuple)) or len(values) != n:
        raise ValueError("raw vector width mismatch")
    result = [finite(v, signed=signed) for v in values]
    if any(v != int(v) for v in result):
        raise ValueError("non-integer raw vector")
    return [int(v) for v in result]


def close(left, right):
    if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError("raw accounting does not reconcile")


def identity_counts(state):
    """Recount raw patient IDs/locations without calling simulator assertions."""
    patients, locations = state["patients"], {}
    counts = Counter()

    def locate(pid, kind, expected_status, facility=None):
        if pid not in patients or pid in locations or patients[pid]["status"] != expected_status:
            raise ValueError("unknown, duplicated or incorrectly located active identity")
        patient = patients[pid]
        if kind == "waiting" and patient["material_facility"] != facility:
            raise ValueError("waiting identity at wrong facility")
        if kind == "production" and patient["manufacturing_facility"] != facility:
            raise ValueError("production identity at wrong facility")
        if kind == "specimen_transit" and patient["material_facility"] is not None:
            raise ValueError("in-transit identity has a material facility")
        locations[pid] = kind
        counts[kind] += 1

    for site, queue in enumerate(state["patient_queues"]):
        for pid in queue:
            locate(pid, "waiting", "waiting", site)
    for site, stages in enumerate(state["in_production_patients"]):
        if stages[0]:
            raise ValueError("unexpected production stage-zero identities")
        for stage in stages[1:]:
            for pid in stage:
                locate(pid, "production", "in_production", site)
    for transit in state["specimen_transits"]:
        locate(transit["patient_id"], "specimen_transit", "in_transit")
    for transit in state["product_return_transits"]:
        locate(transit["patient_id"], "finished_return", "finished")
    for pid, patient in patients.items():
        if pid != patient["patient_id"] or pid != patient["specimen_id"]:
            raise ValueError("patient/specimen identity substitution")
        status = patient["status"]
        if status in ("lost", "delivered"):
            if pid in locations:
                raise ValueError("terminal patient still active")
            counts[status] += 1
        elif pid not in locations:
            raise ValueError("active patient missing from physical compartments")
    if count(state["scalars"]["cumulative_enrolled"]) != len(patients):
        raise ValueError("enrollment counter differs from unique registry")
    if (count(state["scalars"]["cumulative_lost"]) != counts["lost"]
            or count(state["scalars"]["cumulative_served"]) != counts["delivered"]):
        raise ValueError("terminal cumulative counters differ from patient IDs")
    return {"enrolled": len(patients), "active": len(locations),
            **{k: counts[k] for k in ("waiting", "production", "specimen_transit", "finished_return", "lost", "delivered")}}


def verify_episode(header, rows, final_state, config):
    """Each row wraps one raw session event and measured decision-path seconds."""
    objective = config["objective"]
    horizon, n = objective["horizon"], objective["num_facilities"]
    if len(rows) != horizon:
        raise ValueError("partial or extra episode rows")
    initial_state = header["initial_state"]
    if initial_state["scalars"]["t"] != 0 or final_state["scalars"]["t"] != horizon:
        raise ValueError("wrong raw horizon boundaries")
    if initial_state["scalars"]["_episode_seed"] != header["seed"] or final_state["scalars"]["_episode_seed"] != header["seed"]:
        raise ValueError("raw seed mismatch")
    initial, final = identity_counts(initial_state), identity_counts(final_state)
    if not set(initial_state["patients"]).issubset(final_state["patients"]):
        raise ValueError("registered identities disappeared")
    if initial["lost"] or initial["delivered"]:
        raise ValueError("new episode has pre-resolved identities")
    cost = lost = completed = routes = blocked = arrivals = 0
    component_totals = {k: 0. for k in OPERATING + PATIENT}
    causes = {k: 0 for k in CAUSES}
    class_counts, selected_classes, actual_flows, demands = [], set(), set(), []
    reference_choices, seconds = 0, 0.
    token, previous_state = json_hash(initial_state), None
    for step, row in enumerate(rows):
        event = row["event"]
        info, audit = event["info"], event["audit"]
        record, decision = audit["record"], audit["decision"]
        semantics = record["semantics"]
        if (record["origin"] != "trajectory" or record["source_id"] != header["source_id"]
                or record["trajectory_id"] != header["trajectory_id"] or record["step_index"] != step
                or record["state_token"] != token or record["next_state_token"] == token
                or (previous_state is not None and record["state"] != previous_state)
                or record["terminated"] is not (step + 1 == horizon) or record["truncated"] is not False
                or semantics["reward_kind"] != "absolute_environment"
                or semantics["reward_scale"] != objective["reward_scale"] or semantics["gamma"] != objective["gamma"]):
            raise ValueError("raw trajectory/reward/termination contract differs")
        total, base = finite(info["cost"]), finite(info["base_cost"])
        components = {k: finite(info[k]) for k in OPERATING + PATIENT}
        unknown = [k for k in info if k.endswith("_cost") and k not in components
                   and k not in ("base_cost", "specimen_route_cost", "transshipment_cost")]
        if unknown:
            raise ValueError("unknown cost components")
        close(base, sum(components[k] for k in OPERATING))
        close(total, sum(components.values()))
        close(finite(info["specimen_route_cost"]), components["specimen_transfer_cost"])
        close(finite(info["transshipment_cost"]), sum(components[k] for k in OPERATING[-3:]))
        if finite(record["raw_reward"], signed=True) != -total or audit["terminal_cost_added"] != 0:
            raise ValueError("absolute cost reward or terminal-cost mismatch")
        for k in component_totals:
            component_totals[k] += components[k]
        cost += total
        lost_step = sum(vector(info["patients_lost"], n))
        delivered_step = sum(vector(info["patients_completed"], n))
        cause_values = {k: sum(vector(info[k], n)) for k in CAUSES}
        if sum(cause_values.values()) != lost_step:
            raise ValueError("disjoint loss causes do not sum to losses")
        for k, v in cause_values.items():
            causes[k] += v
        lost += lost_step
        completed += delivered_step
        demand = vector(info["demand"], n)
        demands.append(demand)
        arrivals += sum(demand)
        active, terminal = count(info["identity_active_count"]), count(info["identity_terminal_count"])
        if active + terminal != initial["enrolled"] + arrivals or terminal != lost + completed:
            raise ValueError("per-step identity conservation/counters disagree")
        located = sum(sum(vector(info[k], n)) for k in ("waiting_patients", "in_production_patients", "specimen_in_transit"))
        if located > active or (step + 1 == horizon and active != final["active"]):
            raise ValueError("active identity compartments disagree")
        request = record["action"]
        if len(request) != objective["action_width"] or any(abs(finite(v, signed=True)) > 1 for v in request):
            raise ValueError("invalid original normalized action")
        scaled = np.asarray(request[:n]) * objective["transfer_scale"]
        expected = [int(v) for v in np.sign(scaled) * np.floor(np.abs(scaled) + .5)]
        requested, actual = vector(info["specimen_requested_integer_net"], n, signed=True), vector(info["specimen_transfers"], n, signed=True)
        route_count, blocked_count = count(info["specimen_route_count"]), count(info["blocked_specimen_requests"])
        if (requested != expected or sum(actual) != 0 or sum(max(v, 0) for v in actual) != route_count
                or any((r >= 0 and not 0 <= a <= r) or (r < 0 and not r <= a <= 0) for r, a in zip(requested, actual))):
            raise ValueError("requested/executed routing arithmetic differs")
        inbound, outbound = sum(max(v, 0) for v in requested) - route_count, sum(max(-v, 0) for v in requested) - route_count
        if (blocked_count != max(inbound, outbound) or count(info["blocked_specimen_inbound_requests"]) != inbound
                or count(info["blocked_specimen_outbound_requests"]) != outbound):
            raise ValueError("blocked-routing counts differ")
        bank, choice = decision["evaluation"]["candidates"], decision["choice"]
        classes = [tuple(key) for key in bank["class_keys"]]
        requests = bank["requests"]
        if not 2 <= len(requests) <= config["candidate_support"]["max_original_requests"]:
            raise ValueError("candidate support count differs")
        keys = []
        for original in requests:
            if len(original) != objective["action_width"] or list(original[n:]) != list(requests[0][n:]):
                raise ValueError("non-specimen candidate difference or action width mismatch")
            if any(abs(finite(v, signed=True)) > 1 for v in original):
                raise ValueError("candidate outside normalized coordinates")
            lots = np.asarray(original[:n]) * objective["transfer_scale"]
            keys.append(tuple(int(v) for v in np.sign(lots) * np.floor(np.abs(lots) + .5)) + tuple(original[n:]))
        if classes != sorted(set(keys)) or list(bank["request_to_class"]) != [classes.index(k) for k in keys]:
            raise ValueError("raw support canonicalization differs")
        index = choice["class_index"]
        if classes != sorted(set(classes)) or type(index) is not int or not 0 <= index < len(classes):
            raise ValueError("noncanonical or invalid request class")
        if tuple(requested) + tuple(request[n:]) != classes[index] or list(choice["submitted_request"]) != list(request):
            raise ValueError("selected class or original request changed")
        members = [i for i, key in enumerate(keys) if key == classes[index]]
        representative = 0 if 0 in members else 1 if 1 in members else min(members, key=lambda i: tuple(requests[i]))
        if list(request) != list(requests[representative]):
            raise ValueError("original canonical representative was substituted")
        log_probs = [finite(v, signed=True) for v in decision["evaluation"]["log_probs"]]
        if decision["evaluation"]["inference_dtype"] != "float32":
            raise ValueError("P1 requires float32 behavior receipts")
        tolerance = 8 * np.finfo(np.float32).eps
        if len(log_probs) != len(classes) or max(log_probs) > tolerance:
            raise ValueError("invalid behavior class probabilities")
        if not math.isclose(math.fsum(math.exp(v) for v in log_probs), 1., rel_tol=0, abs_tol=tolerance):
            raise ValueError("class probabilities do not normalize at float32 tolerance")
        if header["split"] == "test" and header["role"] in ("frozen", "ppo", "bc_continue") and index != int(np.argmax(log_probs)):
            raise ValueError("final evaluation did not use fixed greedy canonical class")
        if header["role"] == "r4" and list(request) != list(bank["requests"][0]):
            raise ValueError("reference baseline did not retain original R4 request")
        if header["role"] == "mdl2" and list(request) != list(bank["requests"][1]):
            raise ValueError("anchor baseline did not retain full MDL-2 request")
        if decision["evaluation"]["behavior_sha256"] != header["policy_sha256"]:
            raise ValueError("episode policy snapshot changed")
        routes += route_count
        blocked += blocked_count
        class_counts.append(len(classes))
        selected_classes.add(classes[index])
        actual_flows.add(tuple(actual))
        reference_choices += int(index == bank["request_to_class"][0])
        seconds += finite(row["inference_seconds"])
        token, previous_state = record["next_state_token"], record["next_state"]
    if token != json_hash(final_state):
        raise ValueError("final snapshot does not match raw receipt endpoint")
    if lost != final["lost"] or completed != final["delivered"] or final["enrolled"] != initial["enrolled"] + arrivals:
        raise ValueError("episode patient outcomes do not match final registry")
    close(finite(rows[-1]["event"]["info"]["completion_service_level"]), completed / max(final["enrolled"], 1))
    return {k: header[k] for k in ("block", "representation", "role", "world_index", "seed", "trajectory_id", "split")} | {
        "steps": horizon, "cost": cost, "components": component_totals, "losses": lost,
        "completions": completed, "terminal_active": final["active"], "terminal_compartments": final,
        "loss_causes": causes, "initial_enrolled": initial["enrolled"], "enrolled": final["enrolled"],
        "arrivals": demands, "completion_service_level": completed / max(final["enrolled"], 1),
        "route_count": routes, "blocked_requests": blocked, "support_counts": class_counts,
        "distinct_selected_request_classes": len(selected_classes), "distinct_executed_net_flows": len(actual_flows),
        "reference_class_choices": reference_choices, "inference_seconds": seconds,
        "initial_state_sha256": json_hash(initial_state), "final_state_sha256": json_hash(final_state),
        "policy_sha256": header["policy_sha256"], "raw_episode_sha256": json_hash([header, rows, final_state])}


def verify_raw_bundle(root, index, config, streams):
    """Reopen and hash every raw file, then independently recompute all outcomes."""
    files, outcomes = read_raw_episodes(root, index, config)
    analysis = paired_analysis(outcomes, config, streams)
    return {"files": files, "outcomes": outcomes, "analysis": analysis,
            "verification": "raw files reopened, hashes checked, all prescribed outcomes recomputed"}


def read_raw_episodes(root, index, config):
    """Shared raw-only reader for prescribed test and development episode sets."""
    root = Path(root).resolve()
    seen, files, outcomes = set(), [], []

    def read(record, jsonl=False):
        path = root / record["path"]
        if (path.is_symlink() or path.resolve() != path.absolute() or not path.is_file()
                or not path.resolve().is_relative_to(root) or record["path"] in seen):
            raise ValueError("missing, duplicate or redirected raw evidence")
        raw = path.read_bytes()
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("raw file size/hash mismatch")
        seen.add(record["path"])
        files.append(record)
        return [json.loads(line) for line in raw.decode().splitlines()] if jsonl else json.loads(raw)

    for entry in index:
        header = read(entry["header"])
        rows = read(entry["events"], jsonl=True)
        final = read(entry["final_state"])
        outcomes.append(verify_episode(header, rows, final, config))
    return files, outcomes


def paired_analysis(outcomes, config, streams):
    """Prespecified complete-case report. Missing/bad rows fail; never drop them."""
    blocks, worlds = config["blocks"], config["evaluation"]["episodes_per_policy"]
    if len(blocks) != 3:
        raise ValueError("P1's descriptive t interval requires exactly three blocks")
    policies = [(r["name"], role) for r in config["representations"] for role in config["candidate_roles"]]
    policies += [("reference", role) for role in config["reference_roles"]]
    expected = {(b, rep, role, w) for b in blocks for rep, role in policies for w in range(worlds)}
    indexed = {}
    for row in outcomes:
        key = row["block"], row["representation"], row["role"], row["world_index"]
        if key not in expected or key in indexed or row["split"] != "test":
            raise ValueError("duplicate, unexpected or non-test outcome")
        if row["seed"] != streams["environment"]["test"][str(row["block"])][row["world_index"]]:
            raise ValueError("wrong test stream")
        for metric in ("cost", "losses", "completions", "terminal_active", "enrolled"):
            finite(row[metric])
        if row["cost"] <= 0:
            raise ValueError("nonpositive cost denominator")
        indexed[key] = row
    if set(indexed) != expected or len(indexed) != config["evaluation"]["total_episodes"]:
        raise ValueError("missing prescribed final evaluation outcomes")
    if len({row["trajectory_id"] for row in outcomes}) != len(outcomes):
        raise ValueError("duplicate evaluation trajectory identity")
    for b in blocks:
        for world in range(worlds):
            if len({indexed[b, rep, role, world]["initial_state_sha256"] for rep, role in policies}) != 1:
                raise ValueError("paired policies do not share exact initial simulator states")
        for rep, role in policies:
            if len({indexed[b, rep, role, w]["policy_sha256"] for w in range(worlds)}) != 1:
                raise ValueError("test policy changed across evaluation worlds")
    contrasts = [("graph/ppo", other) for other in ("graph/frozen", "graph/bc_continue", "reference/r4", "reference/mdl2")]
    contrasts += [(f"{rep}/ppo", f"{rep}/{role}") for rep in ("self_only", "flat") for role in ("frozen", "bc_continue")]
    contrasts += [(f"graph/{role}", f"self_only/{role}") for role in config["candidate_roles"]]
    contrasts += [("graph/ppo", "flat/ppo")]
    rng = np.random.default_rng(streams["neural"]["analysis/bootstrap"])
    results = []
    for left, right in contrasts:
        lrep, lrole = left.split("/")
        rrep, rrole = right.split("/")
        estimates, all_worlds, baseline_means, left_means = [], [], [], []
        for b in blocks:
            lrows = [indexed[b, lrep, lrole, w] for w in range(worlds)]
            rrows = [indexed[b, rrep, rrole, w] for w in range(worlds)]
            differences = [{"world_index": w, "seed": l["seed"], **{m: l[m] - r[m] for m in
                ("cost", "losses", "completions", "terminal_active", "enrolled")}}
                for w, (l, r) in enumerate(zip(lrows, rrows))]
            delta = np.array([d["cost"] for d in differences], dtype=np.float64)
            indices = rng.integers(0, worlds, size=(config["evaluation"]["paired_world_bootstrap_draws"], worlds))
            interval = np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()
            baseline = float(np.mean([r["cost"] for r in rrows]))
            left_mean = float(np.mean([l["cost"] for l in lrows]))
            estimates.append({"block": b, "mean_cost_difference": float(delta.mean()),
                "relative_cost_gain_percent": 100 * (baseline - left_mean) / baseline,
                "paired_world_bootstrap_ci95": interval,
                "clinical_differences": {m: float(np.mean([d[m] for d in differences])) for m in
                                         ("losses", "completions", "terminal_active", "enrolled")}})
            all_worlds.append({"block": b, "differences": differences})
            baseline_means.append(baseline)
            left_means.append(left_mean)
        effects = np.array([e["mean_cost_difference"] for e in estimates])
        mean, sem = float(effects.mean()), float(effects.std(ddof=1) / math.sqrt(3))
        results.append({"left": left, "right": right, "blocks": estimates, "paired_worlds": all_worlds,
                        "grand_mean_cost_difference": mean,
                        "grand_relative_cost_gain_percent": 100 * (np.mean(baseline_means) - np.mean(left_means)) / np.mean(baseline_means),
                        "descriptive_t_ci95_df2": [mean - 4.302652729911275 * sem, mean + 4.302652729911275 * sem]})
    primary = results[0]
    checks = {"all_blocks_lower_cost": all(e["mean_cost_difference"] < 0 for e in primary["blocks"]),
              "grand_gain_at_least_threshold": primary["grand_relative_cost_gain_percent"] >= config["evaluation"]["practical_relative_cost_gain_percent"],
              "no_worse_secondary_mean_cost": all(c["grand_mean_cost_difference"] <= 0 for c in results[1:4]),
              "no_adverse_clinical_block_direction": all(
                  e["clinical_differences"]["losses"] <= 0 and e["clinical_differences"]["completions"] >= 0
                  and e["clinical_differences"]["terminal_active"] <= 0 for c in results[:4] for e in c["blocks"])}
    decision = ("promising_for_discussion_only" if all(checks.values()) else
                "cost_clinical_tradeoff" if primary["grand_mean_cost_difference"] < 0 and not checks["no_adverse_clinical_block_direction"]
                else "limited_negative_or_inconclusive")
    return {"format": "candidate-pilot-independent-analysis-v1", "evaluation_episodes": len(indexed),
            "independent_training_blocks": 3, "contrasts": results, "triage_checks": checks, "decision": decision,
            "event_level_crn_equality_claim": False, "deployment_adaptation_claim": False,
            "clean_ddpg_superiority_claim": False, "clinical_noninferiority_claim": False,
            "flat_parameter_matched": False, "automatic_followon": False,
            "limitations": ["three training blocks; t interval is descriptive and low power",
                            "within-block bootstrap does not increase independent training sample size",
                            "finite-window cost; unresolved patients are not deaths or free benefit",
                            "action-dependent RNG consumption may change arrivals and denominators",
                            "new candidate policies share a graph-based R4 reference"]}
