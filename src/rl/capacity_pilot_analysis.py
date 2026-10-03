"""Read-only reconstruction of the dynamic-capacity pilot's saved evidence.

No simulator, learner, model or training module is imported. ``run_analysis``
accepts the proposal mapping or its JSON path. It never writes an artifact.
Strict mode raises AnalysisIncomplete *after* building the salvageable report;
the same report is available as ``error.report``. Partial mode returns it.
Missing secondary fields never masquerade as zero or block primary contrasts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


COST_COMPONENTS = (
    "reagent_purchase_cost", "reagent_holding_cost", "reagent_shortage_cost",
    "bioreactor_holding_cost", "bioreactor_shortage_cost", "specimen_transfer_cost",
    "capacity_transfer_cost", "reagent_transfer_cost", "patient_loss_cost", "expiry_cost",
    "urgency_cost", "support_ordinary_staff_cost", "support_flexible_labor_cost",
    "support_switching_cost",
)
TERMINAL = {"delivered", "lost"}
STATUSES = TERMINAL | {"waiting", "in_transit", "in_production", "finished"}
ONLINE = "online_matched_fork"
LEARNED = ("frozen_history", "frozen_matched_exploration", ONLINE)
PHASES = ("teacher_bc_critic_warmup", "offline_ddpg", "six_arm_evaluation")
RAW_NAME = re.compile(r"^(" + "|".join(PHASES) + r")-b(\d+)-c(\d+)-j(\d+)-(.+)\.jsonl(?:\.gz)?$")


class AnalysisIncomplete(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("pilot primary evidence incomplete; use error.report or require_complete=False")


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def _count(value):
    return _finite(value) and value >= 0 and float(value).is_integer()


def _close(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-7)


def _digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(data)
    return hasher.hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _json(text):
    def reject(value):
        raise ValueError(f"non-JSON numeric constant: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result
    return json.loads(text, parse_constant=reject, object_pairs_hook=unique)


def _safe_path(root, value):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("evidence path is outside analysis root")
    return path


def _key(world, role):
    if (not isinstance(world, dict) or world.get("phase") not in PHASES
            or any(type(world.get(k)) is not int for k in ("block", "condition", "replicate", "seed"))
            or not isinstance(role, str)):
        raise ValueError("invalid world/role identity or non-native integer seed")
    return world["phase"], world["block"], world["condition"], world["replicate"], role


def _name(key):
    phase, b, c, j, role = key
    return f"{phase}-b{b}-c{c}-j{j}-{role}"


def _contract(proposal):
    p = copy.deepcopy(proposal) if isinstance(proposal, dict) else _json(Path(proposal).read_text())
    d, a, s = p["design"], p["analysis"], p["synthetic_system"]
    bootstrap = a["bootstrap"]
    if (bootstrap != {"seed": 62690001, "resamples": 2000,
                      "resample": "blocks_then_paired_worlds_within_block_condition",
                      "percentiles": [2.5, 97.5], "descriptive_only": True}
            or (d["blocks"], d["conditions"], d["worlds_per_condition_per_phase_block"], d["evaluation_arms"]) != (3, 3, 4, 6)
            or (s["control_epochs"], s["settlement_epochs"], s["host_total_horizon"]) != (48, 16, 64)
            or a["condition_weighting"] != "equal_thirds_alongside_separate_cells"):
        raise ValueError("unsupported pilot analysis numerical contract")
    roles = tuple(row["id"] for row in p["controllers"])
    expected_roles = (*LEARNED, "adaptive_rule", "id_mpc", "fixed_allocation_reference")
    if set(roles) != set(expected_roles) or len(roles) != 6:
        raise ValueError("six distinct registered roles required")
    contrasts = a["primary_contrasts"] + a["stronger_contrasts"]
    if contrasts != ["online_vs_frozen_history", "online_vs_frozen_matched_exploration",
                     "online_vs_adaptive_rule", "online_vs_id_mpc"]:
        raise ValueError("unsupported predeclared contrasts")
    if [r["index"] for r in p["conditions"]] != [0, 1, 2]:
        raise ValueError("condition indices differ")
    seed_contract = {PHASES[0]: (62600000, 10, "62600000 + 1000*b + 10*c + j", "teacher"),
                     PHASES[1]: (62604000, 10, "62604000 + 1000*b + 10*c + j", "learner"),
                     PHASES[2]: (62610000, 100, "62610000 + 1000*b + 100*c + j", "evaluation")}
    expected = {}
    for phase, (base, stride, formula, seed_name) in seed_contract.items():
        if d[f"{seed_name}_world_seed"] != formula:
            raise ValueError("seed formula differs; expressions are never evaluated")
        phase_roles = roles if phase == PHASES[2] else (("teacher",) if phase == PHASES[0] else ("learner",))
        for b in range(d["blocks"]):
            for c in range(d["conditions"]):
                for j in range(d["worlds_per_condition_per_phase_block"]):
                    world = dict(phase=phase, block=b, condition=c, replicate=j, seed=base + 1000*b + stride*c + j)
                    for role in phase_roles:
                        expected[_key(world, role)] = world
    if len(expected) != d["total_trajectories"] or d["evaluation_trajectories"] != 216:
        raise ValueError("trajectory arithmetic differs")
    return p, expected, roles, contrasts


def _statistics(values):
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "p90": float(np.percentile(values, 90)) if values else None}


def _vector(value):
    return isinstance(value, list) and len(value) == 4 and all(_finite(x) and x >= 0 for x in value)


def _read_trajectory(path, root, key, expected_world, proposal, summary):
    issues, warnings = [], []
    def issue(code, epoch=None):
        item = {"code": code}
        if epoch is not None:
            item["epoch"] = epoch
        issues.append(item)
    rows, digest = [], None
    try:
        digest = _digest(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = _json(line)
                    if not isinstance(row, dict):
                        raise ValueError("row is not an object")
                    rows.append(row)
                except (ValueError, TypeError):
                    issue("malformed_json_row", line_number)
    except (OSError, EOFError, UnicodeError):
        issue("unreadable_or_truncated_raw_file")
    horizon = proposal["synthetic_system"]["host_total_horizon"]
    control = proposal["synthetic_system"]["control_epochs"]
    costs, rewards, epochs, backlogs = [], [], [], {}
    patient_counts, boundary_counts, patient_events, previous_patients = None, {}, {}, {}
    observed_components, component_coverage = defaultdict(list), Counter()
    missing_components, component_mismatch = Counter(), []
    extra_components = set()
    hours = {"booked_flexible": [], "ordinary": [], "matured_flexible": [], "raw_executed_l1": []}
    reset_values, support_completions, component_bad_rows = [], {}, 0
    for row in rows:
        epoch = row.get("epoch")
        if type(epoch) is not int or not 0 <= epoch < horizon:
            issue("invalid_epoch")
            epoch = None
        epochs.append(epoch)
        try:
            row_key = _key(row.get("world"), row.get("role"))
        except ValueError:
            row_key = None
        if row_key != key or row.get("world") != expected_world:
            issue("row_world_or_role_mismatch", epoch)
        cost = row.get("cost")
        if not _finite(cost):
            issue("missing_or_nonfinite_cost", epoch)
        else:
            costs.append(float(cost))
        reward = row.get("reward")
        if _finite(reward):
            rewards.append(float(reward))
        if not _finite(reward) or (_finite(cost) and not _close(-reward, cost)):
            issue("raw_reward_must_equal_negative_unscaled_cost", epoch)
        components = row.get("components")
        if not isinstance(components, dict):
            components = {}
        invalid = False
        for name in COST_COMPONENTS:
            if name not in components or not _finite(components[name]):
                missing_components[name] += 1
                invalid = True
        for name, value in components.items():
            if name not in COST_COMPONENTS:
                extra_components.add(name)
            if _finite(value):
                observed_components[name].append(float(value))
                component_coverage[name] += 1
            else:
                invalid = True
        component_bad_rows += int(invalid)
        if all(_finite(v) for v in components.values()) and _finite(cost):
            recorded_sum = math.fsum(components.values())
            if not _close(recorded_sum, cost):
                component_mismatch.append({"epoch": epoch, "recorded_sum": recorded_sum, "cost": cost,
                                           "unreconciled_cost": cost - recorded_sum})
        info = row.get("info")
        if (isinstance(info, dict) and "cost" in info and _finite(cost)
                and (not _finite(info["cost"]) or not _close(info["cost"], cost))):
            issue("raw_info_cost_mismatch", epoch)
        reported = row.get("cumulative")
        if (not isinstance(reported, dict) or any(not _count(reported.get(k)) for k in ("enrolled", "delivered", "lost"))):
            issue("invalid_cumulative_counts", epoch)
            reported = None
        patients = row.get("patient_records")
        current = {}
        if not isinstance(patients, list):
            issue("missing_full_patient_registry", epoch)
            patients = []
        for patient in patients:
            if (not isinstance(patient, dict) or not isinstance(patient.get("patient_id"), str)
                    or not patient["patient_id"] or patient.get("status") not in STATUSES
                    or type(patient.get("enrollment_epoch")) is not int):
                issue("invalid_public_patient_record", epoch)
                continue
            pid, status, enrolled = patient["patient_id"], patient["status"], patient["enrollment_epoch"]
            if pid in current:
                issue("duplicate_patient_identity", epoch)
            if enrolled < 0 or (epoch is not None and enrolled > epoch):
                issue("future_or_invalid_enrollment", epoch)
            if enrolled >= control:
                issue("enrollment_after_cohort_closure", epoch)
            if pid in previous_patients:
                old = previous_patients[pid]
                if (enrolled != old["enrollment_epoch"] or patient.get("collection_site") != old.get("collection_site")
                        or (old["status"] in TERMINAL and status != old["status"])):
                    issue("patient_identity_changed_or_terminal_reopened", epoch)
            current[pid] = patient
            event = patient_events.setdefault(pid, {"enrollment_epoch": enrolled})
            if epoch is not None:
                if status in ("in_production", "finished", "delivered"):
                    event.setdefault("first_production_observed_epoch", epoch)
                if status in TERMINAL:
                    event.setdefault("terminal_epoch", epoch)
                    event.setdefault("terminal_status", status)
        if not set(previous_patients).issubset(current):
            issue("patient_registry_lost_identities", epoch)
        independent = {"enrolled": len(current),
                       "delivered": sum(p["status"] == "delivered" for p in current.values()),
                       "lost": sum(p["status"] == "lost" for p in current.values())}
        if reported is not None and any(reported[k] != independent[k] for k in independent):
            issue("cumulative_vs_registry_mismatch", epoch)
        patient_counts = independent
        if epoch is not None:
            boundary_counts[epoch + 1] = dict(independent)
        previous_patients = current
        backlog = row.get("support_backlog")
        if _count(backlog) and epoch is not None:
            backlogs[epoch] = int(backlog)
        raw, executed = row.get("requested_hours"), row.get("executed_hours")
        if _vector(executed):
            hours["booked_flexible"].append(math.fsum(executed))
            if epoch is not None and epoch >= control and any(executed):
                issue("nonzero_tail_commitment", epoch)
            if _vector(raw):
                hours["raw_executed_l1"].append(math.fsum(abs(a-b) for a, b in zip(raw, executed)))
        service = row.get("service")
        if isinstance(service, dict):
            for field, output in (("ordinary_hours", "ordinary"), ("applied_hours", "matured_flexible")):
                if _vector(service.get(field)):
                    hours[output].append(math.fsum(service[field]))
            if service.get("epoch") == epoch and isinstance(service.get("completed_ids"), list):
                for group in service["completed_ids"]:
                    if isinstance(group, list):
                        for pid in group:
                            if isinstance(pid, str) and pid in patient_events:
                                support_completions.setdefault(pid, epoch)
        if _count(row.get("filter_resets")):
            reset_values.append(int(row["filter_resets"]))
    sequential = epochs == list(range(horizon))
    if not sequential:
        issue("epochs_not_exactly_0_through_63")
    cost_sum = math.fsum(costs) if costs else None
    cost_complete = sequential and len(costs) == horizon
    if summary is None:
        issue("missing_summary")
    else:
        if summary.get("world") != expected_world or summary.get("role") != key[-1]:
            issue("summary_identity_mismatch")
        if not _finite(summary.get("cost")) or cost_sum is None or not _close(summary["cost"], cost_sum):
            issue("summary_vs_raw_cost_mismatch")
        settlement = summary.get("settlement")
        if summary.get("settled") is not True or not isinstance(settlement, dict) or settlement.get("settled") is not True:
            issue("trajectory_not_settled")
        else:
            if settlement.get("live_ids") != [] or settlement.get("pending_obligations") != 0 or settlement.get("resource_conservation") is not True:
                issue("nonzero_or_unverified_settlement_liability")
            if patient_counts is None or any(settlement.get(k) != patient_counts[k] for k in patient_counts):
                issue("settlement_vs_registry_mismatch")
        if not _sha(summary.get("tape_sha256")):
            issue("missing_or_invalid_tape_digest")
        if key[0] == PHASES[2] and not _sha(summary.get("model_seal_sha256")):
            issue("missing_or_invalid_model_seal")
        if key[2] == 1 and (type(summary.get("change_epoch")) is not int or not 16 <= summary["change_epoch"] <= 28):
            issue("missing_or_invalid_persistent_change_epoch")
    if patient_counts is not None and patient_counts["enrolled"] != patient_counts["delivered"] + patient_counts["lost"]:
        issue("live_patient_liability_at_last_observed_boundary")
    component_complete = (sequential and component_bad_rows == 0 and not extra_components and not component_mismatch)
    if not component_complete:
        warnings.append("cost_components_incomplete_or_nonadditive; raw cost summed independently without imputation")
    secondary, missing = {}, {}
    enrolled = patient_counts["enrolled"] if patient_counts else 0
    secondary["loss_delivery_rates"] = {"denominator": "all_enrolled", "enrolled": enrolled,
        "loss_rate": patient_counts["lost"] / enrolled if enrolled else None,
        "delivery_rate": patient_counts["delivered"] / enrolled if enrolled else None}
    secondary["backlog_and_liability_at_48_and_64"] = {
        str(boundary): {"support_backlog": backlogs.get(boundary - 1),
                        "live_patient_liability": (boundary_counts[boundary]["enrolled"] - boundary_counts[boundary]["delivered"] - boundary_counts[boundary]["lost"])
                        if boundary in boundary_counts else None}
        for boundary in (control, horizon)}
    if any(item["support_backlog"] is None or item["live_patient_liability"] is None
           for item in secondary["backlog_and_liability_at_48_and_64"].values()):
        missing["backlog_and_liability_at_48_and_64"] = "one or more required boundaries/fields absent"
    secondary["ordinary_flexible_booked_matured_productive_idle_hours"] = {
        name: {"observed_sum": math.fsum(values) if values else None, "observed_epochs": len(values), "complete": len(values) == horizon}
        for name, values in hours.items() if name != "raw_executed_l1"}
    missing["ordinary_flexible_booked_matured_productive_idle_hours"] = "productive/idle hour allocation is not observed by public completion events; no latent-response inference"
    private_hours = [row.get("info", {}).get("support_private_audit", {}) for row in rows]
    if len(private_hours) == horizon and all(
            _vector(x.get("productive_total_hours")) and _vector(x.get("idle_total_hours"))
            and x.get("controller_input") is False for x in private_hours):
        secondary["ordinary_flexible_booked_matured_productive_idle_hours"].update(
            productive_total_hours=math.fsum(v for x in private_hours for v in x["productive_total_hours"]),
            idle_total_hours=math.fsum(v for x in private_hours for v in x["idle_total_hours"]),
            source="private simulator audit, never provided to controller",
            ordinary_flexible_productive_split_identified=False)
        del missing["ordinary_flexible_booked_matured_productive_idle_hours"]
    secondary["raw_executed_action_differences"] = {"l1_hours_sum": math.fsum(hours["raw_executed_l1"]) if hours["raw_executed_l1"] else None,
                                                 "observed_epochs": len(hours["raw_executed_l1"])}
    if len(hours["raw_executed_l1"]) != horizon:
        missing["raw_executed_action_differences"] = "missing/invalid requested or executed hours"
    # Runner stores len(controller.filter.reset_events), a cumulative counter.
    resets_complete = (sequential and len(reset_values) == horizon
                       and all(a <= b for a, b in zip(reset_values, reset_values[1:])))
    secondary["filter_resets"] = {"recorded_values": reset_values, "last_recorded": reset_values[-1] if reset_values else None,
                                  "counter_semantics": "cumulative len(controller.filter.reset_events)",
                                  "total_resets": reset_values[-1] if resets_complete else None}
    if not resets_complete:
        missing["filter_resets"] = "missing, nonmonotone or incomplete cumulative reset counter"
    durations, delivered_durations, production_waits, support_delays = [], [], [], []
    events = []
    for pid, event in sorted(patient_events.items()):
        start = event["enrollment_epoch"]
        if "terminal_epoch" in event:
            duration = event["terminal_epoch"] - start
            durations.append(duration)
            if event["terminal_status"] == "delivered":
                delivered_durations.append(duration)
        if "first_production_observed_epoch" in event:
            production_waits.append(event["first_production_observed_epoch"] - start)
        if pid in support_completions:
            event["support_completion_epoch"] = support_completions[pid]
            support_delays.append(support_completions[pid] - start)
        events.append({"patient_id": pid, **event})
    secondary["all_patient_time_to_loss_or_delivery"] = {"definition": "first terminal observed row epoch minus enrollment_epoch; not biological age",
                                                        **_statistics(durations), "events": events}
    secondary["mean_p90_wait_turnaround"] = {"wait_to_first_production_observed": _statistics(production_waits),
                                             "delivered_turnaround": _statistics(delivered_durations),
                                             "definition": "observed row epoch minus enrollment_epoch; outcome-specific, not imputing lost patients"}
    secondary["mean_p90_support_wait"] = {"completion_delay": _statistics(support_delays),
                                          "definition": "public support completion event epoch minus enrollment_epoch; completers only",
                                          "without_observed_completion": len(patient_events) - len(support_delays)}
    if not sequential:
        for metric in ("mean_p90_support_wait", "mean_p90_wait_turnaround", "all_patient_time_to_loss_or_delivery"):
            missing[metric] = "incomplete observation sequence; first-observed times may be censored"
    if len(durations) != enrolled:
        missing["all_patient_time_to_loss_or_delivery"] = "not every enrolled patient has an observed terminal event"
    if not support_delays:
        missing["mean_p90_support_wait"] = "no matched public support completion events"
    missing["training_deployment_query_time_costs"] = "no per-query duration/timing field specified in saved row schema"
    timing = [row.get("timing_seconds",{}) for row in rows]
    if len(timing) == horizon and all(all(_finite(x.get(k)) and x[k]>=0 for k in ("controller","native","filter")) for x in timing):
        secondary["training_deployment_query_time_costs"] = {
            k: math.fsum(x[k] for x in timing) for k in ("controller","native","filter")}
        secondary["training_deployment_query_time_costs"]["scope"] = "wall seconds incl. primitive admission; update durations saved separately in updates"
        del missing["training_deployment_query_time_costs"]
    if not component_complete:
        missing["cost_components"] = "see missing keys, coverage, unexpected keys and cost reconciliation residuals"
    return {"id": _name(key), "world": expected_world, "role": key[-1], "raw_path": str(path.relative_to(root)),
            "raw_sha256": digest, "observed_rows": len(rows), "observed_cost_rows": len(costs),
            "cost": cost_sum, "cost_complete": cost_complete, "patient_counts": patient_counts,
            "patient_cohort_sha256": hashlib.sha256(json.dumps(sorted(
                (pid, patient["enrollment_epoch"], patient.get("collection_site"))
                for pid, patient in previous_patients.items()), separators=(",", ":")).encode()).hexdigest(),
            "primary_complete": not issues, "issues": issues, "warnings": warnings,
            "summary_cost": summary.get("cost") if summary else None,
            "settled_claim": summary.get("settled") if summary else None,
            "tape_sha256": summary.get("tape_sha256") if summary else None,
            "model_seal_sha256": summary.get("model_seal_sha256") if summary else None,
            "change_epoch": summary.get("change_epoch") if summary else None,
            "cost_components": {"complete": component_complete,
                "observed_totals": {k: math.fsum(v) for k, v in sorted(observed_components.items())},
                "observed_epochs": dict(component_coverage), "missing_or_invalid_epochs": dict(missing_components),
                "unexpected_keys": sorted(extra_components), "reconciliation_errors": component_mismatch},
            "reward_observed_sum": math.fsum(rewards) if rewards else None,
            "reward_note": "never used as a substitute for incurred cost or recovery cost",
            "secondary": secondary, "missing_secondary": missing,
            "epoch_costs": [row.get("cost") for row in rows] if cost_complete else None,
            "epoch_executed_hours": [row.get("executed_hours") for row in rows[:control]],
            "epoch_backlogs": [backlogs.get(t) for t in range(horizon)]}


def _pair_issues(left, right):
    issues = []
    if left["tape_sha256"] != right["tape_sha256"]:
        issues.append("paired_tape_mismatch")
    if left["change_epoch"] != right["change_epoch"]:
        issues.append("paired_change_epoch_mismatch")
    if left["model_seal_sha256"] != right["model_seal_sha256"]:
        issues.append("paired_model_seal_mismatch")
    if left["patient_cohort_sha256"] != right["patient_cohort_sha256"]:
        issues.append("paired_patient_cohort_mismatch")
    return issues


def _recovery(online, frozen, proposal):
    spec = proposal["analysis"]["recovery"]
    change = online["change_epoch"]
    if (not online["primary_complete"] or not frozen["primary_complete"] or type(change) is not int
            or any(x is None for x in online["epoch_backlogs"][:spec["censor_at"]])
            or any(x is None for x in frozen["epoch_backlogs"][:spec["censor_at"]])):
        return {"status": "missing", "reason": "complete paired costs/backlogs and change epoch required"}
    trailing, consecutive, streak = spec["trailing_epochs"], spec["consecutive_boundaries"], 0
    for end in range(change + trailing - 1, spec["censor_at"]):
        difference = math.fsum(online["epoch_costs"][end-trailing+1:end+1]) - math.fsum(frozen["epoch_costs"][end-trailing+1:end+1])
        qualifies = difference <= 0 and online["epoch_backlogs"][end] <= frozen["epoch_backlogs"][end]
        streak = streak + 1 if qualifies else 0
        if streak == consecutive:
            return {"status": "recovered", "boundary": end+1, "lag_epochs": end+1-change,
                    "change_epoch": change, "censored": False}
    return {"status": "censored", "boundary": spec["censor_at"], "lag_epochs": None,
            "followup_epochs": spec["censor_at"]-change, "change_epoch": change, "censored": True}


def _bootstrap(all_pairs, contrasts, proposal):
    """Same resampled block/world indices for every arm, contrast and endpoint."""
    d, spec = proposal["design"], proposal["analysis"]["bootstrap"]
    b, c, j, n = d["blocks"], d["conditions"], d["worlds_per_condition_per_phase_block"], spec["resamples"]
    metrics = ("savings", "extra_lost", "extra_delivered", "relative_savings_percent")
    rng = np.random.default_rng(spec["seed"])
    block_draws = rng.integers(b, size=(n, b))
    world_draws = rng.integers(j, size=(n, b, c, j))
    result = {}
    for contrast in contrasts:
        rows = all_pairs[contrast]
        values = np.full((b, c, j, len(metrics)), np.nan)
        for row in rows:
            for k, name in enumerate(metrics):
                if row[name] is not None:
                    values[row["block"], row["condition"], row["replicate"], k] = row[name]
        cells = values[block_draws[:, :, None, None], np.arange(c)[None, None, :, None], world_draws]
        samples = cells.mean(axis=3).mean(axis=1)
        def intervals(sample_matrix, original):
            return {name: ([float(x) for x in np.percentile(sample_matrix[:, k], spec["percentiles"])]
                           if np.isfinite(original[..., k]).all() else None)
                    for k, name in enumerate(metrics)}
        result[contrast] = {"conditions": {str(index): intervals(samples[:, index, :], values[:, index, :, :]) for index in range(c)},
                            "equal_condition_overall": intervals(samples.mean(axis=1), values)}
    return {**spec, "method": "resample 3 blocks, then 4 paired worlds independently within each drawn block/condition; equal block and condition weights",
            "same_draws_across_contrasts": True, "intervals": result,
            "interpretation": "descriptive pilot intervals, not confirmatory significance or clinical noninferiority"}


def _aggregate(rows, expected):
    metrics = ("savings", "relative_savings_percent", "extra_lost", "extra_delivered", "loss_rate_difference", "delivery_rate_difference")
    result = {"n_paired_worlds": len(rows), "expected_paired_worlds": expected, "complete": len(rows) == expected}
    for metric in metrics:
        values = [r[metric] for r in rows if r[metric] is not None]
        result[f"mean_{metric}"] = math.fsum(values)/len(values) if values else None
        result[f"n_{metric}"] = len(values)
    if rows:
        base = math.fsum(r["comparator_cost"] for r in rows)
        result["aggregate_relative_savings_percent"] = 100*math.fsum(r["savings"] for r in rows)/base if base else None
    else:
        result["aggregate_relative_savings_percent"] = None
    return result


def run_analysis(root, proposal, require_complete=True) -> dict:
    """Recompute saved primary evidence; never construct a simulator or model.

    Missing/mismatched primary evidence raises AnalysisIncomplete in strict mode.
    Missing secondary evidence is explicit but is not an additional acceptance
    gate. The returned ``cost`` is always summed from row.cost, not components,
    summary.cost or rewards. Only complete settled pairs enter endpoint contrasts.
    """
    root = Path(root).resolve()
    p, expected, roles, contrasts = _contract(proposal)
    issues, summary_by_path, summary_sources = [], {}, []
    for path in sorted((root / "summaries").glob("*.json")):
        try:
            path = _safe_path(root, path)
            summary = _json(path.read_text())
            key = _key(summary["world"], summary["role"])
            raw = _safe_path(root, summary["raw_path"])
            summary_sources.append({"path": str(path.relative_to(root)), "sha256": _digest(path)})
            if raw in summary_by_path:
                issues.append({"code": "duplicate_summary_for_raw", "path": str(raw.relative_to(root))})
            else:
                summary_by_path[raw] = (key, summary)
        except (ValueError, TypeError, KeyError, OSError):
            issues.append({"code": "invalid_summary", "path": str(path)})
    raw_paths = sorted(set((root / "raw").glob("*.jsonl")) | set((root / "raw").glob("*.jsonl.gz")))
    trajectories, records, observed_paths = [], {}, set()
    for raw in raw_paths:
        try:
            path = _safe_path(root, raw)
        except ValueError:
            issues.append({"code": "raw_path_outside_root", "path": str(raw)})
            continue
        observed_paths.add(path)
        match = RAW_NAME.fullmatch(path.name)
        if match is None:
            issues.append({"code": "unrecognized_raw_filename", "path": str(path.relative_to(root))})
            continue
        phase, b, c, j, role = match.groups()
        key = phase, int(b), int(c), int(j), role
        if key not in expected:
            issues.append({"code": "unexpected_trajectory", "id": _name(key)})
            continue
        summary_pair = summary_by_path.get(path)
        summary = summary_pair[1] if summary_pair else None
        record = _read_trajectory(path, root, key, expected[key], p, summary)
        trajectories.append(record)
        if key in records:
            records[key]["primary_complete"] = record["primary_complete"] = False
            issues.append({"code": "duplicate_trajectory", "id": _name(key)})
        else:
            records[key] = record
        if summary_pair and summary_pair[0] != key:
            record["primary_complete"] = False
            record["issues"].append({"code": "summary_raw_filename_mismatch"})
    for raw in summary_by_path.keys() - observed_paths:
        issues.append({"code": "summary_without_raw", "path": str(raw.relative_to(root))})
    missing = sorted(_name(key) for key in expected.keys() - records.keys())
    if missing:
        issues.append({"code": "missing_trajectories", "count": len(missing)})
    pair_integrity, pair_rows, recovery, exploration = [], {name: [] for name in contrasts}, [], []
    block_seals = defaultdict(set)
    for key, record in records.items():
        if key[0] == PHASES[2] and record["model_seal_sha256"]:
            block_seals[key[1]].add(record["model_seal_sha256"])
    for b, seals in block_seals.items():
        if len(seals) != 1:
            issues.append({"code": "different_evaluation_start_seals_within_block", "block": b})
    for b in range(3):
        for c in range(3):
            for j in range(4):
                arms = {role: records.get((PHASES[2], b, c, j, role)) for role in roles}
                available = {k: v for k, v in arms.items() if v is not None}
                world_issues = []
                if len(available) != 6:
                    world_issues.append("missing_evaluation_arms")
                if any(not row["primary_complete"] for row in available.values()):
                    world_issues.append("incomplete_evaluation_arm")
                if len({r["tape_sha256"] for r in available.values()}) != 1:
                    world_issues.append("six_arm_tape_mismatch")
                if len({r["change_epoch"] for r in available.values()}) != 1:
                    world_issues.append("six_arm_change_epoch_mismatch")
                if len(block_seals[b]) != 1:
                    world_issues.append("same_start_model_seal_mismatch")
                enrolled = {r["patient_counts"]["enrolled"] for r in available.values() if r["patient_counts"]}
                if len(enrolled) > 1:
                    world_issues.append("paired_enrollment_mismatch")
                if len({r["patient_cohort_sha256"] for r in available.values()}) > 1:
                    world_issues.append("paired_patient_cohort_mismatch")
                pair_integrity.append({"block": b, "condition": c, "replicate": j, "complete": not world_issues,
                                       "available_roles": sorted(available), "issues": world_issues})
                for contrast in contrasts:
                    baseline = contrast.removeprefix("online_vs_")
                    online, comparator = arms[ONLINE], arms[baseline]
                    if (online is None or comparator is None or not online["primary_complete"]
                            or not comparator["primary_complete"] or _pair_issues(online, comparator)
                            or online["patient_counts"]["enrolled"] != comparator["patient_counts"]["enrolled"]
                            or len(block_seals[b]) > 1):
                        continue
                    op, cp = online["patient_counts"], comparator["patient_counts"]
                    savings, denominator = comparator["cost"]-online["cost"], comparator["cost"]
                    row = {"block": b, "condition": c, "replicate": j, "seed": online["world"]["seed"],
                           "online_cost": online["cost"], "comparator_cost": comparator["cost"], "savings": savings,
                           "relative_savings_percent": 100*savings/denominator if denominator != 0 else None,
                           "online_patients": op, "comparator_patients": cp,
                           "extra_lost": op["lost"]-cp["lost"], "extra_delivered": op["delivered"]-cp["delivered"],
                           "loss_rate_difference": (op["lost"]-cp["lost"])/op["enrolled"] if op["enrolled"] else None,
                           "delivery_rate_difference": (op["delivered"]-cp["delivered"])/op["enrolled"] if op["enrolled"] else None,
                           "six_arm_complete": not world_issues, "online_raw_path": online["raw_path"],
                           "comparator_raw_path": comparator["raw_path"]}
                    oa, ca = online["epoch_executed_hours"], comparator["epoch_executed_hours"]
                    if len(oa)==len(ca)==48 and all(_vector(x) for x in oa+ca):
                        differences = [math.fsum(abs(x-y) for x,y in zip(a,z)) for a,z in zip(oa,ca)]
                        row["executed_action_difference"] = dict(changed_boundaries=sum(x>1e-9 for x in differences),
                            total_boundaries=48,l1_hours_sum=math.fsum(differences),
                            matched_exploration=baseline=="frozen_matched_exploration",
                            greedy_action_claim=False)
                    pair_rows[contrast].append(row)
                    if baseline == "frozen_history" and c == 1:
                        recovery.append({"block": b, "condition": c, "replicate": j, **_recovery(online, comparator, p)})
                noisy, clean = arms["frozen_matched_exploration"], arms["frozen_history"]
                if noisy and clean and noisy["primary_complete"] and clean["primary_complete"] and not _pair_issues(noisy, clean):
                    exploration.append({"block": b, "condition": c, "replicate": j,
                                        "incremental_cost": noisy["cost"]-clean["cost"],
                                        "extra_lost": noisy["patient_counts"]["lost"]-clean["patient_counts"]["lost"]})
    summaries = {}
    for contrast, rows in pair_rows.items():
        condition_summaries = {}
        for c in range(3):
            condition_rows = [r for r in rows if r["condition"] == c]
            condition_summaries[str(c)] = {"condition_id": p["conditions"][c]["id"], **_aggregate(condition_rows, 12),
                "blocks": {str(b): _aggregate([r for r in condition_rows if r["block"] == b], 4) for b in range(3)}}
        overall = _aggregate(rows, 36)
        for metric in ("savings", "relative_savings_percent", "extra_lost", "extra_delivered", "loss_rate_difference", "delivery_rate_difference"):
            means = [condition_summaries[str(c)][f"mean_{metric}"] for c in range(3)]
            overall[f"equal_condition_mean_{metric}"] = math.fsum(means)/3 if all(v is not None for v in means) else None
        summaries[contrast] = {"worlds": rows, "conditions": condition_summaries, "overall": overall}
    evaluation_complete = all(row["complete"] for row in pair_integrity)
    primary_complete = (not issues and len(records) == len(expected) and evaluation_complete
                        and all(row["primary_complete"] for row in trajectories))
    bootstrap = _bootstrap(pair_rows, contrasts, p) if evaluation_complete else {
        **p["analysis"]["bootstrap"], "intervals": None, "missing": "complete 3x3x4 six-arm paired design required; no partial bootstrap substitution"}
    checks, tradeoffs = {}, []
    for contrast in contrasts:
        for c in range(3):
            cell = summaries[contrast]["conditions"][str(c)]
            if cell["mean_savings"] is not None and cell["mean_savings"] > 0 and cell["mean_extra_lost"] > 0:
                tradeoffs.append({"contrast": contrast, "condition": c, "mean_savings": cell["mean_savings"],
                                 "mean_extra_lost": cell["mean_extra_lost"], "interpretation": p["analysis"]["cost_down_losses_up"]})
    if primary_complete:
        for contrast in p["analysis"]["primary_contrasts"]:
            cell = summaries[contrast]["conditions"]["1"]
            lower = bootstrap["intervals"][contrast]["conditions"]["1"]["savings"][0]
            checks[contrast] = {"all_three_block_savings_positive": all(row["mean_savings"] > 0 for row in cell["blocks"].values()),
                                "descriptive_interval_above_zero": lower > 0,
                                "mean_extra_losses_nonpositive": cell["mean_extra_lost"] <= 0}
    signal = all(all(v.values()) for v in checks.values()) if primary_complete else None
    evaluation_records = [r for r in trajectories if r["world"]["phase"] == PHASES[2]]
    secondary_coverage = {}
    for metric in p["analysis"]["secondary"]:
        expected_units = 216
        if metric == "exploration_cost":
            present = covered = len(exploration)
            expected_units = 36
        elif metric == "recovery_lag":
            present, covered, expected_units = len(recovery), sum(r["status"] != "missing" for r in recovery), 12
        elif metric in ("lost_patients", "delivered_patients"):
            present = sum(r["patient_counts"] is not None for r in evaluation_records)
            covered = sum(r["primary_complete"] for r in evaluation_records)
        elif metric == "cost_components":
            present = sum(bool(r["cost_components"]["observed_totals"]) for r in evaluation_records)
            covered = sum(r["cost_components"]["complete"] for r in evaluation_records)
        else:
            present = sum(metric in r["secondary"] for r in evaluation_records)
            covered = sum(metric in r["secondary"] and metric not in r["missing_secondary"]
                          for r in evaluation_records)
        secondary_coverage[metric] = {"status": "available" if covered == expected_units else ("partial" if present else "missing"),
                                      "present_units": present, "complete_units": covered, "expected_units": expected_units}
    report = {"schema": "capacity-pilot-independent-analysis-v1", "primary_complete": primary_complete,
              "evaluation_complete": evaluation_complete, "status": "complete" if primary_complete else "partial",
              "require_complete": require_complete, "proposal_sha256": hashlib.sha256(json.dumps(p, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "expected_trajectories": len(expected), "observed_trajectories": len(trajectories),
              "expected_evaluation_trajectories": 216,
              "observed_evaluation_trajectories": len(evaluation_records),
              "phase_counts": dict(Counter(r["world"]["phase"] for r in trajectories)),
              "missing_trajectories": missing, "issues": issues, "summary_sources": summary_sources,
              "trajectories": trajectories, "paired_integrity": pair_integrity, "contrasts": summaries,
              "bootstrap": bootstrap, "recovery_lag": recovery, "exploration_cost": exploration,
              "secondary_coverage": secondary_coverage,
              "signal": {"candidate_online_signal": signal, "checks": checks,
                         "classification": ("candidate_online_signal" if signal else p["analysis"]["null_interpretation"]) if primary_complete else p["analysis"]["incomplete_execution"],
                         "negative_findings": [{"contrast": c, "failed_criterion": k} for c, values in checks.items() for k, v in values.items() if not v],
                         "cost_patient_tradeoffs": tradeoffs, "clinical_success_claim": False,
                         "initializer_competence": "not established by this saved-trajectory analysis"},
              "missing_secondary": {r["id"]: r["missing_secondary"] for r in trajectories if r["missing_secondary"]},
              "limitations": ["synthetic development pilot, not confirmation or calibrated clinical/economic evidence",
                              "settlement/resource conservation fields are checked for consistency, not re-simulated",
                              "component sums and raw numeric costs are reported separately; no missing component imputation",
                              "time-to-event summaries use recorded event epochs and outcome-specific denominators"]}
    if require_complete and not primary_complete:
        raise AnalysisIncomplete(report)
    return report
