"""Bounded service/filter mechanics matrix; not a policy performance experiment."""

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np

from evaluation.check_service_effort_mechanics import locked_files, sha, write_json
from src.baselines.completion_response_filter import CompletionResponseFilter
from src.env.completion_feedback_queue import (
    CompletionConfig, CompletionJob, advance, booked_backlog_action, closed,
    completion_probability, initial_observation, unresolved_ledger,
)
from src.env.service_effort_mechanics import ServiceEffortConfig


CONFIG = Path("experiments/configs/completion_feedback_mechanics_20260929.json")
SOURCES = [str(CONFIG), "src/env/completion_feedback_queue.py", "src/baselines/completion_response_filter.py",
           "src/env/service_effort_mechanics.py", "evaluation/check_service_effort_mechanics.py",
           "evaluation/check_completion_feedback_mechanics.py", "tests/test_completion_feedback_queue.py",
           "tests/test_completion_feedback_check.py", "specs/2026-09-29-completion-feedback-mechanics/protocol.md"]
IMMUTABLE = ["src/env/capacity_planning.py", "src/env/patient_capacity_planning.py",
             "src/env/service_queue_network.py", "src/baselines/public_service_response.py",
             "specs/2026-09-29-qualified-support-contract/measurement_contract.json",
             "reports/2026-09-29-queue-cost-sensitivity/run/summary.json",
             "reports/2026-09-29-queue-cost-sensitivity/run/sensitivity.json",
             "reports/2026-09-29-prospective-closed-loop-engineering/check.json"]
REFERENCE = "d4af765"


def make_config(fixture, slots):
    jobs = tuple(CompletionJob(f"site{site}_job{k}", site, release, fixture["downstream_steps"])
                 for site in (0, 1) for k, release in enumerate(fixture["booked_releases"]))
    return CompletionConfig(ServiceEffortConfig(**fixture["effort"]), jobs, slots,
                            fixture["holding_cost"], fixture["max_closure_steps"])


def response_at(fixture, family, epoch):
    if family not in fixture["families"]:
        raise ValueError("unknown response family")
    if epoch < fixture["change_epoch"] or family == "unchanged":
        return tuple(fixture["nominal_response"])
    slow_fast = tuple(fixture["slow_fast_response"])
    if family == "persistent_right_slow" or (family == "rapid_alternating" and (epoch-fixture["change_epoch"]) % 2):
        return slow_fast[::-1]
    return slow_fast


def noise_tape(fixture, seed):
    return np.random.Generator(np.random.PCG64(seed)).random((fixture["horizon"] + fixture["max_closure_steps"], 2))


def run_case(fixture, family, slots, seed):
    cfg, tape = make_config(fixture, slots), noise_tape(fixture, seed)
    filt = CompletionResponseFilter(**fixture["filter"])
    state, rows = initial_observation(cfg), []
    for t in range(fixture["horizon"] + cfg.max_closure_steps):
        if t >= fixture["horizon"] and closed(state):
            break
        action = (0, 0) if t == fixture["horizon"] else booked_backlog_action(state, cfg)
        before, posterior = state, filt.posterior
        state, receipt, cost = advance(state, action, response_at(fixture, family, t), tape[t], cfg)
        filt.observe(receipt)
        rows.append({"before": asdict(before), "action": action, "after": asdict(state),
                     "receipt": asdict(receipt), "cost": cost,
                     "filter_before": posterior, "filter_after": filt.posterior})
    ledger = unresolved_ledger(state, cfg)
    components = {key: sum(r["cost"][key] for r in rows) for key in ("holding", "labor", "switching", "total")}
    return {"family": family, "downstream_slots": slots, "engineering_seed": seed,
            "intervals": len(rows), "decision_intervals": fixture["horizon"],
            "closure_intervals": max(0, len(rows)-fixture["horizon"]),
            "ledger": ledger, "accounting_cost_not_policy_comparison": components,
            "exposed_intervals": filt.exposed_intervals,
            "final_filter_mean_not_accuracy_claim": filt.mean,
            "private_uniform_tape_sha256": hashlib.sha256(tape.astype("<f8").tobytes()).hexdigest()}, rows


def equal(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            equal(a, e)
    elif type(expected) is bool:
        assert type(actual) is bool and actual == expected
    elif type(expected) in (int, float):
        assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def audit_case(fixture, case, rows):
    """Recompute event flow, charges and likelihoods without calling advance/filter."""
    cfg = make_config(fixture, case["downstream_slots"])
    tape = noise_tape(fixture, case["engineering_seed"])
    assert hashlib.sha256(tape.astype("<f8").tobytes()).hexdigest() == case["private_uniform_tape_sha256"]
    current = asdict(initial_observation(cfg))
    posterior = [list(fixture["filter"]["prior"]) for _ in (0, 1)]
    bought, applied, exposures, costs = 0.0, 0.0, [0, 0], dict.fromkeys(("holding", "labor", "switching", "total"), 0.0)
    counts = {"right_censored": 0, "completed": 0, "no_effort": 0, "idle": 0}
    idle_paid = 0.0
    for t, row in enumerate(rows):
        equal(row["before"], current)
        equal(row["filter_before"], posterior)
        request, receipt = row["action"], row["receipt"]
        expected_action = [0, 0] if t == fixture["horizon"] else [
            min(cfg.effort.shared_hour_budget/2, *cfg.effort.site_hour_caps)
            if any(j.site == site and (stage == "support" or j.release == t+1)
                   for j, stage in zip(cfg.jobs, current["stages"])) else 0 for site in (0, 1)]
        equal(request, expected_action)
        equal(receipt["requested_hours"], request)
        equal(receipt["applied_hours"], current["pending_hours"])
        equal([receipt["epoch"], receipt["known_at"]], [t, t+1])
        stages, ready, remaining = list(current["stages"]), list(current["ready_epochs"]), list(current["downstream_remaining"])
        active = [i for i, stage in enumerate(stages) if stage == "active"]
        waiting = sorted((i for i, stage in enumerate(stages) if stage == "waiting"),
                         key=lambda i: (ready[i], cfg.jobs[i].release, cfg.jobs[i].name))
        started = waiting[:cfg.downstream_slots-len(active)]
        finished = []
        for i in active + started:
            remaining[i] = (cfg.jobs[i].downstream_steps if i in started else remaining[i])-1
            stages[i] = "active" if remaining[i] else "done"
            if not remaining[i]:
                finished.append(cfg.jobs[i].name)
        rates = response_at(fixture, case["family"], t)
        for site in (0, 1):
            eligible = sorted((i for i, j in enumerate(cfg.jobs) if j.site == site and current["stages"][i] == "support"),
                              key=lambda i: (cfg.jobs[i].release, cfg.jobs[i].name))
            head = eligible[0] if eligible else None
            hours = current["pending_hours"][site] if eligible else 0.0
            event = bool(hours > 0 and tape[t, site] < 1-math.exp(-rates[site]*hours))
            kind = "idle" if head is None else ("no_effort" if hours == 0 else ("completed" if event else "right_censored"))
            equal(receipt["eligible_tasks"][site], None if head is None else cfg.jobs[head].name)
            equal([receipt["exposure_hours"][site], receipt["completed"][site], receipt["observation_kind"][site]], [hours, event, kind])
            counts[kind] += 1
            if kind == "idle":
                idle_paid += current["pending_hours"][site]
            if event:
                stages[head], ready[head] = "waiting", t+1
            refresh = fixture["filter"]["refresh_probability"]
            raw = [(1-refresh)*p + refresh*q for p, q in zip(posterior[site], fixture["filter"]["prior"])]
            if hours > 0:
                exposures[site] += 1
                raw = [p*((1-math.exp(-rate*hours)) if event else math.exp(-rate*hours))
                       for p, rate in zip(raw, fixture["filter"]["rates"])]
            posterior[site] = [p/sum(raw) for p in raw]
        equal(row["filter_after"], posterior)
        holding = cfg.holding_cost*sum(j.release <= t and stage != "done" for j, stage in zip(cfg.jobs, stages))
        released = []
        for i, j in enumerate(cfg.jobs):
            if stages[i] == "scheduled" and j.release == t+1:
                stages[i] = "support"
                released.append(j.name)
        labor = sum(cfg.effort.hourly_cost*h + cfg.effort.quadratic_cost*h*h for h in request)
        switch = cfg.effort.switching_cost*sum(abs(h-old) for h, old in zip(request, current["previous_request"]))
        equal([receipt["labor_cost"], receipt["switching_cost"]], [labor, switch])
        equal(row["cost"], {"holding": holding, "labor": labor, "switching": switch, "total": holding+labor+switch,
                            "downstream_started": [cfg.jobs[i].name for i in started],
                            "downstream_finished": finished, "released": released})
        bought += sum(request)
        applied += sum(current["pending_hours"])
        equal(bought, applied + sum(request))
        current = {"epoch": t+1, "stages": stages, "ready_epochs": ready, "downstream_remaining": remaining,
                   "pending_hours": request, "previous_request": request}
        equal(row["after"], current)
        assert stages.count("active") <= cfg.downstream_slots
        for key in costs:
            costs[key] += row["cost"][key]
    assert len(rows) == case["intervals"] <= fixture["horizon"]+fixture["max_closure_steps"]
    assert all(s == "done" for s in current["stages"]) and not any(current["pending_hours"]) and not any(current["previous_request"])
    equal(case["ledger"], {"settled": True, "unfinished_jobs": [], "pending_prepaid_hours": [0,0], "previous_request": [0,0]})
    equal(case["accounting_cost_not_policy_comparison"], costs)
    equal(case["exposed_intervals"], exposures)
    equal(case["final_filter_mean_not_accuracy_claim"], [sum(r*p for r, p in zip(fixture["filter"]["rates"], row)) for row in posterior])
    assert counts["completed"] == len(cfg.jobs)
    return {"audit": "passed", "intervals": len(rows), "completion_events": counts["completed"],
            "receipt_counts": counts, "idle_paid_hours": idle_paid, "purchased_and_applied_hours": bought}


def audit_output(output):
    fixture = json.loads(CONFIG.read_text())
    cases = json.loads((output/"cases.json").read_text())
    expected = {f"{f}__slots{s}__seed{r}" for f, s, r in itertools.product(
        fixture["families"], fixture["downstream_slots"], fixture["engineering_seeds"])}
    indexed = {case["case"]: case for case in cases}
    assert len(cases) == len(indexed) == fixture["expected_cases"] and set(indexed) == expected
    groups = {name: [] for name in expected}
    with (output/"steps.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            identity = row.pop("case")
            assert identity in groups
            groups[identity].append(row)
    results = []
    for case in cases:
        identity = case["case"]
        assert identity == f'{case["family"]}__slots{case["downstream_slots"]}__seed{case["engineering_seed"]}'
        results.append({"case": identity, **audit_case(fixture, case, groups[identity])})
    return results


def quadrature(fixture):
    q = fixture["quadrature"]
    uniforms = [(i+0.5)/q["points"] for i in range(q["points"])]
    records = []
    for rate, hours in itertools.product(q["rates"], q["hours"]):
        expected = completion_probability(rate, hours)
        frequency = sum(u < expected for u in uniforms)/q["points"]
        assert abs(expected-frequency) <= 1/q["points"]
        records.append({"rate": rate, "hours": hours, "declared_probability": expected,
                        "midpoint_frequency": frequency, "mean_derivative": rate*math.exp(-rate*hours)})
    return records


def run(output):
    if output.exists():
        raise FileExistsError("refusing to overwrite existing evidence")
    sources = locked_files(SOURCES, "HEAD")
    immutable = locked_files(IMMUTABLE, REFERENCE)
    fixture = json.loads(CONFIG.read_text())
    if fixture["role"] != "hypothetical_completion_feedback_mechanics_not_performance" or fixture["horizon"] != 32:
        raise ValueError("outside the bounded mechanics contract")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"claim.json", {"source_locks": sources, "immutable_locks": immutable,
                                   "reference_commit": REFERENCE, "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                   "python": platform.python_version(), "numpy": np.__version__,
                                   "training": False, "patient_environment": False})
    cases = []
    try:
        with (output/"steps.jsonl").open("x") as stream:
            for family, slots, seed in itertools.product(fixture["families"], fixture["downstream_slots"], fixture["engineering_seeds"]):
                case, rows = run_case(fixture, family, slots, seed)
                identity = f"{family}__slots{slots}__seed{seed}"
                for row in rows:
                    stream.write(json.dumps({"case": identity, **row}, sort_keys=True, allow_nan=False)+"\n")
                cases.append({"case": identity, **case})
                write_json(output/"cases.json", cases)
                audit_case(fixture, case, rows)
        audits = audit_output(output)
        write_json(output/"quadrature.json", quadrature(fixture))
        assert len(cases) == fixture["expected_cases"]
        summary = {"classification": "synthetic_mechanics_pass_not_headroom_or_online_rl_evidence",
                   "cases": len(cases), "recorded_intervals": sum(c["intervals"] for c in cases),
                   "all_jobs_and_prepaid_requests_settled": True,
                   "total_completion_events": sum(a["completion_events"] for a in audits),
                   "case_audits": audits, "quadrature_function_evaluations": 15000,
                   "online_neural_updates": 0, "policy_comparisons": 0,
                   "patient_environment_run": False, "clinical_calibration": False,
                   "formal_holdout_used": False, "statistical_replications_for_effect_claim": 0}
        write_json(output/"summary.json", summary)
        locked_files(IMMUTABLE, REFERENCE)
        locked_files(SOURCES, "HEAD")
        write_json(output/"status.json", {"status": "completed", "exit_code": 0, "mechanics_only": True})
        write_json(output/"inventory.json", {"files": [{"name": p.name, "sha256": sha(p), "bytes": p.stat().st_size}
                                                       for p in sorted(output.iterdir()) if p.is_file()]})
        return summary
    except BaseException as exc:
        write_json(output/"status.json", {"status": "failed", "exit_code": 1, "error": repr(exc), "mechanics_only": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
