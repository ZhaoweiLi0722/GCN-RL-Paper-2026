"""Deterministic mechanics checks only; no patient run, training or headroom claim."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess

from src.baselines.public_service_response import PublicServiceResponseEstimator
from src.env.service_effort_mechanics import ServiceEffortConfig, ServiceEffortMechanics


DEFAULT_CONFIG = Path("experiments/configs/service_effort_mechanics_20260928.json")
REFERENCE_COMMIT = "5638587d843131c8fcda2e2b44442dcccd2899e4"
SOURCES = [
    "src/env/service_effort_mechanics.py", "src/baselines/public_service_response.py",
    "evaluation/check_service_effort_mechanics.py", "tests/test_service_effort_mechanics.py",
    "specs/2026-09-28-service-effort-mechanics/protocol.md",
]
IMMUTABLE = [
    "src/env/capacity_planning.py", "src/env/patient_capacity_planning.py",
    "src/env/development_disruption.py", "experiments/configs/disruption_feasibility_20260928.json",
] + ["reports/2026-09-28-disruption-feasibility/pilot/" + name for name in (
    "episodes.jsonl", "execution.json", "status.json", "inventory.json", "steps.jsonl", "summary.json",
)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def locked_files(paths, revision):
    records = []
    for path in paths:
        expected = subprocess.check_output(["git", "show", f"{revision}:{path}"])
        digest = hashlib.sha256(expected).hexdigest()
        if sha(path) != digest:
            raise ValueError(f"file differs from {revision}: {path}")
        records.append({"path": path, "sha256": digest})
    return records


def check_fixture(fixture):
    if fixture["role"] != "mechanics_fixture_only_not_research_performance":
        raise ValueError("mechanics-only role required")
    if len(fixture["commitments"]) != 8 or [x["name"] for x in fixture["cases"]] != ["nominal", "persistent_shift", "backlog_limited"]:
        raise ValueError("only the three bounded eight-step fixtures are permitted")
    config = ServiceEffortConfig(**fixture["config"])
    if len(config.site_hour_caps) != 3:
        raise ValueError("three-site fixture required")
    rows, cases = [], {}
    for case in fixture["cases"]:
        if len(case["response_tape"]) != 8:
            raise ValueError("exactly eight response rows required")
        model = ServiceEffortMechanics(config, case["jobs"], case["response_tape"])
        estimator = PublicServiceResponseEstimator(fixture["prior"], fixture["estimator_alpha"])
        for action in fixture["commitments"]:
            before = asdict(model.observation())
            estimate_before = estimator.estimate
            after, receipt = model.step(action)
            estimator.observe(receipt)
            rows.append({"case": case["name"], "step": receipt.epoch,
                         "before": before, "receipt": asdict(receipt), "after": asdict(after),
                         "estimate_before": estimate_before, "estimate_after": estimator.estimate})
        cases[case["name"]] = {"final_estimate": estimator.estimate,
                               "uncensored_samples": estimator.samples,
                               "terminal_ledger": model.terminal_ledger()}
    grouped = {name: [r for r in rows if r["case"] == name] for name in cases}
    nominal, shift = grouped["nominal"], grouped["persistent_shift"]
    equal_before = [a["before"] == b["before"] for a, b in zip(nominal, shift)]
    if equal_before != [True] * 4 + [False] * 4:
        raise AssertionError("unexpected information timing")
    if cases["persistent_shift"]["final_estimate"] != (0.5, 1.25, 1.0):
        raise AssertionError("receipt-only estimator did not reproduce the noiseless fixture response")
    if cases["backlog_limited"]["uncensored_samples"] != [0, 0, 0]:
        raise AssertionError("censored output used as exact response")
    summary = {
        "classification": "mechanics_pass_only_not_a_learning_or_headroom_gate",
        "fixture_only": True, "research_performance_evidence": False,
        "training_performed": False, "patient_environment_executed": False,
        "new_scientific_crns_used": False, "clinical_validity_established": False,
        "scenario_count": 3, "transition_rows": len(rows),
        "first_distinguishable_decision_epoch": equal_before.index(False),
        "interpretation": "In this noiseless, fully measured work fixture, a simple estimator identifies the response; no RL need is established.",
        "cases": cases,
    }
    return rows, summary


def run(config_path, output):
    if config_path != DEFAULT_CONFIG:
        raise ValueError("use the committed bounded fixture config")
    sources = locked_files(SOURCES + [str(config_path)], "HEAD")
    immutable = locked_files(IMMUTABLE, REFERENCE_COMMIT)
    execution_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    fixture = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "claim.json", {"fixture_only": True, "execution_commit": execution_commit,
                                      "sources": sources, "immutable_reference_commit": REFERENCE_COMMIT,
                                      "immutable_files": immutable})
    try:
        rows, summary = check_fixture(fixture)
        (output / "transitions.jsonl").write_text("".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows))
        summary["execution_commit"] = execution_commit
        write_json(output / "summary.json", summary)
        locked_files(IMMUTABLE, REFERENCE_COMMIT)
        write_json(output / "status.json", {"status": "completed", "exit_code": 0, "fixture_only": True})
        write_json(output / "inventory.json", {"files": sources + immutable + [
            {"path": str(output / name), "sha256": sha(output / name)}
            for name in ("claim.json", "transitions.jsonl", "summary.json", "status.json")
        ]})
        return summary
    except BaseException as exc:
        write_json(output / "status.json", {"status": "failed", "exit_code": 1, "error": repr(exc)})
        raise


def audit(output):
    """Recompute conservation, timing and charges from persisted rows, not env.step."""
    inventory = json.loads((output / "inventory.json").read_text())
    for item in inventory["files"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError(f"inventory mismatch: {item['path']}")
    fixture = json.loads(DEFAULT_CONFIG.read_text())
    rows = [json.loads(line) for line in (output / "transitions.jsonl").read_text().splitlines()]
    summary = json.loads((output / "summary.json").read_text())
    status = json.loads((output / "status.json").read_text())
    if status["status"] != "completed" or status["exit_code"] != 0:
        raise AssertionError("fixture did not complete")
    keys = {(r["case"], r["step"]) for r in rows}
    expected_keys = {(case["name"], step) for case in fixture["cases"] for step in range(8)}
    if len(rows) != 24 or keys != expected_keys:
        raise AssertionError("missing or duplicate transition")
    cfg = fixture["config"]
    for case in fixture["cases"]:
        series = sorted((r for r in rows if r["case"] == case["name"]), key=lambda r: r["step"])
        total = [0.0] * 3
        estimate = fixture["prior"].copy()
        counts = [0] * 3
        for t, row in enumerate(series):
            pre, post, receipt = row["before"], row["after"], row["receipt"]
            if row["estimate_before"] != estimate or receipt["available_work"] != pre["remaining_work"]:
                raise AssertionError("public information mismatch")
            if t and pre != series[t - 1]["after"]:
                raise AssertionError("broken observation chain")
            action = fixture["commitments"][t]
            applied = fixture["commitments"][t - cfg["commitment_lead_steps"]] if t >= cfg["commitment_lead_steps"] else [0.0] * 3
            if receipt["committed_hours"] != action or receipt["applied_hours"] != applied or pre["epoch"] != t or post["epoch"] != t + 1:
                raise AssertionError("effort timing mismatch")
            if post["pending_hours"] != pre["pending_hours"][1:] + [action]:
                raise AssertionError("pending commitment mismatch")
            labor = sum(cfg["hourly_cost"] * x + cfg["quadratic_cost"] * x * x for x in action)
            change = cfg["switching_cost"] * sum(abs(x - p) for x, p in zip(action, pre["previous_commitment"]))
            if not math.isclose(labor, receipt["labor_cost"]) or not math.isclose(change, receipt["change_cost"]):
                raise AssertionError("cost accounting mismatch")
            for i in range(3):
                delivered = min(pre["remaining_work"][i], applied[i] * case["response_tape"][t][i])
                total[i] += delivered
                if not math.isclose(delivered, receipt["delivered_work"][i], abs_tol=1e-12):
                    raise AssertionError("work response mismatch")
                if not math.isclose(sum(case["jobs"][i]), total[i] + post["remaining_work"][i], abs_tol=1e-12):
                    raise AssertionError("work conservation mismatch")
                if len(case["jobs"][i]) != post["unfinished_jobs"][i] + post["completed_jobs"][i]:
                    raise AssertionError("job conservation mismatch")
                prefix = 0.0
                completed = 0
                for work in case["jobs"][i]:
                    prefix += work
                    if prefix <= total[i] + 1e-12:
                        completed += 1
                if post["completed_jobs"][i] != completed:
                    raise AssertionError("premature or missing whole-job completion")
                kind = "no_effort" if applied[i] == 0 else ("backlog_limited" if delivered >= pre["remaining_work"][i] - 1e-12 else "uncensored")
                if receipt["observation_kind"][i] != kind:
                    raise AssertionError("censoring mismatch")
                if kind == "uncensored":
                    alpha = fixture["estimator_alpha"]
                    estimate[i] = (1 - alpha) * estimate[i] + alpha * delivered / applied[i]
                    counts[i] += 1
            if row["estimate_after"] != estimate:
                raise AssertionError("estimator did not use public evidence only")
        if series[-1]["estimate_after"] != summary["cases"][case["name"]]["final_estimate"]:
            raise AssertionError("summary estimate mismatch")
        if counts != summary["cases"][case["name"]]["uncensored_samples"]:
            raise AssertionError("summary sample count mismatch")
    if summary["research_performance_evidence"] or summary["training_performed"]:
        raise AssertionError("incorrect scientific classification")
    return {"independent_row_audit": "passed", "rows": len(rows), "verified_file_hashes": len(inventory["files"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = audit(args.output) if args.audit_only else run(DEFAULT_CONFIG, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
