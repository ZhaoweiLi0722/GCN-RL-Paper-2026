"""Independent Decimal verification of R2; does not import its auditor or NumPy."""

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
D = Decimal
TOL = D("1e-9")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(), parse_float=D)


def check(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), "mapping keys"
        for key in expected:
            check(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected), "list length"
        for a, e in zip(actual, expected):
            check(a, e)
    elif isinstance(expected, D):
        # Raw CSV costs reach billions; float subtraction has sub-micro rounding.
        assert abs(D(actual) - expected) <= D("0.000002"), "numeric mismatch"
    else:
        assert actual == expected, "exact value mismatch"


def stats(values):
    v = sorted(values)
    n = len(v)
    return {"n": n, "mean": sum(v) / n if n else None,
            "median": (v[(n - 1) // 2] + v[n // 2]) / 2 if n else None,
            "min": v[0] if n else None, "max": v[-1] if n else None,
            "negative": sum(x < -TOL for x in v), "tied": sum(abs(x) <= TOL for x in v),
            "positive": sum(x > TOL for x in v)}


def histogram(v):
    return {"within_top1_tolerance": sum(x <= TOL for x in v),
            "above_tolerance_to_250000": sum(TOL < x <= 250000 for x in v),
            "above_250000_to_1000000": sum(250000 < x <= 1000000 for x in v),
            "above_1000000": sum(x > 1000000 for x in v)}


def summarize(rows):
    output = {"states": len(rows), "selectors": {},
              "fitted_minus_frozen_cost": stats([r["fitted_minus_frozen_cost"] for r in rows])}
    for name in rows[0]["selectors"]:
        values = [r["selectors"][name] for r in rows]
        excess = [r["excess_over_validation_minimum"] for r in values]
        output["selectors"][name] = {
            key: stats([r[key] for r in values]) for key in
            ("validation_cost", "cost_minus_mdl2", "excess_over_validation_minimum")}
        output["selectors"][name].update(
            validation_top1_count=sum(r["validation_top1"] for r in values),
            excess_bins=histogram(excess),
            top1_error_excess_bins=histogram([r["excess_over_validation_minimum"] for r in values
                                            if not r["validation_top1"]]))
    disagreement = [r["selectors"]["discovery_cost_choice"]["excess_over_validation_minimum"]
                    for r in rows if not r["label_best_sets_agree"]]
    output["label_disagreement_excess"] = stats(disagreement)
    output["label_disagreement_excess_bins"] = histogram(disagreement)
    return output


def verify(path):
    result = read(path)
    for name, value in result["input_sha256"].items():
        assert digest(ROOT / name) == value, "changed input"
    config_path = ROOT / "experiments/configs/g1_decision_cost_20260929.json"
    config = read(config_path)
    assert digest(config_path) == result["config_sha256"]
    source = "evaluation/audit_g1_decision_cost.py"
    committed = subprocess.check_output(["git", "show", result["execution_commit"] + ":" + source], cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest() == result["auditor_sha256"] == digest(ROOT / source)
    with (ROOT / config["inputs"]["cost_rows"]["path"]).open(newline="") as f:
        costs = [r for r in csv.DictReader(f) if r["configured_horizon"] == "remaining"]
    with (ROOT / config["inputs"]["prediction_rows"]["path"]).open(newline="") as f:
        predictions = [r for r in csv.DictReader(f) if r["stream"] == "discovery"]
    c = {(int(r["training_seed"]), int(r["step"]), int(r["candidate_index"])): r for r in costs}
    p = {(int(r["held_out_seed"]), int(r["step"]), int(r["candidate_index"])): r for r in predictions}
    assert len(c) == len(costs) == len(p) == len(predictions) == 780
    assert set(c) == set(p)
    rows = []
    for seed, step in sorted({k[:2] for k in c}):
        d = [D(c[seed, step, i]["discovery_total_cost"]) for i in range(5)]
        v = [D(c[seed, step, i]["validation_total_cost"]) for i in range(5)]
        chosen = {"mdl2": 0, "discovery_cost_choice": min(range(5), key=d.__getitem__)}
        for name, field in (("frozen_critic", "baseline_q_advantage"), ("fitted_offline_critic", "fitted_q_advantage")):
            chosen[name] = max(range(5), key=lambda i: D(p[seed, step, i][field]))
        rows.append({"seed": seed, "step": step, "scenario_label_as_recorded": c[seed, step, 0]["scenario"],
                     "discovery_costs": d, "validation_costs": v, "validation_minimum_not_oracle": min(v),
                     "label_best_sets_agree": any(d[i] == min(d) and v[i] == min(v) for i in range(5)),
                     "selectors": {name: {"action_index": i, "validation_cost": v[i],
                                          "cost_minus_mdl2": v[i] - v[0],
                                          "excess_over_validation_minimum": v[i] - min(v),
                                          "validation_top1": abs(v[i] - min(v)) <= TOL}
                                   for name, i in chosen.items()},
                     "fitted_minus_frozen_cost": v[chosen["fitted_offline_critic"]] - v[chosen["frozen_critic"]]})
    check(result["decisions"], rows)
    check(result["summary"], summarize(rows))
    check(result["per_seed"], {str(s): summarize([r for r in rows if r["seed"] == s]) for s in (60, 61, 62)})
    return {"passed": True, "audit_sha256": digest(path), "raw_cost_states": len(rows),
            "selector_choices": 4 * len(rows), "input_hashes_verified": len(result["input_sha256"]),
            "arithmetic": "independent stdlib Decimal from CSV; all decisions and pooled/per-seed summaries",
            "absolute_float_reconciliation_tolerance": "0.000002",
            "environment_queries": 0, "optimizer_updates": 0,
            "verifier_sha256": digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/2026-09-29-g1-decision-cost/audit.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous verification")
    result = verify(args.audit)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
