"""Validate immutable formal rows and export a cost-only audit projection.

This module never imports an environment, loads a policy, or runs simulation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "experiments/evidence/patient_indexed_specimen_routing_primary_ddpg/formal"
ALGORITHMS = ("gcn_residual_mdl2_network_ddpg_afd", "flat_residual_mdl2_network_ddpg_afd")
SEEDS = tuple(range(10, 15))
SCENARIOS = {
    "routing_nominal_history": 91100000,
    "routing_abrupt_regime_shift": 91200000,
    "routing_regional_drift": 91300000,
    "routing_compound_regional_stress": 91400000,
}
FIELDS = ("algorithm", "seed", "training_seed", "evaluation_seed", "replication",
          "scenario", "graph_ablation", "steps", "total_cost")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"hash mismatch: {path}")
    return actual


def validate_rows(rows: list[dict], *, algorithm: str, training_seed: int,
                  anchor: bool, per_scenario: dict) -> dict:
    expected_graph = "full_graph" if algorithm == ALGORITHMS[0] else "flat_state_no_graph"
    table = {}
    for row in rows:
        if not set(FIELDS).issubset(row):
            raise ValueError("missing required field")
        scenario = row["scenario"]
        if scenario not in SCENARIOS:
            raise ValueError("unexpected scenario")
        if (row["algorithm"] != ("mdl2" if anchor else algorithm)
                or int(row["training_seed"]) != training_seed
                or int(row["evaluation_seed"]) != SCENARIOS[scenario]
                or int(row["seed"]) != SCENARIOS[scenario]
                or row["graph_ablation"] != expected_graph
                or int(row["steps"]) != 52):
            raise ValueError("row identity mismatch")
        key = (scenario, int(row["replication"]))
        if key in table:
            raise ValueError("duplicate world key")
        value = float(row["total_cost"])
        if not math.isfinite(value):
            raise ValueError("nonfinite total cost")
        table[key] = value
    expected_keys = {(s, r) for s in SCENARIOS for r in range(100)}
    if set(table) != expected_keys:
        raise ValueError("incomplete or unexpected world keys")
    mean_field = "anchor_cost_mean" if anchor else "candidate_cost_mean"
    errors = {}
    for scenario in SCENARIOS:
        mean = math.fsum(table[scenario, r] for r in range(100)) / 100
        expected = float(per_scenario[scenario][mean_field])
        if not math.isfinite(expected) or not math.isclose(mean, expected, rel_tol=0, abs_tol=1e-5):
            raise ValueError(f"summary mean mismatch: {scenario}")
        errors[scenario] = abs(mean - expected)
    return {"rows": len(rows), "summary_mean_absolute_errors": errors, "world_costs": table}


def recover(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source and output trees must be separate")
    component_path = EVIDENCE / "cost_component_summary.json"
    components = json.loads(component_path.read_text())
    historical = {x["path"]: x for x in components["source_files"]}
    if len(historical) != 10 or any(x["rows"] != 400 for x in historical.values()):
        raise ValueError("unexpected historical hash manifest")
    file_records, summary_records, projections, anchor_reference = [], [], [], None
    input_hashes = {component_path: sha256(component_path)}
    for phase, variant in (("formal_final", "final"), ("formal_pretrain", "pretrain")):
        compact = EVIDENCE / f"{variant}_summary.json"
        raw_summary = source / phase / "summary.json"
        digest = require_hash(raw_summary, sha256(compact))
        input_hashes.update({compact: digest, raw_summary: digest})
        summary = json.loads(compact.read_text())
        if (summary["training_seeds"] != list(SEEDS)
                or set(summary["algorithms"]) != set(ALGORITHMS)
                or summary["scenarios"] != list(SCENARIOS)
                or summary["holdout_seed"] != 91100000
                or summary["holdout_replications"] != 100
                or summary["fixed_checkpoint_variant"] != variant):
            raise ValueError("summary identity mismatch")
        runs = {(r["algorithm"], r["training_seed"]): r for r in summary["runs"]}
        if len(summary["runs"]) != 10 or set(runs) != {(a, s) for a in ALGORITHMS for s in SEEDS}:
            raise ValueError("summary run inventory mismatch")
        summary_records.append({"phase": phase, "source": str(raw_summary),
                                "tracked_reference": str(compact.relative_to(REPO)), "sha256": digest})
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                for anchor in (False, True):
                    filename = "holdout_anchor_rows.csv" if anchor else "holdout_rows.csv"
                    relative = Path(algorithm) / f"seed{seed}" / filename
                    path = source / phase / relative
                    digest = sha256(path)
                    old = historical.get(str(relative)) if phase == "formal_final" else None
                    if old and digest != old["sha256"]:
                        raise ValueError(f"historical CSV hash mismatch: {path}")
                    csv.field_size_limit(2**31 - 1)
                    with path.open(newline="") as handle:
                        reader = csv.DictReader(handle)
                        if not set(FIELDS).issubset(reader.fieldnames or ()):
                            raise ValueError(f"missing CSV fields: {path}")
                        omitted = sorted(set(reader.fieldnames) - set(FIELDS))
                        rows = [{key: row[key] for key in FIELDS} for row in reader]
                    check = validate_rows(rows, algorithm=algorithm, training_seed=seed,
                                          anchor=anchor, per_scenario=runs[algorithm, seed]["holdout"]["per_scenario"])
                    costs = check.pop("world_costs")
                    if anchor:
                        if anchor_reference is None:
                            anchor_reference = costs
                        elif costs != anchor_reference:
                            raise ValueError("anchors differ across phase/algorithm/training seed")
                    require_hash(path, digest)
                    input_hashes[path] = digest
                    rel_output = Path(phase) / relative
                    projections.append((rel_output, rows))
                    file_records.append({"source": str(path), "projection": str(rel_output),
                                         "source_sha256": digest, "historical_csv_hash_verified": bool(old),
                                         "omitted_columns": omitted, **check})
    if sum(r["historical_csv_hash_verified"] for r in file_records) != 10:
        raise ValueError("incomplete historical hash coverage")
    # No output exists until every source file, pairing key and summary has passed.
    output.mkdir(parents=True)
    for (relative, rows), record in zip(projections, file_records):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        record["projection_sha256"] = sha256(target)
    for path, digest in input_hashes.items():
        require_hash(path, digest)
    code_paths = [Path(__file__).resolve(), REPO / "evaluation/crossed_design_bootstrap_audit.py",
                  REPO / "evaluation/aggregate_stats.py", REPO / "evaluation/aggregate_results.py"]
    report = {
        "status": "validated", "scope": "read_only_posthoc_total_cost_reporting_sensitivity",
        "source_root": str(source), "rows": sum(r["rows"] for r in file_records),
        "csv_files": len(file_records), "historically_hash_covered_csv_files": 10,
        "other_csv_files": "30 current hashes plus reconciliation to historical summary means, not historical byte proof",
        "retained_columns": list(FIELDS), "summaries": summary_records, "files": file_records,
        "anchor_costs_identical_across_both_phases_algorithms_and_all_seeds": True,
        "all_input_hashes_unchanged_after_projection": True,
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "analysis_source_hashes": {str(p.relative_to(REPO)): sha256(p) for p in code_paths},
        "historical_cost_manifest_sha256": sha256(component_path),
        "new_training_runs": 0, "new_simulation_runs": 0,
    }
    with (output / "provenance.json").open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = recover(args.source, args.output)
    print(json.dumps({k: report[k] for k in ("status", "csv_files", "rows", "historically_hash_covered_csv_files")}))


if __name__ == "__main__":
    main()
