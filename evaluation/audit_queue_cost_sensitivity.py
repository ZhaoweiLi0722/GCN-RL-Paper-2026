"""Reprice saved finite queue trees; never generate a trajectory or train a policy."""

import argparse
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import subprocess

from evaluation.audit_service_queue_boundary import CONFIG as QUEUE_CONFIG
from evaluation.audit_service_queue_boundary import METHODS, audit, read_rows
from evaluation.check_service_effort_mechanics import locked_files, sha, write_json
from evaluation.diagnose_service_queue_evidence import observation_payload, prefix_optimum


CONFIG = Path("experiments/configs/queue_cost_sensitivity_20260929.json")
COMPONENTS = ("holding", "labor", "switching")
SOURCES = [
    "evaluation/audit_queue_cost_sensitivity.py",
    "tests/test_queue_cost_sensitivity.py",
    "evaluation/audit_service_queue_boundary.py",
    "evaluation/diagnose_service_queue_evidence.py",
    "evaluation/check_service_effort_mechanics.py",
    "specs/2026-09-29-queue-cost-sensitivity/protocol.md",
    str(CONFIG), str(QUEUE_CONFIG),
]


def weights(holding, labor, switching):
    values = (holding, labor, switching)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("strictly positive finite cost multipliers required")
    return dict(zip(COMPONENTS, values))


def components(step):
    result = {key: step["cost"][key] for key in COMPONENTS}
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError("finite nonnegative recorded cost components required")
    return result


def sum_components(rows):
    result = dict.fromkeys(COMPONENTS, 0.0)
    for row in rows:
        for key, value in components(row).items():
            result[key] += value
    return result


def dot(parts, multipliers):
    return sum(parts[key] * multipliers[key] for key in COMPONENTS)


def reprice_edge(edge, closures, multipliers):
    result = deepcopy(edge)
    local = components(edge["step"])
    for key in COMPONENTS:
        result["step"]["cost"][key] = local[key] * multipliers[key]
    result["step"]["cost"]["total"] = dot(local, multipliers)
    result["step"]["receipt"]["labor_cost"] = local["labor"] * multipliers["labor"]
    result["step"]["receipt"]["change_cost"] = local["switching"] * multipliers["switching"]
    closure = dict.fromkeys(COMPONENTS, 0.0)
    if edge["settlement_id"] is not None:
        closure = sum_components(closures[edge["settlement_id"]]["settlement"]["steps"])
    # The continuation is an objective only, never a decision-time observation.
    result["cost"] = dot(local, multipliers) + dot(closure, multipliers)
    return result


def path_components(edges, closures, world, actions):
    parts = dict.fromkeys(COMPONENTS, 0.0)
    for t, action in enumerate(actions):
        edge = edges[world, tuple(actions[:t]), action]
        rows = [edge["step"]]
        if edge["settlement_id"] is not None:
            rows += closures[edge["settlement_id"]]["settlement"]["steps"]
        for key, value in sum_components(rows).items():
            parts[key] += value
    return parts


def enumerate_bounds(edges, worlds, actions, horizon):
    best_open = (math.inf, ())
    perfect = {name: math.inf for name, _ in worlds}
    for path in itertools.product(actions, repeat=horizon):
        expected = 0.0
        for name, probability in worlds:
            cost = sum(edges[name, path[:t], action]["cost"] for t, action in enumerate(path))
            expected += probability * cost
            perfect[name] = min(perfect[name], cost)
        best_open = min(best_open, (expected, path))
    return {"open_loop_cost": best_open[0], "open_loop_actions": best_open[1],
            "clairvoyant_cost": sum(p * perfect[name] for name, p in worlds)}


def unchanged_information(original, repriced, worlds, actions, horizon):
    checked = 0
    for t in range(horizon):
        for prefix in itertools.product(actions, repeat=t):
            for action in actions:
                left = [observation_payload(original[name, prefix, action]) for name, _ in worlds]
                right = [observation_payload(repriced[name, prefix, action]) for name, _ in worlds]
                if (len(set(left)) == 1) != (len(set(right)) == 1):
                    raise AssertionError("repricing changed the public information partition")
                checked += 1
    return checked


def analyze(config, fixture, upstream, edge_rows, closure_rows, episodes):
    closures = {row["id"]: row for row in closure_rows}
    if len(closures) != len(closure_rows):
        raise ValueError("duplicate closure identity")
    actions, horizon = tuple(fixture["actions"]), fixture["horizon"]
    records, decompositions = [], []
    for cell, comparison in sorted(upstream["comparisons"].items()):
        family = next(f for f in fixture["families"] if f["name"] == cell.split("__")[2])
        worlds = tuple((w["name"], w["probability"]) for w in family["worlds"])
        rows = [r for r in edge_rows if r["cell"] == cell]
        edges = {(r["world"], tuple(r["past_actions"]), r["action"]): r for r in rows}
        if len(edges) != len(rows):
            raise ValueError("duplicate edge identity")
        baseline_parts = {}
        for method in METHODS:
            mean = dict.fromkeys(COMPONENTS, 0.0)
            for world, probability in worlds:
                matches = [e for e in episodes if (e["cell"], e["world"], e["method"]) == (cell, world, method)]
                if len(matches) != 1:
                    raise ValueError("expected one archived episode per identity")
                episode = matches[0]
                parts = path_components(edges, closures, world, episode["actions"])
                if not math.isclose(sum(parts.values()), episode["cost"], abs_tol=1e-9):
                    raise AssertionError("component reconstruction disagrees with archived episode")
                decompositions.append({"cell": cell, "world": world, "method": method,
                                       "actions": episode["actions"], "components": parts})
                for key in COMPONENTS:
                    mean[key] += probability * parts[key]
            baseline_parts[method] = mean
        for labor, switching in itertools.product(config["labor_multipliers"], config["switching_multipliers"]):
            multipliers = weights(config["holding_multiplier"], labor, switching)
            repriced = {key: reprice_edge(edge, closures, multipliers) for key, edge in edges.items()}
            count = unchanged_information(edges, repriced, worlds, actions, horizon)
            optimum = prefix_optimum(repriced, worlds, actions, horizon)
            bounds = enumerate_bounds(repriced, worlds, actions, horizon)
            costs = {method: dot(parts, multipliers) for method, parts in baseline_parts.items()}
            gaps = {method: cost - optimum["cost"] for method, cost in costs.items()}
            if not (bounds["clairvoyant_cost"] <= optimum["cost"] + 1e-9 <= bounds["open_loop_cost"] + 2e-9):
                raise AssertionError("information-value bound order failed")
            if min(gaps.values()) < -1e-9:
                raise AssertionError("archived feasible path beat the exact feedback bound")
            if multipliers == dict.fromkeys(COMPONENTS, 1.0):
                expected = comparison["expected_synthetic_cost"]
                if any(not math.isclose(costs[m], expected[m], abs_tol=1e-9) for m in METHODS):
                    raise AssertionError("unit multipliers failed to recover original comparator costs")
                diagnostic = comparison["diagnostic"]
                for actual, key in ((optimum["cost"], "nonanticipative_cost"),
                                    (bounds["open_loop_cost"], "best_open_loop_cost"),
                                    (bounds["clairvoyant_cost"], "clairvoyant_cost")):
                    if not math.isclose(actual, diagnostic[key], abs_tol=1e-9):
                        raise AssertionError("unit multipliers failed to recover original bound")
            records.append({"cell": cell, "multipliers": multipliers,
                            "archived_path_costs_not_reoptimized": costs,
                            "archived_path_gaps_to_known_law_optimum": gaps,
                            "archived_paths_matching_optimum": [m for m, gap in gaps.items() if abs(gap) <= 1e-9],
                            "best_archived_path_gap_descriptive_only": min(gaps.values()),
                            "feedback_value_over_open_loop": bounds["open_loop_cost"] - optimum["cost"],
                            "known_law_feedback_optimum": optimum, "bounds": bounds,
                            "observation_partition_checks": count})
    if len(records) != config["expected_records"]:
        raise AssertionError("unexpected sensitivity record count")
    settings = []
    for labor, switching in itertools.product(config["labor_multipliers"], config["switching_multipliers"]):
        group = [r for r in records if r["multipliers"]["labor"] == labor and r["multipliers"]["switching"] == switching]
        if len(group) != config["expected_cells"]:
            raise AssertionError("unexpected cells per setting")
        settings.append({"labor_multiplier": labor, "switching_multiplier": switching,
                         "cells_matching_any_archived_path": sum(bool(r["archived_paths_matching_optimum"]) for r in group),
                         "cells_with_positive_feedback_value": sum(r["feedback_value_over_open_loop"] > 1e-9 for r in group),
                         "maximum_best_archived_path_gap": max(r["best_archived_path_gap_descriptive_only"] for r in group)})
    summary = {"classification": config["classification"], "record_count": len(records), "settings": settings,
               "unit_multipliers_recover_all_original_cells": True,
               "new_environment_queries": 0, "new_planner_queries": 0, "online_updates": 0,
               "independent_replications": 0, "manufacturing_calibration": False,
               "limitations": ["archived controller paths are not reoptimized for alternative weights",
                               "known-law optimum is itself a frozen history-action map",
                               "rich full-progress observations are not completion-only observations",
                               "batch and booked-flow cells are dependent offset duplicates",
                               "no reward or scientific scenario is selected from these results"]}
    return summary, records, decompositions


def run(output):
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing analysis")
    sources = locked_files(SOURCES, "HEAD")
    config = json.loads(CONFIG.read_text())
    if (len(config["labor_multipliers"]) * len(config["switching_multipliers"])
            != config["expected_settings_per_cell"]):
        raise ValueError("unexpected multiplier grid size")
    root = Path(config["input_root"])
    verified = audit(root)
    inputs = [{"path": str(p), "sha256": sha(p)} for p in sorted(root.iterdir()) if p.is_file()]
    fixture = json.loads(QUEUE_CONFIG.read_text())
    upstream = json.loads((root / "summary.json").read_text())
    summary, records, parts = analyze(config, fixture, upstream,
                                      read_rows(root / "tree_transitions.jsonl.gz"),
                                      read_rows(root / "tree_closures.jsonl.gz"),
                                      read_rows(root / "episodes.jsonl.gz"))
    for item in inputs + sources:
        if sha(item["path"]) != item["sha256"]:
            raise AssertionError("source or input changed during analysis")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    write_json(output / "sensitivity.json", records)
    write_json(output / "baseline_components.json", parts)
    write_json(output / "provenance.json", {
        "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_locks": sources, "input_locks": inputs, "upstream_audit": verified,
        "inputs_unchanged_after_analysis": True})
    write_json(output / "status.json", {"status": "completed", "exit_code": 0,
                                       "analysis_only": True, "scientific_launch": False})
    write_json(output / "inventory.json", {"files": [
        {"name": p.name, "sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(output.iterdir()) if p.is_file()]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
