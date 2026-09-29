"""Replay a fixed count-only rule on immutable recorded edges; no new rollout."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess

from evaluation.audit_service_queue_boundary import CONFIG, audit, equal, jobs_for, read_rows
from evaluation.check_service_effort_mechanics import locked_files, sha, write_json
from src.baselines.completion_count_control import count_observation, completion_count_choice


INPUT = Path("reports/2026-09-29-service-queue-boundary/run")
REFERENCE = Path("reports/2026-09-29-service-queue-boundary/observation_contract/observation_contract.json")
SOURCES = ["src/baselines/completion_count_control.py",
           "evaluation/audit_service_queue_completion_rule.py",
           "tests/test_service_queue_completion_rule.py",
           "specs/2026-09-29-completion-count-comparator/protocol.md"]


def index_edges(rows):
    indexed = {}
    for row in rows:
        key = row["cell"], row["world"], tuple(row["past_actions"]), row["action"]
        if key in indexed:
            raise ValueError("duplicate recorded edge")
        indexed[key] = row
    return indexed


def replay(fixture, cell, world_name, edges):
    """World identity selects evaluator records, never the controller input."""
    sites = tuple(j["site"] for j in jobs_for(fixture, cell))
    prefix, previous_after, rows = (), None, []
    for epoch in range(fixture["horizon"]):
        alternatives = [edges[cell, world_name, prefix, a] for a in fixture["actions"]]
        before = alternatives[0]["step"]["before"]
        equal(before["epoch"], epoch)
        for edge in alternatives:
            equal(edge["step"]["before"], before)
        if previous_after is not None:
            equal(before, previous_after)
        public = count_observation(epoch, before["stages"], sites)
        action = completion_count_choice(public)
        edge = edges[cell, world_name, prefix, action]
        equal(edge["step"]["receipt"]["committed_hours"], fixture["actions"][action])
        decision_cost, charged_cost = edge["step"]["cost"]["total"], edge["cost"]
        if not all(math.isfinite(v) for v in (decision_cost, charged_cost)):
            raise ValueError("nonfinite persisted cost")
        closure = charged_cost - decision_cost
        if closure < -1e-9 or (epoch < fixture["horizon"] - 1 and closure != 0):
            raise ValueError("invalid recorded closure accounting")
        rows.append({"cell": cell, "world": world_name, "past_actions": prefix,
                     "public_input": asdict(public), "action": action,
                     "decision_cost": decision_cost, "closure_cost": closure,
                     "charged_cost": charged_cost, "settlement_id": edge["settlement_id"]})
        prefix += (action,)
        previous_after = edge["step"]["after"]
    return {"cell": cell, "world": world_name, "actions": prefix,
            "cost": sum(r["charged_cost"] for r in rows),
            "decision_cost": sum(r["decision_cost"] for r in rows),
            "closure_cost": sum(r["closure_cost"] for r in rows)}, rows


def compute(fixture, summary, reference, edge_rows):
    if fixture["horizon"] != 5 or fixture["actions"] != {
            "idle": [0, 0], "left": [1, 0], "right": [0, 1], "balanced": [0.5, 0.5]}:
        raise ValueError("unchanged five-decision four-action fixture required")
    edges = index_edges(edge_rows)
    episodes, decisions, comparisons = [], [], {}
    for cell, prior in summary["comparisons"].items():
        family = next(f for f in fixture["families"] if f["name"] == cell.split("__")[2])
        expected, paths = 0.0, {}
        for world in family["worlds"]:
            episode, rows = replay(fixture, cell, world["name"], edges)
            expected += world["probability"] * episode["cost"]
            episodes.append(episode)
            decisions.extend(rows)
            paths[world["name"]] = episode["actions"]
        bound = reference["results"][cell]["contracts"]["completion_events"]["cost"]
        if expected < bound - 1e-9:
            raise AssertionError("count-only feasible rule beat the completion-information bound")
        comparisons[cell] = {
            "count_rule_cost": expected,
            "fixed_balanced_cost": prior["expected_synthetic_cost"]["fixed_balanced"],
            "completion_information_known_law_bound": bound,
            "gap_to_completion_information_bound": expected - bound,
            "matches_bound": math.isclose(expected, bound, abs_tol=1e-9, rel_tol=0),
            "actions_by_world": paths,
        }
    # Identical inputs must select the same action even across cells/worlds.
    choices = {}
    for row in decisions:
        key = json.dumps(row["public_input"], sort_keys=True)
        if key in choices and choices[key] != row["action"]:
            raise AssertionError("controller used information outside its declared input")
        choices[key] = row["action"]
    return {"classification": "post_run_fixed_rule_replay_not_new_simulation",
            "new_simulator_queries": 0, "online_weight_updates": False,
            "operational_measurements_validated": False, "independent_replications": 0,
            "controller_prediction_queries": 0, "replayed_world_paths": len(episodes),
            "decision_rows": len(decisions), "comparisons": comparisons,
            "limitation": "One untuned event-count rule, not an identification-MPC or frozen history-policy comparison. Known-law bound and common continuation are restricted synthetic diagnostics."}, episodes, decisions


def run(output):
    if output.exists():
        raise FileExistsError(output)
    sources = locked_files(SOURCES + [str(CONFIG), str(REFERENCE)], "HEAD")
    before = audit(INPUT)
    input_hash = sha(INPUT / "inventory.json")
    fixture = json.loads(CONFIG.read_text())
    summary = json.loads((INPUT / "summary.json").read_text())
    reference = json.loads(REFERENCE.read_text())
    result, episodes, decisions = compute(fixture, summary, reference, read_rows(INPUT / "tree_transitions.jsonl.gz"))
    after = audit(INPUT)
    equal(before, after)
    if sha(INPUT / "inventory.json") != input_hash:
        raise AssertionError("input inventory changed")
    locked_files(SOURCES + [str(CONFIG), str(REFERENCE)], "HEAD")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", result)
    write_json(output / "episodes.json", episodes)
    write_json(output / "decisions.json", decisions)
    write_json(output / "provenance.json", {
        "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sources": sources, "upstream_audit_before_and_after": after,
        "input_inventory_sha256": input_hash, "new_simulator_queries": 0,
        "outputs": [{"path": str(output / n), "sha256": sha(output / n)}
                    for n in ("summary.json", "episodes.json", "decisions.json")]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
