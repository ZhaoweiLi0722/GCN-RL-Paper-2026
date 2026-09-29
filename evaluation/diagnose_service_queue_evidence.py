"""Read-only structural diagnosis using already recorded queue edges.

The alternative exact check uses a shared-prefix recurrence until the two
deterministic worlds become distinguishable, then individual-world suffixes.
It does not import the simulator, MPC, or original scenario-tree solver.
"""

import argparse
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import subprocess

from evaluation.audit_service_queue_boundary import CONFIG, audit, read_rows
from evaluation.check_service_effort_mechanics import locked_files, sha, write_json


SOURCE = "evaluation/diagnose_service_queue_evidence.py"


def observation_payload(edge):
    return json.dumps(edge["step"], sort_keys=True, allow_nan=False)


def prefix_optimum(edges, worlds, actions, horizon):
    """Independent recurrence specialized to <=2 deterministic finite worlds.

    edges[(world_name, past_actions, next_action)] contains the full public step
    and objective increment. Root observations must match. Once any public
    observation differs, remembered history identifies the deterministic world.
    """
    if not 1 <= len(worlds) <= 2 or not math.isclose(sum(p for _, p in worlds), 1):
        raise ValueError("one or two finite worlds with total probability one required")
    roots = [edges[(name, (), actions[0])]["step"]["before"] for name, _ in worlds]
    if any(root != roots[0] for root in roots):
        raise ValueError("this recurrence requires indistinguishable initial histories")

    @lru_cache(maxsize=None)
    def private_suffix(name, prefix):
        if len(prefix) == horizon:
            return 0.0, ""
        return min((edges[(name, prefix, a)]["cost"] + private_suffix(name, prefix + (a,))[0], a) for a in actions)

    @lru_cache(maxsize=None)
    def common_prefix(prefix):
        if len(prefix) == horizon:
            return 0.0, ""
        alternatives = []
        for action in actions:
            step_edges = [edges[(name, prefix, action)] for name, _ in worlds]
            immediate = sum(p * edge["cost"] for (_, p), edge in zip(worlds, step_edges))
            successor = prefix + (action,)
            distinguishable = len({observation_payload(edge) for edge in step_edges}) > 1
            if distinguishable:
                future = sum(p * private_suffix(name, successor)[0] for name, p in worlds)
            else:
                future = common_prefix(successor)[0]
            alternatives.append((immediate + future, action))
        return min(alternatives)

    value, root_action = common_prefix(())
    witness, replay = [], 0.0
    for name, probability in worlds:
        prefix, identified, cost = (), False, 0.0
        first_identified = None
        for _ in range(horizon):
            action = private_suffix(name, prefix)[1] if identified else common_prefix(prefix)[1]
            edge = edges[(name, prefix, action)]
            cost += edge["cost"]
            if not identified and len({observation_payload(edges[(w, prefix, action)]) for w, _ in worlds}) > 1:
                identified, first_identified = True, len(prefix) + 1
            prefix += (action,)
        replay += probability * cost
        witness.append({"world": name, "actions": prefix, "cost": cost,
                        "first_distinguishable_decision_epoch": first_identified})
    if not math.isclose(replay, value, abs_tol=1e-9):
        raise AssertionError("independent witness replay failed")
    return {"cost": value, "root_action": root_action, "witness": witness,
            "replayed_expected_cost": replay, "online_parameter_updates": False,
            "scope": "known finite-law history policy, five decisions then the declared common continuation"}


def diagnose(fixture, summary, edge_rows, episode_rows):
    comparisons, arrival_checks = {}, {}
    for cell, comparison in summary["comparisons"].items():
        family = next(f for f in fixture["families"] if f["name"] == cell.split("__")[2])
        edges = {(r["world"], tuple(r["past_actions"]), r["action"]): r for r in edge_rows if r["cell"] == cell}
        exact = prefix_optimum(edges, tuple((w["name"], w["probability"]) for w in family["worlds"]), tuple(fixture["actions"]), fixture["horizon"])
        if not math.isclose(exact["cost"], comparison["diagnostic"]["nonanticipative_cost"], abs_tol=1e-9):
            raise AssertionError("independent nonanticipative optimum disagrees")
        means = comparison["expected_synthetic_cost"]
        equals = [name for name, value in means.items() if math.isclose(value, exact["cost"], abs_tol=1e-9)]
        computations = {method: sorted({e["prediction_transition_queries"] for e in episode_rows if e["cell"] == cell and e["method"] == method}) for method in means}
        comparisons[cell] = {"independent_exact_check": exact,
                             "prespecified_baselines_matching_restricted_optimum": equals,
                             "planning_queries_per_episode": computations}
    keyed = {(r["cell"], r["world"], tuple(r["past_actions"]), r["action"]): r for r in edge_rows}
    for downstream in fixture["downstream_regimes"]:
        for family in fixture["families"]:
            name = downstream + "__" + family["name"]
            batch, flow = "batch__" + name, "booked_flow__" + name
            counts = {"matched_edges": 0, "support_progress_difference_edges": 0,
                      "downstream_event_difference_edges": 0, "local_cost_offset_violations": 0,
                      "matched_complete_sequences": 0, "complete_cost_offset_violations": 0}
            # Moving the same booked jobs later removes their pre-release holding
            # charges. Check, do not assume, that nothing else changes the cost.
            shifts = fixture["release_patterns"]["booked_flow"]
            expected_full_offset = 2 * fixture["holding_cost"] * sum(shifts)
            for world in family["worlds"]:
                w = world["name"]
                for t in range(fixture["horizon"]):
                    expected_local = 2 * fixture["holding_cost"] * sum(release > t for release in shifts)
                    for prefix in itertools.product(fixture["actions"], repeat=t):
                        for action in fixture["actions"]:
                            left, right = keyed[batch, w, prefix, action], keyed[flow, w, prefix, action]
                            counts["matched_edges"] += 1
                            if left["step"]["receipt"]["delivered_work"] != right["step"]["receipt"]["delivered_work"]:
                                counts["support_progress_difference_edges"] += 1
                            event_keys = ("downstream_started", "downstream_finished")
                            if any(left["step"]["cost"][k] != right["step"]["cost"][k] for k in event_keys):
                                counts["downstream_event_difference_edges"] += 1
                            if not math.isclose(left["cost"] - right["cost"], expected_local, abs_tol=1e-9):
                                counts["local_cost_offset_violations"] += 1
                for path in itertools.product(fixture["actions"], repeat=fixture["horizon"]):
                    difference = sum(keyed[batch, w, path[:t], action]["cost"] - keyed[flow, w, path[:t], action]["cost"] for t, action in enumerate(path))
                    counts["matched_complete_sequences"] += 1
                    if not math.isclose(difference, expected_full_offset, abs_tol=1e-9):
                        counts["complete_cost_offset_violations"] += 1
            arrival_checks[name] = {**counts, "expected_removed_holding_cost": expected_full_offset,
                                    "cost_equivalent_up_to_release_offset": counts["complete_cost_offset_violations"] == 0}
    return {"classification": "post_run_structural_diagnosis_no_new_simulations",
            "nonanticipative_check": "independent shared-prefix-until-disambiguation recurrence plus witness replay",
            "comparisons": comparisons, "arrival_factor_checks": arrival_checks,
            "all_cells_have_at_least_one_matching_prespecified_baseline": all(
                c["prespecified_baselines_matching_restricted_optimum"] for c in comparisons.values()),
            "qualification": "Per-cell best-baseline summary is descriptive; not a prospectively deployed hybrid controller or an RL result."}


def run(input_root, output):
    source_lock = locked_files([SOURCE, "tests/test_service_queue_evidence.py"], "HEAD")
    audit_result = audit(input_root)
    fixture = json.loads(CONFIG.read_text())
    summary = json.loads((input_root / "summary.json").read_text())
    result = diagnose(fixture, summary, read_rows(input_root / "tree_transitions.jsonl.gz"), read_rows(input_root / "episodes.jsonl.gz"))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "diagnosis.json", result)
    write_json(output / "provenance.json", {"execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                           "source_locks": source_lock, "upstream_audit": audit_result,
                                           "input_inventory_sha256": sha(input_root / "inventory.json"),
                                           "diagnosis_sha256": sha(output / "diagnosis.json"),
                                           "new_simulator_queries": 0})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
