"""Re-solve recorded finite trees under explicit public-measurement contracts.

Post-run diagnostic, no new dynamics, trajectories or tuning. All modes know
the same finite-world law. This is value-of-information analysis, not a fair
comparison of the original full-progress baselines to a restricted controller.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess

from evaluation.audit_service_queue_boundary import CONFIG, audit, read_rows
from evaluation.check_service_effort_mechanics import locked_files, sha, write_json
from evaluation.diagnose_service_queue_evidence import prefix_optimum


MODES = ("exact_progress", "completion_events", "no_feedback")
SOURCE = "evaluation/check_service_queue_observation_contract.py"


def project_edge(edge, mode):
    if mode not in MODES:
        raise ValueError("unknown measurement contract")
    row = copy.deepcopy(edge)
    step = row["step"]
    if mode == "no_feedback":
        row["step"] = {"before": {"epoch": step["before"]["epoch"]},
                       "after": {"epoch": step["after"]["epoch"]}}
    elif mode == "completion_events":
        for key in ("before", "after"):
            del step[key]["remaining_work"]
        for key in ("available_work", "delivered_work", "observation_kind"):
            del step["receipt"][key]
    # Objective increments remain for offline optimization, never included in
    # the observation key apart from the explicitly retained public step cost.
    return row


def compute(fixture, summary, rows):
    result = {}
    for cell, prior in summary["comparisons"].items():
        family = next(f for f in fixture["families"] if f["name"] == cell.split("__")[2])
        local = [row for row in rows if row["cell"] == cell]
        modes = {}
        for mode in MODES:
            edges = {(r["world"], tuple(r["past_actions"]), r["action"]): project_edge(r, mode) for r in local}
            modes[mode] = prefix_optimum(edges, tuple((w["name"], w["probability"]) for w in family["worlds"]),
                                         tuple(fixture["actions"]), fixture["horizon"])
        full, partial, none = (modes[m]["cost"] for m in MODES)
        if not full <= partial + 1e-9 <= none + 2e-9:
            raise AssertionError("measurement information ordering violated")
        if not math.isclose(full, prior["diagnostic"]["nonanticipative_cost"], abs_tol=1e-9):
            raise AssertionError("full-measurement reproduction mismatch")
        if not math.isclose(none, prior["diagnostic"]["best_open_loop_cost"], abs_tol=1e-9):
            raise AssertionError("no-feedback open-loop reproduction mismatch")
        result[cell] = {"contracts": modes, "cost_of_hiding_exact_progress": partial - full,
                        "completion_feedback_value_over_open_loop": none - partial}
    return {"classification": "post_run_measurement_contract_diagnostic", "new_simulator_queries": 0,
            "online_weight_updates": False, "operational_measurements_validated": False,
            "results": result,
            "limitation": "Known finite-world law and fixed continuation; completion-only practical baselines and independent deployment validation have not been run."}


def run(input_root, output):
    sources = locked_files([SOURCE, "tests/test_service_queue_observation_contract.py",
                            "evaluation/diagnose_service_queue_evidence.py"], "HEAD")
    verified = audit(input_root)
    fixture = json.loads(CONFIG.read_text())
    summary = json.loads((input_root / "summary.json").read_text())
    result = compute(fixture, summary, read_rows(input_root / "tree_transitions.jsonl.gz"))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "observation_contract.json", result)
    write_json(output / "provenance.json", {"execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                           "sources": sources, "upstream_audit": verified,
                                           "input_inventory_sha256": sha(input_root / "inventory.json"),
                                           "result_sha256": sha(output / "observation_contract.json"),
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
