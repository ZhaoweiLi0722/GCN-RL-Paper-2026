"""Read archived JSON receipts only; never import an environment or learner.

Reconcile incurred costs with raw rewards independently of the replay helpers.
Distinguish this numerical check from certification of a complete task objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/reward_objective_bridge_20260929.json"


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite numeric receipt required, not a flag or string")
    return float(value)


def close(actual, expected, label):
    if not math.isclose(finite(actual), finite(expected), rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(label)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def read_json(path):
    return json.loads(path.read_text(), object_pairs_hook=_pairs,
                      parse_constant=_invalid_constant)


def check_case(case, config, proposed):
    events = case["events"]
    n = config["expected_steps_per_case"]
    if len(events) != n or case["final"]["steps"] != n:
        raise ValueError("missing or extra receipt")
    records = [event["receipt"]["record"] for event in events]
    semantics = records[0]["semantics"]
    gamma, scale = finite(semantics["gamma"]), finite(semantics["reward_scale"])
    if not 0 <= gamma <= 1 or scale <= 0 or semantics["reward_kind"] != "absolute_environment":
        raise ValueError("invalid absolute cost reward contract")
    if type(semantics["bootstrap_on_truncation"]) is not bool:
        raise ValueError("explicit truncation policy required")
    costs, rewards, subtotals = [], [], []
    previous = None
    for i, event in enumerate(events):
        record, execution = event["receipt"]["record"], event["receipt"]["execution"]
        if (record["semantics"] != semantics or record["origin"] != "trajectory"
                or type(record["step_index"]) is not int or record["step_index"] != i):
            raise ValueError("mixed semantics or trajectory index")
        for flag in ("terminated", "truncated"):
            if type(record[flag]) is not bool:
                raise ValueError("explicit boundary flags required")
        if previous is not None:
            if (previous["next_state"] != record["state"]
                    or previous["next_state_token"] != record["state_token"]
                    or previous["trajectory_id"] != record["trajectory_id"]
                    or previous["source_id"] != record["source_id"]):
                raise ValueError("disconnected recorded trajectory")
        if i < n - 1 and (record["terminated"] or record["truncated"]):
            raise ValueError("trajectory crosses a boundary")
        cost = finite(execution["cost"])
        subtotal = math.fsum(finite(execution[k]) for k in config["top_level_cost_components"])
        close(cost, subtotal, "cost components do not reconcile")
        reward = finite(record["raw_reward"])
        close(reward, -cost, "reward is not negative incurred cost")
        # Specimen-transfer cost is within base_cost, never an extra top-level term.
        costs.append(cost)
        subtotals.append(subtotal)
        rewards.append(reward)
        previous = record
    last = records[-1]
    if not (last["terminated"] or last["truncated"]):
        raise ValueError("archive is an open segment")
    spec = case["specification"]
    if spec["horizon_end"] not in ("terminal", "truncation"):
        raise ValueError("unknown boundary mapping")
    terminal = n >= spec["episode_horizon"] and spec["horizon_end"] == "terminal"
    if last["terminated"] != terminal or last["truncated"] == terminal:
        raise ValueError("boundary mapping differs from declared engineering case")
    lengths = [min(config["archived_return_steps"], n - i) for i in range(n)]
    if case["final"]["emitted_lengths"] != lengths or case["final"]["pending_count"] != 0:
        raise ValueError("missing or duplicated return tail")
    # Deliberately no estimated Q term: these are recorded segment returns only.
    windows = []
    for i, length in enumerate(lengths):
        end = records[i + length - 1]
        bootstrap = not end["terminated"] and (
            not end["truncated"] or semantics["bootstrap_on_truncation"])
        windows.append({"start": i, "n_steps": length,
                        "scaled_return": math.fsum(gamma ** j * rewards[i + j] * scale
                                                   for j in range(length)),
                        "bootstrap_discount": gamma ** length if bootstrap else 0.0})
    incurred = math.fsum(costs)
    discounted = math.fsum(gamma ** i * cost for i, cost in enumerate(costs))
    scaled = math.fsum(reward * scale for reward in rewards)
    close(scaled, -scale * incurred, "scale applied inconsistently")
    mismatches = []
    if gamma != proposed["gamma"]:
        mismatches.append("discounted_engineering_return_not_proposed_undiscounted_objective")
    if config["archived_return_steps"] != proposed["return_steps"]:
        mismatches.append("multi_step_fixture_not_proposed_one_step_replay")
    if last["truncated"]:
        mismatches.append("recorded_segment_has_bootstrap_continuation_not_complete_return")
    active = finite(events[-1]["receipt"]["execution"]["identity_active_count"])
    if active < 0 or not active.is_integer():
        raise ValueError("invalid active patient count")
    return {"case": case["case"], "accounting_passed": True, "steps": n,
            "gamma": gamma, "scale": scale, "raw_cost_sum": incurred,
            "top_level_components_sum": math.fsum(subtotals),
            "scaled_undiscounted_segment_reward": scaled,
            "discounted_segment_cost": discounted,
            "discount_weighting_difference": incurred - discounted,
            "first_step_weight": 1.0, "last_step_weight": gamma ** (n - 1),
            "terminal_mask": last["terminated"], "truncated": last["truncated"],
            "active_patients_at_segment_end": active,
            "complete_liability_settlement_verified": False,
            "settlement_reason": "active patient count is not a commitment/inventory/closure certificate",
            "objective_differences": mismatches, "recomputed_windows": windows,
            "learned_q_targets_independently_recomputed": False}


def audit(config, root=ROOT):
    before, inputs = {}, {}

    def locked(path, digest):
        actual = sha256(root / path)
        if actual != digest:
            raise ValueError("input/source hash mismatch: " + path)
        before[path] = actual
        return root / path

    for key, entry in config["inputs"].items():
        inputs[key] = read_json(locked(entry["path"], entry["sha256"]))
    archive, design, domain = (inputs[k] for k in ("receipts", "design", "domain"))
    if archive["kind"] != "bounded_real_session_mechanics_not_performance" or archive["passed"] is not True:
        raise ValueError("not the completed engineering archive")
    if archive["clinical_terminal_settlement_claimed"] is not False:
        raise ValueError("review changed settlement evidence before reusing this audit")
    for path, digest in archive["source_sha256"].items():
        locked(path, digest)
    ids = [case["case"] for case in archive["cases"]]
    if len(ids) != len(set(ids)) or sorted(ids) != sorted(config["expected_case_ids"]):
        raise ValueError("duplicate, missing or unexpected case")
    folder = Path(config["inputs"]["receipts"]["path"]).parent
    results = []
    for case in archive["cases"]:
        key = case["case"]
        resumed = read_json(locked(str(folder / (key + "-resumed.json")), config["resumed_sha256"][key]))
        if resumed["events"] != case["events"] or resumed["final"] != case["final"]:
            raise ValueError("continuous and resumed recorded receipts differ")
        results.append(check_case(case, config, design["proposed_training_objective"]))
    for path, digest in before.items():
        if sha256(root / path) != digest:
            raise ValueError("input changed during read-only audit: " + path)
    return {"kind": config["kind"], "accounting_audit_passed": True,
            "unique_case_count": len(results), "receipt_count": sum(r["steps"] for r in results),
            "resumed_duplicates_checked_not_extra_replicates": len(results),
            "source_locks_verified": len(archive["source_sha256"]),
            "input_sha256": before, "cases": results,
            "missing_domain_input_ids": [x["id"] for x in domain["domain_inputs"] if x["value"] is None],
            "unfilled_design_evidence_fields": [k for k, v in design["scientific_launch_evidence"].items() if v is None],
            "design_evidence_note": "unfilled template entries are not proof engineering evidence is absent; N6/N7 exist",
            "scientific_launch_authorized": False, "performance_or_reward_improvement_claimed": False,
            "environment_queries": 0, "optimizer_updates": 0,
            "scope_limit": "JSON arithmetic and lineage, not model replay, physical calibration or full liability settlement"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(read_json(args.config))
    result["auditor_sha256"] = sha256(Path(__file__))
    result["config_sha256"] = sha256(args.config)
    result["execution_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
