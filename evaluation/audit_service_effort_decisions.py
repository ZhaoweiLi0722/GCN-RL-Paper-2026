"""Audit persisted finite-fixture arithmetic without executing the simulator."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


CONFIG = Path("experiments/configs/service_effort_decisions_20260929.json")


def read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON token {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def rows(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON token {value}")
    return [json.loads(line, parse_constant=reject) for line in Path(path).read_text().splitlines()]


def close(a, b):
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-12):
        raise AssertionError(f"numeric mismatch {a} != {b}")


def verify_step(row, world, fixture):
    pre, post, receipt = row["before"], row["after"], row["receipt"]
    cfg = fixture["config"]
    action, epoch = receipt["committed_hours"], pre["epoch"]
    if receipt["epoch"] != epoch or post["epoch"] != epoch + 1:
        raise AssertionError("epoch mismatch")
    if list(receipt["applied_hours"]) != list(pre["pending_hours"][0]):
        raise AssertionError("effort delay mismatch")
    if list(post["previous_commitment"]) != list(action) or [list(p) for p in post["pending_hours"]] != [list(p) for p in pre["pending_hours"][1:]] + [list(action)]:
        raise AssertionError("commitment conservation mismatch")
    if len(action) != 2 or any(x < 0 or x > cap for x, cap in zip(action, cfg["site_hour_caps"])) or sum(action) > cfg["shared_hour_budget"] + 1e-12:
        raise AssertionError("illegal action")
    for i in range(2):
        close(receipt["available_work"][i], pre["remaining_work"][i])
        amount = min(pre["remaining_work"][i], receipt["applied_hours"][i] * world["response_tape"][epoch][i])
        close(amount, receipt["delivered_work"][i])
        close(post["remaining_work"][i], pre["remaining_work"][i] - amount)
        unfinished = max(0, math.ceil(post["remaining_work"][i] - 1e-12))
        if post["unfinished_jobs"][i] != unfinished:
            raise AssertionError("whole-job mismatch")
        if post["completed_jobs"][i] != pre["completed_jobs"][i] + pre["unfinished_jobs"][i] - unfinished:
            raise AssertionError("job conservation mismatch")
        kind = "no_effort" if receipt["applied_hours"][i] == 0 else ("backlog_limited" if amount >= pre["remaining_work"][i] - 1e-12 else "uncensored")
        if receipt["observation_kind"][i] != kind:
            raise AssertionError("censoring mismatch")
    labor = sum(cfg["hourly_cost"] * x + cfg["quadratic_cost"] * x * x for x in action)
    change = cfg["switching_cost"] * sum(abs(x - p) for x, p in zip(action, pre["previous_commitment"]))
    holding = sum(w * c for w, c in zip(post["remaining_work"], fixture["holding_weights"]))
    close(labor, receipt["labor_cost"])
    close(change, receipt["change_cost"])
    close(holding, row["holding_cost"])
    close(labor + change + holding, row["cost"])
    return row["cost"]


def verify_settlement(ledger, before, world, fixture):
    total = 0.0
    if len(ledger["drain_rows"]) != fixture["config"]["commitment_lead_steps"]:
        raise AssertionError("incorrect drain duration")
    for row in ledger["drain_rows"]:
        if row["before"] != before or any(row["receipt"]["committed_hours"]):
            raise AssertionError("unaccounted terminal action")
        total += verify_step(row, world, fixture)
        before = row["after"]
    if any(sum(p) for p in before["pending_hours"]) or ledger["external_service_work"] != before["remaining_work"]:
        raise AssertionError("lost terminal obligation")
    if ledger["external_service_jobs"] != before["unfinished_jobs"] or any(ledger["remaining_work_after_external_service"]):
        raise AssertionError("terminal job closure mismatch")
    external = fixture["terminal_work_cost"] * sum(before["remaining_work"])
    close(external, ledger["external_service_cost"])
    close(total + external, ledger["total_settlement_cost"])
    return total + external


def audit(output):
    inventory = read_json(output / "inventory.json")
    for item in inventory["files"]:
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise AssertionError(f"hash mismatch: {item['path']}")
    fixture, summary, status = read_json(CONFIG), read_json(output / "summary.json"), read_json(output / "status.json")
    if status != {"status": "completed", "exit_code": 0, "fixture_only": True}:
        raise AssertionError("noncompleted fixture")
    episodes, decisions, tree = rows(output / "episodes.jsonl"), rows(output / "decisions.jsonl"), rows(output / "tree_transitions.jsonl")
    if (len(episodes), len(decisions), len(tree)) != (15, 75, 4092):
        raise AssertionError("incorrect artifact counts")
    families = {f["name"]: f for f in fixture["families"]}
    methods = {"fixed_balanced", "backlog_rule", "adaptive_backlog_rule", "fixed_model_mpc", "identification_mpc"}
    expected = {(f["name"], w["name"], m) for f in fixture["families"] for w in f["worlds"] for m in methods}
    keys = [(e["family"], e["world"], e["method"]) for e in episodes]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise AssertionError("missing or duplicate baseline episodes")
    means = {name: {m: 0.0 for m in methods} for name in families}
    for episode in episodes:
        f, w, method = episode["family"], episode["world"], episode["method"]
        world = next(x for x in families[f]["worlds"] if x["name"] == w)
        series = sorted((r for r in decisions if (r["family"], r["world"], r["method"]) == (f, w, method)), key=lambda r: r["before"]["epoch"])
        if len(series) != 5 or [r["before"]["epoch"] for r in series] != list(range(5)):
            raise AssertionError("missing/duplicate baseline decision")
        estimate, total = fixture["prior"].copy(), 0.0
        for t, row in enumerate(series):
            if t and row["before"] != series[t - 1]["after"]:
                raise AssertionError("broken public history")
            response = estimate if method in ("adaptive_backlog_rule", "identification_mpc") else fixture["prior"]
            if row["response_used_before_action"] != response:
                raise AssertionError("unobserved information used by controller")
            if row["action"] != episode["decisions"][t] or row["receipt"]["committed_hours"] != fixture["actions"][row["action"]]:
                raise AssertionError("action identity mismatch")
            total += verify_step(row, world, fixture)
            receipt = row["receipt"]
            for i in range(2):
                if receipt["observation_kind"][i] == "uncensored":
                    alpha = fixture["estimator_alpha"]
                    estimate[i] = (1 - alpha) * estimate[i] + alpha * receipt["delivered_work"][i] / receipt["applied_hours"][i]
        total += verify_settlement(episode["settlement"], series[-1]["after"], world, fixture)
        close(total, episode["total_cost"])
        means[f][method] += world["probability"] * total
    for name, costs in means.items():
        for method, cost in costs.items():
            close(cost, summary["comparisons"][name]["expected_synthetic_cost"][method])
    edge = {}
    for item in tree:
        key = (item["family"], item["world"], tuple(item["past_actions"]), item["action"])
        if key in edge:
            raise AssertionError("duplicate exact-tree edge")
        world = families[item["family"]]["worlds"][item["world"]]
        cost = verify_step(item["step"], world, fixture)
        if item["settlement"] is not None:
            if len(item["past_actions"]) != 4:
                raise AssertionError("premature settlement")
            cost += verify_settlement(item["settlement"], item["step"]["after"], world, fixture)
        close(cost, item["cost"])
        edge[key] = item
    for name, family in families.items():
        best_open, best_world = math.inf, [math.inf] * len(family["worlds"])
        for sequence in itertools.product(fixture["actions"], repeat=5):
            expectation = 0.0
            for i, world in enumerate(family["worlds"]):
                cost, previous = 0.0, None
                for t, action in enumerate(sequence):
                    item = edge[(name, i, sequence[:t], action)]
                    if previous is not None and previous != item["step"]["before"]:
                        raise AssertionError("broken exact-tree state chain")
                    previous = item["step"]["after"]
                    cost += item["cost"]
                best_world[i] = min(best_world[i], cost)
                expectation += world["probability"] * cost
            best_open = min(best_open, expectation)
        d = summary["comparisons"][name]["finite_grid_diagnostic"]
        close(best_open, d["best_open_loop_cost"])
        close(sum(w["probability"] * v for w, v in zip(family["worlds"], best_world)), d["clairvoyant_cost"])
        if not d["clairvoyant_cost"] <= d["nonanticipative_cost"] + 1e-10 <= d["best_open_loop_cost"] + 2e-10:
            raise AssertionError("diagnostic ordering mismatch")
    return {"audit": "passed", "baseline_episodes": 15, "decision_rows": 75,
            "tree_transition_rows": 4092, "file_hashes": len(inventory["files"]),
            "scope": "independent costs, timing, evidence parity, open-loop/clairvoyant enumeration; nonanticipative solver is regression-tested, not independently reimplemented"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2))
