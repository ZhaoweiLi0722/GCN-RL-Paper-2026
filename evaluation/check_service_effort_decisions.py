"""Small finite-action decision fixture, never a clinical or RL performance run."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess

from evaluation.check_service_effort_mechanics import locked_files, sha, write_json, IMMUTABLE
from src.baselines.public_service_response import PublicServiceResponseEstimator
from src.baselines.service_effort_control import backlog_action, plan_public_mpc
from src.env.service_effort_mechanics import PublicServiceReceipt, ServiceEffortConfig, ServiceEffortMechanics
from src.utils.finite_scenario_tree import solve_fixture


CONFIG = Path("experiments/configs/service_effort_decisions_20260929.json")
REFERENCE = "1aae687c567eec4f8145b07eb95abd89f40aa32f"
SOURCES = [str(CONFIG), "src/baselines/service_effort_control.py",
           "evaluation/check_service_effort_decisions.py", "evaluation/audit_service_effort_decisions.py",
           "tests/test_service_effort_decisions.py",
           "specs/2026-09-29-service-effort-decisions/protocol.md"]
PRESERVED = IMMUTABLE + ["src/env/service_effort_mechanics.py", "src/baselines/public_service_response.py",
                        "src/utils/finite_scenario_tree.py", "evaluation/check_service_effort_mechanics.py"]
PRESERVED += ["reports/2026-09-28-service-effort-mechanics/fixture/" + name for name in
              ("claim.json", "inventory.json", "status.json", "summary.json", "transitions.jsonl")]
METHODS = ("fixed_balanced", "backlog_rule", "adaptive_backlog_rule", "fixed_model_mpc", "identification_mpc")


def validate(fixture):
    if fixture["role"] != "decision_fixture_only_not_prm_performance" or fixture["horizon"] != 5:
        raise ValueError("only the declared five-decision fixture is supported")
    config = ServiceEffortConfig(**fixture["config"])
    if len(config.site_hour_caps) != 2 or config.commitment_lead_steps != 1:
        raise ValueError("two sites and one delayed interval required")
    if len(fixture["jobs"]) != 2 or any(not row or any(x != 1.0 for x in row) for row in fixture["jobs"]):
        raise ValueError("homogeneous unit-work jobs required")
    if set(fixture["actions"]) != {"idle", "left", "right", "balanced"}:
        raise ValueError("four declared actions required")
    if len(fixture["holding_weights"]) != 2 or any(not math.isfinite(x) or x < 0 for x in fixture["holding_weights"]):
        raise ValueError("invalid holding weights")
    if not math.isfinite(fixture["terminal_work_cost"]) or fixture["terminal_work_cost"] <= 0:
        raise ValueError("positive terminal service charge required")
    if [f["name"] for f in fixture["families"]] != ["no_change", "persistent_change"]:
        raise ValueError("both declared families required")
    for family in fixture["families"]:
        probabilities = [w["probability"] for w in family["worlds"]]
        if not 1 <= len(probabilities) <= 2 or any(not math.isfinite(p) or p <= 0 for p in probabilities) or not math.isclose(sum(probabilities), 1):
            raise ValueError("invalid finite world probabilities")
        for world in family["worlds"]:
            if len(world["response_tape"]) != 6:
                raise ValueError("five decisions plus one commitment-settlement interval required")
            ServiceEffortMechanics(config, fixture["jobs"], world["response_tape"])
    return config


def public_key(observation):
    return json.dumps(asdict(observation), sort_keys=True, allow_nan=False)


def operating_step(model, action, fixture):
    before = model.observation()
    after, receipt = model.step(action)
    holding = sum(w * c for w, c in zip(after.remaining_work, fixture["holding_weights"]))
    cost = receipt.labor_cost + receipt.change_cost + holding
    return cost, {"before": asdict(before), "after": asdict(after), "receipt": asdict(receipt),
                  "holding_cost": holding, "cost": cost}


def settle(model, fixture):
    rows, cost = [], 0.0
    for _ in range(model.config.commitment_lead_steps):
        amount, row = operating_step(model, (0.0, 0.0), fixture)
        rows.append(row)
        cost += amount
    observation = model.observation()
    if any(sum(row) for row in observation.pending_hours):
        raise AssertionError("unsettled effort commitment")
    external = fixture["terminal_work_cost"] * sum(observation.remaining_work)
    return cost + external, {
        "drain_rows": rows, "external_service_work": observation.remaining_work,
        "external_service_jobs": observation.unfinished_jobs, "external_service_cost": external,
        "settlement_scope": "synthetic external service of all remaining work; not a patient release or calibrated outsourcing model",
        "remaining_work_after_external_service": [0.0, 0.0], "total_settlement_cost": cost + external,
    }


class ServiceDecisionTree:
    """Known finite-law diagnostic; private world never enters the history key."""

    def __init__(self, fixture, family):
        self.fixture, self.family = fixture, family
        self.nodes, self.cache, self.records = {}, {}, []
        config = ServiceEffortConfig(**fixture["config"])
        for i, world in enumerate(family["worlds"]):
            model = ServiceEffortMechanics(config, fixture["jobs"], world["response_tape"])
            self.nodes[(i, ())] = (model, (public_key(model.observation()),))

    def information_key(self, state):
        return self.nodes[state][1]

    def transition(self, state, name):
        edge = (state, name)
        if edge in self.cache:
            return self.cache[edge]
        world, path = state
        if len(path) >= self.fixture["horizon"]:
            raise ValueError("decision horizon exhausted")
        original, history = self.nodes[state]
        model = copy.deepcopy(original)
        cost, row = operating_step(model, self.fixture["actions"][name], self.fixture)
        public_history = history + ((name, json.dumps(row, sort_keys=True)),)
        settlement = None
        if len(path) + 1 == self.fixture["horizon"]:
            terminal_cost, settlement = settle(model, self.fixture)
            cost += terminal_cost
        successor = (world, path + (name,))
        self.nodes[successor] = (model, public_history)
        self.cache[edge] = (cost, successor)
        self.records.append({"family": self.family["name"], "world": world, "past_actions": path,
                             "action": name, "cost": cost, "step": row, "settlement": settlement})
        return cost, successor


def evaluate_method(fixture, family, world, method):
    config = ServiceEffortConfig(**fixture["config"])
    model = ServiceEffortMechanics(config, fixture["jobs"], world["response_tape"])
    estimator = PublicServiceResponseEstimator(fixture["prior"], fixture["estimator_alpha"])
    records, total, queries = [], 0.0, 0
    for epoch in range(fixture["horizon"]):
        observation = model.observation()
        response = estimator.estimate if method in ("adaptive_backlog_rule", "identification_mpc") else tuple(fixture["prior"])
        if method == "fixed_balanced":
            name = "balanced"
        elif method in ("backlog_rule", "adaptive_backlog_rule"):
            name = backlog_action(observation, fixture["actions"], response)
        elif method in ("fixed_model_mpc", "identification_mpc"):
            name, _, count = plan_public_mpc(observation, response, config, fixture["actions"],
                                             fixture["horizon"] - epoch, fixture["holding_weights"], fixture["terminal_work_cost"])
            queries += count
        else:
            raise ValueError("unknown baseline")
        # The controller gets only the public observation and its own receipts.
        cost, row = operating_step(model, fixture["actions"][name], fixture)
        row.update({"family": family["name"], "world": world["name"], "method": method,
                    "action": name, "response_used_before_action": response})
        total += cost
        estimator.observe(PublicServiceReceipt(**row["receipt"]))
        records.append(row)
    terminal_cost, ledger = settle(model, fixture)
    total += terminal_cost
    return {"family": family["name"], "world": world["name"], "method": method,
            "total_cost": total, "decisions": [r["action"] for r in records],
            "prediction_transition_queries": queries, "settlement": ledger}, records


def compute(fixture):
    validate(fixture)
    comparisons, episodes, rows, tree_rows = {}, [], [], []
    for family in fixture["families"]:
        model = ServiceDecisionTree(fixture, family)
        diagnostic = solve_fixture(model, worlds=tuple((w["probability"], (i, ())) for i, w in enumerate(family["worlds"])),
                                   actions=tuple(fixture["actions"]), horizon=fixture["horizon"])
        tree_rows.extend(model.records)
        means = {}
        for method in METHODS:
            total = 0.0
            for world in family["worlds"]:
                episode, records = evaluate_method(fixture, family, world, method)
                episodes.append(episode)
                rows.extend(records)
                total += world["probability"] * episode["total_cost"]
            means[method] = total
            if total < diagnostic["nonanticipative_cost"] - 1e-8:
                raise AssertionError("a feasible baseline beat the exact same-grid known-law optimum")
        comparisons[family["name"]] = {"expected_synthetic_cost": means, "finite_grid_diagnostic": diagnostic,
                                      "identification_minus_fixed_mpc": means["identification_mpc"] - means["fixed_model_mpc"]}
    nominal = comparisons["no_change"]["expected_synthetic_cost"]
    if not math.isclose(nominal["identification_mpc"], nominal["fixed_model_mpc"], abs_tol=1e-12):
        raise AssertionError("no-change identification negative control failed")
    return {"classification": "decision_software_fixture_not_prm_or_rl_evidence",
            "clinical_model_validated": False, "online_policy_training": False, "graph_benefit_tested": False,
            "independent_statistical_replications": 0, "comparisons": comparisons,
            "baseline_episodes": len(episodes), "baseline_decision_rows": len(rows), "exact_tree_transition_rows": len(tree_rows)}, episodes, rows, tree_rows


def run(output):
    sources, preserved = locked_files(SOURCES, "HEAD"), locked_files(PRESERVED, REFERENCE)
    fixture = json.loads(CONFIG.read_text())
    validate(fixture)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "claim.json", {"execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                       "role": fixture["role"], "sources": sources, "preserved": preserved})
    try:
        summary, episodes, rows, tree_rows = compute(fixture)
        for name, values in (("episodes", episodes), ("decisions", rows), ("tree_transitions", tree_rows)):
            (output / f"{name}.jsonl").write_text("".join(json.dumps(x, sort_keys=True, allow_nan=False) + "\n" for x in values))
        write_json(output / "summary.json", summary)
        locked_files(PRESERVED, REFERENCE)
        write_json(output / "status.json", {"status": "completed", "exit_code": 0, "fixture_only": True})
        write_json(output / "inventory.json", {"files": sources + preserved + [
            {"path": str(p), "sha256": sha(p)} for p in sorted(output.iterdir())]})
        return summary
    except BaseException as exc:
        write_json(output / "status.json", {"status": "failed", "exit_code": 1, "error": repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
