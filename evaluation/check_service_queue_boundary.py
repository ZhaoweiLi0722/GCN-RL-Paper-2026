"""Frozen eight-cell synthetic queue check; no patient or neural experiment."""

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

from evaluation.check_service_effort_mechanics import locked_files, sha, write_json
from src.baselines.public_service_response import PublicServiceResponseEstimator
from src.baselines.service_queue_control import backlog_choice, plan_queue_mpc
from src.env.service_effort_mechanics import ServiceEffortConfig
from src.env.service_queue_network import BookedJob, QueueConfig, advance, close_episode, initial_observation
from src.utils.finite_scenario_tree import solve_fixture


CONFIG = Path("experiments/configs/service_queue_boundary_20260929.json")
REFERENCE = "5f8226a"
METHODS = ("fixed_balanced", "backlog_rule", "adaptive_backlog_rule", "fixed_model_mpc", "identification_mpc")
SOURCES = [str(CONFIG), "src/env/service_queue_network.py", "src/baselines/service_queue_control.py",
           "evaluation/check_service_queue_boundary.py", "evaluation/audit_service_queue_boundary.py",
           "tests/test_service_queue_network.py", "specs/2026-09-29-service-queue-boundary/protocol.md"]
PRESERVED_ROOTS = [
    "src/env", "src/baselines/public_service_response.py", "src/baselines/service_effort_control.py",
    "src/utils/finite_scenario_tree.py", "evaluation/check_service_effort_mechanics.py",
    "evaluation/check_service_effort_decisions.py", "evaluation/audit_service_effort_decisions.py",
    "reports/2026-09-28-disruption-feasibility", "reports/2026-09-28-service-effort-mechanics",
    "reports/2026-09-29-service-effort-decisions",
]


def validate(fixture):
    if fixture["role"] != "synthetic_mechanism_boundary_not_prm_or_rl_confirmation" or fixture["horizon"] != 5:
        raise ValueError("only the frozen five-decision mechanism scope is allowed")
    if fixture["release_patterns"] != {"batch": [0, 0, 0, 0], "booked_flow": [0, 0, 2, 3]}:
        raise ValueError("release factorial mismatch")
    if fixture["downstream_regimes"] != {"nonbinding": 8, "shared_bottleneck": 1}:
        raise ValueError("downstream factorial mismatch")
    if fixture["actions"] != {"idle": [0, 0], "left": [1, 0], "right": [0, 1], "balanced": [0.5, 0.5]}:
        raise ValueError("action grid mismatch")
    if [f["name"] for f in fixture["families"]] != ["no_change", "persistent_change"]:
        raise ValueError("both response families required")
    for family in fixture["families"]:
        weights = [w["probability"] for w in family["worlds"]]
        if not 1 <= len(weights) <= 2 or any(not math.isfinite(p) or p <= 0 for p in weights) or not math.isclose(sum(weights), 1):
            raise ValueError("invalid finite-world law")
        if len({w["name"] for w in family["worlds"]}) != len(weights):
            raise ValueError("duplicate world identity")
        for world in family["worlds"]:
            if world["change_epoch"] != 2 or len(world["after"]) != 2 or any(x not in (0.5, 1.0, 1.5) for x in world["after"]):
                raise ValueError("response law outside protocol")
    for pattern in fixture["release_patterns"]:
        for downstream in fixture["downstream_regimes"]:
            make_config(fixture, pattern, downstream)


def make_config(fixture, pattern, downstream):
    jobs = tuple(BookedJob(f"site{site}_job{k}", site, release, fixture["job_work"], fixture["downstream_steps"])
                 for site in (0, 1) for k, release in enumerate(fixture["release_patterns"][pattern]))
    return QueueConfig(ServiceEffortConfig(**fixture["effort"]), jobs,
                       fixture["downstream_regimes"][downstream], fixture["holding_cost"], fixture["max_settlement_steps"])


def cells(fixture):
    for pattern in fixture["release_patterns"]:
        for downstream in fixture["downstream_regimes"]:
            config = make_config(fixture, pattern, downstream)
            for family in fixture["families"]:
                yield "__".join((pattern, downstream, family["name"])), config, family


def response_at(world, epoch):
    return tuple(world["after"]) if epoch >= world["change_epoch"] else (1.0, 1.0)


def take_step(observation, action, config, world):
    after, receipt, cost = advance(observation, action, response_at(world, observation.epoch), config)
    return after, receipt, {"before": asdict(observation), "after": asdict(after),
                            "receipt": asdict(receipt), "cost": cost}


def settlement(observation, config, world):
    cost, final, rows = close_episode(observation, config, lambda epoch: response_at(world, epoch), record=True)
    return {"cost": cost, "final": asdict(final), "steps": rows}


class QueueTree:
    def __init__(self, fixture, cell, config, family):
        self.fixture, self.cell, self.config, self.family = fixture, cell, config, family
        initial = initial_observation(config)
        history = (json.dumps(asdict(initial), sort_keys=True),)
        self.nodes = {(i, ()): (initial, history) for i in range(len(family["worlds"]))}
        self.cache, self.records, self.closures = {}, [], {}

    def information_key(self, state):
        return self.nodes[state][1]

    def transition(self, state, action):
        edge = state, action
        if edge in self.cache:
            return self.cache[edge]
        world_id, path = state
        if len(path) >= self.fixture["horizon"]:
            raise ValueError("decision horizon exhausted")
        observation, history = self.nodes[state]
        world = self.family["worlds"][world_id]
        after, _, row = take_step(observation, self.fixture["actions"][action], self.config, world)
        successor = world_id, path + (action,)
        self.nodes[successor] = after, history + ((action, json.dumps(row, sort_keys=True)),)
        closure_id = None
        if len(path) + 1 == self.fixture["horizon"]:
            digest = hashlib.sha256(json.dumps(asdict(after), sort_keys=True).encode()).hexdigest()
            closure_id = "/".join((self.cell, world["name"], digest))
            if closure_id not in self.closures:
                self.closures[closure_id] = {"id": closure_id, "cell": self.cell, "world": world["name"],
                                            "before": asdict(after), "settlement": settlement(after, self.config, world)}
        cost = row["cost"]["total"] + (self.closures[closure_id]["settlement"]["cost"] if closure_id else 0.0)
        self.cache[edge] = cost, successor
        self.records.append({"cell": self.cell, "world": world["name"], "past_actions": path,
                             "action": action, "cost": cost, "step": row, "settlement_id": closure_id})
        return cost, successor


def evaluate(fixture, cell, config, world, method):
    observation = initial_observation(config)
    estimator = PublicServiceResponseEstimator(fixture["prior"], fixture["estimator_alpha"])
    rows, total, queries = [], 0.0, 0
    for epoch in range(fixture["horizon"]):
        estimate = estimator.estimate if method in ("adaptive_backlog_rule", "identification_mpc") else tuple(fixture["prior"])
        if method == "fixed_balanced":
            name = "balanced"
        elif method in ("backlog_rule", "adaptive_backlog_rule"):
            name = backlog_choice(observation, estimate, config)
        elif method in ("fixed_model_mpc", "identification_mpc"):
            name, _, count = plan_queue_mpc(observation, estimate, config, fixture["actions"], fixture["horizon"] - epoch)
            queries += count
        else:
            raise ValueError("unknown method")
        observation, receipt, row = take_step(observation, fixture["actions"][name], config, world)
        estimator.observe(receipt)
        row.update({"cell": cell, "world": world["name"], "method": method, "action": name,
                    "response_before": estimate, "estimator_after": estimator.estimate})
        rows.append(row)
        total += row["cost"]["total"]
    closure = settlement(observation, config, world)
    return {"cell": cell, "world": world["name"], "method": method,
            "cost": total + closure["cost"], "decision_cost": total,
            "actions": [r["action"] for r in rows], "prediction_transition_queries": queries,
            "settlement": closure}, rows


def compute(fixture):
    validate(fixture)
    comparisons, episodes, decisions, transitions, closures = {}, [], [], [], []
    for cell, config, family in cells(fixture):
        tree = QueueTree(fixture, cell, config, family)
        diagnostic = solve_fixture(tree, worlds=tuple((w["probability"], (i, ())) for i, w in enumerate(family["worlds"])),
                                   actions=tuple(fixture["actions"]), horizon=fixture["horizon"])
        transitions.extend(tree.records)
        closures.extend(tree.closures.values())
        means = {}
        for method in METHODS:
            mean = 0.0
            for world in family["worlds"]:
                episode, rows = evaluate(fixture, cell, config, world, method)
                episodes.append(episode)
                decisions.extend(rows)
                mean += world["probability"] * episode["cost"]
            if mean < diagnostic["nonanticipative_cost"] - 1e-8:
                raise AssertionError("feasible baseline beat restricted exact diagnostic")
            means[method] = mean
        comparisons[cell] = {"expected_synthetic_cost": means, "diagnostic": diagnostic,
                             "feedback_value_over_open_loop": diagnostic["best_open_loop_cost"] - diagnostic["nonanticipative_cost"],
                             "identification_gain_over_fixed_mpc": means["fixed_model_mpc"] - means["identification_mpc"],
                             "identification_gap_to_diagnostic": means["identification_mpc"] - diagnostic["nonanticipative_cost"]}
        if family["name"] == "no_change" and not math.isclose(means["fixed_model_mpc"], means["identification_mpc"], abs_tol=1e-10):
            raise AssertionError("no-change MPC negative control failed")
    return {"classification": fixture["role"], "online_weight_updates": False, "patient_model_run": False,
            "operational_calibration_validated": False, "independent_replications": 0,
            "comparisons": comparisons, "baseline_episodes": len(episodes), "decision_rows": len(decisions),
            "exact_tree_edges": len(transitions), "distinct_tree_closures": len(closures)}, episodes, decisions, transitions, closures


def write_compressed(path, rows):
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, allow_nan=False, sort_keys=True) + "\n").encode())


def run(output):
    sources = locked_files(SOURCES, "HEAD")
    preserved_paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", REFERENCE, "--", *PRESERVED_ROOTS], text=True).splitlines()
    preserved = locked_files(preserved_paths, REFERENCE)
    fixture = json.loads(CONFIG.read_text())
    validate(fixture)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "claim.json", {"execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                       "sources": sources, "preserved": preserved, "role": fixture["role"]})
    try:
        summary, episodes, rows, transitions, closures = compute(fixture)
        for name, values in (("episodes", episodes), ("decisions", rows), ("tree_transitions", transitions), ("tree_closures", closures)):
            write_compressed(output / f"{name}.jsonl.gz", values)
        write_json(output / "summary.json", summary)
        locked_files(preserved_paths, REFERENCE)
        write_json(output / "status.json", {"status": "completed", "exit_code": 0, "synthetic_only": True})
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
