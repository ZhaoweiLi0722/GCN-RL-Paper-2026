"""Independently recompute finite-tree values from saved transition rows only."""

import hashlib
import itertools
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path("reports/2026-09-09-online-adaptation-mechanics/fixture")
CONFIG = Path("experiments/configs/online_adaptation_information_mechanics.json")


def check(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    config, summary, claim, status, inventory = [read(p) for p in (
        CONFIG, OUT / "summary.json", OUT / "claim.json", OUT / "status.json", OUT / "inventory.json"
    )]
    check(status == {"status": "completed", "exit_code": 0}, "terminal status")
    check(summary["fixture_only"] and not summary["research_performance_evidence"], "fixture scope")
    check(not summary["training_performed"] and not summary["formal_evaluation_performed"], "execution scope")
    check(claim["execution_commit"] == summary["execution_commit"], "commit mismatch")
    for entry in inventory["files"]:
        path = Path(entry["path"])
        check(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == entry["sha256"], "inventory hash")
        if not path.is_relative_to(OUT):
            committed = subprocess.check_output(["git", "show", claim["execution_commit"] + ":" + str(path)], cwd=ROOT)
            check(hashlib.sha256(committed).hexdigest() == entry["sha256"], "input differs from execution commit")
    with (ROOT / OUT / "transitions.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    actions, horizon = tuple(config["actions"]), config["horizon"]
    expected = {(world, path, action) for world in range(2) for depth in range(horizon)
                for path in itertools.product(actions, repeat=depth) for action in actions}
    edges, information = {}, {}
    caps = [v * config["env"]["max_overtime_fraction"] for v in config["env"]["initial_idle_bioreactors"]]
    budget = sum(caps) * config["env"]["overtime_shared_budget_fraction"]
    lead = config["env"]["overtime_commitment_lead_time"]
    persistence = config["env"]["overtime_commitment_persistence"]

    def request(action):
        values = [f * c for f, c in zip(config["actions"][action], caps)]
        scale = min(1.0, budget / sum(values)) if sum(values) else 1.0
        return [v * scale for v in values]

    max_mechanical_error = 0.0
    for row in rows:
        state = (row["world"], tuple(row["past_actions"]))
        key = (*state, row["action"])
        check(key not in edges and key in expected, "duplicate or unexpected transition")
        check(math.isfinite(row["cost"]), "nonfinite cost")
        edges[key] = row["cost"]
        check(information.setdefault(state, row["public_history_sha256"]) == row["public_history_sha256"], "inconsistent current information")
        child = (state[0], state[1] + (row["action"],))
        check(information.setdefault(child, row["next_public_history_sha256"]) == row["next_public_history_sha256"], "inconsistent successor information")
        active = [0.0, 0.0]
        sequence = state[1] + (row["action"],)
        for epoch in range(len(sequence)):
            mature = request(sequence[epoch - lead]) if epoch >= lead else [0.0, 0.0]
            active = [persistence * old + (1 - persistence) * new for old, new in zip(active, mature)]
        check(sum(row["requested_capacity"]) <= budget + 1e-10, "budget exceeded")
        for actual, expected_values in ((row["active_capacity"], active), (row["requested_capacity"], request(row["action"]))):
            check(len(actual) == 2 and all(math.isfinite(v) for v in actual), "invalid capacity vector")
            max_mechanical_error = max(max_mechanical_error, max(abs(a - b) for a, b in zip(actual, expected_values)))
    check(set(edges) == expected and len(rows) == 10920 == summary["unique_transition_queries"], "transition coverage")
    check(max_mechanical_error <= 1e-10, "lead/persistence mechanics mismatch")
    check(information[(0, ())] == information[(1, ())], "latent root revealed")
    weights = {i: w["probability"] for i, w in enumerate(config["worlds"])}

    def value(states):
        if len(states[0][1]) == horizon:
            return 0.0, {}
        groups = defaultdict(list)
        for state in states:
            groups[information[state]].append(state)
        if len(groups) > 1:
            total, policy = 0.0, {}
            for group in groups.values():
                subtotal, subpolicy = value(group)
                total += subtotal
                policy.update(subpolicy)
            return total, policy
        alternatives = []
        for action in actions:
            subtotal = sum(weights[w] * edges[(w, path, action)] for w, path in states)
            future, policy = value([(w, path + (action,)) for w, path in states])
            alternatives.append((subtotal + future, action, policy))
        best, action, policy = min(alternatives, key=lambda r: (r[0], r[1]))
        return best, {**policy, information[states[0]]: action}

    nonanticipative, policy = value([(0, ()), (1, ())])
    best = {0: math.inf, 1: math.inf}
    openloop = math.inf
    for sequence in itertools.product(actions, repeat=horizon):
        costs = {w: sum(edges[(w, sequence[:t], action)] for t, action in enumerate(sequence)) for w in weights}
        openloop = min(openloop, sum(weights[w] * costs[w] for w in weights))
        best = {w: min(best[w], costs[w]) for w in weights}
    clairvoyant = sum(weights[w] * best[w] for w in weights)
    recomputed = {"nonanticipative_cost": nonanticipative, "clairvoyant_cost": clairvoyant, "best_open_loop_cost": openloop}
    for key, cost in recomputed.items():
        check(math.isclose(cost, summary[key], rel_tol=1e-12, abs_tol=1e-8), "cost recomputation: " + key)
    replay, chosen = 0.0, {}
    for w in weights:
        path = ()
        for _ in range(horizon):
            action = policy[information[(w, path)]]
            replay += weights[w] * edges[(w, path, action)]
            path += (action,)
        chosen[str(w)] = list(path)
    check(math.isclose(replay, nonanticipative, rel_tol=1e-12, abs_tol=1e-8), "independent policy replay")
    check(clairvoyant <= nonanticipative <= openloop, "bound ordering")
    print(json.dumps({"audit_passes": True, "fixture_only": True, "research_performance_evidence": False,
                      "inventory_hashes_verified": len(inventory["files"]), "unique_transitions": len(edges),
                      "maximum_mechanical_error": max_mechanical_error, "recomputed_costs": recomputed,
                      "independently_replayed_cost": replay, "nonanticipative_actions_by_world": chosen,
                      "same_root_action": chosen["0"][0] == chosen["1"][0]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
