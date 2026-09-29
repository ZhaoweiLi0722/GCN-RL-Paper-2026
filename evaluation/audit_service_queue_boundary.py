"""Independent row-accounting audit; never calls the queue transition or planner."""

import argparse
import gzip
import itertools
import json
import math
from pathlib import Path

from evaluation.check_service_effort_mechanics import sha


CONFIG = Path("experiments/configs/service_queue_boundary_20260929.json")
METHODS = ("fixed_balanced", "backlog_rule", "adaptive_backlog_rule", "fixed_model_mpc", "identification_mpc")


def equal(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), "mapping keys differ"
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), "sequence lengths differ"
        for a, e in zip(actual, expected):
            equal(a, e)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-9, rel_tol=1e-10), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def jobs_for(fixture, cell):
    pattern, _, _ = cell.split("__")
    return [{"name": f"site{site}_job{k}", "site": site, "release": release,
             "work": fixture["job_work"], "duration": fixture["downstream_steps"]}
            for site in (0, 1) for k, release in enumerate(fixture["release_patterns"][pattern])]


def initial(fixture, cell):
    jobs = jobs_for(fixture, cell)
    return {"epoch": 0, "stages": ["support" if j["release"] == 0 else "scheduled" for j in jobs],
            "remaining_work": [j["work"] for j in jobs], "ready_epochs": [-1] * len(jobs),
            "downstream_remaining": [0] * len(jobs), "pending_hours": [[0, 0]], "previous_commitment": [0, 0]}


def verify_step(row, fixture, cell, world):
    before, after, receipt, cost = (row[k] for k in ("before", "after", "receipt", "cost"))
    jobs = jobs_for(fixture, cell)
    t = before["epoch"]
    effort = fixture["effort"]
    action, applied = receipt["committed_hours"], receipt["applied_hours"]
    assert len(action) == 2 and all(math.isfinite(x) and 0 <= x <= cap for x, cap in zip(action, effort["site_hour_caps"]))
    assert sum(action) <= effort["shared_hour_budget"] + 1e-12
    equal(applied, before["pending_hours"][0])
    equal(receipt["epoch"], t)
    expected = json.loads(json.dumps(before))
    stages, work = expected["stages"], expected["remaining_work"]
    ready, downstream = expected["ready_epochs"], expected["downstream_remaining"]
    cap = fixture["downstream_regimes"][cell.split("__")[1]]
    active = [i for i, stage in enumerate(stages) if stage == "active"]
    assert len(active) <= cap
    waiting = sorted((i for i, stage in enumerate(stages) if stage == "waiting"),
                     key=lambda i: (ready[i], jobs[i]["release"], jobs[i]["name"]))
    started = waiting[:cap - len(active)]
    finished = []
    for i in active + started:
        duration = jobs[i]["duration"] if i in started else downstream[i]
        assert duration >= 1
        downstream[i] = duration - 1
        stages[i] = "done" if duration == 1 else "active"
        if duration == 1:
            finished.append(i)
    available, delivered, complete, kinds = [], [], [], []
    response = world["after"] if t >= world["change_epoch"] else [1, 1]
    for site in (0, 1):
        indices = sorted((i for i, job in enumerate(jobs) if job["site"] == site and before["stages"][i] == "support"),
                         key=lambda i: (jobs[i]["release"], jobs[i]["name"]))
        total = sum(before["remaining_work"][i] for i in indices)
        amount = min(total, applied[site] * response[site])
        consumed, count = 0.0, 0
        for i in indices:
            progress = min(before["remaining_work"][i], amount - consumed)
            consumed += progress
            work[i] = before["remaining_work"][i] - progress
            if work[i] <= 1e-12:
                work[i], stages[i], ready[i] = 0.0, "waiting", t + 1
                count += 1
        available.append(total)
        delivered.append(amount)
        complete.append(count)
        kinds.append("no_effort" if applied[site] == 0 else ("backlog_limited" if amount >= total - 1e-12 else "uncensored"))
    equal(receipt["available_work"], available)
    equal(receipt["delivered_work"], delivered)
    equal(receipt["completed_jobs"], complete)
    equal(receipt["observation_kind"], kinds)
    equal(sum(before["remaining_work"]) - sum(work), sum(delivered))
    holding = fixture["holding_cost"] * sum(j["release"] <= t and stage != "done" for j, stage in zip(jobs, stages))
    released = []
    for i, job in enumerate(jobs):
        if stages[i] == "scheduled" and job["release"] == t + 1:
            stages[i] = "support"
            released.append(i)
    expected.update(epoch=t + 1, pending_hours=[action], previous_commitment=action)
    equal(after, expected)
    labor = sum(effort["hourly_cost"] * h + effort["quadratic_cost"] * h * h for h in action)
    switch = effort["switching_cost"] * sum(abs(h - old) for h, old in zip(action, before["previous_commitment"]))
    equal(receipt["labor_cost"], labor)
    equal(receipt["change_cost"], switch)
    equal(cost, {"holding": holding, "labor": labor, "switching": switch, "total": holding + labor + switch,
                 "downstream_started": [jobs[i]["name"] for i in started],
                 "downstream_finished": [jobs[i]["name"] for i in finished], "released": [jobs[i]["name"] for i in released]})
    for stage, remaining, duration in zip(after["stages"], after["remaining_work"], after["downstream_remaining"]):
        assert stage in ("scheduled", "support", "waiting", "active", "done")
        if stage in ("waiting", "active", "done"):
            equal(remaining, 0)
        assert (duration > 0) == (stage == "active")


def verify_closure(closure, before, fixture, cell, world):
    assert 0 <= len(closure["steps"]) <= fixture["max_settlement_steps"]
    current, total = before, 0.0
    for offset, row in enumerate(closure["steps"]):
        equal(row["before"], current)
        has_work = any(stage == "support" for stage in current["stages"])
        action = [0, 0] if offset < fixture["effort"]["commitment_lead_steps"] or not has_work else [0.5, 0.5]
        equal(row["receipt"]["committed_hours"], action)
        verify_step(row, fixture, cell, world)
        total += row["cost"]["total"]
        current = row["after"]
    equal(closure["final"], current)
    equal(closure["cost"], total)
    assert all(stage == "done" for stage in current["stages"])
    assert sum(current["remaining_work"]) == 0 and not any(sum(row) for row in current["pending_hours"])
    assert not any(current["previous_commitment"])
    return total


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def audit(output):
    fixture = json.loads(CONFIG.read_text())
    inventory = json.loads((output / "inventory.json").read_text())["files"]
    for item in inventory:
        assert sha(item["path"]) == item["sha256"], item["path"]
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "completed" and status["exit_code"] == 0
    summary = json.loads((output / "summary.json").read_text())
    episodes, decisions, edges, closures = [read_rows(output / f"{name}.jsonl.gz") for name in
                                            ("episodes", "decisions", "tree_transitions", "tree_closures")]
    worlds = {"__".join((pattern, downstream, family["name"])): family["worlds"]
              for pattern in fixture["release_patterns"] for downstream in fixture["downstream_regimes"]
              for family in fixture["families"]}
    world_by_name = {(cell, w["name"]): w for cell, values in worlds.items() for w in values}
    expected = {(cell, w["name"], method) for cell, values in worlds.items() for w in values for method in METHODS}
    keyed_episodes = {(e["cell"], e["world"], e["method"]): e for e in episodes}
    keyed_decisions = {(r["cell"], r["world"], r["method"], r["before"]["epoch"]): r for r in decisions}
    assert len(episodes) == len(keyed_episodes) == 60 and set(keyed_episodes) == expected
    assert len(decisions) == len(keyed_decisions) == 300
    assert set(keyed_decisions) == {key + (t,) for key in expected for t in range(5)}
    for key, episode in keyed_episodes.items():
        cell, name, method = key
        world = world_by_name[cell, name]
        current, total, estimate = initial(fixture, cell), 0.0, list(fixture["prior"])
        actions = []
        for t in range(5):
            row = keyed_decisions[key + (t,)]
            equal(row["before"], current)
            equal(row["receipt"]["committed_hours"], fixture["actions"][row["action"]])
            equal(row["response_before"], estimate if method in ("adaptive_backlog_rule", "identification_mpc") else fixture["prior"])
            verify_step(row, fixture, cell, world)
            receipt = row["receipt"]
            for site in (0, 1):
                if receipt["observation_kind"][site] == "uncensored":
                    a = fixture["estimator_alpha"]
                    estimate[site] = (1 - a) * estimate[site] + a * receipt["delivered_work"][site] / receipt["applied_hours"][site]
            equal(row["estimator_after"], estimate)
            actions.append(row["action"])
            total += row["cost"]["total"]
            current = row["after"]
        equal(episode["actions"], actions)
        equal(episode["decision_cost"], total)
        equal(episode["cost"], total + verify_closure(episode["settlement"], current, fixture, cell, world))
        assert type(episode["prediction_transition_queries"]) is int and episode["prediction_transition_queries"] >= 0
    closure_by_id = {r["id"]: r for r in closures}
    assert len(closure_by_id) == len(closures) == summary["distinct_tree_closures"]
    for record in closures:
        verify_closure(record["settlement"], record["before"], fixture, record["cell"], world_by_name[record["cell"], record["world"]])
    keyed_edges = {(r["cell"], r["world"], tuple(r["past_actions"]), r["action"]): r for r in edges}
    expected_edges = {(cell, w["name"], prefix, action) for cell, values in worlds.items() for w in values
                      for t in range(5) for prefix in itertools.product(fixture["actions"], repeat=t) for action in fixture["actions"]}
    assert len(edges) == len(keyed_edges) == 16368 and set(keyed_edges) == expected_edges
    used_closures = set()
    for (cell, name, prefix, action), edge in keyed_edges.items():
        before = initial(fixture, cell) if not prefix else keyed_edges[cell, name, prefix[:-1], prefix[-1]]["step"]["after"]
        equal(edge["step"]["before"], before)
        equal(edge["step"]["receipt"]["committed_hours"], fixture["actions"][action])
        verify_step(edge["step"], fixture, cell, world_by_name[cell, name])
        terminal = 0
        if len(prefix) == 4:
            closure = closure_by_id[edge["settlement_id"]]
            assert closure["cell"] == cell and closure["world"] == name
            equal(closure["before"], edge["step"]["after"])
            terminal = closure["settlement"]["cost"]
            used_closures.add(edge["settlement_id"])
        else:
            assert edge["settlement_id"] is None
        equal(edge["cost"], edge["step"]["cost"]["total"] + terminal)
    assert used_closures == set(closure_by_id)
    for cell, values in worlds.items():
        comparison = summary["comparisons"][cell]
        means = {method: sum(w["probability"] * keyed_episodes[cell, w["name"], method]["cost"] for w in values) for method in METHODS}
        equal(comparison["expected_synthetic_cost"], means)
        open_loop, perfect = math.inf, {w["name"]: math.inf for w in values}
        for sequence in itertools.product(fixture["actions"], repeat=5):
            expected_cost = 0.0
            for world in values:
                cost = sum(keyed_edges[cell, world["name"], sequence[:t], action]["cost"] for t, action in enumerate(sequence))
                expected_cost += world["probability"] * cost
                perfect[world["name"]] = min(perfect[world["name"]], cost)
            open_loop = min(open_loop, expected_cost)
        diag = comparison["diagnostic"]
        equal(diag["best_open_loop_cost"], open_loop)
        equal(diag["clairvoyant_cost"], sum(w["probability"] * perfect[w["name"]] for w in values))
        assert diag["clairvoyant_cost"] <= diag["nonanticipative_cost"] + 1e-9 <= open_loop + 2e-9
        equal(diag["policy_replay_cost"], diag["nonanticipative_cost"])
        equal(comparison["feedback_value_over_open_loop"], open_loop - diag["nonanticipative_cost"])
        equal(comparison["identification_gain_over_fixed_mpc"], means["fixed_model_mpc"] - means["identification_mpc"])
        equal(comparison["identification_gap_to_diagnostic"], means["identification_mpc"] - diag["nonanticipative_cost"])
        assert min(means.values()) >= diag["nonanticipative_cost"] - 1e-9
    equal([summary["baseline_episodes"], summary["decision_rows"], summary["exact_tree_edges"]], [60, 300, 16368])
    assert not summary["online_weight_updates"] and not summary["patient_model_run"] and not summary["operational_calibration_validated"]
    assert summary["independent_replications"] == 0
    return {"audit": "passed", "baseline_episodes": 60, "decision_rows": 300, "tree_edges": 16368,
            "unique_closures": len(closures), "verified_hashes": len(inventory),
            "limitation": "nonanticipative optimizer is regression-tested, not independently reimplemented; planner query counts are implementation-reported"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2))


if __name__ == "__main__":
    main()
