"""One bounded synthetic conventional-control screen; no neural training."""

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np

from evaluation.check_completion_feedback_mechanics import (
    IMMUTABLE as PRIOR_LOCKS, SOURCES as S1_SOURCES, equal, make_config,
)
from evaluation.check_service_effort_mechanics import locked_files, sha, write_json
from src.baselines.completion_response_filter import CompletionResponseFilter
from src.baselines.completion_rollout_control import (
    action_grid, named_seed, plan_rollout, reservation_action, zero_is_dominant,
)
from src.env.completion_feedback_queue import (
    advance, booked_backlog_action, closed, initial_observation, unresolved_ledger,
)


CONFIG = Path("experiments/configs/completion_control_screen_20260929.json")
SOURCES = [str(CONFIG), "evaluation/run_completion_control_screen.py",
           "src/env/completion_feedback_batch.py", "src/baselines/completion_rollout_control.py",
           "tests/test_completion_rollout_control.py", "tests/test_completion_control_screen.py",
           "specs/2026-09-29-completion-control-screen/protocol.md"]
IMMUTABLE = list(dict.fromkeys(S1_SOURCES + PRIOR_LOCKS + [
    "tests/test_completion_feedback_closure.py",
    "reports/2026-09-29-completion-feedback-mechanics/run/summary.json",
    "reports/2026-09-29-completion-feedback-mechanics/run/steps.jsonl",
    "reports/2026-09-29-completion-feedback-mechanics/repeat/summary.json",
    "reports/2026-09-29-completion-feedback-mechanics/repeat/steps.jsonl",
]))
REFERENCE = "498ea59"
CONTRASTS = [("reservation_rule", "booked_rule"), ("fixed64", "booked_rule"),
             ("id64", "fixed64"), ("fixed64", "fixed16"), ("id64", "id16")]


def world_tapes(screen, fixture, family, split, world):
    """Private evaluator data, independent of forecast randomness and method."""
    if family not in screen["families"] or split not in screen["splits"] or world not in screen["world_indices"]:
        raise ValueError("world outside fixed matrix")
    n = fixture["horizon"] + fixture["max_closure_steps"]
    uniforms = np.random.Generator(np.random.PCG64(named_seed(screen["namespace"], "actual", split, world))).random((n, 2))
    response_rng = np.random.Generator(np.random.PCG64(named_seed(screen["namespace"], "response", split, world)))
    rates = np.tile(fixture["nominal_response"], (n, 1)).astype(float)
    for t in range(fixture["change_epoch"], n):
        orientation = int(response_rng.integers(2)) if family == "fast_iid" else world % 2
        if family != "unchanged":
            rates[t] = fixture["slow_fast_response"][::(-1 if orientation else 1)]
    return uniforms, rates


def digest_array(array):
    return hashlib.sha256(array.astype("<f8").tobytes()).hexdigest()


def case_id(family, slots, split, world, method):
    return f"{split}__{family}__slots{slots}__world{world}__{method}"


def planning_seed(screen, split, world, epoch):
    return named_seed(screen["namespace"], "planning", split, world, epoch)


def choose(method, state, mean, cfg, fixture, grid, seed):
    if state.epoch == fixture["horizon"]:
        return (0.0, 0.0), None, {"kind": "closure_zero", "transition_queries": 0}
    if state.epoch > fixture["horizon"] or method == "booked_rule":
        return booked_backlog_action(state, cfg), None, {"kind": "booked_rule", "transition_queries": 0}
    if method == "reservation_rule":
        return reservation_action(state, mean, cfg, grid), mean, {"kind": method, "transition_queries": 0}
    if method not in ("fixed16", "fixed64", "id16", "id64"):
        raise ValueError("undeclared controller")
    response = tuple(fixture["nominal_response"]) if method.startswith("fixed") else mean
    action, detail = plan_rollout(state, response, cfg, grid, int(method[-2:]), 1, seed, fixture["horizon"])
    return action, response, {"kind": "rollout", "planning_seed": seed, **detail}


def deadline_check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("S2 wall cap reached; preserve partial evidence, no automatic retry")


def episode(screen, fixture, family, slots, split, world, method, *, emit=None, deadline=None):
    cfg, grid = make_config(fixture, slots), action_grid(screen["grid_denominator"])
    noise, rates = world_tapes(screen, fixture, family, split, world)
    state, filt, rows = initial_observation(cfg), CompletionResponseFilter(**fixture["filter"]), []
    probe = None
    for t in range(len(noise)):
        deadline_check(deadline)
        if t >= fixture["horizon"] and closed(state):
            break
        before, posterior = state, filt.posterior
        if method == "booked_rule" and t == screen["probe_epoch"] and split == screen["probe_split"] and world == screen["probe_world"]:
            probe = (before, filt.mean, posterior)
        start = time.perf_counter()
        action, estimate, detail = choose(method, state, filt.mean, cfg, fixture, grid, planning_seed(screen, split, world, t))
        latency = time.perf_counter() - start
        deadline_check(deadline)
        state, receipt, cost = advance(state, action, rates[t], noise[t], cfg)
        filt.observe(receipt)
        row = {"before": asdict(before), "action": action, "after": asdict(state),
               "receipt": asdict(receipt), "cost": cost, "filter_before": posterior,
               "filter_after": filt.posterior, "estimate_used": estimate,
               "decision": detail, "decision_seconds": latency}
        rows.append(row)
        if emit is not None:
            emit(row)
    if not closed(state):
        raise RuntimeError("actual closure cap reached; unfinished cost cannot be omitted")
    identity = {"case": case_id(family, slots, split, world, method), "family": family,
                "slots": slots, "split": split, "world": world, "method": method,
                "uniform_sha256": digest_array(noise), "response_sha256": digest_array(rates)}
    return {**identity, **summarize_episode(rows, state, cfg, fixture)}, rows, probe


def summarize_episode(rows, state, cfg, fixture):
    return {"intervals": len(rows), "closure_intervals": max(0, len(rows)-fixture["horizon"]),
            "ledger": unresolved_ledger(state, cfg),
            "cost": {k: sum(r["cost"][k] for r in rows) for k in ("holding", "labor", "switching", "total")},
            "closure_cost": sum(r["cost"]["total"] for r in rows[fixture["horizon"]:]),
            "idle_paid_hours": sum(h for r in rows for h, task in zip(r["receipt"]["applied_hours"], r["receipt"]["eligible_tasks"]) if task is None),
            "transition_queries": sum(r["decision"]["transition_queries"] for r in rows),
            "decision_seconds": sum(r["decision_seconds"] for r in rows)}


def audit_rollout(detail, action, state, cfg, grid, samples, depth, seed):
    assert detail["kind"] == "rollout" and detail["planning_seed"] == seed
    assert detail["dominant_zero"] == zero_is_dominant(state, cfg)
    if detail["dominant_zero"]:
        equal(action, (0, 0))
        assert detail["transition_queries"] == detail["candidates"] == 0
        return
    prefixes = list(itertools.product(grid, repeat=depth))
    assert detail["samples"] == samples and detail["depth"] == depth
    assert len(detail["candidate_means"]) == len(detail["candidate_mc_se"]) == detail["candidates"] == len(prefixes)
    assert all(np.isfinite(detail["candidate_means"])) and all(np.isfinite(detail["candidate_mc_se"]))
    assert min(detail["candidate_mc_se"]) >= 0 and detail["transition_queries"] > 0
    best = int(np.argmin(detail["candidate_means"]))
    assert detail["selected_index"] == best
    equal(detail["selected_prefix"], prefixes[best])
    equal(action, prefixes[best][0])


def audit_episode(screen, fixture, case, rows, *, replan=False):
    """Replay actual physics and receipts; optionally replay all planner decisions."""
    cfg, grid = make_config(fixture, case["slots"]), action_grid(screen["grid_denominator"])
    noise, rates = world_tapes(screen, fixture, case["family"], case["split"], case["world"])
    assert case["uniform_sha256"] == digest_array(noise) and case["response_sha256"] == digest_array(rates)
    state, filt, purchased, applied = initial_observation(cfg), CompletionResponseFilter(**fixture["filter"]), 0.0, 0.0
    for t, row in enumerate(rows):
        assert t < len(noise)
        assert t < fixture["horizon"] or not closed(state)
        equal(row["before"], asdict(state))
        equal(row["filter_before"], filt.posterior)
        assert tuple(row["action"]) in grid
        assert np.isfinite(row["decision_seconds"]) and row["decision_seconds"] >= 0
        seed, method = planning_seed(screen, case["split"], case["world"], t), case["method"]
        if t < fixture["horizon"] and method in ("fixed16", "fixed64", "id16", "id64") and not replan:
            estimate = tuple(fixture["nominal_response"]) if method.startswith("fixed") else filt.mean
            equal(row["estimate_used"], estimate)
            audit_rollout(row["decision"], row["action"], state, cfg, grid, int(method[-2:]), 1, seed)
        else:
            action, estimate, detail = choose(method, state, filt.mean, cfg, fixture, grid, seed)
            equal([row["action"], row["estimate_used"], row["decision"]], [action, estimate, detail])
        state, receipt, cost = advance(state, row["action"], rates[t], noise[t], cfg)
        filt.observe(receipt)
        equal([row["after"], row["receipt"], row["cost"], row["filter_after"]],
              [asdict(state), asdict(receipt), cost, filt.posterior])
        purchased += sum(receipt.requested_hours)
        applied += sum(receipt.applied_hours)
        equal(purchased, applied + sum(state.pending_hours))
    assert fixture["horizon"] <= len(rows) <= len(noise) and closed(state)
    assert sum(sum(r["receipt"]["completed"]) for r in rows) == len(cfg.jobs)
    expected = summarize_episode(rows, state, cfg, fixture)
    equal({k: case[k] for k in expected}, expected)
    return {"case": case["case"], "audit": "passed", "intervals": len(rows), "full_planner_replay": replan}


def paired_contrasts(screen, cases):
    indexed = {c["case"]: c for c in cases}
    if len(indexed) != len(cases):
        raise ValueError("duplicate episode identity")
    results = []
    for family, slots, split in itertools.product(screen["families"], screen["downstream_slots"], screen["splits"]):
        for left, right in CONTRASTS:
            pairs = []
            for world in screen["world_indices"]:
                a, b = [indexed[case_id(family, slots, split, world, method)] for method in (left, right)]
                pairs.append({"world": world, "left_cost": a["cost"]["total"], "right_cost": b["cost"]["total"],
                              "difference": a["cost"]["total"]-b["cost"]["total"]})
            delta = [p["difference"] for p in pairs]
            results.append({"family": family, "slots": slots, "split": split, "left": left, "right": right,
                            "pairs": pairs, "mean": float(np.mean(delta)), "min": min(delta), "max": max(delta),
                            "lower": sum(x < 0 for x in delta), "equal": sum(x == 0 for x in delta), "higher": sum(x > 0 for x in delta)})
    return results


def adequacy_probe(screen, fixture, family, slots, captured, deadline=None):
    state, mean, posterior = captured
    cfg, grid = make_config(fixture, slots), action_grid(screen["grid_denominator"])
    seed = planning_seed(screen, screen["probe_split"], screen["probe_world"], state.epoch)
    options = []
    for samples, depth in ((16, 1), (64, 1), (64, 2)):
        deadline_check(deadline)
        start = time.perf_counter()
        action, detail = plan_rollout(state, mean, cfg, grid, samples, depth, seed, fixture["horizon"])
        options.append({"samples": samples, "depth": depth, "action": action, "detail": detail,
                        "decision_seconds": time.perf_counter()-start})
        deadline_check(deadline)
    return {"family": family, "slots": slots, "state": asdict(state), "filter_posterior": posterior,
            "response_estimate": mean, "planning_seed": seed, "options": options,
            "dominant_zero": options[0]["detail"]["dominant_zero"],
            "budget_agrees": options[0]["action"] == options[1]["action"],
            "depth_agrees": options[1]["action"] == options[2]["action"]}


def summarize_matrix(screen, cases, audits, probes):
    return {"role": screen["role"], "episodes": len(cases), "audited_intervals": sum(a["intervals"] for a in audits),
            "all_settled": all(c["ledger"]["settled"] for c in cases),
            "probes": len(probes), "nontrivial_probes": sum(not p["dominant_zero"] for p in probes),
            "budget_disagreements": sum(not p["budget_agrees"] for p in probes),
            "depth_disagreements": sum(not p["depth_agrees"] for p in probes),
            "transition_queries": sum(c["transition_queries"] for c in cases),
            "probe_transition_queries": sum(o["detail"]["transition_queries"] for p in probes for o in p["options"]),
            "online_rl_evidence": False, "clinical_calibration": False,
            "strong_frozen_policy_headroom_established": False}


def audit_output(output):
    screen = json.loads((output/"screen.json").read_text())
    fixture = json.loads((output/"mechanics.json").read_text())
    cases = json.loads((output/"cases.json").read_text())
    expected = {case_id(f,s,p,w,m) for f,s,p,w,m in itertools.product(
        screen["families"], screen["downstream_slots"], screen["splits"], screen["world_indices"], screen["methods"])}
    assert len(cases) == len(expected) == screen["expected_episodes"]
    assert {c["case"] for c in cases} == expected
    groups = {name: [] for name in expected}
    with (output/"steps.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            groups[row.pop("case")].append(row)
    audits = []
    for c in cases:
        assert c["case"] == case_id(c["family"], c["slots"], c["split"], c["world"], c["method"])
        replay = c["case"] == case_id("persistent", 1, "discovery", 0, "fixed64")
        audits.append(audit_episode(screen, fixture, c, groups[c["case"]], replan=replay))
    equal(json.loads((output/"contrasts.json").read_text()), paired_contrasts(screen, cases))
    probes = json.loads((output/"probes.json").read_text())
    assert len(probes) == screen["expected_probes"]
    assert {(p["family"], p["slots"]) for p in probes} == set(itertools.product(screen["families"], screen["downstream_slots"]))
    for p in probes:
        rows = groups[case_id(p["family"], p["slots"], screen["probe_split"], screen["probe_world"], "booked_rule")]
        row = rows[screen["probe_epoch"]]
        equal(p["state"], row["before"])
        equal(p["filter_posterior"], row["filter_before"])
        mean = [sum(r*q for r,q in zip(fixture["filter"]["rates"], posterior)) for posterior in p["filter_posterior"]]
        equal(p["response_estimate"], mean)
        # State is reconstructed by replay, never by accepting a private state file.
        cfg = make_config(fixture, p["slots"])
        state = initial_observation(cfg)
        noise, rates = world_tapes(screen, fixture, p["family"], screen["probe_split"], screen["probe_world"])
        for t in range(screen["probe_epoch"]):
            state, _, _ = advance(state, rows[t]["action"], rates[t], noise[t], cfg)
        for option, (samples, depth) in zip(p["options"], ((16,1), (64,1), (64,2))):
            assert option["samples"] == samples and option["depth"] == depth
            audit_rollout({"kind": "rollout", "planning_seed": p["planning_seed"], **option["detail"]},
                          option["action"], state, cfg, action_grid(screen["grid_denominator"]), samples, depth,
                          planning_seed(screen, screen["probe_split"], screen["probe_world"], screen["probe_epoch"]))
        assert len(p["options"]) == 3
        assert p["dominant_zero"] == zero_is_dominant(state, cfg)
        assert p["budget_agrees"] == (p["options"][0]["action"] == p["options"][1]["action"])
        assert p["depth_agrees"] == (p["options"][1]["action"] == p["options"][2]["action"])
    claim = json.loads((output/"claim.json").read_text())
    equal(claim["source_locks"], locked_files(SOURCES, claim["execution_commit"]))
    equal(claim["immutable_locks"], locked_files(IMMUTABLE, REFERENCE))
    equal(screen, json.loads(CONFIG.read_text()))
    equal(fixture, json.loads(Path(screen["mechanics_config"]).read_text()))
    if (output/"summary.json").exists():
        summary = json.loads((output/"summary.json").read_text())
        expected_summary = summarize_matrix(screen, cases, audits, probes)
        equal({k: summary[k] for k in expected_summary}, expected_summary)
    if (output/"inventory.json").exists():
        inventory = json.loads((output/"inventory.json").read_text())
        assert len({record["path"] for record in inventory}) == len(inventory)
        assert {record["path"] for record in inventory} == {p.name for p in output.iterdir() if p.is_file()} - {"status.json", "inventory.json"}
        for record in inventory:
            path = output/record["path"]
            assert path.stat().st_size == record["bytes"] and sha(path) == record["sha256"]
    return audits


def run(output):
    if output.exists():
        raise FileExistsError("refusing to overwrite existing evidence")
    screen = json.loads(CONFIG.read_text())
    fixture = json.loads(Path(screen["mechanics_config"]).read_text())
    if screen["training_authorized"] or screen["patient_integration_authorized"]:
        raise ValueError("synthetic non-training screen only")
    claim = {"execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
             "source_locks": locked_files(SOURCES, "HEAD"), "immutable_locks": locked_files(IMMUTABLE, REFERENCE),
             "python": platform.python_version(), "numpy": np.__version__, "training": False,
             "seed_policy": "actual/response paired across families and capacities; planning independent; splits disjoint"}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"claim.json", claim)
    write_json(output/"screen.json", screen)
    write_json(output/"mechanics.json", fixture)
    started = time.monotonic()
    deadline = started + screen["wall_cap_seconds"]
    cases, probes = [], []
    try:
        with (output/"steps.jsonl").open("x") as stream:
            for family, slots, split, world, method in itertools.product(
                    screen["families"], screen["downstream_slots"], screen["splits"], screen["world_indices"], screen["methods"]):
                identity = case_id(family, slots, split, world, method)
                write_json(output/"status.json", {"status": "running", "case": identity, "completed": len(cases)})
                def emit(row):
                    stream.write(json.dumps({"case": identity, **row}, sort_keys=True, allow_nan=False)+"\n")
                    stream.flush()
                case, rows, captured = episode(screen, fixture, family, slots, split, world, method, emit=emit, deadline=deadline)
                audit_episode(screen, fixture, case, rows)
                cases.append(case)
                write_json(output/"cases.json", cases)
                if captured is not None:
                    probes.append(adequacy_probe(screen, fixture, family, slots, captured, deadline))
                    write_json(output/"probes.json", probes)
                print(f"{len(cases)}/{screen['expected_episodes']} {identity} cost={case['cost']['total']:.4f} elapsed={time.monotonic()-started:.1f}s", flush=True)
        write_json(output/"contrasts.json", paired_contrasts(screen, cases))
        deadline_check(deadline)
        audits = audit_output(output)
        write_json(output/"audit.json", audits)
        deadline_check(deadline)
        summary = {**summarize_matrix(screen, cases, audits, probes), "elapsed_seconds": time.monotonic()-started}
        write_json(output/"summary.json", summary)
        write_json(output/"inventory.json", [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
                                             for p in sorted(output.iterdir()) if p.is_file() and p.name != "status.json"])
        write_json(output/"status.json", {"status": "completed", "exit_code": 0, "completed": len(cases), "elapsed_seconds": time.monotonic()-started})
    except Exception as exc:
        (output/"error.txt").write_text(traceback.format_exc())
        write_json(output/"status.json", {"status": "failed", "exit_code": 1, "completed": len(cases),
                                          "error": str(exc), "elapsed_seconds": time.monotonic()-started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps({"audited_episodes": len(audit_output(args.output))}))
    else:
        run(args.output)
