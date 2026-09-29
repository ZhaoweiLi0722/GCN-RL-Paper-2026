"""One locked S3 conditional-model ranking matrix, not actual policy episodes."""

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback

import numpy as np

from evaluation.check_completion_feedback_mechanics import equal, make_config
from evaluation.check_service_effort_mechanics import locked_files, sha, write_json
from evaluation.run_completion_control_screen import (
    IMMUTABLE as S2_IMMUTABLE, SOURCES as S2_SOURCES,
)
from src.baselines.completion_rollout_control import action_grid, named_seed
from src.env.completion_feedback_batch import forecast
from src.env.completion_feedback_queue import (
    CompletionObservation, advance, booked_backlog_action, closed, validate_observation,
)


CONFIG = Path("experiments/configs/completion_action_ranking_20260929.json")
PROTOCOL = "specs/2026-09-29-completion-action-ranking/protocol.md"
PROTOCOL_COMMIT = "b91dd3b"
SOURCES = [str(CONFIG), PROTOCOL, "evaluation/check_completion_action_ranking.py",
           "tests/test_completion_action_ranking.py"]
# Fixed before execution; all 15 actions are replayed at these path indices.
SCALAR_AUDIT_INDICES = (0, 127, 128, 2047)


def read_json(path):
    return json.loads(Path(path).read_text())


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("S3 900-second cap reached; preserve evidence, no retry")


def observation(context):
    return CompletionObservation(**{k: v if k == "epoch" else tuple(v)
                                    for k, v in context["public"]["observation"].items()})


def context_digest(public):
    return hashlib.sha256(json.dumps(public, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("ascii")).hexdigest()


def canonical_contexts(probes, fixture, grid, expected_probes, expected_contexts):
    if len(probes) != expected_probes:
        raise ValueError("missing or extra S2 probe")
    grouped, identities = {}, set()
    for index, probe in enumerate(probes):
        identity = (probe["family"], probe["slots"])
        if identity in identities:
            raise ValueError("duplicate S2 probe identity")
        identities.add(identity)
        public = {"observation": probe["state"], "response": probe["response_estimate"],
                  "downstream_slots": probe["slots"], "mechanics": fixture}
        digest = context_digest(public)
        references = {}
        for samples in (16, 64):
            matches = [p for p in probe["options"] if p["depth"] == 1 and p["samples"] == samples]
            if len(matches) != 1:
                raise ValueError("missing/duplicate archived comparator")
            references[f"s2_depth1_{samples}"] = grid.index(tuple(matches[0]["action"]))
        context = {"id": digest, "public": public, "references": references, "aliases": []}
        cfg, state = make_config(fixture, probe["slots"]), observation(context)
        validate_observation(state, cfg)
        if state.epoch >= fixture["horizon"] or closed(state):
            raise ValueError("probe outside first-action decision window")
        if len(public["response"]) != 2 or any(not math.isfinite(r) or r <= 0 for r in public["response"]):
            raise ValueError("invalid archived response")
        references["booked_rule"] = grid.index(booked_backlog_action(state, cfg))
        if digest not in grouped:
            grouped[digest] = context
        elif grouped[digest]["references"] != references or grouped[digest]["public"] != public:
            raise ValueError("canonical duplicate has incompatible public content/comparators")
        grouped[digest]["aliases"].append({"source_probe_index": index, "family": probe["family"],
                                            "slots": probe["slots"]})
    if len(grouped) != expected_contexts:
        raise ValueError("unexpected unique context count")
    return [grouped[key] for key in sorted(grouped)]


def noise_chunks(seed, samples, intervals, chunk_size):
    rng = np.random.Generator(np.random.PCG64(seed))
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        yield start, stop, rng.random((stop-start, intervals, 2))


def noise_hash(noise):
    return hashlib.sha256(noise.astype("<f8").tobytes()).hexdigest()


def paired_stats(left, right, *, independent=False):
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired sample sizes must match and exceed one")
    if independent:
        delta = [float(a)-float(b) for a, b in zip(left, right)]
        mean, se = statistics.fmean(delta), statistics.stdev(delta)/math.sqrt(len(delta))
    else:
        delta = np.asarray(left)-np.asarray(right)
        mean, se = float(np.mean(delta)), float(np.std(delta, ddof=1)/math.sqrt(len(delta)))
    if not math.isfinite(mean) or not math.isfinite(se):
        raise ValueError("nonfinite paired metric")
    return {"mean_left_minus_right": mean, "paired_mc_se": se,
            "sign": int(mean > 0)-int(mean < 0)}


def block_statistics(costs, grid, budgets, *, independent=False):
    if costs.shape != (len(grid), max(budgets)) or not np.isfinite(costs).all():
        raise ValueError("incomplete/nonfinite raw cost matrix")
    results = []
    for n in budgets:
        if independent:
            means = [statistics.fmean(map(float, row[:n])) for row in costs]
        else:
            means = costs[:, :n].mean(axis=1).tolist()
        best = min(range(len(grid)), key=lambda i: means[i])
        results.append({"samples": n, "action_means": means, "argmin_index": best,
                        "argmin_action": list(grid[best]),
                        "exact_tied_indices": [i for i, value in enumerate(means) if value == means[best]],
                        "pairs": [{"left": a, "right": b,
                                   **paired_stats(costs[a, :n], costs[b, :n], independent=independent)}
                                  for a, b in itertools.combinations(range(len(grid)), 2)]})
    return results


def frozen_choice(costs, grid):
    if costs.shape[0] != len(grid) or not np.isfinite(costs).all():
        raise ValueError("invalid selection costs")
    index = int(np.argmin(costs.mean(axis=1)))
    return {"index": index, "action": list(grid[index]), "selection_samples": costs.shape[1]}


def validation_comparison(costs, chosen, context, budgets, *, independent=False):
    if type(chosen) is not int or not 0 <= chosen < costs.shape[0]:
        raise ValueError("invalid frozen selection index")
    return [{"samples": n, "comparators": {
        name: {"reference_index": index,
               **paired_stats(costs[chosen, :n], costs[index, :n], independent=independent)}
        for name, index in context["references"].items()}}
        for n in budgets]


def summarize(contexts, blocks, freezes, paths, queries):
    rows = []
    for context in contexts:
        identity = context["id"]
        records = [b for b in blocks if b["context"] == identity]
        argmins = [b["budgets"][-1]["argmin_index"] for b in records]
        directions = {name: [b["validation"][-1]["comparators"][name]["sign"]
                             for b in records if b["block"] != "selection"]
                      for name in context["references"]}
        disagree = len(set(argmins)) > 1 or any(len(set(signs)) > 1 for signs in directions.values())
        rows.append({"context": identity, "aliases": context["aliases"], "frozen_choice": freezes[identity],
                     "block_argmin_indices_2048": argmins, "validation_signs_2048": directions,
                     "budget_agreement_by_block": {b["block"]: len({r["argmin_index"] for r in b["budgets"]}) == 1
                                                    for b in records},
                     "ranking_unresolved": disagree})
    return {"classification": "conditional_model_ranking_only", "contexts": rows,
            "unique_contexts": len(contexts), "source_probe_aliases": sum(len(c["aliases"]) for c in contexts),
            "blocks": len(blocks), "forecast_paths": paths, "transition_queries": queries,
            "unresolved_contexts": sum(r["ranking_unresolved"] for r in rows),
            "actual_environment_episodes": 0, "neural_updates": 0, "online_rl_evidence": False,
            "strong_controller_headroom_established": False, "model_validated": False,
            "clinical_calibration": False}


def scalar_cost(state, first_action, response, tape, cfg, boundary):
    total, count = 0.0, 0
    for offset, draw in enumerate(tape):
        if closed(state):
            break
        if state.epoch == boundary:
            action = (0, 0)
        elif offset == 0 and state.epoch < boundary:
            action = first_action
        else:
            action = booked_backlog_action(state, cfg)
        state, _, charges = advance(state, action, response, draw, cfg)
        total += charges["total"]
        count += 1
    if not closed(state):
        raise RuntimeError("scalar forecast has unsettled tail")
    return total, count


def input_locks(screen):
    root = Path(screen["s2_root"])
    for name in ("probes", "summary"):
        if sha(root/f"{name}.json") != screen[f"s2_{name}_sha256"]:
            raise ValueError(f"S2 {name} hash mismatch")
    paths = list(dict.fromkeys(S2_SOURCES + S2_IMMUTABLE +
                              [str(p) for p in sorted(root.rglob("*")) if p.is_file()]))
    return locked_files(paths, screen["source_commit"])


def check_records(records, contexts, screen):
    expected = {(c["id"], b, i, min(i+screen["sample_chunk_size"], max(screen["sample_budgets"])))
                for c in contexts for b in screen["blocks"]
                for i in range(0, max(screen["sample_budgets"]), screen["sample_chunk_size"])}
    actual = [(r["context"], r["block"], r["start"], r["stop"]) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("missing/duplicate/unexpected raw chunk identity")
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("duplicate raw array path")


def inventory(output):
    return [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(output.rglob("*")) if p.is_file() and p.name not in ("inventory.json", "status.json")]


def audit_output(output, deadline=None, *, require_complete=True):
    screen, claim = read_json(output/"config.json"), read_json(output/"claim.json")
    equal(screen, read_json(CONFIG))
    locked_files([str(CONFIG), PROTOCOL], PROTOCOL_COMMIT)
    equal(claim["source_locks"], locked_files(SOURCES, claim["execution_commit"]))
    equal(claim["input_locks"], input_locks(screen))
    fixture = read_json(Path(screen["s2_root"])/"mechanics.json")
    grid = action_grid(screen["grid_denominator"])
    contexts = canonical_contexts(read_json(Path(screen["s2_root"])/"probes.json"), fixture, grid,
                                  screen["source_probe_count"], screen["expected_unique_contexts"])
    equal(read_json(output/"contexts.json"), contexts)
    equal(read_json(output/"actions.json"), grid)
    records = [json.loads(line) for line in (output/"chunks.jsonl").read_text().splitlines()]
    check_records(records, contexts, screen)
    if {str(p.relative_to(output)) for p in (output/"raw").rglob("*") if p.is_file()} != {r["path"] for r in records}:
        raise ValueError("missing or unindexed raw cost file")
    blocks, freezes, paths, queries, scalar_paths, scalar_queries = [], {}, 0, 0, 0, 0
    expected_events = []
    for block in screen["blocks"]:
        for context in contexts:
            identity, public = context["id"], context["public"]
            state, cfg = observation(context), make_config(fixture, public["downstream_slots"])
            seed = named_seed(screen["namespace"], identity, block)
            intervals = fixture["horizon"]+cfg.max_closure_steps-state.epoch
            chunks = [r for r in records if r["context"] == identity and r["block"] == block]
            chunks.sort(key=lambda r: r["start"])
            arrays = []
            for record, (start, stop, noise) in zip(chunks, noise_chunks(seed, max(screen["sample_budgets"]), intervals,
                                                                       screen["sample_chunk_size"])):
                check_deadline(deadline)
                relative = f"raw/{identity}/{block}/{start:04d}.npy"
                if record["path"] != relative or record["seed"] != seed or record["uniform_sha256"] != noise_hash(noise):
                    raise ValueError("raw identity/seed/tape mismatch")
                expected_events.append({"kind": "chunk", "context": identity, "block": block, "start": start})
                if sha(output/relative) != record["sha256"]:
                    raise ValueError("raw array hash mismatch")
                costs = np.load(output/relative, allow_pickle=False)
                if costs.shape != (len(grid), stop-start) or costs.dtype != np.dtype("float64") or not np.isfinite(costs).all():
                    raise ValueError("raw array shape/type/finite check failed")
                equal(record["shape"], costs.shape)
                if not 0 < record["transition_queries"] <= costs.size*intervals:
                    raise ValueError("invalid query accounting")
                expected_freeze = None if block == "selection" else sha(output/f"selection/{identity}.json")
                if record["selection_freeze_sha256"] != expected_freeze:
                    raise ValueError("validation did not retain frozen selection")
                for index in SCALAR_AUDIT_INDICES:
                    if start <= index < stop:
                        for a, action in enumerate(grid):
                            value, count = scalar_cost(state, action, public["response"], noise[index-start], cfg, fixture["horizon"])
                            equal(float(costs[a, index-start]), value)
                            scalar_paths += 1
                            scalar_queries += count
                arrays.append(costs)
                paths += costs.size
                queries += record["transition_queries"]
            costs = np.concatenate(arrays, axis=1)
            stats = block_statistics(costs, grid, screen["sample_budgets"], independent=True)
            if block == "selection":
                freezes[identity] = {"index": stats[-1]["argmin_index"], "action": stats[-1]["argmin_action"],
                                     "selection_samples": max(screen["sample_budgets"])}
                equal(read_json(output/f"selection/{identity}.json"), freezes[identity])
                expected_events.append({"kind": "freeze", "context": identity})
                comparison, excess = None, None
            else:
                chosen = freezes[identity]["index"]
                comparison = validation_comparison(costs, chosen, context, screen["sample_budgets"], independent=True)
                excess = stats[-1]["action_means"][chosen]-min(stats[-1]["action_means"])
            blocks.append({"context": identity, "block": block, "seed": seed, "budgets": stats,
                           "validation": comparison, "validation_in_sample_excess_2048": excess})
    actual_events = [json.loads(line) for line in (output/"events.jsonl").read_text().splitlines()]
    equal(actual_events, expected_events)
    equal(read_json(output/"blocks.json"), blocks)
    if paths != screen["max_forecast_paths"]:
        raise ValueError("forecast path count mismatch")
    equal(read_json(output/"summary.json"), summarize(contexts, blocks, freezes, paths, queries))
    if require_complete:
        status = read_json(output/"status.json")
        if status["status"] != "completed" or status["exit_code"] != 0:
            raise ValueError("run is not completed successfully")
        if status["completed_chunks"] != len(records) or status["forecast_paths"] != paths:
            raise ValueError("status count mismatch")
        if (output/"error.txt").exists() or status["elapsed_seconds"] > screen["wall_cap_seconds"]:
            raise ValueError("recorded error or wall cap breach")
        equal(read_json(output/"inventory.json"), inventory(output))
    elif (output/"inventory.json").exists():
        equal(read_json(output/"inventory.json"), inventory(output))
    result = {"audit": "passed", "raw_chunks": len(records), "forecast_paths": paths,
            "scalar_replayed_paths": scalar_paths, "scalar_replayed_intervals": scalar_queries,
            "scalar_indices_per_block": list(SCALAR_AUDIT_INDICES),
            "block_budget_records": len(blocks)*len(screen["sample_budgets"]),
            "paired_contrasts_recomputed": len(blocks)*len(screen["sample_budgets"])*math.comb(len(grid), 2),
            "source_locks": len(claim["source_locks"]), "input_locks": len(claim["input_locks"]),
            "full_forecast_matrix_rerun": False, "arithmetic": "statistics.fmean/stdev on paired raw costs"}
    if require_complete:
        equal(read_json(output/"audit.json"), result)
    return result


def run(output):
    # Atomic ownership: no completed, failed or concurrent run may be overwritten.
    output.mkdir(parents=True, exist_ok=False)
    started, completed_chunks = time.monotonic(), 0
    try:
        screen = read_json(CONFIG)
        deadline = started + screen["wall_cap_seconds"]
        locked_files([str(CONFIG), PROTOCOL], PROTOCOL_COMMIT)
        if any(screen[k] for k in ("training_authorized", "new_actual_episodes_authorized", "patient_integration_authorized")):
            raise ValueError("forecast-only authority required")
        claim = {"pid": os.getpid(), "ppid": os.getppid(), "execution_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(), "source_locks": locked_files(SOURCES, "HEAD"),
            "input_locks": input_locks(screen), "python": platform.python_version(), "numpy": np.__version__,
            "seed_derivation": "named_seed(namespace, canonical_context_sha256, block_label); PCG64; path/time/site order",
            "new_actual_episodes": 0, "neural_updates": 0, "automatic_retries": 0}
        write_json(output/"claim.json", claim)
        write_json(output/"config.json", screen)
        fixture = read_json(Path(screen["s2_root"])/"mechanics.json")
        grid = action_grid(screen["grid_denominator"])
        contexts = canonical_contexts(read_json(Path(screen["s2_root"])/"probes.json"), fixture, grid,
                                      screen["source_probe_count"], screen["expected_unique_contexts"])
        expected_paths = len(contexts)*len(screen["blocks"])*max(screen["sample_budgets"])*len(grid)
        if expected_paths != screen["max_forecast_paths"]:
            raise ValueError("matrix exceeds or differs from locked count")
        write_json(output/"contexts.json", contexts)
        write_json(output/"actions.json", grid)
        (output/"selection").mkdir()
        blocks, freezes, paths, queries = [], {}, 0, 0
        with (output/"chunks.jsonl").open("x") as index, (output/"events.jsonl").open("x") as events:
            def event(row):
                events.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                events.flush()
            for block in screen["blocks"]:
                for context in contexts:
                    identity, public = context["id"], context["public"]
                    state, cfg = observation(context), make_config(fixture, public["downstream_slots"])
                    seed = named_seed(screen["namespace"], identity, block)
                    intervals = fixture["horizon"]+cfg.max_closure_steps-state.epoch
                    freeze_hash = None if block == "selection" else sha(output/f"selection/{identity}.json")
                    arrays = []
                    for start, stop, noise in noise_chunks(seed, max(screen["sample_budgets"]), intervals, screen["sample_chunk_size"]):
                        check_deadline(deadline)
                        result = forecast(state, np.asarray(grid)[:, None, :], public["response"], noise, cfg, fixture["horizon"])
                        costs = result["sample_costs"]
                        if costs.shape != (len(grid), stop-start) or not np.isfinite(costs).all():
                            raise ValueError("invalid/nonfinite forecast costs")
                        path = output/f"raw/{identity}/{block}/{start:04d}.npy"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open("xb") as stream:
                            np.save(stream, costs, allow_pickle=False)
                        record = {"context": identity, "block": block, "seed": seed, "start": start, "stop": stop,
                                  "path": str(path.relative_to(output)), "shape": list(costs.shape), "sha256": sha(path),
                                  "uniform_sha256": noise_hash(noise), "transition_queries": result["transition_queries"],
                                  "selection_freeze_sha256": freeze_hash}
                        index.write(json.dumps(record, sort_keys=True, allow_nan=False)+"\n")
                        index.flush()
                        event({"kind": "chunk", "context": identity, "block": block, "start": start})
                        completed_chunks += 1
                        paths += costs.size
                        queries += result["transition_queries"]
                        arrays.append(costs)
                        write_json(output/"status.json", {"status": "running", "pid": os.getpid(), "context": identity,
                                   "block": block, "completed_chunks": completed_chunks, "forecast_paths": paths,
                                   "elapsed_seconds": time.monotonic()-started})
                        check_deadline(deadline)
                    costs = np.concatenate(arrays, axis=1)
                    stats = block_statistics(costs, grid, screen["sample_budgets"])
                    if block == "selection":
                        freezes[identity] = frozen_choice(costs, grid)
                        write_json(output/f"selection/{identity}.json", freezes[identity])
                        event({"kind": "freeze", "context": identity})
                        comparison, excess = None, None
                    else:
                        chosen = freezes[identity]["index"]
                        comparison = validation_comparison(costs, chosen, context, screen["sample_budgets"])
                        excess = stats[-1]["action_means"][chosen]-min(stats[-1]["action_means"])
                    blocks.append({"context": identity, "block": block, "seed": seed, "budgets": stats,
                                   "validation": comparison, "validation_in_sample_excess_2048": excess})
                    write_json(output/"blocks.json", blocks)
                    print(f"{len(blocks)}/16 {block} {identity[:12]} elapsed={time.monotonic()-started:.1f}s", flush=True)
        check_deadline(deadline)
        write_json(output/"summary.json", summarize(contexts, blocks, freezes, paths, queries))
        write_json(output/"audit.json", audit_output(output, deadline, require_complete=False))
        check_deadline(deadline)
        write_json(output/"inventory.json", inventory(output))
        write_json(output/"status.json", {"status": "completed", "exit_code": 0, "pid": os.getpid(),
                   "completed_chunks": completed_chunks, "forecast_paths": paths, "elapsed_seconds": time.monotonic()-started})
    except BaseException as exc:
        (output/"error.txt").write_text(traceback.format_exc())
        write_json(output/"status.json", {"status": "failed", "exit_code": 1, "pid": os.getpid(),
                   "completed_chunks": completed_chunks, "error": repr(exc), "elapsed_seconds": time.monotonic()-started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit_output(args.output), indent=2))
    else:
        run(args.output)
