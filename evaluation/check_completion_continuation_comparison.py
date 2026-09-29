"""Locked S4 model-only continuation matrix and read-only independent audit."""

import argparse
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

from evaluation import check_completion_action_ranking as s3
from evaluation.check_completion_feedback_mechanics import equal, make_config
from evaluation.check_service_effort_mechanics import locked_files, sha, write_json
from src.baselines.completion_feedback_tree import reachable_events, scalar_continuation, second_action_indices
from src.baselines.completion_rollout_control import action_grid, named_seed, reservation_action
from src.env.completion_continuations import forecast_continuations
from src.env.completion_feedback_batch import forecast
from src.env.completion_feedback_queue import advance, booked_backlog_action


CONFIG = Path("experiments/configs/completion_continuation_comparison_20260929.json")
PROTOCOL = "specs/2026-09-29-completion-continuation-comparison/protocol.md"
PROTOCOL_COMMIT = "e4e9fef"
SOURCES = [str(CONFIG), PROTOCOL, "evaluation/check_completion_continuation_comparison.py",
           "src/env/completion_continuations.py", "src/baselines/completion_feedback_tree.py",
           "tests/test_completion_continuation_comparison.py"]
COMPONENTS = ("holding", "labor", "switching")


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("S4 wall cap reached; retain evidence without retry")


def load_inputs(config):
    root = Path(config["s3_root"])
    for name in ("summary", "inventory"):
        if sha(root/f"{name}.json") != config[f"s3_{name}_sha256"]:
            raise ValueError("S3 input hash mismatch")
    claim = s3.read_json(root/"claim.json")
    paths = list(dict.fromkeys([r["path"] for r in claim["source_locks"]+claim["input_locks"]] +
                              [str(p) for p in sorted(root.rglob("*")) if p.is_file()]))
    locks = locked_files(paths, config["source_commit"])
    contexts = s3.read_json(root/"contexts.json")
    if len(contexts) != config["expected_unique_contexts"] or sum(len(c["aliases"]) for c in contexts) != config["source_probe_aliases"]:
        raise ValueError("context/alias count mismatch")
    if len({c["id"] for c in contexts}) != len(contexts):
        raise ValueError("duplicate canonical context")
    for context in contexts:
        if s3.context_digest(context["public"]) != context["id"]:
            raise ValueError("context content hash mismatch")
    return contexts, locks


def save_array(path, key, value):
    if not np.isfinite(value).all():
        raise ValueError("nonfinite raw array")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **{key: value})
    return sha(path)


def load_array(path, key, shape, digest):
    if sha(path) != digest:
        raise ValueError("raw array hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != [key]:
            raise ValueError("raw array schema mismatch")
        value = archive[key]
    if value.shape != tuple(shape) or value.dtype != np.dtype("float64") or not np.isfinite(value).all():
        raise ValueError("raw array shape/type/finite mismatch")
    return value


def means(values, independent=False):
    return [statistics.fmean(map(float, row)) for row in values] if independent else values.mean(axis=1).tolist()


def inner_statistics(costs, budgets, independent=False):
    records = []
    for n in budgets:
        values = means(costs[:, :n], independent)
        chosen = min(range(len(values)), key=lambda a: values[a])
        records.append({"samples": n, "means": values, "index": chosen,
                        "chosen_minus_alternatives": [s3.paired_stats(costs[chosen, :n], costs[a, :n], independent=independent)
                                                       for a in range(len(values))]})
    return records


def rank_statistics(components, config, grid, independent=False):
    expected = (len(config["methods"]), len(grid), max(config["outer_budgets"]), 3)
    if components.shape != expected or not np.isfinite(components).all():
        raise ValueError("incomplete outer component matrix")
    totals = components.sum(axis=-1)
    rows = []
    for m, method in enumerate(config["methods"]):
        for n in config["outer_budgets"]:
            values = means(totals[m, :, :n], independent)
            best = min(range(len(grid)), key=lambda a: values[a])
            rows.append({"method": method, "samples": n, "means": values, "index": best,
                         "action": list(grid[best]), "exact_ties": [i for i, v in enumerate(values) if v == values[best]]})
    return rows


def comparisons(components, config, selection, context, grid, independent=False):
    state, public = s3.observation(context), context["public"]
    cfg = make_config(public["mechanics"], public["downstream_slots"])
    rule_indices = [grid.index(booked_backlog_action(state, cfg)),
                    grid.index(reservation_action(state, public["response"], cfg, grid))]
    frozen = [selection[m] for m in config["methods"]]
    pairs = [("tree128_minus_booked_selected", 3, frozen[3], 0, frozen[0]),
             ("tree128_minus_reservation_selected", 3, frozen[3], 1, frozen[1]),
             ("tree32_minus_booked_selected", 2, frozen[2], 0, frozen[0]),
             ("tree32_minus_reservation_selected", 2, frozen[2], 1, frozen[1]),
             ("tree128_minus_tree32", 3, frozen[3], 2, frozen[2]),
             ("tree128_minus_booked_rule", 3, frozen[3], 0, rule_indices[0]),
             ("tree128_minus_reservation_rule", 3, frozen[3], 1, rule_indices[1])]
    rows = []
    for n in config["outer_budgets"]:
        for name, lm, la, rm, ra in pairs:
            left, right = components[lm, la, :n], components[rm, ra, :n]
            rows.append({"name": name, "samples": n, "left_action_index": la, "right_action_index": ra,
                         **s3.paired_stats(left.sum(axis=1), right.sum(axis=1), independent=independent),
                         "component_differences": {key: s3.paired_stats(left[:, i], right[:, i], independent=independent)
                                                   for i, key in enumerate(COMPONENTS)}})
    return rows


def summary(contexts, trees, blocks, config, inner_paths, outer_paths, queries):
    result = []
    wanted = {f"tree{n}_minus_{m}_selected" for n in config["tree_inner_budgets"] for m in ("booked", "reservation")}
    for context in contexts:
        records = [b for b in blocks if b["context"] == context["id"]]
        full = [r for b in records if b["block"] != "selection" for r in b["comparisons"]
                if r["samples"] == max(config["outer_budgets"]) and r["name"] in wanted]
        if len(full) != 12:
            raise ValueError("incomplete primary validation comparisons")
        result.append({"context": context["id"], "aliases": context["aliases"],
                       "direction_stable_conditional_signal": all(r["mean_left_minus_right"] < 0 for r in full),
                       "validation_differences": {key: [r["mean_left_minus_right"] for r in full if r["name"] == key]
                                                  for key in sorted(wanted)},
                       "continuation_argmins_differ_by_block": {b["block"]: len({r["index"] for r in b["ranks"]
                                                                               if r["samples"] == max(config["outer_budgets"])}) > 1
                                                              for b in records}})
    return {"classification": "conditional_continuation_diagnostic_not_online_rl", "contexts": result,
            "contexts_with_signal": sum(c["direction_stable_conditional_signal"] for c in result),
            "tree_branch_first_rows": sum(len(t["fit_rows"]) for t in trees),
            "tree_budget_action_disagreements": sum(r["budgets"][0]["index"] != r["budgets"][1]["index"]
                                                    for t in trees for r in t["fit_rows"]),
            "inner_model_paths": inner_paths, "outer_model_paths": outer_paths, "transition_queries": queries,
            "actual_environment_episodes": 0, "neural_updates": 0, "reward_weights_changed": False,
            "model_validated": False, "online_rl_evidence": False, "strong_mpc_bound": False}


def verify_counts(inner, outer, queries, config):
    if not 0 < inner <= config["max_inner_paths"] or outer != config["outer_paths"]:
        raise ValueError("model path count outside protocol")
    if inner+outer > config["max_total_paths"] or not 0 < queries <= config["max_transition_queries"]:
        raise ValueError("query/path cap exceeded")


def unique_records(rows, keys):
    mapping = {tuple(row[k] for k in keys): row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("duplicate raw record")
    return mapping


def audit_output(output, deadline=None, require_complete=True):
    config, claim = s3.read_json(output/"config.json"), s3.read_json(output/"claim.json")
    equal(config, s3.read_json(CONFIG))
    locked_files([str(CONFIG), PROTOCOL], PROTOCOL_COMMIT)
    equal(claim["source_locks"], locked_files(SOURCES, claim["execution_commit"]))
    contexts, locks = load_inputs(config)
    equal(claim["input_locks"], locks)
    equal(s3.read_json(output/"contexts.json"), contexts)
    grid = action_grid(config["grid_denominator"])
    records = [json.loads(line) for line in (output/"raw.jsonl").read_text().splitlines()]
    indexed = unique_records(records, ("kind", "context", "label", "index"))
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("duplicate raw path")
    actual_files = {str(p.relative_to(output)) for p in (output/"raw").rglob("*") if p.is_file()}
    if actual_files != {r["path"] for r in records}:
        raise ValueError("missing/unindexed raw file")
    consumed, events, trees, tables_by_context = set(), [], [], {}
    inner_paths = outer_paths = queries = scalar_paths = scalar_steps = 0
    for context in contexts:
        identity, public = context["id"], context["public"]
        state = s3.observation(context)
        fixture, response = public["mechanics"], public["response"]
        cfg = make_config(fixture, public["downstream_slots"])
        branches = reachable_events(state, response, cfg)
        tables = {str(n): np.full((4, len(grid)), -1, dtype=int) for n in config["tree_inner_budgets"]}
        fit_rows = []
        for branch in branches:
            seed = named_seed(config["namespace"], identity, "tree_fit", branch["code"])
            intervals = fixture["horizon"]+cfg.max_closure_steps-state.epoch-1
            noise = np.random.Generator(np.random.PCG64(seed)).random((max(config["tree_inner_budgets"]), intervals, 2))
            for first, action in enumerate(grid):
                check_deadline(deadline)
                key = ("inner", identity, str(branch["code"]), first)
                record = indexed[key]
                consumed.add(key)
                path = f"raw/{identity}/inner/{branch['code']}_{first:02d}.npz"
                if record["path"] != path or record["seed"] != seed or record["uniform_sha256"] != s3.noise_hash(noise):
                    raise ValueError("inner stream/identity mismatch")
                after, receipt, _ = advance(state, action, response, branch["representative_uniforms"], cfg)
                equal(list(receipt.completed), branch["events"])
                costs = load_array(output/path, "costs", (len(grid), len(noise)), record["sha256"])
                for sample in (0, len(noise)-1):
                    for second, request in enumerate(grid):
                        value, count = s3.scalar_cost(after, request, response, noise[sample], cfg, fixture["horizon"])
                        equal(value, float(costs[second, sample]))
                        scalar_paths += 1
                        scalar_steps += count
                stats = inner_statistics(costs, config["tree_inner_budgets"], True)
                for row in stats:
                    tables[str(row["samples"])][branch["code"], first] = row["index"]
                fit_rows.append({"code": branch["code"], "first": first, "budgets": stats})
                if not 0 < record["queries"] <= costs.size*intervals:
                    raise ValueError("inner query count mismatch")
                inner_paths += costs.size
                queries += record["queries"]
                events.append({"kind": "inner", "context": identity, "code": branch["code"], "first": first})
        tree = {"context": identity, "branches": branches, "tables": {k: v.tolist() for k, v in tables.items()}, "fit_rows": fit_rows}
        equal(s3.read_json(output/f"trees/{identity}.json"), tree)
        trees.append(tree)
        tables_by_context[identity] = tables
        events.append({"kind": "freeze_tree", "context": identity})
    blocks, selections = [], {}
    for block in config["blocks"]:
        for context in contexts:
            identity, public = context["id"], context["public"]
            state, fixture = s3.observation(context), public["mechanics"]
            cfg = make_config(fixture, public["downstream_slots"])
            seed = named_seed(config["namespace"], identity, "outer", block)
            intervals = fixture["horizon"]+cfg.max_closure_steps-state.epoch
            arrays = []
            for start, stop, noise in s3.noise_chunks(seed, max(config["outer_budgets"]), intervals, config["chunk_size"]):
                check_deadline(deadline)
                key = ("outer", identity, block, start)
                record = indexed[key]
                consumed.add(key)
                path = f"raw/{identity}/{block}/{start:04d}.npz"
                if record["path"] != path or record["seed"] != seed or record["uniform_sha256"] != s3.noise_hash(noise):
                    raise ValueError("outer stream/identity mismatch")
                if record["tree_sha256"] != sha(output/f"trees/{identity}.json"):
                    raise ValueError("tree not frozen before outer evaluation")
                frozen_hash = None if block == "selection" else sha(output/f"selection/{identity}.json")
                if record["selection_sha256"] != frozen_hash:
                    raise ValueError("validation selection changed")
                costs = load_array(output/path, "components", (len(config["methods"]), len(grid), stop-start, 3), record["sha256"])
                for sample in config["scalar_audit_indices"]:
                    if not start <= sample < stop:
                        continue
                    for m, method in enumerate(config["methods"]):
                        table = tables_by_context[identity].get(method.removeprefix("tree"))
                        for first in range(len(grid)):
                            value, trace = scalar_continuation(state, first, grid, public["response"], noise[sample-start], cfg,
                                                               fixture["horizon"], method, table)
                            equal(value.tolist(), costs[m, first, sample-start].tolist())
                            scalar_paths += 1
                            scalar_steps += len(trace)
                count = int(np.prod(costs.shape[:-1]))
                if not 0 < record["queries"] <= count*intervals:
                    raise ValueError("outer query count mismatch")
                outer_paths += count
                queries += record["queries"]
                arrays.append(costs)
                events.append({"kind": "outer", "context": identity, "block": block, "start": start})
            components = np.concatenate(arrays, axis=2)
            ranks = rank_statistics(components, config, grid, True)
            if block == "selection":
                selections[identity] = {r["method"]: r["index"] for r in ranks if r["samples"] == max(config["outer_budgets"])}
                equal(s3.read_json(output/f"selection/{identity}.json"), selections[identity])
                events.append({"kind": "freeze_selection", "context": identity})
                contrasts = []
            else:
                contrasts = comparisons(components, config, selections[identity], context, grid, True)
            blocks.append({"context": identity, "block": block, "ranks": ranks, "comparisons": contrasts})
    if consumed != set(indexed):
        raise ValueError("unexpected raw record")
    verify_counts(inner_paths, outer_paths, queries, config)
    equal([json.loads(line) for line in (output/"events.jsonl").read_text().splitlines()], events)
    equal(s3.read_json(output/"blocks.json"), blocks)
    equal(s3.read_json(output/"summary.json"), summary(contexts, trees, blocks, config, inner_paths, outer_paths, queries))
    result = {"audit": "passed", "raw_arrays": len(records), "inner_paths": inner_paths, "outer_paths": outer_paths,
              "scalar_replayed_paths": scalar_paths, "scalar_replayed_steps": scalar_steps,
              "source_locks": len(claim["source_locks"]), "input_locks": len(locks),
              "numerical_audit": "independent fmean/stdev on every raw comparison", "full_matrix_rerun": False}
    if require_complete:
        status = s3.read_json(output/"status.json")
        if status["status"] != "completed" or status["exit_code"] != 0 or status["elapsed_seconds"] > config["wall_cap_seconds"]:
            raise ValueError("run not completed within cap")
        if status["inner_paths"] != inner_paths or status["outer_paths"] != outer_paths:
            raise ValueError("status path count mismatch")
        if (output/"error.txt").exists():
            raise ValueError("recorded error")
        equal(s3.read_json(output/"audit.json"), result)
        equal(s3.read_json(output/"inventory.json"), s3.inventory(output))
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inner_paths = outer_paths = queries = 0
    try:
        config = s3.read_json(CONFIG)
        deadline = started+config["wall_cap_seconds"]
        locked_files([str(CONFIG), PROTOCOL], PROTOCOL_COMMIT)
        if any(config[k] for k in ("training_authorized", "new_actual_episodes_authorized", "reward_changes_authorized")):
            raise ValueError("model-only authority required")
        contexts, locks = load_inputs(config)
        claim = {"pid": os.getpid(), "ppid": os.getppid(), "execution_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(), "source_locks": locked_files(SOURCES, "HEAD"),
            "input_locks": locks, "python": platform.python_version(), "numpy": np.__version__,
            "component_axis": list(COMPONENTS), "seed_policy": "PCG64; named_seed(namespace, context, tree_fit/outer, branch/block)",
            "new_actual_episodes": 0, "neural_updates": 0, "reward_changes": False}
        write_json(output/"claim.json", claim)
        write_json(output/"config.json", config)
        write_json(output/"contexts.json", contexts)
        (output/"trees").mkdir()
        (output/"selection").mkdir()
        grid, trees, tables_by_context = action_grid(config["grid_denominator"]), [], {}
        with (output/"raw.jsonl").open("x") as raw, (output/"events.jsonl").open("x") as event_stream:
            def emit(stream, row):
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                stream.flush()
            def progress(phase):
                write_json(output/"status.json", {"status": "running", "pid": os.getpid(), "phase": phase,
                           "inner_paths": inner_paths, "outer_paths": outer_paths, "queries": queries,
                           "elapsed_seconds": time.monotonic()-started})
                if inner_paths > config["max_inner_paths"] or inner_paths+outer_paths > config["max_total_paths"] or queries > config["max_transition_queries"]:
                    raise ValueError("computation cap exceeded")
                check_deadline(deadline)
            for context in contexts:
                identity, public = context["id"], context["public"]
                state, fixture, response = s3.observation(context), public["mechanics"], public["response"]
                cfg = make_config(fixture, public["downstream_slots"])
                branches = reachable_events(state, response, cfg)
                tables = {str(n): np.full((4, len(grid)), -1, dtype=int) for n in config["tree_inner_budgets"]}
                fit_rows = []
                for branch in branches:
                    seed = named_seed(config["namespace"], identity, "tree_fit", branch["code"])
                    intervals = fixture["horizon"]+cfg.max_closure_steps-state.epoch-1
                    noise = np.random.Generator(np.random.PCG64(seed)).random((max(config["tree_inner_budgets"]), intervals, 2))
                    for first, action in enumerate(grid):
                        progress("tree_fit")
                        after, receipt, _ = advance(state, action, response, branch["representative_uniforms"], cfg)
                        equal(list(receipt.completed), branch["events"])
                        result = forecast(after, np.asarray(grid)[:, None, :], response, noise, cfg, fixture["horizon"])
                        costs = result["sample_costs"]
                        path = output/f"raw/{identity}/inner/{branch['code']}_{first:02d}.npz"
                        digest = save_array(path, "costs", costs)
                        emit(raw, {"kind": "inner", "context": identity, "label": str(branch["code"]), "index": first,
                                   "path": str(path.relative_to(output)), "sha256": digest, "seed": seed,
                                   "uniform_sha256": s3.noise_hash(noise), "queries": result["transition_queries"]})
                        emit(event_stream, {"kind": "inner", "context": identity, "code": branch["code"], "first": first})
                        stats = inner_statistics(costs, config["tree_inner_budgets"])
                        for budget, chosen in second_action_indices(costs, config["tree_inner_budgets"]).items():
                            tables[budget][branch["code"], first] = chosen
                        fit_rows.append({"code": branch["code"], "first": first, "budgets": stats})
                        inner_paths += costs.size
                        queries += result["transition_queries"]
                        progress("tree_fit")
                tree = {"context": identity, "branches": branches, "tables": {k: v.tolist() for k, v in tables.items()}, "fit_rows": fit_rows}
                write_json(output/f"trees/{identity}.json", tree)
                emit(event_stream, {"kind": "freeze_tree", "context": identity})
                tables_by_context[identity] = tables
                trees.append(tree)
                print(f"tree {identity[:12]} frozen; inner_paths={inner_paths}", flush=True)
            blocks, selections = [], {}
            for block in config["blocks"]:
                for context in contexts:
                    identity, public = context["id"], context["public"]
                    state, fixture = s3.observation(context), public["mechanics"]
                    cfg = make_config(fixture, public["downstream_slots"])
                    seed = named_seed(config["namespace"], identity, "outer", block)
                    intervals = fixture["horizon"]+cfg.max_closure_steps-state.epoch
                    arrays = []
                    for start, stop, noise in s3.noise_chunks(seed, max(config["outer_budgets"]), intervals, config["chunk_size"]):
                        progress(block)
                        results = [forecast_continuations(state, grid, public["response"], noise, cfg, fixture["horizon"], method,
                                   tables_by_context[identity].get(method.removeprefix("tree"))) for method in config["methods"]]
                        costs = np.stack([r["components"] for r in results])
                        chunk_queries = sum(r["transition_queries"] for r in results)
                        path = output/f"raw/{identity}/{block}/{start:04d}.npz"
                        digest = save_array(path, "components", costs)
                        emit(raw, {"kind": "outer", "context": identity, "label": block, "index": start,
                                   "path": str(path.relative_to(output)), "sha256": digest, "seed": seed,
                                   "uniform_sha256": s3.noise_hash(noise), "queries": chunk_queries,
                                   "tree_sha256": sha(output/f"trees/{identity}.json"),
                                   "selection_sha256": None if block == "selection" else sha(output/f"selection/{identity}.json")})
                        emit(event_stream, {"kind": "outer", "context": identity, "block": block, "start": start})
                        outer_paths += int(np.prod(costs.shape[:-1]))
                        queries += chunk_queries
                        arrays.append(costs)
                        progress(block)
                    components = np.concatenate(arrays, axis=2)
                    ranks = rank_statistics(components, config, grid)
                    if block == "selection":
                        selections[identity] = {r["method"]: r["index"] for r in ranks if r["samples"] == max(config["outer_budgets"])}
                        write_json(output/f"selection/{identity}.json", selections[identity])
                        emit(event_stream, {"kind": "freeze_selection", "context": identity})
                        contrasts = []
                    else:
                        contrasts = comparisons(components, config, selections[identity], context, grid)
                    blocks.append({"context": identity, "block": block, "ranks": ranks, "comparisons": contrasts})
                    write_json(output/"blocks.json", blocks)
                    print(f"outer {len(blocks)}/16 {block} {identity[:12]} elapsed={time.monotonic()-started:.1f}s", flush=True)
        verify_counts(inner_paths, outer_paths, queries, config)
        write_json(output/"summary.json", summary(contexts, trees, blocks, config, inner_paths, outer_paths, queries))
        write_json(output/"audit.json", audit_output(output, deadline, require_complete=False))
        check_deadline(deadline)
        write_json(output/"inventory.json", s3.inventory(output))
        write_json(output/"status.json", {"status": "completed", "exit_code": 0, "pid": os.getpid(),
                   "elapsed_seconds": time.monotonic()-started, "inner_paths": inner_paths, "outer_paths": outer_paths})
    except BaseException as exc:
        (output/"error.txt").write_text(traceback.format_exc())
        write_json(output/"status.json", {"status": "failed", "exit_code": 1, "error": repr(exc),
                   "elapsed_seconds": time.monotonic()-started, "inner_paths": inner_paths, "outer_paths": outer_paths})
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
