"""Stdlib-only verification of raw R3 traces, aliases, selections and contrasts."""

import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b):
    if not (math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=1e-5, rel_tol=1e-12)):
        raise ValueError(f"Raw arithmetic mismatch: {a}, {b}")


def read_trace(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def interval(values):
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    width = 2.364624251 * se
    return {"mean": mean, "se": se, "conditional_descriptive_t7_interval": [mean - width, mean + width]}


def verify(root):
    status = json.loads((root / "status.json").read_text())
    if status["status"] != "completed" or status["exit_code"] != 0:
        raise ValueError("Run did not complete")
    execution = json.loads((root / "execution.json").read_text())
    spec = execution["config"]
    stream_map = json.loads((root / "seed_audit.json").read_text())["streams"]
    for relative, expected in json.loads((root / "artifact_inventory.json").read_text()).items():
        if file_sha(root / relative) != expected:
            raise ValueError(f"Inventory mismatch: {relative}")
    records = [json.loads(line) for line in (root / "outcomes.jsonl").read_text().splitlines()]
    if len(records) != 1152:
        raise ValueError("Expected 1152 logical outcomes")
    lookup, actual_steps, alias_count, identities = {}, 0, 0, {}
    for seed in (60, 61, 62):
        trajectory = read_trace(root / f"seed{seed}_trajectory.jsonl.gz")
        if [r["t"] for r in trajectory] != list(range(52)):
            raise ValueError("Incomplete frozen trajectory")
        actual_steps += len(trajectory)
    states = {}
    for seed in (60, 61, 62):
        for step in (0, 13, 26, 39):
            with gzip.open(root / f"states/seed{seed}_t{step}.json.gz", "rt") as handle:
                state = json.load(handle)
            if digest(state["state"]) != state["state_sha256"]:
                raise ValueError("Snapshot hash mismatch")
            states[seed, step] = state
    for record in records:
        key = tuple(record[k] for k in ("seed", "step", "block", "draw", "action_index"))
        if key in lookup:
            raise ValueError("Duplicate logical outcome")
        seed, step, block, draw, action = key
        state = states[seed, step]
        if (record["source_state_sha256"] != state["state_sha256"]
                or record["continuation_policy_sha256"] != state["policy_sha256"]
                or record["rng_seed"] != stream_map[f"{seed}/{step}/{block}/{draw}"]):
            raise ValueError("Lineage or RNG mismatch")
        own = read_trace(root / record["trace"])
        actual_steps += len(own)
        post = copy.deepcopy(own[0]["post_state"])
        post["scalars"].pop("cumulative_blocked_specimen_requests", None)
        identity = digest({"state": post, "cost": own[0]["info"]["cost"],
                           "routes": own[0]["info"]["specimen_route_events"]})
        if identity != record["execution_identity"] or own[0]["request"] != state["choices"][action]["request"]:
            raise ValueError("Execution identity/request mismatch")
        local_key = (seed, step, block, draw)
        identities.setdefault(local_key, set()).add(identity)
        if record["tail_from"] is not None:
            representative = lookup[(*local_key, record["representative_action"])]
            if (len(own) != 1 or representative["execution_identity"] != identity
                    or representative["trace"] != record["tail_from"] or representative["tail_from"]):
                raise ValueError("Unsafe alias")
            rows = own + read_trace(root / record["tail_from"])[1:]
            alias_count += 1
        else:
            rows = own
        if len(rows) != 52 - step or [r["t"] for r in rows] != list(range(step, 52)):
            raise ValueError("Wrong continuation horizon")
        for i, row in enumerate(rows):
            costs = row["cost_components"]
            close(math.fsum(costs.values()), row["info"]["cost"])
            close(row["reward"], -row["info"]["cost"])
            close(row["scaled_reward"], row["reward"] * 1e-9)
            if (row["next_t"] != row["t"] + 1 or row["truncated"]
                    or row["objective_terminal"] != (i == len(rows) - 1)
                    or row["native_done"] != (i == len(rows) - 1)):
                raise ValueError("Termination/lineage mismatch")
            if i and rows[i - 1]["next_observation_sha256"] != row["observation_sha256"]:
                raise ValueError("Disconnected trajectory")
        out = record["outcome"]
        close(sum(r["info"]["cost"] for r in rows), out["total_cost"])
        close(sum(sum(r["info"]["patients_lost"]) for r in rows), out["patients_lost"])
        close(sum(sum(r["info"]["patients_completed"]) for r in rows), out["patients_completed"])
        close(sum(r["info"]["specimen_route_count"] for r in rows), out["route_count"])
        close(rows[-1]["info"]["completion_service_level"], out["completion_service_level"])
        close(rows[-1]["info"]["patient_ineligibility_during_manufacturing_rate"], out["manufacturing_loss_rate"])
        if out["step_count"] != len(rows) or not out["objective_terminal"] or out["truncated"] or out["obligations"]["t"] != 52:
            raise ValueError("Invalid completed outcome")
        lookup[key] = record
    if actual_steps != status["environment_steps"] or actual_steps > 37596:
        raise ValueError("Actual budget accounting mismatch")
    contrasts = []
    metrics = ("total_cost", "patients_lost", "patients_completed", "completion_service_level", "manufacturing_loss_rate")
    for (seed, step), state in states.items():
        discovery = [lookup[(seed, step, "discovery", d, a)] for d in range(8) for a in range(6)]
        selection = json.loads((root / f"selections/seed{seed}_t{step}.json").read_text())
        means = [statistics.mean(lookup[(seed, step, "discovery", d, a)]["outcome"]["total_cost"] for d in range(8)) for a in range(6)]
        selected = min(range(6), key=lambda a: (means[a], a))
        if selection["selected_action"] != selected or selection["discovery_rows_sha256"] != digest(discovery):
            raise ValueError("Discovery selection mismatch")
        for got, expected in zip(selection["mean_costs"], means):
            close(got, expected)
        comparison = {"seed": seed, "step": step, "selected_action": selected,
                      "selected_label": state["choices"][selected]["label"],
                      "map_reachable_certified": state["choices"][selected]["map_reachable_certified"],
                      "discovery_means": means}
        for reference, name in ((0, "vs_frozen"), (1, "vs_mdl2_first")):
            delta = {metric: [lookup[(seed, step, "validation", d, selected)]["outcome"][metric]
                              - lookup[(seed, step, "validation", d, reference)]["outcome"][metric]
                              for d in range(8)] for metric in metrics}
            comparison[name] = {"cost": interval(delta["total_cost"]), "paired_cost_deltas": delta["total_cost"],
                                "clinical_mean_deltas": {m: statistics.mean(delta[m]) for m in metrics if m != "total_cost"}}
        contrasts.append(comparison)
    return {"verified": True, "logical_records": len(records), "actual_steps": actual_steps,
            "aliased_continuations": alias_count,
            "unique_executed_choices_per_draw": {str(k): len(v) for k, v in identities.items()},
            "contrasts": contrasts, "cost_delta_negative_means_improvement": True,
            "per_seed_descriptive_mean_cost_delta": {
                str(seed): statistics.mean(r["vs_frozen"]["cost"]["mean"] for r in contrasts if r["seed"] == seed)
                for seed in (60, 61, 62)},
            "limitations": ["dependent states", "conditional RNG/keyed-loss uncertainty", "replacement baselines",
                            "not clinical noninferiority", "not a learned selector", "no online/frozen learning contrast"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("contrasts", "unique_executed_choices_per_draw")}, indent=2))
