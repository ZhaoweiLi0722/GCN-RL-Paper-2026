"""Read-only standard-library pilot audit. No simulator imports or rollouts."""

import collections
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path


def check(condition, label):
    if not condition:
        raise AssertionError(label)


def finite(value):
    if isinstance(value, float):
        check(math.isfinite(value), "nonfinite metric")
    elif isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, list):
        for v in value:
            finite(v)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/2026-09-28-disruption-feasibility/pilot")
    load = lambda name: json.loads((root / name).read_text())
    status, execution, summary, inventory = (load(n) for n in ("status.json", "execution.json", "summary.json", "inventory.json"))
    check(status["status"] == "completed" and status["exit_code"] == 0, "completed execution")
    for path, expected_hash in inventory["source_sha256"].items():
        check(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_hash, "source hash: " + path)
    for name, expected_hash in inventory["output_sha256"].items():
        check(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected_hash, "output hash: " + name)
    rows = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
    finite(rows)
    key = lambda r: (r["cell"], r["split"], r["world_seed"], r["policy"])
    index = {key(r): r for r in rows}
    spec = execution["spec"]
    cells = ("capacity_nochange", "capacity_outage", "lead_nochange", "lead_shift")
    policies = [p["name"] for p in spec["policies"]]
    expected = {(c, phase, s, p) for c in cells for phase in ("discovery", "validation")
                for s in spec[phase + "_seeds"] for p in policies}
    check(set(index) == expected and len(rows) == len(expected), "episode Cartesian product")
    horizon = execution["composed_env"]["episode_horizon"]
    seen, costs, errors = set(), collections.defaultdict(float), []
    total_capacity = sum(execution["composed_env"]["initial_idle_bioreactors"])
    with (root / "steps.jsonl").open() as handle:
        for line in handle:
            r = json.loads(line)
            finite(r)
            k = key(r)
            step_key = k + (r["step"],)
            check(step_key not in seen and k in expected, "step uniqueness")
            seen.add(step_key)
            costs[k] += r["cost"]
            errors.append(abs(r["reactor_stock"] - total_capacity))
    check(seen == {k + (t,) for k, t in itertools.product(expected, range(horizon))}, "step Cartesian product")
    check(max(errors) < 1e-7, "capacity conservation")
    for k, r in index.items():
        check(math.isclose(costs[k], r["total_cost"], rel_tol=1e-12), "cost reconciliation")
    for seed in spec["discovery_seeds"] + spec["validation_seeds"]:
        check(len({(r["arrival_sha256"], r["rng_sha256"]) for r in rows if r["world_seed"] == seed}) == 1, "paired worlds")
    comparisons = {}
    for c in cells:
        means = {p: sum(index[c, "discovery", s, p]["total_cost"] for s in spec["discovery_seeds"]) for p in policies}
        check(min(means, key=means.get) == summary["cells"][c]["discovery_selected"], "discovery-only selection")
        comparisons[c] = {}
        for p in policies:
            a = sum(index[c, "validation", s, p]["total_cost"] for s in spec["validation_seeds"])
            b = sum(index[c, "validation", s, "mdl2"]["total_cost"] for s in spec["validation_seeds"])
            change = 100 * (a - b) / b
            reported = summary["cells"][c]["validation_differences_vs_mdl2"][p]["cost_change_pct"]
            check(math.isclose(change, reported, rel_tol=1e-9, abs_tol=1e-9), "paired cost change")
            comparisons[c][p] = change
    for phase in ("discovery", "validation"):
        for seed, p in itertools.product(spec[phase + "_seeds"], policies):
            a, b = (index[c, phase, seed, p] for c in ("capacity_nochange", "lead_nochange"))
            check(all(a[k] == b[k] for k in a if k != "cell"), "identical controls")
    print(json.dumps({"audit_passes": True, "episodes": len(rows), "steps": len(seen),
                      "maximum_capacity_error": max(errors),
                      "source_hashes": len(inventory["source_sha256"]),
                      "output_hashes": len(inventory["output_sha256"]),
                      "recomputed_validation_cost_change_pct": comparisons}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
