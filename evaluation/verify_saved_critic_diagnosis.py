"""Independent stdlib checks of R6 saved-data diagnostic arithmetic."""

import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def read(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
        return json.load(handle)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def near(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"Arithmetic mismatch: {a}, {b}")


def verify(root, report, repeat):
    if sha(report) != sha(repeat):
        raise ValueError("Analysis repeat differs")
    result = read(report)
    for name, value in read(root / "artifact_inventory.json").items():
        if sha(root / name) != value:
            raise ValueError(f"R6 source changed: {name}")
    lookup = {}
    for line in (root / "outcomes.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = r["state_id"], r["draw"], r["action_index"]
        if key in lookup:
            raise ValueError("Duplicate raw outcome")
        lookup[key] = r
    reported_pairs = {(r["state_id"], tuple(r["actions"])): r for r in result["pairs"]}
    expected, groups, equivalents = set(), {"train": Counter(), "test": Counter()}, set()
    sealed_choices = {}
    for seed in (60, 61, 62):
        fit = read(root / f"seed{seed}/fit_summary.json")
        seal = read(root / f"seed{seed}/sealed_test_predictions.json")
        sealed_choices.update(zip(seal["state_ids"], seal["critic_actions"]))
        for role, trajectories in (("train", 4), ("test", 2)):
            predictions = fit["training_predictions"] if role == "train" else seal["advantage_predictions"]
            for i, (trajectory, step) in enumerate(itertools.product(range(trajectories), (0, 13, 26, 39))):
                name = f"seed{seed}/{role}{trajectory}/t{step}"
                state = read(root / f"states/{name}.json.gz")
                support = [a for a, c in enumerate(state["choices"]) if c["map_reachable_certified"]]
                for a, b in itertools.combinations(support, 2):
                    key = name, (a, b)
                    expected.add(key)
                    r = reported_pairs[key]
                    draws = [(lookup[name, d, b]["outcome"]["total_cost"] - lookup[name, d, a]["outcome"]["total_cost"]) * 1e-9 for d in range(8)]
                    first, second = statistics.mean(draws[:4]), statistics.mean(draws[4:])
                    group = ("both_tied" if first == second == 0 else "one_tied" if first == 0 or second == 0
                             else "same_sign" if first * second > 0 else "opposite_sign")
                    prediction = predictions[i][a] - predictions[i][b]
                    near(r["mean_advantage_difference"], statistics.mean(draws))
                    near(r["paired_se"], statistics.stdev(draws) / math.sqrt(8))
                    near(r["prediction_difference"], prediction)
                    if r["group"] != group or r["correct"] != (statistics.mean(draws) * prediction > 0):
                        raise ValueError("Wrong split/ranking result")
                    groups[role][group] += 1
                    groups[role]["comparable"] += statistics.mean(draws) != 0
                    groups[role]["correct"] += r["correct"]
                    if all(lookup[name, d, a]["execution_identity"] == lookup[name, d, b]["execution_identity"] for d in range(8)):
                        equivalents.add(key)
    if expected != set(reported_pairs) or len(expected) != len(result["pairs"]):
        raise ValueError("Missing/duplicate action pairs")
    if equivalents != {(r["state_id"], tuple(r["actions"])) for r in result["sampled_execution_equivalences"]}:
        raise ValueError("Execution-equivalence mismatch")
    for role in groups:
        for group in ("same_sign", "opposite_sign", "both_tied", "one_tied"):
            if groups[role][group] != result["aggregate_label_stability"][role][group]["pairs"]:
                raise ValueError("Group denominator mismatch")
    def components(record):
        with gzip.open(root / record["trace"], "rt") as handle:
            rows = [json.loads(line) for line in handle]
        if record["tail_from"]:
            with gzip.open(root / record["tail_from"], "rt") as handle:
                rows += [json.loads(line) for line in handle][1:]
        return {key: math.fsum(row["cost_components"][key] for row in rows) for key in rows[0]["cost_components"]}
    checks = 0
    conflicts = []
    if (len(result["sealed_test_choices"]) != 24
            or {r["state_id"] for r in result["sealed_test_choices"]} != set(sealed_choices)):
        raise ValueError("Missing/duplicate sealed test choice")
    for row in result["sealed_test_choices"]:
        name, action = row["state_id"], row["action"]
        if action != sealed_choices[name]:
            raise ValueError("Reported choice differs from original seal")
        totals, losses = [], []
        component_draws = []
        for d in range(8):
            chosen, frozen = lookup[name, d, action], lookup[name, d, 0]
            left, right = components(chosen), components(frozen)
            delta = {k: left[k] - right[k] for k in left}
            for k, value in delta.items():
                if not math.isclose(value, row["component_draw_deltas"][d][k], abs_tol=1e-5, rel_tol=1e-9):
                    raise ValueError("Component contrast differs")
            total = chosen["outcome"]["total_cost"] - frozen["outcome"]["total_cost"]
            if not math.isclose(math.fsum(delta.values()), total, abs_tol=1e-5, rel_tol=1e-9):
                raise ValueError("Components do not sum to raw outcome")
            totals.append(total)
            losses.append(chosen["outcome"]["patients_lost"] - frozen["outcome"]["patients_lost"])
            component_draws.append(delta)
            checks += 1
        for k in component_draws[0]:
            if not math.isclose(statistics.mean(d[k] for d in component_draws), row["mean_component_deltas"][k], abs_tol=1e-5, rel_tol=1e-9):
                raise ValueError("Component mean differs")
        conflict = statistics.mean(totals) < 0 and statistics.mean(losses) > 0
        if conflict != row["cheaper_but_more_loss"]:
            raise ValueError("Cost/clinical conflict flag differs")
        if conflict:
            conflicts.append(name)
    return {"status": "passed", "diagnosis_sha256": sha(report), "repeat_byte_identical": True,
            "checked_pairs": len(expected), "checked_execution_equivalences": len(equivalents),
            "checked_sealed_choice_draws": checks, "groups": groups,
            "cheaper_but_more_loss_states": conflicts, "new_steps_or_fitting": 0,
            "scope": "independent raw pair arithmetic and all selected cost components; coverage descriptive, not an OOD test"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("repeat", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = verify(args.root, args.report, args.repeat)
    with args.output.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(value, indent=2))
