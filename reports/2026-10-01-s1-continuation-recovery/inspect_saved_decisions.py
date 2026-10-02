"""Post-hoc descriptive readback only; no checkpoint, model or simulator access."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_bound(root, record):
    path = (root / record["path"]).resolve()
    path.relative_to(root)
    data = path.read_bytes()
    assert len(data) == record["bytes"]
    assert hashlib.sha256(data).hexdigest() == record["sha256"]
    return [json.loads(line) for line in data.splitlines()]


def summary(values):
    return {"n": len(values), "mean": statistics.fmean(values),
            "min": min(values), "max": max(values)}


def inspect(root):
    verification = root / "payload/independent-verification.json"
    data = verification.read_bytes()
    verified = json.loads(data)
    outcomes = verified["outcomes"]
    assert len(outcomes) == 180
    indexed = {(r["block"], r["world_index"], r["role"]): r for r in outcomes}
    assert len(indexed) == 180
    roles = ("own_frozen", "own_ppo", "own_bc_continue")
    stats, shifts, sources = defaultdict(lambda: defaultdict(list)), defaultdict(list), []
    for block in (60, 61, 62):
        for world in range(12):
            traces = {}
            for role in roles:
                outcome = indexed[block, world, role]
                record = outcome["raw_files"]["events"]
                traces[role] = read_bound(root, record)
                sources.append(record)
                assert len(traces[role]) == 52
            for step in range(52):
                decisions = {r: traces[r][step]["event"]["audit"]["decision"] for r in roles}
                base = decisions["own_frozen"]["evaluation"]
                bank = base["candidates"]
                reference = bank["request_to_class"][0]
                assert len(bank["class_keys"]) > 1
                for role, decision in decisions.items():
                    evaluation = decision["evaluation"]
                    assert evaluation["actor_state"] == base["actor_state"]
                    assert evaluation["candidates"] == bank
                    probabilities = [math.exp(x) for x in evaluation["log_probs"]]
                    assert all(math.isfinite(x) for x in probabilities)
                    assert abs(math.fsum(probabilities) - 1.0) < 1e-5
                    assert max(range(len(probabilities)), key=probabilities.__getitem__) == decision["choice"]["class_index"]
                    bucket = stats[block, role]
                    bucket["reference_probability"].append(probabilities[reference])
                    bucket["reference_probability_margin"].append(
                        probabilities[reference] - max(p for i, p in enumerate(probabilities) if i != reference))
                    bucket["reference_chosen"].append(int(decision["choice"]["class_index"] == reference))
                    assert decision["choice"]["submitted_request"] == decisions["own_frozen"]["choice"]["submitted_request"]
                    if role != "own_frozen":
                        base_p = [math.exp(x) for x in base["log_probs"]]
                        shifts[block, role].append(math.fsum(abs(x - y) for x, y in zip(probabilities, base_p)))
    all_path = root / "payload/all-episode-verification.json"
    all_bytes = all_path.read_bytes()
    all_outcomes = json.loads(all_bytes)["outcomes"]
    training = []
    for block in (60, 61, 62):
        for role in roles[1:]:
            rows = [r for r in all_outcomes if r["split"] == "training" and r["block"] == block and r["role"] == role]
            assert len(rows) == 32
            steps = sum(r["steps"] for r in rows)
            reference = sum(r["reference_class_choices"] for r in rows)
            training.append({"block": block, "role": role, "episodes": 32, "steps": steps,
                             "reference_choices": reference, "nonreference_choices": steps - reference,
                             "nonreference_fraction": (steps - reference) / steps})
    return {
        "kind": "posthoc_saved_decision_description_v1",
        "new_model_calls": 0, "new_environment_calls": 0, "new_optimizer_calls": 0,
        "verification_sha256": hashlib.sha256(data).hexdigest(),
        "all_episode_verification_sha256": hashlib.sha256(all_bytes).hexdigest(),
        "raw_event_files": sources,
        "same_context_and_bank_pairs_verified": 3744,
        "paired_test_states": 1872,
        "role_state_rows": 5616,
        "evaluation_by_block_role": [
            {"block": b, "role": r, "reference_chosen": sum(v["reference_chosen"]),
             "rows": len(v["reference_chosen"]),
             "reference_probability": summary(v["reference_probability"]),
             "reference_probability_margin": summary(v["reference_probability_margin"])}
            for (b, r), v in sorted(stats.items())],
        "probability_l1_change_from_frozen": [
            {"block": b, "role": r, **summary(values),
             "nonzero_rows": sum(v > 0 for v in values)} for (b, r), values in sorted(shifts.items())],
        "training_sampling": training,
        "interpretation": "Recorded probabilities can change while greedy requests remain identical; no fresh state, stochastic deployment or headroom experiment was performed.",
        "scope": "Post-hoc descriptive explanation of the fixed primary null; not a new prespecified endpoint or independent confirmation."
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.root.resolve())
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "raw_event_files"}, indent=2))
