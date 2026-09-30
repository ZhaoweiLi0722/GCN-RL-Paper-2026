"""R6 post-run arithmetic only: no simulator, model inference, fitting or RNG."""

import argparse
from collections import Counter
import gzip
import itertools
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from src.models.graph_features import build_graph_spec, flat_state_to_node_features
from src.rl.action_projection import quantize_facility_net_specimen_actions_tensor
from src.utils.research_archive import sha256_file


INVENTORY_SHA = "29ddb322f5da8b003f68008cf8fba95c2cc8489e30f71f0b9bfd27b2cf38bb35"
STEPS = (0, 13, 26, 39)


def load(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
        return json.load(handle)


def finite(values):
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite saved diagnostic input")
    return result


def pair_diagnostic(draw_differences, prediction_difference):
    values = finite(draw_differences)
    if values.shape != (8,) or not math.isfinite(prediction_difference):
        raise ValueError("Expected eight paired draws and a finite prediction")
    means = [float(values[:4].mean()), float(values[4:].mean())]
    signs = np.sign(means)
    if signs[0] == 0 and signs[1] == 0:
        group = "both_tied"
    elif 0 in signs:
        group = "one_tied"
    elif signs[0] == signs[1]:
        group = "same_sign"
    else:
        group = "opposite_sign"
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(8))
    return {"halves": means, "group": group, "mean_advantage_difference": mean,
            "paired_se": se, "abs_mean_gt_two_se": abs(mean) > 2 * se,
            "comparable": mean != 0, "correct": mean * prediction_difference > 0,
            "predicted_tie": prediction_difference == 0,
            "prediction_difference": float(prediction_difference)}


def pair_summary(pairs):
    groups = {}
    for group in ("all", "same_sign", "opposite_sign", "both_tied", "one_tied"):
        rows = pairs if group == "all" else [p for p in pairs if p["group"] == group]
        comparable = sum(p["comparable"] for p in rows)
        correct = sum(p["correct"] for p in rows)
        groups[group] = {"pairs": len(rows), "comparable": comparable, "correct": correct,
                         "accuracy": correct / comparable if comparable else None,
                         "abs_mean_gt_two_se": sum(p["abs_mean_gt_two_se"] for p in rows)}
    return groups


def coverage(train, test, train_steps, test_steps, train_parents, train_ids, test_ids):
    train, test = finite(train), finite(test)
    if (train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]
            or not len(train) or not len(test)
            or any(len(x) != len(train) for x in (train_steps, train_parents, train_ids))
            or any(len(x) != len(test) for x in (test_steps, test_ids))):
        raise ValueError("Feature matrices must have matching widths")
    sd = train.std(axis=0)
    variable = sd > 0
    rows = []
    train_steps = np.asarray(train_steps)
    parents = np.asarray(train_parents)
    for role, values, times, names in (("train", train, train_steps, train_ids),
                                      ("test", test, test_steps, test_ids)):
        for i, (value, step, name) in enumerate(zip(values, times, names)):
            eligible = train_steps == step
            if role == "train":
                eligible = eligible & (parents != parents[i])
            reference = train[eligible]
            if not len(reference):
                raise ValueError("No disjoint same-time reference parent")
            distances = (np.sqrt(np.mean(((reference[:, variable] - value[variable])
                                          / sd[variable]) ** 2, axis=1))
                         if variable.any() else np.zeros(len(reference)))
            rows.append({"state_id": name, "role": role, "step": int(step),
                         "nearest_distance": float(distances.min()),
                         "reference_parents": int(len(set(parents[eligible]))),
                         "outside_same_time_range": int(((value < reference.min(axis=0))
                                                          | (value > reference.max(axis=0))).sum()),
                         "novel_train_constant_coordinates": int((value[~variable] != train[0, ~variable]).sum())})
    for row in rows:
        if row["role"] == "test":
            controls = [x["nearest_distance"] for x in rows if x["role"] == "train" and x["step"] == row["step"]]
            row["above_max_train_leave_parent_out"] = row["nearest_distance"] > max(controls)
            row["train_leave_parent_out_distance_range"] = [min(controls), max(controls)]
    return {"coordinate_count": train.shape[1], "variable_train_coordinates": int(variable.sum()),
            "train_max_absolute_value": float(np.abs(train).max()),
            "train_positive_sd_range": [float(sd[variable].min()), float(sd[variable].max())] if variable.any() else None,
            "rows": rows}


def verify_sources(root):
    if sha256_file(root / "artifact_inventory.json") != INVENTORY_SHA:
        raise ValueError("Wrong R6 inventory")
    inventory = load(root / "artifact_inventory.json")
    execution = load(root / "execution.json")
    for name, sha in inventory.items():
        if sha256_file(root / name) != sha:
            raise ValueError(f"R6 artifact changed: {name}")
    for name, sha in execution["source_locks"].items():
        if sha256_file(Path(name)) != sha:
            raise ValueError(f"Locked source changed: {name}")
    return {"inventory_sha256": INVENTORY_SHA, "files": len(inventory),
            "source_locks": len(execution["source_locks"]),
            "status_sha256": sha256_file(root / "status.json"),
            "execution_commit": execution["commit"]}


def trace_components(root, record):
    def trace(path):
        with gzip.open(root / path, "rt") as handle:
            return [json.loads(line) for line in handle]
    own = trace(record["trace"])
    rows = own + trace(record["tail_from"])[1:] if record["tail_from"] else own
    totals = Counter()
    for row in rows:
        components = row["cost_components"]
        if not math.isclose(math.fsum(components.values()), row["info"]["cost"], rel_tol=1e-12, abs_tol=1e-5):
            raise ValueError("Per-step components do not reconcile")
        totals.update(components)
    if not math.isclose(math.fsum(totals.values()), record["outcome"]["total_cost"], rel_tol=1e-12, abs_tol=1e-5):
        raise ValueError("Remaining-window components do not reconcile")
    return dict(totals)


def component_contrast(left, right):
    keys = set(left) | set(right)
    return {k: left.get(k, 0.0) - right.get(k, 0.0) for k in sorted(keys)}


def analyze(root):
    before = verify_sources(root)
    records = [json.loads(line) for line in (root / "outcomes.jsonl").read_text().splitlines()]
    lookup = {(r["state_id"], r["draw"], r["action_index"]): r for r in records}
    if len(lookup) != 3456 or len(records) != 3456:
        raise ValueError("Missing/duplicate R6 outcomes")
    all_pairs, all_equivalences, models, choices = [], [], {}, []
    for seed in (60, 61, 62):
        folder = root / f"seed{seed}"
        config = load(folder / "effective_config.json")
        graph = build_graph_spec(config, 561)
        fit = load(folder / "fit_summary.json")
        seal = load(folder / "sealed_test_predictions.json")
        public_features, ids_by_role = {}, {}
        model_pairs = []
        for role, count in (("train", 4), ("test", 2)):
            ids = [f"seed{seed}/{role}{i}/t{t}" for i in range(count) for t in STEPS]
            ids_by_role[role] = ids
            states = [load(root / f"states/{name}.json.gz") for name in ids]
            obs = torch.tensor([s["public"]["observation"] for s in states], dtype=torch.float32)
            with torch.no_grad():
                nodes = flat_state_to_node_features(obs, graph).numpy()
            public_features[role] = nodes.reshape(len(states), -1)
            predictions = finite(fit["training_predictions"] if role == "train" else seal["advantage_predictions"])
            for i, (name, state) in enumerate(zip(ids, states)):
                label = load(root / f"labels/{name}.json")
                support = [a for a, c in enumerate(state["choices"]) if c["map_reachable_certified"]]
                draws = finite([v["advantage_draws"] for v in label["targets"]])
                action_tensor = torch.tensor(state["public"]["requests"], dtype=torch.float32)
                if config.get("specimen_action_quantization", {}).get("enabled", False):
                    action_tensor = quantize_facility_net_specimen_actions_tensor(
                        action_tensor, num_facilities=graph.num_facilities,
                        max_specimen_transfer=config["env"]["max_specimen_transfer"])
                quantized = action_tensor.numpy()
                for a, b in itertools.combinations(support, 2):
                    row = pair_diagnostic(draws[a] - draws[b], predictions[i, a] - predictions[i, b])
                    row.update(state_id=name, role=role, seed=seed, actions=[a, b])
                    model_pairs.append(row)
                    equal = all(lookup[name, d, a]["execution_identity"] == lookup[name, d, b]["execution_identity"] for d in range(8))
                    if equal:
                        if not np.array_equal(draws[a], draws[b]):
                            raise ValueError("Recorded exact execution aliases have unequal values")
                        all_equivalences.append({"state_id": name, "role": role, "actions": [a, b],
                                                 "quantized_request_equal": bool(np.array_equal(quantized[a], quantized[b])),
                                                 "predicted_gap_cost_units": float((predictions[i, a] - predictions[i, b]) / 1e-9)})
                if role == "test":
                    selected = seal["critic_actions"][i]
                    deltas, lost, completed = [], [], []
                    for d in range(8):
                        chosen, frozen = lookup[name, d, selected], lookup[name, d, 0]
                        delta = component_contrast(trace_components(root, chosen), trace_components(root, frozen))
                        total = chosen["outcome"]["total_cost"] - frozen["outcome"]["total_cost"]
                        loss = chosen["outcome"]["patients_lost"] - frozen["outcome"]["patients_lost"]
                        if not math.isclose(math.fsum(delta.values()), total, rel_tol=1e-9, abs_tol=1e-5):
                            raise ValueError("Paired components do not sum to paired cost")
                        if not math.isclose(delta["patient_loss_cost"], loss * config["env"]["weight_patient_lost"], abs_tol=1e-5):
                            raise ValueError("Patient-loss weight mismatch")
                        deltas.append(delta)
                        lost.append(loss)
                        completed.append(chosen["outcome"]["patients_completed"] - frozen["outcome"]["patients_completed"])
                    means = {k: float(np.mean([x[k] for x in deltas])) for k in deltas[0]}
                    total, mean_loss = math.fsum(means.values()), float(np.mean(lost))
                    choices.append({"state_id": name, "action": selected, "component_draw_deltas": deltas,
                                    "mean_component_deltas": means, "mean_total_cost_delta": total,
                                    "mean_patients_lost_delta": mean_loss,
                                    "mean_patients_completed_delta": float(np.mean(completed)),
                                    "cheaper_but_more_loss": total < 0 and mean_loss > 0})
        space = coverage(public_features["train"], public_features["test"], list(STEPS) * 4, list(STEPS) * 2,
                         [i for i in range(4) for _ in STEPS], ids_by_role["train"], ids_by_role["test"])
        all_features = np.concatenate([public_features["train"], public_features["test"]])
        groups = {}
        for name, row in zip(ids_by_role["train"] + ids_by_role["test"], all_features):
            groups.setdefault(row.tobytes(), []).append(name)
        models[str(seed)] = {"label_stability": {role: pair_summary([r for r in model_pairs if r["role"] == role]) for role in ("train", "test")},
                             "coverage": space, "node_feature_dim": graph.node_feature_dim,
                             "normalize_node_features": graph.normalize_node_features,
                             "exact_graph_input_duplicate_groups": [v for v in groups.values() if len(v) > 1]}
        all_pairs.extend(model_pairs)
    after = verify_sources(root)
    if before != after:
        raise ValueError("Source changed during analysis")
    return {"kind": "R6_saved_data_posthoc_diagnosis", "source": before,
            "new_simulator_steps": 0, "new_model_inferences": 0, "new_updates": 0,
            "models": models, "pairs": all_pairs, "sampled_execution_equivalences": all_equivalences,
            "sealed_test_choices": choices,
            "aggregate_label_stability": {role: pair_summary([r for r in all_pairs if r["role"] == role]) for role in ("train", "test")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.root)
    result["analysis_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result["analysis_script_sha256"] = sha256_file(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["aggregate_label_stability"], indent=2))
