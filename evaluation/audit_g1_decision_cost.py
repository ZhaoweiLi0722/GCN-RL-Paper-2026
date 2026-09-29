"""Read immutable G1 costs/predictions; no model, fitting or environment calls."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
import subprocess

import numpy as np

from evaluation.audit_reward_objective_bridge import close, read_json, sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/g1_decision_cost_20260929.json"
STREAMS = ("discovery", "validation")


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite archived value")
    return result


def indexed(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError("duplicate row key")
        result[key] = row
    return result


def validate_tables(cost_rows, prediction_rows, arrays, old):
    """Validate joins and recorded contracts without reconstructing a simulator."""
    data = old["dataset"]
    seeds, steps = old["training_seeds"], data["decision_steps"]
    horizons = [str(h) for h in data["horizons"]]
    labels = ["mdl2"] + [f'{x["group"]}:{x["sign"]:+g}x{x["epsilon"]:g}'
                         for x in data["explicit_options"]]
    n, a = len(seeds) * len(steps), len(labels)
    required = {"states", "actions", "training_seeds", "steps", "horizons", "labels",
                "discovery_advantages", "validation_advantages"}
    if set(arrays) != required:
        raise ValueError("unexpected NPZ schema")
    for key in required - {"horizons", "labels"}:
        if not np.isfinite(arrays[key]).all():
            raise ValueError("nonfinite NPZ value")
    if (arrays["states"].ndim != 2 or arrays["states"].shape[0] != n
            or arrays["actions"].ndim != 3 or arrays["actions"].shape[:2] != (n, a)
            or arrays["training_seeds"].shape != (n,) or arrays["steps"].shape != (n,)
            or arrays["training_seeds"].dtype.kind not in "iu"
            or arrays["steps"].dtype.kind not in "iu"
            or arrays["labels"].tolist() != labels or arrays["horizons"].tolist() != horizons
            or any(arrays[s + "_advantages"].shape != (n, len(horizons), a) for s in STREAMS)):
        raise ValueError("NPZ shape/labels/identity contract")
    identities = list(zip(arrays["training_seeds"].tolist(), arrays["steps"].tolist()))
    if len(set(identities)) != n or set(identities) != {(s, t) for s in seeds for t in steps}:
        raise ValueError("missing or duplicated NPZ state")
    costs = indexed(cost_rows, ("training_seed", "step", "configured_horizon", "candidate_index"))
    preds = indexed(prediction_rows, ("held_out_seed", "step", "stream", "candidate_index"))
    expected_c = {(str(s), str(t), h, str(i)) for s in seeds for t in steps
                  for h in horizons for i in range(a)}
    expected_p = {(str(s), str(t), stream, str(i)) for s in seeds for t in steps
                  for stream in STREAMS for i in range(a)}
    if set(costs) != expected_c or set(preds) != expected_p:
        raise ValueError("missing or unexpected table row")
    records = []
    for index, (seed, step) in enumerate(identities):
        record = {"seed": seed, "step": step, "costs": {}, "advantages": {}}
        scenario = old["scenario_by_training_seed"][str(seed)]
        record["scenario_label_as_recorded"] = scenario
        action_ids = None
        for hi, horizon in enumerate(horizons):
            rows = [costs[(str(seed), str(step), horizon, str(i))] for i in range(a)]
            ids = [r["executed_action_id"] for r in rows]
            if len(set(ids)) != a or any(not x for x in ids) or (action_ids is not None and ids != action_ids):
                raise ValueError("executed action identities disagree")
            action_ids = ids
            actual_horizon = data["max_steps_per_episode"] - step
            if horizon != "remaining":
                actual_horizon = min(int(horizon), actual_horizon)
            for i, row in enumerate(rows):
                if (row["scenario"] != scenario or row["candidate_label"] != labels[i]
                        or int(row["horizon"]) != actual_horizon):
                    raise ValueError("recorded scenario/label/horizon mismatch")
            record["costs"][horizon], record["advantages"][horizon] = {}, {}
            for stream in STREAMS:
                values = np.asarray([number(row[stream + "_total_cost"]) for row in rows])
                advantages = values[0] - values
                for i, row in enumerate(rows):
                    start = data[stream + "_rollout_seed"] + seeds.index(seed) * 1_000_000 + step * 100 + hi * 10
                    if (int(row[stream + "_seed_start"]) != start
                            or int(row[stream + "_replications"]) != data[stream + "_replications"]):
                        raise ValueError("CRN or replication mismatch")
                    close(number(row[stream + "_cost_advantage"]), float(advantages[i]), "cost/advantage mismatch")
                    close(float(arrays[stream + "_advantages"][index, hi, i]), float(advantages[i]), "NPZ/CSV mismatch")
                record["costs"][horizon][stream] = values
                record["advantages"][horizon][stream] = advantages
        for stream in STREAMS:
            prediction_values = {"baseline": [], "fitted": []}
            for i in range(a):
                row = preds[(str(seed), str(step), stream, str(i))]
                training = [int(x) for x in row["training_seeds"].split("|")]
                if len(training) != len(seeds) - 1 or set(training) != set(seeds) - {seed}:
                    raise ValueError("held-out fold contamination")
                if (row["scenario"] != scenario or row["candidate_label"] != labels[i]
                        or row["configured_horizon"] != data["primary_horizon"]):
                    raise ValueError("prediction scenario/label/horizon mismatch")
                close(number(row["true_cost_advantage"]),
                      float(record["advantages"][data["primary_horizon"]][stream][i]),
                      "prediction true label mismatch")
                for name in prediction_values:
                    prediction_values[name].append(number(row[name + "_q_advantage"]))
            if stream == "discovery":
                record["predictions"] = prediction_values
            elif prediction_values != record["predictions"]:
                raise ValueError("critic predictions differ between streams")
        if any(values[0] != 0.0 for values in record["predictions"].values()):
            raise ValueError("critic advantage reference is not zero")
        records.append(record)
    return sorted(records, key=lambda r: (r["seed"], r["step"]))


def ranking_metrics(records, name, stream, horizon, gap, material, atol):
    top, pair, selected, headroom = [], [], [], []
    for row in records:
        q = row["predictions"][name]
        adv = row["advantages"][horizon][stream]
        choice = int(np.argmax(q))
        top.append(abs(float(adv[choice] - max(adv))) <= atol)
        selected.append(adv[choice] >= material)
        headroom.append(max(adv) >= material)
        for left, right in combinations(range(len(q)), 2):
            delta = float(adv[left] - adv[right])
            if abs(delta) >= gap:
                product = delta * (q[left] - q[right])
                pair.append(1.0 if product > 0 else 0.5 if product == 0 else 0.0)
    return {"states": len(records), "top1_accuracy": float(np.mean(top)),
            "pairwise_accuracy": statistics.mean(pair) if pair else None,
            "pairwise_comparisons": len(pair),
            "selected_material_improvement_fraction": float(np.mean(selected)),
            "material_headroom_fraction": float(np.mean(headroom))}


def stability(records, horizon, gap, per_seed=True):
    agreements, pairs = [], []
    for row in records:
        discovery, validation = (row["advantages"][horizon][s] for s in STREAMS)
        agreements.append(bool(set(np.flatnonzero(discovery == max(discovery))) &
                               set(np.flatnonzero(validation == max(validation)))))
        for left, right in combinations(range(len(discovery)), 2):
            d, v = discovery[left] - discovery[right], validation[left] - validation[right]
            if min(abs(d), abs(v)) >= gap:
                pairs.append(bool(d * v > 0))
    return {"states": len(records), "best_action_agreement": statistics.mean(agreements),
            "pairwise_sign_agreement": statistics.mean(pairs) if pairs else None,
            "pairwise_comparisons": len(pairs),
            "per_seed": {str(seed): stability([r for r in records if r["seed"] == seed], horizon, gap, False)
                         for seed in sorted({r["seed"] for r in records})} if per_seed else {}}


def compare_metrics(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("historical metric keys differ")
        for key in expected:
            compare_metrics(actual[key], expected[key])
    elif expected is None:
        if actual is not None:
            raise ValueError("historical null metric differs")
    else:
        close(actual, expected, "historical metric value differs")


def describe(values, atol):
    values = [float(x) for x in values]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "negative": sum(x < -atol for x in values), "tied": sum(abs(x) <= atol for x in values),
            "positive": sum(x > atol for x in values)}


def bins(values, boundaries, atol):
    lo, hi = boundaries
    return {"within_top1_tolerance": sum(x <= atol for x in values),
            "above_tolerance_to_250000": sum(atol < x <= lo for x in values),
            "above_250000_to_1000000": sum(lo < x <= hi for x in values),
            "above_1000000": sum(x > hi for x in values)}


def decisions(records, horizon, atol):
    result = []
    for record in records:
        d, v = (record["costs"][horizon][s] for s in STREAMS)
        chosen = {"mdl2": 0, "frozen_critic": int(np.argmax(record["predictions"]["baseline"])),
                  "fitted_offline_critic": int(np.argmax(record["predictions"]["fitted"])),
                  "discovery_cost_choice": int(np.argmin(d))}
        result.append({"seed": record["seed"], "step": record["step"],
                       "scenario_label_as_recorded": record["scenario_label_as_recorded"],
                       "validation_costs": v.tolist(), "discovery_costs": d.tolist(),
                       "validation_minimum_not_oracle": float(min(v)),
                       "label_best_sets_agree": bool(set(np.flatnonzero(d == min(d))) &
                                                     set(np.flatnonzero(v == min(v)))),
                       "selectors": {name: {"action_index": index, "validation_cost": float(v[index]),
                                            "cost_minus_mdl2": float(v[index] - v[0]),
                                            "excess_over_validation_minimum": float(v[index] - min(v)),
                                            "validation_top1": abs(float(v[index] - min(v))) <= atol}
                                     for name, index in chosen.items()},
                       "fitted_minus_frozen_cost": float(v[chosen["fitted_offline_critic"]] - v[chosen["frozen_critic"]])})
    return result


def summarize(rows, boundaries, atol):
    selectors = {}
    for name in rows[0]["selectors"]:
        entries = [r["selectors"][name] for r in rows]
        excess = [r["excess_over_validation_minimum"] for r in entries]
        errors = [r["excess_over_validation_minimum"] for r in entries if not r["validation_top1"]]
        selectors[name] = {"validation_cost": describe([r["validation_cost"] for r in entries], atol),
                           "cost_minus_mdl2": describe([r["cost_minus_mdl2"] for r in entries], atol),
                           "excess_over_validation_minimum": describe(excess, atol),
                           "validation_top1_count": sum(r["validation_top1"] for r in entries),
                           "excess_bins": bins(excess, boundaries, atol),
                           "top1_error_excess_bins": bins(errors, boundaries, atol)}
    disagreement = [r["selectors"]["discovery_cost_choice"]["excess_over_validation_minimum"]
                    for r in rows if not r["label_best_sets_agree"]]
    return {"states": len(rows), "selectors": selectors,
            "fitted_minus_frozen_cost": describe([r["fitted_minus_frozen_cost"] for r in rows], atol),
            "label_disagreement_excess": describe(disagreement, atol),
            "label_disagreement_excess_bins": bins(disagreement, boundaries, atol)}


def audit(config, root=ROOT):
    paths, before = {}, {}
    for key, entry in config["inputs"].items():
        paths[key] = root / entry["path"]
        before[entry["path"]] = sha256(paths[key])
        if before[entry["path"]] != entry["sha256"]:
            raise ValueError("input hash mismatch: " + key)
    old, summary = (read_json(paths[k]) for k in ("original_config", "summary"))
    horizon, atol, bounds = config["primary_horizon"], config["top1_atol"], config["excess_cost_boundaries"]
    gap, material = old["ranker"]["pairwise_minimum_cost_gap"], old["dataset"]["material_improvement"]
    if (horizon != old["dataset"]["primary_horizon"] or horizon != "remaining"
            or bounds != [gap, material] or bounds != [250000.0, 1000000.0] or atol != 1e-9
            or summary["config_sha256"] != sha256(paths["original_config"])):
        raise ValueError("primary metric/reporting contract changed")
    if (summary["actor_updated"] or summary["fitted_checkpoint_saved"]
            or summary["formal_holdout_reused"]):
        raise ValueError("historical scope changed")
    with paths["cost_rows"].open(newline="") as f:
        cost_rows = list(csv.DictReader(f))
    with paths["prediction_rows"].open(newline="") as f:
        prediction_rows = list(csv.DictReader(f))
    with np.load(paths["arrays"], allow_pickle=False) as archive:
        records = validate_tables(cost_rows, prediction_rows, dict(archive), old)
    if len(cost_rows) != summary["dataset"]["rows"] or len(records) != summary["dataset"]["states"]:
        raise ValueError("historical dataset counts differ")
    metrics = {s: {name: ranking_metrics(records, name, s, horizon, gap, material, atol)
                   for name in ("baseline", "fitted")} for s in STREAMS}
    compare_metrics(metrics, summary["aggregate"])
    for seed in old["training_seeds"]:
        subset = [r for r in records if r["seed"] == seed]
        per_seed = {name: ranking_metrics(subset, name, "validation", horizon, gap, material, atol)
                    for name in ("baseline", "fitted")}
        observed = summary["decision"]["observed"]
        close(per_seed["fitted"]["top1_accuracy"], observed["per_seed_validation_top1_accuracy"][str(seed)],
              "per-seed validation accuracy differs")
        close(per_seed["fitted"]["top1_accuracy"] - per_seed["baseline"]["top1_accuracy"],
              observed["per_seed_validation_top1_gain"][str(seed)], "per-seed gain differs")
    replicated = stability(records, horizon, gap)
    compare_metrics(replicated, summary["label_stability"])
    rows = decisions(records, horizon, atol)
    for path, digest in before.items():
        if sha256(root / path) != digest:
            raise ValueError("input changed during audit")
    return {"kind": config["kind"], "integrity_passed": True, "input_sha256": before,
            "cost_rows_checked": len(cost_rows), "prediction_rows_checked": len(prediction_rows),
            "effective_scenario_interpretation": "nominal_history_per_published_correction",
            "original_decision_unchanged": summary["decision"], "reproduced_metrics": metrics,
            "reproduced_label_stability": replicated, "primary_horizon": horizon,
            "summary": summarize(rows, bounds, atol),
            "per_seed": {str(seed): summarize([r for r in rows if r["seed"] == seed], bounds, atol)
                         for seed in old["training_seeds"]}, "decisions": rows,
            "cost_unit": "original_simulator_cost_not_clinically_calibrated_currency",
            "uncertainty": "replication-level costs absent; overlapping states; no CI or significance claim",
            "interpretation": "one-action intervention then archived continuation; not selector deployment or online RL",
            "validation_minimum_is_optimistic_not_true_optimum": True,
            "environment_queries": 0, "optimizer_updates": 0, "scientific_launch_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the existing audit")
    result = audit(read_json(args.config))
    result["auditor_sha256"], result["config_sha256"] = sha256(Path(__file__)), sha256(args.config)
    result["execution_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"integrity_passed": True, "states": len(result["decisions"]),
                      "output": str(args.output), "sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
