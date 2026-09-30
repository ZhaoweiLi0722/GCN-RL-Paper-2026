"""Read-only post-run fit/error accounting; no simulation, fitting or selection."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from src.rl.critic_probe_contract import ranking_counts
from src.rl.frozen_value_probe import write_json
from src.utils.research_archive import sha256_file


def error_accounting(prediction, target, support, scale, standard_error):
    prediction, target, support, standard_error = map(np.asarray, (prediction, target, support, standard_error))
    if (prediction.shape != target.shape or support.shape != target.shape or standard_error.shape != target.shape
            or target.ndim != 2 or not support[:, 0].all() or not np.isfinite(scale) or scale <= 0
            or not all(np.isfinite(x).all() for x in (prediction, target, standard_error))):
        raise ValueError("Invalid saved error-accounting input")
    def average(values):
        return float(np.mean(np.sum(values * support, axis=1) / support.sum(axis=1)))
    return {"normalized_mse": average(((prediction - target) / scale) ** 2),
            "zero_advantage_predictor_mse": average((target / scale) ** 2),
            "estimated_mc_label_mean_variance": average((standard_error / scale) ** 2),
            "variance_caveat": "conditional eight-draw estimate; not identified true prediction error"}


def summarize(root):
    result = {"source_inventory_sha256": sha256_file(root / "artifact_inventory.json"),
              "post_run_descriptive_analysis": True, "new_simulator_steps": 0, "new_updates": 0,
              "models": {}, "test_rows": []}
    for seed in (60, 61, 62):
        directory = root / f"seed{seed}"
        fit = json.loads((directory / "fit_summary.json").read_text())
        seal = json.loads((directory / "sealed_test_predictions.json").read_text())
        model = {}
        for role, n in (("train", 4), ("test", 2)):
            ids = [f"seed{seed}/{role}{i}/t{t}" for i in range(n) for t in (0, 13, 26, 39)]
            states = []
            for name in ids:
                with gzip.open(root / f"states/{name}.json.gz", "rt") as handle:
                    states.append(json.load(handle))
            labels = [json.loads((root / f"labels/{name}.json").read_text()) for name in ids]
            support = np.array([[c["map_reachable_certified"] for c in s["choices"]] for s in states])
            targets = [[t["mean_advantage"] for t in r["targets"]] for r in labels]
            errors = [[t["advantage_mean_se"] for t in r["targets"]] for r in labels]
            prediction = fit["training_predictions"] if role == "train" else seal["advantage_predictions"]
            values = error_accounting(prediction, targets, support, fit["target_scale"], errors)
            ranks = ranking_counts(prediction, [r["mean_costs"] for r in labels], support)
            values["ranking"] = {k: ranks[k] for k in ("correct", "comparable_pairs", "pairwise_accuracy", "label_ties")}
            model[role] = values
        result["models"][str(seed)] = model
        report = json.loads((directory / "test_summary.json").read_text())
        for row in report["states"]:
            result["test_rows"].append({"state_id": row["state_id"], "selected_action": row["selected_action"],
                                        "mean_deltas_vs_frozen": {k: float(np.mean(v)) for k, v in row["contrasts"]["frozen"].items()}})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    write_json(args.output, result)
    print(json.dumps(result["models"], indent=2))
