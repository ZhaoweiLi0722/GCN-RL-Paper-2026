"""Independently verify paired means and crossed bootstrap via count weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def weighted_two_way(diff: np.ndarray, *, resamples: int, seed: int) -> tuple[float, float]:
    """Match the audit RNG schedule, using multiplicities, not array reslicing."""
    ns, nk, nr = diff.shape
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        rng.integers(nr, size=(nk, nr))
    rng.integers(ns, size=(resamples, ns))
    samples = []
    for _ in range(resamples):
        seed_counts = np.bincount(rng.integers(ns, size=ns), minlength=ns)
        world_draws = rng.integers(nr, size=(nk, nr))
        world_counts = np.array([np.bincount(row, minlength=nr) for row in world_draws])
        samples.append(float(np.einsum("s,skr,kr->", seed_counts, diff, world_counts) / (ns * nk * nr)))
    return tuple(float(x) for x in np.percentile(samples, [2.5, 97.5]))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> dict:
    report_path = root / "formal_crossed_bootstrap.json"
    report = json.loads(report_path.read_text())
    if report["alpha"] != .05 or report["resamples"] != 20000 or report["metric"] != "total_cost":
        raise ValueError("unexpected analysis settings")
    projection = root / "projection"
    provenance = json.loads((projection / "provenance.json").read_text())
    if provenance["rows"] != 16000 or provenance["csv_files"] != 40:
        raise ValueError("unexpected projection inventory")
    tables = {}
    for file in provenance["files"]:
        path = projection / file["projection"]
        if digest(path) != file["projection_sha256"]:
            raise ValueError("projection hash mismatch")
        phase, algorithm, _, filename = Path(file["projection"]).parts
        table = tables.setdefault((phase, algorithm, filename), {})
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["training_seed"]), row["scenario"], int(row["replication"]))
                if key in table:
                    raise ValueError("duplicate key")
                table[key] = float(row["total_cost"])
    scenarios = report["scenarios"]
    keys = [(s, k, r) for s in range(10, 15) for k in scenarios for r in range(100)]
    if any(set(table) != set(keys) for table in tables.values()):
        raise ValueError("unbalanced projection")
    gcn = "gcn_residual_mdl2_network_ddpg_afd"
    flat = "flat_residual_mdl2_network_ddpg_afd"
    learned, anchor = "holdout_rows.csv", "holdout_anchor_rows.csv"
    pairs = {
        f"{gcn}_vs_anchor": (("formal_final", gcn, learned), ("formal_final", gcn, anchor)),
        f"{flat}_vs_anchor": (("formal_final", flat, learned), ("formal_final", flat, anchor)),
        f"{gcn}_vs_{flat}": (("formal_final", gcn, learned), ("formal_final", flat, learned)),
        f"{gcn}_final_vs_frozen": (("formal_final", gcn, learned), ("formal_pretrain", gcn, learned)),
        f"{flat}_final_vs_frozen": (("formal_final", flat, learned), ("formal_pretrain", flat, learned)),
    }
    if set(report["contrasts"]) != set(pairs):
        raise ValueError("unexpected contrasts")
    checked = {}
    for name, (candidate, baseline) in pairs.items():
        values = [tables[candidate][key] - tables[baseline][key] for key in keys]
        mean = math.fsum(values) / len(values)
        contrast = report["contrasts"][name]
        if not math.isclose(mean, contrast["mean_difference"], rel_tol=0, abs_tol=1e-6):
            raise ValueError("paired mean mismatch")
        low, high = weighted_two_way(np.array(values).reshape(5, 4, 100), resamples=20000, seed=0)
        interval = contrast["intervals"]["two_way"]
        errors = [abs(low - interval["ci_low"]), abs(high - interval["ci_high"])]
        if max(errors) > 1e-6:
            raise ValueError("independent bootstrap mismatch")
        checked[name] = {"mean_difference": mean, "two_way_max_endpoint_error": max(errors),
                         "two_way_width_increase_vs_seed0_nested_pct":
                             100 * (interval["width"] / contrast["intervals"]["nested_seed_then_row"]["width"] - 1)}
    return {"status": "passed", "method": "independent paired-key arithmetic and multiplicity-weighted crossed resampling",
            "bootstrap_seed": 0, "resamples": 20000, "alpha": .05,
            "contrasts": checked, "projection_csv_hashes_verified": 40,
            "input_hashes": {str(p.relative_to(root)): digest(p) for p in
                             [report_path, root / "formal_crossed_bootstrap.md", projection / "provenance.json"]},
            "verification_source_sha256": digest(Path(__file__)),
            "scope_limit": "checks arithmetic and crossed implementation, not finite-sample coverage or clinical claims"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite verification")
    report = verify(args.root)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "contrasts_verified": len(report["contrasts"])}))


if __name__ == "__main__":
    main()
