"""Merge independently seeded headroom-teacher shards into one formal result."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.aggregate_stats import paired_two_level_summary
from evaluation.evaluate_formal import summarize_rows
from evaluation.network_residual_headroom import (
    headroom_decision,
    smoke_config,
)
from evaluation.run_gcn_residual_sweep import (
    load_local_search_demonstrations,
    save_local_search_demonstrations,
)
from src.rl.config import load_config
from src.rl.experiment import write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shards-root", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke:
        config = smoke_config(config)
    output_root = Path(config["output_root"])
    shards_root = (
        Path(args.shards_root)
        if args.shards_root
        else output_root / "shards"
    )
    shard_dirs = tuple(sorted(shards_root.glob("shard_*_of_*")))
    expected_shards = {
        int(path.name.rsplit("_of_", 1)[1])
        for path in shard_dirs
    }
    if not shard_dirs or len(expected_shards) != 1:
        raise ValueError("Could not identify a complete teacher shard set")
    shard_count = expected_shards.pop()
    if len(shard_dirs) != shard_count:
        raise ValueError(
            f"Expected {shard_count} teacher shards, found {len(shard_dirs)}"
        )

    shard_results = [
        json.loads((path / "summary.json").read_text())
        for path in shard_dirs
    ]
    caches = [
        load_local_search_demonstrations(path / "teacher_cache.npz")
        for path in shard_dirs
    ]
    merged_cache = merge_demonstration_caches(caches)
    demonstration_path = Path(config["demonstration_path"])
    save_local_search_demonstrations(demonstration_path, merged_cache)

    anchor_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    base_evaluation_seed = int(config["seed"]) + 10000
    for path, shard_result in zip(shard_dirs, shard_results):
        start = int(
            shard_result["online_teacher"]["teacher_replication_start"]
        )
        anchor_rows.extend(
            normalized_shard_rows(
                read_csv_rows(path / "anchor.csv"),
                start=start,
                base_evaluation_seed=base_evaluation_seed,
            )
        )
        teacher_rows.extend(
            normalized_shard_rows(
                read_csv_rows(path / "teacher.csv"),
                start=start,
                base_evaluation_seed=base_evaluation_seed,
            )
        )
    anchor_rows.sort(key=lambda row: int(row["replication"]))
    teacher_rows.sort(key=lambda row: int(row["replication"]))
    write_rows(anchor_rows, output_root / "anchor.csv")
    write_rows(teacher_rows, output_root / "teacher.csv")

    paired = {
        metric: paired_two_level_summary(
            teacher_rows,
            anchor_rows,
            metric=metric,
            resamples=20000,
            seed=base_evaluation_seed,
        )
        for metric in (
            "total_cost",
            "completion_service_level",
            "patients_lost",
            "patient_ineligibility_during_manufacturing_rate",
        )
    }
    selected_groups: Counter[str] = Counter()
    total_decisions = 0
    corrected_decisions = 0
    for shard_result in shard_results:
        teacher = shard_result["online_teacher"]
        selected_groups.update(teacher["teacher_selected_group_counts"])
        total_decisions += int(teacher["teacher_total_decisions"])
        corrected_decisions += int(teacher["teacher_corrected_decisions"])
    online_teacher = {
        "anchor_summary": summarize_rows(anchor_rows),
        "teacher_summary": summarize_rows(teacher_rows),
        "paired": paired,
        "teacher_total_decisions": total_decisions,
        "teacher_corrected_decisions": corrected_decisions,
        "teacher_correction_rate": (
            corrected_decisions / total_decisions if total_decisions else 0.0
        ),
        "teacher_selected_group_counts": dict(sorted(selected_groups.items())),
        "teacher_demonstration_path": str(demonstration_path),
        "teacher_demonstration_samples": int(merged_cache["states"].shape[0]),
        "teacher_shard_count": shard_count,
    }
    previous_path = output_root / "summary.json"
    previous = (
        json.loads(previous_path.read_text())
        if previous_path.exists()
        else {}
    )
    result = {
        "name": config.get("name", "network_residual_headroom"),
        "config": config,
        "online_teacher": online_teacher,
    }
    if "state_probe" in previous:
        result["state_probe"] = previous["state_probe"]
    result["decision"] = headroom_decision(result, config)
    previous_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"merged {shard_count} teacher shards into {output_root}", flush=True)


def merge_demonstration_caches(
    caches: list[dict[str, Any]],
) -> dict[str, Any]:
    if not caches:
        raise ValueError("At least one teacher cache is required")
    metadata_keys = ("option_groups", "option_epsilons", "option_signs")
    for cache in caches[1:]:
        for key in metadata_keys:
            matches = (
                np.array_equal(cache[key], caches[0][key])
                if key == "option_groups"
                else np.allclose(
                    cache[key],
                    caches[0][key],
                    rtol=0.0,
                    atol=1e-6,
                )
            )
            if not matches:
                raise ValueError(f"Teacher shard option metadata differs for {key}")
    concatenated_keys = (
        "states",
        "actions",
        "weights",
        "improved_mask",
        "transition_states",
        "transition_actions",
        "transition_rewards",
        "transition_next_states",
        "transition_dones",
        "option_advantages",
        "option_feasible",
    )
    merged = {
        key: np.concatenate([cache[key] for cache in caches], axis=0)
        for key in concatenated_keys
    }
    for key in metadata_keys:
        merged[key] = np.asarray(caches[0][key]).copy()
    merged["improved_steps"] = int(
        sum(int(cache["improved_steps"]) for cache in caches)
    )
    merged["anchor_keep_steps"] = int(
        sum(int(cache["anchor_keep_steps"]) for cache in caches)
    )
    merged["service_rejected_steps"] = int(
        sum(int(cache["service_rejected_steps"]) for cache in caches)
    )
    improved_steps = int(merged["improved_steps"])
    merged["mean_step_improvement"] = (
        sum(
            float(cache["mean_step_improvement"])
            * int(cache["improved_steps"])
            for cache in caches
        )
        / improved_steps
        if improved_steps
        else 0.0
    )
    weights = np.asarray(merged["weights"], dtype=np.float32)
    improved = np.asarray(merged["improved_mask"], dtype=bool)
    total_weight = float(weights.sum())
    merged["improved_weight_fraction"] = (
        float(weights[improved].sum()) / total_weight
        if total_weight > 0.0
        else 0.0
    )
    return merged


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_shard_rows(
    rows: list[dict[str, Any]],
    *,
    start: int,
    base_evaluation_seed: int,
) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        item = copy.deepcopy(row)
        item["replication"] = start + int(item["replication"])
        item["seed"] = base_evaluation_seed
        item["training_seed"] = 0
        item["evaluation_seed"] = base_evaluation_seed
        normalized.append(item)
    return normalized


if __name__ == "__main__":
    main()
