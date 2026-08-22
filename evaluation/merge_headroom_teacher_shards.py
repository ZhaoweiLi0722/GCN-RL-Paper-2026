"""Merge independently seeded headroom-teacher shards into one formal result."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.aggregate_stats import paired_two_level_summary
from evaluation.evaluate_formal import summarize_rows
from evaluation.network_residual_headroom import (
    headroom_decision,
    smoke_config,
    teacher_shard_config,
)
from evaluation.run_gcn_residual_sweep import (
    load_local_search_demonstrations,
    save_local_search_demonstrations,
)
from src.rl.config import load_config
from src.rl.experiment import write_rows


SHARD_PATTERN = re.compile(r"^shard_(\d+)_of_(\d+)$")
CSV_FIELD_SIZE_LIMIT = 64 * 1024 * 1024


def carry_forward_state_probe(
    result: dict[str, Any],
    previous: dict[str, Any],
) -> None:
    for key in ("state_probe", "state_probe_shards"):
        if key in previous:
            result[key] = previous[key]


def discover_teacher_shards(shards_root: Path) -> tuple[list[Path], int]:
    indexed: dict[int, Path] = {}
    shard_count: int | None = None
    for path in sorted(shards_root.glob("shard_*_of_*")):
        match = SHARD_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        index, count = (int(value) for value in match.groups())
        if count <= 0 or not 0 <= index < count:
            raise ValueError(f"Invalid teacher shard directory: {path.name}")
        if shard_count is None:
            shard_count = count
        elif count != shard_count:
            raise ValueError("Teacher shard directories disagree on shard count")
        if index in indexed:
            raise ValueError(f"Duplicate teacher shard index {index}")
        indexed[index] = path
    if shard_count is None:
        raise ValueError("Could not identify any teacher shards")
    expected = set(range(shard_count))
    if set(indexed) != expected:
        raise ValueError(
            "Teacher shard set is incomplete: "
            f"expected={sorted(expected)} found={sorted(indexed)}"
        )
    return [indexed[index] for index in range(shard_count)], shard_count


def validate_teacher_shard_result(
    config: dict[str, Any],
    result: dict[str, Any],
    *,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    expected_config = teacher_shard_config(config, shard_index, shard_count)
    if result.get("config") != expected_config:
        raise ValueError(f"Teacher shard {shard_index} config does not match")
    teacher = result.get("online_teacher")
    if not isinstance(teacher, dict):
        raise ValueError(f"Teacher shard {shard_index} has no online_teacher")
    expected_start = int(expected_config["teacher_replication_start"])
    if int(teacher.get("teacher_replication_start", -1)) != expected_start:
        raise ValueError(f"Teacher shard {shard_index} start does not match")
    expected_offset = int(expected_config["lookahead_decision_offset"])
    if int(teacher.get("lookahead_decision_offset", -1)) != expected_offset:
        raise ValueError(f"Teacher shard {shard_index} offset does not match")
    return expected_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shards-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--demonstration-path", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke:
        config = smoke_config(config)
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    if args.demonstration_path is not None:
        config["demonstration_path"] = str(args.demonstration_path)
    output_root = Path(config["output_root"])
    shards_root = (
        Path(args.shards_root)
        if args.shards_root
        else output_root / "shards"
    )
    shard_dirs, shard_count = discover_teacher_shards(shards_root)
    shard_results = []
    shard_configs = []
    for index, path in enumerate(shard_dirs):
        result = json.loads((path / "summary.json").read_text())
        shard_configs.append(
            validate_teacher_shard_result(
                config,
                result,
                shard_index=index,
                shard_count=shard_count,
            )
        )
        shard_results.append(result)
    caches = [
        load_local_search_demonstrations(path / "teacher_cache.npz")
        for path in shard_dirs
    ]
    merged_cache = merge_demonstration_caches(caches)

    anchor_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    base_evaluation_seed = int(config["seed"]) + 10000
    for path, shard_config in zip(shard_dirs, shard_configs):
        start = int(shard_config["teacher_replication_start"])
        count = int(shard_config["teacher_replications"])
        anchor_rows.extend(
            normalized_shard_rows(
                read_csv_rows(path / "anchor.csv"),
                start=start,
                count=count,
                base_evaluation_seed=base_evaluation_seed,
            )
        )
        teacher_rows.extend(
            normalized_shard_rows(
                read_csv_rows(path / "teacher.csv"),
                start=start,
                count=count,
                base_evaluation_seed=base_evaluation_seed,
            )
        )
    anchor_rows.sort(key=lambda row: int(row["replication"]))
    teacher_rows.sort(key=lambda row: int(row["replication"]))
    total_replications = int(config["teacher_replications"])
    assert_canonical_replications(
        anchor_rows,
        total_replications=total_replications,
        label="anchor",
    )
    assert_canonical_replications(
        teacher_rows,
        total_replications=total_replications,
        label="teacher",
    )

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
    demonstration_path = Path(config["demonstration_path"])
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
        "teacher_shards": {
            "count": shard_count,
            "directories": [str(path) for path in shard_dirs],
            "replications": total_replications,
        },
    }
    carry_forward_state_probe(result, previous)
    result["decision"] = headroom_decision(result, config)

    output_root.mkdir(parents=True, exist_ok=True)
    atomic_save_demonstrations(merged_cache, demonstration_path)
    atomic_write_rows(anchor_rows, output_root / "anchor.csv")
    atomic_write_rows(teacher_rows, output_root / "teacher.csv")
    atomic_write_json(result, previous_path)
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
    csv.field_size_limit(max(csv.field_size_limit(), CSV_FIELD_SIZE_LIMIT))
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_shard_rows(
    rows: list[dict[str, Any]],
    *,
    start: int,
    count: int,
    base_evaluation_seed: int,
) -> list[dict[str, Any]]:
    local_replications = [int(row["replication"]) for row in rows]
    if sorted(local_replications) != list(range(int(count))):
        raise ValueError(
            "Teacher shard rows have invalid local replications: "
            f"expected={list(range(int(count)))} "
            f"found={sorted(local_replications)}"
        )
    normalized = []
    for row in rows:
        item = copy.deepcopy(row)
        item["replication"] = start + int(item["replication"])
        item["seed"] = base_evaluation_seed
        item["training_seed"] = 0
        item["evaluation_seed"] = base_evaluation_seed
        normalized.append(item)
    return normalized


def assert_canonical_replications(
    rows: list[dict[str, Any]],
    *,
    total_replications: int,
    label: str,
) -> None:
    replications = [int(row["replication"]) for row in rows]
    expected = list(range(int(total_replications)))
    if replications != expected:
        raise ValueError(
            f"Merged {label} replications are not canonical: "
            f"expected={expected} found={replications}"
        )


def atomic_save_demonstrations(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    save_local_search_demonstrations(temporary, payload)
    temporary.replace(path)


def atomic_write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    write_rows(rows, temporary)
    temporary.replace(path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
