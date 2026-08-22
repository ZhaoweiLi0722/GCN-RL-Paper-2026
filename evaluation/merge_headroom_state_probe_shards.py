"""Merge deterministic headroom state-probe shards in canonical order."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from pathlib import Path
from typing import Any

from evaluation.network_residual_headroom import (
    headroom_decision,
    smoke_config,
    state_probe_shard_config,
    summarize_state_probe,
)
from src.rl.config import load_config
from src.rl.experiment import write_rows


SHARD_PATTERN = re.compile(r"^shard_(\d+)_of_(\d+)$")


def discover_state_probe_shards(shards_root: Path) -> tuple[list[Path], int]:
    indexed: dict[int, Path] = {}
    shard_count: int | None = None
    for path in sorted(shards_root.glob("shard_*_of_*")):
        match = SHARD_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        index, count = (int(value) for value in match.groups())
        if shard_count is None:
            shard_count = count
        elif count != shard_count:
            raise ValueError("State-probe shard directories disagree on shard count")
        if index in indexed:
            raise ValueError(f"Duplicate state-probe shard index {index}")
        indexed[index] = path
    if shard_count is None:
        raise ValueError("Could not identify any state-probe shards")
    expected = set(range(shard_count))
    if set(indexed) != expected:
        raise ValueError(
            "State-probe shard set is incomplete: "
            f"expected={sorted(expected)} found={sorted(indexed)}"
        )
    return [indexed[index] for index in range(shard_count)], shard_count


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def merge_state_probe_rows(
    shard_rows: list[list[dict[str, Any]]],
    *,
    total_rollouts: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    rows = [copy.deepcopy(row) for group in shard_rows for row in group]
    rows.sort(key=lambda row: (int(row["rollout"]), int(row["step"])))
    keys = [(int(row["rollout"]), int(row["step"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("State-probe shards contain duplicate rollout/step rows")
    by_rollout: dict[int, list[int]] = {}
    for rollout, step in keys:
        if not 0 <= rollout < int(total_rollouts):
            raise ValueError(f"State-probe rollout is out of range: {rollout}")
        if not 0 <= step < int(max_steps):
            raise ValueError(f"State-probe step is out of range: {step}")
        by_rollout.setdefault(rollout, []).append(step)
    expected_rollouts = set(range(int(total_rollouts)))
    if set(by_rollout) != expected_rollouts:
        raise ValueError(
            "State-probe shards do not cover every rollout: "
            f"expected={sorted(expected_rollouts)} "
            f"found={sorted(by_rollout)}"
        )
    for rollout, steps in by_rollout.items():
        if steps != list(range(len(steps))):
            raise ValueError(
                f"State-probe rollout {rollout} has non-contiguous steps: {steps}"
            )
    return rows


def atomic_write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    write_rows(rows, temporary)
    temporary.replace(path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shards-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke:
        config = smoke_config(config)
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    output_root = Path(config["output_root"])
    shards_root = (
        Path(args.shards_root)
        if args.shards_root
        else output_root / "state_probe_shards"
    )
    shard_dirs, shard_count = discover_state_probe_shards(shards_root)
    shard_rows = []
    for index, path in enumerate(shard_dirs):
        result = json.loads((path / "summary.json").read_text())
        expected_config = state_probe_shard_config(config, index, shard_count)
        if result.get("config") != expected_config:
            raise ValueError(f"State-probe shard {index} config does not match")
        shard_rows.append(read_csv_rows(path / "state_probe.csv"))

    rows = merge_state_probe_rows(
        shard_rows,
        total_rollouts=int(config["state_probe_rollouts"]),
        max_steps=int(config["max_steps"]),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_rows(rows, output_root / "state_probe.csv")

    previous_path = output_root / "summary.json"
    previous = (
        json.loads(previous_path.read_text())
        if previous_path.exists()
        else {}
    )
    result: dict[str, Any] = {
        "name": config.get("name", "network_residual_headroom"),
        "config": config,
        "state_probe": summarize_state_probe(rows, config),
    }
    if "online_teacher" in previous:
        result["online_teacher"] = previous["online_teacher"]
    result["state_probe_shards"] = {
        "count": shard_count,
        "directories": [str(path) for path in shard_dirs],
        "states": len(rows),
    }
    result["decision"] = headroom_decision(result, config)
    atomic_write_json(result, previous_path)
    print(
        f"merged {shard_count} state-probe shards into {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
