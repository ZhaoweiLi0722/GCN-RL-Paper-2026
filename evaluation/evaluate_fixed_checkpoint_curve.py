"""Evaluate every prespecified checkpoint on one paired development stream."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluation.evaluate_multiscenario_network_residual import (
    evaluate_multiscenario_agents,
)
from src.rl.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_checkpoint_curve(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    variants = tuple(str(value) for value in config["checkpoint_variants"])
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("Checkpoint curve variants must be unique and nonempty")
    output_root = Path(config["output_root"])
    if output_root.exists():
        raise FileExistsError(output_root)

    summaries = []
    for variant in variants:
        phase_config = copy.deepcopy(config)
        phase_config["name"] = f"{config['name']}__{variant}"
        phase_config["checkpoint_variants"] = [variant]
        phase_config["fixed_checkpoint_variant"] = variant
        phase_config["output_root"] = str(output_root / variant)
        evaluate_multiscenario_agents(phase_config)
        summary_path = output_root / variant / "summary.json"
        summaries.append(
            {
                "checkpoint_variant": variant,
                "summary": str(summary_path),
                "summary_sha256": _sha256(summary_path),
            }
        )

    payload = {
        "config": str(config_path),
        "checkpoint_variants": list(variants),
        "validation_seed": int(config["validation_seed"]),
        "holdout_seed": int(config["holdout_seed"]),
        "holdout_replications": int(config["holdout_replications"]),
        "summaries": summaries,
    }
    output = output_root / "checkpoint_curve_summary.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = evaluate_checkpoint_curve(Path(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
