"""Prepare zero-trajectory source states for the paired Stage F1 DDPG arms."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evaluation.train_multiscenario_network_residual import (
    train_multiscenario_agents,
)
from src.rl.config import load_config


ALGORITHMS = (
    "gcn_residual_mdl2_network_ddpg_afd",
    "flat_residual_mdl2_network_ddpg_afd",
)
SEEDS = (60, 61, 62)
RUN_NAME = (
    "patient_indexed_specimen_routing_ddpg_online_paired_advantage_preonline"
)
DEFAULT_OUTPUT_ROOT = Path(
    "results/patient_indexed_specimen_routing_ddpg_online_paired_advantage_"
    "development/preonline"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    result = prepare_preonline_states(
        Path(args.config),
        output_root=Path(args.output_root),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def prepare_preonline_states(
    config_path: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Build six episode-0 states immediately before online calibration."""

    base = load_config(config_path)
    if int(base["online_episodes"]) != 100:
        raise ValueError("Stage F1 source control must lock 100 episodes")
    if [int(seed) for seed in base["seeds"]] != list(SEEDS):
        raise ValueError("Stage F1 source seeds do not match the lock")
    overrides = dict(base["config_overrides"])
    structured = dict(
        overrides["residual_action"]["structured_exploration"]
    )
    if not bool(structured.get("enabled", False)):
        raise ValueError("Stage F1 source must share structured exploration")
    paired = dict(overrides["online_paired_advantage_critic"])
    if bool(paired.get("enabled", True)):
        raise ValueError("Stage F1 source must use the disabled paired arm")
    if output_root.exists():
        raise FileExistsError(output_root)
    latest: dict[str, Any] | None = None
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            config = copy.deepcopy(base)
            config["name"] = RUN_NAME
            config["experimental_role"] = (
                "zero-trajectory source state shared by Stage F1 control and "
                "paired-advantage candidate"
            )
            config["online_episodes"] = 0
            config["output_root"] = str(output_root)
            state_path = (
                output_root
                / RUN_NAME
                / algorithm
                / f"seed{seed}"
                / "checkpoints"
                / f"{algorithm}_seed{seed}_preonline_training_state.pt"
            )
            run_overrides = dict(config["config_overrides"])
            calibration = dict(
                run_overrides["critic_teacher_advantage_calibration"]
            )
            calibration["updates"] = 0
            run_overrides["critic_teacher_advantage_calibration"] = calibration
            run_overrides["preonline_training_state_path"] = str(state_path)
            config["config_overrides"] = run_overrides
            latest = train_multiscenario_agents(
                config,
                algorithm_filter=algorithm,
                seed_filter=seed,
            )
    if latest is None:
        raise RuntimeError("No Stage F1 pre-online state was prepared")
    return latest


if __name__ == "__main__":
    main()
