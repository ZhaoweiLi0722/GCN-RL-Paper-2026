"""Prepare zero-trajectory source states for the Stage C2 DDPG pairs."""

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
SEEDS = (40, 41, 42)
RUN_NAME = "patient_indexed_specimen_routing_ddpg_persistent_shift_preonline"
DEFAULT_OUTPUT_ROOT = Path(
    "results/patient_indexed_specimen_routing_ddpg_persistent_shift_"
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
        raise ValueError("Stage C2 source control must lock 100 episodes")
    if output_root.exists():
        raise FileExistsError(output_root)
    latest: dict[str, Any] | None = None
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            config = copy.deepcopy(base)
            config["name"] = RUN_NAME
            config["experimental_role"] = (
                "zero-trajectory source state shared by standard and "
                "critic-realigned persistent-shift DDPG"
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
            overrides = dict(config["config_overrides"])
            calibration = dict(
                overrides["critic_teacher_advantage_calibration"]
            )
            calibration["updates"] = 0
            overrides["critic_teacher_advantage_calibration"] = calibration
            overrides["preonline_training_state_path"] = str(state_path)
            config["config_overrides"] = overrides
            latest = train_multiscenario_agents(
                config,
                algorithm_filter=algorithm,
                seed_filter=seed,
            )
    if latest is None:
        raise RuntimeError("No Stage C2 pre-online state was prepared")
    return latest


if __name__ == "__main__":
    main()
