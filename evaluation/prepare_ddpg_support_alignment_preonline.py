"""Create paired episode-0 DDPG states without changing shared trainers."""

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
SEEDS = (30, 31, 32)
DEFAULT_OUTPUT_ROOT = Path(
    "results/patient_indexed_specimen_routing_ddpg_support_alignment_"
    "development/preonline"
)
RUN_NAME = "patient_indexed_specimen_routing_ddpg_support_alignment_preonline"


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
    """Pretrain six matched agents and freeze each complete episode-0 state."""

    base = load_config(config_path)
    if int(base["online_episodes"]) != 100:
        raise ValueError("Source control must lock 100 online episodes")
    if output_root.exists():
        raise FileExistsError(output_root)
    latest: dict[str, Any] | None = None
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            config = copy.deepcopy(base)
            config["name"] = RUN_NAME
            config["experimental_role"] = (
                "episode-0 source state for the paired DDPG support-alignment "
                "development experiment"
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
            overrides = dict(config.get("config_overrides", {}))
            calibration = dict(
                overrides["critic_teacher_advantage_calibration"]
            )
            calibration["enabled"] = False
            calibration["online_ranking_weight"] = 0.0
            calibration.pop("allowed_option_groups", None)
            overrides["critic_teacher_advantage_calibration"] = calibration
            overrides["preonline_training_state_path"] = str(state_path)
            config["config_overrides"] = overrides
            latest = train_multiscenario_agents(
                config,
                algorithm_filter=algorithm,
                seed_filter=seed,
            )
    if latest is None:
        raise RuntimeError("No pre-online state was prepared")
    return latest


if __name__ == "__main__":
    main()
