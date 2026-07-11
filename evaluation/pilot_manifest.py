"""Manifest for the Phase 7 lean two-stage patient pilot.

Rather than ship ~10 near-duplicate per-algorithm JSON configs, this module
composes each agent's training config at runtime from (a) the existing patient
env configs, (b) light per-algorithm hyperparameters, and (c) a step-budget
helper. The two-stage runner (`run_patient_pilot.py`) consumes it.

Decisions (see specs/2026-07-11-pilot-experiments): lean two-stage, 5 seeds,
~30k train steps, blended reward kept as-is. Graph agents receive the full env
config (their `build_graph_spec` needs it); reward is scaled uniformly so the
ranking is fair.
"""

from __future__ import annotations

import math
from typing import Any

from src.rl.config import load_config

STAGE_A_ENV = "experiments/configs/2_clinic_patient_condition.json"     # screen
STAGE_B_ENV = "experiments/configs/20_clinic_patient_condition.json"    # confirm

SEEDS = (0, 1, 2, 3, 4)
DEFAULT_TRAIN_STEPS = 30_000
DEFAULT_EVAL_REPLICATIONS = 20

HEURISTICS = ("myo", "iso", "mdl1", "mdl2", "fmyo", "umyo")
GRAPH_BACKBONES = ("gcn_ddpg", "gcn_td3", "gcn_sac", "gcn_ppo")
# GNN-DDPG is the ablation; the flagship is chosen from the stable backbones.
FLAGSHIP_CANDIDATES = ("gcn_td3", "gcn_sac", "gcn_ppo")
LEARNED = ("flat_ddpg",) + GRAPH_BACKBONES

# Stage A screens the full roster.
STAGE_A_ROSTER = LEARNED + HEURISTICS
# Stage B confirms the top graph backbone (chosen at runtime) plus these anchors:
# GNN-DDPG (ablation), flat-DDPG (graph-vs-flat), MDL-2 (strong heuristic).
STAGE_B_ANCHORS = ("gcn_ddpg", "flat_ddpg", "mdl2")

# Uniform reward scale (patient costs are ~1e5-1e6/step); applied to every agent
# so comparisons stay fair. Blended reward itself is unchanged.
REWARD_SCALE = 1e-6

# Light per-algorithm hyperparameters; anything omitted falls back to the agent's
# verified in-code defaults. Graph agents share encoder/head sizes.
_COMMON_GRAPH = {"gcn_hidden_sizes": [64, 64], "hidden_sizes": [256, 128], "include_global_context": True}
HYPERPARAMS: dict[str, dict[str, Any]] = {
    "flat_ddpg": {"hidden_sizes": [256, 128], "batch_size": 128},
    "gcn_ddpg": {**_COMMON_GRAPH, "batch_size": 128},
    "gcn_td3": {**_COMMON_GRAPH, "batch_size": 128},
    "gcn_sac": {**_COMMON_GRAPH, "batch_size": 256},
    "gcn_ppo": {**_COMMON_GRAPH, "rollout_length": 1040, "minibatch_size": 64, "train_epochs": 10},
}


def train_budget(env_config_path: str, target_steps: int = DEFAULT_TRAIN_STEPS) -> dict[str, int]:
    """Episodes x horizon >= target_steps, using the env's own episode horizon."""

    horizon = int(load_config(env_config_path)["episode_horizon"])
    num_episodes = max(1, math.ceil(target_steps / horizon))
    return {"num_episodes": num_episodes, "max_steps_per_episode": horizon}


def build_agent_config(
    algorithm: str,
    env_config_path: str,
    *,
    seed: int,
    target_steps: int = DEFAULT_TRAIN_STEPS,
) -> dict[str, Any]:
    """Full training config for one algorithm on one patient env at one seed."""

    env_config = load_config(env_config_path)
    config: dict[str, Any] = {
        "algorithm": algorithm,
        "seed": seed,
        "reward_scale": REWARD_SCALE,
        "normalize_observations": False,  # patient obs has a trailing summary block
        "env": env_config,
    }
    config.update(HYPERPARAMS.get(algorithm, {}))
    if algorithm not in HEURISTICS:
        config.update(train_budget(env_config_path, target_steps))
        config["checkpoint_dir"] = f"checkpoints/pilot/{algorithm}_seed{seed}"
    return config


def stage_a_manifest(target_steps: int = DEFAULT_TRAIN_STEPS) -> dict[str, Any]:
    return {
        "stage": "A",
        "env_config": STAGE_A_ENV,
        "roster": list(STAGE_A_ROSTER),
        "seeds": list(SEEDS),
        "target_steps": target_steps,
    }


def stage_b_roster(top_backbone: str) -> list[str]:
    """Top graph backbone from Stage A + fixed anchors (deduped, order-stable)."""

    roster: list[str] = [top_backbone]
    for anchor in STAGE_B_ANCHORS:
        if anchor not in roster:
            roster.append(anchor)
    return roster
