"""Shared experiment helpers for training and evaluation scripts."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.env.capacity_planning import CapacityPlanningConfig, CapacityPlanningEnv
from src.graph.ablation import with_graph_ablation


def build_env(config: dict[str, Any], seed: int) -> CapacityPlanningEnv:
    env_config = dict(config.get("env", {}))
    ablation = env_config.pop("graph_ablation", config.get("graph_ablation", "full_graph"))
    scenario = env_config.pop("scenario_name", config.get("scenario", "default"))
    env_config.pop("supplier_disruption_rate", None)
    env_config.pop("demand_forecast_error", None)
    typed_config = CapacityPlanningConfig(**{key: _to_tuple(value) for key, value in env_config.items()})
    typed_config = apply_graph_ablation(typed_config, ablation)
    env = CapacityPlanningEnv(typed_config, seed=seed)
    env.scenario_name = scenario
    env.graph_ablation = ablation
    return env


def apply_graph_ablation(config: CapacityPlanningConfig, ablation: str) -> CapacityPlanningConfig:
    if ablation in ("full_graph", "", None):
        return config
    if ablation == "no_capacity_sharing_edges":
        return with_graph_ablation(config, remove_capacity_edges=True)
    if ablation == "no_resource_sharing_edges":
        return with_graph_ablation(config, remove_resource_edges=True)
    if ablation == "no_interfacility_edges":
        return with_graph_ablation(
            config,
            remove_specimen_edges=True,
            remove_capacity_edges=True,
            remove_resource_edges=True,
        )
    if ablation == "flat_state_no_graph":
        return config
    raise ValueError(f"Unsupported graph_ablation setting: {ablation}")


def train_off_policy_agent(agent, env: CapacityPlanningEnv, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = int(config.get("seed", 0))
    num_episodes = int(config.get("num_episodes", 1))
    max_steps = int(config.get("max_steps_per_episode", env.config.episode_horizon))
    algorithm = str(config.get("algorithm", agent.algorithm))
    checkpoint_dir = Path(config.get("checkpoint_dir", f"checkpoints/{algorithm}"))
    checkpoint_interval = int(config.get("checkpoint_interval", max(num_episodes, 1)))
    start_time = time.perf_counter()

    for episode in range(num_episodes):
        state = env.reset(seed=seed + episode)
        agent.reset()
        total_reward = 0.0
        metrics = EpisodeMetrics()

        for _step in range(max_steps):
            action = agent.select_action(state, explore=True, env=env)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            agent.update()
            metrics.update(info)
            total_reward += float(reward)
            state = next_state
            if done:
                break

        rows.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "scenario": getattr(env, "scenario_name", "default"),
                "graph_ablation": getattr(env, "graph_ablation", "full_graph"),
                "episode": episode,
                "total_reward": total_reward,
                "total_cost": metrics.total_cost,
                "reagent_shortage_frequency": metrics.reagent_shortage_frequency,
                "bioreactor_shortage_frequency": metrics.bioreactor_shortage_frequency,
                "transshipment_count": metrics.transshipment_count,
                "transshipment_cost": metrics.transshipment_cost,
                "runtime_seconds": time.perf_counter() - start_time,
            }
        )

        if (episode + 1) % checkpoint_interval == 0:
            agent.save(checkpoint_dir / f"{algorithm}_seed{seed}_episode{episode + 1}.pt")

    return rows


def write_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class EpisodeMetrics:
    """Aggregate metrics exposed by the current environment."""

    def __init__(self) -> None:
        self.total_cost = 0.0
        self.steps = 0
        self.reagent_shortage_steps = 0
        self.bioreactor_shortage_steps = 0
        self.transshipment_count = 0
        self.transshipment_cost = 0.0

    def update(self, info: dict[str, Any]) -> None:
        self.steps += 1
        self.total_cost += float(info.get("cost", 0.0))
        under_reagents = np.asarray(info.get("under_reagents", []), dtype=float)
        under_bioreactors = np.asarray(info.get("under_bioreactors", []), dtype=float)
        if under_reagents.size and np.any(under_reagents > 0):
            self.reagent_shortage_steps += 1
        if under_bioreactors.size and np.any(under_bioreactors > 0):
            self.bioreactor_shortage_steps += 1

        specimen_transfers = np.asarray(info.get("specimen_transfers", []), dtype=float)
        capacity_transfers = np.asarray(info.get("capacity_transfers", []), dtype=float)
        reagent_transfers = np.asarray(info.get("reagent_transfers", []), dtype=float)
        transfer_values = np.concatenate((specimen_transfers, capacity_transfers, reagent_transfers))
        self.transshipment_count += int(np.count_nonzero(np.abs(transfer_values) > 1e-8))
        self.transshipment_cost += (
            600.0 * float(np.abs(specimen_transfers).sum())
            + 500.0 * float(np.abs(capacity_transfers).sum())
            + 200.0 * float(np.abs(reagent_transfers).sum())
        )

    @property
    def reagent_shortage_frequency(self) -> float:
        return self.reagent_shortage_steps / self.steps if self.steps else 0.0

    @property
    def bioreactor_shortage_frequency(self) -> float:
        return self.bioreactor_shortage_steps / self.steps if self.steps else 0.0


def _to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_to_tuple(item) for item in value)
    return value
