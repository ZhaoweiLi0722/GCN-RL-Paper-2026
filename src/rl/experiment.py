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
    progress_interval = int(config.get("progress_interval", 0))
    start_time = time.perf_counter()
    pretrain_summary = _maybe_pretrain_agent(agent, env, config)
    elite_config = dict(config.get("elite_imitation", {}))
    elite_enabled = bool(elite_config.get("enabled", False))
    elite_warmup_episodes = int(elite_config.get("warmup_episodes", 0))
    elite_min_improvement = float(elite_config.get("min_improvement", 0.0))
    elite_max_episodes = int(elite_config.get("max_episodes", 5))
    elite_episodes: list[tuple[float, np.ndarray, np.ndarray]] = []
    elite_best_cost = float("inf")
    elite_update_count = 0

    for episode in range(num_episodes):
        state = env.reset(seed=seed + episode)
        agent.reset()
        total_reward = 0.0
        metrics = EpisodeMetrics()
        episode_states: list[np.ndarray] = []
        episode_actions: list[np.ndarray] = []

        for _step in range(max_steps):
            action = agent.select_action(state, explore=True, env=env)
            next_state, reward, done, info = env.step(action)
            if elite_enabled:
                episode_states.append(np.asarray(state, dtype=np.float32))
                episode_actions.append(np.asarray(action, dtype=np.float32))
            agent.observe(state, action, reward, next_state, done)
            agent.update()
            metrics.update(info)
            total_reward += float(reward)
            state = next_state
            if done:
                break

        elite_summary = _maybe_fit_elite_episode(
            agent,
            elite_config,
            enabled=elite_enabled,
            episode=episode,
            warmup_episodes=elite_warmup_episodes,
            min_improvement=elite_min_improvement,
            max_episodes=elite_max_episodes,
            elite_episodes=elite_episodes,
            states=episode_states,
            actions=episode_actions,
            cost=metrics.total_cost,
        )
        if elite_summary:
            elite_best_cost = float(elite_summary.get("elite_best_cost", metrics.total_cost))
            elite_update_count += 1

        rows.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "scenario": getattr(env, "scenario_name", "default"),
                "graph_ablation": getattr(env, "graph_ablation", "full_graph"),
                "episode": episode,
                "total_reward": total_reward,
                "total_cost": metrics.total_cost,
                "service_level": metrics.service_level,
                "average_waiting_time": metrics.average_waiting_time,
                "reagent_shortage_frequency": metrics.reagent_shortage_frequency,
                "bioreactor_shortage_frequency": metrics.bioreactor_shortage_frequency,
                "bioreactor_utilization": metrics.bioreactor_utilization,
                "transshipment_count": metrics.transshipment_count,
                "transshipment_cost": metrics.transshipment_cost,
                "pretrain_samples": pretrain_summary.get("samples", 0),
                "pretrain_final_loss": pretrain_summary.get("final_loss", ""),
                "elite_imitation_updates": elite_update_count,
                "elite_buffer_size": elite_summary.get("elite_buffer_size", len(elite_episodes)),
                "elite_best_cost": "" if not np.isfinite(elite_best_cost) else elite_best_cost,
                "elite_imitation_loss": elite_summary.get("final_loss", ""),
                "runtime_seconds": time.perf_counter() - start_time,
            }
        )

        if (episode + 1) % checkpoint_interval == 0:
            agent.save(checkpoint_dir / f"{algorithm}_seed{seed}_episode{episode + 1}.pt")

        if progress_interval and (
            episode == 0 or (episode + 1) % progress_interval == 0 or episode + 1 == num_episodes
        ):
            print(
                f"{algorithm} seed={seed} episode={episode + 1}/{num_episodes} "
                f"reward={total_reward:.3f} cost={metrics.total_cost:.3f}",
                flush=True,
            )

    return rows


def _maybe_fit_elite_episode(
    agent,
    settings: dict[str, Any],
    *,
    enabled: bool,
    episode: int,
    warmup_episodes: int,
    min_improvement: float,
    max_episodes: int,
    elite_episodes: list[tuple[float, np.ndarray, np.ndarray]],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    cost: float,
) -> dict[str, Any]:
    if not enabled or episode + 1 < warmup_episodes:
        return {}
    if not states or not actions:
        return {}
    state_array = np.asarray(states, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    max_episodes = max(max_episodes, 1)
    if len(elite_episodes) >= max_episodes:
        worst_cost = max(item[0] for item in elite_episodes)
        threshold = worst_cost * (1.0 - min_improvement)
        if cost >= threshold:
            return {}
    elite_episodes.append((float(cost), state_array, action_array))
    elite_episodes.sort(key=lambda item: item[0])
    del elite_episodes[max_episodes:]

    if not any(np.isclose(cost, item[0]) for item in elite_episodes):
        return {}
    if not hasattr(agent, "fit_action_batch"):
        raise ValueError(f"{getattr(agent, 'algorithm', 'agent')} does not support elite_imitation")
    elite_states = np.concatenate([item[1] for item in elite_episodes], axis=0)
    elite_actions = np.concatenate([item[2] for item in elite_episodes], axis=0)
    summary = agent.fit_action_batch(elite_states, elite_actions, settings)
    summary["elite_buffer_size"] = len(elite_episodes)
    summary["elite_best_cost"] = elite_episodes[0][0]
    if summary:
        print(
            "elite_imitation "
            f"episode={episode + 1} cost={cost:.3f} buffer={summary.get('elite_buffer_size')} "
            f"samples={summary.get('samples')} "
            f"final_loss={summary.get('final_loss'):.6f}",
            flush=True,
        )
    return summary


def _maybe_pretrain_agent(agent, env: CapacityPlanningEnv, config: dict[str, Any]) -> dict[str, Any]:
    pretrain_config = dict(config.get("imitation_pretrain", {}))
    if not bool(pretrain_config.get("enabled", False)):
        return {}
    if not hasattr(agent, "pretrain_with_heuristic"):
        raise ValueError(f"{config.get('algorithm', agent.algorithm)} does not support imitation_pretrain")
    summary = agent.pretrain_with_heuristic(env, pretrain_config)
    if summary:
        print(
            "imitation_pretrain "
            f"policy={summary.get('policy')} samples={summary.get('samples')} "
            f"final_loss={summary.get('final_loss'):.6f}",
            flush=True,
        )
    return summary


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
        self.service_level = 0.0
        self.average_waiting_time = 0.0
        self.bioreactor_utilization = 0.0

    def update(self, info: dict[str, Any]) -> None:
        self.steps += 1
        self.total_cost += float(info.get("cost", 0.0))
        self.service_level = float(info.get("service_level", self.service_level))
        self.average_waiting_time = float(info.get("average_waiting_time", self.average_waiting_time))
        self.bioreactor_utilization = float(
            info.get("bioreactor_utilization", self.bioreactor_utilization)
        )
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
