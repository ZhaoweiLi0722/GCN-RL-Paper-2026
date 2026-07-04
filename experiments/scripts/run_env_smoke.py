"""Run a tiny capacity-planning environment smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.capacity_planning import CapacityPlanningConfig, CapacityPlanningEnv
from src.graph.ablation import with_graph_ablation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/configs/smoke_capacity_planning.json",
        help="Path to a JSON environment config.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--remove-capacity-edges", action="store_true")
    parser.add_argument("--remove-resource-edges", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    config = with_graph_ablation(
        config,
        remove_capacity_edges=args.remove_capacity_edges,
        remove_resource_edges=args.remove_resource_edges,
    )
    env = CapacityPlanningEnv(config, seed=args.seed)
    observation = env.reset(seed=args.seed)
    total_cost = 0.0

    for _ in range(config.episode_horizon):
        action = env.sample_random_action() if args.random_policy else env.noop_action()
        observation, _reward, done, info = env.step(action)
        total_cost += float(info["cost"])
        if done:
            break

    graph_obs = env.graph_observation()
    print(f"steps={env.t}")
    print(f"observation_size={observation.shape[0]}")
    print(f"action_size={env.action_size}")
    print(f"node_features_shape={graph_obs['node_features'].shape}")
    print(f"total_cost={total_cost:.2f}")


def load_config(path: Path) -> CapacityPlanningConfig:
    data = json.loads(path.read_text())
    data.pop("scenario_name", None)
    data.pop("graph_ablation", None)
    data.pop("demand_forecast_error", None)
    normalized = {key: _json_to_tuple(value) for key, value in data.items()}
    return CapacityPlanningConfig(**normalized)


def _json_to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_json_to_tuple(item) for item in value)
    return value


if __name__ == "__main__":
    main()
