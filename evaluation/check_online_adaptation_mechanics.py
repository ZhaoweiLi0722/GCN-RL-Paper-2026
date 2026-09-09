"""Bounded deterministic fixtures only; no scientific campaign or learner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from src.env.patient_capacity_planning import PatientConditionCapacityEnv, patient_env_config_from_dict
from src.env.public_demand_information import PublicDemandForecaster
from src.utils.finite_scenario_tree import solve_fixture


DEFAULT_CONFIG = Path("experiments/configs/online_adaptation_information_mechanics.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PatientScenarioFixture:
    """Finite demand tapes wrap the unchanged patient simulator for testing.

    Tapes are test inputs, not a proposed demand model. Oracle code knows their
    finite distribution; actions must still respect public-history partitions.
    No other policy is given this wrapper or private simulator snapshots.
    """

    def __init__(self, config):
        self.config = config
        if config["role"] != "mechanics_fixture_only_not_research_performance" or config["seed"] != 123:
            raise ValueError("only explicit fixture role and seed 123 are permitted")
        env_config = config["env"]
        horizon = config["horizon"]
        if type(horizon) is not int or not 1 <= horizon <= 6 or env_config["episode_horizon"] != horizon:
            raise ValueError("bounded fixture horizon mismatch")
        if env_config["num_facilities"] != 2 or len(config["worlds"]) != 2:
            raise ValueError("fixture requires two facilities and two worlds")
        for key in ("include_demand_forecast_state", "include_demand_history_state", "include_demand_sequence_state"):
            if env_config.get(key, False):
                raise ValueError("legacy forecast-derived channels are forbidden in this fixture")
        if not 1 <= len(config["actions"]) <= 4:
            raise ValueError("invalid bounded action count")
        for fractions in config["actions"].values():
            if len(fractions) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in fractions):
                raise ValueError("invalid fixture action")
        for world in config["worlds"]:
            if len(world["arrivals"]) != horizon + 1 or any(
                len(row) != 2 or any(type(v) is not int or v < 0 or v > 20 for v in row)
                for row in world["arrivals"]
            ):
                raise ValueError("invalid bounded demand tape")
        self.typed = patient_env_config_from_dict(env_config)
        self.nodes, self.edges, self.records = {}, {}, []
        for index, world in enumerate(config["worlds"]):
            env = PatientConditionCapacityEnv(self.typed, seed=123)
            env.demand = np.asarray(world["arrivals"][0], dtype=float)
            forecaster = PublicDemandForecaster(**config["forecast"])
            public = self._observe(env, forecaster)
            self.nodes[(index, ())] = (env.state_dict(), forecaster.state_dict(), (public,))

    def _observe(self, env, forecaster):
        forecaster.observe(epoch=env.t, arrivals=env.demand, announced_exposure=(1.0, 1.0))
        prediction = forecaster.issue(epoch=env.t, future_announced_exposure=[(1.0, 1.0)] * self.config["forecast"]["horizon"])
        # No world index, RNG, scenario name or future tape enters this key.
        return (tuple(float(v) for v in env.observation()), prediction.per_epoch)

    def information_key(self, state):
        return self.nodes[state][2]

    def transition(self, state, action_name):
        edge = (state, action_name)
        if edge in self.edges:
            return self.edges[edge]
        world, path = state
        if len(path) >= self.config["horizon"]:
            raise ValueError("transition beyond fixture horizon")
        snapshot, forecast_state, history = self.nodes[state]
        env = PatientConditionCapacityEnv(self.typed, seed=123)
        env.load_state_dict(snapshot)
        forecaster = PublicDemandForecaster(**self.config["forecast"])
        forecaster.load_state_dict(forecast_state)
        action = env.noop_action()
        n = env.config.num_facilities
        action[4 * n:5 * n] = 2 * np.asarray(self.config["actions"][action_name], dtype=float) - 1
        _, _, _, info = env.step(action)
        cost = float(info["cost"])
        successor = (world, path + (action_name,))
        env.demand = np.asarray(self.config["worlds"][world]["arrivals"][env.t], dtype=float)
        public = self._observe(env, forecaster)
        next_history = history + ((action_name, cost, public),)
        self.nodes[successor] = (env.state_dict(), forecaster.state_dict(), next_history)
        self.edges[edge] = (cost, successor)
        self.records.append({
            "world": world, "past_actions": list(path), "action": action_name, "cost": cost,
            "public_history_sha256": hashlib.sha256(repr(history).encode()).hexdigest(),
            "next_public_history_sha256": hashlib.sha256(repr(next_history).encode()).hexdigest(),
            "active_capacity": info["overtime_active_capacity"].tolist(),
            "requested_capacity": info["overtime_requested_surge"].tolist(),
        })
        return self.edges[edge]


def run(config_path: Path, output: Path):
    config = json.loads(config_path.read_text())
    model = PatientScenarioFixture(config)
    worlds = tuple((world["probability"], (i, ())) for i, world in enumerate(config["worlds"]))
    if len({model.information_key(state) for _, state in worlds}) != 1:
        raise ValueError("fixture root must not reveal its latent world")
    output.mkdir(parents=True, exist_ok=False)
    claim = {"fixture_only": True, "config_sha256": sha(config_path), "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    (output / "claim.json").write_text(json.dumps(claim, indent=2) + "\n")
    try:
        result = solve_fixture(model, worlds=worlds, actions=tuple(config["actions"]), horizon=config["horizon"])
        result.update({"fixture_only": True, "research_performance_evidence": False,
                       "training_performed": False, "formal_evaluation_performed": False,
                       "config_sha256": sha(config_path), "execution_commit": claim["execution_commit"]})
        rows = output / "transitions.jsonl"
        rows.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in model.records))
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        paths = [config_path, Path("evaluation/check_online_adaptation_mechanics.py"), Path("src/env/public_demand_information.py"),
                 Path("src/utils/finite_scenario_tree.py"), Path("src/env/capacity_planning.py"),
                 Path("src/env/patient_capacity_planning.py"), output / "claim.json", rows, output / "summary.json"]
        (output / "inventory.json").write_text(json.dumps({"files": [{"path": str(p), "sha256": sha(p)} for p in paths]}, indent=2) + "\n")
        (output / "status.json").write_text(json.dumps({"status": "completed", "exit_code": 0}) + "\n")
        return result
    except BaseException as exc:
        (output / "status.json").write_text(json.dumps({"status": "failed", "error": repr(exc), "exit_code": 1}) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
