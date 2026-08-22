"""Episode-balanced environment switching for multi-scenario RL training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class EpisodeScenarioEnv:
    """Expose compatible environments through one episode-level interface.

    A scenario is selected only when ``reset`` is called and remains fixed for
    the entire episode. Scenario names are retained for logging but are never
    appended to the policy observation.
    """

    def __init__(
        self,
        environments: Sequence[Any],
        *,
        start_index: int = 0,
    ) -> None:
        self._environments = tuple(environments)
        if not self._environments:
            raise ValueError("At least one scenario environment is required")
        self._validate_compatibility()
        self._start_index = int(start_index) % len(self._environments)
        self._reset_count = 0
        self._current_index = self._start_index

    @property
    def environments(self) -> tuple[Any, ...]:
        return self._environments

    @property
    def current_env(self) -> Any:
        return self._environments[self._current_index]

    @property
    def current_scenario_index(self) -> int:
        return self._current_index

    @property
    def observation_size(self) -> int:
        return int(self.current_env.observation_size)

    @property
    def action_size(self) -> int:
        return int(self.current_env.action_size)

    @property
    def config(self) -> Any:
        return self.current_env.config

    @property
    def scenario_name(self) -> str:
        return str(getattr(self.current_env, "scenario_name", "default"))

    @property
    def graph_ablation(self) -> str:
        return str(
            getattr(self.current_env, "graph_ablation", "full_graph")
        )

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._current_index = (
            self._start_index + self._reset_count
        ) % len(self._environments)
        self._reset_count += 1
        return self.current_env.reset(seed=seed)

    def set_episode_index(self, episode_index: int) -> None:
        """Position the next reset at an episode boundary for resumption."""

        index = int(episode_index)
        if index < 0:
            raise ValueError("episode_index must be non-negative")
        self._reset_count = index
        self._current_index = (
            self._start_index + index
        ) % len(self._environments)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        return self.current_env.step(action)

    def enable_train_randomization(self, **settings: Any) -> None:
        for environment in self._environments:
            environment.enable_train_randomization(**settings)

    def state_dict(self) -> dict[str, Any]:
        """Snapshot scenario position and every stateful child environment."""

        child_states = []
        for environment in self._environments:
            snapshot = getattr(environment, "state_dict", None)
            child_states.append(None if not callable(snapshot) else snapshot())
        return {
            "format_version": 1,
            "start_index": int(self._start_index),
            "reset_count": int(self._reset_count),
            "current_index": int(self._current_index),
            "environments": child_states,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`state_dict`."""

        if int(state.get("format_version", -1)) != 1:
            raise ValueError("Unsupported multi-scenario environment state format")
        if int(state.get("start_index", -1)) != self._start_index:
            raise ValueError("Multi-scenario start index does not match")
        child_states = tuple(state.get("environments", ()))
        if len(child_states) != len(self._environments):
            raise ValueError("Multi-scenario child environment count does not match")
        for environment, child_state in zip(self._environments, child_states):
            if child_state is None:
                continue
            restore = getattr(environment, "load_state_dict", None)
            if not callable(restore):
                raise ValueError("A child environment cannot restore its state")
            restore(dict(child_state))
        self._reset_count = int(state["reset_count"])
        self._current_index = int(state["current_index"])

    def __getattr__(self, name: str) -> Any:
        try:
            environments = object.__getattribute__(
                self,
                "_environments",
            )
            current_index = object.__getattribute__(
                self,
                "_current_index",
            )
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(environments[current_index], name)

    def _validate_compatibility(self) -> None:
        reference = self._environments[0]
        reference_observation = int(reference.observation_size)
        reference_action = int(reference.action_size)
        reference_horizon = int(reference.config.episode_horizon)
        reference_facilities = int(reference.config.num_facilities)
        reference_ablation = str(
            getattr(reference, "graph_ablation", "full_graph")
        )

        for index, environment in enumerate(self._environments[1:], start=1):
            checks = {
                "observation_size": (
                    int(environment.observation_size),
                    reference_observation,
                ),
                "action_size": (
                    int(environment.action_size),
                    reference_action,
                ),
                "episode_horizon": (
                    int(environment.config.episode_horizon),
                    reference_horizon,
                ),
                "num_facilities": (
                    int(environment.config.num_facilities),
                    reference_facilities,
                ),
                "graph_ablation": (
                    str(
                        getattr(
                            environment,
                            "graph_ablation",
                            "full_graph",
                        )
                    ),
                    reference_ablation,
                ),
            }
            mismatches = {
                key: values
                for key, values in checks.items()
                if values[0] != values[1]
            }
            if mismatches:
                raise ValueError(
                    f"Scenario environment {index} is incompatible: "
                    f"{mismatches}"
                )
