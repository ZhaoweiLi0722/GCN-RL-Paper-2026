"""Append a normalized decision-time feature to an existing teacher cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.rl.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-config", required=True)
    args = parser.parse_args()

    env_config = load_config(args.env_config)
    with np.load(args.input, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    augmented, summary = augment_teacher_cache_with_time(payload, env_config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **augmented)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote time-aware teacher cache to {output}", flush=True)


def augment_teacher_cache_with_time(
    payload: dict[str, np.ndarray],
    env_config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return a non-destructive cache copy with ``t / horizon`` appended."""

    if not bool(env_config.get("include_time_state", False)):
        raise ValueError("env config must enable include_time_state")
    horizon = int(env_config.get("episode_horizon", 0))
    if horizon < 1:
        raise ValueError("episode_horizon must be positive")
    required = (
        "states",
        "transition_states",
        "transition_next_states",
        "transition_dones",
    )
    if not all(key in payload for key in required):
        raise ValueError("Teacher cache is missing transition arrays")

    states = np.asarray(payload["states"], dtype=np.float32)
    transition_states = np.asarray(payload["transition_states"], dtype=np.float32)
    next_states = np.asarray(
        payload["transition_next_states"],
        dtype=np.float32,
    )
    dones = np.asarray(payload["transition_dones"], dtype=bool)
    count = int(transition_states.shape[0])
    if (
        states.shape != transition_states.shape
        or next_states.shape != transition_states.shape
        or dones.shape != (count,)
        or not np.allclose(states, transition_states, rtol=0.0, atol=1e-6)
    ):
        raise ValueError("Teacher state and transition arrays are not aligned")

    state_times, next_times, trajectory_lengths = transition_time_features(
        dones,
        horizon,
    )
    augmented = dict(payload)
    augmented["states"] = np.column_stack((states, state_times)).astype(np.float32)
    augmented["transition_states"] = np.column_stack(
        (transition_states, state_times)
    ).astype(np.float32)
    augmented["transition_next_states"] = np.column_stack(
        (next_states, next_times)
    ).astype(np.float32)
    augmented["time_state_version"] = np.asarray(1, dtype=np.int64)
    return augmented, {
        "samples": count,
        "old_state_width": int(states.shape[1]),
        "new_state_width": int(states.shape[1] + 1),
        "episode_horizon": horizon,
        "trajectory_count": len(trajectory_lengths),
        "trajectory_lengths": trajectory_lengths,
        "terminal_transition_count": int(dones.sum()),
        "state_time_min": float(state_times.min(initial=0.0)),
        "state_time_max": float(state_times.max(initial=0.0)),
        "next_time_max": float(next_times.max(initial=0.0)),
    }


def transition_time_features(
    dones: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Recover decision times from ordered finite-horizon transitions."""

    terminal = np.asarray(dones, dtype=bool)
    state_times = np.empty(terminal.size, dtype=np.float32)
    next_times = np.empty(terminal.size, dtype=np.float32)
    trajectory_lengths: list[int] = []
    step = 0
    for index, done in enumerate(terminal):
        state_times[index] = min(step / horizon, 1.0)
        next_times[index] = min((step + 1) / horizon, 1.0)
        step += 1
        if done:
            trajectory_lengths.append(step)
            step = 0
    if step != 0:
        raise ValueError("Teacher cache ends before a terminal transition")
    if not trajectory_lengths:
        raise ValueError("Teacher cache contains no complete trajectory")
    return state_times, next_times, trajectory_lengths


if __name__ == "__main__":
    main()
