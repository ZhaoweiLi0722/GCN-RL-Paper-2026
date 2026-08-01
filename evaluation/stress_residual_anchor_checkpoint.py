"""Stress a residual-policy anchor against replay states from a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.baselines.heuristics import (
    facility_net_action_from_state,
    heuristic_settings_for_policy,
)
from src.rl.networks import require_torch, torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-state", required=True)
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--anchor-policy", default="mdl2")
    parser.add_argument("--calls", type=int, default=150_000)
    parser.add_argument("--minimum-replay-rows", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = stress_checkpoint_anchor(
        training_state_path=Path(args.training_state),
        actor_checkpoint_path=Path(args.actor_checkpoint),
        anchor_policy=str(args.anchor_policy),
        calls=int(args.calls),
        minimum_replay_rows=int(args.minimum_replay_rows),
    )
    print(
        "Residual anchor stress PASS: "
        f"calls={result['calls']}; replay_rows={result['replay_rows']}; "
        f"elapsed_seconds={result['elapsed_seconds']:.3f}; "
        f"actions_per_second={result['actions_per_second']:.1f}; "
        f"baseline_sha256={result['baseline_sha256']}"
    )


def stress_checkpoint_anchor(
    *,
    training_state_path: Path,
    actor_checkpoint_path: Path,
    anchor_policy: str,
    calls: int,
    minimum_replay_rows: int,
) -> dict[str, Any]:
    """Repeatedly evaluate exact replay states and require stable finite actions."""

    if calls < 1:
        raise ValueError("calls must be positive")
    if minimum_replay_rows < 1:
        raise ValueError("minimum_replay_rows must be positive")
    if not training_state_path.is_file():
        raise FileNotFoundError(training_state_path)
    if not actor_checkpoint_path.is_file():
        raise FileNotFoundError(actor_checkpoint_path)

    require_torch()
    training_state = torch.load(
        training_state_path,
        map_location="cpu",
        weights_only=False,
    )
    actor_checkpoint = torch.load(
        actor_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if str(training_state.get("algorithm")) != str(
        actor_checkpoint.get("algorithm")
    ):
        raise ValueError("training-state and actor-checkpoint algorithms differ")

    replay = dict(training_state.get("agent", {}).get("replay_buffer", {}))
    replay_rows = int(replay.get("size", 0))
    if replay_rows < minimum_replay_rows:
        raise ValueError(
            f"replay buffer has {replay_rows} rows; expected at least "
            f"{minimum_replay_rows}"
        )
    states = np.asarray(replay.get("states"), dtype=np.float32)
    next_states = np.asarray(replay.get("next_states"), dtype=np.float32)
    expected_shape = (replay_rows, int(replay.get("state_dim", -1)))
    if states.shape != expected_shape or next_states.shape != expected_shape:
        raise ValueError(
            "replay state shape mismatch: "
            f"states={states.shape}; next_states={next_states.shape}; "
            f"expected={expected_shape}"
        )
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(next_states)):
        raise ValueError("replay states contain NaN or Inf")

    graph_spec = dict(actor_checkpoint.get("graph_spec", {}))
    env_config = graph_spec.get("env_config")
    if not isinstance(env_config, dict) or not env_config:
        raise ValueError("actor checkpoint does not contain graph env_config")
    settings = heuristic_settings_for_policy(anchor_policy)
    replay_views = (states, next_states)
    baseline_actions = []
    digest = hashlib.sha256()
    for replay_view in replay_views:
        view_actions = np.stack(
            [
                facility_net_action_from_state(
                    state,
                    env_config,
                    settings=settings,
                )
                for state in replay_view
            ]
        )
        if not np.all(np.isfinite(view_actions)):
            raise ValueError("anchor produced NaN or Inf during baseline pass")
        baseline_actions.append(view_actions)
        digest.update(view_actions.tobytes(order="C"))

    started = time.perf_counter()
    for call_index in range(calls):
        view_index = call_index & 1
        row_index = (call_index // 2) % replay_rows
        actual = facility_net_action_from_state(
            replay_views[view_index][row_index],
            env_config,
            settings=settings,
        )
        expected = baseline_actions[view_index][row_index]
        if not np.array_equal(actual, expected):
            raise RuntimeError(
                "anchor action changed during stress pass: "
                f"call={call_index}; view={view_index}; row={row_index}"
            )
    elapsed = time.perf_counter() - started
    return {
        "calls": calls,
        "replay_rows": replay_rows,
        "elapsed_seconds": elapsed,
        "actions_per_second": calls / elapsed,
        "baseline_sha256": digest.hexdigest(),
    }


if __name__ == "__main__":
    main()
