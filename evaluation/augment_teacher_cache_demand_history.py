"""Append causal rolling-demand features to an existing teacher cache."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.rl.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Override demand_history_window from the environment config.",
    )
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=True) as source:
        payload = {key: source[key] for key in source.files}
    env_config = load_config(args.env_config)
    if args.window is not None:
        env_config["demand_history_window"] = int(args.window)
    augmented, summary = augment_teacher_cache_with_demand_history(
        payload,
        env_config,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **augmented)
    print(
        f"augmented {summary['samples']} states from "
        f"{summary['source_state_dim']} to {summary['target_state_dim']} features "
        f"with demand_history_window={summary['window']}",
        flush=True,
    )


def augment_teacher_cache_with_demand_history(
    payload: dict[str, Any],
    env_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Insert rolling mean, trend, and forecast error in each facility row."""

    if not bool(env_config.get("include_demand_history_state", False)):
        raise ValueError("Environment config must enable demand history state")
    original_states = np.asarray(payload["states"], dtype=np.float32)
    original_transition_states = np.asarray(
        payload.get("transition_states", original_states),
        dtype=np.float32,
    )
    original_transition_next_states = np.asarray(
        payload["transition_next_states"],
        dtype=np.float32,
    )
    dones = np.asarray(payload["transition_dones"], dtype=bool)
    if (
        original_states.ndim != 2
        or original_transition_states.shape != original_states.shape
        or original_transition_next_states.shape != original_states.shape
        or dones.shape != (original_states.shape[0],)
    ):
        raise ValueError("Teacher transition arrays must align")

    layout = _source_layout(env_config)
    expected_source_dim = (
        layout["num_facilities"]
        * (layout["source_width"] + layout["summary_width"])
        + layout["time_width"]
    )
    expected_augmented_dim = expected_source_dim + 3 * layout["num_facilities"]
    replaced_existing_history = False
    if original_states.shape[1] == expected_source_dim:
        states = original_states
        transition_states = original_transition_states
        transition_next_states = original_transition_next_states
    elif original_states.shape[1] == expected_augmented_dim:
        replaced_existing_history = True
        states = _remove_history(original_states, layout)
        transition_states = _remove_history(
            original_transition_states,
            layout,
        )
        transition_next_states = _remove_history(
            original_transition_next_states,
            layout,
        )
    else:
        raise ValueError(
            "Expected source state_dim="
            f"{expected_source_dim} or history-augmented state_dim="
            f"{expected_augmented_dim}, got {original_states.shape[1]}"
        )

    augmented_states: list[np.ndarray] = []
    augmented_transition_states: list[np.ndarray] = []
    augmented_next_states: list[np.ndarray] = []
    demand_history: list[np.ndarray] = []
    forecast_error_history: list[np.ndarray] = []
    window = int(env_config.get("demand_history_window", 4))
    if window < 1:
        raise ValueError("demand_history_window must be positive")

    for index in range(states.shape[0]):
        demand, forecast_error = _demand_and_forecast_error(
            states[index],
            env_config,
            layout,
        )
        demand_history.append(demand)
        forecast_error_history.append(forecast_error)
        demand_history = demand_history[-window:]
        forecast_error_history = forecast_error_history[-window:]
        history_features = _history_features(
            demand_history,
            forecast_error_history,
        )
        augmented_states.append(
            _insert_history(states[index], history_features, layout)
        )
        augmented_transition_states.append(
            _insert_history(
                transition_states[index],
                history_features,
                layout,
            )
        )

        next_demand, next_forecast_error = _demand_and_forecast_error(
            transition_next_states[index],
            env_config,
            layout,
        )
        next_history = (demand_history + [next_demand])[-window:]
        next_error_history = (
            forecast_error_history + [next_forecast_error]
        )[-window:]
        augmented_next_states.append(
            _insert_history(
                transition_next_states[index],
                _history_features(next_history, next_error_history),
                layout,
            )
        )
        if dones[index]:
            demand_history = []
            forecast_error_history = []

    result = dict(payload)
    result["states"] = np.asarray(augmented_states, dtype=np.float32)
    result["transition_states"] = np.asarray(
        augmented_transition_states,
        dtype=np.float32,
    )
    result["transition_next_states"] = np.asarray(
        augmented_next_states,
        dtype=np.float32,
    )
    result["demand_history_state_version"] = np.asarray(1, dtype=np.int64)
    result["demand_history_window"] = np.asarray(window, dtype=np.int64)
    return result, {
        "samples": int(states.shape[0]),
        "source_state_dim": int(original_states.shape[1]),
        "target_state_dim": int(result["states"].shape[1]),
        "window": window,
        "replaced_existing_history": int(replaced_existing_history),
    }


def _source_layout(env_config: dict[str, Any]) -> dict[str, int]:
    n = int(env_config["num_facilities"])
    lead_time = int(env_config.get("production_lead_time", 3))
    include_supplier = int(bool(env_config.get("include_supplier_state", False)))
    include_forecast = int(
        bool(env_config.get("include_demand_forecast_state", False))
    )
    include_pipeline = int(
        bool(env_config.get("include_transfer_pipeline_state", False))
    )
    source_width = (
        3
        + lead_time
        + include_supplier
        + include_forecast
        + 3 * include_pipeline
    )
    summary_width = 0
    if env_config.get("env_type") == "patient_condition":
        summary_width = 7 + len(
            tuple(env_config.get("survival_bucket_edges", (0.85, 0.90, 0.97)))
        )
    return {
        "num_facilities": n,
        "lead_time": lead_time,
        "include_supplier": include_supplier,
        "include_forecast": include_forecast,
        "source_width": source_width,
        "summary_width": summary_width,
        "time_width": int(bool(env_config.get("include_time_state", False))),
    }


def _demand_and_forecast_error(
    state: np.ndarray,
    env_config: dict[str, Any],
    layout: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    n = layout["num_facilities"]
    facility = np.asarray(
        state[: n * layout["source_width"]],
        dtype=np.float32,
    ).reshape(n, layout["source_width"])
    demand = facility[:, 0].copy()
    if layout["include_forecast"]:
        forecast_col = (
            3 + layout["lead_time"] + layout["include_supplier"]
        )
        forecast = facility[:, forecast_col]
        horizon = max(int(env_config.get("demand_forecast_horizon", 1)), 1)
        forecast_error = demand - forecast / float(horizon)
    else:
        forecast_error = np.zeros(n, dtype=np.float32)
    return demand, np.asarray(forecast_error, dtype=np.float32)


def _history_features(
    demand_history: list[np.ndarray],
    forecast_error_history: list[np.ndarray],
) -> np.ndarray:
    history = np.stack(demand_history, axis=0)
    rolling_mean = history.mean(axis=0)
    if history.shape[0] <= 1:
        trend = np.zeros_like(rolling_mean)
    else:
        trend = (history[-1] - history[0]) / float(history.shape[0] - 1)
    rolling_error = np.stack(forecast_error_history, axis=0).mean(axis=0)
    return np.column_stack((rolling_mean, trend, rolling_error)).astype(
        np.float32
    )


def _insert_history(
    state: np.ndarray,
    history_features: np.ndarray,
    layout: dict[str, int],
) -> np.ndarray:
    n = layout["num_facilities"]
    base_end = n * layout["source_width"]
    base = np.asarray(state[:base_end], dtype=np.float32).reshape(
        n,
        layout["source_width"],
    )
    expanded = np.concatenate((base, history_features), axis=1).reshape(-1)
    return np.concatenate(
        (expanded, np.asarray(state[base_end:], dtype=np.float32))
    ).astype(np.float32)


def _remove_history(
    states: np.ndarray,
    layout: dict[str, int],
) -> np.ndarray:
    """Remove an existing three-column history block from flat states."""

    array = np.asarray(states, dtype=np.float32)
    n = layout["num_facilities"]
    expanded_width = layout["source_width"] + 3
    expanded_end = n * expanded_width
    facility = array[:, :expanded_end].reshape(
        array.shape[0],
        n,
        expanded_width,
    )
    base = facility[:, :, : layout["source_width"]].reshape(
        array.shape[0],
        -1,
    )
    return np.concatenate((base, array[:, expanded_end:]), axis=1).astype(
        np.float32
    )


if __name__ == "__main__":
    main()
