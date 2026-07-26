"""Audit a dense residual-option teacher cache before policy training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.run_gcn_residual_sweep import (
    load_local_search_demonstrations,
)
from src.baselines.heuristics import heuristic_settings_for_policy
from src.rl.config import load_config
from src.rl.residual_options import (
    ResidualOptionSpec,
    residual_option_actions_from_state,
    residual_option_labels_from_advantages,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--anchor-policy", default="mdl2")
    parser.add_argument("--min-advantage", type=float, default=500000.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cache = load_local_search_demonstrations(args.cache)
    env_config = load_config(args.env_config)
    result = audit_option_teacher_cache(
        cache,
        env_config,
        anchor_policy=args.anchor_policy,
        min_advantage=args.min_advantage,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    print(text)


def audit_option_teacher_cache(
    cache: dict[str, Any],
    env_config: dict[str, Any],
    *,
    anchor_policy: str,
    min_advantage: float,
) -> dict[str, Any]:
    states = np.asarray(cache["states"], dtype=np.float32)
    actions = np.asarray(cache["actions"], dtype=np.float32)
    advantages = np.asarray(cache["option_advantages"], dtype=np.float32)
    feasible = np.asarray(cache["option_feasible"], dtype=bool)
    groups = np.asarray(cache["option_groups"], dtype="U32")
    epsilons = np.asarray(cache["option_epsilons"], dtype=np.float32)
    signs = np.asarray(cache["option_signs"], dtype=np.float32)
    sample_count, option_count = advantages.shape
    if states.shape[0] != sample_count or actions.shape[0] != sample_count:
        raise ValueError("Teacher cache state/action rows are not aligned")
    if feasible.shape != advantages.shape:
        raise ValueError("Teacher option feasibility does not match advantages")
    if (
        groups.shape != (option_count,)
        or epsilons.shape != (option_count,)
        or signs.shape != (option_count,)
    ):
        raise ValueError("Teacher option metadata does not match option count")
    if not np.all(feasible[:, 0]):
        raise ValueError("Anchor option must be feasible in every state")
    anchor_advantage_max_abs = float(np.max(np.abs(advantages[:, 0])))
    if anchor_advantage_max_abs > 1e-3:
        raise ValueError("Anchor option advantage is not zero")

    selected_labels = residual_option_labels_from_advantages(
        advantages,
        feasible,
        min_advantage=min_advantage,
    )
    improved = selected_labels != 0
    cached_improved = np.asarray(cache["improved_mask"], dtype=bool)
    if not np.array_equal(improved, cached_improved):
        raise ValueError("Cached improved labels disagree with dense advantages")

    specs = tuple(
        ResidualOptionSpec(str(group), float(epsilon), float(sign))
        for group, epsilon, sign in zip(groups, epsilons, signs)
    )
    anchor_settings = heuristic_settings_for_policy(anchor_policy, {})
    action_errors = np.empty(sample_count, dtype=float)
    for index, (state, target, label) in enumerate(
        zip(states, actions, selected_labels)
    ):
        candidates = residual_option_actions_from_state(
            state,
            env_config,
            anchor_settings,
            specs,
        )
        action_errors[index] = float(
            np.max(np.abs(candidates[int(label)] - target))
        )
    max_action_reconstruction_error = float(action_errors.max(initial=0.0))
    if max_action_reconstruction_error > 1e-5:
        raise ValueError(
            "Cached teacher actions cannot be reconstructed from option metadata"
        )

    feasible_advantages = np.where(feasible, advantages, -np.inf)
    sorted_advantages = np.sort(feasible_advantages, axis=1)
    best_advantages = sorted_advantages[:, -1]
    second_advantages = (
        sorted_advantages[:, -2]
        if option_count > 1
        else np.zeros(sample_count, dtype=float)
    )
    margins = best_advantages - second_advantages
    selected_group_counts = Counter(
        str(groups[label]) for label in selected_labels
    )
    selected_option_counts = Counter(
        (
            str(groups[label]),
            float(signs[label]),
            float(epsilons[label]),
        )
        for label in selected_labels
    )
    return {
        "samples": sample_count,
        "option_count": option_count,
        "states_shape": list(states.shape),
        "actions_shape": list(actions.shape),
        "anchor_advantage_max_abs": anchor_advantage_max_abs,
        "anchor_feasible_fraction": float(np.mean(feasible[:, 0])),
        "mean_feasible_option_fraction": float(np.mean(feasible)),
        "improved_steps": int(improved.sum()),
        "anchor_steps": int((~improved).sum()),
        "improved_fraction": float(np.mean(improved)),
        "best_advantage_quantiles": quantiles(best_advantages),
        "best_second_margin_quantiles": quantiles(margins),
        "max_action_reconstruction_error": max_action_reconstruction_error,
        "selected_group_counts": dict(sorted(selected_group_counts.items())),
        "selected_option_counts": {
            f"{group}:{sign:g}:{epsilon:g}": count
            for (group, sign, epsilon), count in sorted(
                selected_option_counts.items()
            )
        },
        "transition_count": int(
            np.asarray(cache["transition_states"]).shape[0]
        ),
        "terminal_transition_count": int(
            np.asarray(cache["transition_dones"], dtype=bool).sum()
        ),
        "all_finite": bool(
            np.all(np.isfinite(states))
            and np.all(np.isfinite(actions))
            and np.all(np.isfinite(advantages))
        ),
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {}
    return {
        "min": float(np.min(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.quantile(finite, 0.50)),
        "q75": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


if __name__ == "__main__":
    main()
