"""Project a dense option teacher cache onto graph-relevant action groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.option_teacher_config import select_restricted_teacher_options
from evaluation.run_gcn_residual_sweep import (
    load_local_search_demonstrations,
    save_local_search_demonstrations,
)
from src.baselines.heuristics import heuristic_settings_for_policy
from src.rl.config import load_config
from src.rl.residual_options import (
    ResidualOptionSpec,
    residual_option_actions_from_state,
)


DEFAULT_GROUPS = (
    "reagent_transfer",
    "combined_transfer",
    "combined_network",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-snapshot", required=True)
    parser.add_argument("--base-policy", default="mdl2")
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--min-advantage", type=float, default=500_000.0)
    args = parser.parse_args()

    config = load_config(args.config_snapshot)
    residual_config = dict(config.get("residual_action", {}))
    base_settings = heuristic_settings_for_policy(
        args.base_policy,
        dict(residual_config.get("base_policy_config", {})),
    )
    demos = load_local_search_demonstrations(args.input)
    restricted, summary = restrict_teacher_demonstrations(
        demos,
        env_config=dict(config["env"]),
        anchor_settings=base_settings,
        allowed_groups=tuple(args.groups),
        min_advantage=float(args.min_advantage),
    )
    output = Path(args.output)
    save_local_search_demonstrations(output, restricted)
    summary.update(
        {
            "input": str(args.input),
            "output": str(output),
            "config_snapshot": str(args.config_snapshot),
            "base_policy": str(args.base_policy),
        }
    )
    summary_path = output.with_name(f"{output.stem}_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def restrict_teacher_demonstrations(
    demos: dict,
    *,
    env_config: dict,
    anchor_settings,
    allowed_groups: tuple[str, ...],
    min_advantage: float,
) -> tuple[dict, dict]:
    """Return action labels restricted to graph-relevant teacher options."""

    required = (
        "option_groups",
        "option_epsilons",
        "option_signs",
        "option_advantages",
        "option_feasible",
    )
    missing = [key for key in required if key not in demos]
    if missing:
        raise ValueError(f"Teacher cache is missing dense options: {missing}")

    selected_indices, original_labels, selected_advantages = (
        select_restricted_teacher_options(
            demos["option_groups"],
            demos["option_advantages"],
            demos["option_feasible"],
            allowed_groups=allowed_groups,
            min_advantage=min_advantage,
        )
    )
    groups = np.asarray(demos["option_groups"], dtype="U32")
    epsilons = np.asarray(demos["option_epsilons"], dtype=np.float32)
    signs = np.asarray(demos["option_signs"], dtype=np.float32)
    specs = tuple(
        ResidualOptionSpec(
            str(groups[index]),
            float(epsilons[index]),
            float(signs[index]),
        )
        for index in selected_indices
    )
    original_to_local = {
        int(original): local
        for local, original in enumerate(selected_indices)
    }
    local_labels = np.asarray(
        [original_to_local[int(label)] for label in original_labels],
        dtype=np.int64,
    )
    states = np.asarray(demos["states"], dtype=np.float32)
    actions = np.stack(
        [
            residual_option_actions_from_state(
                state,
                env_config,
                anchor_settings,
                specs,
            )[int(label)]
            for state, label in zip(states, local_labels)
        ],
        axis=0,
    ).astype(np.float32)
    improved = local_labels != 0
    weights = np.ones(states.shape[0], dtype=np.float32)
    weights[improved] = np.maximum(
        selected_advantages[improved],
        1.0,
    ).astype(np.float32)
    weight_total = float(weights.sum())

    restricted = {
        key: value
        for key, value in demos.items()
        if not key.startswith("transition_")
    }
    restricted.update(
        {
            "states": states,
            "actions": actions,
            "weights": weights,
            "improved_mask": improved,
            "improved_steps": int(improved.sum()),
            "anchor_keep_steps": int((~improved).sum()),
            "mean_step_improvement": (
                float(selected_advantages[improved].mean())
                if np.any(improved)
                else 0.0
            ),
            "improved_weight_fraction": (
                float(weights[improved].sum()) / weight_total
                if weight_total > 0.0
                else 0.0
            ),
            "option_advantages": np.asarray(
                demos["option_advantages"],
                dtype=np.float32,
            )[:, selected_indices],
            "option_feasible": np.asarray(
                demos["option_feasible"],
                dtype=bool,
            )[:, selected_indices],
            "option_groups": groups[selected_indices],
            "option_epsilons": epsilons[selected_indices],
            "option_signs": signs[selected_indices],
        }
    )
    all_best = np.maximum(
        np.max(
            np.where(
                np.asarray(demos["option_feasible"], dtype=bool),
                np.asarray(demos["option_advantages"], dtype=np.float32),
                -np.inf,
            ),
            axis=1,
        ),
        0.0,
    )
    restricted_positive = np.maximum(selected_advantages, 0.0)
    summary = {
        "rows": int(states.shape[0]),
        "allowed_groups": list(allowed_groups),
        "retained_option_count": int(selected_indices.size),
        "min_advantage": float(min_advantage),
        "correction_rows": int(improved.sum()),
        "correction_rate": float(improved.mean()),
        "dense_positive_advantage_retained_fraction": (
            float(restricted_positive.sum()) / float(all_best.sum())
            if float(all_best.sum()) > 0.0
            else 0.0
        ),
        "mean_selected_advantage_when_correcting": (
            float(selected_advantages[improved].mean())
            if np.any(improved)
            else 0.0
        ),
        "transitions_removed": bool(
            any(key.startswith("transition_") for key in demos)
        ),
    }
    return restricted, summary


if __name__ == "__main__":
    main()
