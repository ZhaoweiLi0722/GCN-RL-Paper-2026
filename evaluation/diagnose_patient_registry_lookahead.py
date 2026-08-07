"""Pinpoint patient-registry corruption in deterministic teacher look-ahead.

This utility replays selected state-probe decisions without writing formal
teacher outputs.  It is intentionally diagnostic-only: the configured policy,
actions, CRN seeds, and environment transitions are unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluation.network_residual_headroom import (
    candidate_action_specs,
    load_env_config,
    lookahead_rollout_seeds,
    make_anchor,
)
from src.env.patient_condition import PatientState
from src.rl.config import load_config
from src.rl.experiment import build_env


@dataclass(frozen=True)
class DiagnosticLocation:
    rollout: int
    step: int
    candidate: int
    replication: int
    lookahead_step: int
    boundary: str


def invalid_registry_values(env: Any) -> list[dict[str, str]]:
    """Describe non-PatientState registry entries without dereferencing them."""

    registry = getattr(env, "patient_registry", None)
    if not isinstance(registry, dict):
        return [
            {
                "patient_id": "<registry>",
                "value_type": type(registry).__name__,
                "value_repr": repr(registry),
            }
        ]
    return [
        {
            "patient_id": str(patient_id),
            "value_type": type(patient).__name__,
            "value_repr": repr(patient),
        }
        for patient_id, patient in registry.items()
        if not isinstance(patient, PatientState)
    ]


def assert_registry_types(env: Any, location: DiagnosticLocation) -> None:
    invalid = invalid_registry_values(env)
    if invalid:
        payload = {
            "location": asdict(location),
            "environment_time": int(getattr(env, "t", -1)),
            "invalid_registry_values": invalid,
        }
        raise RuntimeError(
            "PATIENT_REGISTRY_TYPE_CORRUPTION "
            + json.dumps(payload, sort_keys=True)
        )


def selected_indices(length: int, requested: int | None) -> Iterable[int]:
    if requested is None:
        return range(length)
    if not 0 <= requested < length:
        raise ValueError(f"Requested index {requested} is outside [0, {length})")
    return (requested,)


def replay_window(
    config: dict[str, Any],
    *,
    rollout: int,
    step_start: int,
    step_end: int,
    candidate_index: int | None = None,
    replication_index: int | None = None,
) -> dict[str, int]:
    max_steps = int(config["max_steps"])
    rollout_count = int(config["state_probe_rollouts"])
    if not 0 <= rollout < rollout_count:
        raise ValueError(f"rollout must be in [0, {rollout_count})")
    if not 0 <= step_start <= step_end < max_steps:
        raise ValueError(f"step window must be within [0, {max_steps})")

    env = build_env(load_env_config(config), seed=int(config["seed"]))
    anchor = make_anchor(config, env)
    state = env.reset(seed=int(config["seed"]) + rollout)
    anchor.reset()
    decisions = 0
    lookahead_transitions = 0

    for step in range(step_end + 1):
        assert_registry_types(
            env,
            DiagnosticLocation(rollout, step, -1, -1, -1, "live_before_decision"),
        )
        specs = candidate_action_specs(
            state,
            env,
            anchor,
            epsilons=config["epsilons"],
            candidate_groups=config["candidate_groups"],
            candidate_signs=config.get("candidate_signs", (-1.0, 1.0)),
        )
        if step >= step_start:
            decision_index = rollout * max_steps + step
            seeds = lookahead_rollout_seeds(config, decision_index)
            for candidate in selected_indices(len(specs), candidate_index):
                spec = specs[candidate]
                for replication in selected_indices(
                    len(seeds), replication_index
                ):
                    location = DiagnosticLocation(
                        rollout,
                        step,
                        candidate,
                        replication,
                        -1,
                        "source_before_deepcopy",
                    )
                    assert_registry_types(env, location)
                    rollout_env = copy.deepcopy(env)
                    assert_registry_types(
                        rollout_env,
                        DiagnosticLocation(
                            rollout,
                            step,
                            candidate,
                            replication,
                            -1,
                            "clone_after_deepcopy",
                        ),
                    )
                    rollout_env.rng = np.random.default_rng(seeds[replication])
                    rollout_state = state
                    done = False
                    for lookahead_step in range(int(config["lookahead"])):
                        action = (
                            spec["action"]
                            if lookahead_step == 0
                            else anchor.select_action(
                                rollout_state,
                                explore=False,
                                env=rollout_env,
                            )
                        )
                        assert_registry_types(
                            rollout_env,
                            DiagnosticLocation(
                                rollout,
                                step,
                                candidate,
                                replication,
                                lookahead_step,
                                "before_env_step",
                            ),
                        )
                        try:
                            rollout_state, _reward, done, _info = (
                                rollout_env.step(action)
                            )
                        except Exception as exc:
                            invalid = invalid_registry_values(rollout_env)
                            context = {
                                "location": asdict(
                                    DiagnosticLocation(
                                        rollout,
                                        step,
                                        candidate,
                                        replication,
                                        lookahead_step,
                                        "env_step_exception",
                                    )
                                ),
                                "environment_time": int(
                                    getattr(rollout_env, "t", -1)
                                ),
                                "invalid_registry_values": invalid,
                                "exception_type": type(exc).__name__,
                                "exception": str(exc),
                            }
                            raise RuntimeError(
                                "LOOKAHEAD_DIAGNOSTIC_FAILURE "
                                + json.dumps(context, sort_keys=True)
                            ) from exc
                        lookahead_transitions += 1
                        assert_registry_types(
                            rollout_env,
                            DiagnosticLocation(
                                rollout,
                                step,
                                candidate,
                                replication,
                                lookahead_step,
                                "after_env_step",
                            ),
                        )
                        if done:
                            break
                    decisions += 1
            print(
                "patient_registry_lookahead_pass "
                f"rollout={rollout} step={step} candidates={len(specs)} "
                f"crn={len(seeds)}",
                flush=True,
            )

        state, _reward, done, _info = env.step(specs[0]["action"])
        if done and step < step_end:
            raise RuntimeError(
                f"Live rollout ended at step {step} before requested step {step_end}"
            )

    return {
        "rollout": rollout,
        "step_start": step_start,
        "step_end": step_end,
        "evaluated_candidate_replications": decisions,
        "lookahead_transitions": lookahead_transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rollout", type=int, required=True)
    parser.add_argument("--step-start", type=int, required=True)
    parser.add_argument("--step-end", type=int, required=True)
    parser.add_argument("--candidate-index", type=int, default=None)
    parser.add_argument("--replication-index", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    result = replay_window(
        config,
        rollout=args.rollout,
        step_start=args.step_start,
        step_end=args.step_end,
        candidate_index=args.candidate_index,
        replication_index=args.replication_index,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
