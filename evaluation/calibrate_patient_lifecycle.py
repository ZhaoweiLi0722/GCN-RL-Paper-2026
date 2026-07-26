"""Calibrate patient deterioration and patient-facing objective weights.

The calibration uses deterministic heuristic policies so that clinical-risk
parameters can be screened before spending compute on learned policies. It
reports patient ineligibility during manufacturing explicitly; this is distinct
from a technical product-manufacturing failure.
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.evaluate_formal import evaluate_agent
from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env


def scaled_patient_config(
    reference_patient: dict[str, Any],
    scaled_fields: list[str],
    scale: float,
) -> dict[str, Any]:
    """Return a patient config with selected decay fields scaled."""

    patient = copy.deepcopy(reference_patient)
    for field in scaled_fields:
        if field not in patient:
            raise KeyError(f"Scaled patient field is missing: {field}")
        patient[field] = float(patient[field]) * float(scale)
    return patient


def reweighted_cost_mean(
    rows: list[dict[str, Any]],
    cost_profile: dict[str, float],
) -> float:
    """Recompute mean objective cost without rerunning fixed heuristic actions."""

    values = [
        float(row["base_cost"])
        + float(cost_profile["weight_patient_lost"]) * float(row["patients_lost"])
        + float(cost_profile["weight_expiry"]) * float(row["material_wasted"])
        + float(cost_profile["weight_urgency"]) * float(row["at_risk_unserved"])
        for row in rows
    ]
    return float(np.mean(values))


def calibrate(plan: dict[str, Any], *, smoke: bool = False) -> list[dict[str, Any]]:
    """Evaluate all configured scale-policy pairs."""

    env_config = load_config(plan["env_config"])
    algorithms = list(plan["algorithms"])
    scales = [float(value) for value in plan["decay_scales"]]
    replications = int(plan["replications"])
    if smoke:
        algorithms = algorithms[:1]
        scales = scales[:1]
        replications = min(replications, 2)

    target_low, target_high = (
        float(value) for value in plan["target_manufacturing_ineligibility_rate"]
    )
    seed = int(plan["seed"])
    results: list[dict[str, Any]] = []
    for scale in scales:
        candidate_env = copy.deepcopy(env_config)
        candidate_env["patient"] = scaled_patient_config(
            plan["reference_patient"],
            list(plan["scaled_patient_fields"]),
            scale,
        )
        for algorithm in algorithms:
            config = {"algorithm": algorithm, "env": candidate_env}
            env = build_env(config, seed=seed)
            agent_cls = get_agent_class(algorithm)
            agent = agent_cls(env.observation_size, env.action_size, config)
            rows = evaluate_agent(
                agent,
                env,
                algorithm=algorithm,
                seed=seed,
                replications=replications,
                max_steps=env.config.episode_horizon,
            )
            manufacturing_rate = float(
                np.mean(
                    [
                        float(row["patient_ineligibility_during_manufacturing_rate"])
                        for row in rows
                    ]
                )
            )
            result = {
                "decay_scale": scale,
                "algorithm": algorithm,
                "replications": replications,
                "patient_ineligibility_during_manufacturing_rate": manufacturing_rate,
                "within_target_band": target_low <= manufacturing_rate <= target_high,
                "patients_lost_mean": float(
                    np.mean([float(row["patients_lost"]) for row in rows])
                ),
                "therapies_discarded_mean": float(
                    np.mean([float(row["therapies_discarded"]) for row in rows])
                ),
                "completion_service_level_mean": float(
                    np.mean([float(row["completion_service_level"]) for row in rows])
                ),
                "base_cost_mean": float(
                    np.mean([float(row["base_cost"]) for row in rows])
                ),
            }
            for profile_name, profile in plan["cost_profiles"].items():
                result[f"{profile_name}_total_cost_mean"] = reweighted_cost_mean(
                    rows, profile
                )
            results.append(result)
    return results


def write_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="experiments/configs/patient_lifecycle_calibration.json",
    )
    parser.add_argument(
        "--output",
        default="results/patient_lifecycle_calibration/summary.csv",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    rows = calibrate(load_config(args.plan), smoke=args.smoke)
    write_rows(rows, args.output)
    for row in rows:
        print(
            f"scale={row['decay_scale']:.3f} "
            f"algorithm={row['algorithm']} "
            "manufacturing_ineligibility="
            f"{row['patient_ineligibility_during_manufacturing_rate']:.4f} "
            f"target={row['within_target_band']}"
        )
    print(f"wrote {len(rows)} calibration rows to {args.output}")


if __name__ == "__main__":
    main()
