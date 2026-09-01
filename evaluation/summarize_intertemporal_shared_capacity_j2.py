"""Post-hoc diagnostic for the closed intertemporal-capacity J2 screen."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.screen_intertemporal_shared_capacity_episodes import sha256


DEFAULT_ROOT = Path(
    "results/intertemporal_shared_capacity_development/fixed_state_j2"
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    numeric = {
        "state_index": int,
        "generation_seed": int,
        "epoch": int,
        "world_seed": int,
        "total_cost": float,
    }
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key, converter in numeric.items():
                row[key] = converter(row[key])
            rows.append(row)
    return rows


def mean_costs(
    rows: list[dict[str, Any]], phase: str
) -> dict[tuple[int, str], float]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["phase"] == phase:
            grouped[(row["state_index"], row["candidate"])].append(
                row["total_cost"]
            )
    return {
        key: sum(values) / len(values)
        for key, values in grouped.items()
    }


def best_by_state(
    costs: dict[tuple[int, str], float], candidates: tuple[str, ...]
) -> dict[int, str]:
    states = sorted({state for state, _ in costs})
    return {
        state: min(
            candidates,
            key=lambda candidate: (costs[(state, candidate)], candidate),
        )
        for state in states
    }


def comparison(
    costs: dict[tuple[int, str], float],
    *,
    choices: dict[int, str],
    comparator: str,
    metadata: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {
        "overall": sorted(choices),
    }
    for state, item in metadata.items():
        groups.setdefault(f"scenario:{item['scenario']}", []).append(state)
        groups.setdefault(f"epoch:{item['epoch']}", []).append(state)

    def summarize(states: list[int]) -> dict[str, Any]:
        comparator_cost = sum(costs[(state, comparator)] for state in states)
        treatment_cost = sum(
            costs[(state, choices[state])] for state in states
        )
        state_savings = [
            (costs[(state, comparator)] - costs[(state, choices[state])])
            / costs[(state, comparator)]
            for state in states
        ]
        return {
            "state_count": len(states),
            "relative_saving": (
                comparator_cost - treatment_cost
            ) / comparator_cost,
            "positive_states": sum(value > 0.0 for value in state_savings),
            "material_states_at_0_005": sum(
                value >= 0.005 for value in state_savings
            ),
            "minimum_state_saving": min(state_savings),
            "maximum_state_saving": max(state_savings),
        }

    return {
        key: summarize(states)
        for key, states in sorted(groups.items())
    }


def audit_rng_and_counts(
    rows: list[dict[str, Any]], candidate_count: int
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], row["state_index"], row["world_seed"])].append(
            row
        )
    bad_count = 0
    bad_rng = 0
    for group in grouped.values():
        bad_count += int(len(group) != candidate_count)
        bad_rng += int(len({row["rng_sha256"] for row in group}) != 1)
    return {
        "paired_world_count": len(grouped),
        "candidate_count_mismatch_worlds": bad_count,
        "rng_mismatch_worlds": bad_rng,
        "passes": bad_count == 0 and bad_rng == 0,
    }


def run(root: Path, output: Path) -> dict[str, Any]:
    rows_path = root / "headroom_rows.csv"
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["decision"] != "fixed_state_j2_failed":
        raise ValueError("diagnostic is restricted to the closed J2 result")
    if sha256(rows_path) != summary["provenance"]["rows_sha256"]:
        raise ValueError("J2 rows hash mismatch")

    rows = read_rows(rows_path)
    candidates = tuple(sorted({str(row["candidate"]) for row in rows}))
    metadata = {
        int(item["state_index"]): {
            "scenario": str(item["scenario"]),
            "epoch": int(item["state_id"].rsplit("t", 1)[1]),
        }
        for item in summary["state_results"]
    }
    discovery_costs = mean_costs(rows, "discovery")
    validation_costs = mean_costs(rows, "validation")
    discovery_choices = {
        int(item["state_index"]): str(item["selected_candidate"])
        for item in summary["state_results"]
    }
    validation_oracle_choices = best_by_state(validation_costs, candidates)
    comparator = str(summary["global_candidate"])
    prospective = comparison(
        validation_costs,
        choices=discovery_choices,
        comparator=comparator,
        metadata=metadata,
    )
    optimistic_oracle = comparison(
        validation_costs,
        choices=validation_oracle_choices,
        comparator=comparator,
        metadata=metadata,
    )
    discovery_in_sample = comparison(
        discovery_costs,
        choices=discovery_choices,
        comparator=comparator,
        metadata=metadata,
    )
    agreement = sum(
        discovery_choices[state] == validation_oracle_choices[state]
        for state in discovery_choices
    ) / len(discovery_choices)
    oracle_saving = optimistic_oracle["overall"]["relative_saving"]
    prospective_saving = prospective["overall"]["relative_saving"]
    if oracle_saving < 0.005:
        diagnosis = "candidate_library_saturated_even_under_optimistic_oracle"
    elif prospective_saving < 0.005:
        diagnosis = "discovery_action_labels_do_not_transfer"
    else:
        diagnosis = "prospective_residual_headroom_present"

    result = {
        "name": "intertemporal_shared_capacity_j2_posthoc_diagnostic",
        "experimental_role": (
            "Post-hoc decomposition of a prospectively failed gate; not a new "
            "gate, training authorization, or formal result."
        ),
        "j2_decision": summary["decision"],
        "diagnosis": diagnosis,
        "global_candidate": comparator,
        "selection_agreement_with_validation_oracle": agreement,
        "discovery_selected_candidate_counts": dict(
            sorted(Counter(discovery_choices.values()).items())
        ),
        "validation_oracle_candidate_counts": dict(
            sorted(Counter(validation_oracle_choices.values()).items())
        ),
        "prospective_validation": prospective,
        "optimistic_validation_oracle": optimistic_oracle,
        "discovery_in_sample": discovery_in_sample,
        "paired_rng_audit": audit_rng_and_counts(
            rows, int(summary["candidate_count"])
        ),
        "policy_training_performed": False,
        "formal_confirmation_performed": False,
        "j3_authorized": False,
        "training_authorized": False,
        "provenance": {
            "j2_summary": str(summary_path),
            "j2_summary_sha256": sha256(summary_path),
            "j2_rows": str(rows_path),
            "j2_rows_sha256": sha256(rows_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "posthoc_diagnostic.json"
    run(args.root, output)


if __name__ == "__main__":
    main()
