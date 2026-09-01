"""Unit tests for the episode-level intertemporal capacity pre-screen."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.screen_intertemporal_shared_capacity_episodes import (
    assert_rng_alignment,
    candidate_summary,
    load_json,
    run_episode,
    select_values,
    validate_config,
    write_rows,
)
from evaluation.screen_intertemporal_shared_capacity_states import (
    candidate_specs,
    select_candidates,
    validation_summary,
)
from evaluation.summarize_intertemporal_shared_capacity_j2 import (
    best_by_state,
    comparison,
)


class EpisodeScreenTest(unittest.TestCase):
    def test_rng_alignment_checks_each_phase_scenario_seed_group(self) -> None:
        rows = [
            {
                "phase": "discovery",
                "scenario": "a",
                "seed": 1,
                "rng_sha256": "same",
            },
            {
                "phase": "discovery",
                "scenario": "a",
                "seed": 1,
                "rng_sha256": "same",
            },
        ]
        self.assertEqual(assert_rng_alignment(rows), 1)
        rows[-1]["rng_sha256"] = "different"
        with self.assertRaises(RuntimeError):
            assert_rng_alignment(rows)

    def test_select_values_uses_discovery_mean(self) -> None:
        rows = []
        for family in ("static_ot", "forecast_iot", "graph_forecast_iot"):
            rows.extend(
                [
                    {"family": family, "tuning_value": 0.0, "total_cost": 10.0},
                    {"family": family, "tuning_value": 0.0, "total_cost": 12.0},
                    {"family": family, "tuning_value": 0.5, "total_cost": 8.0},
                    {"family": family, "tuning_value": 0.5, "total_cost": 9.0},
                ]
            )
        self.assertEqual(
            select_values(rows),
            {
                "static_ot": 0.5,
                "forecast_iot": 0.5,
                "graph_forecast_iot": 0.5,
            },
        )

    def test_candidate_summary_clusters_interval_by_seed(self) -> None:
        rows = []
        for scenario in ("a", "b"):
            for seed in (1, 2):
                rows.append(
                    {
                        "family": "static_ot",
                        "scenario": scenario,
                        "seed": seed,
                        "total_cost": 100.0,
                        "patients_lost": 5.0,
                        "completion_service_level": 0.8,
                    }
                )
                rows.append(
                    {
                        "family": "forecast_iot",
                        "scenario": scenario,
                        "seed": seed,
                        "total_cost": 90.0,
                        "patients_lost": 4.0,
                        "completion_service_level": 0.82,
                    }
                )
        summary = candidate_summary(
            rows,
            candidate="forecast_iot",
            scenarios=("a", "b"),
            validation_seeds=(1, 2),
        )
        self.assertAlmostEqual(summary["pooled_seed_clustered"]["mean"], 0.1)
        self.assertEqual(summary["pooled_seed_clustered"]["count"], 2)
        self.assertTrue(summary["clinical"]["noninferior"])

    def test_validate_rejects_overlapping_seed_families(self) -> None:
        config = {
            "discovery_seeds": {"start": 10, "count": 2},
            "validation_seeds": {"start": 11, "count": 2},
            "scenarios": [{}, {}, {}],
            "tuning_grids": {
                "static_ot": [0.0],
                "forecast_iot": [0.0],
                "graph_forecast_iot": [0.0],
            },
            "common_env_overrides": {
                "enable_scheduled_referral_waves": True,
                "enable_overtime_control": True,
                "enable_intertemporal_overtime_commitment": True,
            },
        }
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_real_patient_episode_and_lf_only_csv(self) -> None:
        env = load_json(Path("experiments/configs/20_clinic_patient_condition_geo.json"))
        env.update(
            {
                "scenario_name": "unit_smoke",
                "episode_horizon": 3,
                "include_demand_forecast_state": True,
                "demand_forecast_horizon": 2,
                "enable_scheduled_referral_waves": True,
                "scheduled_referral_clusters": [[0, 1, 2, 3, 4]],
                "scheduled_referral_block_length": 2,
                "scheduled_referral_peak_multiplier": 1.5,
                "enable_overtime_control": True,
                "enable_intertemporal_overtime_commitment": True,
                "overtime_commitment_lead_time": 1,
                "overtime_commitment_persistence": 0.5,
                "overtime_shared_budget_fraction": 0.25,
            }
        )
        row = run_episode(
            env_config=env,
            scenario="unit_smoke",
            seed=123,
            family="forecast_iot",
            tuning_value=0.5,
            phase="test",
        )
        self.assertEqual(row["seed"], 123)
        self.assertGreater(row["total_cost"], 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            write_rows([row], path)
            self.assertNotIn(b"\r\n", path.read_bytes())


class FixedStateScreenTest(unittest.TestCase):
    def test_candidate_specs_validate_library_size_and_bounds(self) -> None:
        config = {
            "action_library": {
                "forecast_overtime_budget_fractions": [0.75, 1.0],
                "graph_forecast_smoothing": [0.0, 0.5],
            },
            "gate": {"minimum_distinct_selected_actions": 3},
        }
        self.assertEqual(len(candidate_specs(config)), 4)
        config["action_library"]["graph_forecast_smoothing"] = [1.1]
        with self.assertRaises(ValueError):
            candidate_specs(config)

    def test_selection_is_discovery_only_and_breaks_ties_stably(self) -> None:
        candidates = (
            {"candidate": "a"},
            {"candidate": "b"},
        )
        rows = [
            {"state_index": 0, "candidate": "a", "total_cost": 10.0},
            {"state_index": 0, "candidate": "b", "total_cost": 9.0},
            {"state_index": 1, "candidate": "a", "total_cost": 8.0},
            {"state_index": 1, "candidate": "b", "total_cost": 9.0},
        ]
        global_candidate, selected = select_candidates(rows, candidates)
        self.assertEqual(global_candidate, "a")
        self.assertEqual(selected, {0: "b", 1: "a"})

    def test_material_and_distinct_gates_use_relative_realized_actions(self) -> None:
        candidates = (
            {"candidate": "global", "budget_fraction": 1.0},
            {"candidate": "selected", "budget_fraction": 0.75},
        )
        rows = []
        for state_index, signature in enumerate(("0.2,0.8", "0.4,0.6", "0.6,0.4")):
            common = {
                "state_index": state_index,
                "state_id": f"s{state_index}",
                "scenario": "wave",
                "generation_seed": state_index,
                "patients_lost": 2.0,
                "completion_service_level": 0.8,
                "local_interior_fraction": 1.0,
                "shared_budget_interior": True,
            }
            rows.append(
                {
                    **common,
                    "candidate": "global",
                    "total_cost": 100.0,
                    "allocation_signature": "0.5,0.5",
                }
            )
            rows.append(
                {
                    **common,
                    "candidate": "selected",
                    "total_cost": 99.0,
                    "allocation_signature": signature,
                }
            )
        summary = validation_summary(
            rows,
            global_candidate="global",
            selected={0: "selected", 1: "selected", 2: "selected"},
            candidates=candidates,
            behavioral_sensitivity_fraction=1.0,
            gate={
                "minimum_behavioral_sensitivity_fraction": 0.9,
                "minimum_material_state_fraction": 0.3,
                "minimum_relative_saving": 0.005,
                "minimum_interior_selected_fraction": 0.3,
                "minimum_distinct_selected_actions": 3,
                "material_relative_saving": 0.005,
                "require_clinical_noninferiority": True,
            },
        )
        self.assertTrue(summary["passes"])
        self.assertEqual(summary["distinct_selected_actions"], 3)
        self.assertEqual(summary["material_state_fraction"], 1.0)

    def test_posthoc_comparison_separates_prospective_and_oracle_choices(self) -> None:
        costs = {
            (0, "a"): 100.0,
            (0, "b"): 90.0,
            (1, "a"): 100.0,
            (1, "b"): 110.0,
        }
        metadata = {
            0: {"scenario": "x", "epoch": 4},
            1: {"scenario": "x", "epoch": 8},
        }
        oracle = best_by_state(costs, ("a", "b"))
        self.assertEqual(oracle, {0: "b", 1: "a"})
        result = comparison(
            costs,
            choices=oracle,
            comparator="a",
            metadata=metadata,
        )
        self.assertAlmostEqual(result["overall"]["relative_saving"], 0.05)
        self.assertEqual(result["overall"]["positive_states"], 1)


if __name__ == "__main__":
    unittest.main()
