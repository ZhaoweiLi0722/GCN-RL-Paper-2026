import copy
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evaluation.compare_ddpg_persistent_shift import (
    VARIANT_EPISODES,
    compare_persistent_shift,
)
from evaluation.evaluate_fixed_checkpoint_curve import (
    evaluate_checkpoint_curve,
)
from evaluation.prepare_ddpg_persistent_shift_preonline import (
    prepare_preonline_states,
)
from evaluation.run_patient_indexed_specimen_routing_ddpg_persistent_shift import (
    ALGORITHMS,
    artifact_inventory,
    clone_paired_states,
    validate_scientific_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "experiments" / "configs"
CONTROL_TRAINING = CONFIG_ROOT / (
    "patient_indexed_specimen_routing_ddpg_persistent_shift_control_100.json"
)
REALIGNED_TRAINING = CONFIG_ROOT / (
    "patient_indexed_specimen_routing_ddpg_persistent_shift_realigned_100.json"
)
CONTROL_EVALUATION = CONFIG_ROOT / (
    "patient_indexed_specimen_routing_ddpg_persistent_shift_control_curve_eval.json"
)
REALIGNED_EVALUATION = CONFIG_ROOT / (
    "patient_indexed_specimen_routing_ddpg_persistent_shift_realigned_curve_eval.json"
)


class PersistentShiftContractTests(unittest.TestCase):
    def test_hotspot_maps_are_static_and_preserve_expected_demand(self) -> None:
        plan = json.loads(
            (CONFIG_ROOT / "patient_indexed_specimen_routing_benchmark.json")
            .read_text(encoding="utf-8")
        )
        demand = json.loads(
            (CONFIG_ROOT / "20_clinic_patient_condition_geo_regional_drift.json")
            .read_text(encoding="utf-8")
        )["demand_rates"]
        scenarios = {
            scenario["name"]: scenario["env_overrides"]
            for scenario in plan["scenarios"]
            if scenario["name"].startswith("routing_persistent_hotspot_")
        }
        expected_hotspots = {
            "routing_persistent_hotspot_cluster1": set(range(5, 10)),
            "routing_persistent_hotspot_cluster2": set(range(10, 15)),
            "routing_persistent_hotspot_cluster3": set(range(15, 20)),
        }

        self.assertEqual(set(scenarios), set(expected_hotspots))
        baseline_total = sum(float(value) for value in demand)
        for name, overrides in scenarios.items():
            initial = overrides["demand_regime_initial_multipliers"]
            final = overrides["demand_regime_final_multipliers"]
            self.assertEqual(initial, final)
            self.assertEqual(overrides["demand_regime_change_step"], 0)
            self.assertEqual(overrides["demand_regime_transition_duration"], 0)
            self.assertEqual(
                {index for index, value in enumerate(initial) if value == 1.4},
                expected_hotspots[name],
            )
            shifted_total = sum(
                float(rate) * float(multiplier)
                for rate, multiplier in zip(demand, initial)
            )
            self.assertAlmostEqual(shifted_total, baseline_total, places=10)

    def test_locked_training_and_evaluation_contracts_are_single_factor(self) -> None:
        validate_scientific_contract(
            {
                "control_training_config": str(CONTROL_TRAINING.relative_to(ROOT)),
                "realigned_training_config": str(
                    REALIGNED_TRAINING.relative_to(ROOT)
                ),
                "control_evaluation_config": str(
                    CONTROL_EVALUATION.relative_to(ROOT)
                ),
                "realigned_evaluation_config": str(
                    REALIGNED_EVALUATION.relative_to(ROOT)
                ),
                "development_crn_seeds": [95_200_000, 95_300_000],
                "forbidden_crn_seeds": [91_100_000, 93_100_000],
                "minimum_final_improvement_pct": 0.02,
            }
        )

    def test_checkpoint_curve_uses_each_fixed_variant_once(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "curve.json"
            output_root = root / "outputs"
            config_path.write_text(
                json.dumps(
                    {
                        "name": "curve_test",
                        "checkpoint_variants": ["pretrain", "final"],
                        "validation_seed": 1,
                        "holdout_seed": 2,
                        "holdout_replications": 1,
                        "output_root": str(output_root),
                    }
                ),
                encoding="utf-8",
            )
            observed = []

            def fake_evaluate(config):
                observed.append(copy.deepcopy(config))
                phase_root = Path(config["output_root"])
                phase_root.mkdir(parents=True)
                (phase_root / "summary.json").write_text(
                    json.dumps({"variant": config["fixed_checkpoint_variant"]}),
                    encoding="utf-8",
                )

            with patch(
                "evaluation.evaluate_fixed_checkpoint_curve."
                "evaluate_multiscenario_agents",
                side_effect=fake_evaluate,
            ):
                summary = evaluate_checkpoint_curve(config_path)

            self.assertEqual(
                [entry["fixed_checkpoint_variant"] for entry in observed],
                ["pretrain", "final"],
            )
            self.assertEqual(
                [entry["checkpoint_variants"] for entry in observed],
                [["pretrain"], ["final"]],
            )
            self.assertEqual(len(summary["summaries"]), 2)

    def test_preonline_preparation_runs_all_pairs_without_online_work(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "source.json"
            config_path.write_text(
                json.dumps(
                    {
                        "online_episodes": 100,
                        "config_overrides": {
                            "critic_teacher_advantage_calibration": {
                                "enabled": True,
                                "updates": 100,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            calls = []

            def fake_train(config, **filters):
                calls.append((copy.deepcopy(config), dict(filters)))
                return {"runs": [filters]}

            with patch(
                "evaluation.prepare_ddpg_persistent_shift_preonline."
                "train_multiscenario_agents",
                side_effect=fake_train,
            ):
                result = prepare_preonline_states(
                    config_path,
                    output_root=root / "preonline",
                )

            self.assertEqual(len(calls), 6)
            self.assertEqual(
                {
                    (filters["algorithm_filter"], filters["seed_filter"])
                    for _config, filters in calls
                },
                {
                    (algorithm, seed)
                    for algorithm in ALGORITHMS
                    for seed in (40, 41, 42)
                },
            )
            for config, _filters in calls:
                self.assertEqual(config["online_episodes"], 0)
                self.assertEqual(
                    config["config_overrides"][
                        "critic_teacher_advantage_calibration"
                    ]["updates"],
                    0,
                )
                self.assertIn(
                    "preonline_training_state_path",
                    config["config_overrides"],
                )
            self.assertIn("runs", result)

    def test_paired_clones_change_contract_metadata_only(self) -> None:
        import torch

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_runs = []
            sources = {}
            for algorithm in ALGORITHMS:
                for seed in (40, 41, 42):
                    source = root / "sources" / f"{algorithm}_{seed}.pt"
                    source.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint = {
                        "training": {"next_episode": 0},
                        "training_contract": {
                            "num_episodes": 0,
                            "history_screen": {
                                "online_episodes": 0,
                                "pretrain_only": True,
                            },
                            "critic_teacher_advantage_calibration": {
                                "enabled": True,
                                "updates": 0,
                            },
                            "online_critic_realignment": {
                                "enabled": False,
                                "mode": "zero_action_columns",
                                "actor_warmup_updates": 0,
                            },
                        },
                        "agent": {
                            "tensor": torch.tensor([seed], dtype=torch.float32),
                            "nested": {"value": [1, 2, 3]},
                        },
                        "environment": {"episode": 0, "seed": seed},
                    }
                    torch.save(checkpoint, source)
                    source_runs.append(
                        {
                            "algorithm": algorithm,
                            "seed": seed,
                            "preonline_training_state": str(source),
                        }
                    )
                    sources[(algorithm, seed)] = checkpoint
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"runs": source_runs}),
                encoding="utf-8",
            )

            provenance = clone_paired_states(
                manifest,
                control_config_path=CONTROL_TRAINING,
                realigned_config_path=REALIGNED_TRAINING,
                output_root=root / "paired",
            )

            self.assertEqual(len(provenance["runs"]), 6)
            for entry in provenance["runs"]:
                key = (entry["algorithm"], entry["seed"])
                source = sources[key]
                for role, warmup in (("control", 0), ("realigned", 500)):
                    clone = torch.load(
                        entry[f"{role}_clone"],
                        map_location="cpu",
                        weights_only=False,
                    )
                    self.assertTrue(
                        torch.equal(
                            clone["agent"]["tensor"],
                            source["agent"]["tensor"],
                        )
                    )
                    self.assertEqual(clone["environment"], source["environment"])
                    self.assertEqual(clone["training_contract"]["num_episodes"], 100)
                    realignment = clone["training_contract"][
                        "online_critic_realignment"
                    ]
                    self.assertEqual(
                        realignment["enabled"],
                        role == "realigned",
                    )
                    self.assertEqual(
                        realignment["actor_warmup_updates"],
                        warmup,
                    )

    def test_comparator_accepts_large_csv_field_and_emits_decision(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            assignment = {
                "40": "routing_persistent_hotspot_cluster1",
                "41": "routing_persistent_hotspot_cluster2",
                "42": "routing_persistent_hotspot_cluster3",
            }
            configs = {}
            for role in ("control", "realigned"):
                output_root = root / role
                config = {
                    "name": role,
                    "algorithms": list(ALGORITHMS),
                    "training_seeds": [40, 41, 42],
                    "checkpoint_variants": list(VARIANT_EPISODES),
                    "scenarios": list(assignment.values()),
                    "scenario_by_training_seed": assignment,
                    "validation_seed": 95_200_000,
                    "holdout_seed": 95_300_000,
                    "holdout_replications": 2,
                    "max_steps": 52,
                    "fixed_deployment_candidate": {
                        "scale": 1.0,
                        "use_checkpoint_group_thresholds": True,
                    },
                    "clinical_noninferiority": {
                        "margins": {
                            "completion_service_level": 0.001,
                            "patients_lost": 1.0,
                            "patient_ineligibility_during_manufacturing_rate": 0.001,
                        }
                    },
                    "output_root": str(output_root),
                }
                training_manifest = root / f"{role}_training_manifest.json"
                training_manifest.write_text(
                    json.dumps(
                        {
                            "runs": [
                                {
                                    "algorithm": algorithm,
                                    "seed": seed,
                                    "actor_drift_from_pretrain": {
                                        "rms": 0.01,
                                        "max_abs": 0.02,
                                        "parameter_count": 100,
                                    },
                                }
                                for algorithm in ALGORITHMS
                                for seed in (40, 41, 42)
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                config["training_manifest"] = str(training_manifest)
                path = root / f"{role}.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                configs[role] = path
                self._write_synthetic_curve(
                    output_root,
                    role=role,
                    assignment=assignment,
                )

            output = root / "comparison" / "summary.json"
            result = compare_persistent_shift(
                configs["control"],
                configs["realigned"],
                output_path=output,
                resamples=100,
                bootstrap_seed=123,
            )

            self.assertEqual(
                result["decision"]["classification"],
                "advance_realigned_ddpg_to_fresh_confirmation",
            )
            self.assertTrue(result["decision"]["realigned_gate_passed"])
            self.assertIn("diagnostics", result)
            self.assertEqual(
                result["diagnostics"]["realigned"][
                    "actor_drift_from_pretrain"
                ][ALGORITHMS[0]]["40"]["rms"],
                0.01,
            )
            self.assertGreater(output.stat().st_size, 0)
            inventory = artifact_inventory(root)
            self.assertIn(str(output), inventory)

    def _write_synthetic_curve(
        self,
        output_root: Path,
        *,
        role: str,
        assignment: dict[str, str],
    ) -> None:
        final_cost = 990.0 if role == "control" else 980.0
        costs = {
            "pretrain": 1000.0,
            "episode10": 998.0 if role == "control" else 997.0,
            "episode25": 996.0 if role == "control" else 994.0,
            "episode50": 994.0 if role == "control" else 990.0,
            "episode75": 992.0 if role == "control" else 985.0,
            "final": final_cost,
        }
        fieldnames = (
            "training_seed",
            "scenario",
            "evaluation_seed",
            "replication",
            "total_cost",
            "completion_service_level",
            "patients_lost",
            "patient_ineligibility_during_manufacturing_rate",
            "diagnostic_payload",
        )
        large_field_written = False
        for variant in VARIANT_EPISODES:
            for algorithm in ALGORITHMS:
                for seed in (40, 41, 42):
                    path = (
                        output_root
                        / variant
                        / algorithm
                        / f"seed{seed}"
                        / "holdout_rows.csv"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fieldnames)
                        writer.writeheader()
                        for replication in range(2):
                            payload = "small"
                            if not large_field_written:
                                payload = "x" * 200_000
                                large_field_written = True
                            writer.writerow(
                                {
                                    "training_seed": seed,
                                    "scenario": assignment[str(seed)],
                                    "evaluation_seed": 95_300_000,
                                    "replication": replication,
                                    "total_cost": costs[variant],
                                    "completion_service_level": 0.95,
                                    "patients_lost": 2.0,
                                    "patient_ineligibility_during_manufacturing_rate": 0.01,
                                    "diagnostic_payload": payload,
                                }
                            )
                    (path.parent / "summary.json").write_text(
                        json.dumps(
                            {
                                "holdout": {
                                    "aggregate": {
                                        "residual_usage": {
                                            "corrected_decisions": 10,
                                            "correction_rate": 0.1,
                                            "applied_residual_l1": 1.0,
                                        }
                                    }
                                }
                            }
                        ),
                        encoding="utf-8",
                    )


if __name__ == "__main__":
    unittest.main()
