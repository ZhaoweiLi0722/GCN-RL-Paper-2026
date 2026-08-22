import copy
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evaluation.compare_ddpg_structured_exploration import (
    compare_structured_exploration,
)
from evaluation.compare_ddpg_persistent_shift import VARIANT_EPISODES
from evaluation.prepare_ddpg_structured_exploration_preonline import (
    prepare_preonline_states,
)
from evaluation.run_patient_indexed_specimen_routing_ddpg_structured_exploration import (
    ALGORITHMS,
    clone_paired_states,
    validate_scientific_contract,
    verify_locked_assets,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "experiments" / "configs"
CONTROL_TRAINING = CONFIG_ROOT / (
    "patient_indexed_specimen_routing_ddpg_structured_exploration_"
    "control_100.json"
)
CANDIDATE_TRAINING = CONFIG_ROOT / (
    "patient_indexed_specimen_routing_ddpg_structured_exploration_"
    "candidate_100.json"
)
CONTROL_EVALUATION = CONFIG_ROOT / (
    "patient_indexed_specimen_routing_ddpg_structured_exploration_"
    "control_curve_eval.json"
)
CANDIDATE_EVALUATION = CONFIG_ROOT / (
    "patient_indexed_specimen_routing_ddpg_structured_exploration_"
    "candidate_curve_eval.json"
)
EXECUTION_SPEC = CONFIG_ROOT / (
    "patient_indexed_specimen_routing_ddpg_structured_exploration_"
    "execution.json"
)


class StructuredExplorationStageC3Tests(unittest.TestCase):
    def test_execution_spec_locks_assets_without_self_reference(self) -> None:
        spec = json.loads(EXECUTION_SPEC.read_text(encoding="utf-8"))
        validate_scientific_contract(spec)
        verified = verify_locked_assets(spec)
        locked_paths = {str(entry["path"]) for entry in spec["locked_files"]}
        self.assertGreaterEqual(len(verified), 40)
        self.assertNotIn(str(EXECUTION_SPEC.relative_to(ROOT)), locked_paths)

    def test_locked_contract_is_single_factor(self) -> None:
        validate_scientific_contract(
            {
                "control_training_config": str(
                    CONTROL_TRAINING.relative_to(ROOT)
                ),
                "candidate_training_config": str(
                    CANDIDATE_TRAINING.relative_to(ROOT)
                ),
                "control_evaluation_config": str(
                    CONTROL_EVALUATION.relative_to(ROOT)
                ),
                "candidate_evaluation_config": str(
                    CANDIDATE_EVALUATION.relative_to(ROOT)
                ),
                "development_crn_seeds": [95_700_000, 95_800_000],
                "forbidden_crn_seeds": [91_100_000, 95_600_000],
                "minimum_final_improvement_pct": 0.02,
            }
        )

    def test_preonline_preparation_consumes_no_online_trajectory(self) -> None:
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
                            },
                            "residual_action": {
                                "structured_exploration": {
                                    "enabled": False,
                                }
                            },
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
                "evaluation.prepare_ddpg_structured_exploration_preonline."
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
                    for seed in (50, 51, 52)
                },
            )
            for config, _filters in calls:
                self.assertEqual(config["online_episodes"], 0)
                self.assertEqual(
                    config["config_overrides"]
                    ["critic_teacher_advantage_calibration"]["updates"],
                    0,
                )
                self.assertFalse(
                    config["config_overrides"]["residual_action"]
                    ["structured_exploration"]["enabled"]
                )
            self.assertIn("runs", result)

    def test_paired_clones_preserve_payload_and_defer_candidate_fork(self) -> None:
        import torch

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_runs = []
            sources = {}
            for algorithm in ALGORITHMS:
                for seed in (50, 51, 52):
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
                            "residual_action": {
                                "structured_exploration": {
                                    "enabled": False,
                                    "selection_probability": 0.2,
                                    "selection_mode": "uniform",
                                    "apply_after_correction_gate": True,
                                    "seed_offset": 684_211,
                                    "options": [],
                                }
                            },
                        },
                        "agent": {
                            "tensor": torch.tensor(
                                [seed], dtype=torch.float32
                            ),
                            "structured_specimen_explorer": {
                                "enabled": False,
                                "total_decisions": 0,
                            },
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
                candidate_config_path=CANDIDATE_TRAINING,
                output_root=root / "paired",
            )

            self.assertEqual(len(provenance["runs"]), 6)
            self.assertTrue(
                provenance["candidate_enabled_by_declared_preonline_fork"]
            )
            for entry in provenance["runs"]:
                key = (entry["algorithm"], entry["seed"])
                for role in ("control", "candidate"):
                    clone = torch.load(
                        entry[f"{role}_clone"],
                        map_location="cpu",
                        weights_only=False,
                    )
                    self.assertTrue(
                        torch.equal(
                            clone["agent"]["tensor"],
                            sources[key]["agent"]["tensor"],
                        )
                    )
                    self.assertEqual(
                        clone["environment"],
                        sources[key]["environment"],
                    )
                    self.assertFalse(
                        clone["training_contract"]["residual_action"]
                        ["structured_exploration"]["enabled"]
                    )

    def test_comparator_advances_only_improving_candidate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            assignment = {
                "50": "routing_persistent_hotspot_cluster1",
                "51": "routing_persistent_hotspot_cluster2",
                "52": "routing_persistent_hotspot_cluster3",
            }
            config_paths = {}
            for role in ("control", "candidate"):
                output_root = root / role
                manifest_path = root / f"{role}_manifest.json"
                manifest_path.write_text(
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
                                    "structured_specimen_exploration": {
                                        "enabled": role == "candidate"
                                    },
                                }
                                for algorithm in ALGORITHMS
                                for seed in (50, 51, 52)
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                config = {
                    "name": role,
                    "algorithms": list(ALGORITHMS),
                    "training_seeds": [50, 51, 52],
                    "checkpoint_variants": list(VARIANT_EPISODES),
                    "scenarios": list(assignment.values()),
                    "scenario_by_training_seed": assignment,
                    "validation_seed": 95_700_000,
                    "holdout_seed": 95_800_000,
                    "holdout_replications": 1,
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
                    "training_manifest": str(manifest_path),
                    "output_root": str(output_root),
                }
                config_path = root / f"{role}.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                config_paths[role] = config_path
                self._write_curve(
                    output_root,
                    role=role,
                    assignment=assignment,
                )

            result = compare_structured_exploration(
                config_paths["control"],
                config_paths["candidate"],
                output_path=root / "comparison" / "summary.json",
                resamples=50,
                bootstrap_seed=123,
            )

            self.assertEqual(
                result["decision"]["classification"],
                "advance_structured_exploration_ddpg_to_fresh_confirmation",
            )
            self.assertTrue(result["decision"]["candidate_gate_passed"])
            self.assertTrue(
                result["decision"]["candidate_no_regression_vs_control"]
            )

    def _write_curve(
        self,
        output_root: Path,
        *,
        role: str,
        assignment: dict[str, str],
    ) -> None:
        final_cost = 990.0 if role == "control" else 970.0
        costs = {
            "pretrain": 1000.0,
            "episode10": 998.0,
            "episode25": 995.0,
            "episode50": 993.0 if role == "control" else 985.0,
            "episode75": 991.0 if role == "control" else 977.0,
            "final": final_cost,
        }
        fields = (
            "training_seed",
            "scenario",
            "evaluation_seed",
            "replication",
            "total_cost",
            "completion_service_level",
            "patients_lost",
            "patient_ineligibility_during_manufacturing_rate",
        )
        for variant in VARIANT_EPISODES:
            for algorithm in ALGORITHMS:
                for seed in (50, 51, 52):
                    run_root = (
                        output_root
                        / variant
                        / algorithm
                        / f"seed{seed}"
                    )
                    run_root.mkdir(parents=True, exist_ok=True)
                    with (run_root / "holdout_rows.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields)
                        writer.writeheader()
                        writer.writerow(
                            {
                                "training_seed": seed,
                                "scenario": assignment[str(seed)],
                                "evaluation_seed": 95_800_000,
                                "replication": 0,
                                "total_cost": costs[variant],
                                "completion_service_level": 0.99,
                                "patients_lost": 1.0,
                                "patient_ineligibility_during_manufacturing_rate": 0.0,
                            }
                        )
                    (run_root / "summary.json").write_text(
                        json.dumps(
                            {
                                "holdout": {
                                    "aggregate": {
                                        "residual_usage": {
                                            "corrected_decisions": 1
                                        }
                                    }
                                }
                            }
                        ),
                        encoding="utf-8",
                    )


if __name__ == "__main__":
    unittest.main()
