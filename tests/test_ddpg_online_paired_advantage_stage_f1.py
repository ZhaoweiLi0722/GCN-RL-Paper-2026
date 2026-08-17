import copy
import csv
import json
import unittest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evaluation.compare_ddpg_online_paired_advantage import (
    compare_online_paired_advantage,
    validate_matching_evaluation_contracts,
)
from evaluation.compare_ddpg_persistent_shift import VARIANT_EPISODES
from evaluation.prepare_ddpg_online_paired_advantage_preonline import (
    ALGORITHMS,
    prepare_preonline_states,
)
from evaluation.run_patient_indexed_specimen_routing_ddpg_online_paired_advantage import (
    audit_training,
    clone_paired_states,
    validate_lock_manifest,
    validate_scientific_contract,
    verify_locked_assets,
)


class OnlinePairedAdvantageStageF1Tests(unittest.TestCase):
    def test_execution_spec_has_complete_verified_lock_manifest(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec_path = Path(
            "experiments/configs/"
            "patient_indexed_specimen_routing_ddpg_online_paired_advantage_"
            "execution.json"
        )
        spec = json.loads((root / spec_path).read_text(encoding="utf-8"))

        validate_scientific_contract(spec)
        validate_lock_manifest(spec_path, spec)
        verified = verify_locked_assets(spec)

        self.assertGreaterEqual(len(verified), 40)
        self.assertNotIn(str(spec_path), verified)

    def test_locked_training_contract_is_single_factor(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config_root = root / "experiments" / "configs"
        validate_scientific_contract(
            {
                "control_training_config": str(
                    config_root
                    / "patient_indexed_specimen_routing_ddpg_online_paired_advantage_control_100.json"
                ),
                "candidate_training_config": str(
                    config_root
                    / "patient_indexed_specimen_routing_ddpg_online_paired_advantage_candidate_100.json"
                ),
                "control_evaluation_config": str(
                    config_root
                    / "patient_indexed_specimen_routing_ddpg_online_paired_advantage_control_curve_eval.json"
                ),
                "candidate_evaluation_config": str(
                    config_root
                    / "patient_indexed_specimen_routing_ddpg_online_paired_advantage_candidate_curve_eval.json"
                ),
                "control_smoke_config": str(
                    config_root
                    / "patient_indexed_specimen_routing_ddpg_online_paired_advantage_control_smoke.json"
                ),
                "candidate_smoke_config": str(
                    config_root
                    / "patient_indexed_specimen_routing_ddpg_online_paired_advantage_smoke.json"
                ),
                "development_crn_seeds": [96_600_000, 96_700_000],
                "forbidden_crn_seeds": [91_100_000, 95_800_000],
            }
        )

    def test_preonline_preparation_preserves_shared_exploration(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "source.json"
            config_path.write_text(
                json.dumps(
                    {
                        "online_episodes": 100,
                        "seeds": [60, 61, 62],
                        "config_overrides": {
                            "critic_teacher_advantage_calibration": {
                                "enabled": True,
                                "updates": 100,
                            },
                            "online_paired_advantage_critic": {
                                "enabled": False,
                            },
                            "residual_action": {
                                "structured_exploration": {
                                    "enabled": True,
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
                "evaluation.prepare_ddpg_online_paired_advantage_preonline."
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
                    for seed in (60, 61, 62)
                },
            )
            for config, _filters in calls:
                self.assertEqual(config["online_episodes"], 0)
                self.assertEqual(
                    config["config_overrides"]
                    ["critic_teacher_advantage_calibration"]["updates"],
                    0,
                )
                self.assertTrue(
                    config["config_overrides"]["residual_action"]
                    ["structured_exploration"]["enabled"]
                )
                self.assertFalse(
                    config["config_overrides"]
                    ["online_paired_advantage_critic"]["enabled"]
                )
            self.assertIn("runs", result)

    def test_comparator_requires_strict_online_gain_and_matched_frozen(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            assignment = {
                "60": "routing_persistent_hotspot_cluster1",
                "61": "routing_persistent_hotspot_cluster2",
                "62": "routing_persistent_hotspot_cluster3",
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
                                }
                                for algorithm in ALGORITHMS
                                for seed in (60, 61, 62)
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                config = {
                    "name": role,
                    "algorithms": list(ALGORITHMS),
                    "training_seeds": [60, 61, 62],
                    "checkpoint_variants": list(VARIANT_EPISODES),
                    "scenarios": list(assignment.values()),
                    "scenario_by_training_seed": assignment,
                    "validation_seed": 96_600_000,
                    "holdout_seed": 96_700_000,
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

            result = compare_online_paired_advantage(
                config_paths["control"],
                config_paths["candidate"],
                output_path=root / "comparison" / "summary.json",
                resamples=50,
                bootstrap_seed=123,
            )

            self.assertEqual(
                result["decision"]["classification"],
                "advance_paired_online_ddpg_to_fresh_confirmation",
            )
            self.assertTrue(result["decision"]["candidate_gate_passed"])
            self.assertTrue(
                result["frozen_identity"][ALGORITHMS[0]]["exact"]
            )

    def test_episode_zero_clones_defer_only_candidate_pair_fork(self) -> None:
        import torch

        root_path = Path(__file__).resolve().parents[1]
        config_root = root_path / "experiments" / "configs"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_runs = []
            source_payloads = {}
            for algorithm in ALGORITHMS:
                for seed in (60, 61, 62):
                    source = root / "source" / f"{algorithm}_{seed}.pt"
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
                            "online_paired_advantage_critic": {
                                "enabled": False,
                                "horizon": 4,
                                "loss_weight": 3.0,
                                "followup_policy": "mdl2",
                                "reward_consistency_atol": 0.000001,
                                "sign_tolerance": 1e-12,
                            },
                            "residual_action": {
                                "structured_exploration": {"enabled": True}
                            },
                        },
                        "agent": {"tensor": torch.tensor([float(seed)])},
                        "environment": {"seed": seed, "episode": 0},
                    }
                    torch.save(checkpoint, source)
                    source_runs.append(
                        {
                            "algorithm": algorithm,
                            "seed": seed,
                            "preonline_training_state": str(source),
                        }
                    )
                    source_payloads[(algorithm, seed)] = checkpoint
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"runs": source_runs}), encoding="utf-8"
            )
            provenance = clone_paired_states(
                manifest,
                control_config_path=(
                    config_root
                    / "patient_indexed_specimen_routing_ddpg_online_paired_advantage_control_100.json"
                ),
                candidate_config_path=(
                    config_root
                    / "patient_indexed_specimen_routing_ddpg_online_paired_advantage_candidate_100.json"
                ),
                output_root=root / "paired",
            )
            self.assertEqual(len(provenance["runs"]), 6)
            self.assertEqual(
                provenance["candidate_fork_path"],
                "online_paired_advantage_critic.enabled",
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
                            source_payloads[key]["agent"]["tensor"],
                        )
                    )
                    self.assertEqual(
                        clone["environment"],
                        source_payloads[key]["environment"],
                    )
                    self.assertFalse(
                        clone["training_contract"]
                        ["online_paired_advantage_critic"]["enabled"]
                    )

    def test_evaluation_contract_rejects_old_crn(self) -> None:
        base = {
            "algorithms": list(ALGORITHMS),
            "training_seeds": [60, 61, 62],
            "checkpoint_variants": list(VARIANT_EPISODES),
            "scenarios": ["a", "b", "c"],
            "scenario_by_training_seed": {"60": "a", "61": "b", "62": "c"},
            "validation_seed": 96_600_000,
            "holdout_seed": 96_700_000,
            "holdout_replications": 50,
            "max_steps": 52,
            "fixed_deployment_candidate": {"scale": 1.0},
            "clinical_noninferiority": {"margins": {}},
        }
        validate_matching_evaluation_contracts(base, copy.deepcopy(base))
        changed = copy.deepcopy(base)
        changed["holdout_seed"] = 95_800_000
        with self.assertRaisesRegex(ValueError, "contracts differ"):
            validate_matching_evaluation_contracts(base, changed)

    def test_training_audit_accepts_complete_control_and_candidate(self) -> None:
        import torch

        with TemporaryDirectory() as directory:
            root = Path(directory)
            for role in ("control", "candidate"):
                runs = []
                for algorithm in ALGORITHMS:
                    for seed in (60, 61, 62):
                        run_root = root / role / algorithm / f"seed{seed}"
                        run_root.mkdir(parents=True)
                        (run_root / "config.json").write_text(
                            "{}\n", encoding="utf-8"
                        )
                        self._write_training_rows(
                            run_root / "training.csv",
                            role=role,
                            seed=seed,
                        )
                        checkpoint_root = run_root / "checkpoints"
                        checkpoint_root.mkdir()
                        for episode in range(5, 101, 5):
                            (checkpoint_root / (
                                f"{algorithm}_seed{seed}_episode{episode}.pt"
                            )).touch()
                        paired_count = 100 if role == "candidate" else 0
                        masks = np.zeros(120, dtype=bool)
                        masks[:paired_count] = True
                        targets = np.zeros((120, 1), dtype=np.float32)
                        targets[:paired_count] = 0.001
                        state_path = checkpoint_root / "training_state.pt"
                        torch.save(
                            {
                                "training_contract": {
                                    "online_paired_advantage_critic": {
                                        "enabled": role == "candidate",
                                        "horizon": 4,
                                        "loss_weight": 3.0,
                                        "followup_policy": "mdl2",
                                        "reward_consistency_atol": 0.000001,
                                        "sign_tolerance": 1e-12,
                                    }
                                },
                                "agent": {
                                    "structured_specimen_explorer": {
                                        "enabled": True,
                                    },
                                    "replay_buffer": {
                                        "size": 120,
                                        "paired_advantage_mask": masks,
                                        "paired_advantages": targets,
                                    },
                                },
                            },
                            state_path,
                        )
                        pretrain = {}
                        if role == "candidate":
                            pretrain["preonline_fork_overrides"] = {
                                "online_paired_advantage_critic.enabled": {
                                    "source": False,
                                    "target": True,
                                }
                            }
                        runs.append(
                            {
                                "algorithm": algorithm,
                                "seed": seed,
                                "parameter_count": (
                                    1000 if algorithm.startswith("gcn") else 995
                                ),
                                "config": str(run_root / "config.json"),
                                "training_state_checkpoint": str(state_path),
                                "structured_specimen_exploration": {
                                    "enabled": True,
                                    "total_decisions": 1000,
                                    "total_selections": 200,
                                    "total_behaviorally_distinct": 100,
                                    "total_selection_rate": 0.2,
                                    "total_correction_selections": 160,
                                    "max_specimen_linf_delta": 0.1,
                                    "total_option_counts": {
                                        "mdl2": 40,
                                        "specimen_transfer:-0.05": 40,
                                        "specimen_transfer:+0.05": 40,
                                        "specimen_transfer:-0.10": 40,
                                        "specimen_transfer:+0.10": 40,
                                    },
                                },
                                "pretrain": pretrain,
                                "actor_drift_from_pretrain": {
                                    "rms": 0.01,
                                    "max_abs": 0.02,
                                    "parameter_count": 100,
                                },
                            }
                        )
                manifest = root / f"{role}_manifest.json"
                manifest.write_text(
                    json.dumps({"runs": runs}), encoding="utf-8"
                )
                result = audit_training(
                    manifest,
                    role=role,
                    scenario_by_seed={
                        60: "routing_persistent_hotspot_cluster1",
                        61: "routing_persistent_hotspot_cluster2",
                        62: "routing_persistent_hotspot_cluster3",
                    },
                    maximum_parameter_gap=0.01,
                    minimum_behavior_delta=1.0 / 120.0,
                )
                self.assertEqual(len(result["runs"]), 6)
                expected_pairs = 100 if role == "candidate" else 0
                self.assertTrue(
                    all(
                        entry["paired_advantage"]["replay_samples"]
                        == expected_pairs
                        for entry in result["runs"]
                    )
                )

    def _write_training_rows(
        self,
        path: Path,
        *,
        role: str,
        seed: int,
    ) -> None:
        fields = [
            "episode",
            "algorithm",
            "scenario",
            "total_cost",
            "completion_service_level",
            "patients_lost",
            "specimen_route_count",
            "online_rl_updates",
            "structured_specimen_episode_decisions",
            "structured_specimen_episode_selections",
            "structured_specimen_episode_behaviorally_distinct",
            "structured_specimen_episode_option_counts_json",
        ]
        if role == "candidate":
            fields.extend(
                [
                    "online_rl_online_paired_advantage_context_count_final",
                    "online_rl_online_paired_advantage_distinct_count_final",
                    "online_rl_online_paired_advantage_reward_error_max_final",
                    "online_rl_critic_online_paired_advantage_samples_mean",
                ]
            )
        scenario = f"routing_persistent_hotspot_cluster{seed - 59}"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for episode in range(100):
                row = {
                    "episode": episode,
                    "algorithm": "fixture",
                    "scenario": scenario,
                    "total_cost": 1000.0,
                    "completion_service_level": 0.99,
                    "patients_lost": 1.0,
                    "specimen_route_count": 1,
                    "online_rl_updates": 1,
                    "structured_specimen_episode_decisions": 10,
                    "structured_specimen_episode_selections": 2,
                    "structured_specimen_episode_behaviorally_distinct": 1,
                    "structured_specimen_episode_option_counts_json": "{}",
                }
                if role == "candidate":
                    row.update(
                        {
                            "online_rl_online_paired_advantage_context_count_final": (
                                (episode + 1) * 10
                            ),
                            "online_rl_online_paired_advantage_distinct_count_final": (
                                episode + 1
                            ),
                            "online_rl_online_paired_advantage_reward_error_max_final": 0.0,
                            "online_rl_critic_online_paired_advantage_samples_mean": 1.0,
                        }
                    )
                writer.writerow(row)

    def _write_curve(
        self,
        output_root: Path,
        *,
        role: str,
        assignment: dict[str, str],
    ) -> None:
        costs = {
            "pretrain": 1000.0,
            "episode10": 999.0,
            "episode25": 998.0,
            "episode50": 997.0 if role == "control" else 985.0,
            "episode75": 996.0 if role == "control" else 977.0,
            "final": 990.0 if role == "control" else 970.0,
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
                for seed in (60, 61, 62):
                    run_root = output_root / variant / algorithm / f"seed{seed}"
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
                                "evaluation_seed": 96_700_000,
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
