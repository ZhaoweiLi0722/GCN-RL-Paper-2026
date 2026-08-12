"""Contract tests for the paired DDPG support-alignment experiment."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.compare_ddpg_support_alignment import compare_support_alignment
from evaluation.run_patient_indexed_specimen_routing_ddpg_support_alignment import (
    ALGORITHMS,
    SEEDS,
    clone_paired_preonline_states,
    materialize_runtime_configs,
    read_csv_rows,
    verify_reused_preonline_source,
    validate_scientific_contract,
    verify_locked_assets,
)
from src.rl.networks import torch
from src.rl.training_state import training_contract_sha256


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / (
    "experiments/configs/"
    "patient_indexed_specimen_routing_ddpg_support_alignment_execution.json"
)
GRAPH = "gcn_residual_mdl2_network_ddpg_afd"
FLAT = "flat_residual_mdl2_network_ddpg_afd"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class DDPGSupportAlignmentTests(unittest.TestCase):
    def test_locked_assets_and_single_factor_contract(self) -> None:
        spec = load_json(SPEC)
        current = Path.cwd()
        try:
            import os

            os.chdir(ROOT)
            verified = verify_locked_assets(spec)
            validate_scientific_contract(spec)
        finally:
            os.chdir(current)
        self.assertEqual(
            verified[
                "experiments/evidence/"
                "patient_indexed_specimen_routing_stage_c/"
                "teacher_train_dagger.npz"
            ],
            "9ba2ac0873c0f68e6ecc4b443e0eace8"
            "230f8e151cceb485fafe218a7ac78d92",
        )

    def test_candidate_is_support_filter_only(self) -> None:
        spec = load_json(SPEC)
        control = load_json(ROOT / spec["control_training_config"])
        candidate = load_json(ROOT / spec["candidate_training_config"])
        self.assertNotIn(
            "allowed_option_groups",
            control["config_overrides"][
                "critic_teacher_advantage_calibration"
            ],
        )
        self.assertEqual(
            candidate["config_overrides"][
                "critic_teacher_advantage_calibration"
            ]["allowed_option_groups"],
            ["specimen_transfer"],
        )
        self.assertEqual(control["seeds"], [30, 31, 32])
        self.assertEqual(candidate["seeds"], [30, 31, 32])

    def test_development_streams_do_not_reuse_prior_evidence(self) -> None:
        spec = load_json(SPEC)
        self.assertEqual(spec["development_crn_seeds"], [95000000, 95100000])
        self.assertTrue(
            set(spec["development_crn_seeds"]).isdisjoint(
                spec["forbidden_crn_seeds"]
            )
        )
        self.assertIn(91100000, spec["forbidden_crn_seeds"])
        self.assertIn(93100000, spec["forbidden_crn_seeds"])

    def test_recovery2_reuses_only_completed_control_and_episode0_clones(
        self,
    ) -> None:
        spec = load_json(SPEC)
        self.assertIn("recovery2", spec["campaign_root"])
        self.assertIn("recovery1", spec["reuse_control_training_manifest"])
        self.assertIn("recovery1", spec["reuse_paired_state_provenance"])
        self.assertNotIn(
            "training_control_gcn_seeds_30_31_32",
            spec["phase_order"],
        )
        self.assertNotIn(
            "training_control_flat_seeds_30_31_32",
            spec["phase_order"],
        )
        self.assertIn(
            "training_candidate_gcn_seeds_30_31_32",
            spec["phase_order"],
        )

    def test_csv_reader_accepts_large_serialized_metric_fields(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            payload = "x" * 200_000
            self._write_csv(
                path,
                [{"episode": 0, "serialized_metric": payload}],
            )
            rows = read_csv_rows(path)

        self.assertEqual(rows[0]["serialized_metric"], payload)

    def test_comparator_detects_support_and_online_gain(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            control_pretrain = self._write_stage(
                root / "control_pretrain",
                variant="pretrain",
                total_cost=100.0,
            )
            control_final = self._write_stage(
                root / "control_final",
                variant="final",
                total_cost=98.0,
            )
            candidate_pretrain = self._write_stage(
                root / "candidate_pretrain",
                variant="pretrain",
                total_cost=100.0,
            )
            candidate_final = self._write_stage(
                root / "candidate_final",
                variant="final",
                total_cost=94.0,
            )
            result = compare_support_alignment(
                control_final_path=control_final,
                control_pretrain_path=control_pretrain,
                candidate_final_path=candidate_final,
                candidate_pretrain_path=candidate_pretrain,
                bootstrap_resamples=200,
                bootstrap_seed=123,
            )

        graph = result["comparisons"][GRAPH]
        self.assertAlmostEqual(
            graph["candidate_final_vs_control_final"]["total_cost"]
            ["mean_difference"],
            -4.0,
        )
        self.assertAlmostEqual(
            graph["candidate_final_vs_candidate_pretrain"]["total_cost"]
            ["mean_difference"],
            -6.0,
        )
        self.assertAlmostEqual(
            graph["online_gain_difference_in_differences"]["total_cost"]
            ["mean_difference"],
            -4.0,
        )
        self.assertEqual(
            result["classifications"][GRAPH]["classification"],
            "pass",
        )

    def test_comparator_accepts_large_serialized_metric_fields(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "control_pretrain": self._write_stage(
                    root / "control_pretrain",
                    variant="pretrain",
                    total_cost=100.0,
                    serialized_metric_size=200_000,
                ),
                "control_final": self._write_stage(
                    root / "control_final",
                    variant="final",
                    total_cost=98.0,
                    serialized_metric_size=200_000,
                ),
                "candidate_pretrain": self._write_stage(
                    root / "candidate_pretrain",
                    variant="pretrain",
                    total_cost=100.0,
                    serialized_metric_size=200_000,
                ),
                "candidate_final": self._write_stage(
                    root / "candidate_final",
                    variant="final",
                    total_cost=94.0,
                    serialized_metric_size=200_000,
                ),
            }
            result = compare_support_alignment(
                control_final_path=paths["control_final"],
                control_pretrain_path=paths["control_pretrain"],
                candidate_final_path=paths["candidate_final"],
                candidate_pretrain_path=paths["candidate_pretrain"],
                bootstrap_resamples=10,
                bootstrap_seed=123,
            )

        self.assertEqual(
            result["classifications"][GRAPH]["classification"],
            "pass",
        )

    @unittest.skipIf(torch is None, "PyTorch is required")
    def test_contract_clones_preserve_state_payload(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_contract = {
                "algorithm": "stub",
                "num_episodes": 0,
                "history_screen": {
                    "online_episodes": 0,
                    "pretrain_only": True,
                },
                "critic_teacher_advantage_calibration": {
                    "enabled": False,
                    "updates": 100,
                    "online_ranking_weight": 0.0,
                },
            }
            runs = []
            for algorithm in ALGORITHMS:
                for seed in SEEDS:
                    state = root / "source" / f"{algorithm}_{seed}.pt"
                    state.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "training": {"next_episode": 0},
                            "training_contract": source_contract,
                            "training_contract_sha256": (
                                training_contract_sha256(source_contract)
                            ),
                            "agent": {"actor": torch.tensor([seed, 1.0])},
                        },
                        state,
                    )
                    runs.append(
                        {
                            "algorithm": algorithm,
                            "seed": seed,
                            "preonline_training_state": str(state),
                        }
                    )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"runs": runs}), encoding="utf-8")
            control_config = root / "control.json"
            candidate_config = root / "candidate.json"
            base_calibration = {
                "enabled": True,
                "updates": 100,
                "online_ranking_weight": 3.0,
            }
            control_config.write_text(
                json.dumps(
                    {
                        "config_overrides": {
                            "critic_teacher_advantage_calibration": (
                                base_calibration
                            )
                        }
                    }
                ),
                encoding="utf-8",
            )
            candidate_config.write_text(
                json.dumps(
                    {
                        "config_overrides": {
                            "critic_teacher_advantage_calibration": {
                                **base_calibration,
                                "allowed_option_groups": [
                                    "specimen_transfer"
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = clone_paired_preonline_states(
                manifest,
                control_config_path=control_config,
                candidate_config_path=candidate_config,
                output_root=root / "clones",
            )

            first = result["runs"][0]
            source = torch.load(
                first["source"],
                map_location="cpu",
                weights_only=False,
            )
            control = torch.load(
                first["control_clone"],
                map_location="cpu",
                weights_only=False,
            )
            candidate = torch.load(
                first["candidate_clone"],
                map_location="cpu",
                weights_only=False,
            )

        self.assertEqual(len(result["runs"]), 6)
        self.assertTrue(
            torch.equal(
                source["agent"]["actor"],
                candidate["agent"]["actor"],
            )
        )
        self.assertEqual(control["training_contract"]["num_episodes"], 100)
        self.assertEqual(
            control["training_contract"]["history_screen"],
            {"online_episodes": 100, "pretrain_only": False},
        )
        self.assertEqual(
            candidate["training_contract"]["history_screen"],
            {"online_episodes": 100, "pretrain_only": False},
        )
        self.assertEqual(
            control["training_contract_sha256"],
            training_contract_sha256(control["training_contract"]),
        )
        self.assertNotIn(
            "allowed_option_groups",
            control["training_contract"][
                "critic_teacher_advantage_calibration"
            ],
        )
        self.assertEqual(
            candidate["training_contract"][
                "critic_teacher_advantage_calibration"
            ]["allowed_option_groups"],
            ["specimen_transfer"],
        )

    def test_runtime_recovery_configs_change_only_output_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_prefix = "results/original"
            target_prefix = "results/recovery1"
            training = root / "training.json"
            evaluation = root / "evaluation.json"
            training.write_text(
                json.dumps(
                    {
                        "online_episodes": 100,
                        "output_root": f"{source_prefix}/training",
                    }
                ),
                encoding="utf-8",
            )
            evaluation.write_text(
                json.dumps(
                    {
                        "holdout_replications": 50,
                        "training_manifest": f"{source_prefix}/manifest.json",
                        "output_root": f"{source_prefix}/evaluation",
                    }
                ),
                encoding="utf-8",
            )
            spec = {
                "runtime_config_path_rewrite": {
                    "source_prefix": source_prefix,
                    "target_prefix": target_prefix,
                }
            }
            for key in (
                "control_training_config",
                "candidate_training_config",
            ):
                spec[key] = str(training)
            for key in (
                "control_final_evaluation_config",
                "control_pretrain_evaluation_config",
                "candidate_final_evaluation_config",
                "candidate_pretrain_evaluation_config",
            ):
                spec[key] = str(evaluation)
            resolved, provenance = materialize_runtime_configs(
                spec,
                root / "launcher",
            )
            runtime_training = load_json(
                Path(resolved["control_training_config"])
            )
            runtime_evaluation = load_json(
                Path(resolved["control_final_evaluation_config"])
            )

        self.assertTrue(provenance["enabled"])
        self.assertFalse(provenance["scientific_values_modified"])
        self.assertEqual(runtime_training["online_episodes"], 100)
        self.assertEqual(
            runtime_training["output_root"],
            f"{target_prefix}/training",
        )
        self.assertEqual(runtime_evaluation["holdout_replications"], 50)
        self.assertEqual(
            runtime_evaluation["training_manifest"],
            f"{target_prefix}/manifest.json",
        )

    def test_runtime_continuation_configs_support_split_manifests(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.json"
            evaluation = root / "evaluation.json"
            training.write_text(
                json.dumps(
                    {
                        "online_episodes": 100,
                        "output_root": "results/original/training",
                    }
                ),
                encoding="utf-8",
            )
            evaluation.write_text(
                json.dumps(
                    {
                        "holdout_replications": 50,
                        "training_manifest": "results/original/manifest.json",
                        "output_root": "results/original/evaluation",
                    }
                ),
                encoding="utf-8",
            )
            spec = {"runtime_config_overrides": {}}
            for key in (
                "control_training_config",
                "candidate_training_config",
            ):
                spec[key] = str(training)
            for key in (
                "control_final_evaluation_config",
                "control_pretrain_evaluation_config",
                "candidate_final_evaluation_config",
                "candidate_pretrain_evaluation_config",
            ):
                spec[key] = str(evaluation)
            spec["runtime_config_overrides"] = {
                "candidate_training_config": {
                    "output_root": "results/recovery2/candidate/training",
                },
                "control_final_evaluation_config": {
                    "output_root": "results/recovery2/control/final",
                    "training_manifest": "results/recovery1/control.json",
                },
                "candidate_final_evaluation_config": {
                    "output_root": "results/recovery2/candidate/final",
                    "training_manifest": "results/recovery2/candidate.json",
                },
            }
            resolved, provenance = materialize_runtime_configs(
                spec,
                root / "launcher",
            )
            control = load_json(
                Path(resolved["control_final_evaluation_config"])
            )
            candidate = load_json(
                Path(resolved["candidate_final_evaluation_config"])
            )

        self.assertEqual(provenance["mode"], "explicit")
        self.assertFalse(provenance["scientific_values_modified"])
        self.assertEqual(
            control["training_manifest"],
            "results/recovery1/control.json",
        )
        self.assertEqual(
            candidate["training_manifest"],
            "results/recovery2/candidate.json",
        )

    def test_reused_preonline_tree_verifies_every_named_artifact(self) -> None:
        import hashlib

        with TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state.pt"
            pretrain = root / "pretrain.pt"
            state.write_bytes(b"episode-zero-state")
            pretrain.write_bytes(b"pretrain-checkpoint")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "preonline_training_state": str(state),
                                "pretrain_checkpoint": str(pretrain),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            inventory = {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((manifest, state, pretrain), key=str)
            }
            tree_sha = hashlib.sha256(
                json.dumps(
                    inventory,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            result = verify_reused_preonline_source(
                manifest,
                expected_manifest_sha256=inventory[str(manifest)],
                expected_tree_sha256=tree_sha,
            )
            state.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact tree"):
                verify_reused_preonline_source(
                    manifest,
                    expected_manifest_sha256=inventory[str(manifest)],
                    expected_tree_sha256=tree_sha,
                )

        self.assertEqual(result["artifact_count"], 3)

    def _write_stage(
        self,
        root: Path,
        *,
        variant: str,
        total_cost: float,
        serialized_metric_size: int = 0,
    ) -> Path:
        runs = []
        for algorithm in (GRAPH, FLAT):
            for seed in (30, 31, 32):
                run_root = root / algorithm / f"seed{seed}"
                rows = []
                for replication in (0, 1):
                    rows.append(
                        {
                            "training_seed": seed,
                            "scenario": "routing_nominal_history",
                            "evaluation_seed": 95100000,
                            "replication": replication,
                            "total_cost": total_cost,
                            "completion_service_level": (
                                0.81 if total_cost < 98.0 else 0.80
                            ),
                            "patients_lost": (
                                9.0 if total_cost < 98.0 else 10.0
                            ),
                            "patient_ineligibility_during_manufacturing_rate": (
                                0.049 if total_cost < 98.0 else 0.05
                            ),
                            "specimen_route_events_json": (
                                "x" * serialized_metric_size
                                if algorithm == GRAPH
                                and seed == 30
                                and replication == 0
                                else "[]"
                            ),
                        }
                    )
                self._write_csv(run_root / "holdout_rows.csv", rows)
                runs.append(
                    {
                        "algorithm": algorithm,
                        "training_seed": seed,
                    }
                )
        summary = {
            "scenarios": ["routing_nominal_history"],
            "holdout_replications": 2,
            "holdout_seed": 95100000,
            "algorithms": [GRAPH, FLAT],
            "training_seeds": [30, 31, 32],
            "fixed_deployment_candidate": {
                "scale": 1.0,
                "use_checkpoint_group_thresholds": True,
            },
            "fixed_checkpoint_variant": variant,
            "clinical_noninferiority": {
                "margins": {
                    "completion_service_level": 0.001,
                    "patients_lost": 1.0,
                    "patient_ineligibility_during_manufacturing_rate": 0.001,
                }
            },
            "runs": runs,
        }
        path = root / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary), encoding="utf-8")
        return path

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
