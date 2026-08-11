import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - optional local dependency
    torch = None

from evaluation.summarize_multiscenario_checkpoint_curve import (
    summarize_checkpoint_curve,
)


GRAPH = "gcn_residual_mdl2_network_ddpg_afd"
FLAT = "flat_residual_mdl2_network_ddpg_afd"
VARIANTS = ("pretrain", "episode25", "episode50", "episode75", "episode100")
CANDIDATE = {"scale": 1.0, "use_checkpoint_group_thresholds": True}
NORMALIZED_CANDIDATE = {
    "group_thresholds": [],
    "scale": 1.0,
    "use_checkpoint_group_thresholds": True,
}


class CheckpointCurveTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is required for actor drift")
    def test_curve_audits_crn_and_applies_late_segment_gate(self) -> None:
        with TemporaryDirectory() as directory:
            spec_path = self._write_campaign(Path(directory))
            result = summarize_checkpoint_curve(
                stage_spec_path=spec_path,
                bootstrap_resamples=200,
                bootstrap_seed=123,
            )

        self.assertTrue(result["crn_audit"]["passed"])
        self.assertEqual(result["crn_audit"]["variant_count"], 5)
        self.assertEqual(result["decision"]["classification"], "improving")
        self.assertEqual(
            result["decision"]["next_step"],
            "matched_ddpg_200_episode_development",
        )
        graph = result["algorithms"][GRAPH]
        late = graph["segments"]["episode75_to_episode100"]
        self.assertAlmostEqual(
            late["metrics"]["total_cost"]["mean_difference"],
            -2.0,
        )
        drift = graph["checkpoints"]["episode100"][
            "actor_drift_from_pretrain"
        ]
        self.assertGreater(drift["per_seed"]["0"]["rms"], 0.0)

    @unittest.skipIf(torch is None, "PyTorch is required for actor drift")
    def test_mismatched_checkpoint_crn_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = self._write_campaign(root)
            row_path = root / "episode50" / GRAPH / "seed0" / "holdout_rows.csv"
            rows = self._read_csv(row_path)
            rows[0]["replication"] = "99"
            self._write_csv(row_path, rows)

            with self.assertRaisesRegex(ValueError, "CRN keys differ"):
                summarize_checkpoint_curve(
                    stage_spec_path=spec_path,
                    bootstrap_resamples=20,
                    bootstrap_seed=123,
                )

    def test_repository_stage_b_configs_are_fixed_and_development_only(self) -> None:
        spec_path = Path(
            "experiments/configs/"
            "patient_indexed_specimen_routing_mac_mps_ddpg_confirmation_100_"
            "online_attribution_stage_b.json"
        )
        spec = json.loads(spec_path.read_text())
        self.assertEqual(spec["development_evaluation_seed"], 93100000)
        self.assertNotEqual(spec["development_evaluation_seed"], 91100000)
        self.assertEqual(spec["checkpoint_variants"], list(VARIANTS))
        output_roots = set()
        for variant in VARIANTS:
            config = json.loads(
                Path(spec["evaluation_configs"][variant]).read_text()
            )
            self.assertEqual(config["checkpoint_variants"], [variant])
            self.assertEqual(config["fixed_checkpoint_variant"], variant)
            self.assertEqual(config["fixed_deployment_candidate"], CANDIDATE)
            self.assertEqual(config["holdout_seed"], 93100000)
            self.assertEqual(config["holdout_replications"], 50)
            self.assertIn("development-only", config["experimental_role"])
            output_roots.add(config["output_root"])
        self.assertEqual(len(output_roots), len(VARIANTS))

    @staticmethod
    def _write_campaign(root: Path) -> Path:
        config_paths = {}
        costs = {
            "pretrain": 100.0,
            "episode25": 99.0,
            "episode50": 98.0,
            "episode75": 97.0,
            "episode100": 95.0,
        }
        for variant_index, variant in enumerate(VARIANTS):
            output_root = root / variant
            config_path = root / f"{variant}.json"
            config = {
                "training_manifest": "training_manifest.json",
                "algorithms": [GRAPH, FLAT],
                "training_seeds": [0, 1, 2],
                "checkpoint_variants": [variant],
                "scenarios": ["scenario_a"],
                "fixed_deployment_candidate": CANDIDATE,
                "fixed_checkpoint_variant": variant,
                "validation_replications": 1,
                "validation_seed": 1000,
                "holdout_replications": 4,
                "holdout_seed": 2000,
                "max_steps": 2,
                "clinical_noninferiority": {
                    "margins": {
                        "completion_service_level": 0.001,
                        "patients_lost": 1.0,
                        "patient_ineligibility_during_manufacturing_rate": 0.001,
                    }
                },
                "output_root": str(output_root),
            }
            config_path.write_text(json.dumps(config))
            config_paths[variant] = str(config_path)
            runs = []
            for algorithm in (GRAPH, FLAT):
                for seed in (0, 1, 2):
                    run_root = output_root / algorithm / f"seed{seed}"
                    actor_path = root / f"{algorithm}_seed{seed}_{variant}.pt"
                    torch.save(
                        {
                            "actor": {
                                "weight": torch.tensor(
                                    [float(variant_index), float(seed)]
                                )
                            }
                        },
                        actor_path,
                    )
                    candidate_cost = (
                        costs[variant]
                        if algorithm == GRAPH
                        else 101.0
                    )
                    candidate_rows = [
                        CheckpointCurveTests._row(
                            seed=seed,
                            replication=replication,
                            total_cost=candidate_cost,
                        )
                        for replication in range(4)
                    ]
                    anchor_rows = [
                        CheckpointCurveTests._row(
                            seed=seed,
                            replication=replication,
                            total_cost=110.0,
                        )
                        for replication in range(4)
                    ]
                    CheckpointCurveTests._write_csv(
                        run_root / "holdout_rows.csv", candidate_rows
                    )
                    CheckpointCurveTests._write_csv(
                        run_root / "holdout_anchor_rows.csv", anchor_rows
                    )
                    runs.append(
                        {
                            "algorithm": algorithm,
                            "training_seed": seed,
                            "checkpoint": str(actor_path),
                            "selected_deployment": {
                                "candidate": NORMALIZED_CANDIDATE,
                                "checkpoint_variant": variant,
                                "selection_source": "pre_registered_fixed",
                            },
                            "holdout": {
                                "aggregate": {
                                    "residual_usage": {
                                        "corrected_decisions": 4,
                                        "applied_residual_l1": 1.0,
                                    }
                                }
                            },
                        }
                    )
            summary = {
                "training_manifest": "training_manifest.json",
                "algorithms": sorted((GRAPH, FLAT)),
                "training_seeds": [0, 1, 2],
                "scenarios": ["scenario_a"],
                "fixed_deployment_candidate": NORMALIZED_CANDIDATE,
                "fixed_checkpoint_variant": variant,
                "validation_replications": 1,
                "validation_seed": 1000,
                "holdout_replications": 4,
                "holdout_seed": 2000,
                "clinical_noninferiority": config["clinical_noninferiority"],
                "runs": runs,
            }
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "summary.json").write_text(json.dumps(summary))

        spec = {
            "checkpoint_variants": list(VARIANTS),
            "evaluation_configs": config_paths,
            "bootstrap_resamples": 200,
            "bootstrap_seed": 123,
            "curve_summary": str(root / "curve.json"),
            "decision_rule": {
                "late_segment_from": "episode75",
                "late_segment_to": "episode100",
                "primary_algorithm": GRAPH,
                "improving_requires_total_cost_ci_high_below_zero": True,
                "improving_requires_seed_wins_at_least": 3,
                "improving_requires_clinical_noninferiority": True,
                "if_improving": "matched_ddpg_200_episode_development",
                "otherwise": (
                    "matched_td3_100_episode_three_seed_development_screen"
                ),
            },
        }
        spec_path = root / "spec.json"
        spec_path.write_text(json.dumps(spec))
        return spec_path

    @staticmethod
    def _row(
        *, seed: int, replication: int, total_cost: float
    ) -> dict[str, object]:
        return {
            "algorithm": "candidate",
            "training_seed": seed,
            "scenario": "scenario_a",
            "evaluation_seed": 2000,
            "replication": replication,
            "total_cost": total_cost,
            "completion_service_level": 0.9,
            "patients_lost": 10.0,
            "patient_ineligibility_during_manufacturing_rate": 0.05,
            "specimen_route_count": 1,
            "specimen_route_distance_miles": 10.0,
            "specimen_route_time_hours": 0.52,
            "average_inference_ms": 1.0,
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
