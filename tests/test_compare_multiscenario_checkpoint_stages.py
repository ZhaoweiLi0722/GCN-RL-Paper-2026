import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.compare_multiscenario_checkpoint_stages import (
    compare_checkpoint_stages,
)


GRAPH = "gcn_residual_mdl2_network_ddpg_afd"
FLAT = "flat_residual_mdl2_network_ddpg_afd"
CANDIDATE = {
    "scale": 0.1,
    "group_thresholds": [0.0, 0.0, 1.0],
}


class CheckpointStageComparisonTests(unittest.TestCase):
    def test_paired_attribution_and_progression_gates(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            final = self._write_stage(root / "final", variant="final")
            frozen = self._write_stage(
                root / "frozen",
                variant="pretrain",
            )

            result = compare_checkpoint_stages(
                final_summary_path=final,
                frozen_summary_path=frozen,
                bootstrap_resamples=200,
                bootstrap_seed=123,
            )

        self.assertTrue(result["crn_audit"]["passed"])
        graph = result["comparisons"][GRAPH]
        self.assertAlmostEqual(
            graph["final_vs_frozen"]["total_cost"]["mean_difference"],
            -3.0,
        )
        did = result["graph_flat_attribution"][
            "graph_minus_flat_difference_in_differences"
        ]["total_cost"]
        self.assertAlmostEqual(did["mean_difference"], -4.0)
        self.assertTrue(
            result["progression_gates"][
                "advance_to_five_seed_confirmation"
            ]
        )

    def test_mismatched_crn_keys_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            final = self._write_stage(root / "final", variant="final")
            frozen = self._write_stage(
                root / "frozen",
                variant="pretrain",
            )
            row_path = (
                frozen.parent
                / GRAPH
                / "seed0"
                / "holdout_rows.csv"
            )
            rows = self._read_csv(row_path)
            rows[0]["replication"] = "99"
            self._write_csv(row_path, rows)

            with self.assertRaisesRegex(ValueError, "CRN keys differ"):
                compare_checkpoint_stages(
                    final_summary_path=final,
                    frozen_summary_path=frozen,
                    bootstrap_resamples=20,
                    bootstrap_seed=123,
                )

    def _write_stage(self, root: Path, *, variant: str) -> Path:
        runs = []
        for algorithm in (GRAPH, FLAT):
            for seed in (0, 1):
                run_root = root / algorithm / f"seed{seed}"
                candidate_rows = []
                anchor_rows = []
                for replication in (0, 1):
                    anchor = self._row(
                        seed=seed,
                        replication=replication,
                        total_cost=100.0,
                        completion=0.80,
                        patients_lost=10.0,
                        ineligibility=0.05,
                        inference_ms=1.0 if variant == "final" else 2.0,
                    )
                    if algorithm == GRAPH and variant == "final":
                        candidate = self._row(
                            seed=seed,
                            replication=replication,
                            total_cost=95.0,
                            completion=0.81,
                            patients_lost=9.0,
                            ineligibility=0.049,
                            inference_ms=3.0,
                        )
                    elif algorithm == GRAPH:
                        candidate = self._row(
                            seed=seed,
                            replication=replication,
                            total_cost=98.0,
                            completion=0.805,
                            patients_lost=9.5,
                            ineligibility=0.0495,
                            inference_ms=3.0,
                        )
                    elif variant == "final":
                        candidate = self._row(
                            seed=seed,
                            replication=replication,
                            total_cost=100.0,
                            completion=0.80,
                            patients_lost=10.0,
                            ineligibility=0.05,
                            inference_ms=3.0,
                        )
                    else:
                        candidate = self._row(
                            seed=seed,
                            replication=replication,
                            total_cost=99.0,
                            completion=0.801,
                            patients_lost=9.8,
                            ineligibility=0.0498,
                            inference_ms=3.0,
                        )
                    candidate_rows.append(candidate)
                    anchor_rows.append(anchor)
                self._write_csv(
                    run_root / "holdout_rows.csv",
                    candidate_rows,
                )
                self._write_csv(
                    run_root / "holdout_anchor_rows.csv",
                    anchor_rows,
                )
                runs.append(
                    {
                        "algorithm": algorithm,
                        "training_seed": seed,
                        "checkpoint": str(
                            root / f"{algorithm}_seed{seed}_{variant}.pt"
                        ),
                        "selected_deployment": {
                            "candidate": CANDIDATE,
                        },
                        "holdout": {
                            "aggregate": {
                                "residual_usage": {
                                    "correction_rate": 0.5,
                                    "corrected_decisions": 2,
                                    "total_decisions": 4,
                                }
                            }
                        },
                    }
                )
        summary = {
            "scenarios": ["scenario_a"],
            "holdout_replications": 2,
            "holdout_seed": 777,
            "algorithms": [GRAPH, FLAT],
            "training_seeds": [0, 1],
            "fixed_deployment_candidate": CANDIDATE,
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
    def _row(
        *,
        seed: int,
        replication: int,
        total_cost: float,
        completion: float,
        patients_lost: float,
        ineligibility: float,
        inference_ms: float,
    ) -> dict[str, object]:
        return {
            "algorithm": "candidate",
            "training_seed": seed,
            "scenario": "scenario_a",
            "evaluation_seed": 777,
            "replication": replication,
            "total_cost": total_cost,
            "completion_service_level": completion,
            "patients_lost": patients_lost,
            "patient_ineligibility_during_manufacturing_rate": ineligibility,
            "average_inference_ms": inference_ms,
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
