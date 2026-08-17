import os
import tempfile
import unittest
from pathlib import Path

from evaluation.build_patient_indexed_specimen_routing_publication_artifacts import (
    build_publication_artifacts,
    read_json,
)


class RoutingPublicationArtifactsTest(unittest.TestCase):
    def test_frozen_evidence_builds_expected_publication_outputs(self) -> None:
        os.environ.setdefault("MPLBACKEND", "Agg")
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            result = build_publication_artifacts(
                repo_root,
                tmp_root / "reports",
                tmp_root / "figures",
                tmp_root / "evidence_map.json",
            )

            self.assertEqual(len(result["primary_rows"]), 3)
            self.assertEqual(len(result["sensitivity_rows"]), 6)
            self.assertEqual(len(result["td3_rows"]), 3)
            self.assertEqual(len(result["online_rows"]), 5)
            self.assertEqual(len(result["cost_rows"]), 12)

            primary = result["primary_rows"]
            self.assertAlmostEqual(
                primary[0]["relative_difference_pct"],
                -0.6584742769523861,
            )
            self.assertAlmostEqual(
                primary[2]["relative_difference_pct"],
                -0.32135654615948883,
            )
            self.assertTrue(all(row["favorable_cost_interval"] for row in primary))

            online_claim = next(
                claim
                for claim in result["evidence_map"]["claims"]
                if claim["id"] == "online_learning_attribution"
            )
            self.assertEqual(online_claim["status"], "not_established")
            self.assertTrue(
                all(
                    not row["supports_online_gain"]
                    for row in result["online_rows"]
                )
            )

            component_path = (
                repo_root
                / "experiments/evidence/"
                "patient_indexed_specimen_routing_primary_ddpg/formal/"
                "cost_component_summary.json"
            )
            components = read_json(component_path)
            top_level_difference = sum(
                values["difference"]
                for values in components["components"].values()
                if values["additive_to_total"]
            )
            self.assertAlmostEqual(
                top_level_difference,
                components["total_cost"]["difference"],
                places=4,
            )
            self.assertFalse(
                components["components"]["specimen_transfer_cost"]["additive_to_total"]
            )

            for name in (
                "primary_comparisons.csv",
                "transport_timing_sensitivity.csv",
                "td3_development_ablation.csv",
                "online_attribution.csv",
                "cost_components.csv",
            ):
                self.assertTrue((tmp_root / "reports" / name).is_file())
            for name in (
                "routing_primary_cost_effects.png",
                "routing_primary_cost_effects.pdf",
                "routing_primary_cost_components.png",
                "routing_primary_cost_components.pdf",
            ):
                self.assertTrue((tmp_root / "figures" / name).is_file())
            self.assertTrue((tmp_root / "evidence_map.json").is_file())
            committed_map = read_json(
                repo_root
                / "experiments/evidence/"
                "patient_indexed_specimen_routing_publication_evidence_map.json"
            )
            self.assertEqual(result["evidence_map"], committed_map)


if __name__ == "__main__":
    unittest.main()
