import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np

from evaluation.audit_patient_indexed_specimen_routing_ddpg import (
    endpoint_support_summary,
    facility_net_group_slices,
    residual_scale_vector,
    read_csv_rows,
    saturation_summary,
    scenario_teacher_config,
)
from evaluation.network_residual_headroom import (
    bound_candidate_specs_to_residual_envelope,
    facility_net_residual_envelope,
)


class PatientIndexedSpecimenRoutingDDPGAuditTests(unittest.TestCase):
    def test_facility_group_slices_cover_four_groups(self) -> None:
        groups = facility_net_group_slices(2)
        self.assertEqual(groups["specimen_transfer"], slice(0, 2))
        self.assertEqual(groups["reagent_transfer"], slice(2, 4))
        self.assertEqual(groups["capacity_transfer"], slice(4, 6))
        self.assertEqual(groups["replenishment"], slice(6, 8))

    def test_group_scales_override_global_scale(self) -> None:
        result = residual_scale_vector(
            8,
            2,
            {
                "scale": 0.25,
                "group_scales": {
                    "specimen_transfer": 1.0,
                    "reagent_transfer": 0.5,
                    "capacity_transfer": 0.25,
                    "replenishment": 0.0,
                },
            },
        )
        np.testing.assert_allclose(
            result,
            [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0],
        )

    def test_saturation_reports_changed_values_at_bounded_gain(self) -> None:
        base = np.zeros((1, 4), dtype=np.float32)
        teacher = np.asarray([[0.2, -0.05, 0.0, 0.0]], dtype=np.float32)
        result = saturation_summary(
            base,
            teacher,
            num_facilities=1,
            base_scale_vector=np.asarray([1.0, 1.0, 1.0, 0.0]),
            multipliers=(1.0, 0.1),
        )
        self.assertEqual(result["changed_values"], 2)
        self.assertEqual(
            result["multipliers"]["1"]["saturated_changed_values"],
            0,
        )
        self.assertEqual(
            result["multipliers"]["0.1"]["saturated_changed_values"],
            1,
        )
        self.assertAlmostEqual(
            result["multipliers"]["0.1"]["saturated_changed_fraction"],
            0.5,
        )

    def test_scenario_config_uses_isolated_output_and_locked_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = {
                "name": "audit",
                "output_root": directory,
                "teacher_seed": 100,
                "lookahead_seed": 200,
                "teacher_replications_per_scenario": 2,
            }
            template = {
                "name": "template",
                "env_config": "old.json",
                "env_overrides": {"scenario_name": "old"},
            }
            scenario = {
                "name": "routing_regional_drift",
                "env_config": "regional.json",
                "env_overrides": {"scenario_name": "routing_regional_drift"},
            }
            result = scenario_teacher_config(audit, template, scenario)
            self.assertEqual(result["env_config"], "regional.json")
            self.assertEqual(result["seed"], 100)
            self.assertEqual(result["lookahead_seed"], 200)
            self.assertEqual(result["teacher_replications"], 2)
            self.assertEqual(result["teacher_replication_start"], 0)
            self.assertEqual(result["lookahead_decision_offset"], 0)
            self.assertEqual(
                Path(result["output_root"]).name,
                "routing_regional_drift",
            )

    def test_csv_reader_accepts_large_serialized_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("replication", "payload"))
                writer.writeheader()
                writer.writerow({"replication": 0, "payload": "x" * 150000})
            rows = read_csv_rows(path)
            self.assertEqual(rows[0]["replication"], "0")
            self.assertEqual(len(rows[0]["payload"]), 150000)

    def test_facility_net_residual_envelope_uses_group_limits(self) -> None:
        result = facility_net_residual_envelope(
            8,
            2,
            {
                "specimen_transfer": 0.1,
                "reagent_transfer": 0.05,
                "capacity_transfer": 0.025,
                "replenishment": 0.0,
            },
        )
        np.testing.assert_allclose(
            result,
            [0.1, 0.1, 0.05, 0.05, 0.025, 0.025, 0.0, 0.0],
        )

    def test_candidate_envelope_is_relative_to_anchor(self) -> None:
        anchor = np.asarray([0.9, -0.9, 0.2, 0.3], dtype=np.float32)
        candidate = np.asarray([-0.9, 0.9, -0.8, -0.7], dtype=np.float32)
        bounded = bound_candidate_specs_to_residual_envelope(
            [
                {"group": "anchor", "action": anchor},
                {"group": "combined_routing_network", "action": candidate},
            ],
            num_facilities=1,
            group_limits={
                "specimen_transfer": 0.1,
                "reagent_transfer": 0.1,
                "capacity_transfer": 0.1,
                "replenishment": 0.0,
            },
        )
        np.testing.assert_allclose(bounded[0]["action"], anchor)
        np.testing.assert_allclose(
            bounded[1]["action"],
            [0.8, -0.8, 0.1, 0.3],
            atol=1e-7,
        )

    def test_endpoint_support_detects_dense_unrepresentable_target(self) -> None:
        base = np.zeros((2, 12), dtype=np.float32)
        teacher = base.copy()
        teacher[0, :3] = (-0.1, 0.05, 0.05)
        teacher[1, :3] = (-0.1, 0.0, 0.1)
        result = endpoint_support_summary(
            base,
            teacher,
            num_facilities=3,
            settings={
                "enabled": True,
                "groups": ["specimen_transfer"],
                "max_endpoints_per_side": 1,
                "min_abs": 0.0,
            },
        )
        self.assertEqual(result["changed_rows"], 2)
        self.assertEqual(result["incompatible_rows"], 1)
        self.assertAlmostEqual(result["incompatible_changed_fraction"], 0.5)
        self.assertEqual(
            result["by_group"]["specimen_transfer"]["incompatible_rows"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
