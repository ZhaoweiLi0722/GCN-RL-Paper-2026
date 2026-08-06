"""Mechanics-gate tests for the patient-indexed routing experiment."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.validate_patient_indexed_specimen_routing import (
    validate_mechanics,
)
from src.rl.config import load_config


class PatientIndexedSpecimenRoutingMechanicsGateTests(unittest.TestCase):
    def test_preregistered_bottleneck_has_legal_mdl2_headroom(self) -> None:
        config = load_config(
            "experiments/configs/"
            "patient_indexed_specimen_routing_mechanics_gate.json"
        )
        with TemporaryDirectory() as directory:
            config["output_path"] = str(Path(directory) / "report.json")
            result = validate_mechanics(config)

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["checks"].values()))
        self.assertGreater(
            result["routing_mdl2"]["specimen_route_count"],
            0.0,
        )
        self.assertGreater(
            result["routing_mdl2"]["patients_completed"],
            result["no_routing_mdl2"]["patients_completed"],
        )

    def test_gate_refuses_to_overwrite_a_report(self) -> None:
        config = load_config(
            "experiments/configs/"
            "patient_indexed_specimen_routing_mechanics_gate.json"
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            config["output_path"] = str(output)
            validate_mechanics(config)
            with self.assertRaises(FileExistsError):
                validate_mechanics(config)


if __name__ == "__main__":
    unittest.main()
