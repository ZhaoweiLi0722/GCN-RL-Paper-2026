"""Read configs and invented metadata only; real constructors/reset/step forbidden."""

from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.env.capacity_planning import CapacityPlanningEnv
from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.rl.candidate_pilot_campaign import PatientBackend
from src.rl.candidate_pilot_compatibility import inspect_patient_layout, audit_reference_layouts, require_supported_layouts
from src.rl.candidate_pilot_resources import stream_manifest
from src.rl.patient_replay_collector import PatientObservationProducer
from tests.test_candidate_collection_boundary import dormant_metadata


class CandidatePilotCompatibilityTests(unittest.TestCase):
    def setUp(self):
        for cls in (CapacityPlanningEnv, PatientConditionCapacityEnv):
            for method in ("__init__", "reset", "step"):
                guard = patch.object(cls, method, side_effect=AssertionError("no real environment operation"))
                guard.start()
                self.addCleanup(guard.stop)
        self.root = Path(__file__).resolve().parents[1]
        self.config = json.loads((self.root / "experiments/configs/candidate_return_pilot_20260930.json").read_text())

    def runtime(self, shell):
        patient = asdict(shell.env_config)
        base = patient.pop("base")
        return {"env": dict(base, **patient, env_type="patient_condition", graph_ablation="full_graph")}

    def test_supported_invented_layout_matches_unchanged_live_producer(self):
        for changes in ({}, {"include_demand_sequence_state": True, "demand_sequence_length": 3},
                        {"include_demand_history_state": True, "include_transfer_pipeline_state": True}):
            shell = dormant_metadata(**changes)
            result = inspect_patient_layout(self.runtime(shell))
            require_supported_layouts(result)
            producer = PatientObservationProducer(shell, enabled=True, gamma=1., reward_scale=1e-9)
            self.assertEqual(result["objective_layout"]["raw_state_width"], producer.raw_width)
            self.assertEqual(result["objective_layout"]["action_width"], shell.action_size)
            self.assertTrue(result["all_relations_identical"])

    def test_hub_and_relation_failures_are_separate_and_not_projected_away(self):
        for changes, reason in (({"include_central_capacity_hub": True}, "unsupported_producer_layout:include_central_capacity_hub"),
                                ({"capacity_edges": ()}, "heterogeneous_relations_cannot_use_single_adjacency")):
            shell = dormant_metadata(**changes)
            result = inspect_patient_layout(self.runtime(shell))
            self.assertIn(reason, result["reasons"])
            with self.assertRaises(ValueError):
                require_supported_layouts(result)
            with self.assertRaises(ValueError):
                PatientObservationProducer(shell, enabled=True, gamma=1., reward_scale=1e-9)

    def test_declared_objective_width_mismatch_cannot_pass(self):
        result = inspect_patient_layout(self.runtime(dormant_metadata()), self.config["objective"])
        self.assertIn("objective_mismatch:num_facilities", result["reasons"])
        self.assertIn("objective_mismatch:raw_state_width", result["reasons"])

    def test_actual_locked_r4_layouts_are_rejected_without_constructing_environment(self):
        report = audit_reference_layouts(self.root, self.config)
        self.assertFalse(report["passed"])
        self.assertTrue(report["effective_environments_equal"])
        self.assertEqual(report["new_environment_constructions"], 0)
        self.assertEqual(len(report["blocks"]), 3)
        for row in report["blocks"]:
            layout = row["layout"]
            self.assertEqual(layout["physical_edge_counts"], {"specimen_edges": 36, "resource_edges": 36,
                "capacity_edges": 190, "information_edges": 36})
            self.assertEqual(layout["reference_graph_nodes"], 21)
            self.assertEqual(len(layout["reference_capacity_graph_edges"]), 20)
            self.assertEqual(layout["objective_layout"]["raw_state_width"], 561)
            self.assertEqual(len(layout["reasons"]), 2)
            self.assertFalse(layout["realized_topology_verified"])
        self.assertEqual(report, audit_reference_layouts(self.root, self.config))

    def test_backend_veto_precedes_reference_load_and_environment_constructor(self):
        backend = PatientBackend(self.root, self.config, stream_manifest(self.config))
        with patch("src.rl.strict_frozen_policy.StrictFrozenPolicy", side_effect=AssertionError("no model load")) as reference:
            with patch("src.rl.experiment.build_env", side_effect=AssertionError("no environment")) as factory:
                with self.assertRaisesRegex(ValueError, "static producer compatibility"):
                    backend.prepare(60, 123)
                reference.assert_not_called()
                factory.assert_not_called()
        self.assertEqual(backend.contexts, {})

    def test_explicit_empty_edges_are_not_replaced_by_defaults(self):
        shell = dormant_metadata(specimen_edges=(), resource_edges=(), capacity_edges=(), information_edges=())
        report = inspect_patient_layout(self.runtime(shell))
        self.assertTrue(report["passed"])
        self.assertEqual(set(report["physical_edge_counts"].values()), {0})

    def test_ablation_is_applied_as_in_factory_not_ignored(self):
        runtime = self.runtime(dormant_metadata())
        runtime["env"]["graph_ablation"] = "no_capacity_sharing_edges"
        report = inspect_patient_layout(runtime)
        self.assertFalse(report["passed"])
        self.assertEqual(report["physical_edge_counts"]["capacity_edges"], 0)
