"""Approved graph adapter on metadata/artificial tensors, never patient episodes."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.env.capacity_planning import CapacityPlanningEnv
from src.env.patient_capacity_planning import PatientConditionCapacityEnv, patient_env_config_from_dict
from src.rl.candidate_pilot_compatibility import inspect_patient_layout, audit_reference_layouts
from src.rl.candidate_pilot_driver import candidate_prototype
from src.rl.candidate_pilot_recovery import recovery_configuration, prior_attempt_receipt
from src.rl.candidate_pilot_resources import stream_manifest
from src.rl.patient_replay_collector import PatientObservationProducer
from src.rl.candidate_patient_session import context_from_public
from src.rl.strict_frozen_policy import StrictFrozenPolicy
from src.rl.prospective_ddpg_kernel import state_digest
from tests.test_candidate_collection_boundary import dormant_metadata, invented_raw


class CandidatePilotRecoveryTests(unittest.TestCase):
    def setUp(self):
        for cls in (CapacityPlanningEnv, PatientConditionCapacityEnv):
            for method in ("__init__", "reset", "step"):
                guard = patch.object(cls, method, side_effect=AssertionError("no real environment operation"))
                guard.start()
                self.addCleanup(guard.stop)
        self.root = Path(__file__).resolve().parents[1]
        self.original = json.loads((self.root / "experiments/configs/candidate_return_pilot_20260930.json").read_text())
        self.cfg = recovery_configuration(self.root, self.original)

    def test_only_graph_and_new_storage_paths_differ(self):
        self.assertEqual({k for k in self.cfg if self.cfg[k] != self.original.get(k)},
                         {"candidate_message_graph", "output_root", "dropbox_directory_proposed"})
        self.assertEqual(stream_manifest(self.original), stream_manifest(self.cfg))
        self.assertFalse(self.cfg["scientific_execution_authorized"])
        self.assertEqual(self.original["output_root"], "results/candidate_return_pilot_20260930")

    def test_prior_failure_zero_step_inventory_is_verified(self):
        prior = prior_attempt_receipt(self.root, self.original)
        self.assertEqual(prior["ledger"]["events"], 2)
        self.assertEqual(prior["ledger"]["counts"], {})
        self.assertEqual(len(prior["files"]), 15)
        self.assertEqual(prior["previous_environment_constructions"], 1)
        self.assertTrue(prior["preflight_ordinal0_is_not_claimed_fresh"])

    def test_declared_projection_keeps_other_topologies_and_detects_drift(self):
        shell = dormant_metadata(include_central_capacity_hub=True, capacity_edges=())
        original_config = copy.deepcopy(shell.env_config)
        producer = PatientObservationProducer(shell, enabled=True, gamma=1., reward_scale=1e-9,
                                              message_graph="specimen_routes")
        obs, _ = producer.observe(invented_raw(shell))
        self.assertEqual(tuple(obs.nodes.shape), (1, 2, shell.features_per_facility + shell.summary_width))
        self.assertEqual(float(producer.links.sum()), 2.)
        self.assertEqual(shell.env_config, original_config)
        self.assertTrue(shell.config.include_central_capacity_hub)
        shell.capacity_edges = ((0, 1),)
        with self.assertRaises(ValueError):
            producer.check_environment(shell)

    def test_projection_is_explicit_separate_contract_not_a_default_migration(self):
        shell = dormant_metadata()
        old = PatientObservationProducer(shell, enabled=True, gamma=1., reward_scale=1e-9)
        new = PatientObservationProducer(shell, enabled=True, gamma=1., reward_scale=1e-9, message_graph="specimen_routes")
        self.assertNotEqual(old.contract, new.contract)
        np.testing.assert_array_equal(old.observe(invented_raw(shell))[1], new.observe(invented_raw(shell))[1])
        with self.assertRaises(ValueError):
            PatientObservationProducer(shell, enabled=True, gamma=1., reward_scale=1e-9, message_graph="union")
        shell = dormant_metadata(include_central_capacity_hub=True)
        with self.assertRaises(ValueError):
            PatientObservationProducer(shell, enabled=True, gamma=1., reward_scale=1e-9)

    def test_unsupported_action_channels_still_rejected(self):
        for field in ("enable_overtime_control", "include_on_order_state", "enable_stochastic_procurement"):
            shell = dormant_metadata(**{field: True})
            with self.assertRaises(ValueError):
                PatientObservationProducer(shell, enabled=True, gamma=1., reward_scale=1e-9, message_graph="specimen_routes")

    def metadata_shell(self, runtime):
        raw = copy.deepcopy(runtime["env"])
        raw.pop("graph_ablation", None)
        raw.pop("scenario_name", None)
        patient = patient_env_config_from_dict(raw)
        shell = object.__new__(PatientConditionCapacityEnv)
        shell.config, shell.env_config = patient.base, patient
        base = patient.base
        shell.features_per_facility = (3 + base.production_lead_time + int(base.include_supplier_state)
            + int(base.include_demand_forecast_state) + 3 * int(base.include_transfer_pipeline_state)
            + 3 * int(base.include_demand_history_state)
            + 3 * base.demand_sequence_length * int(base.include_demand_sequence_state))
        shell.summary_width = 6 + len(patient.survival_bucket_edges) + 1 + 4 * int(patient.include_specimen_routing_state)
        shell.observation_size = base.num_facilities * (shell.features_per_facility + shell.summary_width) + int(base.include_time_state)
        shell.action_size = 4 * base.num_facilities
        report = inspect_patient_layout(runtime, self.cfg["objective"], message_graph="specimen_routes")
        self.assertTrue(report["passed"])
        for name, edges in report["physical_edges"].items():
            setattr(shell, name, tuple(map(tuple, edges)))
        return shell

    def test_full_r4_metadata_schema_counts_and_artificial_raw_inference(self):
        spec = self.cfg["reference"]
        report = audit_reference_layouts(self.root, self.cfg)
        self.assertTrue(report["passed"])
        for block in self.cfg["blocks"]:
            directory = self.root / spec["directory"]
            cfg_path = directory / spec["config_template"].format(seed=block)
            shell = self.metadata_shell(json.loads(cfg_path.read_text()))
            producer = PatientObservationProducer(shell, enabled=True, gamma=1., reward_scale=1e-9, message_graph="specimen_routes")
            tensors = {}
            for rep in self.cfg["representations"]:
                model = candidate_prototype(producer, self.cfg, block=block, representation=rep["name"])
                tensors[rep["name"]] = state_digest(model.state_dict())
            self.assertEqual(tensors["graph"], tensors["self_only"])
            self.assertEqual(producer.links.shape, (20, 20))
            self.assertEqual(float(producer.links.sum()), 72.)
            self.assertEqual(len(shell.capacity_edges), 190)
            # This is an invented numerical array, not an observation from a reset.
            raw = np.arange(561, dtype=np.float32) / 1000.
            with torch.random.fork_rng(devices=[]):
                reference = StrictFrozenPolicy(directory / spec["policy_template"].format(seed=block), cfg_path,
                    checkpoint_sha256=spec["locks"][str(block)]["policy"], config_sha256=spec["locks"][str(block)]["config"])
            before = state_digest(reference._agent.actor.state_dict())
            obs, bank = context_from_public(producer, reference, raw, self.cfg["candidate_support"]["options"], "invented")
            np.testing.assert_array_equal(producer.unpack_raw(obs)[0], raw)
            self.assertEqual(len(bank.requests), 6)
            np.testing.assert_array_equal(np.array(bank.requests[0])[20:], np.array(bank.requests[1])[20:])
            self.assertEqual(reference._agent.graph_spec.num_nodes, 21)
            self.assertEqual(before, state_digest(reference._agent.actor.state_dict()))


if __name__ == "__main__":
    unittest.main()
