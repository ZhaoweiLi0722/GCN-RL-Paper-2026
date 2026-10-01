"""Factory admission and lineage on fake factories only, never patient episodes."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.rl.dynamic_candidate_backend import DynamicPatientBackend
from src.rl.dynamic_candidate_resources import dynamic_stream_manifest
from src.rl.dynamic_candidate_specification import build_dynamic_proposal, validate_dynamic_proposal
from src.rl.networks import torch
from tests.test_dynamic_candidate_session import FakeEnv, FakeProducer, FakeReference, OPTIONS, session


ROOT = Path(__file__).resolve().parents[1]


class DynamicCandidateBackendTests(unittest.TestCase):
    def setUp(self):
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("numerical update forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)

    def test_proposal_preserves_draft_and_disabled_authority(self):
        path = ROOT / "specs/2026-10-01-adaptive-paper-delivery/pilot-budget-draft.json"
        before = path.read_bytes()
        config = build_dynamic_proposal(ROOT)
        self.assertEqual(before, path.read_bytes())
        self.assertFalse(config["scientific_execution_authorized"])
        self.assertFalse(config["ready_to_launch"])
        self.assertEqual(config["model_proposal"]["graph"], "specimen_routes")
        self.assertEqual(config["model_proposal"]["initial_reference_bias"], 0.)
        self.assertEqual(config["totals"]["main_environment_steps"], 22224)
        self.assertTrue(config["candidate_support"]["require_common_non_specimen_actions"])

    def test_unsupported_optimizer_or_graph_or_authority_rejected(self):
        original = build_dynamic_proposal(ROOT)
        for path, value in ((["optimizer", "betas"], [.8, .99]),
                            (["model_proposal", "graph"], "shared_relations"),
                            (["scientific_execution_authorized"], True)):
            config = copy.deepcopy(original)
            target = config
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                validate_dynamic_proposal(config)

    def test_no_default_admission_means_no_factory_or_checkpoint_load(self):
        config = build_dynamic_proposal(ROOT)
        streams = dynamic_stream_manifest(config)
        backend = DynamicPatientBackend(ROOT, config, streams)
        with patch("src.rl.experiment.build_env", side_effect=AssertionError("patient build forbidden")) as env, \
                patch("src.rl.strict_frozen_policy.StrictFrozenPolicy", side_effect=AssertionError("load forbidden")) as policy:
            with self.assertRaises(PermissionError):
                backend.prepare(60, streams["environment"]["60"]["prototype_preflight"][0])
            with self.assertRaises(PermissionError):
                backend.session(60, None, 123, trajectory="no", split="test", selection="greedy")
            env.assert_not_called()
            policy.assert_not_called()
        self.assertEqual(backend.build_attempts, 0)

    def fake_backend(self, root, *, reject_input=False):
        config = build_dynamic_proposal(ROOT)
        streams = dynamic_stream_manifest(config)
        config["reference"].update(directory="invented", config_template="config{block}.json",
                                   policy_template="policy{block}.txt")
        directory = root / "invented"
        directory.mkdir()
        data = {"config": b'{"env":{"invented":true}}', "policy": b"not a model checkpoint"}
        for key, payload in data.items():
            path = directory / config["reference"][key + "_template"].format(block=60)
            path.write_bytes(payload)
            config["reference"]["locks"]["60"][key] = hashlib.sha256(payload).hexdigest()
        if reject_input:
            config["reference"]["locks"]["60"]["config"] = "0" * 64
        config["candidate_support"]["options"] = OPTIONS
        operations = []
        backend = DynamicPatientBackend(root, config, streams, admit_real_calls=operations.append)
        return backend, streams, operations

    def test_fake_prepare_session_and_irreversible_build_counts(self):
        with tempfile.TemporaryDirectory() as name:
            backend, streams, operations = self.fake_backend(Path(name))
            producer = FakeProducer()
            reference = FakeReference(producer)
            with patch("src.rl.dynamic_candidate_backend.inspect_patient_layout", return_value={"passed": True}), \
                    patch("src.rl.experiment.build_env", side_effect=lambda *args: FakeEnv()) as factory, \
                    patch("src.rl.frozen_value_probe.assert_scenario", return_value={"invented": True}), \
                    patch("src.rl.patient_replay_collector.PatientObservationProducer", return_value=producer), \
                    patch("src.rl.strict_frozen_policy.StrictFrozenPolicy", return_value=reference):
                backend.prepare(60, streams["environment"]["60"]["prototype_preflight"][0])
                saved = backend.state_dict()
                run = backend.session(60, session(kind="frozen").learner,
                    streams["environment"]["60"]["test"][0],
                    trajectory="fake/test", split="test", selection="greedy")
                self.assertIs(type(run.env), FakeEnv)
                self.assertEqual(run.index, 0)
                backend.assert_restore_compatible(saved)
                self.assertEqual(backend.episode_builds, 1)
                self.assertEqual(factory.call_count, 2)
                with self.assertRaises(ValueError):
                    backend.prepare(60, streams["environment"]["60"]["prototype_preflight"][0])
                with self.assertRaises(ValueError):
                    backend.session(60, run.learner, 999, trajectory="fake/wrong", split="test", selection="greedy")
                self.assertEqual(factory.call_count, 2)
            self.assertIn("reference_checkpoint_load", operations)
            self.assertEqual(backend.build_attempts, 1)

    def test_hash_failure_is_terminal_before_any_real_factory(self):
        with tempfile.TemporaryDirectory() as name:
            backend, streams, _ = self.fake_backend(Path(name), reject_input=True)
            with patch("src.rl.experiment.build_env", side_effect=AssertionError("forbidden")) as factory:
                seed = streams["environment"]["60"]["prototype_preflight"][0]
                with self.assertRaisesRegex(ValueError, "input changed"):
                    backend.prepare(60, seed)
                with self.assertRaisesRegex(ValueError, "terminal"):
                    backend.prepare(60, seed)
                factory.assert_not_called()
                self.assertEqual(backend.build_attempts, 1)


if __name__ == "__main__":
    unittest.main()
