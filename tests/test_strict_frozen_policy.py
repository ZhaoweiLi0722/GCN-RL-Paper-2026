"""No simulator trajectories or training; synthetic load/inference acceptance."""

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evaluation.audit_replacement_policy_compatibility import compare_outputs, sample_indices
from src.models.gcn_ddpg import GCNDDPGAgent
from src.rl.strict_frozen_policy import POLICY_METADATA, StrictFrozenPolicy, load_policy_payload_strict
from src.utils.research_archive import sha256_file


@dataclass
class GraphFixture:
    width: int = 2


def fixture():
    metadata = dict(algorithm="synthetic", state_dim=2, action_dim=1,
                    correction_gate_mode="classification", correction_gate_threshold=.5,
                    correction_gate_include_proposed_residual_features=True,
                    correction_gate_proposed_residual_feature_mode="clipped_action_delta",
                    correction_gate_group_thresholds=(.5,), correction_safety_gate_threshold=.5,
                    correction_safety_gate_group_thresholds=())
    agent = SimpleNamespace(**metadata, graph_spec=GraphFixture(), actor=torch.nn.Linear(2, 1),
                            correction_gate=torch.nn.Linear(2, 1), correction_safety_gate=None)
    payload = copy.deepcopy(metadata)
    payload.update(graph_spec={"width": 2}, actor=copy.deepcopy(agent.actor.state_dict()),
                   correction_gate=copy.deepcopy(agent.correction_gate.state_dict()),
                   correction_safety_gate=None)
    return agent, payload


class StrictLoadTest(unittest.TestCase):
    def test_exact_load_freezes_all_policy_parameters(self):
        agent, payload = fixture()
        payload["actor"]["weight"].fill_(.23)
        load_policy_payload_strict(agent, payload)
        self.assertTrue(torch.equal(agent.actor.weight, payload["actor"]["weight"]))
        for module in (agent.actor, agent.correction_gate):
            self.assertFalse(module.training)
            self.assertFalse(any(p.requires_grad for p in module.parameters()))

    def test_all_metadata_is_required(self):
        for key in POLICY_METADATA:
            with self.subTest(key=key):
                agent, payload = fixture()
                del payload[key]
                with self.assertRaisesRegex(ValueError, "metadata"):
                    load_policy_payload_strict(agent, payload)

    def test_changed_threshold_or_feature_contract_rejected(self):
        for key, value in (("correction_gate_threshold", .2), ("state_dim", 3),
                           ("correction_gate_proposed_residual_feature_mode", "raw_scaled"),
                           ("correction_gate_group_thresholds", (.4,))):
            with self.subTest(key=key):
                agent, payload = fixture()
                payload[key] = value
                with self.assertRaisesRegex(ValueError, "metadata"):
                    load_policy_payload_strict(agent, payload)

    def test_graph_contract_cannot_silently_change(self):
        agent, payload = fixture()
        payload["graph_spec"]["width"] = 3
        with self.assertRaisesRegex(ValueError, "graph"):
            load_policy_payload_strict(agent, payload)

    def test_gate_failure_does_not_partially_load_actor(self):
        agent, payload = fixture()
        original = agent.actor.weight.clone()
        payload["actor"]["weight"].fill_(77.)
        del payload["correction_gate"]["bias"]
        with self.assertRaisesRegex(ValueError, "keys"):
            load_policy_payload_strict(agent, payload)
        self.assertTrue(torch.equal(agent.actor.weight, original))

    def test_extra_tensor_rejected(self):
        agent, payload = fixture()
        payload["actor"]["unexpected"] = torch.ones(1)
        with self.assertRaisesRegex(ValueError, "keys"):
            load_policy_payload_strict(agent, payload)

    def test_shape_dtype_and_nonfinite_tensor_rejected(self):
        for tensor in (torch.ones(9), torch.ones(1, 2, dtype=torch.float64),
                       torch.full((1, 2), float("nan")), torch.full((1, 2), float("inf"))):
            agent, payload = fixture()
            payload["actor"]["weight"] = tensor
            with self.assertRaisesRegex(ValueError, "tensor"):
                load_policy_payload_strict(agent, payload)

    def test_missing_gate_rejected(self):
        agent, payload = fixture()
        payload["correction_gate"] = None
        with self.assertRaisesRegex(ValueError, "keys"):
            load_policy_payload_strict(agent, payload)

    def test_unexpected_safety_gate_rejected(self):
        agent, payload = fixture()
        payload["correction_safety_gate"] = payload["actor"]
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            load_policy_payload_strict(agent, payload)

    def test_missing_none_module_declaration_rejected(self):
        agent, payload = fixture()
        del payload["correction_safety_gate"]
        with self.assertRaisesRegex(ValueError, "declaration"):
            load_policy_payload_strict(agent, payload)


class OutputCheckTest(unittest.TestCase):
    def setUp(self):
        self.output = {"actor": [0., .1], "gate_scores": [.2], "request": [.1, .2],
                       "hard_gate": [True], "requested_lots": [12, 24]}

    def compare(self, other):
        return compare_outputs(self.output, other, network_atol=1e-5, request_atol=1e-6)

    def test_identical_outputs_pass(self):
        self.assertTrue(all(v == 0 for v in self.compare(self.output).values()))

    def test_gate_flip_fails_even_with_close_continuous_values(self):
        other = copy.deepcopy(self.output)
        other["hard_gate"] = [False]
        with self.assertRaisesRegex(ValueError, "Discrete"):
            self.compare(other)

    def test_requested_lot_mismatch_fails(self):
        other = copy.deepcopy(self.output)
        other["requested_lots"][0] += 1
        with self.assertRaisesRegex(ValueError, "Discrete"):
            self.compare(other)

    def test_tolerance_shape_and_nonfinite_fail(self):
        for values in ([.1, .3], [.1], [.1, float("nan")]):
            other = copy.deepcopy(self.output)
            other["request"] = values
            with self.assertRaises(ValueError):
                self.compare(other)

    def test_sampling_is_fixed_and_without_replacement(self):
        indices = sample_indices(1354, 16)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 1353)
        self.assertEqual(len(set(indices)), 16)
        with self.assertRaises(ValueError):
            sample_indices(4, 16)


class FrozenFacadeTest(unittest.TestCase):
    def test_real_agent_roundtrip_no_environment_or_learning(self):
        config = {"algorithm": "gcn_ddpg", "seed": 7, "device": "cpu", "replay_buffer_size": 1,
                  "gcn_hidden_sizes": [8], "actor_hidden_sizes": [16], "critic_hidden_sizes": [16],
                  "env": {"num_facilities": 20, "production_lead_time": 3,
                          "action_mode": "facility_net", "include_supplier_state": True,
                          "include_central_capacity_hub": True}}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cp, cfg = root / "policy.pt", root / "config.json"
            cfg.write_text(json.dumps(config))
            original = GCNDDPGAgent(140, 80, config)
            original.save(cp)
            with patch.object(GCNDDPGAgent, "load_actor", side_effect=AssertionError("legacy loader")), \
                 patch.object(GCNDDPGAgent, "update", side_effect=AssertionError("update")), \
                 patch.object(torch.optim.Adam, "step", side_effect=AssertionError("optimizer")), \
                 patch("src.env.capacity_planning.CapacityPlanningEnv.step", side_effect=AssertionError("env")), \
                 patch("src.env.capacity_planning.CapacityPlanningEnv.reset", side_effect=AssertionError("reset")):
                policy = StrictFrozenPolicy(cp, cfg, checkpoint_sha256=sha256_file(cp),
                                            config_sha256=sha256_file(cfg))
                observation = np.linspace(0, 1, 140, dtype=np.float32)
                np.testing.assert_array_equal(policy.act(observation), original.select_action(observation, explore=False))
                self.assertEqual(len(policy._agent.replay_buffer), 0)
                self.assertFalse(hasattr(policy, "update"))
                for bad in (np.zeros(139), np.full(140, np.nan)):
                    with self.assertRaises(ValueError):
                        policy.act(bad)
                with self.assertRaisesRegex(ValueError, "Checkpoint hash"):
                    StrictFrozenPolicy(cp, cfg, checkpoint_sha256="bad", config_sha256=sha256_file(cfg))
                with self.assertRaisesRegex(ValueError, "config hash"):
                    StrictFrozenPolicy(cp, cfg, checkpoint_sha256=sha256_file(cp), config_sha256="bad")


if __name__ == "__main__":
    unittest.main()
