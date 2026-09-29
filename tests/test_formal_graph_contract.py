import copy
import unittest
from types import SimpleNamespace

import torch

from evaluation.audit_formal_graph_contract import (
    ALGORITHMS, check_pair, config_differences, historical_function,
    interface_probes, load_historical_modules, reconstruct,
)


def fixture(algorithm=ALGORITHMS[0]):
    return {
        "algorithm": algorithm, "actor_readout_mode": "network_residual",
        "gcn_hidden_sizes": [64, 32], "actor_hidden_sizes": [256, 128, 64],
        "critic_hidden_sizes": [256, 128, 64], "hidden_sizes": [292, 212, 128],
        "include_global_context": True, "include_adaptive_demand_features": True,
        "specimen_routing_head_enabled": True,
        "env": {"num_facilities": 20, "env_type": "patient_condition", "production_lead_time": 3,
                "action_mode": "facility_net", "include_supplier_state": True,
                "include_demand_forecast_state": True, "include_demand_history_state": True,
                "include_transfer_pipeline_state": True, "include_time_state": True,
                "include_central_capacity_hub": True, "include_specimen_routing_state": True,
                "demand_rates": [1.] * 20},
        "residual_action": {"enabled": True, "include_base_action_features": True,
                            "correction_gate": {"enabled": True, "groups": ["specimen_transfer"],
                                                "include_proposed_residual_features": True}},
    }


class FormalGraphContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = load_historical_modules()

    def test_historical_architecture_counts_and_shapes(self):
        for algorithm, expected in zip(ALGORITHMS, (613286, 607338)):
            with self.subTest(algorithm=algorithm):
                result = reconstruct(fixture(algorithm), self.modules)
                self.assertEqual(result["total_parameters"], expected)
                self.assertEqual(result["forward_shapes"], [[2, 80], [2, 1], [2]])
                self.assertTrue(result["finite_untrained_cpu_forward"])

    def test_input_and_environment_boundaries(self):
        probes = interface_probes(fixture(), self.modules)
        for key in ("capacity_pipeline_redistribution_preserves_explicit_graph_raw_block",
                    "raw_flat_states_differ", "flat_ablation_preserves_physical_config",
                    "gcn_proposal_changes_gate_input", "flat_actor_gate_raw_state_preserved"):
            self.assertTrue(probes[key], key)
        self.assertEqual(probes["flat_actor_gate_feature_width"], 641)
        self.assertEqual(probes["gcn_nonzero_proposal_features"], 1)
        self.assertFalse(probes["flat_source_references_proposed_residual_config"])
        self.assertFalse(probes["flat_source_references_adaptive_feature_helper"])

    def test_unreviewed_arm_differences_rejected(self):
        left = fixture()
        right = copy.deepcopy(left)
        right["env"]["num_facilities"] = 19
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            check_pair(left, right)
        right = copy.deepcopy(left)
        right["algorithm"] = ALGORITHMS[1]
        self.assertEqual(set(check_pair(left, right)), {"algorithm"})

    def test_missing_and_none_are_distinguished(self):
        self.assertEqual(config_differences({}, {"a": None})["a"]["gcn_present"], False)
        self.assertEqual(config_differences({"a": None}, {"a": None}), {})

    def test_count_excludes_targets_and_reference(self):
        count = historical_function("evaluation/train_multiscenario_network_residual.py",
                                    "agent_parameter_count", {})
        actor = torch.nn.Linear(2, 3)
        agent = SimpleNamespace(actor=actor, critic=actor, actor_target=torch.nn.Linear(100, 100),
                                pretrain_reference_actor=torch.nn.Linear(100, 100))
        self.assertEqual(count(agent), 9)

    def test_structural_reconstruction_preserves_cpu_rng(self):
        state = torch.random.get_rng_state().clone()
        reconstruct(fixture(), self.modules)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_unsupported_model_mode_rejected(self):
        config = fixture()
        config["temporal_demand_encoder"] = {"enabled": True}
        with self.assertRaisesRegex(ValueError, "non-temporal"):
            reconstruct(config, self.modules)

    def test_missing_proposal_config_rejected(self):
        config = fixture()
        config["residual_action"]["correction_gate"]["include_proposed_residual_features"] = False
        with self.assertRaisesRegex(ValueError, "proposed-residual"):
            reconstruct(config, self.modules)


if __name__ == "__main__":
    unittest.main()
