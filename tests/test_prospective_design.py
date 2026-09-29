"""Count-only preparation checks; no simulation, fitting or outcome comparison."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.audit_prospective_design import (
    ROOT, audit, load_contract, match_flat_width, parameter_counts, validate_config, verify_modules,
)
from src.models.prospective_forward_agent import ProspectiveForwardAgent
from src.rl.networks import torch


@unittest.skipIf(torch is None, "torch unavailable")
class ProspectiveDesignTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "experiments/configs/prospective_development_preflight_20260929.json").read_text())
        self.contract = load_contract(ROOT / self.config["schema_source"], self.config["schema_source_sha256"])

    def match(self, **changes):
        options = dict(graph_width=64, flat_range=[1, 256], component_limit=.01, total_limit=.01)
        return match_flat_width(self.contract, **(options | changes))

    def test_formula_matches_actual_modules_at_several_widths(self):
        for architecture in ("graph", "flat"):
            for width in (1, 8, 64, 68, 256):
                agent = ProspectiveForwardAgent(self.contract, enabled=True, architecture=architecture,
                                                message_mode="physical", hidden_width=width,
                                                residual_scale=.1, seed=0)
                actual = agent.inventory()
                expected = parameter_counts(self.contract, architecture, width)
                self.assertEqual(actual["total_unique"], expected.pop("total"))
                self.assertEqual(actual["components"], expected)

    def test_selected_width_meets_all_component_and_total_limits(self):
        result = self.match()
        chosen = result["selected"]
        self.assertEqual(chosen["width"], 68)
        self.assertTrue(all(value <= .01 for value in chosen["gaps_fraction"].values()))
        eligible = [c for c in result["candidates"] if c["eligible"]]
        self.assertEqual(chosen, min(eligible, key=lambda c: (c["max_component_gap_fraction"],
                                                             c["gaps_fraction"]["total"], c["width"])))

    def test_tiny_original_family_does_not_silently_relax_limits(self):
        self.assertIsNone(self.match(graph_width=8)["selected"])

    def test_unattainable_range_reports_no_match(self):
        result = self.match(flat_range=[1, 2])
        self.assertIsNone(result["selected"])
        self.assertEqual(len(result["candidates"]), 2)

    def test_each_component_matters_not_just_total(self):
        result = self.match(component_limit=0, total_limit=.01)
        self.assertIsNone(result["selected"])
        self.assertTrue(any(c["gaps_fraction"]["total"] < .01 for c in result["candidates"]))

    def test_invalid_bounds_and_limits_rejected(self):
        for changes in ({"flat_range": [0, 256]}, {"flat_range": [3, 2]}, {"flat_range": [1, 99999]},
                        {"flat_range": [True, 256]}, {"component_limit": .02},
                        {"total_limit": float("nan")}, {"graph_width": True}):
            with self.assertRaises(ValueError):
                self.match(**changes)

    def test_operator_isolation_and_no_unused_padding(self):
        result = verify_modules(self.contract, 64, 68)
        self.assertTrue(result["physical_self_only_weights_identical"])
        self.assertTrue(result["operator_only_change"])
        self.assertTrue(result["flat_outputs_invariant_to_neural_adjacency"])
        self.assertTrue(result["weights_unchanged"])
        for arch in ("graph", "flat"):
            self.assertEqual(result["actual_inventory"][arch]["components"], result["backward_participation_numel"][arch])
        self.assertFalse(result["nonzero_or_useful_gradient_claimed"])

    def test_check_leaves_global_torch_rng_unchanged(self):
        before = torch.random.get_rng_state().clone()
        verify_modules(self.contract, 64, 68)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_historical_report_hash_required(self):
        with self.assertRaisesRegex(ValueError, "hash"):
            load_contract(ROOT / self.config["schema_source"], "0" * 64)

    def test_every_execution_permission_must_be_explicitly_false(self):
        for key in self.config["permissions"]:
            changed = copy.deepcopy(self.config)
            changed["permissions"][key] = True
            with self.assertRaises(ValueError):
                validate_config(changed)
        del changed["permissions"][key]
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_objective_and_baselines_are_not_silently_changed(self):
        for key, value in (("gamma", .99), ("return_steps", 4), ("legacy_cache_import", True),
                           ("terminal_bootstrap", True), ("online_gate_parameter_updates", True),
                           ("gamma", True), ("return_steps", 1.0)):
            config = copy.deepcopy(self.config)
            config["proposed_training_objective"][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)
        config = copy.deepcopy(self.config)
        config["required_deployable_comparators"].pop()
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_renaming_missing_evidence_cannot_remove_a_requirement(self):
        config = copy.deepcopy(self.config)
        evidence = config["scientific_launch_evidence"]
        evidence["anything"] = evidence.pop("optimizer_replay_and_exact_resume_acceptance")
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_primary_and_secondary_questions_cannot_be_swapped(self):
        for key, value in (("primary_contrast", "graph_minus_flat"), ("secondary_contrasts", [])):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_count_check_cannot_authorize_launch(self):
        with patch("src.env.patient_capacity_planning.PatientConditionCapacityEnv.step", side_effect=AssertionError("no rollout")), \
                patch("torch.optim.Adam.step", side_effect=AssertionError("no updates")):
            result = audit(self.config)
        self.assertTrue(result["parameter_preflight_passed"])
        self.assertFalse(result["launch_authorized"])
        self.assertEqual(result["environment_steps"], 0)
        self.assertEqual(len(result["scientific_launch_blockers"]), 8)
        config = copy.deepcopy(self.config)
        config["scientific_launch_evidence"] = {key: "claimed approved" for key in config["scientific_launch_evidence"]}
        result = audit(config)
        self.assertFalse(result["launch_authorized"])
        self.assertTrue(all(b["status"] == "unreviewed_reference" for b in result["scientific_launch_blockers"]))

    def test_report_is_repeatable_and_does_not_consume_rewards(self):
        first = audit(self.config)
        second = audit(self.config)
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            original = json.loads((ROOT / self.config["schema_source"]).read_text())
            original["cases"][0]["bellman_targets"] = [999999]
            original["cases"][0]["steps"] = []
            path.write_text(json.dumps(original))
            import hashlib
            contract = load_contract(path, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(contract, self.contract)
            self.assertEqual(parameter_counts(contract, "graph", 64), first["matching"]["reference_counts"])


if __name__ == "__main__":
    unittest.main()
