"""Tests for the Stage E3 label-stability criteria."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from evaluation.label_stability import (
    best_action_agreement,
    label_stability_gate,
    pairwise_sign_agreement,
)

E3_CONFIG = Path("experiments/configs/continuous_overtime_label_stability_e3.json")
E2B_CONFIG = Path("experiments/configs/continuous_overtime_headroom_e2b.json")


class FreshStreamTest(unittest.TestCase):
    """E3 must not reuse the seeds E2b selected on, or the test is circular."""

    def test_e3_seeds_are_disjoint_from_e2b(self) -> None:
        e3 = json.loads(E3_CONFIG.read_text())
        e2b = json.loads(E2B_CONFIG.read_text())
        e3_seeds = set(e3["replications"]["discovery_seeds"]) | set(
            e3["replications"]["validation_seeds"]
        )
        e2b_seeds = set(e2b["replications"]["discovery_seeds"]) | set(
            e2b["replications"]["validation_seeds"]
        )
        self.assertEqual(e3_seeds & e2b_seeds, set())
        self.assertEqual(e3["output_root"], "results/continuous_overtime_label_stability_e3")

    def test_e3_holds_the_e2b_environment_fixed(self) -> None:
        e3 = json.loads(E3_CONFIG.read_text())
        e2b = json.loads(E2B_CONFIG.read_text())
        # Replication is only meaningful if states, ladder and physics match.
        for key in ("overtime_env_overrides", "state_generation", "action_ladder", "scenarios"):
            self.assertEqual(e3[key], e2b[key], f"{key} drifted between E2b and E3")


class BestActionAgreementTest(unittest.TestCase):
    def test_full_agreement(self) -> None:
        a = {("s1", "u_0.50"): 10.0, ("s1", "u_1.00"): 20.0, ("s1", "anchor"): 99.0}
        result = best_action_agreement(a, dict(a))
        self.assertEqual(result["best_action_agreement"], 1.0)

    def test_disagreement_counted(self) -> None:
        a = {("s1", "u_0.50"): 10.0, ("s1", "u_1.00"): 20.0,
             ("s2", "u_0.50"): 10.0, ("s2", "u_1.00"): 20.0}
        b = {("s1", "u_0.50"): 10.0, ("s1", "u_1.00"): 20.0,
             ("s2", "u_0.50"): 30.0, ("s2", "u_1.00"): 20.0}
        result = best_action_agreement(a, b)
        self.assertEqual(result["agreeing_states"], 1)
        self.assertEqual(result["best_action_agreement"], 0.5)

    def test_anchor_is_never_a_candidate(self) -> None:
        a = {("s1", "anchor"): 1.0, ("s1", "u_0.50"): 10.0, ("s1", "u_1.00"): 20.0}
        result = best_action_agreement(a, dict(a))
        self.assertEqual(result["per_state"]["s1"]["stream_a_best"], "u_0.50")

    def test_missing_arm_rejected(self) -> None:
        a = {("s1", "u_0.50"): 1.0, ("s1", "u_1.00"): 2.0}
        with self.assertRaisesRegex(ValueError, "missing arms"):
            best_action_agreement(a, {("s1", "u_0.50"): 1.0})


class PairwiseSignAgreementTest(unittest.TestCase):
    def test_immaterial_pairs_excluded_not_counted_against(self) -> None:
        # The pair differs by 1 in both streams but with opposite sign; below
        # threshold it must be ignored rather than scored as a disagreement.
        a = {("s1", "u_0.50"): 10.0, ("s1", "u_1.00"): 11.0}
        b = {("s1", "u_0.50"): 11.0, ("s1", "u_1.00"): 10.0}
        result = pairwise_sign_agreement(a, b, material_threshold=5.0)
        self.assertEqual(result["material_pairs"], 0)
        self.assertEqual(result["pairwise_sign_agreement"], 0.0)

    def test_material_disagreement_scored(self) -> None:
        a = {("s1", "u_0.50"): 10.0, ("s1", "u_1.00"): 100.0}
        b = {("s1", "u_0.50"): 100.0, ("s1", "u_1.00"): 10.0}
        result = pairwise_sign_agreement(a, b, material_threshold=5.0)
        self.assertEqual(result["material_pairs"], 1)
        self.assertEqual(result["pairwise_sign_agreement"], 0.0)

    def test_material_agreement_scored(self) -> None:
        a = {("s1", "u_0.50"): 10.0, ("s1", "u_1.00"): 100.0}
        b = {("s1", "u_0.50"): 20.0, ("s1", "u_1.00"): 200.0}
        result = pairwise_sign_agreement(a, b, material_threshold=5.0)
        self.assertEqual(result["pairwise_sign_agreement"], 1.0)


class LabelStabilityGateTest(unittest.TestCase):
    def test_routing_channel_shape_fails(self) -> None:
        # Stage G1's actual numbers: 54.5% argmax, 82.5% pairwise.
        decision = label_stability_gate(
            {"best_action_agreement": 0.545},
            {"pairwise_sign_agreement": 0.825, "material_pairs": 1287},
            minimum_best_action_agreement=0.70,
            minimum_pairwise_agreement=0.80,
        )
        self.assertFalse(decision["label_stability_gate_passed"])
        self.assertTrue(decision["pairwise_criterion_passed"])
        self.assertFalse(decision["best_action_criterion_passed"])
        self.assertEqual(decision["classification"], "unstable_counterfactual_labels")
        self.assertFalse(decision["e4_authorized"])

    def test_pairwise_alone_cannot_carry_the_gate(self) -> None:
        decision = label_stability_gate(
            {"best_action_agreement": 0.10},
            {"pairwise_sign_agreement": 1.0, "material_pairs": 10},
            minimum_best_action_agreement=0.70,
            minimum_pairwise_agreement=0.80,
        )
        self.assertFalse(decision["label_stability_gate_passed"])

    def test_both_passing_authorizes_e4(self) -> None:
        decision = label_stability_gate(
            {"best_action_agreement": 0.90},
            {"pairwise_sign_agreement": 0.95, "material_pairs": 500},
            minimum_best_action_agreement=0.70,
            minimum_pairwise_agreement=0.80,
        )
        self.assertTrue(decision["label_stability_gate_passed"])
        self.assertTrue(decision["e4_authorized"])
        self.assertEqual(
            decision["classification"], "labels_replicate_on_fresh_streams"
        )


if __name__ == "__main__":
    unittest.main()
