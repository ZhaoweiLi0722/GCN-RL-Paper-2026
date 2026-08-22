import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluation.evaluate_multiscenario_network_residual import (
    ResidualUsageMonitor,
    apply_multiscenario_env_contract,
    apply_clinical_noninferiority,
    configure_deployment,
    deployment_candidate_label,
    normalized_clinical_noninferiority,
    normalized_deployment_candidate,
    resolve_checkpoint_variants,
    rows_by_scenario,
    select_deployment_candidate,
    select_fixed_deployment_candidate,
)


def _result(
    scale: float,
    cost_difference: float,
    *,
    aggregate_safe: bool = True,
    all_scenarios_safe: bool = True,
) -> dict:
    return {
        "candidate": {
            "scale": scale,
            "group_thresholds": [0.7, 0.8, 1.0],
        },
        "aggregate": {
            "cost_difference_mean": cost_difference,
            "clinically_noninferior": aggregate_safe,
        },
        "all_scenarios_clinically_noninferior": all_scenarios_safe,
    }


class MultiScenarioResidualEvaluationTests(unittest.TestCase):
    def test_eval_restores_shared_observation_contract_only(self) -> None:
        scenario_env = {
            "scenario_name": "abrupt",
            "demand_regime_change_step": 26,
            "include_demand_sequence_state": False,
        }
        snapshot = {
            "multi_scenario_training": {
                "env_overrides": {
                    "include_demand_history_state": True,
                    "demand_history_window": 12,
                    "include_demand_sequence_state": True,
                    "demand_sequence_length": 12,
                }
            }
        }

        merged = apply_multiscenario_env_contract(scenario_env, snapshot)

        self.assertEqual(merged["scenario_name"], "abrupt")
        self.assertEqual(merged["demand_regime_change_step"], 26)
        self.assertTrue(merged["include_demand_sequence_state"])
        self.assertEqual(merged["demand_sequence_length"], 12)
        self.assertFalse(scenario_env["include_demand_sequence_state"])

    def test_residual_usage_monitor_counts_action_groups(self) -> None:
        class Agent:
            def reset(self):
                self.last_residual_action = np.zeros(8, dtype=np.float32)

            def select_action(self, state, explore=False, env=None):
                self.last_residual_action = np.asarray(
                    [0, 0, 0.2, 0, 0, 0, 0, 0],
                    dtype=np.float32,
                )
                return np.zeros(8, dtype=np.float32)

        monitor = ResidualUsageMonitor(Agent(), num_facilities=2)
        monitor.reset()
        monitor.select_action(np.zeros(3, dtype=np.float32))
        summary = monitor.summary()

        self.assertEqual(summary["corrected_decisions"], 1)
        self.assertEqual(
            summary["group_correction_rates"]["reagent_transfer"],
            1.0,
        )
        self.assertEqual(
            summary["group_correction_rates"]["capacity_transfer"],
            0.0,
        )

    def test_residual_usage_monitor_counts_structured_option_delta(self) -> None:
        class Agent:
            def reset(self):
                self._pending_anchor_action = None

            def select_action(self, state, explore=False, env=None):
                self._pending_anchor_action = np.zeros(
                    8,
                    dtype=np.float32,
                )
                action = self._pending_anchor_action.copy()
                action[4] = 0.25
                return action

        monitor = ResidualUsageMonitor(Agent(), num_facilities=2)
        monitor.reset()
        monitor.select_action(np.zeros(3, dtype=np.float32))
        summary = monitor.summary()

        self.assertEqual(summary["corrected_decisions"], 1)
        self.assertEqual(
            summary["group_correction_rates"]["capacity_transfer"],
            1.0,
        )

    def test_monitor_decomposes_capacity_action_effect(self) -> None:
        class Agent:
            def reset(self):
                self.last_residual_action = np.zeros(8, dtype=np.float32)

            def _base_action_from_state_np(self, state):
                return np.asarray(
                    [0, 0, 0, 0, -0.4, 0.3, 0, 0],
                    dtype=np.float32,
                )

            def select_action(self, state, explore=False, env=None):
                self.last_residual_action = np.asarray(
                    [0, 0, 0, 0, -0.2, -0.4, 0, 0],
                    dtype=np.float32,
                )
                return np.asarray(
                    [0, 0, 0, 0, -0.6, -0.1, 0, 0],
                    dtype=np.float32,
                )

        monitor = ResidualUsageMonitor(Agent(), num_facilities=2)
        monitor.reset()
        monitor.select_action(np.zeros(3, dtype=np.float32))
        effect = monitor.summary()["capacity_action_effect"]

        self.assertAlmostEqual(effect["anchor_l1"], 0.7)
        self.assertAlmostEqual(effect["final_l1"], 0.7)
        self.assertAlmostEqual(effect["delta_l1"], 0.6)
        self.assertAlmostEqual(effect["amplification_l1"], 0.2)
        self.assertAlmostEqual(effect["damping_l1"], 0.3)
        self.assertAlmostEqual(effect["reversal_l1"], 0.1)
        self.assertAlmostEqual(effect["new_flow_l1"], 0.0)

    def test_candidate_validation_rejects_bad_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold"):
            normalized_deployment_candidate(
                {
                    "scale": 0.5,
                    "group_thresholds": [0.7, 1.2, 1.0],
                }
            )

    def test_candidate_preserves_temporal_guard_settings(self) -> None:
        candidate = normalized_deployment_candidate(
            {
                "scale": 0.5,
                "group_thresholds": [0.7, 0.8, 1.0],
                "temporal_guard": {
                    "enabled": True,
                    "cooldown_steps": 2,
                },
            }
        )

        self.assertTrue(candidate["temporal_guard"]["enabled"])
        self.assertEqual(
            candidate["temporal_guard"]["cooldown_steps"],
            2,
        )

    def test_candidate_preserves_endpoint_projection_settings(self) -> None:
        candidate = normalized_deployment_candidate(
            {
                "scale": 0.5,
                "group_thresholds": [0.7, 0.8],
                "endpoint_projection": {
                    "enabled": True,
                    "max_endpoints_per_side": 2,
                },
            }
        )

        self.assertTrue(candidate["endpoint_projection"]["enabled"])
        self.assertEqual(
            candidate["endpoint_projection"]["max_endpoints_per_side"],
            2,
        )

    def test_candidate_preserves_nonnegative_anchor_q_margin(self) -> None:
        candidate = normalized_deployment_candidate(
            {
                "scale": 1.0,
                "group_thresholds": [0.0],
                "anchor_q_margin": 0.05,
            }
        )

        self.assertAlmostEqual(candidate["anchor_q_margin"], 0.05)
        self.assertIn("qmargin0.05", deployment_candidate_label(candidate))
        with self.assertRaisesRegex(ValueError, "anchor_q_margin"):
            normalized_deployment_candidate(
                {
                    "scale": 1.0,
                    "group_thresholds": [0.0],
                    "anchor_q_margin": -0.01,
                }
            )

    def test_configure_deployment_applies_temporal_guard(self) -> None:
        class Agent:
            residual_scale_vector = np.ones(8, dtype=np.float32)
            correction_gate_groups = ("reagent_transfer",)
            correction_gate_group_thresholds = (0.5,)

            def configure_residual_temporal_guard(self, settings):
                self.guard_settings = dict(settings)

        agent = Agent()
        configure_deployment(
            agent,
            {
                "scale": 0.25,
                "group_thresholds": [0.75],
                "temporal_guard": {
                    "enabled": True,
                    "cooldown_steps": 1,
                },
            },
        )

        np.testing.assert_allclose(
            agent.residual_scale_vector,
            0.25,
        )
        self.assertEqual(
            agent.correction_gate_group_thresholds,
            (0.75,),
        )
        self.assertEqual(agent.guard_settings["cooldown_steps"], 1)

    def test_configure_deployment_applies_endpoint_projection(self) -> None:
        class Agent:
            residual_scale_vector = np.ones(8, dtype=np.float32)
            correction_gate_groups = ("reagent_transfer",)
            correction_gate_group_thresholds = (0.5,)

            def configure_residual_endpoint_projection(self, settings):
                self.projection_settings = dict(settings)

        agent = Agent()
        configure_deployment(
            agent,
            {
                "scale": 0.25,
                "group_thresholds": [0.75],
                "endpoint_projection": {
                    "enabled": True,
                    "max_endpoints_per_side": 1,
                },
            },
        )

        self.assertTrue(agent.projection_settings["enabled"])
        self.assertEqual(
            agent.projection_settings["max_endpoints_per_side"],
            1,
        )

    def test_configure_deployment_supports_structured_option_gate(self) -> None:
        class Agent:
            correction_gate_threshold = 0.7
            correction_gate_groups = ()

        enabled = Agent()
        configure_deployment(
            enabled,
            {
                "scale": 1.0,
                "group_thresholds": [0.9],
            },
        )
        self.assertAlmostEqual(enabled.correction_gate_threshold, 0.9)

        fallback = Agent()
        configure_deployment(
            fallback,
            {
                "scale": 0.0,
                "group_thresholds": [1.0],
            },
        )
        self.assertAlmostEqual(fallback.correction_gate_threshold, 1.0)

    def test_configure_deployment_applies_anchor_q_margin(self) -> None:
        class Agent:
            correction_gate_threshold = 0.0
            correction_gate_groups = ()
            anchor_q_margin = 0.0

        agent = Agent()
        configure_deployment(
            agent,
            {
                "scale": 1.0,
                "group_thresholds": [0.0],
                "anchor_q_margin": 0.075,
            },
        )

        self.assertAlmostEqual(agent.anchor_q_margin, 0.075)

    def test_monitor_reports_temporal_suppression(self) -> None:
        class Agent:
            def reset(self):
                self.last_residual_action = np.zeros(8, dtype=np.float32)

            def select_action(self, state, explore=False, env=None):
                self.last_residual_action = np.asarray(
                    [0, 0, 0.1, -0.1, 0, 0, 0, 0],
                    dtype=np.float32,
                )
                self.last_residual_guard_info = {
                    "raw_l1": 0.8,
                    "applied_l1": 0.2,
                }
                return np.zeros(8, dtype=np.float32)

        monitor = ResidualUsageMonitor(Agent(), num_facilities=2)
        monitor.reset()
        monitor.select_action(np.zeros(3, dtype=np.float32))
        summary = monitor.summary()

        self.assertEqual(summary["temporally_suppressed_decisions"], 1)
        self.assertAlmostEqual(summary["raw_residual_l1"], 0.8)
        self.assertAlmostEqual(summary["applied_residual_l1"], 0.2)
        self.assertAlmostEqual(summary["temporally_suppressed_l1"], 0.6)

    def test_deployment_can_preserve_checkpoint_group_thresholds(self) -> None:
        class Agent:
            residual_scale_vector = np.ones(4, dtype=np.float32)
            correction_gate_groups = ("capacity_transfer",)
            correction_gate_group_thresholds = (0.47,)

        agent = Agent()
        candidate = normalized_deployment_candidate(
            {
                "scale": 0.5,
                "use_checkpoint_group_thresholds": True,
            }
        )

        configure_deployment(agent, candidate)

        self.assertEqual(agent.correction_gate_group_thresholds, (0.47,))
        np.testing.assert_allclose(
            agent.residual_scale_vector,
            np.full(4, 0.5, dtype=np.float32),
        )

    def test_selection_uses_validation_cost_with_clinical_guardrail(self) -> None:
        selected = select_deployment_candidate(
            [
                _result(0.0, 0.0),
                _result(0.25, -10.0),
                _result(0.5, -20.0, aggregate_safe=False),
            ],
            strict_scenario_guardrail=False,
        )

        self.assertEqual(selected["candidate"]["scale"], 0.25)
        self.assertEqual(selected["selection_source"], "validation_only")

    def test_strict_selection_rejects_single_scenario_harm(self) -> None:
        selected = select_deployment_candidate(
            [
                _result(0.0, 0.0),
                _result(0.25, -10.0, all_scenarios_safe=False),
            ],
            strict_scenario_guardrail=True,
        )

        self.assertEqual(selected["candidate"]["scale"], 0.0)

    def test_selection_can_choose_pretrain_over_online_final(self) -> None:
        final = _result(0.25, -5.0)
        final["checkpoint_variant"] = "final"
        pretrain = _result(0.25, -10.0)
        pretrain["checkpoint_variant"] = "pretrain"

        selected = select_deployment_candidate(
            [final, pretrain],
            strict_scenario_guardrail=False,
        )

        self.assertEqual(selected["checkpoint_variant"], "pretrain")

    def test_fixed_selection_ignores_validation_ranking(self) -> None:
        anchor = _result(0.0, 0.0)
        anchor["checkpoint_variant"] = "final"
        fixed = _result(
            0.5,
            10.0,
            aggregate_safe=False,
            all_scenarios_safe=False,
        )
        fixed["checkpoint_variant"] = "final"

        selected = select_fixed_deployment_candidate(
            [anchor, fixed],
            candidate=fixed["candidate"],
            checkpoint_variant="final",
        )

        self.assertEqual(selected["candidate"]["scale"], 0.5)
        self.assertEqual(
            selected["selection_source"],
            "pre_registered_fixed",
        )
        self.assertFalse(
            selected["all_scenarios_clinically_noninferior"]
        )

    def test_fixed_selection_requires_exactly_one_match(self) -> None:
        fixed = _result(0.5, -10.0)
        fixed["checkpoint_variant"] = "final"

        with self.assertRaises(ValueError):
            select_fixed_deployment_candidate(
                [fixed, fixed],
                candidate=fixed["candidate"],
                checkpoint_variant="final",
            )

    def test_selection_applies_preregistered_statistical_guardrails(self) -> None:
        anchor = _result(0.0, 0.0)
        anchor["aggregate"].update(
            {
                "cost_gap_pct": 0.0,
                "cost_difference_ci_high": 0.0,
                "paired_win_rate": 0.0,
            }
        )
        noisy = _result(0.25, -20.0)
        noisy["aggregate"].update(
            {
                "cost_gap_pct": -0.005,
                "cost_difference_ci_high": 1.0,
                "paired_win_rate": 0.7,
            }
        )
        supported = _result(0.5, -10.0)
        supported["aggregate"].update(
            {
                "cost_gap_pct": -0.02,
                "cost_difference_ci_high": -1.0,
                "paired_win_rate": 0.6,
            }
        )

        selected = select_deployment_candidate(
            [anchor, noisy, supported],
            strict_scenario_guardrail=True,
            selection_guardrails={
                "max_cost_gap_pct": -0.01,
                "max_cost_difference_ci_high": 0.0,
                "min_paired_win_rate": 0.55,
            },
        )

        self.assertEqual(selected["candidate"]["scale"], 0.5)

    def test_statistical_guardrail_falls_back_to_anchor(self) -> None:
        anchor = _result(0.0, 0.0)
        anchor["aggregate"]["cost_gap_pct"] = 0.0
        candidate = _result(0.25, -10.0)
        candidate["aggregate"]["cost_gap_pct"] = -0.005

        selected = select_deployment_candidate(
            [anchor, candidate],
            strict_scenario_guardrail=False,
            selection_guardrails={"max_cost_gap_pct": -0.01},
        )

        self.assertEqual(selected["candidate"]["scale"], 0.0)

    def test_scenario_cost_guardrail_rejects_hidden_nominal_harm(self) -> None:
        anchor = _result(0.0, 0.0)
        anchor["per_scenario"] = {
            "regional": {"cost_gap_pct": 0.0},
            "nominal": {"cost_gap_pct": 0.0},
        }
        candidate = _result(0.25, -10.0)
        candidate["per_scenario"] = {
            "regional": {"cost_gap_pct": -1.0},
            "nominal": {"cost_gap_pct": 0.1},
        }

        selected = select_deployment_candidate(
            [anchor, candidate],
            strict_scenario_guardrail=True,
            selection_guardrails={
                "max_scenario_cost_gap_pct": 0.0,
            },
        )

        self.assertEqual(selected["candidate"]["scale"], 0.0)

    def test_paired_ci_clinical_noninferiority_uses_margins(self) -> None:
        anchor = [
            {
                "completion_service_level": 0.5,
                "patients_lost": 100.0,
                "patient_ineligibility_during_manufacturing_rate": 0.08,
            },
            {
                "completion_service_level": 0.5,
                "patients_lost": 100.0,
                "patient_ineligibility_during_manufacturing_rate": 0.08,
            },
        ]
        candidate = [
            {
                "completion_service_level": 0.498,
                "patients_lost": 98.0,
                "patient_ineligibility_during_manufacturing_rate": 0.078,
            },
            {
                "completion_service_level": 0.502,
                "patients_lost": 100.0,
                "patient_ineligibility_during_manufacturing_rate": 0.082,
            },
        ]
        summary = {}

        apply_clinical_noninferiority(
            summary,
            candidate,
            anchor,
            {
                "mode": "paired_ci",
                "z_value": 1.96,
                "margins": {
                    "completion_service_level": 0.005,
                    "patients_lost": 1.0,
                    "patient_ineligibility_during_manufacturing_rate": 0.005,
                },
            },
        )

        self.assertTrue(summary["clinically_noninferior"])
        self.assertEqual(
            summary["clinical_noninferiority_mode"],
            "paired_ci",
        )

    def test_paired_ci_rejects_uncertain_zero_margin_result(self) -> None:
        anchor = [
            {
                "completion_service_level": 0.5,
                "patients_lost": 100.0,
                "patient_ineligibility_during_manufacturing_rate": 0.08,
            },
            {
                "completion_service_level": 0.5,
                "patients_lost": 100.0,
                "patient_ineligibility_during_manufacturing_rate": 0.08,
            },
        ]
        candidate = [
            {
                "completion_service_level": 0.51,
                "patients_lost": 100.0,
                "patient_ineligibility_during_manufacturing_rate": 0.08,
            },
            {
                "completion_service_level": 0.495,
                "patients_lost": 100.0,
                "patient_ineligibility_during_manufacturing_rate": 0.08,
            },
        ]
        summary = {}

        apply_clinical_noninferiority(
            summary,
            candidate,
            anchor,
            {"mode": "paired_ci"},
        )

        self.assertFalse(summary["clinically_noninferior"])

    def test_clinical_noninferiority_rejects_negative_margin(self) -> None:
        with self.assertRaises(ValueError):
            normalized_clinical_noninferiority(
                {
                    "margins": {
                        "patients_lost": -1.0,
                    }
                }
            )

    def test_checkpoint_variants_deduplicate_identical_files(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"test")

            variants = resolve_checkpoint_variants(
                {
                    "checkpoint": str(checkpoint),
                    "pretrain_checkpoint": str(checkpoint),
                },
                ("pretrain", "final"),
            )

        self.assertEqual(tuple(variants), ("pretrain",))

    def test_checkpoint_variants_resolve_intermediate_episode(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            final = checkpoint_dir / "gcn_seed2_episode100.pt"
            intermediate = checkpoint_dir / "gcn_seed2_episode40.pt"
            final.write_bytes(b"final")
            intermediate.write_bytes(b"intermediate")

            variants = resolve_checkpoint_variants(
                {
                    "algorithm": "gcn",
                    "seed": 2,
                    "checkpoint": str(final),
                },
                ("episode40",),
            )

        self.assertEqual(variants, {"episode40": intermediate})

    def test_checkpoint_variants_reject_episode_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
            resolve_checkpoint_variants({}, ("episode0",))

    def test_anchor_rows_are_grouped_without_aliasing(self) -> None:
        rows = [
            {"scenario": "nominal", "total_cost": 1.0},
            {"scenario": "drift", "total_cost": 2.0},
        ]

        grouped = rows_by_scenario(rows)
        grouped["nominal"][0]["total_cost"] = 99.0

        self.assertEqual(rows[0]["total_cost"], 1.0)
        self.assertEqual(tuple(grouped), ("nominal", "drift"))


if __name__ == "__main__":
    unittest.main()
