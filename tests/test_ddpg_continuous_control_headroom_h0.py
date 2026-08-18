import copy
import json
import unittest
from pathlib import Path

from evaluation.audit_ddpg_continuous_control_headroom import (
    FORMAL_HOLDOUT_SEED,
    diagnostic_crn_seed_sequence,
    load_audit_config,
    runtime_for_scenario,
    smoke_config,
    stage_h0_decision,
    summarize_group,
    validate_config,
)
from src.rl.experiment import build_env


class DDPGContinuousControlHeadroomH0Tests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = (
            root
            / "experiments/configs/"
            "patient_indexed_specimen_routing_ddpg_continuous_control_"
            "headroom_h0.json"
        )
        self.config = json.loads(path.read_text(encoding="utf-8"))

    def test_locked_config_uses_fresh_disjoint_streams(self) -> None:
        validate_config(self.config)
        used = diagnostic_crn_seed_sequence(self.config)

        self.assertEqual(len(used), 246)
        self.assertEqual(len(used), len(set(used)))
        self.assertNotIn(FORMAL_HOLDOUT_SEED, used)
        self.assertGreater(min(used), 110_999_999)

        smoke = smoke_config(self.config)
        validate_config(smoke)
        self.assertEqual(smoke["audit"]["decision_steps"], [0])
        self.assertEqual(smoke["audit"]["horizon"], 4)
        self.assertEqual(len(diagnostic_crn_seed_sequence(smoke)), 4)

    def test_runtime_rebuilds_declared_online_scenario_not_reference_env(self) -> None:
        runtime = {
            "seed": 60,
            "env": {"scenario_name": "routing_nominal_history"},
            "multi_scenario_training": {
                "scenarios": ["routing_persistent_hotspot_cluster1"],
                "env_overrides": {},
            },
        }
        rebuilt = runtime_for_scenario(
            self.config,
            runtime=runtime,
            algorithm="gcn_residual_mdl2_network_ddpg_afd",
            scenario="routing_persistent_hotspot_cluster1",
        )
        env = build_env(rebuilt, seed=1)

        self.assertEqual(runtime["env"]["scenario_name"], "routing_nominal_history")
        self.assertEqual(env.scenario_name, "routing_persistent_hotspot_cluster1")
        self.assertEqual(
            env.config.demand_regime_initial_multipliers[5],
            1.4,
        )

    def test_nonstationary_child_config_inherits_locked_protocol(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / (
            "experiments/configs/patient_indexed_specimen_routing_ddpg_"
            "continuous_control_nonstationary_h1.json"
        )
        config = load_audit_config(path)

        validate_config(config)
        self.assertEqual(len(config["audit"]["explicit_options"]), 12)
        self.assertEqual(config["audit"]["material_improvement"], 1_000_000.0)
        self.assertTrue(
            config["audit"]["allow_out_of_distribution_scenario_probe"]
        )
        self.assertEqual(len(config["locked_files"]), 6)
        self.assertGreater(min(diagnostic_crn_seed_sequence(config)), 124_999_999)

    def test_group_summary_separates_execution_and_prospective_success(self) -> None:
        rows = []
        for step in (0, 1):
            rows.append(
                {
                    "training_seed": 60,
                    "step": step,
                    "candidate_index": 0,
                    "candidate_group": "anchor",
                    "execution_id": "full-anchor",
                    "discovery_cost_advantage": 0.0,
                    "validation_cost_advantage": 0.0,
                    "discovery_clinical_noninferior": True,
                    "validation_clinical_noninferior": True,
                }
            )
            for option in range(1, 5):
                best = option == 1
                rows.append(
                    {
                        "training_seed": 60,
                        "step": step,
                        "candidate_index": option,
                        "candidate_group": "reagent_transfer",
                        "execution_survived": option != 4,
                        "anchor_execution_id": f"anchor-{step}",
                        "execution_id": (
                            f"option-{step}-{option}"
                            if option != 4
                            else f"anchor-{step}"
                        ),
                        "discovery_cost_advantage": 2_000_000.0 if best else 0.0,
                        "validation_cost_advantage": 1_500_000.0 if best else 0.0,
                        "discovery_clinical_noninferior": True,
                        "validation_clinical_noninferior": True,
                    }
                )

        result = summarize_group(
            rows,
            group="reagent_transfer",
            material_improvement=1_000_000.0,
            minimum_pairwise_cost_gap=250_000.0,
        )

        self.assertEqual(result["states"], 2)
        self.assertEqual(result["nonanchor_execution_survival_fraction"], 0.75)
        self.assertEqual(result["fully_distinct_execution_state_fraction"], 0.0)
        self.assertEqual(result["discovery_validation_best_action_agreement"], 1.0)
        self.assertEqual(
            result[
                "validation_material_clinically_noninferior_opportunity_fraction"
            ],
            1.0,
        )
        self.assertEqual(
            result["discovery_selected_validation_material_success_fraction"],
            1.0,
        )

    def test_decision_only_authorizes_larger_frozen_audit(self) -> None:
        passing = {
            "nonanchor_execution_survival_fraction": 0.90,
            "discovery_validation_best_action_agreement": 0.80,
            "discovery_validation_pairwise_sign_agreement": 0.90,
            "validation_material_clinically_noninferior_opportunity_fraction": 0.20,
            "discovery_selected_validation_material_success_fraction": 0.10,
        }
        failing = copy.deepcopy(passing)
        failing["nonanchor_execution_survival_fraction"] = 0.20
        metrics = {
            "specimen_transfer": failing,
            "reagent_transfer": passing,
            "combined_transfer": failing,
        }

        result = stage_h0_decision(
            metrics,
            gates=self.config["stage_h0_gate"],
            smoke=False,
        )
        smoke = stage_h0_decision(
            metrics,
            gates=self.config["stage_h0_gate"],
            smoke=True,
        )

        self.assertEqual(result["selected_group"], "reagent_transfer")
        self.assertTrue(result["full_156_state_frozen_audit_authorized"])
        self.assertFalse(result["online_training_authorized"])
        self.assertFalse(result["formal_confirmation_authorized"])
        self.assertEqual(smoke["classification"], "smoke_only_no_scientific_decision")
        self.assertFalse(smoke["full_156_state_frozen_audit_authorized"])


if __name__ == "__main__":
    unittest.main()
