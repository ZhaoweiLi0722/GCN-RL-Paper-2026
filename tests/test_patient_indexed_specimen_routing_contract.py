"""Cross-module and experiment-contract tests for specimen routing."""

from __future__ import annotations

import base64
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from evaluation.compare_patient_indexed_specimen_routing import (
    compare_attribution,
)
from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    make_training_config,
    resolve_budget,
    select_scenarios,
)
from evaluation.train_multiscenario_network_residual import (
    actor_checkpoint_drift,
    agent_parameter_count,
)
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env
from src.rl.networks import torch
from src.rl.residual_options import (
    make_explicit_residual_option_specs,
    residual_option_actions_from_patterns,
)


PLAN_PATH = (
    "experiments/configs/"
    "patient_indexed_specimen_routing_benchmark.json"
)
GCN = "gcn_residual_mdl2_network_ddpg_afd"
FLAT = "flat_residual_mdl2_network_ddpg_afd"
RESULT_ROOT = "results/patient_indexed_specimen_routing_recovery7/"
MAC_RESULT_ROOT = "results/patient_indexed_specimen_routing_mac_mps_primary/"
STAGE_C_RESULT_ROOT = (
    "results/patient_indexed_specimen_routing_stage_c_td3_development"
)
DDPG_SUPPORT_RESULT_ROOT = (
    "results/patient_indexed_specimen_routing_ddpg_support_alignment_"
    "development"
)
DDPG_PERSISTENT_SHIFT_RESULT_ROOT = (
    "results/patient_indexed_specimen_routing_ddpg_persistent_shift_"
    "development"
)
DDPG_STRUCTURED_EXPLORATION_RESULT_ROOT = (
    "results/patient_indexed_specimen_routing_ddpg_structured_exploration_"
    "development"
)
DDPG_ONLINE_IDENTIFIABILITY_RESULT_ROOT = (
    "results/patient_indexed_specimen_routing_ddpg_online_identifiability_"
    "f0_development"
)
DDPG_ONLINE_PAIRED_ADVANTAGE_RESULT_ROOT = (
    "results/patient_indexed_specimen_routing_ddpg_online_paired_advantage_"
    "development"
)
DDPG_ACTOR_PROJECTION_RESULT_ROOT = (
    "results/patient_indexed_specimen_routing_ddpg_actor_projection_transfer_"
    "g0_development"
)
DDPG_LEGAL_ACTION_RANKER_RESULT_ROOT = (
    "results/patient_indexed_specimen_routing_ddpg_legal_action_ranker_"
    "g1_development"
)
DDPG_CONTINUOUS_HEADROOM_RESULT_ROOT = (
    "results/patient_indexed_specimen_routing_ddpg_continuous_control_"
    "headroom_h0_development"
)
DDPG_CONTINUOUS_NONSTATIONARY_RESULT_ROOT = (
    "results/patient_indexed_specimen_routing_ddpg_continuous_control_"
    "nonstationary_h1_development"
)
DDPG_POSTHOC_RESULT_ROOTS = {
    "patient_indexed_specimen_routing_ddpg_actor_projection_transfer_g0.json": (
        DDPG_ONLINE_PAIRED_ADVANTAGE_RESULT_ROOT,
        DDPG_ACTOR_PROJECTION_RESULT_ROOT,
    ),
    "patient_indexed_specimen_routing_ddpg_legal_action_ranker_g1.json": (
        DDPG_ONLINE_PAIRED_ADVANTAGE_RESULT_ROOT,
        DDPG_LEGAL_ACTION_RANKER_RESULT_ROOT,
    ),
    (
        "patient_indexed_specimen_routing_ddpg_continuous_control_"
        "headroom_h0.json"
    ): (
        DDPG_ONLINE_PAIRED_ADVANTAGE_RESULT_ROOT,
        DDPG_CONTINUOUS_HEADROOM_RESULT_ROOT,
    ),
    (
        "patient_indexed_specimen_routing_ddpg_continuous_control_"
        "nonstationary_h1.json"
    ): (
        DDPG_CONTINUOUS_HEADROOM_RESULT_ROOT,
        DDPG_CONTINUOUS_NONSTATIONARY_RESULT_ROOT,
    ),
}
FROZEN_TEACHER_CONFIG = (
    "patient_indexed_specimen_routing_teacher_routing.json"
)
FROZEN_TEACHER_ROOT = (
    "results/patient_indexed_specimen_routing_recovery4/teachers/routing"
)


def _scenario(plan: dict, name: str) -> dict:
    return next(item for item in plan["scenarios"] if item["name"] == name)


def _config(plan: dict, algorithm: str, scenario: dict) -> dict:
    budget = resolve_budget(plan, "routing_smoke")
    config = make_training_config(
        plan,
        "routing_smoke",
        budget,
        algorithm,
        scenario,
        0,
    )
    config["device"] = "cpu"
    return config


class RoutingExperimentContractTests(unittest.TestCase):
    def test_ddpg_confirmation_uses_untouched_seeds_and_frozen_candidate(self) -> None:
        config_root = Path("experiments/configs")
        candidate = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_bounded_gain01_"
                    "specimenonly_ddpg_sharedpreonline_quantizedspecimenste_"
                    "anchorrelreward_nstep4_materialadv_dualconfirm_"
                    "balancedcycle8.json"
                )
            ).read_text()
        )
        confirmation = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_"
                    "ddpg_confirmation_100.json"
                )
            ).read_text()
        )

        self.assertEqual(candidate["online_episodes"], 8)
        self.assertEqual(confirmation["online_episodes"], 100)
        self.assertEqual(confirmation["seeds"], [10, 11, 12, 13, 14])
        self.assertTrue(set(confirmation["seeds"]).isdisjoint({0, 1, 2}))
        self.assertEqual(
            [entry["name"] for entry in confirmation["algorithms"]],
            [GCN, FLAT],
        )
        self.assertTrue(
            all(
                entry["seeds"] == confirmation["seeds"]
                for entry in confirmation["algorithms"]
            )
        )

        candidate_contract = dict(candidate)
        confirmation_contract = dict(confirmation)
        self.assertEqual(
            confirmation_contract.pop("teacher_cache_sha256"),
            "9ba2ac0873c0f68e6ecc4b443e0eace8230f8e151cceb485fafe218a7ac78d92",
        )
        for payload in (candidate_contract, confirmation_contract):
            for key in (
                "name",
                "experimental_role",
                "budget",
                "online_episodes",
                "seeds",
                "algorithms",
                "output_root",
            ):
                payload.pop(key)
            overrides = dict(payload["config_overrides"])
            for key in (
                "checkpoint_interval",
                "training_state_checkpoint_interval",
                "progress_interval",
            ):
                overrides.pop(key)
            payload["config_overrides"] = overrides
        self.assertEqual(confirmation_contract, candidate_contract)

        overrides = confirmation["config_overrides"]
        self.assertEqual(overrides["batch_size"], 64)
        self.assertEqual(overrides["actor_lr"], 1e-5)
        self.assertEqual(overrides["actor_update_frequency"], 2)
        self.assertEqual(overrides["update_frequency"], 1)
        self.assertEqual(overrides["checkpoint_interval"], 5)
        self.assertEqual(overrides["residual_action"]["scale"], 0.1)
        self.assertEqual(
            overrides["online_advantage_self_imitation"]["minimum_return"],
            0.0005,
        )

    def test_ddpg_confirmation_evaluation_is_fixed_and_paired(self) -> None:
        config_root = Path("experiments/configs")
        manifest = (
            "results/patient_indexed_specimen_routing_mac_mps_primary/"
            "confirmation/ddpg_routing_primary_100/"
            "patient_indexed_specimen_routing_mac_mps_ddpg_confirmation_100/"
            "training_manifest.json"
        )
        filenames = {
            "final": (
                "patient_indexed_specimen_routing_mac_mps_"
                "ddpg_confirmation_100_final_eval.json"
            ),
            "pretrain": (
                "patient_indexed_specimen_routing_mac_mps_"
                "ddpg_confirmation_100_pretrain_eval.json"
            ),
        }
        evaluations = {
            variant: json.loads((config_root / filename).read_text())
            for variant, filename in filenames.items()
        }
        for variant, evaluation in evaluations.items():
            self.assertEqual(evaluation["training_manifest"], manifest)
            self.assertEqual(evaluation["algorithms"], [GCN, FLAT])
            self.assertEqual(
                evaluation["training_seeds"],
                [10, 11, 12, 13, 14],
            )
            self.assertEqual(evaluation["checkpoint_variants"], [variant])
            self.assertEqual(
                evaluation["fixed_checkpoint_variant"],
                variant,
            )
            self.assertEqual(evaluation["validation_replications"], 1)
            self.assertEqual(evaluation["holdout_replications"], 100)
            self.assertEqual(
                evaluation["fixed_deployment_candidate"],
                {
                    "scale": 1.0,
                    "use_checkpoint_group_thresholds": True,
                },
            )
            self.assertEqual(
                [item["scale"] for item in evaluation["deployment_candidates"]],
                [0.0, 1.0],
            )
        self.assertEqual(
            evaluations["final"]["validation_seed"],
            evaluations["pretrain"]["validation_seed"],
        )
        self.assertEqual(
            evaluations["final"]["holdout_seed"],
            evaluations["pretrain"]["holdout_seed"],
        )

        smoke = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_"
                    "ddpg_confirmation_100_eval_smoke.json"
                )
            ).read_text()
        )
        self.assertEqual(smoke["training_manifest"], manifest)
        self.assertEqual(smoke["training_seeds"], [10])
        self.assertEqual(smoke["checkpoint_variants"], ["pretrain", "final"])
        self.assertEqual(smoke["validation_replications"], 1)
        self.assertEqual(smoke["holdout_replications"], 1)
        self.assertNotEqual(
            smoke["holdout_seed"],
            evaluations["final"]["holdout_seed"],
        )

    def test_ddpg_confirmation_sensitivity_is_fixed_policy_and_paired(
        self,
    ) -> None:
        config_root = Path("experiments/configs")
        final = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_"
                    "ddpg_confirmation_100_final_eval.json"
                )
            ).read_text()
        )
        manifest = final["training_manifest"]
        plan = json.loads(Path(final["plan"]).read_text())
        scenarios = {
            scenario["name"]: scenario
            for scenario in plan["scenarios"]
        }
        nominal_overrides = scenarios["routing_nominal_history"][
            "env_overrides"
        ]
        filenames = {
            "routing_nominal_lead0_sensitivity": (
                "patient_indexed_specimen_routing_mac_mps_"
                "lead0_sensitivity_eval.json"
            ),
            "routing_nominal_return1_sensitivity": (
                "patient_indexed_specimen_routing_mac_mps_"
                "return1_sensitivity_eval.json"
            ),
        }
        timing_changes = {
            "routing_nominal_lead0_sensitivity": (
                "specimen_routing_lead_time_epochs",
                0,
            ),
            "routing_nominal_return1_sensitivity": (
                "finished_product_return_lead_time_epochs",
                1,
            ),
        }
        for scenario, filename in filenames.items():
            sensitivity = json.loads((config_root / filename).read_text())
            timing_field, timing_value = timing_changes[scenario]
            expected_overrides = dict(nominal_overrides)
            expected_overrides["scenario_name"] = scenario
            expected_overrides[timing_field] = timing_value
            self.assertEqual(
                scenarios[scenario]["env_config"],
                scenarios["routing_nominal_history"]["env_config"],
            )
            self.assertEqual(
                scenarios[scenario]["env_overrides"],
                expected_overrides,
            )
            self.assertEqual(sensitivity["training_manifest"], manifest)
            self.assertEqual(sensitivity["algorithms"], [GCN, FLAT])
            self.assertEqual(
                sensitivity["training_seeds"],
                [10, 11, 12, 13, 14],
            )
            self.assertEqual(sensitivity["checkpoint_variants"], ["final"])
            self.assertEqual(
                sensitivity["fixed_checkpoint_variant"],
                "final",
            )
            self.assertEqual(sensitivity["scenarios"], [scenario])
            self.assertEqual(
                sensitivity["deployment_candidates"],
                final["deployment_candidates"],
            )
            self.assertEqual(
                sensitivity["fixed_deployment_candidate"],
                final["fixed_deployment_candidate"],
            )
            self.assertEqual(
                sensitivity["validation_replications"],
                final["validation_replications"],
            )
            self.assertEqual(
                sensitivity["validation_seed"],
                final["validation_seed"],
            )
            self.assertEqual(
                sensitivity["holdout_replications"],
                final["holdout_replications"],
            )
            self.assertEqual(
                sensitivity["holdout_seed"],
                final["holdout_seed"],
            )
            self.assertEqual(
                sensitivity["clinical_noninferiority"],
                final["clinical_noninferiority"],
            )
            self.assertNotIn("config_overrides", sensitivity)

        smoke = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_"
                    "ddpg_confirmation_100_sensitivity_eval_smoke.json"
                )
            ).read_text()
        )
        self.assertEqual(smoke["training_manifest"], manifest)
        self.assertEqual(smoke["algorithms"], [GCN, FLAT])
        self.assertEqual(smoke["training_seeds"], [10])
        self.assertEqual(smoke["checkpoint_variants"], ["final"])
        self.assertEqual(smoke["scenarios"], list(filenames))
        self.assertEqual(
            smoke["fixed_deployment_candidate"],
            final["fixed_deployment_candidate"],
        )
        self.assertEqual(smoke["validation_replications"], 1)
        self.assertEqual(smoke["holdout_replications"], 1)
        self.assertNotEqual(smoke["holdout_seed"], final["holdout_seed"])

    def test_mac_mps_campaign_is_matched_and_routing_primary(self) -> None:
        config_root = Path("experiments/configs")
        smoke = json.loads(
            (config_root / "patient_indexed_specimen_routing_mac_mps_smoke.json").read_text()
        )
        primary = json.loads(
            (config_root / "patient_indexed_specimen_routing_mac_mps_primary.json").read_text()
        )
        expected_teacher = (
            "results/patient_indexed_specimen_routing_mac_mps_primary/"
            "teachers/routing/teacher_cache.npz"
        )
        for config in (smoke, primary):
            self.assertEqual(config["config_overrides"]["device"], "mps")
            self.assertEqual(config["teacher_cache"], expected_teacher)
            self.assertEqual(
                [entry["name"] for entry in config["algorithms"]],
                [GCN, FLAT],
            )
            self.assertTrue(
                all(name.startswith("routing_") for name in config["scenarios"])
            )
        self.assertEqual(smoke["online_episodes"], 5)
        self.assertEqual(smoke["seeds"], [0])
        self.assertEqual(primary["online_episodes"], 100)
        self.assertEqual(primary["seeds"], [0, 1, 2])
        self.assertEqual(primary["pretrain_epochs"], 300)

        manifest = (
            "results/patient_indexed_specimen_routing_mac_mps_primary/"
            "training/patient_indexed_specimen_routing_mac_mps_primary/"
            "training_manifest.json"
        )
        for filename, variant in (
            ("patient_indexed_specimen_routing_mac_mps_primary_eval.json", "final"),
            ("patient_indexed_specimen_routing_mac_mps_pretrain_eval.json", "pretrain"),
        ):
            evaluation = json.loads((config_root / filename).read_text())
            self.assertEqual(evaluation["training_manifest"], manifest)
            self.assertEqual(evaluation["training_seeds"], [0, 1, 2])
            self.assertEqual(evaluation["checkpoint_variants"], [variant])
            self.assertEqual(evaluation["holdout_replications"], 100)

    def test_update_frequency_sensitivity_is_single_factor_and_prespecified(self) -> None:
        config_root = Path("experiments/configs")
        primary = json.loads(
            (config_root / "patient_indexed_specimen_routing_mac_mps_primary.json").read_text()
        )
        candidate = json.loads(
            (
                config_root
                / "patient_indexed_specimen_routing_mac_mps_updatefreq4_seed0.json"
            ).read_text()
        )
        smoke = json.loads(
            (
                config_root
                / "patient_indexed_specimen_routing_mac_mps_updatefreq4_smoke.json"
            ).read_text()
        )
        protocol = json.loads(
            (
                config_root
                / "patient_indexed_specimen_routing_mac_mps_updatefreq4_protocol.json"
            ).read_text()
        )
        plan = load_benchmark_plan(PLAN_PATH)

        self.assertEqual(
            plan["algorithm_settings"][GCN]["config_overrides"]["update_frequency"],
            1,
        )
        self.assertEqual(
            plan["algorithm_settings"][GCN]["config_overrides"]["updates_per_update"],
            1,
        )
        self.assertEqual(protocol["factor"]["baseline"], 1)
        self.assertEqual(protocol["factor"]["candidate"], 4)
        self.assertEqual(candidate["config_overrides"]["update_frequency"], 4)
        self.assertEqual(smoke["config_overrides"]["update_frequency"], 4)

        primary_contract = dict(primary)
        candidate_contract = dict(candidate)
        for key in ("name", "seeds", "algorithms", "output_root"):
            primary_contract.pop(key)
            candidate_contract.pop(key)
        candidate_overrides = dict(candidate_contract["config_overrides"])
        candidate_overrides.pop("update_frequency")
        candidate_contract["config_overrides"] = candidate_overrides
        self.assertEqual(candidate_contract, primary_contract)

        fixed = protocol["fixed_training_contract"]
        self.assertEqual(fixed["baseline_expected_online_updates"], 5200)
        self.assertEqual(fixed["candidate_expected_online_updates"], 1300)
        self.assertEqual(fixed["deployment_scale"], 0.1)
        self.assertEqual(protocol["screening"]["training_seeds"], [0])
        self.assertNotEqual(
            protocol["screening"]["screening_confirmation_seed"],
            protocol["screening"]["official_holdout_seed_reserved"],
        )
        self.assertTrue(
            protocol["confirmation"]["run_only_after_screening_gate_passes"]
        )

        baseline_eval = json.loads(
            (
                config_root
                / "patient_indexed_specimen_routing_mac_mps_updatefreq4_baseline_screen_eval.json"
            ).read_text()
        )
        candidate_eval = json.loads(
            (
                config_root
                / "patient_indexed_specimen_routing_mac_mps_updatefreq4_candidate_screen_eval.json"
            ).read_text()
        )
        for evaluation in (baseline_eval, candidate_eval):
            self.assertEqual(evaluation["training_seeds"], [0])
            self.assertEqual(evaluation["validation_seed"], 8500000)
            self.assertEqual(evaluation["validation_replications"], 5)
            self.assertEqual(evaluation["holdout_seed"], 8510000)
            self.assertEqual(evaluation["holdout_replications"], 20)
            self.assertNotEqual(evaluation["holdout_seed"], 8400000)
            self.assertEqual(evaluation["fixed_checkpoint_variant"], "final")
            self.assertEqual(evaluation["fixed_deployment_candidate"]["scale"], 0.1)

        comparable_baseline = dict(baseline_eval)
        comparable_candidate = dict(candidate_eval)
        for payload in (comparable_baseline, comparable_candidate):
            for key in ("name", "training_manifest", "output_root"):
                payload.pop(key)
        self.assertEqual(comparable_candidate, comparable_baseline)

        confirmation_train = json.loads(
            (
                config_root
                / "patient_indexed_specimen_routing_mac_mps_updatefreq4_seeds1_2.json"
            ).read_text()
        )
        candidate_train_contract = dict(candidate)
        confirmation_train_contract = dict(confirmation_train)
        for payload in (candidate_train_contract, confirmation_train_contract):
            payload.pop("seeds")
            payload.pop("algorithms")
        self.assertEqual(confirmation_train_contract, candidate_train_contract)
        self.assertEqual(confirmation_train["seeds"], [1, 2])
        self.assertTrue(
            all(entry["seeds"] == [1, 2] for entry in confirmation_train["algorithms"])
        )

        confirmation_manifest = candidate_eval["training_manifest"]
        for filename, variant in (
            (
                "patient_indexed_specimen_routing_mac_mps_updatefreq4_confirm_final_eval.json",
                "final",
            ),
            (
                "patient_indexed_specimen_routing_mac_mps_updatefreq4_confirm_pretrain_eval.json",
                "pretrain",
            ),
        ):
            evaluation = json.loads((config_root / filename).read_text())
            self.assertEqual(evaluation["training_manifest"], confirmation_manifest)
            self.assertEqual(evaluation["training_seeds"], [0, 1, 2])
            self.assertEqual(evaluation["checkpoint_variants"], [variant])
            self.assertEqual(evaluation["fixed_checkpoint_variant"], variant)
            self.assertEqual(evaluation["holdout_seed"], 8400000)
            self.assertEqual(evaluation["holdout_replications"], 100)
            self.assertEqual(evaluation["fixed_deployment_candidate"]["scale"], 0.1)

        attribution = json.loads(
            (
                config_root
                / "patient_indexed_specimen_routing_mac_mps_updatefreq4_confirm_attribution.json"
            ).read_text()
        )
        self.assertEqual(attribution["bootstrap_resamples"], 20000)
        self.assertEqual(attribution["bootstrap_seed"], 8520000)
        self.assertIn("update_frequency_4/confirmation/final_scale01", attribution["routing_final_root"])
        self.assertIn(
            "update_frequency_4/confirmation/pretrain_scale01",
            attribution["routing_pretrain_root"],
        )

    def test_bounded_gain_teacher_and_ddpg_smoke_share_action_envelope(self) -> None:
        config_root = Path("experiments/configs")
        teacher = json.loads(
            (
                config_root
                / "patient_indexed_specimen_routing_mac_mps_teacher_bounded_gain01.json"
            ).read_text()
        )
        smoke = json.loads(
            (
                config_root
                / "patient_indexed_specimen_routing_mac_mps_bounded_gain01_gcn_smoke.json"
            ).read_text()
        )
        envelope = teacher["residual_action_envelope"]
        scales = smoke["config_overrides"]["residual_action"]["group_scales"]
        self.assertEqual(envelope, scales)
        self.assertEqual(smoke["config_overrides"]["update_frequency"], 1)
        self.assertEqual(
            [entry["name"] for entry in smoke["algorithms"]],
            [GCN],
        )
        self.assertTrue(
            all(
                float(option["epsilon"]) <= 0.1
                for option in teacher["explicit_options"]
            )
        )
        for variant in ("pretrain", "final"):
            evaluation = json.loads(
                (
                    config_root
                    / (
                        "patient_indexed_specimen_routing_mac_mps_"
                        f"bounded_gain01_gcn_smoke_{variant}_eval.json"
                    )
                ).read_text()
            )
            self.assertEqual(
                evaluation["fixed_deployment_candidate"]["scale"],
                1.0,
            )
            self.assertEqual(
                evaluation["fixed_checkpoint_variant"],
                variant,
            )
            self.assertNotEqual(evaluation["holdout_seed"], 8400000)

        pretrain300 = json.loads(
            (
                config_root
                / "patient_indexed_specimen_routing_mac_mps_bounded_gain01_gcn_pretrain300_screen.json"
            ).read_text()
        )
        smoke_contract = dict(smoke)
        pretrain300_contract = dict(pretrain300)
        for payload in (smoke_contract, pretrain300_contract):
            for key in ("name", "experimental_role", "pretrain_epochs", "output_root"):
                payload.pop(key)
        self.assertEqual(pretrain300_contract, smoke_contract)
        self.assertEqual(smoke["pretrain_epochs"], 5)
        self.assertEqual(pretrain300["pretrain_epochs"], 300)

        dense = json.loads(
            (
                config_root
                / "patient_indexed_specimen_routing_mac_mps_bounded_gain01_gcn_dense_pretrain300_screen.json"
            ).read_text()
        )
        pretrain300_contract = dict(pretrain300)
        dense_contract = dict(dense)
        for payload in (pretrain300_contract, dense_contract):
            for key in ("name", "experimental_role", "output_root"):
                payload.pop(key)
        dense_residual = dict(
            dense_contract["config_overrides"]["residual_action"]
        )
        self.assertFalse(dense_residual.pop("endpoint_projection")["enabled"])
        dense_contract["config_overrides"] = dict(
            dense_contract["config_overrides"]
        )
        dense_contract["config_overrides"]["residual_action"] = dense_residual
        self.assertEqual(dense_contract, pretrain300_contract)

        dense_manifest = (
            "results/patient_indexed_specimen_routing_mac_mps_primary/"
            "sensitivity/bounded_gain01/"
            "gcn_dense_pretrain300_screen_training_mps/"
            "patient_indexed_specimen_routing_mac_mps_"
            "bounded_gain01_gcn_dense_pretrain300_screen/"
            "training_manifest.json"
        )
        comparable_evaluations = []
        for variant in ("pretrain", "final"):
            evaluation = json.loads(
                (
                    config_root
                    / (
                        "patient_indexed_specimen_routing_mac_mps_"
                        "bounded_gain01_gcn_dense_pretrain300_screen_"
                        f"{variant}_eval.json"
                    )
                ).read_text()
            )
            self.assertEqual(evaluation["training_manifest"], dense_manifest)
            self.assertEqual(evaluation["fixed_checkpoint_variant"], variant)
            self.assertEqual(evaluation["fixed_deployment_candidate"]["scale"], 1.0)
            self.assertFalse(
                evaluation["fixed_deployment_candidate"]
                ["endpoint_projection"]["enabled"]
            )
            comparable = dict(evaluation)
            for key in (
                "name",
                "experimental_role",
                "checkpoint_variants",
                "fixed_checkpoint_variant",
                "output_root",
            ):
                comparable.pop(key)
            comparable_evaluations.append(comparable)
        self.assertEqual(*comparable_evaluations)

        actor_lr = json.loads(
            (
                config_root
                / "patient_indexed_specimen_routing_mac_mps_bounded_gain01_gcn_dense_actorlr1e4_screen.json"
            ).read_text()
        )
        dense_contract = dict(dense)
        actor_lr_contract = dict(actor_lr)
        for payload in (dense_contract, actor_lr_contract):
            for key in ("name", "experimental_role", "output_root"):
                payload.pop(key)
        actor_lr_overrides = dict(actor_lr_contract["config_overrides"])
        self.assertEqual(actor_lr_overrides.pop("actor_lr"), 0.0001)
        actor_lr_contract["config_overrides"] = actor_lr_overrides
        self.assertEqual(actor_lr_contract, dense_contract)

        actor_lr_manifest = (
            "results/patient_indexed_specimen_routing_mac_mps_primary/"
            "sensitivity/bounded_gain01/"
            "gcn_dense_actorlr1e4_screen_training_mps/"
            "patient_indexed_specimen_routing_mac_mps_"
            "bounded_gain01_gcn_dense_actorlr1e4_screen/"
            "training_manifest.json"
        )
        for variant in ("pretrain", "final"):
            evaluation = json.loads(
                (
                    config_root
                    / (
                        "patient_indexed_specimen_routing_mac_mps_"
                        "bounded_gain01_gcn_dense_actorlr1e4_screen_"
                        f"{variant}_eval.json"
                    )
                ).read_text()
            )
            self.assertEqual(evaluation["training_manifest"], actor_lr_manifest)
            self.assertEqual(evaluation["fixed_checkpoint_variant"], variant)
            self.assertEqual(evaluation["validation_seed"], 8680000)
            self.assertEqual(evaluation["holdout_seed"], 8690000)

        group_gate = json.loads(
            (
                config_root
                / "patient_indexed_specimen_routing_mac_mps_bounded_gain01_gcn_dense_actorlr1e4_groupgate_screen.json"
            ).read_text()
        )
        actor_lr_contract = dict(actor_lr)
        group_gate_contract = dict(group_gate)
        for payload in (actor_lr_contract, group_gate_contract):
            for key in ("name", "experimental_role", "output_root"):
                payload.pop(key)
        gated_residual = dict(
            group_gate_contract["config_overrides"]["residual_action"]
        )
        gate = gated_residual.pop("correction_gate")
        group_gate_contract["config_overrides"] = dict(
            group_gate_contract["config_overrides"]
        )
        group_gate_contract["config_overrides"]["residual_action"] = (
            gated_residual
        )
        self.assertEqual(group_gate_contract, actor_lr_contract)
        self.assertTrue(gate["enabled"])
        self.assertEqual(
            gate["groups"],
            [
                "specimen_transfer",
                "reagent_transfer",
                "capacity_transfer",
            ],
        )
        self.assertEqual(gate["threshold"], 0.5)
        self.assertEqual(gate["target_delta"], 0.04)

        phase_lr = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_"
                    "bounded_gain01_gcn_dense_pretrainlr1e4_"
                    "onlinelr1e5_groupgate_screen.json"
                )
            ).read_text()
        )
        group_gate_contract = dict(group_gate)
        phase_lr_contract = dict(phase_lr)
        for payload in (group_gate_contract, phase_lr_contract):
            for key in ("name", "experimental_role", "output_root"):
                payload.pop(key)
        group_gate_overrides = dict(
            group_gate_contract["config_overrides"]
        )
        phase_lr_overrides = dict(
            phase_lr_contract["config_overrides"]
        )
        self.assertEqual(group_gate_overrides.pop("actor_lr"), 0.0001)
        self.assertEqual(phase_lr_overrides.pop("actor_lr"), 0.00001)
        pretrain_optimizer = phase_lr_overrides.pop(
            "advantage_distillation_pretrain"
        )
        self.assertEqual(pretrain_optimizer, {"actor_lr": 0.0001})
        group_gate_contract["config_overrides"] = group_gate_overrides
        phase_lr_contract["config_overrides"] = phase_lr_overrides
        self.assertEqual(phase_lr_contract, group_gate_contract)

        phase_lr_manifest = (
            "results/patient_indexed_specimen_routing_mac_mps_primary/"
            "sensitivity/bounded_gain01/"
            "gcn_dense_pretrainlr1e4_onlinelr1e5_groupgate_"
            "screen_training_mps/"
            "patient_indexed_specimen_routing_mac_mps_bounded_gain01_"
            "gcn_dense_pretrainlr1e4_onlinelr1e5_groupgate_screen/"
            "training_manifest.json"
        )
        phase_lr_evaluations = []
        for variant in ("pretrain", "final"):
            evaluation = json.loads(
                (
                    config_root
                    / (
                        "patient_indexed_specimen_routing_mac_mps_"
                        "bounded_gain01_gcn_dense_pretrainlr1e4_"
                        "onlinelr1e5_groupgate_screen_"
                        f"{variant}_eval.json"
                    )
                ).read_text()
            )
            self.assertEqual(
                evaluation["training_manifest"],
                phase_lr_manifest,
            )
            self.assertEqual(
                evaluation["fixed_checkpoint_variant"],
                variant,
            )
            comparable = dict(evaluation)
            for key in (
                "name",
                "experimental_role",
                "checkpoint_variants",
                "fixed_checkpoint_variant",
                "output_root",
            ):
                comparable.pop(key)
            phase_lr_evaluations.append(comparable)
        self.assertEqual(*phase_lr_evaluations)

        threshold_calibration = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_"
                    "bounded_gain01_gcn_dense_pretrainlr1e4_"
                    "onlinelr1e5_groupgate_threshold_calibration_"
                    "final_eval.json"
                )
            ).read_text()
        )
        self.assertEqual(
            threshold_calibration["training_manifest"],
            phase_lr_manifest,
        )
        self.assertEqual(
            threshold_calibration["checkpoint_variants"],
            ["final"],
        )
        self.assertNotIn(
            "fixed_deployment_candidate",
            threshold_calibration,
        )
        self.assertEqual(
            [
                candidate["group_thresholds"]
                for candidate in threshold_calibration[
                    "deployment_candidates"
                ]
            ],
            [
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5],
                [0.43, 0.15, 0.15],
                [0.5, 0.2, 0.2],
                [0.5, 0.15, 0.5],
                [0.5, 0.5, 0.15],
                [0.1, 0.1, 0.2],
            ],
        )
        self.assertEqual(threshold_calibration["validation_seed"], 8710000)
        self.assertEqual(threshold_calibration["holdout_seed"], 8720000)

        broad_fixed = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_"
                    "bounded_gain01_gcn_dense_pretrainlr1e4_"
                    "onlinelr1e5_groupgate_broad_fixed_final_eval.json"
                )
            ).read_text()
        )
        self.assertEqual(broad_fixed["training_manifest"], phase_lr_manifest)
        self.assertEqual(
            broad_fixed["fixed_deployment_candidate"]["group_thresholds"],
            [0.1, 0.1, 0.2],
        )
        self.assertEqual(broad_fixed["fixed_checkpoint_variant"], "final")
        self.assertEqual(broad_fixed["validation_seed"], 8730000)
        self.assertEqual(broad_fixed["holdout_seed"], 8740000)

        dagger_collection = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_"
                    "bounded_gain01_groupgate_broad_dagger1.json"
                )
            ).read_text()
        )
        self.assertEqual(
            dagger_collection["training_manifest"],
            phase_lr_manifest,
        )
        self.assertEqual(
            dagger_collection["selection_summary"],
            threshold_calibration["output_root"] + "/summary.json",
        )
        self.assertEqual(
            dagger_collection["base_teacher_cache"],
            phase_lr["teacher_cache"],
        )
        self.assertEqual(dagger_collection["checkpoint_variant"], "final")
        self.assertEqual(
            dagger_collection["behavior_deployment_candidate"],
            {
                "scale": 1.0,
                "group_thresholds": [0.1, 0.1, 0.2],
                "endpoint_projection": {
                    "enabled": False,
                    "groups": [],
                },
            },
        )
        self.assertTrue(dagger_collection["filter_base_cache_causally"])
        self.assertTrue(dagger_collection["include_base_cache"])
        self.assertEqual(
            dagger_collection["teacher_behavior_probability"],
            0.0,
        )
        self.assertEqual(dagger_collection["rollouts_per_policy"], 1)

        dagger_training = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_"
                    "bounded_gain01_gcn_dense_pretrainlr1e4_"
                    "onlinelr1e5_groupgate_dagger1_screen.json"
                )
            ).read_text()
        )
        phase_lr_single_factor = json.loads(json.dumps(phase_lr))
        dagger_single_factor = json.loads(json.dumps(dagger_training))
        for payload in (phase_lr_single_factor, dagger_single_factor):
            for key in ("name", "experimental_role", "output_root"):
                payload.pop(key)
        self.assertEqual(
            phase_lr_single_factor.pop("teacher_cache"),
            (
                "results/patient_indexed_specimen_routing_mac_mps_primary/"
                "teachers/bounded_gain01_screen/teacher_train.npz"
            ),
        )
        self.assertEqual(
            dagger_single_factor.pop("teacher_cache"),
            (
                "results/patient_indexed_specimen_routing_mac_mps_primary/"
                "teachers/bounded_gain01_dagger1/teacher_train_dagger.npz"
            ),
        )
        self.assertEqual(dagger_single_factor, phase_lr_single_factor)

        dagger_manifest = (
            "results/patient_indexed_specimen_routing_mac_mps_primary/"
            "sensitivity/bounded_gain01/"
            "gcn_dense_pretrainlr1e4_onlinelr1e5_groupgate_dagger1_"
            "screen_training_mps/"
            "patient_indexed_specimen_routing_mac_mps_bounded_gain01_"
            "gcn_dense_pretrainlr1e4_onlinelr1e5_groupgate_dagger1_"
            "screen/training_manifest.json"
        )
        dagger_evaluation = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_"
                    "bounded_gain01_gcn_dense_pretrainlr1e4_"
                    "onlinelr1e5_groupgate_dagger1_broad_fixed_"
                    "final_eval.json"
                )
            ).read_text()
        )
        self.assertEqual(
            dagger_evaluation["training_manifest"],
            dagger_manifest,
        )
        self.assertEqual(
            dagger_evaluation["fixed_deployment_candidate"],
            broad_fixed["fixed_deployment_candidate"],
        )
        self.assertEqual(dagger_evaluation["validation_seed"], 8760000)
        self.assertEqual(dagger_evaluation["holdout_seed"], 8770000)
        broad_protocol = json.loads(json.dumps(broad_fixed))
        dagger_protocol = json.loads(json.dumps(dagger_evaluation))
        for payload in (broad_protocol, dagger_protocol):
            for key in (
                "name",
                "experimental_role",
                "training_manifest",
                "validation_seed",
                "holdout_seed",
                "output_root",
            ):
                payload.pop(key)
        self.assertEqual(dagger_protocol, broad_protocol)

        capacity_gate = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_"
                    "bounded_gain01_gcn_dense_pretrainlr1e4_"
                    "onlinelr1e5_groupgate_dagger1_capacitygate05_"
                    "fixed_final_eval.json"
                )
            ).read_text()
        )
        self.assertEqual(capacity_gate["training_manifest"], dagger_manifest)
        self.assertEqual(
            capacity_gate["fixed_deployment_candidate"]["group_thresholds"],
            [0.1, 0.1, 0.5],
        )
        self.assertEqual(capacity_gate["validation_seed"], 8780000)
        self.assertEqual(capacity_gate["holdout_seed"], 8790000)
        broad_dagger_protocol = json.loads(json.dumps(dagger_evaluation))
        capacity_protocol = json.loads(json.dumps(capacity_gate))
        for payload in (broad_dagger_protocol, capacity_protocol):
            for key in (
                "name",
                "experimental_role",
                "deployment_candidates",
                "fixed_deployment_candidate",
                "validation_seed",
                "holdout_seed",
                "output_root",
            ):
                payload.pop(key)
        self.assertEqual(capacity_protocol, broad_dagger_protocol)

        reagent_gate = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_"
                    "bounded_gain01_gcn_dense_pretrainlr1e4_"
                    "onlinelr1e5_groupgate_dagger1_reagentgate02_"
                    "capacitygate05_fixed_final_eval.json"
                )
            ).read_text()
        )
        self.assertEqual(reagent_gate["training_manifest"], dagger_manifest)
        self.assertEqual(
            reagent_gate["fixed_deployment_candidate"]["group_thresholds"],
            [0.1, 0.2, 0.5],
        )
        self.assertEqual(reagent_gate["validation_seed"], 8800000)
        self.assertEqual(reagent_gate["holdout_seed"], 8810000)
        capacity_single_factor = json.loads(json.dumps(capacity_gate))
        reagent_single_factor = json.loads(json.dumps(reagent_gate))
        for payload in (capacity_single_factor, reagent_single_factor):
            for key in (
                "name",
                "experimental_role",
                "deployment_candidates",
                "fixed_deployment_candidate",
                "validation_seed",
                "holdout_seed",
                "output_root",
            ):
                payload.pop(key)
        self.assertEqual(reagent_single_factor, capacity_single_factor)

        scale_075 = json.loads(
            (
                config_root
                / (
                    "patient_indexed_specimen_routing_mac_mps_"
                    "bounded_gain01_gcn_dense_pretrainlr1e4_"
                    "onlinelr1e5_groupgate_dagger1_reagentgate02_"
                    "capacitygate05_scale075_fixed_final_eval.json"
                )
            ).read_text()
        )
        self.assertEqual(scale_075["training_manifest"], dagger_manifest)
        self.assertEqual(
            scale_075["fixed_deployment_candidate"],
            {
                "scale": 0.75,
                "group_thresholds": [0.1, 0.2, 0.5],
                "endpoint_projection": {
                    "enabled": False,
                    "groups": [],
                },
            },
        )
        self.assertEqual(scale_075["validation_seed"], 8820000)
        self.assertEqual(scale_075["holdout_seed"], 8830000)
        reagent_scale_contract = json.loads(json.dumps(reagent_gate))
        scale_contract = json.loads(json.dumps(scale_075))
        for payload in (reagent_scale_contract, scale_contract):
            for key in (
                "name",
                "experimental_role",
                "deployment_candidates",
                "fixed_deployment_candidate",
                "validation_seed",
                "holdout_seed",
                "output_root",
            ):
                payload.pop(key)
        self.assertEqual(scale_contract, reagent_scale_contract)

    def test_routing_primary_plan_and_optional_controls_are_explicit(self) -> None:
        plan = load_benchmark_plan(PLAN_PATH)
        self.assertEqual(resolve_budget(plan, "routing_smoke")["num_episodes"], 5)
        pilot = resolve_budget(plan, "routing_pilot")
        self.assertEqual(pilot["seeds"], [0, 1, 2])
        self.assertEqual(pilot["num_episodes"], 100)
        self.assertEqual(pilot["checkpoint_interval"], 5)
        self.assertEqual(plan["parameter_matching"]["gcn_count"], 563397)
        self.assertEqual(plan["parameter_matching"]["flat_count"], 564137)
        self.assertLess(
            plan["parameter_matching"]["relative_gap"],
            plan["parameter_matching"]["maximum_relative_gap"],
        )
        endpoint_projection = plan["algorithm_settings"][GCN][
            "config_overrides"
        ]["residual_action"]["endpoint_projection"]
        self.assertTrue(endpoint_projection["straight_through_gradient"])

        default_names = {
            scenario["name"] for scenario in select_scenarios(plan, None)
        }
        self.assertEqual(
            default_names,
            {
                "routing_nominal_history",
                "routing_abrupt_regime_shift",
                "routing_regional_drift",
                "routing_compound_regional_stress",
            },
        )
        self.assertTrue(all(name.startswith("routing_") for name in default_names))

        routing = _scenario(plan, "routing_nominal_history")["env_overrides"]
        control = _scenario(plan, "no_routing_nominal_history")["env_overrides"]
        self.assertTrue(routing["enable_specimen_routing"])
        self.assertFalse(control["enable_specimen_routing"])
        self.assertTrue(routing["include_specimen_routing_state"])
        self.assertTrue(control["include_specimen_routing_state"])
        self.assertEqual(routing["specimen_edges"], control["specimen_edges"])
        self.assertEqual(routing["specimen_routing_lead_time_epochs"], 1)
        self.assertEqual(routing["finished_product_return_lead_time_epochs"], 0)
        self.assertEqual(
            _scenario(plan, "routing_nominal_lead0_sensitivity")["env_overrides"][
                "specimen_routing_lead_time_epochs"
            ],
            0,
        )
        self.assertEqual(
            _scenario(plan, "routing_nominal_return1_sensitivity")["env_overrides"][
                "finished_product_return_lead_time_epochs"
            ],
            1,
        )
        routing_teacher = json.loads(
            Path(
                "experiments/configs/"
                "patient_indexed_specimen_routing_teacher_routing.json"
            ).read_text(encoding="utf-8")
        )
        control_teacher = json.loads(
            Path(
                "experiments/configs/"
                "patient_indexed_specimen_routing_teacher_no_routing.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            control_teacher["experimental_role"],
            "optional_no_routing_supplement_only",
        )
        self.assertTrue(control_teacher["requires_separate_approval"])
        self.assertEqual(routing_teacher["seed"], control_teacher["seed"])
        self.assertEqual(
            routing_teacher["lookahead_seed"],
            control_teacher["lookahead_seed"],
        )
        self.assertEqual(
            routing_teacher["explicit_options"],
            control_teacher["explicit_options"],
        )
        afd = plan["algorithm_settings"][GCN]["config_overrides"][
            "advantage_distillation_pretrain"
        ]
        for key in ("epsilons", "candidate_groups", "candidate_signs"):
            self.assertIn(key, routing_teacher)
            self.assertEqual(routing_teacher[key], afd[key])
            self.assertEqual(control_teacher[key], afd[key])

    @unittest.skipIf(torch is None, "PyTorch is required for model matching")
    def test_gcn_flat_and_control_contracts_are_shape_and_parameter_matched(self) -> None:
        plan = load_benchmark_plan(PLAN_PATH)
        routing_scenario = _scenario(plan, "routing_nominal_history")
        control_scenario = _scenario(plan, "no_routing_nominal_history")

        agents = {}
        environments = {}
        for arm, scenario in (("routing", routing_scenario), ("control", control_scenario)):
            for algorithm in (GCN, FLAT):
                config = _config(plan, algorithm, scenario)
                env = build_env(config, seed=0)
                agent = get_agent_class(algorithm)(
                    env.observation_size,
                    env.action_size,
                    config,
                )
                environments[(arm, algorithm)] = env
                agents[(arm, algorithm)] = agent

        dimensions = {
            (env.observation_size, env.action_size)
            for env in environments.values()
        }
        self.assertEqual(dimensions, {(561, 80)})
        self.assertTrue(hasattr(agents[("routing", GCN)].actor, "specimen_edge_head"))
        self.assertTrue(hasattr(agents[("control", GCN)].actor, "specimen_edge_head"))

        routing_gcn = agent_parameter_count(agents[("routing", GCN)])
        routing_flat = agent_parameter_count(agents[("routing", FLAT)])
        self.assertEqual(routing_gcn, 563397)
        self.assertEqual(routing_flat, 564137)
        self.assertEqual(
            routing_gcn,
            agent_parameter_count(agents[("control", GCN)]),
        )
        self.assertEqual(
            routing_flat,
            agent_parameter_count(agents[("control", FLAT)]),
        )
        self.assertLess(
            abs(routing_gcn - routing_flat) / max(routing_gcn, routing_flat),
            0.01,
        )

        for key, agent in agents.items():
            env = environments[key]
            state = env.reset(seed=1234)
            action = agent.select_action(state, explore=False, env=env)
            self.assertEqual(action.shape, (80,))
            self.assertTrue(np.isfinite(action).all())

    def test_specimen_residual_options_are_conserved_and_shared(self) -> None:
        specs = make_explicit_residual_option_specs(
            [
                {"group": "specimen_transfer", "epsilon": 0.5, "sign": 1.0},
                {"group": "combined_routing_network", "epsilon": 0.5, "sign": 1.0},
            ]
        )
        anchor = np.zeros(12, dtype=np.float32)
        specimen = np.asarray((-1.0, 0.25, 0.75), dtype=np.float32)
        resource = np.asarray((-0.5, 0.0, 0.5), dtype=np.float32)
        capacity = np.asarray((0.5, -0.5, 0.0), dtype=np.float32)
        actions = residual_option_actions_from_patterns(
            anchor,
            resource,
            capacity,
            specs,
            specimen_pattern=specimen,
        )
        self.assertEqual(len(actions), 3)
        np.testing.assert_allclose(actions[1][:3], 0.5 * specimen)
        self.assertAlmostEqual(float(actions[1][:3].sum()), 0.0)
        np.testing.assert_allclose(actions[2][:3], 0.5 * specimen)
        self.assertGreater(float(np.abs(actions[2][3:9]).sum()), 0.0)

    def test_new_configs_never_reference_legacy_results(self) -> None:
        paths = sorted(
            Path("experiments/configs").glob(
                "patient_indexed_specimen_routing_*.json"
            )
        )
        self.assertGreaterEqual(len(paths), 10)

        def leaves(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from leaves(item)
            elif isinstance(value, list):
                for item in value:
                    yield from leaves(item)

        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for value in leaves(payload):
                if value.startswith("results/"):
                    frozen_teacher_reference = (
                        path.name == FROZEN_TEACHER_CONFIG
                        and value
                        in {
                            FROZEN_TEACHER_ROOT,
                            f"{FROZEN_TEACHER_ROOT}/teacher_cache.npz",
                        }
                    )
                    mac_primary_reference = (
                        path.name.startswith(
                            "patient_indexed_specimen_routing_mac_mps_"
                        )
                        and value.startswith(MAC_RESULT_ROOT)
                    )
                    stage_c_reference = (
                        path.name.startswith(
                            "patient_indexed_specimen_routing_stage_c_td3_"
                        )
                        and (
                            value == STAGE_C_RESULT_ROOT
                            or value.startswith(f"{STAGE_C_RESULT_ROOT}/")
                        )
                    )
                    ddpg_support_reference = (
                        path.name.startswith(
                            "patient_indexed_specimen_routing_ddpg_"
                            "support_alignment_"
                        )
                        and (
                            value == DDPG_SUPPORT_RESULT_ROOT
                            or value.startswith(
                                f"{DDPG_SUPPORT_RESULT_ROOT}/"
                            )
                        )
                    )
                    ddpg_persistent_shift_reference = (
                        path.name.startswith(
                            "patient_indexed_specimen_routing_ddpg_"
                            "persistent_shift_"
                        )
                        and (
                            value == DDPG_PERSISTENT_SHIFT_RESULT_ROOT
                            or value.startswith(
                                f"{DDPG_PERSISTENT_SHIFT_RESULT_ROOT}/"
                            )
                        )
                    )
                    ddpg_structured_reference = (
                        path.name.startswith(
                            "patient_indexed_specimen_routing_ddpg_"
                            "structured_exploration_"
                        )
                        and (
                            value == DDPG_STRUCTURED_EXPLORATION_RESULT_ROOT
                            or value.startswith(
                                f"{DDPG_STRUCTURED_EXPLORATION_RESULT_ROOT}/"
                            )
                        )
                    )
                    frozen_route_headroom_reference = (
                        path.name
                        == (
                            "patient_indexed_specimen_routing_ddpg_frozen_"
                            "route_headroom_audit.json"
                        )
                        and (
                            value.startswith(
                                f"{DDPG_STRUCTURED_EXPLORATION_RESULT_ROOT}/"
                            )
                            or value.startswith(
                                f"{DDPG_PERSISTENT_SHIFT_RESULT_ROOT}/"
                            )
                        )
                    )
                    ddpg_identifiability_reference = (
                        path.name
                        == (
                            "patient_indexed_specimen_routing_ddpg_online_"
                            "identifiability_f0.json"
                        )
                        and (
                            value
                            == DDPG_ONLINE_IDENTIFIABILITY_RESULT_ROOT
                            or value.startswith(
                                f"{DDPG_ONLINE_IDENTIFIABILITY_RESULT_ROOT}/"
                            )
                            or value.startswith(
                                f"{DDPG_STRUCTURED_EXPLORATION_RESULT_ROOT}/"
                            )
                        )
                    )
                    ddpg_online_paired_reference = (
                        path.name.startswith(
                            "patient_indexed_specimen_routing_ddpg_online_"
                            "paired_advantage_"
                        )
                        and (
                            value == DDPG_ONLINE_PAIRED_ADVANTAGE_RESULT_ROOT
                            or value.startswith(
                                f"{DDPG_ONLINE_PAIRED_ADVANTAGE_RESULT_ROOT}/"
                            )
                            or value.startswith(
                                f"{DDPG_ONLINE_IDENTIFIABILITY_RESULT_ROOT}/"
                            )
                        )
                    )
                    ddpg_posthoc_reference = any(
                        value == root or value.startswith(f"{root}/")
                        for root in DDPG_POSTHOC_RESULT_ROOTS.get(path.name, ())
                    )
                    self.assertTrue(
                        value.startswith(RESULT_ROOT)
                        or frozen_teacher_reference
                        or mac_primary_reference
                        or stage_c_reference
                        or ddpg_support_reference
                        or ddpg_persistent_shift_reference
                        or ddpg_structured_reference
                        or frozen_route_headroom_reference
                        or ddpg_identifiability_reference
                        or ddpg_online_paired_reference
                        or ddpg_posthoc_reference,
                        f"{path}: {value}",
                    )

    @unittest.skipIf(torch is None, "PyTorch is required for actor drift")
    def test_actor_drift_is_measured_against_frozen_pretrain(self) -> None:
        actor = torch.nn.Linear(2, 1, bias=False)
        agent = SimpleNamespace(actor=actor)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pretrain.pt"
            torch.save({"actor": actor.state_dict()}, checkpoint)
            with torch.no_grad():
                actor.weight.add_(0.25)
            drift = actor_checkpoint_drift(agent, checkpoint)
        self.assertGreater(drift["rms"], 0.0)
        self.assertAlmostEqual(drift["max_abs"], 0.25)
        self.assertEqual(drift["parameter_count"], 2.0)

    def test_attribution_integrates_routing_primary_cells(self) -> None:
        def write_root(
            root: Path,
            *,
            scenario: str,
            gcn_cost: float,
            flat_cost: float,
            anchor_cost: float,
        ) -> None:
            root.mkdir(parents=True)
            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            for algorithm, cost in ((GCN, gcn_cost), (FLAT, flat_cost)):
                run = root / algorithm / "seed0"
                run.mkdir(parents=True)
                row = {
                    "algorithm": algorithm,
                    "training_seed": 0,
                    "evaluation_seed": 8400000,
                    "replication": 0,
                    "scenario": scenario,
                    "total_cost": cost,
                    "specimen_route_events_json": "x" * 200_000,
                }
                with (run / "holdout_rows.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                if algorithm == GCN:
                    anchor = dict(row)
                    anchor["algorithm"] = "mdl2"
                    anchor["total_cost"] = anchor_cost
                    with (run / "holdout_anchor_rows.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(anchor))
                        writer.writeheader()
                        writer.writerow(anchor)

        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            routing_final = temporary / "routing_final"
            routing_pretrain = temporary / "routing_pretrain"
            write_root(
                routing_final,
                scenario="routing_nominal_history",
                gcn_cost=80.0,
                flat_cost=90.0,
                anchor_cost=95.0,
            )
            write_root(
                routing_pretrain,
                scenario="routing_nominal_history",
                gcn_cost=85.0,
                flat_cost=95.0,
                anchor_cost=95.0,
            )
            output = temporary / "attribution.json"
            config = {
                "routing_final_root": str(routing_final),
                "routing_pretrain_root": str(routing_pretrain),
                "algorithms": [GCN, FLAT],
                "metrics": {"total_cost": "lower"},
                "bootstrap_resamples": 20,
                "bootstrap_seed": 1,
                "output_path": str(output),
            }
            result = compare_attribution(config)
            self.assertTrue(result["decision"]["gcn_vs_flat_total_cost_supported"])
            self.assertTrue(result["decision"]["gcn_vs_mdl2_total_cost_supported"])
            self.assertTrue(
                result["decision"]["gcn_final_vs_pretrain_total_cost_supported"]
            )
            self.assertEqual(
                result["decision"]["interpretation"],
                "Under patient-indexed specimen routing, the preregistered "
                "total-cost comparisons support GCN residual control over both "
                "matched flat residual control and MDL-2.",
            )
            self.assertTrue(output.is_file())
            with self.assertRaises(FileExistsError):
                compare_attribution(config)

    def test_locked_runner_is_staged_and_non_destructive(self) -> None:
        runner = Path(
            "scripts/run_patient_indexed_specimen_routing.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("codex/patient-indexed-specimen-routing", runner)
        self.assertIn("ce9b6274419c8e0e7adf800f434e47d96c18c1dc", runner)
        self.assertIn(
            r'$ResultRoot = "results\patient_indexed_specimen_routing_recovery7"',
            runner,
        )
        self.assertIn(
            r'$SupersededResultRoot = "results\patient_indexed_specimen_routing_recovery6"',
            runner,
        )
        self.assertIn("d647e491f4d55c3d1e19e2c27889b11f63acd10a", runner)
        self.assertIn(
            "optional empty-string Base64 parameter binding",
            runner,
        )
        self.assertIn(
            "5264bbabb6517a515aaf1cb75c662b04f00bdabbc4611f9e2cb84da1c6e0fa03",
            runner,
        )
        self.assertIn(
            "aaa429d5684f62752250d04e0173de41f8cb8e8d713c530a434ba9ef9d1da34c",
            runner,
        )
        self.assertIn("ControlProcessIds", runner)
        self.assertIn("$ExplicitControlProcessIds", runner)
        self.assertIn("$IsPythonWorkload", runner)
        self.assertIn("$IsDetachedRoutingLauncher", runner)
        self.assertNotIn(
            '$_.CommandLine -match "patient_indexed_specimen_routing"',
            runner,
        )
        self.assertIn("superseded_outputs_reused = $false", runner)
        for phase in ("Preflight", "ImportTeacher", "Smoke", "Pilot", "Evaluate"):
            self.assertIn(f'"{phase}"', runner)
        self.assertNotIn('[ValidateSet("Validate"', runner)
        self.assertIn(
            "evaluation.verify_patient_indexed_specimen_routing_validation",
            runner,
        )
        self.assertIn("verify frozen Mac validation evidence", runner)
        self.assertIn("Recovery5ValidateEvidenceSha256", runner)
        self.assertIn("7f8b1f3774ab0e9e2ae5453615052e273a027c5c4f34a90a240159c8eb925e27", runner)
        self.assertNotIn('"-m", "unittest"', runner)
        self.assertNotIn('"-m", "compileall"', runner)
        self.assertNotIn(
            "evaluation.validate_patient_indexed_specimen_routing",
            runner,
        )
        self.assertNotIn('"Teachers"', runner)
        self.assertIn("evaluation.headroom_teacher_bundle", runner)
        self.assertIn('"extract"', runner)
        self.assertNotIn('"evaluation.network_residual_headroom"', runner)
        self.assertIn("-ApprovePilot", runner)
        self.assertNotIn("teacher_no_routing.json", runner)
        self.assertNotIn("smoke_no_routing.json", runner)
        self.assertNotIn("pilot_no_routing.json", runner)
        self.assertNotIn("pilot_no_routing_eval.json", runner)
        self.assertIn("learned_runs = 2", runner)
        self.assertIn("learned_runs = 6", runner)
        for forbidden in (
            "--force",
            "Remove-Item",
            "Stop-Process",
            "Compress-Archive",
            "Copy-Item",
            "resume-training-state",
        ):
            self.assertNotIn(forbidden, runner)

    def test_recovery7_launcher_detaches_and_preserves_control_chain(self) -> None:
        launcher = Path(
            "scripts/start_patient_indexed_specimen_routing_phase.ps1"
        ).read_text(encoding="utf-8")
        wrapper = Path(
            "scripts/invoke_patient_indexed_specimen_routing_phase_detached.ps1"
        ).read_text(encoding="utf-8")
        runner = Path(
            "scripts/run_patient_indexed_specimen_routing.ps1"
        ).read_text(encoding="utf-8")

        self.assertIn("Start-Process", launcher)
        self.assertIn("-RedirectStandardOutput", launcher)
        self.assertIn("-RedirectStandardError", launcher)
        self.assertIn("-PassThru", launcher)
        self.assertNotIn("-Wait", launcher)
        self.assertIn("launcher-logs", launcher)
        self.assertIn("patient_indexed_specimen_routing_recovery7", launcher)
        self.assertIn("TeacherBundle", launcher)
        self.assertIn("TeacherBundle", wrapper)
        self.assertIn("TeacherBundleBase64", launcher)
        self.assertIn("TeacherBundleBase64", wrapper)
        self.assertIn("ExpectedTeacherBundleSha256", launcher)
        self.assertIn("ExpectedTeacherBundleSha256", wrapper)
        self.assertIn("ExpectedTeacherBundleSha256", runner)
        self.assertIn("ConvertTo-Utf8Base64", launcher)
        self.assertIn("ConvertFrom-Utf8Base64", wrapper)
        self.assertIn("[Convert]::FromBase64String", wrapper)
        self.assertIn("if ($TeacherBundleBase64)", wrapper)
        self.assertIn("TransportProbe", wrapper)
        self.assertNotIn("TransportProbe", launcher)
        self.assertIn("if (-not $TransportProbe)", wrapper)
        self.assertIn("transport_probe = [bool]$TransportProbe", wrapper)
        self.assertIn("& $Runner @RunnerParameters", wrapper)
        self.assertNotIn("& powershell.exe @RunnerArguments", wrapper)
        self.assertNotIn('@("-TeacherBundle", $TeacherBundle)', launcher)
        self.assertIn("Get-ControlProcessIds", launcher)
        self.assertIn("SerializedControlProcessIds", launcher)
        self.assertIn("ControlProcessIds", launcher)
        self.assertIn("ControlProcessIds", wrapper)
        self.assertIn("control_process_ids", launcher)
        self.assertIn("control_process_ids", wrapper)
        self.assertIn("PID=$($Process.Id)", launcher)
        self.assertIn("run_patient_indexed_specimen_routing.ps1", wrapper)
        self.assertIn("exit $ExitCode", wrapper)
        self.assertIn("status.json", launcher)
        self.assertIn("Resolve-RoutingPython", launcher)
        self.assertIn("[Convert]::ToBase64String", launcher)
        self.assertIn("base64.b64decode", launcher)
        self.assertNotIn("-c $ProbeScript", launcher)
        self.assertLess(
            launcher.index("$PythonProbe = Resolve-RoutingPython"),
            launcher.index("$ResultRoot = Join-Path"),
        )
        self.assertIn("Python 3.11.9", launcher)
        self.assertIn('python_version -ne "3.11.9"', launcher)
        self.assertIn('numpy_version -ne "2.0.2"', launcher)
        self.assertIn('cuda_device_name -notmatch "(?i)RTX\\s*4090"', launcher)
        self.assertIn(r'C:\gcnrl\.venv\Scripts\python.exe', launcher)
        self.assertIn("requested_python", launcher)
        self.assertIn("python_sha256", launcher)
        self.assertIn('argument_transport = "utf8_base64"', launcher)
        self.assertIn('argument_transport = "utf8_base64"', wrapper)
        self.assertIn('"Preflight"', launcher)
        self.assertIn('"Preflight"', wrapper)
        self.assertNotIn('"Validate"', launcher)
        self.assertNotIn('"Validate"', wrapper)
        self.assertNotIn("Missing requested Python executable", launcher)
        for forbidden in (
            "--force",
            "Remove-Item",
            "Stop-Process",
            "Copy-Item",
            "git reset",
            "git clean",
        ):
            self.assertNotIn(forbidden, launcher)
            self.assertNotIn(forbidden, wrapper)

    def test_launcher_python_probe_survives_native_argument_parsing(self) -> None:
        launcher = Path(
            "scripts/start_patient_indexed_specimen_routing_phase.ps1"
        ).read_text(encoding="utf-8")
        probe = launcher.split("$ProbeScript = @'\n", 1)[1].split("\n'@", 1)[0]
        encoded = base64.b64encode(probe.encode("utf-8")).decode("ascii")
        prelude = (
            "import sys,types;"
            "numpy=types.ModuleType('numpy');numpy.__version__='2.0.2';"
            "torch=types.ModuleType('torch');torch.__version__='test';"
            "torch.cuda=types.SimpleNamespace(is_available=lambda:False);"
            "evaluation=types.ModuleType('evaluation');evaluation.__path__=[];"
            "training=types.ModuleType("
            "'evaluation.train_multiscenario_network_residual');"
            "sys.modules.update({'numpy':numpy,'torch':torch,"
            "'evaluation':evaluation,"
            "'evaluation.train_multiscenario_network_residual':training});"
        )
        command = (
            prelude
            + f"import base64;exec(base64.b64decode('{encoded}'))"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path.cwd())

        completed = subprocess.run(
            (sys.executable, "-c", command),
            cwd=Path.cwd(),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        line = next(
            item
            for item in completed.stdout.splitlines()
            if item.startswith("ROUTING_PYTHON_PROBE=")
        )
        payload = json.loads(line.removeprefix("ROUTING_PYTHON_PROBE="))
        self.assertIn("python_version", payload)
        self.assertIn("numpy_version", payload)
        self.assertIn("cuda_available", payload)


if __name__ == "__main__":
    unittest.main()
