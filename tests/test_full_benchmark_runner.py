"""Tests for the full benchmark manifest and runner helpers."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluation.check_training_stability import summarize_training_stability
from evaluation.run_gcn_residual_sweep import (
    best_variant_summary,
    collect_local_search_demonstrations,
    elite_sample_weights,
    local_search_candidate_actions,
    local_search_metric_score,
    make_residual_sweep_config,
    residual_variant_name,
    summary_metadata,
)
from evaluation.run_full_benchmark import (
    advantage_distillation_settings,
    algorithm_config_overrides,
    anchor_fallback_candidate_diagnostics,
    anchor_fallback_settings,
    config_snapshot_path,
    evaluation_csv_path,
    evaluation_outputs_complete,
    evaluation_summary_path,
    final_checkpoint_path,
    checkpoint_label,
    learned_checkpoint_candidates,
    local_search_checkpoint_path,
    local_search_candidate_groups,
    local_search_candidate_signs,
    load_benchmark_plan,
    make_evaluation_config,
    make_scenario_env_config,
    make_training_config,
    residual_deployment_scale_candidates,
    resolve_budget,
    select_algorithms,
    select_anchor_fallback_policy,
    select_residual_deployment_candidate,
    select_scenarios,
    training_csv_path,
    training_outputs_complete,
)
from src.baselines.heuristics import MyopicPolicy, get_heuristic_class
from src.env.capacity_planning import CapacityPlanningEnv, make_legacy_two_facility_config
from src.rl.config import load_config
from src.rl.experiment import build_env


class FullBenchmarkRunnerTests(unittest.TestCase):
    def test_smoke_budget_and_selection(self) -> None:
        plan = load_benchmark_plan()
        budget = resolve_budget(plan, "smoke")

        self.assertEqual(budget["seeds"], [0])
        self.assertEqual(budget["evaluation_replications"], 1)
        self.assertIn("sac", select_algorithms(plan, None))
        self.assertNotIn("sac", select_algorithms(plan, None, primary_only=True))
        self.assertEqual(select_scenarios(plan, ["disruption_0_3"])[0]["name"], "disruption_0_3")

    def test_graph_stress_plan_loads(self) -> None:
        plan = load_benchmark_plan("experiments/configs/graph_stress_benchmark.json")
        budget = resolve_budget(plan, "pilot")

        self.assertEqual(plan["name"], "20_clinic_graph_stress_benchmark")
        self.assertEqual(budget["evaluation_replications"], 100)
        self.assertEqual(
            select_scenarios(plan, ["graph_stress_supply_cluster"])[0]["name"],
            "graph_stress_supply_cluster",
        )

    def test_residual_policy_plan_loads(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "pilot")
        mini_pilot = resolve_budget(plan, "mini_pilot")
        targeted_100 = resolve_budget(plan, "targeted_100")
        targeted_300 = resolve_budget(plan, "targeted_300")

        self.assertEqual(plan["name"], "20_clinic_residual_policy_benchmark")
        self.assertEqual(budget["num_episodes"], 300)
        self.assertEqual(mini_pilot["num_episodes"], 10)
        self.assertEqual(targeted_100["num_episodes"], 100)
        self.assertEqual(targeted_100["anchor_fallback"]["validation_replications"], 20)
        self.assertEqual(targeted_100["anchor_fallback"]["min_improvement"], 0.0)
        self.assertEqual(targeted_100["anchor_fallback"]["min_service_level_delta"], 0.0)
        self.assertEqual(
            targeted_100["anchor_fallback"]["deployment_scale_candidates"],
            [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        )
        self.assertTrue(targeted_100["checkpoint_selection"]["enabled"])
        self.assertEqual(targeted_100["checkpoint_selection"]["service_level_weight"], 100000000.0)
        self.assertEqual(targeted_100["checkpoint_selection"]["eligibility_rate_weight"], 100000000.0)
        self.assertEqual(targeted_100["checkpoint_selection"]["at_risk_unserved_weight"], 50000.0)
        self.assertEqual(targeted_100["checkpoint_selection"]["patients_lost_weight"], 500000.0)
        self.assertEqual(targeted_300["seeds"], [0, 1, 2])
        self.assertEqual(targeted_300["num_episodes"], 300)
        self.assertEqual(targeted_300["checkpoint_interval"], 50)
        self.assertEqual(targeted_300["evaluation_replications"], 100)
        self.assertEqual(targeted_300["anchor_fallback"]["validation_replications"], 50)
        self.assertEqual(
            targeted_300["anchor_fallback"]["deployment_scale_candidates"][-1],
            3.0,
        )
        self.assertTrue(targeted_300["checkpoint_selection"]["enabled"])
        self.assertEqual(targeted_300["checkpoint_selection"]["validation_replications"], 20)
        self.assertEqual(targeted_100["local_search"]["gcn_residual_mdl2"]["min_improvement"], 0.0)
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo"]["service_level_weight"],
            100000000.0,
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo"]["eligibility_rate_weight"],
            100000000.0,
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo"]["at_risk_unserved_weight"],
            50000.0,
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo"]["patients_lost_weight"],
            500000.0,
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo_td3"]["service_level_weight"],
            100000000.0,
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo_td3"]["patients_lost_weight"],
            500000.0,
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo_risk_pressure_td3"]["candidate_groups"],
            ["replenishment_patient_risk_pressure", "replenishment_positive_pressure"],
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo_risk_pressure_td3"]["min_service_level_delta"],
            0.0,
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo_risk_replenish_td3"]["candidate_groups"],
            ["replenishment_patient_risk"],
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo_risk_replenish_td3"]["min_service_level_delta"],
            0.002,
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo_rebalance_td3"]["patients_lost_weight"],
            500000.0,
        )
        self.assertFalse(
            plan["algorithm_settings"]["gcn_residual_pmyo_shield_td3"]["local_search"]["enabled"]
        )
        self.assertFalse(plan["algorithm_settings"]["gcn_mdl2_shield_selector"]["local_search"]["enabled"])
        self.assertFalse(plan["algorithm_settings"]["gcn_pmyo_shield_selector"]["local_search"]["enabled"])
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo_transfer_td3_bc"]["epochs"],
            40,
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo_transfer_td3_bc"]["lookahead"],
            6,
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo_transfer_td3_bc"]["min_service_level_delta"],
            0.002,
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo_transfer_td3_bc"]["anchor_keep_weight"],
            500000.0,
        )
        self.assertEqual(
            targeted_100["local_search"]["gcn_residual_pmyo_transfer_td3"]["patients_lost_weight"],
            500000.0,
        )
        self.assertIn("flat_residual_mdl2", select_algorithms(plan, None, primary_only=True))
        self.assertIn("flat_residual_pmyo", select_algorithms(plan, None, primary_only=True))
        self.assertIn("gcn_residual_mdl2", select_algorithms(plan, None, primary_only=True))
        self.assertIn(
            "gcn_residual_mdl2_replenish_ddpg",
            select_algorithms(plan, None, primary_only=True),
        )
        self.assertIn(
            "gcn_residual_mdl2_replenish_ddpg_afd",
            select_algorithms(plan, None, primary_only=True),
        )
        self.assertIn("gcn_residual_mdl2_replenish_td3", select_algorithms(plan, None, primary_only=True))
        self.assertIn("gcn_residual_mdl2_td3", select_algorithms(plan, None, primary_only=True))
        self.assertIn("gcn_residual_mdl2_shield_td3", select_algorithms(plan, None, primary_only=True))
        self.assertIn("gcn_residual_pmyo_td3", select_algorithms(plan, None, primary_only=True))
        self.assertIn("gcn_residual_pmyo_risk_pressure_td3", select_algorithms(plan, None, primary_only=True))
        self.assertIn("gcn_residual_pmyo_risk_replenish_td3", select_algorithms(plan, None, primary_only=True))
        self.assertIn("gcn_residual_pmyo_rebalance_td3", select_algorithms(plan, None, primary_only=True))
        self.assertIn("gcn_residual_pmyo_shield_td3", select_algorithms(plan, None, primary_only=True))
        self.assertIn("gcn_mdl2_shield_selector", select_algorithms(plan, None, primary_only=True))
        self.assertIn("gcn_pmyo_shield_selector", select_algorithms(plan, None, primary_only=True))
        self.assertIn("gcn_residual_pmyo_transfer_td3_bc", select_algorithms(plan, None, primary_only=True))
        self.assertIn("gcn_residual_pmyo_transfer_td3", select_algorithms(plan, None, primary_only=True))
        self.assertIn("pmyo", select_algorithms(plan, None, primary_only=True))
        self.assertIn("pmyo_shield", select_algorithms(plan, None, primary_only=True))
        self.assertEqual(
            select_scenarios(plan, ["graph_dynamic_transfer_delay"])[0]["name"],
            "graph_dynamic_transfer_delay",
        )
        self.assertEqual(
            select_scenarios(plan, ["patient_condition_stress"])[0]["name"],
            "patient_condition_stress",
        )
        self.assertEqual(
            select_scenarios(plan, ["patient_condition_geo"])[0]["name"],
            "patient_condition_geo",
        )
        self.assertEqual(
            select_scenarios(plan, ["patient_condition_geo_demand_drift"])[0]["name"],
            "patient_condition_geo_demand_drift",
        )

    def test_heuristic_evaluation_config_applies_algorithm_overrides(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["patient_condition_geo"])[0]

        config = make_evaluation_config(plan, "smoke", budget, "pmyo_shield", scenario, seed=0)

        self.assertEqual(config["algorithm"], "pmyo_shield")
        self.assertEqual(config["shield_lookahead"], 3)
        self.assertEqual(config["shield_epsilons"], [0.005, 0.01])
        self.assertIn("replenishment_patient_risk_pressure", config["candidate_groups"])
        self.assertEqual(config["env"]["scenario_name"], "patient_condition_geo")

    def test_training_config_preserves_algorithm_graph_ablation(self) -> None:
        plan = load_benchmark_plan()
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["disruption_0_05"])[0]

        flat_config = make_training_config(plan, "smoke", budget, "flat_ddpg", scenario, seed=7)
        gcn_config = make_training_config(plan, "smoke", budget, "gcn_ddpg", scenario, seed=7)

        self.assertEqual(flat_config["env"]["supplier_disruption_rate"], 0.05)
        self.assertEqual(flat_config["env"]["graph_ablation"], "flat_state_no_graph")
        self.assertEqual(gcn_config["env"]["graph_ablation"], "full_graph")
        self.assertEqual(flat_config["num_episodes"], 1)
        self.assertEqual(flat_config["max_steps_per_episode"], 3)
        self.assertEqual(flat_config["checkpoint_interval"], 1)
        self.assertEqual(flat_config["progress_interval"], 1)

    def test_final_checkpoint_path_matches_training_config(self) -> None:
        plan = load_benchmark_plan()
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["disruption_0_6"])[0]
        config = make_training_config(plan, "smoke", budget, "td3", scenario, seed=2)

        self.assertEqual(
            final_checkpoint_path(plan, "smoke", budget, "td3", scenario, seed=2),
            Path(config["checkpoint_dir"]) / "td3_seed2_episode1.pt",
        )

    def test_learned_checkpoint_candidates_include_episode_and_local_search(self) -> None:
        with TemporaryDirectory() as tmpdir:
            plan = {"output_root": tmpdir}
            scenario = {"name": "s"}
            checkpoint_dir = Path(tmpdir) / "b" / "checkpoints" / "s" / "algo_seed0"
            checkpoint_dir.mkdir(parents=True)
            episode100 = checkpoint_dir / "algo_seed0_episode100.pt"
            episode50 = checkpoint_dir / "algo_seed0_episode50.pt"
            local_search = checkpoint_dir / "algo_seed0_local_search.pt"
            for path in (episode100, episode50, local_search):
                path.touch()

            candidates = learned_checkpoint_candidates(
                plan,
                "b",
                "algo",
                scenario,
                0,
                local_search,
            )

            self.assertEqual(candidates, (episode50, episode100, local_search))
            self.assertEqual(checkpoint_label(episode50), "episode50")
            self.assertEqual(checkpoint_label(local_search), "local_search")

    def test_algorithm_overrides_and_local_search_checkpoint(self) -> None:
        plan = load_benchmark_plan("experiments/configs/patient_forecast_benchmark.json")
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["graph_dynamic_patient_forecast"])[0]
        config = make_training_config(plan, "smoke", budget, "gcn_ddpg", scenario, seed=0)

        overrides = algorithm_config_overrides(plan, "gcn_ddpg")

        self.assertEqual(overrides["residual_action"]["base_policy"], "myo")
        self.assertEqual(config["env"]["scenario_name"], "graph_dynamic_patient_forecast")
        self.assertEqual(config["residual_action"]["scale"], 0.05)
        self.assertEqual(config["residual_action"]["group_scales"]["replenishment"], 0.05)
        self.assertEqual(config["residual_action"]["group_scales"]["specimen_transfer"], 0.0)
        self.assertEqual(
            final_checkpoint_path(plan, "smoke", budget, "gcn_ddpg", scenario, seed=0),
            local_search_checkpoint_path(plan, "smoke", "gcn_ddpg", scenario, seed=0),
        )

    def test_residual_policy_plan_overrides_anchor_and_checkpoint(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["graph_dynamic_transfer_delay"])[0]
        config = make_training_config(plan, "smoke", budget, "gcn_residual_mdl2", scenario, seed=0)

        self.assertEqual(config["algorithm"], "gcn_residual_mdl2")
        self.assertEqual(config["residual_action"]["base_policy"], "mdl2")
        self.assertEqual(config["imitation_pretrain"]["policy"], "mdl2")
        self.assertEqual(config["residual_action"]["group_scales"]["reagent_transfer"], 0.05)
        self.assertEqual(config["env"]["scenario_name"], "graph_dynamic_transfer_delay")
        self.assertEqual(
            final_checkpoint_path(plan, "smoke", budget, "gcn_residual_mdl2", scenario, seed=0),
            local_search_checkpoint_path(
                plan,
                "smoke",
                "gcn_residual_mdl2",
                scenario,
                seed=0,
            ),
        )

    def test_heuristic_evaluation_uses_reference_env_defaults(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "targeted_100")
        scenario = select_scenarios(plan, ["patient_condition_stress"])[0]

        gcn_config = make_evaluation_config(
            plan,
            "targeted_100",
            budget,
            "gcn_residual_mdl2",
            scenario,
            seed=0,
        )
        mdl2_config = make_evaluation_config(plan, "targeted_100", budget, "mdl2", scenario, seed=0)
        env_config = make_scenario_env_config(plan, "mdl2", scenario)

        self.assertEqual(plan["heuristic_env_reference_algorithm"], "gcn_residual_mdl2")
        self.assertEqual(mdl2_config["env"], env_config)
        self.assertEqual(mdl2_config["env"], gcn_config["env"])
        self.assertEqual(mdl2_config["env"]["demand_shock_probability"], 0.12)
        self.assertEqual(mdl2_config["env"]["demand_shock_multiplier"], 2.4)

    def test_flat_residual_policy_plan_uses_flat_graph_ablation(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["patient_condition_stress"])[0]
        config = make_training_config(plan, "smoke", budget, "flat_residual_mdl2", scenario, seed=0)

        self.assertEqual(config["algorithm"], "flat_residual_mdl2")
        self.assertEqual(config["env"]["env_type"], "patient_condition")
        self.assertEqual(config["env"]["graph_ablation"], "flat_state_no_graph")
        self.assertEqual(config["env"]["transfer_lead_time"], 0)
        self.assertFalse(config["env"]["include_transfer_pipeline_state"])
        self.assertEqual(config["residual_action"]["base_policy"], "mdl2")
        self.assertEqual(
            final_checkpoint_path(plan, "smoke", budget, "flat_residual_mdl2", scenario, seed=0),
            local_search_checkpoint_path(
                plan,
                "smoke",
                "flat_residual_mdl2",
                scenario,
                seed=0,
            ),
        )

    def test_flat_afd_ablation_matches_gcn_training_recipe(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["patient_condition_geo_demand_drift"])[0]
        flat_algorithm = "flat_residual_mdl2_replenish_ddpg_afd"
        gcn_algorithm = "gcn_residual_mdl2_replenish_ddpg_afd"
        flat_config = make_training_config(
            plan,
            "smoke",
            budget,
            flat_algorithm,
            scenario,
            seed=0,
        )
        gcn_config = make_training_config(
            plan,
            "smoke",
            budget,
            gcn_algorithm,
            scenario,
            seed=0,
        )

        for key in (
            "residual_action",
            "imitation_pretrain",
            "advantage_distillation_pretrain",
            "anchor_advantage_actor_loss",
            "patient_service_proxy_actor_loss",
            "train_randomization",
            "exploration_noise",
            "update_frequency",
            "updates_per_update",
        ):
            self.assertEqual(flat_config[key], gcn_config[key])
        self.assertEqual(flat_config["env"]["graph_ablation"], "flat_state_no_graph")
        self.assertEqual(gcn_config["env"]["graph_ablation"], "full_graph")
        self.assertEqual(
            advantage_distillation_settings(flat_config, budget, flat_algorithm),
            advantage_distillation_settings(gcn_config, budget, gcn_algorithm),
        )

    def test_patient_priority_residual_plan_sets_patient_aware_anchor(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["patient_condition_stress"])[0]
        flat_config = make_training_config(plan, "smoke", budget, "flat_residual_pmyo", scenario, seed=0)
        gcn_config = make_training_config(plan, "smoke", budget, "gcn_residual_pmyo", scenario, seed=0)
        mdl2_replenish_ddpg_config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_residual_mdl2_replenish_ddpg",
            scenario,
            seed=0,
        )
        mdl2_replenish_ddpg_afd_config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_residual_mdl2_replenish_ddpg_afd",
            scenario,
            seed=0,
        )
        mdl2_replenish_td3_config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_residual_mdl2_replenish_td3",
            scenario,
            seed=0,
        )
        mdl2_td3_config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_residual_mdl2_td3",
            scenario,
            seed=0,
        )
        mdl2_shield_td3_config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_residual_mdl2_shield_td3",
            scenario,
            seed=0,
        )
        td3_config = make_training_config(plan, "smoke", budget, "gcn_residual_pmyo_td3", scenario, seed=0)
        risk_replenish_config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_residual_pmyo_risk_replenish_td3",
            scenario,
            seed=0,
        )
        risk_pressure_config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_residual_pmyo_risk_pressure_td3",
            scenario,
            seed=0,
        )
        rebalance_config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_residual_pmyo_rebalance_td3",
            scenario,
            seed=0,
        )
        shield_teacher_config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_residual_pmyo_shield_td3",
            scenario,
            seed=0,
        )
        shield_selector_config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_pmyo_shield_selector",
            scenario,
            seed=0,
        )
        mdl2_shield_selector_config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_mdl2_shield_selector",
            scenario,
            seed=0,
        )
        transfer_config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_residual_pmyo_transfer_td3",
            scenario,
            seed=0,
        )
        transfer_bc_config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_residual_pmyo_transfer_td3_bc",
            scenario,
            seed=0,
        )

        self.assertEqual(flat_config["residual_action"]["base_policy"], "pmyo")
        self.assertEqual(flat_config["imitation_pretrain"]["policy"], "pmyo")
        self.assertEqual(gcn_config["residual_action"]["base_policy"], "pmyo")
        self.assertEqual(gcn_config["imitation_pretrain"]["policy"], "pmyo")
        self.assertEqual(
            mdl2_replenish_ddpg_config["algorithm"],
            "gcn_residual_mdl2_replenish_ddpg",
        )
        self.assertEqual(
            mdl2_replenish_ddpg_config["residual_action"]["base_policy"],
            "mdl2",
        )
        self.assertEqual(
            mdl2_replenish_ddpg_config["imitation_pretrain"]["policy"],
            "mdl2_shield",
        )
        self.assertEqual(
            mdl2_replenish_ddpg_config["residual_action"]["group_scales"][
                "reagent_transfer"
            ],
            0.0,
        )
        self.assertIn(
            "replenishment",
            mdl2_replenish_ddpg_config["residual_action"]["positive_only_groups"],
        )
        self.assertTrue(mdl2_replenish_ddpg_config["anchor_fallback"]["enabled"])
        self.assertTrue(
            mdl2_replenish_ddpg_config["anchor_advantage_actor_loss"]["enabled"]
        )
        self.assertTrue(
            mdl2_replenish_ddpg_config["patient_service_proxy_actor_loss"]["enabled"]
        )
        self.assertTrue(mdl2_replenish_ddpg_config["train_randomization"]["enabled"])
        self.assertEqual(mdl2_replenish_ddpg_config["exploration_noise"]["sigma"], 0.005)
        self.assertFalse(
            plan["algorithm_settings"]["gcn_residual_mdl2_replenish_ddpg"][
                "local_search"
            ]["enabled"]
        )
        self.assertEqual(
            mdl2_replenish_ddpg_afd_config["residual_action"],
            mdl2_replenish_ddpg_config["residual_action"],
        )
        self.assertEqual(
            mdl2_replenish_ddpg_afd_config["algorithm"],
            "gcn_residual_mdl2_replenish_ddpg_afd",
        )
        afd_settings = advantage_distillation_settings(
            mdl2_replenish_ddpg_afd_config,
            budget,
            "gcn_residual_mdl2_replenish_ddpg_afd",
        )
        self.assertTrue(afd_settings["enabled"])
        self.assertEqual(afd_settings["rollouts"], 1)
        self.assertEqual(afd_settings["lookahead"], 1)
        self.assertEqual(afd_settings["max_steps"], 2)
        self.assertEqual(afd_settings["anchor_keep_probability"], 1.0)
        self.assertFalse(afd_settings["anchor_keep_on_improved"])
        self.assertTrue(afd_settings["balance_label_weights"])
        self.assertTrue(afd_settings["retain_for_regularization"])
        self.assertEqual(afd_settings["epsilons"], [0.005, 0.01, 0.02, 0.05])
        self.assertEqual(
            afd_settings["candidate_groups"],
            ["replenishment_patient_risk_pressure", "replenishment_positive_pressure"],
        )
        self.assertFalse(mdl2_replenish_ddpg_afd_config["elite_imitation"]["enabled"])
        self.assertEqual(
            mdl2_replenish_ddpg_afd_config["imitation_pretrain"]["regularization_weight"],
            1.0,
        )
        self.assertEqual(mdl2_replenish_td3_config["algorithm"], "gcn_residual_mdl2_replenish_td3")
        self.assertEqual(mdl2_replenish_td3_config["residual_action"]["base_policy"], "mdl2")
        self.assertEqual(mdl2_replenish_td3_config["imitation_pretrain"]["policy"], "mdl2_shield")
        self.assertEqual(
            mdl2_replenish_td3_config["residual_action"]["group_scales"]["reagent_transfer"],
            0.0,
        )
        self.assertEqual(
            mdl2_replenish_td3_config["residual_action"]["group_scales"]["capacity_transfer"],
            0.0,
        )
        self.assertIn(
            "replenishment",
            mdl2_replenish_td3_config["residual_action"]["positive_only_groups"],
        )
        self.assertTrue(mdl2_replenish_td3_config["train_randomization"]["enabled"])
        self.assertTrue(mdl2_replenish_td3_config["patient_service_proxy_actor_loss"]["enabled"])
        self.assertEqual(
            mdl2_replenish_td3_config["patient_service_proxy_actor_loss"]["group"],
            "replenishment",
        )
        self.assertGreater(
            mdl2_replenish_td3_config["anchor_advantage_actor_loss"]["negative_penalty_weight"],
            0.0,
        )
        self.assertGreater(
            mdl2_replenish_td3_config["patient_service_proxy_actor_loss"]["cost_weight"],
            0.0,
        )
        self.assertFalse(
            plan["algorithm_settings"]["gcn_residual_mdl2_replenish_td3"]["local_search"]["enabled"]
        )
        self.assertEqual(mdl2_td3_config["algorithm"], "gcn_residual_mdl2_td3")
        self.assertEqual(mdl2_td3_config["residual_action"]["base_policy"], "mdl2")
        self.assertEqual(mdl2_td3_config["imitation_pretrain"]["policy"], "mdl2")
        self.assertTrue(mdl2_td3_config["residual_action"]["pressure_projection"]["enabled"])
        self.assertEqual(
            mdl2_td3_config["residual_action"]["pressure_projection"]["groups"],
            ["reagent_transfer", "capacity_transfer", "replenishment"],
        )
        self.assertTrue(mdl2_td3_config["anchor_fallback"]["enabled"])
        self.assertTrue(mdl2_td3_config["anchor_advantage_actor_loss"]["enabled"])
        self.assertTrue(mdl2_td3_config["anchor_advantage_actor_loss"]["use_twin_min"])
        self.assertEqual(mdl2_td3_config["update_frequency"], 4)
        self.assertEqual(mdl2_td3_config["updates_per_update"], 1)
        self.assertFalse(plan["algorithm_settings"]["gcn_residual_mdl2_td3"]["local_search"]["enabled"])
        self.assertEqual(mdl2_shield_td3_config["algorithm"], "gcn_residual_mdl2_shield_td3")
        self.assertEqual(mdl2_shield_td3_config["residual_action"]["base_policy"], "mdl2")
        self.assertEqual(mdl2_shield_td3_config["imitation_pretrain"]["policy"], "mdl2_shield")
        self.assertEqual(
            mdl2_shield_td3_config["imitation_pretrain"]["policy_config"]["anchor_policy"],
            "mdl2",
        )
        self.assertTrue(mdl2_shield_td3_config["anchor_advantage_actor_loss"]["enabled"])
        self.assertNotIn("freeze_actor_updates", mdl2_shield_td3_config)
        self.assertFalse(plan["algorithm_settings"]["gcn_residual_mdl2_shield_td3"]["local_search"]["enabled"])
        self.assertEqual(td3_config["residual_action"]["base_policy"], "pmyo")
        self.assertEqual(td3_config["imitation_pretrain"]["policy"], "pmyo")
        self.assertEqual(risk_replenish_config["residual_action"]["base_policy"], "pmyo")
        self.assertEqual(risk_replenish_config["imitation_pretrain"]["policy"], "pmyo")
        self.assertEqual(risk_pressure_config["residual_action"]["base_policy"], "pmyo")
        self.assertEqual(risk_pressure_config["imitation_pretrain"]["policy"], "pmyo")
        self.assertEqual(rebalance_config["residual_action"]["base_policy"], "pmyo")
        self.assertEqual(rebalance_config["imitation_pretrain"]["policy"], "pmyo")
        self.assertEqual(shield_teacher_config["residual_action"]["base_policy"], "pmyo")
        self.assertEqual(shield_teacher_config["imitation_pretrain"]["policy"], "pmyo_shield")
        self.assertEqual(
            shield_teacher_config["imitation_pretrain"]["policy_config"]["shield_lookahead"],
            2,
        )
        self.assertTrue(shield_teacher_config["freeze_actor_updates"])
        self.assertEqual(shield_teacher_config["exploration_noise_std"], 0.0)
        self.assertFalse(plan["algorithm_settings"]["gcn_residual_pmyo_shield_td3"]["local_search"]["enabled"])
        self.assertEqual(shield_selector_config["shield_selector"]["anchor_policy"], "pmyo")
        self.assertEqual(
            shield_selector_config["shield_selector"]["teacher_policy_config"]["shield_lookahead"],
            2,
        )
        self.assertEqual(shield_selector_config["shield_selector"]["candidate_epsilons"], [0.005])
        self.assertTrue(shield_selector_config["residual_action"]["include_base_action_features"])
        self.assertFalse(plan["algorithm_settings"]["gcn_pmyo_shield_selector"]["local_search"]["enabled"])
        self.assertEqual(mdl2_shield_selector_config["residual_action"]["base_policy"], "mdl2")
        self.assertEqual(mdl2_shield_selector_config["shield_selector"]["anchor_policy"], "mdl2")
        self.assertEqual(mdl2_shield_selector_config["shield_selector"]["confidence_threshold"], 0.6)
        self.assertEqual(mdl2_shield_selector_config["imitation_pretrain"]["class_weight_power"], 1.0)
        self.assertEqual(
            mdl2_shield_selector_config["shield_selector"]["teacher_policy_config"]["anchor_policy"],
            "mdl2",
        )
        self.assertFalse(plan["algorithm_settings"]["gcn_mdl2_shield_selector"]["local_search"]["enabled"])
        self.assertEqual(transfer_config["residual_action"]["base_policy"], "pmyo")
        self.assertEqual(transfer_config["imitation_pretrain"]["policy"], "pmyo")
        self.assertEqual(transfer_bc_config["residual_action"]["base_policy"], "pmyo")
        self.assertEqual(transfer_bc_config["imitation_pretrain"]["policy"], "pmyo")
        self.assertEqual(flat_config["residual_action"]["group_scales"]["replenishment"], 0.02)
        self.assertEqual(flat_config["residual_action"]["group_scales"]["reagent_transfer"], 0.0)
        self.assertEqual(flat_config["residual_action"]["group_scales"]["capacity_transfer"], 0.0)
        self.assertEqual(flat_config["residual_action"]["center_groups"], [])
        self.assertEqual(flat_config["residual_action"]["positive_only_groups"], ["replenishment"])
        self.assertEqual(flat_config["residual_action"]["state_gate"]["groups"], ["replenishment"])
        self.assertEqual(flat_config["residual_action"]["state_gate"]["threshold"], 0.0)
        self.assertEqual(gcn_config["residual_action"]["group_scales"]["reagent_transfer"], 0.0)
        self.assertEqual(gcn_config["residual_action"]["group_scales"]["capacity_transfer"], 0.0)
        self.assertEqual(gcn_config["residual_action"]["group_scales"]["replenishment"], 0.02)
        self.assertEqual(gcn_config["residual_action"]["center_groups"], [])
        self.assertEqual(gcn_config["residual_action"]["positive_only_groups"], ["replenishment"])
        self.assertEqual(gcn_config["residual_action"]["state_gate"]["groups"], ["replenishment"])
        self.assertEqual(gcn_config["residual_action"]["state_gate"]["threshold"], 0.0)
        self.assertEqual(gcn_config["residual_action"]["l2_weight"], 0.05)
        flat_pmyo_search = plan["algorithm_settings"]["flat_residual_pmyo"]["local_search"]
        gcn_pmyo_search = plan["algorithm_settings"]["gcn_residual_pmyo"]["local_search"]
        td3_pmyo_search = plan["algorithm_settings"]["gcn_residual_pmyo_td3"]["local_search"]
        risk_replenish_search = plan["algorithm_settings"]["gcn_residual_pmyo_risk_replenish_td3"]["local_search"]
        risk_pressure_search = plan["algorithm_settings"]["gcn_residual_pmyo_risk_pressure_td3"]["local_search"]
        rebalance_search = plan["algorithm_settings"]["gcn_residual_pmyo_rebalance_td3"]["local_search"]
        transfer_search = plan["algorithm_settings"]["gcn_residual_pmyo_transfer_td3"]["local_search"]
        transfer_bc_search = plan["algorithm_settings"]["gcn_residual_pmyo_transfer_td3_bc"]["local_search"]
        self.assertEqual(
            local_search_candidate_groups(flat_pmyo_search),
            ("replenishment_uniform", "replenishment_positive_pressure"),
        )
        self.assertEqual(
            local_search_candidate_groups(gcn_pmyo_search),
            ("replenishment_uniform", "replenishment_positive_pressure"),
        )
        self.assertEqual(
            local_search_candidate_signs(gcn_pmyo_search),
            (1.0,),
        )
        self.assertEqual(td3_config["residual_action"]["group_scales"]["reagent_transfer"], 0.0)
        self.assertEqual(td3_config["residual_action"]["group_scales"]["capacity_transfer"], 0.0)
        self.assertEqual(td3_config["residual_action"]["group_scales"]["replenishment"], 0.02)
        self.assertEqual(td3_config["residual_action"]["positive_only_groups"], ["replenishment"])
        self.assertEqual(td3_config["residual_action"]["state_gate"]["groups"], ["replenishment"])
        self.assertEqual(td3_config["residual_action"]["state_gate"]["threshold"], 0.0)
        self.assertTrue(td3_config["anchor_fallback"]["enabled"])
        self.assertEqual(td3_config["exploration_noise_std"], 0.03)
        self.assertEqual(
            local_search_candidate_groups(td3_pmyo_search),
            ("replenishment_uniform", "replenishment_positive_pressure"),
        )
        self.assertEqual(
            local_search_candidate_signs(td3_pmyo_search),
            (1.0,),
        )
        self.assertEqual(risk_replenish_config["residual_action"]["group_scales"]["reagent_transfer"], 0.0)
        self.assertEqual(risk_replenish_config["residual_action"]["group_scales"]["capacity_transfer"], 0.0)
        self.assertEqual(risk_replenish_config["residual_action"]["group_scales"]["replenishment"], 0.015)
        self.assertEqual(risk_replenish_config["residual_action"]["positive_only_groups"], ["replenishment"])
        self.assertEqual(risk_replenish_config["exploration_noise_std"], 0.01)
        self.assertEqual(risk_pressure_config["residual_action"]["group_scales"]["reagent_transfer"], 0.0)
        self.assertEqual(risk_pressure_config["residual_action"]["group_scales"]["capacity_transfer"], 0.0)
        self.assertEqual(risk_pressure_config["residual_action"]["group_scales"]["replenishment"], 0.015)
        self.assertEqual(risk_pressure_config["residual_action"]["positive_only_groups"], ["replenishment"])
        self.assertEqual(risk_pressure_config["exploration_noise_std"], 0.01)
        self.assertEqual(
            local_search_candidate_groups(risk_replenish_search),
            ("replenishment_patient_risk",),
        )
        self.assertEqual(local_search_candidate_signs(risk_replenish_search), (1.0,))
        self.assertEqual(risk_replenish_search["min_service_level_delta"], 0.002)
        self.assertEqual(
            local_search_candidate_groups(risk_pressure_search),
            ("replenishment_patient_risk_pressure", "replenishment_positive_pressure"),
        )
        self.assertEqual(local_search_candidate_signs(risk_pressure_search), (1.0,))
        self.assertEqual(risk_pressure_search["min_service_level_delta"], 0.0)
        self.assertEqual(rebalance_config["residual_action"]["group_scales"]["reagent_transfer"], 0.0)
        self.assertEqual(rebalance_config["residual_action"]["group_scales"]["capacity_transfer"], 0.0)
        self.assertEqual(rebalance_config["residual_action"]["group_scales"]["replenishment"], 0.02)
        self.assertEqual(rebalance_config["residual_action"]["center_groups"], ["replenishment"])
        self.assertNotIn("positive_only_groups", rebalance_config["residual_action"])
        self.assertEqual(
            local_search_candidate_groups(rebalance_search),
            ("replenishment_pressure",),
        )
        self.assertEqual(
            local_search_candidate_signs(rebalance_search),
            (1.0,),
        )
        self.assertEqual(transfer_config["residual_action"]["group_scales"]["reagent_transfer"], 0.01)
        self.assertEqual(transfer_config["residual_action"]["group_scales"]["capacity_transfer"], 0.01)
        self.assertEqual(transfer_config["residual_action"]["group_scales"]["replenishment"], 0.02)
        self.assertEqual(transfer_bc_config["residual_action"]["group_scales"]["reagent_transfer"], 0.005)
        self.assertEqual(transfer_bc_config["residual_action"]["group_scales"]["capacity_transfer"], 0.005)
        self.assertEqual(transfer_bc_config["residual_action"]["group_scales"]["replenishment"], 0.015)
        self.assertEqual(transfer_bc_config["residual_action"]["l2_weight"], 0.12)
        self.assertEqual(transfer_bc_config["exploration_noise_std"], 0.01)
        self.assertEqual(transfer_bc_config["policy_noise"], 0.015)
        self.assertEqual(transfer_bc_config["noise_clip"], 0.03)
        self.assertEqual(
            transfer_config["residual_action"]["center_groups"],
            ["reagent_transfer", "capacity_transfer"],
        )
        self.assertEqual(
            transfer_bc_config["residual_action"]["center_groups"],
            ["reagent_transfer", "capacity_transfer"],
        )
        self.assertEqual(transfer_config["residual_action"]["positive_only_groups"], ["replenishment"])
        self.assertEqual(transfer_bc_config["residual_action"]["positive_only_groups"], ["replenishment"])
        self.assertEqual(
            local_search_candidate_groups(transfer_search),
            (
                "reagent_transfer",
                "capacity_transfer",
                "combined_transfer",
                "replenishment_positive_pressure",
            ),
        )
        self.assertEqual(local_search_candidate_signs(transfer_search), (-1.0, 1.0))
        self.assertEqual(
            local_search_candidate_groups(transfer_bc_search),
            (
                "reagent_transfer",
                "capacity_transfer",
                "combined_transfer",
                "replenishment_positive_pressure",
            ),
        )
        self.assertEqual(local_search_candidate_signs(transfer_bc_search), (-1.0, 1.0))
        self.assertEqual(transfer_bc_search["lookahead"], 8)
        self.assertEqual(transfer_bc_search["epochs"], 32)
        self.assertEqual(transfer_bc_search["min_service_level_delta"], 0.002)
        self.assertEqual(transfer_bc_search["anchor_keep_weight"], 500000.0)
        self.assertEqual(flat_config["env"]["env_type"], "patient_condition")

    def test_local_search_candidate_actions_can_match_replenishment_only_residual(self) -> None:
        env_config = load_config("experiments/configs/2_clinic_patient_condition.json")
        env = build_env({"env": env_config}, seed=0)
        state = env.reset(seed=0)
        baseline = get_heuristic_class("pmyo")(
            state_dim=env.observation_size,
            action_dim=env.action_size,
            config={},
        )
        baseline_action = baseline.select_action(state, explore=False, env=env)

        actions = local_search_candidate_actions(
            state,
            env,
            baseline,
            epsilons=(0.02,),
            candidate_groups=("replenishment_uniform", "replenishment_positive_pressure"),
            candidate_signs=(1.0,),
        )

        self.assertEqual(len(actions), 3)
        n = env.config.num_facilities
        for action in actions[1:]:
            np.testing.assert_allclose(action[: 3 * n], baseline_action[: 3 * n])
            self.assertTrue(np.all(action[3 * n : 4 * n] >= baseline_action[3 * n : 4 * n]))

    def test_local_search_candidate_actions_can_target_patient_risk_replenishment(self) -> None:
        n = 2

        class ZeroBaseline:
            def select_action(self, state, explore=False, env=None):
                return np.zeros(4 * n, dtype=np.float32)

        env = SimpleNamespace(
            config=SimpleNamespace(num_facilities=n),
            demand=np.zeros(n, dtype=float),
            demand_forecast=np.zeros(n, dtype=float),
            specimens=np.zeros(n, dtype=float),
            reagents=np.zeros(n, dtype=float),
            bioreactors=np.zeros((n, 1), dtype=float),
            at_risk_counts=lambda: np.array([0.0, 4.0]),
            near_expiry_counts=lambda: np.array([0.0, 0.0]),
        )

        actions = local_search_candidate_actions(
            np.zeros(1, dtype=np.float32),
            env,
            ZeroBaseline(),
            epsilons=(0.1,),
            candidate_groups=("replenishment_patient_risk",),
            candidate_signs=(1.0,),
        )

        self.assertEqual(len(actions), 2)
        np.testing.assert_allclose(actions[1][3 * n : 4 * n], np.array([0.0, 0.1], dtype=np.float32))

    def test_local_search_candidate_actions_can_target_patient_risk_pressure_replenishment(self) -> None:
        n = 2

        class ZeroBaseline:
            def select_action(self, state, explore=False, env=None):
                return np.zeros(4 * n, dtype=np.float32)

        env = SimpleNamespace(
            config=SimpleNamespace(num_facilities=n),
            demand=np.array([0.0, 10.0]),
            demand_forecast=np.array([0.0, 10.0]),
            specimens=np.array([0.0, 10.0]),
            reagents=np.array([0.0, 0.0]),
            bioreactors=np.zeros((n, 1), dtype=float),
            at_risk_counts=lambda: np.array([4.0, 4.0]),
            near_expiry_counts=lambda: np.array([0.0, 0.0]),
        )

        actions = local_search_candidate_actions(
            np.zeros(1, dtype=np.float32),
            env,
            ZeroBaseline(),
            epsilons=(0.1,),
            candidate_groups=("replenishment_patient_risk_pressure",),
            candidate_signs=(1.0,),
        )

        self.assertEqual(len(actions), 2)
        self.assertGreater(actions[1][3 * n + 1], actions[1][3 * n])

    def test_local_search_candidate_actions_account_for_pending_transfer_arrivals(self) -> None:
        config = replace(
            make_legacy_two_facility_config(episode_horizon=2),
            action_mode="facility_net",
            transfer_lead_time=2,
            include_transfer_pipeline_state=True,
        )
        baseline_env = CapacityPlanningEnv(config, seed=0)
        pipeline_env = CapacityPlanningEnv(config, seed=0)
        for env in (baseline_env, pipeline_env):
            env.reset(seed=0)
            env.demand = np.array([5.0, 20.0])
            env.demand_forecast = np.array([5.0, 20.0])
            env.specimens = np.array([5.0, 20.0])
            env.reagents = np.array([25.0, 0.0])
        pipeline_env.reagent_transfer_pipeline[:, 1] = 60.0

        baseline_actions = local_search_candidate_actions(
            baseline_env.observation(),
            baseline_env,
            MyopicPolicy(),
            epsilons=(0.05,),
            candidate_groups=("reagent_transfer",),
            candidate_signs=(1.0,),
        )
        pipeline_actions = local_search_candidate_actions(
            pipeline_env.observation(),
            pipeline_env,
            MyopicPolicy(),
            epsilons=(0.05,),
            candidate_groups=("reagent_transfer",),
            candidate_signs=(1.0,),
        )

        n = baseline_env.config.num_facilities
        self.assertGreater(baseline_actions[1][n + 1], baseline_actions[1][n])
        self.assertLess(pipeline_actions[1][n + 1], pipeline_actions[1][n])

    def test_patient_condition_geo_plan_combines_patient_and_geography(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["patient_condition_geo"])[0]
        config = make_training_config(plan, "smoke", budget, "gcn_residual_mdl2", scenario, seed=0)

        self.assertEqual(config["env"]["env_type"], "patient_condition")
        self.assertEqual(config["env"]["scenario_name"], "patient_condition_geo")
        self.assertEqual(config["env"]["graph_ablation"], "full_graph")
        self.assertEqual(len(config["env"]["clinic_coordinates"]), 20)
        self.assertGreater(config["env"]["geographic_transfer_cost_scale"], 0.0)
        self.assertGreater(config["env"]["geographic_transfer_time_cost_scale"], 0.0)
        self.assertGreater(config["env"]["regional_supplier_disruption_probability"], 0.0)
        self.assertEqual(config["env"]["transfer_lead_time"], 3)
        self.assertTrue(config["env"]["include_transfer_pipeline_state"])
        self.assertEqual(config["env"]["transfer_lead_time_distance_thresholds"], [500.0, 1500.0])

    def test_patient_condition_geo_demand_drift_plan_splits_truth_from_prior(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["patient_condition_geo_demand_drift"])[0]
        config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_residual_mdl2_shield_td3",
            scenario,
            seed=0,
        )

        self.assertEqual(config["env"]["scenario_name"], "patient_condition_geo_demand_drift")
        self.assertEqual(config["env"]["env_type"], "patient_condition")
        self.assertNotEqual(config["env"]["demand_rates"], config["env"]["demand_rate_estimates"])
        self.assertEqual(len(config["env"]["demand_rate_estimates"]), 20)
        self.assertEqual(config["train_randomization"]["demand_rate_multiplier_range"], [0.8, 1.5])
        self.assertEqual(config["train_randomization"]["forecast_error_range"], [0.0, 0.35])
        self.assertEqual(config["env"]["transfer_lead_time"], 3)
        self.assertEqual(len(config["env"]["clinic_coordinates"]), 20)

    def test_patient_condition_geo_severe_demand_drift_raises_true_rates(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["patient_condition_geo_demand_drift_severe"])[0]
        config = make_training_config(
            plan,
            "smoke",
            budget,
            "gcn_residual_mdl2_replenish_td3",
            scenario,
            seed=0,
        )

        estimates = config["env"]["demand_rate_estimates"]
        truth = config["env"]["demand_rates"]
        self.assertEqual(config["env"]["scenario_name"], "patient_condition_geo_demand_drift_severe")
        self.assertEqual(len(truth), 20)
        self.assertEqual(len(estimates), 20)
        self.assertTrue(all(true > prior for true, prior in zip(truth, estimates)))
        self.assertEqual(config["env"]["transfer_lead_time"], 3)
        self.assertTrue(config["env"]["include_transfer_pipeline_state"])

    def test_anchor_fallback_settings_and_decision_rule(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["patient_condition_stress"])[0]
        config = make_training_config(plan, "smoke", budget, "gcn_residual_pmyo", scenario, seed=0)

        settings = anchor_fallback_settings(config, budget)

        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["validation_replications"], 1)
        self.assertEqual(
            residual_deployment_scale_candidates(settings, has_residual_scale=True),
            (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
        )
        self.assertEqual(
            select_anchor_fallback_policy(95.0, 100.0, min_improvement=0.0),
            "learned",
        )
        self.assertEqual(
            select_anchor_fallback_policy(
                95.0,
                100.0,
                min_improvement=0.0,
                learned_service_level=0.89,
                anchor_service_level=0.90,
                min_service_level_delta=0.0,
            ),
            "anchor",
        )
        self.assertEqual(
            select_anchor_fallback_policy(
                95.0,
                100.0,
                min_improvement=0.0,
                learned_service_level=0.90,
                anchor_service_level=0.90,
                min_service_level_delta=0.0,
            ),
            "learned",
        )
        self.assertEqual(
            select_anchor_fallback_policy(100.0, 100.0, min_improvement=0.05),
            "anchor",
        )
        self.assertEqual(
            select_anchor_fallback_policy(float("nan"), 100.0, min_improvement=0.0),
            "anchor",
        )

    def test_residual_deployment_candidate_selection_uses_partial_trust_region(self) -> None:
        anchor = {"total_cost_mean": 100.0, "service_level_mean": 0.90}
        candidates = [
            {"residual_scale": 0.0, "summary": {"total_cost_mean": 100.0, "service_level_mean": 0.90}},
            {"residual_scale": 0.5, "summary": {"total_cost_mean": 98.0, "service_level_mean": 0.90}},
            {"residual_scale": 1.0, "summary": {"total_cost_mean": 96.0, "service_level_mean": 0.85}},
        ]

        selected, decision = select_residual_deployment_candidate(
            candidates,
            anchor,
            {"min_improvement": 0.0, "min_service_level_delta": 0.0},
        )

        self.assertEqual(decision, "learned")
        self.assertEqual(selected["residual_scale"], 0.5)

    def test_residual_deployment_candidate_selection_falls_back_when_only_zero_scale_passes(self) -> None:
        anchor = {"total_cost_mean": 100.0, "service_level_mean": 0.90}
        candidates = [
            {"residual_scale": 0.0, "summary": {"total_cost_mean": 100.0, "service_level_mean": 0.90}},
            {"residual_scale": 1.0, "summary": {"total_cost_mean": 98.0, "service_level_mean": 0.88}},
        ]

        selected, decision = select_residual_deployment_candidate(
            candidates,
            anchor,
            {"min_improvement": 0.0, "min_service_level_delta": 0.0},
        )

        self.assertEqual(decision, "anchor")
        self.assertEqual(selected["residual_scale"], 1.0)

    def test_anchor_fallback_candidate_diagnostics_records_scale_gates(self) -> None:
        anchor = {
            "total_cost_mean": 100.0,
            "service_level_mean": 0.90,
        }
        candidates = [
            {
                "residual_scale": 0.0,
                "summary": {
                    "total_cost_mean": 100.0,
                    "service_level_mean": 0.90,
                    "eligibility_rate_mean_mean": 0.95,
                    "patients_lost_mean": 2.0,
                    "at_risk_unserved_mean": 3.0,
                },
            },
            {
                "residual_scale": 0.25,
                "summary": {
                    "total_cost_mean": 99.0,
                    "service_level_mean": 0.91,
                    "eligibility_rate_mean_mean": 0.96,
                    "patients_lost_mean": 1.0,
                    "at_risk_unserved_mean": 2.0,
                },
            },
            {
                "residual_scale": 1.0,
                "summary": {
                    "total_cost_mean": 98.0,
                    "service_level_mean": 0.88,
                    "eligibility_rate_mean_mean": 0.94,
                    "patients_lost_mean": 5.0,
                    "at_risk_unserved_mean": 8.0,
                },
            },
        ]

        metadata = anchor_fallback_candidate_diagnostics(
            candidates,
            anchor,
            {"min_improvement": 0.0, "min_service_level_delta": 0.0},
        )

        self.assertEqual(metadata["anchor_fallback_validation_candidate_scales"], "0|0.25|1")
        self.assertEqual(
            metadata["anchor_fallback_validation_candidate_decisions"],
            "anchor_equivalent|learned|anchor",
        )
        self.assertEqual(metadata["anchor_fallback_validation_best_nonzero_scale"], 1.0)
        self.assertEqual(metadata["anchor_fallback_validation_best_nonzero_decision"], "anchor")
        self.assertAlmostEqual(
            float(metadata["anchor_fallback_validation_best_nonzero_service_gap"]),
            -0.02,
        )

    def test_pure_gcn_ddpg_plan_disables_residual_and_imitation(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["graph_dynamic_transfer_delay"])[0]
        config = make_training_config(plan, "smoke", budget, "gcn_pure_ddpg", scenario, seed=0)

        self.assertFalse(config["residual_action"]["enabled"])
        self.assertFalse(config["imitation_pretrain"]["enabled"])
        self.assertFalse(config["elite_imitation"]["enabled"])

    def test_output_completion_helpers_require_all_expected_files(self) -> None:
        plan = load_benchmark_plan()
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["disruption_0_3"])[0]
        with TemporaryDirectory() as tmpdir:
            plan = dict(plan)
            plan["output_root"] = tmpdir

            self.assertFalse(training_outputs_complete(plan, "smoke", budget, "td3", scenario, 0))
            for path in (
                training_csv_path(plan, "smoke", "td3", scenario, 0),
                config_snapshot_path(plan, "smoke", "td3", scenario, 0),
                final_checkpoint_path(plan, "smoke", budget, "td3", scenario, 0),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ok")
            self.assertTrue(training_outputs_complete(plan, "smoke", budget, "td3", scenario, 0))

            self.assertFalse(evaluation_outputs_complete(plan, "smoke", "td3", scenario, 0))
            for path in (
                evaluation_csv_path(plan, "smoke", "td3", scenario, 0),
                evaluation_summary_path(plan, "smoke", "td3", scenario, 0),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ok")
            self.assertTrue(evaluation_outputs_complete(plan, "smoke", "td3", scenario, 0))

    def test_residual_sweep_config_sets_anchor_and_paths(self) -> None:
        base_config = {
            "algorithm": "gcn_ddpg",
            "env": {"num_facilities": 20, "graph_ablation": "full_graph"},
            "imitation_pretrain": {"enabled": True, "policy": "mdl2"},
        }
        env_config = {"scenario_name": "dynamic", "supplier_disruption_rate": 0.3}
        variant = residual_variant_name(
            "myo",
            0.2,
            0.02,
            center_residual_groups=("replenishment",),
            transfer_scale=0.04,
            replenishment_scale=0.2,
        )

        config = make_residual_sweep_config(
            base_config,
            env_config,
            base_policy="myo",
            scale=0.2,
            transfer_scale=0.04,
            replenishment_scale=0.2,
            l2_weight=0.02,
            seed=3,
            episodes=5,
            steps=7,
            batch_size=11,
            checkpoint_interval=2,
            elite_epochs=4,
            output_root=Path("results/sweep"),
            scenario_name="dynamic",
            variant=variant,
            progress_interval=2,
            center_residual_groups=("replenishment",),
        )

        self.assertEqual(config["residual_action"]["base_policy"], "myo")
        self.assertEqual(config["residual_action"]["group_scales"]["specimen_transfer"], 0.04)
        self.assertEqual(config["residual_action"]["group_scales"]["replenishment"], 0.2)
        self.assertEqual(config["residual_action"]["center_groups"], ["replenishment"])
        self.assertEqual(config["imitation_pretrain"]["policy"], "myo")
        self.assertEqual(config["max_steps_per_episode"], 7)
        self.assertEqual(config["batch_size"], 11)
        self.assertEqual(config["checkpoint_interval"], 2)
        self.assertEqual(config["elite_imitation"]["epochs"], 4)
        self.assertIn(variant, config["result_csv_path"])
        self.assertIn("centerreplenishment", variant)

    def test_residual_sweep_metadata_and_best_selector(self) -> None:
        metadata = summary_metadata(
            variant="v",
            base_policy="myo",
            scale=0.2,
            transfer_scale=0.0,
            replenishment_scale=0.2,
            l2_weight=0.02,
            elite_epochs=6,
            seed=0,
            eval_seed=50000,
            checkpoint_path=Path("ckpt.pt"),
            checkpoint_episode="offline_elite",
            selection_stage="offline_elite",
        )
        rows = [
            {"variant": "v", "training_seed": 0, "total_cost_mean": "4.0"},
            {"variant": "v", "training_seed": 0, "total_cost_mean": "3.5", **metadata},
        ]

        best = best_variant_summary(rows, "v", 0)

        self.assertEqual(metadata["selection_stage"], "offline_elite")
        self.assertEqual(metadata["checkpoint_episode"], "offline_elite")
        self.assertEqual(best["total_cost_mean"], "3.5")

    def test_offline_elite_sample_weights_follow_rollout_advantage(self) -> None:
        elites = [
            (0.0, 9.0, 12.0, 3.0, np.zeros((2, 3), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)),
            (1.0, 11.0, 12.0, 1.0, np.zeros((3, 3), dtype=np.float32), np.zeros((3, 1), dtype=np.float32)),
        ]

        weights = elite_sample_weights(elites, weighting="improvement", power=1.0, floor=0.0)

        self.assertEqual(tuple(weights.shape), (5,))
        self.assertGreater(float(weights[0]), float(weights[-1]))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        rank_weights = elite_sample_weights(elites, weighting="rank", power=1.0, floor=0.0)
        self.assertEqual(tuple(rank_weights.shape), (5,))
        self.assertGreater(float(rank_weights[0]), float(rank_weights[-1]))
        self.assertIsNone(elite_sample_weights(elites, weighting="none", power=1.0, floor=0.0))

    def test_local_search_metric_score_values_patient_outcomes(self) -> None:
        weights = {
            "service_level": 100.0,
            "eligibility_rate": 50.0,
            "at_risk_unserved": 10.0,
            "patients_lost": 1000.0,
        }
        safe = {
            "total_cost": 1000.0,
            "service_level": 0.9,
            "eligibility_rate": 0.95,
            "at_risk_unserved": 1.0,
            "patients_lost": 0.0,
        }
        risky = {
            "total_cost": 900.0,
            "service_level": 0.8,
            "eligibility_rate": 0.9,
            "at_risk_unserved": 5.0,
            "patients_lost": 1.0,
        }

        self.assertLess(local_search_metric_score(safe, weights), local_search_metric_score(risky, weights))

    def test_local_search_candidate_actions_are_valid(self) -> None:
        config = replace(make_legacy_two_facility_config(episode_horizon=2), action_mode="facility_net")
        env = CapacityPlanningEnv(config, seed=5)
        state = env.reset(seed=5)

        actions = local_search_candidate_actions(state, env, MyopicPolicy(), epsilons=(0.05,))

        self.assertGreaterEqual(len(actions), 11)
        self.assertTrue(any(np.any(action[env.config.num_facilities : 2 * env.config.num_facilities]) for action in actions))
        self.assertTrue(any(np.any(action[2 * env.config.num_facilities : 3 * env.config.num_facilities]) for action in actions))
        for action in actions:
            self.assertEqual(action.shape, (env.action_size,))
            self.assertTrue(np.all(action >= -1.0))
            self.assertTrue(np.all(action <= 1.0))

    def test_local_search_demo_collection_shapes(self) -> None:
        config = replace(make_legacy_two_facility_config(episode_horizon=2), action_mode="facility_net")
        env = CapacityPlanningEnv(config, seed=7)

        demos = collect_local_search_demonstrations(
            env,
            seed=7,
            rollouts=1,
            lookahead=1,
            epsilons=(0.05,),
            max_steps=2,
            baseline_policy="myo",
            min_improvement=-1.0,
        )

        self.assertEqual(demos["states"].shape[1], env.observation_size)
        self.assertEqual(demos["actions"].shape[1], env.action_size)
        self.assertEqual(demos["weights"].shape[0], demos["states"].shape[0])

    def test_local_search_demo_collection_keeps_anchor_references(self) -> None:
        config = replace(make_legacy_two_facility_config(episode_horizon=2), action_mode="facility_net")
        env = CapacityPlanningEnv(config, seed=8)

        demos = collect_local_search_demonstrations(
            env,
            seed=8,
            rollouts=1,
            lookahead=1,
            epsilons=(),
            max_steps=2,
            baseline_policy="myo",
            min_improvement=0.0,
            anchor_keep_probability=1.0,
            anchor_keep_weight=7.0,
        )

        self.assertEqual(demos["improved_steps"], 0)
        self.assertEqual(demos["anchor_keep_steps"], 2)
        self.assertEqual(demos["states"].shape[0], 2)
        np.testing.assert_allclose(demos["weights"], np.full(2, 7.0, dtype=np.float32))

    def test_advantage_filtered_demos_do_not_add_conflicting_anchor_labels(self) -> None:
        config = replace(
            make_legacy_two_facility_config(episode_horizon=2),
            action_mode="facility_net",
        )
        env = CapacityPlanningEnv(config, seed=9)
        anchor_action = np.zeros(env.action_size, dtype=np.float32)
        improved_action = anchor_action.copy()
        improved_action[3 * env.config.num_facilities] = 0.01
        baseline_metrics = {
            "total_cost": 100.0,
            "service_level": 0.8,
            "eligibility_rate": 0.9,
            "at_risk_unserved": 2.0,
            "patients_lost": 1.0,
        }
        improved_metrics = dict(baseline_metrics, total_cost=90.0)

        with patch(
            "evaluation.run_gcn_residual_sweep.local_search_candidate_actions",
            return_value=[anchor_action, improved_action],
        ), patch(
            "evaluation.run_gcn_residual_sweep.rollout_metrics_after_action",
            side_effect=[
                baseline_metrics,
                improved_metrics,
                baseline_metrics,
                improved_metrics,
            ],
        ):
            demos = collect_local_search_demonstrations(
                env,
                seed=9,
                rollouts=1,
                lookahead=1,
                epsilons=(0.01,),
                max_steps=2,
                baseline_policy="myo",
                min_improvement=0.0,
                anchor_keep_probability=1.0,
                anchor_keep_weight=7.0,
                anchor_keep_on_improved=False,
                balance_label_weights=True,
            )

        self.assertEqual(demos["improved_steps"], 2)
        self.assertEqual(demos["anchor_keep_steps"], 0)
        self.assertEqual(demos["states"].shape[0], 2)
        np.testing.assert_allclose(
            demos["actions"],
            np.stack([improved_action, improved_action]),
        )
        self.assertEqual(demos["improved_weight_fraction"], 1.0)

    def test_advantage_filtered_demo_weights_balance_positive_and_anchor_labels(self) -> None:
        config = replace(
            make_legacy_two_facility_config(episode_horizon=2),
            action_mode="facility_net",
        )
        env = CapacityPlanningEnv(config, seed=10)
        anchor_action = np.zeros(env.action_size, dtype=np.float32)
        improved_action = anchor_action.copy()
        improved_action[3 * env.config.num_facilities] = 0.01
        baseline_metrics = {
            "total_cost": 100.0,
            "service_level": 0.8,
            "eligibility_rate": 0.9,
            "at_risk_unserved": 2.0,
            "patients_lost": 1.0,
        }
        improved_metrics = dict(baseline_metrics, total_cost=90.0)
        worse_metrics = dict(baseline_metrics, total_cost=110.0)

        with patch(
            "evaluation.run_gcn_residual_sweep.local_search_candidate_actions",
            return_value=[anchor_action, improved_action],
        ), patch(
            "evaluation.run_gcn_residual_sweep.rollout_metrics_after_action",
            side_effect=[
                baseline_metrics,
                improved_metrics,
                baseline_metrics,
                worse_metrics,
            ],
        ):
            demos = collect_local_search_demonstrations(
                env,
                seed=10,
                rollouts=1,
                lookahead=1,
                epsilons=(0.01,),
                max_steps=2,
                baseline_policy="myo",
                min_improvement=0.0,
                anchor_keep_probability=1.0,
                anchor_keep_weight=1.0,
                anchor_keep_on_improved=False,
                balance_label_weights=True,
            )

        self.assertEqual(demos["improved_steps"], 1)
        self.assertEqual(demos["anchor_keep_steps"], 1)
        self.assertAlmostEqual(demos["improved_weight_fraction"], 0.5, places=6)


class TrainingStabilityTests(unittest.TestCase):
    def test_stability_summary_flags_non_finite_values(self) -> None:
        rows = [
            {
                "algorithm": "flat_ddpg",
                "scenario": "s",
                "graph_ablation": "flat_state_no_graph",
                "seed": "0",
                "episode": "0",
                "total_reward": "1.0",
                "total_cost": "2.0",
            },
            {
                "algorithm": "flat_ddpg",
                "scenario": "s",
                "graph_ablation": "flat_state_no_graph",
                "seed": "0",
                "episode": "1",
                "total_reward": "nan",
                "total_cost": "3.0",
            },
        ]

        summary = summarize_training_stability(rows, window=1, min_episodes=1)

        self.assertEqual(summary[0]["status"], "check")
        self.assertIn("non_finite_metric", summary[0]["issues"])


if __name__ == "__main__":
    unittest.main()
