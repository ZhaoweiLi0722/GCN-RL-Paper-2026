"""Tests for the full benchmark manifest and runner helpers."""

from __future__ import annotations

from dataclasses import replace
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluation.check_training_stability import summarize_training_stability
from evaluation.run_gcn_residual_sweep import (
    best_variant_summary,
    collect_local_search_demonstrations,
    elite_sample_weights,
    local_search_candidate_actions,
    make_residual_sweep_config,
    residual_variant_name,
    summary_metadata,
)
from evaluation.run_full_benchmark import (
    algorithm_config_overrides,
    anchor_fallback_settings,
    config_snapshot_path,
    evaluation_csv_path,
    evaluation_outputs_complete,
    evaluation_summary_path,
    final_checkpoint_path,
    checkpoint_label,
    learned_checkpoint_candidates,
    local_search_checkpoint_path,
    load_benchmark_plan,
    make_evaluation_config,
    make_scenario_env_config,
    make_training_config,
    resolve_budget,
    select_algorithms,
    select_anchor_fallback_policy,
    select_scenarios,
    training_csv_path,
    training_outputs_complete,
)
from src.baselines.heuristics import MyopicPolicy
from src.env.capacity_planning import CapacityPlanningEnv, make_legacy_two_facility_config


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

        self.assertEqual(plan["name"], "20_clinic_residual_policy_benchmark")
        self.assertEqual(budget["num_episodes"], 300)
        self.assertEqual(mini_pilot["num_episodes"], 10)
        self.assertEqual(targeted_100["num_episodes"], 100)
        self.assertEqual(targeted_100["anchor_fallback"]["validation_replications"], 10)
        self.assertEqual(targeted_100["anchor_fallback"]["min_improvement"], 0.005)
        self.assertTrue(targeted_100["checkpoint_selection"]["enabled"])
        self.assertEqual(targeted_100["local_search"]["gcn_residual_mdl2"]["min_improvement"], 0.0)
        self.assertIn("flat_residual_mdl2", select_algorithms(plan, None, primary_only=True))
        self.assertIn("flat_residual_pmyo", select_algorithms(plan, None, primary_only=True))
        self.assertIn("gcn_residual_mdl2", select_algorithms(plan, None, primary_only=True))
        self.assertIn("pmyo", select_algorithms(plan, None, primary_only=True))
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

    def test_patient_priority_residual_plan_sets_patient_aware_anchor(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["patient_condition_stress"])[0]
        flat_config = make_training_config(plan, "smoke", budget, "flat_residual_pmyo", scenario, seed=0)
        gcn_config = make_training_config(plan, "smoke", budget, "gcn_residual_pmyo", scenario, seed=0)

        self.assertEqual(flat_config["residual_action"]["base_policy"], "pmyo")
        self.assertEqual(flat_config["imitation_pretrain"]["policy"], "pmyo")
        self.assertEqual(gcn_config["residual_action"]["base_policy"], "pmyo")
        self.assertEqual(gcn_config["imitation_pretrain"]["policy"], "pmyo")
        self.assertEqual(flat_config["env"]["env_type"], "patient_condition")

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
        self.assertEqual(config["env"]["transfer_lead_time"], 0)
        self.assertFalse(config["env"]["include_transfer_pipeline_state"])

    def test_anchor_fallback_settings_and_decision_rule(self) -> None:
        plan = load_benchmark_plan("experiments/configs/residual_policy_benchmark.json")
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["patient_condition_stress"])[0]
        config = make_training_config(plan, "smoke", budget, "gcn_residual_pmyo", scenario, seed=0)

        settings = anchor_fallback_settings(config, budget)

        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["validation_replications"], 1)
        self.assertEqual(
            select_anchor_fallback_policy(95.0, 100.0, min_improvement=0.0),
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
