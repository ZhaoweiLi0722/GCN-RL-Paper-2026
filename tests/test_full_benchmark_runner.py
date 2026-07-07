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
    config_snapshot_path,
    evaluation_csv_path,
    evaluation_outputs_complete,
    evaluation_summary_path,
    final_checkpoint_path,
    load_benchmark_plan,
    make_training_config,
    resolve_budget,
    select_algorithms,
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

        self.assertEqual(len(actions), 5)
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
