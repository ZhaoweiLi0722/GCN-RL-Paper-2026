"""Tests for the full benchmark manifest and runner helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from evaluation.check_training_stability import summarize_training_stability
from evaluation.run_full_benchmark import (
    final_checkpoint_path,
    load_benchmark_plan,
    make_training_config,
    resolve_budget,
    select_algorithms,
    select_scenarios,
)


class FullBenchmarkRunnerTests(unittest.TestCase):
    def test_smoke_budget_and_selection(self) -> None:
        plan = load_benchmark_plan()
        budget = resolve_budget(plan, "smoke")

        self.assertEqual(budget["seeds"], [0])
        self.assertEqual(budget["evaluation_replications"], 1)
        self.assertIn("sac", select_algorithms(plan, None))
        self.assertNotIn("sac", select_algorithms(plan, None, primary_only=True))
        self.assertEqual(select_scenarios(plan, ["disruption_0_3"])[0]["name"], "disruption_0_3")

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

    def test_final_checkpoint_path_matches_training_config(self) -> None:
        plan = load_benchmark_plan()
        budget = resolve_budget(plan, "smoke")
        scenario = select_scenarios(plan, ["disruption_0_6"])[0]
        config = make_training_config(plan, "smoke", budget, "td3", scenario, seed=2)

        self.assertEqual(
            final_checkpoint_path(plan, "smoke", budget, "td3", scenario, seed=2),
            Path(config["checkpoint_dir"]) / "td3_seed2_episode1.pt",
        )


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
