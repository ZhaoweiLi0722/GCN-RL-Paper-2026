from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    resolve_budget,
    select_scenarios,
)
from evaluation.train_network_residual_history_screen import (
    make_history_screen_config,
)


class NetworkResidualHistoryScreenTests(unittest.TestCase):
    def test_config_uses_matching_cache_and_environment_window(self) -> None:
        plan = load_benchmark_plan(
            "experiments/configs/residual_policy_benchmark.json"
        )
        budget = resolve_budget(plan, "diagnostic_pretrain")
        scenario = select_scenarios(
            plan,
            ("patient_condition_geo_demand_drift",),
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "teacher.npz"
            np.savez_compressed(
                cache,
                demand_history_window=np.asarray(12, dtype=np.int64),
            )
            config = make_history_screen_config(
                plan,
                budget_name="history12_pretrain",
                budget=budget,
                algorithm="gcn_residual_mdl2_network_ddpg_afd",
                scenario=scenario,
                seed=2,
                teacher_cache=cache,
                demand_history_window=12,
                base_policy="fmdl2",
                gcn_hidden_sizes=(32, 16),
                actor_hidden_sizes=(64, 32),
                gate_hidden_sizes=(32, 16),
                offline_updates=500,
                teacher_advantage_weight_scale=500_000.0,
                teacher_advantage_weight_cap=50.0,
            )

        self.assertEqual(config["num_episodes"], 0)
        self.assertEqual(config["env"]["demand_history_window"], 12)
        self.assertEqual(config["residual_action"]["base_policy"], "fmdl2")
        self.assertEqual(config["imitation_pretrain"]["policy"], "fmdl2")
        self.assertEqual(
            config["advantage_distillation_pretrain"]["baseline_policy"],
            "fmdl2",
        )
        self.assertEqual(
            config["advantage_distillation_pretrain"]["demonstration_path"],
            str(cache),
        )
        self.assertEqual(config["gcn_hidden_sizes"], [32, 16])
        self.assertEqual(config["actor_hidden_sizes"], [64, 32])
        self.assertEqual(
            config["residual_action"]["correction_gate"]["hidden_sizes"],
            [32, 16],
        )
        self.assertEqual(
            config["advantage_distillation_pretrain"][
                "dense_advantage_weight_scale"
            ],
            500_000.0,
        )
        self.assertEqual(
            config["advantage_distillation_pretrain"][
                "dense_advantage_weight_cap"
            ],
            50.0,
        )
        self.assertEqual(
            config["history_screen"]["teacher_advantage_weight_scale"],
            500_000.0,
        )
        self.assertEqual(
            config["history_screen"]["teacher_advantage_weight_cap"],
            50.0,
        )
        self.assertEqual(config["history_screen"]["offline_updates"], 500)

    def test_rejects_cache_window_mismatch(self) -> None:
        plan = load_benchmark_plan(
            "experiments/configs/residual_policy_benchmark.json"
        )
        budget = resolve_budget(plan, "diagnostic_pretrain")
        scenario = select_scenarios(
            plan,
            ("patient_condition_geo_demand_drift",),
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "teacher.npz"
            np.savez_compressed(
                cache,
                demand_history_window=np.asarray(8, dtype=np.int64),
            )
            with self.assertRaisesRegex(
                ValueError,
                "does not match requested window 12",
            ):
                make_history_screen_config(
                    plan,
                    budget_name="history12_pretrain",
                    budget=budget,
                    algorithm="gcn_residual_mdl2_network_ddpg_afd",
                    scenario=scenario,
                    seed=0,
                    teacher_cache=cache,
                    demand_history_window=12,
                )

    def test_flat_screen_applies_parameter_matched_mlp_widths(self) -> None:
        plan = load_benchmark_plan(
            "experiments/configs/residual_policy_benchmark.json"
        )
        budget = resolve_budget(plan, "diagnostic_pretrain")
        scenario = select_scenarios(
            plan,
            ("patient_condition_geo_abrupt_regime_shift",),
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "teacher.npz"
            np.savez_compressed(
                cache,
                demand_history_window=np.asarray(12, dtype=np.int64),
            )
            config = make_history_screen_config(
                plan,
                budget_name="abrupt_shift_pretrain",
                budget=budget,
                algorithm="flat_residual_mdl2_network_ddpg_afd",
                scenario=scenario,
                seed=0,
                teacher_cache=cache,
                demand_history_window=12,
                actor_hidden_sizes=(280, 256),
                gate_hidden_sizes=(80, 48),
            )

        self.assertEqual(config["hidden_sizes"], [280, 256])
        self.assertEqual(config["history_screen"]["actor_hidden_sizes"], [])
        self.assertEqual(
            config["residual_action"]["correction_gate"]["hidden_sizes"],
            [80, 48],
        )
        self.assertEqual(config["history_screen"]["hidden_sizes"], [280, 256])

    def test_option_screen_uses_all_ordered_teacher_options(self) -> None:
        plan = load_benchmark_plan(
            "experiments/configs/residual_policy_benchmark.json"
        )
        budget = resolve_budget(plan, "diagnostic_pretrain")
        scenario = select_scenarios(
            plan,
            ("patient_condition_geo_abrupt_regime_shift",),
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "teacher.npz"
            np.savez_compressed(
                cache,
                demand_history_window=np.asarray(12, dtype=np.int64),
                option_groups=np.asarray(
                    ["anchor", "reagent_transfer", "combined_network"],
                    dtype="U32",
                ),
                option_epsilons=np.asarray([0.0, 0.32, 0.48]),
                option_signs=np.asarray([0.0, -1.0, 1.0]),
            )
            config = make_history_screen_config(
                plan,
                budget_name="abrupt_shift_option_pretrain",
                budget=budget,
                algorithm="gcn_residual_mdl2_option_dqn_afd",
                scenario=scenario,
                seed=0,
                teacher_cache=cache,
                demand_history_window=12,
                option_hidden_sizes=(96, 48),
                use_all_teacher_options=True,
            )

        self.assertEqual(
            config["residual_option"]["explicit_options"],
            [
                {
                    "group": "reagent_transfer",
                    "epsilon": 0.32,
                    "sign": -1.0,
                },
                {
                    "group": "combined_network",
                    "epsilon": 0.48,
                    "sign": 1.0,
                },
            ],
        )
        self.assertEqual(config["residual_option"]["hidden_sizes"], [96, 48])
        self.assertEqual(config["history_screen"]["teacher_option_count"], 3)

    def test_option_screen_configures_online_fine_tuning(self) -> None:
        plan = load_benchmark_plan(
            "experiments/configs/residual_policy_benchmark.json"
        )
        budget = resolve_budget(plan, "network_targeted_300")
        scenario = select_scenarios(
            plan,
            ("patient_condition_geo_abrupt_regime_shift",),
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "teacher.npz"
            np.savez_compressed(
                cache,
                demand_history_window=np.asarray(12, dtype=np.int64),
            )
            config = make_history_screen_config(
                plan,
                budget_name="abrupt_shift_option_online",
                budget=budget,
                algorithm="gcn_residual_mdl2_option_dqn_afd",
                scenario=scenario,
                seed=0,
                teacher_cache=cache,
                demand_history_window=12,
                online_episodes=100,
                pretrain_epochs=250,
                option_advantage_scale=10_000_000.0,
                option_ranking_loss_weight=1.0,
                correction_gate_threshold=0.9,
                online_reward_mode="one_step_anchor_relative",
                imitation_regularization_weight=1.0,
            )

        self.assertEqual(config["num_episodes"], 100)
        self.assertFalse(config["history_screen"]["pretrain_only"])
        self.assertEqual(
            config["advantage_distillation_pretrain"]["epochs"],
            250,
        )
        self.assertEqual(
            config["residual_option"]["option_advantage_scale"],
            10_000_000.0,
        )
        self.assertEqual(
            config["residual_option"]["ranking_loss_weight"],
            1.0,
        )
        self.assertEqual(
            config["residual_option"]["correction_gate_threshold"],
            0.9,
        )
        self.assertEqual(
            config["residual_option"]["online_reward_mode"],
            "one_step_anchor_relative",
        )
        self.assertEqual(
            config["residual_option"]["imitation_regularization_weight"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
