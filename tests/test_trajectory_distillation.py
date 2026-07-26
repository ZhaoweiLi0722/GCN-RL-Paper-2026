"""Tests for trajectory-held-out residual distillation."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.run_gcn_residual_sweep import (
    fit_action_batch_with_early_stopping,
    split_demonstrations_by_trajectory,
)
from src.rl.networks import require_torch, torch


def make_demonstrations() -> dict[str, object]:
    row_count = 8
    return {
        "states": np.arange(row_count, dtype=np.float32).reshape(-1, 1),
        "actions": np.zeros((row_count, 1), dtype=np.float32),
        "weights": np.ones(row_count, dtype=np.float32),
        "improved_mask": np.asarray(
            [True, False, True, False, True, True, False, False],
            dtype=bool,
        ),
        "improved_steps": 4,
        "anchor_keep_steps": 4,
        "service_rejected_steps": 0,
        "mean_step_improvement": 1.0,
        "improved_weight_fraction": 0.5,
        "scenario_ids": np.repeat([0, 1], 4).astype(np.int64),
        "trajectory_ids": np.repeat(
            [10, 11, 20, 21],
            2,
        ).astype(np.int64),
        "trajectory_steps": np.tile([0, 1], 4).astype(np.int64),
        "scenario_names": np.asarray(["a", "b"], dtype="U96"),
        "option_groups": np.asarray(
            ["anchor", "reagent_transfer"],
            dtype="U32",
        ),
        "option_epsilons": np.asarray([0.0, 0.1], dtype=np.float32),
        "option_signs": np.asarray([0.0, 1.0], dtype=np.float32),
        "option_advantages": np.zeros((row_count, 2), dtype=np.float32),
        "option_feasible": np.ones((row_count, 2), dtype=bool),
    }


class TrajectoryDistillationTests(unittest.TestCase):
    def test_split_holds_out_complete_trajectories_per_scenario(self) -> None:
        train, validation, summary = split_demonstrations_by_trajectory(
            make_demonstrations(),
            validation_fraction=0.5,
            min_per_scenario=1,
            seed=17,
        )

        train_ids = set(np.asarray(train["trajectory_ids"]).tolist())
        validation_ids = set(
            np.asarray(validation["trajectory_ids"]).tolist()
        )
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(len(train_ids), 2)
        self.assertEqual(len(validation_ids), 2)
        self.assertEqual(
            set(np.asarray(train["scenario_ids"]).tolist()),
            {0, 1},
        )
        self.assertEqual(
            set(np.asarray(validation["scenario_ids"]).tolist()),
            {0, 1},
        )
        self.assertEqual(
            np.asarray(train["option_advantages"]).shape[0],
            int(summary["train_samples"]),
        )
        np.testing.assert_array_equal(
            train["option_groups"],
            np.asarray(["anchor", "reagent_transfer"]),
        )

    def test_split_rejects_missing_provenance(self) -> None:
        demos = make_demonstrations()
        del demos["trajectory_ids"]
        with self.assertRaisesRegex(
            ValueError,
            "scenario_ids and trajectory_ids",
        ):
            split_demonstrations_by_trajectory(
                demos,
                validation_fraction=0.25,
                min_per_scenario=1,
                seed=3,
            )

    def test_early_stopping_restores_best_epoch(self) -> None:
        require_torch()

        class FakeAgent:
            def __init__(self) -> None:
                self.actor = torch.nn.Linear(1, 1, bias=False)
                self.actor_target = torch.nn.Linear(1, 1, bias=False)
                torch.nn.init.zeros_(self.actor.weight)
                torch.nn.init.zeros_(self.actor_target.weight)
                self.actor_optimizer = torch.optim.SGD(
                    self.actor.parameters(),
                    lr=0.1,
                )

            def fit_action_batch(
                self,
                states,
                actions,
                settings,
                weights=None,
            ):
                with torch.no_grad():
                    self.actor.weight.add_(
                        float(settings["epochs"])
                    )
                    self.actor_target.load_state_dict(
                        self.actor.state_dict()
                    )
                return {
                    "samples": int(len(states)),
                    "final_loss": 0.0,
                    "target_mode": "residual",
                }

            def evaluate_action_batch(
                self,
                states,
                actions,
                *,
                weights=None,
            ):
                value = float(self.actor.weight.item())
                loss = (value - 2.0) ** 2
                return {
                    "samples": int(len(states)),
                    "loss": loss,
                    "actor_loss": loss,
                    "target_mode": "residual",
                }

        demos = make_demonstrations()
        train, validation, _summary = split_demonstrations_by_trajectory(
            demos,
            validation_fraction=0.5,
            min_per_scenario=1,
            seed=17,
        )
        agent = FakeAgent()
        summary = fit_action_batch_with_early_stopping(
            agent,
            train,
            validation,
            epochs=8,
            batch_size=4,
            seed=5,
            retain_for_regularization=False,
            patience=2,
            check_interval=1,
            min_delta=0.0,
        )

        self.assertEqual(summary["best_epoch"], 2)
        self.assertEqual(summary["epochs_completed"], 4)
        self.assertTrue(summary["early_stopped"])
        self.assertAlmostEqual(float(agent.actor.weight.item()), 2.0)
        self.assertAlmostEqual(summary["validation_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
