from pathlib import Path
import unittest

import numpy as np

from evaluation.collect_multiscenario_dagger import (
    causally_filter_base_cache,
    causal_query_start_step,
    require_cache_provenance,
    resolve_behavior_checkpoint,
    selected_deployments,
)


class MultiScenarioDaggerTest(unittest.TestCase):
    def test_query_starts_after_change_and_detection_delay(self) -> None:
        config = {
            "episode_horizon": 52,
            "demand_regime_initial_multipliers": [1.0, 1.0],
            "demand_regime_final_multipliers": [1.4, 0.7],
            "demand_regime_change_step": 26,
        }
        self.assertEqual(
            causal_query_start_step(config, detection_delay=4),
            30,
        )

    def test_stationary_scenario_can_query_from_reset(self) -> None:
        config = {
            "episode_horizon": 52,
            "demand_regime_initial_multipliers": [1.0, 1.0],
            "demand_regime_final_multipliers": [1.0, 1.0],
            "demand_regime_change_step": 13,
        }
        self.assertEqual(
            causal_query_start_step(config, detection_delay=4),
            0,
        )

    def test_query_override_is_validated(self) -> None:
        config = {"episode_horizon": 10}
        self.assertEqual(
            causal_query_start_step(
                config,
                detection_delay=4,
                override=7,
            ),
            7,
        )
        with self.assertRaisesRegex(ValueError, "within horizon"):
            causal_query_start_step(
                config,
                detection_delay=4,
                override=10,
            )

    def test_deployment_requires_validation_only_selection(self) -> None:
        summary = {
            "runs": [
                {
                    "algorithm": "gcn",
                    "training_seed": 2,
                    "selected_deployment": {
                        "selection_source": "validation_only",
                        "candidate": {
                            "scale": 0.1,
                            "group_thresholds": [0.8, 0.9, 0.8],
                        },
                    },
                }
            ]
        }
        deployments = selected_deployments(summary, algorithm="gcn")
        self.assertEqual(deployments[2]["scale"], 0.1)
        self.assertEqual(
            deployments[2]["group_thresholds"],
            [0.8, 0.9, 0.8],
        )

        summary["runs"][0]["selected_deployment"][
            "selection_source"
        ] = "holdout"
        with self.assertRaisesRegex(ValueError, "validation only"):
            selected_deployments(summary, algorithm="gcn")

    def test_behavior_override_must_match_validation_candidate(self) -> None:
        broad = {
            "scale": 1.0,
            "group_thresholds": [0.1, 0.1, 0.2],
        }
        summary = {
            "runs": [
                {
                    "algorithm": "gcn",
                    "training_seed": 0,
                    "selected_deployment": {
                        "selection_source": "validation_only",
                        "checkpoint_variant": "final",
                        "candidate": {
                            "scale": 0.0,
                            "group_thresholds": [1.0, 1.0, 1.0],
                        },
                    },
                    "validation": [
                        {
                            "checkpoint_variant": "final",
                            "candidate": broad,
                        }
                    ],
                }
            ]
        }

        deployments = selected_deployments(
            summary,
            algorithm="gcn",
            behavior_candidate=broad,
            checkpoint_variant="final",
        )
        self.assertEqual(deployments[0], broad)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            selected_deployments(
                summary,
                algorithm="gcn",
                behavior_candidate={
                    "scale": 1.0,
                    "group_thresholds": [0.2, 0.2, 0.2],
                },
                checkpoint_variant="final",
            )

    def test_behavior_checkpoint_supports_explicit_episode_boundary(self) -> None:
        run = {
            "checkpoint": "final.pt",
            "pretrain_checkpoint": "pretrain.pt",
        }
        self.assertEqual(
            resolve_behavior_checkpoint(
                run,
                training_seed=0,
                checkpoint_variant="episode4",
                behavior_checkpoints={0: "episode4.pt"},
            ),
            Path("episode4.pt"),
        )
        with self.assertRaisesRegex(ValueError, "explicit"):
            resolve_behavior_checkpoint(
                run,
                training_seed=0,
                checkpoint_variant="episode4",
            )

    def test_base_cache_requires_complete_provenance(self) -> None:
        cache = {
            "states": np.zeros((2, 3), dtype=np.float32),
            "scenario_names": np.asarray(["a"], dtype="U96"),
            "scenario_ids": np.zeros(2, dtype=np.int64),
            "trajectory_ids": np.zeros(2, dtype=np.int64),
            "trajectory_steps": np.arange(2, dtype=np.int64),
        }
        require_cache_provenance(cache)
        del cache["trajectory_steps"]
        with self.assertRaisesRegex(ValueError, "trajectory_steps"):
            require_cache_provenance(cache)

    def test_causal_filter_removes_prechange_teacher_rows(self) -> None:
        row_count = 8
        cache = {
            "states": np.zeros((row_count, 3), dtype=np.float32),
            "actions": np.zeros((row_count, 4), dtype=np.float32),
            "weights": np.ones(row_count, dtype=np.float32),
            "improved_mask": np.zeros(row_count, dtype=bool),
            "scenario_names": np.asarray(["nominal", "shift"], dtype="U96"),
            "scenario_ids": np.repeat([0, 1], 4).astype(np.int64),
            "trajectory_ids": np.repeat([0, 1], 4).astype(np.int64),
            "trajectory_steps": np.tile(np.arange(4), 2).astype(np.int64),
        }
        filtered = causally_filter_base_cache(
            cache,
            scenario_env_configs=[
                {
                    "scenario_name": "nominal",
                    "episode_horizon": 4,
                },
                {
                    "scenario_name": "shift",
                    "episode_horizon": 4,
                    "demand_regime_initial_multipliers": [1.0],
                    "demand_regime_final_multipliers": [2.0],
                    "demand_regime_change_step": 1,
                },
            ],
            detection_delay=1,
        )
        self.assertEqual(filtered["states"].shape[0], 6)
        np.testing.assert_array_equal(
            filtered["trajectory_steps"],
            np.asarray([0, 1, 2, 3, 2, 3]),
        )


if __name__ == "__main__":
    unittest.main()
