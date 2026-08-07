"""Tests for the calibrated network-residual headroom diagnostic."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluation.augment_teacher_cache_time_state import (
    augment_teacher_cache_with_time,
)
from evaluation.merge_headroom_teacher_shards import (
    discover_teacher_shards,
    merge_demonstration_caches,
    normalized_shard_rows,
    validate_teacher_shard_result,
)
from evaluation.merge_headroom_state_probe_shards import (
    merge_state_probe_rows,
)
from evaluation.network_residual_headroom import (
    headroom_decision,
    lookahead_rollout_seeds,
    select_clinical_candidate,
    state_probe_shard_config,
    teacher_shard_config,
)
from evaluation.probe_option_distillation import pretrain_gate_decision


class NetworkResidualHeadroomTests(unittest.TestCase):
    def test_teacher_cache_time_augmentation_preserves_rows(self) -> None:
        payload = {
            "states": np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
            "transition_states": np.asarray(
                [[1.0], [2.0], [3.0], [4.0]],
                dtype=np.float32,
            ),
            "transition_next_states": np.asarray(
                [[2.0], [0.0], [4.0], [0.0]],
                dtype=np.float32,
            ),
            "transition_dones": np.asarray([False, True, False, True]),
            "actions": np.ones((4, 1), dtype=np.float32),
        }

        augmented, summary = augment_teacher_cache_with_time(
            payload,
            {"include_time_state": True, "episode_horizon": 2},
        )

        np.testing.assert_allclose(augmented["states"][:, -1], [0.0, 0.5, 0.0, 0.5])
        np.testing.assert_allclose(
            augmented["transition_next_states"][:, -1],
            [0.5, 1.0, 0.5, 1.0],
        )
        np.testing.assert_array_equal(augmented["actions"], payload["actions"])
        self.assertEqual(summary["trajectory_lengths"], [2, 2])
    def test_lookahead_seeds_are_common_and_decision_specific(self) -> None:
        config = {
            "seed": 10,
            "lookahead_seed": 1000,
            "lookahead_replications": 3,
        }

        self.assertEqual(lookahead_rollout_seeds(config, 0), (1000, 1001, 1002))
        self.assertEqual(lookahead_rollout_seeds(config, 1), (1003, 1004, 1005))

    def test_teacher_shards_preserve_global_replication_and_decision_offsets(self) -> None:
        config = {
            "teacher_replications": 10,
            "max_steps": 52,
            "output_root": "results/test",
            "demonstration_path": "results/test/cache.npz",
        }

        shard = teacher_shard_config(config, 3, 5)

        self.assertEqual(shard["teacher_replications"], 2)
        self.assertEqual(shard["teacher_replication_start"], 6)
        self.assertEqual(shard["lookahead_decision_offset"], 312)
        self.assertTrue(shard["output_root"].endswith("shard_03_of_05"))

    def test_teacher_shard_discovery_rejects_incomplete_set(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "shard_00_of_02").mkdir()

            with self.assertRaisesRegex(ValueError, "incomplete"):
                discover_teacher_shards(root)

    def test_teacher_shard_result_requires_exact_locked_config(self) -> None:
        config = {
            "name": "test_teacher",
            "teacher_replications": 3,
            "max_steps": 52,
            "output_root": "results/test",
            "demonstration_path": "results/test/teacher_cache.npz",
        }
        expected = teacher_shard_config(config, 1, 3)
        result = {
            "config": expected,
            "online_teacher": {
                "teacher_replication_start": 1,
                "lookahead_decision_offset": 52,
            },
        }

        validated = validate_teacher_shard_result(
            config,
            result,
            shard_index=1,
            shard_count=3,
        )

        self.assertEqual(validated, expected)
        result["config"] = {**expected, "max_steps": 51}
        with self.assertRaisesRegex(ValueError, "config does not match"):
            validate_teacher_shard_result(
                config,
                result,
                shard_index=1,
                shard_count=3,
            )

    def test_teacher_shard_rows_require_complete_local_replications(self) -> None:
        with self.assertRaisesRegex(ValueError, "local replications"):
            normalized_shard_rows(
                [
                    {"replication": "0"},
                    {"replication": "0"},
                ],
                start=0,
                count=2,
                base_evaluation_seed=100,
            )

    def test_state_probe_shards_preserve_global_rollout_indices(self) -> None:
        config = {
            "name": "test_probe",
            "state_probe_rollouts": 10,
            "max_steps": 52,
            "output_root": "results/test",
        }

        shard = state_probe_shard_config(config, 3, 4)

        self.assertEqual(shard["state_probe_rollouts"], 2)
        self.assertEqual(shard["state_probe_rollout_start"], 8)
        self.assertEqual(shard["state_probe_total_rollouts"], 10)
        self.assertEqual(
            Path(shard["output_root"]).parts[-2:],
            ("state_probe_shards", "shard_03_of_04"),
        )

    def test_state_probe_rows_merge_in_canonical_order(self) -> None:
        shard_rows = [
            [
                {"rollout": "1", "step": "1", "selected_group": "anchor"},
                {"rollout": "1", "step": "0", "selected_group": "anchor"},
            ],
            [
                {"rollout": "0", "step": "1", "selected_group": "anchor"},
                {"rollout": "0", "step": "0", "selected_group": "anchor"},
            ],
        ]

        merged = merge_state_probe_rows(
            shard_rows,
            total_rollouts=2,
            max_steps=2,
        )

        self.assertEqual(
            [(int(row["rollout"]), int(row["step"])) for row in merged],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )

    def test_state_probe_merge_rejects_missing_rollout(self) -> None:
        with self.assertRaisesRegex(ValueError, "cover every rollout"):
            merge_state_probe_rows(
                [[{"rollout": "0", "step": "0"}]],
                total_rollouts=2,
                max_steps=2,
            )

    def test_teacher_shard_caches_merge_in_replication_order(self) -> None:
        def cache(value: float) -> dict:
            return {
                "states": np.asarray([[value, value]], dtype=np.float32),
                "actions": np.asarray([[value]], dtype=np.float32),
                "weights": np.asarray([1.0], dtype=np.float32),
                "improved_mask": np.asarray([value > 0.0]),
                "transition_states": np.asarray(
                    [[value, value]],
                    dtype=np.float32,
                ),
                "transition_actions": np.asarray([[value]], dtype=np.float32),
                "transition_rewards": np.asarray([-value], dtype=np.float32),
                "transition_next_states": np.asarray(
                    [[value + 1.0, value + 1.0]],
                    dtype=np.float32,
                ),
                "transition_dones": np.asarray([True]),
                "option_advantages": np.asarray(
                    [[0.0, value]],
                    dtype=np.float32,
                ),
                "option_feasible": np.asarray([[True, True]]),
                "option_groups": np.asarray(
                    ["anchor", "reagent_transfer"],
                ),
                "option_epsilons": np.asarray([0.0, 0.32], dtype=np.float32),
                "option_signs": np.asarray([0.0, 1.0], dtype=np.float32),
                "improved_steps": int(value > 0.0),
                "anchor_keep_steps": int(value <= 0.0),
                "service_rejected_steps": 0,
                "mean_step_improvement": max(value, 0.0),
                "improved_weight_fraction": float(value > 0.0),
            }

        merged = merge_demonstration_caches([cache(0.0), cache(2.0)])

        np.testing.assert_allclose(merged["states"][:, 0], [0.0, 2.0])
        self.assertEqual(merged["improved_steps"], 1)
        self.assertEqual(merged["anchor_keep_steps"], 1)
        self.assertAlmostEqual(merged["mean_step_improvement"], 2.0)

    def test_clinical_selection_rejects_patient_harm(self) -> None:
        evaluated = [
            {
                "metrics": {
                    "total_cost": 100.0,
                    "service_level": 0.9,
                    "completion_service_level": 0.8,
                    "eligibility_rate": 0.9,
                    "patient_ineligibility_during_manufacturing_rate": 0.1,
                    "at_risk_unserved": 1.0,
                    "patients_lost": 2.0,
                }
            },
            {
                "metrics": {
                    "total_cost": 90.0,
                    "service_level": 0.9,
                    "completion_service_level": 0.79,
                    "eligibility_rate": 0.9,
                    "patient_ineligibility_during_manufacturing_rate": 0.1,
                    "at_risk_unserved": 1.0,
                    "patients_lost": 2.0,
                }
            },
            {
                "metrics": {
                    "total_cost": 95.0,
                    "service_level": 0.9,
                    "completion_service_level": 0.8,
                    "eligibility_rate": 0.9,
                    "patient_ineligibility_during_manufacturing_rate": 0.1,
                    "at_risk_unserved": 1.0,
                    "patients_lost": 2.0,
                }
            },
        ]

        selected, _scores, feasible = select_clinical_candidate(
            evaluated,
            score_weights={},
            guardrails={
                "min_completion_service_level_delta": 0.0,
                "max_patients_lost_delta": 0.0,
                "max_patient_ineligibility_during_manufacturing_rate_delta": 0.0,
            },
        )

        self.assertEqual(selected, 2)
        self.assertEqual(feasible, [True, False, True])

    def test_headroom_gate_requires_material_significant_gain(self) -> None:
        result = {
            "state_probe": {"opportunity_rate": 0.2},
            "online_teacher": {
                "paired": {
                    "total_cost": {
                        "mean_gap_pct": -1.5,
                        "ci_high": -100.0,
                    }
                }
            },
        }
        config = {
            "materiality": {
                "minimum_episode_cost_improvement_pct": 1.0,
                "minimum_opportunity_rate": 0.05,
            }
        }

        decision = headroom_decision(result, config)

        self.assertTrue(decision["advance_to_network_residual_training"])
        self.assertTrue(np.isfinite(decision["teacher_cost_gap_pct"]))

    def test_headroom_gate_uses_online_teacher_opportunity_without_state_probe(
        self,
    ) -> None:
        result = {
            "online_teacher": {
                "teacher_correction_rate": 0.25,
                "paired": {
                    "total_cost": {
                        "mean_gap_pct": -1.5,
                        "ci_high": -100.0,
                    }
                },
            }
        }
        config = {
            "materiality": {
                "minimum_episode_cost_improvement_pct": 1.0,
                "minimum_opportunity_rate": 0.05,
            }
        }

        decision = headroom_decision(result, config)

        self.assertTrue(decision["advance_to_network_residual_training"])
        self.assertEqual(decision["opportunity_rate_source"], "online_teacher")
        self.assertAlmostEqual(decision["state_opportunity_rate"], 0.25)

    def test_headroom_gate_rejects_episode_level_clinical_harm(self) -> None:
        result = {
            "state_probe": {"opportunity_rate": 0.2},
            "online_teacher": {
                "paired": {
                    "total_cost": {
                        "mean_gap_pct": -2.0,
                        "ci_high": -100.0,
                    },
                    "completion_service_level": {"mean_difference": -0.01},
                    "patients_lost": {"mean_difference": -2.0},
                    "patient_ineligibility_during_manufacturing_rate": {
                        "mean_difference": -0.01
                    },
                }
            },
        }
        config = {
            "materiality": {
                "minimum_episode_cost_improvement_pct": 1.0,
                "minimum_opportunity_rate": 0.05,
                "require_episode_clinical_noninferiority": True,
            }
        }

        decision = headroom_decision(result, config)

        self.assertFalse(decision["advance_to_network_residual_training"])
        self.assertFalse(decision["episode_clinical_noninferiority"])

    def test_option_pretrain_gate_requires_real_safe_correction_gain(self) -> None:
        result = {
            "paired": {
                "total_cost": {
                    "mean_difference": -5.0,
                    "mean_gap_pct": -1.2,
                    "ci_high": -1.0,
                },
                "completion_service_level": {"mean_difference": 0.01},
                "patients_lost": {"mean_difference": -2.0},
                "patient_ineligibility_during_manufacturing_rate": {
                    "mean_difference": -0.01
                },
            },
            "option_diagnostics": {"correction_rate": 0.25},
        }

        decision = pretrain_gate_decision(result)

        self.assertTrue(decision["advance_to_online_rl"])


if __name__ == "__main__":
    unittest.main()
