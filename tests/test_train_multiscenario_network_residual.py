import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from evaluation.train_multiscenario_network_residual import (
    configure_online_imitation_regularization,
    maybe_load_initial_checkpoint,
    merge_multiscenario_env_overrides,
    multiscenario_env_overrides,
    train_multiscenario_agents,
    verify_file_sha256,
)


class MultiscenarioEnvOverrideTests(unittest.TestCase):
    def test_optional_file_sha256_lock_rejects_changed_teacher(self):
        with TemporaryDirectory() as directory:
            cache = Path(directory) / "teacher.npz"
            cache.write_bytes(b"locked teacher")
            expected = hashlib.sha256(b"locked teacher").hexdigest()
            self.assertEqual(verify_file_sha256(cache, expected), expected)
            with self.assertRaisesRegex(ValueError, "File SHA256 mismatch"):
                verify_file_sha256(cache, "0" * 64)

    def test_online_imitation_cache_is_loaded_with_provenance(self):
        class Agent:
            algorithm = "test_ddpg"

            def __init__(self):
                self.received = None

            def configure_online_imitation_regularization(
                self,
                states,
                actions,
                *,
                weights,
                seed,
            ):
                self.received = (states, actions, weights, seed)
                return {
                    "samples": int(states.shape[0]),
                    "weighted": weights is not None,
                    "seed": seed,
                }

        demonstrations = {
            "states": np.zeros((2, 3), dtype=np.float32),
            "actions": np.ones((2, 4), dtype=np.float32),
            "weights": np.asarray([0.5, 1.5], dtype=np.float32),
            "improved_steps": 1,
            "anchor_keep_steps": 1,
            "improved_weight_fraction": 0.75,
        }
        agent = Agent()
        with TemporaryDirectory() as directory:
            cache = Path(directory) / "teacher.npz"
            cache.write_bytes(b"locked teacher")
            with (
                patch(
                    "evaluation.train_multiscenario_network_residual."
                    "load_local_search_demonstrations",
                    return_value=demonstrations,
                ),
                patch(
                    "evaluation.train_multiscenario_network_residual."
                    "balance_demonstration_label_weights",
                    side_effect=lambda value: value,
                ) as balance,
            ):
                summary = configure_online_imitation_regularization(
                    agent,
                    {
                        "seed": 7,
                        "imitation_pretrain": {
                            "regularization_weight": 1.0,
                        },
                        "online_imitation_regularization": {
                            "enabled": True,
                            "demonstration_path": str(cache),
                            "balance_label_weights": True,
                            "seed": 99,
                        },
                    },
                )

        balance.assert_called_once_with(demonstrations)
        self.assertEqual(agent.received[3], 99)
        self.assertEqual(summary["online_imitation_samples"], 2)
        self.assertEqual(summary["online_imitation_improved_steps"], 1)
        self.assertEqual(summary["online_imitation_anchor_keep_steps"], 1)
        self.assertEqual(
            summary["online_imitation_demonstration_sha256"],
            hashlib.sha256(b"locked teacher").hexdigest(),
        )

    def test_resume_requires_single_algorithm_and_seed(self):
        with self.assertRaisesRegex(ValueError, "one algorithm and one seed"):
            train_multiscenario_agents(
                {},
                resume_training_state="partial.pt",
            )

    def test_optional_initial_checkpoint_is_loaded(self):
        class Agent:
            def __init__(self):
                self.loaded = None

            def load_actor(self, path):
                self.loaded = Path(path)

        agent = Agent()
        self.assertIsNone(maybe_load_initial_checkpoint(agent, {}))
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "actor.pt"
            checkpoint.touch()
            loaded = maybe_load_initial_checkpoint(
                agent,
                {"initial_checkpoint": str(checkpoint)},
            )

        self.assertEqual(loaded, str(checkpoint))
        self.assertEqual(agent.loaded, checkpoint)

    def test_initial_checkpoint_template_uses_algorithm_and_seed(self):
        class Agent:
            def __init__(self):
                self.loaded = None

            def load_actor(self, path):
                self.loaded = Path(path)

        agent = Agent()
        with TemporaryDirectory() as directory:
            checkpoint = (
                Path(directory)
                / "gcn_residual"
                / "seed2"
                / "gcn_residual_seed2_pretrain.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            loaded = maybe_load_initial_checkpoint(
                agent,
                {
                    "algorithm": "gcn_residual",
                    "seed": 2,
                    "initial_checkpoint": str(
                        Path(directory)
                        / "{algorithm}"
                        / "seed{seed}"
                        / "{algorithm}_seed{seed}_pretrain.pt"
                    ),
                },
            )

        self.assertEqual(loaded, str(checkpoint))
        self.assertEqual(agent.loaded, checkpoint)

    def test_env_overrides_preserve_scenario_and_apply_algorithm_precedence(self):
        scenario_env = {
            "scenario_name": "regional_drift",
            "regional_demand_shift": {"enabled": True, "magnitude": 0.4},
            "include_demand_sequence_state": False,
            "demand_sequence_length": 4,
        }

        merged = merge_multiscenario_env_overrides(
            scenario_env,
            common_overrides={
                "env": {
                    "include_demand_sequence_state": True,
                    "demand_sequence_length": 12,
                }
            },
            algorithm_overrides={
                "env": {
                    "demand_sequence_length": 8,
                }
            },
            demand_history_window=12,
        )

        self.assertEqual(merged["scenario_name"], "regional_drift")
        self.assertEqual(
            merged["regional_demand_shift"],
            {"enabled": True, "magnitude": 0.4},
        )
        self.assertTrue(merged["include_demand_sequence_state"])
        self.assertEqual(merged["demand_sequence_length"], 8)
        self.assertTrue(merged["include_demand_history_state"])
        self.assertEqual(merged["demand_history_window"], 12)
        self.assertFalse(scenario_env["include_demand_sequence_state"])

    def test_shared_contract_contains_only_explicit_env_settings(self):
        contract = multiscenario_env_overrides(
            common_overrides={
                "batch_size": 64,
                "env": {"include_demand_sequence_state": True},
            },
            algorithm_overrides={
                "actor_lr": 1e-4,
                "env": {"demand_sequence_length": 12},
            },
            demand_history_window=8,
        )

        self.assertEqual(
            contract,
            {
                "include_demand_sequence_state": True,
                "demand_sequence_length": 12,
                "demand_history_window": 8,
                "include_demand_history_state": True,
            },
        )

    def test_non_mapping_env_override_is_rejected(self):
        with self.assertRaises(TypeError):
            merge_multiscenario_env_overrides(
                {},
                common_overrides={"env": "invalid"},
                algorithm_overrides={},
                demand_history_window=12,
            )


if __name__ == "__main__":
    unittest.main()
