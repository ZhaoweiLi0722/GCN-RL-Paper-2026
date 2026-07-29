from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.train_multiscenario_network_residual import (
    maybe_load_initial_checkpoint,
    merge_multiscenario_env_overrides,
    multiscenario_env_overrides,
)


class MultiscenarioEnvOverrideTests(unittest.TestCase):
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
