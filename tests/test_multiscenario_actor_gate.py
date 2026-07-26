import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluation.calibrate_multiscenario_actor_gate import (
    actor_gate_weights,
    grouped_actor_actions,
    load_actor_gate_examples,
)


class MultiScenarioActorGateTest(unittest.TestCase):
    def test_grouped_actor_actions_change_only_requested_group(self) -> None:
        anchor = np.zeros(8, dtype=np.float32)
        actor = np.arange(8, dtype=np.float32) / 8.0
        reagent, replenishment = grouped_actor_actions(
            anchor,
            actor,
            group_names=("reagent_transfer", "replenishment"),
            num_facilities=2,
        )
        np.testing.assert_allclose(reagent[2:4], actor[2:4])
        np.testing.assert_allclose(reagent[:2], 0.0)
        np.testing.assert_allclose(reagent[4:], 0.0)
        np.testing.assert_allclose(replenishment[6:8], actor[6:8])
        np.testing.assert_allclose(replenishment[:6], 0.0)

    def test_actor_gate_weights_emphasize_large_advantage_magnitude(self) -> None:
        labels = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        advantages = np.asarray(
            [[-100_000.0, -50_000.0], [3_000_000.0, 0.0], [0.0, 500_000.0]],
            dtype=np.float32,
        )
        weights = actor_gate_weights(
            labels,
            advantages,
            positive_mass=None,
        )
        self.assertGreater(weights[1], weights[2])
        self.assertGreater(weights[2], weights[0])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_actor_gate_positive_mass_is_sample_level(self) -> None:
        labels = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        advantages = np.ones_like(labels)
        weights = actor_gate_weights(
            labels,
            advantages,
            positive_mass=0.25,
        )
        positive = np.any(labels > 0.5, axis=1)
        self.assertAlmostEqual(
            float(weights[positive].sum() / weights.sum()),
            0.25,
            places=6,
        )

    def test_cached_actor_examples_can_use_a_new_fixed_margin(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "examples.npz"
            np.savez_compressed(
                path,
                states=np.zeros((3, 4), dtype=np.float32),
                advantages=np.asarray(
                    [[50_000.0], [100_000.0], [250_000.0]],
                    dtype=np.float32,
                ),
                feasible=np.asarray(
                    [[True], [False], [True]],
                    dtype=bool,
                ),
                scenario_ids=np.asarray([0, 0, 0], dtype=np.int64),
                trajectory_ids=np.asarray([0, 0, 1], dtype=np.int64),
                trajectory_steps=np.asarray([0, 1, 0], dtype=np.int64),
                scenario_names=np.asarray(["scenario"], dtype="U96"),
                group_names=np.asarray(
                    ["reagent_transfer"],
                    dtype="U32",
                ),
            )

            examples = load_actor_gate_examples(
                path,
                scenario_names=("scenario",),
                min_improvement=100_000.0,
            )

        np.testing.assert_array_equal(
            examples["labels"].reshape(-1),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(examples["group_names"], ("reagent_transfer",))


if __name__ == "__main__":
    unittest.main()
