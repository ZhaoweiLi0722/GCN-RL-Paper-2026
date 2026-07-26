import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.merge_scenario_teacher_caches import (
    merge_scenario_parts,
    scenario_balanced_label_weights,
    validation_trajectory_count,
)
from evaluation.run_gcn_residual_sweep import (
    load_local_search_demonstrations,
    save_local_search_demonstrations,
)


def _part(scenario_id: int) -> dict:
    rows = 4
    return {
        "states": np.full((rows, 3), scenario_id, dtype=np.float32),
        "actions": np.zeros((rows, 2), dtype=np.float32),
        "weights": np.asarray([1.0, 4.0, 1.0, 2.0], dtype=np.float32),
        "improved_mask": np.asarray([False, True, False, True]),
        "option_advantages": np.asarray(
            [[0.0, 0.0], [0.0, 4.0], [0.0, 0.0], [0.0, 2.0]],
            dtype=np.float32,
        ),
        "option_feasible": np.ones((rows, 2), dtype=bool),
        "option_groups": np.asarray(["anchor", "reagent_transfer"]),
        "option_epsilons": np.asarray([0.0, 0.32], dtype=np.float32),
        "option_signs": np.asarray([0.0, 1.0], dtype=np.float32),
        "scenario_ids": np.full(rows, scenario_id, dtype=np.int64),
        "trajectory_ids": np.asarray(
            [2 * scenario_id, 2 * scenario_id, 2 * scenario_id + 1, 2 * scenario_id + 1],
            dtype=np.int64,
        ),
        "trajectory_steps": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "demand_history_window": 12,
    }


class ScenarioTeacherCacheTests(unittest.TestCase):
    def test_validation_split_retains_a_training_trajectory(self) -> None:
        self.assertEqual(validation_trajectory_count(1, 0.5), 0)
        self.assertEqual(validation_trajectory_count(2, 0.5), 1)
        self.assertEqual(validation_trajectory_count(10, 0.2), 2)

    def test_scenario_and_label_masses_are_balanced(self) -> None:
        scenario_ids = np.asarray([0, 0, 0, 1, 1, 1])
        improved = np.asarray([False, True, True, False, False, True])
        source = np.asarray([1.0, 9.0, 1.0, 2.0, 4.0, 8.0])

        weights = scenario_balanced_label_weights(
            scenario_ids,
            improved,
            source,
        )

        for scenario_id in (0, 1):
            self.assertAlmostEqual(
                float(weights[scenario_ids == scenario_id].sum()),
                3.0,
                places=5,
            )
            for label in (False, True):
                self.assertAlmostEqual(
                    float(
                        weights[
                            (scenario_ids == scenario_id)
                            & (improved == label)
                        ].sum()
                    ),
                    1.5,
                    places=5,
                )

    def test_merged_provenance_round_trips_without_pickle(self) -> None:
        merged = merge_scenario_parts(
            [_part(0), _part(1)],
            scenario_names=("nominal", "shift"),
            balance_scenarios=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.npz"
            save_local_search_demonstrations(path, merged)
            loaded = load_local_search_demonstrations(path)

        self.assertEqual(loaded["states"].shape, (8, 3))
        self.assertEqual(loaded["scenario_names"].tolist(), ["nominal", "shift"])
        self.assertEqual(loaded["scenario_cache_version"], 1)
        np.testing.assert_array_equal(
            loaded["scenario_ids"],
            merged["scenario_ids"],
        )
        np.testing.assert_array_equal(
            loaded["trajectory_ids"],
            merged["trajectory_ids"],
        )


if __name__ == "__main__":
    unittest.main()
