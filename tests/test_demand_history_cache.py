import unittest

import numpy as np

from evaluation.augment_teacher_cache_demand_history import (
    augment_teacher_cache_with_demand_history,
)


def _config(window: int) -> dict:
    return {
        "num_facilities": 2,
        "production_lead_time": 1,
        "include_supplier_state": False,
        "include_demand_forecast_state": False,
        "include_transfer_pipeline_state": False,
        "include_demand_history_state": True,
        "demand_history_window": window,
    }


def _payload() -> dict:
    states = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0, 0.0, 18.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return {
        "states": states,
        "transition_states": states.copy(),
        "transition_next_states": states.copy(),
        "transition_dones": np.asarray([False, False, True]),
    }


class DemandHistoryCacheTests(unittest.TestCase):
    def test_history_window_changes_causal_rolling_features(self) -> None:
        history_two, summary_two = augment_teacher_cache_with_demand_history(
            _payload(),
            _config(2),
        )
        history_three, summary_three = augment_teacher_cache_with_demand_history(
            _payload(),
            _config(3),
        )

        self.assertEqual(summary_two["window"], 2)
        self.assertEqual(summary_three["window"], 3)
        self.assertEqual(int(history_two["demand_history_window"]), 2)
        self.assertEqual(int(history_three["demand_history_window"]), 3)
        # Facility 0 rolling mean at the third state: (3 + 7) / 2 vs
        # (1 + 3 + 7) / 3. History features follow its four source features.
        self.assertAlmostEqual(float(history_two["states"][2, 4]), 5.0)
        self.assertAlmostEqual(
            float(history_three["states"][2, 4]),
            11.0 / 3.0,
            places=6,
        )

    def test_history_window_must_be_positive(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "demand_history_window must be positive",
        ):
            augment_teacher_cache_with_demand_history(
                _payload(),
                _config(0),
            )

    def test_existing_history_can_be_rewindowed_without_appending_columns(self) -> None:
        history_two, _summary = augment_teacher_cache_with_demand_history(
            _payload(),
            _config(2),
        )

        rewindowed, rewindowed_summary = (
            augment_teacher_cache_with_demand_history(
                history_two,
                _config(3),
            )
        )
        direct, _direct_summary = augment_teacher_cache_with_demand_history(
            _payload(),
            _config(3),
        )

        self.assertEqual(rewindowed_summary["replaced_existing_history"], 1)
        np.testing.assert_allclose(rewindowed["states"], direct["states"])
        np.testing.assert_allclose(
            rewindowed["transition_next_states"],
            direct["transition_next_states"],
        )


if __name__ == "__main__":
    unittest.main()
