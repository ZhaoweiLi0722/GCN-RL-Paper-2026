from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.option_teacher_config import (
    explicit_options_from_teacher_cache,
    select_restricted_teacher_options,
)


class OptionTeacherConfigTests(unittest.TestCase):
    def test_selects_best_allowed_teacher_option(self) -> None:
        indices, labels, advantages = select_restricted_teacher_options(
            np.asarray(
                ["anchor", "reagent_transfer", "replenishment_uniform"],
                dtype="U32",
            ),
            np.asarray(
                [
                    [0.0, 2_000_000.0, 4_000_000.0],
                    [0.0, 400_000.0, 5_000_000.0],
                ],
                dtype=np.float32,
            ),
            np.ones((2, 3), dtype=bool),
            allowed_groups=("reagent_transfer",),
            min_advantage=500_000.0,
        )

        np.testing.assert_array_equal(indices, np.asarray([0, 1]))
        np.testing.assert_array_equal(labels, np.asarray([1, 0]))
        np.testing.assert_allclose(
            advantages,
            np.asarray([2_000_000.0, 0.0]),
        )

    def test_requires_anchor_as_first_option(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "teacher.npz"
            np.savez_compressed(
                cache,
                option_groups=np.asarray(
                    ["reagent_transfer", "anchor"],
                    dtype="U32",
                ),
                option_epsilons=np.asarray([0.32, 0.0]),
                option_signs=np.asarray([1.0, 0.0]),
            )
            with self.assertRaisesRegex(ValueError, "option zero must be anchor"):
                explicit_options_from_teacher_cache(cache)

    def test_rejects_duplicate_correction_option(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "teacher.npz"
            np.savez_compressed(
                cache,
                option_groups=np.asarray(
                    ["anchor", "reagent_transfer", "reagent_transfer"],
                    dtype="U32",
                ),
                option_epsilons=np.asarray([0.0, 0.32, 0.32]),
                option_signs=np.asarray([0.0, 1.0, 1.0]),
            )
            with self.assertRaisesRegex(ValueError, "duplicate option"):
                explicit_options_from_teacher_cache(cache)


if __name__ == "__main__":
    unittest.main()
