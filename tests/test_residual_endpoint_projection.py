"""Tests for conservative endpoint-sparse transfer projection."""

from __future__ import annotations

import unittest

import numpy as np

from src.rl.residual_endpoint_projection import ResidualEndpointProjection

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


class ResidualEndpointProjectionTests(unittest.TestCase):
    def projection(self, top_k: int = 1) -> ResidualEndpointProjection:
        return ResidualEndpointProjection(
            num_facilities=4,
            action_dim=16,
            settings={
                "enabled": True,
                "groups": [
                    "reagent_transfer",
                    "capacity_transfer",
                ],
                "max_endpoints_per_side": top_k,
            },
        )

    def test_numpy_projection_is_sparse_balanced_and_non_amplifying(self) -> None:
        action = np.zeros(16, dtype=np.float32)
        action[4:8] = np.asarray(
            [-0.7, -0.2, 0.6, 0.3],
            dtype=np.float32,
        )

        projected = self.projection(top_k=1).apply_numpy(action)
        reagent = projected[4:8]

        self.assertEqual(int(np.count_nonzero(reagent)), 2)
        self.assertAlmostEqual(float(reagent.sum()), 0.0, places=6)
        self.assertLessEqual(
            float(np.abs(reagent).sum()),
            float(np.abs(action[4:8]).sum()),
        )
        self.assertLess(float(reagent[0]), 0.0)
        self.assertGreater(float(reagent[2]), 0.0)

    def test_projection_leaves_unconfigured_groups_unchanged(self) -> None:
        action = np.linspace(-0.8, 0.8, 16, dtype=np.float32)

        projected = self.projection(top_k=1).apply_numpy(action)

        np.testing.assert_allclose(projected[:4], action[:4])
        np.testing.assert_allclose(projected[12:], action[12:])

    def test_minimum_magnitude_can_suppress_a_group(self) -> None:
        projection = ResidualEndpointProjection(
            num_facilities=4,
            action_dim=16,
            settings={
                "enabled": True,
                "groups": ["reagent_transfer"],
                "max_endpoints_per_side": 1,
                "min_abs": 0.5,
            },
        )
        action = np.zeros(16, dtype=np.float32)
        action[4:8] = np.asarray(
            [-0.4, -0.2, 0.3, 0.1],
            dtype=np.float32,
        )

        projected = projection.apply_numpy(action)

        np.testing.assert_allclose(projected[4:8], 0.0)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_tensor_projection_matches_numpy_projection(self) -> None:
        actions = np.zeros((2, 16), dtype=np.float32)
        actions[0, 4:8] = [-0.7, -0.2, 0.6, 0.3]
        actions[1, 8:12] = [-0.1, -0.5, 0.2, 0.9]
        projection = self.projection(top_k=1)

        tensor_result = projection.apply_tensor(
            torch.as_tensor(actions)
        ).numpy()
        numpy_result = projection.apply_numpy(actions)

        np.testing.assert_allclose(
            tensor_result,
            numpy_result,
            atol=1e-7,
        )
