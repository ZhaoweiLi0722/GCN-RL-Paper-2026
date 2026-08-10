import unittest

import numpy as np
import torch

from src.rl.critic_advantage import (
    teacher_advantage_loss,
    teacher_advantage_support_mask,
)


class CriticAdvantageLossTests(unittest.TestCase):
    def test_support_mask_keeps_anchor_and_allowed_winners(self) -> None:
        mask = teacher_advantage_support_mask(
            np.asarray(
                [
                    [0.0, 2.0, 1.0],
                    [0.0, 1.0, 2.0],
                    [0.0, -1.0, -2.0],
                ]
            ),
            np.ones((3, 3), dtype=bool),
            np.asarray(["anchor", "specimen_transfer", "reagent_transfer"]),
            allowed_option_groups=("specimen_transfer",),
        )
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_default_loss_is_exact_mse(self) -> None:
        predictions = torch.tensor([[0.0], [2.0]])
        targets = torch.tensor([[1.0], [0.0]])
        total, regression, margin, pairwise = teacher_advantage_loss(
            predictions,
            targets,
        )
        torch.testing.assert_close(total, torch.tensor(2.5))
        torch.testing.assert_close(regression, torch.tensor(2.5))
        torch.testing.assert_close(margin, torch.tensor(0.0))
        torch.testing.assert_close(pairwise, torch.tensor(0.0))

    def test_margin_penalizes_only_material_positive_shortfall(self) -> None:
        predictions = torch.tensor([[0.0], [0.0], [-0.001]])
        targets = torch.tensor([[0.0001], [0.002], [0.0]])
        total, regression, margin, pairwise = teacher_advantage_loss(
            predictions,
            targets,
            positive_margin=0.0005,
            positive_margin_weight=2.0,
        )
        expected_margin = torch.tensor(0.0005**2)
        torch.testing.assert_close(margin, expected_margin)
        torch.testing.assert_close(total, regression + 2.0 * margin)
        torch.testing.assert_close(pairwise, torch.tensor(0.0))

    def test_margin_has_directional_gradient_below_threshold(self) -> None:
        predictions = torch.tensor([[-0.001]], requires_grad=True)
        targets = torch.tensor([[0.002]])
        total, _, _, _ = teacher_advantage_loss(
            predictions,
            targets,
            positive_margin=0.0005,
            positive_margin_weight=1.0,
        )
        total.backward()
        self.assertLess(float(predictions.grad.item()), 0.0)

    def test_pairwise_loss_matches_advantage_differences(self) -> None:
        predictions = torch.tensor([[0.0], [1.0], [1.0]])
        targets = torch.tensor([[0.0], [1.0], [2.0]])
        total, regression, margin, pairwise = teacher_advantage_loss(
            predictions,
            targets,
            pairwise_difference_weight=1.0,
        )
        self.assertGreater(float(pairwise.item()), 0.0)
        torch.testing.assert_close(total, regression + pairwise)
        torch.testing.assert_close(margin, torch.tensor(0.0))


if __name__ == "__main__":
    unittest.main()
