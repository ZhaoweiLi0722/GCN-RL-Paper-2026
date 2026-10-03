"""Artificial receipts/tensors only; no simulator, fitted model or optimizer."""

from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import patch

from src.env.service_effort_mechanics import ServiceEffortConfig
from src.rl.capacity_adaptation_interface import (
    PublicCapacityHistory, PublicCapacityReceipt, project_effort, project_effort_tensor,
)
from src.rl.networks import torch


def limits(lead=1):
    return ServiceEffortConfig((1.0, 2.0, 1.5), 2.0, lead, 2.0, 0.5, 0.25)


def history(lead=1):
    return PublicCapacityHistory(limits(lead), site_ids=("A", "B", "C"), window=2, max_epochs=4)


def receipt(h, raw, completed=(0, 0, 0), eligible=(2, 2, 2), ordinary=(0, 0, 0)):
    c = h.config
    committed = project_effort(raw, c)
    previous = h.receipts[-1].committed_hours if h.receipts else (0, 0, 0)
    return PublicCapacityReceipt(
        h.next_decision_epoch, h.next_decision_epoch + 1, raw, committed,
        h.pending_hours[0], ordinary, eligible, completed,
        sum(c.hourly_cost * x + c.quadratic_cost * x * x for x in committed),
        c.switching_cost * sum(abs(x - y) for x, y in zip(committed, previous)),
    )


class CapacityInterfaceTests(unittest.TestCase):
    def test_projection_retains_all_feasible_requests_and_unused_budget(self):
        for raw in ((0, 0, 0), (0.123456789, 0.2, 0.3), (1, 0, 1)):
            self.assertEqual(project_effort(raw, limits()), raw)

    def test_shared_and_site_caps(self):
        self.assertEqual(project_effort((5, 0, 0), limits()), (1, 0, 0))
        actual = project_effort((1, 2, 1), limits())
        self.assertEqual(actual, (0.5, 1.0, 0.5))

    def test_site_permutation_equivariance(self):
        c, raw, order = limits(), (0.4, 1.8, 0.9), (2, 0, 1)
        moved = replace(c, site_hour_caps=tuple(c.site_hour_caps[i] for i in order))
        result = project_effort(raw, c)
        self.assertEqual(project_effort(tuple(raw[i] for i in order), moved),
                         tuple(result[i] for i in order))

    def test_request_rejects_nonfinite_negative_boolean_and_wrong_size(self):
        for raw in ((-1, 0, 0), (float('inf'), 0, 0), (float('nan'), 0, 0),
                    (True, 0, 0), (0, 0), ('1', 0, 0)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                project_effort(raw, limits())

    def test_no_future_information_and_masked_history(self):
        h = history()
        self.assertEqual(h.node_history(decision_epoch=0), (((0.0,) * 6,) * 2,) * 3)
        r = receipt(h, (0.25, 0.5, 0.75))
        with self.assertRaises(ValueError):
            h.observe(r, decision_epoch=0)
        h.observe(r, decision_epoch=1)
        self.assertEqual(h.node_history(decision_epoch=1)[0][-1], (0.25, 0, 0, 2, 0, 1))
        with self.assertRaises(ValueError):
            h.node_history(decision_epoch=2)
        forbidden = {"efficiency", "remaining_work", "response", "future", "uniforms"}
        self.assertFalse(forbidden.intersection(asdict(r)))

    def test_delay_and_paid_idle_effort(self):
        h = history(lead=2)
        first = receipt(h, (0.5, 0, 0), eligible=(0, 0, 0))
        self.assertGreater(first.labor_cost, 0)
        h.observe(first, decision_epoch=1)
        self.assertEqual(h.pending_hours[0], (0, 0, 0))
        h.observe(receipt(h, (0, 0, 0)), decision_epoch=2)
        self.assertEqual(h.pending_hours[0], (0.5, 0, 0))
        h.observe(receipt(h, (0, 0, 0), completed=(1, 0, 0)), decision_epoch=3)

    def test_invalid_receipt_atomic_and_duplicate_rejected(self):
        h = history()
        r = receipt(h, (0.2, 0.3, 0.4))
        bad = [replace(r, applied_hours=(0.2, 0, 0)),
               replace(r, committed_hours=(0.2, 0.3, 0.5)),
               replace(r, labor_cost=0), replace(r, switching_cost=float('nan')),
               replace(r, known_at=2), replace(r, completed_jobs=(1, 0, 0)),
               replace(r, eligible_jobs=(0.5, 2, 2)), replace(r, eligible_jobs=(True, 2, 2))]
        before = h.snapshot()
        for invalid in bad:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                h.observe(invalid, decision_epoch=1)
            self.assertEqual(h.snapshot(), before)
        h.observe(r, decision_epoch=1)
        with self.assertRaises(ValueError):
            h.observe(r, decision_epoch=1)
        with self.assertRaises(TypeError):
            h.observe(asdict(r), decision_epoch=2)

    def test_zero_completions_not_relabelled_as_zero_efficiency(self):
        h = history()
        h.observe(receipt(h, (0.5, 0.5, 0)), decision_epoch=1)
        h.observe(receipt(h, (0, 0, 0)), decision_epoch=2)
        row = h.node_history(decision_epoch=2)[0][-1]
        self.assertEqual(row, (0, 0.5, 0, 2, 0, 1))
        self.assertFalse(hasattr(h, 'estimated_efficiency'))

    def test_base_staffing_is_not_fictitious_flexible_effort(self):
        h = history()
        r = receipt(h, (0, 0, 0), completed=(1, 0, 0), ordinary=(1, 1, 1))
        h.observe(r, decision_epoch=1)
        self.assertEqual(h.receipts[0].applied_hours, (0, 0, 0))
        self.assertEqual(h.receipts[0].labor_cost, 0)
        self.assertEqual(h.node_history(decision_epoch=1)[0][-1], (0, 0, 1, 2, 1, 1))

    def test_json_restore_same_history_pipeline_and_next_receipt(self):
        h = history(lead=2)
        h.observe(receipt(h, (0.5, 0.5, 0)), decision_epoch=1)
        h.observe(receipt(h, (0.1, 0, 0.3)), decision_epoch=2)
        snapshot = json.loads(json.dumps(h.snapshot()))
        other = PublicCapacityHistory.restore(snapshot, config=limits(2), site_ids=h.site_ids,
                                              window=2, max_epochs=4)
        self.assertEqual(other.snapshot(), h.snapshot())
        self.assertEqual(other.pending_hours, h.pending_hours)
        next_receipt = receipt(h, (0, 0, 0), completed=(1, 0, 0))
        for target in (h, other):
            target.observe(next_receipt, decision_epoch=3)
        self.assertEqual(h.node_history(decision_epoch=3), other.node_history(decision_epoch=3))
        with self.assertRaises(ValueError):
            PublicCapacityHistory.restore(snapshot, config=limits(1), site_ids=h.site_ids,
                                          window=2, max_epochs=4)
        snapshot['hidden_efficiency'] = [1, 1, 1]
        with self.assertRaises(ValueError):
            PublicCapacityHistory.restore(snapshot, config=limits(2), site_ids=h.site_ids,
                                          window=2, max_epochs=4)

    def test_history_cap_and_no_alias_mutation(self):
        h = history()
        raw = [0.5, 0, 0]
        r = receipt(h, raw)
        raw[0] = 1
        self.assertEqual(r.raw_requested_hours, (0.5, 0, 0))
        for t in range(4):
            h.observe(receipt(h, (0.5, 0, 0)), decision_epoch=t + 1)
        with self.assertRaises(ValueError):
            h.observe(receipt(h, (0.5, 0, 0)), decision_epoch=5)

    @unittest.skipIf(torch is None, 'torch not installed')
    def test_tensor_parity_and_gradient_no_optimizer(self):
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('no optimizer permitted')):
            raw = torch.tensor([[0.2, 0.3, 0.1], [1, 2, 1]], dtype=torch.float64, requires_grad=True)
            projected = project_effort_tensor(raw, limits())
            self.assertEqual(projected.tolist(), [list(project_effort(row, limits())) for row in raw.tolist()])
            grad = torch.autograd.grad(projected[0].sum(), raw)[0]
            self.assertEqual(grad[0].tolist(), [1, 1, 1])
            self.assertTrue(torch.isfinite(grad).all().item())
            self.assertTrue(torch.autograd.gradcheck(lambda x: project_effort_tensor(x, limits()),
                                                    (raw[:1].detach().requires_grad_(),)))

    @unittest.skipIf(torch is None, 'torch not installed')
    def test_tensor_zero_purchase_and_invalid_inputs(self):
        z = torch.zeros((2, 3), dtype=torch.float32, requires_grad=True)
        y = project_effort_tensor(z, limits())
        self.assertEqual(y.sum().item(), 0)
        self.assertTrue(torch.isfinite(torch.autograd.grad(y.sum(), z)[0]).all().item())
        for x in (torch.ones(2), torch.ones(3, dtype=torch.int64),
                  torch.tensor([float('nan'), 0, 0]), torch.tensor([-1., 0, 0])):
            with self.assertRaises(ValueError):
                project_effort_tensor(x, limits())


if __name__ == '__main__':
    unittest.main()
