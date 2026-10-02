"""Production entry wiring with authored models, fake updates and no patient calls."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.rl.conservative_cohort_plan import budget_plan
from src.rl.conservative_cohort_training import fit_round
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.prospective_ddpg_kernel import state_digest
from tests.test_conservative_cohort_plan import config


class Prototype:
    def __init__(self, value):
        self.value = value
    def snapshot_sha256(self):
        return state_digest(self.value)


class FakeActor:
    def __init__(self, prototype, contract, settings, examples, **kw):
        self.value, self.steps, self.round = prototype.value, 0, kw["round_index"]
        self.q = kw["reference_logits"]
        assert len(self.q) == len(examples) == 6
        assert all(q.dtype == torch.float64 for q in self.q)
    def state_dict(self):
        return dict(engineering_fixture=True, value=self.value, steps=self.steps, round=self.round, q=self.q)
    def update(self, *, before_optimizer_step, before_compute):
        before_compute()
        before_optimizer_step("actor")
        self.steps += 1
        self.value += 1
        return dict(step=self.steps, critic_optimizer_steps=0, engineering_fixture=True)


class FitTests(unittest.TestCase):
    def test_two_explicit_rounds_continue_weights_but_reset_steps(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(torch.optim.Adam, "step", side_effect=AssertionError("real fitting prohibited")) as real:
            root = Path(directory)
            cfg = config()
            budget = DynamicCandidateBudget(root / "budget.jsonl", budget_plan(cfg), enabled=True, clock=lambda: 0.)
            calls, start = [], Prototype(7)
            try:
                for rnd in (0, 1):
                    budget.begin(f"paired_actor/round{rnd}/block60")
                    owner, record = fit_round(root, cfg, 60, rnd, "paired_cost", start, "fake", 
                        dict(block=60, round=rnd), budget, admit=calls.append, owner_type=FakeActor,
                        decode=lambda *args: (tuple(range(6)), "a" * 64),
                        logits=lambda prototype, example: torch.tensor([prototype.value, example], dtype=torch.float32))
                    self.assertEqual(owner.steps, 64)
                    self.assertEqual(owner.round, rnd + 1)
                    self.assertTrue((root / record["path"]).is_file())
                    start = Prototype(owner.value)
                    budget.finish()
                self.assertEqual(start.value, 135)
                self.assertEqual(calls.count("round_start_logits"), 12)
                self.assertEqual(budget.counts, dict(environment=0, optimizer=128))
                self.assertEqual(budget.owner_counts.get("global:critic", 0), 0)
            finally:
                budget.close()
            real.assert_not_called()

    def test_wrong_round_rejected_before_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config()
            budget = DynamicCandidateBudget(Path(directory) / "b.jsonl", budget_plan(cfg), enabled=True, clock=lambda: 0.)
            try:
                budget.begin("paired_actor/round0/block60")
                with self.assertRaises(ValueError):
                    fit_round(directory, cfg, 60, 0, "paired_cost", Prototype(0), "fake",
                        dict(block=60, round=1), budget, admit=lambda *a: None,
                        decode=lambda *a: self.fail("must reject before decoding"))
                self.assertEqual(budget.counts, dict(environment=0, optimizer=0))
            finally:
                budget.close()

    def test_failure_keeps_attempt_and_saves_boundary(self):
        class FailedActor(FakeActor):
            def update(self, **kw):
                kw["before_optimizer_step"]("actor")
                raise RuntimeError("invented post-debit failure")
        with tempfile.TemporaryDirectory() as directory:
            cfg = config()
            budget = DynamicCandidateBudget(Path(directory) / "b.jsonl", budget_plan(cfg), enabled=True, clock=lambda: 0.)
            try:
                budget.begin("paired_actor/round0/block60")
                with self.assertRaises(RuntimeError):
                    fit_round(directory, cfg, 60, 0, "paired_cost", Prototype(0), "fake",
                        dict(block=60, round=0), budget, admit=lambda *a: None, owner_type=FailedActor,
                        decode=lambda *a: (tuple(range(6)), "a" * 64),
                        logits=lambda *a: torch.zeros(2))
                self.assertEqual(budget.counts["optimizer"], 1)
                self.assertTrue((Path(directory) / "payload/models/round0/block60/paired_cost/failure.pt").is_file())
            finally:
                budget.close()


if __name__ == "__main__":
    unittest.main()
