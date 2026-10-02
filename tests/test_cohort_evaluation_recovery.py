"""Zero-fit synthetic tests for the bounded evaluation-only continuation."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.rl.cohort_evaluation_recovery import (BLOCKS, ROLES, INTENT, RecoveryCampaign,
    authorization, budget_plan, descriptors, no_optimizer_updates)
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.cohort_evaluation_models import restore_evaluation_owner
from tests.test_cohort_evaluation_models import invented_owner, CohortEvaluationModelTests


def intent():
    return json.loads((Path(__file__).resolve().parents[1] / INTENT).read_text())


def streams():
    return {"environment": {str(b): {"test": list(range(100 * b, 100 * b + 12))} for b in BLOCKS}}


class Owner:
    def state_dict(self):
        return {"weight": 3}


class Backend:
    def session(self, block, owner, seed, **metadata):
        return SimpleNamespace(learner=owner, producer=SimpleNamespace(anchor_config={}), metadata=metadata)


class Recorder:
    def __init__(self, root, path, prefix, config, **metadata):
        self.metadata = metadata
    def finish_prefix(self, *args):
        return {}
    def record_prefix(self, *args):
        pass
    def record_tail(self, *args):
        pass
    def finish(self, run):
        return dict(fake=True, **self.metadata)


class Collection:
    def __init__(self, prefix, **kwargs):
        self.index, self.closed = 0, False
    def step(self, *, before_step):
        before_step()
        self.index += 1
        self.closed = self.index == 63


class RecoveryTests(unittest.TestCase):
    def test_scope_partition_and_no_training_budget(self):
        plan = budget_plan(intent())
        self.assertEqual(plan["limits"], dict(trajectory=12915, clone=0, actor=0, critic=0, seconds=7200))
        self.assertEqual(sum(r["seconds"] for r in plan["phases"].values()), 6720)
        self.assertEqual(len(plan["sections"]), 22)
        for key, value in (("new_evaluations", 206), ("optimizer_calls", 1), ("seconds", 7201), ("attempts", 2)):
            changed = intent() | {key: value}
            with self.assertRaises(PermissionError):
                budget_plan(changed)

    def test_exact_remaining_original_worlds(self):
        rows = [row for b in BLOCKS for role in ROLES for row in descriptors(streams(), b, role)]
        self.assertEqual(len(rows), 205)
        self.assertEqual(len({r[-1] for r in rows}), 205)
        self.assertEqual(rows[0][:3], (60, "own_frozen", 11))
        self.assertTrue(all(r[4] == "test" and r[5] != "sample" for r in rows))
        with self.assertRaises(ValueError):
            descriptors(streams(), 63, "own_frozen")

    def test_complete_fake_serial_chain_and_no_refund(self):
        with TemporaryDirectory() as temp:
            now = [10.0]
            plan = budget_plan(intent())
            budget = DynamicCandidateBudget(Path(temp) / "budget.jsonl", plan, enabled=True, clock=lambda: now[0])
            packet = dict(streams=streams(), scientific_config={}, reused_index=[{"reused": n} for n in range(11)])
            owners = {f"block{b}/graph/{role}": Owner() for b in BLOCKS for role in ROLES[:4]}
            campaign = RecoveryCampaign(temp, packet, budget, Backend(), owners, recorder=Recorder, collection=Collection)
            with patch("builtins.print"):
                for b in BLOCKS:
                    for role in ROLES:
                        budget.begin(f"final_evaluation/block{b}/{role}")
                        rows = descriptors(packet["streams"], b, role)
                        for row in rows:
                            campaign.episode(row)
                        with self.assertRaises(ValueError):
                            campaign.episode(rows[-1])
                        budget.finish()
            self.assertEqual(len(campaign.index), 216)
            self.assertEqual(budget.counts, {"environment": 12915, "optimizer": 0})
            budget.close()
            counts = read_dynamic_ledger(budget.path)["counts"]
            self.assertEqual({k: counts.get(k, 0) for k in budget.counts}, budget.counts)

    def test_owner_deadline_and_update_rejected(self):
        with TemporaryDirectory() as temp:
            clock = [1.0]
            budget = DynamicCandidateBudget(Path(temp) / "ledger", budget_plan(intent()), enabled=True, clock=lambda: clock[0])
            budget.begin("final_evaluation/block60/own_frozen")
            with self.assertRaises(RuntimeError):
                budget.debit_optimizer("actor")
            self.assertEqual(budget.counts["optimizer"], 0)
            budget.close()
        with TemporaryDirectory() as temp:
            clock = [1.0]
            budget = DynamicCandidateBudget(Path(temp) / "ledger", budget_plan(intent()), enabled=True, clock=lambda: clock[0])
            budget.begin("final_evaluation/block60/own_frozen")
            clock[0] += 241
            with self.assertRaises(TimeoutError):
                budget.check()
            budget.close()

    def test_restoration_works_inside_update_guard(self):
        fixture_owner = CohortEvaluationModelTests()
        fixture_owner.setUp()
        self.addCleanup(fixture_owner.doCleanups)
        for role in ROLES[:4]:
            owner = invented_owner(role, completed=role != "own_frozen")
            with no_optimizer_updates():
                restored = restore_evaluation_owner(copy.deepcopy(owner.state_dict()))
                self.assertTrue(restored.evaluation_only)
                param = torch.nn.Parameter(torch.tensor(1.0))
                with self.assertRaises(PermissionError):
                    torch.optim.Adam([param]).step()

    def test_failed_episode_never_enters_index(self):
        class Failing(Collection):
            def step(self, *, before_step):
                super().step(before_step=before_step)
                raise RuntimeError("invented failure")
        with TemporaryDirectory() as temp:
            budget = DynamicCandidateBudget(Path(temp) / "ledger", budget_plan(intent()), enabled=True, clock=lambda: 1.0)
            budget.begin("final_evaluation/block60/own_frozen")
            packet = dict(streams=streams(), scientific_config={}, reused_index=[{}] * 11)
            campaign = RecoveryCampaign(temp, packet, budget, Backend(), {"block60/graph/own_frozen": Owner()},
                recorder=Recorder, collection=Failing)
            row = descriptors(streams(), 60, "own_frozen")[0]
            with self.assertRaises(RuntimeError):
                campaign.episode(row)
            self.assertEqual(len(campaign.index), 11)
            self.assertEqual(budget.counts["environment"], 1)
            with self.assertRaises(ValueError):
                campaign.episode(row)
            budget.close()


if __name__ == "__main__":
    unittest.main()
