"""Exact new entry with fake backends; all real fits/native construction forbidden."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.rl.capacity_native import CapacityWorldTape, CapacityPilotEnv
from src.rl.fixed_budget_capacity_analysis import run_analysis
from src.rl.fixed_budget_capacity_execution import build_runner, stream_manifest
from src.rl.fixed_budget_capacity_resources import PHASES, merged_proposal, numeric_contract, FixedBudgetCapacityBudget
from tests.test_capacity_completion_control import PROPOSAL
from tests.test_patient_constrained_runner import Budget as OldBudget, Env as OldEnv, Learner as OldLearner, capture
from tests.test_capacity_pilot_runner import FakeController


ROOT = Path(__file__).resolve().parents[1]
INHERITED = json.loads((ROOT/"experiments/configs/patient_constrained_capacity_20261003.json").read_text())
STUDY = json.loads((ROOT/"experiments/configs/fixed_budget_capacity_20261003.json").read_text())
P = merged_proposal(PROPOSAL, INHERITED, STUDY)


class Budget(OldBudget):
    def __init__(self):
        self.contract = numeric_contract(STUDY)
        self.limits = self.contract["limits"]
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase, self.chunks = None, {}
        self.phases = {p:dict.fromkeys(self.limits, 0) for p in PHASES}


class Env(OldEnv):
    def __init__(self, *args):
        super().__init__(*args)
        self.pending = [[0.]*4, [0.]*4]

    def step(self, action):
        result = super().step(action)
        result[3]["support_public_receipt"]["applied_hours"] = self.pending.pop(0)
        self.pending.append(list(action.requested_hours))
        return result


class Learner(OldLearner):
    @classmethod
    def from_snapshot(cls, seal, *, before_forward, before_optimizer):
        if seal.state["arm"] is not None:
            assert sum(s.state["arm"] is not None for s in cls.seals) == 3, "test before all final seals"
        obj = cls(before_forward, before_optimizer)
        for k,v in copy.deepcopy(seal.state).items():
            setattr(obj, k, v)
        cls.restorations.append(seal.sha256)
        return obj


class EntryTests(unittest.TestCase):
    def test_full_one_arm_entry_raw_readout_and_no_hidden_work(self):
        Learner.seals, Learner.restorations = [], []
        budget = Budget()
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(CapacityPilotEnv, "__init__", side_effect=AssertionError("native forbidden")), \
                patch.object(torch.optim.Adam, "step", side_effect=AssertionError("optimizer forbidden")):
            runner = build_runner(temp, P, budget, admission=dict(verified=True, scope="fixed-budget-capacity-v1"),
                learner_factory=Learner, env_factory=Env, controller_factory=FakeController,
                tape_factory=lambda p,w:CapacityWorldTape(w, (), (), (), 20 if w["condition"]==1 else None),
                capture=capture, features=lambda *a:np.zeros((4,2), dtype=np.float32),
                base_control=lambda *a,**k:np.zeros(16))
            with patch.object(runner, "_save_state"):
                result = runner.run()
            self.assertEqual(result["trajectories"], 180)
            self.assertEqual(budget.counts, budget.limits)
            self.assertEqual(budget.counts["total_optimizer_steps"], 6720)
            self.assertEqual(len(Learner.seals), 6)
            self.assertEqual(len(Learner.restorations), 39)
            for seal in Learner.seals:
                terminal = [r for r in seal.state["reference"] if r.done]
                self.assertEqual(len(terminal), 12)
                self.assertTrue(all(r.total_losses==1 and r.full_cost==17 for r in terminal))
            analysis = run_analysis(temp, STUDY)
            self.assertTrue(analysis["complete"])
            self.assertFalse(analysis["patient_preserving_training_signal"])
            self.assertEqual(analysis["raw_rows"], 11520)
            self.assertEqual(analysis["evaluation_trajectories"], 108)
            self.assertEqual(set(analysis["contrasts"]), {"fixed_allocation_reference", "id_mpc"})
            self.assertTrue(all(r["total_applied_hours_including_tail"]==384 for r in analysis["trajectories"]))
            path = next((Path(temp)/"summaries").glob("evaluation*.json"))
            s = json.loads(path.read_text()); s["cost"] += 1; path.write_text(json.dumps(s))
            with self.assertRaises(ValueError):
                run_analysis(temp, STUDY)
            with self.assertRaises(RuntimeError):
                runner.run()

    def test_admission_streams_and_arithmetic(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(PermissionError):
                build_runner(Path(temp)/"none", P, None, admission={})
            self.assertFalse((Path(temp)/"none").exists())
        streams = stream_manifest(P)
        self.assertEqual(len(streams["rows"]), 72)
        self.assertEqual(len(streams["allocations"]), len(set(streams["allocations"])))
        self.assertEqual(streams["rows"][0]["world"]["seed"], 62800000)
        bad = copy.deepcopy(STUDY); bad["budget"]["forwards"] += 1
        with self.assertRaises(ValueError):
            numeric_contract(bad)

    def test_no_evaluation_updates_no_refund_and_job_cap(self):
        now = [0.]
        with tempfile.TemporaryDirectory() as temp:
            b = FixedBudgetCapacityBudget(Path(temp)/"launcher", P, clock=lambda:now[0])
            b.enter(PHASES[2], PHASES[2])
            with self.assertRaises(RuntimeError): b.debit({"actor_optimizer_steps":1})
            self.assertEqual(b.counts["actor_optimizer_steps"], 0)
            with self.assertRaises(RuntimeError): b.debit({"native_steps":1})
            b.close()
        with tempfile.TemporaryDirectory() as temp:
            b = FixedBudgetCapacityBudget(Path(temp)/"launcher", P, clock=lambda:now[0])
            b.enter(PHASES[0], PHASES[0]); b.job("b0", "reference_block")
            b.debit({"native_steps":1}); now[0] = 201.
            with self.assertRaises(TimeoutError): b.check()
            self.assertEqual(b.counts["native_steps"], 1)
            b.close()


if __name__ == "__main__":
    unittest.main()
