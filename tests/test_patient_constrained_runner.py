"""Real entry and persisted schemas with fake worlds and forbidden real updates."""

import copy
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.rl.capacity_native import CapacityWorldTape, CapacityPilotEnv
from src.rl.patient_constrained_resources import PHASES, merged_proposal, numeric_contract, PatientConstrainedBudget
from src.rl.patient_constrained_execution import build_runner, stream_manifest
from src.rl.patient_constrained_analysis import run_analysis
from tests.test_capacity_completion_control import PROPOSAL, view, patient
from tests.test_capacity_pilot_runner import FakeBudget, FakeEnv, FakeController


ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT/"experiments/configs/patient_constrained_capacity_20261003.json").read_text())
P = merged_proposal(PROPOSAL,STUDY)


class Budget(FakeBudget):
    def __init__(self):
        self.contract = numeric_contract(STUDY)
        self.limits = self.contract["limits"]
        self.counts = dict.fromkeys(self.limits,0)
        self.phase, self.chunks = None, {}
        self.phases = {p:dict.fromkeys(self.limits,0) for p in PHASES}
    def debit(self, charges):
        for k,v in charges.items():
            assert 0 < v and self.phases[self.phase][k]+v <= self.contract["phase_limits"][self.phase][k], k
        super().debit(charges)
        for k,v in charges.items(): self.phases[self.phase][k] += v
    def job(self, *args): pass


class Env(FakeEnv):
    def __init__(self,p,tape,before_native):
        super().__init__(p,tape,before_native)
        self.seed=tape.world["seed"]
    def settlement(self):
        return dict(settled=True,enrolled=1,lost=1,delivered=0,
                    live_ids=[],pending_obligations=0,resource_conservation=True)


def capture(env):
    return view(env.t,(patient(f"world{env.seed}:patient0",status="lost" if env.t>=50 else "waiting"),))


@dataclass
class Seal:
    payload: bytes
    sha256: str
    state: dict


class Learner:
    seals, restorations = [], []
    @classmethod
    def from_proposal(cls,p,**kw): return cls(kw["before_forward"],kw["before_optimizer"])
    def __init__(self,forward,optimizer):
        self.forward,self.optimizer=forward,optimizer
        self.reference,self.training,self.counts,self.arm=[],[],{},None
    def add_reference(self,row): self.reference.append(row)
    def add_training(self,row): self.training.append(row)
    def set_arm(self,arm):
        assert self.arm is None and not self.training
        self.arm=arm
    def act(self,features,**kw):
        self.forward("behavior",1)
        return np.full(4,2.),np.full(4,2.)
    def fit(self,mode,condition):
        assert any(r.condition==condition for r in self.reference)
        if mode=="training": assert any(r.condition==condition for r in self.training)
        for _ in range(5 if mode=="warmup" else 8): self.forward("fake",64)
        for name in (("reward_critic","loss_critic") if mode=="warmup" else ("reward_critic","loss_critic","actor")):
            self.optimizer(name,64)
        self.counts[mode]=self.counts.get(mode,0)+1
        return dict(fake_not_optimizer=True,condition=condition)
    def warmup_step(self,c): return self.fit("warmup",c)
    def training_step(self,c): return self.fit("training",c)
    def update_multiplier(self,c,loss,reference):
        assert self.arm=="constrained" and loss==reference==1
        return dict(condition=c,learned=loss,reference=reference)
    def state_dict(self):
        return copy.deepcopy(dict(reference=self.reference,training=self.training,counts=self.counts,arm=self.arm))
    def snapshot(self):
        state=self.state_dict(); raw=pickle.dumps(state)
        seal=Seal(raw,hashlib.sha256(raw).hexdigest(),state)
        self.seals.append(seal)
        return seal
    @classmethod
    def from_snapshot(cls,seal,*,before_forward,before_optimizer):
        if seal.state["arm"] is not None:
            assert sum(s.state["arm"] is not None for s in cls.seals)==6,"test before all final seals"
        obj=cls(before_forward,before_optimizer)
        for k,v in copy.deepcopy(seal.state).items(): setattr(obj,k,v)
        cls.restorations.append(seal.sha256)
        return obj


class EntryTests(unittest.TestCase):
    def test_full_exact_schedule_tail_losses_and_reader_without_science(self):
        Learner.seals,Learner.restorations=[],[]
        budget=Budget()
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(CapacityPilotEnv,"__init__",side_effect=AssertionError("native forbidden")), \
                patch.object(torch.optim.Adam,"step",side_effect=AssertionError("optimizer forbidden")):
            runner=build_runner(temp,P,budget,admission=dict(verified=True,scope="patient-constrained-capacity-v1"),
                learner_factory=Learner,env_factory=Env,controller_factory=FakeController,
                tape_factory=lambda p,w:CapacityWorldTape(w,(),(),(),20 if w["condition"]==1 else None),
                capture=capture,features=lambda *a:np.zeros((4,2),dtype=np.float32),
                base_control=lambda *a,**k:np.zeros(16))
            with patch.object(runner,"_save_state"):
                result=runner.run()
            self.assertEqual(result["trajectories"],252)
            self.assertEqual(budget.counts,budget.limits)
            self.assertEqual(len(Learner.seals),9)
            self.assertEqual(len(Learner.restorations),78)
            for seal in Learner.seals:
                terminal=[r for r in seal.state["reference"] if r.done]
                self.assertEqual(len(terminal),12)
                self.assertTrue(all(r.total_losses==1 and r.full_cost==17 for r in terminal))
            analysis=run_analysis(temp,STUDY)
            self.assertTrue(analysis["complete"])
            self.assertFalse(analysis["patient_preserving_training_signal"])
            self.assertEqual(analysis["raw_rows"],16128)
            self.assertEqual(analysis["contrasts"]["fixed_allocation_reference"]["1"]["means"]["savings"],0.)
            path=next((Path(temp)/"summaries").glob("evaluation*.json"))
            s=json.loads(path.read_text()); s["cost"]+=1; path.write_text(json.dumps(s))
            with self.assertRaises(ValueError): run_analysis(temp,STUDY)
            with self.assertRaises(RuntimeError): runner.run()

    def test_no_admission_and_disjoint_streams(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(PermissionError):
                build_runner(Path(temp)/"none",P,None,admission={})
            self.assertFalse((Path(temp)/"none").exists())
        streams=stream_manifest(P)
        self.assertEqual(len(streams["rows"]),72)
        self.assertEqual(len(streams["allocations"]),len(set(streams["allocations"])))

    def test_zero_phase_allowance_no_refund_and_job_wall_cap(self):
        now=[0.]
        with tempfile.TemporaryDirectory() as temp:
            b=PatientConstrainedBudget(Path(temp)/"launcher",P,clock=lambda:now[0])
            b.enter(PHASES[2],PHASES[2])
            with self.assertRaises(RuntimeError): b.debit({"actor_optimizer_steps":1})
            self.assertEqual(b.counts["actor_optimizer_steps"],0)
            with self.assertRaises(RuntimeError): b.debit({"native_steps":1})
            b.close()
        with tempfile.TemporaryDirectory() as temp:
            b=PatientConstrainedBudget(Path(temp)/"launcher",P,clock=lambda:now[0])
            b.enter(PHASES[0],PHASES[0]); b.job("b0","reference_block")
            b.debit({"native_steps":1}); now[0]=401.
            with self.assertRaises(TimeoutError): b.check()
            self.assertEqual(b.counts["native_steps"],1)
            b.close()


if __name__=="__main__": unittest.main()
