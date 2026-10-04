"""Actual new entry and full fixed schedule on fake backends only."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from src.rl.capacity_native import CapacityWorldTape
from src.rl.capacity_planner_tail_execution import build_runner, stream_manifest
from src.rl.capacity_planner_tail_analysis import run_analysis
from src.rl.capacity_planner_tail_learner import CapacityPlannerTailLearner
from src.rl.capacity_planner_tail_resources import SCOPE, numeric_contract, merged_proposal
from tests.test_capacity_planner_tail_design import STUDY
from tests.test_capacity_pilot_runner import FakeController
from tests.test_capacity_value_mpc import (
    ZeroUpdateCase, ValueBudget, FakeValue, ReceiptEnv, FakePublic, FakeCommon,
    fake_features, PROPOSAL,
)


class Budget(ValueBudget):
    def __init__(self):
        self.contract=numeric_contract(STUDY)
        self.limits=self.contract["limits"]
        self.counts=dict.fromkeys(self.limits,0)
        self.phase_counts={p:dict.fromkeys(self.limits,0) for p in self.contract["phase_limits"]}
        self.phase,self.chunks,self.jobs=None,{},[]


class Value(FakeValue):
    @classmethod
    def fork_weights(cls,raw,config,*,seed,expected_sha256,before_forward,before_optimizer):
        assert hashlib.sha256(raw).hexdigest()==expected_sha256
        out=cls(config,seed=seed,feature_dim=31,before_forward=before_forward,before_optimizer=before_optimizer)
        out.ancestor_sha256=expected_sha256
        return out

    def admit_observed(self,data):
        self.forward("value",64)
        self.pending=dict(targets=np.zeros(64),source_hashes=["a"*64],completed=0)

    def admit_tails(self,tails):
        n=sum(len(t["costs"]) for t in tails)
        if self.config["method"]=="planner_tail_td":
            for i in range(0,n,64):
                self.forward("value",min(64,n-i))
        self.pending=dict(targets=np.zeros(n),source_hashes=["a"*64]*96,completed=0)

    def state_dict(self):
        return {**super().state_dict(),"ancestor_sha256":getattr(self,"ancestor_sha256","a"*64)}

    @classmethod
    def from_bytes(cls,raw,*,before_forward,before_optimizer):
        out=super().from_bytes(raw,before_forward=before_forward,before_optimizer=before_optimizer)
        out.ancestor_sha256=json.loads(raw)["ancestor_sha256"]
        return out


class Controller(FakeController):
    def __init__(self,*args,value=None,planning_horizon=8,capture_epochs=(),
                 before_tail_clone=None,before_tail_step=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.value,self.horizon,self.capture=value,planning_horizon,capture_epochs
        self.clone,self.step=before_tail_clone,before_tail_step
        self.last_plan,self.last_training_tails={},[]

    def act(self,view,role,*,before_query,before_filter):
        self.last_training_tails=[]
        before_query(dict(model_epochs=48*self.horizon))
        if self.value is not None:
            self.value.residuals(np.zeros((48,4,31),dtype=np.float32))
        if view.common.epoch in self.capture:
            start=view.common.epoch+8
            for k in range(48):
                self.clone(dict(start_epoch=start))
                for t in range(start,64):
                    self.step(dict(epoch=t))
                self.last_training_tails.append(dict(start_epoch=start,epochs=list(range(start,65)),
                    features=np.zeros((65-start,4,31),dtype=np.float32),heuristics=np.zeros(65-start),
                    costs=np.ones(64-start),candidate=k//3,quantile=[.1,.5,.9][k%3],
                    decision_epoch=view.common.epoch,prefix_cost=8.))
        self.last_plan=dict(fake_predictor=True,planning_horizon=self.horizon)
        return np.full(4,2.)


class RawReceiptEnv(ReceiptEnv):
    def step(self, action):
        observation, reward, done, info = super().step(action)
        info["support_public_receipt"].update(epoch=self.t - 1, known_at=self.t,
                                             raw_requested_hours=list(action.requested_hours))
        return observation, reward, done, info


class RunnerTests(ZeroUpdateCase):
    def test_actual_entry_full480world_schedule_no_real_science(self):
        study=copy.deepcopy(STUDY)
        budget=Budget()
        with tempfile.TemporaryDirectory() as tmp,patch("os.fsync"), \
                patch.object(CapacityPlannerTailLearner,"__init__",side_effect=AssertionError("real learner forbidden")):
            work=Path(tmp)
            initial=work/"initial"
            initial.mkdir()
            study["initial_models"]["root"]="initial"
            for block,name in enumerate(study["initial_models"]["names"]):
                raw=json.dumps(dict(config=dict(STUDY["value"],method="observed_td"),
                    seed=block,feature_dim=31,updates=1536,ancestor_sha256="a"*64),sort_keys=True).encode()
                (initial/name).write_bytes(raw)
                study["initial_models"]["sha256"][block]=hashlib.sha256(raw).hexdigest()
            p=json.loads(json.dumps(merged_proposal(PROPOSAL,study)))
            manifest=stream_manifest(p)
            self.assertEqual(len(manifest["rows"]),180)
            root=work/"payload"
            root.mkdir()
            runner=build_runner(root,p,budget,admission=dict(verified=True,scope=SCOPE),workspace=work,
                value_factory=Value,frozen_factory=Value,controller_factory=Controller,env_factory=RawReceiptEnv,
                tape_factory=lambda p,w:CapacityWorldTape(w,(),(),(),None),
                capture=lambda env:FakePublic(FakeCommon(env.t)),feature_adapter=fake_features,
                base_control=lambda *a,**k:np.zeros(16))
            status=runner.status
            tests=[]
            def record(event,**values):
                if event=="trajectory_started" and runner.active["phase"]=="evaluation":
                    self.assertEqual(len(runner.final),15)
                    self.assertEqual(len(runner.ancestors),5)
                    self.assertEqual(budget.counts["total_optimizer_steps"],11520)
                    self.assertTrue((root/"models/all-sealed.json").is_file())
                    tests.append(runner.active.copy())
                return status(event,**values)
            runner.status=record
            result=runner.run()
            self.assertEqual(result["trajectories"],480)
            self.assertEqual(len(tests),360)
            self.assertEqual(budget.counts,budget.limits)
            self.assertEqual(budget.phase_counts,budget.contract["phase_limits"])
            self.assertEqual(budget.chunks,{})
            self.assertEqual(len(list((root/"training-tails").glob("*.pkl.gz"))),120)
            report=run_analysis(root,study)
            self.assertEqual(report["trajectory_count"],480)
            self.assertEqual(len(report["contrasts"]),15)
            self.assertTrue(report["tape_artifacts_verified"])
            self.assertTrue(report["progress_evidence"]["barrier_verified"])
            self.assertEqual(len(report["model_evidence"]["byte_verified"]),20)
            self.assertFalse(report["strong_baseline_development_signal"])
            with self.assertRaises(RuntimeError):
                runner.run()
