"""The real serial entry with fake metadata backends and forbidden native fits."""

import copy
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.rl.capacity_adaptation_campaign import numeric_contract, PHASES
from src.rl.capacity_native import CapacityWorldTape, CapacityPilotEnv
from src.rl.capacity_pilot_resources import CapacityPilotBudget
from src.rl.capacity_pilot_runner import CapacityPilotRunner, COST_KEYS, worlds
from src.rl.capacity_public_features import public_features, resource_adjacency
from src.models.capacity_ddpg import CapacityActor, CapacityModelConfig
from src.baselines.capacity_completion_control import CompletionIntervalFilter
from tests.test_capacity_completion_control import PROPOSAL, view


class FakeBudget:
    def __init__(self, proposal):
        self.limits = numeric_contract(proposal)["limits"]
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase = None
        self.chunks = {}
    def debit(self, charges):
        for key, count in charges.items():
            assert self.counts[key] + count <= self.limits[key], key
            self.counts[key] += count
    def enter(self, owner, phase): self.phase = phase
    def check(self, **kwargs): pass
    def snapshot(self): return dict(self.counts)
    def chunk_call(self, kind, total, width):
        if kind not in self.chunks:
            self.debit({kind: total})
            self.chunks[kind] = [total, 0]
        self.chunks[kind][1] += width
    def finish_chunk(self, kind):
        total, done = self.chunks.pop(kind)
        assert total == done


class FakeEnv:
    def __init__(self, proposal, tape, before_native):
        self.before, self.t = before_native, 0
        before_native("construction")
        before_native("construction_reset")
    def step(self, action):
        self.before("step")
        self.t += 1
        costs = dict.fromkeys(COST_KEYS, 0.)
        costs["support_ordinary_staff_cost"] = 1.
        return None, -1., self.t == 64, {**costs, "cost": 1., "support_public_receipt": {"committed_hours": list(action.requested_hours)}}
    def settlement(self): return dict(settled=True, enrolled=0, delivered=0, lost=0)
    def state_dict(self): return dict(fake_epoch=self.t)


class FakeFilter:
    reset_events = []
    def node_summaries(self): return np.zeros((4, 9))


class FakeController:
    def __init__(self, *args, **kwargs): self.filter, self.epoch = FakeFilter(), -1
    def observe(self, public, before_filter=None):
        epoch = public.common.epoch
        if epoch != self.epoch and epoch:
            before_filter({"hypothesis_transitions": 100})
        self.epoch = epoch
    def record_operation(self, *args): pass
    def act(self, public, role, before_query=None, before_filter=None):
        if role == "id_mpc": before_query({"model_epochs": 384})
        return np.full(4, 2.)
    def state_dict(self): return dict(fake=True)


@dataclass
class FakeSeal:
    payload: bytes
    sha256: str
    rows: list


class FakeLearner:
    seals = []
    restored = []
    @classmethod
    def from_proposal(cls, proposal, **kwargs):
        return cls(kwargs["before_forward"], kwargs["before_optimizer"])
    def __init__(self, before_forward, before_optimizer):
        self.forward, self.optimizer = before_forward, before_optimizer
        self.offline_replay, self.world_replay, self.counts = [], [], {}
    def add_offline(self, row): self.offline_replay.append(row)
    def add_world(self, row): self.world_replay.append(row)
    def act(self, features, **kwargs):
        self.forward("behavior", 1)
        return np.full(4, 2.), np.full(4, 2.)
    def fit(self, mode):
        n = {"bc": 1, "warmup": 3, "ddpg": 5}[mode]
        for _ in range(n): self.forward("fake", 64)
        for name in (("actor",) if mode == "bc" else ("critic",) if mode == "warmup" else ("critic", "actor")):
            self.optimizer(name, 64)
        return {"fixture_not_optimizer": True}
    def bc_step(self): return self.fit("bc")
    def critic_warmup_step(self): return self.fit("warmup")
    def offline_ddpg_step(self): return self.fit("ddpg")
    def online_ddpg_step(self): return self.fit("ddpg")
    def state_dict(self): return {"offline_replay": self.offline_replay, "world_replay": self.world_replay}
    def snapshot(self):
        seal = FakeSeal(b"fake", str(len(self.seals)), list(self.offline_replay))
        self.seals.append(seal)
        return seal
    @classmethod
    def from_snapshot(cls, seal, before_forward, before_optimizer):
        assert len(cls.seals) == 3, "test access before all seals"
        obj = cls(before_forward, before_optimizer)
        obj.offline_replay = list(seal.rows)
        cls.restored.append(seal.sha256)
        return obj
    def begin_world(self, world_id, **kwargs):
        assert not self.world_replay


class RunnerTests(unittest.TestCase):
    def test_full_frozen_numeric_schedule_on_real_entry_fake_backend(self):
        FakeLearner.seals, FakeLearner.restored = [], []
        budget = FakeBudget(PROPOSAL)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(CapacityPilotEnv, "__init__", side_effect=AssertionError("forbidden real patient")), \
                patch.object(torch.optim.Adam, "step", side_effect=AssertionError("forbidden optimizer")):
            runner = CapacityPilotRunner(temporary, PROPOSAL, budget,
                admission={"verified": True, "scope": "dynamic-capacity-pilot-v1"},
                env_factory=FakeEnv, learner_factory=FakeLearner, controller_factory=FakeController,
                tape_factory=lambda p,w: CapacityWorldTape(w, (), (), (), None),
                capture=lambda host: view(host.t), features=lambda *args: np.zeros((4, 2), dtype=np.float32),
                base_control=lambda *args,**kwargs: np.zeros(16))
            with patch.object(runner, "_save_state"):
                result = runner.run()
            self.assertEqual(result["trajectories"], 288)
            self.assertEqual(budget.counts, budget.limits)
            self.assertEqual(len(FakeLearner.restored), 108)
            for seal in FakeLearner.seals:
                self.assertEqual(len(seal.rows), 1152)
                terminal = [r for r in seal.rows if r.done]
                self.assertEqual(len(terminal), 24)
                self.assertTrue(all(r.full_cost == 17 and r.settlement_cost == 16 for r in terminal))
            self.assertEqual(len(list((Path(temporary)/"summaries").glob("*.json"))), 288)
            with self.assertRaises(RuntimeError): runner.run()

    def test_fixed_public_features_and_graph_real_model_without_forward(self):
        f = CompletionIntervalFilter(PROPOSAL, scientific=False)
        v = view()
        f.update(v)
        a = public_features(v, f, PROPOSAL)
        self.assertEqual(a.shape[0], 4)
        self.assertEqual(a.dtype, np.float32)
        with patch.object(CapacityActor, "forward", side_effect=AssertionError("no research scoring")):
            actor = CapacityActor(CapacityModelConfig(a.shape[1], (32,32), (32,1), (4,)*4, 8, 2, 4),
                                  adjacency=resource_adjacency(PROPOSAL))
        self.assertEqual(actor.config.node_input_dim, a.shape[1])
        f.update(view(1))
        self.assertEqual(public_features(view(1), f, PROPOSAL).shape, a.shape)

    def test_no_admission_no_directory_or_constructor(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(PermissionError):
                CapacityPilotRunner(Path(root)/"absent", PROPOSAL, None, admission={})
            self.assertFalse((Path(root)/"absent").exists())

    def test_declared_seed_matrix_disjoint_and_native_integer(self):
        streams = [w for phase in PHASES for block in range(3) for w in worlds(PROPOSAL, phase, block)]
        self.assertEqual(len(streams), 108)
        self.assertEqual(len({w["seed"] for w in streams}), 108)
        self.assertTrue(all(type(w["seed"]) is int for w in streams))


class DurableBudgetTests(unittest.TestCase):
    def test_prepaid_query_chunk_is_not_refunded_or_claimed_completed(self):
        with tempfile.TemporaryDirectory() as root:
            b = CapacityPilotBudget(Path(root)/"launcher", PROPOSAL)
            b.enter("initialization_training_and_seals", PHASES[0])
            b.chunk_call("planner_total_model_epochs", 384, 1)
            self.assertEqual(b.counts["planner_total_model_epochs"], 384)
            with self.assertRaises(ValueError): b.finish_chunk("planner_total_model_epochs")
            self.assertEqual(b.snapshot()["verified_completed_chunk_calls"]["planner_total_model_epochs"], 0)
            with self.assertRaises(RuntimeError): b.enter("evaluation_and_settlement", PHASES[2])
            b.close()

    def test_budget_caps_and_failure_latch(self):
        with tempfile.TemporaryDirectory() as root:
            b = CapacityPilotBudget(Path(root)/"launcher", PROPOSAL)
            b.enter("initialization_training_and_seals", PHASES[0])
            with self.assertRaises(RuntimeError): b.debit({"historical_model_loads": 1})
            with self.assertRaises(RuntimeError): b.debit({"native_steps": 1})
            self.assertEqual(b.counts["native_steps"], 0)
            b.close()


if __name__ == "__main__":
    unittest.main()
