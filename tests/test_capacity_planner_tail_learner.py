"""Artificial weight/target fixtures; real forward and optimizer steps forbidden."""

import copy
import hashlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.models.capacity_terminal_value import CapacityTerminalValue
from src.rl.capacity_planner_tail_learner import CapacityPlannerTailLearner
from tests.test_capacity_planner_tail_design import STUDY


class LearnerTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(torch.optim.Adam, "step", side_effect=AssertionError("real optimizer forbidden")).start()
        patch.object(CapacityTerminalValue, "forward", side_effect=AssertionError("real model forward forbidden")).start()

    def make(self, method="planner_tail_td", **callbacks):
        c = dict(STUDY["value"], method=method, architecture="graph", max_new_updates=768)
        return CapacityPlannerTailLearner(c, seed=7, feature_dim=31, ancestor_sha256="1"*64,
            before_forward=callbacks.get("forward", lambda *args: None),
            before_optimizer=callbacks.get("optimizer", lambda *args: None))

    def records(self):
        return [dict(features=np.full((10,4,31), i/100, dtype=np.float32),
                     heuristics=np.zeros(10), costs=np.arange(1,10,dtype=float), epochs=list(range(55,65)))
                for i in range(96)]

    def fake_forward(self, learner, seen):
        def forward(x):
            learner.before_forward("value", len(x))
            learner.counts["forwards"] += 1
            seen.append(len(x))
            return torch.zeros(len(x), dtype=torch.float32, requires_grad=True)
        learner._forward = forward

    def test_variable_rows_identical_data_and_minibatch_rng(self):
        td, mc, seen = self.make(), self.make("planner_tail_mc"), []
        self.fake_forward(td, seen)
        td.admit_tails(self.records())
        mc.admit_tails(self.records())
        self.assertEqual(len(td.pending["states"]), 864)
        self.assertEqual(seen, [64]*13+[32])
        self.assertEqual(td.pending["source_hashes"], mc.pending["source_hashes"])
        np.testing.assert_array_equal(td.pending["states"], mc.pending["states"])
        np.testing.assert_array_equal(td.rng.integers(864,size=64), mc.rng.integers(864,size=64))
        self.assertNotEqual(td.pending["targets"][0], mc.pending["targets"][0])
        np.testing.assert_array_equal(td.pending["targets"][1:9], mc.pending["targets"][1:9])

    def test_full_snapshot_variable_pending_and_binding(self):
        model = self.make("planner_tail_mc")
        model.admit_tails(self.records())
        raw, sha = model.snapshot()
        restored = CapacityPlannerTailLearner.from_bytes(raw,before_forward=lambda *a:None,before_optimizer=lambda *a:None)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
        np.testing.assert_array_equal(model.pending["targets"], restored.pending["targets"])
        self.assertEqual(model.rng.bit_generator.state, restored.rng.bit_generator.state)
        wrong = copy.deepcopy(model.state_dict())
        wrong["ancestor_sha256"] = "2"*64
        with self.assertRaises(ValueError):
            restored.load_state_dict(wrong)
        wrong = copy.deepcopy(model.state_dict())
        wrong["pending"]["completed"] = 2
        with self.assertRaises(ValueError):
            restored.load_state_dict(wrong)

    def test_failed_step_rolls_back_rng_model_targets_but_not_charges(self):
        learner, calls = self.make("planner_tail_mc"), []
        learner.admit_tails(self.records())
        self.fake_forward(learner, calls)
        before = learner.state_dict()
        with self.assertRaisesRegex(AssertionError, "real optimizer forbidden"):
            learner.update()
        self.assertEqual(learner.updates, 0)
        self.assertEqual(learner.pending["completed"], 0)
        self.assertEqual(learner.rng.bit_generator.state, before["rng"])
        self.assertEqual(learner.counts, dict(forwards=1, optimizer_steps=1))
        for key, value in before["model"].items():
            self.assertTrue(torch.equal(value, learner.model.state_dict()[key]))

    def test_artificial_weight_fork_resets_all_training_state(self):
        fixture = self.make().state_dict()
        fixture.update(format="capacity-value-comparison-td-v1", updates=1536, pending=None,
                       counts=dict(forwards=10000,optimizer_steps=1536))
        stream = io.BytesIO()
        torch.save(fixture, stream)
        raw = stream.getvalue()
        sha = hashlib.sha256(raw).hexdigest()
        config = dict(STUDY["value"], method="planner_tail_mc", architecture="graph", max_new_updates=768)
        fork = CapacityPlannerTailLearner.fork_weights(raw, config, seed=11, expected_sha256=sha,
            before_forward=lambda *a:None,before_optimizer=lambda *a:None)
        self.assertEqual(fork.updates,0)
        self.assertEqual(fork.counts,dict(forwards=0,optimizer_steps=0))
        self.assertEqual(len(fork.optimizer.state),0)
        for key, value in fixture["model"].items():
            self.assertTrue(torch.equal(value,fork.model.state_dict()[key]))
        with self.assertRaises(ValueError):
            CapacityPlannerTailLearner.fork_weights(raw,config,seed=11,expected_sha256="0"*64,
                before_forward=lambda *a:None,before_optimizer=lambda *a:None)

    def test_observed_admission_and_update_limit(self):
        learner, seen = self.make("observed_td"), []
        self.fake_forward(learner,seen)
        learner.admit_observed(dict(features=np.zeros((65,4,31),dtype=np.float32),
                                    heuristics=np.zeros(65),costs=np.ones(64)))
        self.assertEqual(seen,[64])
        self.assertEqual(learner.pending["targets"].dtype,np.float64)
        learner.pending=None
        learner.updates=768
        count_before=len(seen)
        with self.assertRaises(ValueError):
            learner.admit_observed(dict(features=np.zeros((65,4,31),dtype=np.float32),
                                        heuristics=np.zeros(65),costs=np.ones(64)))
        self.assertEqual(len(seen),count_before)
        with self.assertRaises(ValueError):
            learner._admit(np.zeros((64,4,31)), np.zeros(64), ["0"*64])
        with self.assertRaises(RuntimeError):
            learner.update()


if __name__ == "__main__":
    unittest.main()
