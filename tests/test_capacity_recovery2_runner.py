"""Remaining-only real entry with fake native/learner/replay backends."""

from dataclasses import asdict, replace
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

from src.rl.capacity_adaptation_campaign import PHASES
from src.rl.capacity_ddpg_learner import CapacityTransition
from src.rl.capacity_native import CapacityWorldTape, CapacityPilotEnv
from src.rl.capacity_pilot_runner import worlds, trajectory_id, jsonable
from src.rl.capacity_recovery2_resources import remaining_contract, CapacityRecovery2Budget
from src.rl import capacity_recovery2_runner as module
from src.rl.capacity_pilot_recovery2_execution import build_runner
from tests.test_capacity_completion_control import PROPOSAL, view
from tests.test_capacity_pilot_runner import FakeBudget, FakeEnv, FakeController, FakeLearner, FakeSeal


REMAINDER = json.loads((Path(__file__).resolve().parents[1] /
    'specs/2026-10-03-dynamic-capacity-adaptation/recovery2/proposal.json').read_text())


def public_view(epoch):
    result = view(epoch)
    return replace(result, common=replace(result.common, base_observation=(0.,)))


class Budget(FakeBudget):
    def __init__(self):
        self.limits = remaining_contract(PROPOSAL, REMAINDER)['limits']
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase, self.chunks = None, {}
    def debit(self, charges):
        charges = dict(charges)
        n = charges.get('estimator_hypothesis_transitions', 0) + charges.get('saved_reconstruction_hypothesis_transitions', 0)
        if n: charges['combined_filter_transitions'] = n
        super().debit(charges)


class Env(FakeEnv):
    steps = []
    def load_state_dict(self, state): self.t = state['fake_epoch']
    def step(self, action):
        self.steps.append(self.t)
        return super().step(action)


class Control(FakeController):
    def load_recovery1_boundary(self, state): self.epoch = state['filter']['last_view']['common']['epoch']


def replay(rows, end_state, proposal, *, before_receipt, before_filter, after_receipt):
    for _ in rows:
        before_receipt()
        before_filter({'hypothesis_transitions': 100})
        after_receipt()
    n = min(len(rows), 48)
    return [CapacityTransition.from_cost(state=np.zeros((4, 2)), next_state=np.zeros((4, 2)),
        executed_hours=(2.,)*4, world_id=trajectory_id(rows[0]['world'], 'teacher'),
        control_cost=1., reward_divisor=100000., done=(t == 47 and len(rows) == 64),
        settlement_cost=16. if t == 47 and len(rows) == 64 else None) for t in range(n)]


def fixture(root):
    for d in ('raw', 'summaries', 'states', 'models', 'tapes'): (root/d).mkdir()
    incomplete = list(worlds(PROPOSAL, PHASES[0], 1))[-1]
    for phase, block in ((PHASES[0], 0), (PHASES[1], 0), (PHASES[0], 1)):
        role = 'teacher' if phase == PHASES[0] else 'learner'
        for w in worlds(PROPOSAL, phase, block):
            ident = trajectory_id(w, role)
            tape = CapacityWorldTape(w, (), (), (), None)
            (root/'tapes'/f'{trajectory_id(w,"exogenous")}.json').write_text(json.dumps(asdict(tape)))
            length = 47 if w == incomplete else 64
            rows = [dict(epoch=i, cost=1., world=w, role=role) for i in range(length)]
            with gzip.open(root/'raw'/f'{ident}.jsonl.gz', 'wt') as h:
                for row in rows: h.write(json.dumps(row)+'\n')
            if length == 64:
                (root/'summaries'/f'{ident}.json').write_text('{}')
            if block == 1:
                state = dict(active=dict(w, role=role), epoch=length, learner=None,
                    environment=dict(fake_epoch=length, world_tape_sha256=tape.digest()),
                    controller={'filter': {'last_view': jsonable(public_view(length))}})
                path = root/'failure-state.pkl.gz' if length == 47 else root/'states'/f'{ident}.pkl.gz'
                with gzip.open(path, 'wb') as h: pickle.dump(state, h)
    (root/'models/block0-seal.pt').write_bytes(b'fake-seal')
    (root/'models/block0-seal.json').write_text('{}')
    return {p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.rglob('*') if p.is_file()}


class RemainingRunnerTests(unittest.TestCase):
    def test_all_remaining_counts_partial_append_seals_before_test_no_old_fit(self):
        FakeLearner.seals, FakeLearner.restored, Env.steps = [], [], []
        budget = Budget()
        def snapshot(payload, sha):
            seal = FakeSeal(payload, sha, [None]*1152)
            FakeLearner.seals.append(seal)
            return seal
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(CapacityPilotEnv, '__init__', side_effect=AssertionError('native forbidden')), \
                patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('optimizer forbidden')), \
                patch.object(module, 'reconstruct_teacher', side_effect=replay):
            prior, out = Path(temp)/'prior', Path(temp)/'out'
            prior.mkdir(); out.mkdir()
            inputs = fixture(prior)
            runner = build_runner(out, PROPOSAL, budget, prior_root=prior,
                input_files=inputs, admission=dict(verified=True,scope='dynamic-capacity-pilot-v1',remaining_scope='dynamic-capacity-recovery2'),
                env_factory=Env, learner_factory=FakeLearner, controller_factory=Control, snapshot_factory=snapshot,
                tape_factory=lambda p,w: CapacityWorldTape(w, (), (), (), None),
                capture=lambda e:public_view(e.t), features=lambda *a:np.zeros((4,2),dtype=np.float32),
                base_control=lambda *a,**k:np.zeros(16))
            with patch.object(runner, '_save_state'):
                result = runner.run()
            self.assertEqual(budget.counts, budget.limits)
            self.assertEqual(result['trajectories'],288)
            self.assertEqual(len(Env.steps),16145)
            self.assertEqual(Env.steps[:17],list(range(47,64)))
            self.assertEqual(len(FakeLearner.seals),3)
            self.assertEqual(len(FakeLearner.restored),108)
            partial = trajectory_id(list(worlds(PROPOSAL,PHASES[0],1))[-1], 'teacher')
            with gzip.open(out/'raw'/f'{partial}.jsonl.gz','rt') as h:
                self.assertEqual([json.loads(x)['epoch'] for x in h],list(range(64)))
            for name, record in inputs.items():
                self.assertEqual(hashlib.sha256((prior/name).read_bytes()).hexdigest(),record['sha256'])
            with self.assertRaises(RuntimeError): runner.run()

    def test_remaining_budget_reserves_filter_and_rejects_refund(self):
        with tempfile.TemporaryDirectory() as temp:
            b = CapacityRecovery2Budget(Path(temp)/'launcher',PROPOSAL,REMAINDER)
            b.enter('initialization_training_and_seals',PHASES[0])
            b.chunk_call('saved_reconstruction_hypothesis_transitions',100,1)
            self.assertEqual(b.counts['combined_filter_transitions'],100)
            with self.assertRaises(ValueError): b.finish_chunk('saved_reconstruction_hypothesis_transitions')
            with self.assertRaises(RuntimeError): b.enter('evaluation_and_settlement',PHASES[2])
            with self.assertRaises(RuntimeError): b.debit({'native_steps':-1})
            with self.assertRaises(RuntimeError): b.debit({'native_steps':1})
            b.close()

    def test_no_remaining_permission_no_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(PermissionError):
                build_runner(Path(temp)/'no',PROPOSAL,None,
                    prior_root=temp,input_files={},admission={'verified':True})
            self.assertFalse((Path(temp)/'no').exists())


if __name__ == '__main__': unittest.main()
