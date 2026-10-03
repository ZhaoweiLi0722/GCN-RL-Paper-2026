"""Correction entry admission and controller binding; no native or optimizer."""

import copy
import tempfile
import unittest
from unittest.mock import patch

from src.rl import capacity_pilot_recovery1_execution as execution
from src.rl.candidate_pilot_resources import digest
from src.baselines.capacity_completion_control_recovery1 import CapacityCompletionControl
from src.rl.capacity_native import CapacityPilotEnv
from tests.test_capacity_pilot_execution import packet as old_packet
from tests.test_capacity_pilot_runner import FakeBudget, FakeEnv, FakeLearner
from tests.test_capacity_completion_control import PROPOSAL, view


def packet():
    p = old_packet()
    p.pop('packet_sha256')
    p['format'] = 'dynamic-capacity-recovery1-frozen-v1'
    p['reused_streams_from_packet'] = execution.OLD_PACKET_SHA256
    p['recovery_protocol'] = {'sha256': 'fixture-amendment'}
    p['intent']['recovery_protocol_sha256'] = 'fixture-amendment'
    return dict(p, packet_sha256=digest(p))


class RecoveryEntryTests(unittest.TestCase):
    def test_new_directory_and_exact_unchanged_budget(self):
        self.assertNotEqual(execution.RUN, execution.OLD_RUN)
        self.assertTrue(execution.RUN.endswith('_recovery1'))
        auth = execution.authorization(packet())
        self.assertEqual(auth['limits']['limits']['total_optimizer_steps'], 7872)
        self.assertEqual(auth['limits']['limits']['native_steps'], 18432)
        self.assertEqual(auth['attempts'], 1)
        self.assertFalse(auth['automatic_retry'])

    def test_correction_authority_and_old_attempt_are_bound(self):
        for field, value in [('reused_streams_from_packet', 'different'),
                             ('recovery_protocol', {'sha256': 'changed'})]:
            p = copy.deepcopy(packet())
            p.pop('packet_sha256')
            p[field] = value
            p['packet_sha256'] = digest(p)
            with self.assertRaises(PermissionError): execution.authorization(p)

    def test_actual_child_builder_injects_versioned_controller(self):
        with tempfile.TemporaryDirectory() as path, \
                patch.object(CapacityPilotEnv, '__init__', side_effect=AssertionError('native forbidden')):
            runner = execution.build_runner(path, PROPOSAL, FakeBudget(PROPOSAL),
                admission={'verified': True, 'scope': 'dynamic-capacity-pilot-v1'},
                env_factory=FakeEnv, learner_factory=FakeLearner)
            self.assertIs(runner.controller_factory, CapacityCompletionControl)
            self.assertIsNone(runner.env)
            self.assertIsNone(runner.learner)
            runner.controller = runner.controller_factory(PROPOSAL,
                base_control=runner.base_control, scientific=True)
            runner._observe(view())
            runner.controller.act(view(), role='id_mpc', before_query=runner._query,
                                  before_filter=runner._filter)
            runner.budget.finish_chunk('planner_total_model_epochs')
            self.assertEqual(runner.budget.counts['planner_total_model_epochs'], 384)
            self.assertEqual(runner.budget.counts['total_optimizer_steps'], 0)

    def test_no_admission_no_scientific_child(self):
        with patch.object(execution, 'approved', side_effect=PermissionError('not frozen')), \
                patch.object(execution, 'build_runner', side_effect=AssertionError('runner forbidden')):
            with self.assertRaises(PermissionError): execution.child('.')


if __name__ == '__main__':
    unittest.main()
