"""Zero-fit admission and real-controller/public-feature interface fixtures."""

import copy
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.rl import capacity_pilot_execution as execution
from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_adaptation_campaign import numeric_contract
from src.rl.capacity_native import CapacityPilotEnv, common_operation
from src.rl.capacity_pilot_runner import CapacityPilotRunner
from src.rl.capacity_public_features import public_features
from src.baselines.capacity_completion_control import CapacityCompletionControl
from tests.test_capacity_completion_control import PROPOSAL, view, patient
from tests.test_capacity_pilot_runner import FakeBudget, FakeEnv, FakeLearner


def packet():
    row = dict(format="dynamic-capacity-frozen-v1",proposal={"sha256":execution.EXPECTED_PROPOSAL},
        protocol={"sha256":execution.EXPECTED_PROTOCOL},intent={"scope_approved":True},
        seed_audit={"passed":True,"collisions":[]},config=PROPOSAL,contract=numeric_contract(PROPOSAL),
        implementation_commit="fake-committed-source",runtime={"fixture":True})
    return dict(row,packet_sha256=digest(row))


class EntryTests(unittest.TestCase):
    def test_authorization_binds_scope_and_immutable_packet_without_science(self):
        p = packet()
        a = execution.authorization(p)
        self.assertTrue(a["scientific_execution_authorized"])
        self.assertEqual(a["limits"]["limits"]["total_optimizer_steps"],7872)
        self.assertFalse(a["automatic_retry"])
        p["config"] = copy.deepcopy(p["config"])
        p["config"]["learner"]["offline_ddpg_pairs_per_block"] += 1
        with self.assertRaises(ValueError): execution.authorization(p)

    def test_no_permission_or_source_lock_no_scientific_child(self):
        with patch.object(execution,"approved",side_effect=PermissionError("fixture source mismatch")), \
                patch.object(CapacityPilotEnv,"__init__",side_effect=AssertionError("native forbidden")):
            with self.assertRaises(PermissionError): execution.child(".")

    def test_substreams_match_prospective_matrix_and_are_unique(self):
        streams = execution.stream_manifest(PROPOSAL)
        self.assertEqual(len(streams["rows"]),108)
        self.assertEqual(len(streams["allocations"]),652)
        self.assertEqual(len(set(streams["allocations"])),652)

    def test_real_public_controller_feature_and_common_operation_binding(self):
        with tempfile.TemporaryDirectory() as root:
            budget = FakeBudget(PROPOSAL)
            r = CapacityPilotRunner(root,PROPOSAL,budget,
                admission={"verified":True,"scope":"dynamic-capacity-pilot-v1"},
                env_factory=FakeEnv,learner_factory=FakeLearner)
            r.controller = CapacityCompletionControl(PROPOSAL,base_control=common_operation,scientific=True)
            v = view(patients=(patient("a"),patient("b",site=1)),arrivals=(1.,0.,2.,1.))
            r._observe(v)
            before = public_features(v,r.controller.filter,PROPOSAL)
            hours = r.controller.act(v,role="id_mpc",before_query=r._query,before_filter=r._filter)
            budget.finish_chunk("planner_total_model_epochs")
            self.assertEqual(budget.counts["planner_total_model_epochs"],384)
            self.assertTrue(np.isfinite(hours).all())
            self.assertLessEqual(hours.sum(),8+1e-12)
            base = common_operation(v,PROPOSAL)
            self.assertEqual(base.shape,(16,))
            r.controller.record_operation(v,base)
            # This is an artificial zero-completion public receipt, not a patient step.
            next_view = view(1,patients=(patient("a"),patient("b",site=1)),
                eligible=(("a",),("b",),(),()),ordinary=(0.,)*4,arrivals=(1.,0.,2.,1.))
            r._observe(next_view)
            self.assertEqual(budget.counts["estimator_hypothesis_transitions"],100)
            self.assertEqual(public_features(next_view,r.controller.filter,PROPOSAL).shape,before.shape)
            self.assertFalse(budget.chunks)


if __name__ == "__main__":
    unittest.main()
