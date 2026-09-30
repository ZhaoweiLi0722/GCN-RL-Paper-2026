"""Pure/invented collector checks. Patient construction/reset/step are forbidden."""

import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from src.baselines.heuristics import facility_net_action_from_state
from src.env.capacity_planning import CapacityPlanningConfig
from src.env.patient_capacity_planning import PatientConditionCapacityEnv, PatientEnvConfig
from src.models.candidate_policy import CandidatePolicy
from src.models.matched_inputs import build_matched_inputs
from src.rl.candidate_collection_boundary import (
    CollectionBoundary, OPERATING_COSTS, PATIENT_COSTS, audit_candidate_step,
    candidate_submission, public_candidate_context,
)
from src.rl.candidate_rollout import CandidateDecision, evaluate_candidate_policy, prepare_candidate_segment
from src.rl.networks import torch
from src.rl.patient_replay_collector import PatientObservationProducer
from src.rl.routing_candidate_contract import choose_candidate
from tests.test_candidate_policy_rollout import candidates, contract, observation, policy, request


def dormant_metadata(**changes):
    """Metadata shell only: no constructor, patient, simulator RNG or episode."""
    options = dict(num_facilities=2, production_lead_time=2, episode_horizon=4,
                   action_mode="facility_net", include_supplier_state=True, include_time_state=True,
                   demand_rates=(1., 2.), max_reagent_replenishment=(4., 4.),
                   max_specimen_transfer=4., specimen_edges=((0, 1),), resource_edges=((0, 1),),
                   capacity_edges=((0, 1),), information_edges=((0, 1),))
    base = CapacityPlanningConfig(**(options | changes))
    config = PatientEnvConfig(base=base, enable_specimen_routing=True, include_specimen_routing_state=True)
    shell = object.__new__(PatientConditionCapacityEnv)
    shell.config, shell.env_config = base, config
    shell.features_per_facility = (3 + base.production_lead_time + int(base.include_supplier_state)
                                    + int(base.include_demand_forecast_state)
                                    + 3 * int(base.include_transfer_pipeline_state)
                                    + 3 * int(base.include_demand_history_state)
                                    + 3 * base.demand_sequence_length * int(base.include_demand_sequence_state))
    shell.summary_width = 6 + len(config.survival_bucket_edges) + 1 + 4
    shell.observation_size = 2 * (shell.features_per_facility + shell.summary_width) + int(base.include_time_state)
    shell.action_size = 8
    for name in ("specimen_edges", "resource_edges", "capacity_edges", "information_edges"):
        setattr(shell, name, getattr(base, name))
    return shell


def invented_raw(shell):
    width = shell.features_per_facility
    base = np.zeros((2, width), dtype=np.float32)
    base[:, :5] = [[3, 2, 4, 1, 2], [1, 3, 2, 2, 1]]
    if shell.config.include_supplier_state:
        base[:, 5] = 1
    tail = np.arange(2 * shell.summary_width, dtype=np.float32).reshape(2, shell.summary_width) / 10
    clock = [.25] if shell.config.include_time_state else []
    return np.concatenate((base.flatten(), tail.flatten(), np.array(clock, dtype=np.float32)))


def invented_info():
    parts = {name: float(i + 1) for i, name in enumerate(OPERATING_COSTS + PATIENT_COSTS)}
    base = sum(parts[k] for k in OPERATING_COSTS)
    return dict(parts, base_cost=base, cost=base + sum(parts[k] for k in PATIENT_COSTS),
                specimen_route_cost=parts["specimen_transfer_cost"],
                transshipment_cost=sum(parts[k] for k in OPERATING_COSTS[-3:]),
                specimen_requested_integer_net=np.array([-1., 1.]), specimen_transfers=np.array([-1., 1.]),
                specimen_route_count=1., blocked_specimen_requests=0., blocked_specimen_inbound_requests=0.,
                blocked_specimen_outbound_requests=0., patients_lost=np.array([1., 0.]),
                patients_completed=np.array([0., 1.]), identity_active_count=4., identity_terminal_count=2.,
                waiting_patients=np.array([1., 0.]), in_production_patients=np.array([0., 1.]),
                specimen_in_transit=np.array([0., 1.]))


@unittest.skipIf(torch is None, "torch unavailable")
class CandidateCollectionBoundaryTests(unittest.TestCase):
    def setUp(self):
        for method in ("__init__", "reset", "step"):
            guard = patch.object(PatientConditionCapacityEnv, method,
                                 side_effect=AssertionError("patient execution forbidden in this packet"))
            guard.start()
            self.addCleanup(guard.stop)
        self.policy = policy()
        self.bank = candidates()
        evaluation = evaluate_candidate_policy(self.policy, observation(), self.bank, contract())
        self.decision = CandidateDecision(evaluation, choose_candidate(self.bank, 0))
        self.submission = candidate_submission(self.policy, observation(), self.decision,
                                               current_state_token=self.bank.state_token)

    def audit(self, **changes):
        info = invented_info()
        kwargs = dict(next_observation=observation(1), next_anchor=torch.tensor([candidates(1).requests[1]]),
                      next_state_token="invented-state-1", raw_reward=-info["cost"], environment_done=False,
                      info=info, boundary=CollectionBoundary(4, 1, "truncation"),
                      source_id="invented-source", trajectory_id="invented-trajectory", step_index=0)
        return audit_candidate_step(self.decision, self.submission, **(kwargs | changes))

    def test_public_producer_layout_and_anchor_without_simulator(self):
        for options in ({}, {"include_supplier_state": False, "include_time_state": False},
                        {"include_demand_forecast_state": True, "include_transfer_pipeline_state": True,
                         "include_demand_history_state": True, "include_demand_sequence_state": True,
                         "demand_sequence_length": 3}):
            shell = dormant_metadata(**options)
            producer = PatientObservationProducer(shell, enabled=True, gamma=.9, reward_scale=.01)
            raw = invented_raw(shell)
            obs, anchor = producer.observe(raw)
            np.testing.assert_array_equal(producer.unpack_raw(obs)[0], raw)
            np.testing.assert_array_equal(anchor[0].numpy(), facility_net_action_from_state(
                raw, producer.anchor_config, settings=producer.settings))
            width = shell.features_per_facility
            np.testing.assert_array_equal(obs.nodes[0, :, :width].numpy(), raw[:2 * width].reshape(2, width))
            np.testing.assert_array_equal(obs.nodes[0, :, width:].numpy(),
                                           raw[2 * width:2 * (width + shell.summary_width)].reshape(2, shell.summary_width))
            self.assertEqual(obs.nodes.shape[-1], len(producer.contract.inputs.node_feature_names))

    def test_public_context_keeps_full_mdl2_anchor_and_raw_input_parity(self):
        shell = dormant_metadata()
        producer = PatientObservationProducer(shell, enabled=True, gamma=.9, reward_scale=.01)
        raw = invented_raw(shell)
        obs, anchor = producer.observe(raw)
        reference = anchor[0].numpy().copy()
        reference[:2] = [-.125, .125]
        view, bank = public_candidate_context(producer, raw, reference_request=reference,
                                              option_requests=[], state_token="invented-token", max_candidates=2)
        np.testing.assert_array_equal(bank.requests[1], anchor[0].numpy())
        flat = build_matched_inputs(view, producer.contract.inputs, anchor, role="actor", message_mode="physical").flat
        for architecture, mode in (("graph", "physical"), ("graph", "self_only"), ("flat", "self_only")):
            model = CandidatePolicy(producer.contract.inputs, enabled=True, architecture=architecture,
                                     message_mode=mode, encoder_width=3, head_width=5, seed=9)
            self.assertEqual(model(obs, bank).actor_state, tuple(flat[0].tolist()))

    def test_full_anchor_mismatch_is_not_silently_replaced(self):
        shell = dormant_metadata()
        producer = PatientObservationProducer(shell, enabled=True, gamma=.9, reward_scale=.01)
        raw = invented_raw(shell)
        _, anchor = producer.observe(raw)
        reference = anchor[0].numpy().copy()
        reference[2] = .25 if reference[2] != .25 else .5
        with self.assertRaisesRegex(ValueError, "other request groups"):
            public_candidate_context(producer, raw, reference_request=reference,
                                      option_requests=[], state_token="s", max_candidates=2)

    def test_unsupported_layouts_and_topologies_remain_rejected(self):
        for options in ({"enable_overtime_control": True}, {"include_central_capacity_hub": True},
                        {"enable_stochastic_procurement": True}, {"include_on_order_state": True},
                        {"reagent_purchase_lead_time": 2}, {"capacity_edges": ()}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PatientObservationProducer(dormant_metadata(**options), enabled=True, gamma=.9, reward_scale=.01)

    def test_hidden_metadata_not_accepted_or_added_to_input(self):
        shell = dormant_metadata()
        producer = PatientObservationProducer(shell, enabled=True, gamma=.9, reward_scale=.01)
        raw = invented_raw(shell)
        first, anchor = producer.observe(raw)
        shell.patient_registry = {"secret": object()}
        shell.rng = "must-not-be-read"
        second, other = producer.observe(raw)
        self.assertTrue(torch.equal(first.nodes, second.nodes))
        self.assertTrue(torch.equal(anchor, other))
        with self.assertRaises(TypeError):
            producer.observe({"raw": raw, "hidden_registry": shell.patient_registry})
        for bad in (raw.astype(np.float64), raw[:-1], raw * np.nan):
            with self.assertRaises((TypeError, ValueError)):
                producer.observe(bad)

    def test_disabled_routing_context_rejected_without_query(self):
        shell = dormant_metadata()
        shell.env_config = replace(shell.env_config, enable_specimen_routing=False)
        producer = PatientObservationProducer(shell, enabled=True, gamma=.9, reward_scale=.01)
        with self.assertRaisesRegex(ValueError, "enabled routing"):
            public_candidate_context(producer, invented_raw(shell), reference_request=request(0.),
                                      option_requests=[], state_token="s", max_candidates=2)

    def test_submission_keeps_original_not_class_center(self):
        bank = candidates(reference_request=request(-.14), option_requests=[])
        e = evaluate_candidate_policy(self.policy, observation(), bank, contract())
        decision = CandidateDecision(e, choose_candidate(bank, bank.reference_class))
        submitted = candidate_submission(self.policy, observation(), decision, current_state_token=bank.state_token)
        self.assertEqual(tuple(submitted), decision.choice.submitted_request)
        self.assertNotEqual(submitted[0], bank.class_features[bank.reference_class][0])
        self.assertEqual(submitted.dtype, np.float64)
        self.assertFalse(submitted.flags.writeable)

    def test_stale_state_old_weights_wrong_input_and_precision_fail_before_submit(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            candidate_submission(self.policy, observation(), self.decision, current_state_token="wrong")
        with self.assertRaisesRegex(ValueError, "reproduce"):
            candidate_submission(self.policy, observation(1), self.decision, current_state_token=self.bank.state_token)
        with torch.no_grad():
            self.policy.value_head[-1].bias.add_(1)
        with self.assertRaisesRegex(ValueError, "reproduce"):
            candidate_submission(self.policy, observation(), self.decision, current_state_token=self.bank.state_token)
        bank = candidates(reference_request=request(.125 - 1e-9), option_requests=[])
        e = evaluate_candidate_policy(self.policy, observation(), bank, contract())
        d = CandidateDecision(e, choose_candidate(bank, bank.reference_class))
        with self.assertRaisesRegex(ValueError, "precision conversion"):
            candidate_submission(self.policy, observation(), d, current_state_token=bank.state_token)

    def test_receipt_reconciles_raw_cost_action_and_unresolved_boundary(self):
        audit = self.audit()
        self.assertEqual(audit.record.raw_reward, -66.)
        self.assertEqual(audit.record.action, self.decision.choice.submitted_request)
        self.assertTrue(audit.record.truncated)
        self.assertFalse(audit.record.terminated)
        self.assertEqual(audit.unresolved_at_boundary, 4)
        self.assertEqual(audit.other_active_patients, 1)
        self.assertEqual(audit.terminal_cost_added, 0.)
        self.assertEqual(audit.candidate_class_count, 3)
        self.assertTrue(audit.selected_request_differs_from_reference)
        self.assertEqual(audit.route_count, 1)

    def test_boundary_mapping_is_explicit_and_does_not_resolve_patients(self):
        for horizon_end, flags in (("terminal", (True, False)), ("truncation", (False, True))):
            audit = self.audit(boundary=CollectionBoundary(1, 2, horizon_end), environment_done=True)
            self.assertEqual((audit.record.terminated, audit.record.truncated), flags)
            self.assertEqual(audit.unresolved_at_boundary, 4)
        open_step = self.audit(boundary=CollectionBoundary(4, 3, "terminal"))
        self.assertIsNone(open_step.unresolved_at_boundary)
        self.assertFalse(open_step.record.terminated or open_step.record.truncated)
        for args in ((0, 1, "terminal"), (2, True, "terminal"), (2, 3, "guess")):
            with self.assertRaises(ValueError):
                CollectionBoundary(*args)
        for index, done in ((-1, False), (4, True), (0, True), (0, 0)):
            with self.assertRaises(ValueError):
                CollectionBoundary(4, 2, "terminal").flags(index, done)

    def test_cost_mismatch_double_scaling_and_extra_cost_fail(self):
        for key in ("base_cost", "cost", "patient_loss_cost", "specimen_route_cost", "transshipment_cost"):
            info = invented_info()
            info[key] += .1
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit(info=info)
        with self.assertRaisesRegex(ValueError, "unscaled"):
            self.audit(raw_reward=-6.6)
        with self.assertRaisesRegex(ValueError, "unsupported cost"):
            self.audit(info=dict(invented_info(), overtime_cost=0.))
        missing = invented_info()
        missing.pop("patient_loss_cost")
        with self.assertRaises(KeyError):
            self.audit(info=missing)
        for bad in (float("nan"), float("inf"), True, -1):
            info = invented_info()
            info["reagent_purchase_cost"] = bad
            with self.assertRaises(ValueError):
                self.audit(info=info)

    def test_route_zero_allowed_but_not_called_behaviorally_distinct_execution(self):
        info = invented_info()
        info.update(specimen_transfers=[0., 0.], specimen_route_count=0., blocked_specimen_requests=1.,
                    blocked_specimen_inbound_requests=1., blocked_specimen_outbound_requests=1.)
        audit = self.audit(info=info)
        self.assertEqual(audit.route_count, 0)
        self.assertEqual(audit.blocked_requests, 1)
        self.assertTrue(audit.selected_request_differs_from_reference)

    def test_bad_requests_flow_counts_and_identity_fields_rejected(self):
        changes = [dict(specimen_requested_integer_net=[-2, 2]), dict(specimen_transfers=[0, 1]),
                   dict(specimen_transfers=[1, -1]), dict(specimen_route_count=0),
                   dict(blocked_specimen_requests=2), dict(blocked_specimen_inbound_requests=1),
                   dict(identity_active_count=2), dict(patients_lost=[.5, 0]),
                   dict(patients_completed=[True, 1]), dict(specimen_transfers=[-1]),
                   dict(waiting_patients=[float("nan"), 0]), dict(identity_terminal_count=-1)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.audit(info=invented_info() | change)

    def test_loss_report_can_differ_without_changing_neural_input(self):
        first = self.audit()
        changed = invented_info()
        changed["patient_registry"] = {"hidden": "audit metadata only"}
        changed["patient_loss_cost"] += 1
        changed["cost"] += 1
        second = self.audit(info=changed, raw_reward=-changed["cost"])
        self.assertEqual(first.record.state, second.record.state)
        self.assertEqual(first.decision, second.decision)
        self.assertNotEqual(first.record.raw_reward, second.record.raw_reward)

    def test_next_observation_type_lineage_and_submitted_precision_checked(self):
        for change in (dict(next_state_token=self.bank.state_token),
                       dict(next_observation=observation(1, torch.float64), next_anchor=torch.zeros(1, 8, dtype=torch.float64)),
                       dict(next_anchor=torch.zeros(1, 7))):
            with self.assertRaises(ValueError):
                self.audit(**change)
        saved = self.submission
        try:
            self.submission = saved.astype(np.float32)
            with self.assertRaisesRegex(ValueError, "exact submitted"):
                self.audit()
        finally:
            self.submission = saved

    def test_accounting_does_not_mutate_inputs_and_audit_arrays_are_copied(self):
        info = invented_info()
        before = copy.deepcopy(info)
        audit = self.audit(info=info)
        for key, value in info.items():
            np.testing.assert_array_equal(value, before[key])
        info["patients_lost"][0] = 9
        self.assertEqual(audit.patients_lost, (1, 0))

    def test_audited_terminal_record_flows_into_existing_gae_without_training(self):
        audit = self.audit(boundary=CollectionBoundary(1, 1, "terminal"), environment_done=True)
        segment = prepare_candidate_segment((audit.decision,), (audit.record,), contract(),
                                             behavior_sha256=self.policy.snapshot_sha256(), bootstrap=None,
                                             gae_lambda=.8, max_steps=1)
        self.assertAlmostEqual(segment.returns[0], -6.6, places=5)
        self.assertTrue(all(p.grad is None for p in self.policy.parameters()))


if __name__ == "__main__":
    unittest.main()
