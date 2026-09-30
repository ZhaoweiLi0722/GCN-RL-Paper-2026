"""Invented numeric/queue fixtures only; no environment episodes or fitting."""

import copy
from dataclasses import FrozenInstanceError, replace
import itertools
import unittest

import numpy as np

from src.env.patient_condition import PatientState
from src.env.specimen_routing import execute_patient_indexed_routes
from src.models.matched_inputs import InputSchema, ObservationBatch
from src.rl.networks import torch
from src.rl.prospective_adapter import ReplayInputContract, pack_actor_state, prepare_replay_batch
from src.rl.routing_candidate_contract import (
    RoutingRequestSchema, build_request_candidates, choose_candidate, class_distribution,
    validate_candidate_transition,
)
from src.rl.validated_returns import OneStepRecord, ReplaySemantics, make_return


def request(x, y=None):
    return [x, -x if y is None else y, 0., 0., 0., 0., -.5, -.5]


def payload():
    return dict(state_token="synthetic-state-0", reference_request=request(.01),
                anchor_request=request(0),
                option_requests=[request(.02), request(.125), request(-.125), request(.14)])


def schema(**changes):
    values = dict(action_schema_id="synthetic-candidate-v1/action", num_facilities=2,
                  max_specimen_transfer=4., max_candidates=6)
    return RoutingRequestSchema(**(values | changes))


def bank():
    return build_request_candidates(payload(), schema())


def semantics(**changes):
    values = dict(reward_kind="absolute_environment", reward_definition_id="synthetic-cost-v1",
                  reward_scale=.1, gamma=.5, state_schema_id="synthetic-candidate-v1/actor-flat",
                  action_schema_id=schema().action_schema_id, state_dim=15, action_dim=8,
                  bootstrap_on_truncation=False)
    return ReplaySemantics(**(values | changes))


def record(choice, **changes):
    values = dict(semantics=semantics(), source_id="invented-fixture", origin="trajectory",
                  trajectory_id="fixture-0", step_index=0, state_token=choice.state_token,
                  next_state_token="synthetic-state-1", state=(0.,) * 15,
                  next_state=(0.,) * 15, action=choice.submitted_request, raw_reward=-2.,
                  terminated=True, truncated=False)
    return OneStepRecord(**(values | changes))


class CandidateRequestTests(unittest.TestCase):
    def test_reference_anchor_and_aliases_are_retained(self):
        b = bank()
        self.assertEqual(len(b.requests), 6)
        self.assertEqual(len(b.class_keys), 3)
        self.assertEqual(b.reference_class, b.anchor_class)
        self.assertEqual(b.members[b.reference_class], (0, 1, 2))
        self.assertEqual(b.representatives[b.reference_class], 0)
        self.assertEqual(b.request_to_class[3], b.request_to_class[5])
        self.assertEqual(choose_candidate(b, b.reference_class).submitted_request, tuple(request(.01)))

    def test_exact_half_away_rounding_and_no_float32_input_coercion(self):
        p = payload()
        p["option_requests"] = [request(.125 - 1e-9), request(.125),
                                request(-.125 + 1e-9), request(-.125)]
        b = build_request_candidates(p, schema())
        self.assertEqual(b.request_to_class[0], b.request_to_class[2])
        self.assertEqual(b.request_to_class[0], b.request_to_class[4])
        self.assertNotEqual(b.request_to_class[0], b.request_to_class[3])
        self.assertNotEqual(b.request_to_class[0], b.request_to_class[5])

    def test_one_ulp_boundary_retains_actual_decoder_float64_behavior(self):
        # Adding .5 in the legacy float64 decoder rounds this value up again.
        p = payload() | {"option_requests": [request(np.nextafter(.125, 0)), request(.125)]}
        b = build_request_candidates(p, schema())
        self.assertEqual(b.request_to_class[2], b.request_to_class[3])
        self.assertNotEqual(b.request_to_class[0], b.request_to_class[2])

    def test_option_order_does_not_change_classes_or_representative_requests(self):
        expected = bank()
        expected_requests = tuple(choose_candidate(expected, i).submitted_request for i in range(3))
        for order in itertools.permutations(payload()["option_requests"]):
            b = build_request_candidates(payload() | {"option_requests": order}, schema())
            self.assertEqual(b.class_keys, expected.class_keys)
            self.assertEqual(tuple(choose_candidate(b, i).submitted_request for i in range(3)), expected_requests)

    def test_bank_is_an_immutable_copy(self):
        p = payload()
        b = build_request_candidates(p, schema())
        before = b.sha256
        p["reference_request"][0] = .9
        p["option_requests"][0][0] = -.9
        self.assertEqual(b.sha256, before)
        with self.assertRaises(FrozenInstanceError):
            b.state_token = "changed"

    def test_anchor_is_preserved_when_its_class_has_no_reference(self):
        p = payload() | {"anchor_request": request(.13)}
        b = build_request_candidates(p, schema())
        self.assertNotEqual(b.anchor_class, b.reference_class)
        self.assertEqual(choose_candidate(b, b.anchor_class).submitted_request, tuple(request(.13)))

    def test_identical_candidates_have_one_class(self):
        b = build_request_candidates(payload() | {"option_requests": []}, schema())
        self.assertEqual(len(b.class_keys), 1)
        self.assertEqual(b.reference_class, 0)

    def test_schema_validation(self):
        for changes in ({"num_facilities": True}, {"num_facilities": 0}, {"action_schema_id": ""},
                        {"max_candidates": 1}, {"max_candidates": 2.5},
                        {"max_specimen_transfer": 0}, {"max_specimen_transfer": True},
                        {"max_specimen_transfer": float("inf")}, {"max_specimen_transfer": 2**40}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                schema(**changes)

    def test_hidden_data_and_external_masks_rejected(self):
        for key in ("patient_registry", "environment", "future_costs", "outcome_mask", "feasible_mask"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "whitelist"):
                build_request_candidates(payload() | {key: []}, schema())
        p = payload()
        del p["anchor_request"]
        with self.assertRaisesRegex(ValueError, "whitelist"):
            build_request_candidates(p, schema())

    def test_invalid_requests_and_non_specimen_changes_rejected(self):
        for value in (float("nan"), float("inf"), 1.01, -.0 + 1j, True, ".2"):
            p = payload()
            p["option_requests"][0][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_request_candidates(p, schema())
        for changed in (request(0)[:-1], [[0] * 8]):
            with self.assertRaisesRegex(ValueError, "shape"):
                build_request_candidates(payload() | {"anchor_request": changed}, schema())
        p = payload()
        p["option_requests"][0][3] = .25
        with self.assertRaisesRegex(ValueError, "other request groups"):
            build_request_candidates(p, schema())

    def test_declared_count_and_state_token_required(self):
        with self.assertRaisesRegex(ValueError, "cap"):
            build_request_candidates(payload(), schema(max_candidates=5))
        with self.assertRaisesRegex(ValueError, "state token"):
            build_request_candidates(payload() | {"state_token": ""}, schema())
        with self.assertRaises(TypeError):
            build_request_candidates(payload() | {"option_requests": None}, schema())

    def test_digest_separates_state_schema_scale_and_raw_aliases(self):
        original = bank().sha256
        variants = [build_request_candidates(payload() | {"state_token": "state-1"}, schema()),
                    build_request_candidates(payload(), schema(action_schema_id="other/action")),
                    build_request_candidates(payload(), schema(max_specimen_transfer=5.)),
                    build_request_candidates(payload() | {"reference_request": request(.02)}, schema())]
        self.assertEqual(len({original, *(v.sha256 for v in variants)}), 5)

    def test_class_index_is_not_silently_clipped_or_coerced(self):
        for index in (-1, 3, True, 1., np.int64(1)):
            with self.assertRaises(ValueError):
                choose_candidate(bank(), index)

    def test_decoder_features_are_not_submitted_actions(self):
        b = bank()
        choice = choose_candidate(b, b.request_to_class[3])
        self.assertNotEqual(choice.submitted_request, b.class_features[choice.class_index])
        with self.assertRaisesRegex(ValueError, "original submitted"):
            validate_candidate_transition(choice, b, record(choice, action=b.class_features[choice.class_index]),
                                          semantics(), replay_dtype=np.float64)

    def test_distinct_requests_with_sampled_empty_execution_are_not_merged(self):
        b = bank()
        results = []
        for i in range(len(b.class_keys)):
            action = choose_candidate(b, i).submitted_request
            result = execute_patient_indexed_routes([[], []], np.asarray(action[:2]) * 4, [(0, 1)],
                edge_priorities=None, edge_distances=[1.], edge_transport_hours=[1.],
                edge_transfer_costs=[1.], lead_time_epochs=1, max_transfers_per_patient=1, route_epoch=0)
            results.append(result)
        self.assertTrue(all(np.array_equal(r.actual_net, [0., 0.]) for r in results))
        self.assertEqual(len(b.class_keys), 3)
        self.assertGreater(results[0].blocked_requests, results[1].blocked_requests)

    def test_same_class_matches_existing_decoder_on_invented_queues(self):
        def patient(index, origin):
            return PatientState(.5, 100., 0, patient_id=f"p{index}", specimen_id=f"p{index}",
                                collection_facility=origin, material_facility=origin)
        for lead in (0, 1):
            queues = [[patient(0, 0), patient(1, 0)], [patient(2, 1), patient(3, 1)]]
            outputs = []
            for x in (.125, .14):
                q = copy.deepcopy(queues)
                out = execute_patient_indexed_routes(q, np.asarray(request(x)[:2]) * 4, [(0, 1)],
                    edge_priorities=None, edge_distances=[1.], edge_transport_hours=[1.],
                    edge_transfer_costs=[1.], lead_time_epochs=lead, max_transfers_per_patient=1, route_epoch=0)
                outputs.append((out, q))
            for field_name in ("requested_integer_net", "actual_net", "edge_flows", "immediate_arrivals"):
                np.testing.assert_array_equal(getattr(outputs[0][0], field_name), getattr(outputs[1][0], field_name))
            for field_name in ("transits", "events", "blocked_requests"):
                self.assertEqual(getattr(outputs[0][0], field_name), getattr(outputs[1][0], field_name))
            self.assertEqual(outputs[0][1], outputs[1][1])

    def test_receipt_binds_original_request_and_reward_contract(self):
        b = bank()
        c = choose_candidate(b, b.reference_class)
        r = record(c)
        validate_candidate_transition(c, b, r, semantics(), replay_dtype=np.float32)
        self.assertAlmostEqual(make_return([r], semantics()).reward, -.2)
        for changed in (replace(c, candidate_sha256="wrong"), replace(c, state_token="wrong"),
                        replace(c, submitted_request=tuple(request(0)))):
            with self.assertRaisesRegex(ValueError, "seal"):
                validate_candidate_transition(changed, b, r, semantics(), replay_dtype=np.float32)
        for changed in (record(c, state_token="wrong"), record(c, action=tuple(request(0)))):
            with self.assertRaisesRegex(ValueError, "original submitted"):
                validate_candidate_transition(c, b, changed, semantics(), replay_dtype=np.float32)
        for settings in ({"reward_kind": "anchor_relative"}, {"gamma": .9}, {"reward_scale": .2},
                         {"action_schema_id": "category-index/action"}):
            with self.assertRaisesRegex(ValueError, "contracts differ"):
                validate_candidate_transition(c, b, r, semantics(**settings), replay_dtype=np.float32)

    def test_counterfactual_is_not_behavior_receipt(self):
        b = bank()
        c = choose_candidate(b, 0)
        cf = record(c, origin="counterfactual", trajectory_id=None, step_index=None)
        with self.assertRaisesRegex(ValueError, "trajectory receipt"):
            validate_candidate_transition(c, b, cf, semantics(), replay_dtype=np.float32)

    def test_float32_boundary_crossing_rejected_without_changing_record(self):
        p = payload() | {"reference_request": request(.125 - 1e-9)}
        b = build_request_candidates(p, schema())
        c = choose_candidate(b, b.reference_class)
        r = record(c)
        validate_candidate_transition(c, b, r, semantics(), replay_dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "precision conversion"):
            validate_candidate_transition(c, b, r, semantics(), replay_dtype=np.float32)
        self.assertEqual(r.action, c.submitted_request)
        with self.assertRaisesRegex(ValueError, "precision"):
            validate_candidate_transition(c, b, r, semantics(), replay_dtype=np.int64)


@unittest.skipIf(torch is None, "torch unavailable")
class CandidateTensorTests(unittest.TestCase):
    def test_alias_multiplicity_does_not_change_probability(self):
        b = bank()
        dist = class_distribution(torch.zeros(3, dtype=torch.float64), b)
        torch.testing.assert_close(dist.probs, torch.full((3,), 1 / 3, dtype=torch.float64))
        p = payload()
        p["option_requests"] = [request(.125), request(-.125)]
        small = build_request_candidates(p, schema())
        self.assertEqual(small.class_keys, b.class_keys)
        torch.testing.assert_close(class_distribution(torch.zeros(3, dtype=torch.float64), small).probs, dist.probs)
        with self.assertRaisesRegex(ValueError, "per canonical class"):
            class_distribution(torch.zeros(6), b)

    def test_distribution_is_differentiable_and_does_not_sample_global_rng(self):
        before = torch.get_rng_state().clone()
        logits = torch.tensor([1., -2., .5], dtype=torch.float64, requires_grad=True)
        dist = class_distribution(logits, bank())
        loss = -dist.log_prob(torch.tensor(0))
        loss.backward()
        expected = dist.probs.detach().clone()
        expected[0] -= 1
        torch.testing.assert_close(logits.grad, expected)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_single_class_and_invalid_logits(self):
        b = build_request_candidates(payload() | {"option_requests": []}, schema())
        d = class_distribution(torch.tensor([1000.]), b)
        self.assertEqual(d.probs.item(), 1.)
        self.assertEqual(d.entropy().item(), 0.)
        for logits in (torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                       torch.zeros((1, 1)), torch.ones(1, dtype=torch.int64), [0.]):
            with self.assertRaises(ValueError):
                class_distribution(logits, b)

    def test_large_common_logit_offset_retains_normalized_log_probabilities(self):
        d = class_distribution(torch.full((3,), 1e30), bank())
        torch.testing.assert_close(d.log_prob(torch.arange(3)), torch.full((3,), -np.log(3), dtype=torch.float32))
        limit = torch.finfo(torch.float32).max
        with self.assertRaisesRegex(ValueError, "range overflows"):
            class_distribution(torch.tensor([-limit, 0., limit]), bank())

    def test_original_request_reaches_existing_ddpg_adapter_without_target_changes(self):
        b = bank()
        choice = choose_candidate(b, b.reference_class)
        inputs = InputSchema("synthetic-candidate-v1", ("A", "B"), ("queue",), ("time",),
                             tuple(f"request_{i}" for i in range(8)))
        binding = ReplayInputContract(inputs, semantics())
        links = torch.tensor([[[0., 1.], [1., 0.]]], dtype=torch.float64)
        obs = ObservationBatch(inputs, torch.tensor([[[2.], [3.]]], dtype=torch.float64),
                               torch.tensor([[0.]], dtype=torch.float64), links)
        anchor = torch.tensor([b.requests[1]], dtype=torch.float64)
        state = tuple(pack_actor_state(obs, anchor, binding)[0].tolist())
        r = record(choice, state=state, next_state=state)
        validate_candidate_transition(choice, b, r, binding.replay, replay_dtype=np.float64)
        for mode in ("physical", "self_only"):
            batch = prepare_replay_batch([[r]], binding, dtype=torch.float64, device="cpu", message_mode=mode)
            self.assertEqual(tuple(batch.current_critic.context[0, -8:].tolist()), choice.submitted_request)
            self.assertEqual(batch.bootstrap_discounts.item(), 0.)
            self.assertAlmostEqual(batch.targets(torch.tensor([[999.]], dtype=torch.float64)).item(), -.2)


if __name__ == "__main__":
    unittest.main()
