"""Invented tensors/records only; patient execution, fitting and loads forbidden."""

import copy
from dataclasses import asdict
import random
import unittest
from unittest.mock import patch

import numpy as np

from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.rl import cohort_public, patient_replay_collector
from src.rl.candidate_rollout import CandidateDecision
from src.rl.cohort_evaluation_models import restore_evaluation_owner
from src.rl.cohort_ppo import CohortPPOKernel
from src.rl.cohort_public import CohortPrefixSession
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.dynamic_candidate_rollout import evaluate_dynamic_policy
from src.rl.dynamic_candidate_session import dynamic_candidate_submission
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import decode_arrays, encode_arrays
from src.rl.routing_candidate_contract import choose_candidate
from tests.test_cohort_public import ArrayLayout, MANIFEST, SOURCE, fixture, kernel
from tests.test_dynamic_candidate_session import FakeReference, OPTIONS


ROLES = {"own_frozen": "frozen", "window_ppo": "window",
         "cohort_ppo": "cohort", "bc_continue": "bc"}


def invented_moments(optimizer_state, parameters, steps):
    """Fabricate serialization fixtures directly, never simulate Adam.step."""
    ids = [i for group in optimizer_state["param_groups"] for i in group["params"]]
    optimizer_state["state"] = {
        index: dict(step=torch.tensor(float(steps), dtype=torch.float32),
                    exp_avg=torch.full_like(parameter, .031),
                    exp_avg_sq=torch.full_like(parameter, .09))
        for index, parameter in zip(ids, parameters)}


def invented_owner(role, *, completed=False, dtype=None):
    _, _, _, producer = fixture()
    owner = kernel(producer, ROLES[role], dtype=dtype)
    if role == "own_frozen":
        prototype = copy.deepcopy(owner.policy)
        with torch.no_grad():
            for parameter in prototype.actor_parameters():
                parameter.add_(.017)
        owner = DynamicCandidatePPOKernel(prototype, owner.contract, owner.settings,
            enabled=True, mode="frozen", sampling_seed=41, shuffle_seed=43)
    # Private streams deliberately no longer equal their initial seeded states.
    torch.rand(7, generator=owner.sampling_rng)
    torch.rand(5, generator=owner.rng if role == "bc_continue" else owner.shuffle_rng)
    if not completed:
        return owner
    saved = owner.state_dict()
    for name, value in saved["policy"].items():
        if name.startswith("actor.") or role != "bc_continue":
            value.add_(.013)
    if role == "bc_continue":
        saved["steps"] = 2
        saved["history"] = [dict(identities=["invented/0", "invented/1"],
                                  examples_sha256="a" * 64, steps=2)]
        invented_moments(saved["optimizer"], owner.policy.actor_parameters(), 2)
    else:
        saved["history"] = [4]
        saved["consumed"] = [(SOURCE, row["trajectory_id"], t) for row in MANIFEST[:2] for t in range(2)]
        closures = [dict(row, source_id=SOURCE, split="training", tail_costs=(3., 4., 5.),
                         terminal_active=0, prefix_state_sha256="b" * 64,
                         final_state_sha256="c" * 64, tail_rows_sha256="d" * 64)
                    for row in MANIFEST[:2]]
        saved["target_history"] = [owner._receipt(
            ((-1., -2.), (-3., -4.)), ((.1, .2), (.3, .4)), closures,
            ("e" * 64,) * 2, ("f" * 64,) * 2, 0)]
        for name in ("actor", "critic"):
            invented_moments(saved["optimizer"][name],
                             getattr(owner.policy, name + "_parameters")(), 2)
    owner.load_state_dict(saved)
    return owner


def prefix(owner, *, boundary=False):
    _, _, env, producer = fixture()
    return CohortPrefixSession(env, producer, FakeReference(producer, boundary=boundary), owner,
        enabled=True, options=OPTIONS, trajectory_id="test/invented/evaluation", split="test",
        selection="greedy", source_id=SOURCE, environment_seed=101)


@unittest.skipIf(torch is None, "torch unavailable")
class CohortEvaluationModelTests(unittest.TestCase):
    def setUp(self):
        for target, method in ((PatientConditionCapacityEnv, "__init__"),
                               (PatientConditionCapacityEnv, "step"),
                               (PatientConditionCapacityEnv, "observation"),
                               (torch.optim.Adam, "step"), (torch.optim.SGD, "step"),
                               (torch, "load"), (torch.Tensor, "backward"),
                               (DynamicCandidateImitationKernel, "fit"),
                               (DynamicCandidatePPOKernel, "update"), (CohortPPOKernel, "update")):
            guard = patch.object(target, method, side_effect=AssertionError("scientific execution forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)
        # Reuse the audited array fixture, replacing only declared engine type
        # bindings. Production prefix constructor/step/audit paths run unchanged.
        for module in (patient_replay_collector, cohort_public):
            binding = patch.object(module, "PatientConditionCapacityEnv", ArrayLayout)
            binding.start()
            self.addCleanup(binding.stop)

    def assert_state_equal(self, left, right):
        self.assertEqual(state_digest(left), state_digest(right))

    def test_all_four_formats_reproduce_greedy_receipts_and_entire_state(self):
        for role in ROLES:
            for completed in ((False,) if role == "own_frozen" else (False, True)):
                with self.subTest(role=role, completed=completed):
                    original = invented_owner(role, completed=completed)
                    saved = decode_arrays(encode_arrays(original.state_dict()))
                    before = state_digest(saved)
                    restored = restore_evaluation_owner(saved)
                    self.assertIs(type(restored), type(original))
                    self.assertTrue(restored.evaluation_only)
                    self.assertEqual(restored.contract, original.contract)
                    self.assertFalse(restored.policy.training)
                    self.assertTrue(all(not p.requires_grad and p.grad is None
                                        for p in restored.policy.parameters()))
                    a, b = prefix(original), prefix(restored)
                    self.assertEqual(a.manifest, b.manifest)
                    while not a.closed:
                        observation, bank = a._context(a.env.observation(), a.expected_token)
                        evaluation = evaluate_dynamic_policy(original.policy, observation, bank, original.contract)
                        expected = CandidateDecision(evaluation, choose_candidate(bank, int(np.argmax(evaluation.log_probs))))
                        actual = restored.greedy(observation, bank)
                        self.assertEqual(expected, actual)
                        self.assertEqual(expected.old_log_prob, actual.old_log_prob)
                        self.assertEqual(evaluation.log_probs, actual.evaluation.log_probs)
                        request = dynamic_candidate_submission(restored.policy, observation, actual,
                                                               current_state_token=a.expected_token)
                        self.assertEqual(tuple(request), expected.choice.submitted_request)
                        self.assert_state_equal(encode_arrays(a.step(before_step=lambda: None)),
                                                encode_arrays(b.step(before_step=lambda: None)))
                        self.assertEqual(before, state_digest(restored.state_dict()))
                    self.assert_state_equal(a.state_dict(), b.state_dict())
                    self.assertEqual(before, state_digest(saved))
                    self.assertEqual(before, state_digest(original.state_dict()))
                    self.assertEqual(before, state_digest(restored.state_dict()))

    def test_double_precision_and_original_boundary_requests_survive(self):
        for role in ("own_frozen", "window_ppo", "cohort_ppo"):
            with self.subTest(role=role):
                original = invented_owner(role, completed=role != "own_frozen", dtype=torch.float64)
                restored = restore_evaluation_owner(original.state_dict())
                self.assertEqual(next(restored.policy.parameters()).dtype, torch.float64)
                a, b = prefix(original, boundary=True), prefix(restored, boundary=True)
                while not a.closed:
                    self.assert_state_equal(encode_arrays(a.step(before_step=lambda: None)),
                                            encode_arrays(b.step(before_step=lambda: None)))
                self.assert_state_equal(original.state_dict(), restored.state_dict())

    def test_architecture_message_mode_and_tie_breaking_come_from_saved_definition(self):
        _, _, _, producer = fixture()
        settings = kernel(producer, "frozen").settings
        for architecture, message_mode in (("graph", "physical"), ("graph", "self_only"),
                                           ("flat", "self_only")):
            with self.subTest(architecture=architecture, message_mode=message_mode):
                model = DynamicCandidatePolicy(producer.contract.inputs, enabled=True,
                    architecture=architecture, message_mode=message_mode, encoder_width=4, head_width=6,
                    actor_seed=137, critic_seed=139, initial_reference_bias=.2)
                with torch.no_grad():
                    for parameter in model.actor_parameters():
                        parameter.zero_()
                    model.critic.value_head[-1].bias.fill_(.41)
                original = DynamicCandidatePPOKernel(model, producer.contract, settings,
                    enabled=True, mode="frozen", sampling_seed=149, shuffle_seed=151)
                owner = restore_evaluation_owner(original.state_dict())
                run = prefix(owner)
                observation, bank = run._context(run.env.observation(), run.expected_token)
                expected = evaluate_dynamic_policy(original.policy, observation, bank, original.contract)
                decision = owner.greedy(observation, bank)
                self.assertEqual(decision.evaluation, expected)
                self.assertEqual(decision.choice.class_index, 0)
                self.assertEqual(len(set(expected.log_probs)), 1)
                self.assert_state_equal(original.state_dict(), owner.state_dict())

    def test_greedy_session_clone_and_restore_keep_read_only_bindings(self):
        for role in ROLES:
            with self.subTest(role=role):
                owner = restore_evaluation_owner(invented_owner(role, completed=role != "own_frozen").state_dict())
                run = prefix(owner)
                run.step(before_step=lambda: None)
                saved = run.state_dict()
                clone = copy.deepcopy(run)
                clone.load_state_dict(saved)
                self.assert_state_equal(run.state_dict(), clone.state_dict())
                self.assert_state_equal(encode_arrays(run.step(before_step=lambda: None)),
                                        encode_arrays(clone.step(before_step=lambda: None)))
                self.assertIs(clone.learner.greedy.__self__, clone.learner)
                self.assert_state_equal(owner.state_dict(), clone.learner.state_dict())
                with self.assertRaisesRegex(ValueError, "evaluation-only"):
                    clone.learner.update()

    def test_no_sampling_fit_admission_optimizer_or_changed_state_reload(self):
        for role in ROLES:
            with self.subTest(role=role):
                owner = restore_evaluation_owner(invented_owner(role, completed=role != "own_frozen").state_dict())
                before = state_digest(owner.state_dict())
                for method in ("decide", "fit", "_fit", "update", "_update_in_place",
                               "add_segment", "load", "save"):
                    with self.assertRaisesRegex(ValueError, "evaluation-only"):
                        getattr(owner, method)()
                optimizers = [owner.optimizer] if role == "bc_continue" else owner.optimizers.values()
                for optimizer in optimizers:
                    if optimizer is not None:
                        self.assertNotIsInstance(optimizer, torch.optim.Optimizer)
                        with self.assertRaisesRegex(ValueError, "evaluation-only"):
                            optimizer.step()
                        with self.assertRaisesRegex(ValueError, "evaluation-only"):
                            optimizer.load_state_dict({})
                owner.load_state_dict(owner.state_dict())
                changed = owner.state_dict()
                changed["policy"]["actor.reference_bias"].add_(1.)
                with self.assertRaisesRegex(ValueError, "unchanged"):
                    owner.load_state_dict(changed)
                owner._set_modes()
                self.assertTrue(all(not p.requires_grad for p in owner.policy.parameters()))
                self.assertEqual(before, state_digest(owner.state_dict()))

    def test_restore_does_not_alias_caller_state_or_change_global_rng(self):
        for role in ROLES:
            with self.subTest(role=role):
                state = invented_owner(role, completed=role != "own_frozen").state_dict()
                python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
                owner = restore_evaluation_owner(state)
                run = prefix(owner)
                run.step(before_step=lambda: None)
                self.assertEqual(python_rng, random.getstate())
                for a, b in zip(numpy_rng, np.random.get_state()):
                    np.testing.assert_equal(a, b)
                self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
                before = state_digest(owner.state_dict())
                state["policy"]["actor.reference_bias"].add_(1.)
                state["sampling_rng"].zero_()
                state["manifest"]["policy"]["encoder_width"] = 999
                exported = owner.state_dict()
                exported["policy"]["actor.reference_bias"].add_(2.)
                exported["history"].clear()
                self.assertEqual(before, state_digest(owner.state_dict()))

    def test_pending_prefix_records_and_tail_closures_are_preserved(self):
        for role in ("window_ppo", "cohort_ppo"):
            with self.subTest(role=role):
                owner = invented_owner(role)
                run = prefix(owner)
                while not run.closed:
                    run.step(before_step=lambda: None)
                # Invent training lineage over the array receipts, without a
                # training session, sampling, admission or update invocation.
                from src.rl.candidate_patient_session import restore_decision
                from src.rl.candidate_rollout import prepare_candidate_segment
                from src.rl.prospective_patient_session import restore_record
                decisions = [restore_decision(e["audit"]["decision"], owner.contract) for e in run.events]
                records = []
                for event in run.events:
                    record = copy.deepcopy(event["audit"]["record"])
                    record["trajectory_id"] = MANIFEST[0]["trajectory_id"]
                    records.append(restore_record(record))
                segment = prepare_candidate_segment(decisions, records, owner.contract,
                    behavior_sha256=owner.policy.snapshot_sha256(), bootstrap=None,
                    gae_lambda=1., max_steps=4)
                state = owner.state_dict()
                state["pending"] = [asdict(segment)]
                state["pending_cohorts"] = [dict(MANIFEST[0], source_id=SOURCE, split="training",
                    tail_costs=(1., 2., 3.), terminal_active=0, prefix_state_sha256="a" * 64,
                    final_state_sha256="b" * 64, tail_rows_sha256="c" * 64)]
                restored = restore_evaluation_owner(state)
                self.assert_state_equal(state, restored.state_dict())
                evaluation = prefix(restored)
                evaluation.step(before_step=lambda: None)
                self.assert_state_equal(state, restored.state_dict())

    def test_malformed_failed_incompatible_or_non_model_inputs_are_rejected(self):
        for bad in (None, "unused.pt", {}, {"state": {}, "sha256": "a" * 64}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                restore_evaluation_owner(bad)
        for role in ROLES:
            state = invented_owner(role, completed=role != "own_frozen").state_dict()
            corruptions = [
                lambda s: s.update(failure={"message": "invented failure"}),
                lambda s: s["manifest"].update(format="unsupported"),
                lambda s: s["policy"].pop("actor.reference_bias"),
                lambda s: s["policy"].update(extra=torch.tensor(0.)),
                lambda s: s["policy"].update({"actor.reference_bias": torch.tensor(float("nan"))}),
                lambda s: s["policy"].update({"actor.reference_bias": torch.ones(2)}),
                lambda s: s["policy"].update({"actor.reference_bias": torch.tensor(1., dtype=torch.float64)}),
                lambda s: s["manifest"]["policy"].update(encoder_width=999),
                lambda s: s["manifest"]["policy"]["initialization"].update(torch_version="wrong-runtime"),
                lambda s: s["manifest"]["contract"]["replay"].update(action_dim=999),
                lambda s: s.update(sampling_rng=torch.zeros(1, dtype=torch.uint8)),
            ]
            if role != "bc_continue":
                corruptions.append(lambda s: s.update(manifest_sha256="0" * 64))
            if role in ("window_ppo", "cohort_ppo"):
                corruptions.append(lambda s: s["target_history"].clear())
                corruptions.append(lambda s: s["optimizer"]["actor"]["state"][0]["step"].fill_(99.))
            elif role == "bc_continue":
                corruptions.append(lambda s: s["manifest"]["settings"].update(allowed_split="demonstration"))
                corruptions.append(lambda s: s["optimizer"]["state"][0]["step"].fill_(99.))
            else:
                corruptions.append(lambda s: s["manifest"].update(mode="online"))
            for mutate in corruptions:
                bad = copy.deepcopy(state)
                mutate(bad)
                with self.subTest(role=role, change=corruptions.index(mutate)), self.assertRaises((ValueError, TypeError)):
                    restore_evaluation_owner(bad)


if __name__ == "__main__":
    unittest.main()
