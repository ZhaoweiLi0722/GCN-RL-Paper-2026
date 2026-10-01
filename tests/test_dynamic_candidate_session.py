"""Purpose-built fake trajectories only; no patient objects or optimizer steps."""

import copy
from dataclasses import asdict, replace
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.baselines.heuristics import facility_net_action_from_state, heuristic_settings_for_policy
from src.env.specimen_routing import round_facility_net_requests
from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.models.matched_inputs import InputSchema, ObservationBatch
from src.rl.candidate_collection_boundary import OPERATING_COSTS, PATIENT_COSTS
from src.rl.candidate_imitation import ImitationSettings
from src.rl.candidate_ppo_kernel import CandidatePPOSettings
from src.rl.candidate_rollout import CandidateDecision
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.dynamic_candidate_rollout import evaluate_dynamic_policy
from src.rl.dynamic_candidate_session import (
    DynamicCandidateSession, dynamic_candidate_submission, dynamic_context_from_public, load_envelope,
)
from src.rl.networks import torch
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import encode_arrays
from src.rl.routing_candidate_contract import choose_candidate
from src.rl.validated_returns import ReplaySemantics


OPTIONS = [{"group": "specimen_transfer", "epsilon": .5, "sign": -1},
           {"group": "specimen_transfer", "epsilon": .5, "sign": 1}]


def fake_info(action):
    parts = {name: float(i + 1) for i, name in enumerate(OPERATING_COSTS + PATIENT_COSTS)}
    base = sum(parts[name] for name in OPERATING_COSTS)
    requested = round_facility_net_requests(action[:2] * 4.)
    inbound = float(np.maximum(requested, 0).sum())
    outbound = float(np.maximum(-requested, 0).sum())
    return dict(parts, base_cost=base, cost=base + sum(parts[name] for name in PATIENT_COSTS),
                specimen_route_cost=parts["specimen_transfer_cost"],
                transshipment_cost=sum(parts[name] for name in OPERATING_COSTS[-3:]),
                specimen_requested_integer_net=requested, specimen_transfers=np.zeros(2),
                specimen_route_count=0., blocked_specimen_requests=max(inbound, outbound),
                blocked_specimen_inbound_requests=inbound, blocked_specimen_outbound_requests=outbound,
                patients_lost=np.array([1., 0.]), patients_completed=np.array([0., 1.]),
                identity_active_count=4., identity_terminal_count=2., waiting_patients=np.array([1., 0.]),
                in_production_patients=np.array([0., 1.]), specimen_in_transit=np.array([0., 1.]))


class FakeEnv:
    """Two deterministic array transitions, deliberately no simulator inheritance."""
    def __init__(self):
        self.config = SimpleNamespace(episode_horizon=2)
        self.t = 0
        self.raw = np.array([1., 5., 2., 2., 2., 1., 6., 6., 0.], dtype=np.float32)
        self.actions = []

    def observation(self):
        return self.raw.copy()

    def state_dict(self):
        return {"t": self.t, "raw": self.raw.copy(), "actions": copy.deepcopy(self.actions)}

    def load_state_dict(self, state):
        self.t, self.raw = state["t"], state["raw"].copy()
        self.actions = copy.deepcopy(state["actions"])

    def step(self, action):
        if (action.dtype != np.float64 or action.flags.writeable or action.shape != (8,)
                or self.t >= self.config.episode_horizon):
            raise AssertionError("invalid fake submission")
        self.actions.append(action.copy())
        self.t += 1
        self.raw[-1] = self.t / self.config.episode_horizon
        info = fake_info(action)
        return self.observation(), -info["cost"], self.t == self.config.episode_horizon, info


class FakeProducer:
    def __init__(self):
        self.anchor_config = dict(num_facilities=2, production_lead_time=1,
                                  enable_specimen_routing=True, max_specimen_transfer=4.,
                                  max_bioreactor_transfer=4., max_reagent_transfer=4.,
                                  max_reagent_replenishment=(4., 4.), demand_rates=(1., 2.),
                                  specimen_edges=((0, 1),), resource_edges=((0, 1),),
                                  capacity_edges=((0, 1),))
        self.settings = heuristic_settings_for_policy("mdl2")
        inputs = InputSchema("fake-dynamic-session-v1", ("A", "B"),
                             ("demand", "specimens", "reagents", "capacity"), ("time",),
                             tuple(f"request_{i}" for i in range(8)))
        self.contract = ReplayInputContract(inputs, ReplaySemantics(
            "absolute_environment", "invented-cost-v1", .01, 1.,
            inputs.definition_id + "/actor-flat", inputs.definition_id + "/action", 21, 8, True))

    def check_environment(self, env):
        if type(env) is not FakeEnv or env.config.episode_horizon != 2:
            raise ValueError("wrong fake environment")

    def observe(self, raw):
        if not isinstance(raw, np.ndarray) or raw.dtype != np.float32 or raw.shape != (9,):
            raise ValueError("raw fake public observation mismatch")
        obs = ObservationBatch(self.contract.inputs, torch.tensor(raw[:8].reshape(1, 2, 4)),
                               torch.tensor(raw[None, 8:]), torch.tensor([[[0., 1.], [1., 0.]]]))
        anchor = facility_net_action_from_state(raw, self.anchor_config, settings=self.settings)
        return obs, torch.tensor(anchor[None])

    def unpack_raw(self, observation):
        return np.concatenate((observation.nodes.detach().numpy().reshape(1, 8),
                               observation.globals.detach().numpy()), axis=1)


class FakeReference:
    def __init__(self, producer, *, boundary=False):
        self.producer, self.boundary = producer, boundary
        self.checkpoint_sha256 = ("b" if boundary else "a") * 64  # Invented identity, no checkpoint.

    def act(self, raw):
        _, anchor = self.producer.observe(raw)
        request = anchor[0].numpy().astype(np.float64)
        if self.boundary:
            request[:2] = [.125 - 1e-9, -.125 + 1e-9]
        else:
            request[0] += .0001 if request[0] <= 0 else -.0001
        return request


def session(selection="sample", split="training", *, kind="ppo", dtype=None,
            boundary=False, enabled=True, bias=0.):
    producer, env = FakeProducer(), FakeEnv()
    dtype = torch.float32 if dtype is None else dtype
    model = DynamicCandidatePolicy(producer.contract.inputs, enabled=True, architecture="graph",
                                   message_mode="physical", encoder_width=3, head_width=5,
                                   actor_seed=31, critic_seed=37, initial_reference_bias=bias).to(dtype=dtype)
    if kind in ("ppo", "frozen"):
        settings = CandidatePPOSettings(.001, .2, .5, .01, .5, 1., True, 1, 2, 4, 2, 8)
        learner = DynamicCandidatePPOKernel(model, producer.contract, settings, enabled=True,
                                            mode="frozen" if kind == "frozen" else "online",
                                            sampling_seed=41, shuffle_seed=43)
    elif kind in ("bc", "initialization"):
        settings = ImitationSettings(.001, .5, 2, 4, 2,
                                     "demonstration" if kind == "initialization" else "training")
        learner = DynamicCandidateImitationKernel(model, producer.contract, settings, enabled=True,
                                                  shuffle_seed=43, sampling_seed=None if kind == "initialization" else 41)
    else:
        raise ValueError("unknown fake learner")
    return DynamicCandidateSession(env, producer, FakeReference(producer, boundary=boundary), learner,
                                    enabled=enabled, options=OPTIONS, trajectory_id="fake/episode",
                                    split=split, selection=selection, source_id="fake-session-v1")


@unittest.skipIf(torch is None, "torch unavailable")
class DynamicCandidateSessionTests(unittest.TestCase):
    def setUp(self):
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("optimizer step forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)

    def finish(self, run):
        while not run.closed:
            run.step(before_step=lambda: None)

    def assert_rolled_back_with_failure(self, run, before_digest):
        state = run.state_dict()
        self.assertIsNotNone(state["failure"])
        state["failure"] = None
        self.assertEqual(before_digest, state_digest(state))

    def test_two_step_interruption_resume_rng_and_receipts_match(self):
        for kind, dtype in (("ppo", torch.float32), ("bc", torch.float32),
                            ("frozen", torch.float32), ("ppo", torch.float64)):
            with self.subTest(kind=kind, dtype=dtype):
                left, right = session(kind=kind, dtype=dtype), session(kind=kind, dtype=dtype)
                charges = []
                left.step(before_step=lambda: charges.append("first"))
                self.assertFalse(left.closed)
                with self.assertRaisesRegex(ValueError, "complete sampled"):
                    left.segment(1.)
                right.load_state_dict(left.state_dict())
                self.assertEqual(state_digest(left.state_dict()), state_digest(right.state_dict()))
                a = left.step(before_step=lambda: charges.append("second"))
                b = right.step(before_step=lambda: None)
                self.assertEqual(charges, ["first", "second"])
                self.assertEqual(state_digest(encode_arrays(a)), state_digest(encode_arrays(b)))
                self.assertEqual(left.segment(1.), right.segment(1.))
                self.assertEqual(state_digest(left.state_dict()), state_digest(right.state_dict()))
                self.assertEqual(a["audit"]["unresolved_at_boundary"], 4)
                self.assertEqual(a["audit"]["terminal_cost_added"], 0.)
                self.assertTrue(a["audit"]["record"]["terminated"])
                self.assertFalse(a["audit"]["record"]["truncated"])
                with self.assertRaisesRegex(ValueError, "closed"):
                    left.step(before_step=lambda: None)

    def test_closed_segment_admission_and_once_scaled_cost_no_optimizer(self):
        run = session()
        self.finish(run)
        segment = run.segment(1.)
        step_cost = run.events[0]["info"]["cost"]
        np.testing.assert_allclose(segment.returns, [-2 * step_cost * .01, -step_cost * .01], rtol=1e-6)
        run.learner.add_segment(segment)
        self.assertEqual(run.learner.total_optimizer_steps, 0)
        restored = session()
        restored.load_state_dict(run.state_dict())
        self.assertEqual(state_digest(run.state_dict()), state_digest(restored.state_dict()))
        self.assertEqual(len(restored.learner.pending), 1)

    def test_reference_and_greedy_qualification_restore_without_training(self):
        for kind in ("ppo", "frozen", "bc", "initialization"):
            for selection in ("reference", "greedy"):
                with self.subTest(kind=kind, selection=selection):
                    run = session(selection, "qualification", kind=kind, bias=-100.)
                    sampler = run.learner.sampling_rng
                    before_rng = None if sampler is None else sampler.get_state().clone()
                    self.finish(run)
                    first = run.events[0]["audit"]["decision"]
                    bank = run.examples[0]["candidates"]
                    selected = first["choice"]["class_index"]
                    if selection == "greedy":
                        self.assertEqual(selected, int(np.argmax(first["evaluation"]["log_probs"])))
                        self.assertNotEqual(selected, bank["request_to_class"][0])
                    else:
                        self.assertEqual(first["choice"]["submitted_request"], bank["requests"][0])
                    if sampler is not None:
                        self.assertTrue(torch.equal(before_rng, sampler.get_state()))
                    restored = session(selection, "qualification", kind=kind, bias=-100.)
                    restored.load_state_dict(run.state_dict())
                    self.assertEqual(state_digest(run.state_dict()), state_digest(restored.state_dict()))
                    self.assertTrue(all(e["split"] == "qualification" for e in run.examples))
                    with self.assertRaisesRegex(ValueError, "complete sampled"):
                        run.segment(1.)

    def test_full_mdl2_is_not_replaced_by_alias_reference(self):
        anchor_run = session("anchor", "test")
        raw = anchor_run.env.observation()
        _, anchor = anchor_run.producer.observe(raw)
        reference = anchor_run.reference.act(raw)
        self.assertFalse(np.array_equal(anchor[0].numpy(), reference))
        anchor_event = anchor_run.step(before_step=lambda: None)
        reference_run = session("reference", "test")
        reference_event = reference_run.step(before_step=lambda: None)
        self.assertEqual(anchor_event["audit"]["record"]["action"], tuple(anchor[0].tolist()))
        self.assertEqual(reference_event["audit"]["record"]["action"], tuple(reference))
        self.assertNotEqual(anchor_event["audit"]["record"]["action"], reference_event["audit"]["record"]["action"])
        self.assertEqual(anchor_event["audit"]["requested_integer_net"], reference_event["audit"]["requested_integer_net"])
        restored = session("anchor", "test")
        restored.load_state_dict(anchor_run.state_dict())
        self.assertEqual(state_digest(anchor_run.state_dict()), state_digest(restored.state_dict()))

    def test_reference_demonstration_and_preflight_labels_never_become_training(self):
        for selection, split, kind in (("reference", "demonstration", "initialization"),
                                        ("sample", "preflight", "ppo"), ("greedy", "test", "frozen")):
            run = session(selection, split, kind=kind)
            self.finish(run)
            self.assertEqual({e["split"] for e in run.examples}, {"test" if split == "preflight" else split})
            with self.assertRaises(ValueError):
                run.segment(1.)
            restored = session(selection, split, kind=kind)
            restored.load_state_dict(run.state_dict())
            self.assertEqual(state_digest(run.state_dict()), state_digest(restored.state_dict()))

    def test_invalid_split_selection_opt_in_and_missing_sampler_rejected(self):
        for selection, split in (("sample", "test"), ("sample", "qualification"),
                                 ("sample", "demonstration"), ("anchor", "qualification"),
                                 ("greedy", "demonstration"), ("greedy", "training"),
                                 ("reference", "training"), ("anchor", "training"),
                                 ("anchor", "demonstration"), ("guess", "test"), ("greedy", "holdout")):
            with self.subTest(selection=selection, split=split), self.assertRaises(ValueError):
                session(selection, split)
        with self.assertRaisesRegex(ValueError, "enablement"):
            session(enabled=False)
        with self.assertRaisesRegex(ValueError, "private generator"):
            session(kind="initialization")
        with self.assertRaises(TypeError):
            session().step()
        with self.assertRaisesRegex(TypeError, "before_step"):
            session().step(before_step=None)

    def test_debit_failure_rolls_back_sampler_and_env_without_refund(self):
        run, durable = session(), []
        good = run.state_dict()
        before = state_digest(good)
        def fail_after_debit():
            durable.append("nonrefundable")
            raise RuntimeError("debit acknowledgement failed")
        with patch.object(run.env, "step", side_effect=AssertionError("step forbidden")) as step:
            with self.assertRaisesRegex(RuntimeError, "debit acknowledgement"):
                run.step(before_step=fail_after_debit)
            step.assert_not_called()
        self.assertEqual(durable, ["nonrefundable"])
        self.assert_rolled_back_with_failure(run, before)
        failed = run.state_dict()
        failure = failed["failure"]
        self.assertEqual(failure["error_type"], "RuntimeError")
        self.assertEqual(failure["message"], "debit acknowledgement failed")
        self.assertEqual(failure["stage"], "before_step")
        self.assertTrue(failure["debit_callback_started"])
        self.assertFalse(failure["debit_callback_acknowledged"])
        self.assertFalse(failure["environment_step_started"])
        self.assertEqual(failure["resource_authority"], "external_non_refundable_ledger")
        self.assertFalse(torch.equal(failure["partial_transaction"]["sampling_rng"], good["kernel"]["sampling_rng"]))
        for call in (lambda: run.step(before_step=lambda: durable.append("retry")),
                     lambda: run.load_state_dict(good), lambda: run.segment(1.),
                     lambda: session().load_state_dict(failed)):
            with self.assertRaisesRegex(ValueError, "failed session"):
                call()
        self.assertEqual(durable, ["nonrefundable"])
        self.assertEqual(state_digest(failed), state_digest(run.state_dict()))

    def test_post_step_audit_failure_rolls_back_not_external_charge(self):
        run, durable = session(), []
        before = state_digest(run.state_dict())
        initial_model = state_digest(run.learner.policy.state_dict())
        original_step = run.env.step
        def bad_reward(action):
            raw, reward, done, info = original_step(action)
            with torch.no_grad():
                run.learner.policy.actor.reference_bias.add_(.25)
            return raw, reward + 1., done, info
        with patch.object(run.env, "step", side_effect=bad_reward):
            with self.assertRaisesRegex(ValueError, "unscaled negative"):
                run.step(before_step=lambda: durable.append("charged"))
        self.assertEqual(durable, ["charged"])
        self.assert_rolled_back_with_failure(run, before)
        failed = run.state_dict()
        failure = failed["failure"]
        self.assertEqual(failure["stage"], "step_audit")
        self.assertTrue(failure["debit_callback_acknowledged"])
        self.assertTrue(failure["environment_step_started"])
        self.assertEqual(failure["partial_transaction"]["environment"]["t"], 1)
        self.assertNotEqual(initial_model, state_digest(failure["partial_transaction"]["kernel"]["policy"]))
        self.assertEqual(initial_model, state_digest(run.learner.policy.state_dict()))
        self.assertEqual(failure["rollback_errors"], {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed-collection.pt"
            run.save(path)
            evidence = load_envelope(path)
            self.assertEqual(state_digest(failed), state_digest(evidence))
            with self.assertRaisesRegex(ValueError, "failed session evidence"):
                session().load_state_dict(evidence)
            with self.assertRaises(FileExistsError):
                run.save(path)

    def test_exact_float64_submission_and_precision_failure_before_debit(self):
        run, durable = session("reference", "qualification", boundary=True), []
        before = state_digest(run.state_dict())
        with self.assertRaisesRegex(ValueError, "precision conversion"):
            run.step(before_step=lambda: durable.append(1))
        self.assertEqual(durable, [])
        self.assert_rolled_back_with_failure(run, before)
        self.assertEqual(run.state_dict()["failure"]["stage"], "inference")
        double = session("reference", "qualification", boundary=True, dtype=torch.float64)
        expected = double.reference.act(double.env.observation())
        double.step(before_step=lambda: durable.append(1))
        np.testing.assert_array_equal(double.env.actions[0], expected)
        self.assertEqual(double.env.actions[0].dtype, np.float64)
        self.assertEqual(double.env.actions[0][0], .125 - 1e-9)
        restored = session("reference", "qualification", boundary=True, dtype=torch.float64)
        restored.load_state_dict(double.state_dict())
        self.assertEqual(state_digest(double.state_dict()), state_digest(restored.state_dict()))

    def test_bad_pre_call_arguments_do_not_latch_or_consume_step(self):
        run = session()
        before = state_digest(run.state_dict())
        with self.assertRaises(TypeError):
            run.step(before_step=None)
        with self.assertRaises(ValueError):
            run.segment(1.)
        self.assertIsNone(run.state_dict()["failure"])
        self.assertEqual(before, state_digest(run.state_dict()))
        charged = []
        run.step(before_step=lambda: charged.append(1))
        self.assertEqual(charged, [1])

    def test_nonfinite_inference_failure_is_saved_as_terminal_evidence(self):
        run, charged = session(), []
        before = state_digest(run.state_dict())
        original = run.reference.act
        def corrupt_model(raw):
            request = original(raw)
            with torch.no_grad():
                run.learner.policy.actor.reference_bias.fill_(float("nan"))
            return request
        with patch.object(run.reference, "act", side_effect=corrupt_model):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                run.step(before_step=lambda: charged.append(1))
        self.assertEqual(charged, [])
        self.assert_rolled_back_with_failure(run, before)
        failed = run.state_dict()
        evidence = failed["failure"]["partial_transaction"]["kernel"]["policy"]["actor.reference_bias"]
        self.assertEqual(evidence["nonfinite_tensor"]["dtype"], "torch.float32")
        self.assertEqual(evidence["nonfinite_tensor"]["shape"], ())
        raw_bytes = bytes.fromhex(evidence["nonfinite_tensor"]["bytes_hex"])
        self.assertTrue(np.isnan(np.frombuffer(raw_bytes, dtype=np.float32)[0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nonfinite-evidence.pt"
            run.save(path)
            self.assertEqual(state_digest(failed), state_digest(load_envelope(path)))
        with self.assertRaisesRegex(ValueError, "failed session"):
            run.step(before_step=lambda: charged.append(1))

    def test_interrupted_environment_transaction_latches_after_partial_step(self):
        for error in (RuntimeError("step interrupted"), KeyboardInterrupt("step interrupted")):
            run, charged = session(), []
            before = state_digest(run.state_dict())
            original = run.env.step
            def partial_step(action):
                original(action)
                raise error
            with patch.object(run.env, "step", side_effect=partial_step):
                with self.assertRaises(type(error)):
                    run.step(before_step=lambda: charged.append(1))
            self.assertEqual(charged, [1])
            self.assert_rolled_back_with_failure(run, before)
            failed = run.state_dict()["failure"]
            self.assertEqual(failed["stage"], "environment_step")
            self.assertEqual(failed["error_type"], type(error).__name__)
            self.assertEqual(failed["partial_transaction"]["environment"]["t"], 1)
            self.assertIsNone(failed["partial_transaction"]["step_result"])

    def test_rollback_error_preserves_primary_failure_and_restores_sampler(self):
        run = session()
        before_rng = run.learner.sampling_rng.get_state().clone()
        original = run.env.step
        def partial_step(action):
            original(action)
            raise RuntimeError("primary step failure")
        with patch.object(run.env, "step", side_effect=partial_step), \
                patch.object(run.env, "load_state_dict", side_effect=OSError("rollback failed")):
            with self.assertRaisesRegex(RuntimeError, "primary step failure"):
                run.step(before_step=lambda: None)
        failure = run.state_dict()["failure"]
        self.assertEqual(failure["message"], "primary step failure")
        self.assertEqual(failure["rollback_errors"]["environment"]["message"], "rollback failed")
        self.assertTrue(torch.equal(before_rng, run.learner.sampling_rng.get_state()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollback-failure.pt"
            run.save(path)
            self.assertEqual(load_envelope(path)["failure"]["rollback_errors"], failure["rollback_errors"])
        with self.assertRaisesRegex(ValueError, "failed session"):
            run.step(before_step=lambda: None)

    def test_submission_rejects_stale_or_forged_receipt_without_legacy_verifier(self):
        run = session()
        obs, bank = run._context(run.env.observation(), run.expected_token)
        evaluation = evaluate_dynamic_policy(run.learner.policy, obs, bank, run.learner.contract)
        decision = CandidateDecision(evaluation, choose_candidate(bank, bank.reference_class))
        action = dynamic_candidate_submission(run.learner.policy, obs, decision, current_state_token=run.expected_token)
        self.assertFalse(action.flags.writeable)
        self.assertEqual(tuple(action), decision.choice.submitted_request)
        with self.assertRaisesRegex(ValueError, "stale"):
            dynamic_candidate_submission(run.learner.policy, obs, decision, current_state_token="different")
        forged = replace(decision, evaluation=replace(evaluation, value=evaluation.value + 1.))
        with self.assertRaisesRegex(ValueError, "reproduce"):
            dynamic_candidate_submission(run.learner.policy, obs, forged, current_state_token=run.expected_token)

    def test_kernel_terminal_failure_cannot_be_cleared_by_session_restore(self):
        for kind in ("ppo", "bc"):
            run = session(kind=kind)
            self.finish(run)
            if kind == "ppo":
                run.learner.add_segment(run.segment(1.))
            good = run.state_dict()
            durable = []
            def failed_debit(*args):
                durable.append("charged")
                raise RuntimeError("kernel debit failure")
            with self.assertRaisesRegex(RuntimeError, "kernel debit failure"):
                if kind == "ppo":
                    run.learner.update(before_optimizer_step=failed_debit, before_minibatch=lambda: None)
                else:
                    run.learner.fit(run.examples, epochs=1, before_step=failed_debit)
            self.assertEqual(durable, ["charged"])
            self.assertIsNotNone(run.learner.state_dict()["failure"])
            failed = run.state_dict()
            for call in (lambda: run.load_state_dict(good), lambda: run.segment(1.),
                         lambda: run.step(before_step=lambda: None),
                         lambda: session(kind=kind).load_state_dict(failed)):
                with self.assertRaisesRegex(ValueError, "failed"):
                    call()
            self.assertEqual(state_digest(failed), state_digest(run.state_dict()))

    def test_corrupt_session_state_is_rejected_atomically(self):
        run = session()
        run.step(before_step=lambda: None)
        saved = run.state_dict()
        before = state_digest(saved)
        changes = [lambda s: s.update(index=True), lambda s: s["events"].clear(),
                   lambda s: s["examples"][0].update(split="test"),
                   lambda s: s["events"][0]["audit"].update(route_count=100),
                   lambda s: s["events"][0]["info"].update(cost=1.),
                   lambda s: s["events"][0]["audit"]["decision"]["evaluation"].update(value=100.),
                   lambda s: s["kernel"]["sampling_rng"].zero_(),
                   lambda s: s.update(initial_rng=s["initial_rng"].float()),
                   lambda s: s["manifest"].update(selection="reference"),
                   lambda s: s["examples"][0]["candidates"].update(state_token="forged")]
        for mutate in changes:
            bad = copy.deepcopy(saved)
            mutate(bad)
            with self.assertRaises((ValueError, TypeError, RuntimeError)):
                run.load_state_dict(bad)
            self.assertEqual(before, state_digest(run.state_dict()))

    def test_save_load_envelope_is_exact_and_no_overwrite(self):
        run = session()
        run.step(before_step=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake-session.pt"
            run.save(path)
            restored = session()
            restored.load_state_dict(load_envelope(path))
            self.assertEqual(state_digest(run.state_dict()), state_digest(restored.state_dict()))
            with self.assertRaises(FileExistsError):
                run.save(path)
            broken = path.with_name("corrupt.pt")
            payload = torch.load(path, weights_only=True)
            payload["sha256"] = "0" * 64
            torch.save(payload, broken)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_envelope(broken)

    def test_public_source_drift_and_private_sampler_drift_fail(self):
        run = session()
        run.env.raw[0] += 1.
        with self.assertRaisesRegex(ValueError, "drift"):
            run.step(before_step=lambda: None)
        run = session()
        with torch.no_grad():
            next(run.learner.policy.parameters()).add_(1.)
        with self.assertRaisesRegex(ValueError, "drift"):
            run.step(before_step=lambda: None)
        run = session()
        run.learner.sampling_rng.manual_seed(1)
        with self.assertRaisesRegex(ValueError, "RNG drift"):
            run.step(before_step=lambda: None)

    def test_public_context_preserves_support_and_rejects_other_action_groups(self):
        run = session()
        raw = run.env.observation()
        obs, first = dynamic_context_from_public(run.producer, run.reference, raw, OPTIONS, "one")
        _, permuted = dynamic_context_from_public(run.producer, run.reference, raw, list(reversed(OPTIONS)), "two")
        self.assertEqual(first.class_keys, permuted.class_keys)
        self.assertEqual(first.requests[0], tuple(run.reference.act(raw)))
        self.assertEqual(first.requests[1], tuple(run.producer.observe(raw)[1][0].tolist()))
        self.assertEqual(obs.schema, run.learner.contract.inputs)
        bad = run.reference.act(raw)
        bad[2] = .75 if bad[2] != .75 else .25
        with patch.object(run.reference, "act", return_value=bad):
            with self.assertRaisesRegex(ValueError, "other request groups"):
                run.step(before_step=lambda: None)

    def test_global_rng_unchanged_by_collection_and_restore(self):
        before_torch = torch.get_rng_state().clone()
        before_numpy, before_python = np.random.get_state(), random.getstate()
        run = session()
        run.step(before_step=lambda: None)
        other = session()
        other.load_state_dict(run.state_dict())
        other.step(before_step=lambda: None)
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        for left, right in zip(before_numpy, np.random.get_state()):
            np.testing.assert_equal(left, right)


if __name__ == "__main__":
    unittest.main()
