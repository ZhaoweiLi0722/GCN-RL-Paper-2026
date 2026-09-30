"""Whole-collector recovery on invented metadata/steps, never patient dynamics."""

import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.baselines.heuristics import facility_net_action_from_state
from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.env.specimen_routing import round_facility_net_requests
from src.models.candidate_policy import CandidatePolicy
from src.rl.candidate_patient_session import CandidatePatientSession, load_envelope, save_envelope
from src.rl.candidate_ppo_kernel import CandidatePPOKernel
from src.rl.networks import torch
from src.rl.patient_replay_collector import PatientObservationProducer
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import decode_arrays, encode_arrays
from tests.test_candidate_collection_boundary import dormant_metadata, invented_raw, invented_info
from tests.test_candidate_ppo_kernel import settings


def fixture_state(env):
    return {"t": env.t, "raw": env.fake_raw.copy(), "invented_counter": env.fake_counter}


def fixture_load(env, state):
    env.t, env.fake_raw, env.fake_counter = state["t"], state["raw"].copy(), state["invented_counter"]


def fixture_step(env, action):
    """Invented accounting, not simulated clinical or production behavior."""
    info = invented_info()
    requested = round_facility_net_requests(action[:2] * env.config.max_specimen_transfer)
    inbound = float(np.maximum(requested, 0).sum())
    outbound = float(np.maximum(-requested, 0).sum())
    info.update(specimen_requested_integer_net=requested, specimen_transfers=np.zeros(2),
                specimen_route_count=0., blocked_specimen_requests=max(inbound, outbound),
                blocked_specimen_inbound_requests=inbound, blocked_specimen_outbound_requests=outbound)
    env.t += 1
    env.fake_counter += 7
    env.fake_raw[-1] = env.t / env.config.episode_horizon
    return env.fake_raw.copy(), -info["cost"], env.t == env.config.episode_horizon, info


class InventedReference:
    checkpoint_sha256 = "a" * 64

    def __init__(self, producer):
        self.producer = producer

    def act(self, raw):
        anchor = facility_net_action_from_state(raw, self.producer.anchor_config, settings=self.producer.settings)
        # Distinct raw request, same integer-request class: exercise alias preservation.
        anchor[0] += .0001
        return anchor


OPTIONS = [{"group": "specimen_transfer", "epsilon": .1, "sign": -1},
           {"group": "specimen_transfer", "epsilon": .1, "sign": 1}]


def session(selection="sample", split="training"):
    env = dormant_metadata()
    env.t, env.fake_raw, env.fake_counter = 0, invented_raw(env), 0
    producer = PatientObservationProducer(env, enabled=True, gamma=1., reward_scale=.01)
    model = CandidatePolicy(producer.contract.inputs, enabled=True, architecture="graph", message_mode="physical",
                             encoder_width=3, head_width=5, seed=19)
    learner = CandidatePPOKernel(model, producer.contract, settings(gae_lambda=1.), enabled=True,
                                 mode="online", sampling_seed=119, shuffle_seed=121)
    return CandidatePatientSession(env, producer, InventedReference(producer), learner, enabled=True,
                                    options=OPTIONS, trajectory_id="invented/episode", split=split,
                                    selection=selection, source_id="invented-session-v1")


class CandidatePatientSessionTests(unittest.TestCase):
    def setUp(self):
        patches = [patch.object(PatientConditionCapacityEnv, name, side_effect=AssertionError("patient execution forbidden"))
                   for name in ("__init__", "reset")]
        patches += [patch.object(PatientConditionCapacityEnv, "state_dict", fixture_state),
                    patch.object(PatientConditionCapacityEnv, "load_state_dict", fixture_load),
                    patch.object(PatientConditionCapacityEnv, "observation", lambda env: env.fake_raw.copy()),
                    patch.object(PatientConditionCapacityEnv, "step", fixture_step)]
        for guard in patches:
            guard.start()
            self.addCleanup(guard.stop)

    def test_complete_episode_preserves_raw_cost_and_absolute_return(self):
        run, spent = session(), []
        while not run.closed:
            run.step(before_step=lambda: spent.append(1))
        self.assertEqual(len(spent), 4)
        segment = run.segment(1.)
        expected = -invented_info()["cost"] * .01
        np.testing.assert_allclose(segment.returns, [expected * n for n in (4, 3, 2, 1)], rtol=1e-6)
        self.assertTrue(segment.records[-1].terminated)
        self.assertFalse(segment.records[-1].truncated)
        self.assertEqual(run.events[-1]["audit"]["unresolved_at_boundary"], 4)
        run.learner.add_segment(segment)
        run.learner.update()

    def test_mid_episode_restore_next_step_and_finished_update_exact(self):
        run, restored = session(), session()
        run.step(before_step=lambda: None)
        restored.load_state_dict(run.state_dict())
        self.assertEqual(state_digest(run.state_dict()), state_digest(restored.state_dict()))
        while not run.closed:
            left, right = run.step(before_step=lambda: None), restored.step(before_step=lambda: None)
            self.assertEqual(state_digest(encode_arrays(left)), state_digest(encode_arrays(right)))
        self.assertEqual(run.segment(1.), restored.segment(1.))
        for target in (run, restored):
            target.learner.add_segment(target.segment(1.))
        self.assertEqual(run.learner.update(), restored.learner.update())
        self.assertEqual(state_digest(run.learner.state_dict()), state_digest(restored.learner.state_dict()))

    def test_full_anchor_baseline_does_not_submit_alias_reference(self):
        run = session("anchor", "test")
        raw = run.env.observation()
        expected = facility_net_action_from_state(raw, run.producer.anchor_config, settings=run.producer.settings)
        different = run.reference.act(raw)
        self.assertFalse(np.array_equal(expected, different))
        event = run.step(before_step=lambda: None)
        self.assertEqual(event["audit"]["record"]["action"], tuple(expected.astype(float)))
        restored = session("anchor", "test")
        restored.load_state_dict(run.state_dict())
        self.assertEqual(state_digest(restored.state_dict()), state_digest(run.state_dict()))

    def test_reference_baseline_preserves_original_and_never_becomes_ppo_data(self):
        run = session("reference", "demonstration")
        expected = tuple(run.reference.act(run.env.observation()).astype(float))
        self.assertEqual(run.step(before_step=lambda: None)["audit"]["record"]["action"], expected)
        while not run.closed:
            run.step(before_step=lambda: None)
        with self.assertRaises(ValueError):
            run.segment(1.)

    def test_bad_split_modes_and_unbounded_step_rejected(self):
        for selection, split in (("sample", "test"), ("greedy", "training"),
                                 ("anchor", "demonstration"), ("greedy", "qualification")):
            with self.assertRaises(ValueError):
                session(selection, split)
        with self.assertRaises(TypeError):
            session().step()

    def test_step_audit_failure_rolls_back_env_sampler_not_external_spend(self):
        run, spent = session(), []
        before = state_digest(run.state_dict())
        def bad_step(env, action):
            raw, reward, done, info = fixture_step(env, action)
            return raw, reward + 1., done, info
        with patch.object(PatientConditionCapacityEnv, "step", bad_step), self.assertRaises(ValueError):
            run.step(before_step=lambda: spent.append(1))
        self.assertEqual(spent, [1])
        self.assertEqual(before, state_digest(run.state_dict()))

    def test_budget_failure_prevents_even_invented_step(self):
        run = session()
        before = state_digest(run.state_dict())
        def exhausted():
            raise RuntimeError("spent cap")
        with patch.object(PatientConditionCapacityEnv, "step", side_effect=AssertionError("not called")):
            with self.assertRaisesRegex(RuntimeError, "spent cap"):
                run.step(before_step=exhausted)
        self.assertEqual(before, state_digest(run.state_dict()))

    def test_external_cursor_weights_or_rng_drift_rejected(self):
        run = session()
        run.env.fake_counter += 1
        with self.assertRaises(ValueError):
            run.step(before_step=lambda: None)
        run = session()
        with torch.no_grad():
            next(run.learner.policy.parameters()).add_(1)
        with self.assertRaises(ValueError):
            run.step(before_step=lambda: None)
        run = session()
        run.step(before_step=lambda: None)
        run.learner.sampling_rng.manual_seed(2)
        with self.assertRaises(ValueError):
            session().load_state_dict(run.state_dict())

    def test_corrupt_checkpoint_receipts_support_costs_rng_fail_atomically(self):
        run = session()
        run.step(before_step=lambda: None)
        saved, before = run.state_dict(), state_digest(run.state_dict())
        changes = [lambda s: s.update(index=True), lambda s: s["events"].clear(),
                   lambda s: s["examples"][0].update(split="test"),
                   lambda s: s["events"][0]["audit"].update(route_count=100),
                   lambda s: s["events"][0]["info"].update(cost=1.),
                   lambda s: s["events"][0]["audit"]["decision"]["evaluation"].update(value=100.),
                   lambda s: s["kernel"]["sampling_rng"].zero_(),
                   lambda s: s.update(initial_rng=s["initial_rng"].float())]
        for mutate in changes:
            bad = copy.deepcopy(saved)
            mutate(bad)
            with self.assertRaises((ValueError, TypeError, RuntimeError)):
                run.load_state_dict(bad)
            self.assertEqual(before, state_digest(run.state_dict()))

    def test_checkpoint_file_roundtrip_no_overwrite_and_failed_publication(self):
        run = session()
        run.step(before_step=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invented.pt"
            run.save(path)
            restored = session()
            restored.load_state_dict(load_envelope(path))
            self.assertEqual(state_digest(restored.state_dict()), state_digest(run.state_dict()))
            with self.assertRaises(FileExistsError):
                run.save(path)
            bad = path.with_name("bad.pt")
            with patch("src.rl.candidate_patient_session.os.link", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    run.save(bad)
            self.assertFalse(bad.exists())
            envelope = torch.load(path, weights_only=True)
            envelope["sha256"] = "0" * 64
            torch.save(envelope, bad)
            with self.assertRaises(ValueError):
                load_envelope(bad)


if __name__ == "__main__":
    unittest.main()
