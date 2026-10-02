"""Artificial tensors and array clocks only; scientific calls are forbidden."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.rl import cohort_public, patient_replay_collector
from src.rl.cohort_public import CohortPrefixSession
from src.rl.paired_cohort_backend import ReferenceContextSession, PairedCohortBackend
from src.rl.paired_cohort_episode import run_episode, recycle_prefix_template
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import decode_arrays
from tests.test_cohort_campaign import Recorder
from tests.test_cohort_public import ArrayLayout, ArrayCohort, fixture, kernel, FakeReference, OPTIONS
from tests.test_dynamic_candidate_session import fake_info


class Budget:
    def __init__(self):
        self.charges = []
    def check(self):
        return 0.
    def debit_environment(self, kind):
        self.charges.append(kind)
    def snapshot(self):
        return dict(charges=list(self.charges))


class PairedBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
    def setUp(self):
        for cls, method in ((PatientConditionCapacityEnv, "__init__"), (PatientConditionCapacityEnv, "step"),
                            (torch.optim.Adam, "step"), (torch.optim.SGD, "step")):
            guard = patch.object(cls, method, side_effect=AssertionError("real science forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)
        for module in (cohort_public, patient_replay_collector):
            guard = patch.object(module, "PatientConditionCapacityEnv", ArrayLayout)
            guard.start()
            self.addCleanup(guard.stop)
        original = ArrayLayout.step
        def step(env, action):
            if not env._cohort_closed:
                return original(env, action)
            env.actions.append(action.copy())
            env.t += 1
            env.raw[-1] = env.t / env.config.episode_horizon
            info = fake_info(action)
            return env.observation(), -info["cost"], env.t == env.cohort_spec.enrollment_steps + env.cohort_spec.accounting_steps, info
        guard = patch.object(ArrayLayout, "step", step)
        guard.start()
        self.addCleanup(guard.stop)

    def session(self, horizon=2):
        _, _, env, producer = fixture(horizon, accounting=11)
        owner = kernel(producer, "frozen")
        return ReferenceContextSession(env, producer, FakeReference(producer), owner, enabled=True,
            options=OPTIONS, trajectory_id="invented/reference", source_id="invented", environment_seed=101)

    def test_new_reference_training_contract_and_exact_restore_without_old_guard_bypass(self):
        session = self.session()
        before = state_digest(session.learner.state_dict())
        with self.assertRaises(ValueError):
            CohortPrefixSession(session.env, session.producer, session.reference, session.learner,
                enabled=True, options=OPTIONS, trajectory_id="invented", split="training", selection="reference",
                source_id="invented", environment_seed=101)
        session.step(before_step=lambda: None)
        restored = self.session()
        restored.load_state_dict(session.state_dict())
        self.assertEqual(state_digest(restored.state_dict()), state_digest(session.state_dict()))
        self.assertEqual(restored.examples[0]["split"], "training")
        self.assertEqual(restored.selection, "reference")
        self.assertEqual(state_digest(restored.learner.state_dict()), before)
        with self.assertRaisesRegex(ValueError, "cannot enter"):
            restored.segment(1.)

    def test_actual_episode_dispatch_captures_4_20_36_then_common_tail_and_recycles(self):
        session, budget = self.session(52), Budget()
        class Backend:
            def session(self, *args, **kwargs):
                return session
        with tempfile.TemporaryDirectory() as temp:
            result = run_episode(temp, {}, dict(context_after_prefix_steps=[4, 20, 36], economic_endpoint=63),
                dict(namespace="invented"), Backend(), session.learner, budget, admit=lambda _: None,
                block=60, role="r4", world=0, seed=101, split="training", selection="reference",
                name="invented/train", recorder_type=Recorder)
            self.assertEqual(set(result["contexts"]), {(0, 4), (0, 20), (0, 36)})
            self.assertEqual(budget.charges, ["trajectory"] * 63)
            self.assertEqual(result["environment"].t, 63)
            for (_, t), saved in result["contexts"].items():
                context = decode_arrays(saved)
                self.assertEqual(context["environment"]["t"], t)
                self.assertEqual(context["public_example"]["split"], "training")
            template = recycle_prefix_template(result["environment"], result["contexts"][0, 36])
            self.assertIs(template, result["environment"])
            self.assertEqual(template.t, 36)
            self.assertFalse(template._cohort_closed)
            self.assertTrue((Path(temp) / "payload/contexts/block60/cohort0/after36.pt").is_file())

    def test_counted_full_same_start_clone_no_extra_fresh_environment(self):
        session, budget = self.session(52), Budget()
        session = CohortPrefixSession(session.env, session.producer, session.reference, session.learner,
            enabled=True, options=OPTIONS, trajectory_id="invented/preflight", split="preflight", selection="reference",
            source_id="invented", environment_seed=101)
        calls = []
        class Backend:
            def session(self, *args, **kwargs):
                calls.append("fresh")
                return session
        with tempfile.TemporaryDirectory() as temp:
            result = run_episode(temp, {}, dict(context_after_prefix_steps=[4, 20, 36], economic_endpoint=63),
                dict(namespace="invented"), Backend(), session.learner, budget, admit=calls.append,
                block=60, role="r4", world=0, seed=101, split="preflight", selection="reference",
                name="invented/preflight", clone=True, recorder_type=Recorder)
            self.assertEqual(calls.count("fresh"), 1)
            self.assertEqual(calls.count("preflight_clone"), 1)
            self.assertEqual(budget.charges.count("trajectory"), 63)
            self.assertEqual(budget.charges.count("clone"), 63)
            self.assertIsNotNone(result["clone_index"])

    def test_native_backend_is_disabled_without_execution_admission(self):
        backend = PairedCohortBackend(".", {}, {})
        with self.assertRaises(PermissionError):
            backend.session(60, None, 101, trajectory="x", split="training", selection="reference")

    def test_native_training_factory_retains_true_split_and_large_seed(self):
        _, layout, env, producer = fixture(52, accounting=11)
        seed = 2**110 + 123
        admission = []
        cfg = dict(candidate_support=dict(options=OPTIONS), totals=dict(fresh_episode_builds=231),
            objective=dict(scenario="invented"),
            cohort_proposal=dict(enrollment_steps=52, patient_resolution_steps=2, accounting_steps=11))
        backend = PairedCohortBackend(".", cfg, dict(namespace="invented", environment={"60": {"context": [seed]}}),
                                     admit_real_calls=admission.append)
        backend.contexts[60] = ({"env": {}, "scenario": "invented"}, layout, FakeReference(producer))
        with patch("src.env.patient_capacity_planning.patient_env_config_from_dict", return_value=env.env_config), \
             patch("src.rl.experiment.apply_graph_ablation", side_effect=lambda base, _: base), \
             patch("src.rl.frozen_value_probe.assert_scenario", return_value={}), \
             patch("src.rl.paired_cohort_backend.cohort_environment_class", return_value=ArrayCohort):
            session = backend.session(60, kernel(producer, "frozen"), seed, trajectory="invented/train",
                                      split="training", selection="reference")
        self.assertIs(type(session), ReferenceContextSession)
        self.assertEqual(session.env._episode_seed, seed)
        self.assertEqual(session.manifest["split"], "training")
        self.assertEqual(session.manifest["environment_seed"], seed)
        self.assertEqual(admission, ["episode_build"])
        self.assertEqual(backend.episode_builds, 1)

    def test_episode_failure_keeps_spend_and_partial_state(self):
        session, budget = self.session(52), Budget()
        class Backend:
            def session(self, *args, **kwargs):
                return session
        def fail_after_charge(kind):
            budget.charges.append(kind)
            raise RuntimeError("invented exhausted budget")
        budget.debit_environment = fail_after_charge
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                run_episode(temp, {}, dict(context_after_prefix_steps=[4, 20, 36], economic_endpoint=63),
                    dict(namespace="invented"), Backend(), session.learner, budget, admit=lambda _: None,
                    block=60, role="r4", world=0, seed=101, split="training", selection="reference",
                    name="invented/failure", recorder_type=Recorder)
            self.assertEqual(budget.charges, ["trajectory"])
            self.assertTrue((Path(temp) / "launcher/episode-failures/invented/failure/failure.json").is_file())
            self.assertIsNotNone(session._failure)


if __name__ == "__main__":
    unittest.main()
