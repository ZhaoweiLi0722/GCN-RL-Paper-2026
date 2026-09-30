"""Complete P1 phase wiring on tiny invented bookkeeping, never patient dynamics."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.models.candidate_policy import CandidatePolicy
from src.rl.candidate_patient_session import CandidatePatientSession
from src.rl.candidate_pilot_campaign import PilotCampaign, global_rng_state
from src.rl.candidate_pilot_resources import PilotBudget, stream_manifest
from src.rl.candidate_pilot_verification import CAUSES
from src.rl.prospective_ddpg_kernel import state_digest
from tests import test_candidate_patient_session as fixtures
from tests.test_candidate_pilot_resources import config
from tests.test_candidate_pilot_verification import patient


class InventedBackend:
    def __init__(self, cfg):
        self.cfg, self.producers, self.calls = cfg, {}, []

    def prepare(self, block, seed):
        self.producers[block] = fixtures.session().producer
        return {"invented_not_clinical": True, "block": block}

    def producer(self, block):
        return self.producers[block]

    def session(self, block, kernel, seed, *, trajectory, split, selection):
        self.calls.append((block, split, seed, trajectory))
        template = fixtures.session()
        template.env.fake_seed = seed
        return CandidatePatientSession(template.env, self.producer(block), fixtures.InventedReference(self.producer(block)),
            kernel, enabled=True, options=self.cfg["candidate_support"]["options"], trajectory_id=trajectory,
            split=split, selection=selection, source_id=self.cfg["rng"]["namespace"])


class CandidatePilotCampaignTests(unittest.TestCase):
    def setUp(self):
        fixtures.CandidatePatientSessionTests.setUp(self)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve() / "campaign"
        self.root.mkdir()
        self.cfg = config()
        self.cfg["objective"].update(horizon=4, reward_scale=.01, num_facilities=2)
        self.cfg["model"].update(encoder_width=3, head_width=5)
        template = fixtures.session()
        self.cfg["objective"].update(action_width=template.env.action_size, raw_state_width=template.env.observation_size)
        self.cfg["objective"]["transfer_scale"] = template.env.config.max_specimen_transfer
        self.cfg["candidate_support"]["options"] = fixtures.OPTIONS
        for rep in self.cfg["representations"]:
            model = CandidatePolicy(template.producer.contract.inputs, enabled=True, architecture=rep["architecture"],
                message_mode=rep["message_mode"], encoder_width=3, head_width=5, seed=19)
            rep["expected_parameters"] = sum(p.numel() for p in model.parameters())
        self.cfg["initialization"].update(demonstration_episodes_per_block=1, qualification_episodes_per_block=1,
            optimizer_steps_per_model=1, batch_size=4, minimum_greedy_reference_agreement_each_model=0.)
        for role in ("ppo", "bc_continue"):
            self.cfg[role].update(episodes_per_model=2, episodes_per_rollout=1, max_updates=2,
                                  max_optimizer_steps=2, epochs=1, batch_size=4)
        self.cfg["ppo"]["max_rollout_steps"] = 4
        self.cfg["evaluation"].update(episodes_per_policy=1, total_episodes=33, paired_world_bootstrap_draws=10)
        self.cfg["rng"]["ordinal_ranges"] = {"preflight": [0, 11], "demonstration": [12, 14],
            "qualification": [15, 17], "training": [18, 23], "test": [24, 26]}
        self.cfg["caps"]["environment_steps"] = {"preflight_including_clones": 54, "demonstrations": 12,
            "qualification": 12, "ppo": 72, "bc_continue": 72, "evaluation": 132}
        self.cfg["caps"]["optimizer_steps"] = {"initialization": 9, "ppo": 18, "bc_continue": 18}
        self.cfg["dropbox_directory_proposed"] = str(self.root.parent / "fake-local-backup")
        def state(env):
            value = fixtures.fixture_state(env)
            value.update(scalars={"t": env.t, "_episode_seed": getattr(env, "fake_seed", 123),
                "cumulative_enrolled": 12, "cumulative_lost": env.t, "cumulative_served": env.t},
                patients={f"p{i}": patient(f"p{i}", "lost" if i % 2 else "delivered")
                    if i < 2 * env.t else patient(f"p{i}") for i in range(12)},
                patient_queues=[[f"p{i}" for i in range(2 * env.t, 12)], []],
                in_production_patients=[[[], []], [[], []]], specimen_transits=[], product_return_transits=[])
            return value
        def step(env, action):
            raw, reward, done, info = fixtures.fixture_step(env, action)
            info.update({k: np.zeros(2) for k in CAUSES})
            info.update(demand=np.zeros(2), patients_lost_waiting_ineligible=np.array([1., 0.]),
                identity_active_count=12 - 2 * env.t, identity_terminal_count=2 * env.t,
                waiting_patients=np.array([12 - 2 * env.t, 0.]), in_production_patients=np.zeros(2),
                specimen_in_transit=np.zeros(2), completion_service_level=env.t / 12)
            return raw, reward, done, info
        for guard in (patch.object(PatientConditionCapacityEnv, "state_dict", state),
                      patch.object(PatientConditionCapacityEnv, "step", step)):
            guard.start()
            self.addCleanup(guard.stop)

    def campaign(self):
        streams = stream_manifest(self.cfg)
        budget = PilotBudget(self.root / "launcher/budget.jsonl", self.cfg)
        self.addCleanup(budget.close)
        backend = InventedBackend(self.cfg)
        return PilotCampaign(self.root, self.cfg, streams, budget, backend)

    def test_complete_fake_application_real_kernels_recorder_verifier_and_archive(self):
        run = self.campaign()
        with patch("builtins.print"):
            result = run.run()
        if result:
            self.fail((self.root / "launcher/failure.json").read_text())
        self.assertEqual(len(run.sequence.completed), 66)
        self.assertEqual(run.budget.counts, {"environment": 354, "optimizer": 45})
        self.assertEqual(len(run.test_index), 33)
        first_test = next(i for i, row in enumerate(run.backend.calls) if row[1] == "test")
        self.assertTrue(all(row[1] == "test" for row in run.backend.calls[first_test:]))
        self.assertEqual(len(run.sequence.seals), 27)
        report = json.loads((self.root / "payload/independent-verification.json").read_text())
        self.assertEqual(report["analysis"]["decision"], "limited_negative_or_inconclusive")
        self.assertEqual(report["analysis"]["evaluation_episodes"], 33)
        self.assertTrue((self.root / "launcher/completed.json").exists())
        receipt = json.loads((self.root / "launcher/archive-receipt.json").read_text())
        self.assertFalse(receipt["cloud_sync_verified"])
        self.assertTrue(all(c["local_copy_verified"] for c in receipt["local_copies"]))

    def test_qualification_failure_preserves_all_nine_scores_and_never_trains_or_tests(self):
        run = self.campaign()
        def rejected(kernel, examples, **kwargs):
            return {"passed": False, "agreement": .5, "kernel_sha256": state_digest(kernel.state_dict())}
        with patch("src.rl.candidate_pilot_campaign.qualify_initializer", side_effect=rejected), patch("builtins.print"):
            self.assertEqual(run.run(), 1)
        self.assertEqual(len(run.qualified), 9)
        self.assertEqual(run.models, {})
        self.assertFalse(any(row[1] in ("training", "test") for row in run.backend.calls))
        self.assertEqual(run.budget.counts, {"environment": 78, "optimizer": 9})
        self.assertTrue((self.root / "launcher/failure-state.pt").exists())
        self.assertFalse((self.root / "launcher/completed.json").exists())
        with self.assertRaises(ValueError):
            run.sequence.begin("qualification")

    def test_preflight_clone_failure_is_terminal_and_spend_retained(self):
        run = self.campaign()
        original = CandidatePatientSession.load_state_dict
        def corrupted(session, state):
            original(session, state)
            session.env.fake_counter += 1
        with patch.object(CandidatePatientSession, "load_state_dict", corrupted), patch("builtins.print"):
            self.assertEqual(run.run(), 1)
        self.assertEqual(run.budget.counts["optimizer"], 0)
        self.assertGreater(run.budget.counts["environment"], 0)
        self.assertEqual(run.sequence.failure["job"], "preflight_including_clones")
        self.assertFalse(any(row[1] != "preflight" for row in run.backend.calls))

    def test_campaign_boundary_restore_is_atomic_and_cannot_refund_budget(self):
        run = self.campaign()
        run.sequence.begin("preflight_including_clones")
        run.budget.begin("preflight_including_clones")
        run.preflight()
        saved = run.checkpoint_state()
        before = state_digest(saved)
        run.restore_checkpoint_state(saved)
        self.assertEqual(state_digest(run.checkpoint_state()), before)
        bad = copy.deepcopy(saved)
        bad["budget"]["counts"]["environment"] = 0
        with self.assertRaises(ValueError):
            run.restore_checkpoint_state(bad)
        self.assertEqual(state_digest(run.checkpoint_state()), before)
        self.assertEqual(state_digest(global_rng_state()), state_digest(saved["global_rng"]))

    def test_campaign_mid_collection_and_update_restore_keep_recorder_and_kernel_bound(self):
        from src.rl.candidate_pilot_driver import CandidateContinuation
        run = self.campaign()
        with patch("builtins.print"):
            while run.sequence.next_job != "block60/graph/ppo":
                job = run.sequence.next_job
                run.sequence.begin(job)
                run.budget.begin(job)
                evidence = run.dispatch(job)
                run.budget.finish()
                run.sequence.finish(evidence)
        scope = run.sequence.next_job
        run.sequence.begin(scope)
        run.budget.begin(scope)
        run.continuation = CandidateContinuation(run.models[scope], run.budget, run.config, role="ppo", scope=scope,
            seeds=run.streams["environment"]["training"]["60"], session_factory=run.continuation_factory(scope))
        run.live_session = run.continuation.start_episode()
        run.recorder = run.recorder_type(run.root, "payload/episodes/recovery-fixture", run.live_session, run.config,
            block=60, representation="graph", role="ppo", world_index=0, seed=run.continuation.seeds[0])
        self.addCleanup(run.recorder.handle.close)
        run.recorder.append(run.continuation.step())
        saved = run.checkpoint_state()
        run.restore_checkpoint_state(saved)
        self.assertEqual(state_digest(run.checkpoint_state()), state_digest(saved))
        self.assertIs(run.recorder.session, run.continuation.active)
        self.assertIs(run.models[scope], run.continuation.kernel)
        while not run.continuation.update_due:
            run.recorder.append(run.continuation.step())
        run.recorder.finish()
        run.recorder, run.live_session = None, None
        run.continuation.update()
        saved = run.checkpoint_state()
        run.restore_checkpoint_state(saved)
        self.assertEqual(state_digest(run.checkpoint_state()), state_digest(saved))
        self.assertIs(run.models[scope], run.continuation.kernel)
        bad = copy.deepcopy(saved)
        bad["global_rng"]["torch"] = bad["global_rng"]["torch"][:2]
        with self.assertRaises(RuntimeError):
            run.restore_checkpoint_state(bad)
        self.assertEqual(state_digest(run.checkpoint_state()), state_digest(saved))


if __name__ == "__main__":
    unittest.main()
