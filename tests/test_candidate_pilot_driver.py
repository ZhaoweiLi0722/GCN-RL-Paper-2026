"""P1 scheduling/recovery on invented tensors and fake accounting only."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl.candidate_imitation import CandidateImitationKernel, ImitationSettings
from src.rl.candidate_patient_session import CandidatePatientSession
from src.rl.candidate_pilot_driver import (
    CandidateContinuation, PilotSequence, assert_budget_ancestor, candidate_prototype,
    fork_initializer, initialization_kernel, pilot_jobs, qualify_initializer, required_models,
)
from src.rl.candidate_pilot_resources import PilotBudget, stream_manifest
from src.rl.candidate_ppo_kernel import CandidatePPOKernel
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import encode_arrays
from src.rl.networks import torch
from tests import test_candidate_patient_session as session_tests
from tests.test_candidate_patient_session import session, OPTIONS, InventedReference
from tests.test_candidate_ppo_kernel import settings
from tests.test_candidate_pilot_resources import config


class CandidatePilotDriverTests(unittest.TestCase):
    def setUp(self):
        session_tests.CandidatePatientSessionTests.setUp(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.counter = 0

    def make_driver(self, role="ppo", budget=None):
        cfg = config()
        cfg["objective"]["horizon"] = 4
        template = session()
        scope = f"block60/graph/{role}"
        if budget is None:
            self.counter += 1
            budget = PilotBudget(self.root / f"budget{self.counter}.jsonl", cfg)
            self.addCleanup(budget.close)
            budget.begin(scope)
        cfg[role].update(episodes_per_model=4, episodes_per_rollout=2, max_updates=2,
                         max_optimizer_steps=4, epochs=1, batch_size=4)
        if role == "ppo":
            kernel = CandidatePPOKernel(template.learner.policy, template.learner.contract,
                settings(epochs=1, batch_size=4, max_rollout_steps=8, max_updates=2,
                         max_optimizer_steps=4, gae_lambda=1.), enabled=True,
                mode="online", sampling_seed=119, shuffle_seed=121)
        else:
            kernel = CandidateImitationKernel(template.learner.policy, template.learner.contract,
                ImitationSettings(.001, .5, 4, 4, 2, "training"), enabled=True,
                sampling_seed=119, shuffle_seed=121)

        def factory(learner, episode, seed):
            fresh = session()
            return CandidatePatientSession(fresh.env, fresh.producer, InventedReference(fresh.producer),
                learner, enabled=True, options=OPTIONS, trajectory_id=f"invented/{role}/{episode}/{seed}",
                split="training", selection="sample", source_id="invented-campaign")
        return CandidateContinuation(kernel, budget, cfg, role=role, scope=scope,
                                     seeds=[101, 102, 103, 104], session_factory=factory)

    def reach_update(self, driver):
        while not driver.update_due:
            driver.step()

    def test_serial_order_and_no_test_entry_before_all_seals(self):
        jobs = pilot_jobs(config())
        self.assertEqual(len(jobs), 66)
        self.assertLess(jobs.index("qualification"), jobs.index("block60/graph/ppo"))
        self.assertLess(jobs.index("block62/flat/bc_continue"), jobs.index("seal_all_models"))
        sequence = PilotSequence(config(), self.root)
        with self.assertRaises(ValueError):
            sequence.begin("block60/graph/frozen/evaluation")
        sequence.begin(jobs[0])
        with self.assertRaises(ValueError):
            sequence.begin(jobs[0])
        evidence = self.root / "dummy.json"
        evidence.write_text('{"invented":true}')
        sequence.finish([evidence])
        with self.assertRaises(ValueError):
            sequence.begin(jobs[0])
        restored = PilotSequence(config(), self.root)
        restored.load_state_dict(sequence.state_dict())
        self.assertEqual(restored.state_dict(), sequence.state_dict())
        restored.fail("invented failure")
        with self.assertRaises(ValueError):
            restored.begin(jobs[1])

    def test_seal_barrier_checks_every_file_and_restore_cannot_skip_phase(self):
        sequence = PilotSequence(config(), self.root)
        evidence = self.root / "dummy.json"
        evidence.write_text('{"invented":true}')
        while sequence.next_job != "seal_all_models":
            sequence.begin(sequence.next_job)
            sequence.finish([evidence])
        sequence.begin("seal_all_models")
        with self.assertRaises(ValueError):
            sequence.finish([evidence])
        paths = {}
        for i, name in enumerate(required_models(config())):
            path = self.root / f"model{i}.invented"
            path.write_text(str(i))
            paths[name] = path
        sequence.seal_models(paths)
        sequence.finish([evidence])
        restored = PilotSequence(config(), self.root)
        restored.load_state_dict(sequence.state_dict())
        self.assertEqual(restored.next_job, "block60/graph/frozen/evaluation")
        state = sequence.state_dict()
        state["completed"].pop(2)
        with self.assertRaises(ValueError):
            restored.load_state_dict(state)
        next(iter(paths.values())).write_text("changed")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            sequence.begin(sequence.next_job)

    def test_mid_episode_restore_and_next_update_exact_for_both_roles(self):
        for role in ("ppo", "bc_continue"):
            left = self.make_driver(role)
            left.step()
            saved = left.state_dict()
            right = self.make_driver(role, left.budget)
            right.load_state_dict(saved)
            while not left.update_due:
                self.assertEqual(state_digest(encode_arrays(left.step())), state_digest(encode_arrays(right.step())))
            self.assertEqual(left.update(), right.update())
            self.assertEqual(state_digest(left.kernel.state_dict()), state_digest(right.kernel.state_dict()))
            self.assertEqual(left.episode, 2)
            self.assertEqual(left.updates, 1)

    def test_post_update_boundary_has_preupdate_collector_and_new_kernel(self):
        for role in ("ppo", "bc_continue"):
            left = self.make_driver(role)
            self.reach_update(left)
            left.update()
            saved = left.state_dict()
            self.assertNotEqual(state_digest(saved["last_closed"]["kernel"]), state_digest(saved["kernel"]))
            right = self.make_driver(role, left.budget)
            right.load_state_dict(saved)
            self.assertEqual(state_digest(encode_arrays(left.step())), state_digest(encode_arrays(right.step())))
            while not left.update_due:
                self.assertEqual(state_digest(encode_arrays(left.step())), state_digest(encode_arrays(right.step())))
            self.assertEqual(left.update(), right.update())
            self.assertTrue(left.done)
            self.assertTrue(right.done)
            with self.assertRaises(ValueError):
                left.step()

    def test_update_failure_rolls_back_kernel_but_never_external_budget(self):
        for role in ("ppo", "bc_continue"):
            run = self.make_driver(role)
            self.reach_update(run)
            saved = run.state_dict()
            original = torch.optim.Adam.step
            calls = []
            def interrupted(optimizer, *args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("invented second minibatch failure")
                return original(optimizer, *args, **kwargs)
            with patch.object(torch.optim.Adam, "step", interrupted), self.assertRaises(RuntimeError):
                run.update()
            self.assertEqual(state_digest(run.kernel.state_dict()), state_digest(saved["kernel"]))
            self.assertEqual(run.budget.counts["optimizer"], 2)
            run.load_state_dict(saved)
            self.assertEqual(run.budget.counts["optimizer"], 2)
            self.assertTrue(run.update_due)

    def test_checkpoint_counter_tampering_rejected_without_mutation(self):
        run = self.make_driver()
        run.step()
        saved, before = run.state_dict(), state_digest(run.kernel.state_dict())
        for mutate in (lambda s: s.update(episode=True), lambda s: s["budget"]["counts"].update(environment=0),
                       lambda s: s["budget"].update(ledger_sha256="0" * 64),
                       lambda s: s["active"].update(index=3)):
            bad = copy.deepcopy(saved)
            mutate(bad)
            with self.assertRaises((ValueError, RuntimeError)):
                run.load_state_dict(bad)
            self.assertEqual(state_digest(run.kernel.state_dict()), before)
        run.budget.finish()
        with self.assertRaises(ValueError):
            assert_budget_ancestor(saved["budget"], run.budget)

    def test_file_recovery_is_no_overwrite_and_exact(self):
        run = self.make_driver("bc_continue")
        self.reach_update(run)
        run.update()
        path = self.root / "boundary.pt"
        run.save(path)
        other = self.make_driver("bc_continue", run.budget)
        other.load(path)
        self.assertEqual(state_digest(run.state_dict()), state_digest(other.state_dict()))
        with self.assertRaises(FileExistsError):
            run.save(path)

    def test_qualification_is_frozen_split_checked_and_reports_support(self):
        run = session("reference", "qualification")
        while not run.closed:
            run.step(before_step=lambda: None)
        before = state_digest(run.learner.state_dict())
        output = qualify_initializer(run.learner, run.examples, expected_rows=4, minimum_agreement=.95)
        self.assertEqual(output["rows"], 4)
        self.assertEqual(output["passed"], output["agreement"] >= .95)
        self.assertEqual(before, state_digest(run.learner.state_dict()))
        bad = copy.deepcopy(run.examples)
        bad[0]["split"] = "test"
        with self.assertRaises(ValueError):
            qualify_initializer(run.learner, bad, expected_rows=4, minimum_agreement=.95)

    def test_forks_share_weights_sampler_but_discard_initialization_moments(self):
        cfg = config()
        cfg["initialization"].update(optimizer_steps_per_model=1, batch_size=2)
        streams = stream_manifest(cfg)
        run = session("reference", "demonstration")
        while not run.closed:
            run.step(before_step=lambda: None)
        kernel = initialization_kernel(run.learner.policy, run.learner.contract, cfg, streams, block=60, representation="graph")
        kernel.fit(run.examples, replacement_steps=1)
        self.assertTrue(kernel.optimizer.state_dict()["state"])
        qualification = {"passed": True, "kernel_sha256": state_digest(kernel.state_dict())}
        forks = fork_initializer(kernel, qualification, cfg, streams, block=60, representation="graph")
        self.assertIsNone(forks["frozen"].optimizer)
        for model in forks.values():
            self.assertEqual(model.policy.snapshot_sha256(), kernel.policy.snapshot_sha256())
        for role in ("ppo", "bc_continue"):
            self.assertEqual(forks[role].optimizer.state_dict()["state"], {})
        self.assertEqual(state_digest(forks["ppo"].sampling_rng.get_state()), state_digest(forks["bc_continue"].sampling_rng.get_state()))
        with self.assertRaises(ValueError):
            fork_initializer(kernel, {**qualification, "passed": False}, cfg, streams, block=60, representation="graph")

    def test_parameter_count_gate_and_graph_self_initial_tensors(self):
        cfg, template = config(), session()
        cfg["model"].update(encoder_width=3, head_width=5)
        for row in cfg["representations"][:2]:
            row["expected_parameters"] = sum(p.numel() for p in template.learner.policy.parameters())
        graph = candidate_prototype(template.producer, cfg, block=60, representation="graph")
        own = candidate_prototype(template.producer, cfg, block=60, representation="self_only")
        self.assertEqual(state_digest(graph.state_dict()), state_digest(own.state_dict()))
        cfg["representations"][0]["expected_parameters"] += 1
        with self.assertRaises(ValueError):
            candidate_prototype(template.producer, cfg, block=60, representation="graph")


if __name__ == "__main__":
    unittest.main()
