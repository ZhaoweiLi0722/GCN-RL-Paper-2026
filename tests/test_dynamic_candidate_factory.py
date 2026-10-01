"""Prospective wiring on invented public arrays; zero numerical optimizer calls."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.rl.dynamic_candidate_factory import (
    dynamic_template, dynamic_initializer, ppo_settings, qualify_dynamic_path,
    qualify_dynamic_outcomes, fork_dynamic_initializer,
)
from src.rl.dynamic_candidate_resources import dynamic_stream_manifest
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from tests.test_dynamic_candidate_session import FakeProducer, session


def proposal():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "specs/2026-10-01-adaptive-paper-delivery/pilot-budget-draft.json").read_text())
    # A test-only explicit binding, not an amendment of the original draft.
    config["candidate_message_graph"] = "specimen_routes"
    config["model_proposal"].update(graph="specimen_routes", initial_reference_bias=0.)
    return config


@unittest.skipIf(torch is None, "torch unavailable")
class DynamicCandidateFactoryTests(unittest.TestCase):
    def setUp(self):
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("real update forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)

    def test_settings_count_two_owners_and_fixed_rollout(self):
        settings = ppo_settings(proposal())
        self.assertEqual(settings.max_optimizer_steps, 256)
        self.assertEqual(settings.max_rollout_steps, 208)
        self.assertEqual(settings.max_updates, 8)
        self.assertEqual(settings.gae_lambda, 1.)

    def test_template_private_seeds_and_independent_value(self):
        config = proposal()
        streams = dynamic_stream_manifest(config)
        before = torch.get_rng_state().clone()
        one = dynamic_template(FakeProducer(), config, streams, 60)
        two = dynamic_template(FakeProducer(), config, streams, 60)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(one.policy.snapshot_sha256(), two.policy.snapshot_sha256())
        self.assertEqual(one.mode, "frozen")
        actor = {id(p) for p in one.policy.actor_parameters()}
        critic = {id(p) for p in one.policy.critic_parameters()}
        self.assertFalse(actor & critic)

    def test_no_implicit_graph_fallback(self):
        config = proposal()
        streams = dynamic_stream_manifest(config)
        config["model_proposal"]["graph"] = "physical_shared_relations"
        with self.assertRaisesRegex(ValueError, "no graph fallback"):
            dynamic_template(FakeProducer(), config, streams, 60)

    def test_unfitted_initializer_cannot_be_forked(self):
        config = proposal()
        streams = dynamic_stream_manifest(config)
        initializer = dynamic_initializer(dynamic_template(FakeProducer(), config, streams, 60), config, streams, 60)
        self.assertEqual(initializer.steps, 0)
        self.assertFalse(initializer.optimizer.state)
        receipt = {"passed": True, "kernel_sha256": state_digest(initializer.state_dict())}
        with self.assertRaisesRegex(ValueError, "complete qualified"):
            fork_dynamic_initializer(initializer, receipt, config, streams, 60)

    def collect_qualification(self):
        run = session(selection="reference", split="qualification", kind="initialization", bias=10.)
        while not run.closed:
            run.step(before_step=lambda: None)
        config = proposal()
        config["objective"]["horizon"] = 2
        config["qualification"]["fresh_worlds_per_block"] = 1
        return run, config

    def test_readonly_reference_agreement_and_multiclass(self):
        run, config = self.collect_qualification()
        charges = []
        result = qualify_dynamic_path(run.learner, run.examples, config,
                                      before_forward=lambda: charges.append("check"))
        self.assertTrue(result["passed"])
        self.assertEqual(result["agreement"], 1.)
        self.assertEqual(result["multiclass_agreement"], 1.)
        self.assertEqual(len(charges), 2)

    def test_qualification_rejects_duplicate_or_wrong_split(self):
        run, config = self.collect_qualification()
        for examples in ([run.examples[0]] * 2, copy.deepcopy(run.examples)):
            if examples[0] is not examples[1]:
                examples[0]["split"] = "test"
            with self.assertRaises(ValueError):
                qualify_dynamic_path(run.learner, examples, config, before_forward=lambda: None)

    def test_qualification_needs_multiclass_coverage(self):
        run, config = self.collect_qualification()
        config["qualification"]["minimum_multiclass_rows_each_path"] = 3
        result = qualify_dynamic_path(run.learner, run.examples, config, before_forward=lambda: None)
        self.assertFalse(result["passed"])

    def outcomes(self):
        return [dict(block=60, world_index=w, seed=100 + w, role=role, cost=100., losses=1.,
                     terminal_active=2., completions=3.)
                for w in range(2) for role in ("r4", "initializer_greedy")]

    def test_paired_outcome_gate_and_adverse_patient_direction(self):
        rows = self.outcomes()
        result = qualify_dynamic_outcomes(rows, proposal(), 60)
        self.assertTrue(result["passed"])
        self.assertFalse(result["clinical_noninferiority_claim"])
        rows[1]["cost"] -= 5
        rows[1]["losses"] += 1
        self.assertFalse(qualify_dynamic_outcomes(rows, proposal(), 60)["passed"])

    def test_outcome_integrity_is_not_a_soft_gate(self):
        rows = self.outcomes()
        variants = [rows[:-1], rows + [rows[0]]]
        for metric, value in (("cost", float("nan")), ("seed", 999)):
            bad = copy.deepcopy(rows)
            bad[1][metric] = value
            variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                qualify_dynamic_outcomes(bad, proposal(), 60)


if __name__ == "__main__":
    unittest.main()
