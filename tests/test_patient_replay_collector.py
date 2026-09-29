"""N4 real-environment mechanics tests, never training or measuring performance."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

from src.env.patient_capacity_planning import PatientConditionCapacityEnv, patient_env_config_from_dict
from src.models.matched_inputs import build_matched_inputs
from src.models.prospective_forward_agent import ForwardDecision, ProspectiveForwardAgent
from src.rl.networks import torch
from src.rl.patient_replay_collector import PatientObservationProducer, PatientReplayCollector, evidence_digest
from src.rl.prospective_adapter import completed_segment_windows, prepare_replay_batch
from src.rl.prospective_engineering_check import run_engineering_check

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return json.loads((ROOT / "experiments/configs/prospective_collector_engineering_20260929.json").read_text())


@unittest.skipIf(torch is None, "torch unavailable")
class PatientReplayCollectorTests(unittest.TestCase):
    def setUp(self):
        self.config = fixture()
        self.env = PatientConditionCapacityEnv(patient_env_config_from_dict(self.config["env"]), seed=0)
        self.producer = self.produce(self.env)

    def produce(self, env, **changes):
        return PatientObservationProducer(env, **(dict(enabled=True, gamma=.9, reward_scale=1e-5) | changes))

    def agent(self, architecture="graph", mode="physical", **changes):
        options = dict(enabled=True, architecture=architecture, message_mode=mode,
                       hidden_width=8, residual_scale=.1, seed=0)
        return ProspectiveForwardAgent(self.producer.contract, **(options | changes))

    def collector(self, **changes):
        options = dict(enabled=True, trajectory_id="unit", max_steps=4, horizon_end="truncation")
        return PatientReplayCollector(self.env, self.producer, **(options | changes))

    def decision(self):
        return self.agent().select_action(*self.producer.observe(self.env.observation()))

    def test_explicit_opt_in_required_for_all_components(self):
        for call in (lambda: self.produce(self.env, enabled=False), lambda: self.agent(enabled=False),
                     lambda: self.collector(enabled=False)):
            with self.assertRaises(ValueError):
                call()

    def test_raw_roundtrip_and_anchor_match_real_helper(self):
        from src.baselines.heuristics import get_heuristic_class
        raw = self.env.observation()
        obs, anchor = self.producer.observe(raw)
        np.testing.assert_array_equal(raw, self.producer.unpack_raw(obs)[0])
        expected = get_heuristic_class("mdl2")().select_action(raw, env=self.env)
        np.testing.assert_array_equal(anchor[0].numpy(), expected)

    def test_optional_observation_blocks_lossless(self):
        options = dict(self.config["env"], include_demand_forecast_state=True,
                       include_transfer_pipeline_state=True, include_demand_history_state=True,
                       include_demand_sequence_state=True, demand_sequence_length=3)
        env = PatientConditionCapacityEnv(patient_env_config_from_dict(options), seed=0)
        producer = self.produce(env)
        for _ in range(3):
            raw = env.observation()
            obs, _ = producer.observe(raw)
            np.testing.assert_array_equal(producer.unpack_raw(obs)[0], raw)
            env.step(env.noop_action())

    def test_reject_unsupported_relations_and_flags(self):
        for change in ({"capacity_edges": [[0, 2]]}, {"include_central_capacity_hub": True},
                       {"enable_overtime_control": True}, {"reagent_purchase_lead_time": 2}):
            env = PatientConditionCapacityEnv(patient_env_config_from_dict(self.config["env"] | change), seed=0)
            with self.assertRaises(ValueError):
                self.produce(env)

    def test_observation_rejects_hidden_snapshot_wrong_width_dtype_nan(self):
        bad = (self.env.state_dict(), self.env.observation()[:-1], self.env.observation().astype(np.float64),
               np.full_like(self.env.observation(), np.nan))
        for value in bad:
            with self.assertRaises((TypeError, ValueError)):
                self.producer.observe(value)

    def test_rng_metadata_is_not_neural_input(self):
        raw = self.env.observation()
        obs1, anchor1 = self.producer.observe(raw)
        token1 = evidence_digest(self.env.state_dict())
        self.env.rng.random()
        self.assertNotEqual(token1, evidence_digest(self.env.state_dict()))
        obs2, anchor2 = self.producer.observe(self.env.observation())
        torch.testing.assert_close(obs1.nodes, obs2.nodes, rtol=0, atol=0)
        torch.testing.assert_close(anchor1, anchor2, rtol=0, atol=0)

    def test_neural_views_have_equal_information_under_message_ablation(self):
        obs, anchor = self.producer.observe(self.env.observation())
        for role in ("actor", "critic", "gate"):
            kwargs = {} if role == "actor" else {"action": anchor + .01}
            if role == "gate":
                kwargs["detach_proposal"] = True
            views = [build_matched_inputs(obs, self.producer.contract.inputs, anchor, role=role,
                                          message_mode=mode, **kwargs) for mode in ("physical", "self_only")]
            torch.testing.assert_close(views[0].flat, views[1].flat, rtol=0, atol=0)
            self.assertFalse(torch.equal(views[0].message_adjacency, views[1].message_adjacency))
            for arch in ("graph", "flat"):
                head = getattr(self.agent(arch), role)
                self.assertTrue(torch.isfinite(head(views[0])).all())

    def test_gate_sees_proposal_in_both_architectures(self):
        for arch in ("graph", "flat"):
            agent, captured = self.agent(arch), []
            hook = agent.gate.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
            obs, anchor = self.producer.observe(self.env.observation())
            decision = agent.select_action(obs, anchor)
            hook.remove()
            np.testing.assert_array_equal(captured[0].context[0, -self.env.action_size:].numpy(),
                                          decision.proposal - anchor[0].numpy())

    def test_clone_reproduction_receipts_and_integer_execution(self):
        collector = self.collector(max_steps=1)
        proposal = self.env.noop_action()
        proposal[0], proposal[1] = -.8, .8
        receipt = collector.step(ForwardDecision(proposal, np.ones_like(proposal), proposal))
        self.assertTrue(receipt.clone_verified)
        np.testing.assert_array_equal(receipt.record.action, proposal)
        self.assertGreater(receipt.execution["specimen_route_count"], 0)
        self.assertFalse(np.array_equal(receipt.execution["specimen_requested_integer_net"], proposal[:3]))
        self.assertEqual(receipt.record.raw_reward, -receipt.execution["cost"])
        self.assertTrue(receipt.record.truncated)

    def test_invalid_decision_does_not_step_env(self):
        collector = self.collector()
        decision = self.decision()
        for bad in (replace(decision, request=np.full(self.env.action_size, np.nan)),
                    replace(decision, gate=np.ones(self.env.action_size) * 2),
                    replace(decision, proposal=np.zeros(1)),
                    replace(decision, request=-decision.request)):
            token = evidence_digest(self.env.state_dict())
            with self.assertRaises(ValueError):
                collector.step(bad)
            self.assertEqual(token, evidence_digest(self.env.state_dict()))

    def test_reset_rng_and_config_drift_rejected(self):
        collector = self.collector()
        decision = self.decision()
        self.env.rng.random()
        with self.assertRaisesRegex(ValueError, "drift"):
            collector.step(decision)
        self.env.reset(seed=0)
        collector = self.collector()
        collector.step(self.decision())
        self.env.reset(seed=0)
        with self.assertRaisesRegex(ValueError, "drift"):
            collector.step(self.decision())
        self.env.env_config = replace(self.env.env_config, weight_patient_lost=123.)
        with self.assertRaisesRegex(ValueError, "config"):
            self.producer.check_environment(self.env)

    def test_boundary_required_and_no_step_after_closed_or_resume(self):
        with self.assertRaises(ValueError):
            self.collector(horizon_end="guess")
        collector = self.collector(max_steps=1)
        collector.step(self.decision())
        token = evidence_digest(self.env.state_dict())
        with self.assertRaisesRegex(ValueError, "closed"):
            collector.step(self.decision())
        self.assertEqual(token, evidence_digest(self.env.state_dict()))
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.collector()

    def test_endpoint_anchor_and_action_are_used_for_target(self):
        agent, collector = self.agent(), self.collector()
        records = []
        while not collector.closed:
            obs, anchor = self.producer.observe(self.env.observation())
            records.append(collector.step(agent.select_action(obs, anchor)).record)
        windows = completed_segment_windows(records, self.producer.contract.replay, max_steps=3)
        batch = prepare_replay_batch(windows, self.producer.contract, dtype=torch.float32,
                                     device="cpu", message_mode="physical")
        captured = []
        hook = agent.target_critic.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        before = agent.weights_digest()
        _, next_q, targets = agent.replay_forward(batch)
        hook.remove()
        _, _, expected_request = agent.policy_tensors(batch.next_actor, target=True)
        torch.testing.assert_close(captured[0].context[:, -self.env.action_size:], expected_request, rtol=0, atol=0)
        torch.testing.assert_close(captured[0].flat[:, :-self.env.action_size], batch.next_actor.flat, rtol=0, atol=0)
        for i, window in enumerate(windows):
            np.testing.assert_array_equal(batch.next_actor.flat[i].numpy(), window[-1].next_state)
            np.testing.assert_array_equal(batch.current_critic.context[i, -self.env.action_size:].numpy(), window[0].action)
        torch.testing.assert_close(targets, batch.rewards + batch.bootstrap_discounts * next_q)
        self.assertEqual(before, agent.weights_digest())
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in agent.parameters()))

    def test_wrong_schema_and_reward_contract_rejected(self):
        obs, anchor = self.producer.observe(self.env.observation())
        with self.assertRaises(ValueError):
            self.agent().select_action(replace(obs, schema=replace(obs.schema, definition_id="wrong")), anchor)
        receipt = self.collector(max_steps=1).step(self.decision())
        wrong = replace(self.producer.contract, replay=replace(self.producer.contract.replay, gamma=.5))
        with self.assertRaises(ValueError):
            prepare_replay_batch([(receipt.record,)], wrong, dtype=torch.float32, device="cpu", message_mode="physical")

    def test_initialization_preserves_rng_and_copies_targets(self):
        before = torch.random.get_rng_state().clone()
        agent = self.agent()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for name in ("actor", "critic", "gate"):
            for original, target in zip(getattr(agent, name).parameters(), getattr(agent, "target_" + name).parameters()):
                self.assertTrue(torch.equal(original, target))
                self.assertIsNot(original, target)

    def test_full_bounded_harness_is_reproducible(self):
        first = run_engineering_check(self.config)
        second = run_engineering_check(copy.deepcopy(self.config))
        self.assertEqual(evidence_digest(first), evidence_digest(second))
        self.assertEqual(first["decision_count"], 36)
        self.assertEqual(first["optimizer_updates"], 0)
        self.assertFalse(first["online_gain_claimed"])
        self.assertTrue(all(c["all_clone_steps_exact"] for c in first["cases"]))

    def test_source_inventory_excludes_dynamic_pseudo_files(self):
        from experiments.scripts.check_prospective_collector import local_source_hashes
        modules = [SimpleNamespace(__file__="_ops.py"), SimpleNamespace(__file__="<generated>"),
                   SimpleNamespace(__file__=str(Path(__file__).resolve())), SimpleNamespace()]
        hashes = local_source_hashes(modules, ROOT)
        self.assertEqual(list(hashes), ["tests/test_patient_replay_collector.py"])
        self.assertEqual(len(next(iter(hashes.values()))), 64)


if __name__ == "__main__":
    unittest.main()
