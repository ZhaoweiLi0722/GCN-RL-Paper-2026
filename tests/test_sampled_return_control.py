"""Artificial tensors only. Real optimizer and patient calls are forbidden."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.rl.actor_positive_control import rng_snapshot
from src.rl.candidate_patient_session import load_envelope, save_envelope
from src.rl.candidate_calibration_engineering import scores
from src.rl.networks import torch
from src.rl.sampled_return_control import (CONFIG, ArtificialBudget, SampledReturns, collect_samples,
    make_models, observed_reward, public_fixture, sampled_losses, snapshot, update_pair)
from src.rl.sampled_return_verification import evaluation_metrics, verify_charge, verify_sample_batch
from src.rl import sampled_return_campaign as campaign
from src.rl.sampled_return_packet_verification import ReceiptReader, verify_optimizer
from src.rl.candidate_pilot_recording import write_json_once


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, "torch unavailable")
class SampledReturnTests(unittest.TestCase):
    def setUp(self):
        for target in ("torch.optim.Adam.step", "torch.optim.SGD.step"):
            guard = patch(target, side_effect=AssertionError("real optimizer calls forbidden"))
            guard.start()
            self.addCleanup(guard.stop)
        for method in ("__init__", "reset", "step"):
            guard = patch.object(PatientConditionCapacityEnv, method, side_effect=AssertionError("patient calls forbidden"))
            guard.start()
            self.addCleanup(guard.stop)
        self.cfg = json.loads((ROOT / CONFIG).read_text())
        self.obs, self.bank = public_fixture(self.cfg, self.cfg["training_times"])
        self.actor, self.critic = make_models(self.cfg, self.bank, "graph", 101)
        self.budget = ArtificialBudget(self.cfg)
        self.events = []

    def collect(self, rng=None, reward=None):
        return collect_samples(self.actor, self.critic, self.obs, self.bank,
            sampling_rng=rng if rng is not None else torch.Generator().manual_seed(2), repetitions=4,
            reward_observer=reward or (lambda o, b, a: observed_reward(self.cfg, o, b, a)),
            charge_observations=lambda n: self.budget.charge("graph-101", "observations", n, self.events.append))

    def test_pending_authorization_and_exact_budget_arithmetic(self):
        self.assertFalse(self.cfg["artificial_execution_authorized"])
        self.assertFalse(self.cfg["scientific_execution_authorized"])
        self.assertFalse(self.cfg["execution_implementation_ready"])
        self.assertEqual(self.cfg["patient_calls_authorized"], 0)
        samples = len(self.cfg["training_times"])*len(self.cfg["cues"])*4*32
        self.assertEqual(samples, 1536)
        self.assertEqual(samples*9, 13824)
        self.assertEqual((samples//12)*2*9, 2304)
        old = json.loads((ROOT / "experiments/configs/candidate_actor_positive_control_20261001.json").read_text())
        for name in ("units", "request_values", "training_times", "heldout_times", "invented_return"):
            self.assertEqual(self.cfg[name], old[name])

    def test_models_disjoint_and_critic_does_not_consume_rng(self):
        before = rng_snapshot()
        for rep in self.cfg["representations"]:
            actor, critic = make_models(self.cfg, self.bank, rep, 101)
            self.assertFalse({p.data_ptr() for p in actor.parameters()} & {p.data_ptr() for p in critic.parameters()})
            self.assertEqual(float(critic(self.obs[0], self.bank).detach()), 0.)
        self.assertTrue(torch.equal(before["torch_cpu"], torch.get_rng_state()))

    def test_task_returns_only_selected_action_outcome(self):
        for index in (0, 6):
            obs = self.obs[index]
            rewards = [observed_reward(self.cfg, obs, self.bank, action) for action in range(6)]
            self.assertAlmostEqual(max(rewards)-min(rewards), .25)
            self.assertEqual(sum(x == max(rewards) for x in rewards), 1)
        for action in (-1, 6, True):
            with self.assertRaises(ValueError):
                observed_reward(self.cfg, self.obs[0], self.bank, action)

    def test_sampling_charged_before_observer_private_rng_and_receipts(self):
        before = torch.get_rng_state().clone()
        calls = []
        def observer(o, b, a):
            self.assertEqual(self.budget.observations, 48)
            calls.append(a)
            return observed_reward(self.cfg, o, b, a)
        rng = torch.Generator().manual_seed(610101)
        saved_rng = rng.get_state().clone()
        batch = self.collect(rng, observer)
        self.assertEqual(len(calls), 48)
        self.assertEqual(batch.actions.tolist(), calls)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertFalse(torch.equal(saved_rng, rng.get_state()))
        recreated = torch.Generator()
        recreated.set_state(saved_rng)
        torch.testing.assert_close(batch.actions, self.collect(recreated).actions)
        self.assertEqual(set(vars(batch)), {"context_indices", "actions", "old_log_probs", "old_values", "returns", "advantages", "behavior_logits"})

    def test_arbitrary_observed_returns_not_oracle_labels_drive_advantages(self):
        batch = self.collect(reward=lambda o, b, a: 7.)
        self.assertTrue((batch.returns == 7.).all())
        self.assertTrue((batch.advantages == 0.).all())
        with self.assertRaises(TypeError):
            SampledReturns(**vars(batch), q_values=torch.zeros(48, 6))

    def test_corrupted_probabilities_advantages_shapes_and_support_rejected(self):
        batch = self.collect()
        for bad in (replace(batch, old_log_probs=batch.old_log_probs+.1),
                    replace(batch, advantages=batch.advantages+.1),
                    replace(batch, actions=torch.full_like(batch.actions, 6)),
                    replace(batch, returns=batch.returns[:, None]),
                    replace(batch, returns=torch.full_like(batch.returns, float("nan"))),
                    replace(batch, old_values=batch.old_values.clone().requires_grad_())):
            with self.assertRaises(ValueError):
                bad.validate(12, 6)

    def test_loss_arithmetic_and_gradient_separation(self):
        batch, indices = self.collect(), torch.arange(12)
        loss, value_loss = sampled_losses(self.actor, self.critic, self.obs, self.bank, batch, indices, self.cfg["optimizer"])
        logits, _ = scores(self.actor, [self.obs[i] for i in batch.context_indices[indices]], self.bank)
        logp = logits.log_softmax(1)
        actual_logp = logp.gather(1, batch.actions[indices, None]).flatten()
        ratio = (actual_logp-batch.old_log_probs[indices]).exp()
        expected = -torch.minimum(ratio*batch.advantages[indices], ratio.clamp(.8, 1.2)*batch.advantages[indices]).mean()
        expected += .01*(logp.exp()*logp).sum(1).mean()
        torch.testing.assert_close(loss.total, expected)
        torch.testing.assert_close(value_loss, batch.returns[indices].square().mean())
        self.assertTrue(all(g is None for g in torch.autograd.grad(loss.total, tuple(self.critic.parameters()), allow_unused=True)))
        self.assertTrue(all(g is None for g in torch.autograd.grad(value_loss, tuple(self.actor.parameters()), allow_unused=True)))
        before = self.actor.snapshot_sha256()
        loss.total.backward()
        value_loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for m in (self.actor, self.critic) for p in m.parameters()))
        self.assertEqual(self.actor.snapshot_sha256(), before)

    def test_clipping_uses_recorded_behavior_not_new_distribution(self):
        batch = self.collect()
        with torch.no_grad():
            self.actor.actor_head.bias[0] += .1
        loss, _ = sampled_losses(self.actor, self.critic, self.obs, self.bank, batch, torch.arange(12), self.cfg["optimizer"])
        self.assertGreater(float(loss.clip_fraction), 0.)

    def test_minibatch_indices_must_be_unique_valid_integer_vector(self):
        batch = self.collect()
        for indices in (torch.tensor([0, 0]), torch.tensor([48]), torch.tensor([-1]), torch.tensor([0.])):
            with self.assertRaises(ValueError):
                sampled_losses(self.actor, self.critic, self.obs, self.bank, batch, indices, self.cfg["optimizer"])

    def test_critic_no_future_or_label_inputs_and_invalid_units_rejected(self):
        with self.assertRaises(TypeError):
            self.critic(self.obs[0], self.bank, targets=torch.zeros(1))
        for change in (0., float("nan"), True):
            cfg = copy.deepcopy(self.cfg)
            cfg["critic"]["output_gain"] = change
            with self.assertRaises(ValueError):
                make_models(cfg, self.bank, "graph", 101)
        with self.assertRaises(ValueError):
            self.critic(replace(self.obs[0], globals=torch.full_like(self.obs[0].globals, float("inf"))), self.bank)

    def test_call_limits_and_persistence_failure_do_not_refund(self):
        for role in ("actor", "critic"):
            for _ in range(128):
                self.budget.charge("graph-101", role, 1, self.events.append)
            with self.assertRaises(RuntimeError):
                self.budget.charge("graph-101", role, 1, self.events.append)
        self.budget.charge("graph-101", "observations", 1536, self.events.append)
        with self.assertRaises(RuntimeError):
            self.budget.charge("graph-101", "observations", 1, self.events.append)
        with self.assertRaises(OSError):
            self.budget.charge("graph-102", "actor", 1, lambda _: (_ for _ in ()).throw(OSError("disk")))
        self.assertEqual(self.budget.optimizer_calls, 257)
        self.assertEqual(self.budget.by_fit["graph-102"]["actor"], 1)

    def test_failed_pair_rolls_back_both_models_but_keeps_attempt_charges(self):
        batch = self.collect()
        optimizers = [torch.optim.Adam(m.parameters(), lr=.0003) for m in (self.actor, self.critic)]
        before = [copy.deepcopy(m.state_dict()) for m in (self.actor, self.critic)]
        def fake_actor_call():
            with torch.no_grad():
                next(self.actor.parameters()).add_(1.)
        def failed_critic_call():
            raise RuntimeError("invented failure; no actual Adam call")
        with patch.object(optimizers[0], "step", side_effect=fake_actor_call), patch.object(optimizers[1], "step", side_effect=failed_critic_call):
            with self.assertRaisesRegex(RuntimeError, "invented failure"):
                update_pair(self.actor, self.critic, *optimizers, self.obs, self.bank, batch, torch.arange(12),
                    self.cfg["optimizer"], charge_optimizer=lambda k: self.budget.charge("graph-101", k, 1, self.events.append))
        for m, state in zip((self.actor, self.critic), before):
            for name, value in m.state_dict().items():
                torch.testing.assert_close(value, state[name])
        self.assertEqual(self.budget.optimizer_calls, 2)
        self.assertTrue(all(not o.state for o in optimizers))

    def test_wrong_optimizer_ownership_rejected_before_charge(self):
        batch = self.collect()
        opts = [torch.optim.Adam(self.actor.parameters(), lr=.0003) for _ in range(2)]
        with self.assertRaises(ValueError):
            update_pair(self.actor, self.critic, *opts, self.obs, self.bank, batch, torch.arange(12),
                self.cfg["optimizer"], charge_optimizer=lambda k: self.events.append(k))
        self.assertEqual(self.budget.optimizer_calls, 0)

    def test_partial_snapshot_round_trip_without_optimization(self):
        batch = self.collect()
        optimizers = [torch.optim.Adam(m.parameters(), lr=.0003) for m in (self.actor, self.critic)]
        generators = [torch.Generator().manual_seed(n) for n in (610101, 620101)]
        permutation = torch.randperm(48, generator=generators[1])
        state = snapshot(self.actor, self.critic, optimizers, generators, batch=batch,
                         permutation=permutation, next_minibatch=0, budget=self.budget)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "artificial-state.pt"
            save_envelope(path, state)
            loaded = load_envelope(path)
        self.assertFalse(loaded["resume_authorized"])
        self.assertEqual(loaded["budget"]["observations"], 48)
        SampledReturns(**loaded["batch"]).validate(12, 6)
        other_actor, other_critic = make_models(self.cfg, self.bank, "graph", 999)
        other_actor.load_state_dict(loaded["actor"])
        other_critic.load_state_dict(loaded["critic"])
        self.assertEqual(self.actor.snapshot_sha256(), other_actor.snapshot_sha256())
        restored = torch.Generator()
        restored.set_state(loaded["shuffle_rng"])
        torch.testing.assert_close(torch.randperm(48, generator=restored), torch.randperm(48, generator=generators[1]))
        torch.testing.assert_close(loaded["permutation"], permutation)

    def test_independent_scalar_sample_and_charge_audit(self):
        batch = self.collect()
        receipt = {key: tensor.tolist() for key, tensor in vars(batch).items()}
        result = verify_sample_batch(receipt, self.cfg)
        self.assertEqual(result["sample_count"], 48)
        self.assertEqual(sum(result["coverage"].values()), 48)
        previous = {"observations": 0, "optimizer_calls": 0, "by_fit": {}}
        self.assertEqual(verify_charge(previous, self.events[0], self.cfg), self.budget.state())
        for key in ("returns", "advantages", "old_log_probs"):
            bad = copy.deepcopy(receipt)
            bad[key][0] += .01
            with self.assertRaises(ValueError):
                verify_sample_batch(bad, self.cfg)
        bad = copy.deepcopy(self.events[0])
        bad["budget"]["observations"] -= 1
        with self.assertRaises(ValueError):
            verify_charge(previous, bad, self.cfg)

    def test_independent_evaluation_arithmetic_reports_failures(self):
        logits = [[0.]*6 for _ in range(8)]
        coverage = {f"{i}:{j}": 1 for i in range(12) for j in range(6)}
        result = evaluation_metrics(logits, [0.]*8, -1.8, coverage, self.cfg)
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["accuracy"])
        self.assertFalse(result["gates"]["margin"])
        self.assertTrue(result["gates"]["coverage"])
        self.assertGreater(result["critic_mse"], 0.)
        coverage["0:0"] = 0
        self.assertFalse(evaluation_metrics(logits, [0.]*8, -1.8, coverage, self.cfg)["gates"]["coverage"])
        with self.assertRaises(ValueError):
            evaluation_metrics(logits[:7], [0.]*8, -1.8, coverage, self.cfg)

    def test_exact_approval_bound_to_source_runtime_protocol_and_caps(self):
        runtime, hashes = {"invented_runtime": True}, {"src/example.py": "3"*64}
        auth = {"scope": campaign.SCOPE, "decision": "approved_by_zhaowei", "implementation_ready": True,
            "user_approval_evidence": "UNIT TEST ONLY NOT AN APPROVAL", "implementation_commit": "1"*40,
            "caps": copy.deepcopy(campaign.CAPS), "config_sha256": "1"*64, "protocol_sha256": "2"*64,
            "source_hashes": hashes, "runtime": runtime}
        def check(item):
            campaign.validate_authorization(self.cfg, item, config_sha="1"*64, protocol_sha="2"*64,
                                            source_hashes=hashes, current_runtime=runtime)
        check(auth)
        for key, value in (("scope", "patient"), ("decision", "pending"), ("implementation_ready", False),
                           ("user_approval_evidence", ""), ("implementation_commit", "main"), ("caps", {}),
                           ("config_sha256", "x"), ("source_hashes", {}), ("runtime", {})):
            with self.assertRaises(ValueError):
                check(auth | {key: value})
        changed = copy.deepcopy(self.cfg)
        changed["sampling"]["rollouts_per_fit"] += 1
        with self.assertRaises(ValueError):
            campaign.validate_config(changed)

    def test_missing_approval_cannot_create_output_or_construct_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json_once(root / CONFIG, self.cfg)
            with patch.object(campaign, "make_models", side_effect=AssertionError("no models before approval")):
                with self.assertRaises(FileNotFoundError):
                    campaign.run(root)
            self.assertFalse((root / self.cfg["output"]).exists())

    def test_recorder_wall_cap_and_nonrefundable_write_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            now = [0.]
            recorder = campaign.Recorder(temp, self.cfg, clock=lambda: now[0])
            recorder.charge("graph-101", "observations", 48)
            reader = ReceiptReader(temp, self.cfg)
            reader.charge("graph-101", "observations")
            self.assertEqual(reader.budget, recorder.budget.state())
            with patch.object(campaign, "write_json_once", side_effect=OSError("invented disk failure")):
                with self.assertRaises(OSError):
                    recorder.charge("graph-101", "actor", 1)
            self.assertEqual(recorder.budget.optimizer_calls, 1)
            now[0] = 1800.
            with self.assertRaises(TimeoutError):
                recorder.charge("graph-101", "critic", 1)
            self.assertEqual(recorder.budget.optimizer_calls, 1)

    def test_independent_optimizer_counters_without_calling_optimizer(self):
        opt = torch.optim.Adam(self.critic.parameters(), lr=.0003)
        verify_optimizer(opt.state_dict(), self.critic.state_dict(), 0, self.cfg["optimizer"])
        # Invent saved Adam moments; do not invoke an update to generate them.
        for param in self.critic.parameters():
            opt.state[param] = {"step": torch.tensor(1.), "exp_avg": torch.zeros_like(param),
                                "exp_avg_sq": torch.zeros_like(param)}
        verify_optimizer(opt.state_dict(), self.critic.state_dict(), 1, self.cfg["optimizer"])
        with self.assertRaises(ValueError):
            verify_optimizer(opt.state_dict(), self.critic.state_dict(), 2, self.cfg["optimizer"])

    def test_terminal_failure_preserves_claim_partial_and_forbids_second_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json_once(root / campaign.AUTHORIZATION, {"test_only": True})
            with patch.object(campaign.subprocess, "check_output", return_value="1"*40), \
                 patch.object(campaign, "runtime", return_value={"unit_test_only": True}), \
                 patch.object(campaign, "collect_samples", side_effect=RuntimeError("mock collector failed")):
                with self.assertRaisesRegex(RuntimeError, "mock collector failed"):
                    campaign.execute(root, self.cfg, {"test_only": True}, {}, {})
            output = root / self.cfg["output"]
            self.assertTrue((output / "claim.json").is_file())
            self.assertTrue((output / "partial-on-failure.pt").is_file())
            failure = json.loads((output / "terminal-failure.json").read_text())
            self.assertFalse(failure["retry_permitted"])
            self.assertEqual(failure["budget"]["optimizer_calls"], 0)
            with self.assertRaises(FileExistsError):
                campaign.execute(root, self.cfg, {}, {}, {})

    def test_entire_serial_schedule_with_mocked_observer_optimizer_and_evaluation(self):
        from src.rl.sampled_return_packet_verification import verify
        from src.utils.research_archive import inventory, sha256_file
        from src.rl.candidate_ppo_objective import normalize_rollout_advantages
        from src.models.independent_artificial_value import IndependentArtificialValue
        ordered, reference = sorted(self.cfg["request_values"]), self.bank.reference_class
        forwards, updates, samples = [], [], []
        def fake_scores(policy, observations, bank):
            forwards.append(len(observations))
            logits = torch.zeros(len(observations), 6)
            logits[:, reference] = self.cfg["initial_reference_logit"]
            return logits, torch.zeros(len(observations))
        def fake_collect(actor, critic, observations, bank, **kwargs):
            samples.append(1)
            kwargs["charge_observations"](48)
            contexts = torch.arange(12).repeat(4)
            logits = torch.zeros(48, 6)
            logits[:, reference] = self.cfg["initial_reference_logit"]
            actions = torch.multinomial(logits.softmax(1), 1, generator=kwargs["sampling_rng"]).flatten()
            # Hand-built receipt fixture, not the task's reward_observer.
            returns = []
            for context, action in zip(contexts.tolist(), actions.tolist()):
                cue, t = self.cfg["cues"][context//6], self.cfg["training_times"][context%6]
                winner = .75 if cue > 0 else 0.
                returns.append(-.25-2.75*(1-t)-(0 if ordered[action] == winner else .25))
            values = torch.tensor(returns, dtype=torch.float32)
            return SampledReturns(contexts, actions, logits.log_softmax(1).gather(1, actions[:, None]).flatten(),
                torch.zeros(48), values, normalize_rollout_advantages(values, enabled=True), logits)
        def fake_update(actor, critic, aopt, copt, *unused, charge_optimizer):
            updates.append(1)
            for kind, optimizer in (("actor", aopt), ("critic", copt)):
                charge_optimizer(kind)
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        step = float(optimizer.state.get(parameter, {}).get("step", 0.))+1
                        optimizer.state[parameter] = {"step": torch.tensor(step), "exp_avg": torch.zeros_like(parameter),
                                                     "exp_avg_sq": torch.zeros_like(parameter)}
            return dict.fromkeys(("policy_loss", "entropy", "actor_total", "critic_mse", "clip_fraction",
                                  "actor_grad_norm", "critic_grad_norm"), 0.)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json_once(root / CONFIG, self.cfg)
            write_json_once(root / campaign.PROTOCOL, {"unit_test_protocol": True})
            auth = {"decision": "approved_by_zhaowei", "implementation_ready": True,
                    "scope": campaign.SCOPE, "caps": copy.deepcopy(campaign.CAPS),
                    "source_hashes": {}, "config_sha256": sha256_file(root / CONFIG),
                    "protocol_sha256": sha256_file(root / campaign.PROTOCOL),
                    "unit_test_only_not_user_approval": True}
            write_json_once(root / campaign.AUTHORIZATION, auth)
            prior_roots = {"p2_payload": "candidate_reference_prior_pilot_20261001/payload",
                "p2_launcher": "candidate_reference_prior_pilot_20261001/launcher", "calibration": "candidate_calibration_engineering_20261001",
                "actor_positive_control": "candidate_actor_positive_control_20261001"}
            prior = {}
            for key, relative in prior_roots.items():
                write_json_once(root / "results" / relative / "mock.json", {"unit_test_only": True})
                prior[key] = inventory(root / "results" / relative)
            with patch.object(campaign.subprocess, "check_output", return_value="1"*40), \
                 patch.object(campaign, "runtime", return_value={"unit_test_only": True}), \
                 patch.object(campaign, "historical_evidence", return_value=prior), \
                 patch.object(campaign, "scores", side_effect=fake_scores), \
                 patch.object(campaign, "collect_samples", side_effect=fake_collect), \
                 patch.object(campaign, "update_pair", side_effect=fake_update), \
                 patch.object(IndependentArtificialValue, "forward", return_value=torch.tensor(0.)), \
                 patch("builtins.print"):
                terminal = campaign.execute(root, self.cfg, auth, {}, prior)
            self.assertFalse(terminal["engineering_passed"])
            self.assertEqual(len(samples), 288)
            self.assertEqual(len(updates), 1152)
            self.assertEqual(forwards, [8]*18)
            with patch.object(campaign, "make_models", side_effect=AssertionError("verifier must not construct models")):
                report = verify(root)
            self.assertEqual(report["passed_fixtures"], 0)
            self.assertEqual(report["training_observations_verified"], 13824)
            self.assertEqual(report["optimizer_charges_verified"], 2304)
            self.assertTrue(report["private_rng_replay_verified"])


if __name__ == "__main__":
    unittest.main()
