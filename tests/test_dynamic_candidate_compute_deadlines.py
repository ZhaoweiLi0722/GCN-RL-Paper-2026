"""Bounded invented receipts, deadline injection and fake optimizer metadata.

Real Adam/SGD steps are sentinels. No patient environment or fitted weights.
"""

import copy
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
import src.rl.dynamic_candidate_imitation as imitation_module
import src.rl.dynamic_candidate_ppo as ppo_module
from src.rl.dynamic_candidate_resources import read_dynamic_ledger
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from tests import test_dynamic_candidate_continuation as continuation_fixtures
from tests.test_dynamic_candidate_updates import examples, fake_adam, imitation, learner, segment


class Deadline:
    def __init__(self):
        self.expired = False
        self.calls = 0

    def check(self):
        self.calls += 1
        if self.expired:
            raise TimeoutError("invented compute deadline")

    def __deepcopy__(self, memo):
        raise AssertionError("external compute callback must not be copied")


@unittest.skipIf(torch is None, "torch unavailable")
class DynamicComputeDeadlineTests(unittest.TestCase):
    def setUp(self):
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("real step forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)

    def build(self, role):
        if role == "ppo":
            kernel = learner(epochs=1, batch_size=1)
            kernel.add_segment(segment(kernel, length=1))
            return kernel
        return imitation("demonstration" if role == "initialization" else "training", batch_size=1)

    def invoke(self, kernel, charges, before_compute=None, *, omit_callback=False):
        options = {} if omit_callback else {"before_compute": before_compute}
        if isinstance(kernel, ppo_module.DynamicCandidatePPOKernel):
            return kernel.update(before_optimizer_step=charges.append, before_minibatch=lambda: None,
                                 **options)
        split = kernel.settings.allowed_split
        count = {"replacement_steps": 1} if split == "demonstration" else {"epochs": 1}
        return kernel.fit(examples(split, length=1), before_step=lambda: charges.append("actor"),
                          **count, **options)

    def assert_clean_guards(self, kernel):
        self.assertNotIn("_before_compute", kernel.__dict__)
        self.assertTrue(all(not module._forward_pre_hooks for module in kernel.policy.modules()))

    def assert_terminal(self, kernel, original, charges):
        failed = kernel.state_dict()
        self.assertEqual(state_digest(dict(failed, failure=None)), state_digest(original))
        failure = failed["failure"]
        self.assertEqual(failure["error_type"], "TimeoutError")
        self.assertEqual(failure["acknowledged_charge_owners"], charges)
        self.assertEqual(failure["resource_authority"], "external_non_refundable_ledger")
        partial = failure["partial_transaction"]
        self.assertEqual(state_digest(partial["policy"]), state_digest(original["policy"]))
        self.assertIsNone(partial["failure"])
        self.assert_clean_guards(kernel)
        with self.assertRaisesRegex(ValueError, "failed"):
            kernel.load_state_dict(original)
        with self.assertRaisesRegex(ValueError, "failed"):
            self.invoke(kernel, [], lambda: None)
        with self.assertRaisesRegex(ValueError, "failed"):
            copy.deepcopy(kernel)._restore(failed)
        self.assertEqual(state_digest(failed), state_digest(kernel.state_dict()))
        return partial

    @contextmanager
    def expire_after_loss(self, role, expire):
        if role == "ppo":
            objective = ppo_module.candidate_ppo_loss

            def loss(*args, **kwargs):
                result = objective(*args, **kwargs)
                expire()
                return result

            with patch.object(ppo_module, "candidate_ppo_loss", new=loss):
                yield
        else:
            stack = torch.stack

            def loss(items, *args, **kwargs):
                result = stack(items, *args, **kwargs)
                if all(item.ndim == 0 and item.requires_grad for item in items):
                    expire()
                return result

            with patch.object(torch, "stack", new=loss):
                yield

    def test_expired_entry_blocks_forward_backward_and_optimizer_restore(self):
        for role in ("ppo", "bc", "initialization"):
            with self.subTest(role=role):
                kernel = self.build(role)
                original, charges, deadline = kernel.state_dict(), [], Deadline()
                deadline.expired = True
                with patch.object(DynamicCandidatePolicy, "forward") as forward, \
                        patch.object(torch.Tensor, "backward") as backward, \
                        patch.object(torch.optim.Adam, "load_state_dict") as restore:
                    with self.assertRaises(TimeoutError):
                        self.invoke(kernel, charges, deadline.check)
                    forward.assert_not_called()
                    backward.assert_not_called()
                    restore.assert_not_called()
                self.assertEqual(charges, [])
                self.assert_terminal(kernel, original, [])

    def test_forward_guard_blocks_expiry_inside_adapter_before_actual_forward(self):
        for role in ("ppo", "bc"):
            with self.subTest(role=role):
                kernel = self.build(role)
                original, charges, deadline = kernel.state_dict(), [], Deadline()
                call = torch.nn.Module._call_impl

                def expire_at_call(module, *args, **kwargs):
                    if type(module) is DynamicCandidatePolicy:
                        deadline.expired = True
                    return call(module, *args, **kwargs)

                with patch.object(torch.nn.Module, "_call_impl", new=expire_at_call), \
                        patch.object(DynamicCandidatePolicy, "forward") as forward, \
                        patch.object(torch.Tensor, "backward") as backward:
                    with self.assertRaises(TimeoutError):
                        self.invoke(kernel, charges, deadline.check)
                    forward.assert_not_called()
                    backward.assert_not_called()
                self.assertTrue(deadline.expired)
                self.assert_terminal(kernel, original, [])

    def test_critic_forward_is_checked_after_actor_forward(self):
        for role in ("ppo", "bc"):
            with self.subTest(role=role):
                kernel = self.build(role)
                original, deadline = kernel.state_dict(), Deadline()
                actor_class = type(kernel.policy.actor)
                forward = actor_class.forward

                def actor(module, *args, **kwargs):
                    result = forward(module, *args, **kwargs)
                    deadline.expired = True
                    return result

                with patch.object(actor_class, "forward", new=actor), \
                        patch.object(type(kernel.policy.critic), "forward") as critic, \
                        patch.object(torch.Tensor, "backward") as backward:
                    with self.assertRaises(TimeoutError):
                        self.invoke(kernel, [], deadline.check)
                    critic.assert_not_called()
                    backward.assert_not_called()
                self.assert_terminal(kernel, original, [])

    def test_expiry_after_loss_blocks_first_backward(self):
        for role in ("ppo", "bc", "initialization"):
            with self.subTest(role=role):
                kernel = self.build(role)
                original, charges, deadline = kernel.state_dict(), [], Deadline()
                with self.expire_after_loss(role, lambda: setattr(deadline, "expired", True)), \
                        patch.object(torch.Tensor, "backward") as backward:
                    with self.assertRaises(TimeoutError):
                        self.invoke(kernel, charges, deadline.check)
                    backward.assert_not_called()
                self.assertTrue(deadline.expired)
                self.assertEqual(charges, [])
                self.assert_terminal(kernel, original, [])

    def test_expiry_after_pair_admission_blocks_minibatch_forward(self):
        kernel, deadline, charges = self.build("ppo"), Deadline(), []
        original = kernel.state_dict()
        with patch.object(ppo_module, "reevaluate_dynamic_decision") as evaluate, \
                patch.object(torch.Tensor, "backward") as backward:
            with self.assertRaises(TimeoutError):
                kernel.update(before_optimizer_step=charges.append,
                              before_minibatch=lambda: setattr(deadline, "expired", True),
                              before_compute=deadline.check)
            evaluate.assert_not_called()
            backward.assert_not_called()
        self.assertEqual(charges, [])
        self.assert_terminal(kernel, original, [])

    def test_expiry_after_backward_blocks_gradient_clip_and_debit(self):
        for role in ("ppo", "bc"):
            with self.subTest(role=role):
                kernel = self.build(role)
                original, deadline, charges = kernel.state_dict(), Deadline(), []
                backward = torch.Tensor.backward

                def expire(tensor, *args, **kwargs):
                    result = backward(tensor, *args, **kwargs)
                    deadline.expired = True
                    return result

                with patch.object(torch.Tensor, "backward", new=expire), \
                        patch.object(torch.nn.utils, "clip_grad_norm_") as clip:
                    with self.assertRaises(TimeoutError):
                        self.invoke(kernel, charges, deadline.check)
                    clip.assert_not_called()
                self.assertTrue(deadline.expired)
                self.assertEqual(charges, [])
                self.assert_terminal(kernel, original, [])

    def test_expiry_after_adam_validation_blocks_state_restore(self):
        for role in ("ppo", "bc"):
            with self.subTest(role=role):
                kernel = self.build(role)
                original, deadline = kernel.state_dict(), Deadline()

                def digest(value):
                    result = state_digest(value)
                    if isinstance(value, dict) and set(value) == {"state", "param_groups"}:
                        deadline.expired = True
                    return result

                with patch.object(ppo_module, "state_digest", new=digest), \
                        patch.object(torch.optim.Adam, "load_state_dict") as restore:
                    with self.assertRaises(TimeoutError):
                        self.invoke(kernel, [], deadline.check)
                    restore.assert_not_called()
                self.assertTrue(deadline.expired)
                self.assert_terminal(kernel, original, [])

    def test_expiry_during_decode_blocks_next_example_and_forward(self):
        kernel, deadline = imitation(batch_size=1), Deadline()
        original = kernel.state_dict()
        decode = imitation_module.decode_example

        def expire(*args):
            result = decode(*args)
            deadline.expired = True
            return result

        with patch.object(imitation_module, "decode_example", side_effect=expire) as decoded, \
                patch.object(DynamicCandidatePolicy, "forward") as forward:
            with self.assertRaises(TimeoutError):
                kernel.fit(examples(length=2), epochs=1, before_step=lambda: self.fail("unexpected debit"),
                           before_compute=deadline.check)
            self.assertEqual(decoded.call_count, 1)
            forward.assert_not_called()
        self.assert_terminal(kernel, original, [])

    def test_expiry_after_actor_debit_retains_charge_without_optimizer_call(self):
        for role in ("ppo", "bc"):
            with self.subTest(role=role):
                kernel = self.build(role)
                original, deadline, charges = kernel.state_dict(), Deadline(), []

                class Charges:
                    def append(self, owner):
                        charges.append(owner)
                        deadline.expired = True

                with self.assertRaises(TimeoutError):
                    self.invoke(kernel, Charges(), deadline.check)
                self.assertEqual(charges, ["actor"])
                self.assert_terminal(kernel, original, charges)

    def test_critic_backward_timeout_retains_only_actor_metadata_and_debit(self):
        kernel, deadline, charges, calls = self.build("ppo"), Deadline(), [], []
        original = kernel.state_dict()
        zero_grad, backward = DynamicCandidatePolicy.zero_grad, torch.Tensor.backward

        def metadata(optimizer, *args, **kwargs):
            fake_adam(optimizer, *args, **kwargs)
            calls.append("actor")

        def expire_after_zero(module, *args, **kwargs):
            zero_grad(module, *args, **kwargs)
            if calls:
                deadline.expired = True

        with patch.object(torch.optim.Adam, "step", new=metadata), \
                patch.object(DynamicCandidatePolicy, "zero_grad", new=expire_after_zero), \
                patch.object(torch.Tensor, "backward", autospec=True, side_effect=backward) as backwards:
            with self.assertRaises(TimeoutError):
                self.invoke(kernel, charges, deadline.check)
            self.assertEqual(backwards.call_count, 1)
        self.assertEqual(charges, ["actor"])
        partial = self.assert_terminal(kernel, original, charges)
        self.assertEqual({s["step"].item() for s in partial["optimizer"]["actor"]["state"].values()}, {1.})
        self.assertEqual(partial["optimizer"]["critic"]["state"], {})

    def test_final_restore_timeout_retains_both_owner_charges_and_payload(self):
        for role in ("ppo", "bc"):
            with self.subTest(role=role):
                kernel = self.build(role)
                original, deadline, charges = kernel.state_dict(), Deadline(), []
                restore = torch.optim.Adam.load_state_dict

                def expire(optimizer, saved):
                    result = restore(optimizer, saved)
                    if saved["state"]:
                        deadline.expired = True
                    return result

                with patch.object(torch.optim.Adam, "step", new=fake_adam), \
                        patch.object(torch.optim.Adam, "load_state_dict", new=expire):
                    with self.assertRaises(TimeoutError):
                        self.invoke(kernel, charges, deadline.check)
                expected = ["actor", "critic"] if role == "ppo" else ["actor"]
                self.assertEqual(charges, expected)
                partial = self.assert_terminal(kernel, original, expected)
                owners = partial["optimizer"] if role == "ppo" else {"actor": partial["optimizer"]}
                for owner in expected:
                    self.assertEqual({s["step"].item() for s in owners[owner]["state"].values()}, {1.})
                self.assertEqual(len(partial["history"]), 1)

    def test_default_callback_compatibility_and_explicit_callback_does_not_leak(self):
        for role in ("ppo", "bc", "initialization"):
            for omit in (False, True):
                with self.subTest(role=role, omit=omit):
                    kernel = self.build(role)
                    original, deadline, charges = kernel.policy.snapshot_sha256(), Deadline(), []
                    with patch.object(torch.optim.Adam, "step", new=fake_adam):
                        report = self.invoke(kernel, charges, deadline.check, omit_callback=omit)
                    self.assertEqual(report["optimizer_steps"], 2 if role == "ppo" else 1)
                    self.assertEqual(charges, ["actor", "critic"] if role == "ppo" else ["actor"])
                    self.assertEqual(kernel.policy.snapshot_sha256(), original)
                    self.assertEqual(deadline.calls > 0, not omit)
                    self.assert_clean_guards(kernel)
                    self.assertEqual(state_digest(kernel.state_dict()), state_digest(copy.deepcopy(kernel).state_dict()))

    def test_noncallable_compute_callback_is_rejected_without_transaction(self):
        for role in ("ppo", "bc"):
            kernel = self.build(role)
            original = state_digest(kernel.state_dict())
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "before_compute"):
                self.invoke(kernel, [], None)
            self.assertEqual(state_digest(kernel.state_dict()), original)

    def test_continuation_wires_budget_check_and_preserves_pending_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            self.root, self.serial = Path(directory), 0
            for role in ("ppo", "bc_continue"):
                with self.subTest(role=role):
                    run = continuation_fixtures.DynamicContinuationTests.build(self, role)
                    for _ in range(4):
                        run.step()
                    self.assertTrue(run.update_due)
                    original, deadline = run.kernel.state_dict(), Deadline()
                    pending_examples = copy.deepcopy(run.examples)
                    run.budget.clock = lambda: 101. if deadline.expired else 0.
                    with self.expire_after_loss(role, lambda: setattr(deadline, "expired", True)), \
                            patch.object(torch.Tensor, "backward") as backward:
                        with self.assertRaises(TimeoutError):
                            run.update()
                        backward.assert_not_called()
                    self.assertTrue(run.budget.failed)
                    self.assertEqual(run.failure["error_type"], "TimeoutError")
                    self.assertEqual(run.examples, pending_examples)
                    self.assert_terminal(run.kernel, original, [])
                    payload = run.state_dict()
                    self.assertIsNotNone(payload["kernel"]["failure"]["partial_transaction"])
                    ledger = read_dynamic_ledger(run.budget.path)
                    self.assertEqual(ledger["counts"]["environment"], 4)
                    self.assertEqual(ledger["counts"].get("optimizer", 0), 0)
                    with self.assertRaisesRegex(ValueError, "failed"):
                        run.update()


if __name__ == "__main__":
    unittest.main()
