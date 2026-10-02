"""Invented wrapper/admission fixtures; no research input, model or patient call."""

import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.rl import dynamic_candidate_recovery_execution as execution
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.prospective_ddpg_kernel import state_digest


def fixture_packet(root):
    sections = {
        name: {"seconds": 2}
        for name in ("runtime_input_binding", "same_start_preflight",
                     "supervisor_dispatch_terminal_closure")
    }
    record = {"path": "invented", "bytes": 0, "sha256": "a" * 64}
    packet = {
        "scientific_execution_authorized": False,
        "ready_to_launch": False,
        "workspace": str(root),
        "branch": execution.BRANCH,
        "implementation_commit": "b" * 40,
        "original_config": {"fixture": "no scientific configuration"},
        "scientific_config": {
            "totals": {"fresh_episode_builds": 2},
            "evaluation": {"total_episodes": 1},
        },
        "streams": {"fixture": "no experimental streams"},
        "budget_plan": {
            "limits": {"trajectory": 4, "clone": 0, "actor": 0,
                       "critic": 0, "seconds": 8},
            "sections": sections,
        },
        "source_files": {"src/invented.py": dict(record)},
        "runtime": {"fixture": "no runtime import needed"},
        "proposal": dict(record, path=execution.PROPOSAL),
        "protocol": dict(record, path=execution.PROTOCOL),
        "input_files": {"invented.pt": dict(record, path="invented.pt")},
    }
    packet["packet_sha256"] = digest(packet)
    return packet


def fixture_authorization(packet):
    return {
        "format": "dynamic-continuation-recovery-authorization-v1",
        "approved": True,
        "packet_sha256": packet["packet_sha256"],
        "implementation_commit": packet["implementation_commit"],
        "limits": execution.approved_limits(packet),
        "automatic_retry": False,
        "remote_or_dropbox_actions": False,
        "reuse_completed_qualification_without_rescoring": True,
        "old_s1_remains_terminal": True,
        "user_approval": {
            "user": "Zhaowei",
            "verbatim": "INVENTED TEST FIXTURE ONLY, NOT USER AUTHORIZATION",
            "recorded_at_utc": "fixture",
        },
    }


class RecoveryAdmissionFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / execution.RUN
        self.packet = fixture_packet(self.root)
        self.auth = fixture_authorization(self.packet)
        for target in (
            "src.rl.experiment.build_env",
            "src.rl.strict_frozen_policy.StrictFrozenPolicy",
            "src.rl.dynamic_candidate_recovery_execution.restore_saved_policy",
            "torch.load", "torch.nn.Module._call_impl",
            "torch.optim.Adam.step", "torch.optim.SGD.step",
        ):
            guard = patch(target, side_effect=AssertionError("scientific work forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)

    def admission(self):
        budget = SimpleNamespace(
            started=100.0,
            active="runtime_input_binding",
            plan=copy.deepcopy(self.packet["budget_plan"]),
            path=self.output / "launcher/budget.jsonl",
            check=Mock(return_value=100.0),
        )
        claim = {
            "pid": os.getppid(),
            "packet_sha256": self.packet["packet_sha256"],
            "authorization_sha256": digest(self.auth),
            "clock_id": execution.CLOCK_ID,
            "started": budget.started,
        }
        write_json_once(self.output / "launcher/claim.json", claim)
        admitted = execution.RecoveryAdmission(
            self.root, self.packet, self.auth, claim, budget)
        return admitted, budget, claim

    def binding_mocks(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        mocks = {}
        values = {
            "source_locks": self.packet["source_files"],
            "runtime_record": self.packet["runtime"],
            "recovery_config": self.packet["scientific_config"],
            "recovery_budget_plan": self.packet["budget_plan"],
        }
        for name, value in values.items():
            mocks[name] = stack.enter_context(
                patch.object(execution, name, return_value=copy.deepcopy(value)))
        records = dict(self.packet["input_files"])
        records[execution.PROPOSAL] = self.packet["proposal"]
        records[execution.PROTOCOL] = self.packet["protocol"]
        mocks["file_record"] = stack.enter_context(patch.object(
            execution, "file_record", side_effect=lambda root, name: copy.deepcopy(records[name])))
        mocks["git_run"] = stack.enter_context(patch.object(execution.subprocess, "run"))
        return mocks


class RecoveryAuthorizationTests(RecoveryAdmissionFixture):
    def test_every_limit_is_exact_and_binds_six_historical_loads(self):
        limits = execution.approved_limits(self.packet)
        self.assertEqual(limits["historical_initializer_loads"], 3)
        self.assertEqual(limits["reference_loads"], 3)
        self.assertEqual(limits["layout_builds"], 3)
        self.assertEqual(limits["episode_builds"], 2)
        self.assertEqual(limits["final_test_episodes"], 1)
        execution.validate_authorization(self.auth, self.packet)
        for key in limits:
            changed = copy.deepcopy(self.auth)
            changed["limits"][key] += 1
            with self.subTest(limit=key), self.assertRaises(PermissionError):
                execution.validate_authorization(changed, self.packet)

    def test_missing_approval_identity_and_reuse_terms_fail_closed(self):
        for key in ("approved", "packet_sha256", "implementation_commit",
                    "reuse_completed_qualification_without_rescoring",
                    "old_s1_remains_terminal", "remote_or_dropbox_actions"):
            changed = copy.deepcopy(self.auth)
            changed.pop(key)
            with self.subTest(field=key), self.assertRaises(PermissionError):
                execution.validate_authorization(changed, self.packet)
        for approval in ({}, {"user": "Zhaowei", "verbatim": " ", "recorded_at_utc": "fixture"},
                         {"user": "Zhaowei", "verbatim": "fixture"}):
            with self.subTest(approval=approval), self.assertRaises(PermissionError):
                execution.validate_authorization(dict(self.auth, user_approval=approval), self.packet)

    def test_packet_mutation_fails_even_when_authorization_hash_is_unchanged(self):
        altered = copy.deepcopy(self.packet)
        altered["streams"]["fixture"] = "changed"
        with self.assertRaisesRegex(ValueError, "packet changed"):
            execution.validate_authorization(self.auth, altered)

    def test_preparation_flags_cannot_be_used_as_authorization(self):
        altered = copy.deepcopy(self.packet)
        altered["scientific_execution_authorized"] = True
        altered.pop("packet_sha256")
        altered["packet_sha256"] = digest(altered)
        with self.assertRaises(PermissionError):
            execution.validate_authorization(fixture_authorization(altered), altered)


class RecoveryOperationAdmissionTests(RecoveryAdmissionFixture):
    def test_claim_binds_parent_packet_authorization_clock_budget_and_path(self):
        _, budget, claim = self.admission()
        for field, value in (("pid", -1), ("packet_sha256", "changed"),
                             ("authorization_sha256", "changed"), ("clock_id", "changed"),
                             ("started", 101.0)):
            with self.subTest(field=field), self.assertRaises(PermissionError):
                execution.RecoveryAdmission(self.root, self.packet, self.auth,
                                            dict(claim, **{field: value}), budget)
        for field, value in (("plan", {}), ("path", self.root / "old-budget.jsonl")):
            changed = copy.copy(budget)
            setattr(changed, field, value)
            with self.subTest(field=field), self.assertRaises(PermissionError):
                execution.RecoveryAdmission(self.root, self.packet, self.auth, claim, changed)

    def test_each_load_and_layout_cap_debits_once_before_the_operation(self):
        admitted, _, _ = self.admission()
        with patch.object(execution, "shared_monotonic", return_value=100.0):
            for operation in ("checkpoint_load", "reference_checkpoint_load", "layout_environment_build"):
                for count in range(1, 4):
                    admitted(operation)
                    row = json.loads((self.output / f"launcher/operations/{operation}-{count:04d}.json").read_text())
                    self.assertEqual(row, {"operation": operation, "count": count,
                        "phase": "runtime_input_binding", "clock": 100.0, "before_operation": True})
                with self.subTest(operation=operation), self.assertRaises(ValueError):
                    admitted(operation)
                self.assertEqual(admitted.counts[operation], 3)

    def test_episode_build_cap_uses_effective_config_and_existing_scope(self):
        admitted, budget, _ = self.admission()
        budget.active = "same_start_preflight"
        admitted("episode_build")
        admitted("episode_build")
        with self.assertRaises(ValueError):
            admitted("episode_build")
        self.assertEqual(admitted.counts, {"episode_build": 2})

    def test_load_and_layout_rejected_after_binding_without_debit(self):
        admitted, budget, _ = self.admission()
        budget.active = "ppo_continuation/block60"
        for operation in ("checkpoint_load", "reference_checkpoint_load", "layout_environment_build"):
            with self.subTest(operation=operation), self.assertRaises(PermissionError):
                admitted(operation)
        self.assertEqual(admitted.counts, {})

    def test_episode_build_rejected_in_non_episode_phases(self):
        admitted, budget, _ = self.admission()
        for phase in ("runtime_input_binding", "all_model_seal", "saved_data_verification_analysis",
                      "local_archive_verification", "supervisor_dispatch_terminal_closure"):
            budget.active = phase
            with self.assertRaises(PermissionError, msg=phase):
                admitted("episode_build")
        self.assertEqual(admitted.counts, {})

    def test_deadline_and_no_active_phase_prevent_debits(self):
        admitted, budget, _ = self.admission()
        budget.check.side_effect = TimeoutError("invented deadline")
        with self.assertRaises(TimeoutError):
            admitted("checkpoint_load")
        budget.check.side_effect = None
        budget.active = None
        with self.assertRaises(PermissionError):
            admitted("checkpoint_load")
        self.assertEqual(admitted.counts, {})
        self.assertFalse((self.output / "launcher/operations").exists())

    def test_unknown_operation_and_old_qualification_scoring_are_rejected(self):
        admitted, _, _ = self.admission()
        for operation in ("scoring", "qualification", "optimizer", "resume", "unknown"):
            with self.subTest(operation=operation), self.assertRaises(PermissionError):
                admitted(operation)
        self.assertEqual(admitted.counts, {})

    def test_terminal_parent_and_owner_identity_stop_further_admission(self):
        admitted, _, claim = self.admission()
        with patch.object(execution.os, "getpid", return_value=admitted.pid + 1):
            with self.assertRaises(PermissionError):
                admitted("checkpoint_load")
        with patch.object(execution.os, "getppid", return_value=claim["pid"] + 1):
            with self.assertRaises(PermissionError):
                admitted("checkpoint_load")
        write_json_once(self.output / "launcher/terminal.json", {"status": "failed"})
        with self.assertRaises(PermissionError):
            admitted("checkpoint_load")
        self.assertEqual(admitted.counts, {})

    def test_claim_file_mutation_is_rejected(self):
        admitted, _, claim = self.admission()
        path = self.output / "launcher/claim.json"
        path.write_text(json.dumps(dict(claim, unexpected="changed")))
        with self.assertRaises(PermissionError):
            admitted("checkpoint_load")

    def test_duplicate_operation_receipt_is_never_overwritten(self):
        admitted, _, _ = self.admission()
        path = self.output / "launcher/operations/checkpoint_load-0001.json"
        write_json_once(path, {"existing": "immutable fixture"})
        with self.assertRaises(FileExistsError):
            admitted("checkpoint_load")
        self.assertEqual(json.loads(path.read_text()), {"existing": "immutable fixture"})
        self.assertEqual(admitted.counts, {})

    def test_fake_loader_observes_durable_debit_and_does_not_refund_failure(self):
        admitted, _, _ = self.admission()
        path = self.root / "invented.pt"
        raw = b"invented bytes, never a research checkpoint"
        path.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        def loader(handle, **options):
            self.assertEqual(handle.read(), raw)
            self.assertEqual(options, {"map_location": "cpu", "weights_only": True})
            self.assertEqual(admitted.counts["checkpoint_load"], 1)
            self.assertTrue((self.output / "launcher/operations/checkpoint_load-0001.json").exists())
            raise RuntimeError("invented parser failure")
        with self.assertRaisesRegex(RuntimeError, "invented parser failure"):
            execution.load_envelope(path, sha, admitted, loader=loader)
        self.assertEqual(admitted.counts["checkpoint_load"], 1)

    def test_fake_loader_hash_guard_and_weights_only_success(self):
        admitted, _, _ = self.admission()
        path = self.root / "invented.pt"
        raw = b"not a model"
        path.write_bytes(raw)
        state = {"invented": [1, 2]}
        loader = Mock(return_value={"state": state, "sha256": state_digest(state)})
        with self.assertRaisesRegex(ValueError, "hash differs"):
            execution.load_envelope(path, "0" * 64, admitted, loader=loader)
        loader.assert_not_called()
        self.assertEqual(admitted.counts, {})
        self.assertEqual(execution.load_envelope(path, hashlib.sha256(raw).hexdigest(),
                                               admitted, loader=loader), state)
        self.assertEqual(loader.call_args.kwargs, {"map_location": "cpu", "weights_only": True})

    def test_campaign_binding_rejects_other_root_config_stream_budget_and_backend(self):
        admitted, budget, _ = self.admission()
        class InventedBackend:
            pass
        backend = InventedBackend()
        backend.admit = admitted
        args = [self.output, self.packet["scientific_config"], self.packet["streams"], budget, backend]
        with patch.object(execution, "DynamicPatientBackend", InventedBackend):
            admitted.bind_campaign(*args)
            for index, wrong in ((0, self.root / "other"), (1, {}), (2, {}),
                                 (3, copy.copy(budget)), (4, object())):
                changed = list(args)
                changed[index] = wrong
                with self.subTest(index=index), self.assertRaises(PermissionError):
                    admitted.bind_campaign(*changed)
            backend.admit = object()
            with self.assertRaises(PermissionError):
                admitted.bind_campaign(*args)


class RecoveryBindingTests(RecoveryAdmissionFixture):
    def test_source_and_input_verification_uses_hashes_without_loading_models(self):
        mocks = self.binding_mocks()
        execution.verify_bindings(self.root, self.packet)
        self.assertEqual(mocks["git_run"].call_count, 2)
        mocks["recovery_config"].assert_called_once_with(self.packet["original_config"])
        mocks["recovery_budget_plan"].assert_called_once_with(self.packet["original_config"])
        self.assertEqual(mocks["file_record"].call_count, 3)

    def test_source_runtime_effective_config_and_budget_drift_are_rejected(self):
        mocks = self.binding_mocks()
        for name in ("source_locks", "runtime_record", "recovery_config", "recovery_budget_plan"):
            original = mocks[name].return_value
            mocks[name].return_value = {"changed": True}
            with self.subTest(name=name), self.assertRaises(ValueError):
                execution.verify_bindings(self.root, self.packet)
            mocks[name].return_value = original

    def test_protocol_proposal_and_any_input_drift_are_rejected(self):
        mocks = self.binding_mocks()
        original = mocks["file_record"].side_effect
        for changed_name in (execution.PROPOSAL, execution.PROTOCOL, "invented.pt"):
            def record(root, name):
                result = original(root, name)
                if name == changed_name:
                    result["sha256"] = "f" * 64
                return result
            mocks["file_record"].side_effect = record
            with self.subTest(name=changed_name), self.assertRaises(ValueError):
                execution.verify_bindings(self.root, self.packet)

    def test_git_ancestry_failure_stops_before_hash_or_runtime_work(self):
        mocks = self.binding_mocks()
        mocks["git_run"].side_effect = execution.subprocess.CalledProcessError(1, "invented git")
        with self.assertRaises(execution.subprocess.CalledProcessError):
            execution.verify_bindings(self.root, self.packet)
        mocks["source_locks"].assert_not_called()
        mocks["runtime_record"].assert_not_called()

    def test_wrong_workspace_branch_or_dirty_tree_stops_before_source_binding(self):
        with patch.object(execution, "committed_json", side_effect=lambda root, name:
                          self.auth if name == execution.AUTHORIZATION else self.packet), \
                patch.object(execution, "verify_bindings") as verify:
            for branch, status in (("wrong-branch", ""), (execution.BRANCH, "dirty")):
                with patch.object(execution, "git", side_effect=[branch, status]):
                    with self.subTest(branch=branch, status=status), self.assertRaises(ValueError):
                        execution.approved(self.root)
            with patch.object(execution, "git", return_value=execution.BRANCH):
                with self.assertRaises(ValueError):
                    execution.approved(self.root / "different-workspace")
            verify.assert_not_called()


class RecoveryLaunchTests(RecoveryAdmissionFixture):
    def launch_mocks(self, supervisor_result=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(execution, "approved", return_value=(self.packet, self.auth)))
        stack.enter_context(patch.object(execution, "git", return_value="invented-head"))
        clock = stack.enter_context(patch.object(execution, "shared_monotonic", return_value=100.0))
        supervisor = stack.enter_context(patch.object(execution, "supervise_dynamic", return_value=supervisor_result))
        return clock, supervisor

    def test_missing_authorization_has_no_claim_or_child(self):
        with patch.object(execution, "committed_json", side_effect=PermissionError("not approved")), \
                patch.object(execution, "supervise_dynamic") as supervisor:
            with self.assertRaises(PermissionError):
                execution.launch(self.root)
        supervisor.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_expired_startup_origin_prevents_claim_and_child(self):
        clock, supervisor = self.launch_mocks()
        clock.return_value = 102.0
        with self.assertRaises(TimeoutError):
            execution.launch(self.root, started=100.0)
        supervisor.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failed_dispatch_consumes_exclusive_root_and_does_not_retry(self):
        _, supervisor = self.launch_mocks()
        supervisor.side_effect = RuntimeError("invented launch failure")
        self.assertEqual(execution.launch(self.root, started=100.0), 1)
        self.assertTrue((self.output / "launcher/launch-failure.json").exists())
        with self.assertRaises(FileExistsError):
            execution.launch(self.root, started=100.0)
        supervisor.assert_called_once()

    def test_original_started_value_is_forwarded_and_closure_includes_terminal(self):
        result = {"passed": True, "final_deadline": 108.0}
        _, supervisor = self.launch_mocks(result)
        def supervise(*args, **kwargs):
            self.assertEqual(kwargs["started"], 99.0)
            (kwargs["launcher"] / "stderr.log").write_bytes(b"")
            return result
        supervisor.side_effect = supervise
        def archive(source, destination):
            self.assertEqual(source, self.output / "launcher")
            self.assertEqual(destination, self.output / "archives/completed-launcher.tar.gz")
            terminal = json.loads((source / "terminal.json").read_text())
            self.assertEqual(terminal["status"], "completed")
            self.assertTrue(terminal["requires_successful_closure_archive"])
            return {"fixture": "archive mocked; no research payload"}
        with patch.object(execution, "verify_completed", return_value=True), \
                patch.object(execution, "create_archive", side_effect=archive) as create:
            self.assertEqual(execution.launch(self.root, started=99.0), 0)
        create.assert_called_once()
        receipt = json.loads((self.output / "closure-archive-receipt.json").read_text())
        self.assertTrue(receipt["local_archive_verified"])
        self.assertFalse(receipt["dropbox_exported"])

    def test_closure_archive_overrun_is_failure_without_retry(self):
        result = {"passed": True, "final_deadline": 108.0}
        clock, supervisor = self.launch_mocks(result)
        def supervise(*args, **kwargs):
            (kwargs["launcher"] / "stderr.log").write_bytes(b"")
            return result
        supervisor.side_effect = supervise
        def archive(*args):
            clock.return_value = 109.0
            return {"fixture": True}
        with patch.object(execution, "verify_completed", return_value=True), \
                patch.object(execution, "create_archive", side_effect=archive):
            self.assertEqual(execution.launch(self.root, started=100.0), 1)
        self.assertTrue((self.output / "launcher/launch-failure.json").exists())
        self.assertFalse((self.output / "closure-archive-receipt.json").exists())
        clock.return_value = 100.0
        with self.assertRaises(FileExistsError):
            execution.launch(self.root, started=100.0)
        supervisor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
