"""Invented admission and dispatch receipts; all scientific calls are blocked."""

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

from src.rl import time_baseline_execution as execution
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest, audit_stream_collisions
from tests.test_time_baseline_campaign import proposed, saved_design


def fixture(root):
    proposal, original = proposed(), saved_design()
    packet = {"format": "time-baseline-frozen-v1", "scientific_execution_authorized": False,
        "ready_to_launch": False, "workspace": str(root), "branch": execution.BRANCH,
        "implementation_commit": "a" * 40, "proposal_data": proposal, "original_config": original,
        "initializer_streams": {"fixture": "old initialization metadata only"},
        "scientific_config": execution.time_baseline_config(original, proposal),
        "streams": execution.time_baseline_stream_manifest(proposal),
        "budget_plan": execution.time_baseline_budget_plan(proposal),
        "source_files": {}, "input_files": {}, "runtime": {"invented": True},
        "local_seed_collision_audit": {"files": []},
        "proposal": {}, "protocol": {}, "reward_pivot": {}}
    packet["packet_sha256"] = digest(packet)
    auth = {"format": "time-baseline-authorization-v1", "approved": True,
        "packet_sha256": packet["packet_sha256"], "implementation_commit": packet["implementation_commit"],
        "limits": execution.approved_limits(packet), "automatic_retry": False,
        "remote_or_dropbox_actions": False, "reward_change_authorized": False,
        "old_attempts_remain_terminal": True, "reuse_completed_qualification_without_rescoring": True,
        "user_approval": {"user": "Zhaowei", "verbatim": "INVENTED UNIT TEST, NOT ACTUAL AUTHORIZATION",
                          "recorded_at_utc": "fixture", "reply_to_numeric_package": True}}
    return packet, auth


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / execution.RUN
        self.packet, self.auth = fixture(self.root)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("src.rl.experiment.build_env", "src.rl.strict_frozen_policy.StrictFrozenPolicy",
                     "src.rl.time_baseline_execution.restore_saved_policy", "torch.load",
                     "torch.nn.Module._call_impl", "torch.optim.Adam.step", "torch.optim.SGD.step"):
            guard = self.stack.enter_context(patch(name, side_effect=AssertionError("scientific work forbidden")))
            self.addCleanup(guard.assert_not_called)

    def admission(self):
        budget = SimpleNamespace(started=100., active="runtime_input_binding", plan=self.packet["budget_plan"],
            path=self.output / "launcher/budget.jsonl", check=Mock(return_value=100.))
        claim = dict(pid=os.getppid(), packet_sha256=self.packet["packet_sha256"],
                     authorization_sha256=digest(self.auth), clock_id=execution.CLOCK_ID, started=100.)
        write_json_once(self.output / "launcher/claim.json", claim)
        return execution.TimeBaselineAdmission(self.root, self.packet, self.auth, claim, budget), budget, claim

    def test_complete_numeric_approval_and_phase_plan_required(self):
        execution.validate_authorization(self.auth, self.packet)
        self.assertEqual(self.auth["limits"]["episodes"], 522)
        self.assertEqual(self.auth["limits"]["environment_steps"], 27216)
        self.assertEqual(self.auth["limits"]["optimizer_calls"], 1920)
        self.assertEqual(self.auth["limits"]["seconds"], 10800)
        for name in self.auth["limits"]:
            altered = copy.deepcopy(self.auth)
            altered["limits"][name] = "changed"
            with self.subTest(limit=name), self.assertRaises(PermissionError):
                execution.validate_authorization(altered, self.packet)
        for name in ("approved", "packet_sha256", "implementation_commit", "automatic_retry",
                     "remote_or_dropbox_actions", "reward_change_authorized", "old_attempts_remain_terminal",
                     "reuse_completed_qualification_without_rescoring"):
            altered = copy.deepcopy(self.auth)
            altered.pop(name)
            with self.subTest(field=name), self.assertRaises(PermissionError):
                execution.validate_authorization(altered, self.packet)
        altered = copy.deepcopy(self.auth)
        altered["user_approval"]["reply_to_numeric_package"] = False
        with self.assertRaises(PermissionError):
            execution.validate_authorization(altered, self.packet)

    def test_packet_and_preparation_flags_cannot_grant_permission(self):
        altered = copy.deepcopy(self.packet)
        altered["streams"]["namespace"] = "changed"
        with self.assertRaises(ValueError):
            execution.validate_authorization(self.auth, altered)
        altered = copy.deepcopy(self.packet)
        altered["ready_to_launch"] = True
        altered.pop("packet_sha256")
        altered["packet_sha256"] = digest(altered)
        auth = dict(self.auth, packet_sha256=altered["packet_sha256"])
        with self.assertRaises(PermissionError):
            execution.validate_authorization(auth, altered)

    def test_live_claim_parent_clock_budget_and_terminal(self):
        admitted, budget, claim = self.admission()
        for name, value in (("pid", -1), ("clock_id", "wrong"), ("started", 99),
                            ("packet_sha256", "wrong"), ("authorization_sha256", "wrong")):
            with self.subTest(field=name), self.assertRaises(PermissionError):
                execution.TimeBaselineAdmission(self.root, self.packet, self.auth, dict(claim, **{name: value}), budget)
        with patch.object(execution.os, "getppid", return_value=-1), self.assertRaises(PermissionError):
            admitted("checkpoint_load")
        write_json_once(self.output / "launcher/terminal.json", {"status": "failed"})
        with self.assertRaises(PermissionError):
            admitted("checkpoint_load")
        self.assertEqual(admitted.counts, {})

    def test_new_episode_phases_and_caps_no_old_continuation(self):
        admitted, budget, _ = self.admission()
        for name in ("runtime_input_binding", "all_model_seal", "raw_verification", "payload_archive",
                     "supervisor_closure", "ppo_continuation/block60"):
            budget.active = name
            with self.subTest(phase=name), self.assertRaises(PermissionError):
                admitted("episode_build")
        for name in ("same_start_preflight", "current_ppo/block60", "time_baseline_ppo/block60",
                     "bc_continue/block60", "final_evaluation/block60/own_frozen"):
            budget.active = name
            admitted("episode_build")
        self.assertEqual(admitted.counts["episode_build"], 5)
        admitted.counts["episode_build"] = admitted.caps["episode_build"]
        with self.assertRaises(ValueError):
            admitted("episode_build")
        with self.assertRaises(PermissionError):
            admitted("checkpoint_load")

    def test_predebit_and_no_refund_or_receipt_overwrite(self):
        admitted, budget, _ = self.admission()
        path = self.root / "invented.pt"
        path.write_bytes(b"not a model")
        def fake_loader(*args, **kwargs):
            self.assertEqual(admitted.counts["checkpoint_load"], 1)
            self.assertTrue((self.output / "launcher/operations/checkpoint_load-0001.json").exists())
            raise RuntimeError("fake parser failure")
        with self.assertRaises(RuntimeError):
            execution.load_envelope(path, hashlib.sha256(path.read_bytes()).hexdigest(), admitted, loader=fake_loader)
        self.assertEqual(admitted.counts["checkpoint_load"], 1)
        for _ in range(2):
            admitted("checkpoint_load")
        with self.assertRaises(ValueError):
            admitted("checkpoint_load")
        write_json_once(self.output / "launcher/operations/layout_environment_build-0001.json", {"preserved": True})
        with self.assertRaises(FileExistsError):
            admitted("layout_environment_build")
        self.assertNotIn("layout_environment_build", admitted.counts)
        budget.check.side_effect = TimeoutError("fixture deadline")
        with self.assertRaises(TimeoutError):
            admitted("reference_checkpoint_load")

    def test_exact_backend_and_stream_binding(self):
        admitted, budget, _ = self.admission()
        class FakeBackend:
            pass
        backend = FakeBackend()
        backend.admit = admitted
        args = [self.output, self.packet["scientific_config"], self.packet["streams"], budget, backend]
        with patch.object(execution, "TimeBaselinePatientBackend", FakeBackend):
            admitted.bind_campaign(*args)
            for i, replacement in ((0, self.root), (1, {}), (2, {}), (3, object()), (4, object())):
                changed = list(args)
                changed[i] = replacement
                with self.assertRaises(PermissionError):
                    admitted.bind_campaign(*changed)

    def test_binding_checks_new_plan_streams_and_seed_files_without_scoring(self):
        self.stack.enter_context(patch.object(execution.subprocess, "run"))
        for name, value in (("source_locks", {}), ("runtime_record", self.packet["runtime"]), ("seed_inventory", [])):
            self.stack.enter_context(patch.object(execution, name, return_value=value))
        record = self.stack.enter_context(patch.object(execution, "file_record", return_value={}))
        execution.verify_bindings(self.root, self.packet)
        self.assertEqual(record.call_count, 3)
        for field in ("scientific_config", "budget_plan", "streams"):
            changed = copy.deepcopy(self.packet)
            changed[field] = {}
            with self.subTest(field=field), self.assertRaises(ValueError):
                execution.verify_bindings(self.root, changed)
        with patch.object(execution, "seed_inventory", return_value=[self.root / "new.json"]), self.assertRaises(ValueError):
            execution.verify_bindings(self.root, self.packet)

    def test_seed_scope_includes_preserved_old_attempt_and_exact_decimal_strings(self):
        files = ["experiments/configs/prior.json", execution.PRIOR_PACKET,
                 "results/dynamic_candidate_pilot_20261001/payload/old-seed-manifest.json"]
        seed = self.packet["streams"]["environment"]["60"]["test"][0]
        for name in files:
            write_json_once(self.root / name, {"old_seed": 5})
        write_json_once(self.root / execution.DIRECTORY / "prospective-streams.json", {"seed": seed})
        with patch.object(execution, "git", return_value=files[0]):
            found = execution.seed_inventory(self.root)
        self.assertEqual(set(found), {self.root / n for n in files})
        audit = audit_stream_collisions({"seed": seed}, found)
        self.assertTrue(audit["passed"])
        (self.root / files[2]).write_text(json.dumps({"seed": seed}))
        self.assertFalse(audit_stream_collisions({"seed": seed}, found)["passed"])

    def test_missing_numeric_authorization_cannot_claim_or_dispatch(self):
        with patch.object(execution, "committed_json", side_effect=PermissionError("not approved")), \
                patch("src.rl.time_baseline_watchdog.supervise_time_baseline") as supervisor:
            with self.assertRaises(PermissionError):
                execution.launch(self.root)
            supervisor.assert_not_called()
        self.assertFalse(self.output.exists())

    def launch_mocks(self):
        self.stack.enter_context(patch.object(execution, "approved", return_value=(self.packet, self.auth)))
        self.stack.enter_context(patch.object(execution, "git", return_value="fixture-head"))
        clock = self.stack.enter_context(patch.object(execution, "shared_monotonic", return_value=100.))
        supervisor = self.stack.enter_context(patch("src.rl.time_baseline_watchdog.supervise_time_baseline"))
        return clock, supervisor

    def test_failed_launch_consumes_root_and_cannot_retry(self):
        _, supervisor = self.launch_mocks()
        supervisor.side_effect = RuntimeError("invented dispatch failure")
        self.assertEqual(execution.launch(self.root, started=100.), 1)
        self.assertTrue((self.output / "launcher/launch-failure.json").exists())
        with self.assertRaises(FileExistsError):
            execution.launch(self.root, started=100.)
        supervisor.assert_called_once()

    def test_complete_closure_and_overrun_no_successful_archive_receipt(self):
        clock, supervisor = self.launch_mocks()
        def supervised(*args, **kwargs):
            self.assertEqual(kwargs["started"], 99.)
            self.assertEqual(args[0][-2:], ["experiments.scripts.run_time_baseline_comparison", "--child"])
            (kwargs["launcher"] / "stderr.log").write_bytes(b"")
            return {"passed": True, "final_deadline": 108.}
        supervisor.side_effect = supervised
        def archive(source, destination):
            self.assertEqual(json.loads((source / "terminal.json").read_text())["status"], "completed")
            clock.return_value = 109.
            return {"invented_archive": True}
        with patch.object(execution, "verify_completed", return_value=True), \
                patch.object(execution, "create_archive", side_effect=archive):
            self.assertEqual(execution.launch(self.root, started=99.), 1)
        self.assertFalse((self.output / "closure-archive-receipt.json").exists())
        self.assertTrue((self.output / "launcher/launch-failure.json").exists())


if __name__ == "__main__":
    unittest.main()
