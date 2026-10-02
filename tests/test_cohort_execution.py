"""Invented admission/dispatch only; no models, patients or optimizer calls."""

import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.rl import cohort_execution as execution
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import audit_stream_collisions, digest
from tests.test_cohort_objective_plan import fixture as scope_fixture
from tests.test_time_baseline_campaign import saved_design


def fixture(root):
    base, proposal = scope_fixture()
    base["scientific_execution_authorized"] = False
    proposal.update(base_proposal=execution.BASE_PROPOSAL, base_proposal_content_sha256=digest(base),
        economic_endpoint=63, attempts=1, automatic_retry=False, automatic_followon=False,
        no_new_penalty_or_salvage_weights=True)
    original = saved_design()
    record = {"sha256": "b" * 64}
    packet = {"format": "cohort-objective-frozen-v1", "scientific_execution_authorized": False,
        "ready_to_launch": False, "workspace": str(root), "branch": execution.BRANCH,
        "result_root": execution.RUN, "frozen_at_utc": "2026-10-02T10:00:00+00:00",
        "implementation_commit": "a" * 40, "proposal_data": proposal, "base_proposal_data": base,
        "original_config": original, "initializer_streams": {"fixture": "metadata only"},
        "initializers": {str(b): {"path": f"invented/block{b}.pt", "sha256": "c" * 64} for b in (60, 61, 62)},
        "qualified_blocks": {str(b): {"passed": True} for b in (60, 61, 62)},
        "scientific_config": execution.cohort_config(original, base, proposal),
        "streams": execution.cohort_stream_manifest(base, proposal),
        "budget_plan": execution.cohort_budget_plan(base, proposal),
        "source_files": {}, "input_files": {}, "runtime": {"invented": True},
        "local_seed_collision_audit": {"files": []},
        "proposal": record, "protocol": record, "base_proposal": record, "bound": record,
        "approval_intent": record,
        "approval_intent_data": {
            "format": "cohort-objective-approval-intent-v1", "recorded_at_utc": "2026-10-02T09:00:00Z",
            "user_literal": "INVENTED START FIXTURE, NOT HUMAN AUTHORIZATION",
            "proposal_sha256": record["sha256"], "protocol_sha256": record["sha256"],
            "objective_amendment_approved": True, "main_episodes": 522,
            "environment_calls_max": 33186, "optimizer_calls_max": 1920,
            "wall_seconds_max": 10800, "attempts": 1, "automatic_retry": False,
            "automatic_followon": False, "no_new_weights_or_search": True,
            "approval_precedes_final_source_freeze": True, "execution_started": False,
            "requires_final_packet_bound_authorization_before_claim": True}}
    packet["packet_sha256"] = digest(packet)
    auth = {"format": "cohort-objective-authorization-v1", "approved": True,
        "packet_sha256": packet["packet_sha256"], "implementation_commit": packet["implementation_commit"],
        "numeric_scope_sha256": digest(proposal), "protocol_sha256": packet["protocol"]["sha256"],
        "runtime_sha256": digest(packet["runtime"]), "limits": execution.approved_limits(packet),
        "approval_intent_sha256": packet["approval_intent"]["sha256"],
        "automatic_retry": False, "automatic_followon": False, "remote_or_dropbox_actions": False,
        "objective_amendment_approved": True, "new_weights_or_search_authorized": False,
        "old_attempts_remain_terminal": True, "reuse_completed_qualification_without_rescoring": True,
        "user_approval": {"user": "Zhaowei", "verbatim": "INVENTED START FIXTURE, NOT HUMAN AUTHORIZATION",
            "recorded_at_utc": "2026-10-02T09:00:00Z",
            "request_precedes_freeze": True, "numeric_scope_sha256": digest(proposal),
            "protocol_sha256": packet["protocol"]["sha256"]}}
    return packet, auth


class CohortExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / execution.RUN
        self.packet, self.auth = fixture(self.root)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("src.rl.experiment.build_env", "src.rl.strict_frozen_policy.StrictFrozenPolicy",
                     "src.rl.cohort_execution.restore_saved_policy", "torch.load",
                     "torch.nn.Module._call_impl", "torch.optim.Adam.step", "torch.optim.SGD.step"):
            guard = self.stack.enter_context(patch(name, side_effect=AssertionError("scientific work forbidden")))
            self.addCleanup(guard.assert_not_called)

    def admission(self):
        budget = SimpleNamespace(started=100., active="runtime_input_binding", plan=self.packet["budget_plan"],
            path=self.output / "launcher/budget.jsonl", check=Mock(return_value=100.))
        claim = dict(pid=os.getppid(), packet_sha256=self.packet["packet_sha256"],
                     authorization_sha256=digest(self.auth), clock_id=execution.CLOCK_ID, started=100.)
        write_json_once(self.output / "launcher/claim.json", claim)
        return execution.CohortAdmission(self.root, self.packet, self.auth, claim, budget), budget, claim

    def test_exact_scope_and_auth_fields(self):
        execution.validate_authorization(self.auth, self.packet)
        self.assertEqual([self.auth["limits"][k] for k in ("episodes", "environment_steps", "optimizer_calls", "seconds")],
                         [522, 33186, 1920, 10800])
        self.assertEqual(len(self.packet["budget_plan"]["sections"]), 33)
        for key in self.auth["limits"]:
            changed = copy.deepcopy(self.auth)
            changed["limits"][key] = "changed"
            with self.subTest(limit=key), self.assertRaises(PermissionError):
                execution.validate_authorization(changed, self.packet)
        for key in self.auth:
            changed = copy.deepcopy(self.auth)
            changed.pop(key)
            with self.subTest(field=key), self.assertRaises(PermissionError):
                execution.validate_authorization(changed, self.packet)

    def test_truthful_pre_freeze_request_and_separate_runtime_binding(self):
        for key, value in (("recorded_at_utc", "2026-10-02T10:00:01Z"),
                           ("recorded_at_utc", "2026-10-02T09:00:00"),
                           ("request_precedes_freeze", False), ("numeric_scope_sha256", "changed"),
                           ("protocol_sha256", "changed"), ("verbatim", " "),
                           ("verbatim", "different request"), ("recorded_at_utc", "2026-10-02T08:41:13Z")):
            changed = copy.deepcopy(self.auth)
            changed["user_approval"][key] = value
            with self.subTest(field=key), self.assertRaises(PermissionError):
                execution.validate_authorization(changed, self.packet)
        for key in ("runtime_sha256", "numeric_scope_sha256", "protocol_sha256"):
            with self.subTest(field=key), self.assertRaises(PermissionError):
                execution.validate_authorization(dict(self.auth, **{key: "changed"}), self.packet)

    def test_approval_intent_scope_hashes_and_global_cap_are_binding(self):
        for key, value in (("environment_calls_max", 33187), ("proposal_sha256", "changed"),
                           ("protocol_sha256", "changed"), ("no_new_weights_or_search", False),
                           ("recorded_at_utc", "2026-10-02T10:01:00Z")):
            changed = copy.deepcopy(self.packet)
            changed["approval_intent_data"][key] = value
            changed.pop("packet_sha256")
            changed["packet_sha256"] = digest(changed)
            auth = dict(self.auth, packet_sha256=changed["packet_sha256"])
            with self.subTest(field=key), self.assertRaises(PermissionError):
                execution.validate_authorization(auth, changed)
        with self.assertRaises(PermissionError):
            execution.validate_authorization(dict(self.auth, approval_intent_sha256="changed"), self.packet)

    def test_packet_flags_never_grant_permission(self):
        changed = copy.deepcopy(self.packet)
        changed["streams"]["namespace"] = "changed"
        with self.assertRaises(ValueError):
            execution.validate_authorization(self.auth, changed)
        for key in ("ready_to_launch", "scientific_execution_authorized"):
            changed = copy.deepcopy(self.packet)
            changed[key] = True
            changed.pop("packet_sha256")
            changed["packet_sha256"] = digest(changed)
            with self.subTest(field=key), self.assertRaises(PermissionError):
                execution.validate_authorization(dict(self.auth, packet_sha256=changed["packet_sha256"]), changed)
        base, proposal = copy.deepcopy((self.packet["base_proposal_data"], self.packet["proposal_data"]))
        proposal["no_new_penalty_or_salvage_weights"] = False
        with self.assertRaises(ValueError):
            execution._scope(base, proposal)

    def test_live_parent_claim_clock_budget_and_terminal(self):
        admitted, budget, claim = self.admission()
        for key, value in (("pid", -1), ("clock_id", "wrong"), ("started", 99),
                           ("packet_sha256", "wrong"), ("authorization_sha256", "wrong")):
            with self.subTest(field=key), self.assertRaises(PermissionError):
                execution.CohortAdmission(self.root, self.packet, self.auth, dict(claim, **{key: value}), budget)
        with patch.object(execution.os, "getppid", return_value=-1), self.assertRaises(PermissionError):
            admitted("checkpoint_load")
        write_json_once(self.output / "launcher/terminal.json", {"status": "failed"})
        with self.assertRaises(PermissionError):
            admitted("checkpoint_load")
        self.assertEqual(admitted.counts, {})

    def test_fixed_caps_new_episode_phases_and_preflight_only_parity(self):
        admitted, budget, _ = self.admission()
        self.assertEqual(admitted.caps, dict(checkpoint_load=3, reference_checkpoint_load=3,
            layout_environment_build=3, episode_build=522, parity_environment_build=3))
        for phase in self.packet["budget_plan"]["sections"]:
            budget.active = phase
            if phase != "same_start_preflight":
                with self.subTest(phase=phase), self.assertRaises(PermissionError):
                    admitted("parity_environment_build")
        for phase in ("runtime_input_binding", "all_model_seal", "raw_verification", "current_ppo/block60",
                      "time_baseline_ppo/block60", "cohort_ppo/block99", "final_evaluation/unknown"):
            budget.active = phase
            with self.subTest(phase=phase), self.assertRaises(PermissionError):
                admitted("episode_build")
        for phase in ("same_start_preflight", "window_ppo/block60", "cohort_ppo/block60", "bc_continue/block60"):
            budget.active = phase
            admitted("episode_build")
        budget.active = next(p for p in self.packet["budget_plan"]["sections"] if p.startswith("final_evaluation/"))
        admitted("episode_build")
        with self.assertRaises(PermissionError):
            admitted("checkpoint_load")
        budget.active = "same_start_preflight"
        for _ in range(3):
            admitted("parity_environment_build")
        with self.assertRaises(ValueError):
            admitted("parity_environment_build")
        admitted.counts["episode_build"] = 522
        with self.assertRaises(ValueError):
            admitted("episode_build")

    def test_predebit_failure_no_refund_or_overwrite(self):
        admitted, budget, _ = self.admission()
        path = self.root / "invented.pt"
        path.write_bytes(b"not a model")

        def fake_loader(*args, **kwargs):
            self.assertEqual(admitted.counts["checkpoint_load"], 1)
            self.assertTrue((self.output / "launcher/operations/checkpoint_load-0001.json").exists())
            raise RuntimeError("invented parser failure")

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
        budget.check.side_effect = TimeoutError("fixture deadline")
        with self.assertRaises(TimeoutError):
            admitted("reference_checkpoint_load")

    def test_exact_cohort_backend_and_campaign_bindings(self):
        admitted, budget, _ = self.admission()

        class FakeBackend:
            pass

        backend = FakeBackend()
        backend.admit = admitted
        args = [self.output, self.packet["scientific_config"], self.packet["streams"], budget, backend]
        with patch.object(execution, "_backend_class", return_value=FakeBackend):
            admitted.bind_campaign(*args)
            for i, replacement in ((0, self.root), (1, {}), (2, {}), (3, object()), (4, object())):
                changed = list(args)
                changed[i] = replacement
                with self.subTest(argument=i), self.assertRaises(PermissionError):
                    admitted.bind_campaign(*changed)

    def test_binding_reconstruction_and_metadata_without_scoring(self):
        self.stack.enter_context(patch.object(execution.subprocess, "run"))
        for name, value in (("source_locks", {}), ("runtime_record", self.packet["runtime"]), ("seed_inventory", [])):
            self.stack.enter_context(patch.object(execution, name, return_value=value))
        self.stack.enter_context(patch.object(execution, "file_record", return_value=self.packet["protocol"]))
        self.stack.enter_context(patch.object(execution, "committed_json", return_value=self.packet["approval_intent_data"]))
        execution.verify_bindings(self.root, self.packet)
        for field in ("scientific_config", "budget_plan", "streams", "runtime", "source_files"):
            changed = copy.deepcopy(self.packet)
            changed[field] = {"changed": True}
            with self.subTest(field=field), self.assertRaises(ValueError):
                execution.verify_bindings(self.root, changed)
        with patch.object(execution, "seed_inventory", return_value=[self.root / "new.json"]), self.assertRaises(ValueError):
            execution.verify_bindings(self.root, self.packet)

    def test_seed_scope_keeps_closed_time_baseline_and_exact_seed_strings(self):
        names = ["experiments/configs/prior.json", execution.PRIOR_PACKET,
                 "results/dynamic_candidate_time_baseline_20261002/payload/old-streams.json",
                 "specs/2026-10-02-time-baseline-comparison/frozen.json"]
        for name in names:
            write_json_once(self.root / name, {"old_seed": 5})
        seed = self.packet["streams"]["environment"]["60"]["test"][0]
        write_json_once(self.root / execution.DIRECTORY / "prospective-streams.json", {"seed": seed})
        write_json_once(self.output / "payload/seed-manifest.json", {"seed": seed})
        with patch.object(execution, "git", return_value=names[0]):
            found = execution.seed_inventory(self.root)
        self.assertEqual(set(found), {self.root / n for n in names})
        self.assertTrue(audit_stream_collisions({"seed": seed}, found)["passed"])
        (self.root / names[2]).write_text(json.dumps({"seed": seed}))
        self.assertFalse(audit_stream_collisions({"seed": seed}, found)["passed"])

    def test_fake_only_freeze_reuses_inputs_and_keeps_proposals_false(self):
        before = copy.deepcopy(self.packet)
        instant = execution.datetime.fromisoformat(self.packet["frozen_at_utc"])
        clock = self.stack.enter_context(patch.object(execution, "datetime", wraps=execution.datetime))
        clock.now.return_value = instant
        prior = {key: self.packet[key] for key in ("original_config", "qualified_blocks", "initializers")}
        prior["streams"] = self.packet["initializer_streams"]
        self.stack.enter_context(patch.object(execution, "git", side_effect=[execution.BRANCH, "", "a" * 40]))
        self.stack.enter_context(patch.object(execution, "committed_json", side_effect=[
            self.packet["base_proposal_data"], self.packet["proposal_data"], self.packet["approval_intent_data"]]))
        reused = self.stack.enter_context(patch.object(execution, "input_bindings", return_value=(prior, {})))
        self.stack.enter_context(patch.object(execution, "seed_inventory", return_value=[]))
        self.stack.enter_context(patch.object(execution, "audit_stream_collisions", return_value={"passed": True, "files": []}))
        self.stack.enter_context(patch.object(execution, "source_locks", return_value={}))
        self.stack.enter_context(patch.object(execution, "runtime_record", return_value=self.packet["runtime"]))
        self.stack.enter_context(patch.object(execution, "file_record", return_value=self.packet["protocol"]))
        packet = execution.freeze(self.root)
        reused.assert_called_once_with(self.root)
        self.assertEqual(self.packet, before)
        self.assertFalse(packet["scientific_execution_authorized"])
        self.assertFalse(packet["base_proposal_data"]["scientific_execution_authorized"])
        self.assertEqual(packet["new_checkpoint_loads"], 0)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_approved_reads_only_committed_scope_before_bindings(self):
        with patch.object(execution, "committed_json", side_effect=[self.auth, self.packet]) as read, \
                patch.object(execution, "git", side_effect=[execution.BRANCH, ""]), \
                patch.object(execution, "verify_bindings") as verify:
            self.assertEqual(execution.approved(self.root), (self.packet, self.auth))
        self.assertEqual([c.args[1] for c in read.call_args_list], [execution.AUTHORIZATION, execution.FROZEN])
        verify.assert_called_once_with(self.root, self.packet)

    def test_missing_approval_cannot_claim_or_dispatch(self):
        with patch.object(execution, "committed_json", side_effect=PermissionError("not approved")), \
                patch.object(execution, "supervise_time_baseline") as supervisor:
            with self.assertRaises(PermissionError):
                execution.launch(self.root)
            supervisor.assert_not_called()
        self.assertFalse(self.output.exists())

    def launch_mocks(self):
        self.stack.enter_context(patch.object(execution, "approved", return_value=(self.packet, self.auth)))
        self.stack.enter_context(patch.object(execution, "git", return_value="fixture-head"))
        clock = self.stack.enter_context(patch.object(execution, "shared_monotonic", return_value=100.))
        supervisor = self.stack.enter_context(patch.object(execution, "supervise_time_baseline"))
        return clock, supervisor

    def test_failed_dispatch_consumes_attempt_and_setup_timeout_does_not_claim(self):
        clock, supervisor = self.launch_mocks()
        clock.return_value = 401.
        with self.assertRaises(TimeoutError):
            execution.launch(self.root, started=100.)
        self.assertFalse(self.output.exists())
        clock.return_value = 100.
        supervisor.side_effect = RuntimeError("invented dispatch failure")
        self.assertEqual(execution.launch(self.root, started=100.), 1)
        self.assertTrue((self.output / "launcher/launch-failure.json").exists())
        with self.assertRaises(FileExistsError):
            execution.launch(self.root, started=100.)
        supervisor.assert_called_once()

    def test_successful_closure_and_deadline_overrun(self):
        clock, supervisor = self.launch_mocks()

        def supervised(command, **kwargs):
            self.assertEqual(command[-2:], ["experiments.scripts.run_cohort_objective", "--child"])
            self.assertEqual(kwargs["plan"], self.packet["budget_plan"])
            (kwargs["launcher"] / "stderr.log").write_bytes(b"")
            return {"passed": True, "final_deadline": 108.}

        supervisor.side_effect = supervised
        with patch.object(execution, "verify_completed", return_value=True), \
                patch.object(execution, "create_archive", return_value={"invented_archive": True}):
            self.assertEqual(execution.launch(self.root, started=99.), 0)
        receipt = json.loads((self.output / "closure-archive-receipt.json").read_text())
        self.assertTrue(receipt["local_archive_verified"])
        self.assertFalse(receipt["dropbox_exported"])

    def test_archive_overrun_has_no_successful_closure_receipt(self):
        clock, supervisor = self.launch_mocks()

        def supervised(*args, **kwargs):
            (kwargs["launcher"] / "stderr.log").write_bytes(b"")
            return {"passed": True, "final_deadline": 108.}

        def archive(*args):
            clock.return_value = 109.
            return {"invented_archive": True}

        supervisor.side_effect = supervised
        with patch.object(execution, "verify_completed", return_value=True), \
                patch.object(execution, "create_archive", side_effect=archive):
            self.assertEqual(execution.launch(self.root, started=99.), 1)
        self.assertFalse((self.output / "closure-archive-receipt.json").exists())
        self.assertTrue((self.output / "launcher/launch-failure.json").exists())

    def test_child_wires_cohort_base_and_preserves_locks_without_models(self):
        admitted, budget, claim = self.admission()
        claim["head"] = "fixture-head"
        (self.output / "launcher/claim.json").write_text(json.dumps(claim))
        campaign = Mock()
        campaign.run.return_value = 0
        module = ModuleType("src.rl.cohort_campaign")
        module.CohortCampaign = Mock(return_value=campaign)
        with patch.object(execution, "approved", return_value=(self.packet, self.auth)), \
                patch.object(execution, "git", return_value="fixture-head"), \
                patch.object(execution, "DynamicCandidateBudget", return_value=budget), \
                patch.object(execution, "_backend_class", return_value=Mock()), \
                patch.object(execution, "copy_verified") as copy_lock, \
                patch.dict("sys.modules", {"src.rl.cohort_campaign": module}):
            self.assertEqual(execution.child(self.root), 0)
        args, kwargs = module.CohortCampaign.call_args
        self.assertEqual(args[:5], (self.output, self.packet["original_config"], self.packet["base_proposal_data"],
                                   self.packet["proposal_data"], self.packet["streams"]))
        self.assertFalse(kwargs["engineering_only"])
        self.assertIsInstance(kwargs["execution_admission"], execution.CohortAdmission)
        self.assertEqual(copy_lock.call_count, 7)
        campaign.begin_next.assert_called_once()
        campaign.run.assert_called_once()
        with patch.object(execution, "verify_bindings"), self.assertRaises(ValueError):
            kwargs["final_lock_check"]()


if __name__ == "__main__":
    unittest.main()
