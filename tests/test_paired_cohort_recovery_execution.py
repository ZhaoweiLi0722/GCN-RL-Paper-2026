"""Invented packet/claim/clock fixtures; no freeze, model or scientific calls."""

import copy
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl import paired_cohort_recovery_execution as execution
from src.rl.candidate_pilot_resources import digest
from src.rl.paired_cohort_plan import branch_plan, context_keys
from tests.test_paired_cohort_execution import save, reseal, synthetic_packet as old_packet
from tests.test_paired_cohort_recovery_resources import recovery


CLASS_COUNTS = {60: [(5, 6, 6), (5, 5, 5), (5, 5, 6), (5, 5, 6)],
                61: [(6, 6, 6), (6, 6, 6), (6, 5, 6), (6, 5, 6)],
                62: [(5, 6, 6), (6, 6, 5), (5, 6, 6), (5, 6, 5)]}


def artificial_manifest(packet):
    cfg, streams = packet["config"], packet["streams"]
    counts = {(b, c, t): CLASS_COUNTS[b][c][i] for b, c, t in context_keys(cfg)
              for i in [cfg["context_after_prefix_steps"].index(t)]}
    schedule = []
    for row in branch_plan(cfg, counts):
        name = f"block{row.block}/cohort{row.cohort}/after{row.after_prefix_steps}"
        schedule.append(dict(block=row.block, cohort=row.cohort, after_prefix_steps=row.after_prefix_steps,
            replication=row.replication, candidate_index=row.candidate, environment_calls=row.environment_calls,
            future_seed=streams["conditional_future"][name][row.replication],
            branch_id=f"branches/{name}/rep{row.replication}/class{row.candidate}"))
    def record(path):
        return dict(path=path, bytes=1, sha256="a" * 64)
    result = dict(format="paired-cohort-recovery-data-v1", old_run=execution.original.RUN,
        original_packet=packet["original_packet"],
        contexts=[dict(key=list(key), file=record(f"invented-context-{key}")) for key in counts],
        completed=[dict(branch_id=row["branch_id"], boundary=record("invented-boundary"), files={}) for row in schedule[:117]],
        remaining=schedule[117:], interrupted=[record(f"invented-partial-{i}") for i in range(3)],
        remaining_by_block={b: {k: v[k] for k in ("branches", "environment_calls")}
                            for b, v in packet["recovery_proposal"]["remaining_by_block"].items()},
        old_environment_charge=6246, preserved_interrupted_charge=1, new_environment_charge=25655,
        cumulative_environment_charge=31901, original_attempt_remains_terminal=True)
    return reseal(result, "manifest_sha256")


def synthetic_packet(test, root, *, base=None, manifest=None):
    """Invent authorization directly; freeze is never called, including fixtures."""
    old = old_packet(root)
    original_record = save(root, execution.ORIGINAL_PACKET, old)
    proposal_record = save(root, execution.PROPOSAL, recovery())
    for name, value in (("ORIGINAL_PACKET_SHA256", original_record["sha256"]),
                        ("PROPOSAL_SHA256", proposal_record["sha256"])):
        mocked = patch.object(execution, name, value)
        mocked.start()
        test.addCleanup(mocked.stop)
    packet = execution._metadata(root, None)
    if base is not None:
        for key in execution.INHERITED_FIELDS:
            if key in base:
                packet[key] = copy.deepcopy(base[key])
        packet["seed_reuse"]["original_streams_sha256"] = digest(packet["streams"])
    if manifest is None:
        manifest = artificial_manifest(packet)
    else:
        manifest = copy.deepcopy(manifest)
        manifest["original_packet"] = packet["original_packet"]
        reseal(manifest, "manifest_sha256")
    packet["recovery_manifest"] = manifest
    protocol = save(root, execution.PROTOCOL, dict(invented_protocol=True))
    intent = dict(format="paired-cohort-recovery-approval-intent-v1", approved=True,
        user_literal="SYNTHETIC AUTHORIZATION ONLY", original_attempt_remains_terminal=True,
        proposal_sha256=proposal_record["sha256"], protocol_sha256=protocol["sha256"])
    packet.update(format="paired-cohort-recovery-frozen-v1", source_frozen=True,
        proposal=proposal_record, protocol=protocol, intent=intent, intent_file=save(root, execution.INTENT, intent),
        source_files=old["source_files"], runtime=dict(invented=True), implementation_commit="f" * 40)
    return reseal(packet)


def labels_fixture(packet):
    cfg = packet["config"]
    counts = {(b, c, t): CLASS_COUNTS[b][c][cfg["context_after_prefix_steps"].index(t)] for b, c, t in context_keys(cfg)}
    labels = [dict(block=b, cohort=c, after_prefix_steps=t, class_keys=[[k] for k in range(n)],
        public_example={"candidates": {"class_keys": [[k] for k in range(n)]}}) for (b, c, t), n in counts.items()]
    return reseal(dict(format="paired-cohort-label-dataset-v1", config_sha256=digest(cfg), labels=labels,
        branch_outcomes=[dict(block=b.block, cohort=b.cohort, after_prefix_steps=b.after_prefix_steps,
            replication=b.replication, candidate_index=b.candidate) for b in branch_plan(cfg, counts)]), "dataset_sha256")


class MetadataTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.packet = synthetic_packet(self, self.root)

    def test_prepare_preserves_original_fields_and_intentional_seed_reuse(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.object(execution, "recovery_manifest", return_value=self.packet["recovery_manifest"]), \
                patch.object(execution, "runtime_record", side_effect=AssertionError("no runtime")):
            prepared = execution.prepare(self.root)
        for field in execution.INHERITED_FIELDS:
            self.assertEqual(prepared[field], self.packet[field])
        self.assertEqual(prepared["new_environment_calls"], 0)
        self.assertEqual(prepared["new_checkpoint_loads"], 0)
        self.assertFalse(prepared["ready_to_launch"])
        self.assertFalse(prepared["seed_reuse"]["fresh_independent_replication_claimed"])
        self.assertNotIn("recovery", prepared)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_pinned_original_and_proposal_bytes_cannot_change(self):
        (self.root / execution.ORIGINAL_PACKET).write_text("{}")
        with patch.object(execution, "recovery_manifest") as reader, self.assertRaisesRegex(ValueError, "pinned"):
            execution.prepare(self.root)
        reader.assert_not_called()

    def test_scope_specific_intent_and_separate_authorization(self):
        self.assertTrue(execution.authorization(self.packet)["approved"])
        for changes in (dict(format="paired-cohort-approval-intent-v1"), dict(approved=False),
                dict(user_literal=""), dict(protocol_sha256="0" * 64), dict(original_attempt_remains_terminal=False)):
            packet = copy.deepcopy(self.packet)
            packet["intent"].update(changes)
            with self.assertRaises(PermissionError):
                execution.authorization(reseal(packet))

    def test_original_source_and_stream_bindings_cannot_change(self):
        packet = copy.deepcopy(self.packet)
        packet["original_source_files"]["src/original.py"] = dict(path="src/original.py", bytes=1, sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "original locked"):
            execution.authorization(reseal(packet))
        packet = copy.deepcopy(self.packet)
        packet["streams"]["bootstrap"] = "1"
        with self.assertRaises(PermissionError):
            execution.authorization(reseal(packet))

    def test_remaining_schedule_seed_charge_and_completed_overlap_rejected(self):
        for mutate in (lambda m: m["remaining"][0].update(future_seed="1"),
                       lambda m: m.update(preserved_interrupted_charge=0),
                       lambda m: m["completed"][0].update(branch_id=m["remaining"][0]["branch_id"])):
            packet = copy.deepcopy(self.packet)
            mutate(packet["recovery_manifest"])
            reseal(packet["recovery_manifest"], "manifest_sha256")
            with self.assertRaises(ValueError):
                execution.authorization(reseal(packet))

    def test_unapproved_launch_child_and_freeze_stop_before_science(self):
        with patch.object(execution, "approved", side_effect=PermissionError("unapproved")), \
                patch.object(execution, "supervise_paired") as supervisor, patch.object(execution, "PairedBudget") as budget:
            for method in (execution.launch, execution.child):
                with self.assertRaises(PermissionError):
                    method(self.root)
            supervisor.assert_not_called()
            budget.assert_not_called()
        with patch.object(execution, "_clean", side_effect=PermissionError("uncommitted")), \
                patch.object(execution, "prepare") as prepare:
            with self.assertRaises(PermissionError):
                execution.freeze(self.root)
            prepare.assert_not_called()
        self.assertFalse((self.root / execution.RUN).exists())

    def test_authorize_is_committed_packet_generator_only(self):
        with patch.object(execution, "_clean"), patch.object(execution, "committed_json", return_value=self.packet) as read, \
                patch.object(execution, "verify_bindings") as verify:
            self.assertEqual(execution.authorize(self.root), execution.authorization(self.packet))
            read.assert_called_once_with(self.root, execution.FROZEN)
            verify.assert_called_once_with(self.root, self.packet)
        self.assertFalse((self.root / execution.AUTHORIZATION).exists())

    def test_cli_defaults_to_preparation(self):
        from experiments.scripts.run_paired_cohort_recovery import main
        with patch.object(execution, "prepare", return_value=dict(ready_to_launch=False)) as prepare, \
                patch.object(execution, "freeze", side_effect=AssertionError("no freeze")), \
                patch.object(execution, "launch", side_effect=AssertionError("no launch")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main([]), 0)
            self.assertFalse(json.loads(output.getvalue())["ready_to_launch"])
            prepare.assert_called_once()


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.packet = synthetic_packet(self, self.root)
        self.auth, self.now = execution.authorization(self.packet), 10.
        for target in ("src.rl.dynamic_candidate_resources.shared_monotonic", "src.rl.paired_cohort_execution.shared_monotonic"):
            mocked = patch(target, side_effect=lambda: self.now)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.output = self.root / execution.RUN
        self.claim = dict(pid=os.getppid(), packet_sha256=self.packet["packet_sha256"], authorization_sha256=digest(self.auth),
                          started=self.now, clock_id=execution.CLOCK_ID, head="fake")
        save(self.output, "launcher/claim.json", self.claim)
        self.budget = execution.PairedBudget(self.output / "launcher/budget.jsonl", self.packet["budget_plan"], enabled=True, started=self.now)
        self.addCleanup(self.budget.close)
        self.admit = execution.RecoveryAdmission(self.root, self.packet, self.auth, self.claim, self.budget)
        self.addCleanup(self.admit.close)

    def test_admit_before_begin_and_exact_native_caps(self):
        self.admit("binding")
        self.assertIsNone(self.budget.active)
        self.assertEqual(execution.read_admission(self.admit.path, self.packet)["jobs"], ["binding"])
        self.budget.begin("binding")
        for op, cap in self.admit.limits["binding"].items():
            for _ in range(cap):
                self.admit(op)
        self.assertEqual(self.admit.counts["branch_import"], 117)
        self.assertEqual(self.admit.counts["context_load"], 36)
        self.assertEqual(self.admit.caps["branch_step"], 12047)
        self.assertEqual(self.budget.counts, dict(environment=0, optimizer=0))

    def test_no_preflight_or_actor_calls_during_binding_and_no_retry(self):
        self.admit("binding")
        self.budget.begin("binding")
        self.admit("context_load")
        with self.assertRaises(PermissionError):
            self.admit("preflight_clone")
        with self.assertRaises(PermissionError):
            self.admit("context_load")
        self.assertEqual(self.admit.counts["context_load"], 1)
        with self.assertRaises(ValueError):
            execution.read_admission(self.admit.path, self.packet)

    def test_existing_admission_claim_and_terminal_cannot_be_adopted(self):
        with self.assertRaises(FileExistsError):
            execution.RecoveryAdmission(self.root, self.packet, self.auth, self.claim, self.budget)
        save(self.output, "launcher/terminal.json", dict(status="failed"))
        with self.assertRaises(PermissionError):
            self.admit("binding")

    def test_hash_tamper_and_serial_skip_rejected(self):
        self.admit("binding")
        raw = self.admit.path.read_bytes()
        self.admit.path.write_bytes(raw.replace(b'"before_operation": true', b'"before_operation": false'))
        with self.assertRaises(ValueError):
            execution.read_admission(self.admit.path, self.packet)
        with self.assertRaises(PermissionError):
            self.admit("paired_branches/block61")

    def test_binding_setup_is_not_refunded(self):
        self.now += 301
        with self.assertRaises(PermissionError):
            self.admit("binding")


class CompletionTests(unittest.TestCase):
    setUp = MetadataTests.setUp

    def test_exact_full_matrix_and_new_only_owner_totals(self):
        expected = execution.expected_completion(self.packet, labels_fixture(self.packet))
        self.assertEqual(expected["environment_calls"], 25655)
        self.assertEqual(expected["cumulative_environment_calls"], 31901)
        self.assertEqual(expected["owners"]["paired_branches/block60"]["clone"], 297)
        self.assertEqual(expected["conditional_branches"], 402)
        self.assertEqual(expected["new_branches"], 285)

    def completion_files(self):
        output = self.root / execution.RUN
        save(output, "payload/labels.json", labels_fixture(self.packet))
        seals = {}
        for b in self.packet["config"]["blocks"]:
            for role in self.packet["config"]["evaluation_controllers"][:4]:
                name = f"block{b}/{role}"
                seals[name] = save(output, f"payload/sealed/{name}.pt", dict(invented=name))
            for arm in self.packet["config"]["training_arms"]:
                save(output, f"payload/models/block{b}/{arm}/final.pt", dict(invented=arm))
        save(output, "payload/model-seals.json", seals)
        save(output, "payload/evaluation-index.json", [dict(invented=i) for i in range(216)])
        expected = execution.expected_completion(self.packet, labels_fixture(self.packet))
        jobs = list(expected["operations"])
        admitted = jobs[:jobs.index("raw_verification") + 1]
        counts = {job: 1 for job in admitted}
        for job in admitted:
            for op, count in expected["operations"][job].items():
                counts[op] = counts.get(op, 0) + count
        operations = dict(jobs=admitted, by_section={job: caps for job, caps in expected["operations"].items() if caps},
            counts=counts, sha256="a" * 64)
        ledger = dict(plan_sha256=digest(self.packet["budget_plan"]), active="raw_verification", closed=admitted[:-1],
            counts=dict(environment=25655, optimizer=768), phase_seconds={},
            owner_counts={f"section/{job}:{key}": value for job, row in expected["owners"].items() for key, value in row.items()})
        return output, ledger, operations

    def test_raw_boundary_needs_only_admitted_jobs_and_exact_owner_debits(self):
        output, ledger, operations = self.completion_files()
        with patch.object(execution, "read_dynamic_ledger", return_value=ledger), patch.object(execution, "read_admission", return_value=operations):
            result = execution.verify_completion(output, self.packet, require_closed=False)
            self.assertEqual(result["old_partial_debit_preserved"], 1)
            ledger["owner_counts"]["section/paired_branches/block60:clone"] -= 1
            ledger["owner_counts"]["section/paired_branches/block61:clone"] += 1
            with self.assertRaisesRegex(ValueError, "per-owner"):
                execution.verify_completion(output, self.packet, require_closed=False)


class LauncherTests(unittest.TestCase):
    setUp = MetadataTests.setUp

    def run_fake(self, *, fail_archive=False, late_terminal=False):
        now = [10.]
        output = self.root / execution.RUN
        def supervise(command, **kwargs):
            self.assertIn("experiments.scripts.run_paired_cohort_recovery", command)
            for name in ("stdout.log", "stderr.log"):
                (kwargs["launcher"] / name).write_bytes(b"")
            return dict(passed=True, final_deadline=100.)
        def archive(source, target):
            self.assertFalse((source / "terminal.json").exists())
            if fail_archive:
                raise OSError("invented archive failure")
            return dict(invented=True)
        writer = execution.write_json_once
        def write(path, value):
            writer(path, value)
            if late_terminal and Path(path).name == "terminal.json" and value["status"] == "completed":
                now[0] = 101.
        with ExitStack() as stack:
            for name, kwargs in (("approved", dict(return_value=(self.packet, execution.authorization(self.packet)))),
                    ("git", dict(return_value="f" * 40)), ("shared_monotonic", dict(side_effect=lambda: now[0])),
                    ("supervise_paired", dict(side_effect=supervise)), ("verify_completion", dict(return_value={})),
                    ("create_archive", dict(side_effect=archive)), ("write_json_once", dict(side_effect=write))):
                stack.enter_context(patch.object(execution, name, **kwargs))
            stack.enter_context(patch.object(execution.subprocess, "Popen", side_effect=AssertionError("no process")))
            code = execution.launch(self.root, started=10.)
        return code, execution.read_terminal(output)

    def test_completion_only_after_archive(self):
        code, terminal = self.run_fake()
        self.assertEqual(code, 0)
        self.assertTrue(terminal["scientific_completion_verified"])

    def test_archive_failure_is_authoritative(self):
        code, terminal = self.run_fake(fail_archive=True)
        self.assertEqual(code, 1)
        self.assertEqual(terminal["status"], "failed")
        self.assertFalse(terminal["scientific_completion_verified"])

    def test_terminal_deadline_failure_overrides_completed_receipt(self):
        code, terminal = self.run_fake(late_terminal=True)
        self.assertEqual(code, 1)
        self.assertEqual(terminal["authoritative_source"], "launcher/launch-failure.json")
        self.assertFalse(terminal["scientific_completion_verified"])


if __name__ == "__main__":
    unittest.main()
