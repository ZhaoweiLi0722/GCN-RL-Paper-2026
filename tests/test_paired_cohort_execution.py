"""Artificial metadata, receipt and clock tests. No freeze or scientific work."""

import copy
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.rl import paired_cohort_execution as execution
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import read_dynamic_ledger
from src.rl.paired_cohort_plan import branch_plan, context_keys
from tests.test_paired_cohort_resources import config

file_record = execution.file_record


def save(root, name, value):
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return file_record(root, name)


def reseal(value, key="packet_sha256"):
    value.pop(key, None)
    value[key] = digest(value)
    return value


def history(root):
    cfg = config()
    save(root, execution.PROPOSAL, cfg)
    reference = dict(directory="results/fake-reference", config_template="block{block}/config.json",
        policy_template="block{block}/model.pt", locks={str(b): dict(config="c" * 64, policy="d" * 64) for b in cfg["blocks"]})
    inputs = {reference["directory"] + "/" + reference[kind + "_template"].format(block=b):
        dict(path=reference["directory"] + "/" + reference[kind + "_template"].format(block=b),
             bytes=7, sha256=reference["locks"][str(b)][kind]) for b in cfg["blocks"] for kind in ("config", "policy")}
    inputs["results/unrelated/outcome.json"] = dict(path="results/unrelated/outcome.json", bytes=1, sha256="0" * 64)
    old = dict(blocks=cfg["blocks"], candidate_message_graph="specimen_routes",
        reference=reference, model_proposal={"fixture": "graph"},
        objective={"fixture": "old objective"}, candidate_support={"max_original_requests": 6},
        cohort_proposal=dict(enrollment_steps=52, economic_endpoint=63, accounting_steps=11, patient_resolution_steps=8))
    def record(name):
        return dict(path="results/fake-history/" + name + ".pt", bytes=7, sha256="a" * 64)
    original = reseal(dict(scientific_config=old, original_config={"original": "initializer recipe"},
        initializer_streams={"original": "initializer streams"}, qualified_blocks={str(b): dict(passed=True,
            kernel_sha256="b" * 64) for b in cfg["blocks"]},
        initializers={str(b): record(f"initializer{b}") for b in cfg["blocks"]}, input_files=inputs,
        streams={"environment": {str(b): {"layout": [str(b)]} for b in cfg["blocks"]}}))
    evaluation = reseal(dict(scientific_config=copy.deepcopy(old),
        models={f"block{b}/graph/cohort_ppo": record(f"ppo{b}") for b in cfg["blocks"]}))
    return {name: save(root, name, data)["sha256"] for name, data in
            ((execution.INITIALIZER_PACKET, original), (execution.EVALUATION_PACKET, evaluation))}


def synthetic_packet(root):
    # Construct an invented authorization fixture directly. Do NOT call freeze.
    pins = history(root)
    with patch.object(execution, "HISTORICAL_JSONS", pins):
        packet = execution.prepare(root)
    protocol = save(root, execution.PROTOCOL, {"synthetic": True})
    proposal = file_record(root, execution.PROPOSAL)
    intent = dict(approved=True, user_literal="SYNTHETIC TEST ONLY", proposal_sha256=proposal["sha256"],
                  protocol_sha256=protocol["sha256"])
    packet.update(format="paired-cohort-frozen-v1", source_frozen=True, implementation_commit="f" * 40,
        intent=intent, intent_file=save(root, execution.INTENT, intent), protocol=protocol, proposal=proposal,
        source_files={}, runtime={"fake": "no native runtime"},
        local_seed_collision_audit=dict(passed=True, collisions=[], files=[],
                                       manifest_sha256=digest(packet["streams"]["allocations"])))
    return reseal(packet)


def labels_fixture(cfg, classes=2):
    counts = {key: classes for key in context_keys(cfg)}
    labels = [dict(block=b, cohort=c, after_prefix_steps=t, class_keys=[[k] for k in range(classes)],
                   public_example={"candidates": {"class_keys": [[k] for k in range(classes)]}})
              for b, c, t in counts]
    outcomes = [dict(block=b.block, cohort=b.cohort, after_prefix_steps=b.after_prefix_steps,
                     replication=b.replication, candidate_index=b.candidate) for b in branch_plan(cfg, counts)]
    return reseal(dict(format="paired-cohort-label-dataset-v1", config_sha256=digest(cfg),
                       labels=labels, branch_outcomes=outcomes), "dataset_sha256")


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.pins = history(self.root)
        pin = patch.object(execution, "HISTORICAL_JSONS", self.pins)
        pin.start()
        self.addCleanup(pin.stop)

    def test_preparation_reads_only_pinned_metadata_and_leaves_draft_false(self):
        before = {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        campaign_imported = "src.rl.paired_cohort_campaign" in sys.modules
        with patch.object(execution, "runtime_record", side_effect=AssertionError("no runtime during preparation")), \
                patch.object(execution, "seed_inventory", side_effect=AssertionError("no seed audit before approval")):
            result = execution.prepare(self.root)
        self.assertEqual(set(result["historical_jsons"]), set(self.pins))
        self.assertEqual(result["initializer_config"], {"original": "initializer recipe"})
        self.assertEqual(result["initializer_streams"], {"original": "initializer streams"})
        self.assertEqual(len(result["inherited"]["model_inputs"]), 6)
        self.assertEqual(len(result["input_files"]), 14)
        self.assertNotIn("results/unrelated/outcome.json", result["input_files"])
        self.assertEqual(result["streams"]["format"], "paired-cohort-streams-v1")
        self.assertIs(type(result["streams"]["bootstrap"]), str)
        self.assertFalse(result["config"]["scientific_execution_authorized"])
        self.assertFalse(result["ready_to_launch"])
        self.assertFalse(result["source_frozen"])
        self.assertEqual(result["new_checkpoint_loads"], 0)
        self.assertEqual(before, {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        self.assertEqual("src.rl.paired_cohort_campaign" in sys.modules, campaign_imported)

    def test_changed_pinned_metadata_and_internal_digest_rejected(self):
        path = self.root / execution.INITIALIZER_PACKET
        original = path.read_text()
        path.write_text(original + " ")
        with self.assertRaisesRegex(ValueError, "pinned historical"):
            execution.prepare(self.root)
        changed = json.loads(original)
        changed["original_config"]["new"] = True
        self.pins[execution.INITIALIZER_PACKET] = save(self.root, execution.INITIALIZER_PACKET, changed)["sha256"]
        with self.assertRaisesRegex(ValueError, "packet content digest"):
            execution.prepare(self.root)

    def test_scope_cannot_expand_budget_or_enable_draft(self):
        cfg = config()
        for field, value in (("scientific_execution_authorized", True), ("attempts", 2), ("automatic_retry", True)):
            changed = copy.deepcopy(cfg)
            changed[field] = value
            with self.assertRaises(ValueError):
                execution._scope(changed)
        changed = copy.deepcopy(cfg)
        changed["per_owner_seconds"]["evaluation_controller_block"] = 241
        with self.assertRaises(ValueError):
            execution._scope(changed)

    def test_seed_inventory_includes_prior_frozen_and_only_excludes_own_attempt(self):
        prior = "specs/older/frozen.json"
        new_spec = execution.DIRECTORY + "/future-streams.json"
        new_run = execution.RUN + "/config.json"
        nested = "specs/older/frozen-proposal/proposal.json"
        for name in (prior, nested, new_spec, new_run, "results/previous/seed-manifest.json"):
            save(self.root, name, {"seed": "12345678901234567890"})
        with patch.object(execution, "git", return_value=execution.PROPOSAL):
            names = {p.relative_to(self.root).as_posix() for p in execution.seed_inventory(self.root)}
        self.assertIn(prior, names)
        self.assertIn(nested, names)
        self.assertIn(execution.INITIALIZER_PACKET, names)
        self.assertIn(execution.EVALUATION_PACKET, names)
        self.assertIn(execution.PROPOSAL, names)
        self.assertNotIn(new_spec, names)
        self.assertNotIn(new_run, names)
        audit = execution.audit_stream_collisions({"seed": "12345678901234567890"}, [self.root / prior])
        self.assertFalse(audit["passed"])

    def test_authorization_requires_true_literal_and_exact_new_scope(self):
        packet = synthetic_packet(self.root)
        auth = execution.authorization(packet)
        self.assertTrue(auth["approved"])
        self.assertEqual(auth["packet_sha256"], packet["packet_sha256"])
        self.assertFalse(auth["automatic_retry"])
        for changes in (dict(approved=False), dict(user_literal=""), dict(proposal_sha256="0" * 64),
                        dict(protocol_sha256="0" * 64)):
            altered = copy.deepcopy(packet)
            altered["intent"].update(changes)
            with self.assertRaises(PermissionError):
                execution.authorization(reseal(altered))
        with self.assertRaises((KeyError, ValueError)):
            execution.authorization(execution.prepare(self.root))

    def test_native_launch_and_child_stop_at_unapproved_boundary(self):
        with patch.object(execution, "approved", side_effect=PermissionError("not approved")), \
                patch.object(execution, "supervise_paired") as supervisor, patch.object(execution, "PairedBudget") as budget:
            for call in (execution.launch, execution.child):
                with self.assertRaisesRegex(PermissionError, "not approved"):
                    call(self.root)
            supervisor.assert_not_called()
            budget.assert_not_called()
        self.assertFalse((self.root / execution.RUN).exists())

    def test_cli_default_is_read_only_preparation(self):
        from experiments.scripts.run_paired_cohort_improvement import main
        with patch.object(execution, "prepare", return_value={"ready_to_launch": False}) as prepare, \
                patch.object(execution, "launch", side_effect=AssertionError("no launch")), \
                patch.object(execution, "freeze", side_effect=AssertionError("no freeze")), \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(main([]), 0)
            self.assertFalse(json.loads(stdout.getvalue())["ready_to_launch"])
            prepare.assert_called_once()


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.packet = synthetic_packet(self.root)
        self.auth = execution.authorization(self.packet)
        self.output = self.root / execution.RUN
        self.now = 10.
        for target in ("src.rl.dynamic_candidate_resources.shared_monotonic", "src.rl.paired_cohort_execution.shared_monotonic"):
            mocked = patch(target, side_effect=lambda: self.now)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.claim = dict(pid=os.getppid(), packet_sha256=self.packet["packet_sha256"], authorization_sha256=digest(self.auth),
                          started=self.now, clock_id=execution.CLOCK_ID, head="fake")
        save(self.output, "launcher/claim.json", self.claim)
        self.budget = execution.PairedBudget(self.output / "launcher/budget.jsonl", self.packet["budget_plan"],
                                            enabled=True, started=self.now)
        self.addCleanup(self.budget.close)
        self.admit = execution.PairedAdmission(self.root, self.packet, self.auth, self.claim, self.budget)
        self.addCleanup(self.admit.close)

    def start(self, job="binding"):
        self.admit(job)
        self.budget.begin(job)

    def test_jobs_are_prospective_and_binding_operations_are_all_counted(self):
        self.admit("binding")
        self.assertIsNone(self.budget.active)
        receipts = execution.read_admission(self.admit.path, self.packet)
        self.assertEqual(receipts["jobs"], ["binding"])
        self.budget.begin("binding")
        for name, cap in execution.operation_limits(self.packet["config"])["binding"].items():
            for _ in range(cap):
                self.admit(name)
        receipts = execution.read_admission(self.admit.path, self.packet)
        self.assertEqual(receipts["counts"], self.admit.counts)
        self.assertEqual(self.admit.counts["input_hash_verification"], 12)
        self.assertEqual(self.admit.counts["checkpoint_load"], 6)
        self.assertEqual(self.budget.counts, dict(environment=0, optimizer=0))
        self.budget.finish()
        self.start("same_start_preflight/block60")
        for name in ("preflight_clone", "episode_dispatch", "episode_build", "episode_step", "clone_step"):
            self.admit(name)
        self.assertEqual(self.admit.by_section["same_start_preflight/block60"]["clone_step"], 1)

    def test_rejected_operation_closes_attempt_and_spent_receipts_remain(self):
        self.start()
        self.admit("checkpoint_load")
        with self.assertRaisesRegex(PermissionError, "out-of-phase"):
            self.admit("actor_update")
        self.assertEqual(self.admit.counts["checkpoint_load"], 1)
        with self.assertRaisesRegex(PermissionError, "terminal"):
            self.admit("checkpoint_load")
        with self.assertRaisesRegex(ValueError, "failed"):
            execution.read_admission(self.admit.path, self.packet)

    def test_serial_skip_repeat_and_budget_before_job_are_rejected(self):
        with self.assertRaisesRegex(PermissionError, "serial job"):
            self.admit("same_start_preflight/block60")

    def test_native_operation_caps_and_all_phase_owners(self):
        self.assertEqual(self.admit.caps, dict(input_hash_verification=12, reference_and_layout_build=3,
            reference_checkpoint_load=3, layout_environment_build=3, checkpoint_load=6,
            episode_build=231, episode_dispatch=231, episode_step=14553, preflight_clone=3, clone_step=189,
            conditional_branch_clone=432, branch_step=18576, actor_fork=6, actor_update=768))
        for job, caps in self.admit.limits.items():
            self.start(job)
            for operation in caps:
                self.admit(operation)
            self.budget.finish()
        self.assertEqual(execution.read_admission(self.admit.path, self.packet)["jobs"], self.admit.jobs)
        self.assertEqual(self.budget.counts, dict(environment=0, optimizer=0))

    def test_repeated_job_and_work_before_budget_begin_are_rejected(self):
        self.admit("binding")
        with self.assertRaisesRegex(PermissionError, "out-of-phase"):
            self.admit("checkpoint_load")

    def test_cap_is_per_owner_and_no_extra_historical_loads(self):
        self.start()
        for _ in range(3):
            self.admit("reference_checkpoint_load")
        with self.assertRaisesRegex(PermissionError, "exhausted"):
            self.admit("reference_checkpoint_load")
        self.assertEqual(self.admit.counts["reference_checkpoint_load"], 3)

    def test_fresh_operation_ledger_cannot_be_adopted(self):
        with self.assertRaises(FileExistsError):
            execution.PairedAdmission(self.root, self.packet, self.auth, self.claim, self.budget)

    def test_parent_claim_and_terminal_boundaries_remain_live(self):
        self.start()
        save(self.output, "launcher/terminal.json", {"status": "failed"})
        with self.assertRaisesRegex(PermissionError, "terminal"):
            self.admit("checkpoint_load")

    def test_modified_claim_and_authorization_rejected(self):
        altered = dict(self.auth, automatic_retry=True)
        with self.assertRaises(PermissionError):
            execution.PairedAdmission(self.root, self.packet, altered, self.claim, self.budget)
        save(self.output, "launcher/claim.json", self.claim | dict(pid=1))
        with self.assertRaisesRegex(PermissionError, "owner changed"):
            self.admit("binding")

    def test_operation_chain_tamper_is_rejected(self):
        self.start()
        self.admit("checkpoint_load")
        raw = self.admit.path.read_bytes()
        self.admit.path.write_bytes(raw.replace(b'"count": 1', b'"count": 2'))
        with self.assertRaisesRegex(ValueError, "hash chain"):
            execution.read_admission(self.admit.path, self.packet)

    def test_binding_setup_time_cannot_be_refunded(self):
        self.now += 301
        with self.assertRaisesRegex(PermissionError, "setup cap"):
            self.admit("binding")


class CompletionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.packet = synthetic_packet(self.root)
        self.cfg = self.packet["config"]
        self.labels = labels_fixture(self.cfg)
        self.expected = execution.expected_completion(self.cfg, self.labels)

    def test_deduplicated_actual_branch_calls_not_maximum_are_exact(self):
        expected = self.expected
        self.assertEqual(expected["conditional_branches"], 144)
        self.assertEqual(expected["branch_steps"], 6192)
        self.assertEqual(expected["environment_calls"], 14742 + 6192)
        self.assertEqual(expected["operations"]["paired_branches/block60"], dict(conditional_branch_clone=48, branch_step=2064))
        self.assertEqual(expected["optimizer_calls"], 768)
        self.assertEqual(expected["owners"]["paired_actor/block60"]["actor"], 128)
        maximum = execution.expected_completion(self.cfg, labels_fixture(self.cfg, classes=6))
        self.assertEqual(maximum["conditional_branches"], 432)
        self.assertEqual(maximum["environment_calls"], 33318)

    def test_missing_duplicate_context_or_branch_is_not_unused_capacity(self):
        for mutate in (lambda x: x["labels"].pop(), lambda x: x["labels"].append(x["labels"][0]),
                       lambda x: x["branch_outcomes"].pop(), lambda x: x["branch_outcomes"].append(x["branch_outcomes"][0])):
            labels = copy.deepcopy(self.labels)
            mutate(labels)
            with self.assertRaises(ValueError):
                execution.expected_completion(self.cfg, reseal(labels, "dataset_sha256"))

    def fake_completed(self):
        output = self.root / execution.RUN
        save(output, "payload/labels.json", self.labels)
        save(output, "payload/evaluation-index.json", [{"artificial": i} for i in range(216)])
        seals = {}
        for b in self.cfg["blocks"]:
            for role in self.cfg["evaluation_controllers"][:4]:
                name = f"block{b}/{role}"
                # Artificial bytes are hashed only; never decoded as models.
                seals[name] = save(output, f"payload/sealed/{name}.pt", {"fake": name})
            for arm in self.cfg["training_arms"]:
                save(output, f"payload/models/block{b}/{arm}/final.pt", {"fake": arm})
        save(output, "payload/model-seals.json", seals)
        counts = dict(environment=self.expected["environment_calls"], optimizer=768)
        jobs = list(self.expected["operations"])
        by_section = {k: v for k, v in self.expected["operations"].items() if v}
        operations = dict(counts={k: 1 for k in jobs}, jobs=jobs, by_section=by_section, sha256="a" * 64)
        for caps in by_section.values():
            for k, v in caps.items():
                operations["counts"][k] = operations["counts"].get(k, 0) + v
        ledger = dict(plan_sha256=digest(self.packet["budget_plan"]), counts=counts, active=None, closed=jobs,
            owner_counts={f"section/{k}:{o}": n for k, owners in self.expected["owners"].items() for o, n in owners.items()},
            phase_seconds={})
        save(output, "launcher/closure.json", {k: self.expected[k] for k in
            ("evaluation_cohorts", "conditional_branches", "contexts", "new_fits")} | dict(
                complete=True, engineering_fixture=False, counts=counts, operations=operations["counts"]))
        return output, ledger, operations

    def test_final_verification_checks_exact_per_owner_not_only_global_total(self):
        output, ledger, operations = self.fake_completed()
        with patch.object(execution, "read_dynamic_ledger", return_value=ledger), \
                patch.object(execution, "read_admission", return_value=operations):
            result = execution.verify_completion(output, self.packet)
            self.assertEqual(result["conditional_branches"], 144)
            self.assertFalse(result["unused_branch_capacity_reallocated"])
            ledger["owner_counts"]["section/paired_actor/block60:actor"] -= 1
            ledger["owner_counts"]["section/bc_actor/block60:actor"] += 1
            with self.assertRaisesRegex(ValueError, "per-owner exact"):
                execution.verify_completion(output, self.packet)

    def test_final_verification_rejects_missing_seal_or_fit(self):
        output, ledger, operations = self.fake_completed()
        (output / "payload/models/block60/paired_cost/final.pt").unlink()
        with patch.object(execution, "read_dynamic_ledger", return_value=ledger), \
                patch.object(execution, "read_admission", return_value=operations), self.assertRaises((ValueError, FileNotFoundError)):
            execution.verify_completion(output, self.packet)


class DeadlineTests(unittest.TestCase):
    def test_setup_aggregate_phase_and_closure_deadline_survive_finished_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            now = [10.]
            plan = execution.budget_plan(config())
            path = Path(temp) / "budget.jsonl"
            with patch("src.rl.dynamic_candidate_resources.shared_monotonic", side_effect=lambda: now[0]):
                budget = execution.PairedBudget(path, plan, enabled=True, started=10.)
                try:
                    watcher = execution.PairedLedgerDeadline(path, plan, 10.)
                    self.assertEqual(watcher.deadline(99999), 310.)
                    now[0] = 20.
                    budget.begin("binding")
                    now[0] = 30.
                    budget.finish()
                    for job in list(plan["sections"])[1:]:
                        budget.begin(job)
                        now[0] += 1
                        budget.finish()
                    watcher.poll()
                    self.assertIsNone(watcher.initial_deadline)
                    self.assertEqual(watcher.spent["binding"], 20.)
                    self.assertEqual(watcher.spent["evaluation"], 18.)
                    self.assertEqual(watcher.closure_deadline, now[0] - 1 + 600)
                    self.assertEqual(watcher.deadline(99999), watcher.closure_deadline)
                    self.assertEqual(read_dynamic_ledger(path)["closed"], list(plan["sections"]))
                finally:
                    budget.close()

    def test_supervisor_refuses_existing_ledger_without_spawning_anything(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "budget.jsonl").write_text("old evidence")
            with patch.object(execution.subprocess, "Popen", side_effect=AssertionError("no process")), \
                    self.assertRaises(FileExistsError):
                execution.supervise_paired(["fake-child"], cwd=root, launcher=root,
                    plan=execution.budget_plan(config()), started=10.)


class LauncherTests(unittest.TestCase):
    """Fake supervisor/archive only; no native launch, subprocess or science."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.packet = synthetic_packet(self.root)
        self.output = self.root / execution.RUN
        self.now = 10.

    def run_fake(self, archive, writer=None):
        def supervise(command, **kwargs):
            self.assertEqual(kwargs["plan"], self.packet["budget_plan"])
            for name in ("stdout.log", "stderr.log"):
                (kwargs["launcher"] / name).write_bytes(b"")
            return dict(passed=True, final_deadline=100.)
        with ExitStack() as stack:
            for target, kwargs in (
                ("approved", dict(return_value=(self.packet, execution.authorization(self.packet)))),
                ("git", dict(return_value="f" * 40)), ("shared_monotonic", dict(side_effect=lambda: self.now)),
                ("supervise_paired", dict(side_effect=supervise)), ("verify_completion", dict(return_value={})),
                ("create_archive", dict(side_effect=archive))):
                stack.enter_context(patch.object(execution, target, **kwargs))
            stack.enter_context(patch.object(execution.subprocess, "Popen", side_effect=AssertionError("no subprocess")))
            if writer is not None:
                stack.enter_context(patch.object(execution, "write_json_once", side_effect=writer))
            return execution.launch(self.root, started=10.)

    def archive(self, source, target):
        self.assertFalse((source / "terminal.json").exists())
        self.assertEqual(json.loads((source / "archive-pending.json").read_text())["status"], "pending_archive")
        return dict(files={"artificial": "f" * 64}, archive_sha256="e" * 64)

    def test_completed_terminal_only_after_verified_archive_and_receipt(self):
        self.assertEqual(self.run_fake(self.archive), 0)
        terminal = execution.read_terminal(self.output)
        self.assertEqual(terminal["status"], "completed")
        self.assertTrue(terminal["scientific_completion_verified"])
        self.assertEqual(terminal["closure_archive_receipt"], file_record(self.output, "closure-archive-receipt.json"))

    def test_archive_failure_never_leaves_a_completed_terminal(self):
        def failed(source, target):
            self.archive(source, target)
            raise OSError("synthetic archive failure")
        self.assertEqual(self.run_fake(failed), 1)
        self.assertEqual(json.loads((self.output / "launcher/terminal.json").read_text())["status"], "failed")
        terminal = execution.read_terminal(self.output)
        self.assertFalse(terminal["scientific_completion_verified"])
        self.assertTrue(terminal["authoritative"])

    def test_archive_timeout_never_leaves_a_completed_terminal(self):
        def slow(source, target):
            receipt = self.archive(source, target)
            self.now = 101.
            return receipt
        self.assertEqual(self.run_fake(slow), 1)
        self.assertEqual(execution.read_terminal(self.output)["status"], "failed")
        self.assertFalse((self.output / "closure-archive-receipt.json").exists())

    def test_late_terminal_fsync_failure_override_is_authoritative(self):
        original = execution.write_json_once
        def late(path, value):
            original(path, value)
            if Path(path).name == "terminal.json" and value["status"] == "completed":
                self.now = 101.
        self.assertEqual(self.run_fake(self.archive, writer=late), 1)
        terminal = execution.read_terminal(self.output)
        self.assertEqual(terminal["status"], "failed")
        self.assertFalse(terminal["scientific_completion_verified"])
        self.assertEqual(terminal["authoritative_source"], "launcher/launch-failure.json")


if __name__ == "__main__":
    unittest.main()
