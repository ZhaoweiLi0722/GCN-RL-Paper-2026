"""Separately approved one-attempt continuation; preparation is JSON/hash only."""

import json
import os
from pathlib import Path
import subprocess
import sys

from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_execution import configure_runtime, runtime_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_backend import DynamicPatientBackend
from src.rl.dynamic_candidate_execution import committed_json, confined, verify_completed
from src.rl.dynamic_candidate_preparation import BRANCH, git, source_locks
from src.rl.dynamic_candidate_recovery_plan import recovery_budget_plan, recovery_config, REMOVED
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.dynamic_candidate_saved_execution import load_envelope
from src.rl.dynamic_candidate_saved_qualification import restore_saved_policy
from src.rl.dynamic_candidate_watchdog import supervise_dynamic
from src.utils.research_archive import copy_verified, create_archive
from src.utils.research_clock import CLOCK_ID, shared_monotonic


DIRECTORY = "specs/2026-10-01-adaptive-paper-delivery"
PROPOSAL = DIRECTORY + "/continuation-recovery-proposal.json"
PROTOCOL = DIRECTORY + "/continuation-recovery-protocol.md"
FROZEN = DIRECTORY + "/continuation-recovery-frozen.json"
AUTHORIZATION = DIRECTORY + "/continuation-recovery-authorization.json"
OLD = "results/dynamic_candidate_pilot_20261001"
QUALIFICATION = "results/dynamic_candidate_saved_qualification_20261001"
RUN = "results/dynamic_candidate_continuation_recovery_20261001"
ORIGINAL = OLD + "/payload/locks/worktree/" + DIRECTORY + "/frozen-proposal/proposal.json"
QUALIFICATION_SHA = "f1b271a72b4bfa1d0b9d16deb3651af60f1d56aedbf4dc541ded0f133257b9b6"
TERMINAL_SHA = "5617d5fc053efcea92073f18f2a0d46e7b4efd1f4925bf455c770dcb3ce28906"


def input_bindings(root):
    """Reuse the passed receipt and audit unused phases, not the 39 old outcomes."""
    root = Path(root).resolve()
    qname, tname = QUALIFICATION + "/qualification.json", QUALIFICATION + "/terminal.json"
    files = {name: file_record(root, name) for name in (qname, tname, ORIGINAL)}
    if files[qname]["sha256"] != QUALIFICATION_SHA or files[tname]["sha256"] != TERMINAL_SHA:
        raise ValueError("exact completed saved qualification required")
    q, terminal, original = (json.loads(confined(root, name).read_text()) for name in (qname, tname, ORIGINAL))
    if (q["passed"] is not True or terminal["status"] != "completed" or terminal["exit_code"] != 0
            or q["source_unchanged"] is not True or set(q["blocks"]) != {"60", "61", "62"}
            or q["counts"] != {"checkpoint_load": 4, "scoring": 624}
            or files[ORIGINAL] != q["input_files_after"][ORIGINAL]):
        raise ValueError("completed qualification lineage differs")
    initializers = {}
    for b in original["scientific_config"]["blocks"]:
        name = OLD + f"/payload/initialization/block{b}/graph/final.pt"
        record = file_record(root, name)
        if record != q["input_files_after"][name] or record != q["input_files_before"][name]:
            raise ValueError("qualified initializer bytes changed")
        initializers[str(b)] = record
        files[name] = record
    for name, expected in original["static_inputs"]["inputs"].items():
        record = file_record(root, name)
        if record != expected:
            raise ValueError("locked reference/config input changed")
        files[name] = record
    ledger_name, failure_name = OLD + "/launcher/budget.jsonl", OLD + "/launcher/failure.json"
    ledger = read_dynamic_ledger(root / ledger_name)
    failure = json.loads((root / failure_name).read_text())
    if (failure["status"] != "failed" or failure["sequence"]["active"] != "qualification"
            or ledger["counts"] != {"environment": 2040, "optimizer": 768}
            or any(row["job"].split("/")[0] not in REMOVED | {"runtime_input_binding"}
                   for row in failure["sequence"]["completed"])):
        raise ValueError("original terminal attempt did not stop at qualification")
    untouched = ("same_start_preflight", "ppo_continuation", "bc_continuation", "final_evaluation")
    for phase in untouched:
        if any(ledger["owner_counts"].get(f"phase/{phase}:{owner}", 0)
               for owner in ("trajectory", "clone", "actor", "critic")):
            raise ValueError("continuation/test allocation was already consumed")
    headers = sorted((root / OLD / "payload/episodes").rglob("header.json"))
    if len(headers) != 39:
        raise ValueError("old attempt raw episode inventory changed")
    streams = original["streams"]["environment"]
    retained = {seed for block in streams.values() for role in ("fork_preflight", "training", "test")
                for seed in block[role]}
    for path in headers:
        row = json.loads(path.read_text())
        if row["seed"] in retained or row["split"] in ("training", "test"):
            raise ValueError("declared unused stream appears in original raw evidence")
    for name in (ledger_name, failure_name, *(p.relative_to(root).as_posix() for p in headers)):
        files[name] = file_record(root, name)
    audit = {"original_attempt_terminal": True, "old_counts": ledger["counts"],
        "unconsumed_phases": list(untouched), "old_raw_headers": len(headers),
        "retained_world_allocations": len(retained), "passed": True,
        "scope": "reuse original allocation and freshness receipt; actual old ledger and headers show no continuation/test use; unavailable external runs not covered"}
    return files, original, q, initializers, audit


def validate_proposal(proposal, original):
    config, plan = recovery_config(original["scientific_config"]), recovery_budget_plan(original["scientific_config"])
    if (proposal.get("schema") != "dynamic-candidate-continuation-recovery-proposal-v1"
            or proposal.get("approved") is not False or proposal.get("scientific_execution_authorized") is not False
            or proposal.get("ready_to_launch") is not False or not proposal.get("approvals")
            or any(value is not False for value in proposal["approvals"].values())
            or proposal["result_root"] != RUN or proposal["scientific_config"] != original["scientific_config"]
            or proposal["inherited_stream_manifest"] != original["streams"] or proposal["budget_plan"] != plan
            or proposal["execution_budget"]["phase_budgets"] != config["phase_budgets"]
            or proposal["attempt"]["maximum_attempts"] != 1
            or proposal["attempt"]["automatic_retry"] is not False
            or proposal["attempt"]["old_s1_reopened"] is not False):
        raise ValueError("prospective recovery proposal differs from the exact retained design")
    totals = proposal["execution_budget"]["totals"]
    expected = {"full_episodes": 387, "trajectory_steps": 20124, "mandatory_clone_steps": 60,
        "environment_step_calls": 20184, "actor_optimizer_calls": 768, "critic_optimizer_calls": 384,
        "optimizer_step_calls": 1152, "phase_seconds": 17100, "global_elapsed_seconds": 17400,
        "environment_builds_max": 390, "historical_checkpoint_loads_max": 6,
        "qualification_rescorings": 0, "new_initialization_optimizer_calls": 0}
    if any(totals.get(k) != value for k, value in expected.items()):
        raise ValueError("complete prospective budget differs")


def freeze(root):
    root = Path(root).resolve()
    if git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain"):
        raise ValueError("freeze requires clean committed integration branch")
    proposal = committed_json(root, PROPOSAL)
    if proposal.get("scientific_execution_authorized") is not False or proposal.get("ready_to_launch") is not False:
        raise ValueError("prospective proposal must remain unapproved")
    files, original, q, initializers, audit = input_bindings(root)
    validate_proposal(proposal, original)
    for record in proposal["reuse"]["receipt_files"] + proposal["reuse"]["reference_input_files"]:
        if file_record(root, record["path"]) != record:
            raise ValueError("prospective reused receipt bytes changed")
        files[record["path"]] = record
    config = recovery_config(original["scientific_config"])
    packet = {"format": "dynamic-continuation-recovery-frozen-v1", "scientific_execution_authorized": False,
        "ready_to_launch": False, "workspace": str(root), "branch": BRANCH,
        "implementation_commit": git(root, "rev-parse", "HEAD"), "source_frozen": True,
        "proposal": file_record(root, PROPOSAL),
        "protocol": file_record(root, PROTOCOL), "original_config": original["scientific_config"],
        "scientific_config": config, "streams": original["streams"],
        "budget_plan": recovery_budget_plan(original["scientific_config"]),
        "qualified_blocks": q["blocks"], "initializers": initializers,
        "input_files": files, "source_files": source_locks(root), "runtime": runtime_record(),
        "stream_reuse_audit": audit, "new_checkpoint_loads": 0, "new_environment_calls": 0,
        "new_optimizer_calls": 0, "new_model_forwards": 0}
    packet["packet_sha256"] = digest(packet)
    return packet


def approved_limits(packet):
    plan, config = packet["budget_plan"], packet["scientific_config"]
    return {"attempts": 1, "environment_steps": plan["limits"]["trajectory"] + plan["limits"]["clone"],
        "optimizer_calls": plan["limits"]["actor"] + plan["limits"]["critic"],
        "seconds": plan["limits"]["seconds"], "historical_initializer_loads": 3,
        "reference_loads": 3, "layout_builds": 3, "episode_builds": config["totals"]["fresh_episode_builds"],
        "final_test_episodes": config["evaluation"]["total_episodes"]}


def validate_authorization(auth, packet):
    body = dict(packet)
    sha = body.pop("packet_sha256", None)
    if digest(body) != sha:
        raise ValueError("recovery frozen packet changed")
    if (packet["scientific_execution_authorized"] is not False or packet["ready_to_launch"] is not False
            or auth.get("format") != "dynamic-continuation-recovery-authorization-v1"
            or auth.get("approved") is not True or auth.get("packet_sha256") != sha
            or auth.get("implementation_commit") != packet["implementation_commit"]
            or auth.get("limits") != approved_limits(packet)
            or auth.get("automatic_retry") is not False or auth.get("remote_or_dropbox_actions") is not False
            or auth.get("reuse_completed_qualification_without_rescoring") is not True
            or auth.get("old_s1_remains_terminal") is not True):
        raise PermissionError("complete separate continuation and evaluation approval required")
    user = auth.get("user_approval", {})
    if (user.get("user") != "Zhaowei" or not isinstance(user.get("verbatim"), str)
            or not user["verbatim"].strip() or not user.get("recorded_at_utc")):
        raise PermissionError("actual scope-specific user approval required")


def verify_bindings(root, packet):
    subprocess.run(["git", "merge-base", "--is-ancestor", packet["implementation_commit"], "HEAD"], cwd=root, check=True)
    subprocess.run(["git", "diff", "--exit-code", packet["implementation_commit"], "--",
                    "src", "experiments/scripts", "tests", "AGENTS.md"], cwd=root, check=True, stdout=subprocess.DEVNULL)
    if source_locks(root) != packet["source_files"] or runtime_record() != packet["runtime"]:
        raise ValueError("frozen recovery source/runtime changed")
    for key, name in (("proposal", PROPOSAL), ("protocol", PROTOCOL)):
        if file_record(root, name) != packet[key]:
            raise ValueError("prospective recovery protocol/proposal changed")
    if {name: file_record(root, name) for name in packet["input_files"]} != packet["input_files"]:
        raise ValueError("frozen recovery inputs changed")
    if (recovery_config(packet["original_config"]) != packet["scientific_config"]
            or recovery_budget_plan(packet["original_config"]) != packet["budget_plan"]):
        raise ValueError("continuation schedule differs from original scientific design")


def approved(root):
    auth, packet = committed_json(root, AUTHORIZATION), committed_json(root, FROZEN)
    validate_authorization(auth, packet)
    if (packet["workspace"] != str(Path(root).resolve()) or packet["branch"] != BRANCH
            or git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain")):
        raise ValueError("clean declared recovery workspace required")
    verify_bindings(root, packet)
    return packet, auth


class RecoveryAdmission:
    def __init__(self, workspace, packet, authorization, claim, budget):
        validate_authorization(authorization, packet)
        self.workspace, self.root = Path(workspace).resolve(), confined(workspace, RUN)
        self.packet, self.claim, self.budget = packet, claim, budget
        if (claim.get("pid") != os.getppid() or claim.get("packet_sha256") != packet["packet_sha256"]
                or claim.get("authorization_sha256") != digest(authorization) or claim.get("clock_id") != CLOCK_ID
                or claim.get("started") != budget.started or budget.plan != packet["budget_plan"]
                or budget.path.resolve() != self.root / "launcher/budget.jsonl"):
            raise PermissionError("live exclusive new claim and budget required")
        self.pid, self.counts = os.getpid(), {}
        self.caps = {"checkpoint_load": 3, "reference_checkpoint_load": 3,
                     "layout_environment_build": 3, "episode_build": packet["scientific_config"]["totals"]["fresh_episode_builds"]}

    def check(self):
        if (os.getpid() != self.pid or os.getppid() != self.claim["pid"]
                or json.loads((self.root / "launcher/claim.json").read_text()) != self.claim
                or (self.root / "launcher/terminal.json").exists()):
            raise PermissionError("recovery owner changed or attempt is terminal")
        self.budget.check()
        if self.budget.active is None:
            raise PermissionError("numerical work outside an active recovery scope")

    def __call__(self, operation):
        self.check()
        if operation in self.caps:
            self.debit(operation)
        elif operation not in ("reference_and_layout_build", "input_hash_verification"):
            raise PermissionError("undeclared scientific operation")

    def debit(self, operation):
        self.check()
        if operation not in self.caps or self.counts.get(operation, 0) >= self.caps[operation]:
            raise ValueError("recovery construction/load cap exceeded")
        if operation != "episode_build" and self.budget.active != "runtime_input_binding":
            raise PermissionError("historical models/layouts may only be loaded once during binding")
        if (operation == "episode_build" and self.budget.active != "same_start_preflight"
                and not self.budget.active.startswith(("ppo_continuation/", "bc_continuation/", "final_evaluation/"))):
            raise PermissionError("episode construction forbidden outside preflight, continuation or final evaluation")
        count = self.counts.get(operation, 0) + 1
        write_json_once(self.root / f"launcher/operations/{operation}-{count:04d}.json", {
            "operation": operation, "count": count, "phase": self.budget.active,
            "clock": shared_monotonic(), "before_operation": True})
        self.counts[operation] = count

    def bind_campaign(self, root, config, streams, budget, backend):
        if (Path(root).resolve() != self.root or config != self.packet["scientific_config"]
                or streams != self.packet["streams"] or budget is not self.budget
                or type(backend) is not DynamicPatientBackend or backend.admit is not self):
            raise PermissionError("campaign does not match the new recovery admission")


def child(root):
    packet, auth = approved(root)
    output = confined(root, RUN)
    claim = json.loads((output / "launcher/claim.json").read_text())
    if claim["head"] != git(root, "rev-parse", "HEAD"):
        raise PermissionError("claim execution commit changed")
    write_json_once(output / "launcher/child-claim.json", {"pid": os.getpid(), "ppid": os.getppid()})
    budget = DynamicCandidateBudget(output / "launcher/budget.jsonl", packet["budget_plan"],
                                    enabled=True, started=claim["started"])
    campaign = None
    try:
        admission = RecoveryAdmission(root, packet, auth, claim, budget)
        backend = DynamicPatientBackend(root, packet["scientific_config"], packet["streams"], admit_real_calls=admission)
        loaded = set()
        def initializer_loader(block):
            if block in loaded:
                raise ValueError("initializer cannot be loaded twice")
            loaded.add(block)
            record = packet["initializers"][str(block)]
            state = load_envelope(confined(root, record["path"]), record["sha256"], admission)
            view = restore_saved_policy(state, packet["original_config"], packet["streams"], block)
            view.steps = state["steps"]
            return view
        def final_lock_check():
            verify_bindings(root, packet)
            if admission.counts != admission.caps:
                raise ValueError("incomplete recovery load/build matrix")
        from src.rl.dynamic_candidate_recovery_campaign import RecoveryCampaign
        campaign = RecoveryCampaign(output, packet["original_config"], packet["streams"], budget, backend,
            initializer_loader=initializer_loader, qualifications=packet["qualified_blocks"],
            enabled=True, engineering_only=False, execution_admission=admission, final_lock_check=final_lock_check)
        campaign.begin_next()
        paths = set(packet["source_files"]) | set(packet["input_files"]) | {FROZEN, AUTHORIZATION, PROPOSAL, PROTOCOL}
        for name in sorted(paths):
            budget.check()
            copy_verified(confined(root, name), output / "payload/locks/worktree" / name)
        write_json_once(output / "payload/locks/execution.json", {
            "head": claim["head"], "packet_sha256": packet["packet_sha256"], "authorization_sha256": digest(auth),
            "old_attempt_remains_terminal": True, "new_independent_budget": True, "dropbox_exported": False})
        return campaign.run()
    except BaseException as error:
        if campaign is not None:
            campaign.fail(error)
        else:
            write_json_once(output / "launcher/child-failure.json", {"error": repr(error), "automatic_retry": False})
            budget.close()
        return 1


def launch(root, *, started=None):
    started = shared_monotonic() if started is None else started
    packet, auth = approved(root)
    if shared_monotonic() - started >= packet["budget_plan"]["sections"]["runtime_input_binding"]["seconds"]:
        raise TimeoutError("recovery setup deadline before claim")
    output = confined(root, RUN)
    output.mkdir(parents=True, exist_ok=False)
    launcher = output / "launcher"
    write_json_once(launcher / "claim.json", {"pid": os.getpid(), "ppid": os.getppid(),
        "head": git(root, "rev-parse", "HEAD"), "packet_sha256": packet["packet_sha256"],
        "authorization_sha256": digest(auth), "started": started, "clock_id": CLOCK_ID,
        "automatic_retry": False})
    try:
        result = supervise_dynamic([sys.executable, "-m", "experiments.scripts.run_dynamic_candidate_recovery", "--child"],
            cwd=root, launcher=launcher, plan=packet["budget_plan"], started=started)
        success = verify_completed(output, packet, result) and (launcher / "stderr.log").stat().st_size == 0
        success = success and shared_monotonic() <= result["final_deadline"]
        write_json_once(launcher / "terminal.json", {"status": "completed" if success else "failed",
            "exit_code": 0 if success else 1, "scientific_completion_verified": success,
            "supervisor": result, "automatic_retry": False, "automatic_followon": False,
            "dropbox_exported": False, "requires_successful_closure_archive": success})
        if shared_monotonic() > result["final_deadline"]:
            write_json_once(launcher / "terminal-overrun.json", {"status": "failed", "automatic_retry": False})
            return 1
        if success:
            archive = create_archive(launcher, output / "archives/completed-launcher.tar.gz")
            if shared_monotonic() > result["final_deadline"]:
                raise TimeoutError("closure archive exceeded phase/global deadline; attempt not complete")
            write_json_once(output / "closure-archive-receipt.json", {"archive": archive,
                "local_archive_verified": True, "dropbox_exported": False,
                "cloud_sync_verified": False, "howard_access_verified": False})
            if shared_monotonic() > result["final_deadline"]:
                raise TimeoutError("closure receipt exceeded deadline")
        return 0 if success else 1
    except BaseException as error:
        write_json_once(launcher / "launch-failure.json", {"status": "failed", "error": repr(error), "automatic_retry": False})
        return 1
