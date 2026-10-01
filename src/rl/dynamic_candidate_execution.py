"""Source-bound, one-attempt dynamic-pilot execution after explicit approval.

No authorization is shipped. The fixed-path approval record and frozen proposal
must be committed and agree before an exclusive claim or numerical call exists.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_execution import configure_runtime, runtime_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import audit_stream_collisions, digest
from src.rl.dynamic_candidate_backend import DynamicPatientBackend
from src.rl.dynamic_candidate_preparation import (
    BRANCH, PACKET_DIRECTORY, RUN_DIRECTORY, PROTOCOL, git, prepare_proposal,
    seed_evidence_inventory, source_locks,
)
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, dynamic_budget_plan, read_dynamic_ledger
from src.rl.dynamic_candidate_watchdog import supervise_dynamic
from src.utils.research_archive import copy_verified
from src.utils.research_clock import CLOCK_ID, shared_monotonic


PACKET = PACKET_DIRECTORY + "/proposal.json"
AUTHORIZATION = "specs/2026-10-01-adaptive-paper-delivery/execution-authorization.json"
REQUIRED_AMENDMENTS = {"preserve_failed_a1_replace_prerequisite": True,
    "specimen_routes_instead_of_draft_graph_label": True,
    "neutral_trainable_reference_bias": True, "unchanged_reward_and_scenario": True}


def confined(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("worktree-relative path required")
    result = Path(root) / path
    if result.absolute() != result.resolve() or not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("symlinked or external execution path forbidden")
    return result


def committed_json(root, name):
    path = confined(root, name)
    if not path.is_file():
        raise PermissionError("missing committed prospective approval/packet: " + name)
    raw = path.read_bytes()
    committed = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=root)
    if raw != committed:
        raise ValueError("execution record must be committed without modifications")
    return json.loads(raw)


def verify_frozen_proposal(root, packet, *, require_clean=True, refresh_seeds=True):
    root = Path(root).resolve()
    if git(root, "branch", "--show-current") != BRANCH or (require_clean and git(root, "status", "--porcelain")):
        raise ValueError("clean declared branch required")
    sealed = dict(packet)
    content_sha = sealed.pop("packet_content_sha256", None)
    if (digest(sealed) != content_sha or packet.get("source_frozen") is not True
            or packet.get("scientific_execution_authorized") is not False or packet.get("ready_to_launch") is not False
            or packet.get("workspace") != str(root) or packet.get("branch") != BRANCH):
        raise ValueError("invalid unapproved frozen proposal receipt")
    subprocess.run(["git", "merge-base", "--is-ancestor", packet["implementation_commit"], "HEAD"], cwd=root, check=True)
    current = prepare_proposal(root)
    for key in ("scientific_config", "static_inputs", "parameter_counts", "streams", "budget_plan", "protocol"):
        if current[key] != packet[key]:
            raise ValueError("prospective configuration or input binding changed: " + key)
    if source_locks(root) != packet["source_files"] or runtime_record() != packet["runtime"]:
        raise ValueError("source/runtime differs from frozen proposal")
    for name, record in packet["source_files"].items():
        content = subprocess.check_output(["git", "show", packet["implementation_commit"] + ":" + name], cwd=root)
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("source inventory is not bound to implementation commit")
    if refresh_seeds:
        streams = packet["streams"]
        audit = audit_stream_collisions({"environment": streams["environment"], "neural": streams["neural"]},
                                        seed_evidence_inventory(root))
        for record in audit["files"] + audit["collisions"]:
            record["path"] = Path(record["path"]).relative_to(root).as_posix()
        if audit != packet["seed_collision_audit"] or audit["passed"] is not True:
            raise ValueError("local seed declaration inventory changed")
    return packet["scientific_config"]


def validate_authorization(authorization, packet):
    expected_limits = {"environment_steps": packet["budget_plan"]["limits"]["trajectory"] + packet["budget_plan"]["limits"]["clone"],
        "optimizer_steps": packet["budget_plan"]["limits"]["actor"] + packet["budget_plan"]["limits"]["critic"],
        "seconds": packet["budget_plan"]["limits"]["seconds"], "attempts": 1}
    if (authorization.get("format") != "dynamic-pilot-explicit-authorization-v1"
            or authorization.get("approved") is not True
            or authorization.get("proposal_content_sha256") != packet["packet_content_sha256"]
            or authorization.get("implementation_commit") != packet["implementation_commit"]
            or authorization.get("scientific_config_sha256") != digest(packet["scientific_config"])
            or authorization.get("amendments") != REQUIRED_AMENDMENTS
            or authorization.get("limits") != expected_limits
            or authorization.get("automatic_retry") is not False
            or authorization.get("remote_or_dropbox_actions") is not False):
        raise PermissionError("a complete explicit approval of this exact prospective packet is required")
    approval = authorization.get("user_approval", {})
    if (approval.get("user") != "Zhaowei" or not isinstance(approval.get("verbatim"), str)
            or not approval["verbatim"].strip() or not approval.get("recorded_at_utc")):
        raise PermissionError("actual user approval text and timestamp required; no inferred sign-off")
    if (authorization.get("protocol") != PROTOCOL
            or authorization.get("protocol_sha256") != packet["protocol"]["sha256"]):
        raise PermissionError("committed protocol binding required")


def approved_packet(root):
    # Authorization is checked before potentially expensive source/input reads.
    authorization = committed_json(root, AUTHORIZATION)
    packet = committed_json(root, PACKET)
    validate_authorization(authorization, packet)
    protocol = confined(root, PROTOCOL)
    raw = protocol.read_bytes()
    if (hashlib.sha256(raw).hexdigest() != authorization["protocol_sha256"]
            or subprocess.check_output(["git", "show", "HEAD:" + PROTOCOL], cwd=root) != raw):
        raise ValueError("approved protocol changed or is not committed")
    verify_frozen_proposal(root, packet)
    return packet, authorization


class ScientificAdmission:
    """Live owned child capability; not a substitute for the recorded approval."""
    def __init__(self, workspace, packet, authorization, claim, budget):
        validate_authorization(authorization, packet)
        self.workspace = Path(workspace).resolve()
        self.root = confined(self.workspace, RUN_DIRECTORY)
        self.packet, self.authorization, self.claim, self.budget = packet, authorization, claim, budget
        if (claim.get("format") != "dynamic-exclusive-claim-v1" or claim.get("pid") != os.getppid()
                or claim.get("proposal_content_sha256") != packet["packet_content_sha256"]
                or claim.get("authorization_sha256") != digest(authorization)
                or claim.get("clock_id") != CLOCK_ID or claim.get("started") != budget.started
                or budget.path.resolve() != self.root / "launcher/budget.jsonl"
                or budget.plan != packet["budget_plan"]):
            raise PermissionError("live exclusive parent claim and exact irreversible budget required")
        self.pid = os.getpid()

    def __call__(self, operation):
        if os.getpid() != self.pid or os.getppid() != self.claim["pid"]:
            raise PermissionError("execution owner changed")
        claim = json.loads((self.root / "launcher/claim.json").read_text())
        if claim != self.claim or (self.root / "launcher/terminal.json").exists():
            raise PermissionError("claim changed or attempt is terminal")
        self.budget.check()
        if self.budget.active is None:
            raise PermissionError("scientific work outside a declared phase")

    def bind_campaign(self, root, config, streams, budget, backend):
        if (Path(root).resolve() != self.root or config != self.packet["scientific_config"]
                or streams != self.packet["streams"] or budget is not self.budget
                or type(backend) is not DynamicPatientBackend or backend.admit is not self):
            raise PermissionError("campaign/backend differs from the admitted packet")
        if os.getpid() != self.pid or os.getppid() != self.claim["pid"]:
            raise PermissionError("not the admitted process owner")


def preserve_inputs(root, output, packet, authorization, budget):
    paths = set(packet["source_files"]) | set(packet["static_inputs"]["inputs"])
    paths.update((PACKET, AUTHORIZATION, PROTOCOL))
    paths.update(row["path"] for row in packet["scientific_config"]["source_bindings"].values())
    for name in sorted(paths):
        budget.check()
        copy_verified(confined(root, name), output / "payload/locks/worktree" / name)
    budget.check()
    write_json_once(output / "payload/locks/execution.json", {
        "head": git(root, "rev-parse", "HEAD"), "implementation_commit": packet["implementation_commit"],
        "proposal_content_sha256": packet["packet_content_sha256"], "authorization_sha256": digest(authorization),
        "all_local_inputs_and_source_copied": True, "dropbox_exported": False})


def child(root):
    root = Path(root).resolve()
    packet, authorization = approved_packet(root)
    output = confined(root, RUN_DIRECTORY)
    claim = json.loads((output / "launcher/claim.json").read_text())
    if claim["pid"] != os.getppid() or claim["head"] != git(root, "rev-parse", "HEAD"):
        raise PermissionError("child must belong to the exact claiming supervisor")
    write_json_once(output / "launcher/child-claim.json", {"pid": os.getpid(), "ppid": os.getppid(),
        "proposal_content_sha256": packet["packet_content_sha256"]})
    budget = DynamicCandidateBudget(output / "launcher/budget.jsonl", packet["budget_plan"],
                                    enabled=True, started=claim["started"])
    campaign = None
    try:
        admission = ScientificAdmission(root, packet, authorization, claim, budget)
        backend = DynamicPatientBackend(root, packet["scientific_config"], packet["streams"], admit_real_calls=admission)
        from src.rl.dynamic_candidate_campaign import DynamicCandidateCampaign
        campaign = DynamicCandidateCampaign(output, packet["scientific_config"], packet["streams"], budget, backend,
            enabled=True, engineering_only=False, execution_admission=admission,
            final_lock_check=lambda: verify_frozen_proposal(root, packet, require_clean=False, refresh_seeds=False))
        campaign.begin_next()
        preserve_inputs(root, output, packet, authorization, budget)
        return campaign.run()
    except BaseException as error:
        if campaign is not None:
            campaign.fail(error)
        else:
            write_json_once(output / "launcher/child-failure.json", {"error": repr(error), "retry_permitted": False})
            budget.close()
        return 1


def verify_completed(output, packet, supervisor):
    if supervisor["passed"] is not True:
        return False
    completed = json.loads((output / "launcher/completed.json").read_text())
    if (completed.get("status") != "completed" or completed.get("scientific_execution") is not True
            or completed.get("engineering_fixture") is not False or completed.get("exit_code") != 0):
        return False
    ledger = read_dynamic_ledger(output / "launcher/budget.jsonl")
    limits = packet["budget_plan"]["limits"]
    if (ledger["plan_sha256"] != digest(packet["budget_plan"])
            or set(ledger["closed"]) != set(packet["budget_plan"]["sections"])
            or {k: ledger["counts"].get(k, 0) for k in ("environment", "optimizer")} != {
                "environment": limits["trajectory"] + limits["clone"], "optimizer": limits["actor"] + limits["critic"]}):
        return False
    receipt = json.loads((output / "launcher/archive-receipt.json").read_text())
    expected = receipt["archive"]
    archive = output / "archives/completed-payload.tar.gz"
    if (expected.get("verified_by_reading_every_archive_member") is not True
            or expected.get("archive") != archive.name or archive.is_symlink()
            or expected["size_bytes"] != archive.stat().st_size):
        return False
    # Member verification and unchanged-payload checks already ran under the
    # child watchdog. Confirm the resulting archive bytes once, not another
    # full member and source-tree audit in the supervisor's closure phase.
    checksum = hashlib.sha256()
    with archive.open("rb") as handle:
        while True:
            if shared_monotonic() > supervisor.get("final_deadline", float("inf")):
                raise TimeoutError("terminal archive readback deadline")
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            checksum.update(chunk)
    if checksum.hexdigest() != expected["archive_sha256"]:
        return False
    return not (output / "launcher/failure.json").exists()


def launch(root):
    root = Path(root).resolve()
    started = shared_monotonic()
    packet, authorization = approved_packet(root)
    plan = packet["budget_plan"]
    if shared_monotonic() - started >= plan["sections"]["runtime_input_binding"]["seconds"]:
        raise TimeoutError("source binding exceeded initial phase before claim")
    output = confined(root, RUN_DIRECTORY)
    output.mkdir(parents=True, exist_ok=False)
    launcher = output / "launcher"
    launcher.mkdir()
    claim = {"format": "dynamic-exclusive-claim-v1", "pid": os.getpid(), "ppid": os.getppid(),
        "head": git(root, "rev-parse", "HEAD"), "implementation_commit": packet["implementation_commit"],
        "proposal_content_sha256": packet["packet_content_sha256"], "authorization_sha256": digest(authorization),
        "started": started, "clock_id": CLOCK_ID, "automatic_retry": False}
    write_json_once(launcher / "claim.json", claim)
    try:
        command = [sys.executable, "-m", "experiments.scripts.run_dynamic_candidate_pilot", "--child"]
        result = supervise_dynamic(command, cwd=root, launcher=launcher, plan=plan, started=started)
        success = verify_completed(output, packet, result)
        success = success and shared_monotonic() <= result["final_deadline"]
        write_json_once(launcher / "terminal.json", {"status": "completed" if success else "failed",
            "scientific_completion_verified": success, "supervisor": result,
            "automatic_retry": False, "dropbox_exported": False})
        if shared_monotonic() > result["final_deadline"]:
            write_json_once(launcher / "terminal-overrun.json", {"status": "failed", "automatic_retry": False,
                "reason": "terminal preservation exceeded deadline; no numerical resume"})
            return 1
        return 0 if success else 1
    except BaseException as error:
        write_json_once(launcher / "launch-failure.json", {"status": "failed", "error": repr(error), "automatic_retry": False})
        return 1
