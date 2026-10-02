"""Hash-only preparation and separately authorized one-attempt execution.

Freezing a packet never opens a model, builds an environment or authorizes a
scientific forward. Admission requires a committed, scope-specific human reply.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_execution import configure_runtime, runtime_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import audit_stream_collisions, digest
from src.rl.dynamic_candidate_execution import committed_json, confined, verify_completed
from src.rl.dynamic_candidate_preparation import BRANCH, git, source_locks
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget
from src.rl.dynamic_candidate_saved_execution import load_envelope
from src.rl.dynamic_candidate_saved_qualification import restore_saved_policy
from src.rl.time_baseline_backend import TimeBaselinePatientBackend
from src.rl.time_baseline_factory import time_baseline_config
from src.rl.time_baseline_plan import time_baseline_budget_plan, time_baseline_stream_manifest
from src.utils.research_archive import copy_verified, create_archive
from src.utils.research_clock import CLOCK_ID, shared_monotonic


DIRECTORY = "specs/2026-10-02-time-baseline-comparison"
PROPOSAL, PROTOCOL = DIRECTORY + "/proposal.json", DIRECTORY + "/protocol.md"
FROZEN, AUTHORIZATION = DIRECTORY + "/frozen.json", DIRECTORY + "/authorization.json"
REWARD_PIVOT = DIRECTORY + "/reward-pivot.md"
RUN = "results/dynamic_candidate_time_baseline_20261002"
PRIOR_PACKET = "specs/2026-10-01-adaptive-paper-delivery/continuation-recovery-frozen.json"
QUALIFICATION = "results/dynamic_candidate_saved_qualification_20261001"
QUALIFICATION_SHA = "f1b271a72b4bfa1d0b9d16deb3651af60f1d56aedbf4dc541ded0f133257b9b6"
TERMINAL_SHA = "5617d5fc053efcea92073f18f2a0d46e7b4efd1f4925bf455c770dcb3ce28906"


def input_bindings(root):
    """Read only reused qualification metadata and hash its three saved models."""
    prior = committed_json(root, PRIOR_PACKET)
    body = dict(prior)
    if digest({k: v for k, v in body.items() if k != "packet_sha256"}) != body["packet_sha256"]:
        raise ValueError("prior input receipt changed")
    names = [QUALIFICATION + "/qualification.json", QUALIFICATION + "/terminal.json"]
    files = {name: file_record(root, name) for name in [PRIOR_PACKET, *names]}
    if files[names[0]]["sha256"] != QUALIFICATION_SHA or files[names[1]]["sha256"] != TERMINAL_SHA:
        raise ValueError("exact completed saved qualification required")
    qualification, terminal = [json.loads(confined(root, name).read_text()) for name in names]
    if (qualification["passed"] is not True or qualification["source_unchanged"] is not True
            or terminal["status"] != "completed" or terminal["exit_code"] != 0
            or qualification["blocks"] != prior["qualified_blocks"]):
        raise ValueError("saved qualification is not the bound completed receipt")
    for record in prior["initializers"].values():
        name = record["path"]
        if (file_record(root, name) != record or record != qualification["input_files_before"][name]
                or record != qualification["input_files_after"][name]):
            raise ValueError("qualified initializer bytes changed")
        files[name] = record
    reference = prior["original_config"]["reference"]
    for block in prior["original_config"]["blocks"]:
        for kind in ("config", "policy"):
            name = reference["directory"] + "/" + reference[kind + "_template"].format(block=block)
            record = file_record(root, name)
            if record["sha256"] != reference["locks"][str(block)][kind]:
                raise ValueError("locked R4 input changed")
            files[name] = record
    return prior, files


def seed_inventory(root):
    """Tracked configs and locally present named manifests, excluding only this attempt.

    Include the preserved original attempt and older frozen packets. Unavailable
    external files cannot be covered. No old trajectories or model payloads are
    decoded, and prospective seeds are never silently replaced after a collision.
    """
    root = Path(root).resolve()
    files = {confined(root, name) for name in git(root, "ls-files", "experiments/configs").splitlines()
             if name.endswith(".json")}
    for directory in ("results", "reports", "specs"):
        for path in (root / directory).rglob("*"):
            name = path.relative_to(root).as_posix()
            if name.startswith((DIRECTORY + "/", RUN + "/")) or not path.is_file():
                continue
            if not any(word in path.name.lower() for word in ("seed", "stream", "manifest", "config", "frozen")):
                continue
            if path.suffix in (".json", ".jsonl"):
                files.add(path)
            elif path.suffix in (".csv", ".yaml", ".yml", ".toml"):
                raise ValueError("unhandled local seed declaration format: " + name)
    files.add(confined(root, PRIOR_PACKET))
    return sorted(files)


def freeze(root):
    root = Path(root).resolve()
    if git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain"):
        raise ValueError("freeze requires clean committed integration branch")
    proposal = committed_json(root, PROPOSAL)
    if (proposal.get("scientific_execution_authorized") is not False or proposal["result_root"] != RUN
            or proposal["blocks"] != [60, 61, 62]):
        raise ValueError("original unapproved numeric proposal required")
    plan, streams = time_baseline_budget_plan(proposal), time_baseline_stream_manifest(proposal)
    prior, files = input_bindings(root)
    audit = audit_stream_collisions({k: streams[k] for k in ("environment", "neural")}, seed_inventory(root))
    if not audit["passed"]:
        raise ValueError("prospective seeds collide with local history: " + repr(audit["collisions"]))
    packet = {"format": "time-baseline-frozen-v1", "scientific_execution_authorized": False,
        "ready_to_launch": False, "workspace": str(root), "branch": BRANCH,
        "implementation_commit": git(root, "rev-parse", "HEAD"), "source_frozen": True,
        "proposal": file_record(root, PROPOSAL), "protocol": file_record(root, PROTOCOL),
        "reward_pivot": file_record(root, REWARD_PIVOT), "proposal_data": proposal,
        "original_config": prior["original_config"], "initializer_streams": prior["streams"],
        "scientific_config": time_baseline_config(prior["original_config"], proposal),
        "streams": streams, "budget_plan": plan, "qualified_blocks": prior["qualified_blocks"],
        "initializers": prior["initializers"], "input_files": files,
        "source_files": source_locks(root), "runtime": runtime_record(),
        "local_seed_collision_audit": audit,
        "seed_audit_scope": "tracked JSON configs and all local named seed/stream/manifest/config/frozen JSON records in results/reports/specs; this prospective directory excluded; unavailable external runs not covered",
        "new_checkpoint_loads": 0, "new_environment_calls": 0,
        "new_optimizer_calls": 0, "new_model_forwards": 0}
    packet["packet_sha256"] = digest(packet)
    return packet


def approved_limits(packet):
    proposal = packet["proposal_data"]
    return {"attempts": 1, "blocks": proposal["blocks"], "episodes": proposal["totals"]["episodes"],
        "training_episodes_per_arm_block": proposal["episodes_per_training_arm_block"],
        "environment_steps": proposal["totals"]["environment"],
        "optimizer_calls": proposal["totals"]["optimizer"], "seconds": proposal["totals"]["global_seconds"],
        "historical_initializer_loads": 3, "reference_loads": 3, "layout_builds": 3,
        "episode_builds": proposal["episode_builds"],
        "final_test_episodes": packet["scientific_config"]["evaluation"]["total_episodes"],
        "phase_and_owner_plan_sha256": digest(packet["budget_plan"])}


def validate_authorization(auth, packet):
    body = dict(packet)
    sha = body.pop("packet_sha256", None)
    if digest(body) != sha:
        raise ValueError("time-baseline packet changed")
    if (packet.get("format") != "time-baseline-frozen-v1"
            or packet["scientific_execution_authorized"] is not False or packet["ready_to_launch"] is not False
            or auth.get("format") != "time-baseline-authorization-v1" or auth.get("approved") is not True
            or auth.get("packet_sha256") != sha or auth.get("implementation_commit") != packet["implementation_commit"]
            or auth.get("limits") != approved_limits(packet) or auth.get("automatic_retry") is not False
            or auth.get("remote_or_dropbox_actions") is not False or auth.get("reward_change_authorized") is not False
            or auth.get("old_attempts_remain_terminal") is not True
            or auth.get("reuse_completed_qualification_without_rescoring") is not True):
        raise PermissionError("complete new numeric package approval required")
    user = auth.get("user_approval", {})
    if (user.get("user") != "Zhaowei" or not isinstance(user.get("verbatim"), str)
            or not user["verbatim"].strip() or not user.get("recorded_at_utc")
            or user.get("reply_to_numeric_package") is not True):
        raise PermissionError("actual scope-specific reply after the numeric package required")


def verify_bindings(root, packet):
    subprocess.run(["git", "merge-base", "--is-ancestor", packet["implementation_commit"], "HEAD"], cwd=root, check=True)
    subprocess.run(["git", "diff", "--exit-code", packet["implementation_commit"], "--",
                    "src", "experiments/scripts", "tests", "AGENTS.md"], cwd=root, check=True, stdout=subprocess.DEVNULL)
    if source_locks(root) != packet["source_files"] or runtime_record() != packet["runtime"]:
        raise ValueError("frozen comparison source/runtime changed")
    for key, name in (("proposal", PROPOSAL), ("protocol", PROTOCOL), ("reward_pivot", REWARD_PIVOT)):
        if file_record(root, name) != packet[key]:
            raise ValueError("prospective comparison protocol changed")
    if {name: file_record(root, name) for name in packet["input_files"]} != packet["input_files"]:
        raise ValueError("frozen comparison inputs changed")
    files = seed_inventory(root)
    if [file_record(root, p) | {"path": str(p)} for p in files] != packet["local_seed_collision_audit"]["files"]:
        raise ValueError("local historical seed inventory changed since freeze")
    proposal = packet["proposal_data"]
    if (time_baseline_config(packet["original_config"], proposal) != packet["scientific_config"]
            or time_baseline_budget_plan(proposal) != packet["budget_plan"]
            or time_baseline_stream_manifest(proposal) != packet["streams"]):
        raise ValueError("comparison bindings differ from the prospective design")


def approved(root):
    auth, packet = committed_json(root, AUTHORIZATION), committed_json(root, FROZEN)
    validate_authorization(auth, packet)
    if (packet["workspace"] != str(Path(root).resolve()) or packet["branch"] != BRANCH
            or git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain")):
        raise ValueError("clean declared comparison workspace required")
    verify_bindings(root, packet)
    return packet, auth


class TimeBaselineAdmission:
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
            "layout_environment_build": 3, "episode_build": packet["proposal_data"]["episode_builds"]}

    def check(self):
        if (os.getpid() != self.pid or os.getppid() != self.claim["pid"]
                or json.loads((self.root / "launcher/claim.json").read_text()) != self.claim
                or (self.root / "launcher/terminal.json").exists()):
            raise PermissionError("comparison owner changed or attempt is terminal")
        self.budget.check()
        if self.budget.active is None:
            raise PermissionError("numerical work outside active comparison scope")

    def __call__(self, operation):
        self.check()
        if operation in self.caps:
            self.debit(operation)
        elif operation not in ("reference_and_layout_build", "input_hash_verification"):
            raise PermissionError("undeclared scientific operation")

    def debit(self, operation):
        self.check()
        if operation not in self.caps or self.counts.get(operation, 0) >= self.caps[operation]:
            raise ValueError("comparison construction/load cap exceeded")
        if operation != "episode_build" and self.budget.active != "runtime_input_binding":
            raise PermissionError("saved models/layouts may only be loaded once during binding")
        if (operation == "episode_build" and self.budget.active != "same_start_preflight"
                and not self.budget.active.startswith(("current_ppo/", "time_baseline_ppo/", "bc_continue/", "final_evaluation/"))):
            raise PermissionError("episode construction outside declared phase")
        count = self.counts.get(operation, 0) + 1
        write_json_once(self.root / f"launcher/operations/{operation}-{count:04d}.json", {
            "operation": operation, "count": count, "phase": self.budget.active,
            "clock": shared_monotonic(), "before_operation": True})
        self.counts[operation] = count

    def bind_campaign(self, root, config, streams, budget, backend):
        if (Path(root).resolve() != self.root or config != self.packet["scientific_config"]
                or streams != self.packet["streams"] or budget is not self.budget
                or type(backend) is not TimeBaselinePatientBackend or backend.admit is not self):
            raise PermissionError("campaign does not match the new comparison admission")


def child(root):
    packet, auth = approved(root)
    output = confined(root, RUN)
    claim = json.loads((output / "launcher/claim.json").read_text())
    if claim["head"] != git(root, "rev-parse", "HEAD"):
        raise PermissionError("claim execution commit changed")
    write_json_once(output / "launcher/child-claim.json", {"pid": os.getpid(), "ppid": os.getppid()})
    budget = DynamicCandidateBudget(output / "launcher/budget.jsonl", packet["budget_plan"], enabled=True, started=claim["started"])
    campaign = None
    try:
        admission = TimeBaselineAdmission(root, packet, auth, claim, budget)
        backend = TimeBaselinePatientBackend(root, packet["scientific_config"], packet["streams"], admit_real_calls=admission)
        loaded = set()
        def initializer_loader(block):
            if block in loaded:
                raise ValueError("initializer cannot be loaded twice")
            loaded.add(block)
            record = packet["initializers"][str(block)]
            state = load_envelope(confined(root, record["path"]), record["sha256"], admission)
            view = restore_saved_policy(state, packet["original_config"], packet["initializer_streams"], block)
            view.steps = state["steps"]
            return view
        def final_lock_check():
            verify_bindings(root, packet)
            if admission.counts != admission.caps:
                raise ValueError("incomplete comparison load/build matrix")
        from src.rl.time_baseline_campaign import TimeBaselineCampaign
        campaign = TimeBaselineCampaign(output, packet["original_config"], packet["proposal_data"], packet["streams"],
            budget, backend, initializer_loader=initializer_loader, qualifications=packet["qualified_blocks"],
            enabled=True, engineering_only=False, execution_admission=admission, final_lock_check=final_lock_check)
        campaign.begin_next()
        seed_files = {str(Path(record["path"]).relative_to(Path(root).resolve()))
                      for record in packet["local_seed_collision_audit"]["files"]}
        paths = set(packet["source_files"]) | set(packet["input_files"]) | seed_files | {FROZEN, AUTHORIZATION, PROPOSAL, PROTOCOL, REWARD_PIVOT}
        for name in sorted(paths):
            budget.check()
            copy_verified(confined(root, name), output / "payload/locks/worktree" / name)
        write_json_once(output / "payload/locks/execution.json", {
            "head": claim["head"], "packet_sha256": packet["packet_sha256"], "authorization_sha256": digest(auth),
            "old_attempts_remain_terminal": True, "new_independent_budget": True, "dropbox_exported": False})
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
        raise TimeoutError("comparison setup deadline before claim")
    output = confined(root, RUN)
    output.mkdir(parents=True, exist_ok=False)
    launcher = output / "launcher"
    write_json_once(launcher / "claim.json", {"pid": os.getpid(), "ppid": os.getppid(),
        "head": git(root, "rev-parse", "HEAD"), "packet_sha256": packet["packet_sha256"],
        "authorization_sha256": digest(auth), "started": started, "clock_id": CLOCK_ID, "automatic_retry": False})
    try:
        from src.rl.time_baseline_watchdog import supervise_time_baseline
        result = supervise_time_baseline([sys.executable, "-m", "experiments.scripts.run_time_baseline_comparison", "--child"],
            cwd=root, launcher=launcher, plan=packet["budget_plan"], started=started)
        success = verify_completed(output, packet, result) and (launcher / "stderr.log").stat().st_size == 0
        success = success and shared_monotonic() <= result["final_deadline"]
        write_json_once(launcher / "terminal.json", {"status": "completed" if success else "failed",
            "exit_code": 0 if success else 1, "scientific_completion_verified": success, "supervisor": result,
            "automatic_retry": False, "automatic_followon": False, "dropbox_exported": False,
            "requires_successful_closure_archive": success})
        if shared_monotonic() > result["final_deadline"]:
            write_json_once(launcher / "terminal-overrun.json", {"status": "failed", "automatic_retry": False})
            return 1
        if success:
            archive = create_archive(launcher, output / "archives/completed-launcher.tar.gz")
            if shared_monotonic() > result["final_deadline"]:
                raise TimeoutError("closure archive exceeded deadline; attempt not complete")
            write_json_once(output / "closure-archive-receipt.json", {"archive": archive,
                "local_archive_verified": True, "dropbox_exported": False,
                "cloud_sync_verified": False, "howard_access_verified": False})
            if shared_monotonic() > result["final_deadline"]:
                raise TimeoutError("closure receipt exceeded deadline")
        return 0 if success else 1
    except BaseException as error:
        write_json_once(launcher / "launch-failure.json", {"status": "failed", "error": repr(error), "automatic_retry": False})
        return 1
