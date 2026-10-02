"""Scope-bound cohort preparation and one-attempt execution, never implicit approval.

The human START request precedes the final freeze. Its original scope/protocol
binding is distinct from the coordinator's later packet/runtime binding.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_execution import configure_runtime, runtime_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import audit_stream_collisions, digest
from src.rl.cohort_factory import cohort_config
from src.rl.cohort_objective_plan import cohort_budget_plan, cohort_stream_manifest
from src.rl.dynamic_candidate_execution import committed_json, confined, verify_completed
from src.rl.dynamic_candidate_preparation import BRANCH, git, source_locks
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget
from src.rl.dynamic_candidate_saved_execution import load_envelope
from src.rl.dynamic_candidate_saved_qualification import restore_saved_policy
from src.rl.time_baseline_execution import PRIOR_PACKET, input_bindings
from src.rl.time_baseline_watchdog import supervise_time_baseline
from src.utils.research_archive import copy_verified, create_archive
from src.utils.research_clock import CLOCK_ID, shared_monotonic


DIRECTORY = "specs/2026-10-02-terminal-obligation"
PROPOSAL, PROTOCOL = DIRECTORY + "/proposal.json", DIRECTORY + "/protocol.md"
FROZEN, AUTHORIZATION = DIRECTORY + "/frozen.json", DIRECTORY + "/authorization.json"
APPROVAL_INTENT = DIRECTORY + "/approval-intent.json"
BASE_PROPOSAL = "specs/2026-10-02-time-baseline-comparison/proposal.json"
BOUND = DIRECTORY + "/bound-and-settlement.md"
RUN = "results/dynamic_candidate_cohort_objective_20261002"


def _backend_class():
    from src.rl.cohort_backend import CohortPatientBackend
    return CohortPatientBackend


def _scope(base, proposal):
    plan = cohort_budget_plan(base, proposal)
    totals = proposal["totals"]
    if (base.get("scientific_execution_authorized") is not False
            or proposal.get("base_proposal") != BASE_PROPOSAL
            or proposal["blocks"] != [60, 61, 62]
            or [totals[k] for k in ("episodes", "environment", "optimizer", "global_seconds")] != [522, 33186, 1920, 10800]
            or [proposal[k] for k in ("enrollment_steps", "patient_resolution_steps", "accounting_steps", "economic_endpoint")] != [52, 8, 11, 63]
            or proposal.get("attempts") != 1 or proposal.get("automatic_retry") is not False
            or proposal.get("automatic_followon") is not False
            or proposal.get("no_new_penalty_or_salvage_weights") is not True):
        raise ValueError("exact one-shot cohort amendment required; no expanded scope")
    return plan


def seed_inventory(root):
    """Named local seed/config metadata only; preserve all prior attempt evidence."""
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
    base, proposal = committed_json(root, BASE_PROPOSAL), committed_json(root, PROPOSAL)
    plan, streams = _scope(base, proposal), cohort_stream_manifest(base, proposal)
    prior, files = input_bindings(root)
    audit = audit_stream_collisions({k: streams[k] for k in ("environment", "neural")}, seed_inventory(root))
    if not audit["passed"]:
        raise ValueError("prospective seeds collide with local history: " + repr(audit["collisions"]))
    packet = {"format": "cohort-objective-frozen-v1", "scientific_execution_authorized": False,
        "ready_to_launch": False, "workspace": str(root), "branch": BRANCH, "result_root": RUN,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": git(root, "rev-parse", "HEAD"), "source_frozen": True,
        "proposal": file_record(root, PROPOSAL), "protocol": file_record(root, PROTOCOL),
        "approval_intent": file_record(root, APPROVAL_INTENT),
        "approval_intent_data": committed_json(root, APPROVAL_INTENT),
        "base_proposal": file_record(root, BASE_PROPOSAL), "bound": file_record(root, BOUND),
        "proposal_data": proposal, "base_proposal_data": base,
        "original_config": prior["original_config"], "initializer_streams": prior["streams"],
        "scientific_config": cohort_config(prior["original_config"], base, proposal),
        "streams": streams, "budget_plan": plan, "qualified_blocks": prior["qualified_blocks"],
        "initializers": prior["initializers"], "input_files": files,
        "source_files": source_locks(root), "runtime": runtime_record(),
        "local_seed_collision_audit": audit,
        "seed_audit_scope": "tracked JSON configs and local named seed/stream/manifest/config/frozen JSON records in results/reports/specs; only this cohort directory/run excluded; unavailable external runs not covered",
        "new_checkpoint_loads": 0, "new_environment_calls": 0,
        "new_optimizer_calls": 0, "new_model_forwards": 0}
    _validate_intent(packet)
    packet["packet_sha256"] = digest(packet)
    return packet


def approved_limits(packet):
    proposal = packet["proposal_data"]
    return {"attempts": 1, "blocks": proposal["blocks"], "episodes": proposal["totals"]["episodes"],
        "training_episodes_per_arm_block": proposal["episodes_per_training_arm_block"],
        "environment_steps": proposal["totals"]["environment"],
        "optimizer_calls": proposal["totals"]["optimizer"], "seconds": proposal["totals"]["global_seconds"],
        "historical_initializer_loads": 3, "reference_loads": 3, "layout_builds": 3,
        "episode_builds": 522, "parity_environment_builds": 3,
        "prefix_steps": proposal["enrollment_steps"], "accounting_steps": proposal["accounting_steps"],
        "economic_endpoint": proposal["economic_endpoint"], "clone_instances": proposal["clone_instances"],
        "final_test_episodes": packet["scientific_config"]["evaluation"]["total_episodes"],
        "phase_and_owner_plan_sha256": digest(packet["budget_plan"])}


def _utc(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            raise ValueError("missing timezone")
        return parsed.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError) as error:
        raise PermissionError("explicit timezone-aware approval/freeze timestamps required") from error


def _validate_intent(packet):
    intent = packet["approval_intent_data"]
    expected = {"format": "cohort-objective-approval-intent-v1",
        "proposal_sha256": packet["proposal"]["sha256"],
        "protocol_sha256": packet["protocol"]["sha256"],
        "objective_amendment_approved": True, "main_episodes": 522,
        "environment_calls_max": 33186, "optimizer_calls_max": 1920,
        "wall_seconds_max": 10800, "attempts": 1, "automatic_retry": False,
        "automatic_followon": False, "no_new_weights_or_search": True,
        "approval_precedes_final_source_freeze": True, "execution_started": False,
        "requires_final_packet_bound_authorization_before_claim": True}
    if (any(type(intent.get(k)) is not type(v) or intent.get(k) != v for k, v in expected.items())
            or not isinstance(intent.get("user_literal"), str) or not intent["user_literal"].strip()
            or _utc(intent.get("recorded_at_utc")) > _utc(packet.get("frozen_at_utc"))):
        raise PermissionError("committed pre-freeze approval intent must bind the exact cohort scope/protocol")
    return intent


def validate_authorization(auth, packet):
    body = dict(packet)
    sha = body.pop("packet_sha256", None)
    if digest(body) != sha:
        raise ValueError("cohort packet changed")
    intent = _validate_intent(packet)
    plan = _scope(packet["base_proposal_data"], packet["proposal_data"])
    if (packet.get("format") != "cohort-objective-frozen-v1"
            or packet.get("scientific_execution_authorized") is not False
            or packet.get("ready_to_launch") is not False or packet.get("result_root") != RUN
            or plan != packet["budget_plan"]
            or auth.get("format") != "cohort-objective-authorization-v1" or auth.get("approved") is not True
            or auth.get("packet_sha256") != sha or auth.get("implementation_commit") != packet["implementation_commit"]
            or auth.get("runtime_sha256") != digest(packet["runtime"])
            or auth.get("approval_intent_sha256") != packet["approval_intent"]["sha256"]
            or auth.get("numeric_scope_sha256") != digest(packet["proposal_data"])
            or auth.get("protocol_sha256") != packet["protocol"]["sha256"]
            or auth.get("limits") != approved_limits(packet)
            or auth.get("automatic_retry") is not False or auth.get("automatic_followon") is not False
            or auth.get("remote_or_dropbox_actions") is not False
            or auth.get("objective_amendment_approved") is not True
            or auth.get("new_weights_or_search_authorized") is not False
            or auth.get("old_attempts_remain_terminal") is not True
            or auth.get("reuse_completed_qualification_without_rescoring") is not True):
        raise PermissionError("exact cohort START scope and final packet/runtime binding required")
    user = auth.get("user_approval", {})
    if (user.get("user") != "Zhaowei" or not isinstance(user.get("verbatim"), str)
            or not user["verbatim"].strip() or user.get("request_precedes_freeze") is not True
            or user["verbatim"] != intent["user_literal"]
            or user.get("recorded_at_utc") != intent["recorded_at_utc"]
            or user.get("numeric_scope_sha256") != auth["numeric_scope_sha256"]
            or user.get("protocol_sha256") != auth["protocol_sha256"]
            or _utc(user.get("recorded_at_utc")) > _utc(packet.get("frozen_at_utc"))):
        raise PermissionError("literal pre-freeze START request bound to this numeric scope/protocol required")


def verify_bindings(root, packet):
    subprocess.run(["git", "merge-base", "--is-ancestor", packet["implementation_commit"], "HEAD"], cwd=root, check=True)
    subprocess.run(["git", "diff", "--exit-code", packet["implementation_commit"], "--",
                    "src", "experiments/scripts", "tests", "AGENTS.md"], cwd=root, check=True, stdout=subprocess.DEVNULL)
    if source_locks(root) != packet["source_files"] or runtime_record() != packet["runtime"]:
        raise ValueError("frozen cohort source/runtime changed")
    for key, name in (("proposal", PROPOSAL), ("protocol", PROTOCOL), ("base_proposal", BASE_PROPOSAL),
                      ("bound", BOUND), ("approval_intent", APPROVAL_INTENT)):
        if file_record(root, name) != packet[key]:
            raise ValueError("cohort scope/protocol changed")
    if committed_json(root, APPROVAL_INTENT) != packet["approval_intent_data"]:
        raise ValueError("committed cohort approval intent differs from frozen receipt")
    if {name: file_record(root, name) for name in packet["input_files"]} != packet["input_files"]:
        raise ValueError("frozen cohort inputs changed")
    if [file_record(root, p) | {"path": str(p)} for p in seed_inventory(root)] != packet["local_seed_collision_audit"]["files"]:
        raise ValueError("local historical seed inventory changed since freeze")
    base, proposal = packet["base_proposal_data"], packet["proposal_data"]
    if (cohort_config(packet["original_config"], base, proposal) != packet["scientific_config"]
            or _scope(base, proposal) != packet["budget_plan"]
            or cohort_stream_manifest(base, proposal) != packet["streams"]):
        raise ValueError("cohort bindings differ from the fixed prospective design")


def approved(root):
    auth, packet = committed_json(root, AUTHORIZATION), committed_json(root, FROZEN)
    validate_authorization(auth, packet)
    if (packet["workspace"] != str(Path(root).resolve()) or packet["branch"] != BRANCH
            or git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain")):
        raise ValueError("clean declared cohort workspace required")
    verify_bindings(root, packet)
    return packet, auth


class CohortAdmission:
    def __init__(self, workspace, packet, authorization, claim, budget):
        validate_authorization(authorization, packet)
        self.workspace, self.root = Path(workspace).resolve(), confined(workspace, RUN)
        self.packet, self.claim, self.budget = packet, dict(claim), budget
        if (claim.get("pid") != os.getppid() or claim.get("packet_sha256") != packet["packet_sha256"]
                or claim.get("authorization_sha256") != digest(authorization) or claim.get("clock_id") != CLOCK_ID
                or claim.get("started") != budget.started or budget.plan != packet["budget_plan"]
                or budget.path.resolve() != self.root / "launcher/budget.jsonl"):
            raise PermissionError("live exclusive cohort claim and budget required")
        self.pid, self.counts = os.getpid(), {}
        self.caps = {"checkpoint_load": 3, "reference_checkpoint_load": 3,
            "layout_environment_build": 3, "episode_build": 522, "parity_environment_build": 3}

    def check(self):
        if (os.getpid() != self.pid or os.getppid() != self.claim["pid"]
                or json.loads((self.root / "launcher/claim.json").read_text()) != self.claim
                or (self.root / "launcher/terminal.json").exists()):
            raise PermissionError("cohort owner changed or attempt is terminal")
        self.budget.check()
        if self.budget.active is None:
            raise PermissionError("numerical work outside active cohort scope")

    def __call__(self, operation):
        self.check()
        if operation in self.caps:
            self.debit(operation)
        elif (operation not in ("reference_and_layout_build", "input_hash_verification")
                or self.budget.active != "runtime_input_binding"):
            raise PermissionError("undeclared scientific operation")

    def debit(self, operation):
        self.check()
        if operation not in self.caps or self.counts.get(operation, 0) >= self.caps[operation]:
            raise ValueError("cohort construction/load cap exceeded")
        phase = self.budget.active
        if operation == "parity_environment_build":
            allowed = phase == "same_start_preflight"
        elif operation == "episode_build":
            allowed = phase in self.packet["budget_plan"]["sections"] and (phase == "same_start_preflight"
                or phase.startswith(("window_ppo/", "cohort_ppo/", "bc_continue/", "final_evaluation/")))
        else:
            allowed = phase == "runtime_input_binding"
        if not allowed:
            raise PermissionError("cohort load/build outside its declared phase")
        count = self.counts.get(operation, 0) + 1
        write_json_once(self.root / f"launcher/operations/{operation}-{count:04d}.json", {
            "operation": operation, "count": count, "phase": phase,
            "clock": shared_monotonic(), "before_operation": True})
        self.counts[operation] = count

    def bind_campaign(self, root, config, streams, budget, backend):
        if (Path(root).resolve() != self.root or config != self.packet["scientific_config"]
                or streams != self.packet["streams"] or budget is not self.budget
                or type(backend) is not _backend_class() or backend.admit is not self):
            raise PermissionError("campaign does not match the cohort admission")


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
        admission = CohortAdmission(root, packet, auth, claim, budget)
        backend = _backend_class()(root, packet["scientific_config"], packet["streams"], admit_real_calls=admission)
        loaded = set()

        def initializer_loader(block):
            if str(block) not in packet["initializers"] or block in loaded:
                raise ValueError("initializer must be declared and loaded only once")
            loaded.add(block)
            record = packet["initializers"][str(block)]
            state = load_envelope(confined(root, record["path"]), record["sha256"], admission)
            view = restore_saved_policy(state, packet["original_config"], packet["initializer_streams"], block)
            view.steps = state["steps"]
            return view

        def final_lock_check():
            verify_bindings(root, packet)
            if admission.counts != admission.caps:
                raise ValueError("incomplete cohort load/build matrix")

        from src.rl.cohort_campaign import CohortCampaign
        campaign = CohortCampaign(output, packet["original_config"], packet["base_proposal_data"],
            packet["proposal_data"], packet["streams"], budget, backend,
            initializer_loader=initializer_loader, qualifications=packet["qualified_blocks"],
            enabled=True, engineering_only=False, execution_admission=admission, final_lock_check=final_lock_check)
        campaign.begin_next()
        seed_files = {str(Path(record["path"]).relative_to(Path(root).resolve()))
                      for record in packet["local_seed_collision_audit"]["files"]}
        paths = set(packet["source_files"]) | set(packet["input_files"]) | seed_files | {
            FROZEN, AUTHORIZATION, PROPOSAL, PROTOCOL, BASE_PROPOSAL, BOUND, APPROVAL_INTENT}
        for name in sorted(paths):
            budget.check()
            copy_verified(confined(root, name), output / "payload/locks/worktree" / name)
        write_json_once(output / "payload/locks/execution.json", {
            "head": claim["head"], "packet_sha256": packet["packet_sha256"], "authorization_sha256": digest(auth),
            "old_attempts_remain_terminal": True, "new_independent_budget": True,
            "objective_amendment_approved": True, "new_weights_or_search_authorized": False, "dropbox_exported": False})
        return campaign.run()
    except BaseException as error:
        if campaign is not None:
            campaign.fail(error)
        else:
            write_json_once(output / "launcher/child-failure.json", {"error": repr(error), "automatic_retry": False})
            budget.close()
        return 1


def launch(root, started=None):
    started = shared_monotonic() if started is None else started
    packet, auth = approved(root)
    if shared_monotonic() - started >= packet["budget_plan"]["sections"]["runtime_input_binding"]["seconds"]:
        raise TimeoutError("cohort setup deadline before claim")
    output = confined(root, RUN)
    output.mkdir(parents=True, exist_ok=False)
    launcher = output / "launcher"
    write_json_once(launcher / "claim.json", {"pid": os.getpid(), "ppid": os.getppid(),
        "head": git(root, "rev-parse", "HEAD"), "packet_sha256": packet["packet_sha256"],
        "authorization_sha256": digest(auth), "started": started, "clock_id": CLOCK_ID, "automatic_retry": False})
    try:
        result = supervise_time_baseline([sys.executable, "-m", "experiments.scripts.run_cohort_objective", "--child"],
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
