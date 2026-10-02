"""Prospective one-attempt entry; committed scope approval is mandatory.

Preparation is metadata-only. No launch, scientific checkpoint load or forward
occurs merely by importing this module or constructing a proposal.
"""

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from src.rl import paired_cohort_execution as base
from src.rl.candidate_pilot_resources import audit_stream_collisions, digest
from src.rl.conservative_cohort_binding import DIRECTORY, PROPOSAL, RUN, prepare
from src.rl.conservative_cohort_plan import budget_plan, schedule, validate_config
from src.rl.dynamic_candidate_resources import read_dynamic_ledger
from src.utils.research_archive import copy_verified
from src.utils.research_clock import CLOCK_ID, shared_monotonic


PROTOCOL = DIRECTORY + "/protocol.md"
INTENT, FROZEN, AUTHORIZATION = (DIRECTORY + "/" + name for name in
                                ("approval-intent.json", "frozen.json", "authorization.json"))
BRANCH = base.BRANCH
git, confined, file_record = base.git, base.confined, base.file_record
write_json_once, committed_json = base.write_json_once, base.committed_json
runtime_record, _clean = base.runtime_record, base._clean


def source_locks(root):
    records = base.source_locks(root)
    required = {"src/rl/conservative_cohort_" + name + ".py" for name in
        ("actor", "binding", "campaign", "collection", "comparison", "execution", "plan", "training", "verification")}
    required.add("experiments/scripts/run_conservative_cohort_improvement.py")
    if not required <= set(records):
        raise ValueError("complete additive source must be committed before freeze")
    return records


def seed_inventory(root):
    paths = [p for p in base.seed_inventory(root)
             if not p.relative_to(root).as_posix().startswith((DIRECTORY + "/", RUN + "/"))]
    # The reused allocator excluded the old original root; its immutable packet
    # must be included explicitly to cover those consumed seeds too.
    paths.append(confined(root, base.FROZEN))
    return sorted(set(paths))


def freeze(root):
    root = Path(root).resolve()
    _clean(root)
    proposal, protocol = file_record(root, PROPOSAL), file_record(root, PROTOCOL)
    intent = base._intent(committed_json(root, INTENT), proposal, protocol)
    committed_json(root, PROPOSAL)
    if subprocess.check_output(["git", "show", "HEAD:" + PROTOCOL], cwd=root) != confined(root, PROTOCOL).read_bytes():
        raise ValueError("protocol must be committed unchanged")
    packet, locks = prepare(root), source_locks(root)
    for name, record in packet["input_files"].items():
        if file_record(root, name) != record:
            raise ValueError("pinned initializer/reference input changed")
    audit = audit_stream_collisions(packet["streams"]["allocations"], seed_inventory(root))
    for record in audit["files"] + audit["collisions"]:
        record["path"] = Path(record["path"]).relative_to(root).as_posix()
    if not audit["passed"]:
        raise ValueError("prospective seeds collide with local prior evidence")
    packet.update(format="conservative-cohort-frozen-v1", source_frozen=True,
        implementation_commit=git(root, "rev-parse", "HEAD"),
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), proposal=proposal, protocol=protocol,
        intent=intent, intent_file=file_record(root, INTENT), source_files=locks,
        runtime=runtime_record(), local_seed_collision_audit=audit)
    packet["packet_sha256"] = digest(packet)
    return packet


def authorization(packet):
    base._sealed_body(packet)
    base._intent(packet["intent"], packet["proposal"], packet["protocol"])
    validate_config(packet["config"])
    if (packet["format"] != "conservative-cohort-frozen-v1" or packet["source_frozen"] is not True
            or packet["scientific_execution_authorized"] is not False or packet["ready_to_launch"] is not False
            or packet["branch"] != BRANCH or packet["result_root"] != RUN
            or packet["local_seed_collision_audit"]["passed"] is not True
            or packet["local_seed_collision_audit"]["collisions"]
            or packet["budget_plan"] != budget_plan(packet["config"])):
        raise PermissionError("exact approved single-attempt scope and frozen locks required")
    return dict(format="conservative-cohort-authorization-v1", approved=True,
        packet_sha256=packet["packet_sha256"], implementation_commit=packet["implementation_commit"],
        runtime_sha256=digest(packet["runtime"]), proposal_sha256=packet["proposal"]["sha256"],
        protocol_sha256=packet["protocol"]["sha256"], user_approval=packet["intent"],
        budget_plan_sha256=digest(packet["budget_plan"]), attempts=1, automatic_retry=False,
        automatic_followon=False, remote_or_dropbox_actions=False)


def verify_bindings(root, packet):
    authorization(packet)
    root = Path(root).resolve()
    actual = prepare(root)
    if any(packet[k] != v for k, v in actual.items() if k not in ("format", "source_frozen")):
        raise ValueError("prospective metadata/input scope changed")
    if source_locks(root) != packet["source_files"] or runtime_record() != packet["runtime"]:
        raise ValueError("source/runtime lock changed")
    for name, record in packet["input_files"].items():
        if file_record(root, name) != record:
            raise ValueError("scientific input bytes changed")
    for key, name in (("proposal", PROPOSAL), ("protocol", PROTOCOL), ("intent_file", INTENT)):
        if file_record(root, name) != packet[key]:
            raise ValueError("scope or approval bytes changed")
    if committed_json(root, INTENT) != packet["intent"]:
        raise PermissionError("scope approval is not committed unchanged")
    audit = packet["local_seed_collision_audit"]
    if (audit["manifest_sha256"] != digest(packet["streams"]["allocations"])
            or [file_record(root, p.relative_to(root).as_posix()) for p in seed_inventory(root)] != audit["files"]):
        raise ValueError("seed inventory changed since freeze")
    subprocess.run(["git", "merge-base", "--is-ancestor", packet["implementation_commit"], "HEAD"], cwd=root, check=True)
    subprocess.run(["git", "diff", "--exit-code", packet["implementation_commit"], "--", "src", "tests",
                    "experiments/scripts", "AGENTS.md"], cwd=root, check=True, stdout=subprocess.DEVNULL)


def approved(root):
    _clean(root)
    packet, auth = committed_json(root, FROZEN), committed_json(root, AUTHORIZATION)
    if packet["workspace"] != str(Path(root).resolve()) or auth != authorization(packet):
        raise PermissionError("committed workspace-bound execution authorization required")
    verify_bindings(root, packet)
    return packet, auth


def operation_limits(config):
    caps = dict(binding=dict(input_hash_verification=9, reference_and_layout_build=3,
        reference_checkpoint_load=3, layout_environment_build=3, checkpoint_load=3),
        preflight=dict(episode_dispatch=1, episode_build=1, episode_step=63, preflight_clone=1, clone_step=63),
        contexts=dict(episode_dispatch=2, episode_build=2, episode_step=126),
        branches=dict(conditional_branch_clone=144, branch_step=6192),
        paired_actor=dict(actor_fork=1, round_start_logits=6, actor_update=64),
        bc_actor=dict(actor_fork=1, round_start_logits=6, actor_update=64),
        evaluation=dict(episode_dispatch=12, episode_build=12, episode_step=756))
    return {job["id"]: dict(caps.get(job["phase"], {})) for job in schedule(config)}


class ConservativeAdmission(base.PairedAdmission):
    def __init__(self, workspace, packet, auth, claim, budget):
        self.workspace, self.root = Path(workspace).resolve(), confined(workspace, RUN)
        if (auth != authorization(packet) or packet["workspace"] != str(self.workspace)
                or claim["pid"] != os.getppid() or claim["packet_sha256"] != packet["packet_sha256"]
                or claim["authorization_sha256"] != digest(auth) or claim["clock_id"] != CLOCK_ID
                or claim["started"] != budget.started or budget.plan != packet["budget_plan"]
                or budget.path.resolve() != self.root / "launcher/budget.jsonl"
                or json.loads((self.root / "launcher/claim.json").read_text()) != claim
                or budget.active is not None or budget.closed):
            raise PermissionError("live exclusive new-attempt claim and budget required")
        self.packet, self.claim, self.budget = packet, copy.deepcopy(claim), budget
        self.pid, self.failed, self.counts, self.by_section = os.getpid(), False, {}, {}
        self.limits = operation_limits(packet["config"])
        self.jobs, self.admitted_jobs, self.current = list(self.limits), [], None
        self.previous, self.sequence = "0" * 64, 0
        self.path = self.root / "launcher/admission.jsonl"
        self.handle = self.path.open("x", encoding="utf-8")
        self._append(dict(event="claim", packet_sha256=packet["packet_sha256"], authorization_sha256=digest(auth),
            started=budget.started, clock_id=CLOCK_ID, pid=self.pid, supervisor_pid=claim["pid"]))


def child(root):
    packet, auth = approved(root)
    output = confined(root, RUN)
    claim = json.loads((output / "launcher/claim.json").read_text())
    if claim["head"] != git(root, "rev-parse", "HEAD"):
        raise PermissionError("execution commit changed")
    write_json_once(output / "launcher/child-claim.json", dict(pid=os.getpid(), ppid=os.getppid()))
    budget = base.PairedBudget(output / "launcher/budget.jsonl", packet["budget_plan"], enabled=True, started=claim["started"])
    admission = None
    try:
        admission = ConservativeAdmission(root, packet, auth, claim, budget)
        from src.rl.conservative_cohort_campaign import ConservativeCohortCampaign
        from src.rl.conservative_cohort_comparison import verify_bundle
        campaign = ConservativeCohortCampaign(root, output, packet, budget, admission, verifier=verify_bundle,
            final_lock_check=lambda: verify_bindings(root, packet))
        native_binder = campaign.binder
        def bind(*args, **kwargs):
            paths = set(packet["source_files"]) | set(packet["input_files"]) | {FROZEN, AUTHORIZATION, INTENT, PROPOSAL, PROTOCOL}
            for name in sorted(paths):
                budget.check()
                copy_verified(confined(root, name), output / "payload/locks/worktree" / name)
            write_json_once(output / "payload/locks/execution.json", dict(head=claim["head"],
                packet_sha256=packet["packet_sha256"], authorization_sha256=digest(auth)))
            return native_binder(*args, **kwargs)
        campaign.binder = bind
        campaign.run()
        return 0
    except BaseException as error:
        write_json_once(output / "launcher/child-failure.json", dict(error=repr(error), automatic_retry=False))
        return 1
    finally:
        if admission is not None:
            admission.close()
        budget.close()


def verify_completion(output, packet):
    ledger = read_dynamic_ledger(output / "launcher/budget.jsonl")
    closure = json.loads((output / "launcher/closure.json").read_text())
    expected_env = 12474 + sum(v["result"]["environment_calls"] for v in
        json.loads((output / "payload/branch-index.json").read_text()))
    if (ledger["closed"] != [j["id"] for j in schedule(packet["config"])] or ledger["active"] is not None
            or ledger["counts"] != dict(environment=expected_env, optimizer=768)
            or ledger["owner_counts"].get("global:critic", 0) != 0
            or closure["complete"] is not True or closure["engineering_fixture"] is not False
            or closure["models"] != 9 or closure["round_fits"] != 12
            or closure["evaluations"] != 180 or closure["contexts"] != 36):
        raise ValueError("complete numerical inventory/ledger differs")
    return ledger


def launch(root, started=None):
    started = shared_monotonic() if started is None else started
    packet, auth = approved(root)
    if shared_monotonic() - started >= 300:
        raise TimeoutError("binding setup cap exhausted")
    output = confined(root, RUN)
    output.mkdir(parents=True, exist_ok=False)
    launcher = output / "launcher"
    write_json_once(launcher / "claim.json", dict(pid=os.getpid(), ppid=os.getppid(), head=git(root, "rev-parse", "HEAD"),
        packet_sha256=packet["packet_sha256"], authorization_sha256=digest(auth), started=started,
        clock_id=CLOCK_ID, automatic_retry=False))
    try:
        result = base.supervise_paired([sys.executable, "-m", "experiments.scripts.run_conservative_cohort_improvement", "--child"],
            cwd=root, launcher=launcher, plan=packet["budget_plan"], started=started)
        success = result["passed"] and (launcher / "stderr.log").stat().st_size == 0
        if success:
            verify_completion(output, packet)
        success = bool(success and shared_monotonic() <= result["final_deadline"])
        write_json_once(launcher / "terminal.json", dict(status="completed" if success else "failed",
            exit_code=0 if success else 1, scientific_completion_verified=success,
            supervisor=result, automatic_retry=False, automatic_followon=False,
            final_terminal_outside_archive=True))
        if shared_monotonic() > result["final_deadline"]:
            raise TimeoutError("final terminal receipt exceeded watched closure deadline")
        return 0 if success else 1
    except BaseException as error:
        write_json_once(launcher / "launch-failure.json", dict(status="failed", error=repr(error),
            authoritative=True, overrides_terminal=True, automatic_retry=False))
        return 1
