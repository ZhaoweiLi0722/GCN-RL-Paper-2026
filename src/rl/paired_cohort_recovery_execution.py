"""Additive, separately authorized remaining-work entry; never reopen the old run.

Preparation reads saved metadata/data only. Freeze/authorization return data;
native execution requires committed locks and an exclusive one-attempt claim.
"""

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from src.rl import paired_cohort_execution as original
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import read_dynamic_ledger
from src.rl.paired_cohort_recovery_resources import budget_plan, operation_limits
from src.utils.research_archive import copy_verified, create_archive
from src.utils.research_clock import CLOCK_ID, shared_monotonic


BRANCH = original.BRANCH
DIRECTORY = original.DIRECTORY + "/recovery1"
PROPOSAL = original.DIRECTORY + "/recovery-proposal.json"
PROTOCOL = DIRECTORY + "/protocol.md"
INTENT, FROZEN, AUTHORIZATION = (DIRECTORY + "/" + name for name in
                                ("approval-intent.json", "frozen.json", "authorization.json"))
RUN = "results/paired_cohort_improvement_20261002_recovery1"
ORIGINAL_PACKET = original.FROZEN
ORIGINAL_PACKET_SHA256 = "01012b0007e8553806e7c295f17761e54ec6a6e2ec2ba9d54997600e0de294ca"
PROPOSAL_SHA256 = "9a3d0fe2121de3b89b65f5607d75a49e959b125c5780204cacf3f6ac4a556638"
REQUIRED_SOURCE = tuple("src/rl/paired_cohort_recovery_" + name + ".py" for name in
    ("execution", "resources", "sequence", "data", "training", "campaign")) + (
    "experiments/scripts/run_paired_cohort_recovery.py",
    "tests/test_paired_cohort_recovery_execution.py", "tests/test_paired_cohort_recovery_resources.py",
    "tests/test_paired_cohort_recovery_sequence.py", "tests/test_paired_cohort_recovery_campaign.py",
    "tests/test_paired_cohort_recovery_native.py")
INHERITED_FIELDS = ("config", "inherited", "inherited_scientific_config", "initializer_config",
                    "initializer_streams", "qualified_blocks", "streams", "historical_jsons")

git, confined, file_record = original.git, original.confined, original.file_record
write_json_once, committed_json = original.write_json_once, original.committed_json
runtime_record, _clean = original.runtime_record, original._clean
PairedBudget, PairedLedgerDeadline = original.PairedBudget, original.PairedLedgerDeadline
supervise_paired, read_terminal = original.supervise_paired, original.read_terminal


def recovery_manifest(workspace):
    from src.rl.paired_cohort_recovery_data import recovery_manifest as build
    return build(workspace)


def verify_data_manifest(workspace, manifest):
    from src.rl.paired_cohort_recovery_data import verify_data_manifest as verify
    return verify(workspace, manifest)


def _metadata(root, manifest):
    root = Path(root).resolve()
    old_record, proposal_record = file_record(root, ORIGINAL_PACKET), file_record(root, PROPOSAL)
    if old_record["sha256"] != ORIGINAL_PACKET_SHA256 or proposal_record["sha256"] != PROPOSAL_SHA256:
        raise ValueError("pinned original packet or committed recovery proposal changed")
    old = json.loads(confined(root, ORIGINAL_PACKET).read_text())
    original._sealed_body(old)
    original.authorization(old)
    proposal = json.loads(confined(root, PROPOSAL).read_text())
    if old["workspace"] != str(root) or old["result_root"] != original.RUN:
        raise ValueError("original immutable packet belongs to another worktree/run")
    return dict(format="paired-cohort-recovery-preparation-v1", scientific_execution_authorized=False,
        ready_to_launch=False, source_frozen=False, workspace=str(root), branch=BRANCH, result_root=RUN,
        **{key: copy.deepcopy(old[key]) for key in INHERITED_FIELDS},
        recovery_proposal=proposal, recovery_manifest=copy.deepcopy(manifest),
        original_packet=old_record, original_packet_sha256=old["packet_sha256"],
        original_source_files=copy.deepcopy(old["source_files"]),
        input_files=copy.deepcopy(old["input_files"]) | {ORIGINAL_PACKET: old_record},
        budget_plan=budget_plan(old["config"], proposal),
        seed_reuse=dict(format="paired-cohort-intentional-seed-reuse-v1",
            original_streams_sha256=digest(old["streams"]), new_seed_allocations=0,
            fresh_independent_replication_claimed=False, interrupted_branch_same_seed=True),
        new_checkpoint_loads=0, new_model_forwards=0, new_environment_calls=0, new_optimizer_calls=0)


def prepare(root):
    """Saved-data preparation, no runtime setup, frozen file, or scientific calls."""
    packet = _metadata(root, None)
    packet["recovery_manifest"] = recovery_manifest(Path(root).resolve())
    validate_manifest(packet)
    return packet


def validate_manifest(packet):
    """Bind saved-data identity and remaining seeds without opening any tensor."""
    manifest = copy.deepcopy(packet["recovery_manifest"])
    sha = manifest.pop("manifest_sha256")
    cfg, streams = packet["config"], packet["streams"]
    if (digest(manifest) != sha or manifest["format"] != "paired-cohort-recovery-data-v1"
            or manifest["old_run"] != original.RUN or manifest["original_packet"] != packet["original_packet"]
            or manifest["old_environment_charge"] != 6246 or manifest["preserved_interrupted_charge"] != 1
            or manifest["new_environment_charge"] != 25655 or manifest["cumulative_environment_charge"] != 31901
            or manifest["original_attempt_remains_terminal"] is not True
            or len(manifest["contexts"]) != 36 or len(manifest["completed"]) != 117
            or len(manifest["remaining"]) != 285 or len(manifest["interrupted"]) != 3):
        raise ValueError("exact immutable saved-work manifest and consumed charges required")
    keys = {(b, c, t) for b in cfg["blocks"] for c in range(4) for t in (4, 20, 36)}
    if {tuple(row["key"]) for row in manifest["contexts"]} != keys:
        raise ValueError("all original 36 saved contexts required")
    completed = [row["branch_id"] for row in manifest["completed"]]
    remaining, slots = [], []
    counts = {str(b): dict(branches=0, environment_calls=0) for b in cfg["blocks"]}
    for row in manifest["remaining"]:
        b, c, t, rep, candidate = (row[k] for k in
            ("block", "cohort", "after_prefix_steps", "replication", "candidate_index"))
        key = f"block{b}/cohort{c}/after{t}"
        if (any(type(v) is not int for v in (b, c, t, rep, candidate)) or (b, c, t) not in keys
                or rep not in (0, 1) or not 0 <= candidate < 6
                or row["environment_calls"] != 63 - t
                or row["future_seed"] != streams["conditional_future"][key][rep]
                or row["branch_id"] != f"branches/{key}/rep{rep}/class{candidate}"):
            raise ValueError("remaining branch identity, original seed or horizon differs")
        slots.append((b, c, t, rep, candidate))
        remaining.append(row["branch_id"])
        counts[str(b)]["branches"] += 1
        counts[str(b)]["environment_calls"] += row["environment_calls"]
    wanted = {b: {key: value[key] for key in ("branches", "environment_calls")}
              for b, value in packet["recovery_proposal"]["remaining_by_block"].items()}
    if (slots != sorted(set(slots)) or slots[0] != (60, 3, 36, 0, 1)
            or len(set(completed + remaining)) != 402 or counts != wanted
            or manifest["remaining_by_block"] != wanted):
        raise ValueError("remaining schedule overlaps completed work or changes exact owners")


def source_locks(root):
    records = original.source_locks(root)
    if not set(REQUIRED_SOURCE).issubset(records):
        raise ValueError("all additive native recovery sources must be committed")
    return records


def _unchanged_original_sources(packet, current):
    if any(current.get(name) != record for name, record in packet["original_source_files"].items()):
        raise ValueError("an original locked source file changed; recovery must remain additive")


def _intent(intent, proposal, protocol):
    if (intent.get("format") != "paired-cohort-recovery-approval-intent-v1"
            or intent.get("approved") is not True or not isinstance(intent.get("user_literal"), str)
            or not intent["user_literal"].strip() or intent.get("proposal_sha256") != proposal["sha256"]
            or intent.get("protocol_sha256") != protocol["sha256"]
            or intent.get("original_attempt_remains_terminal") is not True):
        raise PermissionError("committed recovery-specific literal, terminal boundary and scope hashes required")
    return intent


def freeze(root):
    """Build only after committed contextual intent; do not write or launch."""
    root = Path(root).resolve()
    _clean(root)
    proposal, protocol = file_record(root, PROPOSAL), file_record(root, PROTOCOL)
    intent = _intent(committed_json(root, INTENT), proposal, protocol)
    committed_json(root, PROPOSAL)
    committed_json(root, ORIGINAL_PACKET)
    raw = subprocess.check_output(["git", "show", "HEAD:" + PROTOCOL], cwd=root)
    if hashlib.sha256(raw).hexdigest() != protocol["sha256"]:
        raise ValueError("recovery protocol must be committed unchanged")
    packet = prepare(root)
    locks, head = source_locks(root), git(root, "rev-parse", "HEAD")
    _unchanged_original_sources(packet, locks)
    for name, record in locks.items():
        raw = subprocess.check_output(["git", "show", head + ":" + name], cwd=root)
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("recovery implementation differs from committed source")
    for name, record in packet["input_files"].items():
        if file_record(root, name) != record:
            raise ValueError("immutable scientific input changed: " + name)
    packet.update(format="paired-cohort-recovery-frozen-v1", source_frozen=True,
        implementation_commit=head, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        proposal=proposal, protocol=protocol, intent=intent, intent_file=file_record(root, INTENT),
        source_files=locks, runtime=runtime_record())
    packet["packet_sha256"] = digest(packet)
    return packet


def authorization(packet):
    original._sealed_body(packet)
    _intent(packet["intent"], packet["proposal"], packet["protocol"])
    if (packet["format"] != "paired-cohort-recovery-frozen-v1" or packet["source_frozen"] is not True
            or packet["scientific_execution_authorized"] is not False or packet["ready_to_launch"] is not False
            or packet["result_root"] != RUN or packet["branch"] != BRANCH
            or packet["original_packet"]["path"] != ORIGINAL_PACKET
            or packet["original_packet"]["sha256"] != ORIGINAL_PACKET_SHA256
            or packet["proposal"]["sha256"] != PROPOSAL_SHA256
            or not isinstance(packet["recovery_manifest"], dict) or not packet["recovery_manifest"]
            or packet["seed_reuse"] != dict(format="paired-cohort-intentional-seed-reuse-v1",
                original_streams_sha256=digest(packet["streams"]), new_seed_allocations=0,
                fresh_independent_replication_claimed=False, interrupted_branch_same_seed=True)
            or packet["budget_plan"] != budget_plan(packet["config"], packet["recovery_proposal"])):
        raise PermissionError("frozen exact recovery scope, original packet and intentional seed reuse required")
    validate_manifest(packet)
    _unchanged_original_sources(packet, packet["source_files"])
    return dict(format="paired-cohort-recovery-authorization-v1", approved=True,
        packet_sha256=packet["packet_sha256"], implementation_commit=packet["implementation_commit"],
        runtime_sha256=digest(packet["runtime"]), proposal_sha256=packet["proposal"]["sha256"],
        protocol_sha256=packet["protocol"]["sha256"], intent_sha256=packet["intent_file"]["sha256"],
        user_approval=copy.deepcopy(packet["intent"]), original_packet=packet["original_packet"],
        recovery_manifest_sha256=digest(packet["recovery_manifest"]),
        budget_plan_sha256=digest(packet["budget_plan"]), attempts=1,
        automatic_retry=False, automatic_followon=False, remote_or_dropbox_actions=False)


def verify_bindings(root, packet):
    authorization(packet)
    root = Path(root).resolve()
    current = _metadata(root, packet["recovery_manifest"])
    if any(packet[key] != value for key, value in current.items() if key not in ("format", "source_frozen")):
        raise ValueError("original scientific metadata or additive recovery scope changed")
    locks = source_locks(root)
    _unchanged_original_sources(packet, locks)
    if packet["source_files"] != locks or packet["runtime"] != runtime_record():
        raise ValueError("recovery source/runtime changed")
    for name, record in packet["input_files"].items():
        if file_record(root, name) != record:
            raise ValueError("pinned recovery input changed: " + name)
    for key, name in (("proposal", PROPOSAL), ("protocol", PROTOCOL), ("intent_file", INTENT)):
        if file_record(root, name) != packet[key]:
            raise ValueError("committed recovery scope/intent bytes changed")
    if committed_json(root, INTENT) != packet["intent"]:
        raise PermissionError("committed contextual intent changed")
    verify_data_manifest(root, packet["recovery_manifest"])
    subprocess.run(["git", "merge-base", "--is-ancestor", packet["implementation_commit"], "HEAD"], cwd=root, check=True)
    subprocess.run(["git", "diff", "--exit-code", packet["implementation_commit"], "--",
        "src", "experiments/scripts", "tests", "AGENTS.md"], cwd=root, check=True, stdout=subprocess.DEVNULL)


def approved(root):
    _clean(root)
    packet, auth = committed_json(root, FROZEN), committed_json(root, AUTHORIZATION)
    if packet["workspace"] != str(Path(root).resolve()) or auth != authorization(packet):
        raise PermissionError("committed recovery packet and workspace authorization required")
    verify_bindings(root, packet)
    return packet, auth


def authorize(root):
    """Return authorization for an already committed packet; never write it."""
    _clean(root)
    packet = committed_json(root, FROZEN)
    verify_bindings(root, packet)
    return authorization(packet)


class RecoveryAdmission(original.PairedAdmission):
    """Exact recovery type; reuse original prospective writes and failure latching."""

    def __init__(self, workspace, packet, auth, claim, budget):
        self.workspace = Path(workspace).resolve()
        self.root = confined(workspace, RUN)
        if (auth != authorization(packet) or packet["workspace"] != str(self.workspace)
                or claim["pid"] != os.getppid() or claim["packet_sha256"] != packet["packet_sha256"]
                or claim["authorization_sha256"] != digest(auth) or claim["clock_id"] != CLOCK_ID
                or claim["started"] != budget.started or budget.clock_id != CLOCK_ID
                or budget.plan != packet["budget_plan"] or budget.path.resolve() != self.root / "launcher/budget.jsonl"
                or json.loads((self.root / "launcher/claim.json").read_text()) != claim
                or budget.active is not None or budget.closed):
            raise PermissionError("live exclusive recovery packet/claim/budget required")
        self.packet, self.claim, self.budget = packet, copy.deepcopy(claim), budget
        self.pid, self.failed, self.counts, self.by_section = os.getpid(), False, {}, {}
        self.limits = operation_limits(packet["config"], packet["recovery_proposal"])
        self.jobs, self.admitted_jobs, self.current = list(self.limits), [], None
        self.caps = {key: sum(c.get(key, 0) for c in self.limits.values()) for c in self.limits.values() for key in c}
        self.previous, self.sequence = "0" * 64, 0
        self.path = self.root / "launcher/admission.jsonl"
        self.handle = self.path.open("x", encoding="utf-8")
        self._append(dict(event="claim", packet_sha256=packet["packet_sha256"], authorization_sha256=digest(auth),
            started=budget.started, clock_id=CLOCK_ID, pid=self.pid, supervisor_pid=claim["pid"]))


def read_admission(path, packet):
    limits = operation_limits(packet["config"], packet["recovery_proposal"])
    counts, by_section, jobs = {}, {}, []
    previous, current, last_clock = "0" * 64, None, None
    path = Path(path)
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError("redirected recovery admission evidence")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("partial recovery admission receipt")
    for i, line in enumerate(raw.splitlines()):
        row = json.loads(line)
        sha = row.pop("sha256")
        if row["sequence"] != i or row["previous"] != previous or digest(row) != sha:
            raise ValueError("recovery admission hash chain differs")
        previous = sha
        if i == 0:
            if (row["event"] != "claim" or row["packet_sha256"] != packet["packet_sha256"]
                    or row["authorization_sha256"] != digest(authorization(packet)) or row["clock_id"] != CLOCK_ID):
                raise ValueError("recovery admission claim differs")
            last_clock = row["started"]
            if type(last_clock) not in (int, float) or not math.isfinite(last_clock) or last_clock < 0:
                raise ValueError("invalid recovery clock origin")
            continue
        now = row["clock"]
        if type(now) not in (int, float) or not math.isfinite(now) or now < last_clock:
            raise ValueError("recovery admission clock regressed")
        last_clock = now
        if row["event"] == "job":
            if len(jobs) >= len(limits) or row["operation"] != list(limits)[len(jobs)]:
                raise ValueError("recovery admission serial order differs")
            current = row["operation"]
            jobs.append(current)
            counts[current] = 1
        elif row["event"] == "operation":
            op, section = row["operation"], row["section"]
            value = by_section.get(section, {}).get(op, 0) + 1
            if (section != current or section not in limits or op not in limits[section]
                    or value > limits[section][op] or type(row["count"]) is not int
                    or row["count"] != counts.get(op, 0) + 1
                    or type(row["section_count"]) is not int or row["section_count"] != value):
                raise ValueError("recovery operation phase/count/cap differs")
            counts[op] = row["count"]
            by_section.setdefault(section, {})[op] = value
        else:
            raise ValueError("failed or unknown recovery admission event")
        if row["before_operation"] is not True:
            raise ValueError("recovery operation receipt was not prospective")
    return dict(counts=counts, by_section=by_section, jobs=jobs, sha256=previous)


def expected_completion(packet, labels):
    validate_manifest(packet)
    complete = original.expected_completion(packet["config"], labels)
    if complete["conditional_branches"] != 402 or complete["branch_steps"] != 17158:
        raise ValueError("complete reused plus new canonical branch matrix required")
    counts = {tuple(row[k] for k in ("block", "cohort", "after_prefix_steps")): len(row["class_keys"])
              for row in labels["labels"]}
    branches = original.branch_plan(packet["config"], counts)
    ordered = [f"branches/block{b.block}/cohort{b.cohort}/after{b.after_prefix_steps}/rep{b.replication}/class{b.candidate}"
               for b in branches]
    manifest = packet["recovery_manifest"]
    if ordered != [r["branch_id"] for r in manifest["completed"] + manifest["remaining"]]:
        raise ValueError("assembled canonical matrix differs from imported plus remaining schedule")
    operations = operation_limits(packet["config"], packet["recovery_proposal"])
    plan = budget_plan(packet["config"], packet["recovery_proposal"])
    owners = {name: {key: row[key] for key in ("trajectory", "clone", "actor", "critic")}
              for name, row in plan["sections"].items()}
    return dict(operations=operations, owners=owners, conditional_branches=402, branch_steps=17158,
        imported_branches=117, new_branches=285, new_branch_steps=12047,
        environment_calls=25655, optimizer_calls=768, critic_calls=0, contexts=36,
        new_fits=6, sealed_models=12, evaluation_cohorts=216, old_environment_calls=6246,
        old_partial_debit_preserved=1, cumulative_environment_calls=31901)


def verify_completion(output, packet, *, require_closed=True, admission=None, campaign=None):
    output = Path(output)
    labels = json.loads((output / "payload/labels.json").read_text())
    expected = expected_completion(packet, labels)
    ledger = read_dynamic_ledger(output / "launcher/budget.jsonl")
    operations = read_admission(output / "launcher/admission.jsonl", packet)
    jobs = list(expected["operations"])
    if ledger["plan_sha256"] != digest(packet["budget_plan"]):
        raise ValueError("recovery ledger plan differs")
    if require_closed:
        if ledger["active"] is not None or ledger["closed"] != jobs or operations["jobs"] != jobs:
            raise ValueError("all recovery serial jobs must close exactly once")
    elif (ledger["active"] not in ("raw_verification", "closure")
            or ledger["closed"] != jobs[:jobs.index(ledger["active"])]
            or operations["jobs"] != jobs[:jobs.index(ledger["active"]) + 1]):
        raise ValueError("recovery numerical verification at wrong serial boundary")
    want_counts = dict(environment=25655, optimizer=768)
    if ledger["counts"] != want_counts:
        raise ValueError("exact new-work debits required; consumed old calls cannot be refunded or adopted")
    for job, owners in expected["owners"].items():
        if any(ledger["owner_counts"].get(f"section/{job}:{key}", 0) != value for key, value in owners.items()):
            raise ValueError("recovery per-owner ledger differs: " + job)
        if job in operations["jobs"] and operations["by_section"].get(job, {}) != expected["operations"][job]:
            raise ValueError("recovery per-owner admitted operation inventory differs: " + job)
    if admission is not None and (operations["counts"] != admission.counts or operations["by_section"] != admission.by_section):
        raise ValueError("live recovery admission differs from durable receipts")
    cfg = packet["config"]
    models = {f"block{b}/{role}" for b in cfg["blocks"] for role in cfg["evaluation_controllers"][:4]}
    seals = json.loads((output / "payload/model-seals.json").read_text())
    if set(seals) != models or len({r["path"] for r in seals.values()}) != 12:
        raise ValueError("twelve separately sealed recovery models required")
    for record in seals.values():
        if file_record(output, record["path"]) != record:
            raise ValueError("recovery model seal bytes changed")
    fits = {f"block{b}/{arm}" for b in cfg["blocks"] for arm in cfg["training_arms"]}
    for name in fits:
        file_record(output, f"payload/models/{name}/final.pt")
    if len(json.loads((output / "payload/evaluation-index.json").read_text())) != 216:
        raise ValueError("216 complete unchanged evaluation worlds required")
    if campaign is not None and (len(campaign.contexts) != 36 or set(campaign.fitted) != fits
            or len(campaign.branch_indexes) != 402 or len(campaign.evaluation_indexes) != 216
            or set(campaign.sequence.seals) != models):
        raise ValueError("recovery live campaign inventory differs")
    if require_closed:
        closure = json.loads((output / "launcher/closure.json").read_text())
        if (closure["complete"] is not True or closure["engineering_fixture"] is not False
                or any(closure[key] != expected[key] for key in ("evaluation_cohorts", "conditional_branches", "contexts", "new_fits"))
                or closure["counts"] != want_counts or closure["operations"] != operations["counts"]):
            raise ValueError("recovery closure receipt differs")
    return dict(format="paired-cohort-recovery-compute-verification-v1", **expected,
        counts=want_counts, admitted_counts=operations["counts"], admission_sha256=operations["sha256"],
        owner_counts=ledger["owner_counts"], phase_seconds=ledger["phase_seconds"],
        automatic_retry=False, old_attempt_reopened=False, refunded_calls=0)


def child(root):
    packet, auth = approved(root)
    output = confined(root, RUN)
    claim = json.loads((output / "launcher/claim.json").read_text())
    if claim["head"] != git(root, "rev-parse", "HEAD"):
        raise PermissionError("recovery execution commit changed after claim")
    write_json_once(output / "launcher/child-claim.json", dict(pid=os.getpid(), ppid=os.getppid()))
    budget = PairedBudget(output / "launcher/budget.jsonl", packet["budget_plan"], enabled=True, started=claim["started"])
    admission = None
    try:
        admission = RecoveryAdmission(root, packet, auth, claim, budget)
        from src.rl.paired_cohort_recovery_campaign import PairedCohortRecoveryCampaign

        def final_check():
            verify_bindings(root, packet)
            report = verify_completion(output, packet, require_closed=False, admission=admission, campaign=campaign)
            if budget.active == "raw_verification":
                write_json_once(output / "payload/compute-accounting.json", report)

        campaign = PairedCohortRecoveryCampaign(root, output, packet, budget, admission, final_lock_check=final_check)
        original_binder = campaign.binder

        def binder(*args, **kwargs):
            paths = set(packet["source_files"]) | set(packet["input_files"]) | {
                FROZEN, AUTHORIZATION, PROPOSAL, PROTOCOL, INTENT}
            for name in sorted(paths):
                budget.check()
                copy_verified(confined(root, name), output / "payload/locks/worktree" / name)
            write_json_once(output / "payload/locks/recovery-manifest.json", packet["recovery_manifest"])
            write_json_once(output / "payload/locks/execution.json", dict(head=claim["head"],
                packet_sha256=packet["packet_sha256"], authorization_sha256=digest(auth),
                original_packet=packet["original_packet"], old_attempt_reopened=False,
                intentional_seed_reuse=True, automatic_followon=False))
            return original_binder(*args, **kwargs)

        campaign.binder = binder
        campaign.run()
        verify_completion(output, packet, admission=admission, campaign=campaign)
        return 0
    except BaseException as error:
        write_json_once(output / "launcher/child-failure.json", dict(error=repr(error), automatic_retry=False))
        return 1
    finally:
        if admission is not None:
            admission.close()
        budget.close()


def launch(root, started=None):
    started = shared_monotonic() if started is None else started
    packet, auth = approved(root)
    if shared_monotonic() - started >= 300:
        raise TimeoutError("recovery binding cap exhausted before exclusive claim")
    output = confined(root, RUN)
    output.mkdir(parents=True, exist_ok=False)
    launcher = output / "launcher"
    write_json_once(launcher / "claim.json", dict(pid=os.getpid(), ppid=os.getppid(), head=git(root, "rev-parse", "HEAD"),
        packet_sha256=packet["packet_sha256"], authorization_sha256=digest(auth), started=started,
        clock_id=CLOCK_ID, automatic_retry=False))
    try:
        result = supervise_paired([sys.executable, "-m", "experiments.scripts.run_paired_cohort_recovery", "--child"],
            cwd=root, launcher=launcher, plan=packet["budget_plan"], started=started)
        success = result["passed"] and (launcher / "stderr.log").stat().st_size == 0
        if success:
            verify_completion(output, packet)
        success = bool(success and shared_monotonic() <= result["final_deadline"])
        if not success:
            write_json_once(launcher / "terminal.json", dict(status="failed", exit_code=1,
                scientific_completion_verified=False, supervisor=result, automatic_retry=False, automatic_followon=False))
            return 1
        write_json_once(launcher / "archive-pending.json", dict(status="pending_archive",
            scientific_completion_verified=False, supervisor=result, automatic_retry=False))
        receipt = create_archive(launcher, output / "archives/completed-launcher.tar.gz")
        if shared_monotonic() > result["final_deadline"]:
            raise TimeoutError("recovery launcher archive exceeded closure deadline")
        write_json_once(output / "closure-archive-receipt.json", dict(archive=receipt,
            local_archive_verified=True, dropbox_exported=False, cloud_sync_verified=False,
            archive_scope="launcher snapshot before final terminal receipt"))
        if shared_monotonic() > result["final_deadline"]:
            raise TimeoutError("recovery closure archive receipt exceeded deadline")
        write_json_once(launcher / "terminal.json", dict(status="completed", exit_code=0,
            scientific_completion_verified=True, supervisor=result,
            closure_archive_receipt=file_record(output, "closure-archive-receipt.json"),
            failure_override="launcher/launch-failure.json", automatic_retry=False, automatic_followon=False))
        if shared_monotonic() > result["final_deadline"]:
            raise TimeoutError("recovery terminal receipt exceeded deadline")
        return 0
    except BaseException as error:
        write_json_once(launcher / "launch-failure.json", dict(status="failed", error=repr(error),
            authoritative=True, overrides_terminal=True, scientific_completion_verified=False, automatic_retry=False))
        if not (launcher / "terminal.json").exists():
            write_json_once(launcher / "terminal.json", dict(status="failed", exit_code=1,
                scientific_completion_verified=False, automatic_retry=False, automatic_followon=False))
        return 1
