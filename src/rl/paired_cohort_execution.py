"""Prospective, one-attempt paired-cohort entry. Import/preparation do no science.

Freeze and authorization builders return data, never approve a draft themselves.
Native work requires committed scope-specific approval, locked inputs/runtime,
an exclusive supervisor claim, and durable pre-operation receipts. Historical
attempts are inputs only, never resumed. No checkpoint is decoded by this module.
"""

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from src.rl.candidate_pilot_resources import audit_stream_collisions, digest
from src.rl.candidate_pilot_watchdog import LedgerDeadline
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, _combined, read_dynamic_ledger
from src.rl.paired_cohort_plan import branch_plan
from src.rl.paired_cohort_resources import budget_plan, inherited_metadata, stream_manifest
from src.utils.research_archive import copy_verified, create_archive, sha256_file
from src.utils.research_clock import CLOCK_ID, shared_monotonic


BRANCH = "codex/september-research-integration"
DIRECTORY = "specs/2026-10-02-paired-cohort-improvement"
PROPOSAL = "experiments/configs/paired_cohort_improvement_20261002.json"
PROTOCOL = DIRECTORY + "/protocol.md"
INTENT, FROZEN, AUTHORIZATION = (DIRECTORY + "/" + p for p in
                                  ("approval-intent.json", "frozen.json", "authorization.json"))
RUN = "results/paired_cohort_improvement_20261002"
INITIALIZER_PACKET = "specs/2026-10-02-terminal-obligation/frozen.json"
EVALUATION_PACKET = "specs/2026-10-02-cohort-evaluation-recovery2/frozen.json"
HISTORICAL_JSONS = {
    INITIALIZER_PACKET: "b7513d6afbcbe4315df8687fceac72c4bc7903cc6029cff0123569daad7dae5a",
    EVALUATION_PACKET: "a38c8d6ce68fd33f211f0ad77b818a3f63902be1ce1a06eb89a53df9edd262ec",
}
REQUIRED_SOURCE = ("src/rl/paired_cohort_execution.py", "src/rl/paired_cohort_campaign.py",
                   "src/rl/paired_cohort_backend.py", "src/rl/paired_cohort_binding.py",
                   "src/rl/paired_cohort_episode.py", "src/rl/paired_cohort_comparison.py",
                   "experiments/scripts/run_paired_cohort_improvement.py", "tests/test_paired_cohort_execution.py")


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def confined(root, name):
    root, path = Path(root).resolve(), Path(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ValueError("canonical worktree-relative path required")
    result = root / path
    if result.absolute() != result.resolve() or not result.resolve().is_relative_to(root):
        raise ValueError("symlinked or external execution path forbidden")
    return result


def file_record(root, name):
    """Same raw record schema as the recorder, without importing model owners."""
    path = confined(root, name)
    if not path.is_file():
        raise ValueError("evidence must be a regular non-symlink file")
    return dict(path=name, bytes=path.stat().st_size, sha256=sha256_file(path))


def write_json_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def committed_json(root, name):
    path = confined(root, name)
    if not path.is_file():
        raise PermissionError("missing committed approval/packet: " + name)
    raw = path.read_bytes()
    if raw != subprocess.check_output(["git", "show", "HEAD:" + name], cwd=root):
        raise ValueError("execution metadata must be committed unchanged")
    return json.loads(raw)


def runtime_record():
    # Lazy: unapproved metadata preparation also works without PyTorch installed.
    from src.rl.candidate_pilot_execution import runtime_record as record
    return record()


def source_locks(root):
    names = git(root, "ls-files", "src", "experiments/scripts", "tests", "AGENTS.md").splitlines()
    if not set(REQUIRED_SOURCE).issubset(names):
        raise ValueError("all native implementation files must be committed before freeze")
    return {name: file_record(root, name) for name in names if confined(root, name).is_file()}


def _scope(config):
    plan = budget_plan(config)
    if (config["schema"] != "paired-cohort-improvement-proposal-v1"
            or config["scientific_execution_authorized"] is not False
            or config["blocks"] != [60, 61, 62] or config["context_cohorts_per_block"] != 4
            or config["context_after_prefix_steps"] != [4, 20, 36]
            or config["enrollment_steps"] != 52 or config["economic_endpoint"] != 63
            or config["test_worlds_per_block"] != 12 or config["future_replications"] != 2
            or config["actor_updates_per_arm_block"] != 128 or config["critic_updates"] != 0
            or config["attempts"] != 1 or config["automatic_retry"] is not False
            or config["automatic_followon"] is not False
            or plan["limits"] != dict(trajectory=14553, clone=18765, actor=768, critic=0, seconds=14400)
            or {k: v["seconds"] for k, v in plan["phases"].items()} != dict(binding=300,
                same_start_preflight=600, reference_contexts=600, paired_branches=3600,
                paired_actor=1200, bc_actor=600, seal=120, evaluation=4320,
                raw_verification=600, archive=1200, closure=600)
            or config["per_owner_seconds"] != dict(preflight_block=200, context_block=200,
                branch_block=1200, paired_fit_block=400, bc_fit_block=200, evaluation_controller_block=240)):
        raise ValueError("unchanged prospective one-attempt numerical package required")
    return plan


def _sealed_body(packet, field="packet_sha256"):
    body = dict(packet)
    if digest({k: v for k, v in body.items() if k != field}) != body.get(field):
        raise ValueError("packet content digest differs")


def prepare(root):
    """Read the proposal and exactly two pinned historical JSONs; write nothing.

    Historical model files, old test outcomes, runtime configuration and seed
    inventory are not opened here. This returns native campaign metadata only.
    """
    root = Path(root).resolve()
    config = json.loads(confined(root, PROPOSAL).read_text())
    plan = _scope(config)
    old = {}
    records = {}
    for name, sha in HISTORICAL_JSONS.items():
        record = file_record(root, name)
        if record["sha256"] != sha:
            raise ValueError("pinned historical JSON changed: " + name)
        packet = json.loads(confined(root, name).read_text())
        _sealed_body(packet)
        old[name], records[name] = packet, record
    initializer, evaluation = old[INITIALIZER_PACKET], old[EVALUATION_PACKET]
    inherited = inherited_metadata(config, initializer, evaluation)
    if (set(initializer["qualified_blocks"]) != set(map(str, config["blocks"]))
            or any(v["passed"] is not True for v in initializer["qualified_blocks"].values())):
        raise ValueError("three already-qualified initializer metadata records required")
    inputs = {}
    reference = inherited["backend_config"]["reference"]
    for block in config["blocks"]:
        for kind in ("config", "policy"):
            name = reference["directory"] + "/" + reference[kind + "_template"].format(block=block)
            record = initializer["input_files"].get(name)
            if not record or record["path"] != name or record["sha256"] != reference["locks"][str(block)][kind]:
                raise ValueError("required historical reference input lock missing")
            inputs[name] = copy.deepcopy(record)
    for record in inherited["model_inputs"].values():
        if record["path"] in inputs and inputs[record["path"]] != record:
            raise ValueError("conflicting immutable input records")
        inputs[record["path"]] = copy.deepcopy(record)
    inputs.update(records)
    return dict(format="paired-cohort-preparation-v1", scientific_execution_authorized=False,
        ready_to_launch=False, source_frozen=False, workspace=str(root), branch=BRANCH, result_root=RUN,
        config=config, inherited=inherited, inherited_scientific_config=copy.deepcopy(initializer["scientific_config"]),
        initializer_config=copy.deepcopy(initializer["original_config"]),
        initializer_streams=copy.deepcopy(initializer["initializer_streams"]),
        qualified_blocks=copy.deepcopy(initializer["qualified_blocks"]),
        streams=stream_manifest(config, inherited["layout_seeds"]), budget_plan=plan,
        input_files=inputs, historical_jsons=records, new_checkpoint_loads=0,
        new_environment_calls=0, new_optimizer_calls=0, new_model_forwards=0)


def seed_inventory(root):
    """Local named seed metadata, including prior frozen packets, not raw episodes.

    Exclude ONLY this new specification/result directory. The new proposal has
    no allocated numeric seeds and remains part of the tracked config inventory.
    """
    root, files = Path(root).resolve(), set()
    for name in git(root, "ls-files", "experiments/configs").splitlines():
        path = confined(root, name)
        if path.name == ".gitkeep" and not path.read_bytes().strip():
            continue
        if path.suffix != ".json":
            raise ValueError("unhandled tracked seed configuration: " + name)
        files.add(path)
    for directory in ("results", "reports", "specs"):
        for path in (root / directory).rglob("*"):
            name = path.relative_to(root).as_posix()
            if name.startswith((DIRECTORY + "/", RUN + "/")) or not path.is_file():
                continue
            metadata = any(word in path.name.lower() for word in ("seed", "stream", "manifest", "config", "frozen"))
            frozen_spec = directory == "specs" and "frozen" in name.lower()
            if metadata or frozen_spec:
                if path.suffix in (".json", ".jsonl"):
                    files.add(confined(root, name))
                elif path.suffix in (".csv", ".yaml", ".yml", ".toml"):
                    raise ValueError("unhandled local seed declaration: " + name)
    files.update(confined(root, name) for name in HISTORICAL_JSONS)
    return sorted(files)


def _clean(root):
    if git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain"):
        raise PermissionError("clean committed integration branch required")


def _intent(intent, proposal, protocol):
    if (intent.get("approved") is not True or not isinstance(intent.get("user_literal"), str)
            or not intent["user_literal"].strip() or intent.get("proposal_sha256") != proposal["sha256"]
            or intent.get("protocol_sha256") != protocol["sha256"]):
        raise PermissionError("explicit committed scope-specific approval literal and proposal/protocol hashes required")
    return intent


def freeze(root):
    """Build (do not write) a frozen packet ONLY after committed explicit approval."""
    root = Path(root).resolve()
    _clean(root)
    proposal, protocol = file_record(root, PROPOSAL), file_record(root, PROTOCOL)
    intent = _intent(committed_json(root, INTENT), proposal, protocol)
    if committed_json(root, PROPOSAL)["scientific_execution_authorized"] is not False:
        raise PermissionError("proposal must remain an unapproved execution draft")
    committed_protocol = subprocess.check_output(["git", "show", "HEAD:" + PROTOCOL], cwd=root)
    if hashlib.sha256(committed_protocol).hexdigest() != protocol["sha256"]:
        raise ValueError("protocol must be committed unchanged before freeze")
    packet = prepare(root)
    for name in HISTORICAL_JSONS:
        committed_json(root, name)
    locks, head = source_locks(root), git(root, "rev-parse", "HEAD")
    for name, record in locks.items():
        raw = subprocess.check_output(["git", "show", head + ":" + name], cwd=root)
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("source bytes differ from implementation commit")
    for name, record in packet["input_files"].items():
        if file_record(root, name) != record:
            raise ValueError("pinned input bytes changed: " + name)
    # Immutable layout seeds are deliberately reused, not fresh allocations.
    audit = audit_stream_collisions(packet["streams"]["allocations"], seed_inventory(root))
    for record in audit["files"] + audit["collisions"]:
        record["path"] = Path(record["path"]).relative_to(root).as_posix()
    if audit["passed"] is not True:
        raise ValueError("new prospective seeds collide with local history")
    packet.update(format="paired-cohort-frozen-v1", source_frozen=True, implementation_commit=head,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), intent=intent, intent_file=file_record(root, INTENT),
        proposal=proposal, protocol=protocol, source_files=locks, runtime=runtime_record(),
        local_seed_collision_audit=audit,
        seed_audit_scope="local tracked configs and named seed/stream/manifest/config/frozen JSON/JSONL; only own spec/run excluded; no external freshness claim")
    packet["packet_sha256"] = digest(packet)
    return packet


def authorization(packet):
    """Generate the separate packet-bound authorization, never from a draft."""
    _sealed_body(packet)
    _scope(packet["config"])
    _intent(packet["intent"], packet["proposal"], packet["protocol"])
    if (packet["format"] != "paired-cohort-frozen-v1" or packet["source_frozen"] is not True
            or packet["scientific_execution_authorized"] is not False or packet["ready_to_launch"] is not False
            or packet["result_root"] != RUN or packet["branch"] != BRANCH
            or packet["local_seed_collision_audit"]["passed"] is not True
            or packet["local_seed_collision_audit"]["collisions"]
            or packet["budget_plan"] != budget_plan(packet["config"])):
        raise PermissionError("approved scope, frozen source/runtime and successful local seed audit required")
    return dict(format="paired-cohort-authorization-v1", approved=True, packet_sha256=packet["packet_sha256"],
        implementation_commit=packet["implementation_commit"], runtime_sha256=digest(packet["runtime"]),
        proposal_sha256=packet["proposal"]["sha256"], protocol_sha256=packet["protocol"]["sha256"],
        intent_sha256=packet["intent_file"]["sha256"], user_approval=copy.deepcopy(packet["intent"]),
        budget_plan_sha256=digest(packet["budget_plan"]), attempts=1,
        automatic_retry=False, automatic_followon=False, remote_or_dropbox_actions=False)


def verify_bindings(root, packet):
    _sealed_body(packet)
    root = Path(root).resolve()
    current = prepare(root)
    if any(packet[k] != v for k, v in current.items() if k not in ("format", "source_frozen")):
        raise ValueError("native preparation/config/input/stream metadata changed")
    if packet["source_files"] != source_locks(root) or packet["runtime"] != runtime_record():
        raise ValueError("source or runtime lock changed")
    for name, record in packet["input_files"].items():
        if file_record(root, name) != record:
            raise ValueError("pinned input changed: " + name)
    for key, name in (("proposal", PROPOSAL), ("protocol", PROTOCOL), ("intent_file", INTENT)):
        if file_record(root, name) != packet[key]:
            raise ValueError("scope/approval bytes changed")
    if committed_json(root, INTENT) != packet["intent"]:
        raise PermissionError("committed scope-specific approval differs")
    audit = packet["local_seed_collision_audit"]
    if (audit["manifest_sha256"] != digest(packet["streams"]["allocations"])
            or [file_record(root, p.relative_to(root).as_posix()) for p in seed_inventory(root)] != audit["files"]):
        raise ValueError("historical seed evidence changed since freeze")
    subprocess.run(["git", "merge-base", "--is-ancestor", packet["implementation_commit"], "HEAD"], cwd=root, check=True)
    subprocess.run(["git", "diff", "--exit-code", packet["implementation_commit"], "--",
                    "src", "experiments/scripts", "tests", "AGENTS.md"], cwd=root, check=True, stdout=subprocess.DEVNULL)


def approved(root):
    _clean(root)
    packet, auth = committed_json(root, FROZEN), committed_json(root, AUTHORIZATION)
    if packet["workspace"] != str(Path(root).resolve()) or auth != authorization(packet):
        raise PermissionError("committed workspace/packet-specific authorization required")
    verify_bindings(root, packet)
    return packet, auth


def operation_limits(config):
    """Per-owner caps for the parent's native operations, including no-cost builds."""
    sections = budget_plan(config)["sections"]
    result = {}
    for name in sections:
        phase = sections[name]["phase"]
        if phase == "binding":
            caps = dict(input_hash_verification=12, reference_and_layout_build=3,
                        reference_checkpoint_load=3, layout_environment_build=3, checkpoint_load=6)
        elif phase in ("same_start_preflight", "reference_contexts", "evaluation"):
            episodes = {"same_start_preflight": 1, "reference_contexts": 4, "evaluation": 12}[phase]
            caps = dict(episode_build=episodes, episode_dispatch=episodes, episode_step=63 * episodes)
            if phase == "same_start_preflight":
                caps.update(preflight_clone=1, clone_step=63)
        elif phase == "paired_branches":
            caps = dict(conditional_branch_clone=144, branch_step=6192)
        elif phase in ("paired_actor", "bc_actor"):
            caps = dict(actor_fork=1, actor_update=128)
        else:
            caps = {}
        result[name] = caps
    return result


class PairedBudget(DynamicCandidateBudget):
    """The new binding owner charges supervisor setup from the shared origin."""

    def begin(self, section):
        if section == "binding" and not self.closed and self.active is None:
            now = self.check()
            if now - self.started > self.plan["sections"][section]["seconds"]:
                self.failed = True
                raise TimeoutError("binding including supervisor setup exceeded cap")
            self._append(dict(event="begin", section=section, clock=self.started))
            self.active, self.section_started = section, self.started
            return
        super().begin(section)


class PairedAdmission:
    """Exact native campaign admission. Every job/operation is fsynced before work."""

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
            raise PermissionError("live exclusive native packet/claim/budget required")
        self.packet, self.claim, self.budget = packet, copy.deepcopy(claim), budget
        self.pid, self.failed, self.counts, self.by_section = os.getpid(), False, {}, {}
        self.limits = operation_limits(packet["config"])
        self.jobs, self.admitted_jobs, self.current = list(self.limits), [], None
        self.caps = {k: sum(c.get(k, 0) for c in self.limits.values()) for c in self.limits.values() for k in c}
        self.previous, self.sequence = "0" * 64, 0
        self.path = self.root / "launcher/admission.jsonl"
        self.handle = self.path.open("x", encoding="utf-8")
        self._append(dict(event="claim", packet_sha256=packet["packet_sha256"], authorization_sha256=digest(auth),
            started=budget.started, clock_id=CLOCK_ID, pid=self.pid, supervisor_pid=claim["pid"]))

    def _append(self, row):
        row = dict(row, sequence=self.sequence, previous=self.previous)
        row["sha256"] = digest(row)
        self.handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.previous, self.sequence = row["sha256"], self.sequence + 1

    def _deny(self, reason):
        self.failed = self.budget.failed = True
        self._append(dict(event="denied", reason=reason, clock=shared_monotonic()))
        raise PermissionError(reason)

    def check(self):
        if self.failed:
            raise PermissionError("admission failure is terminal; no retry")
        if (os.getpid() != self.pid or os.getppid() != self.claim["pid"]
                or json.loads((self.root / "launcher/claim.json").read_text()) != self.claim
                or any((self.root / "launcher" / name).exists() for name in
                       ("terminal.json", "child-failure.json", "launch-failure.json", "supervisor-overrun.json"))):
            self._deny("native owner changed or attempt is terminal")
        self.budget.check()
        if "binding" not in self.budget.closed and shared_monotonic() - self.budget.started > 300:
            self._deny("binding setup cap exhausted")

    def __call__(self, operation):
        self.check()
        if operation in self.jobs:
            if (self.budget.active is not None or self.budget.closed != self.admitted_jobs
                    or len(self.admitted_jobs) >= len(self.jobs) or self.jobs[len(self.admitted_jobs)] != operation):
                self._deny("serial job must be admitted once in order before budget.begin")
            self._append(dict(event="job", operation=operation, clock=shared_monotonic(), before_operation=True))
            self.admitted_jobs.append(operation)
            self.current = operation
            self.counts[operation] = 1
            return
        section = self.budget.active
        caps = self.limits.get(section, {})
        used = self.by_section.get(section, {}).get(operation, 0)
        if section != self.current or operation not in caps or used >= caps[operation]:
            self._deny("undeclared, out-of-phase or exhausted native operation: " + str(operation))
        total = self.counts.get(operation, 0) + 1
        self._append(dict(event="operation", operation=operation, section=section, count=total,
            section_count=used + 1, clock=shared_monotonic(), before_operation=True))
        self.counts[operation] = total
        self.by_section.setdefault(section, {})[operation] = used + 1

    def close(self):
        self.handle.close()


def read_admission(path, packet):
    """Independently recount the write-ahead operation chain without adopting it."""
    limits, counts, by_section, jobs = operation_limits(packet["config"]), {}, {}, []
    previous, current, last_clock = "0" * 64, None, None
    path = Path(path)
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError("redirected admission evidence")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("partial admission receipt")
    for i, line in enumerate(raw.splitlines()):
        row = json.loads(line)
        sha = row.pop("sha256")
        if row["sequence"] != i or row["previous"] != previous or digest(row) != sha:
            raise ValueError("admission hash chain differs")
        previous = sha
        if i == 0:
            if (row["event"] != "claim" or row["packet_sha256"] != packet["packet_sha256"]
                    or row["authorization_sha256"] != digest(authorization(packet)) or row["clock_id"] != CLOCK_ID):
                raise ValueError("admission claim differs")
            last_clock = row["started"]
            continue
        now = row["clock"]
        if type(now) not in (int, float) or not math.isfinite(now) or now < last_clock:
            raise ValueError("admission clock regressed")
        last_clock = now
        if row["event"] == "job":
            if len(jobs) >= len(limits) or row["operation"] != list(limits)[len(jobs)]:
                raise ValueError("admission serial job order differs")
            current = row["operation"]
            jobs.append(current)
            counts[current] = 1
        elif row["event"] == "operation":
            op, section = row["operation"], row["section"]
            value = by_section.get(section, {}).get(op, 0) + 1
            if (section != current or op not in limits[section] or value > limits[section][op]
                    or type(row["count"]) is not int or row["count"] != counts.get(op, 0) + 1
                    or type(row["section_count"]) is not int or row["section_count"] != value):
                raise ValueError("operation phase/count/cap differs")
            counts[op] = row["count"]
            by_section.setdefault(section, {})[op] = value
        else:
            raise ValueError("failed or unknown admission event")
        if row["before_operation"] is not True:
            raise ValueError("operation receipt was not prospective")
    return dict(counts=counts, by_section=by_section, jobs=jobs, sha256=previous)


def expected_completion(config, labels):
    """Deduplicated branch schedule from the verified context support, not 432."""
    body = copy.deepcopy(labels)
    seal = body.pop("dataset_sha256")
    if digest(body) != seal or body["format"] != "paired-cohort-label-dataset-v1" or body["config_sha256"] != digest(config):
        raise ValueError("complete independently verified label dataset required")
    counts = {}
    for row in body["labels"]:
        key = tuple(row[k] for k in ("block", "cohort", "after_prefix_steps"))
        keys = row["class_keys"]
        if (key in counts or keys != sorted(map(list, set(map(tuple, keys))))
                or keys != row["public_example"]["candidates"]["class_keys"]):
            raise ValueError("duplicate context or noncanonical label support")
        counts[key] = len(keys)
    branches = branch_plan(config, counts)
    expected_slots = {(b.block, b.cohort, b.after_prefix_steps, b.replication, b.candidate) for b in branches}
    actual = [tuple(r[k] for k in ("block", "cohort", "after_prefix_steps", "replication", "candidate_index"))
              for r in body["branch_outcomes"]]
    if len(actual) != len(expected_slots) or set(actual) != expected_slots:
        raise ValueError("branch outcome inventory differs from deduplicated support")
    operations = operation_limits(config)
    owner = {job: {k: 0 for k in ("trajectory", "clone", "actor", "critic")} for job in operations}
    for job, caps in operations.items():
        phase = job.split("/")[0]
        if phase == "paired_branches":
            block = int(job.split("block")[1])
            selected = [b for b in branches if b.block == block]
            caps["conditional_branch_clone"] = len(selected)
            caps["branch_step"] = sum(b.environment_calls for b in selected)
        owner[job].update(trajectory=caps.get("episode_step", 0),
            clone=caps.get("clone_step", 0) + caps.get("branch_step", 0), actor=caps.get("actor_update", 0))
    calls = sum(b.environment_calls for b in branches)
    return dict(operations=operations, owners=owner, conditional_branches=len(branches), branch_steps=calls,
                environment_calls=14742 + calls, optimizer_calls=768, critic_calls=0,
                contexts=36, new_fits=6, sealed_models=12, evaluation_cohorts=216)


def verify_completion(output, packet, *, require_closed=True, admission=None, campaign=None):
    """Compare exact per-owner durable spend with the actual deduplicated matrix."""
    output = Path(output)
    labels = json.loads((output / "payload/labels.json").read_text())
    expected = expected_completion(packet["config"], labels)
    ledger = read_dynamic_ledger(output / "launcher/budget.jsonl")
    operations = read_admission(output / "launcher/admission.jsonl", packet)
    jobs = list(operation_limits(packet["config"]))
    if ledger["plan_sha256"] != digest(packet["budget_plan"]):
        raise ValueError("final budget plan differs")
    if require_closed:
        if ledger["active"] is not None or ledger["closed"] != jobs or operations["jobs"] != jobs:
            raise ValueError("all serial scopes must close exactly once")
    elif (ledger["active"] not in ("raw_verification", "closure")
          or ledger["closed"] != jobs[:jobs.index(ledger["active"])]
          or operations["jobs"] != jobs[:jobs.index(ledger["active"]) + 1]):
        raise ValueError("final numerical check must occur at its declared serial boundary")
    want_counts = dict(environment=expected["environment_calls"], optimizer=768)
    if ledger["counts"] != want_counts:
        raise ValueError("total native environment/optimizer accounting differs")
    for job, owners in expected["owners"].items():
        if any(ledger["owner_counts"].get(f"section/{job}:{kind}", 0) != value for kind, value in owners.items()):
            raise ValueError("per-owner exact ledger differs: " + job)
        if operations["by_section"].get(job, {}) != expected["operations"][job]:
            raise ValueError("per-owner admitted operation inventory differs: " + job)
    if admission is not None and (operations["counts"] != admission.counts or operations["by_section"] != admission.by_section):
        raise ValueError("live admission differs from persisted receipts")
    config = packet["config"]
    models = {f"block{b}/{r}" for b in config["blocks"] for r in config["evaluation_controllers"][:4]}
    seals = json.loads((output / "payload/model-seals.json").read_text())
    if set(seals) != models or len({v["path"] for v in seals.values()}) != 12:
        raise ValueError("exact twelve separately sealed native models required")
    for record in seals.values():
        if file_record(output, record["path"]) != record:
            raise ValueError("sealed model bytes changed")
    fits = {f"block{b}/{arm}" for b in config["blocks"] for arm in config["training_arms"]}
    for name in fits:
        file_record(output, f"payload/models/{name}/final.pt")
    index = json.loads((output / "payload/evaluation-index.json").read_text())
    if len(index) != 216:
        raise ValueError("exact 216 fresh complete evaluations required")
    if campaign is not None and (len(campaign.contexts) != 36 or set(campaign.fitted) != fits
            or len(campaign.branch_indexes) != expected["conditional_branches"]
            or len(campaign.evaluation_indexes) != 216 or set(campaign.sequence.seals) != models):
        raise ValueError("native campaign inventory differs from persisted accounting")
    if require_closed:
        closure = json.loads((output / "launcher/closure.json").read_text())
        if (closure["complete"] is not True or closure["engineering_fixture"] is not False
                or any(closure[k] != expected[k] for k in ("evaluation_cohorts", "conditional_branches", "contexts", "new_fits"))
                or closure["counts"] != want_counts or closure["operations"] != operations["counts"]):
            raise ValueError("native closure receipt differs")
    return dict(format="paired-cohort-compute-verification-v1", **expected, counts=want_counts,
                admitted_counts=operations["counts"], admission_sha256=operations["sha256"],
                owner_counts=ledger["owner_counts"], phase_seconds=ledger["phase_seconds"],
                automatic_retry=False, unused_branch_capacity_reallocated=False)


class PairedLedgerDeadline(LedgerDeadline):
    """Native binding/closure names over the existing incremental hash reader."""

    def __init__(self, path, plan, started):
        super().__init__(path)
        self.plan, self.origin = copy.deepcopy(plan), started
        self.initial_deadline = started + plan["sections"]["binding"]["seconds"]
        self.phase_deadline = self.closure_deadline = None
        self.spent, self.extra_partial, self.owner, self.began = {}, b"", None, None

    def poll(self):
        offset = self.offset
        super().poll()
        if offset == self.offset:
            return
        with self.path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read(self.offset - offset)
        lines = (self.extra_partial + chunk).split(b"\n")
        self.extra_partial = lines.pop()
        for line in lines:
            row = json.loads(line)
            if row["event"] == "claim":
                sections = {k: _combined(v) | {"phase": v["phase"]} for k, v in self.plan["sections"].items()}
                phases = {k: {p: _combined(v)[k] for p, v in self.plan["phases"].items()} for k in ("environment", "optimizer")}
                if (row["dynamic_plan"] != self.plan or row["started"] != self.origin
                        or row["limits"] != _combined(self.plan["limits"]) or row["sections"] != sections
                        or row["phase_limits"] != phases):
                    raise ValueError("watchdog child plan/origin differs")
                continue
            now = row["clock"]
            if self.initial_deadline is not None and now > self.initial_deadline:
                raise TimeoutError("binding including setup exceeded deadline")
            if row["event"] == "begin":
                self.owner, self.began = row["section"], now
                phase = self.plan["sections"][self.owner]["phase"]
                self.phase_deadline = now + self.plan["phases"][phase]["seconds"] - self.spent.get(phase, 0.)
                if self.owner == "closure":
                    self.closure_deadline = min(self.phase_deadline, now + self.plan["sections"][self.owner]["seconds"])
            else:
                if self.phase_deadline is None or now > self.phase_deadline:
                    raise TimeoutError("aggregate paired phase deadline exceeded")
                if row["event"] == "finish":
                    phase = self.plan["sections"][self.owner]["phase"]
                    elapsed = now - self.began
                    total = self.spent.get(phase, 0.) + elapsed
                    if row["seconds"] != elapsed or row["phase_seconds"] != total:
                        raise ValueError("aggregate paired phase receipt differs")
                    self.spent[phase] = total
                    if self.owner == "binding":
                        self.initial_deadline = None
                    self.owner = self.began = self.phase_deadline = None

    def deadline(self, global_deadline):
        return min([super().deadline(global_deadline)] + [v for v in
            (self.initial_deadline, self.phase_deadline, self.closure_deadline) if v is not None])


def supervise_paired(command, *, cwd, launcher, plan, started, poll_seconds=.1, termination_grace_seconds=1.):
    """One owned process group; retain all deadline and failure receipts, no retry."""
    if (not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command)
            or type(started) not in (int, float) or not math.isfinite(started) or started < 0
            or not 0 < poll_seconds <= 1 or not 0 <= termination_grace_seconds <= 2):
        raise ValueError("explicit bounded supervisor arguments required")
    launcher = Path(launcher)
    watcher = PairedLedgerDeadline(launcher / "budget.jsonl", plan, started)
    paths = [launcher / name for name in ("stdout.log", "stderr.log", "supervisor.json")]
    if any(p.exists() or p.is_symlink() for p in [*paths, watcher.path, launcher / "supervisor-overrun.json"]):
        raise FileExistsError("fresh supervisor evidence required; no ledger adoption")
    deadline = started + plan["limits"]["seconds"]
    child_process, reason, error, forced = None, None, None, False
    launcher.mkdir(parents=True, exist_ok=True)
    with paths[0].open("xb") as stdout, paths[1].open("xb") as stderr:
        try:
            now = shared_monotonic()
            if now < started or now >= watcher.deadline(deadline):
                raise TimeoutError("binding deadline exhausted before child")
            child_process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True)
            while child_process.poll() is None:
                watcher.poll()
                if shared_monotonic() >= watcher.deadline(deadline):
                    reason = "wall_clock_deadline"
                    break
                time.sleep(poll_seconds)
            if reason is None:
                watcher.poll()
                reason = "child_exited" if child_process.returncode == 0 else "child_failed"
                if watcher.claim is None or watcher.partial or watcher.active is not None or set(watcher.closed) != set(plan["sections"]):
                    reason = "incomplete_budget_scopes"
                if shared_monotonic() > watcher.deadline(deadline):
                    reason = "wall_clock_deadline"
        except BaseException as exc:
            reason, error = "watchdog_or_launch_error", repr(exc)
        finally:
            if child_process is not None:
                if child_process.poll() is None:
                    try:
                        os.killpg(child_process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child_process.wait(timeout=termination_grace_seconds)
                    except subprocess.TimeoutExpired:
                        forced = True
                        try:
                            os.killpg(child_process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child_process.wait()
                else:
                    child_process.wait()
            for stream in (stdout, stderr):
                stream.flush()
                os.fsync(stream.fileno())
    result = dict(format="paired-cohort-supervisor-v1", command=command, cwd=str(cwd),
        pid=None if child_process is None else child_process.pid, ppid=os.getpid(),
        exit_code=None if child_process is None else child_process.returncode, reason=reason, error=error,
        forced_kill=forced, clock_id=CLOCK_ID, started=started, elapsed_seconds=shared_monotonic() - started,
        maximum_seconds=plan["limits"]["seconds"], poll_seconds=poll_seconds,
        termination_grace_seconds=termination_grace_seconds, final_deadline=watcher.deadline(deadline),
        closure_deadline=watcher.closure_deadline, last_ledger_sha256=watcher.previous,
        automatic_retry=False, passed=reason == "child_exited" and child_process is not None and child_process.returncode == 0)
    write_json_once(paths[2], result)
    if result["passed"] and shared_monotonic() > watcher.deadline(deadline):
        result.update(passed=False, reason="supervisor_receipt_exceeded_deadline")
        write_json_once(launcher / "supervisor-overrun.json", dict(status="failed", automatic_retry=False,
                        final_deadline=watcher.deadline(deadline)))
    return result


def child(root):
    packet, auth = approved(root)
    output = confined(root, RUN)
    claim = json.loads((output / "launcher/claim.json").read_text())
    if claim["head"] != git(root, "rev-parse", "HEAD"):
        raise PermissionError("execution commit changed since exclusive claim")
    write_json_once(output / "launcher/child-claim.json", dict(pid=os.getpid(), ppid=os.getppid()))
    budget = PairedBudget(output / "launcher/budget.jsonl", packet["budget_plan"], enabled=True, started=claim["started"])
    admission = None
    try:
        admission = PairedAdmission(root, packet, auth, claim, budget)
        from src.rl.paired_cohort_campaign import PairedCohortCampaign
        campaign = PairedCohortCampaign(root, output, packet, budget, admission)
        original_binder = campaign.binder

        def binder(*args, **kwargs):
            # Called within admitted binding, so lock copying uses its deadline.
            paths = set(packet["source_files"]) | set(packet["input_files"]) | {
                r["path"] for r in packet["local_seed_collision_audit"]["files"]} | {FROZEN, AUTHORIZATION, PROPOSAL, PROTOCOL, INTENT}
            for name in sorted(paths):
                budget.check()
                copy_verified(confined(root, name), output / "payload/locks/worktree" / name)
            write_json_once(output / "payload/locks/execution.json", dict(head=claim["head"],
                packet_sha256=packet["packet_sha256"], authorization_sha256=digest(auth),
                new_independent_attempt=True, no_old_attempt_retry=True, automatic_followon=False))
            return original_binder(*args, **kwargs)

        def final_check():
            verify_bindings(root, packet)
            report = verify_completion(output, packet, require_closed=False, admission=admission, campaign=campaign)
            if budget.active == "raw_verification":
                write_json_once(output / "payload/compute-accounting.json", report)

        campaign.binder, campaign.final_lock_check = binder, final_check
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


def read_terminal(output):
    """Failure receipts dominate a terminal, including late archive/receipt I/O."""
    launcher = Path(output) / "launcher"
    for name in ("launch-failure.json", "supervisor-overrun.json", "child-failure.json"):
        path = launcher / name
        if path.exists():
            return json.loads(path.read_text()) | dict(status="failed", scientific_completion_verified=False,
                                                      authoritative_source="launcher/" + name)
    return json.loads((launcher / "terminal.json").read_text())


def launch(root, started=None):
    started = shared_monotonic() if started is None else started
    packet, auth = approved(root)
    if shared_monotonic() - started >= 300:
        raise TimeoutError("binding cap exhausted before exclusive claim")
    output = confined(root, RUN)
    output.mkdir(parents=True, exist_ok=False)
    launcher = output / "launcher"
    write_json_once(launcher / "claim.json", dict(pid=os.getpid(), ppid=os.getppid(), head=git(root, "rev-parse", "HEAD"),
        packet_sha256=packet["packet_sha256"], authorization_sha256=digest(auth), started=started,
        clock_id=CLOCK_ID, automatic_retry=False))
    try:
        result = supervise_paired([sys.executable, "-m", "experiments.scripts.run_paired_cohort_improvement", "--child"],
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
            raise TimeoutError("launcher archive exceeded closure deadline")
        write_json_once(output / "closure-archive-receipt.json", dict(archive=receipt,
            local_archive_verified=True, dropbox_exported=False, cloud_sync_verified=False, howard_access_verified=False,
            archive_scope="launcher snapshot before final terminal receipt"))
        if shared_monotonic() > result["final_deadline"]:
            raise TimeoutError("closure archive receipt exceeded deadline")
        # The final terminal is outside the archived snapshot by construction.
        # A later failure receipt always overrides it, including its own fsync.
        write_json_once(launcher / "terminal.json", dict(status="completed", exit_code=0,
            scientific_completion_verified=True, supervisor=result,
            closure_archive_receipt=file_record(output, "closure-archive-receipt.json"),
            failure_override="launcher/launch-failure.json", automatic_retry=False, automatic_followon=False))
        if shared_monotonic() > result["final_deadline"]:
            raise TimeoutError("final terminal receipt exceeded closure deadline")
        return 0
    except BaseException as error:
        write_json_once(launcher / "launch-failure.json", dict(status="failed", error=repr(error),
            authoritative=True, overrides_terminal=True, scientific_completion_verified=False, automatic_retry=False))
        if not (launcher / "terminal.json").exists():
            write_json_once(launcher / "terminal.json", dict(status="failed", exit_code=1,
                scientific_completion_verified=False, automatic_retry=False, automatic_followon=False))
        return 1
