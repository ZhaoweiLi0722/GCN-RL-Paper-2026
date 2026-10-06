"""Committed one-attempt authority and detached supervision for family selection."""

from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import traceback

from src.rl import paired_cohort_execution as base
from src.rl.candidate_pilot_resources import audit_stream_collisions, digest
from src.rl.capacity_confirmation_execution import seed_files as prior_seed_files
from src.rl.capacity_family_design import PHASES, SCOPE, merged_proposal, numeric_contract, seeds, worlds
from src.rl.capacity_family_resources import CapacityFamilyBudget
from src.rl.capacity_native import keyed_seed
from src.rl.capacity_value_comparison_execution import configure_runtime
from src.utils.research_archive import create_archive, inventory
from src.utils.research_clock import CLOCK_ID, shared_monotonic

DIRECTORY = "specs/2026-10-06-family-selection"
PROPOSAL = "experiments/configs/capacity_family_selection_20261006.json"
PROTOCOL = DIRECTORY+"/protocol.md"
INTENT, FROZEN, AUTH = (DIRECTORY+"/"+n for n in ("approval-intent.json", "frozen.json", "authorization.json"))
RUN = "results/capacity_family_selection_20261006"
SCRIPT = "experiments.scripts.run_capacity_family_selection"


def source_locks(root):
    records = base.source_locks(root)
    required = ["src/rl/capacity_family_"+n+".py" for n in
                ("design", "resources", "learner", "value", "runner", "analysis", "execution")]
    required.append("experiments/scripts/run_capacity_family_selection.py")
    if not set(required).issubset(records):
        raise ValueError("commit complete implementation before freeze")
    return records


def seed_files(root):
    files = set(prior_seed_files(root))
    for directory in ("specs/2026-10-06-conditional-value-confirmation", "results/capacity_confirmation_20261006"):
        for path in (root/directory).rglob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl") and any(
                    w in path.name.lower() for w in ("seed", "stream", "manifest", "config", "frozen")):
                files.add(path)
    files.add(root/"experiments/configs/capacity_confirmation_20261006.json")
    return sorted(p for p in files if p.relative_to(root).as_posix() != PROPOSAL
                  and not p.relative_to(root).as_posix().startswith((DIRECTORY+"/", RUN+"/")))


def stream_manifest(p):
    study = p["family_study"]
    allocations, rows = [study["streams"]["bootstrap_seed"]], []
    for phase in PHASES:
        blocks = 3 if phase.startswith("development") else 5
        for block in range(blocks):
            if phase.endswith("training"):
                allocations.extend(seeds(study, phase, block))
            for world in worlds(study, phase, block):
                streams = {k: keyed_seed(p, world, k) for k in study["streams"]["purposes"]}
                rows.append(dict(world=world, substreams=streams))
                allocations.extend([world["seed"], *streams.values()])
    if len(set(allocations)) != len(allocations):
        raise ValueError("prospective seed allocation collision")
    return dict(rows=rows, allocations=allocations,
                intentional_sharing="same stage/block tapes and model/sampler seeds across families, learning rates and adjacency arms")


def input_locks(root, study):
    inherited = study["inherit"]
    result = {inherited["proposal"]: base.file_record(root, inherited["proposal"])}
    if result[inherited["proposal"]]["sha256"] != inherited["sha256"]:
        raise ValueError("inherited physical/reward configuration changed")
    for name, sha in zip(study["legacy"]["names"], study["legacy"]["sha256"]):
        path = study["legacy"]["root"]+"/"+name
        result[path] = base.file_record(root, path)
        if result[path]["sha256"] != sha:
            raise ValueError("predeclared historical reference differs")
        path = path.removesuffix(".pt")+".json"
        result[path] = base.file_record(root, path)
    return result


def freeze(root):
    root = Path(root).resolve()
    base._clean(root)
    study, intent = base.committed_json(root, PROPOSAL), base.committed_json(root, INTENT)
    proposal, protocol = base.file_record(root, PROPOSAL), base.file_record(root, PROTOCOL)
    if (study["scientific_execution_authorized"] is not False or intent.get("scope") != SCOPE
            or intent.get("scope_approved") is not True or not intent.get("user_literal")
            or intent.get("proposal_sha256") != proposal["sha256"]
            or intent.get("protocol_sha256") != protocol["sha256"]):
        raise PermissionError("one complete-package approval binding required")
    p = merged_proposal(base.committed_json(root, study["inherit"]["proposal"]), study)
    streams = stream_manifest(p)
    audit = audit_stream_collisions(streams["allocations"], seed_files(root))
    for row in audit["files"]+audit["collisions"]:
        row["path"] = Path(row["path"]).relative_to(root).as_posix()
    if not audit["passed"]:
        raise ValueError("historical seed collision; no silent reseed")
    packet = dict(format=SCOPE, workspace=str(root), branch=base.BRANCH,
        implementation_commit=base.git(root, "rev-parse", "HEAD"), result_root=RUN,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), proposal=proposal, protocol=protocol,
        intent=intent, intent_file=base.file_record(root, INTENT), inputs=input_locks(root, study),
        config=p, contract=numeric_contract(study), streams=streams, seed_audit=audit,
        source_files=source_locks(root), runtime=base.runtime_record())
    packet["packet_sha256"] = digest(packet)
    return packet


def authorization(packet):
    base._sealed_body(packet)
    if (packet["format"] != SCOPE or packet["intent"].get("scope_approved") is not True
            or packet["intent"].get("scope") != SCOPE or not packet["intent"].get("user_literal")
            or not packet["seed_audit"]["passed"]
            or packet["contract"] != numeric_contract(packet["config"]["family_study"])):
        raise PermissionError("invalid complete-package binding")
    return dict(format=SCOPE, scientific_execution_authorized=True,
        packet_sha256=packet["packet_sha256"], implementation_commit=packet["implementation_commit"],
        user_approval=packet["intent"], attempts=1, automatic_retry=False, automatic_resume=False,
        automatic_follow_on=False, remote_export=False)


def approved(root):
    root = Path(root).resolve()
    base._clean(root)
    packet, auth = base.committed_json(root, FROZEN), base.committed_json(root, AUTH)
    if (packet["workspace"] != str(root) or packet["branch"] != base.git(root, "branch", "--show-current")
            or auth != authorization(packet) or packet["source_files"] != source_locks(root)
            or packet["runtime"] != base.runtime_record()
            or packet["inputs"] != input_locks(root, packet["config"]["family_study"])):
        raise PermissionError("source/input/runtime/authority mismatch")
    for field, path in (("proposal", PROPOSAL), ("protocol", PROTOCOL), ("intent_file", INTENT)):
        if packet[field] != base.file_record(root, path):
            raise ValueError("locked protocol/config/authority changed")
    if [base.file_record(root, p.relative_to(root).as_posix()) for p in seed_files(root)] != packet["seed_audit"]["files"]:
        raise ValueError("seed declarations changed after freeze")
    subprocess.run(["git", "merge-base", "--is-ancestor", packet["implementation_commit"], "HEAD"], cwd=root, check=True)
    return packet, auth


def child(root):
    root = Path(root).resolve()
    packet, _ = approved(root)
    output, launcher, payload = root/RUN, root/RUN/"launcher", root/RUN/"payload"
    claim = json.loads((launcher/"claim.json").read_text())
    if claim["pid"] != os.getppid() or claim["packet_sha256"] != packet["packet_sha256"]:
        raise PermissionError("exclusive matching parent required")
    base.write_json_once(launcher/"child-claim.json", dict(pid=os.getpid(), ppid=os.getppid(), packet_sha256=packet["packet_sha256"]))
    budget = CapacityFamilyBudget(launcher, packet["config"], started=claim["started"])
    payload.mkdir(exist_ok=False)
    runner = None
    try:
        from src.rl.capacity_family_runner import CapacityFamilyRunner
        from src.rl.capacity_family_analysis import analyze
        runner = CapacityFamilyRunner(payload, packet["config"], budget,
            admission=dict(verified=True, scope=SCOPE), workspace=root)
        result = runner.run()
        comparison = analyze(payload, packet["config"]["family_study"], result["selection"])
        base.write_json_once(payload/"comparison.json", comparison)
        base.write_json_once(payload/"completion.json", dict(**result, packet_sha256=packet["packet_sha256"]))
        base.write_json_once(payload/"artifact-inventory.json", inventory(payload))
        budget.check(storage=True)
        receipt = create_archive(payload, output/"archives/payload.tar.gz")
        base.write_json_once(output/"archive-receipt.json", dict(archive=receipt, local_archive_verified=True,
            dropbox_exported=False, cloud_sync_verified=False, collaborator_access_verified=False))
        budget.check(storage=True)
        base.write_json_once(launcher/"child-completed.json", dict(budget=budget.snapshot(),
            payload_verified=True, packet_sha256=packet["packet_sha256"]))
        return 0
    except BaseException as error:
        base.write_json_once(launcher/"child-failure.json", dict(status="failed", error=repr(error),
            traceback=traceback.format_exc(), budget=budget.snapshot(), automatic_retry=False))
        if runner is not None:
            try:
                with gzip.open(payload/"failure-state.pkl.gz", "xb") as stream:
                    pickle.dump(runner.state_dict(), stream, protocol=5)
            except BaseException as saved:
                base.write_json_once(launcher/"failure-preservation-error.json", dict(error=repr(saved)))
        return 1
    finally:
        budget.close()


def launch(root):
    root, started = Path(root).resolve(), shared_monotonic()
    packet, auth = approved(root)
    caps = packet["config"]["proposed_budget"]["time_seconds"]
    if shared_monotonic()-started > caps["admission_lock_binding"]:
        raise TimeoutError("admission cap exhausted")
    output = root/RUN
    output.mkdir(parents=True, exist_ok=False)
    launcher = output/"launcher"
    base.write_json_once(launcher/"claim.json", dict(pid=os.getpid(), ppid=os.getppid(), started=started,
        head=base.git(root, "rev-parse", "HEAD"), packet_sha256=packet["packet_sha256"],
        authorization_sha256=digest(auth), clock_id=CLOCK_ID))
    owner, owner_started, times = "admission_lock_binding", started, {}
    job, job_started, job_consumed, job_cap = None, started, 0., 0.
    offset, process, failure_started = 0, None, None
    try:
        with (launcher/"stdout.log").open("xb") as stdout, (launcher/"stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable, "-m", SCRIPT, "--child"], cwd=root,
                stdout=stdout, stderr=stderr, start_new_session=True)
            while process.poll() is None:
                ledger = launcher/"budget.jsonl"
                if ledger.exists():
                    with ledger.open() as stream:
                        stream.seek(offset)
                        while True:
                            before = stream.tell()
                            line = stream.readline()
                            if not line or not line.endswith("\n"):
                                offset = before
                                break
                            row = json.loads(line)
                            if row["event"] == "enter":
                                owner, owner_started, times, job = row["owner"], row["clock"], row["owner_times"], None
                            elif row["event"] == "job_enter":
                                job, job_started, job_consumed, job_cap = row["job"], row["clock"], row["consumed"], row["cap"]
                now = shared_monotonic()
                if (launcher/"child-failure.json").exists() and failure_started is None:
                    failure_started = now
                if failure_started is not None:
                    owner, owner_started, times, job = "failure_flush_shutdown_reserve", failure_started, {}, None
                if (now-started > caps["global"] or times.get(owner, 0.)+now-owner_started > caps[owner]
                        or job is not None and job_consumed+now-job_started > job_cap):
                    base.write_json_once(launcher/"supervisor-overrun.json", dict(owner=owner, job=job,
                        elapsed=now-started, automatic_retry=False))
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                time.sleep(.2)
        failure = any((launcher/n).exists() for n in ("child-failure.json", "supervisor-overrun.json", "failure-preservation-error.json"))
        success = (not failure and process.returncode == 0 and (launcher/"stderr.log").stat().st_size == 0
                   and (launcher/"child-completed.json").is_file())
        if success:
            done = json.loads((launcher/"child-completed.json").read_text())
            success = (done["payload_verified"] and done["budget"]["counts"]["trajectories"] == 6744
                       and not done["budget"]["pending_chunks"])
        base.write_json_once(launcher/"terminal.json", dict(status="completed" if success else "failed",
            exit_code=0 if success else 1, child_exit_code=process.returncode,
            scientific_completion_verified=bool(success), elapsed=shared_monotonic()-started,
            packet_sha256=packet["packet_sha256"], automatic_retry=False))
        return 0 if success else 1
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        base.write_json_once(launcher/"launch-failure.json", dict(error=repr(error),
            traceback=traceback.format_exc(), automatic_retry=False))
        return 1
