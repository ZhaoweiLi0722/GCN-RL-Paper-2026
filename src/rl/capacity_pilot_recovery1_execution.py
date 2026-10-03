"""Separate one-shot precision correction, retaining the consumed pilot."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from src.rl import paired_cohort_execution as base
from src.rl.candidate_pilot_resources import audit_stream_collisions, digest
from src.rl.capacity_adaptation_campaign import numeric_contract, PHASES
from src.rl.capacity_native import keyed_seed
from src.rl.capacity_pilot_runner import worlds
from src.utils.research_archive import create_archive, inventory
from src.utils.research_clock import shared_monotonic, CLOCK_ID


BRANCH = "codex/september-research-integration"
ORIGINAL_DIRECTORY = "specs/2026-10-03-dynamic-capacity-adaptation"
DIRECTORY = ORIGINAL_DIRECTORY + "/recovery1"
PROPOSAL, PROTOCOL = ORIGINAL_DIRECTORY + "/pilot-proposal.json", ORIGINAL_DIRECTORY + "/pilot-proposal.md"
RECOVERY_PROTOCOL = DIRECTORY + "/protocol.md"
OLD_RUN = "results/dynamic_capacity_adaptation_20261003"
OLD_PACKET_SHA256 = "95b9e866e5419d6233a0059f811a368a825823c142f047ac3faa990f7f06e772"
INTENT, FROZEN, AUTH = (DIRECTORY + "/" + name for name in ("approval-intent.json", "frozen.json", "authorization.json"))
RUN = "results/dynamic_capacity_adaptation_20261003_recovery1"
EXPECTED_PROPOSAL = "f46f6d5ae417a67c5974516b1c9e2b5b05ba9e2de56681aaedbfe276e9f355a9"
EXPECTED_PROTOCOL = "4edcb1caeac4653dd27f9c2643bf562a276bbcd2c5e2d9a266e63d83d5075bc9"
SCRIPT = "experiments.scripts.run_dynamic_capacity_recovery1"


def source_locks(root):
    names = base.git(root, "ls-files", "src", "tests", "experiments/scripts", "AGENTS.md").splitlines()
    required = ["src/rl/capacity_pilot_recovery1_execution.py",
                "src/baselines/capacity_completion_control_recovery1.py", "src/rl/capacity_pilot_runner.py",
                "src/rl/capacity_pilot_analysis.py", "src/rl/capacity_native.py",
                "src/rl/capacity_ddpg_learner.py", "src/baselines/capacity_completion_control.py",
                "experiments/scripts/run_dynamic_capacity_recovery1.py"]
    if not set(required).issubset(names):
        raise ValueError("commit all real execution bindings first")
    return {name: base.file_record(root, name) for name in names}


def seed_files(root):
    paths = [p for p in base.seed_inventory(root) if not p.relative_to(root).as_posix().startswith((ORIGINAL_DIRECTORY+"/", OLD_RUN+"/", RUN+"/"))]
    # The reused inventory excludes its original attempt; retain those consumed seeds.
    paths += [base.confined(root, base.FROZEN)]
    return sorted(set(paths))


def stream_manifest(proposal):
    rows = []
    for phase in PHASES:
        for block in range(proposal["design"]["blocks"]):
            for world in worlds(proposal, phase, block):
                rows.append(dict(world=world, substreams={purpose: keyed_seed(proposal, world, purpose)
                    for purpose in proposal["design"]["substream_purposes"]}))
    # Only seed values enter collision matching, not dimensions/block indices.
    allocations = proposal["design"]["training_seeds"] + [proposal["analysis"]["bootstrap"]["seed"]]
    for row in rows:
        allocations += [row["world"]["seed"], *row["substreams"].values()]
    if len(allocations) != len(set(allocations)):
        raise ValueError("prospective random streams collide")
    return dict(rows=rows, allocations=allocations)


def freeze(root):
    root = Path(root).resolve()
    base._clean(root)
    if base.git(root, "branch", "--show-current") != BRANCH:
        raise PermissionError("wrong research branch")
    p = base.committed_json(root, PROPOSAL)
    intent = base.committed_json(root, INTENT)
    proposal, protocol = base.file_record(root, PROPOSAL), base.file_record(root, PROTOCOL)
    if (proposal["sha256"] != EXPECTED_PROPOSAL or protocol["sha256"] != EXPECTED_PROTOCOL
            or not intent["scope_approved"] or intent["proposal_sha256"] != proposal["sha256"]
            or intent["protocol_sha256"] != protocol["sha256"]):
        raise PermissionError("exact approved prospective scope required")
    previous = base.committed_json(root, ORIGINAL_DIRECTORY + "/frozen.json")
    base._sealed_body(previous)
    terminal = json.loads(base.confined(root, OLD_RUN + "/launcher/terminal.json").read_text())
    failure = json.loads(base.confined(root, OLD_RUN + "/launcher/child-failure.json").read_text())
    if (previous["packet_sha256"] != OLD_PACKET_SHA256 or terminal["status"] != "failed"
            or terminal["exit_code"] != 1 or failure["budget"]["counts"]["total_optimizer_steps"] != 0):
        raise PermissionError("expected consumed pre-fit attempt required")
    recovery_protocol = base.file_record(root, RECOVERY_PROTOCOL)
    if intent["recovery_protocol_sha256"] != recovery_protocol["sha256"]:
        raise PermissionError("correction protocol not approved")
    streams = stream_manifest(p)
    if streams != previous["streams"]:
        raise PermissionError("same-design correction must retain original streams")
    prior_records = {name: base.file_record(root, name) for name in (
        ORIGINAL_DIRECTORY + "/frozen.json", OLD_RUN + "/launcher/terminal.json",
        OLD_RUN + "/launcher/child-failure.json")}
    audit = audit_stream_collisions(streams["allocations"], seed_files(root))
    for entry in audit["files"] + audit["collisions"]:
        entry["path"] = Path(entry["path"]).relative_to(root).as_posix()
    if not audit["passed"]:
        raise ValueError("new stream collision in local metadata")
    packet = dict(format="dynamic-capacity-recovery1-frozen-v1", workspace=str(root), branch=BRANCH,
        implementation_commit=base.git(root,"rev-parse","HEAD"), result_root=RUN,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), proposal=proposal, protocol=protocol,
        intent=intent, intent_file=base.file_record(root, INTENT), config=p, contract=numeric_contract(p),
        recovery_protocol=recovery_protocol, prior_attempt=prior_records, reused_streams_from_packet=OLD_PACKET_SHA256,
        streams=streams, seed_audit=audit, source_files=source_locks(root), runtime=base.runtime_record())
    packet["packet_sha256"] = digest(packet)
    return packet


def authorization(packet):
    base._sealed_body(packet)
    if (packet["format"] != "dynamic-capacity-recovery1-frozen-v1" or packet["proposal"]["sha256"] != EXPECTED_PROPOSAL
            or packet["protocol"]["sha256"] != EXPECTED_PROTOCOL or not packet["intent"]["scope_approved"]
            or not packet["seed_audit"]["passed"] or packet["seed_audit"]["collisions"]
            or packet["contract"] != numeric_contract(packet["config"])
            or packet["reused_streams_from_packet"] != OLD_PACKET_SHA256
            or packet["recovery_protocol"]["sha256"] != packet["intent"]["recovery_protocol_sha256"]):
        raise PermissionError("unapproved or unfrozen capacity execution")
    return dict(format="dynamic-capacity-recovery1-effective-authorization-v1", scientific_execution_authorized=True,
        packet_sha256=packet["packet_sha256"], implementation_commit=packet["implementation_commit"],
        proposal_sha256=EXPECTED_PROPOSAL, protocol_sha256=EXPECTED_PROTOCOL, user_approval=packet["intent"],
        recovery_protocol_sha256=packet["recovery_protocol"]["sha256"], prior_attempt_packet=OLD_PACKET_SHA256,
        limits=packet["contract"], runtime_sha256=digest(packet["runtime"]), attempts=1,
        automatic_retry=False, automatic_resume=False, automatic_followon=False, remote_export=False)


def approved(root):
    root = Path(root).resolve()
    base._clean(root)
    packet, auth = base.committed_json(root,FROZEN), base.committed_json(root,AUTH)
    if (packet["workspace"] != str(root) or base.git(root,"branch","--show-current") != BRANCH
            or auth != authorization(packet) or packet["source_files"] != source_locks(root)
            or packet["runtime"] != base.runtime_record()):
        raise PermissionError("execution/source/runtime/authorization mismatch")
    for field, name in (("proposal",PROPOSAL),("protocol",PROTOCOL),("intent_file",INTENT)):
        if base.file_record(root,name) != packet[field]:
            raise ValueError("scientific input bytes changed")
    if base.file_record(root, RECOVERY_PROTOCOL) != packet["recovery_protocol"]:
        raise ValueError("correction protocol changed")
    if any(base.file_record(root, name) != record for name, record in packet["prior_attempt"].items()):
        raise ValueError("prior failure evidence changed")
    records = [base.file_record(root,p.relative_to(root).as_posix()) for p in seed_files(root)]
    if records != packet["seed_audit"]["files"]:
        raise ValueError("local seed metadata changed after freeze")
    subprocess.run(["git","merge-base","--is-ancestor",packet["implementation_commit"],"HEAD"],cwd=root,check=True)
    return packet, auth


def child(root):
    root = Path(root).resolve()
    packet, _ = approved(root)
    output = base.confined(root,RUN)
    launcher = output/"launcher"
    claim = json.loads((launcher/"claim.json").read_text())
    if claim["pid"] != os.getppid() or claim["packet_sha256"] != packet["packet_sha256"]:
        raise PermissionError("child is not owned by this exclusive supervisor")
    base.write_json_once(launcher/"child-claim.json",dict(pid=os.getpid(),ppid=os.getppid(),packet_sha256=packet["packet_sha256"]))
    from src.rl.capacity_pilot_resources import CapacityPilotBudget
    import torch
    torch.set_num_threads(packet["config"]["proposed_budget"]["compute_threads"])
    torch.set_num_interop_threads(1)
    budget = CapacityPilotBudget(launcher,packet["config"],started=claim["started"])
    runner = None
    payload = output/"payload"
    payload.mkdir(exist_ok=False)
    try:
        budget.check(storage=True)
        runner = build_runner(payload,packet["config"],budget,
            admission=dict(verified=True,scope="dynamic-capacity-pilot-v1",packet_sha256=packet["packet_sha256"]))
        result = runner.run()
        budget.enter("analysis_archive_verification",None)
        from src.rl.capacity_pilot_analysis import run_analysis
        analysis = run_analysis(payload,packet["config"],require_complete=True)
        base.write_json_once(payload/"comparison.json",analysis)
        base.write_json_once(payload/"completion.json",dict(**result,packet_sha256=packet["packet_sha256"]))
        base.write_json_once(payload/"artifact-inventory.json",inventory(payload))
        budget.check(storage=True)
        receipt = create_archive(payload,output/"archives/payload.tar.gz")
        base.write_json_once(output/"archive-receipt.json",dict(archive=receipt,local_archive_verified=True,
            dropbox_exported=False,cloud_sync_verified=False,howard_access_verified=False))
        budget.check(storage=True)
        base.write_json_once(launcher/"child-completed.json",dict(budget=budget.snapshot(),payload_verified=True,
            packet_sha256=packet["packet_sha256"]))
        budget.check(storage=True)
        return 0
    except BaseException as error:
        evidence = dict(status="failed",error=repr(error),traceback=traceback.format_exc(),
                        budget=budget.snapshot(),automatic_retry=False)
        # Preservation may not execute science or clear a consumed failure latch.
        base.write_json_once(launcher/"child-failure.json",evidence)
        if hasattr(error,"report"):
            base.write_json_once(payload/"comparison-partial.json",error.report)
        if runner is not None:
            try:
                import gzip, pickle
                state = dict(active=runner.active,epoch=runner.epoch,
                    environment=None if runner.env is None else runner.env.state_dict(),
                    controller=None if runner.controller is None else runner.controller.state_dict(),
                    learner=None if runner.learner is None else runner.learner.state_dict(),
                    budget=budget.snapshot(),resume_authorized=False)
                with gzip.open(payload/"failure-state.pkl.gz","xb") as handle:
                    pickle.dump(state,handle,protocol=5)
            except BaseException as preservation_error:
                base.write_json_once(launcher/"failure-preservation-error.json",dict(error=repr(preservation_error)))
        return 1
    finally:
        budget.close()


def build_runner(payload, proposal, budget, *, admission, **test_backends):
    from src.rl.capacity_pilot_runner import CapacityPilotRunner
    from src.baselines.capacity_completion_control_recovery1 import CapacityCompletionControl
    return CapacityPilotRunner(payload, proposal, budget, admission=admission,
        controller_factory=CapacityCompletionControl, **test_backends)


def launch(root):
    root = Path(root).resolve()
    started = shared_monotonic()
    packet, auth = approved(root)
    caps = packet["config"]["proposed_budget"]["time_seconds"]
    if shared_monotonic()-started > caps["admission_lock_binding"]:
        raise TimeoutError("admission exhausted before launch")
    output = base.confined(root,RUN)
    output.mkdir(parents=True,exist_ok=False)
    launcher = output/"launcher"
    base.write_json_once(launcher/"claim.json",dict(pid=os.getpid(),ppid=os.getppid(),started=started,
        head=base.git(root,"rev-parse","HEAD"),packet_sha256=packet["packet_sha256"],authorization_sha256=digest(auth),clock_id=CLOCK_ID))
    owner, owner_started, owner_times = "admission_lock_binding", started, {}
    offset = 0
    failure_started = None
    process = None
    try:
        with (launcher/"stdout.log").open("xb") as stdout, (launcher/"stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable,"-m",SCRIPT,"--child"],cwd=root,stdout=stdout,stderr=stderr,start_new_session=True)
            while process.poll() is None:
                path = launcher/"budget.jsonl"
                if path.exists():
                    with path.open("r") as handle:
                        handle.seek(offset)
                        while True:
                            before = handle.tell()
                            line = handle.readline()
                            if not line or not line.endswith("\n"):
                                offset = before
                                break
                            event = json.loads(line)
                            if event["event"] == "enter":
                                owner, owner_started, owner_times = event["owner"], event["clock"], event["owner_times"]
                now = shared_monotonic()
                if (launcher/"child-failure.json").exists() and failure_started is None:
                    failure_started = now
                if failure_started is not None:
                    owner, owner_started, owner_times = "failure_flush_shutdown_reserve", failure_started, {}
                if now-started > caps["global"] or owner_times.get(owner,0.)+now-owner_started > caps[owner]:
                    base.write_json_once(launcher/"supervisor-overrun.json",dict(owner=owner,elapsed=now-started,automatic_retry=False))
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
                    break
                time.sleep(.2)
        failed = any((launcher/name).exists() for name in ("child-failure.json","supervisor-overrun.json","failure-preservation-error.json"))
        success = not failed and process.returncode == 0 and (launcher/"stderr.log").stat().st_size == 0 and (launcher/"child-completed.json").is_file()
        if success:
            done = json.loads((launcher/"child-completed.json").read_text())
            success = done["budget"]["counts"] == packet["contract"]["limits"] and done["payload_verified"]
        base.write_json_once(launcher/"terminal.json",dict(status="completed" if success else "failed",exit_code=0 if success else 1,
            child_exit_code=process.returncode,scientific_completion_verified=bool(success),automatic_retry=False,
            elapsed=shared_monotonic()-started,packet_sha256=packet["packet_sha256"]))
        return 0 if success else 1
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        base.write_json_once(launcher/"launch-failure.json",dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        return 1
