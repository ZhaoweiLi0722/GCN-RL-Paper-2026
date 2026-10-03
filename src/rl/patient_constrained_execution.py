"""Exact approved single-attempt binding and supervised end-to-end execution."""

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
from src.rl.capacity_native import keyed_seed
from src.rl.patient_constrained_resources import merged_proposal, numeric_contract, PatientConstrainedBudget
from src.rl.patient_constrained_runner import worlds
from src.utils.research_archive import create_archive, inventory
from src.utils.research_clock import shared_monotonic, CLOCK_ID


DIRECTORY = "specs/2026-10-03-patient-constrained-improvement"
PROPOSAL = "experiments/configs/patient_constrained_capacity_20261003.json"
PROTOCOL = DIRECTORY+"/protocol.md"
INTENT, FROZEN, AUTH = (DIRECTORY+"/"+n for n in ("approval-intent.json", "frozen.json", "authorization.json"))
RUN = "results/patient_constrained_capacity_20261003"
SCRIPT = "experiments.scripts.run_patient_constrained_capacity"
EXPECTED_CONFIG = "8bcf5672916985d8769720047dee3a0a24cfa7a1ee78467906c24e5c98914bae"
EXPECTED_PROTOCOL = "a562c0a2222c039381b18156e7f9d60fcae02ecd863ef058a956d007723ed22c"
SCOPE = "patient-constrained-capacity-v1"


def configure_runtime():
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)


def source_locks(root):
    records = base.source_locks(root)
    required = ["src/rl/patient_constrained_"+n+".py" for n in ("resources", "runner", "learner", "analysis", "execution")]
    required.append("experiments/scripts/run_patient_constrained_capacity.py")
    if not set(required).issubset(records):
        raise ValueError("new entry must be committed before freeze")
    return records


def seed_files(root):
    names = set(base.seed_inventory(root))
    names.add(root/base.FROZEN)
    return sorted(p for p in names if p.relative_to(root).as_posix() != PROPOSAL
                  and not p.relative_to(root).as_posix().startswith((DIRECTORY+"/", RUN+"/")))


def stream_manifest(p):
    allocations = list(p["design"]["training_seeds"])+[p["patient_constrained"]["readout"]["bootstrap"]["seed"]]
    rows = []
    for phase in ("training", "evaluation"):
        for block in range(3):
            for world in worlds(p,phase,block):
                streams = {purpose:keyed_seed(p,world,purpose) for purpose in p["design"]["substream_purposes"]}
                rows.append(dict(world=world, substreams=streams))
                allocations += [world["seed"], *streams.values()]
    if len(set(allocations)) != len(allocations):
        raise ValueError("new random stream collision")
    return dict(rows=rows, allocations=allocations, shared_training_worlds_between_three_arms=True)


def freeze(root):
    root = Path(root).resolve()
    base._clean(root)
    cfg, intent = base.committed_json(root,PROPOSAL), base.committed_json(root,INTENT)
    config_file, protocol_file = base.file_record(root,PROPOSAL), base.file_record(root,PROTOCOL)
    if (config_file["sha256"] != EXPECTED_CONFIG or protocol_file["sha256"] != EXPECTED_PROTOCOL
            or not intent["scope_approved"] or intent["proposal_sha256"] != EXPECTED_CONFIG
            or intent["protocol_sha256"] != EXPECTED_PROTOCOL or cfg["scientific_execution_authorized"] is not False):
        raise PermissionError("exact numeric package approval required")
    original_file = base.file_record(root,cfg["inherit"]["proposal"])
    if original_file["sha256"] != cfg["inherit"]["sha256"]:
        raise ValueError("inherited simulator contract changed")
    p = merged_proposal(base.committed_json(root,cfg["inherit"]["proposal"]),cfg)
    streams = stream_manifest(p)
    audit = audit_stream_collisions(streams["allocations"],seed_files(root))
    for entry in audit["files"]+audit["collisions"]:
        entry["path"] = Path(entry["path"]).relative_to(root).as_posix()
    if not audit["passed"]:
        raise ValueError("prospective seed collision: "+repr(audit["collisions"]))
    packet = dict(format="patient-constrained-frozen-v1", workspace=str(root), branch=base.BRANCH,
        implementation_commit=base.git(root,"rev-parse","HEAD"), result_root=RUN,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), proposal=config_file, protocol=protocol_file,
        inherited_file=original_file, intent=intent, intent_file=base.file_record(root,INTENT),
        config=p, contract=numeric_contract(cfg), streams=streams, seed_audit=audit,
        source_files=source_locks(root), runtime=base.runtime_record())
    packet["packet_sha256"] = digest(packet)
    return packet


def authorization(packet):
    base._sealed_body(packet)
    if (packet["format"] != "patient-constrained-frozen-v1" or packet["proposal"]["sha256"] != EXPECTED_CONFIG
            or packet["protocol"]["sha256"] != EXPECTED_PROTOCOL or not packet["intent"]["scope_approved"]
            or not packet["seed_audit"]["passed"] or packet["contract"] != numeric_contract(packet["config"]["patient_constrained"])):
        raise PermissionError("invalid patient-constrained packet")
    return dict(format="patient-constrained-effective-authorization-v1", scientific_execution_authorized=True,
        packet_sha256=packet["packet_sha256"], implementation_commit=packet["implementation_commit"],
        user_approval=packet["intent"], limits=packet["contract"], runtime_sha256=digest(packet["runtime"]),
        attempts=1, automatic_retry=False, automatic_resume=False, automatic_follow_on=False, remote_export=False)


def approved(root):
    root = Path(root).resolve()
    base._clean(root)
    packet, auth = base.committed_json(root,FROZEN), base.committed_json(root,AUTH)
    if (packet["workspace"] != str(root) or packet["branch"] != base.git(root,"branch","--show-current")
            or auth != authorization(packet) or packet["source_files"] != source_locks(root)
            or packet["runtime"] != base.runtime_record()):
        raise PermissionError("current source/runtime/authorization differs from frozen execution")
    for field,name in (("proposal",PROPOSAL),("protocol",PROTOCOL),("intent_file",INTENT),
                       ("inherited_file",packet["config"]["patient_constrained"]["inherit"]["proposal"])):
        if base.file_record(root,name) != packet[field]:
            raise ValueError("frozen scientific input changed: "+name)
    if [base.file_record(root,p.relative_to(root).as_posix()) for p in seed_files(root)] != packet["seed_audit"]["files"]:
        raise ValueError("seed metadata changed since freeze")
    subprocess.run(["git","merge-base","--is-ancestor",packet["implementation_commit"],"HEAD"],cwd=root,check=True)
    return packet,auth


def build_runner(payload, proposal, budget, *, admission, **backends):
    from src.rl.patient_constrained_runner import PatientConstrainedRunner
    return PatientConstrainedRunner(payload,proposal,budget,admission=admission,**backends)


def child(root):
    root = Path(root).resolve()
    packet,_ = approved(root)
    output, payload = root/RUN, root/RUN/"payload"
    launcher = output/"launcher"
    claim = json.loads((launcher/"claim.json").read_text())
    if claim["pid"] != os.getppid() or claim["packet_sha256"] != packet["packet_sha256"]:
        raise PermissionError("child lacks exclusive parent claim")
    base.write_json_once(launcher/"child-claim.json",dict(pid=os.getpid(),ppid=os.getppid(),packet_sha256=packet["packet_sha256"]))
    budget = PatientConstrainedBudget(launcher,packet["config"],started=claim["started"])
    payload.mkdir(exist_ok=False)
    runner = None
    try:
        runner = build_runner(payload,packet["config"],budget,admission=dict(verified=True,scope=SCOPE))
        result = runner.run()
        budget.enter("analysis_archive_verification")
        from src.rl.patient_constrained_analysis import run_analysis
        analysis = run_analysis(payload,packet["config"]["patient_constrained"])
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
        base.write_json_once(launcher/"child-failure.json",dict(status="failed",error=repr(error),
            traceback=traceback.format_exc(),budget=budget.snapshot(),automatic_retry=False))
        if runner is not None:
            try:
                with gzip.open(payload/"failure-state.pkl.gz","xb") as handle:
                    pickle.dump(runner.state_dict(),handle,protocol=5)
            except BaseException as preservation_error:
                base.write_json_once(launcher/"failure-preservation-error.json",dict(error=repr(preservation_error)))
        return 1
    finally:
        budget.close()


def launch(root):
    root = Path(root).resolve()
    started = shared_monotonic()
    packet,auth = approved(root)
    caps = packet["config"]["proposed_budget"]["time_seconds"]
    if shared_monotonic()-started > caps["admission_lock_binding"]:
        raise TimeoutError("admission budget exhausted")
    output = root/RUN
    output.mkdir(parents=True,exist_ok=False)
    launcher = output/"launcher"
    base.write_json_once(launcher/"claim.json",dict(pid=os.getpid(),ppid=os.getppid(),started=started,
        head=base.git(root,"rev-parse","HEAD"),packet_sha256=packet["packet_sha256"],authorization_sha256=digest(auth),clock_id=CLOCK_ID))
    owner, owner_started, owner_times = "admission_lock_binding", started, {}
    job, job_started, job_consumed, job_cap = None, started, 0., 0.
    offset, process, failure_started = 0,None,None
    try:
        with (launcher/"stdout.log").open("xb") as stdout, (launcher/"stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable,"-m",SCRIPT,"--child"],cwd=root,
                stdout=stdout,stderr=stderr,start_new_session=True)
            while process.poll() is None:
                ledger = launcher/"budget.jsonl"
                if ledger.exists():
                    with ledger.open() as handle:
                        handle.seek(offset)
                        while True:
                            before = handle.tell()
                            line = handle.readline()
                            if not line or not line.endswith("\n"):
                                offset = before
                                break
                            row = json.loads(line)
                            if row["event"] == "enter":
                                owner,owner_started,owner_times = row["owner"],row["clock"],row["owner_times"]
                                job = None
                            elif row["event"] == "job_enter":
                                job,job_started,job_consumed,job_cap = row["job"],row["clock"],row["consumed"],row["cap"]
                now = shared_monotonic()
                if (launcher/"child-failure.json").exists() and failure_started is None:
                    failure_started = now
                if failure_started is not None:
                    owner,owner_started,owner_times,job = "failure_flush_shutdown_reserve",failure_started,{},None
                if (now-started > caps["global"] or owner_times.get(owner,0.)+now-owner_started > caps[owner]
                        or job is not None and job_consumed+now-job_started > job_cap):
                    base.write_json_once(launcher/"supervisor-overrun.json",dict(owner=owner,job=job,elapsed=now-started,automatic_retry=False))
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
            completed = json.loads((launcher/"child-completed.json").read_text())
            success = completed["budget"]["counts"] == packet["contract"]["limits"] and completed["payload_verified"]
        base.write_json_once(launcher/"terminal.json",dict(status="completed" if success else "failed",exit_code=0 if success else 1,
            child_exit_code=process.returncode,scientific_completion_verified=bool(success),elapsed=shared_monotonic()-started,
            packet_sha256=packet["packet_sha256"],automatic_retry=False))
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
