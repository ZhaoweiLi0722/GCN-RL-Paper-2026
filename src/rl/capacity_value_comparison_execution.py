"""New value-MPC authorization boundary with the existing supervised IO design."""

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
from src.rl.capacity_value_comparison_resources import merged_proposal, numeric_contract, CapacityValueComparisonBudget, worlds
from src.utils.research_archive import create_archive, inventory
from src.utils.research_clock import shared_monotonic, CLOCK_ID

DIRECTORY = "specs/2026-10-04-value-mpc-comparison"
PROPOSAL = "experiments/configs/capacity_value_comparison_20261004.json"
PROTOCOL = DIRECTORY+"/protocol.md"
INTENT, FROZEN, AUTH = (DIRECTORY+"/"+n for n in ("approval-intent.json","frozen.json","authorization.json"))
RUN = "results/capacity_value_comparison_20261004"
SCRIPT = "experiments.scripts.run_capacity_value_comparison"
SCOPE = "capacity-value-comparison-v1"


def configure_runtime():
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)


def source_locks(root):
    records = base.source_locks(root)
    required = ["src/rl/capacity_value_"+name+".py" for name in ("features","learner","resources","runner","analysis","execution")]
    required += ["src/rl/capacity_value_comparison_"+name+".py" for name in ("learner","resources","runner","analysis","execution")]
    required += ["src/models/capacity_terminal_value.py", "src/baselines/capacity_value_mpc.py",
                 "src/baselines/capacity_matched_value.py", "experiments/scripts/run_capacity_value_comparison.py"]
    if not set(required).issubset(records):
        raise ValueError("commit new implementation before freeze")
    return records


def seed_files(root):
    # The reused inventory excludes its own older campaign. It is historical
    # evidence here and must be included in this new scope's seed check.
    files = set(base.seed_inventory(root))
    for directory in (base.DIRECTORY, base.RUN):
        for path in (root/directory).rglob("*"):
            if (path.is_file() and path.suffix in (".json", ".jsonl")
                    and any(word in path.name.lower() for word in ("seed", "stream", "manifest", "config", "frozen"))):
                files.add(base.confined(root, path.relative_to(root).as_posix()))
    return sorted(p for p in files
                  if p.relative_to(root).as_posix() != PROPOSAL
                  and not p.relative_to(root).as_posix().startswith((DIRECTORY+"/",RUN+"/")))


def stream_manifest(p):
    study = p["value_comparison_study"]
    seeds = [s for a in ("graph","flat") for s in study["design"]["model_seeds"][a]]
    allocations = seeds+[s+1 for s in seeds]+[study["streams"]["bootstrap_seed"]]
    rows = []
    for phase in ("initial","continuation","evaluation"):
        for b in range(5):
            for w in worlds(study,phase,b):
                streams = {k:keyed_seed(p,w,k) for k in study["streams"]["purposes"]}
                rows.append(dict(world=w,substreams=streams))
                allocations += [w["seed"],*streams.values()]
    if len(set(allocations)) != len(allocations):
        raise ValueError("prospective value-MPC seed collision")
    return dict(rows=rows,allocations=allocations)


def freeze(root):
    root = Path(root).resolve()
    base._clean(root)
    study, intent = base.committed_json(root,PROPOSAL), base.committed_json(root,INTENT)
    proposal, protocol = base.file_record(root,PROPOSAL), base.file_record(root,PROTOCOL)
    if (intent.get("scope_approved") is not True or intent.get("scope") != SCOPE
            or not isinstance(intent.get("user_literal"), str) or not intent["user_literal"].strip()
            or intent.get("proposal_sha256") != proposal["sha256"]
            or intent.get("protocol_sha256") != protocol["sha256"]
            or study["scientific_execution_authorized"] is not False):
        raise PermissionError("exact complete numerical package approval required")
    original_file = base.file_record(root,study["inherit"]["proposal"])
    if original_file["sha256"] != study["inherit"]["sha256"]:
        raise ValueError("original synthetic system changed")
    p = merged_proposal(base.committed_json(root,study["inherit"]["proposal"]),study)
    streams = stream_manifest(p)
    audit = audit_stream_collisions(streams["allocations"],seed_files(root))
    for row in audit["files"]+audit["collisions"]:
        row["path"] = Path(row["path"]).relative_to(root).as_posix()
    if not audit["passed"]:
        raise ValueError("historical stream collision; no silent substitution")
    packet = dict(format="capacity-value-comparison-frozen-v1",workspace=str(root),branch=base.BRANCH,
        implementation_commit=base.git(root,"rev-parse","HEAD"),result_root=RUN,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),proposal=proposal,protocol=protocol,
        original_file=original_file,intent=intent,intent_file=base.file_record(root,INTENT),
        config=p,contract=numeric_contract(study),streams=streams,seed_audit=audit,
        source_files=source_locks(root),runtime=base.runtime_record())
    packet["packet_sha256"] = digest(packet)
    return packet


def authorization(packet):
    base._sealed_body(packet)
    if (packet["format"] != "capacity-value-comparison-frozen-v1" or not packet["intent"]["scope_approved"]
            or packet["intent"].get("scope") != SCOPE or not packet["intent"].get("user_literal")
            or not packet["seed_audit"]["passed"] or packet["contract"] != numeric_contract(packet["config"]["value_comparison_study"])):
        raise PermissionError("invalid value-MPC frozen packet")
    return dict(format="capacity-value-comparison-authorization-v1",scientific_execution_authorized=True,
        packet_sha256=packet["packet_sha256"],implementation_commit=packet["implementation_commit"],
        user_approval=packet["intent"],attempts=1,automatic_retry=False,automatic_resume=False,
        automatic_follow_on=False,remote_export=False)


def approved(root):
    root = Path(root).resolve()
    base._clean(root)
    packet, auth = base.committed_json(root,FROZEN),base.committed_json(root,AUTH)
    if (packet["workspace"] != str(root) or packet["branch"] != base.git(root,"branch","--show-current")
            or auth != authorization(packet) or packet["source_files"] != source_locks(root)
            or packet["runtime"] != base.runtime_record()):
        raise PermissionError("current value-MPC source/runtime/authority mismatch")
    for field,name in (("proposal",PROPOSAL),("protocol",PROTOCOL),("intent_file",INTENT),
                        ("original_file",packet["config"]["value_comparison_study"]["inherit"]["proposal"])):
        if packet[field] != base.file_record(root,name):
            raise ValueError("value-MPC locked input changed")
    if [base.file_record(root,p.relative_to(root).as_posix()) for p in seed_files(root)] != packet["seed_audit"]["files"]:
        raise ValueError("seed metadata changed since freeze")
    subprocess.run(["git","merge-base","--is-ancestor",packet["implementation_commit"],"HEAD"],cwd=root,check=True)
    return packet,auth


def build_runner(payload, proposal, budget, *, admission, **backends):
    from src.rl.capacity_value_comparison_runner import CapacityValueComparisonRunner
    return CapacityValueComparisonRunner(payload,proposal,budget,admission=admission,**backends)


def child(root):
    root = Path(root).resolve()
    packet,_ = approved(root)
    output, payload = root/RUN,root/RUN/"payload"
    launcher = output/"launcher"
    claim = json.loads((launcher/"claim.json").read_text())
    if claim["pid"] != os.getppid() or claim["packet_sha256"] != packet["packet_sha256"]:
        raise PermissionError("exclusive parent claim required")
    base.write_json_once(launcher/"child-claim.json",dict(pid=os.getpid(),ppid=os.getppid(),packet_sha256=packet["packet_sha256"]))
    budget = CapacityValueComparisonBudget(launcher,packet["config"],started=claim["started"])
    payload.mkdir(exist_ok=False)
    runner = None
    try:
        runner = build_runner(payload,packet["config"],budget,admission=dict(verified=True,scope=SCOPE))
        result = runner.run()
        budget.enter("analysis_archive_verification")
        from src.rl.capacity_value_comparison_analysis import run_analysis
        analysis = run_analysis(payload,packet["config"]["value_comparison_study"])
        base.write_json_once(payload/"comparison.json",analysis)
        base.write_json_once(payload/"completion.json",dict(**result,packet_sha256=packet["packet_sha256"]))
        base.write_json_once(payload/"artifact-inventory.json",inventory(payload))
        budget.check(storage=True)
        receipt = create_archive(payload,output/"archives/payload.tar.gz")
        base.write_json_once(output/"archive-receipt.json",dict(archive=receipt,local_archive_verified=True,
            dropbox_exported=False,cloud_sync_verified=False,howard_access_verified=False))
        budget.check(storage=True)
        base.write_json_once(launcher/"child-completed.json",dict(budget=budget.snapshot(),payload_verified=True,packet_sha256=packet["packet_sha256"]))
        budget.check(storage=True)
        return 0
    except BaseException as error:
        base.write_json_once(launcher/"child-failure.json",dict(status="failed",error=repr(error),traceback=traceback.format_exc(),budget=budget.snapshot(),automatic_retry=False))
        if runner is not None:
            try:
                with gzip.open(payload/"failure-state.pkl.gz","xb") as handle:
                    pickle.dump(runner.state_dict(),handle,protocol=5)
            except BaseException as error:
                base.write_json_once(launcher/"failure-preservation-error.json",dict(error=repr(error)))
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
    owner,owner_started,owner_times = "admission_lock_binding",started,{}
    job,job_started,job_consumed,job_cap = None,started,0.,0.
    offset,process,failure_started = 0,None,None
    try:
        with (launcher/"stdout.log").open("xb") as stdout,(launcher/"stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable,"-m",SCRIPT,"--child"],cwd=root,stdout=stdout,stderr=stderr,start_new_session=True)
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
