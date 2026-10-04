"""Single authorized remaining-only comparison recovery; no retry interface."""

from datetime import datetime, timezone
import gzip
import hashlib
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
from src.rl import capacity_value_comparison_execution as original
from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_value_comparison_recovery1_resources import (
    CapacityValueComparisonRecovery1Budget, RECOVERY_SCOPE, PARTIAL, remainder)
from src.rl.capacity_value_comparison_resources import SCOPE
from src.utils.research_archive import create_archive, inventory, copy_verified
from src.utils.research_clock import shared_monotonic, CLOCK_ID

DIRECTORY = original.DIRECTORY+"/recovery1"
PROPOSAL = original.PROPOSAL
PROTOCOL = DIRECTORY+"/protocol.md"
INTENT, FROZEN, AUTH = (DIRECTORY+"/"+n for n in ("approval-intent.json","frozen.json","authorization.json"))
RUN = original.RUN+"_recovery1"
SCRIPT = "experiments.scripts.run_capacity_value_comparison_recovery1"
configure_runtime = original.configure_runtime


def source_locks(root):
    records = original.source_locks(root)
    required = ["src/rl/capacity_value_comparison_recovery1_"+s+".py"
                for s in ("runner","resources","execution")]
    required += ["src/baselines/capacity_forecast_recovery3.py",
                 "src/baselines/capacity_value_mpc_recovery1.py",
                 "experiments/scripts/run_capacity_value_comparison_recovery1.py"]
    if not set(required).issubset(records):
        raise ValueError("commit recovery implementation before freeze")
    return records


def old_evidence(root):
    oldroot = root/original.RUN
    hashes = inventory(oldroot)
    storage = dict(files=len(hashes), total_bytes=0, raw_bytes=0, archive_bytes=0)
    for relative in hashes:
        size = (oldroot/relative).stat().st_size
        storage["total_bytes"] += size
        storage["archive_bytes" if "archives" in Path(relative).parts else "raw_bytes"] += size
    old = json.loads((oldroot/"launcher/child-failure.json").read_text())["budget"]
    terminal = json.loads((oldroot/"launcher/terminal.json").read_text())
    return hashes, storage, old, terminal


def freeze(root):
    root = Path(root).resolve()
    base._clean(root)
    previous = base.committed_json(root,original.FROZEN)
    base._sealed_body(previous)
    intent = base.committed_json(root,INTENT)
    protocol = base.file_record(root,PROTOCOL)
    if (intent.get("scope") != RECOVERY_SCOPE or intent.get("scope_approved") is not True
            or intent.get("user_literal") != "批准 继续跑"
            or intent.get("protocol_sha256") != protocol["sha256"]
            or intent.get("original_packet_sha256") != previous["packet_sha256"]):
        raise PermissionError("exact remaining-only approval required")
    records = source_locks(root)
    if any(records.get(k) != v for k,v in previous["source_files"].items()):
        raise ValueError("historical scientific source changed")
    hashes, storage, old, terminal = old_evidence(root)
    if hashes["payload/failure-state.pkl.gz"] != "b9a624487e5deda8b80a940d26e28e167bbbcd7a1d38fec603ef24b34d427b2e":
        raise ValueError("wrong saved recovery state")
    locked_names = [original.FROZEN, original.AUTH, original.PROTOCOL, original.PROPOSAL,
                    previous["config"]["value_comparison_study"]["inherit"]["proposal"]]
    packet = dict(format="capacity-value-comparison-recovery1-frozen-v1",
        workspace=str(root),branch=base.BRANCH,implementation_commit=base.git(root,"rev-parse","HEAD"),
        result_root=RUN,previous_result_root=original.RUN,original_packet_sha256=previous["packet_sha256"],
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),protocol=protocol,
        intent=intent,intent_file=base.file_record(root,INTENT),
        config=previous["config"], contract=remainder(previous["config"]["value_comparison_study"],old,terminal,storage),
        prior_files=hashes, original_locks={p:base.file_record(root,p) for p in locked_names},
        streams=previous["streams"], seed_audit=previous["seed_audit"],
        seed_reuse="intentional_remaining_only_not_independent_confirmation",
        source_files=records,runtime=base.runtime_record())
    packet["packet_sha256"] = digest(packet)
    return packet


def authorization(packet):
    base._sealed_body(packet)
    if (packet["format"] != "capacity-value-comparison-recovery1-frozen-v1"
            or packet["intent"].get("scope") != RECOVERY_SCOPE
            or packet["intent"].get("scope_approved") is not True
            or packet["intent"].get("user_literal") != "批准 继续跑"
            or packet["contract"]["limits"]["native_steps"] != 33819
            or packet["contract"]["limits"]["total_optimizer_steps"] != 12320):
        raise PermissionError("invalid remaining-only frozen packet")
    return dict(format="capacity-value-comparison-recovery1-authorization-v1",
        scientific_execution_authorized=True,packet_sha256=packet["packet_sha256"],
        implementation_commit=packet["implementation_commit"],user_approval=packet["intent"],
        attempts=1,automatic_retry=False,automatic_resume=False,automatic_follow_on=False,remote_export=False)


def approved(root):
    root = Path(root).resolve()
    base._clean(root)
    packet,auth = base.committed_json(root,FROZEN),base.committed_json(root,AUTH)
    if (packet["workspace"] != str(root) or packet["branch"] != base.git(root,"branch","--show-current")
            or auth != authorization(packet) or packet["source_files"] != source_locks(root)
            or packet["runtime"] != base.runtime_record()
            or packet["protocol"] != base.file_record(root,PROTOCOL)
            or packet["intent_file"] != base.file_record(root,INTENT)):
        raise PermissionError("recovery source/runtime/authority mismatch")
    for path,expected in packet["original_locks"].items():
        if base.file_record(root,path) != expected:
            raise ValueError("historical locked input changed")
    hashes,storage,old,terminal = old_evidence(root)
    if (hashes != packet["prior_files"] or
            remainder(packet["config"]["value_comparison_study"],old,terminal,storage) != packet["contract"]):
        raise ValueError("historical state/remaining budgets changed")
    subprocess.run(["git","merge-base","--is-ancestor",packet["implementation_commit"],"HEAD"],cwd=root,check=True)
    return packet,auth


def import_payload(root, payload, packet):
    old = root/original.RUN/"payload"
    partial = "raw/"+PARTIAL+".jsonl.gz"
    copied = {}
    for relative,sha in packet["prior_files"].items():
        if not relative.startswith("payload/"):
            continue
        name = relative.removeprefix("payload/")
        if name != "progress.jsonl" and Path(name).parts[0] not in ("raw","summaries","models","states","tapes","updates"):
            continue
        target = "recovery-prefix/"+Path(name).name if name == partial else name
        receipt = copy_verified(old/name,payload/target)
        if receipt["sha256"] != sha:
            raise ValueError("copied recovery input hash mismatch")
        copied[target] = sha
    base.write_json_once(payload/"recovery-inputs.json",dict(original_root=original.RUN,
        files=copied,old_counts=packet["contract"]["old_counts"],
        retained_reservation=packet["contract"]["retained_interrupted_reservation"],
        mixed_predictor_history=True,seed_reuse=packet["seed_reuse"]))
    return payload/"recovery-prefix"/Path(partial).name


def build_runner(payload, proposal, budget, *, admission, failure_state, partial_raw_path, **backends):
    from src.rl.capacity_value_comparison_recovery1_runner import CapacityValueComparisonRecovery1Runner
    return CapacityValueComparisonRecovery1Runner(payload,proposal,budget,admission=admission,
        failure_state=failure_state,partial_raw_path=partial_raw_path,**backends)


def child(root):
    root = Path(root).resolve()
    packet,_ = approved(root)
    output,payload = root/RUN,root/RUN/"payload"
    launcher = output/"launcher"
    claim = json.loads((launcher/"claim.json").read_text())
    if claim["pid"] != os.getppid() or claim["packet_sha256"] != packet["packet_sha256"]:
        raise PermissionError("exclusive recovery parent claim required")
    base.write_json_once(launcher/"child-claim.json",dict(pid=os.getpid(),ppid=os.getppid(),packet_sha256=packet["packet_sha256"]))
    budget = CapacityValueComparisonRecovery1Budget(launcher,packet["config"],packet["contract"],started=claim["started"])
    payload.mkdir(exist_ok=False)
    runner = None
    try:
        prefix = import_payload(root,payload,packet)
        with gzip.open(root/original.RUN/"payload/failure-state.pkl.gz","rb") as handle:
            state = pickle.load(handle)
        budget.check(storage=True)
        runner = build_runner(payload,packet["config"],budget,
            admission=dict(verified=True,scope=SCOPE,remaining_scope=RECOVERY_SCOPE),
            failure_state=state,partial_raw_path=prefix)
        result = runner.run()
        budget.enter("analysis_archive_verification")
        from src.rl.capacity_value_comparison_analysis import run_analysis
        analysis = run_analysis(payload,packet["config"]["value_comparison_study"])
        analysis["recovery_provenance"] = dict(original_packet_sha256=packet["original_packet_sha256"],
            resumed_from_epoch=37,old_optimizer_steps=3040,new_optimizer_steps=12320,
            mixed_predictor_training_history=True,all_evaluation_uses_repaired_predictor=True,
            independent_confirmation=False)
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
    caps = packet["contract"]["seconds"]
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

