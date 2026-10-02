"""Additive recovery entry with lossless persisted-seed normalization.

The failed adapter remains immutable. Collection, restoration, raw comparison,
budgets and supervision are reused; only admission and seed transport are new.
"""

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from src.rl import cohort_evaluation_recovery as previous
from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_execution import runtime_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.cohort_evaluation_models import restore_evaluation_owner
from src.rl.dynamic_candidate_execution import committed_json, confined
from src.rl.dynamic_candidate_preparation import BRANCH, git, source_locks
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.time_baseline_verification import _decimal_seed
from src.rl.time_baseline_watchdog import supervise_time_baseline
from src.utils.research_archive import copy_verified, create_archive, inventory
from src.utils.research_clock import CLOCK_ID, shared_monotonic


DIRECTORY = "specs/2026-10-02-cohort-evaluation-recovery2"
PROTOCOL, PROPOSAL = DIRECTORY + "/protocol.md", DIRECTORY + "/proposal.json"
INTENT, FROZEN, AUTHORIZATION = (DIRECTORY + "/" + p for p in ("approval-intent.json", "frozen.json", "authorization.json"))
RUN = previous.OLD_RUN + "_evaluation_recovery2"
CAPS = dict(checkpoint_load=12, reference_checkpoint_load=3, layout_environment_build=3, episode_build=205)


def runtime_streams(streams):
    """Canonical decimal strings to arbitrary-precision ints, never via float."""
    result = copy.deepcopy(streams)
    for splits in result["environment"].values():
        for role, values in splits.items():
            splits[role] = [_decimal_seed(value) for value in values]
    return result


def bind_inputs(root, output, packet, budget, admission, *, backend_type=None,
                owner_loader=None, restore_owner=None):
    if backend_type is None:
        from src.rl.cohort_backend import CohortPatientBackend
        backend_type = CohortPatientBackend
    owner_loader = load_envelope if owner_loader is None else owner_loader
    restore_owner = restore_evaluation_owner if restore_owner is None else restore_owner
    runtime_packet = dict(packet, streams=runtime_streams(packet["streams"]))
    backend = backend_type(root, packet["scientific_config"], runtime_packet["streams"], admit_real_calls=admission)
    models = {}
    for block in previous.BLOCKS:
        budget.check()
        seed = runtime_packet["streams"]["environment"][str(block)]["layout"][0]
        receipt = backend.prepare(block, seed)
        write_json_once(Path(output) / f"payload/binding/block{block}.json", receipt)
        for role in previous.ROLES[:4]:
            key = f"block{block}/graph/{role}"
            record = packet["models"][key]
            admission("checkpoint_load")
            models[key] = restore_owner(owner_loader(Path(root) / record["path"]))
            if models[key].contract != backend.producer(block).contract:
                raise ValueError("restored policy and original input contract differ")
    write_json_once(Path(output) / "payload/model-bindings.json", dict(models=packet["models"],
        states={k: state_digest(v.state_dict()) for k, v in models.items()}, sealed_before_original_tests=True,
        persisted_streams_sha256=digest(packet["streams"]), runtime_seed_conversion="canonical-decimal-to-int"))
    return backend, models, runtime_packet


def run_evaluations(output, packet, budget, backend, models, *, campaign_type=previous.RecoveryCampaign):
    campaign = campaign_type(output, packet, budget, backend, models)
    campaign.status("binding_complete")
    budget.finish()
    try:
        for block in previous.BLOCKS:
            for role in previous.ROLES:
                budget.begin(f"final_evaluation/block{block}/{role}")
                for descriptor in previous.descriptors(packet["streams"], block, role):
                    campaign.episode(descriptor)
                budget.finish()
    except BaseException:
        if campaign.recorder is not None:
            campaign.recorder.close_partial()
        raise
    return campaign


def freeze(root):
    root = Path(root).resolve()
    if git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain"):
        raise ValueError("clean committed integration worktree required")
    prior = committed_json(root, previous.FROZEN)
    intent, proposal = committed_json(root, INTENT), committed_json(root, PROPOSAL)
    if (intent.get("approved") is not True or intent.get("user") != "Zhaowei"
            or not isinstance(intent.get("user_literal"), str) or not intent["user_literal"].strip()
            or intent.get("proposal_sha256") != file_record(root, PROPOSAL)["sha256"]
            or intent.get("protocol_sha256") != file_record(root, PROTOCOL)["sha256"]
            or proposal.get("scientific_execution_authorized") is not False
            or proposal.get("scope") != {k: prior["intent"][k] for k in proposal["scope"]}
            or set(proposal["scope"]) != {"reused_evaluations", "new_evaluations", "environment_calls",
                "optimizer_calls", "seconds", "owner_seconds", "policy_loads", "reference_loads",
                "layout_builds", "episode_builds", "attempts", "automatic_retry", "scientific_settings_changed",
                "old_attempt_remains_terminal"}
            or proposal.get("phase_seconds") != {k: v["seconds"] for k, v in prior["budget_plan"]["phases"].items()}
            or proposal.get("automatic_followon") is not False or proposal.get("remote_or_dropbox_actions") is not False):
        raise PermissionError("new explicit proposal/protocol-bound restart acceptance required")
    for name, record in prior["source_files"].items():
        if file_record(root, name) != record:
            raise ValueError("historical source changed: " + name)
    inputs = dict(prior["input_files"])
    for name in (previous.FROZEN, previous.AUTHORIZATION, previous.RUN + "/launcher/terminal.json",
                 previous.RUN + "/launcher/child-failure.json", previous.RUN + "/launcher/budget.jsonl"):
        inputs[name] = file_record(root, name)
    if json.loads((root / previous.RUN / "launcher/terminal.json").read_text())["status"] != "failed":
        raise ValueError("prior attempt must remain terminal")
    packet = dict(format="cohort-evaluation-recovery2-frozen-v1", scientific_execution_authorized=False,
        workspace=str(root), result_root=RUN, implementation_commit=git(root, "rev-parse", "HEAD"),
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), intent=intent,
        intent_file=file_record(root, INTENT), protocol=file_record(root, PROTOCOL), proposal=file_record(root, PROPOSAL),
        previous_packet_sha256=prior["packet_sha256"], source_files=source_locks(root), input_files=inputs,
        runtime=runtime_record(), **{k: prior[k] for k in ("scientific_config", "streams", "models", "reused_index", "budget_plan")})
    packet["budget_plan"] = copy.deepcopy(packet["budget_plan"])
    packet["budget_plan"]["draft_sha256"] = digest(proposal)
    runtime_streams(packet["streams"])
    packet["packet_sha256"] = digest(packet)
    return packet


def authorization(packet):
    return dict(format="cohort-evaluation-recovery2-authorization-v1", approved=True,
        packet_sha256=packet["packet_sha256"], implementation_commit=packet["implementation_commit"],
        user_approval=packet["intent"], runtime_sha256=digest(packet["runtime"]),
        automatic_retry=False, new_training=False, remote_or_dropbox_actions=False)


def verify_bindings(root, packet):
    body = dict(packet)
    sha = body.pop("packet_sha256")
    prior, proposal = committed_json(root, previous.FROZEN), committed_json(root, PROPOSAL)
    plan = copy.deepcopy(prior["budget_plan"])
    plan["draft_sha256"] = digest(proposal)
    if (digest(body) != sha or packet["result_root"] != RUN or packet["budget_plan"] != plan
            or packet["source_files"] != source_locks(root) or packet["runtime"] != runtime_record()
            or packet["previous_packet_sha256"] != prior["packet_sha256"]
            or any(packet[k] != prior[k] for k in ("scientific_config", "streams", "models", "reused_index"))):
        raise ValueError("sealed scientific configuration, source, runtime or budget changed")
    for name, record in packet["input_files"].items():
        if file_record(root, name) != record:
            raise ValueError("frozen input changed: " + name)
    for key, name in (("protocol", PROTOCOL), ("proposal", PROPOSAL), ("intent_file", INTENT)):
        if packet[key] != file_record(root, name):
            raise ValueError("bound authorization changed")
    if packet["intent"] != committed_json(root, INTENT) or packet["intent"]["approved"] is not True:
        raise PermissionError("new acceptance missing")
    subprocess.run(["git", "merge-base", "--is-ancestor", packet["implementation_commit"], "HEAD"], cwd=root, check=True)


def approved(root):
    packet, auth = committed_json(root, FROZEN), committed_json(root, AUTHORIZATION)
    if (packet["workspace"] != str(Path(root).resolve()) or auth != authorization(packet)
            or git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain")):
        raise PermissionError("committed packet-specific authorization and clean declared worktree required")
    verify_bindings(root, packet)
    return packet, auth


class Recovery2Admission(previous.RecoveryAdmission):
    def __init__(self, root, packet, auth, claim, budget):
        if (auth != authorization(packet) or claim["pid"] != os.getppid()
                or claim["packet_sha256"] != packet["packet_sha256"]
                or claim["authorization_sha256"] != digest(auth) or claim["started"] != budget.started
                or claim["clock_id"] != CLOCK_ID or budget.plan != packet["budget_plan"]):
            raise PermissionError("live recovery2 authorization/claim required")
        self.root, self.packet, self.claim, self.budget = Path(root) / RUN, packet, claim, budget
        self.pid, self.counts, self.caps = os.getpid(), {}, dict(CAPS)


def child(root):
    packet, auth = approved(root)
    output = Path(root) / RUN
    claim = json.loads((output / "launcher/claim.json").read_text())
    if claim["head"] != git(root, "rev-parse", "HEAD"):
        raise PermissionError("execution commit changed")
    write_json_once(output / "launcher/child-claim.json", dict(pid=os.getpid(), ppid=os.getppid()))
    budget = DynamicCandidateBudget(output / "launcher/budget.jsonl", packet["budget_plan"], enabled=True, started=claim["started"])
    try:
        with previous.no_optimizer_updates():
            budget.begin("runtime_input_binding")
            admission = Recovery2Admission(root, packet, auth, claim, budget)
            paths = set(packet["source_files"]) | set(packet["input_files"]) | {FROZEN, AUTHORIZATION, PROTOCOL, PROPOSAL, INTENT}
            for name in sorted(paths):
                budget.check()
                copy_verified(confined(root, name), output / "payload/locks/worktree" / name)
            for idx in packet["reused_index"]:
                for part in idx.values():
                    for record in part.values():
                        copy_verified(Path(root) / previous.OLD_RUN / record["path"], output / record["path"])
            backend, models, runtime_packet = bind_inputs(root, output, packet, budget, admission)
            campaign = run_evaluations(output, runtime_packet, budget, backend, models)
            budget.begin("raw_verification")
            if len(campaign.index) != 216 or admission.counts != admission.caps:
                raise ValueError("incomplete evaluation or operation matrix")
            write_json_once(output / "payload/evaluation-index.json", campaign.index)
            report = previous.verify_cohort_raw_bundle(output, campaign.index, packet["scientific_config"], packet["streams"])
            write_json_once(output / "payload/independent-verification.json", report)
            write_json_once(output / "payload/comparison.json", report["analysis"])
            ledger = read_dynamic_ledger(budget.path)
            counts = {k: ledger["counts"].get(k, 0) for k in ("environment", "optimizer")}
            if counts != {"environment": 12915, "optimizer": 0}:
                raise ValueError("call accounting differs")
            write_json_once(output / "payload/compute-accounting.json", dict(counts=counts, operations=admission.counts,
                reused_evaluations=11, new_evaluations=205, old_partial_calls_spent=17))
            verify_bindings(root, packet)
            campaign.status("raw_verification_complete", decision=report["analysis"]["decision"])
            budget.finish()
            budget.begin("payload_archive")
            write_json_once(output / "payload/artifact-inventory.json", inventory(output / "payload"))
            archive = create_archive(output / "payload", output / "archives/completed-payload.tar.gz")
            write_json_once(output / "launcher/archive-receipt.json", dict(archive=archive, local_archive_verified=True,
                dropbox_exported=False, cloud_sync_verified=False, howard_access_verified=False))
            budget.finish()
            budget.begin("supervisor_closure")
            if inventory(output / "payload") != archive["files"]:
                raise ValueError("payload changed after archive")
            verify_bindings(root, packet)
            write_json_once(output / "launcher/closure.json", dict(complete=True, evaluations=216,
                new_training_jobs=0, payload_unchanged=True, automatic_followon=False))
            campaign.status("child_complete")
            budget.finish()
        budget.close()
        return 0
    except BaseException as error:
        write_json_once(output / "launcher/child-failure.json", dict(error=repr(error), automatic_retry=False))
        budget.close()
        return 1


def launch(root, started=None):
    started = shared_monotonic() if started is None else started
    packet, auth = approved(root)
    if shared_monotonic() - started >= 300:
        raise TimeoutError("binding cap exhausted")
    output = Path(root) / RUN
    output.mkdir(parents=True, exist_ok=False)
    launcher = output / "launcher"
    write_json_once(launcher / "claim.json", dict(pid=os.getpid(), ppid=os.getppid(), head=git(root, "rev-parse", "HEAD"),
        packet_sha256=packet["packet_sha256"], authorization_sha256=digest(auth), started=started, clock_id=CLOCK_ID))
    try:
        result = supervise_time_baseline([sys.executable, "-m", "experiments.scripts.run_cohort_evaluation_recovery2", "--child"],
            cwd=root, launcher=launcher, plan=packet["budget_plan"], started=started)
        success = result["passed"] and (launcher / "stderr.log").stat().st_size == 0
        if success:
            ledger = read_dynamic_ledger(launcher / "budget.jsonl")
            success = (ledger["counts"].get("environment", 0) == 12915 and ledger["counts"].get("optimizer", 0) == 0
                and ledger["active"] is None and set(ledger["closed"]) == set(packet["budget_plan"]["sections"])
                and json.loads((launcher / "closure.json").read_text())["complete"])
        success = success and shared_monotonic() <= result["final_deadline"]
        write_json_once(launcher / "terminal.json", dict(status="completed" if success else "failed",
            exit_code=0 if success else 1, scientific_completion_verified=bool(success), supervisor=result,
            automatic_retry=False, automatic_followon=False))
        if success:
            receipt = create_archive(launcher, output / "archives/completed-launcher.tar.gz")
            if shared_monotonic() > result["final_deadline"]:
                raise TimeoutError("closure archive exceeded deadline")
            write_json_once(output / "closure-archive-receipt.json", dict(archive=receipt, local_archive_verified=True,
                dropbox_exported=False, cloud_sync_verified=False, howard_access_verified=False))
        return 0 if success else 1
    except BaseException as error:
        write_json_once(launcher / "launch-failure.json", dict(status="failed", error=repr(error), automatic_retry=False))
        return 1
