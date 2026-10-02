"""One-shot evaluation of unchanged sealed models; no training path exists."""

from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch

from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_execution import runtime_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.cohort_bundle_verification import verify_cohort_raw_bundle
from src.rl.cohort_collection import CohortCollection
from src.rl.cohort_recording import CohortRecorder
from src.rl.dynamic_candidate_execution import committed_json, confined
from src.rl.dynamic_candidate_preparation import BRANCH, git, source_locks
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger, validate_budget_plan
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.time_baseline_watchdog import supervise_time_baseline
from src.utils.research_archive import copy_verified, create_archive, inventory
from src.utils.research_clock import CLOCK_ID, shared_monotonic


DIRECTORY = "specs/2026-10-02-cohort-evaluation-recovery"
PROTOCOL, INTENT = DIRECTORY + "/protocol.md", DIRECTORY + "/approval-intent.json"
FROZEN, AUTHORIZATION = DIRECTORY + "/frozen.json", DIRECTORY + "/authorization.json"
OLD_PACKET = "specs/2026-10-02-terminal-obligation/frozen.json"
OLD_RUN = "results/dynamic_candidate_cohort_objective_20261002"
RUN = OLD_RUN + "_evaluation_recovery1"
CLOSURE = "reports/2026-10-02-cohort-objective-closure"
BLOCKS = (60, 61, 62)
ROLES = ("own_frozen", "window_ppo", "cohort_ppo", "bc_continue", "r4", "full_mdl2")


def descriptors(streams, block, role):
    if block not in BLOCKS or role not in ROLES:
        raise ValueError("undeclared evaluation owner")
    seeds = streams["environment"][str(block)]["test"]
    if len(seeds) != 12:
        raise ValueError("original twelve test worlds required")
    return [(block, role, w, seed, "test",
             "reference" if role == "r4" else "anchor" if role == "full_mdl2" else "greedy",
             f"evaluation/block{block}/{role}/world{w:02d}")
            for w, seed in enumerate(seeds) if not (block == 60 and role == "own_frozen" and w < 11)]


def budget_plan(intent):
    expected = dict(reused_evaluations=11, new_evaluations=205, environment_calls=12915,
        optimizer_calls=0, seconds=7200, owner_seconds=240, policy_loads=12,
        reference_loads=3, layout_builds=3, episode_builds=205, attempts=1,
        automatic_retry=False, scientific_settings_changed=False, old_attempt_remains_terminal=True)
    if (any(type(intent.get(k)) is not type(v) or intent.get(k) != v for k, v in expected.items())
            or intent.get("user") != "Zhaowei" or intent.get("user_literal") != "继续"):
        raise PermissionError("exact bounded evaluation-only acceptance required")
    def row(steps, seconds):
        return dict(trajectory=steps, clone=0, actor=0, critic=0, seconds=seconds)
    phases = {"runtime_input_binding": row(0, 300), "final_evaluation": row(12915, 4320),
        "raw_verification": row(0, 600), "payload_archive": row(0, 900), "supervisor_closure": row(0, 600)}
    sections = {"runtime_input_binding": phases["runtime_input_binding"] | {"phase": "runtime_input_binding"}}
    for b in BLOCKS:
        for role in ROLES:
            count = 1 if b == 60 and role == "own_frozen" else 12
            sections[f"final_evaluation/block{b}/{role}"] = row(63 * count, 240) | {"phase": "final_evaluation"}
    for name in ("raw_verification", "payload_archive", "supervisor_closure"):
        sections[name] = phases[name] | {"phase": name}
    plan = dict(format="dynamic-candidate-budget-plan-v1", draft_sha256=digest(intent),
                limits=row(12915, 7200), phases=phases, sections=sections)
    validate_budget_plan(plan)
    return plan


def reuse_index(root):
    """Only the eleven previously completed raw bundles, never the partial twelfth."""
    source = Path(root) / OLD_RUN
    indexes = []
    for world in range(11):
        directory = source / f"payload/episodes/evaluation/block60/own_frozen/world{world:02d}"
        tail = json.loads((directory / "tail/header.json").read_text())
        if (tail["split"] != "test" or tail["block"] != 60 or tail["role"] != "own_frozen"
                or tail["world_index"] != world or not (directory / "tail/outcome.json").is_file()):
            raise ValueError("invalid completed reuse boundary")
        prefix = tail["prefix"]
        for rec in prefix.values():
            if file_record(source, rec["path"]) != rec:
                raise ValueError("reused prefix changed")
        indexes.append(dict(prefix=prefix, tail={key: file_record(source, directory / "tail" / filename)
            for key, filename in (("header", "header.json"), ("events", "events.jsonl"), ("final_state", "final_state.json"))}))
    return indexes


def freeze(root):
    root = Path(root).resolve()
    if git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain"):
        raise ValueError("clean committed integration worktree required")
    old, intent = committed_json(root, OLD_PACKET), committed_json(root, INTENT)
    if (old["scientific_config"]["blocks"] != list(BLOCKS)
            or old["scientific_config"]["final_controllers"] != list(ROLES)
            or json.loads((root / OLD_RUN / "launcher/terminal.json").read_text())["status"] != "failed"):
        raise ValueError("exact terminal cohort source required")
    for name, record in old["source_files"].items():
        if file_record(root, name) != record:
            raise ValueError("historical implementation changed: " + name)
    seals = json.loads((root / OLD_RUN / "payload/model-seals.json").read_text())
    expected = {f"block{b}/graph/{r}" for b in BLOCKS for r in ROLES[:4]}
    if set(seals) != expected:
        raise ValueError("twelve sealed models required")
    paths = {OLD_PACKET, "specs/2026-10-02-terminal-obligation/authorization.json",
        OLD_RUN + "/payload/model-seals.json", OLD_RUN + "/launcher/terminal.json",
        OLD_RUN + "/launcher/budget.jsonl", CLOSURE + "/independent-training-targets.json",
        CLOSURE + "/metadata-verification.json", CLOSURE + "/closure-index.json",
        CLOSURE + "/archives/failed-run.tar.gz.manifest.json"}
    models = {}
    for key, record in seals.items():
        if file_record(root / OLD_RUN, record["path"]) != record:
            raise ValueError("sealed model changed")
        name = OLD_RUN + "/" + record["path"]
        models[key] = file_record(root, name)
        paths.add(name)
    reference = old["scientific_config"]["reference"]
    for b in BLOCKS:
        for kind in ("config", "policy"):
            name = reference["directory"] + "/" + reference[kind + "_template"].format(block=b)
            if file_record(root, name)["sha256"] != reference["locks"][str(b)][kind]:
                raise ValueError("original R4 reference changed")
            paths.add(name)
    reused = reuse_index(root)
    paths.update(OLD_RUN + "/" + rec["path"] for idx in reused for part in idx.values() for rec in part.values())
    packet = dict(format="cohort-evaluation-recovery-frozen-v1", scientific_execution_authorized=False,
        workspace=str(root), branch=BRANCH, result_root=RUN, old_result_root=OLD_RUN,
        implementation_commit=git(root, "rev-parse", "HEAD"), frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        intent=intent, intent_file=file_record(root, INTENT), protocol=file_record(root, PROTOCOL),
        scientific_config=old["scientific_config"], streams=old["streams"],
        original_packet_sha256=old["packet_sha256"], budget_plan=budget_plan(intent),
        reused_index=reused, models=models, runtime=runtime_record(), source_files=source_locks(root),
        input_files={p: file_record(root, p) for p in sorted(paths)},
        new_checkpoint_loads=0, new_environment_calls=0, new_optimizer_calls=0, new_model_forwards=0)
    packet["packet_sha256"] = digest(packet)
    return packet


def authorization(packet):
    return dict(format="cohort-evaluation-recovery-authorization-v1", approved=True,
        packet_sha256=packet["packet_sha256"], implementation_commit=packet["implementation_commit"],
        runtime_sha256=digest(packet["runtime"]), user_approval=packet["intent"],
        protocol_sha256=packet["protocol"]["sha256"], budget_sha256=digest(packet["budget_plan"]),
        automatic_retry=False, training_authorized=False, old_attempt_remains_terminal=True,
        remote_or_dropbox_actions=False)


def verify_bindings(root, packet):
    body = dict(packet)
    if body.pop("packet_sha256", None) != digest(body):
        raise ValueError("packet changed")
    if (packet["result_root"] != RUN or packet["old_result_root"] != OLD_RUN
            or packet["budget_plan"] != budget_plan(packet["intent"])
            or packet["runtime"] != runtime_record() or packet["source_files"] != source_locks(root)):
        raise ValueError("scope, runtime or implementation changed")
    old = committed_json(root, OLD_PACKET)
    if (packet["scientific_config"] != old["scientific_config"] or packet["streams"] != old["streams"]
            or packet["original_packet_sha256"] != old["packet_sha256"]):
        raise ValueError("original scientific design changed")
    for name, record in packet["input_files"].items():
        if file_record(root, name) != record:
            raise ValueError("frozen input changed: " + name)
    if (file_record(root, INTENT) != packet["intent_file"] or file_record(root, PROTOCOL) != packet["protocol"]
            or committed_json(root, INTENT) != packet["intent"]):
        raise ValueError("approval or protocol changed")
    subprocess.run(["git", "merge-base", "--is-ancestor", packet["implementation_commit"], "HEAD"], cwd=root, check=True)


def approved(root):
    packet, auth = committed_json(root, FROZEN), committed_json(root, AUTHORIZATION)
    if auth != authorization(packet):
        raise PermissionError("committed exact packet-bound authorization required")
    if (packet["workspace"] != str(Path(root).resolve()) or git(root, "branch", "--show-current") != BRANCH
            or git(root, "status", "--porcelain")):
        raise ValueError("clean declared worktree required")
    verify_bindings(root, packet)
    return packet, auth


class RecoveryAdmission:
    def __init__(self, root, packet, auth, claim, budget):
        self.root, self.packet, self.claim, self.budget = Path(root) / RUN, packet, claim, budget
        if (auth != authorization(packet) or claim["pid"] != os.getppid()
                or claim["packet_sha256"] != packet["packet_sha256"]
                or claim["authorization_sha256"] != digest(auth) or claim["started"] != budget.started
                or claim["clock_id"] != CLOCK_ID or budget.plan != packet["budget_plan"]):
            raise PermissionError("live exact evaluation claim required")
        self.pid, self.counts = os.getpid(), {}
        self.caps = dict(checkpoint_load=12, reference_checkpoint_load=3, layout_environment_build=3, episode_build=205)

    def __call__(self, operation):
        self.budget.check()
        if (os.getpid() != self.pid or os.getppid() != self.claim["pid"]
                or json.loads((self.root / "launcher/claim.json").read_text()) != self.claim
                or (self.root / "launcher/terminal.json").exists()):
            raise PermissionError("evaluation claim changed or terminal")
        phase = self.budget.active
        if operation in ("input_hash_verification", "reference_and_layout_build") and phase == "runtime_input_binding":
            return
        if (operation not in self.caps or self.counts.get(operation, 0) >= self.caps[operation]
                or not (isinstance(phase, str) and phase.startswith("final_evaluation/") if operation == "episode_build"
                        else phase == "runtime_input_binding")):
            raise PermissionError("undeclared or exhausted operation")
        count = self.counts.get(operation, 0) + 1
        write_json_once(self.root / f"launcher/operations/{operation}-{count:04d}.json",
            dict(operation=operation, count=count, phase=phase, before_operation=True, clock=shared_monotonic()))
        self.counts[operation] = count


@contextmanager
def no_optimizer_updates():
    """Defense in depth: the evaluator has neither an update path nor a budget."""
    from contextlib import ExitStack
    with ExitStack() as stack:
        for name in dir(torch.optim):
            cls = getattr(torch.optim, name)
            if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer) and "step" in cls.__dict__:
                stack.enter_context(patch.object(cls, "step", side_effect=PermissionError("evaluation forbids optimizer.step")))
        yield


class RecoveryCampaign:
    def __init__(self, root, packet, budget, backend, models, *, recorder=CohortRecorder, collection=CohortCollection):
        self.root, self.packet, self.budget, self.backend = Path(root), packet, budget, backend
        self.models, self.recorder_type, self.collection_type = models, recorder, collection
        self.index, self.serial, self.recorder = copy.deepcopy(packet["reused_index"]), 0, None
        self.completed = set()

    def status(self, stage, **values):
        self.serial += 1
        write_json_once(self.root / f"launcher/progress/{self.serial:06d}.json",
            dict(stage=stage, completed_evaluations=len(self.index), reused=11,
                 counts=self.budget.counts, clock=shared_monotonic(), **values))
        print(json.dumps(dict(stage=stage, completed_evaluations=len(self.index), **values)), flush=True)

    def episode(self, descriptor):
        b, role, world, seed, split, selection, name = descriptor
        if name in self.completed or descriptor not in descriptors(self.packet["streams"], b, role):
            raise ValueError("duplicate or undeclared evaluation")
        expected_scope = f"final_evaluation/block{b}/{role}"
        if self.budget.active != expected_scope:
            raise PermissionError("wrong owner scope")
        self.completed.add(name)
        key = f"block{b}/graph/{role if role in ROLES[:4] else 'own_frozen'}"
        owner = self.models[key]
        before = state_digest(owner.state_dict())
        prefix = self.backend.session(b, owner, seed, trajectory=name, split=split, selection=selection)
        self.recorder = self.recorder_type(self.root, "payload/episodes/" + name, prefix,
            self.packet["scientific_config"], block=b, role=role, world_index=world, seed=seed,
            representation="reference" if role in ROLES[4:] else "graph")
        run = self.collection_type(prefix, enabled=True, objective="none", split="test", trajectory_id=name,
            followup_action=lambda env: env.common_followup_request(prefix.producer.anchor_config),
            finish_prefix=self.recorder.finish_prefix, record_prefix=self.recorder.record_prefix,
            record_tail=self.recorder.record_tail)
        steps = 0
        while not run.closed:
            run.step(before_step=lambda: self.budget.debit_environment("trajectory"))
            steps += 1
        if steps != 63 or state_digest(owner.state_dict()) != before:
            raise ValueError("incomplete endpoint or changed deterministic model state")
        self.index.append(self.recorder.finish(run))
        self.recorder = None
        self.budget.check()
        write_json_once(self.root / f"launcher/boundaries/{len(self.index):03d}.json",
            dict(index=self.index[-1], completed_evaluations=len(self.index), trajectory=name,
                 model_state_sha256=before, budget=self.budget.snapshot()))
        self.status("episode_complete", block=b, role=role, world=world)


def child(root):
    packet, auth = approved(root)
    output = Path(root) / RUN
    claim = json.loads((output / "launcher/claim.json").read_text())
    if claim["head"] != git(root, "rev-parse", "HEAD"):
        raise PermissionError("execution commit changed")
    write_json_once(output / "launcher/child-claim.json", dict(pid=os.getpid(), ppid=os.getppid()))
    budget = DynamicCandidateBudget(output / "launcher/budget.jsonl", packet["budget_plan"], enabled=True, started=claim["started"])
    campaign = None
    try:
        with no_optimizer_updates():
            budget.begin("runtime_input_binding")
            admission = RecoveryAdmission(root, packet, auth, claim, budget)
            from src.rl.cohort_backend import CohortPatientBackend
            from src.rl.cohort_evaluation_models import restore_evaluation_owner
            backend = CohortPatientBackend(root, packet["scientific_config"], packet["streams"], admit_real_calls=admission)
            paths = set(packet["source_files"]) | set(packet["input_files"]) | {FROZEN, AUTHORIZATION, PROTOCOL, INTENT}
            for name in sorted(paths):
                budget.check()
                copy_verified(confined(root, name), output / "payload/locks/worktree" / name)
            for idx in packet["reused_index"]:
                for part in idx.values():
                    for record in part.values():
                        copy_verified(Path(root) / OLD_RUN / record["path"], output / record["path"])
            models = {}
            for b in BLOCKS:
                receipt = backend.prepare(b, packet["streams"]["environment"][str(b)]["layout"][0])
                write_json_once(output / f"payload/binding/block{b}.json", receipt)
                for role in ROLES[:4]:
                    key = f"block{b}/graph/{role}"
                    record = packet["models"][key]
                    admission("checkpoint_load")
                    state = load_envelope(Path(root) / record["path"])
                    models[key] = restore_evaluation_owner(state)
                    if models[key].contract != backend.producer(b).contract:
                        raise ValueError("saved model and original public input contract differ")
            write_json_once(output / "payload/model-bindings.json", dict(models=packet["models"],
                states={k: state_digest(v.state_dict()) for k, v in models.items()}, sealed_before_original_tests=True))
            campaign = RecoveryCampaign(output, packet, budget, backend, models)
            campaign.status("binding_complete")
            budget.finish()
            for b in BLOCKS:
                for role in ROLES:
                    budget.begin(f"final_evaluation/block{b}/{role}")
                    for descriptor in descriptors(packet["streams"], b, role):
                        campaign.episode(descriptor)
                    budget.finish()
            budget.begin("raw_verification")
            if len(campaign.index) != 216 or admission.counts != admission.caps:
                raise ValueError("incomplete evaluation or operation inventory")
            write_json_once(output / "payload/evaluation-index.json", campaign.index)
            report = verify_cohort_raw_bundle(output, campaign.index, packet["scientific_config"], packet["streams"])
            write_json_once(output / "payload/independent-verification.json", report)
            write_json_once(output / "payload/comparison.json", report["analysis"])
            ledger = read_dynamic_ledger(budget.path)
            counts = {k: ledger["counts"].get(k, 0) for k in ("environment", "optimizer")}
            if counts != {"environment": 12915, "optimizer": 0}:
                raise ValueError("evaluation call accounting differs")
            write_json_once(output / "payload/compute-accounting.json", dict(counts=counts, operations=admission.counts,
                reused_evaluations=11, new_evaluations=205, old_partial_calls_spent=17, old_attempt_remains_terminal=True))
            verify_bindings(root, packet)
            campaign.status("raw_verification_complete", decision=report["analysis"]["decision"])
            budget.finish()
            budget.begin("payload_archive")
            write_json_once(output / "payload/artifact-inventory.json", inventory(output / "payload"))
            receipt = create_archive(output / "payload", output / "archives/completed-payload.tar.gz")
            write_json_once(output / "launcher/archive-receipt.json", dict(archive=receipt, local_archive_verified=True,
                dropbox_exported=False, cloud_sync_verified=False, howard_access_verified=False))
            budget.finish()
            budget.begin("supervisor_closure")
            if inventory(output / "payload") != receipt["files"]:
                raise ValueError("payload changed after archive")
            verify_bindings(root, packet)
            write_json_once(output / "launcher/closure.json", dict(complete=True, payload_unchanged=True,
                comparison_verified=True, evaluations=216, new_training_jobs=0, automatic_followon=False))
            campaign.status("child_complete")
            budget.finish()
        budget.close()
        return 0
    except BaseException as error:
        if campaign is not None and campaign.recorder is not None:
            campaign.recorder.close_partial()
        write_json_once(output / "launcher/child-failure.json", dict(error=repr(error), automatic_retry=False))
        budget.close()
        return 1


def launch(root, started=None):
    started = shared_monotonic() if started is None else started
    packet, auth = approved(root)
    if shared_monotonic() - started >= 300:
        raise TimeoutError("binding cap exhausted before claim")
    output = Path(root) / RUN
    output.mkdir(parents=True, exist_ok=False)
    launcher = output / "launcher"
    write_json_once(launcher / "claim.json", dict(pid=os.getpid(), ppid=os.getppid(), head=git(root, "rev-parse", "HEAD"),
        packet_sha256=packet["packet_sha256"], authorization_sha256=digest(auth), started=started,
        clock_id=CLOCK_ID, automatic_retry=False))
    try:
        result = supervise_time_baseline([sys.executable, "-m", "experiments.scripts.run_cohort_evaluation_recovery", "--child"],
            cwd=root, launcher=launcher, plan=packet["budget_plan"], started=started)
        success = result["passed"] and (launcher / "stderr.log").stat().st_size == 0
        if success:
            closure = json.loads((launcher / "closure.json").read_text())
            ledger = read_dynamic_ledger(launcher / "budget.jsonl")
            counts = {k: ledger["counts"].get(k, 0) for k in ("environment", "optimizer")}
            success = (closure["complete"] and counts == {"environment": 12915, "optimizer": 0}
                and set(ledger["closed"]) == set(packet["budget_plan"]["sections"]) and ledger["active"] is None)
        success = success and shared_monotonic() <= result["final_deadline"]
        write_json_once(launcher / "terminal.json", dict(status="completed" if success else "failed",
            exit_code=0 if success else 1, scientific_completion_verified=bool(success), supervisor=result,
            automatic_retry=False, automatic_followon=False, dropbox_exported=False))
        if success:
            receipt = create_archive(launcher, output / "archives/completed-launcher.tar.gz")
            if shared_monotonic() > result["final_deadline"]:
                raise TimeoutError("closure archive exceeded cap")
            write_json_once(output / "closure-archive-receipt.json", dict(archive=receipt, local_archive_verified=True,
                dropbox_exported=False, cloud_sync_verified=False, howard_access_verified=False))
        return 0 if success else 1
    except BaseException as error:
        write_json_once(launcher / "launch-failure.json", dict(status="failed", error=repr(error), automatic_retry=False))
        return 1
