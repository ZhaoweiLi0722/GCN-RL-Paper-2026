"""One additional fixed-policy role on saved tapes; never trains a model."""

from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from src.env.patient_support_capacity import PatientSupportAction
from src.rl.capacity_native import CapacityPilotEnv, CapacityWorldTape, common_operation
from src.rl.capacity_pilot_runner import COST_KEYS, jsonable, trajectory_id
from src.rl.public_support_input import PublicSupportControlInput
from src.utils.research_archive import create_archive, sha256_file

SPEC = "specs/2026-10-08-mdl2-fixed-reference"
CONFIG = "experiments/configs/capacity_fixed_reference_20261008.json"
FROZEN = SPEC + "/frozen.json"
NEW_SOURCE = (
    "src/rl/capacity_fixed_reference.py",
    "src/rl/capacity_fixed_reference_analysis.py",
    "experiments/scripts/run_capacity_fixed_reference.py",
    "tests/test_capacity_fixed_reference.py",
    "tests/test_capacity_fixed_reference_analysis.py",
    CONFIG, SPEC + "/protocol.md", SPEC + "/approval-intent.json",
)


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(jsonable(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def file_record(root, relative):
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError("expected regular bound input: " + relative)
    return dict(path=relative, bytes=path.stat().st_size, sha256=sha256_file(path))


def verify_records(root, records):
    for relative, record in records.items():
        if file_record(root, relative) != record:
            raise ValueError("locked bytes changed: " + relative)


def runtime():
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                numpy=importlib.metadata.version("numpy"), torch=importlib.metadata.version("torch"))


def fixed_hours(epoch, config):
    if not 0 <= epoch < config["control_epochs"] + config["settlement_epochs"]:
        raise ValueError("outside fixed horizon")
    return tuple(config["fixed_hours"]) if epoch < config["control_epochs"] else (0.,) * 4


def validate_config(config):
    expected = dict(role="mdl2-fixed2", fixed_hours=[2., 2., 2., 2.], control_epochs=48,
                    settlement_epochs=16, blocks=5, conditions=3, worlds_per_condition_block=8,
                    max_trajectories=120, max_native_steps=7680, max_native_operations=7920,
                    optimizer_updates=0, neural_forwards=0, planner_steps=0, filter_transitions=0,
                    clones=0, attempts=1, automatic_retry=False, automatic_resume=False,
                    automatic_follow_on=False)
    if any(config.get(k) != value for k, value in expected.items()):
        raise PermissionError("fixed2 scope differs from explicit user approval")


def paired_inputs(root, config):
    old = root / config["old_run"] / "payload"
    rows = [read(path) for path in sorted((old / "summaries").glob("confirmation_evaluation-*-plain_h8.json"))]
    keys = {(r["world"]["block"], r["world"]["condition"], r["world"]["replicate"]) for r in rows}
    if keys != {(b, c, j) for b in range(5) for c in range(3) for j in range(8)} or len(rows) != 120:
        raise ValueError("exact original120worlds required")
    inputs, pairs = {}, []
    for h8 in rows:
        world = h8["world"]
        tape_path = old / "tapes" / (trajectory_id(world, "exogenous") + ".json")
        tape = CapacityWorldTape(**read(tape_path))
        if tape.world != world or tape.digest() != h8["tape_sha256"]:
            raise ValueError("tape digest/world mismatch")
        paths = [tape_path]
        references = []
        for role in config["reference_roles"]:
            path = old / "summaries" / (trajectory_id(world, role) + ".json")
            row = read(path)
            if (row["world"] != world or row["role"] != role or not row["settled"]
                    or row["tape_sha256"] != tape.digest() or row["optimizer_updates_during_trajectory"] != 0):
                raise ValueError("reference pairing/settlement mismatch")
            paths.append(path)
            references.append(dict(role=role, model_seal_sha256=row["model_seal_sha256"]))
        for path in paths:
            relative = path.relative_to(root).as_posix()
            inputs[relative] = file_record(root, relative)
        pairs.append(dict(world=world, tape_path=tape_path.relative_to(root).as_posix(),
                          tape_sha256=tape.digest(), references=references))
    return inputs, pairs


def prepare(root):
    root = Path(root).resolve()
    if git(root, "status", "--porcelain"):
        raise PermissionError("commit the amendment and implementation first")
    if git(root, "branch", "--show-current") != "codex/september-research-integration":
        raise PermissionError("wrong worktree branch")
    config, intent = read(root / CONFIG), read(root / SPEC / "approval-intent.json")
    validate_config(config)
    if intent["scope_approved"] is not True or intent["user_literal"] != "是的":
        raise PermissionError("explicit fixed2approval missing")
    old = read(root / config["old_frozen"])
    verify_records(root, old["source_files"])
    inputs, pairs = paired_inputs(root, config)
    for relative in (config["old_frozen"], config["old_run"] + "/payload/comparison.json",
                     config["old_run"] + "/terminal-readout/world-readout.json",
                     config["old_run"] + "/terminal-readout/verification.json"):
        inputs[relative] = file_record(root, relative)
    sources = dict(old["source_files"])
    sources.update({p: file_record(root, p) for p in NEW_SOURCE})
    packet = dict(scope=config["scope"], workspace=str(root), implementation_commit=git(root, "rev-parse", "HEAD"),
                  created_at_utc=datetime.now(timezone.utc).isoformat(), config=config, approval=intent,
                  original_packet_sha256=old["packet_sha256"], source_files=sources, inputs=inputs,
                  worlds=pairs, runtime=runtime(), old_config=old["config"],
                  reused_worlds=True, new_independent_test=False)
    write_once(root / FROZEN, packet)
    print("Prepared fixed2 supplement:120saved worlds, no scientific calls", flush=True)


class Budget:
    def __init__(self, root, config):
        self.root, self.config, self.started = Path(root), config, time.monotonic()
        self.world_started = None
        self.active = None
        self.epoch = 0
        self.counts = dict.fromkeys(("trajectories", "native_constructions", "construction_triggered_resets",
            "native_steps", "control_steps", "tail_steps", "total_native_operations", "total_optimizer_steps",
            "actor_optimizer_steps", "critic_optimizer_steps", "value_optimizer_steps", "fit_batches",
            "optimizer_example_presentations", "neural_forward_module_calls", "planner_total_model_epochs",
            "planner_total_decisions", "planner_candidate_rollouts", "estimator_receipt_updates",
            "estimator_hypothesis_transitions", "native_clones"), 0)

    def check(self, storage=False):
        now = time.monotonic()
        if now - self.started > self.config["wall_seconds"]:
            raise TimeoutError("fixed supplement wall cap")
        if self.world_started is not None and now - self.world_started > self.config["world_seconds"]:
            raise TimeoutError("fixed supplement world cap")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
        if rss > self.config["rss_bytes"]:
            raise MemoryError("fixed supplement RSS cap")
        if storage:
            files = [p for p in self.root.rglob("*") if p.is_file()]
            raw = sum(p.stat().st_size for p in files if "archives" not in p.relative_to(self.root).parts)
            archive = sum(p.stat().st_size for p in files if "archives" in p.relative_to(self.root).parts)
            if (raw > self.config["raw_bytes"] or archive > self.config["archive_bytes"]
                    or len(files) > self.config["file_cap"]):
                raise RuntimeError("fixed supplement storage/file cap")

    def begin(self, world):
        self.check()
        if self.counts["trajectories"] >= self.config["max_trajectories"]:
            raise RuntimeError("no extra trajectory")
        self.active, self.epoch, self.world_started = world, 0, time.monotonic()
        self.counts["trajectories"] += 1

    def native(self, kind):
        self.check()
        key, cap = {"construction": ("native_constructions", 120),
                    "construction_reset": ("construction_triggered_resets", 120),
                    "step": ("native_steps", self.config["max_native_steps"])}[kind]
        if (self.counts[key] >= cap or
                self.counts["total_native_operations"] >= self.config["max_native_operations"]):
            raise RuntimeError("native admission cap exhausted")
        self.counts[key] += 1
        self.counts["total_native_operations"] += 1
        if kind == "step":
            self.counts["control_steps" if self.epoch < 48 else "tail_steps"] += 1

    def event(self, name, **fields):
        value = dict(event=name, utc=datetime.now(timezone.utc).isoformat(),
                     elapsed=time.monotonic() - self.started, active=self.active, epoch=self.epoch,
                     counts=self.counts, **fields)
        with (self.root / "payload/progress.jsonl").open("a") as stream:
            stream.write(json.dumps(jsonable(value), sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def episode(payload, config, proposal, tape, budget):
    world, role = tape.world, config["role"]
    start_counts = dict(budget.counts)
    budget.begin(world)
    start, ident = time.monotonic(), trajectory_id(world, role)
    env = CapacityPilotEnv(proposal, tape, budget.native)
    public = PublicSupportControlInput.capture(env)
    costs, lost_ids = [], set()
    budget.event("trajectory_started")
    with gzip.open(payload / "raw" / (ident + ".jsonl.gz"), "xt") as stream:
        for epoch in range(64):
            budget.epoch = epoch
            hours = fixed_hours(epoch, config)
            base = common_operation(public, proposal, tail=epoch >= 48)
            _, reward, done, info = env.step(PatientSupportAction(tuple(map(float, base)), hours))
            cost = float(info["cost"])
            components = {k: float(info[k]) for k in COST_KEYS}
            if (not math.isfinite(cost) or cost < 0 or bool(done) != (epoch == 63)
                    or not math.isclose(math.fsum(components.values()), cost, rel_tol=1e-12, abs_tol=1e-6)
                    or not math.isclose(-float(reward), cost, rel_tol=1e-12, abs_tol=1e-6)):
                raise ValueError("raw cost/reward/horizon mismatch")
            following = PublicSupportControlInput.capture(env)
            patients = following.operations.patients
            losses = {p.patient_id for p in patients if p.status == "lost"}
            committed = info["support_public_receipt"]["committed_hours"]
            if not lost_ids.issubset(losses) or not np.allclose(committed, hours, atol=1e-12, rtol=0):
                raise ValueError("fixed commitment or absorbing loss mismatch")
            row = dict(epoch=epoch, world=world, role=role, cost=cost, reward=float(reward),
                       components=components, new_lost_patients=len(losses - lost_ids),
                       patient_records=patients, requested_hours=hours, executed_hours=committed,
                       public_input=public, info=info)
            stream.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + "\n")
            costs.append(cost)
            lost_ids, public = losses, following
            if (epoch + 1) % 8 == 0:
                stream.flush()
                os.fsync(stream.fileno())
                budget.check(storage=True)
                budget.event("epoch_boundary")
    settlement = env.settlement()
    if not settlement["settled"] or settlement["lost"] != len(lost_ids):
        raise ValueError("fixed world did not fully settle")
    summary = dict(world=world, role=role, raw_path="raw/" + ident + ".jsonl.gz",
                   cost=math.fsum(costs), lost=len(lost_ids), settled=True, settlement=settlement,
                   tape_sha256=tape.digest(), model_seal_sha256=None, optimizer_updates_during_trajectory=0,
                   elapsed=time.monotonic() - start,
                   compute={k: v - start_counts[k] for k, v in budget.counts.items()})
    write_once(payload / "summaries" / (ident + ".json"), summary)
    budget.world_started = None
    budget.event("trajectory_completed", summary=summary)
    return summary


def run(root):
    from src.rl.capacity_fixed_reference_analysis import analyze
    root = Path(root).resolve()
    if git(root, "status", "--porcelain"):
        raise PermissionError("commit frozen packet before execution")
    packet = read(root / FROZEN)
    if (packet["workspace"] != str(root) or packet["runtime"] != runtime()
            or git(root, "branch", "--show-current") != "codex/september-research-integration"):
        raise PermissionError("workspace/runtime differs from prepared packet")
    verify_records(root, packet["source_files"])
    verify_records(root, packet["inputs"])
    subprocess.run(["git", "merge-base", "--is-ancestor", packet["implementation_commit"], "HEAD"], cwd=root, check=True)
    config, proposal = packet["config"], packet["old_config"]
    validate_config(config)
    output = root / config["result_root"]
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "payload"
    for name in ("raw", "summaries", "authority", "tapes"):
        (payload / name).mkdir(parents=True, exist_ok=False)
    budget = Budget(output, config)
    write_once(output / "claim.json", dict(pid=os.getpid(), ppid=os.getppid(),
        command=sys.argv, head=git(root, "rev-parse", "HEAD"), frozen_sha256=sha256_file(root / FROZEN),
        started_at_utc=datetime.now(timezone.utc).isoformat(), no_retry=True))
    shutil.copyfile(root / FROZEN, payload / "authority/frozen.json")
    for relative in packet["source_files"]:
        target = payload / "authority/source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    status, error = "failed", None
    def timeout(signum, frame):
        raise TimeoutError("fixed supplement hard wall timeout")
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, config["wall_seconds"])
    try:
        for item in packet["worlds"]:
            path = root / item["tape_path"]
            shutil.copyfile(path, payload / "tapes" / path.name)
            tape = CapacityWorldTape(**read(path))
            if tape.digest() != item["tape_sha256"] or tape.world != item["world"]:
                raise ValueError("tape changed since preparation")
            summary = episode(payload, config, proposal, tape, budget)
            print(json.dumps(dict(event="world_completed", completed=budget.counts["trajectories"],
                elapsed=time.monotonic() - budget.started, last_seconds=summary["elapsed"])), flush=True)
        if (budget.counts["native_steps"] != 7680 or budget.counts["total_native_operations"] != 7920
                or budget.counts["trajectories"] != 120):
            raise ValueError("incomplete fixed supplement allocation")
        budget.event("all_trajectories_complete")
        result = analyze(payload, root / config["old_run"] / "payload")
        write_once(payload / "comparison.json", result)
        write_once(payload / "completion.json", dict(counts=budget.counts, trajectories=120,
            new_independent_test=False, old_results_unchanged=True, automatic_follow_on=False))
        verify_records(root, packet["source_files"])
        verify_records(root, packet["inputs"])
        budget.check(storage=True)
        budget.event("analysis_complete")
        status = "completed"
    except BaseException as exc:
        error = dict(error=repr(exc), traceback=traceback.format_exc())
        write_once(output / "failure.json", error)
    finally:
        budget.world_started = None
        budget.event("terminal", status=status, error=error)
        try:
            receipt = create_archive(payload, output / "archives/payload.tar.gz")
            write_once(output / "archive-receipt.json", receipt)
            budget.check(storage=True)
        except BaseException as exc:
            status = "failed"
            write_once(output / "archive-failure.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        write_once(output / "terminal.json", dict(status=status, counts=budget.counts,
            elapsed_seconds=time.monotonic() - budget.started, scientific_completion_verified=status == "completed",
            automatic_retry=False, automatic_resume=False))
    print(json.dumps(read(output / "terminal.json")), flush=True)
    if status != "completed":
        raise SystemExit(1)
