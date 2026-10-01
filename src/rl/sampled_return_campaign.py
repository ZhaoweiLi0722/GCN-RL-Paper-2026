"""Single artificial packet, guarded by a separate committed authorization.

No authorization file is supplied by engineering preparation. This runner does
not create one, change the draft, tune a failed result, or use a patient env.
"""

import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np

from src.rl.actor_positive_control import prior_evidence
from src.rl.candidate_calibration_engineering import read, scores
from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.networks import torch
from src.rl.sampled_return_control import (CONFIG, PROTOCOL, ArtificialBudget, collect_samples,
    make_models, observed_reward, public_fixture, snapshot, update_pair)
from src.rl.sampled_return_verification import actor_metrics, evaluation_metrics
from src.utils.research_archive import inventory, sha256_file


AUTHORIZATION = "specs/2026-10-01-sampled-return-control/execution_authorization.json"
SCOPE = "one-artificial-sampled-return-packet-no-patient-calls"
CAPS = {"fits": 9, "actor_calls_per_fit": 128, "critic_calls_per_fit": 128,
        "observations_per_fit": 1536, "total_optimizer_calls": 2304,
        "total_training_observations": 13824, "numerical_wall_seconds": 1800, "attempts": 1}


def runtime():
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": "cpu", "dtype": "float32", "torch_threads": 1}


def validate_config(config):
    if (not config["engineering_only"] or config["artificial_execution_authorized"]
            or config["scientific_execution_authorized"] or config["patient_calls_authorized"] != 0
            or config["registered_with_patient_runner"] or config["automatic_retry"]):
        raise ValueError("preserve the non-authorizing draft and artificial-only scope")
    if config["caps"] != CAPS or config["representations"] != ["graph", "self_only", "flat"] or config["initialization_seeds"] != [101, 102, 103]:
        raise ValueError("fixed nine-fit scope and caps required")
    s = config["sampling"]
    if s != {"rollouts_per_fit": 32, "context_repetitions": 4, "minibatch_size": 12, "epochs": 1,
             "sampling_seed_offset": 610000, "shuffle_seed_offset": 620000, "normalize_advantages_once_per_rollout": True}:
        raise ValueError("fixed sampling schedule required")
    if len(config["training_times"])*len(config["cues"]) != 12 or len(config["heldout_times"])*len(config["cues"]) != 8:
        raise ValueError("fixed context counts required")
    if config["evaluation"] != {"contexts_per_fit": 8, "actor_snapshots_per_fit": 2, "critic_snapshots_per_fit": 1,
            "all_finals_sealed_before_evaluation": True, "oracle_table_available_only_to_generator_and_evaluator": True}:
        raise ValueError("fixed final evaluation required")
    if config["output"] != "results/candidate_sampled_return_control_20261001":
        raise ValueError("one fixed output location required; no alternate attempt")


def validate_authorization(config, authorization, *, config_sha, protocol_sha, source_hashes, current_runtime):
    validate_config(config)
    if (authorization.get("scope") != SCOPE or authorization.get("decision") != "approved_by_zhaowei"
            or authorization.get("implementation_ready") is not True
            or not isinstance(authorization.get("user_approval_evidence"), str)
            or not authorization["user_approval_evidence"].strip()
            or authorization.get("caps") != CAPS
            or authorization.get("config_sha256") != config_sha
            or authorization.get("protocol_sha256") != protocol_sha
            or authorization.get("source_hashes") != source_hashes
            or authorization.get("runtime") != current_runtime):
        raise ValueError("explicit exact-scope approval and frozen source/runtime locks required")
    commit = authorization.get("implementation_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("exact frozen implementation commit required")


def historical_evidence(root):
    previous = prior_evidence(root)
    actor = inventory(root / "results/candidate_actor_positive_control_20261001")
    if actor != read(root / "reports/2026-10-01-actor-positive-control/verification.json")["artifact_inventory"]:
        raise ValueError("closed actor-control evidence changed")
    if previous != read(root / "results/candidate_actor_positive_control_20261001/prior-evidence.json"):
        raise ValueError("historical trees differ from the closed actor-control input snapshot")
    return previous | {"actor_positive_control": actor}


def preflight(root):
    """Read-only: reject missing approval before constructing any artificial data."""
    root = Path(root).resolve()
    config, authorization = read(root / CONFIG), read(root / AUTHORIZATION)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("commit implementation, tests, protocol and authorization first")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    if branch != "codex/september-research-integration":
        raise ValueError("wrong research worktree branch")
    tracked = subprocess.check_output(["git", "ls-files", "src", "experiments/scripts", "tests", "AGENTS.md", CONFIG, PROTOCOL],
                                     cwd=root, text=True).splitlines()
    hashes = {p: sha256_file(root / p) for p in tracked}
    validate_authorization(config, authorization, config_sha=sha256_file(root / CONFIG),
                           protocol_sha=sha256_file(root / PROTOCOL), source_hashes=hashes, current_runtime=runtime())
    commit = authorization["implementation_commit"]
    subprocess.check_call(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=root)
    changed = subprocess.check_output(["git", "diff", "--name-only", commit, "HEAD", "--", *tracked], cwd=root, text=True)
    if changed.strip():
        raise ValueError("execution differs from frozen implementation")
    if subprocess.check_output(["git", "ls-files", AUTHORIZATION], cwd=root, text=True).strip() != AUTHORIZATION:
        raise ValueError("authorization must itself be committed")
    if (root / config["output"]).exists():
        raise FileExistsError("attempt exists; no resume, overwrite or retry")
    return config, authorization, hashes, historical_evidence(root)


class Recorder:
    def __init__(self, output, config, *, clock=time.monotonic):
        self.output, self.config, self.clock = Path(output), config, clock
        self.started, self.events, self.charges = clock(), 0, 0
        self.budget = ArtificialBudget(config)

    def check_time(self):
        if self.clock()-self.started >= self.config["caps"]["numerical_wall_seconds"]:
            raise TimeoutError("single-attempt numerical cap exhausted")

    def event(self, kind, **payload):
        self.events += 1
        write_json_once(self.output / "events" / f"{self.events:05d}.json",
            {"sequence": self.events, "kind": kind, "elapsed_seconds": self.clock()-self.started, **payload})

    def charge(self, fixture, kind, amount):
        self.check_time()
        def persist(receipt):
            self.charges += 1
            relative = f"charges/{self.charges:05d}.json"
            write_json_once(self.output / relative, receipt)
            self.event("charge", path=relative)
        self.budget.charge(fixture, kind, amount, persist)


def record_state(recorder, relative, actor, critic, optimizers, generators, *, batch,
                 permutation, next_minibatch, fit, rollout):
    state = snapshot(actor, critic, optimizers, generators, batch=batch, permutation=permutation,
                     next_minibatch=next_minibatch, budget=recorder.budget)
    state.update({"config": recorder.config, "fixture": fit, "rollout": rollout,
                  "elapsed_seconds": recorder.clock()-recorder.started,
                  "critic_parameter_count": sum(p.numel() for p in critic.parameters())})
    save_envelope(recorder.output / relative, state)
    return sha256_file(recorder.output / relative)


def execute(root, config, authorization, hashes, before):
    """Internal orchestration; call run, never bypass its read-only preflight."""
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(output, config)
    old_threads = torch.get_num_threads()
    fit, rollout, next_minibatch = None, 0, 0
    actor = critic = optimizers = generators = batch = permutation = None
    try:
        torch.set_num_threads(1)
        write_json_once(output / "claim.json", {"scope": SCOPE, "pid": os.getpid(), "ppid": os.getppid(),
            "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime(),
            "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "authorization_sha256": sha256_file(root / AUTHORIZATION), "source_hashes": hashes})
        write_json_once(output / "authorization.json", authorization)
        write_json_once(output / "config.json", config)
        write_json_once(output / "prior-evidence.json", before)
        observations, bank = public_fixture(config, config["training_times"])
        trained, seals = {}, {}
        for representation in config["representations"]:
            for seed in config["initialization_seeds"]:
                recorder.check_time()
                fit, rollout, next_minibatch = f"{representation}-{seed}", 0, 0
                batch = permutation = None
                actor, critic = make_models(config, bank, representation, seed)
                frozen = copy.deepcopy(actor)
                initial_sha = actor.snapshot_sha256()
                optimizers = [torch.optim.Adam(m.parameters(), lr=config["optimizer"]["learning_rate"]) for m in (actor, critic)]
                generators = [torch.Generator().manual_seed(config["sampling"][key]+seed)
                              for key in ("sampling_seed_offset", "shuffle_seed_offset")]
                args = (recorder, actor, critic, optimizers, generators)
                def save(relative):
                    return record_state(args[0], relative, *args[1:], batch=batch, permutation=permutation,
                                        next_minibatch=next_minibatch, fit=fit, rollout=rollout)
                save(f"{fit}/initial.pt")
                recorder.event("fit_started", fixture=fit, initial_actor_sha256=initial_sha)
                coverage = {f"{i}:{a}": 0 for i in range(12) for a in range(6)}
                observed_returns = []
                for rollout in range(32):
                    recorder.check_time()
                    batch = permutation = None
                    next_minibatch = 0
                    save(f"{fit}/rollout-{rollout:02d}/before.pt")
                    batch = collect_samples(actor, critic, observations, bank, sampling_rng=generators[0], repetitions=4,
                        reward_observer=lambda o, b, a: observed_reward(config, o, b, a),
                        charge_observations=lambda n: recorder.charge(fit, "observations", n))
                    permutation = torch.randperm(48, generator=generators[1])
                    raw = {key: value.tolist() for key, value in vars(batch).items()}
                    relative = f"{fit}/rollout-{rollout:02d}/samples.json"
                    write_json_once(output / relative, raw)
                    save(f"{fit}/rollout-{rollout:02d}/collected.pt")
                    recorder.event("samples", fixture=fit, rollout=rollout, path=relative)
                    observed_returns.extend(raw["returns"])
                    for i, a in zip(raw["context_indices"], raw["actions"]):
                        coverage[f"{i}:{a}"] += 1
                    for index in range(4):
                        recorder.check_time()
                        selected = permutation[12*index:12*(index+1)]
                        metrics = update_pair(actor, critic, *optimizers, observations, bank, batch, selected,
                            config["optimizer"], charge_optimizer=lambda kind: recorder.charge(fit, kind, 1))
                        next_minibatch = index+1
                        prefix = f"{fit}/rollout-{rollout:02d}/update-{index}"
                        write_json_once(output / (prefix+".json"), {"selected": selected.tolist(), **metrics})
                        digest = save(prefix+".pt")
                        recorder.event("update", fixture=fit, rollout=rollout, minibatch=index,
                                       state=prefix+".pt", sha256=digest)
                if frozen.snapshot_sha256() != initial_sha:
                    raise ValueError("frozen actor changed")
                seals[fit] = save(f"{fit}/final.pt")
                record = {"initial_actor_sha256": initial_sha, "final_actor_sha256": actor.snapshot_sha256(),
                    "frozen_actor_sha256": frozen.snapshot_sha256(), "coverage": coverage,
                    "training_return_mean": math.fsum(observed_returns)/len(observed_returns),
                    "counts": recorder.budget.by_fit[fit], "actor_manifest": actor.manifest(),
                    "actor_parameter_count": sum(p.numel() for p in actor.parameters()),
                    "critic_parameter_count": sum(p.numel() for p in critic.parameters())}
                write_json_once(output / f"{fit}/training.json", record)
                recorder.event("fit_sealed", fixture=fit, sha256=seals[fit])
                trained[fit] = actor, critic, frozen, record
                print(json.dumps({"fixture": fit, "budget": recorder.budget.state()}), flush=True)
        write_json_once(output / "all-models-sealed.json", {"models": seals, "budget": recorder.budget.state()})
        recorder.event("all_models_sealed", models=seals)
        heldout, heldout_bank = public_fixture(config, config["heldout_times"])
        checks = {}
        for name, (final, value, frozen, record) in trained.items():
            recorder.check_time()
            with torch.no_grad():
                final_logits, _ = scores(final, heldout, heldout_bank)
                frozen_logits, _ = scores(frozen, heldout, heldout_bank)
                values = [float(value(o, heldout_bank)) for o in heldout]
            raw = {"final_logits": final_logits.tolist(), "frozen_logits": frozen_logits.tolist(),
                   "critic_values": values}
            write_json_once(output / name / "heldout.json", raw)
            checks[name] = {"continued": evaluation_metrics(raw["final_logits"], values,
                record["training_return_mean"], record["coverage"], config),
                "frozen": actor_metrics(raw["frozen_logits"], config)}
            recorder.event("evaluation", fixture=name)
        recorder.check_time()
        numerical_seconds = recorder.clock()-recorder.started
        if recorder.budget.observations != 13824 or recorder.budget.optimizer_calls != 2304:
            raise ValueError("incomplete fixed packet")
        if historical_evidence(root) != before or any(sha256_file(root / p) != h for p, h in hashes.items()):
            raise ValueError("historical evidence or frozen source changed")
        if sha256_file(root / AUTHORIZATION) != read(output / "claim.json")["authorization_sha256"]:
            raise ValueError("authorization changed")
        terminal = {"status": "completed", "exit_code": 0, "checks": checks,
            "engineering_passed": all(v["continued"]["passed"] for v in checks.values()),
            "budget": recorder.budget.state(), "numerical_seconds": numerical_seconds,
            "new_patient_calls": 0, "new_patient_fits": 0, "artificial_fits": 9,
            "decision": "artificial_diagnostic_only_no_patient_launch"}
        write_json_once(output / "terminal.json", terminal)
        write_json_once(output / "inventory.json", inventory(output))
        print(json.dumps({k: v for k, v in terminal.items() if k != "checks"}), flush=True)
        return terminal
    except BaseException:
        error, checkpoint_error = traceback.format_exc(), None
        if actor is not None and optimizers is not None:
            try:
                record_state(recorder, "partial-on-failure.pt", actor, critic, optimizers, generators,
                    batch=batch, permutation=permutation, next_minibatch=next_minibatch, fit=fit, rollout=rollout)
            except BaseException:
                checkpoint_error = traceback.format_exc()
        write_json_once(output / "terminal-failure.json", {"status": "failed", "traceback": error,
            "partial_checkpoint_error": checkpoint_error, "budget": recorder.budget.state(),
            "retry_permitted": False, "elapsed_seconds": recorder.clock()-recorder.started})
        raise
    finally:
        torch.set_num_threads(old_threads)


def run(root):
    root = Path(root).resolve()
    return execute(root, *preflight(root))
