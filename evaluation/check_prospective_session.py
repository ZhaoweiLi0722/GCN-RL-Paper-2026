"""Six finite live-collector/update/resume cases, not a policy comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments.scripts.check_prospective_collector import local_source_hashes
from src.env.patient_capacity_planning import PatientConditionCapacityEnv, patient_env_config_from_dict
from src.models.prospective_forward_agent import ProspectiveForwardAgent
from src.rl.networks import torch
from src.rl.patient_replay_collector import PatientObservationProducer, evidence_digest
from src.rl.prospective_ddpg_kernel import KernelSettings, ProspectiveDDPGKernel, state_digest
from src.rl.prospective_patient_session import PatientLearningSession, scheduled_windows, restore_record

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/prospective_closed_loop_engineering_20260929.json"
PROTOCOL = "specs/2026-09-29-prospective-closed-loop-engineering/protocol.md"


def write_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_fixture(config):
    expected = {"kind": "bounded_live_collector_learner_resume_engineering", "enabled": True,
                "device": "cpu", "fixture_seed": 0, "max_steps": 6, "interrupt_after_steps": 4,
                "return_steps": 3, "gamma": .9, "reward_scale": 1e-5, "residual_scale": .1,
                "performance_evaluation": False, "scientific_launch_authorized": False}
    for name, value in expected.items():
        if config.get(name) != value or type(config.get(name)) is not type(value):
            raise ValueError(f"bounded N7 configuration differs: {name}")
    cases = [
        {"architecture": "graph", "message_mode": "physical", "hidden_width": 64,
         "episode_horizon": 8, "horizon_end": "truncation", "boundary": "collector_cutoff"},
        {"architecture": "graph", "message_mode": "self_only", "hidden_width": 64,
         "episode_horizon": 6, "horizon_end": "truncation", "boundary": "environment_truncation"},
        {"architecture": "flat", "message_mode": "physical", "hidden_width": 68,
         "episode_horizon": 6, "horizon_end": "terminal", "boundary": "terminal_mapping"},
    ]
    if (config["cases"] != cases or config["modes"] != ["online", "frozen"]
            or config["noise"] != {"mu": 0., "theta": .15, "sigma": .05}
            or KernelSettings(**config["learner"]) != KernelSettings(.001, .002, .05, 2, 4, 3, 3, True)):
        raise ValueError("declared cases/noise/learner settings differ")
    contents = {}
    for key in ("environment_source", "learner_evidence"):
        payload = (ROOT / config[key]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != config[key + "_sha256"]:
            raise ValueError(f"immutable {key} hash mismatch")
        contents[key] = json.loads(payload)
    if contents["learner_evidence"]["passed"] is not True:
        raise ValueError("N6 acceptance prerequisite missing")
    return contents["environment_source"]["env"]


def make_session(config, index, mode):
    env_config = load_fixture(config) | {"episode_horizon": config["cases"][index]["episode_horizon"]}
    if mode not in config["modes"]:
        raise ValueError("unknown mode")
    env = PatientConditionCapacityEnv(patient_env_config_from_dict(env_config), seed=0)
    producer = PatientObservationProducer(env, enabled=True, gamma=config["gamma"], reward_scale=config["reward_scale"])
    case = config["cases"][index]
    prototype = ProspectiveForwardAgent(producer.contract, enabled=True,
                                         **{key: case[key] for key in ("architecture", "message_mode", "hidden_width")},
                                         residual_scale=config["residual_scale"], seed=0)
    learner = ProspectiveDDPGKernel(prototype, KernelSettings(**config["learner"]), enabled=True, mode=mode, replay_seed=0)
    return PatientLearningSession(env, producer, learner, enabled=True, trajectory_id=f"n7-fixture-{index}-{mode}",
                                   max_steps=config["max_steps"], return_steps=config["return_steps"],
                                   horizon_end=case["horizon_end"], noise_options=config["noise"], noise_seed=0)


def advance(session, steps):
    for _ in range(steps):
        session.step()


def summary(session):
    learner, collector = session.learner, session.collector
    records = [restore_record(event["receipt"]["record"]) for event in session.events]
    windows, pending = scheduled_windows(records, learner.contract.replay, session.return_steps, collector.closed)
    return {"state_sha256": state_digest(session.state_dict()), "environment_sha256": evidence_digest(collector.env.state_dict()),
            "kernel_state_sha256": state_digest(learner.state_dict()), "weights_sha256": learner.agent.weights_digest(),
            "steps": collector.index, "closed": collector.closed, "updates": learner.total_updates,
            "pending_count": len(pending), "emitted_lengths": [len(window) for window in windows],
            "replay_size": len(learner.windows), "replay_position": learner.position,
            "routing_count": sum(event["receipt"]["execution"]["specimen_route_count"] for event in session.events),
            "all_clone_steps_exact": all(event["receipt"]["clone_verified"] for event in session.events)}


def worker(config, index, mode, checkpoint, output):
    session = make_session(config, index, mode)
    session.load(checkpoint)
    status = summary(session)
    if (status["steps"] != 4 or status["closed"] or status["pending_count"] != 2
            or status["updates"] != int(mode == "online") or status["replay_size"] != 2):
        raise ValueError("worker checkpoint is not the declared live four-step boundary")
    advance(session, 2)
    session.save(output.with_suffix(".pt"))
    write_json(output, {"events": session.events, "final": summary(session), "fresh_process": True})


def audit(config, output):
    load_fixture(config)
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for index, specification in enumerate(config["cases"]):
        for mode in config["modes"]:
            name = f"case-{index}-{mode}"
            continuous = make_session(config, index, mode)
            initial = summary(continuous)
            advance(continuous, 6)
            final = summary(continuous)
            continuous.save(output / f"{name}-continuous.pt")
            interrupted = make_session(config, index, mode)
            advance(interrupted, 4)
            boundary = summary(interrupted)
            checkpoint = output / f"{name}-prefix.pt"
            interrupted.save(checkpoint)
            if continuous.events[:4] != interrupted.events:
                raise AssertionError("continuous and interrupted prefixes differ")
            del interrupted
            worker_path = output / f"{name}-resumed.json"
            child = subprocess.run([sys.executable, "-m", "evaluation.check_prospective_session",
                                    "--config", str(CONFIG), "--resume-case", str(index), "--mode", mode,
                                    "--checkpoint", str(checkpoint), "--output", str(worker_path)],
                                   cwd=ROOT, text=True, capture_output=True, timeout=120)
            write_json(output / f"{name}-worker-log.json", {"exit_code": child.returncode,
                                                          "stdout": child.stdout, "stderr": child.stderr})
            if child.returncode != 0 or child.stderr:
                raise AssertionError(f"fresh worker failure/stderr: {child.stderr}")
            restored = json.loads(worker_path.read_text())
            if restored["events"] != continuous.events or restored["final"] != final:
                raise AssertionError("continuous/restarted complete session differs")
            if (not final["closed"] or final["pending_count"] or final["emitted_lengths"] != [3, 3, 3, 3, 2, 1]
                    or final["replay_size"] != 4 or final["replay_position"] != 2
                    or final["updates"] != (3 if mode == "online" else 0)):
                raise AssertionError("incorrect closure, replay or update schedule")
            if (final["weights_sha256"] == initial["weights_sha256"]) != (mode == "frozen"):
                raise AssertionError("online/frozen parameter-update invariant failed")
            if not final["all_clone_steps_exact"] or final["routing_count"] <= 0:
                raise AssertionError("fixture clone/routing coverage missing")
            before = state_digest(continuous.state_dict())
            try:
                continuous.step()
            except ValueError:
                pass
            else:
                raise AssertionError("closed session unexpectedly advanced")
            if state_digest(continuous.state_dict()) != before:
                raise AssertionError("closed-session rejection mutated state")
            cases.append({"case": name, "specification": specification, "mode": mode,
                          "initial": initial, "interruption": boundary, "final": final,
                          "events": continuous.events, "exact_fresh_process_resume": True,
                          "closed_session_refuses_steps": True})
    inventory = []
    for path in sorted(output.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if state_digest(payload["state"]) != payload["state_sha256"]:
            raise AssertionError("saved checkpoint checksum mismatch")
        inventory.append({"path": path.name, "bytes": path.stat().st_size,
                          "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "semantic_state_sha256": payload["state_sha256"]})
    write_json(output / "checkpoint_inventory.json", {"checkpoints": inventory, "git_tracking": "local_ignored_pt"})
    return {"kind": "bounded_real_session_mechanics_not_performance", "passed": True, "cases": cases,
            "primary_environment_steps": 72, "clone_verification_steps": 72,
            "kernel_updates": 18, "adam_steps": 36, "fresh_process_continuations": 6,
            "scientific_launch_authorized": False, "online_gain_claimed": False,
            "clinical_terminal_settlement_claimed": False, "source_environment_config_sha256": config["environment_source_sha256"],
            "source_learner_report_sha256": config["learner_evidence_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-case", type=int, choices=range(3))
    parser.add_argument("--mode", choices=("online", "frozen"))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.config.resolve() != CONFIG or args.output.exists():
        parser.error("committed N7 config and fresh output required")
    if any(x is not None for x in (args.resume_case, args.mode, args.checkpoint)) and any(
            x is None for x in (args.resume_case, args.mode, args.checkpoint)):
        parser.error("all three resume arguments required together")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", "src",
                                     "evaluation/check_prospective_session.py", str(CONFIG.relative_to(ROOT)), PROTOCOL],
                                    cwd=ROOT, text=True)
    if dirty.strip():
        parser.error("commit source/config/protocol before the recorded check")
    torch.set_num_threads(1)
    config = json.loads(CONFIG.read_text())
    if args.resume_case is not None:
        worker(config, args.resume_case, args.mode, args.checkpoint, args.output)
        return
    result = audit(config, args.output.resolve())
    result["execution_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result["config_sha256"] = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    result["source_sha256"] = local_source_hashes(tuple(sys.modules.values()), ROOT)
    result["runtime"] = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": str(torch.__version__),
                         "device": "cpu", "threads": torch.get_num_threads()}
    write_json(args.output / "check.json", result)
    print(json.dumps({"passed": True, "primary_steps": 72, "clone_steps": 72,
                      "kernel_updates": 18, "output": str(args.output)}))


if __name__ == "__main__":
    main()
