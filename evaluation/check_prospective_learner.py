"""Bounded synthetic optimizer/resume check. Never constructs an environment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from evaluation.audit_prospective_design import load_contract
from src.models.matched_inputs import ObservationBatch
from src.models.prospective_forward_agent import ProspectiveForwardAgent
from src.rl.networks import torch
from src.rl.prospective_adapter import ReplayInputContract, pack_actor_state, completed_segment_windows
from src.rl.prospective_ddpg_kernel import KernelSettings, ProspectiveDDPGKernel, state_digest
from src.rl.validated_returns import OneStepRecord, ReplaySemantics

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/prospective_learner_engineering_20260929.json"
PROTOCOL = "specs/2026-09-29-prospective-learner-engineering/protocol.md"
SOURCES = ("evaluation/check_prospective_learner.py", "evaluation/audit_prospective_design.py",
           "src/rl/prospective_ddpg_kernel.py", "src/rl/prospective_adapter.py",
           "src/rl/validated_returns.py", "src/rl/training_state.py",
           "src/models/prospective_forward_agent.py", "src/models/matched_inputs.py",
           "src/models/gcn.py", "src/rl/networks.py")


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def fixture(config, root=ROOT):
    """Use upstream dimensions only, with new synthetic provenance and values."""
    expected = {"kind": "bounded_ddpg_update_resume_engineering", "enabled": True,
                "device": "cpu", "dtype": "float32", "fixture_seed": 0,
                "synthetic_records": 12, "prefix_records": 6, "interruption_after_updates": 3,
                "residual_scale": .1, "environment_steps": 0, "performance_evaluation": False,
                "scientific_launch_authorized": False}
    for key, value in expected.items():
        if config.get(key) != value or type(config.get(key)) is not type(value):
            raise ValueError(f"bounded engineering config differs: {key}")
    settings = KernelSettings(**config["learner"])
    if settings != KernelSettings(.001, .002, .05, 4, 8, 6, 1, True):
        raise ValueError("recorded check must use the declared finite learner settings")
    cases = [{"architecture": "graph", "message_mode": mode, "hidden_width": 64}
             for mode in ("physical", "self_only")]
    cases.append({"architecture": "flat", "message_mode": "physical", "hidden_width": 68})
    if config["cases"] != cases:
        raise ValueError("recorded matrix differs")
    old = load_contract(root / config["schema_source"], config["schema_source_sha256"])
    content = (root / config["count_source"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != config["count_source_sha256"]:
        raise ValueError("N5 count-source hash mismatch")
    counts = json.loads(content)
    if not counts["parameter_preflight_passed"] or counts["matching"]["selected"]["width"] != 68:
        raise ValueError("N5 matched-width prerequisite failed")
    inputs = replace(old.inputs, definition_id="invented-optimizer-fixture-v1")
    semantics = ReplaySemantics("absolute_environment", "invented-numeric-reward-v1", 1., 1.,
                                inputs.definition_id + "/actor-flat", inputs.definition_id + "/action",
                                old.replay.state_dim, old.replay.action_dim, True)
    binding = ReplayInputContract(inputs, semantics)
    n, f, g, a = (len(inputs.node_ids), len(inputs.node_feature_names),
                  len(inputs.global_feature_names), len(inputs.action_names))
    links = torch.zeros(1, n, n)
    for index in range(n - 1):
        links[0, index, index + 1] = links[0, index + 1, index] = 1.

    def state(episode, step):
        nodes = torch.arange(1, n * f + 1, dtype=torch.float32).reshape(1, n, f) / (n * f)
        nodes = nodes + step * .01 + episode * .02
        observation = ObservationBatch(inputs, nodes, torch.full((1, g), step / 6.), links)
        anchor = torch.full((1, a), .01 * episode)
        return tuple(pack_actor_state(observation, anchor, binding)[0].tolist())

    windows = []
    for episode in range(2):
        rows = []
        for step in range(6):
            rows.append(OneStepRecord(
                semantics, "invented-numbers-not-simulation", "trajectory", f"invented-{episode}", step,
                f"invented-{episode}-{step}", f"invented-{episode}-{step + 1}", state(episode, step),
                tuple(((-1) ** index) * .02 * (step + 1) for index in range(a)),
                -.2 - .03 * (episode * 6 + step), state(episode, step + 1),
                terminated=episode == 0 and step == 5, truncated=episode == 1 and step == 5))
        windows.extend(completed_segment_windows(rows, semantics, max_steps=1))
    return binding, settings, tuple(windows)


def kernel(config, binding, settings, index, mode="online"):
    prototype = ProspectiveForwardAgent(binding, enabled=True, **config["cases"][index],
                                         residual_scale=config["residual_scale"], seed=config["fixture_seed"])
    return ProspectiveDDPGKernel(prototype, settings, enabled=True, mode=mode,
                                 replay_seed=config["fixture_seed"])


def update_rows(learner, count):
    rows = []
    for _ in range(count):
        diagnostic = learner.update()
        diagnostic["state_sha256"] = state_digest(learner.state_dict())
        rows.append(diagnostic)
    return rows


def final_summary(learner, windows):
    batch = learner._prepare(windows)
    request = learner.agent.policy_tensors(batch.current_actor)[2]
    state = learner.state_dict()
    return {"state_sha256": state_digest(state), "weights_sha256": learner.agent.weights_digest(),
            "actor_sha256": state_digest(learner.agent.actor.state_dict()),
            "critic_sha256": state_digest(learner.agent.critic.state_dict()),
            "gate_sha256": state_digest(learner.agent.gate.state_dict()),
            "target_gate_sha256": state_digest(learner.agent.target_gate.state_dict()),
            "total_updates": learner.total_updates, "replay_size": len(learner.windows),
            "replay_position": learner.position, "requests": request.tolist()}


def resume_worker(config, index, checkpoint, output):
    binding, settings, windows = fixture(config)
    learner = kernel(config, binding, settings, index)
    learner.load(checkpoint)
    if learner.total_updates != 3 or len(learner.windows) != 6 or learner.position != 6:
        raise ValueError("worker requires the declared three-update prefix")
    learner.add_windows(windows[6:])
    rows = update_rows(learner, 3)
    final = final_summary(learner, windows)
    learner.save(output.with_suffix(".pt"))
    write_json(output, {"updates": rows, "final": final, "fresh_process": True})


def audit(config, output, config_path=CONFIG):
    binding, settings, windows = fixture(config)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cases = []
    for index, specification in enumerate(config["cases"]):
        uninterrupted = kernel(config, binding, settings, index)
        frozen = kernel(config, binding, settings, index, "frozen")
        original = final_summary(frozen, windows)
        if original["weights_sha256"] != uninterrupted.agent.weights_digest():
            raise AssertionError("online/frozen initial weights differ")
        uninterrupted.add_windows(windows[:6])
        continuous = update_rows(uninterrupted, 3)
        uninterrupted.add_windows(windows[6:])
        continuous += update_rows(uninterrupted, 3)
        final = final_summary(uninterrupted, windows)
        uninterrupted.save(output / f"case-{index}-continuous.pt")
        interrupted = kernel(config, binding, settings, index)
        interrupted.add_windows(windows[:6])
        prefix = update_rows(interrupted, 3)
        checkpoint = output / f"case-{index}-prefix.pt"
        interrupted.save(checkpoint)
        del interrupted
        worker_path = output / f"case-{index}-resumed.json"
        child = subprocess.run([sys.executable, "-m", "evaluation.check_prospective_learner",
                                "--config", str(config_path), "--resume-case", str(index),
                                "--checkpoint", str(checkpoint), "--output", str(worker_path)],
                               cwd=ROOT, text=True, capture_output=True, timeout=120)
        write_json(output / f"case-{index}-worker-log.json",
                   {"exit_code": child.returncode, "stdout": child.stdout, "stderr": child.stderr})
        if child.returncode != 0 or child.stderr:
            raise AssertionError(f"resume worker failed or emitted stderr: {child.stderr}")
        resumed = json.loads(worker_path.read_text())
        if continuous != prefix + resumed["updates"] or final != resumed["final"]:
            raise AssertionError("continuous and fresh-process resumed states/diagnostics differ")
        if original != final_summary(frozen, windows):
            raise AssertionError("frozen reference changed")
        if any(final[key] == original[key] for key in ("actor_sha256", "critic_sha256")):
            raise AssertionError("fixture did not exercise actual actor and critic changes")
        if any(final[key] != original[key] for key in ("gate_sha256", "target_gate_sha256")):
            raise AssertionError("fixed gate changed")
        if final["replay_position"] != 4 or final["replay_size"] != 8:
            raise AssertionError("fixture did not exercise replay wraparound")
        cases.append({"specification": specification, "initial": original, "updates": continuous,
                      "final": final, "exact_fresh_process_resume": True, "frozen_unchanged": True,
                      "actor_and_critic_changed": True, "fixed_gates_unchanged": True,
                      "manifest_sha256": uninterrupted.manifest_sha256})
    return {"kind": "synthetic_update_resume_not_performance", "contract": asdict(binding),
            "cases": cases, "kernel_updates": 36, "adam_steps": 72, "environment_steps": 0,
            "online_gain_claimed": False, "scientific_launch_authorized": False,
            "full_environment_resume_claimed": False, "passed": True,
            "schema_source_sha256": config["schema_source_sha256"],
            "count_source_sha256": config["count_source_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-case", type=int, choices=range(3))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.config.resolve() != CONFIG or args.output.exists():
        parser.error("use the committed N6 config and a new output path")
    if (args.resume_case is None) != (args.checkpoint is None):
        parser.error("checkpoint and resume case must be supplied together")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--",
                                     "src", *SOURCES, str(CONFIG.relative_to(ROOT)), PROTOCOL],
                                    cwd=ROOT, text=True)
    if dirty.strip():
        parser.error("commit source/config/protocol before the recorded check")
    config = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    if args.resume_case is not None:
        resume_worker(config, args.resume_case, args.checkpoint, args.output)
        return
    result = audit(config, args.output.resolve())
    result["execution_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result["config_sha256"] = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    result["audited_source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                       for name in SOURCES}
    result["runtime"] = {"python": sys.version.split()[0], "torch": str(torch.__version__),
                         "numpy": np.__version__, "device": "cpu", "threads": torch.get_num_threads()}
    write_json(args.output / "check.json", result)
    print(json.dumps({"passed": result["passed"], "kernel_updates": result["kernel_updates"],
                      "environment_steps": 0, "output": str(args.output)}))


if __name__ == "__main__":
    main()
