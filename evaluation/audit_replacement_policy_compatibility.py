"""R5 saved-observation inference audit. Never constructs or steps an env."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import traceback

import numpy as np
import torch

from src.models.graph_features import flat_state_to_node_features
from src.rl.strict_frozen_policy import (
    POLICY_MODULES, StrictFrozenPolicy, load_policy_payload_strict,
)
from src.rl.training_state import training_contract_sha256
from src.utils.research_archive import sha256_file


DEFAULT_CONFIG = "experiments/configs/replacement_policy_compatibility_20260929.json"


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def verify_inputs(spec):
    manifest_path = Path(spec["archive_manifest"])
    if sha256_file(manifest_path) != spec["archive_manifest_sha256"]:
        raise ValueError("R4 manifest changed")
    if sha256_file(Path(spec["r3_design"])) != spec["r3_design_sha256"]:
        raise ValueError("Historical R3 design changed")
    manifest = json.loads(manifest_path.read_text())
    payload = Path(spec["payload_root"])
    for relative, digest in manifest["files"].items():
        if sha256_file(payload / relative) != digest:
            raise ValueError(f"R4 payload changed: {relative}")
    if json.loads((payload / "execution.json").read_text())["commit"] != spec["rebuild_commit"]:
        raise ValueError("R4 execution identity mismatch")
    source_locks = {}
    with tarfile.open(payload / "source.tar.gz", "r:gz") as archive:
        for member in archive:
            if member.isfile() and member.name.startswith("src/"):
                archived = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                if sha256_file(Path(member.name)) != archived:
                    raise ValueError(f"R4 source changed: {member.name}")
                source_locks[member.name] = archived
    if not source_locks:
        raise ValueError("No archived source locks")
    return manifest, source_locks


def sample_indices(size, count):
    if size < count or count < 2:
        raise ValueError("Insufficient recorded observations")
    return np.linspace(0, size - 1, count, dtype=int).tolist()


def outputs(policy, observation):
    agent = policy._agent
    state = torch.as_tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        raw = agent.actor(flat_state_to_node_features(state, agent.graph_spec))
        residual = agent._ungated_policy_residuals_tensor(state, raw)
        gate = agent.correction_gate(agent.correction_gate_node_features(state, residuals=residual))
        values = torch.sigmoid(gate) if agent.correction_gate_mode == "classification" else gate
        threshold = agent._correction_gate_threshold_tensor(values)
        hard = values >= threshold
        margin = (values - threshold).abs()
    request = policy.act(observation)
    n = agent.graph_spec.num_facilities
    scaled = request[:n] * agent.specimen_action_quantization_max_transfer
    lots = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    return {"actor": raw.cpu().numpy()[0].tolist(),
            "gate_scores": gate.cpu().numpy()[0].tolist(),
            "hard_gate": hard.cpu().numpy()[0].tolist(),
            "gate_margin": margin.cpu().numpy()[0].tolist(),
            "request": request.tolist(), "requested_lots": lots.astype(int).tolist()}


def compare_outputs(reference, candidate, *, network_atol, request_atol):
    differences = {}
    for key in ("actor", "gate_scores", "request"):
        a, b = np.asarray(reference[key]), np.asarray(candidate[key])
        if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError(f"Invalid output shape/value: {key}")
        delta = float(np.max(np.abs(a - b)))
        if delta > (request_atol if key == "request" else network_atol):
            raise ValueError(f"Output tolerance failed: {key}, {delta}")
        differences[key] = delta
    for key in ("hard_gate", "requested_lots"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Discrete output differs: {key}")
    return differences


def assert_unchanged(policy, payload):
    for name in POLICY_MODULES:
        module = getattr(policy._agent, name)
        if module is not None:
            for key, value in module.state_dict().items():
                if not torch.equal(value.cpu(), payload[name][key]):
                    raise ValueError(f"Inference changed tensor: {name}.{key}")
            if any(p.requires_grad for p in module.parameters()):
                raise ValueError("Policy parameters are not frozen")
    if len(policy._agent.replay_buffer) or policy._agent.total_updates:
        raise ValueError("Inference used replay or learner updates")
    for name in ("actor_optimizer", "critic_optimizer", "correction_gate_optimizer"):
        optimizer = getattr(policy._agent, name)
        if optimizer is not None and optimizer.state:
            raise ValueError("Optimizer acquired state during inference")


def audit_seed(spec, manifest, seed):
    payload_root = Path(spec["payload_root"])
    base = Path("training/frozen_baseline_rebuild_20260929") / spec["algorithm"] / f"seed{seed}"
    config_path = base / "config.json"
    checkpoint_path = base / "checkpoints" / f"{spec['algorithm']}_seed{seed}_pretrain.pt"
    state_path = base / "checkpoints" / f"{spec['algorithm']}_seed{seed}_preonline_training_state.pt"
    config = json.loads((payload_root / config_path).read_text())
    saved = torch.load(payload_root / state_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(payload_root / checkpoint_path, map_location="cpu", weights_only=True)
    if (saved["seed"] != seed or config["seed"] != seed
            or saved["algorithm"] != spec["algorithm"]
            or saved["training_contract_sha256"] != training_contract_sha256(config)
            or saved["training_contract_sha256"] != training_contract_sha256(saved["training_contract"])):
        raise ValueError("Saved training/config identity mismatch")
    if saved["training"]["global_step"] != 0 or saved["training"]["next_episode"] != 0:
        raise ValueError("Not a pre-online baseline")
    states = np.asarray(saved["agent"]["replay_buffer"]["states"])
    if states.shape != (saved["agent"]["replay_buffer"]["size"], checkpoint["state_dim"]):
        raise ValueError("Saved observations shape mismatch")
    selected = sample_indices(len(states), spec["recorded_observations_per_seed"])
    args = dict(checkpoint=payload_root / checkpoint_path,
                effective_config=payload_root / config_path,
                checkpoint_sha256=manifest["files"][str(checkpoint_path)],
                config_sha256=manifest["files"][str(config_path)])
    cpu = StrictFrozenPolicy(**args, device="cpu")
    full = StrictFrozenPolicy(**args, device="cpu")
    reference_payload = copy.deepcopy(checkpoint)
    for name in POLICY_MODULES:
        reference_payload[name] = saved["agent"]["modules"].get(name)
    load_policy_payload_strict(full._agent, reference_payload)
    assert_unchanged(full, checkpoint)
    mps = StrictFrozenPolicy(**args, device="mps")
    graph = cpu._agent.graph_spec
    if not graph.include_time_state or config["env"]["episode_horizon"] != 52:
        raise ValueError("Missing fixed-window time coordinate")
    time_index = graph.num_facilities * (graph.features_per_facility + graph.patient_summary_width)
    rows = []
    for index in selected:
        obs = states[index]
        a, b, c = outputs(cpu, obs), outputs(full, obs), outputs(mps, obs)
        row = {"replay_index": index, "observation": obs.tolist(),
               "time_coordinate": float(obs[time_index]),
               "cpu_policy": a, "cpu_full_state": b, "mps_policy": c}
        # Persist inputs/outputs even if subsequent acceptance rejects a row.
        rows.append(row)
    for policy in (cpu, full, mps):
        assert_unchanged(policy, checkpoint)
    return rows, {"seed": seed, "state_dim": checkpoint["state_dim"],
                  "action_dim": checkpoint["action_dim"], "replay_rows": len(states),
                  "indices": selected, "time_index": time_index,
                  "checkpoint_sha256": args["checkpoint_sha256"],
                  "config_sha256": args["config_sha256"],
                  "policy_parameters_unchanged": True,
                  "saved_learning_state_not_restored": True}


def summarize_rows(rows, spec):
    maxima = {"actor": 0., "gate_scores": 0., "request": 0.}
    for row in rows:
        compare_outputs(row["cpu_policy"], row["cpu_full_state"], network_atol=0., request_atol=0.)
        diff = compare_outputs(row["cpu_policy"], row["mps_policy"],
                               network_atol=spec["network_gate_atol"], request_atol=spec["request_atol"])
        maxima = {k: max(maxima[k], diff[k]) for k in maxima}
    return {"observations": len(rows), "max_cpu_mps_absolute_difference": maxima,
            "cpu_full_state_exact": True, "hard_gate_and_requested_lots_identical": True,
            "minimum_gate_margin": min(float(x) for r in rows for x in r["cpu_policy"]["gate_margin"]),
            "sampled_time_range": [min(r["time_coordinate"] for r in rows),
                                   max(r["time_coordinate"] for r in rows)]}


def run(config_path, expected_commit):
    spec = json.loads(Path(config_path).read_text())
    if (spec["seeds"] != [60, 61, 62] or spec["recorded_observations_per_seed"] != 16
            or spec["maximum_environment_steps"] != 0 or spec["optimizer_updates"] != 0
            or spec["scientific_pilot_authorized"] or spec["historical_output_parity_claim"]):
        raise ValueError("R5 scope changed")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != expected_commit or subprocess.check_output(["git", "status", "--porcelain"], text=True):
        raise ValueError("Exact clean source required")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or not torch.backends.mps.is_available():
        raise RuntimeError("Explicit MPS inference required, no fallback")
    manifest, sources = verify_inputs(spec)
    root = Path(spec["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "execution.json", {"commit": head, "spec_sha256": sha256_file(Path(config_path)),
                                        "pid": os.getpid(), "torch_version": torch.__version__,
                                        "mps_fallback": "0", "source_locks": sources})
    summary = {"status": "failed", "seeds": [], "environment_steps": 0, "optimizer_updates": 0,
               "historical_output_parity_claim": False, "scientific_pilot_authorized": False}
    try:
        for seed in spec["seeds"]:
            rows, info = audit_seed(spec, manifest, seed)
            raw = root / f"seed{seed}_raw.json"
            write_new(raw, rows)
            info.update(summarize_rows(rows, spec))
            info["raw_sha256"] = sha256_file(raw)
            summary["seeds"].append(info)
        verify_inputs(spec)
        summary.update(status="completed", exit_code=0, verified_payload_files=len(manifest["files"]),
                       verified_existing_source_files=len(sources))
    except BaseException as error:
        summary.update(exit_code=1, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write_new(root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.expected_commit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
