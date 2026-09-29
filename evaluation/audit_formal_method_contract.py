"""Read-only extraction of the executed routing-primary method contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TRAIN_REF = "9c0718b0da49d34f7f878ed2f036e96bf7bb3693"
EVAL_REF = "62a344e7c8f501862c6ae1dcf3be5ef59964ea0d"
MANIFEST_HASH = "0311c648fc1570843daf659f5e45ad27cceb967a9265710d03bd45c60a5e31e2"
ALGORITHMS = ("gcn_residual_mdl2_network_ddpg_afd", "flat_residual_mdl2_network_ddpg_afd")
EXPECTED = {
    "gamma": .99, "tau": .005, "reward_scale": 1e-9,
    "num_episodes": 100, "max_steps_per_episode": 52, "batch_size": 64,
    "actor_lr": 1e-5, "critic_lr": 3e-4, "actor_update_frequency": 2,
    "critic_warmup_updates": 500, "update_frequency": 1, "updates_per_update": 1,
    "exploration_noise.theta": .15, "exploration_noise.sigma": .005,
    "residual_action.online_reward_mode": "n_step_anchor_relative",
    "residual_action.online_reward_n_step_horizon": 4,
    "residual_action.group_scales.specimen_transfer": .1,
    "residual_action.group_scales.reagent_transfer": 0,
    "residual_action.group_scales.capacity_transfer": 0,
    "residual_action.group_scales.replenishment": 0,
    "residual_action.l2_weight": .05,
    "residual_action.correction_gate.align_online_policy": True,
    "residual_action.correction_gate.threshold": .5,
    "specimen_action_quantization.enabled": True,
    "specimen_action_quantization.actor_gradient": "straight_through",
    "pretrain_reference_actor_loss.enabled": True,
    "pretrain_reference_actor_loss.weight": 500,
    "online_advantage_self_imitation.enabled": True,
    "online_advantage_self_imitation.weight": 1,
    "online_advantage_self_imitation.minimum_return": .0005,
    "online_advantage_self_imitation.require_positive_one_step_return": True,
    "online_advantage_self_imitation.release_pretrain_reference": True,
    "imitation_pretrain.regularization_weight": 1,
    "critic_teacher_advantage_calibration.enabled": True,
    "critic_teacher_advantage_calibration.online_ranking_weight": 3,
    "anchor_fallback.enabled": False,
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_config(config: dict, algorithm: str, seed: int) -> dict:
    if config.get("algorithm") != algorithm or config.get("seed") != seed:
        raise ValueError("run identity mismatch")
    observed = {}
    for field, expected in EXPECTED.items():
        value = config
        try:
            for key in field.split("."):
                value = value[key]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"missing contract field: {field}") from exc
        if value != expected:
            raise ValueError(f"contract mismatch: {field}: {value!r}")
        observed[field] = value
    return observed


def git_bytes(ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=REPO)


def historical_source(path: str, names: list[str]) -> dict:
    data = git_bytes(TRAIN_REF, path)
    tree = ast.parse(data)
    symbols = {}
    for name in names:
        matches = [n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]
        if not matches:
            raise ValueError(f"missing historical symbol: {path}:{name}")
        node = max(matches, key=lambda n: n.end_lineno - n.lineno)
        symbols[name] = {"start": node.lineno, "end": node.end_lineno,
                         "source": "\n".join(data.decode().splitlines()[node.lineno - 1:node.end_lineno])}
    return {"path": path, "commit": TRAIN_REF, "sha256": sha(data), "symbols": symbols}


def audit(root: Path) -> dict:
    manifest_path = root / "training_manifest.json"
    manifest_data = manifest_path.read_bytes()
    if sha(manifest_data) != MANIFEST_HASH:
        raise ValueError("historical training manifest hash mismatch")
    manifest = json.loads(manifest_data)
    expected_runs = {(a, s) for a in ALGORITHMS for s in range(10, 15)}
    runs = {(r["algorithm"], r["seed"]): r for r in manifest["runs"]}
    if len(manifest["runs"]) != 10 or set(runs) != expected_runs:
        raise ValueError("run inventory mismatch")
    records = []
    for (algorithm, seed), run in sorted(runs.items()):
        path = root / algorithm / f"seed{seed}" / "config.json"
        data = path.read_bytes()
        config = json.loads(data)
        profile = validate_config(config, algorithm, seed)
        if not run["config"].endswith(f"/{algorithm}/seed{seed}/config.json"):
            raise ValueError("manifest config path mismatch")
        records.append({"algorithm": algorithm, "seed": seed, "config_path": str(path),
                        "config_current_sha256": sha(data), "contract": profile,
                        "parameter_count": run["parameter_count"],
                        "provenance_limit": "current config hash; path and run identity linked to byte-verified historical manifest"})
    sources = [historical_source(path, names) for path, names in {
        "src/models/gcn_ddpg.py": ["capture_training_reward_context", "transform_training_reward",
                                   "_emit_n_step_reward_transition", "observe", "select_action", "update",
                                   "_pretrain_reference_action_loss", "_online_advantage_self_imitation_loss",
                                   "_compose_actions_tensor", "_policy_residuals_tensor"],
        "src/baselines/flat_ddpg.py": ["transform_training_reward", "_emit_n_step_reward_transition"],
        "src/rl/action_projection.py": ["project_action", "quantize_facility_net_specimen_actions_tensor"],
        "src/models/gcn.py": ["GCNCritic", "graph_readout"],
    }.items()]
    evaluation = []
    for variant in ("final", "pretrain"):
        path = f"experiments/configs/patient_indexed_specimen_routing_mac_mps_ddpg_confirmation_100_{variant}_eval.json"
        data = git_bytes(EVAL_REF, path)
        config = json.loads(data)
        if (config["fixed_checkpoint_variant"] != variant
                or config["fixed_deployment_candidate"] != {"scale": 1.0, "use_checkpoint_group_thresholds": True}):
            raise ValueError("unexpected fixed deployment contract")
        evaluation.append({"path": path, "commit": EVAL_REF, "sha256": sha(data), "config": config})
    if sha(manifest_path.read_bytes()) != MANIFEST_HASH:
        raise ValueError("manifest changed during audit")
    for record in records:
        if sha(Path(record["config_path"]).read_bytes()) != record["config_current_sha256"]:
            raise ValueError("config changed during audit")
    return {"status": "passed", "scope": "read_only_method_reporting_audit_not_reexecution",
            "training_commit": TRAIN_REF, "evaluation_commit": EVAL_REF,
            "historical_manifest_sha256_verified": MANIFEST_HASH, "runs": records,
            "historical_sources": sources, "evaluation_configs": evaluation,
            "manual_source_findings": {
                "replay_action": "bounded normalized request, not the final matched patient-event allocation",
                "online_reward": "one-step actual minus cloned-anchor environment reward at each actual visited state",
                "online_return": "up to four discounted relative rewards, scaled by 1e-9; gamma^n bootstrap",
                "reference_loss_default": "network output MSE, uniform mode except self-imitation release mask",
                "gate_contract": "hard behavior/target gate; soft detached actor gate by historical defaults",
                "encoder_contract": "separate actor and critic encoders, not shared learned parameters",
                "objective_equivalence": "not established between shaped training surrogate and undiscounted evaluation cost",
            }, "new_training_runs": 0, "new_evaluation_runs": 0,
            "followup": "teacher/replay mixture semantics and graph feature parity remain separate queue packets"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite audit evidence")
    report = audit(args.training_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "verified_run_configs": len(report["runs"])}))


if __name__ == "__main__":
    main()
