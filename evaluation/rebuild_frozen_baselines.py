"""One bounded replacement-baseline rebuild with per-seed verified archives."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from src.utils.research_archive import create_archive, copy_verified, sha256_file


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def build_config(spec: dict, source: dict, seed: int, payload: Path) -> dict:
    if (spec["seeds"] != [60, 61, 62] or seed not in spec["seeds"]
            or spec["algorithm"] != "gcn_residual_mdl2_network_ddpg_afd"
            or spec["online_episodes"] != 0
            or spec["pretrain_epochs"] != 300 or spec["offline_updates"] != 500
            or spec["maximum_environment_steps_per_seed"] != 1040
            or spec["maximum_seconds"] != 3600
            or spec["scientific_evaluation_authorized"]
            or spec["historical_reproduction_claim"]):
        raise ValueError("Rebuild budget/identity changed")
    if source["pretrain_epochs"] != 300 or source["offline_updates"] != 500:
        raise ValueError("Historical pretraining budget differs")
    result = copy.deepcopy(source)
    result.update(name=spec["name"], online_episodes=0,
                  output_root=str(payload / "training"),
                  experimental_role="Replacement pretrained baseline; NOT recovered F1 evidence")
    result["algorithms"] = [{"name": spec["algorithm"], "seeds": spec["seeds"]}]
    overrides = result["config_overrides"]
    if overrides["device"] != "mps":
        raise ValueError("MPS required")
    overrides["critic_teacher_advantage_calibration"]["updates"] = 0
    run = payload / "training" / spec["name"] / spec["algorithm"] / f"seed{seed}"
    overrides["preonline_training_state_path"] = str(
        run / "checkpoints" / f"{spec['algorithm']}_seed{seed}_preonline_training_state.pt")
    return result


def require_finite(value, path="root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            require_finite(item, f"{path}[{i}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite metric: {path}")


def train_seed(spec: dict, source: dict, seed: int, payload: Path) -> dict:
    from src.env.multi_scenario import EpisodeScenarioEnv
    from evaluation.train_multiscenario_network_residual import train_multiscenario_agents

    config = build_config(spec, source, seed, payload)
    steps = 0
    original_step = EpisodeScenarioEnv.step

    def bounded_step(env, action):
        nonlocal steps
        if steps >= spec["maximum_environment_steps_per_seed"]:
            raise RuntimeError("Heuristic demonstration step cap reached")
        steps += 1
        return original_step(env, action)

    # Counts only initialization demonstrations; any unexpected rollout fails.
    with patch.object(EpisodeScenarioEnv, "step", bounded_step):
        result = train_multiscenario_agents(config, algorithm_filter=spec["algorithm"],
                                           seed_filter=seed)
    run = next(x for x in result["runs"] if x["seed"] == seed)
    require_finite(run)
    if run["online_episodes"] != 0 or run["scenario_episode_counts"]:
        raise RuntimeError("Unexpected online trajectory")
    if run["pretrain"]["offline_rl_updates"] != 500:
        raise RuntimeError("Offline update budget not completed")
    if run["pretrain"].get("local_search_demonstration_source") != "cache":
        raise RuntimeError("Teacher cache not used")
    import torch
    state = torch.load(run["preonline_training_state"], map_location="cpu", weights_only=False)
    if state["training"]["global_step"] != 0 or state["training"]["next_episode"] != 0:
        raise RuntimeError("Invalid pre-online state boundary")
    actor = torch.load(run["pretrain_checkpoint"], map_location="cpu", weights_only=False)
    def check_tensors(value):
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise RuntimeError("Nonfinite saved tensor")
        if isinstance(value, dict):
            for item in value.values():
                check_tensors(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check_tensors(item)
    check_tensors(state["agent"])
    for name in ("actor", "correction_gate"):
        if not actor.get(name) or any(not torch.isfinite(x).all() for x in actor[name].values()):
            raise RuntimeError(f"Missing/nonfinite {name}")
        saved = state["agent"]["modules"][name]
        if actor[name].keys() != saved.keys() or any(
                not torch.equal(actor[name][key], saved[key]) for key in saved):
            raise RuntimeError(f"Policy/full-state {name} mismatch")
    # The legacy full-state format records CPU/CUDA RNG, but omits MPS RNG.
    rng_path = Path(run["preonline_training_state"]).with_name("mps_rng_state.pt")
    with rng_path.open("xb") as handle:
        torch.save(torch.mps.get_rng_state(), handle)
    receipt = {"seed": seed, "heuristic_demonstration_steps": steps,
               "online_episodes": 0, "checkpoint_sha256": sha256_file(Path(run["pretrain_checkpoint"])),
               "full_state_sha256": sha256_file(Path(run["preonline_training_state"])),
               "finite_saved_tensors": True,
               "actor_gate_match_full_state": True,
               "supplemental_mps_rng_sha256": sha256_file(rng_path),
               "replacement_not_historical_recovery": True}
    write_json(payload / f"seed{seed}_receipt.json", receipt)
    return receipt


def run(spec_path: Path, expected_commit: str) -> None:
    import torch
    spec = json.loads(spec_path.read_text())
    source_path = Path(spec["source_config"])
    source = json.loads(source_path.read_text())
    if sha256_file(source_path) != spec["source_config_sha256"]:
        raise ValueError("Historical source config changed")
    if sha256_file(Path(source["teacher_cache"])) != spec["teacher_sha256"]:
        raise ValueError("Teacher changed")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if head != expected_commit or dirty:
        raise ValueError("Execution requires exact committed clean source")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable or CPU fallback not explicitly disabled")
    root = Path(spec["campaign_root"])
    root.mkdir(parents=True, exist_ok=False)
    payload = root / "payload"
    payload.mkdir()
    logs = payload / "logs"
    logs.mkdir()
    write_json(payload / "execution.json", {
        "commit": head, "spec_sha256": sha256_file(spec_path), "spec": spec,
        "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
        "pid": os.getpid(), "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "cloud_sync_verified": False})
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(payload / "source.tar.gz"),
                    head, "AGENTS.md", "requirements.txt", "src", "evaluation", "tests",
                    "experiments/configs", source["teacher_cache"],
                    "specs/2026-09-29-frozen-baseline-rebuild"], check=True)
    status = {"status": "running", "commit": head, "pid": os.getpid(),
              "completed_seeds": [], "backups": [], "started_at_unix": time.time()}
    write_json(root / "status.json", status)

    def timeout(_signum, _frame):
        raise TimeoutError("One-hour rebuild cap exceeded; no automatic retry")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(int(spec["maximum_seconds"]))
    try:
        for seed in spec["seeds"]:
            status["current_seed"] = seed
            write_json(root / "status.json", status)
            print(f"Starting replacement pretraining seed={seed}", flush=True)
            with (logs / f"seed{seed}.log").open("x") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    receipt = train_seed(spec, source, seed, payload)
            status["completed_seeds"].append(receipt)
            archive = root / "archives" / f"through_seed{seed}.tar.gz"
            info = create_archive(payload, archive)
            copies = [copy_verified(path, Path(spec["dropbox_directory"]) / path.name)
                      for path in (archive, archive.with_name(archive.name + ".manifest.json"))]
            status["backups"].append({"archive": str(archive), "sha256": info["archive_sha256"],
                                      "copies": copies})
            write_json(root / "status.json", status)
            print(f"Completed and backed up seed={seed}; cloud sync unverified", flush=True)
        status.update(status="completed", exit_code=0, completed_at_unix=time.time())
    except BaseException:
        status.update(status="failed", exit_code=1, traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        write_json(root / "status.json", status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    run(args.spec, args.expected_commit)


if __name__ == "__main__":
    main()
