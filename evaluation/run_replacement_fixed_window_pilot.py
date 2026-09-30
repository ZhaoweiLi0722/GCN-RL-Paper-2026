"""One-shot R3 replacement-baseline pilot or its distinct bounded smoke."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time
import traceback

import numpy as np
import torch

from evaluation.audit_replacement_policy_compatibility import verify_inputs, assert_unchanged
from src.rl.experiment import build_env
from src.rl.frozen_value_probe import (
    Budget, append_row, assert_scenario, candidates, collect_draw, discovery_choice,
    recorded_step, result_from_steps, streams, write_json,
)
from src.rl.patient_replay_collector import evidence_digest
from src.rl.strict_frozen_policy import StrictFrozenPolicy
from src.utils.research_archive import inventory, sha256_file


DEFAULT_CONFIG = "experiments/configs/replacement_fixed_window_pilot_20260929.json"


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def seed_audit(spec):
    allocated = streams(spec)
    base = int(spec["rng_namespace"])
    hits, files = [], {}
    for name in git("ls-files").splitlines():
        if ("replacement_fixed_window" in name or "replacement-fixed-window" in name
                or name.endswith("frozen_value_probe.py")):
            continue
        path = Path(name)
        if path.suffix not in (".py", ".json", ".md", ".yaml", ".yml", ".toml", ".sh"):
            continue
        content = path.read_text(errors="replace")
        files[name] = sha256_file(path)
        for token in re.findall(r"(?<![\w.])\d{10,}(?![\w.])", content):
            if int(token) >> 16 == base >> 16:
                hits.append({"path": name, "number": token})
    if hits:
        raise ValueError(f"Seed namespace previously registered: {hits}")
    return {"files": files, "namespace_hits": hits, "streams": allocated,
            "engineering_seed": base + spec["engineering_seed_offset"],
            "scope": "tracked prior source/config/docs; unrecorded external streams unknown"}


def load_model(lock, manifest, seed, device):
    base = Path("training/frozen_baseline_rebuild_20260929") / lock["algorithm"] / f"seed{seed}"
    c = base / "config.json"
    p = base / "checkpoints" / f"{lock['algorithm']}_seed{seed}_pretrain.pt"
    root = Path(lock["payload_root"])
    policy = StrictFrozenPolicy(root / p, root / c, checkpoint_sha256=manifest["files"][str(p)],
                                config_sha256=manifest["files"][str(c)], device=device)
    return policy, json.loads((root / c).read_text()), torch.load(root / p, map_location="cpu", weights_only=True)


def smoke(spec, lock, manifest, budget, root):
    policy, config, payload = load_model(lock, manifest, 60, spec["device"])
    seed = int(spec["rng_namespace"]) + spec["engineering_seed_offset"]
    env = build_env(config, seed=seed)
    assert_scenario(env, config, spec["scenario"])
    choices = candidates(policy, env, spec["options"])
    before = evidence_digest(env.state_dict())
    checks = []
    for choice in choices:
        attempts = []
        for replicate in range(2):
            clone = copy.deepcopy(env)
            rows = [recorded_step(clone, choice["request"], budget)]
            rows.append(recorded_step(clone, policy.act(clone.observation(), env=clone), budget))
            rows[-1]["truncated"] = True
            if rows[-1]["objective_terminal"]:
                raise ValueError("Smoke unexpectedly reached terminal")
            try:
                result_from_steps(rows, clone)
            except ValueError:
                pass
            else:
                raise ValueError("Truncation was accepted as a completed value")
            receipt = {"rows": rows, "end_state": clone.state_dict()}
            attempts.append(evidence_digest(receipt))
            write_json(root / f"action{choice['index']}_replicate{replicate}.json.gz", receipt)
        if attempts[0] != attempts[1]:
            raise ValueError("Exact cloned-path reproduction failed")
        checks.append({"action": choice["index"], "exact_clone_match": True, "digest": attempts[0]})
    if evidence_digest(env.state_dict()) != before:
        raise ValueError("Counterfactual altered parent state")
    assert_unchanged(policy, payload)
    return {"checks": checks, "environment_steps": budget.steps, "optimizer_updates": 0,
            "scope": "engineering only; two-step truncated paths; no performance conclusion"}


def pilot(spec, lock, manifest, budget, root, stream_map):
    count = 0
    env_identity = None
    with (root / "outcomes.jsonl").open("x") as outcomes, (root / "progress.jsonl").open("x") as progress:
        for seed in spec["seeds"]:
            policy, config, payload = load_model(lock, manifest, seed, spec["device"])
            env = build_env(config, seed=stream_map[f"trajectory/{seed}"])
            effective = assert_scenario(env, config, spec["scenario"])
            digest = evidence_digest(effective)
            if env_identity is not None and digest != env_identity:
                raise ValueError("Different environment across baseline seeds")
            env_identity = digest
            write_json(root / f"seed{seed}_effective_env.json", effective)
            snapshots = {}
            with gzip.open(root / f"seed{seed}_trajectory.jsonl.gz", "xt") as trace:
                while env.t < spec["horizon"]:
                    if env.t in spec["decision_steps"]:
                        snapshots[int(env.t)] = copy.deepcopy(env)
                    row = recorded_step(env, policy.act(env.observation(), env=env), budget)
                    append_row(trace, row)
                    if row["native_done"]:
                        break
            if list(snapshots) != spec["decision_steps"]:
                raise ValueError("Missing prespecified states; no replacements allowed")
            for step, snapshot in snapshots.items():
                choices = candidates(policy, snapshot, spec["options"])
                state_path = root / f"states/seed{seed}_t{step}.json.gz"
                original_digest = evidence_digest(snapshot.state_dict())
                write_json(state_path, {"seed": seed, "step": step, "observation": snapshot.observation(),
                                       "state": snapshot.state_dict(), "state_sha256": original_digest,
                                       "policy_sha256": policy.checkpoint_sha256,
                                       "config_sha256": policy.config_sha256, "choices": choices})
                for block in ("discovery", "validation"):
                    block_rows = []
                    for draw in range(spec["draws_per_block"]):
                        stem = f"seed{seed}_t{step}/{block}/draw{draw}"
                        rows = collect_draw(snapshot, policy, choices,
                                            stream_map[f"{seed}/{step}/{block}/{draw}"], budget, root, stem)
                        for row in rows:
                            row.update(seed=seed, step=step, block=block, draw=draw,
                                       source_state_sha256=original_digest,
                                       continuation_policy_sha256=policy.checkpoint_sha256)
                            append_row(outcomes, row)
                        block_rows.extend(rows)
                        count += len(rows)
                    if block == "discovery":
                        selected = discovery_choice(block_rows)
                        selected.update(seed=seed, step=step,
                                        discovery_rows_sha256=evidence_digest(block_rows),
                                        selected_clinical_outcomes=[r["outcome"] for r in block_rows
                                                                  if r["action_index"] == selected["selected_action"]])
                        write_json(root / f"selections/seed{seed}_t{step}.json", selected)
                    append_row(progress, {"seed": seed, "step": step, "completed_block": block,
                                          "logical_records": count, "actual_steps": budget.steps,
                                          "utc": datetime.now(timezone.utc).isoformat()})
                    print(f"seed={seed} t={step} {block} records={count} steps={budget.steps}", flush=True)
                if evidence_digest(snapshot.state_dict()) != original_digest:
                    raise ValueError("Parent snapshot changed")
            assert_unchanged(policy, payload)
    if count != spec["maximum_records"]:
        raise ValueError("Incomplete pilot records")
    return {"logical_records": count, "states": 12, "environment_steps": budget.steps,
            "optimizer_updates": 0, "environment_sha256": env_identity,
            "online_benefit_claim": False, "automatic_training_authorized": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--mode", required=True, choices=("engineering", "pilot"))
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    if git("status", "--porcelain"):
        raise ValueError("Commit source/config/docs before recording")
    if spec["device"] != "mps" or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or not torch.backends.mps.is_available():
        raise RuntimeError("MPS required without CPU fallback")
    root = Path(spec["engineering_root"] if args.mode == "engineering" else spec["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    budget = Budget(spec["engineering_maximum_steps"] if args.mode == "engineering"
                    else spec["maximum_environment_steps"], spec["maximum_seconds"])
    def timeout(*_):
        raise TimeoutError("Locked one-hour wall-time exceeded")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(spec["maximum_seconds"])
    try:
        head = git("rev-parse", "HEAD")
        write_json(root / "claim.json", {"pid": os.getpid(), "ppid": os.getppid(), "commit": head,
                                         "mode": args.mode, "utc": datetime.now(timezone.utc).isoformat()})
        lock = json.loads(Path(spec["input_lock_config"]).read_text())
        manifest, source_locks = verify_inputs(lock)
        audit = seed_audit(spec)
        write_json(root / "seed_audit.json", audit)
        write_json(root / "execution.json", {"commit": head, "config": spec,
                                            "spec_sha256": sha256_file(Path(args.config)),
                                            "input_lock_sha256": sha256_file(Path(spec["input_lock_config"])),
                                            "source_locks": source_locks, "torch": torch.__version__,
                                            "numpy": np.__version__, "python": platform.python_version(),
                                            "device": "mps", "mps_fallback": "0"})
        with (root / "source.tar.gz").open("xb") as archive:
            subprocess.run(["git", "archive", "--format=tar.gz", "HEAD"], stdout=archive, check=True)
        if args.mode == "engineering":
            summary = smoke(spec, lock, manifest, budget, root)
        else:
            acceptance = Path(spec["engineering_root"])
            if (json.loads((acceptance / "status.json").read_text())["status"] != "completed"
                    or json.loads((acceptance / "execution.json").read_text())["commit"] != head):
                raise ValueError("Matching committed engineering acceptance required")
            summary = pilot(spec, lock, manifest, budget, root, audit["streams"])
        verify_inputs(lock)
        write_json(root / "summary.json", summary)
        write_json(root / "artifact_inventory.json", inventory(root))
        write_json(root / "status.json", {"status": "completed", "exit_code": 0,
                                          "seconds": time.monotonic() - started,
                                          "environment_steps": budget.steps})
    except BaseException as error:
        write_json(root / "failure.json", {"exception": repr(error), "traceback": traceback.format_exc(),
                                          "environment_steps": budget.steps})
        write_json(root / "status.json", {"status": "failed", "exit_code": 1,
                                          "seconds": time.monotonic() - started,
                                          "environment_steps": budget.steps})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
