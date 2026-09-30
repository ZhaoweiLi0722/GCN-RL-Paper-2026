"""Single bounded R6 attempt; trained critic never controls a trajectory."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import time
import traceback

import numpy as np
import torch

from evaluation.audit_replacement_policy_compatibility import assert_unchanged, verify_inputs
from evaluation.run_replacement_fixed_window_pilot import git, load_model
from src.rl.clean_critic_probe import checkpoint, fit, fresh_critic, seal_predictions
from src.rl.critic_probe_contract import (
    StateLineage, paired_advantage, public_inputs, ranking_counts, validate_partitions,
)
from src.rl.experiment import build_env
from src.rl.frozen_value_probe import (
    Budget, append_row, assert_scenario, candidates, collect_draw, recorded_step, write_json,
)
from src.rl.patient_replay_collector import evidence_digest
from src.utils.research_archive import inventory, sha256_file


DEFAULT = "experiments/configs/clean_critic_generalization_20260930.json"
METRICS = ("total_cost", "patients_lost", "patients_completed",
           "completion_service_level", "manufacturing_loss_rate")


def utc():
    return datetime.now(timezone.utc).isoformat()


def allocate_streams(spec):
    base = int.from_bytes(hashlib.sha256(spec["rng_namespace_text"].encode()).digest()[:12], "big") << 16
    result = {}
    for seed in spec["seeds"]:
        for role in ("train", "test"):
            for trajectory in range(spec["trajectories"][role]):
                name = f"seed{seed}/{role}{trajectory}"
                result[name] = base + len(result)
                for step in spec["decision_steps"]:
                    for draw in range(spec["draws_per_state"]):
                        result[f"{name}/t{step}/draw{draw}"] = base + len(result)
    if len(result) != 594 or len(set(result.values())) != 594:
        raise ValueError("Expected594 distinct starts")
    return base, result


def audit_streams(spec):
    base, streams = allocate_streams(spec)
    hits, files = [], {}
    for name in git("ls-files").splitlines():
        if "clean_critic" in name or "clean-critic" in name:
            continue
        path = Path(name)
        if path.suffix not in (".py", ".json", ".md", ".yaml", ".yml", ".toml", ".sh"):
            continue
        content = path.read_text(errors="replace")
        files[name] = sha256_file(path)
        for token in re.findall(r"(?<![\w.])\d{6,}(?![\w.])", content):
            number = int(token)
            if number >> 16 == base >> 16 or number in spec["critic_init_seeds"]:
                hits.append({"path": name, "number": token})
    if hits:
        raise ValueError(f"Previously allocated seeds: {hits}")
    return {"streams": streams, "prior_files": files, "namespace_hits": hits,
            "scope": "committed prior files; external unrecorded streams unknown"}


def check_prior(spec):
    for name, expected in spec["prior_locks"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"Prior lock changed: {name}")
    prior = Path("results/replacement_fixed_window_pilot_20260929")
    for name, expected in json.loads((prior / "artifact_inventory.json").read_text()).items():
        if sha256_file(prior / name) != expected:
            raise ValueError(f"Prior R3 evidence changed: {name}")


def collect_parents(policy, config, spec, seed, role, streams, root, budget, progress):
    states = []
    for trajectory in range(spec["trajectories"][role]):
        name = f"seed{seed}/{role}{trajectory}"
        env = build_env(config, seed=streams[name])
        environment = assert_scenario(env, config, spec["scenario"])
        env_sha = evidence_digest(environment)
        snapshots = {}
        path = root / f"parents/{name}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "xt") as trace:
            while env.t < spec["horizon"]:
                if env.t in spec["decision_steps"]:
                    snapshots[int(env.t)] = copy.deepcopy(env)
                row = recorded_step(env, policy.act(env.observation(), env=env), budget)
                append_row(trace, row)
                if row["native_done"]:
                    break
        if list(snapshots) != spec["decision_steps"]:
            raise ValueError("Missing states; no replacement allowed")
        for step, snapshot in snapshots.items():
            choices = candidates(policy, snapshot, spec["options"])
            state_id = f"{name}/t{step}"
            state_sha = evidence_digest(snapshot.state_dict())
            public = {"observation": snapshot.observation(),
                      "requests": [c["request"] for c in choices],
                      "reference_request": choices[0]["request"]}
            inputs = public_inputs(public, step)
            lineage = StateLineage(state_id, state_sha, policy.checkpoint_sha256, env_sha,
                                   streams[name], step, "prospective_unseen")
            write_json(root / f"states/{state_id}.json.gz",
                       {"lineage": asdict(lineage), "seed": seed, "role": role,
                        "trajectory": trajectory, "state": snapshot.state_dict(),
                        "public": public, "choices": choices})
            states.append({"lineage": lineage, "seed": seed, "role": role,
                           "trajectory": trajectory, "snapshot": snapshot, "choices": choices,
                           "public": inputs, "support": [c["map_reachable_certified"] for c in choices]})
        progress({"phase": "parents", "seed": seed, "role": role, "trajectory": trajectory})
    return states


def collect_labels(states, policy, spec, streams, root, budget, outcomes, progress):
    labels = []
    for state in states:
        identity = state["lineage"]
        records = []
        for draw in range(spec["draws_per_state"]):
            stem = f"{identity.state_id}/draw{draw}"
            rows = collect_draw(state["snapshot"], policy, state["choices"], streams[stem],
                                budget, root, stem)
            for row in rows:
                row.update(state_id=identity.state_id, seed=state["seed"], role=state["role"],
                           trajectory=state["trajectory"], step=identity.decision_step, draw=draw,
                           source_state_sha256=identity.snapshot_sha256,
                           continuation_policy_sha256=identity.continuation_sha256)
                append_row(outcomes, row)
            records.extend(rows)
        costs = [[r["outcome"]["total_cost"] for r in records if r["action_index"] == a]
                 for a in range(6)]
        seeds = [streams[f"{identity.state_id}/draw{d}"] for d in range(spec["draws_per_state"])]
        targets = [paired_advantage(c, costs[0], seeds, seeds, reward_scale=spec["reward_scale"])
                   for c in costs]
        row = {"state_id": identity.state_id, "role": state["role"], "targets": targets,
               "mean_costs": np.mean(costs, axis=1).tolist(), "records_sha256": evidence_digest(records)}
        write_json(root / f"labels/{identity.state_id}.json", row)
        labels.append(row)
        if evidence_digest(state["snapshot"].state_dict()) != identity.snapshot_sha256:
            raise ValueError("Counterfactual changed parent")
        progress({"phase": "labels", "state_id": identity.state_id, "role": state["role"]})
    return labels


def summarize_test(states, labels, seal, records):
    support = [s["support"] for s in states]
    costs = [x["mean_costs"] for x in labels]
    result = {"ranking": ranking_counts(seal["advantage_predictions"], costs, support),
              "states": [], "trajectories": []}
    lookup = {(r["state_id"], r["draw"], r["action_index"]): r["outcome"] for r in records}
    for i, state in enumerate(states):
        state_id = state["lineage"].state_id
        chosen = seal["critic_actions"][i]
        row = {"state_id": state_id, "trajectory": state["trajectory"], "selected_action": chosen,
               "support": support[i], "contrasts": {}}
        for baseline in ("frozen", "mdl2", "constant"):
            reference = seal[f"{baseline}_actions"][i]
            row["contrasts"][baseline] = {metric: [lookup[state_id, d, chosen][metric]
                                                  - lookup[state_id, d, reference][metric]
                                                  for d in range(8)] for metric in METRICS}
        result["states"].append(row)
    for trajectory in range(2):
        rows = [r for r in result["states"] if r["trajectory"] == trajectory]
        comparison = {b: {m: float(np.mean([r["contrasts"][b][m] for r in rows])) for m in METRICS}
                      for b in ("frozen", "mdl2", "constant")}
        frozen = comparison["frozen"]
        adverse = (frozen["patients_lost"] > 0 or frozen["patients_completed"] < 0
                   or frozen["completion_service_level"] < 0 or frozen["manufacturing_loss_rate"] > 0)
        result["trajectories"].append({"trajectory": trajectory, "mean_contrasts": comparison,
                                      "adverse_clinical_mean_vs_frozen": adverse})
    return result


def run(spec, root, budget, lock, manifest, streams):
    all_states, summaries = [], {}
    with (root / "outcomes.jsonl").open("x") as outcomes, (root / "progress.jsonl").open("x") as progress_file:
        def progress(row):
            row.update(utc=utc(), actual_steps=budget.steps)
            append_row(progress_file, row)
            print(json.dumps(row), flush=True)
        for seed, init_seed in zip(spec["seeds"], spec["critic_init_seeds"]):
            policy, config, payload = load_model(lock, manifest, seed, spec["device"])
            write_json(root / f"seed{seed}/effective_config.json", config)
            base = Path("training/frozen_baseline_rebuild_20260929") / lock["algorithm"] / f"seed{seed}"
            original = Path(lock["payload_root"]) / base / "checkpoints" / f"{lock['algorithm']}_seed{seed}_pretrain.pt"
            shutil.copy2(original, root / f"seed{seed}/frozen_actor_gate_source.pt")
            training = collect_parents(policy, config, spec, seed, "train", streams, root, budget, progress)
            train_labels = collect_labels(training, policy, spec, streams, root, budget, outcomes, progress)
            agent = policy._agent
            model = fresh_critic(agent, init_seed)
            directory = root / f"seed{seed}"
            checkpoint(directory / "critic_initial.pt", model, None, updates=0, scale=None, seed=init_seed)
            train_costs = [r["mean_costs"] for r in train_labels]
            train_support = [s["support"] for s in training]
            targets = [[t["mean_advantage"] for t in r["targets"]] for r in train_labels]
            with (directory / "fit.jsonl").open("x") as fit_log:
                optimizer, fit_report = fit(model, agent, [s["public"] for s in training], targets,
                                            train_support, ["train"] * len(training), spec["fit"], budget,
                                            lambda row: append_row(fit_log, row))
            checkpoint(directory / "critic_final.pt", model, optimizer, updates=fit_report["updates"],
                       scale=fit_report["target_scale"], seed=init_seed)
            write_json(directory / "fit_summary.json", fit_report)
            progress({"phase": "critic_fit_complete", "seed": seed, "updates": fit_report["updates"]})
            test = collect_parents(policy, config, spec, seed, "test", streams, root, budget, progress)
            all_states.extend(training + test)
            validate_partitions([s["lineage"] for s in all_states],
                                {s["lineage"].state_id: s["role"] for s in all_states})
            seal = seal_predictions(model, agent, [s["public"] for s in test], [s["support"] for s in test],
                                    fit_report["target_scale"], train_costs, train_support)
            seal.update(state_ids=[s["lineage"].state_id for s in test],
                        checkpoint_sha256=sha256_file(directory / "critic_final.pt"),
                        train_labels_sha256=evidence_digest(train_labels), utc=utc(),
                        public_test_sha256=evidence_digest([asdict(s["public"]) for s in test]))
            write_json(directory / "sealed_test_predictions.json", seal)
            seal_sha = sha256_file(directory / "sealed_test_predictions.json")
            progress({"phase": "test_predictions_sealed", "seed": seed, "seal_sha256": seal_sha})
            test_labels = collect_labels(test, policy, spec, streams, root, budget, outcomes, progress)
            if sha256_file(directory / "sealed_test_predictions.json") != seal_sha:
                raise ValueError("Test prediction seal changed")
            if sha256_file(directory / "critic_final.pt") != seal["checkpoint_sha256"]:
                raise ValueError("Critic changed after test seal")
            assert_unchanged(policy, payload)
            records = [json.loads(line) for line in (root / "outcomes.jsonl").read_text().splitlines()]
            report = summarize_test(test, test_labels, seal, records)
            write_json(directory / "test_summary.json", report)
            summaries[str(seed)] = report
            progress({"phase": "policy_complete", "seed": seed, "actor_gate_unchanged": True})
    if len(records) != spec["maximum_records"] or len(all_states) != 72:
        raise ValueError("Incomplete recorded matrix")
    partition = validate_partitions([s["lineage"] for s in all_states],
                                    {s["lineage"].state_id: s["role"] for s in all_states})
    trajectories = [t for r in summaries.values() for t in r["trajectories"]]
    cost_gate = all(t["mean_contrasts"][b]["total_cost"] < 0 for t in trajectories for b in ("frozen", "constant"))
    ranking_gate = all(r["ranking"]["pairwise_accuracy"] is not None
                       and r["ranking"]["pairwise_accuracy"] > .5 for r in summaries.values())
    clinical_gate = not any(t["adverse_clinical_mean_vs_frozen"] for t in trajectories)
    return {"states": 72, "trajectories": 18, "logical_records": len(records),
            "environment_steps": budget.steps, "supervised_critic_updates": 3000,
            "actor_updates": 0, "ddpg_updates": 0, "partition": partition,
            "triage": {"all_trajectory_cost": cost_gate, "all_policy_ranking": ranking_gate,
                       "clinical_directions": clinical_gate,
                       "followup_discussion_only": cost_gate and ranking_gate and clinical_gate},
            "automatic_actor_training": False, "online_benefit_claim": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT)
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    if git("status", "--porcelain"):
        raise ValueError("Commit source/config/protocol before recording")
    if spec["device"] != "mps" or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or not torch.backends.mps.is_available():
        raise RuntimeError("MPS with fallback0 required")
    root = Path(spec["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    budget = Budget(spec["maximum_environment_steps"], spec["maximum_seconds"])
    def timeout(*_):
        raise TimeoutError("Locked two-hour wall time exhausted; do not retry")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(spec["maximum_seconds"])
    try:
        head = git("rev-parse", "HEAD")
        write_json(root / "claim.json", {"pid": os.getpid(), "ppid": os.getppid(), "commit": head, "utc": utc()})
        lock = json.loads(Path(spec["input_lock_config"]).read_text())
        manifest, source_locks = verify_inputs(lock)
        check_prior(spec)
        audit = audit_streams(spec)
        write_json(root / "seed_audit.json", audit)
        write_json(root / "execution.json", {"commit": head, "config": spec, "spec_sha256": sha256_file(Path(args.config)),
                                            "source_locks": source_locks, "inputs": manifest,
                                            "torch": torch.__version__, "numpy": np.__version__,
                                            "python": platform.python_version(), "device": "mps", "mps_fallback": "0"})
        with (root / "source.tar.gz").open("xb") as archive:
            subprocess.run(["git", "archive", "--format=tar.gz", "HEAD"], stdout=archive, check=True)
        summary = run(spec, root, budget, lock, manifest, audit["streams"])
        verify_inputs(lock)
        check_prior(spec)
        write_json(root / "summary.json", summary)
        write_json(root / "artifact_inventory.json", inventory(root))
        write_json(root / "status.json", {"status": "completed", "exit_code": 0, "seconds": time.monotonic() - start,
                                          "environment_steps": budget.steps})
    except BaseException as error:
        write_json(root / "failure.json", {"exception": repr(error), "traceback": traceback.format_exc(),
                                          "environment_steps": budget.steps})
        write_json(root / "status.json", {"status": "failed", "exit_code": 1, "seconds": time.monotonic() - start,
                                          "environment_steps": budget.steps})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
