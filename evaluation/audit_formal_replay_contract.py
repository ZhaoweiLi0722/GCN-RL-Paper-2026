"""Read-only teacher/replay audit, without importing an agent or environment."""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import textwrap
from collections import Counter, deque
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from evaluation.audit_formal_method_contract import (
    ALGORITHMS, MANIFEST_HASH, TRAIN_REF, git_bytes, historical_source, sha,
    validate_config,
)

TEACHER_HASH = "9ba2ac0873c0f68e6ecc4b443e0eace8230f8e151cceb485fafe218a7ac78d92"
AGENT_PATHS = ("src/models/gcn_ddpg.py", "src/baselines/flat_ddpg.py")
METHODS = ("observe", "_emit_n_step_reward_transition", "finalize_training_episode")
METRICS = (
    "critic_bellman_loss", "critic_teacher_advantage_weighted_ranking_loss",
    "actor_loss", "imitation_loss", "pretrain_reference_weighted_loss",
    "online_advantage_self_imitation_weighted_loss", "residual_l2_loss",
    "online_advantage_self_imitation_active_fraction",
    "online_advantage_self_imitation_eligible_fraction",
    "pretrain_reference_parameter_drift_rms",
)


def historical_function(path: str, name: str):
    """Extract only a reviewed numeric function, never import historical modules."""
    source = git_bytes(TRAIN_REF, path).decode()
    nodes = [n for n in ast.walk(ast.parse(source))
             if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f"ambiguous historical function: {path}:{name}")
    segment = textwrap.dedent("\n".join(source.splitlines()[nodes[0].lineno - 1:nodes[0].end_lineno]))
    namespace = {"np": np}
    exec(compile("from __future__ import annotations\n" + segment,
                 f"{TRAIN_REF}:{path}:{name}", "exec"), namespace)
    return namespace[name]


class Recorder:
    def __init__(self):
        self.rows = []

    def add(self, state, action, reward, next_state, done, **metadata):
        self.rows.append((np.array(state), np.array(action), float(reward),
                          np.array(next_state), bool(done), metadata))


def reference_windows(demos: dict, horizon: int = 4) -> list[tuple[int, int]]:
    """Independent index recurrence; do not cross explicit terminal flags."""
    result = []
    dones = demos["transition_dones"]
    for start in range(len(dones)):
        stop = min(start + horizon, len(dones))
        terminal = np.flatnonzero(dones[start:stop])
        if terminal.size:
            stop = start + int(terminal[0]) + 1
        elif stop - start < horizon:
            continue  # The loader itself does not flush a nonterminal tail.
        result.append((start, stop))
    return result


def replay_probe(demos: dict, path: str) -> dict:
    recorder = Recorder()
    agent = SimpleNamespace(replay_buffer=recorder, online_reward_mode="n_step_anchor_relative",
                            online_reward_n_step_horizon=4, gamma=.99, reward_scale=1e-9,
                            _n_step_reward_queue=deque())
    for name in METHODS:
        setattr(agent, name, MethodType(historical_function(path, name), agent))
    loader = historical_function("evaluation/run_gcn_residual_sweep.py",
                                 "populate_agent_replay_from_demonstrations")
    inserted = loader(agent, demos)
    windows = reference_windows(demos)
    if len(recorder.rows) != len(windows):
        raise ValueError("historical queue differs from independent window recurrence")
    broken = 0
    spans = Counter()
    for row, (start, stop) in zip(recorder.rows, windows):
        state, action, reward, next_state, done, meta = row
        length = stop - start
        expected_reward = sum(.99 ** (i - start) * float(demos["transition_rewards"][i])
                              for i in range(start, stop)) * 1e-9
        if (not np.array_equal(state, demos["transition_states"][start])
                or not np.array_equal(action, demos["transition_actions"][start])
                or not np.array_equal(next_state, demos["transition_next_states"][stop - 1])
                or done != bool(demos["transition_dones"][stop - 1])
                or not math.isclose(reward, expected_reward, abs_tol=1e-12)
                or not math.isclose(meta["one_step_reward"], float(demos["transition_rewards"][start]) * 1e-9, abs_tol=1e-12)
                or not math.isclose(meta["discount_multiplier"], .99 ** (length - 1), abs_tol=1e-12)):
            raise ValueError("historical replay item mismatch")
        broken += any(not np.array_equal(demos["transition_next_states"][i],
                                         demos["transition_states"][i + 1])
                      for i in range(start, stop - 1))
        spans[length] += 1
    return {"agent_source": path, "loader_reported_rows": inserted,
            "emitted_rows": len(recorder.rows), "pending_tail": len(agent._n_step_reward_queue),
            "horizon_counts": dict(sorted(spans.items())),
            "emitted_windows_with_discontinuous_observed_state": broken,
            "independent_recurrence_matches": True,
            "scope": "numeric replay insertion only; no environment steps or network updates"}


def teacher_profile(demos: dict) -> dict:
    rewards = np.asarray(demos["transition_rewards"])
    advantages = np.asarray(demos["option_advantages"])
    feasible = np.asarray(demos["option_feasible"])
    if not np.isfinite(rewards).all() or not np.isfinite(advantages).all():
        raise ValueError("nonfinite teacher values")
    if feasible.shape != advantages.shape or not feasible.any(axis=1).all():
        raise ValueError("invalid option feasibility")
    best = np.argmax(np.where(feasible, advantages, -np.inf), axis=1)
    groups = np.asarray(demos["option_groups"])
    adjacent_equal = np.all(demos["transition_next_states"][:-1] == demos["transition_states"][1:], axis=1)
    nonterminal = ~np.asarray(demos["transition_dones"][:-1], dtype=bool)
    targets = np.maximum(advantages[np.arange(len(best)), best], 0) * 1e-9
    return {"rows": len(rewards), "states_shape": list(demos["states"].shape),
            "options": [{"group": str(g), "epsilon": float(e), "sign": float(s)}
                        for g, e, s in zip(groups, demos["option_epsilons"], demos["option_signs"])],
            "best_option_groups": dict(Counter(str(g) for g in groups[best])),
            "best_option_outside_anchor_or_specimen": int((~np.isin(groups[best], ["anchor", "specimen_transfer"])).sum()),
            "improved_rows": int(demos["improved_mask"].sum()),
            "terminal_rows": int(demos["transition_dones"].sum()),
            "nonterminal_adjacent_pairs": int(nonterminal.sum()),
            "nonterminal_state_discontinuities": int((nonterminal & ~adjacent_equal).sum()),
            "cached_reward_min_mean_max": [float(f(rewards)) for f in (np.min, np.mean, np.max)],
            "critic_target_mean": float(targets.mean()),
            "critic_target_positive_fraction": float((targets > 0).mean()),
            "label_horizon_serialized_in_cache": any("lookahead" in k or "horizon" in k for k in demos),
            "limitation": "observed-state continuity is necessary but not sufficient for full simulator-state continuity"}


def loss_profile(rows: list[dict]) -> dict:
    if [int(r["episode"]) for r in rows] != list(range(100)):
        raise ValueError("expected exactly episodes 0 through 99")
    result = {}
    for metric in METRICS:
        key = f"online_rl_{metric}_mean"
        values = [float(r[key]) for r in rows]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"nonfinite persisted metric: {key}")
        result[metric] = {"episode_mean_average": float(np.mean(values)),
                          "min_episode_mean": min(values), "max_episode_mean": max(values),
                          "first_episode_mean": values[0], "last_episode_mean": values[-1]}
    updates = [int(r["online_rl_updates"]) for r in rows]
    actors = [float(r["online_rl_actor_updated_mean"]) for r in rows]
    if updates != [52] * 100 or actors != [.5] * 100:
        raise ValueError("unexpected historical update counts")
    return {"online_critic_updates": sum(updates), "online_actor_updates": int(sum(u*a for u,a in zip(updates, actors))),
            "metrics": result, "aggregation": "arithmetic average of 100 persisted episode means; equal update counts checked",
            "not_measured": ["gradient norms", "gradient interference", "causal loss ablations", "per-minibatch online fraction"]}


def audit(root: Path, teacher: Path) -> dict:
    inputs = {}
    def read(path):
        data = path.read_bytes()
        inputs[str(path)] = sha(data)
        return data
    manifest = json.loads(read(root / "training_manifest.json"))
    if inputs[str(root / "training_manifest.json")] != MANIFEST_HASH:
        raise ValueError("manifest hash mismatch")
    teacher_bytes = read(teacher)
    if sha(teacher_bytes) != TEACHER_HASH:
        raise ValueError("teacher hash mismatch")
    with np.load(io.BytesIO(teacher_bytes), allow_pickle=False) as archive:
        demos = {k: archive[k] for k in archive.files}
    profile = teacher_profile(demos)
    probes = [replay_probe(demos, path) for path in AGENT_PATHS]
    runs = []
    csv.field_size_limit(10_000_000)
    for algorithm in ALGORITHMS:
        for seed in range(10, 15):
            directory = root / algorithm / f"seed{seed}"
            config = json.loads(read(directory / "config.json"))
            validate_config(config, algorithm, seed)
            pre = json.loads(read(directory / "summary.json"))["pretrain"]
            rows = list(csv.DictReader(io.StringIO(read(directory / "training.csv").decode())))
            checks = {"offline_rl_updates": 500, "offline_rl_actor_updated_mean": 0,
                      "critic_teacher_advantage_samples": 314,
                      "critic_teacher_advantage_excluded_samples": 0,
                      "advantage_distillation_replay_transitions": 314}
            if any(pre.get(k) != v for k, v in checks.items()):
                raise ValueError(f"pretrain summary mismatch: {algorithm}/{seed}")
            if any(int(row["pretrain_samples"]) != 1040 for row in rows):
                raise ValueError("unexpected heuristic pretraining sample count")
            for key, observed in (("critic_teacher_advantage_target_mean", profile["critic_target_mean"]),
                                  ("critic_teacher_advantage_target_positive_fraction", profile["critic_target_positive_fraction"])):
                if not math.isclose(float(pre[key]), observed, rel_tol=1e-6, abs_tol=1e-10):
                    raise ValueError(f"teacher target reconciliation failed: {key}")
            if config.get("online_replay_fraction") is not None:
                raise ValueError("unexpected explicit online replay fraction")
            runs.append({"algorithm": algorithm, "seed": seed, "losses": loss_profile(rows),
                         "pretrain_checks": checks, "heuristic_pretrain_samples": 1040,
                         "logged_but_not_verified_cache_generation_metadata": {
                             k: pre[k] for k in ("advantage_distillation_lookahead",
                                                "advantage_distillation_epsilons",
                                                "advantage_distillation_candidate_groups")}})
    sources = [historical_source(p, names) for p, names in {
        "src/models/gcn_ddpg.py": [*METHODS, "pretrain_with_heuristic", "prepare_online_finetuning",
                                   "_critic_replay_bellman_loss", "configure_critic_teacher_advantage_calibration"],
        "src/baselines/flat_ddpg.py": list(METHODS),
        "src/rl/replay_buffer.py": ["add", "sample"],
        "src/rl/critic_advantage.py": ["teacher_advantage_loss"],
        "evaluation/run_gcn_residual_sweep.py": ["populate_agent_replay_from_demonstrations"],
    }.items()]
    for path, digest in inputs.items():
        if sha(Path(path).read_bytes()) != digest:
            raise ValueError(f"source changed during audit: {path}")
    if len(manifest["runs"]) != len(runs):
        raise ValueError("run count mismatch")
    return {"status": "completed_with_methodology_findings", "training_commit": TRAIN_REF,
            "teacher": profile, "teacher_sha256_verified": TEACHER_HASH,
            "replay_probes": probes, "runs": runs, "input_current_sha256": inputs,
            "historical_sources": sources, "new_training_runs": 0, "new_evaluation_runs": 0,
            "claim_limit": "static and numeric data-path evidence, not a causal explanation or an optimization experiment"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite audit evidence")
    report = audit(args.training_root, args.teacher)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": report["status"], "teacher": report["teacher"], "probes": report["replay_probes"]}))


if __name__ == "__main__":
    main()
