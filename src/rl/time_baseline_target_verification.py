"""Raw-file reconciliation of both PPO arms' targets; no learner/model imports."""

import math
from pathlib import Path

import numpy as np

from src.rl.candidate_pilot_driver import file_record
from src.rl.dynamic_candidate_verification import _reread, _sha


def _unchanged_targets(rewards, values, scale):
    # Independently reproduce the existing terminal float32 GAE arithmetic,
    # including rounding of stored advantages before adding collected values.
    r = np.asarray([v * scale for v in rewards], dtype=np.float32)
    v = np.asarray(values, dtype=np.float32)
    a = np.zeros_like(r)
    gae, next_value = 0., 0.
    for t in reversed(range(len(r))):
        nonterminal = float(t != len(r) - 1)
        delta = r[t] + next_value * nonterminal - v[t]
        gae = delta + nonterminal * gae
        a[t], next_value = gae, v[t]
    return a, a + v


def verify_time_baseline_targets(root, index, config, streams):
    root = Path(root).resolve()
    cfg, horizon = config["continuation"], config["objective"]["horizon"]
    count, updates = cfg["episodes_per_rollout"], cfg["rollouts_per_arm_per_block"]
    if (count < 2 or cfg["episodes_per_arm_per_block"] != count * updates
            or config["objective"]["gamma"] != 1. or config["objective"]["gae_lambda"] != 1.):
        raise ValueError("fixed terminal MC target contract required")
    raw, files = {}, []
    roles = {"current_ppo": "collected_value", "time_baseline_ppo": "leave_one_episode_out_time"}
    for entry in index:
        header = _reread(root, entry["header"])
        if header["split"] != "training" or header["role"] not in roles:
            continue
        key = header["block"], header["role"], header["world_index"]
        if key in raw or header["source_id"] != streams["namespace"]:
            raise ValueError("duplicate/foreign training target episode")
        rows = _reread(root, entry["events"], jsonl=True)
        if len(rows) != horizon:
            raise ValueError("incomplete training target episode")
        raw[key] = header, rows
        files.extend((entry["header"], entry["events"]))
    expected = {(b, role, e) for b in config["blocks"] for role in roles
                for e in range(cfg["episodes_per_arm_per_block"])}
    if set(raw) != expected:
        raise ValueError("missing/extra PPO training raw target episodes")
    results = []
    fields = {"trajectory_ids", "environment_seeds", "behavior_sha256s", "terminal_flags", "method",
              "returns", "collected_values", "baselines", "advantages", "raw_rewards", "segment_sha256s"}
    for block in config["blocks"]:
        for role, method in roles.items():
            record = file_record(root, f"payload/phases/{role}/block{block}.json")
            phase = _reread(root, record)
            files.append(record)
            if phase["job"] != f"{role}/block{block}" or len(phase["updates"]) != updates:
                raise ValueError("exact PPO phase/update inventory required")
            for update_index, update in enumerate(phase["updates"]):
                receipt = update["target_receipt"]
                if (set(receipt) != fields or receipt["method"] != method
                        or update["update"] != update_index + 1 or update["rollout_steps"] != count * horizon):
                    raise ValueError("target method or update cursor mismatch")
                rewards, values, old_adv, returns, ids, seeds, behaviors = [], [], [], [], [], [], []
                for i in range(count):
                    e = update_index * count + i
                    header, rows = raw[block, role, e]
                    trajectory = f"training/block{block}/graph/{role}/episode{e:02d}"
                    seed = int(streams["environment"][str(block)]["training"][e])
                    if header["trajectory_id"] != trajectory or header["seed"] != seed:
                        raise ValueError("raw target episode identity or seed mismatch")
                    rr, vv = [], []
                    for t, row in enumerate(rows):
                        transition, decision = row["event"]["audit"]["record"], row["event"]["audit"]["decision"]
                        evaluation = decision["evaluation"]
                        if (transition["source_id"] != streams["namespace"] or transition["trajectory_id"] != trajectory
                                or transition["step_index"] != t or transition["truncated"] is not False
                                or transition["terminated"] is not (t == horizon - 1)
                                or evaluation["behavior_sha256"] != header["policy_sha256"]
                                or evaluation["inference_dtype"] != "float32"):
                            raise ValueError("raw target lineage/terminal/behavior mismatch")
                        rr.append(transition["raw_reward"])
                        vv.append(evaluation["value"])
                    a, g = _unchanged_targets(rr, vv, config["objective"]["reward_scale"])
                    rewards.append(rr)
                    values.append(vv)
                    old_adv.append(a.tolist())
                    returns.append(g.tolist())
                    ids.append(trajectory)
                    seeds.append(seed)
                    behaviors.append(header["policy_sha256"])
                if len(set(behaviors)) != 1 or len(set(seeds)) != count:
                    raise ValueError("target batch not fixed-behavior independent episodes")
                baselines = (values if method == "collected_value" else
                    [[math.fsum(returns[j][t] for j in range(count) if j != i) / (count - 1)
                      for t in range(horizon)] for i in range(count)])
                advantages = (old_adv if method == "collected_value" else
                    [[returns[i][t] - baselines[i][t] for t in range(horizon)] for i in range(count)])
                expected_receipt = dict(trajectory_ids=ids, environment_seeds=seeds, behavior_sha256s=behaviors,
                    terminal_flags=[True] * count, returns=returns, collected_values=values,
                    raw_rewards=rewards, baselines=baselines, advantages=advantages)
                for field, value in expected_receipt.items():
                    if receipt[field] != value:
                        raise ValueError(f"raw training/target receipt differs: {field}")
                if len(receipt["segment_sha256s"]) != count:
                    raise ValueError("segment hash inventory differs")
                for sha in receipt["segment_sha256s"]:
                    _sha(sha)
                arrays = [np.asarray(x, dtype=np.float64) for x in (returns, values, baselines, advantages)]
                if not all(np.isfinite(a).all() for a in arrays):
                    raise ValueError("nonfinite raw target diagnostics")
                g, v, baseline, adv = arrays
                results.append(dict(block=block, role=role, update=update_index + 1, method=method,
                    rows=count * horizon, collected_value_mse=float(np.mean((g - v) ** 2)),
                    applied_baseline_mse=float(np.mean((g - baseline) ** 2)),
                    raw_advantage_variance=float(np.var(adv)),
                    time_mean_advantage_variance=float(np.var(adv.mean(axis=0))),
                    gradient_variance_measured=False))
    return dict(format="time-baseline-raw-target-verification-v1", files=files, rollouts=results,
                raw_reward_and_collected_value_binding_verified=True, unchanged_critic_targets_verified=True,
                entire_trajectory_exclusion_verified=True,
                segment_digest_payload_reconstruction_verified=False,
                scope="raw-file lineage, target arithmetic and scalar diagnostics; not model gradients or patient benefit")
