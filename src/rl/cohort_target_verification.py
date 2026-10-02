"""Read-only cohort target reconciliation; never construct or forward a model.

torch.load(weights_only=True) is used solely as a container decoder. Tests mock
that boundary with invented containers; historical scientific files are not tests.
"""

import copy
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_verification import finite, identity_counts, json_hash
from src.rl.cohort_bundle_verification import _check_index, _read_parts
from src.rl.dynamic_candidate_verification import _reread, _sha
from src.rl.time_baseline_target_verification import _unchanged_targets


def _container_digest(value):
    # Same serialization contract as state_digest, without importing model code.
    import torch

    def encode(item):
        if isinstance(item, torch.Tensor):
            return dict(tensor_dtype=str(item.dtype), shape=list(item.shape),
                        values=item.detach().cpu().tolist())
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError("unsupported serialized state value")

    raw = json.dumps(value, default=encode, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _plain(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        if set(value) == {"ndarray_dtype", "tensor"}:
            tensor = value["tensor"]
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or tensor.numpy().dtype.str != value["ndarray_dtype"]):
                raise ValueError("invalid encoded array container")
            return tensor.tolist()
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _load_container(root, record):
    import torch

    root = Path(root).resolve()
    path = root / record["path"]
    if path.resolve() != path.absolute() or not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError("redirected checkpoint container")
    raw = path.read_bytes()
    if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("checkpoint bytes differ")
    envelope = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (not isinstance(envelope, dict) or set(envelope) != {"state", "sha256"}
            or _container_digest(envelope["state"]) != envelope["sha256"]):
        raise ValueError("checkpoint envelope checksum differs")
    return _plain(envelope["state"])


def _target(raw, costs, objective, trajectory):
    tail = math.fsum(costs)
    window = math.fsum(-r for r in raw)
    rewards = list(raw)
    if objective == "cohort":
        rewards[-1] -= tail
    return dict(format="cohort-objective-target-v1", objective=objective,
        trajectory_id=trajectory, split="training", raw_prefix_rewards=list(raw),
        training_raw_rewards=rewards, window_cost=window, tail_cost=tail,
        cohort_cost=math.fsum((window, tail)), added_terminal_charge=tail if objective == "cohort" else 0.,
        environment_reward_unchanged=True, reward_scale_applied=False, bootstrap=0., tail_has_learned_actions=False)


def _episode(root, entry, config, streams, files):
    (header, rows, prefix), (tail_header, tail_rows, final) = _read_parts(root, entry)
    obj, proposal = config["objective"], config["cohort_proposal"]
    role, block, episode = header["role"], header["block"], header["world_index"]
    objective = "window" if role == "window_ppo" else "cohort"
    trajectory = f"training/block{block}/graph/{role}/episode{episode:02d}"
    seed = int(streams["environment"][str(block)]["training"][episode])
    if (header["trajectory_id"] != trajectory or header["seed"] != seed
            or header["source_id"] != streams["namespace"] or len(rows) != obj["horizon"]
            or len(tail_rows) != proposal["accounting_steps"]):
        raise ValueError("raw target trajectory/source/seed/window differs")
    rewards, values, records, decisions = [], [], [], []
    for t, row in enumerate(rows):
        event = row["event"]
        record, decision = event["audit"]["record"], event["audit"]["decision"]
        evaluation = decision["evaluation"]
        if (record["trajectory_id"] != trajectory or record["source_id"] != streams["namespace"]
                or record["step_index"] != t or record["terminated"] is not (t + 1 == len(rows))
                or record["truncated"] is not False or evaluation["inference_dtype"] != "float32"
                or evaluation["behavior_sha256"] != header["policy_sha256"]
                or finite(record["raw_reward"], signed=True) != -finite(event["info"]["cost"])
                or event["audit"]["terminal_cost_added"] != 0):
            raise ValueError("raw target lineage/behavior/reward differs")
        rewards.append(record["raw_reward"])
        values.append(finite(evaluation["value"], signed=True))
        records.append(record)
        decisions.append(decision)
    costs = [finite(r["cost"]) for r in tail_rows]
    if (any(r["cost"] != r["info"]["cost"] or r["raw_reward"] != -r["cost"]
            or r["index"] != i + 1 or r["accounting_done"] is not (i + 1 == len(tail_rows))
            for i, r in enumerate(tail_rows))
            or identity_counts(final)["active"] != 0
            or tail_header["prefix_final_sha256"] != json_hash(prefix)
            or tail_header["prefix"] != entry["prefix"]):
        raise ValueError("incomplete or altered target closure")
    closure = dict(trajectory_id=trajectory, environment_seed=seed, source_id=streams["namespace"],
        split="training", tail_costs=costs, terminal_active=0, prefix_state_sha256=json_hash(prefix),
        final_state_sha256=json_hash(final), tail_rows_sha256=json_hash(tail_rows))
    target = _target(rewards, costs, objective, trajectory)
    dirs = {part: Path(entry[part]["header"]["path"]).parent for part in ("prefix", "tail")}

    def read(part, filename, container=False):
        rec = file_record(root, dirs[part] / filename)
        files.append(rec)
        return _load_container(root, rec) if container else _reread(root, rec)

    if read("tail", "training-target.json") != target or read("tail", "training-lineage.json") != closure:
        raise ValueError("once-only suffix target or closure hash differs")
    original, collected = read("prefix", "collector.pt", True), read("tail", "collector.pt", True)
    if (original["failure"] is not None or original["index"] != len(rows)
            or original["environment"] != prefix or original["events"] != [r["event"] for r in rows]
            or original["manifest"] != header["session_manifest"]
            or collected["format"] != "cohort-collection-v1" or collected["failure"] is not None
            or collected["objective"] != objective or collected["split"] != "training"
            or collected["trajectory_id"] != trajectory or collected["prefix_live"] is not None
            or collected["prefix_snapshot"] != original or collected["prefix_receipt"] != entry["prefix"]
            or collected["prefix_costs"] != [-r for r in rewards] or collected["tail_events"] != tail_rows):
        raise ValueError("raw prefix/collector snapshot changed or tail collector differs")
    followup = collected["followup"]
    if (followup["failure"] is not None or followup["closed"] is not True
            or followup["steps"] != len(tail_rows) or followup["costs"] != costs
            or followup["environment"] != final or followup["prefix_state"] != prefix
            or followup["contract"] != tail_header["contract"]):
        raise ValueError("collector followup closure differs")
    adv, ret = _unchanged_targets(rewards, values, obj["reward_scale"])
    segment = dict(decisions=decisions, records=records, bootstrap=None, gae_lambda=obj["gae_lambda"],
                   advantages=adv.tolist(), returns=ret.tolist())
    return dict(header=header, raw_rewards=rewards, collected_values=values, closure=closure,
                target=target, segment_sha256=json_hash(segment))


def verify_cohort_targets(root, raw_index, config, streams):
    """Bind both PPO arms' saved kernel histories to immutable raw acquisitions.

    Expects final.pt at payload/models/blockN/graph/ROLE/final.pt and the existing
    phase receipt at payload/phases/ROLE/blockN.json. Accepts a training-only index
    or a full campaign index; non-PPO rows remain the raw bundle reader's scope.
    """
    root = Path(root).resolve()
    _check_index(raw_index)
    cfg, obj = config["continuation"], config["objective"]
    count, updates, episodes = (cfg[k] for k in
        ("episodes_per_rollout", "rollouts_per_arm_per_block", "episodes_per_arm_per_block"))
    if (any(type(x) is not int or x < 1 for x in (count, updates, episodes))
            or episodes != count * updates or obj["gamma"] != 1. or obj["gae_lambda"] != 1.
            or finite(obj["reward_scale"]) <= 0):
        raise ValueError("complete terminal gamma1/lambda1 target contract required")
    roles = {"window_ppo": "window", "cohort_ppo": "cohort"}
    expected = {(b, role, e) for b in config["blocks"] for role in roles for e in range(episodes)}
    raw, files, inventory = {}, [], {}
    for entry in raw_index:
        header = _reread(root, entry["prefix"]["header"])
        if header["split"] != "training":
            continue
        key = header["block"], header["role"], header["world_index"]
        if (key in inventory or header["role"] not in (*roles, "bc_continue")
                or header["block"] not in config["blocks"] or type(header["world_index"]) is not int
                or not 0 <= header["world_index"] < episodes or header["source_id"] != streams["namespace"]
                or header["seed"] != int(streams["environment"][str(header["block"])]["training"][header["world_index"]])):
            raise ValueError("duplicate/foreign training inventory")
        inventory[key] = copy.deepcopy(entry)
        if header["role"] == "bc_continue":
            continue
        if key in raw or key not in expected:
            raise ValueError("duplicate/unexpected PPO training acquisition")
        files.extend(copy.deepcopy(rec) for part in entry.values() for rec in part.values())
        raw[key] = _episode(root, entry, config, streams, files)
    if set(raw) != expected:
        raise ValueError("missing complete PPO training acquisition inventory")
    full_expected = {(b, role, e) for b in config["blocks"] for role in (*roles, "bc_continue") for e in range(episodes)}
    if set(inventory) != full_expected:
        raise ValueError("missing complete three-arm training inventory")
    phases = {}
    for b in config["blocks"]:
        for role in (*roles, "bc_continue"):
            rec = file_record(root, f"payload/phases/{role}/block{b}.json")
            files.append(rec)
            phase = _reread(root, rec)
            if (phase["job"] != f"{role}/block{b}" or len(phase["updates"]) != updates
                    or phase["indexes"] != [inventory[b, role, e] for e in range(episodes)]):
                raise ValueError("phase raw acquisition/update inventory differs")
            phases[b, role] = phase
    results = []
    for b in config["blocks"]:
        for role, objective in roles.items():
            model_rec = file_record(root, f"payload/models/block{b}/graph/{role}/final.pt")
            files.append(model_rec)
            state, phase = _load_container(root, model_rec), phases[b, role]
            manifest = state["manifest"]
            origins = [dict(trajectory_id=raw[b, role, e]["header"]["trajectory_id"],
                            environment_seed=raw[b, role, e]["header"]["seed"]) for e in range(episodes)]
            consumed = [[streams["namespace"], r["trajectory_id"], t] for r in origins for t in range(obj["horizon"])]
            if (manifest["format"] != "cohort-ppo-kernel-v1" or manifest["objective"] != objective
                    or manifest["target_method"] != "collected_value" or manifest["mode"] != "online"
                    or manifest["accounting_steps"] != config["cohort_proposal"]["accounting_steps"]
                    or manifest["episode_horizon"] != obj["horizon"] or manifest["episodes_per_rollout"] != count
                    or manifest["training_source_id"] != streams["namespace"] or manifest["training_manifest"] != origins
                    or state["manifest_sha256"] != json_hash(manifest)
                    or state["failure"] is not None or state["pending"] or state["pending_cohorts"]
                    or state["history"] != [count * obj["horizon"]] * updates or state["consumed"] != consumed
                    or len(state["target_history"]) != updates or len(phase["updates"]) != updates
                    or phase["job"] != f"{role}/block{b}"):
                raise ValueError("final CohortPPOKernel manifest/progress/inventory differs")
            for u in range(updates):
                batch = [raw[b, role, u * count + i] for i in range(count)]
                behaviors = [r["header"]["policy_sha256"] for r in batch]
                if len(set(behaviors)) != 1:
                    raise ValueError("mixed behavior policy within target rollout")
                for sha in behaviors:
                    _sha(sha)
                advantages, returns = [], []
                for r in batch:
                    a, g = _unchanged_targets(r["target"]["training_raw_rewards"], r["collected_values"], obj["reward_scale"])
                    if not np.isfinite(a).all() or not np.isfinite(g).all():
                        raise ValueError("nonfinite cohort training targets")
                    advantages.append(a.tolist())
                    returns.append(g.tolist())
                receipt = dict(method="collected_value", objective=objective,
                    trajectory_ids=[r["header"]["trajectory_id"] for r in batch],
                    environment_seeds=[r["header"]["seed"] for r in batch], behavior_sha256s=behaviors,
                    terminal_flags=[True] * count, raw_rewards=[r["raw_rewards"] for r in batch],
                    collected_values=[r["collected_values"] for r in batch],
                    baselines=[r["collected_values"] for r in batch], returns=returns, advantages=advantages,
                    segment_sha256s=[r["segment_sha256"] for r in batch],
                    cohort_receipts=[r["target"] for r in batch], closures=[r["closure"] for r in batch])
                update = phase["updates"][u]
                if (state["target_history"][u] != receipt or update["target_receipt"] != receipt
                        or update["update"] != u + 1 or update["rollout_steps"] != count * obj["horizon"]):
                    raise ValueError("raw cohort arithmetic/segment/closure differs from saved target history")
                g, v, a = (np.asarray(x, dtype=np.float64) for x in
                           (returns, receipt["collected_values"], advantages))
                results.append(dict(block=b, role=role, update=u + 1, objective=objective,
                    rows=count * obj["horizon"], collected_value_mse=float(np.mean((g - v) ** 2)),
                    raw_advantage_variance=float(np.var(a)), target_receipt_sha256=json_hash(receipt)))
    return dict(format="cohort-raw-target-verification-v1", files=files, rollouts=results,
        raw_prefix_immutable_verified=True, once_only_suffix_charge_verified=True,
        three_arm_training_inventory_verified=True,
        closure_hashes_verified=True, segment_digest_payload_reconstruction_verified=True,
        saved_kernel_target_history_verified=True, checkpoint_inference_performed=False,
        gradient_or_optimizer_reexecution_performed=False,
        scope="serialized receipts and raw target arithmetic; not learned-policy inference or scientific authorization")
