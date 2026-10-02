"""Saved JSON arithmetic only: no model, torch, environment or optimizer imports."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def summary(xs):
    xs = list(xs)
    if not xs or not all(math.isfinite(x) for x in xs):
        raise ValueError("nonempty finite values required")
    return {"n": len(xs), "mean": st.fmean(xs), "sd": st.pstdev(xs),
            "min": min(xs), "max": max(xs)}


def corr(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("paired lengths differ")
    dx = [x - st.fmean(xs) for x in xs]
    dy = [y - st.fmean(ys) for y in ys]
    denom = math.sqrt(math.fsum(x*x for x in dx) * math.fsum(y*y for y in dy))
    return math.fsum(x*y for x, y in zip(dx, dy)) / denom if denom else None


def returns_to_go(rewards):
    result, total = [], 0.0
    for reward in reversed(rewards):
        total += reward
        result.append(total)
    return list(reversed(result))


def normalize(xs):
    mean, sd = st.fmean(xs), st.pstdev(xs)
    return [(x - mean)/(sd + 1e-8) for x in xs]


def centered(rows, key, groups):
    values = defaultdict(list)
    for r in rows:
        values[tuple(r[g] for g in groups)].append(r[key])
    means = {k: st.fmean(v) for k, v in values.items()}
    return [r[key] - means[tuple(r[g] for g in groups)] for r in rows]


def logit_signals(probabilities, selected, advantage, entropy_coef=.01):
    entropy = -math.fsum(p*math.log(p) for p in probabilities)
    policy = [advantage*(int(i == selected)-p) for i, p in enumerate(probabilities)]
    entropy_gradient = [-entropy_coef*p*(math.log(p)+entropy) for p in probabilities]
    return policy, entropy_gradient, entropy


def subset_description(rows):
    residual = centered(rows, "return", ("update", "step"))
    values = defaultdict(list)
    for r, value in zip(rows, residual):
        values["reference" if r["reference_chosen"] else "nonreference"].append(value)
    return {
        "rows": len(rows),
        "return_step_correlation": corr([r["return"] for r in rows], [r["step"] for r in rows]),
        "advantage_step_correlation": corr([r["advantage"] for r in rows], [r["step"] for r in rows]),
        "within_update_time_return_sd": st.pstdev(residual),
        "within_update_time_nonreference_minus_reference_mean_return": (
            st.fmean(values["nonreference"])-st.fmean(values["reference"])),
        "group_counts": {k: len(v) for k, v in values.items()},
        "mean_reference_logit_policy_ascent_signal": st.fmean(r["reference_policy_signal"] for r in rows),
        "mean_reference_logit_entropy_ascent_signal": st.fmean(r["reference_entropy_signal"] for r in rows),
        "interpretation": "Descriptive on-policy association; states and action probabilities differ. Not a counterfactual advantage or significance test.",
    }


def analyze(root):
    payload = root / "payload"
    inventory_path = payload / "artifact-inventory.json"
    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_bytes)
    used = []

    def read(relative, record=None):
        path = (payload / relative).resolve()
        path.relative_to(payload)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if inventory[relative] != digest:
            raise ValueError("inventory hash mismatch: " + relative)
        if record is not None and (record["bytes"] != len(data) or record["sha256"] != digest):
            raise ValueError("phase index mismatch: " + relative)
        used.append({"path": "payload/"+relative, "sha256": digest, "bytes": len(data)})
        return data

    blocks = []
    for block in (60, 61, 62):
        phase = json.loads(read("phases/ppo_continuation/block%d.json" % block))
        if len(phase["indexes"]) != 32 or len(phase["updates"]) != 8:
            raise ValueError("unexpected phase coverage")
        rows = []
        for episode, index in enumerate(phase["indexes"]):
            record = index["events"]
            relative = "episodes/training/block%d/graph/own_ppo/episode%02d/events.jsonl" % (block, episode)
            if record["path"] != "payload/"+relative:
                raise ValueError("episode order differs")
            events = [json.loads(line)["event"]["audit"] for line in read(relative, record).splitlines()]
            if len(events) != 52:
                raise ValueError("episode length differs")
            rewards = []
            for step, event in enumerate(events):
                rec = event["record"]
                sem = rec["semantics"]
                if (rec["step_index"] != step or sem["gamma"] != 1.0
                        or sem["reward_scale"] != 1e-9 or rec["truncated"]
                        or rec["terminated"] != (step == 51)):
                    raise ValueError("closed return semantics differ")
                rewards.append(rec["raw_reward"]*sem["reward_scale"])
            for step, (event, ret) in enumerate(zip(events, returns_to_go(rewards))):
                decision = event["decision"]
                ev, choice = decision["evaluation"], decision["choice"]
                bank = ev["candidates"]
                ps = [math.exp(x) for x in ev["log_probs"]]
                if abs(math.fsum(ps)-1) > 1e-5 or len(ps) < 2:
                    raise ValueError("invalid support probabilities")
                ref, selected = bank["request_to_class"][0], choice["class_index"]
                rows.append({"episode": episode, "update": episode//4, "step": step,
                             "reward": rewards[step], "return": ret, "value": ev["value"],
                             "advantage": ret-ev["value"], "probabilities": ps,
                             "selected": selected, "reference": ref,
                             "reference_chosen": selected == ref})
        updates = []
        for update, receipt in enumerate(phase["updates"]):
            chunk = rows[update*208:(update+1)*208]
            adv = normalize([r["advantage"] for r in chunk])
            for r, a in zip(chunk, adv):
                policy, entropy, h = logit_signals(r["probabilities"], r["selected"], a)
                r.update(normalized_advantage=a, entropy=h,
                         entropy_gap=math.log(len(r["probabilities"]))-h,
                         reference_policy_signal=policy[r["reference"]],
                         reference_entropy_signal=entropy[r["reference"]],
                         policy_logit_l2=math.sqrt(math.fsum(x*x for x in policy)),
                         entropy_logit_l2=math.sqrt(math.fsum(x*x for x in entropy)))
            minibatches = receipt["minibatches"]
            if len(minibatches) != 16 or receipt["rollout_steps"] != 208:
                raise ValueError("update coverage differs")
            for epoch in range(4):
                indices = [i for b in minibatches if b["epoch"] == epoch for i in b["indices"]]
                if sorted(indices) != list(range(208)):
                    raise ValueError("minibatch epoch not an exact partition")
            first = minibatches[0]
            predicted_policy = -st.fmean(adv[i] for i in first["indices"])
            predicted_value = st.fmean(chunk[i]["advantage"]**2 for i in first["indices"])
            errors = {"policy": abs(predicted_policy-first["policy_loss"]),
                      "value": abs(predicted_value-first["value_loss"])}
            if max(errors.values()) > 2e-5:
                raise ValueError("float64 saved reconstruction disagrees with first float32 minibatch")
            rets = [r["return"] for r in chunk]
            errors_v = [r["advantage"] for r in chunk]
            variance = st.pvariance(rets)
            updates.append({"update": update+1, "rows": 208,
                            "returns": summary(rets), "values": summary(r["value"] for r in chunk),
                            "advantage": summary(errors_v),
                            "value_explained_variance": 1-st.pvariance(errors_v)/variance if variance else None,
                            "value_rmse": math.sqrt(st.fmean(x*x for x in errors_v)),
                            "first_minibatch_reconstruction_error": errors,
                            "recorded_minibatches": {
                                k: summary(b[k] for b in minibatches)
                                for k in ("policy_loss", "value_loss", "entropy", "clip_fraction",
                                          "actor_grad_norm_before_clip", "critic_grad_norm_before_clip")},
                            "fraction_actor_minibatches_clipped": st.fmean(b["actor_grad_norm_before_clip"] > .5 for b in minibatches),
                            "fraction_critic_minibatches_clipped": st.fmean(b["critic_grad_norm_before_clip"] > .5 for b in minibatches),
                            "old_policy_entropy_gap": summary(r["entropy_gap"] for r in chunk),
                            "old_policy_logit_norm_ratio_entropy_to_surrogate": (
                                st.fmean(r["entropy_logit_l2"] for r in chunk)/st.fmean(r["policy_logit_l2"] for r in chunk)),
                            "descriptive": subset_description(chunk)})
        blocks.append({"block": block, "episodes": 32, "rows": len(rows), "updates": updates,
                       "all": subset_description(rows),
                       "first16episodes": subset_description(rows[:832]),
                       "last16episodes": subset_description(rows[832:])})
    return {"schema": "saved-return-ranking-diagnostic-v1", "blocks": blocks,
            "source_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            "inputs": used, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scientific_calls": {"model_loads": 0, "model_forwards": 0, "environment": 0, "optimizer": 0},
            "method": "Float64 arithmetic from saved raw rewards/values; gamma=lambda=1, terminal bootstrap=0. Normalize within each208-row rollout. First-minibatch losses reconciled to saved float32 receipts within2e-5.",
            "limits": ["On-policy state/action confounding remains after update/time centering.",
                       "Step rows are correlated, not independent replicates; no new significance claims.",
                       "Logit gradients at the OLD policy are analytic score-space quantities, not parameter gradients or actual Adam updates.",
                       "Saved actor norm combines policy and entropy; separate parameter-gradient contribution is unavailable.",
                       "No counterfactual headroom, stochastic deployment benefit, reward defect or global optimality is established."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root.resolve())
    with args.output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"blocks": [{"block": b["block"], "all": b["all"],
                                "initial_ev": b["updates"][0]["value_explained_variance"],
                                "last_ev": b["updates"][-1]["value_explained_variance"]} for b in result["blocks"]],
                      "input_files": len(result["inputs"]), "scientific_calls": result["scientific_calls"]}, indent=2))
