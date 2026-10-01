"""Posthoc P2 arithmetic and fixed-weight bounds; never run a neural model.

Run with PYTHONPATH=. from the persistent worktree. Existing tensors are read
using checksummed weights-only envelopes, not instantiated learner objects.
"""

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import traceback

import numpy as np

from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_recording import write_json_once
from src.utils.research_archive import inventory, sha256_file


ROOT = Path.cwd()
RUN = ROOT / "results/candidate_reference_prior_pilot_20261001"
OUT = ROOT / "reports/2026-10-01-p2-training-diagnostic/result.json"


def read(path):
    return json.loads(path.read_text())


def numbers(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size or not np.isfinite(a).all():
        raise ValueError("empty or nonfinite numerical evidence")
    return a


def stats(values):
    a = numbers(values).ravel()
    return {"n": int(a.size), "min": float(a.min()), "mean": float(a.mean()),
            "median": float(np.median(a)), "max": float(a.max())}


def score_bound(weights, nonreference_mass=.1):
    if not 0 < nonreference_mass < .5:
        raise ValueError("this certificate requires reference mass above one half")
    span = 2 * float(np.abs(numbers(weights)).sum())
    minimum_prior = math.log((1 - nonreference_mass) / nonreference_mass)
    return {"residual_pairwise_span_upper_bound": span,
            "minimum_prior_margin_k2": minimum_prior,
            "prior_margin_k5": math.log(4 * (1 - nonreference_mass) / nonreference_mass),
            "prior_margin_k6": math.log(5 * (1 - nonreference_mass) / nonreference_mass),
            "reference_certified_for_all_k_ge_2": span + 1e-5 < minimum_prior,
            "minimum_mathematical_margin_k2": minimum_prior - span,
            "singleton_has_no_choice": True}


def value_range(weights, bias):
    b = numbers(bias).ravel()
    if len(b) != 1:
        raise ValueError("scalar value bias required")
    radius = float(np.abs(numbers(weights)).sum())
    return float(b[0] - radius), float(b[0] + radius)


def target_range_readback(targets, bounds):
    values = numbers(targets)
    lo, hi = bounds
    if not math.isfinite(lo + hi) or lo > hi:
        raise ValueError("invalid output range")
    excess = np.maximum(np.maximum(lo - values, values - hi), 0)
    return {"output_range": [lo, hi], "target_range": [float(values.min()), float(values.max())],
            "outside_count": int(np.count_nonzero(excess > 1e-5)), "targets": int(values.size),
            "outside_fraction": float(np.mean(excess > 1e-5)),
            "mse_lower_bound_from_output_range": float(np.mean(excess**2))}


def reverse_returns(rewards, values):
    rewards, values = numbers(rewards), numbers(values)
    if rewards.ndim != 1 or rewards.shape != values.shape:
        raise ValueError("aligned one-dimensional terminal episode required")
    returns = np.cumsum(rewards[::-1], dtype=np.float64)[::-1]
    return returns, returns - values


def normalize(values):
    a = numbers(values)
    return (a - a.mean()) / (a.std() + 1e-8)


def reference_logit_signal(advantages, chosen_reference, probability, entropy):
    a, p, ent = numbers(advantages), numbers(probability), numbers(entropy)
    chosen = np.asarray(chosen_reference, dtype=bool)
    if not a.shape == p.shape == ent.shape == chosen.shape or np.any((p <= 0) | (p > 1)):
        raise ValueError("aligned valid reference probabilities required")
    return a * (chosen.astype(float) - p), -.01 * p * (np.log(p) + ent)


def position_share(values):
    a = numbers(values)
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError("episode-by-position matrix required")
    centered = a - a.mean()
    total = float(np.sum(centered**2))
    return None if total == 0 else float(a.shape[0] * np.sum((a.mean(0) - a.mean())**2) / total)


def array(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    terminal = read(RUN / "launcher/terminal.json")
    assert terminal["status"] == "completed" and terminal["supervisor"]["exit_code"] == 0
    processes = subprocess.check_output(["ps", "-ax", "-o", "pid,ppid,etime,command"], text=True)
    assert not [s for s in processes.splitlines() if " -m experiments.scripts.run_candidate_reference_pilot" in s]
    payload_before, launcher_before = inventory(RUN / "payload"), inventory(RUN / "launcher")
    assert payload_before == read(RUN / "archives/completed-payload.tar.gz.manifest.json")["files"]
    packet = read(RUN / "payload/locks/effective-execution.json")
    # Verify the frozen subset, not today's larger tracked source-file set.
    for name, expected in (packet["source_files"] | packet["document_locks"]).items():
        assert sha256_file(ROOT / name) == expected, name
    config = packet["scientific_config"]
    settings = config["ppo"]
    assert settings["gae_lambda"] == 1 and settings["normalize_advantages"] is True
    assert (settings["value_loss_coef"], settings["entropy_coef"], settings["max_grad_norm"]) == (.5, .01, .5)
    inputs = {}

    def source(path):
        relative = str(path.relative_to(RUN))
        expected = payload_before[relative.removeprefix("payload/")]
        inputs[relative] = expected
        return read(path)

    def state(path):
        relative = str(path.relative_to(RUN))
        inputs[relative] = payload_before[relative.removeprefix("payload/")]
        return load_envelope(path)

    models = {}
    for block in (60, 61, 62):
        for rep in ("graph", "self_only", "flat"):
            key = f"block{block}/{rep}/ppo"
            base = RUN / "payload/models" / key
            initial, final = state(base / "initial.pt"), state(base / "final.pt")
            assert final["manifest"]["policy"]["reference_prior"]["nonreference_mass"] == .1
            train = source(base / "training.json")
            assert train["episodes"] == 32 and len(train["updates"]) == 8
            bounds = value_range(array(final["policy"]["value_head.2.weight"]), array(final["policy"]["value_head.2.bias"]))
            final_score = score_bound(array(final["policy"]["score_head.2.weight"]))
            initial_score = score_bound(array(initial["policy"]["score_head.2.weight"]))
            episodes, components, class_counts = [], Counter(), Counter()
            for i, entry in enumerate(train["raw_episodes"]):
                hp = RUN / entry["header"]["path"]
                header = source(hp)
                assert inputs[str(hp.relative_to(RUN))] == entry["header"]["sha256"]
                assert (header["split"], header["block"], header["representation"], header["role"], header["world_index"]) == ("training", block, rep, "ppo", i)
                event_path = RUN / entry["events"]["path"]
                event_relative = str(event_path.relative_to(RUN))
                assert entry["events"]["sha256"] == payload_before[event_relative.removeprefix("payload/")]
                inputs[event_relative] = entry["events"]["sha256"]
                rows = [json.loads(s)["event"] for s in event_path.read_text().splitlines()]
                assert len(rows) == 52
                rewards, values, probability, reference, entropy, margins = [], [], [], [], [], []
                for t, e in enumerate(rows):
                    audit, info = e["audit"], e["info"]
                    rec, decision = audit["record"], audit["decision"]
                    ev = decision["evaluation"]
                    assert rec["step_index"] == t and rec["terminated"] == (t == 51) and not rec["truncated"]
                    assert rec["raw_reward"] == -info["cost"]
                    assert rec["semantics"]["gamma"] == 1 and rec["semantics"]["reward_scale"] == 1e-9
                    lp = numbers(ev["log_probs"]); probabilities = np.exp(lp)
                    assert abs(float(probabilities.sum()) - 1) < 1e-6
                    ref = ev["candidates"]["request_to_class"][0]
                    assert 0 <= ref < len(lp) and 0 <= decision["choice"]["class_index"] < len(lp)
                    class_counts[len(lp)] += 1
                    rewards.append(rec["raw_reward"] * 1e-9); values.append(ev["value"])
                    probability.append(probabilities[ref]); reference.append(decision["choice"]["class_index"] == ref)
                    entropy.append(-float(np.sum(probabilities * lp)))
                    if len(lp) > 1:
                        margins.append(float(lp[ref] - max(v for j, v in enumerate(lp) if j != ref)))
                    for component, cost in audit["cost_components"]:
                        components[component] += cost
                    assert math.isclose(sum(c for _, c in audit["cost_components"]), info["cost"], rel_tol=1e-12)
                returns, advantages = reverse_returns(rewards, values)
                episodes.append({"directory": hp.parent, "returns": returns, "advantages": advantages,
                                 "values": values, "reference": reference, "p_ref": probability,
                                 "entropy": entropy, "margins": margins})
            rollouts = []
            for u, update in enumerate(train["updates"]):
                group = episodes[4*u:4*u+4]
                saved = state(group[-1]["directory"] / "collector.pt")["kernel"]
                assert len(saved["pending"]) == 4
                persisted_a = np.array([s["advantages"] for s in saved["pending"]])
                persisted_g = np.array([s["returns"] for s in saved["pending"]])
                recomputed_g = np.array([e["returns"] for e in group])
                recomputed_a = np.array([e["advantages"] for e in group])
                err = max(float(np.max(abs(persisted_g-recomputed_g))), float(np.max(abs(persisted_a-recomputed_a))))
                assert err < 1e-5
                for e, seg in zip(group, saved["pending"]):
                    assert seg["bootstrap"] is None and seg["gae_lambda"] == 1
                    assert len(seg["records"]) == 52
                    # Full receipt hashes were verified by the original run; bind the ordered episode identity here.
                    assert seg["records"][0]["trajectory_id"].endswith(e["directory"].name)
                a = normalize(persisted_a).ravel(); g = persisted_g.ravel()
                v = np.array([e["values"] for e in group]).ravel()
                ref = np.array([e["reference"] for e in group]).ravel()
                p = np.array([e["p_ref"] for e in group]).ravel()
                ent = np.array([e["entropy"] for e in group]).ravel()
                logs = update["minibatches"]
                assert len(logs) == 16 and update["rollout_steps"] == 208
                for epoch in range(4):
                    assert sorted(i for m in logs if m["epoch"] == epoch for i in m["indices"]) == list(range(208))
                first = logs[0]; idx = first["indices"]
                expected = {"policy_loss": -float(a[idx].mean()), "value_loss": float(np.mean((v[idx]-g[idx])**2)), "entropy": float(ent[idx].mean())}
                first_error = {k: abs(first[k]-x) for k, x in expected.items()}
                assert max(first_error.values()) < 1e-5
                for mb in logs:
                    numbers([mb[k] for k in ("policy_loss", "value_loss", "entropy", "total_loss", "clip_fraction", "grad_norm_before_clip")])
                    total = mb["policy_loss"] + .5*mb["value_loss"] - .01*mb["entropy"]
                    assert math.isclose(total, mb["total_loss"], abs_tol=1e-6, rel_tol=1e-6)
                current_bounds = value_range(array(saved["policy"]["value_head.2.weight"]), array(saved["policy"]["value_head.2.bias"]))
                assert min(v) >= current_bounds[0]-1e-5 and max(v) <= current_bounds[1]+1e-5
                centered = a.reshape(4,52) - a.reshape(4,52).mean(0)
                # Score-function signal for the reference logit at ratio=1; not a neural parameter gradient.
                assert ref.any() and (~ref).any() and np.var(g) > 0
                score_signal, entropy_signal = reference_logit_signal(a, ref, p, ent)
                rollouts.append({"update":u+1,"return_reconstruction_max_error":err,"first_minibatch_errors":first_error,
                    "advantage":stats(persisted_a),"time_position_advantage_variation_share":position_share(persisted_a),
                    "reference_fraction":float(ref.mean()),"reference_probability":stats(p),"reference_margin":stats([m for e in group for m in e['margins']]),
                    "mean_normalized_advantage_reference":float(a[ref].mean()),"mean_normalized_advantage_nonreference":float(a[~ref].mean()),
                    "time_centered_advantage_reference":float(centered.ravel()[ref].mean()),
                    "time_centered_advantage_nonreference":float(centered.ravel()[~ref].mean()),
                    "reference_logit_return_signal_mean":float(score_signal.mean()),
                    "reference_logit_entropy_signal_mean":float(entropy_signal.mean()),
                    "reference_logit_combined_signal_mean":float((score_signal+entropy_signal).mean()),
                    "value_explained_variance":float(1-np.var(g-v)/np.var(g)),
                    "actual_value_mse":float(np.mean((g-v)**2)),
                    "preupdate_target_range":target_range_readback(g,current_bounds),
                    "preupdate_score_bound":score_bound(array(saved["policy"]["score_head.2.weight"]))})
            minibatches = [m for u in train["updates"] for m in u["minibatches"]]
            norms = np.array([m["grad_norm_before_clip"] for m in minibatches])
            all_targets = np.concatenate([e["returns"] for e in episodes])
            total_cost = float(sum(components.values()))
            models[key] = {"initial_score_bound":initial_score,"final_score_bound":final_score,
                "class_count_histogram":dict(class_counts),"final_value_target_range":target_range_readback(all_targets,bounds),
                "cost_components":dict(components),"cost_component_shares":{k:v/total_cost for k,v in components.items()},
                "final_adam_steps":sorted({float(v["step"]) for v in final["optimizer"]["state"].values()}),
                "minibatches":len(minibatches),"probability_clip_fraction":stats([m["clip_fraction"] for m in minibatches]),
                "gradient_norm":stats(norms),"gradient_clipped_batches":int(np.count_nonzero(norms>.5)),
                "common_gradient_multiplier":stats(np.minimum(1.,.5/(norms+1e-6))),
                "weighted_value_loss":stats([.5*m["value_loss"] for m in minibatches]),
                "absolute_policy_loss":stats([abs(m["policy_loss"]) for m in minibatches]),
                "rollouts":rollouts}
            print(json.dumps({"model":key,"score_bound":final_score["residual_pairwise_span_upper_bound"],
                              "clipped_gradient_batches":models[key]["gradient_clipped_batches"]}),flush=True)
    assert inventory(RUN / "payload") == payload_before and inventory(RUN / "launcher") == launcher_before
    report = {"format":"p2-saved-training-diagnostic-v1","diagnostic_commit":subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        "models":models,"input_files":inputs,"original_payload_launcher_unchanged":True,"frozen_source_document_locks_verified":True,
        "finished_utc":datetime.now(timezone.utc).isoformat(),
        "coverage":{"models":9,"episodes":288,"training_steps":14976,"rollouts":72,"minibatches":1152},
        "new_simulator_calls":0,"new_neural_forwards":0,"new_neural_backwards":0,"new_optimizer_steps":0,
        "new_test_evaluations":0,"posthoc_not_independent_confirmation":True,
        "limits":["score bounds fix observed weights, not all future trained models", "loss magnitudes/global clipping do not identify component gradients",
                  "sampled action associations are not same-state counterfactuals", "three blocks; correlated time steps are not independent experiments"]}
    write_json_once(OUT,report)
    print(str(OUT))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        write_json_once(OUT.parent / "readback-errors" / f"{stamp}.json", {
            "traceback":traceback.format_exc(), "scientific_retry":False,
            "diagnostic_source_sha256":sha256_file(Path(__file__)),
            "head":subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
        raise
