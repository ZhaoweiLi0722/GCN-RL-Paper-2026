"""Scalar readback of sealed artificial receipts, without neural evaluation."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_recording import write_json_once
from src.utils.research_archive import inventory, sha256_file


ROOT = Path.cwd()
RUN = ROOT / "results/candidate_calibration_engineering_20261001"
OUT = ROOT / "reports/2026-10-01-candidate-calibration-engineering/verification.json"


def read(path):
    return json.loads(path.read_text())


def weight_digest(state):
    digest = hashlib.sha256(json.dumps(state["manifest"], sort_keys=True).encode())
    for name, tensor in state["policy"].items():
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def mean(values):
    return math.fsum(values)/len(values)


def variance(values):
    m = mean(values)
    return mean([(x-m)**2 for x in values])


def softmax(logits):
    v = [math.exp(x-max(logits)) for x in logits]
    s = math.fsum(v)
    return [x/s for x in v]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    before = inventory(RUN)
    terminal, claim, seals = (read(RUN / p) for p in ("terminal.json", "claim.json", "all-models-sealed.json"))
    assert terminal["status"] == "completed" and terminal["exit_code"] == 0
    cfg_path = ROOT / "experiments/configs/candidate_calibration_engineering_20261001.json"
    assert sha256_file(cfg_path) == claim["config_sha256"]
    cfg = read(cfg_path)
    for relative, digest in claim["source_hashes"].items():
        assert sha256_file(ROOT / relative) == digest, relative
    assert 0 < terminal["numerical_seconds"] <= 1800
    assert terminal["new_patient_calls"] == terminal["new_scientific_fits"] == 0
    charges = sorted((RUN / "charges").glob("*.json"))
    assert len(charges) == 1152
    counts = Counter()
    for i, p in enumerate(charges, 1):
        r = read(p)
        counts[r["fixture"]] += 1
        assert r["total_charged_optimizer_calls"] == i and r["fixture_calls"] == counts[r["fixture"]]
    assert len(counts) == 9 and set(counts.values()) == {128}
    expected_names = {f"{r}-{seed}" for r in cfg["representations"] for seed in cfg["initialization_seeds"]}
    assert set(counts) == set(seals["models"]) == set(terminal["checks"]) == expected_names
    rows, grade = {}, {}
    class_values = sorted(cfg["request_values"])
    reference = class_values.index(0.)
    for name in sorted(expected_names):
        initial, final = (load_envelope(RUN / name / p) for p in ("initial.pt", "final.pt"))
        training = read(RUN / name / "training.json")
        assert sha256_file(RUN / name / "final.pt") == seals["models"][name]
        assert weight_digest(initial) == training["initial_sha256"]
        assert weight_digest(final) == training["final_sha256"] != training["initial_sha256"]
        assert final["optimizer_calls"] == 128 and initial["optimizer_calls"] == 0
        assert {float(s["step"]) for s in final["optimizer"]["state"].values()} == {128.}
        assert len(training["updates"]) == len(list((RUN / name / "updates").glob("*.json"))) == 128
        for i, row in enumerate(training["updates"], 1):
            assert row == read(RUN / name / "updates" / f"{i:03d}.json") and row["step"] == i
            assert all(math.isfinite(v) for v in row.values() if v is not None)
            expected_total = row["policy_loss"]+.5*row["value_loss_normalized"]-.01*row["entropy"]
            assert math.isclose(expected_total, row["total_loss"], abs_tol=1e-6)
            a, c, total_norm = row["actor_norm"], row["value_norm"], row["total_grad_norm_before_clip"]
            assert abs(a-c)-1e-5 <= total_norm <= a+c+1e-5
            cosine = row["shared_gradient_cosine"]
            assert cosine is None or -1.00001 <= cosine <= 1.00001
        hp = RUN / name / "heldout.json"
        assert hp.stat().st_mtime_ns >= (RUN / "all-models-sealed.json").stat().st_mtime_ns
        heldout = read(hp)
        assert heldout == terminal["checks"][name]
        recomputed = {}
        for arm, receipt in heldout.items():
            assert len(receipt["logits"]) == len(cfg["heldout_times"])*len(cfg["cues"]) == 8
            all_values, true_values, margins, predictions, labels, ref_probs = [], [], [], [], [], []
            index = 0
            for cue in cfg["cues"]:
                winning_request = cfg["positive_cue_winner_request"] if cue > 0 else cfg["negative_cue_winner_request"]
                winner = class_values.index(winning_request)
                for t in cfg["heldout_times"]:
                    logits, q = receipt["logits"][index], receipt["q_values"][index]
                    assert len(logits) == len(q) == 6 and all(math.isfinite(v) for v in logits+q)
                    oracle = [cfg["invented_return"]["intercept"]+cfg["invented_return"]["remaining_time_slope"]*(1-t)
                              +(0 if j == winner else cfg["invented_return"]["wrong_action_penalty"]) for j in range(6)]
                    assert all(abs(a-b) < 2e-6 for a, b in zip(q, oracle))
                    probs = softmax(logits)
                    true_value = math.fsum(p*v for p, v in zip(probs, q))
                    assert abs(true_value-receipt["true_frozen_policy_values"][index]) < 1e-6
                    true_values.append(true_value)
                    all_values.append(receipt["values"][index])
                    margins.append(logits[winner]-max(x for j, x in enumerate(logits) if j != winner))
                    predictions.append(max(range(6), key=logits.__getitem__))
                    labels.append(winner)
                    ref_probs.append(probs[reference])
                    index += 1
            errors = [a-b for a, b in zip(true_values, all_values)]
            accuracy = mean([float(a == b) for a, b in zip(predictions, labels)])
            rmse, ev = math.sqrt(mean([e*e for e in errors])), 1-variance(errors)/variance(true_values)
            metrics = {"accuracy": accuracy, "minimum_winner_margin": min(margins), "value_ev": ev, "value_rmse": rmse}
            assert all(abs(value-receipt[key]) < 2e-6 for key, value in metrics.items())
            gates = {"accuracy": accuracy >= cfg["gates"]["heldout_greedy_accuracy"],
                     "margin": min(margins) >= cfg["gates"]["heldout_minimum_winner_logit_margin"],
                     "value_ev": ev >= cfg["gates"]["heldout_value_explained_variance"],
                     "value_rmse": rmse <= cfg["gates"]["heldout_value_rmse"]}
            assert gates == receipt["gates"] and all(gates.values()) == receipt["passed"]
            if arm == "frozen":
                assert predictions == [reference]*8 and max(abs(p-.9) for p in ref_probs) < 1e-6
            recomputed[arm] = metrics | {"gates": gates, "reference_probability_range": [min(ref_probs), max(ref_probs)]}
        gain = final["manifest"]["calibration"]["logit_gain"]
        span = 2*gain*math.fsum(abs(x) for x in final["policy"]["score_head.2.weight"].tolist()[0])
        history = training["updates"]
        row = {"recomputed": recomputed, "scaled_score_span_upper_bound": span,
               "k6_prior_gap": math.log(45), "old_certificate_still_excludes_any_flip": span < math.log(45),
               "parameter_count": training["parameter_count"],
               "gradient_clipped_updates": sum(h["total_grad_norm_before_clip"] > .5 for h in history),
               "ratio_clipped_updates": sum(h["ratio_clip_fraction"] > 0 for h in history),
               "initial_value_actor_gradient_norm_ratio": history[0]["value_norm"]/history[0]["actor_norm"],
               "negative_shared_gradient_cosine_updates": sum(h["shared_gradient_cosine"] is not None and h["shared_gradient_cosine"] < 0 for h in history)}
        rows[name] = row
        grade[name] = all(recomputed["continued"]["gates"].values())
    assert all(grade.values()) == terminal["engineering_passed"]
    assert inventory(RUN) == before
    report = {"format": "calibration-engineering-scalar-verification-v1", "source_commit": terminal["source_commit"],
              "records": rows, "passed_fixtures": sum(grade.values()), "total_fixtures": 9,
              "optimizer_calls_verified": 1152, "engineering_passed": terminal["engineering_passed"],
              "all_models_sealed_before_heldout": True, "artifact_inventory": before,
              "new_neural_forwards_or_optimizer_calls": 0, "original_acceptance_artifacts_unchanged": True,
              "limits": "Invented oracle labels and interpolation only; not patient RL, not independent scientific evidence"}
    write_json_once(OUT, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("records", "artifact_inventory")}))


if __name__ == "__main__":
    main()
