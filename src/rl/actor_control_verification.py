"""Independent scalar receipt audit; no model construction, forward or update."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_recording import write_json_once
from src.utils.research_archive import inventory, sha256_file


def read(path):
    return json.loads(path.read_text())


def softmax(row):
    exps = [math.exp(v-max(row)) for v in row]
    return [v/math.fsum(exps) for v in exps]


def weight_digest(state):
    digest = hashlib.sha256(json.dumps(state["manifest"], sort_keys=True).encode())
    for name, tensor in state["policy"].items():
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def scalar_metrics(logits, labels):
    if not logits or len(logits) != len(labels):
        raise ValueError("nonempty matching decisions required")
    if any(not row or not all(math.isfinite(v) for v in row) or not 0 <= label < len(row)
           or len(row) < 2 for row, label in zip(logits, labels)):
        raise ValueError("finite non-singleton logits and known labels required")
    choices = [max(range(len(row)), key=row.__getitem__) for row in logits]
    margins = [row[label]-max(v for j, v in enumerate(row) if j != label) for row, label in zip(logits, labels)]
    return {"accuracy": sum(a == b for a, b in zip(choices, labels))/len(labels),
            "minimum_winner_margin": min(margins), "greedy_classes": choices}


def verify(root):
    root = Path(root)
    run = root / "results/candidate_actor_positive_control_20261001"
    out = root / "reports/2026-10-01-actor-positive-control/verification.json"
    if out.exists():
        raise FileExistsError(out)
    before = inventory(run)
    terminal, claim, seals, cfg = (read(run / p) for p in ("terminal.json", "claim.json", "all-models-sealed.json", "config.json"))
    assert terminal["status"] == "completed" and terminal["exit_code"] == 0
    assert terminal["new_patient_calls"] == terminal["new_scientific_fits"] == 0
    assert terminal["source_commit"] == claim["commit"]
    assert 0 < terminal["numerical_seconds"] <= cfg["caps"]["numerical_wall_seconds"]
    config_path = root / "experiments/configs/candidate_actor_positive_control_20261001.json"
    assert read(config_path) == cfg and sha256_file(config_path) == claim["config_sha256"]
    for relative, digest in claim["source_hashes"].items():
        assert sha256_file(root / relative) == digest, relative
    counts = Counter()
    charges = sorted((run / "charges").glob("*.json"))
    assert len(charges) == terminal["optimizer_calls"] == seals["optimizer_calls"] == 1152
    for i, path in enumerate(charges, 1):
        receipt = read(path)
        counts[receipt["fixture"]] += 1
        assert receipt["total_charged_optimizer_calls"] == i
        assert receipt["fixture_calls"] == counts[receipt["fixture"]] <= 128
        assert 0 <= receipt["elapsed_seconds"] <= 1800
    names = {f"{rep}-{seed}" for rep in cfg["representations"] for seed in cfg["initialization_seeds"]}
    assert set(counts) == set(seals["models"]) == set(terminal["checks"]) == names
    assert set(counts.values()) == {128}
    ordered_actions = sorted(cfg["request_values"])
    reference = ordered_actions.index(cfg["request_values"][0])
    rows, passed = {}, []
    for name in sorted(names):
        initial, final = (load_envelope(run / name / f"{phase}.pt") for phase in ("initial", "final"))
        training = read(run / name / "training.json")
        assert sha256_file(run / name / "final.pt") == seals["models"][name]
        assert weight_digest(initial) == training["initial_sha256"]
        assert weight_digest(final) == training["final_sha256"] != training["initial_sha256"]
        assert initial["optimizer_calls"] == 0 and final["optimizer_calls"] == final["next_step"] == 128
        assert initial["config"] == final["config"] == cfg and not final["resume_authorized"]
        assert {float(s["step"]) for s in final["optimizer"]["state"].values()} == {128.}
        assert set(initial["rng"]) == set(final["rng"]) == {"python", "numpy", "torch_cpu"}
        assert final["manifest"]["critic"] is False and final["manifest"]["fixed_prior"] is False
        assert not any("value" in n or "critic" in n for n in final["policy"])
        assert sum(t.numel() for t in final["policy"].values()) == training["parameter_count"]
        assert (initial["policy"]["actor_head.weight"] == 0).all()
        for i, bias in enumerate(initial["policy"]["actor_head.bias"].tolist()):
            assert abs(bias*cfg["units"]["logit_gain"]-(cfg["initial_reference_logit"] if i == reference else 0)) < 1e-7
        assert len(training["updates"]) == len(list((run / name / "updates").glob("*.json"))) == 128
        for i, row in enumerate(training["updates"], 1):
            assert row == read(run / name / "updates" / f"{i:03d}.json") and row["step"] == i
            assert all(math.isfinite(v) for v in row.values()) and row["value_loss"] == 0
            assert abs(row["total_loss"]-(row["policy_loss"]-cfg["optimizer"]["entropy_coef"]*row["entropy"])) < 2e-6
            assert row["grad_norm_before_clip"] >= 0 and 0 <= row["ratio_clip_fraction"] <= 1
        heldout_path = run / name / "heldout.json"
        assert heldout_path.stat().st_mtime_ns >= (run / "all-models-sealed.json").stat().st_mtime_ns
        heldout = read(heldout_path)
        assert heldout == terminal["checks"][name] and set(heldout) == {"continued", "frozen"}
        arms = {}
        for arm, receipt in heldout.items():
            assert len(receipt["logits"]) == 8
            labels, expected_returns, greedy_returns = [], [], []
            for index, (cue, t) in enumerate((cue, t) for cue in cfg["cues"] for t in cfg["heldout_times"]):
                winner_value = cfg["positive_cue_winner_request"] if cue > 0 else cfg["negative_cue_winner_request"]
                winner = ordered_actions.index(winner_value)
                labels.append(winner)
                logits = receipt["logits"][index]
                assert len(logits) == 6
                oracle = [cfg["invented_return"]["intercept"]+cfg["invented_return"]["remaining_time_slope"]*(1-t)
                          +(0 if j == winner else cfg["invented_return"]["wrong_action_penalty"]) for j in range(6)]
                assert len(receipt["q_values"][index]) == 6
                assert max(abs(a-b) for a, b in zip(oracle, receipt["q_values"][index])) < 2e-6
                probs = softmax(logits)
                assert max(abs(a-b) for a, b in zip(probs, receipt["probabilities"][index])) < 2e-6
                expected_returns.append(math.fsum(p*q for p, q in zip(probs, oracle)))
                greedy_returns.append(oracle[max(range(6), key=logits.__getitem__)])
                if arm == "frozen":
                    assert max(abs(v-(cfg["initial_reference_logit"] if j == reference else 0)) for j, v in enumerate(logits)) < 1e-7
            metrics = scalar_metrics(receipt["logits"], labels)
            assert labels == receipt["labels"] and metrics["greedy_classes"] == receipt["greedy_classes"]
            assert abs(metrics["accuracy"]-receipt["accuracy"]) < 1e-7
            assert abs(metrics["minimum_winner_margin"]-receipt["minimum_winner_margin"]) < 2e-6
            for predicted, recorded in ((expected_returns, receipt["expected_invented_return"]), (greedy_returns, receipt["greedy_invented_return"])):
                assert len(predicted) == len(recorded) and max(abs(a-b) for a, b in zip(predicted, recorded)) < 2e-6
            gates = {"accuracy": metrics["accuracy"] >= cfg["gates"]["heldout_greedy_accuracy"],
                     "margin": metrics["minimum_winner_margin"] >= cfg["gates"]["heldout_minimum_winner_logit_margin"]}
            assert gates == receipt["gates"] and all(gates.values()) == receipt["passed"]
            arms[arm] = metrics | {"gates": gates, "mean_expected_invented_return": math.fsum(expected_returns)/8,
                                  "mean_greedy_invented_return": math.fsum(greedy_returns)/8}
        passed.append(all(arms["continued"]["gates"].values()))
        rows[name] = {"arms": arms, "parameter_count": training["parameter_count"],
                      "ratio_clipped_updates": sum(r["ratio_clip_fraction"] > 0 for r in training["updates"]),
                      "gradient_clipped_updates": sum(r["grad_norm_before_clip"] > cfg["optimizer"]["max_grad_norm"] for r in training["updates"])}
    assert all(passed) == terminal["engineering_passed"]
    prior = read(run / "prior-evidence.json")
    prior_roots = {"p2_payload": "candidate_reference_prior_pilot_20261001/payload",
                   "p2_launcher": "candidate_reference_prior_pilot_20261001/launcher",
                   "calibration": "candidate_calibration_engineering_20261001"}
    for key, relative in prior_roots.items():
        assert inventory(root / "results" / relative) == prior[key]
    assert inventory(run) == before
    report = {"format": "actor-control-independent-scalar-verification-v1", "source_commit": claim["commit"],
              "passed_fixtures": sum(passed), "total_fixtures": 9, "optimizer_calls_verified": 1152,
              "engineering_passed": all(passed), "records": rows, "artifact_inventory": before,
              "prior_evidence_unchanged": True, "all_models_sealed_before_heldout": True,
              "new_neural_forwards_or_optimizer_calls": 0,
              "limits": "Arithmetic and hashes of recorded logits, not an independent neural-forward replay or patient-science confirmation"}
    write_json_once(out, report)
    return {key: value for key, value in report.items() if key not in ("records", "artifact_inventory")}
