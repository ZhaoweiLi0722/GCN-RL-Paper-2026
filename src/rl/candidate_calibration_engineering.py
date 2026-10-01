"""Single bounded invented-tensor acceptance, never a patient campaign."""

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import traceback

from src.models.calibrated_reference_candidate import CandidateCalibration, CalibratedReferenceCandidatePolicy
from src.models.matched_inputs import InputSchema, ObservationBatch
from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_ppo_objective import candidate_ppo_loss
from src.rl.networks import torch
from src.rl.routing_candidate_contract import RoutingRequestSchema, build_request_candidates
from src.utils.research_archive import inventory, sha256_file


CONFIG = "experiments/configs/candidate_calibration_engineering_20261001.json"
PROTOCOL = "specs/2026-10-01-candidate-calibration-engineering/protocol.md"


def read(path):
    return json.loads(path.read_text())


def fixture_schema():
    return InputSchema("invented-calibration-only-v1", ("A", "B"), ("invented_count",),
                       ("time", "public_cue"), tuple(f"request_{i}" for i in range(8)))


def model(config, representation, seed):
    calibration = dict(config["calibration"])
    for name in ("node_divisors", "global_divisors", "action_divisors"):
        calibration[name] = tuple(calibration[name])
    return CalibratedReferenceCandidatePolicy(fixture_schema(), enabled=True,
        architecture="flat" if representation == "flat" else "graph",
        message_mode="physical" if representation == "graph" else "self_only",
        encoder_width=config["encoder_width"], head_width=config["head_width"], seed=seed,
        nonreference_mass=config["nonreference_mass"], calibration=CandidateCalibration(**calibration))


def invented_data(config, times):
    schema = fixture_schema()
    requests = [[v, -v, 0., 0., 0., 0., -.5, -.5] for v in config["request_values"]]
    bank = build_request_candidates({"state_token": "invented-not-a-patient",
        "reference_request": requests[0], "anchor_request": requests[1], "option_requests": requests[2:]},
        RoutingRequestSchema(schema.definition_id + "/action", 2, 120., 6))
    observations, labels, targets = [], [], []
    for cue in config["cues"]:
        winner_request = config["positive_cue_winner_request"] if cue > 0 else config["negative_cue_winner_request"]
        winner = next(i for i, row in enumerate(bank.class_features) if row[0] == winner_request)
        for t in times:
            observations.append(ObservationBatch(schema, torch.tensor([[[1000.], [2000.]]]),
                torch.tensor([[t, cue]]), torch.tensor([[[0., 100000.], [100000., 0.]]])))
            q = config["invented_return"]
            base = q["intercept"] + q["remaining_time_slope"]*(1-t)
            targets.append([base + (0 if j == winner else q["wrong_action_penalty"])
                            for j in range(len(bank.class_keys))])
            labels.append(winner)
    return observations, bank, torch.tensor(targets), torch.tensor(labels)


def scores(policy, observations, bank):
    out = [policy(obs, bank) for obs in observations]
    return torch.stack([x.logits for x in out]), torch.stack([x.value for x in out])


def enumerated_objective(logits, values, old_log_probs, q_values, settings):
    """Exact old-policy weighted surrogate over invented known-answer actions."""
    logp = logits.log_softmax(1)
    old = old_log_probs.detach().exp()
    expected = (old*q_values).sum(1).detach()
    advantages = (q_values-expected[:, None]).detach()
    weighted_advantages = advantages * old * logits.shape[1]
    entropy = -(logp.exp()*logp).sum(1)
    unit = settings["value_loss_unit"]
    return candidate_ppo_loss(logp.flatten(), (values/unit)[:, None].expand_as(logp).flatten(),
        entropy[:, None].expand_as(logp).flatten(), old_log_probs.detach().flatten(),
        weighted_advantages.flatten(), (expected/unit)[:, None].expand_as(logp).flatten(),
        clip_ratio=settings["clip_ratio"], value_loss_coef=settings["value_loss_coef"],
        entropy_coef=settings["entropy_coef"])


def gradient_readback(policy, loss, settings):
    named = list(policy.named_parameters())
    parameters = [p for _, p in named]
    actor = torch.autograd.grad(loss.policy-settings["entropy_coef"]*loss.entropy,
                                parameters, retain_graph=True, allow_unused=True)
    critic = torch.autograd.grad(settings["value_loss_coef"]*loss.value,
                                 parameters, retain_graph=True, allow_unused=True)
    def vector(gradients, shared=False):
        selected = [g.flatten() for (name, _), g in zip(named, gradients)
                    if g is not None and (not shared or name.startswith("encoder."))]
        return torch.cat(selected) if selected else torch.zeros(1)
    a, v, sa, sv = vector(actor), vector(critic), vector(actor, True), vector(critic, True)
    norms = [float(x.norm()) for x in (a, v, sa, sv)]
    cosine = float(torch.dot(sa, sv)/(sa.norm()*sv.norm())) if norms[2] > 0 and norms[3] > 0 else None
    assert all(math.isfinite(x) for x in norms) and (cosine is None or math.isfinite(cosine))
    return dict(zip(("actor_norm", "value_norm", "shared_actor_norm", "shared_value_norm"), norms)) | {
        "shared_gradient_cosine": cosine}


def evaluate(policy, data, gates):
    observations, bank, q, labels = data
    with torch.no_grad():
        logits, values = scores(policy, observations, bank)
        true_values = (logits.softmax(1)*q).sum(1)
        error = true_values-values
        masked = logits.clone()
        masked[torch.arange(len(labels)), labels] = -torch.inf
        margins = logits[torch.arange(len(labels)), labels]-masked.max(1).values
        ev = float(1-error.var(unbiased=False)/true_values.var(unbiased=False))
        rmse = float(error.square().mean().sqrt())
        accuracy = float((logits.argmax(1) == labels).float().mean())
        checks = {"accuracy": accuracy >= gates["heldout_greedy_accuracy"],
                  "margin": float(margins.min()) >= gates["heldout_minimum_winner_logit_margin"],
                  "value_ev": ev >= gates["heldout_value_explained_variance"],
                  "value_rmse": rmse <= gates["heldout_value_rmse"]}
        return {"gates": checks, "passed": all(checks.values()), "accuracy": accuracy,
                "minimum_winner_margin": float(margins.min()), "value_ev": ev, "value_rmse": rmse,
                "winner_labels": labels.tolist(), "greedy_classes": logits.argmax(1).tolist(),
                "logits": logits.tolist(), "values": values.tolist(), "q_values": q.tolist(),
                "true_frozen_policy_values": true_values.tolist()}


def charge(total, fixture_steps, cap, per_fixture):
    if total >= cap or fixture_steps >= per_fixture:
        raise RuntimeError("optimizer call budget exhausted; no refund/retry")
    return total+1, fixture_steps+1


def run(root):
    root = Path(root).resolve()
    config = read(root / CONFIG)
    assert config["engineering_only"] and not config["scientific_execution_authorized"]
    assert config["patient_calls_authorized"] == 0 and config["no_retry_or_expansion"]
    assert config["caps"] == {"fixtures": 9, "total_optimizer_step_calls": 1152, "numerical_wall_seconds": 1800}
    assert len(config["representations"])*len(config["initialization_seeds"]) == 9
    assert config["optimizer"]["steps_per_fixture"] == 128
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    output = root / config["output"]
    if output.exists():
        raise FileExistsError("existing artificial attempt is immutable")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    p2 = root / "results/candidate_reference_prior_pilot_20261001"
    old_packet = read(p2 / "payload/locks/effective-execution.json")
    for relative, digest in (old_packet["source_files"] | old_packet["document_locks"]).items():
        assert sha256_file(root / relative) == digest, relative
    before = {part: inventory(p2 / part) for part in ("payload", "launcher")}
    assert before["payload"] == read(p2 / "archives/completed-payload.tar.gz.manifest.json")["files"]
    tracked = subprocess.check_output(["git", "ls-files", "src", "experiments/scripts", "tests", CONFIG, PROTOCOL],
                                      cwd=root, text=True).splitlines()
    write_json_once(output / "claim.json", {"commit": commit, "config_sha256": sha256_file(root / CONFIG),
        "protocol_sha256": sha256_file(root / PROTOCOL), "mode": "artificial-only-single-attempt",
        "source_hashes": {p: sha256_file(root / p) for p in tracked},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "dtype": "float32"},
        "started_utc": datetime.now(timezone.utc).isoformat()})
    started, total, old_threads = time.monotonic(), 0, torch.get_num_threads()
    models, artifacts = {}, {}
    policy, optimizer = None, None
    try:
        torch.set_num_threads(1)
        data = invented_data(config, config["training_times"])
        observations, bank, q, _ = data
        for representation in config["representations"]:
            for seed in config["initialization_seeds"]:
                name = f"{representation}-{seed}"
                policy = model(config, representation, seed)
                frozen = copy.deepcopy(policy)
                initial_sha = policy.snapshot_sha256()
                assert initial_sha == frozen.snapshot_sha256()
                with torch.no_grad():
                    initial, initial_values = scores(policy, observations, bank)
                    error = float((initial.softmax(1)[:, bank.reference_class]-.9).abs().max())
                    assert error <= config["gates"]["initial_reference_probability_absolute_error"]
                    assert (initial.argmax(1) == bank.reference_class).all() and (initial_values == 0).all()
                optimizer = torch.optim.Adam(policy.parameters(), lr=config["optimizer"]["learning_rate"])
                save_envelope(output / name / "initial.pt", {"manifest": policy.manifest(), "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(), "optimizer_calls": 0, "rng_cpu": torch.get_rng_state()})
                history, used = [], 0
                for step in range(128):
                    if time.monotonic()-started >= 1800:
                        raise TimeoutError("numerical acceptance wall cap exhausted")
                    optimizer.zero_grad(set_to_none=True)
                    logits, values = scores(policy, observations, bank)
                    if step % config["optimizer"]["old_distribution_refresh_every"] == 0:
                        old_log_probs = logits.detach().log_softmax(1)
                    loss = enumerated_objective(logits, values, old_log_probs, q, config["optimizer"])
                    gradients = gradient_readback(policy, loss, config["optimizer"])
                    loss.total.backward()
                    norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), config["optimizer"]["max_grad_norm"]))
                    if not math.isfinite(norm):
                        raise ValueError("nonfinite artificial gradient")
                    if time.monotonic()-started >= 1800:
                        raise TimeoutError("numerical acceptance wall cap exhausted before update")
                    total, used = charge(total, used, 1152, 128)
                    write_json_once(output / "charges" / f"{total:04d}.json", {
                        "fixture": name, "total_charged_optimizer_calls": total, "fixture_calls": used,
                        "elapsed_seconds": time.monotonic()-started})
                    optimizer.step()
                    if not all(torch.isfinite(p).all() for p in policy.parameters()):
                        raise ValueError("nonfinite artificial fitted weights")
                    row = {"step": used, "policy_loss": float(loss.policy.detach()), "value_loss_normalized": float(loss.value.detach()),
                           "entropy": float(loss.entropy.detach()), "total_loss": float(loss.total.detach()),
                           "ratio_clip_fraction": float(loss.clip_fraction), "total_grad_norm_before_clip": norm, **gradients}
                    write_json_once(output / name / "updates" / f"{used:03d}.json", row)
                    history.append(row)
                assert initial_sha == frozen.snapshot_sha256()
                save_envelope(output / name / "final.pt", {"manifest": policy.manifest(), "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(), "optimizer_calls": used, "rng_cpu": torch.get_rng_state(),
                    "old_log_probs": old_log_probs, "invented_q_values": q, "next_step": 128})
                write_json_once(output / name / "training.json", {"optimizer_calls": used, "initial_sha256": initial_sha,
                    "final_sha256": policy.snapshot_sha256(), "parameter_count": sum(p.numel() for p in policy.parameters()),
                    "manifest": policy.manifest(), "updates": history, "initial_probability_max_error": error})
                models[name] = (policy, frozen)
                artifacts[name] = sha256_file(output / name / "final.pt")
                print(json.dumps({"fixture": name, "optimizer_calls": used, "total_optimizer_calls": total}), flush=True)
        write_json_once(output / "all-models-sealed.json", {"models": artifacts, "optimizer_calls": total})
        heldout = invented_data(config, config["heldout_times"])
        checks = {}
        for name, (policy, frozen) in models.items():
            if time.monotonic()-started >= 1800:
                raise TimeoutError("numerical acceptance wall cap exhausted before heldout readback")
            checks[name] = {"continued": evaluate(policy, heldout, config["gates"]),
                            "frozen": evaluate(frozen, heldout, config["gates"])}
            write_json_once(output / name / "heldout.json", checks[name])
        numerical_seconds = time.monotonic()-started
        assert total == 1152
        for relative, digest in read(output / "claim.json")["source_hashes"].items():
            assert sha256_file(root / relative) == digest, "frozen engineering source changed"
        for part in before:
            assert inventory(p2 / part) == before[part], "original evidence changed"
        result = {"status": "completed", "exit_code": 0, "engineering_passed": all(x["continued"]["passed"] for x in checks.values()),
            "checks": checks, "optimizer_calls": total, "numerical_seconds": numerical_seconds,
            "new_patient_calls": 0, "new_scientific_fits": 0, "artificial_fits": 9,
            "original_p2_payload_launcher_unchanged": True, "original_source_document_locks_verified": True,
            "source_commit": commit, "decision": "artificial_acceptance_only_no_patient_pilot_authorized"}
        write_json_once(output / "terminal.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "checks"}), flush=True)
    except BaseException:
        error = traceback.format_exc()
        partial_error = None
        if policy is not None and optimizer is not None:
            try:
                save_envelope(output / "partial-on-failure.pt", {"manifest": policy.manifest(), "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(), "charged_optimizer_calls": total, "rng_cpu": torch.get_rng_state()})
            except BaseException:
                partial_error = traceback.format_exc()
        write_json_once(output / "terminal-failure.json", {"status": "failed", "traceback": error,
            "charged_optimizer_calls": total, "elapsed_seconds": time.monotonic()-started,
            "partial_checkpoint_error": partial_error,
            "scientific_retry_permitted": False, "artificial_retry_permitted": False})
        raise
    finally:
        torch.set_num_threads(old_threads)
