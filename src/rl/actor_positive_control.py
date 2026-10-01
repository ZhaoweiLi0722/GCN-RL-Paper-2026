"""Bounded exact-label actor engineering check; no patient environment or critic."""

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time
import traceback

import numpy as np

from src.models.fixed_bank_actor import FixedBankActor
from src.rl.candidate_calibration_engineering import charge, fixture_schema, invented_data, read, scores
from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_ppo_objective import candidate_ppo_loss
from src.rl.networks import torch
from src.utils.research_archive import inventory, sha256_file


CONFIG = "experiments/configs/candidate_actor_positive_control_20261001.json"
PROTOCOL = "specs/2026-10-01-actor-positive-control/protocol.md"


def model(config, bank, representation, seed):
    if representation not in ("graph", "self_only", "flat"):
        raise ValueError("unknown representation")
    return FixedBankActor(fixture_schema(), bank,
        architecture="flat" if representation == "flat" else "graph",
        message_mode="physical" if representation == "graph" else "self_only",
        encoder_width=config["encoder_width"], seed=seed, units=config["units"],
        initial_reference_logit=config["initial_reference_logit"])


def actor_objective(logits, old_log_probs, q_values, settings):
    if logits.ndim != 2 or old_log_probs.shape != logits.shape or q_values.shape != logits.shape:
        raise ValueError("matching decision-by-class matrices required")
    if any(x.dtype != logits.dtype or x.device != logits.device or not torch.isfinite(x).all()
           for x in (old_log_probs, q_values)):
        raise ValueError("finite matching precision required")
    old = old_log_probs.detach().exp()
    if not torch.allclose(old.sum(1), torch.ones_like(old[:, 0]), atol=1e-6, rtol=0):
        raise ValueError("old class probabilities must sum to one")
    logp = logits.log_softmax(1)
    advantages = (q_values-(old*q_values).sum(1, keepdim=True)).detach()
    weighted = logits.shape[1]*old*advantages
    entropy = -(logp.exp()*logp).sum(1)
    zero = torch.zeros_like(logp).flatten()
    return candidate_ppo_loss(logp.flatten(), zero, entropy[:, None].expand_as(logp).flatten(),
        old_log_probs.detach().flatten(), weighted.flatten(), zero,
        clip_ratio=settings["clip_ratio"], value_loss_coef=0., entropy_coef=settings["entropy_coef"])


def evaluate(policy, data, gates):
    observations, bank, q, labels = data
    with torch.no_grad():
        logits, sentinel = scores(policy, observations, bank)
        assert (sentinel == 0).all()
        probabilities = logits.softmax(1)
        rivals = logits.clone()
        rivals[torch.arange(len(labels)), labels] = -torch.inf
        margins = logits[torch.arange(len(labels)), labels]-rivals.max(1).values
        accuracy = float((logits.argmax(1) == labels).float().mean())
        checks = {"accuracy": accuracy >= gates["heldout_greedy_accuracy"],
                  "margin": float(margins.min()) >= gates["heldout_minimum_winner_logit_margin"]}
        return {"accuracy": accuracy, "minimum_winner_margin": float(margins.min()),
            "logits": logits.tolist(), "probabilities": probabilities.tolist(),
            "q_values": q.tolist(), "labels": labels.tolist(), "greedy_classes": logits.argmax(1).tolist(),
            "expected_invented_return": (probabilities*q).sum(1).tolist(),
            "greedy_invented_return": q[torch.arange(len(labels)), logits.argmax(1)].tolist(),
            "passed": all(checks.values()), "gates": checks}


def rng_snapshot():
    algorithm, keys, position, has_gaussian, gaussian = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": algorithm, "keys_uint32": keys.tolist(), "position": position,
                      "has_gaussian": has_gaussian, "cached_gaussian": gaussian},
            "torch_cpu": torch.get_rng_state()}


def checkpoint(policy, optimizer, config, total, used, started, old_log_probs):
    return {"manifest": policy.manifest(), "policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
            "config": config, "rng": rng_snapshot(), "optimizer_calls": used,
            "total_charged_optimizer_calls": total, "next_step": used,
            "elapsed_seconds": time.monotonic()-started, "old_log_probs": old_log_probs,
            "resume_authorized": False}


def remaining_time(started, limit):
    if time.monotonic()-started >= limit:
        raise TimeoutError("single-attempt numerical time cap exhausted; no retry")


def prior_evidence(root):
    p2 = root / "results/candidate_reference_prior_pilot_20261001"
    packet = read(p2 / "payload/locks/effective-execution.json")
    for relative, digest in (packet["source_files"] | packet["document_locks"]).items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"immutable P2 source/document mismatch: {relative}")
    old_run = root / "results/candidate_calibration_engineering_20261001"
    old_report = root / "reports/2026-10-01-candidate-calibration-engineering/verification.json"
    trees = {"p2_payload": inventory(p2 / "payload"), "p2_launcher": inventory(p2 / "launcher"),
             "calibration": inventory(old_run)}
    assert trees["p2_payload"] == read(p2 / "archives/completed-payload.tar.gz.manifest.json")["files"]
    assert trees["calibration"] == read(old_report)["artifact_inventory"]
    return trees


def run(root):
    root = Path(root).resolve()
    config = read(root / CONFIG)
    assert config["engineering_only"] and not config["scientific_execution_authorized"]
    assert config["patient_calls_authorized"] == 0 and config["no_retry_or_expansion"]
    assert config["caps"] == {"fixtures": 9, "total_optimizer_step_calls": 1152, "numerical_wall_seconds": 1800}
    assert config["representations"] == ["graph", "self_only", "flat"]
    assert config["initialization_seeds"] == [101, 102, 103]
    assert config["optimizer"]["steps_per_fixture"] == 128
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    output = root / config["output"]
    if output.exists():
        raise FileExistsError("preserve the existing attempt; no relaunch")
    before = prior_evidence(root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tracked = subprocess.check_output(["git", "ls-files", "src", "experiments/scripts", "tests", "AGENTS.md", CONFIG, PROTOCOL],
                                     cwd=root, text=True).splitlines()
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(output / "claim.json", {"commit": commit, "config_sha256": sha256_file(root / CONFIG),
        "protocol_sha256": sha256_file(root / PROTOCOL), "source_hashes": {p: sha256_file(root / p) for p in tracked},
        "mode": "single-artificial-actor-only-attempt", "started_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                    "platform": platform.platform(), "device": "cpu", "dtype": "float32", "torch_threads": 1}})
    write_json_once(output / "config.json", config)
    write_json_once(output / "prior-evidence.json", before)
    started, total, used = time.monotonic(), 0, 0
    old_threads = torch.get_num_threads()
    policies, seals = {}, {}
    policy, optimizer, old_log_probs = None, None, None
    try:
        torch.set_num_threads(1)
        observations, bank, q, _ = invented_data(config, config["training_times"])
        for representation in config["representations"]:
            for seed in config["initialization_seeds"]:
                remaining_time(started, 1800)
                name = f"{representation}-{seed}"
                policy = model(config, bank, representation, seed)
                frozen = copy.deepcopy(policy)
                initial_sha = policy.snapshot_sha256()
                assert frozen.snapshot_sha256() == initial_sha
                with torch.no_grad():
                    initial, sentinel = scores(policy, observations, bank)
                    expected = torch.zeros_like(initial)
                    expected[:, bank.reference_class] = config["initial_reference_logit"]
                    error = float((initial-expected).abs().max())
                    assert error <= config["gates"]["initial_reference_logit_absolute_error"]
                    assert (initial.argmax(1) == bank.reference_class).all() and (sentinel == 0).all()
                optimizer = torch.optim.Adam(policy.parameters(), lr=config["optimizer"]["learning_rate"])
                used, old_log_probs, history = 0, None, []
                save_envelope(output / name / "initial.pt", checkpoint(policy, optimizer, config, total, used, started, old_log_probs))
                for step in range(128):
                    remaining_time(started, 1800)
                    optimizer.zero_grad(set_to_none=True)
                    logits, _ = scores(policy, observations, bank)
                    if step % config["optimizer"]["old_distribution_refresh_every"] == 0:
                        old_log_probs = logits.detach().log_softmax(1)
                    loss = actor_objective(logits, old_log_probs, q, config["optimizer"])
                    loss.total.backward()
                    norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), config["optimizer"]["max_grad_norm"]))
                    if not math.isfinite(norm):
                        raise ValueError("nonfinite actor gradient")
                    remaining_time(started, 1800)
                    total, used = charge(total, used, 1152, 128)
                    write_json_once(output / "charges" / f"{total:04d}.json", {"fixture": name,
                        "total_charged_optimizer_calls": total, "fixture_calls": used, "elapsed_seconds": time.monotonic()-started})
                    optimizer.step()
                    if not all(torch.isfinite(p).all() for p in policy.parameters()):
                        raise ValueError("nonfinite actor weights")
                    row = {"step": used, "policy_loss": float(loss.policy.detach()), "entropy": float(loss.entropy.detach()),
                           "total_loss": float(loss.total.detach()), "value_loss": float(loss.value.detach()),
                           "ratio_clip_fraction": float(loss.clip_fraction), "grad_norm_before_clip": norm}
                    write_json_once(output / name / "updates" / f"{used:03d}.json", row)
                    history.append(row)
                assert frozen.snapshot_sha256() == initial_sha
                save_envelope(output / name / "final.pt", checkpoint(policy, optimizer, config, total, used, started, old_log_probs))
                write_json_once(output / name / "training.json", {"initial_sha256": initial_sha,
                    "final_sha256": policy.snapshot_sha256(), "optimizer_calls": used, "updates": history,
                    "manifest": policy.manifest(), "initial_logit_max_error": error,
                    "parameter_count": sum(p.numel() for p in policy.parameters())})
                policies[name] = (policy, frozen)
                seals[name] = sha256_file(output / name / "final.pt")
                print(json.dumps({"fixture": name, "optimizer_calls": used, "total_optimizer_calls": total}), flush=True)
        write_json_once(output / "all-models-sealed.json", {"models": seals, "optimizer_calls": total})
        heldout = invented_data(config, config["heldout_times"])
        checks = {}
        for name, (policy, frozen) in policies.items():
            remaining_time(started, 1800)
            checks[name] = {"continued": evaluate(policy, heldout, config["gates"]),
                            "frozen": evaluate(frozen, heldout, config["gates"])}
            write_json_once(output / name / "heldout.json", checks[name])
        remaining_time(started, 1800)
        numerical_seconds = time.monotonic()-started
        assert total == 1152
        assert prior_evidence(root) == before, "historical evidence changed"
        for relative, digest in read(output / "claim.json")["source_hashes"].items():
            assert sha256_file(root / relative) == digest, "frozen source changed"
        result = {"status": "completed", "exit_code": 0, "engineering_passed": all(x["continued"]["passed"] for x in checks.values()),
            "checks": checks, "optimizer_calls": total, "numerical_seconds": numerical_seconds,
            "new_patient_calls": 0, "new_scientific_fits": 0, "artificial_fits": 9, "prior_evidence_unchanged": True,
            "source_commit": commit, "decision": "positive_control_only_no_patient_pilot_authorized"}
        write_json_once(output / "terminal.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "checks"}), flush=True)
    except BaseException:
        error, partial_error = traceback.format_exc(), None
        if policy is not None and optimizer is not None:
            try:
                save_envelope(output / "partial-on-failure.pt", checkpoint(policy, optimizer, config, total, used, started, old_log_probs))
            except BaseException:
                partial_error = traceback.format_exc()
        write_json_once(output / "terminal-failure.json", {"status": "failed", "traceback": error,
            "charged_optimizer_calls": total, "elapsed_seconds": time.monotonic()-started,
            "partial_checkpoint_error": partial_error, "retry_permitted": False})
        raise
    finally:
        torch.set_num_threads(old_threads)
