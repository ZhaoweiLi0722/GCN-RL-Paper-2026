"""Audit saved sampled-control artifacts without model construction or inference.

Torch is used only to read tensor envelopes and replay private categorical RNG
from saved logits. No model forward, reward observation or optimizer is called.
"""

import json
import math
from pathlib import Path

from src.rl.actor_control_verification import weight_digest
from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.networks import torch
from src.rl.sampled_return_verification import actor_metrics, evaluation_metrics, verify_charge, verify_sample_batch
from src.rl.prospective_ddpg_kernel import state_digest
from src.utils.research_archive import inventory, sha256_file


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_optimizer(state, weights, expected_steps, settings):
    require(len(state["param_groups"]) == 1, "one optimizer group required")
    group = state["param_groups"][0]
    require(group["lr"] == settings["learning_rate"] and group["weight_decay"] == 0
            and tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8, "optimizer settings changed")
    require(len(group["params"]) == len(set(group["params"])) == len(weights), "optimizer parameter count differs")
    if expected_steps == 0:
        require(not state["state"], "initial optimizer already consumed")
        return
    require(set(state["state"]) == set(group["params"]), "missing optimizer state")
    for identity, tensor in zip(group["params"], weights.values()):
        moment = state["state"][identity]
        require(float(moment["step"]) == expected_steps, "Adam step counter differs from charged updates")
        for key in ("exp_avg", "exp_avg_sq"):
            require(moment[key].shape == tensor.shape and moment[key].dtype == torch.float32
                    and torch.isfinite(moment[key]).all().item(), "invalid Adam moment")
        require((moment["exp_avg_sq"] >= 0).all().item(), "negative Adam squared moment")


def same_fields(a, b, fields):
    for key in fields:
        require(state_digest(a[key]) == state_digest(b[key]), f"unexpected state change: {key}")


class ReceiptReader:
    def __init__(self, run, config):
        self.run, self.config = Path(run), config
        self.sequence, self.charges, self.last_elapsed = 0, 0, 0.
        self.budget = {"observations": 0, "optimizer_calls": 0, "by_fit": {}}

    def event(self, kind, **identity):
        self.sequence += 1
        event = read(self.run / "events" / f"{self.sequence:05d}.json")
        require(event["sequence"] == self.sequence and event["kind"] == kind, "invalid event order")
        elapsed = event["elapsed_seconds"]
        require(math.isfinite(elapsed) and self.last_elapsed <= elapsed < 1800, "event outside numerical clock")
        self.last_elapsed = elapsed
        for key, value in identity.items():
            require(event.get(key) == value, f"event identity differs: {key}")
        return event

    def charge(self, fit, kind):
        self.charges += 1
        path = f"charges/{self.charges:05d}.json"
        self.event("charge", path=path)
        receipt = read(self.run / path)
        require(receipt["fixture"] == fit and receipt["kind"] == kind, "charge arm/phase mismatch")
        self.budget = verify_charge(self.budget, receipt, self.config)

    def checkpoint(self, relative, name, rollout, cursor):
        state = load_envelope(self.run / relative)
        require(state["fixture"] == name and state["config"] == self.config and state["rollout"] == rollout
                and state["next_minibatch"] == cursor and state["budget"] == self.budget
                and state["resume_authorized"] is False, "checkpoint lineage/budget differs")
        require(set(state["global_rng"]) == {"python", "numpy", "torch_cpu"}, "incomplete global RNG")
        for role in ("actor", "critic"):
            steps = self.budget["by_fit"].get(name, {}).get(role, 0)
            weights = state[role]
            require(all(t.dtype == torch.float32 and t.device.type == "cpu" and torch.isfinite(t).all().item()
                        for t in weights.values()), "invalid saved weights")
            verify_optimizer(state[role+"_optimizer"], weights, steps, self.config["optimizer"])
        return state


def verify(root):
    root = Path(root).resolve()
    run = root / "results/candidate_sampled_return_control_20261001"
    report_path = root / "reports/2026-10-01-sampled-return-control/verification.json"
    if report_path.exists():
        raise FileExistsError("preserve the existing verification")
    before = inventory(run)
    require({k: v for k, v in before.items() if k != "inventory.json"} == read(run / "inventory.json"),
            "recorded artifact inventory differs")
    config, claim, authorization, terminal, seals = (read(run / p) for p in
        ("config.json", "claim.json", "authorization.json", "terminal.json", "all-models-sealed.json"))
    require(terminal["status"] == "completed" and terminal["exit_code"] == 0
            and terminal["new_patient_calls"] == terminal["new_patient_fits"] == 0
            and terminal["artificial_fits"] == 9, "completed artificial-only packet required")
    require(0 < terminal["numerical_seconds"] < 1800, "numerical cap exceeded")
    require(authorization["decision"] == "approved_by_zhaowei" and authorization["implementation_ready"] is True,
            "missing recorded approval")
    require(authorization.get("scope") == "one-artificial-sampled-return-packet-no-patient-calls"
            and authorization.get("caps") == config["caps"]
            and config["caps"] == {"fits": 9, "actor_calls_per_fit": 128, "critic_calls_per_fit": 128,
                "observations_per_fit": 1536, "total_optimizer_calls": 2304, "total_training_observations": 13824,
                "numerical_wall_seconds": 1800, "attempts": 1}
            and config["representations"] == ["graph", "self_only", "flat"]
            and config["initialization_seeds"] == [101, 102, 103]
            and config["scientific_execution_authorized"] is False and config["patient_calls_authorized"] == 0,
            "authorization scope or fixed packet differs")
    require(claim["source_hashes"] == authorization["source_hashes"], "source locks differ")
    for relative, digest in claim["source_hashes"].items():
        require(sha256_file(root / relative) == digest, f"source changed: {relative}")
    config_path = root / "experiments/configs/candidate_sampled_return_control_20261001.json"
    protocol_path = root / "specs/2026-10-01-sampled-return-control/protocol.md"
    authorization_path = protocol_path.with_name("execution_authorization.json")
    require(read(config_path) == config and sha256_file(config_path) == authorization["config_sha256"]
            and sha256_file(protocol_path) == authorization["protocol_sha256"]
            and read(authorization_path) == authorization
            and sha256_file(authorization_path) == claim["authorization_sha256"], "authorization/config/protocol changed")
    names = [f"{r}-{s}" for r in config["representations"] for s in config["initialization_seeds"]]
    require(len(names) == 9 and set(names) == set(seals["models"]) == set(terminal["checks"]), "missing/extra fit")
    reader, records = ReceiptReader(run, config), {}
    model_fields = ("actor", "critic", "actor_optimizer", "critic_optimizer", "actor_manifest", "global_rng")
    rng_fields = ("sampling_rng", "shuffle_rng")
    reference = sorted(config["request_values"]).index(config["request_values"][0])
    for name in names:
        seed = int(name.rsplit("-", 1)[1])
        initial = reader.checkpoint(f"{name}/initial.pt", name, 0, 0)
        initial_digest = weight_digest({"manifest": initial["actor_manifest"], "policy": initial["actor"]})
        reader.event("fit_started", fixture=name, initial_actor_sha256=initial_digest)
        require(all((tensor == 0).all().item() for tensor in initial["critic"].values()), "critic did not start at zero")
        require((initial["actor"]["actor_head.weight"] == 0).all().item(), "actor head did not start at zero")
        for action, value in enumerate(initial["actor"]["actor_head.bias"].tolist()):
            require(abs(value*config["units"]["logit_gain"]-(config["initial_reference_logit"] if action == reference else 0)) < 1e-7,
                    "initial actor prior differs")
        sample_rng = torch.Generator().manual_seed(config["sampling"]["sampling_seed_offset"]+seed)
        shuffle_rng = torch.Generator().manual_seed(config["sampling"]["shuffle_seed_offset"]+seed)
        require(torch.equal(initial["sampling_rng"], sample_rng.get_state())
                and torch.equal(initial["shuffle_rng"], shuffle_rng.get_state()), "initial private RNG differs")
        previous, returns, coverage = initial, [], {f"{i}:{a}": 0 for i in range(12) for a in range(6)}
        for rollout in range(32):
            prefix = f"{name}/rollout-{rollout:02d}"
            start = reader.checkpoint(prefix+"/before.pt", name, rollout, 0)
            require(start["batch"] is None and start["permutation"] is None, "rollout starts with stale batch")
            same_fields(start, previous, model_fields+rng_fields)
            reader.charge(name, "observations")
            sample_path = prefix+"/samples.json"
            receipt = read(run / sample_path)
            arithmetic = verify_sample_batch(receipt, config)
            collected = reader.checkpoint(prefix+"/collected.pt", name, rollout, 0)
            same_fields(start, collected, model_fields)
            require({k: v.tolist() for k, v in collected["batch"].items()} == receipt, "batch checkpoint differs")
            # RNG replay uses saved probability inputs only, not a model or task call.
            logits = torch.tensor(receipt["behavior_logits"], dtype=torch.float32)
            actions = torch.multinomial(logits.softmax(1), 1, generator=sample_rng).flatten().tolist()
            permutation = torch.randperm(48, generator=shuffle_rng)
            require(actions == receipt["actions"] and torch.equal(permutation, collected["permutation"])
                    and torch.equal(sample_rng.get_state(), collected["sampling_rng"])
                    and torch.equal(shuffle_rng.get_state(), collected["shuffle_rng"]), "sampling/shuffle RNG replay differs")
            reader.event("samples", fixture=name, rollout=rollout, path=sample_path)
            returns.extend(receipt["returns"])
            coverage = {key: value+arithmetic["coverage"][key] for key, value in coverage.items()}
            previous = collected
            for index in range(4):
                reader.charge(name, "actor")
                reader.charge(name, "critic")
                update_path = prefix+f"/update-{index}"
                metrics = read(run / (update_path+".json"))
                require(metrics.pop("selected") == permutation[12*index:12*(index+1)].tolist(), "minibatch membership differs")
                require(all(type(v) in (int, float) and math.isfinite(v) for v in metrics.values()), "nonfinite update metric")
                require(0 <= metrics["clip_fraction"] <= 1 and min(metrics["actor_grad_norm"], metrics["critic_grad_norm"], metrics["critic_mse"]) >= 0,
                        "invalid gradient/loss metric")
                require(abs(metrics["actor_total"]-(metrics["policy_loss"]-config["optimizer"]["entropy_coef"]*metrics["entropy"])) < 2e-6,
                        "actor objective arithmetic differs")
                updated = reader.checkpoint(update_path+".pt", name, rollout, index+1)
                same_fields(collected, updated, ("batch", "permutation", "global_rng")+rng_fields)
                reader.event("update", fixture=name, rollout=rollout, minibatch=index,
                             state=update_path+".pt", sha256=sha256_file(run / (update_path+".pt")))
                previous = updated
        final = reader.checkpoint(f"{name}/final.pt", name, 31, 4)
        same_fields(final, previous, model_fields+rng_fields+("batch", "permutation"))
        training = read(run / name / "training.json")
        final_digest = weight_digest({"manifest": final["actor_manifest"], "policy": final["actor"]})
        require(training["initial_actor_sha256"] == training["frozen_actor_sha256"] == initial_digest
                and training["final_actor_sha256"] == final_digest, "model digest differs")
        require(training["actor_manifest"] == json.loads(json.dumps(final["actor_manifest"])), "actor manifest differs")
        require(training["counts"] == {"observations": 1536, "actor": 128, "critic": 128}
                and training["coverage"] == coverage, "training count/coverage differs")
        require(abs(training["training_return_mean"]-math.fsum(returns)/1536) < 1e-12, "training baseline differs")
        for role in ("actor", "critic"):
            require(training[role+"_parameter_count"] == sum(t.numel() for t in final[role].values()), "parameter count differs")
        digest = sha256_file(run / name / "final.pt")
        require(seals["models"][name] == digest, "final seal changed")
        reader.event("fit_sealed", fixture=name, sha256=digest)
        records[name] = {"coverage": coverage, "training_return_mean": math.fsum(returns)/1536,
                         "actor_parameter_count": training["actor_parameter_count"],
                         "critic_parameter_count": training["critic_parameter_count"]}
    reader.event("all_models_sealed", models=seals["models"])
    require(seals["budget"] == terminal["budget"] == reader.budget
            and reader.budget["observations"] == 13824 and reader.budget["optimizer_calls"] == 2304, "final budget differs")
    for name in names:
        raw = read(run / name / "heldout.json")
        require(set(raw) == {"final_logits", "frozen_logits", "critic_values"}, "unexpected heldout fields")
        for row in raw["frozen_logits"]:
            require(len(row) == 6 and all(abs(v-(config["initial_reference_logit"] if i == reference else 0)) < 1e-7
                                        for i, v in enumerate(row)), "frozen logits differ from start")
        stats = records[name]
        checks = {"continued": evaluation_metrics(raw["final_logits"], raw["critic_values"],
                      stats["training_return_mean"], stats["coverage"], config),
                  "frozen": actor_metrics(raw["frozen_logits"], config)}
        require(checks == terminal["checks"][name], "reported heldout arithmetic differs")
        records[name]["checks"] = checks
        reader.event("evaluation", fixture=name)
    require(len(list((run / "events").glob("*.json"))) == reader.sequence
            and len(list((run / "charges").glob("*.json"))) == reader.charges == 2592, "extra/missing receipts")
    require(reader.last_elapsed <= terminal["numerical_seconds"], "terminal clock predates final evaluation")
    passed = sum(item["checks"]["continued"]["passed"] for item in records.values())
    require(terminal["engineering_passed"] == (passed == 9), "overall gate differs")
    prior_roots = {"p2_payload": "candidate_reference_prior_pilot_20261001/payload",
        "p2_launcher": "candidate_reference_prior_pilot_20261001/launcher", "calibration": "candidate_calibration_engineering_20261001",
        "actor_positive_control": "candidate_actor_positive_control_20261001"}
    old = read(run / "prior-evidence.json")
    for key, relative in prior_roots.items():
        require(inventory(root / "results" / relative) == old[key], "historical evidence changed")
    require(inventory(run) == before, "packet changed during independent verification")
    report = {"format": "sampled-return-packet-verification-v1", "passed_fixtures": passed,
        "total_fixtures": 9, "optimizer_charges_verified": 2304, "training_observations_verified": 13824,
        "private_rng_replay_verified": True, "all_models_sealed_before_evaluation": True,
        "prior_evidence_unchanged": True, "new_model_forwards_optimizer_calls_patient_calls": 0,
        "records": records, "artifact_inventory": before,
        "limits": "Saved-logit arithmetic, RNG and state audit; not a second neural-forward execution or patient-performance confirmation."}
    write_json_once(report_path, report)
    return {k: v for k, v in report.items() if k not in ("records", "artifact_inventory")}
