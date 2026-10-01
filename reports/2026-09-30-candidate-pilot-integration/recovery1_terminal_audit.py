"""Read back the closed P1-R1 attempt; no simulator or optimizer operations.

Run from the repository root with PYTHONPATH=. using the locked interpreter.
Only saved demonstration/qualification states are scored; no new test data.
The output is exclusive and outside the immutable experimental result tree.
"""

from collections import Counter
import json
from pathlib import Path
import subprocess

import torch

from src.models.candidate_policy import CandidatePolicy
from src.models.matched_inputs import InputSchema
from src.rl.candidate_imitation import decode_example
from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_execution import configure_runtime, verify_packet
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import read_ledger
from src.rl.candidate_pilot_verification import verify_episode
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.validated_returns import ReplaySemantics
from src.utils.research_archive import inventory, sha256_file


ROOT = Path.cwd()
RUN = ROOT / "results/candidate_return_pilot_20260930_recovery1"
OUT = ROOT / "reports/2026-09-30-candidate-pilot-integration/recovery1-terminal-audit.json"


def read(path):
    return json.loads(path.read_text())


def score_saved(model, contract, examples):
    rows = []
    with torch.inference_mode():
        for example in examples:
            obs, bank = decode_example(example, contract)
            logits = model(obs, bank).logits
            probabilities = torch.softmax(logits, dim=0)
            picked = int(logits.argmax())
            reference = bank.reference_class
            other = torch.cat((logits[:reference], logits[reference + 1:]))
            rows.append({"identity": example["identity"], "selected_class": picked,
                         "reference_class": reference, "classes": len(bank.class_keys),
                         "correct": picked == reference,
                         "reference_probability": float(probabilities[reference]),
                         "reference_margin": float(logits[reference] - other.max()),
                         "entropy": float(-(probabilities * probabilities.log()).sum())})
    hits = sum(row["correct"] for row in rows)
    return {"rows": len(rows), "hits": hits, "agreement": hits / len(rows),
            "mean_reference_probability": sum(r["reference_probability"] for r in rows) / len(rows),
            "mean_entropy": sum(r["entropy"] for r in rows) / len(rows),
            "minimum_reference_margin": min(r["reference_margin"] for r in rows),
            "support_counts": dict(sorted(Counter(r["classes"] for r in rows).items())),
            "misses": [r for r in rows if not r["correct"]]}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    configure_runtime()
    before = inventory(RUN)
    effective_path = ROOT / "experiments/configs/candidate_return_pilot_20260930_recovery1_execution.json"
    packet = read(effective_path)
    config = verify_packet(ROOT, packet, require_clean=False, recovery=True)
    terminal, failure = read(RUN / "launcher/terminal.json"), read(RUN / "launcher/failure.json")
    assert terminal["status"] == "failed" and terminal["supervisor"]["exit_code"] == 1
    assert failure["sequence"]["active"] == "qualification" and failure["retry_permitted"] is False
    ledger = read_ledger(RUN / "launcher/budget.jsonl")
    assert ledger["counts"] == failure["budget"]["counts"] == {"environment": 2064, "optimizer": 2304}
    assert ledger["last_sha256"] == failure["budget"]["ledger_sha256"]
    saved = load_envelope(RUN / "launcher/failure-state.pt")
    assert not saved["models"] and not saved["test_index"] and not saved["model_paths"]
    assert saved["continuation"] is None and saved["budget"] == failure["budget"]
    assert not (RUN / "payload/episodes/training").exists()
    assert not (RUN / "payload/episodes/test").exists()

    raw_examples, outcomes = {}, []
    for header_path in sorted((RUN / "payload/episodes").rglob("header.json")):
        directory = header_path.parent
        header, final = read(header_path), read(directory / "final_state.json")
        rows = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        outcome = verify_episode(header, rows, final, config)
        assert outcome == read(directory / "outcome.json")
        outcomes.append(outcome)
        if header["split"] in ("demonstration", "qualification"):
            for row in rows:
                audit = row["event"]["audit"]
                record = audit["record"]
                identity = f"{record['trajectory_id']}/{record['step_index']}"
                assert identity not in raw_examples
                raw_examples[identity] = (record["state"], audit["decision"]["evaluation"]["candidates"])
    split_counts = Counter(o["split"] for o in outcomes)
    assert split_counts == {"preflight": 9, "demonstration": 24, "qualification": 6}
    assert sum(o["steps"] for o in outcomes) + 36 == ledger["counts"]["environment"]
    for group in ("demonstrations", "qualification"):
        for block, examples in saved[group].items():
            assert len(examples) == (416 if group == "demonstrations" else 104)
            for example in examples:
                raw_state, raw_bank = raw_examples.pop(example["identity"])
                assert list(example["actor_state"]) == raw_state
                assert json.dumps(example["candidates"], sort_keys=True) == json.dumps(raw_bank, sort_keys=True)
    assert not raw_examples

    qualifications = read(RUN / "payload/qualification.json")
    models = {}
    for key, state in sorted(saved["initializers"].items()):
        fit_path = RUN / "payload/initialization" / key / "fit.pt"
        assert state_digest(load_envelope(fit_path)) == state_digest(state) == qualifications[key]["kernel_sha256"]
        assert state["steps"] == 256
        manifest = state["manifest"]
        contract = ReplayInputContract(InputSchema(**manifest["contract"]["inputs"]),
                                       ReplaySemantics(**manifest["contract"]["replay"]))
        spec = manifest["policy"]
        model = CandidatePolicy(contract.inputs, enabled=True, seed=0,
                                architecture=spec["architecture"], message_mode=spec["message_mode"],
                                encoder_width=spec["encoder_width"], head_width=spec["head_width"])
        model.load_state_dict(state["policy"])
        model.requires_grad_(False).eval()
        model_before = model.snapshot_sha256()
        block = int(key.split("/")[0][5:])
        demo = score_saved(model, contract, saved["demonstrations"][block])
        qual = score_saved(model, contract, saved["qualification"][block])
        assert model.snapshot_sha256() == model_before
        assert qual["agreement"] == qualifications[key]["agreement"]
        assert (qual["agreement"] >= .95) == qualifications[key]["passed"]
        assert qualifications[key]["multi_class_rows"] == 104
        logs = read(fit_path.with_suffix(".json"))["minibatches"]
        assert len(logs) == 256 and all(torch.isfinite(torch.tensor([r["cross_entropy"], r["grad_norm"]])).all() for r in logs)
        models[key] = {"parameters": sum(p.numel() for p in model.parameters()),
                       "optimizer_steps": state["steps"], "policy_sha256": model_before,
                       "demonstration": demo, "qualification": qual,
                       "first_minibatch_ce": logs[0]["cross_entropy"],
                       "last_minibatch_ce": logs[-1]["cross_entropy"]}
    assert len(models) == 9
    assert inventory(RUN) == before
    report = {"kind": "closed-p1-r1-saved-evidence-readback", "execution_commit":
              subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "effective_sha256": sha256_file(effective_path), "run_inventory": before,
              "run_unchanged": True, "source_runtime_prior_input_locks_verified": True,
              "ledger": ledger, "terminal": terminal, "models": models,
              "raw_episode_counts": dict(split_counts), "raw_outcomes": outcomes,
              "persisted_examples_match_raw_events": True,
              "qualification_independently_recomputed": True,
              "additional_simulator_calls": 0, "additional_optimizer_steps": 0,
              "test_data_read": False, "ppo_started": False,
              "imitation_continuation_started": False,
              "stderr_bytes": (RUN / "launcher/stderr.log").stat().st_size,
              "claim": "Initialization gate failure, not an RL performance contrast.",
              "diagnostic_limit": "Training and qualification inferences are descriptive readbacks; no fresh test evidence."}
    write_json_once(OUT, report)
    print(json.dumps({"report": str(OUT), "files": len(before), "counts": ledger["counts"],
                      "qualification": {k: {"demonstration_agreement": v["demonstration"]["agreement"],
                            "qualification_agreement": v["qualification"]["agreement"],
                            "reference_probability": v["qualification"]["mean_reference_probability"]}
                            for k, v in models.items()}}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
