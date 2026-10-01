"""Closed P2 saved-data audit; no model forward, simulation or optimizer calls.

Run from the worktree with PYTHONPATH=. and the locked interpreter. Output is
exclusive and outside the immutable result tree. This is posthoc interpretation,
not fresh confirmation or a new candidate selection procedure.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import subprocess

import torch

from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_execution import configure_runtime
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import read_ledger
from src.rl.candidate_pilot_verification import verify_episode, json_hash
from src.rl.candidate_reference_execution import verify_packet
from src.utils.research_archive import inventory, sha256_file


ROOT = Path.cwd()
RUN = ROOT / "results/candidate_reference_prior_pilot_20261001"
OUT = ROOT / "reports/2026-10-01-reference-prior-integration/terminal-audit.json"


def read(path):
    return json.loads(path.read_text())


def distribution_row(decision):
    evaluation = decision["evaluation"]
    logp = evaluation["log_probs"]
    ref = evaluation["candidates"]["request_to_class"][0]
    chosen = decision["choice"]["class_index"]
    if (not logp or any(not math.isfinite(v) for v in logp)
            or not 0 <= ref < len(logp) or not 0 <= chosen < len(logp)):
        raise ValueError("invalid saved probability receipt")
    if not math.isclose(math.fsum(math.exp(v) for v in logp), 1., abs_tol=1e-6):
        raise ValueError("saved probabilities do not normalize")
    margin = None if len(logp) == 1 else logp[ref] - max(v for i, v in enumerate(logp) if i != ref)
    return {"reference_probability": math.exp(logp[ref]), "chosen_reference": chosen == ref,
            "classes": len(logp), "margin": margin,
            "learned_margin": None if margin is None else margin - math.log(9 * (len(logp) - 1))}


def distribution_summary(rows):
    if not rows:
        raise ValueError("empty receipt group")
    probabilities = [r["reference_probability"] for r in rows]
    margins = [r["margin"] for r in rows if r["margin"] is not None]
    learned = [r["learned_margin"] for r in rows if r["learned_margin"] is not None]
    return {"decisions": len(rows), "nonreference_choices": sum(not r["chosen_reference"] for r in rows),
            "classes": dict(Counter(r["classes"] for r in rows)),
            "reference_probability_min_mean_max": [min(probabilities), statistics.mean(probabilities), max(probabilities)],
            "minimum_reference_margin": min(margins) if margins else None,
            "learned_margin_min_max": [min(learned), max(learned)] if learned else None}


def weight_readback(initial, final):
    if initial["policy"].keys() != final["policy"].keys():
        raise ValueError("incompatible saved weights")
    changes = {}
    for name, before in initial["policy"].items():
        after = final["policy"][name]
        if before.shape != after.shape or not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise ValueError("invalid saved weights")
        changes[name] = float((after.double() - before.double()).abs().max())
    kind = final["manifest"]["format"]
    if kind == "candidate-imitation-v1":
        if "pending" in final or sum(h["steps"] for h in final["history"]) != final["steps"]:
            raise ValueError("invalid imitation update history")
        pending = None  # This synchronous BC state has no pending-segment field.
    elif kind == "candidate-ppo-kernel-v1" and "pending" in final:
        pending = len(final["pending"])
    else:
        raise ValueError("unrecognized checkpoint pending-state contract")
    return {"max_abs_delta_by_tensor": changes, "changed_tensors": sum(v > 0 for v in changes.values()),
            "tensor_count": len(changes), "optimizer_steps_per_parameter": sorted({
                float(v["step"]) for v in final["optimizer"]["state"].values()}),
            "completed_rollouts": len(final["history"]), "pending_segments": pending,
            "checkpoint_format": kind}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    configure_runtime()
    packet_path = ROOT / "experiments/configs/candidate_reference_prior_pilot_20261001_execution.json"
    config = verify_packet(ROOT, read(packet_path), require_clean=False)
    terminal, completed = read(RUN / "launcher/terminal.json"), read(RUN / "launcher/completed.json")
    assert terminal["status"] == completed["status"] == "completed"
    assert terminal["supervisor"]["exit_code"] == completed["exit_code"] == 0
    assert terminal["supervisor"]["passed"] and not terminal["supervisor"]["forced_kill"]
    assert len(completed["sequence"]["completed"]) == 56 and completed["sequence"]["active"] is None
    processes = subprocess.check_output(["ps", "-ax", "-o", "pid,ppid,etime,command"], text=True)
    related = [line for line in processes.splitlines() if " -m experiments.scripts.run_candidate_reference_pilot" in line]
    assert not related, related
    launcher_before = inventory(RUN / "launcher")
    payload_before = inventory(RUN / "payload")
    archived = read(RUN / "archives/completed-payload.tar.gz.manifest.json")
    assert payload_before == archived["files"]
    assert sha256_file(RUN / "archives/completed-payload.tar.gz") == archived["archive_sha256"]
    ledger = read_ledger(RUN / "launcher/budget.jsonl")
    assert ledger["counts"] == completed["budget"]["counts"] == {"environment": 51360, "optimizer": 2304}
    assert ledger["last_sha256"] == completed["budget"]["ledger_sha256"]
    recorded = read(RUN / "payload/independent-verification.json")
    outcomes, traces, distributions = [], {}, defaultdict(list)
    for path in sorted((RUN / "payload/episodes").rglob("header.json")):
        header, final = read(path), read(path.with_name("final_state.json"))
        rows = [json.loads(line) for line in path.with_name("events.jsonl").read_text().splitlines()]
        result = verify_episode(header, rows, final, config)
        assert result == read(path.with_name("outcome.json"))
        # This direct sum does not use the collector or analysis cost summarizer.
        raw_sum = math.fsum(row["event"]["info"]["cost"] for row in rows)
        assert math.isclose(raw_sum, result["cost"], rel_tol=1e-12, abs_tol=1e-6)
        outcomes.append(result)
        if result["split"] == "test":
            key = result["block"], result["representation"], result["role"], result["world_index"]
            assert key not in traces
            physical = [{"info": r["event"]["info"], "record": {k: r["event"]["audit"]["record"][k]
                for k in ("state", "next_state", "action", "raw_reward", "state_token", "next_state_token", "terminated", "truncated")}}
                for r in rows]
            traces[key] = json_hash(physical)
            distributions[key[:3]].extend(distribution_row(r["event"]["audit"]["decision"]) for r in rows)
    counts = Counter(o["split"] for o in outcomes)
    assert counts == {"preflight": 9, "qualification": 6, "training": 576, "test": 396}
    assert sum(o["steps"] for o in outcomes) + 36 == ledger["counts"]["environment"]
    tests = {(o["block"], o["representation"], o["role"], o["world_index"]): o for o in outcomes if o["split"] == "test"}
    assert len(tests) == 396
    assert tests == {(o["block"], o["representation"], o["role"], o["world_index"]): o for o in recorded["outcomes"]}
    aliases = []
    for key in sorted(tests):
        block, rep, role, world = key
        if rep == "reference":
            continue
        ref = (block, "reference", "r4", world)
        assert traces[key] == traces[ref]
        aliases.append({"block": block, "representation": rep, "role": role, "world": world,
                        "trace_sha256": traces[key], "equals_r4": True})
    recomputed = []
    for contrast in recorded["analysis"]["contrasts"]:
        lr, la = contrast["left"].split("/")
        rr, ra = contrast["right"].split("/")
        left, right, diffs = [], [], []
        clinical = defaultdict(list)
        for block in (60, 61, 62):
            for world in range(12):
                l, r = tests[block, lr, la, world], tests[block, rr, ra, world]
                assert l["seed"] == r["seed"]
                left.append(l["cost"]); right.append(r["cost"]); diffs.append(l["cost"] - r["cost"])
                for k in ("losses", "completions", "terminal_active", "enrolled"):
                    clinical[k].append(l[k] - r[k])
        delta = statistics.mean(diffs)
        gain = -100 * delta / statistics.mean(right)
        assert math.isclose(delta, contrast["grand_mean_cost_difference"], rel_tol=1e-12, abs_tol=1e-6)
        assert math.isclose(gain, contrast["grand_relative_cost_gain_percent"], rel_tol=1e-12, abs_tol=1e-10)
        recomputed.append({"left": contrast["left"], "right": contrast["right"], "paired_worlds": 36,
                           "mean_cost_difference": delta, "relative_gain_percent": gain,
                           "clinical_mean_differences": {k: statistics.mean(v) for k, v in clinical.items()}})
    models = {}
    seals = read(RUN / "payload/model-seals.json")
    assert len(seals) == 27
    for key, seal in seals.items():
        assert sha256_file(RUN / seal["path"]) == seal["sha256"]
        block, rep, role = key.split("/")
        initial = load_envelope(RUN / "payload/models" / key / "initial.pt")
        peers = [load_envelope(RUN / "payload/models" / block / rep / r / "initial.pt") for r in ("frozen", "ppo", "bc_continue")]
        assert all(all(torch.equal(p["policy"][k], v) for k, v in initial["policy"].items()) for p in peers)
        assert all(torch.equal(p["sampling_rng"], initial["sampling_rng"]) for p in peers)
        if role == "frozen":
            continue
        final = load_envelope(RUN / seal["path"])
        weights = weight_readback(initial, final)
        assert weights["changed_tensors"] > 0 and weights["optimizer_steps_per_parameter"] == [128.]
        assert weights["completed_rollouts"] == 8
        assert weights["pending_segments"] == (0 if role == "ppo" else None)
        models[key] = {**weights, "evaluation": distribution_summary(distributions[int(block[5:]), rep, role])}
    assert len(models) == 18
    logs = (RUN / "launcher/stdout.log").read_text() + (RUN / "launcher/stderr.log").read_text()
    assert not any(token in logs.lower() for token in ("traceback", "segmentation fault", "out of memory", "fatal error"))
    assert (RUN / "launcher/stderr.log").stat().st_size == 0
    assert inventory(RUN / "payload") == payload_before
    assert inventory(RUN / "launcher") == launcher_before
    report = {"format": "closed-p2-posthoc-saved-data-audit-v1", "execution_commit": read(RUN / "launcher/claim.json")["head"],
        "effective_sha256": sha256_file(packet_path), "terminal": terminal, "ledger": ledger,
        "source_runtime_prior_input_locks_verified": True, "payload_archive_members_match_originals": True,
        "payload_unchanged": True, "launcher_unchanged": True, "related_processes_after_exit": related,
        "raw_episode_counts": dict(counts), "raw_recorded_steps": sum(o["steps"] for o in outcomes), "clone_steps": 36,
        "full_closed_loop_aliases": aliases, "independent_contrast_arithmetic": recomputed,
        "models": models, "raw_outcomes": outcomes, "decision": recorded["analysis"]["decision"],
        "additional_simulator_calls": 0, "additional_optimizer_steps": 0, "additional_model_forwards": 0,
        "posthoc_not_independent_confirmation": True, "stderr_bytes": 0,
        "limitation": "Greedy decisions equal R4 on these worlds; not global optimality, absence of headroom or universal RL failure."}
    write_json_once(OUT, report)
    print(json.dumps({"report": str(OUT), "raw_episode_counts": dict(counts), "models": len(models),
                      "r4_trace_aliases": len(aliases), "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
