"""Saved-data selection and independent raw reconciliation; no scientific calls."""

import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle

import numpy as np

from src.rl.capacity_family_design import METHODS
from src.rl.capacity_pilot_runner import COST_KEYS


def summaries(root, phase):
    return [r for p in sorted((Path(root)/"summaries").glob("*.json"))
            if (r := json.loads(p.read_text()))["world"]["phase"] == phase]


def select_finalists(rows, study):
    keyed = {(r["world"]["block"], r["world"]["condition"], r["world"]["replicate"], r["role"]): r for r in rows}
    if len(rows) != 864 or len(keyed) != len(rows):
        raise ValueError("complete 3-block development comparison required")
    ranked = []
    for mi, method in enumerate(METHODS):
        trials = []
        for li, lr in enumerate(study["learning_rates"]):
            role = f"{method}-lr{li}-graph"
            pairs = [(keyed[b, c, j, role], keyed[b, c, j, "plain_h8"])
                     for b in range(3) for c in range(3) for j in range(8)]
            cost = float(np.mean([a["cost"] for a, _ in pairs]))
            loss = float(np.mean([a["lost"]-h["lost"] for a, h in pairs]))
            score = (0, cost, 0., mi, li) if loss <= 0 else (1, loss, cost, mi, li)
            trials.append(dict(method=method, lr_index=li, lr=lr, role=role,
                               cost=cost, extra_losses=loss, rank_key=score))
        ranked.append(min(trials, key=lambda r: r["rank_key"]))
    ranked.sort(key=lambda r: r["rank_key"])
    return dict(finalists=ranked[:2], family_ranking=ranked,
                basis="development_only_not_a_winner", test_results_consulted=False)


def reconcile_raw(root, rows):
    records = []
    for summary in rows:
        path = Path(root)/summary["raw_path"]
        costs, lost, requested, committed, applied = [], 0, 0., 0., 0.
        components = dict.fromkeys(COST_KEYS, 0.)
        with gzip.open(path, "rt") as stream:
            for epoch, line in enumerate(stream):
                row = json.loads(line)
                if row["epoch"] != epoch or row["world"] != summary["world"] or row["role"] != summary["role"]:
                    raise ValueError("raw trajectory identity/epoch mismatch")
                costs.append(row["cost"])
                if not math.isclose(sum(row["components"].values()), row["cost"], rel_tol=1e-12, abs_tol=1e-6):
                    raise ValueError("raw component sum mismatch")
                for key in COST_KEYS:
                    components[key] += row["components"][key]
                lost += row["new_lost_patients"]
                requested += sum(row["requested_hours"])
                committed += sum(row["executed_hours"])
                applied += sum(row["info"]["support_public_receipt"]["applied_hours"])
        if (len(costs) != 64 or lost != summary["lost"] or not summary["settled"]
                or not math.isclose(math.fsum(costs), summary["cost"], rel_tol=1e-12, abs_tol=1e-6)):
            raise ValueError("independent raw/summary mismatch")
        records.append(dict(**summary, components=components, requested_hours=requested,
            committed_hours=committed, applied_hours=applied, raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return records


def pair_readout(rows, left, right, blocks, rng, resamples=2000):
    keyed = {(r["world"]["block"], r["world"]["condition"], r["world"]["replicate"], r["role"]): r for r in rows}
    a = [keyed[b, c, j, left] for b in range(blocks) for c in range(3) for j in range(8)]
    h = [keyed[b, c, j, right] for b in range(blocks) for c in range(3) for j in range(8)]
    if any(x["tape_sha256"] != y["tape_sha256"] for x, y in zip(a, h)):
        raise ValueError("within-condition pairing mismatch")
    saving = np.asarray([y["cost"]-x["cost"] for x, y in zip(a, h)]).reshape(blocks, 3, 8)
    losses = np.asarray([x["lost"]-y["lost"] for x, y in zip(a, h)]).reshape(blocks, 3, 8)
    ref = np.asarray([y["cost"] for y in h]).reshape(blocks, 3, 8)
    # Resample the same training blocks across conditions, independent worlds
    # inside each condition. A replicate label is not cross-condition pairing.
    b = rng.integers(0, blocks, (resamples, blocks, 1, 1))
    j = rng.integers(0, 8, (resamples, blocks, 3, 8))
    c = np.arange(3)[None, None, :, None]
    bs, bl = saving[b, c, j].mean((1, 3)), losses[b, c, j].mean((1, 3))
    conditions = []
    for k in range(3):
        conditions.append(dict(condition=k, cost_saving=float(saving[:, k].mean()),
            saving_fraction=float(saving[:, k].mean()/ref[:, k].mean()),
            cost_ci95=np.quantile(bs[:, k], [.025, .975]).tolist(),
            extra_patient_losses=float(losses[:, k].mean()),
            patient_ci95=np.quantile(bl[:, k], [.025, .975]).tolist(),
            block_savings=saving[:, k].mean(1).tolist(),
            cost_harmed_worlds=int((saving[:, k] < 0).sum()),
            patient_harmed_worlds=int((losses[:, k] > 0).sum())))
    metrics = {}
    for key in ("cost", "lost", "elapsed", "requested_hours", "committed_hours", "applied_hours"):
        if key in a[0]:
            metrics[key] = dict(candidate_mean=float(np.mean([x[key] for x in a])),
                                reference_mean=float(np.mean([x[key] for x in h])))
    return dict(candidate=left, reference=right, conditions=conditions, metrics=metrics,
        cost_saving=float(saving.mean()), saving_fraction=float(saving.mean()/ref.mean()),
        cost_ci95=np.quantile(bs.mean(1), [.025, .975]).tolist(),
        extra_patient_losses=float(losses.mean()), patient_ci95=np.quantile(bl.mean(1), [.025, .975]).tolist(),
        block_savings=saving.mean((1, 2)).tolist(),
        world_savings=saving.tolist(), world_extra_losses=losses.tolist(),
        fast_minus_stable_saving=float((saving[:, 2].mean(1)-saving[:, 0].mean(1)).mean()),
        fast_minus_persistent_saving=float((saving[:, 2].mean(1)-saving[:, 1].mean(1)).mean()))


def analyze(root, study, selection):
    all_rows = [json.loads(p.read_text()) for p in sorted((Path(root)/"summaries").glob("*.json"))]
    if len(all_rows) != 6744:
        raise ValueError("complete fixed package required for final ranking")
    raw = reconcile_raw(root, all_rows)
    final = [r for r in raw if r["world"]["phase"] == "confirmation_evaluation"]
    roles = sorted({r["role"] for r in final})
    if len(final) != 1080 or len(roles) != 9:
        raise ValueError("all final controls must be retained")
    def contrast(data, a, b, blocks):
        seed = int.from_bytes(hashlib.sha256(f"{study['streams']['bootstrap_seed']}:{blocks}:{a}:{b}".encode()).digest()[:8], "big")
        return pair_readout(data, a, b, blocks, np.random.default_rng(seed), study["readout"]["bootstrap_resamples"])
    pairs = [contrast(final, a, b, 5)
             for a, b in itertools.combinations(roles, 2)]
    development = [r for r in raw if r["world"]["phase"] == "development_evaluation"]
    dev_roles = sorted({r["role"] for r in development})
    dev_pairs = [contrast(development, a, b, 3) for a, b in itertools.combinations(dev_roles, 2)]
    screens = []
    for entry in selection["finalists"]:
        role = entry["method"]+"-graph-final"
        result = contrast(final, role, "plain_h8", 5)
        passed = (result["saving_fraction"] >= study["readout"]["minimum_cost_saving_fraction"]
                  and min(result["block_savings"]) > 0 and result["cost_ci95"][0] > 0
                  and result["extra_patient_losses"] <= 0)
        screens.append(dict(role=role, passed=bool(passed), result=result))
    eligible = [s for s in screens if s["passed"]]
    winner, reason = None, "no_candidate_met_prespecified_cost_service_screen"
    if len(eligible) == 1:
        winner, reason = eligible[0]["role"], "only_candidate_meeting_screen_not_universal_superiority"
    elif len(eligible) == 2:
        candidate = min(eligible, key=lambda s: s["result"]["metrics"]["cost"]["candidate_mean"])
        other = next(s for s in eligible if s is not candidate)
        head = contrast(final, candidate["role"], other["role"], 5)
        if head["cost_ci95"][0] > 0 and head["extra_patient_losses"] <= 0:
            winner, reason = candidate["role"], "screen_plus_descriptive_head_to_head"
        else:
            reason = "no_unique_winner_keep_pareto_alternatives"
    return dict(selection=selection, winner=winner, reason=reason, screens=screens, pairs=pairs,
        development_pairs=dev_pairs, fit_evidence=fit_evidence(root),
        raw_reconciliation=raw, native_trajectories=len(raw), final_test_updates=0,
        synthetic_only=True, formal_multiplicity_adjusted=False,
        globally_optimal=False, additional_experiment_authorized=False)


def fit_evidence(root):
    """Verify real completed optimizer receipts and saved after-fit Adam steps."""
    root = Path(root)
    files = [p for p in sorted((root/"updates").glob("*.json")) if not p.name.endswith("-targets.json")]
    if len(files) != 4800:
        raise ValueError("missing fit receipt files")
    dispatches = 0
    for path in files:
        record = json.loads(path.read_text())
        if len(record["receipts"]) != 32:
            raise ValueError("incomplete fit receipt batch")
        with gzip.open(root/"states"/(path.stem+"-after-fit.pkl.gz"), "rb") as stream:
            state = pickle.load(stream)["learner"]
        if "optimizers" in state:
            for name, optimizer in state["optimizers"].items():
                steps = state["optimizer_module_steps"][name]
                if not optimizer["state"] or any(float(v["step"]) != steps for v in optimizer["state"].values()):
                    raise ValueError("saved Adam steps differ from completed receipts")
            if not state["modules"] or state["counts"] != record["counts"]:
                raise ValueError("after-fit model/counters absent or inconsistent")
            for receipt in record["receipts"]:
                for value in receipt["losses"].values():
                    if (value["optimizer_completed"] is not True or not math.isfinite(value["loss"])
                            or not math.isfinite(value["grad_norm"])):
                        raise ValueError("invalid actual optimizer receipt")
                    dispatches += 1
        else:
            if (not state["model"] or not state["optimizer"]["state"]
                    or any(float(v["step"]) != state["updates"] for v in state["optimizer"]["state"].values())
                    or state["counts"] != record["counts"]):
                raise ValueError("missing value model/Adam evidence")
            for receipt in record["receipts"]:
                if not math.isfinite(receipt["loss"]) or not math.isfinite(receipt["gradient_norm"]):
                    raise ValueError("nonfinite value update receipt")
                dispatches += 1
    return dict(after_fit_states=len(files), minibatch_iterations=4800*32,
                optimizer_dispatches=dispatches, proves_learning_execution_not_benefit=True)
