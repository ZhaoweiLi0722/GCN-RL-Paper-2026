"""Development-only saved-data diagnostic; stdlib, no scientific execution."""

import json
import math
from pathlib import Path
from statistics import fmean


BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = BASE / "results/capacity_family_selection_20261006"
READOUT = RUN / "terminal-readout"
ROLES = ("value_td-lr0-graph", "value_td-lr1-graph")
CONDITIONS = ("no_change", "persistent_change", "fast_fluctuation")


def read(path):
    return json.loads(path.read_text())


def mean(rows, key):
    return fmean(row[key] for row in rows)


def correlation(x, y):
    x = [v - fmean(x) for v in x]
    y = [v - fmean(y) for v in y]
    denominator = math.sqrt(sum(v * v for v in x) * sum(v * v for v in y))
    return sum(a * b for a, b in zip(x, y)) / denominator if denominator else None


def describe(pairs, worlds, role, condition=None):
    oriented = next(p for p in pairs if {p["candidate"], p["reference"]} == {role, "plain_h8"})
    sign = 1 if oriented["candidate"] == role else -1
    rows = [p for p in oriented["worlds"] if condition is None or p["condition"] == condition]
    candidate = [w for w in worlds if w["role"] == role and (condition is None or w["world"]["condition"] == condition)]
    reference = [w for w in worlds if w["role"] == "plain_h8" and (condition is None or w["world"]["condition"] == condition)]
    saved = oriented["saved_descriptive_readout"]
    if condition is not None:
        saved = next(c for c in saved["conditions"] if c["condition"] == condition)
    savings = [sign * p["cost_saving"] for p in rows]
    losses = [sign * p["extra_patient_losses"] for p in rows]
    hours = [sign * p["metric_delta"]["applied_hours"] for p in rows]
    components = {k: sign * fmean(p["component_delta"][k] for p in rows) for k in rows[0]["component_delta"]}
    # Reverse signed intervals, but recompute percentages with H8 as denominator.
    ci = sorted(sign * v for v in saved["cost_ci95"])
    assert math.isclose(fmean(savings), sign * saved["cost_saving"], abs_tol=1e-6)
    assert math.isclose(sum(components.values()), -fmean(savings), abs_tol=1e-6)
    centered_hours, centered_savings = [], []
    for p, h, s in zip(rows, hours, savings):
        cell = [i for i, q in enumerate(rows) if (q["block"], q["condition"]) == (p["block"], p["condition"])]
        centered_hours.append(h - fmean(hours[i] for i in cell))
        centered_savings.append(s - fmean(savings[i] for i in cell))
    return {
        "condition": "all_equal_weight" if condition is None else CONDITIONS[condition],
        "worlds": len(rows), "blocks": 3,
        "saving": fmean(savings), "saving_percent": 100 * fmean(savings) / mean(reference, "cost"),
        "saved_descriptive_cost_ci95": ci,
        "extra_losses": fmean(losses),
        "saved_descriptive_extra_loss_ci95": sorted(sign * v for v in saved["patient_ci95"]),
        "block_savings": [sign * v for v in saved["block_savings"]],
        "cost_harmed_worlds": sum(s < 0 for s in savings),
        "patient_harmed_worlds": sum(x > 0 for x in losses),
        "candidate_cost": mean(candidate, "cost"), "h8_cost": mean(reference, "cost"),
        "candidate_applied_hours": mean(candidate, "applied_hours"),
        "h8_applied_hours": mean(reference, "applied_hours"),
        "applied_hour_delta": fmean(hours),
        "candidate_fraction_of_max_bookable_hours": mean(candidate, "applied_hours") / (48 * 8),
        "requested_hour_delta": sign * fmean(p["metric_delta"]["requested_hours"] for p in rows),
        "committed_hour_delta": sign * fmean(p["metric_delta"]["committed_hours"] for p in rows),
        "changed_control_boundaries_mean": mean(rows, "changed_control_boundaries"),
        "action_l1_hours_mean": mean(rows, "action_l1_hours"),
        "component_delta_candidate_minus_h8": components,
        "decomposition_residual": sum(components.values()) + fmean(savings),
        "support_delta_vs_saving_pearson": correlation(hours, savings),
        "within_block_condition_centered_pearson": correlation(centered_hours, centered_savings),
        "association_is_causal": False,
    }


def fit_summary(training, jobs, config):
    output, receipts_used = [], []
    for role in ROLES:
        for block in range(config["development_blocks"]):
            selected = [w for w in training if w["role"] == role and w["world"]["block"] == block]
            fits = []
            for world in selected:
                name = Path(world["raw_path"]).name.removesuffix(".jsonl.gz") + ".json"
                path = RUN / "payload/updates" / name
                receipt = read(path)
                records = receipt["receipts"]
                assert len(records) == config["updates_per_world"]
                fits.append((records[0]["update"], world, records))
                receipts_used.append(str(path.relative_to(BASE)))
            fits.sort(key=lambda item: item[0])
            records = [record for _, _, cohort in fits for record in cohort]
            assert len(fits) == config["train_worlds"]
            assert [r["update"] for r in records] == list(range(1, len(records) + 1))
            assert all(math.isfinite(r[k]) for r in records for k in ("loss", "gradient_norm"))
            job = next(j for j in jobs if j["role"] == role and j["block"] == block)
            assert job["fit_optimizer_dispatches"]["value"] == len(records)
            # Fixed descriptive windows, not checkpoint selection or validation loss.
            first, last = fits[:24], fits[-24:]
            window_loss = lambda window: fmean(r["loss"] for _, _, cohort in window for r in cohort)
            condition_losses = []
            for condition in range(3):
                subset = [item for item in fits if item[1]["world"]["condition"] == condition]
                condition_losses.append({"condition": CONDITIONS[condition], "worlds": len(subset),
                                         "first_8_world_mean_loss": window_loss(subset[:8]),
                                         "last_8_world_mean_loss": window_loss(subset[-8:])})
            output.append({
                "role": role, "block": block, "training_worlds": len(fits),
                "fit_updates": len(records), "finite_receipts": True,
                "first_24_world_mean_loss": window_loss(first),
                "last_24_world_mean_loss": window_loss(last),
                "within_world_first_8_updates_mean_loss": fmean(r["loss"] for _, _, cohort in fits for r in cohort[:8]),
                "within_world_last_8_updates_mean_loss": fmean(r["loss"] for _, _, cohort in fits for r in cohort[-8:]),
                "gradient_norm_above_cap_fraction": fmean(r["gradient_norm"] > config["value"]["gradient_norm_cap"] for r in records),
                "maximum_recorded_gradient_norm": max(r["gradient_norm"] for r in records),
                "condition_fit_receipts": condition_losses,
            })
    return output, sorted(receipts_used)


def main():
    config = read(BASE / "experiments/configs/capacity_family_selection_20261006.json")
    verification = read(READOUT / "verification.json")
    assert verification["matched_raw_summary_comparison"] and verification["all_pair_world_means_reconciled"]
    selection = read(RUN / "payload/selection.json")
    assert selection["basis"] == "development_only_not_a_winner" and not selection["test_results_consulted"]
    # Mixed-stage containers are filtered immediately. Final/test numbers never enter outputs.
    development = [w for w in read(READOUT / "world-readout.json") if w["world"]["phase"] in ("development_training", "development_evaluation")]
    evaluation = [w for w in development if w["world"]["phase"] == "development_evaluation"]
    training = [w for w in development if w["world"]["phase"] == "development_training" and w["role"] in ROLES]
    pairs = read(READOUT / "all-pairs-readout.json")["development_evaluation"]
    jobs = [j for j in read(READOUT / "job-and-role-readout.json")["jobs"] if j["phase"] == "development_training" and j["role"] in ROLES]
    assert len(evaluation) == 864 and len(training) == 576 and len(jobs) == 6
    diagnostics = {role: {"overall": describe(pairs, evaluation, role),
                          "conditions": [describe(pairs, evaluation, role, c) for c in range(3)]} for role in ROLES}
    fits, receipts = fit_summary(training, jobs, config)
    result = {
        "schema": "value-td-development-diagnostic-v1",
        "question": "Which development weaknesses justify one prospective comparison?",
        "scope": {
            "outcome_phases": ["development_evaluation"],
            "fit_phase": "development_training",
            "excluded": ["confirmation_training", "confirmation_evaluation", "MDL2 supplement", "old raw trajectories", "archives", "model states"],
            "same_worktree_only": True, "scientific_calls": 0,
            "new_resampling": False, "new_experiment_authorized": False,
            "selection_exposed_data_not_validation": True,
        },
        "definitions": {
            "saving": "H8 minus Value-TD; positive is cheaper; synthetic cost units",
            "extra_losses": "Value-TD minus H8; negative is fewer losses per complete 64-step world",
            "component_delta": "Value-TD minus H8; mutually exclusive components, no duplicate base_cost",
            "saving_percent": "100 * mean paired saving / mean H8 cost; not mean world ratios",
            "weights": "24 worlds/condition, 3 blocks, 8 worlds/block/condition; equal condition weights",
            "uncertainty": "Reuse signed existing paired descriptive bootstrap intervals; no new inference or multiplicity correction",
            "hours": "Whole-world totals including settlement; /384 is bookable-volume utilization, NOT control-step saturation frequency",
            "fit_loss": "Recorded normalized TD MSE on sampled own-world targets; changing data/targets, not held-out fit or calibration",
            "gradient_norm": "Logged clip_grad_norm_ return, before clipping; configured cap=5; no clipping-effect attribution",
        },
        "development_selection": selection,
        "value_td_vs_h8": diagnostics,
        "development_fit_receipts": fits,
        "recommendation": {
            "status": "one hypothesis only; not implemented, approved, run, or validated",
            "hypothesis": "A single prospectively fixed attenuation of the learned TD residual toward the H8 zero-residual anchor may improve the cost/service tradeoff compared with unattenuated Value-TD.",
            "comparison": "Attenuated vs unattenuated Value-TD using matched sealed models; retain unchanged H8 as the strong reference.",
            "freeze_before_execution": "One nonzero attenuation below full weight, model eligibility, metrics, decision rule and finite budget; no coefficient grid or per-condition oracle switch.",
            "held_fixed": "Reward weights, physical dynamics, public information, condition mix, planning horizon and training recipe; do not infer a training or graph benefit from this controller ablation.",
            "required_future_evaluation": "New untouched paired worlds, blocked by model/training seed; old final/test and MDL2 worlds remain consumed. Original H8 screen remains unchanged. Predeclare a patient-loss guardrail against the unattenuated policy and H8, and report all conditions, blocks, harms, cost components, hours and compute.",
            "risk": "Attenuation may remove the patient gains without lowering cost; support utilization and cost components are associations, not evidence of excessive or causal support spending.",
            "budget": None,
        },
        "sources": {
            "readouts": [str((READOUT / name).relative_to(BASE)) for name in ("world-readout.json", "all-pairs-readout.json", "job-and-role-readout.json", "verification.json")],
            "selection": str((RUN / "payload/selection.json").relative_to(BASE)),
            "config": "experiments/configs/capacity_family_selection_20261006.json",
            "fit_receipts": receipts,
            "semantics_read_only": ["src/rl/capacity_value_learner.py:56-91", "src/rl/capacity_family_value.py:11-29"],
            "authority_context_only": ["AGENTS.md", "docs/patient_indexed_specimen_routing_locked_execution_plan.md", "specs/2026-10-01-adaptive-paper-delivery/workflow.json", "specs/2026-10-06-family-selection/protocol.md", "specs/2026-10-06-family-selection/terminal-readout.md"],
            "verification_reused_at": verification["at_utc"],
            "note": "Receipt paths are read once; no raw/archive/state re-audit. Mixed containers include other stages but their outcomes are excluded. Authority narratives disclose final/posthoc status, not inputs for choosing the hypothesis.",
        },
    }
    output = OUT / "diagnostic.json"
    if output.exists():
        assert read(output) == result, "Existing diagnostic differs; refusing to overwrite"
    else:
        with output.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"output": str(OUT / "diagnostic.json"), "training_receipt_files": len(receipts),
                      "value_fit_updates": sum(x["fit_updates"] for x in fits),
                      "selected_development": diagnostics[ROLES[0]]["overall"]}, indent=2))


if __name__ == "__main__":
    main()
