"""Terminal readout of saved files only; no project/scientific imports."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time

BASE = Path(__file__).resolve().parents[2]
RUN = BASE / "results/capacity_family_selection_20261006"
PAYLOAD = RUN / "payload"
OUT = RUN / "terminal-readout"
OUT.mkdir(exist_ok=True)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-6), (a, b)


def mean(values):
    return statistics.fmean(values)


def save(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


started = time.monotonic()
terminal = read(RUN / "launcher/terminal.json")
assert terminal["status"] == "completed" and terminal["scientific_completion_verified"]
budget = read(RUN / "launcher/child-completed.json")["budget"]
comparison = read(PAYLOAD / "comparison.json")
frozen = read(BASE / "specs/2026-10-06-family-selection/frozen.json")
manifest = read(RUN / "archives/payload.tar.gz.manifest.json")
prior = {r["raw_path"]: r for r in comparison["raw_reconciliation"]}
assert len(prior) == 6744
lock_counts = {}
for group in ("source_files", "inputs"):
    for item in frozen[group].values():
        path = BASE / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert digest(path) == item["sha256"], str(path)
    lock_counts[group] = len(frozen[group])

records, actions, jobs, eval_keys = [], {}, defaultdict(list), {}
fit_counts = Counter()
for number, path in enumerate(sorted((PAYLOAD / "summaries").glob("*.json")), 1):
    s = read(path)
    raw_path = PAYLOAD / s["raw_path"]
    raw_hash = digest(raw_path)
    assert raw_hash == manifest["files"][s["raw_path"]] == prior[s["raw_path"]]["raw_sha256"]
    assert digest(path) == manifest["files"][str(path.relative_to(PAYLOAD))]
    costs, requested, committed, applied, waiting, turnaround = [], [], [], [], [], []
    components, lost = Counter(), 0
    executed_actions = []
    with gzip.open(raw_path, "rt") as stream:
        for epoch, line in enumerate(stream):
            r = json.loads(line)
            assert (r["epoch"], r["world"], r["role"]) == (epoch, s["world"], s["role"])
            close(sum(r["components"].values()), r["cost"])
            close(-r["reward"], r["cost"])
            costs.append(r["cost"])
            components.update(r["components"])
            lost += r["new_lost_patients"]
            requested.append(sum(r["requested_hours"]))
            committed.append(sum(r["executed_hours"]))
            applied.append(sum(r["info"]["support_public_receipt"]["applied_hours"]))
            waiting.append(r["info"]["average_waiting_time"])
            turnaround.append(r["info"]["average_turnaround_time"])
            if epoch < 48:
                executed_actions.append(r["executed_hours"])
    assert len(costs) == 64 and s["settled"] and lost == s["lost"]
    assert s["optimizer_updates_during_trajectory"] == 0
    close(math.fsum(costs), s["cost"])
    old = prior[s["raw_path"]]
    for key, value in components.items():
        close(value, old["components"][key])
    for key, values in (("requested_hours", requested), ("committed_hours", committed), ("applied_hours", applied)):
        close(sum(values), old[key])
    record = dict(s, raw_sha256=raw_hash, components=dict(components),
                  requested_hours=sum(requested), committed_hours=sum(committed),
                  applied_hours=sum(applied), epoch_mean_reported_wait=mean(waiting),
                  epoch_mean_reported_turnaround=mean(turnaround))
    world = s["world"]
    training = world["phase"].endswith("training")
    job = (world["phase"], world["block"], s["role"])
    jobs[job].append(record)
    if training:
        update = PAYLOAD / "updates" / (path.stem + ".json")
        after = PAYLOAD / "states" / (path.stem + "-after-fit.pkl.gz")
        for artifact in (update, after):
            assert digest(artifact) == manifest["files"][str(artifact.relative_to(PAYLOAD))]
        receipt = read(update)
        assert len(receipt["receipts"]) == 32
        actual = Counter()
        for item in receipt["receipts"]:
            if "losses" in item:
                for module, value in item["losses"].items():
                    assert value["optimizer_completed"] is True
                    assert math.isfinite(value["loss"]) and math.isfinite(value["grad_norm"])
                    actual["critic" if module.startswith("critic") else module] += 1
            else:
                assert math.isfinite(item["loss"]) and math.isfinite(item["gradient_norm"])
                actual["value"] += 1
        record["fit_optimizer_dispatches"] = dict(actual)
        fit_counts.update(actual)
    else:
        assert s["compute"]["total_optimizer_steps"] == s["compute"]["fit_batches"] == 0
        key = (world["phase"], world["block"], world["condition"], world["replicate"], s["role"])
        assert key not in eval_keys
        eval_keys[key] = record
        actions[key] = executed_actions
    records.append(record)
    if number % 1000 == 0:
        print("saved trajectories reconciled", number, flush=True)

assert len(records) == 6744
assert dict(fit_counts) == {"actor": 95232, "critic": 153600, "value": 67584}, fit_counts
models = []
for path in sorted((PAYLOAD / "models").glob("*.json")):
    if path.name == "all-finalists-sealed.json":
        continue
    model = read(path)
    target = PAYLOAD / model["path"]
    assert digest(target) == model["sha256"] == manifest["files"][model["path"]]
    assert target.stat().st_size == model["bytes"]
    models.append(model)
assert len(models) == 100

job_rows = []
for (phase, block, role), rows in sorted(jobs.items()):
    training = phase.endswith("training")
    assert len(rows) == (96 if training else 24)
    compute = Counter()
    updates = Counter()
    for r in rows:
        compute.update(r["compute"])
        updates.update(r.get("fit_optimizer_dispatches", {}))
    job_rows.append(dict(phase=phase, block=block, role=role, worlds=len(rows),
                         fit_iterations=32 * len(rows) if training else 0,
                         fit_optimizer_dispatches=dict(updates),
                         trajectory_compute=dict(compute),
                         trajectory_elapsed_seconds=sum(r["elapsed"] for r in rows)))

metric_names = ("cost", "lost", "requested_hours", "committed_hours", "applied_hours", "elapsed",
                "epoch_mean_reported_wait", "epoch_mean_reported_turnaround")


def verify_pair(pair, phase, blocks):
    a, b = pair["candidate"], pair["reference"]
    pairs = []
    for block in range(blocks):
        for condition in range(3):
            for replicate in range(8):
                base = (phase, block, condition, replicate)
                x, y = eval_keys[base + (a,)], eval_keys[base + (b,)]
                assert x["tape_sha256"] == y["tape_sha256"]
                saving, loss = y["cost"] - x["cost"], x["lost"] - y["lost"]
                close(saving, pair["world_savings"][block][condition][replicate])
                close(loss, pair["world_extra_losses"][block][condition][replicate])
                ax, ay = actions[base + (a,)], actions[base + (b,)]
                changed = sum(any(abs(u-v) > 2e-6 for u, v in zip(s, t)) for s, t in zip(ax, ay))
                pairs.append(dict(block=block, condition=condition, replicate=replicate,
                    cost_saving=saving, extra_patient_losses=loss,
                    metric_delta={k: x[k]-y[k] for k in metric_names},
                    component_delta={k: x["components"][k]-y["components"][k] for k in x["components"]},
                    compute_delta={k: x["compute"][k]-y["compute"][k] for k in x["compute"]},
                    changed_control_boundaries=changed,
                    action_l1_hours=sum(abs(u-v) for s,t in zip(ax,ay) for u,v in zip(s,t))))
    close(mean(p["cost_saving"] for p in pairs), pair["cost_saving"])
    close(mean(p["extra_patient_losses"] for p in pairs), pair["extra_patient_losses"])
    ref_mean = mean(eval_keys[(phase,p["block"],p["condition"],p["replicate"],b)]["cost"] for p in pairs)
    close(pair["cost_saving"]/ref_mean, pair["saving_fraction"])
    for block in range(blocks):
        close(mean(p["cost_saving"] for p in pairs if p["block"] == block), pair["block_savings"][block])
    return dict(candidate=a, reference=b, saved_descriptive_readout=pair, worlds=pairs,
        cells=[dict(block=block, condition=condition, worlds=8,
            metric_delta_mean={k: mean(p["metric_delta"][k] for p in pairs if p["block"] == block and p["condition"] == condition) for k in metric_names},
            cost_harmed_worlds=sum(p["cost_saving"] < 0 for p in pairs if p["block"] == block and p["condition"] == condition),
            patient_harmed_worlds=sum(p["extra_patient_losses"] > 0 for p in pairs if p["block"] == block and p["condition"] == condition))
            for block in range(blocks) for condition in range(3)])


pair_sets = {}
for phase, field, blocks, count in (("development_evaluation", "development_pairs", 3, 66),
                                   ("confirmation_evaluation", "pairs", 5, 36)):
    roles = sorted({k[-1] for k in eval_keys if k[0] == phase})
    assert {(p["candidate"], p["reference"]) for p in comparison[field]} == set(itertools.combinations(roles, 2))
    assert len(comparison[field]) == count
    pair_sets[phase] = [verify_pair(p, phase, blocks) for p in comparison[field]]
for screen in comparison["screens"]:
    verify_pair(screen["result"], "confirmation_evaluation", 5)
    r = screen["result"]
    passed = r["saving_fraction"] >= .005 and min(r["block_savings"]) > 0 and r["cost_ci95"][0] > 0 and r["extra_patient_losses"] <= 0
    assert passed == screen["passed"] == False
assert comparison["winner"] is None
for key, value in budget["counts"].items():
    assert value <= frozen["contract"]["limits"][key]

role_metrics = []
for phase in ("development_evaluation", "confirmation_evaluation"):
    for role in sorted({k[-1] for k in eval_keys if k[0] == phase}):
        rows = [r for k,r in eval_keys.items() if k[0] == phase and k[-1] == role]
        role_metrics.append(dict(phase=phase, role=role, worlds=len(rows),
            means={k:mean(r[k] for r in rows) for k in metric_names},
            mean_compute={k:mean(r["compute"][k] for r in rows) for k in rows[0]["compute"]}))

receipt = dict(at_utc=datetime.now(timezone.utc).isoformat(), scientific_calls=0,
    terminal=terminal, raw_trajectories=6744, raw_epochs=6744*64,
    matched_raw_summary_comparison=True, frozen_locks_verified=lock_counts,
    model_seals_hashed=100, after_fit_states_hashed=4800, fit_receipt_files=4800,
    actual_optimizer_dispatches=dict(fit_counts), optimizer_total=sum(fit_counts.values()),
    prior_all_state_Adam_validation_reused=comparison["fit_evidence"],
    phase_counts=dict(Counter(r["world"]["phase"] for r in records)),
    budget_counts=budget["counts"], budget_phases=budget["phases"], owner_seconds=budget["times"],
    development_pairs=66, confirmation_pairs=36, all_pair_world_means_reconciled=True,
    interval_method="Reused frozen 2000 resamples: shared training-block draws, independent within-condition world draws, equal condition weights; descriptive, not multiplicity-adjusted.",
    latency_definition="Whole-trajectory elapsed seconds, including recorded rollout I/O; per-decision latency not separately recorded.",
    delay_definition="Epoch mean of simulator-reported average waiting/turnaround; not a pooled patient-level duration or a clinical estimate. Full epoch values remain in raw archive.",
    action_difference_tolerance_hours=2e-6, winner=None, reason=comparison["reason"],
    archive_sha256=manifest["archive_sha256"], archive_bytes=manifest["size_bytes"],
    archive_member_verification_reused=manifest["verified_by_reading_every_archive_member"],
    elapsed_saved_data_readout_seconds=time.monotonic()-started)
save("world-readout.json", records)
save("all-pairs-readout.json", pair_sets)
save("job-and-role-readout.json", dict(jobs=job_rows, role_metrics=role_metrics, model_seals=models))
save("verification.json", receipt)
print(json.dumps(receipt, indent=2), flush=True)
