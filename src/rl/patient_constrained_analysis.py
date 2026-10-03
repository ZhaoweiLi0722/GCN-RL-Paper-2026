"""Independent saved-row arithmetic; no simulator, neural model or runner imports."""

from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.rl.capacity_pilot_analysis import COST_COMPONENTS, _json, _digest, _close


def expected_worlds(study):
    roles = study["design"]["evaluation_arms"]
    for phase, base, names in (("training", 62700000, ["fixed_allocation_reference", "constrained", "cost_only"]),
                               ("evaluation", 62720000, roles)):
        for b in range(3):
            for c in range(3):
                for j in range(4):
                    w = dict(phase=phase, block=b, condition=c, replicate=j, seed=base+1000*b+100*c+j)
                    for role in names:
                        yield w, role, f"{phase}-b{b}-c{c}-j{j}-{role}"


def read_trajectory(root, world, role, ident):
    path, summary_path = root/"raw"/f"{ident}.jsonl.gz", root/"summaries"/f"{ident}.json"
    summary = _json(summary_path.read_text())
    if (summary["world"] != world or summary["role"] != role or summary["raw_path"] != f"raw/{ident}.jsonl.gz"
            or summary.get("optimizer_updates_during_trajectory") != 0):
        raise ValueError("summary scope or rollout freezing mismatch: "+ident)
    costs, components, hours, patient_times = [], {k: [] for k in COST_COMPONENTS}, [], {}
    previous, previous_lost = {}, set()
    with gzip.open(path, "rt") as handle:
        for t, line in enumerate(handle):
            row = _json(line)
            if row["world"] != world or row["role"] != role or row["epoch"] != t:
                raise ValueError("nonsequential or unpaired raw identity")
            comp, cost = row["components"], row["cost"]
            if (set(comp) != set(COST_COMPONENTS) or not all(math.isfinite(v) for v in comp.values())
                    or not math.isfinite(cost) or cost < 0 or not _close(math.fsum(comp.values()), cost)
                    or not _close(row["reward"], -cost)):
                raise ValueError("raw component/cost/reward mismatch")
            costs.append(cost)
            for k,v in comp.items(): components[k].append(v)
            current = {x["patient_id"]: x for x in row["patient_records"]}
            if len(current) != len(row["patient_records"]) or not set(previous).issubset(current):
                raise ValueError("duplicate or missing patient identity")
            for pid, patient in current.items():
                if (patient["status"] not in {"waiting", "in_transit", "in_production", "finished", "delivered", "lost"}
                        or type(patient["enrollment_epoch"]) is not int or not 0 <= patient["enrollment_epoch"] < min(t+1, 48)):
                    raise ValueError("invalid patient state or enrollment")
                if pid in previous:
                    old = previous[pid]
                    if (old["enrollment_epoch"] != patient["enrollment_epoch"] or old["collection_site"] != patient["collection_site"]
                            or old["status"] in {"lost", "delivered"} and old["status"] != patient["status"]):
                        raise ValueError("patient identity changed or terminal state reopened")
                if patient["status"] in {"lost", "delivered"}:
                    patient_times.setdefault(pid, t-patient["enrollment_epoch"])
            lost = {pid for pid,p in current.items() if p["status"] == "lost"}
            if not previous_lost.issubset(lost) or row["new_lost_patients"] != len(lost-previous_lost):
                raise ValueError("loss target differs from independently reconstructed patient events")
            counts = dict(enrolled=len(current), lost=len(lost), delivered=sum(p["status"] == "delivered" for p in current.values()))
            if row["cumulative"] != counts:
                raise ValueError("raw patient counts disagree")
            action = row["executed_hours"]
            if (len(action) != 4 or any(not math.isfinite(h) or not 0 <= h <= 4 for h in action)
                    or sum(action) > 8+1e-6 or t >= 48 and any(action)):
                raise ValueError("illegal hours or nonzero settlement action")
            if t < 48: hours.append(action)
            previous, previous_lost = current, lost
    settled = summary["settlement"]
    if (len(costs) != 64 or not summary["settled"] or not settled["settled"]
            or not _close(math.fsum(costs), summary["cost"]) or summary["lost"] != len(previous_lost)
            or any(settled[k] != counts[k] for k in counts)
            or counts["enrolled"] != counts["lost"]+counts["delivered"]
            or settled["live_ids"] or settled["pending_obligations"] != 0 or not settled["resource_conservation"]):
        raise ValueError("raw full-horizon or settlement reconciliation failed: "+ident)
    cohort = sorted((pid,p["enrollment_epoch"],p["collection_site"]) for pid,p in previous.items())
    return dict(world=world, role=role, cost=math.fsum(costs), **counts,
        component_costs={k:math.fsum(v) for k,v in components.items()}, actions=hours,
        cohort_sha256=hashlib.sha256(json.dumps(cohort,separators=(",", ":")).encode()).hexdigest(),
        tape_sha256=summary["tape_sha256"], raw_sha256=_digest(path), summary_sha256=_digest(summary_path),
        model_seal_sha256=summary["model_seal_sha256"], mean_time_to_outcome=float(np.mean(list(patient_times.values()))) if patient_times else None)


def summarize_pairs(pairs, rng):
    def mean(rows, key): return float(np.mean([r[key] for r in rows]))
    by_block = [[r for r in pairs if r["block"]==b] for b in range(3)]
    intervals = {k: [] for k in ("savings", "extra_lost", "percent_savings")}
    for _ in range(2000):
        sample = []
        for b in rng.integers(0,3,3):
            group = by_block[int(b)]
            sample.extend(group[int(j)] for j in rng.integers(0,len(group),len(group)))
        for key in intervals: intervals[key].append(mean(sample,key))
    return dict(n_worlds=len(pairs), independent_blocks=3,
        means={k:mean(pairs,k) for k in ("savings", "extra_lost", "extra_delivered", "percent_savings")},
        blocks=[dict(block=b, **{k:mean(rows,k) for k in ("savings", "extra_lost", "percent_savings")}) for b,rows in enumerate(by_block)],
        descriptive95={k:np.percentile(values,[2.5,97.5]).tolist() for k,values in intervals.items()}, pairs=pairs)


def run_analysis(root, study):
    root = Path(root)
    expected = list(expected_worlds(study))
    names = {f"{ident}.json" for _,_,ident in expected}
    if {p.name for p in (root/"summaries").glob("*.json")} != names:
        raise ValueError("missing or extra completed trajectories")
    if {p.name for p in (root/"raw").glob("*.jsonl.gz")} != {n[:-5]+".jsonl.gz" for n in names}:
        raise ValueError("missing or extra raw trajectories")
    trajectories = {}
    for world,role,ident in expected:
        trajectories[(world["phase"],world["block"],world["condition"],world["replicate"],role)] = read_trajectory(root,world,role,ident)
    for phase in ("training", "evaluation"):
        roles = ["fixed_allocation_reference","constrained","cost_only"] if phase=="training" else study["design"]["evaluation_arms"]
        for b in range(3):
            for c in range(3):
                for j in range(4):
                    rows = [trajectories[(phase,b,c,j,r)] for r in roles]
                    if any(len({r[k] for r in rows}) != 1 for k in ("tape_sha256", "cohort_sha256", "enrolled")):
                        raise ValueError("paired exogenous world/patient cohort mismatch")
    updates, multipliers = Counter(), []
    for path in sorted((root/"updates").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = _json(line)
            updates[row["mode"]] += 1
            if row["mode"] == "multiplier": multipliers.append(row)
    if updates != Counter(warmup=768, training=3456, multiplier=36):
        raise ValueError("missing update receipts or unexpected evaluation training")
    rng = np.random.default_rng(study["readout"]["bootstrap"]["seed"])
    contrasts = {}
    for reference in ("fixed_allocation_reference", "cost_only_greedy_frozen", "id_mpc"):
        conditions = {}
        for c in range(3):
            pairs = []
            for b in range(3):
                for j in range(4):
                    learned = trajectories[("evaluation",b,c,j,"constrained_greedy_frozen")]
                    ref = trajectories[("evaluation",b,c,j,reference)]
                    delta = np.asarray(learned["actions"])-np.asarray(ref["actions"])
                    pairs.append(dict(block=b, replicate=j, seed=learned["world"]["seed"],
                        savings=ref["cost"]-learned["cost"], percent_savings=100*(ref["cost"]-learned["cost"])/ref["cost"],
                        extra_lost=learned["lost"]-ref["lost"], extra_delivered=learned["delivered"]-ref["delivered"],
                        changed_action_boundaries=int((np.abs(delta).sum(axis=1)>1e-9).sum()),
                        action_l1_hours=float(np.abs(delta).sum()),
                        component_savings={k:ref["component_costs"][k]-learned["component_costs"][k] for k in COST_COMPONENTS}))
            conditions[str(c)] = summarize_pairs(pairs,rng)
        contrasts[reference] = conditions
    primary = contrasts["fixed_allocation_reference"]
    criteria = dict(all_persistent_blocks_cost_positive=all(x["savings"]>0 for x in primary["1"]["blocks"]),
        persistent_descriptive_cost_interval_above_zero=primary["1"]["descriptive95"]["savings"][0]>0,
        no_mean_extra_losses_in_any_condition=all(x["means"]["extra_lost"]<=0 for x in primary.values()))
    return dict(format="patient-constrained-raw-readout-v1", complete=True,
        trajectory_count=len(trajectories), raw_rows=len(trajectories)*64, evaluation_trajectories=144,
        update_receipts=dict(updates), multiplier_receipts=multipliers,
        trajectories=list(trajectories.values()), contrasts=contrasts, criteria=criteria,
        patient_preserving_training_signal=all(criteria.values()),
        claim_scope="simulation_RL_training_not_deployment_online_adaptation_not_graph_attribution",
        clinical_safety_established=False, independent_confirmation=False,
        automatic_follow_on=False)
