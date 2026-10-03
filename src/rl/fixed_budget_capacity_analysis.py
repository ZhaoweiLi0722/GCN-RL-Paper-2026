"""Independent raw cost/patient readout with the new fixed-total action contract."""

from collections import Counter
import gzip
import math
from pathlib import Path

import numpy as np

from src.rl.capacity_pilot_analysis import COST_COMPONENTS, _json
from src.rl.patient_constrained_analysis import read_trajectory, summarize_pairs


def expected_worlds(study):
    for phase, names in (("training", ["fixed_allocation_reference", "constrained"]),
                         ("evaluation", study["design"]["evaluation_arms"])):
        base = study["streams"][phase+"_base"]
        for b in range(3):
            for c in range(3):
                for j in range(4):
                    w = dict(phase=phase, block=b, condition=c, replicate=j, seed=base+1000*b+100*c+j)
                    for role in names:
                        yield w, role, f"{phase}-b{b}-c{c}-j{j}-{role}"


def allocation_receipts(root, role, ident):
    committed, applied, residuals, projection = [], [], [], []
    bounded = role in ("constrained", "constrained_greedy_frozen", "fixed_allocation_reference")
    with gzip.open(root/"raw"/f"{ident}.jsonl.gz", "rt") as handle:
        for line in handle:
            row = _json(line)
            request, execute = row["requested_hours"], row["executed_hours"]
            actual = row["info"]["support_public_receipt"]["applied_hours"]
            if (len(request) != 4 or len(actual) != 4
                    or any(not math.isfinite(h) or h < 0 or h > 4 for h in (*request, *actual))):
                raise ValueError("invalid raw requested/applied hours")
            applied.append(math.fsum(actual))
            if row["epoch"] < 48:
                committed.append(math.fsum(execute))
                residuals.append(abs(math.fsum(request)-8.))
                projection.append(max(abs(a-b) for a,b in zip(request, execute)))
                if bounded and (any(h < .5 or h > 3.5 for h in request)
                        or sum(request) > 8 or math.fsum(request) > 8
                        or residuals[-1] > 1e-12 or projection[-1] > 1e-12):
                    raise ValueError("fixed-resource action contract violated")
            elif any(request):
                raise ValueError("nonzero requested settlement action")
    return dict(mean_committed_control_hours=float(np.mean(committed)),
        total_committed_control_hours=math.fsum(committed),
        total_applied_hours_including_tail=math.fsum(applied),
        max_control_request_total_residual=max(residuals),
        max_native_projection_change=max(projection),
        allocation_contract_checked=bounded)


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
        row = read_trajectory(root,world,role,ident)
        row.update(allocation_receipts(root, role, ident))
        trajectories[(world["phase"],world["block"],world["condition"],world["replicate"],role)] = row
    for phase in ("training", "evaluation"):
        roles = ["fixed_allocation_reference","constrained"] if phase=="training" else study["design"]["evaluation_arms"]
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
    if updates != Counter(warmup=768, training=1728, multiplier=36):
        raise ValueError("missing update receipts or unexpected evaluation training")
    rng = np.random.default_rng(study["streams"]["bootstrap_seed"])
    contrasts = {}
    for reference in ("fixed_allocation_reference", "id_mpc"):
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
    return dict(format="fixed-budget-capacity-raw-readout-v1", complete=True,
        trajectory_count=len(trajectories), raw_rows=len(trajectories)*64, evaluation_trajectories=108,
        update_receipts=dict(updates), multiplier_receipts=multipliers,
        trajectories=list(trajectories.values()), contrasts=contrasts, criteria=criteria,
        patient_preserving_training_signal=all(criteria.values()),
        claim_scope="simulation_RL_training_not_deployment_online_adaptation_not_graph_attribution",
        clinical_safety_established=False, independent_confirmation=False,
        automatic_follow_on=False)
