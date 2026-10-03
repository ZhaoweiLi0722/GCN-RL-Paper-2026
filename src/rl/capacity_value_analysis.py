"""Read-only full-cost and patient reconstruction; no neural or environment imports."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from src.rl.capacity_pilot_analysis import COST_COMPONENTS
from src.rl.patient_constrained_analysis import read_trajectory, summarize_pairs
from src.rl.fixed_budget_capacity_analysis import allocation_receipts


def run_analysis(root, study):
    root = Path(root)
    expected, rows = [], {}
    for phase, roles, n in (("initial", ["plain_mpc"], 24),
                            ("continuation", ["continuation"], 24),
                            ("evaluation", study["design"]["roles"], 12)):
        for block in range(3):
            for index in range(n):
                c, j = index % 3, index // 3
                world = dict(phase=phase, block=block, condition=c, replicate=j,
                    seed=study["streams"][phase+"_base"]+1000*block+100*c+j)
                for role in roles:
                    name = f"{phase}-b{block}-c{c}-j{j}-{role}"
                    expected.append(name)
                    row = read_trajectory(root, world, role, name)
                    row.update(allocation_receipts(root, role, name))
                    rows[(phase,block,c,j,role)] = row
    if ({p.stem for p in (root/"summaries").glob("*.json")} != set(expected)
            or {p.name.removesuffix(".jsonl.gz") for p in (root/"raw").glob("*.jsonl.gz")} != set(expected)):
        raise ValueError("incomplete or extra value-MPC trajectories")
    for b in range(3):
        initial = json.loads((root/"models"/f"block{b}-initial.json").read_text())
        final = json.loads((root/"models"/f"block{b}-final.json").read_text())
        if (initial["updates"], final["updates"]) != (768, 1536):
            raise ValueError("wrong initial/final update boundary")
        for kind, metadata in (("initial", initial), ("final", final)):
            raw = (root/"models"/f"block{b}-{kind}.pt").read_bytes()
            if hashlib.sha256(raw).hexdigest() != metadata["sha256"] or len(raw) != metadata["bytes"]:
                raise ValueError("sealed value model bytes changed")
        for c in range(3):
            for j in range(4):
                group = [rows[("evaluation",b,c,j,r)] for r in study["design"]["roles"]]
                if any(len({r[k] for r in group}) != 1 for k in ("tape_sha256","cohort_sha256","enrolled")):
                    raise ValueError("unpaired exogenous or patient evaluation world")
                for role, seal in (("initial_value_mpc",initial),("updated_value_mpc",final)):
                    if rows[("evaluation",b,c,j,role)]["model_seal_sha256"] != seal["sha256"]:
                        raise ValueError("wrong evaluation value model")
    counts = Counter()
    for path in (root/"updates").glob("*.jsonl"):
        records = [json.loads(l) for l in path.read_text().splitlines()]
        if len(records) != 32 or [r["cohort_update"] for r in records] != list(range(1,33)):
            raise ValueError("incomplete value update receipts")
        if any(not np.isfinite([r["loss"], r["gradient_norm"]]).all() for r in records):
            raise ValueError("nonfinite persisted value update")
        counts[path.name.split("-b")[0]] += len(records)
    if counts != Counter(initial=2304, continuation=2304):
        raise ValueError("unexpected value optimizer counts")
    rng = np.random.default_rng(study["streams"]["bootstrap_seed"])
    contrasts = {}
    for learned, reference in (("updated_value_mpc","initial_value_mpc"),
                                ("updated_value_mpc","plain_mpc"),
                                ("updated_value_mpc","fixed_allocation_reference"),
                                ("initial_value_mpc","plain_mpc")):
        conditions = {}
        for c in range(3):
            pairs = []
            for b in range(3):
                for j in range(4):
                    a, r = rows[("evaluation",b,c,j,learned)], rows[("evaluation",b,c,j,reference)]
                    delta = np.asarray(a["actions"])-np.asarray(r["actions"])
                    pairs.append(dict(block=b, replicate=j, seed=a["world"]["seed"],
                        savings=r["cost"]-a["cost"], percent_savings=100*(r["cost"]-a["cost"])/r["cost"],
                        extra_lost=a["lost"]-r["lost"], extra_delivered=a["delivered"]-r["delivered"],
                        changed_action_boundaries=int((np.abs(delta).sum(1)>1e-9).sum()),
                        action_l1_hours=float(np.abs(delta).sum()),
                        component_savings={k:r["component_costs"][k]-a["component_costs"][k] for k in COST_COMPONENTS}))
            conditions[str(c)] = summarize_pairs(pairs,rng)
        contrasts[learned+"_vs_"+reference] = conditions
    primary = contrasts["updated_value_mpc_vs_initial_value_mpc"]
    criteria = dict(all_persistent_blocks_cost_positive=all(x["savings"]>0 for x in primary["1"]["blocks"]),
        persistent_descriptive_cost_interval_above_zero=primary["1"]["descriptive95"]["savings"][0]>0,
        no_mean_extra_losses_in_any_condition=all(x["means"]["extra_lost"]<=0 for x in primary.values()))
    return dict(format="capacity-value-mpc-readout-v1", complete=True, trajectories=list(rows.values()),
        trajectory_count=len(rows), raw_rows=len(rows)*64, evaluation_trajectories=144,
        updates=dict(counts), contrasts=contrasts, criteria=criteria, training_signal=all(criteria.values()),
        clinical_safety_established=False, deployment_online_adaptation=False,
        isolated_graph_contribution=False, automatic_follow_on=False)
