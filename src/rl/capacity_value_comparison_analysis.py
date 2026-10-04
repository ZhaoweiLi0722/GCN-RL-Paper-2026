"""Independent raw-ledger arithmetic for the five-block graph/flat comparison."""

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_pilot_analysis import COST_COMPONENTS
from src.rl.capacity_value_comparison_resources import ARCHITECTURES, ROLES, worlds, numeric_contract
from src.rl.fixed_budget_capacity_analysis import allocation_receipts
from src.rl.patient_constrained_analysis import read_trajectory


def summarize(pairs, rng, *, resamples=2000):
    keys = ("savings", "extra_lost", "extra_delivered", "percent_savings")
    groups = [[r for r in pairs if r["block"] == b] for b in range(5)]
    if any(len(g) != 4 or {r["replicate"] for r in g} != set(range(4)) for g in groups):
        raise ValueError("five complete independent blocks required")
    data = np.asarray([[[r[k] for k in keys] for r in group] for group in groups])
    blocks = rng.integers(0, 5, (resamples, 5))
    within = rng.integers(0, 4, (resamples, 5, 4))
    sampled = data[blocks[..., None], within].mean(axis=(1, 2))
    means = data.mean(axis=(0, 1))
    return dict(n_worlds=len(pairs), independent_blocks=5,
        means=dict(zip(keys, map(float, means))),
        blocks=[dict(block=b, **dict(zip(keys, map(float, group.mean(0))))) for b,group in enumerate(data)],
        descriptive95={k:np.percentile(sampled[:,i],[2.5,97.5]).tolist() for i,k in enumerate(keys)},
        pairs=pairs)


def _criteria(conditions):
    return dict(all_persistent_blocks_cost_positive=all(x["savings"]>0 for x in conditions["1"]["blocks"]),
        persistent_descriptive_cost_interval_above_zero=conditions["1"]["descriptive95"]["savings"][0]>0,
        no_mean_extra_losses_in_any_condition=all(x["means"]["extra_lost"]<=0 for x in conditions.values()))


def run_analysis(root, study):
    numeric_contract(study)
    root, rows, expected, input_digests = Path(root), {}, set(), {}
    counts, fit_names, seals = Counter(), set(), {}
    for phase in ("initial", "continuation", "evaluation"):
        roles = ["plain_mpc"] if phase == "initial" else [a+"_value_mpc" for a in ARCHITECTURES] if phase == "continuation" else ROLES
        for block in range(5):
            for w in worlds(study, phase, block):
                for role in roles:
                    name = f"{phase}-b{block}-c{w['condition']}-j{w['replicate']}-{role}"
                    expected.add(name)
                    row = read_trajectory(root, w, role, name)
                    row.update(allocation_receipts(root, role, name))
                    rows[(phase,block,w["condition"],w["replicate"],role)] = row
                    if phase != "evaluation":
                        features, heuristics, costs = [], [], []
                        with gzip.open(root/"raw"/(name+".jsonl.gz"), "rt") as handle:
                            for line in handle:
                                raw = json.loads(line)
                                features.append(raw["value_features"])
                                heuristics.append(raw["value_base"])
                                costs.append(raw["cost"])
                        features.append(raw["next_value_features"])
                        heuristics.append(raw["next_value_base"])
                        input_digests[name] = digest(dict(features=features, heuristics=heuristics, costs=costs))
    if ({p.stem for p in (root/"summaries").glob("*.json")} != expected
            or {p.name.removesuffix(".jsonl.gz") for p in (root/"raw").glob("*.jsonl.gz")} != expected):
        raise ValueError("missing or extra comparison trajectories")
    for block in range(5):
        parameters = {}
        for architecture in ARCHITECTURES:
            for kind,expected_updates in (("initial",768),("final",1536)):
                name = f"block{block}-{architecture}-{kind}"
                metadata = json.loads((root/"models"/(name+".json")).read_text())
                raw = (root/"models"/(name+".pt")).read_bytes()
                if (metadata["updates"] != expected_updates or metadata["architecture"] != architecture
                        or metadata["config"] != dict(study["value"], architecture=architecture)
                        or hashlib.sha256(raw).hexdigest() != metadata["sha256"] or len(raw) != metadata["bytes"]):
                    raise ValueError("wrong comparison model seal")
                parameters[architecture] = metadata["parameters"]
                seals[(block,architecture,kind)] = metadata["sha256"]
            for phase, offset in (("initial",0),("continuation",768)):
                for index,w in enumerate(worlds(study, phase, block)):
                    name = f"{phase}-b{block}-c{w['condition']}-j{w['replicate']}-{architecture}_value_fit"
                    fit_names.add(name)
                    records = [json.loads(l) for l in (root/"updates"/(name+".jsonl")).read_text().splitlines()]
                    if (len(records) != 32 or [r["cohort_update"] for r in records] != list(range(1,33))
                            or [r["update"] for r in records] != list(range(offset+32*index+1,offset+32*(index+1)+1))
                            or any(not np.isfinite([r["loss"],r["gradient_norm"]]).all() for r in records)):
                        raise ValueError("incomplete or nonfinite comparison value updates")
                    counts[phase+"_"+architecture] += len(records)
                    source_role = "plain_mpc" if phase == "initial" else architecture+"_value_mpc"
                    source = f"{phase}-b{block}-c{w['condition']}-j{w['replicate']}-{source_role}"
                    meta = json.loads((root/"updates"/(name+"-input.json")).read_text())
                    if meta != dict(architecture=architecture, source_trajectory=source,
                                    data_sha256=input_digests[source], updates_after=offset+32*(index+1)):
                        raise ValueError("fit inputs do not match saved cohort data")
                    targets = json.loads((root/"updates"/(name+"-targets.json")).read_text())
                    if (targets["td_horizon"] != 8 or targets["terminal_bootstrap_zero"] is not True
                            or np.asarray(targets["targets"]).shape != (64,) or not np.isfinite(targets["targets"]).all()
                            or not (root/"states"/(name+"-after-fit.pkl.gz")).is_file()):
                        raise ValueError("missing complete update boundary")
        if (any(type(n) is not int or n <= 0 for n in parameters.values())
                or abs(parameters["graph"]-parameters["flat"])/parameters["graph"] > .01):
            raise ValueError("unmatched graph/flat parameters")
        for phase in ("continuation", "evaluation"):
            roles = [a+"_value_mpc" for a in ARCHITECTURES] if phase == "continuation" else ROLES
            for w in worlds(study, phase, block):
                group = [rows[(phase,block,w["condition"],w["replicate"],r)] for r in roles]
                if any(len({r[k] for r in group}) != 1 for k in ("tape_sha256", "cohort_sha256", "enrolled")):
                    raise ValueError("unpaired comparison tapes or patient cohort")
                if phase == "evaluation":
                    for r in group:
                        architecture = r["role"].split("_")[0]
                        wanted = seals.get((block,architecture,"final"))
                        if r["model_seal_sha256"] != wanted:
                            raise ValueError("wrong frozen evaluation model")
    if {p.stem for p in (root/"updates").glob("*.jsonl")} != fit_names or counts != Counter(
            initial_graph=3840,initial_flat=3840,continuation_graph=3840,continuation_flat=3840):
        raise ValueError("unexpected comparison fit receipt counts")
    events = [json.loads(l) for l in (root/"progress.jsonl").read_text().splitlines()]
    barriers = [i for i,e in enumerate(events) if e["event"] == "all_models_sealed"]
    if len(barriers) != 1 or sum(e["event"] == "final_sealed" for e in events[:barriers[0]]) != 10:
        raise ValueError("missing all-models-sealed barrier")
    for i,event in enumerate(events):
        if event["event"] == "trajectory_started" and event["active"]["phase"] == "evaluation":
            if i <= barriers[0] or event["counts"]["total_optimizer_steps"] != 15360:
                raise ValueError("test opened before complete training barrier")
    rng, contrasts = np.random.default_rng(study["streams"]["bootstrap_seed"]), {}
    for learned,reference in (("graph_value_mpc","plain_mpc"),("flat_value_mpc","plain_mpc"),
            ("graph_value_mpc","flat_value_mpc"),("graph_value_mpc","fixed_allocation_reference"),
            ("flat_value_mpc","fixed_allocation_reference"),("plain_mpc","fixed_allocation_reference")):
        conditions = {}
        for c in range(3):
            pairs = []
            for b in range(5):
                for j in range(4):
                    a,ref = rows[("evaluation",b,c,j,learned)],rows[("evaluation",b,c,j,reference)]
                    delta = np.asarray(a["actions"])-np.asarray(ref["actions"])
                    if ref["cost"] <= 0:
                        raise ValueError("undefined relative savings with nonpositive reference cost")
                    pairs.append(dict(block=b,replicate=j,seed=a["world"]["seed"],
                        savings=ref["cost"]-a["cost"], percent_savings=100*(ref["cost"]-a["cost"])/ref["cost"],
                        extra_lost=a["lost"]-ref["lost"], extra_delivered=a["delivered"]-ref["delivered"],
                        changed_action_boundaries=int((np.abs(delta).sum(1)>1e-9).sum()),
                        action_l1_hours=float(np.abs(delta).sum()),
                        component_savings={k:ref["component_costs"][k]-a["component_costs"][k] for k in COST_COMPONENTS}))
            conditions[str(c)] = summarize(pairs,rng,resamples=study["readout"]["bootstrap_resamples"])
        contrasts[learned+"_vs_"+reference] = conditions
    primary = _criteria(contrasts["graph_value_mpc_vs_plain_mpc"])
    graph = _criteria(contrasts["graph_value_mpc_vs_flat_value_mpc"])
    return dict(format="capacity-value-comparison-readout-v1", complete=True,
        trajectory_count=len(rows), raw_rows=len(rows)*64, evaluation_trajectories=240,
        updates=dict(counts), trajectories=list(rows.values()), contrasts=contrasts,
        primary_criteria=primary, secondary_graph_criteria=graph,
        strong_baseline_development_signal=all(primary.values()),
        graph_inductive_bias_development_signal=all(graph.values()),
        clinical_safety_established=False, deployment_online_adaptation=False,
        isolated_edge_effect=False, formal_confirmation=False, automatic_follow_on=False)
