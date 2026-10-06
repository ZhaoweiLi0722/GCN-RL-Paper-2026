"""Saved-data checks for all fresh blocks, seven roles and conditional contrasts."""

from itertools import combinations
import gzip
import math
from pathlib import Path
import pickle

import numpy as np

from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_confirmation_design import (EVAL_ROLES, TRAIN_ROLES, WARM_ROLES,
    worlds, validate, capture_epochs, candidate_index, warm_config, tail_config)
from src.rl.capacity_pilot_analysis import COST_COMPONENTS, _close, _digest, _json, _safe_path, _sha
from src.rl.capacity_native_tail_analysis import _ident, _pair, _summarize
from src.rl.fixed_budget_capacity_analysis import allocation_receipts
from src.rl.patient_constrained_analysis import read_trajectory

def _read_json(path):
    return _json(path.read_text())


def _latency(values, expected):
    return dict(observed_decisions=len(values), expected_decisions=expected,
                complete=len(values) == expected,
                median_seconds=float(np.median(values)) if values else None,
                p95_seconds=float(np.percentile(values, 95)) if values else None,
                max_seconds=float(max(values)) if values else None)


def read_saved_trajectory(payload, world, role, ident=None):
    """Independently reconstruct raw costs/patients and delayed labor receipts."""
    root = Path(payload).resolve()
    ident = _ident(world, role) if ident is None else ident
    result = read_trajectory(root, world, role, ident)
    result.update(allocation_receipts(root, role, ident))
    if not _sha(result["tape_sha256"]):
        raise ValueError("invalid tape hash")
    requests, executed, applied, latencies = [], [], [], []
    with gzip.open(root / "raw" / (ident + ".jsonl.gz"), "rt") as handle:
        for line in handle:
            raw = _json(line)
            t, receipt = raw["epoch"], raw["info"]["support_public_receipt"]
            request = np.asarray(raw["requested_hours"], dtype=np.float64)
            action = np.asarray(raw["executed_hours"], dtype=np.float64)
            actual = np.asarray(receipt["applied_hours"], dtype=np.float64)
            if (request.shape != (4,) or action.shape != (4,) or actual.shape != (4,)
                    or not np.isfinite([request, action, actual]).all()
                    or math.fsum(request) > 8. + 1e-6
                    or receipt["epoch"] != t or receipt["known_at"] != t + 1
                    or not np.array_equal(request, receipt["raw_requested_hours"])
                    or not np.array_equal(action, receipt["committed_hours"])):
                raise ValueError("requested/committed receipt identity mismatch")
            # The inherited fixed physical contract has a two-step commitment lag.
            matured = executed[t - 2] if t >= 2 else np.zeros(4)
            if not np.allclose(actual, matured, rtol=0., atol=1e-9):
                raise ValueError("applied hours differ from two-step committed history")
            for field, component in (("labor_cost", "support_flexible_labor_cost"),
                                     ("switching_cost", "support_switching_cost")):
                if field in receipt and not _close(receipt[field], raw["components"][component]):
                    raise ValueError("labor cost receipt/component mismatch")
            requests.append(request)
            executed.append(action)
            applied.append(actual)
            plan = raw.get("plan")
            if t < 48 and isinstance(plan, dict) and "decision_wall_seconds" in plan:
                value = plan["decision_wall_seconds"]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid decision latency")
                latencies.append(float(value))
    summary = _read_json(root / "summaries" / (ident + ".json"))
    for key in ("enrolled", "lost", "delivered"):
        if key in summary and summary[key] != result[key]:
            raise ValueError("summary patient counts disagree with raw records")
    if world["phase"] == "evaluation":
        updates = 1536 if role in ("existing_frozen", "fresh_frozen") else 768 if role in TRAIN_ROLES else None
        if summary["value_updates_before_trajectory"] != updates:
            raise ValueError("wrong evaluation model update-count convention")
    result.update(requested_actions=np.asarray(requests[:48]).tolist(),
                  applied_actions=np.asarray(applied).tolist(),
                  total_requested_control_hours=math.fsum(np.asarray(requests[:48]).flat),
                  labor_receipt_identity_checked=True,
                  decision_wall_seconds=latencies, decision_latency=_latency(latencies, 48))
    tape_path = root / "tapes" / (_ident(world, "exogenous") + ".json")
    result["tape_artifact_verified"] = tape_path.is_file()
    if tape_path.is_file():
        tape = _read_json(tape_path)
        if tape.get("world") != world or digest(tape) != result["tape_sha256"]:
            raise ValueError("saved tape world/hash differs from trajectory")
    return result


def saved_native_branches(root, study):
    """Independently sum raw branch costs; do not run a model or simulator."""
    output, seen = [], set()
    expected_files = {_ident(w, "plain_h8") + ".pkl.gz"
        for b in range(5) for w in worlds(study, "reference", b)}
    if {p.name for p in (root / "training-tails").glob("*.pkl.gz")} != expected_files:
        raise ValueError("incomplete native training cohort files")
    for b in range(5):
        for world in worlds(study, "reference", b):
            path = root / "training-tails" / (_ident(world, "plain_h8") + ".pkl.gz")
            tape = _read_json(root / "tapes" / (_ident(world, "exogenous") + ".json"))
            if tape.get("world") != world:
                raise ValueError("native branch reference tape identity mismatch")
            tape_sha = digest(tape)
            with gzip.open(path, "rb") as handle:
                cohort = pickle.load(handle)
            records = cohort["native"]
            if len(records) != 2:
                raise ValueError("exactly two native branches required")
            roots = set()
            for record in records:
                epoch = record["root_epoch"]
                if epoch not in capture_epochs(world["index"]) or epoch in roots:
                    raise ValueError("wrong or duplicate native branch root")
                roots.add(epoch)
                candidate = candidate_index(world["index"], epoch)
                start = epoch + 8
                if (record["format"] != "capacity-native-tail-record-v1"
                        or record["candidate"] != candidate or record["start_epoch"] != start
                        or record["epochs"] != list(range(start, 65))
                        or record["label_policy"] != "frozen_mpc"
                        or record["provenance"] != "native_patient_branch"
                        or record["continuation_sha256"] != _read_json(root / "models" / f"block{b}-graph-warm.json")["sha256"]
                        or not _sha(record["source_state_sha256"]) or record["tape_sha256"] != tape_sha):
                    raise ValueError("invalid native branch identity")
                raw_path = _safe_path(root, record["raw_path"])
                if raw_path in seen:
                    raise ValueError("branch raw file reused")
                seen.add(raw_path)
                with gzip.open(raw_path, "rt") as handle:
                    raw = [_json(line) for line in handle]
                if [r["epoch"] for r in raw] != list(range(epoch, 64)):
                    raise ValueError("incomplete raw branch")
                costs, lost = [], None
                requested, executed = [], []
                for row in raw:
                    t = row["epoch"]
                    if any(row["metadata"].get(k) != record[k] for k in (
                            "format", "root_epoch", "candidate", "start_epoch", "label_policy",
                            "continuation_sha256", "source_state_sha256", "tape_sha256", "provenance")):
                        raise ValueError("raw branch metadata mismatch")
                    components = row["components"]
                    cost = float(row["cost"])
                    if (set(components) != set(COST_COMPONENTS) or not math.isfinite(cost) or cost < 0
                            or not _close(math.fsum(components.values()), cost)
                            or not _close(-float(row["reward"]), cost)
                            or row["prefix"] != (t < start)):
                        raise ValueError("raw native objective mismatch")
                    patients = row["patient_records"]
                    ids = [p["patient_id"] for p in patients]
                    current = {p["patient_id"] for p in patients if p["status"] == "lost"}
                    prior_patients = row["public_input"]["operations"]["patients"]
                    prior_lost = {p["patient_id"] for p in prior_patients if p["status"] == "lost"}
                    if (len(set(ids)) != len(ids) or not prior_lost.issubset(current)
                            or lost is not None and lost != prior_lost
                            or type(row["new_lost_patients"]) is not int
                            or row["new_lost_patients"] != len(current - prior_lost)):
                        raise ValueError("raw native patient identity or loss reversal")
                    lost = current
                    request = np.asarray(row["requested_hours"], dtype=np.float64)
                    committed = np.asarray(row["executed_hours"], dtype=np.float64)
                    receipt = row["info"]["support_public_receipt"]
                    if (request.shape != (4,) or committed.shape != (4,)
                            or not np.isfinite([request, committed]).all()
                            or (request < 0).any() or math.fsum(request) > 8. + 1e-6
                            or t >= 48 and np.count_nonzero(request)
                            or not np.array_equal(request, receipt["raw_requested_hours"])
                            or not np.array_equal(committed, receipt["committed_hours"])
                            or receipt["epoch"] != t or receipt["known_at"] != t + 1):
                        raise ValueError("raw native request or receipt mismatch")
                    if len(executed) >= 2 and not np.allclose(receipt["applied_hours"], executed[-2], rtol=0., atol=1e-9):
                        raise ValueError("native branch labor lag mismatch")
                    requested.append(request)
                    executed.append(committed)
                    costs.append(cost)
                stored = np.asarray(record["costs"])
                prefix = np.asarray(record["prefix_costs"])
                if (stored.dtype != np.dtype("float64") or prefix.dtype != np.dtype("float64")
                        or not np.array_equal(stored, np.asarray(costs[8:], dtype=np.float64))
                        or not np.array_equal(prefix, np.asarray(costs[:8], dtype=np.float64))
                        or not _close(math.fsum(costs), record["total_branch_cost"])
                        or not record["settlement"]["settled"]
                        or record["settlement"]["lost"] != len(lost)):
                    raise ValueError("raw cost/suffix/settlement differs from admitted native data")
                output.append(dict(block=b, world_index=world["index"], root_epoch=epoch,
                    candidate=candidate, branch_steps=len(costs), suffix_rows=len(stored),
                    native_suffix_cost=math.fsum(stored), total_branch_cost=math.fsum(costs),
                    lost=len(lost), raw_sha256=_digest(raw_path), tape_sha256=record["tape_sha256"]))
    if len(output) != 240 or sum(r["branch_steps"] for r in output) != 9720:
        raise ValueError("incomplete native branch matrix")
    return dict(branches=output, branch_count=240, native_steps=9720,
        same_reference_future_tapes=True, independent_worlds=0,
        costs_verified_from_raw=True, new_scientific_calls=0,
        common_parent_shared_native_data=True)



def interaction(conditions, rng, resamples):
    """Shared training-block resampling; condition worlds are not paired."""
    output = {}
    for c in (0, 1):
        arrays = [np.asarray([[p["savings"] for p in conditions[str(k)]["pairs"] if p["block"] == b]
                              for b in range(5)]) for k in (2, c)]
        if any(a.shape != (5, 4) for a in arrays):
            raise ValueError("incomplete interaction blocks")
        blocks = rng.integers(0, 5, (resamples, 5))
        within_fast = rng.integers(0, 4, (resamples, 5, 4))
        within_other = rng.integers(0, 4, (resamples, 5, 4))
        sampled = (arrays[0][blocks[..., None], within_fast].mean((1, 2))
                   - arrays[1][blocks[..., None], within_other].mean((1, 2)))
        output["fast_minus_" + str(c)] = dict(
            savings_difference=float(arrays[0].mean() - arrays[1].mean()),
            block_differences=(arrays[0].mean(1) - arrays[1].mean(1)).tolist(),
            descriptive95=np.percentile(sampled, [2.5, 97.5]).tolist(),
            condition_worlds_paired=False, independent_training_blocks=5)
    return output


def model_evidence(root, study):
    warm, final, legacy = {}, {}, {}
    for b in range(5):
        for role in WARM_ROLES:
            key = f"block{b}-{role}"
            name = key + "-warm"
            meta = _read_json(root / "models" / (name + ".json"))
            path = root / "models" / (name + ".pt")
            if (meta["updates"] != 1536 or meta["role"] != role or meta["block"] != b
                    or meta["parameters"] != 3169 or meta["config"] != warm_config(study, role)
                    or meta["sha256"] != _digest(path) or meta["bytes"] != path.stat().st_size):
                raise ValueError("warm model/config/byte mismatch")
            warm[key] = meta["sha256"]
        for role in TRAIN_ROLES:
            key, parent = f"block{b}-{role}", warm[f"block{b}-graph"]
            meta = _read_json(root / "models" / (key + "-final.json"))
            path = root / "models" / (key + "-final.pt")
            architecture = "self_only" if role == "self_only_td" else "graph"
            if (meta["updates"] != 768 or meta["role"] != role or meta["block"] != b
                    or meta["parameters"] != 3169 or meta["config"] != tail_config(study, role, parent)
                    or meta["own_ancestor_sha256"] != warm[f"block{b}-{architecture}"]
                    or meta["continuation_sha256"] != parent
                    or meta["sha256"] != _digest(path) or meta["bytes"] != path.stat().st_size):
                raise ValueError("final model/own ancestor/common parent mismatch")
            final[key] = meta["sha256"]
        meta = _read_json(root / "models" / f"block{b}-legacy.json")
        if meta["sha256"] != study["initial_models"]["sha256"][b] or not meta["historical_reference_only"]:
            raise ValueError("legacy reference binding mismatch")
        legacy[str(b)] = meta["sha256"]
    expected = dict(warm=warm, final=final, legacy=legacy)
    if _read_json(root / "models/all-sealed.json") != expected:
        raise ValueError("all-sealed manifest mismatch")
    names = {k + "-warm" for k in warm} | {k + "-final" for k in final}
    if {p.stem for p in (root / "models").glob("*.pt")} != names:
        raise ValueError("missing or extra model bytes")
    return expected


def fit_evidence(root, study):
    expected, total = set(), 0
    for phase, roles in (("warmup", WARM_ROLES), ("reference", TRAIN_ROLES)):
        for b in range(5):
            for w in worlds(study, phase, b):
                batches = []
                hashes = []
                for role in roles:
                    name = _ident(w, role)
                    expected.add(name)
                    rows = [_json(line) for line in (root / "updates" / (name + ".jsonl")).read_text().splitlines()]
                    if (len(rows) != 32 or [r["update"] for r in rows] != list(range(32*w["index"]+1, 32*(w["index"]+1)+1))
                            or any(r["cohort_update"] != i+1 or not np.isfinite([r["loss"], r["gradient_norm"]]).all()
                                   for i, r in enumerate(rows))
                            or not (root / "states" / (name + "-after-fit.pkl.gz")).is_file()):
                        raise ValueError("missing finite update receipt/after-fit state")
                    batches.append([r["indices"] for r in rows])
                    targets = _read_json(root / "updates" / (name + "-targets.json"))
                    if targets["source_native"] != _ident(w, "plain_h8") or targets["method"] != role:
                        raise ValueError("training data source mismatch")
                    hashes.append(targets["source_hashes"])
                    total += 32
                if any(x != batches[0] for x in batches[1:]) or any(x != hashes[0] for x in hashes[1:]):
                    raise ValueError("unmatched training row sources or sampling indices")
    if total != 26880 or {p.stem for p in (root / "updates").glob("*.jsonl")} != expected:
        raise ValueError("incomplete training matrix")
    return dict(total_updates=total, paired_sources_and_indices=True, after_fit_states=len(expected))


def run_analysis(payload, study):
    validate(study)
    root, rows, names = Path(payload).resolve(), {}, set()
    models = model_evidence(root, study)
    fit = fit_evidence(root, study)
    for phase in ("warmup", "reference", "evaluation"):
        roles = EVAL_ROLES if phase == "evaluation" else ("plain_h8",)
        for b in range(5):
            for w in worlds(study, phase, b):
                for role in roles:
                    name = _ident(w, role)
                    names.add(name)
                    row = read_saved_trajectory(root, w, role)
                    wanted = (models["warm"][f"block{b}-graph"] if role == "fresh_frozen"
                              else models["legacy"][str(b)] if role == "existing_frozen"
                              else models["final"].get(f"block{b}-{role}")) if phase == "evaluation" else None
                    if row["model_seal_sha256"] != wanted:
                        raise ValueError("wrong evaluation model binding")
                    rows[(phase, b, w["condition"], w["replicate"], role)] = row
    if ({p.stem for p in (root / "summaries").glob("*.json")} != names
            or {p.name.removesuffix(".jsonl.gz") for p in (root / "raw").glob("*.jsonl.gz")} != names):
        raise ValueError("missing or extra raw trajectories")
    events = [_json(line) for line in (root / "progress.jsonl").read_text().splitlines()]
    barriers = [i for i, e in enumerate(events) if e["event"] == "all_models_sealed"]
    if len(barriers) != 1:
        raise ValueError("one complete seal barrier required")
    before = events[:barriers[0]]
    if sum(e["event"] == "warm_sealed" for e in before) != 10 or sum(e["event"] == "final_sealed" for e in before) != 15:
        raise ValueError("incomplete warm/final seal events")
    opened = set()
    for i, e in enumerate(events):
        if e["event"] == "trajectory_started" and e["active"]["phase"] == "evaluation":
            name = _ident(e["active"], e["active"]["role"])
            if i <= barriers[0] or name in opened or e["counts"]["total_optimizer_steps"] != 26880:
                raise ValueError("test before barrier or test updates")
            opened.add(name)
    wanted = {_ident(w, r) for b in range(5) for w in worlds(study, "evaluation", b) for r in EVAL_ROLES}
    if opened != wanted:
        raise ValueError("incomplete test starts")
    for b in range(5):
        for w in worlds(study, "evaluation", b):
            group = [rows[("evaluation", b, w["condition"], w["replicate"], r)] for r in EVAL_ROLES]
            if any(len({r[k] for r in group}) != 1 for k in ("tape_sha256", "cohort_sha256", "enrolled")):
                raise ValueError("unpaired controller worlds")
    rng = np.random.default_rng(study["streams"]["bootstrap_seed"])
    contrasts = {}
    ordered = ("graph_td",) + tuple(r for r in EVAL_ROLES if r != "graph_td")
    for a, ref in combinations(ordered, 2):
        contrasts[a + "_vs_" + ref] = {str(c): _summarize([
            _pair(rows[("evaluation", b, c, j, a)], rows[("evaluation", b, c, j, ref)])
            for b in range(5) for j in range(4)], rng, 2000) for c in range(3)}
    primary, interactions = {}, {}
    for ref in ("plain_h8", "fresh_frozen"):
        conditions = contrasts["graph_td_vs_" + ref]
        fast = conditions["2"]
        primary[ref] = dict(all_five_fast_blocks_cost_positive=all(b["savings"] > 0 for b in fast["blocks"]),
            fast_cost_descriptive95_above_zero=fast["descriptive95"]["savings"][0] > 0,
            no_fast_mean_extra_losses=fast["means"]["extra_lost"] <= 0)
        interactions[ref] = interaction(conditions, rng, 2000)
    return dict(format="capacity-confirmation-readout-v1", complete=True, trajectory_count=780,
        raw_rows=49920, evaluation_trajectories=420, paired_test_worlds=60,
        fresh_training_blocks=5, trajectories=list(rows.values()), contrasts=contrasts,
        primary_criteria=primary, joint_primary_development_signal=all(all(x.values()) for x in primary.values()),
        condition_interactions=interactions, model_evidence=models, fit_evidence=fit,
        barrier_verified=True, test_updates=0, saved_native_branches=saved_native_branches(root, study),
        latency={r: _latency([v for row in rows.values() if row["world"]["phase"] == "evaluation"
                            and row["role"] == r for v in row["decision_wall_seconds"]], 2880) for r in EVAL_ROLES},
        planned_caps=study["budget"]["counts"], planned_caps_are_not_measured_resources=True,
        formal_multiplicity_adjusted_confirmation=False, clinical_safety=False,
        deployment_online_adaptation=False, exploratory_condition_selection_disclosed=True,
        common_graph_parent_conditions_ablation=True, automatic_follow_on=False)
