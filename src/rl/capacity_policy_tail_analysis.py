"""Read-only six-role analysis of saved planner-tail evidence; no execution.

``run_analysis(payload, study)`` expects CapacityValueRunner raw/summaries named
``{phase}-b{block}-c{condition}-j{replicate}-{role}``, with worlds from the new
planner-tail design (including ``index``). It reads 120 reference plain_h8 and
360 evaluation trajectories, never historical trajectories or model objects.

Model evidence defaults to ``models/block{b}-ancestor.json/.pt`` and
``models/block{b}-{role}-final.json/.pt`` plus ``models/all-sealed.json``.
The latter maps ``ancestors`` (block{b}->sha) and ``final_models``
(block{b}-{role}->sha). Model bytes are hashed, never deserialized.
Alternatively, ``study['analysis_inputs']['model_manifest']`` may supply a
mapping or payload-relative JSON path with format
``capacity-policy-tail-model-manifest-v1`` containing ``ancestors`` (5)
and ``final_models`` (15). Each record has block, sha256, updates; final records
also have role and ancestor_sha256. Ancestor updates are 1536; final NEW updates
are 768. Optional path (and optional bytes count) verifies saved bytes. Hash-only
records are explicitly reported as manifest bindings, NOT byte verification.

Optional ``tapes/*-exogenous.json`` use the existing canonical tape digest.
Optional ``progress.jsonl`` must contain one all_models_sealed barrier, then
all 360 evaluation trajectory_started events with active=world+role and counts
at 11520 value_optimizer_steps (total_optimizer_steps is also accepted).
If per-model ancestor_bound/final_sealed events are supplied, they must also
match the complete manifests and precede the barrier.
No optional evidence is silently claimed as checked when absent.
"""

import copy
import gzip
from itertools import combinations
import math
import pickle
from pathlib import Path

import numpy as np

from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_pilot_analysis import COST_COMPONENTS, _close, _digest, _json, _safe_path, _sha
from src.rl.capacity_policy_tail_design import (EVAL_ROLES, TRAIN_ROLES, numeric_contract,
    worlds, learner_config, capture_epochs, candidate_index)
from src.rl.capacity_value_comparison_analysis import summarize
from src.rl.fixed_budget_capacity_analysis import allocation_receipts
from src.rl.patient_constrained_analysis import read_trajectory


def expected_trajectories(study):
    """Yield (world, role, basename) for the exact new 480-trajectory matrix."""
    numeric_contract(study)
    for phase in ("reference", "evaluation"):
        for block in range(5):
            for world in worlds(study, phase, block):
                for role in ("plain_h8",) if phase == "reference" else EVAL_ROLES:
                    yield world, role, _ident(world, role)


def _ident(world, role):
    return f"{world['phase']}-b{world['block']}-c{world['condition']}-j{world['replicate']}-{role}"


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
        updates = 1536 if role == "existing_frozen" else 768 if role in TRAIN_ROLES else None
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


def _model_evidence(root, study):
    root = Path(root).resolve()
    source = study.get("analysis_inputs", {}).get("model_manifest")
    if source is None:
        names = {f"block{b}-ancestor" for b in range(5)} | {
            f"block{b}-{r}-final" for b in range(5) for r in TRAIN_ROLES}
        if ({p.stem for p in (root / "models").glob("*.pt")} != names
                or {p.stem for p in (root / "models").glob("block*.json")} != names):
            raise ValueError("missing or extra final/ancestor model artifacts")
        manifest = dict(format="capacity-policy-tail-model-manifest-v1", ancestors=[], final_models=[])
        for block in range(5):
            for role in ("ancestor", *TRAIN_ROLES):
                name = f"block{block}-ancestor" if role == "ancestor" else f"block{block}-{role}-final"
                record = _read_json(root / "models" / (name + ".json"))
                if role != "ancestor" and (record["method"] != role or record["config"] != learner_config(study, role, study["initial_models"]["sha256"][block])):
                    raise ValueError("final model method/config differs from study")
                record = dict(record, block=block, path="models/" + name + ".pt")
                if role != "ancestor":
                    record["role"] = role
                manifest["ancestors" if role == "ancestor" else "final_models"].append(record)
    else:
        manifest = copy.deepcopy(source) if isinstance(source, dict) else _read_json(_safe_path(root, source))
    if manifest.get("format") != "capacity-policy-tail-model-manifest-v1":
        raise ValueError("explicit planner-tail model manifest required")
    ancestors, finals, verified, declared = {}, {}, [], []
    expected_hashes = study["initial_models"]["sha256"]
    if len(expected_hashes) != 5 or any(not _sha(sha) for sha in expected_hashes):
        raise ValueError("five configured ancestor hashes required")
    for kind, records in (("ancestor", manifest["ancestors"]), ("final", manifest["final_models"])):
        for record in records:
            b, sha = record["block"], record["sha256"]
            if type(b) is not int or b not in range(5) or not _sha(sha):
                raise ValueError("invalid model block or hash")
            if kind == "ancestor":
                key = b
                collection, updates = ancestors, 1536
                if sha != expected_hashes[b]:
                    raise ValueError("ancestor binding differs from frozen study input")
            else:
                key = (b, record["role"])
                collection, updates = finals, 768
                if record["role"] not in TRAIN_ROLES or record["ancestor_sha256"] != expected_hashes[b]:
                    raise ValueError("wrong final role or ancestor binding")
                config = learner_config(study, record["role"], expected_hashes[b])
                if record.get("method", record["role"]) != record["role"] or record.get("config", config) != config:
                    raise ValueError("final model method/config differs from study")
            if (key in collection or type(record["updates"]) is not int or record["updates"] != updates
                    or record.get("parameters", 3169) != 3169):
                raise ValueError("duplicate model or wrong sealed update/parameter count")
            collection[key] = sha
            label = f"block{b}-{record.get('role', 'ancestor')}"
            if "path" in record:
                path = _safe_path(root, record["path"])
                if (not path.is_file() or _digest(path) != sha or ("bytes" in record and
                        (type(record["bytes"]) is not int or record["bytes"] <= 0
                         or path.stat().st_size != record["bytes"]))):
                    raise ValueError("model byte hash/length mismatch")
                verified.append(label)
            else:
                declared.append(label)
    if set(ancestors) != set(range(5)) or set(finals) != {(b, r) for b in range(5) for r in TRAIN_ROLES}:
        raise ValueError("exactly five ancestors and fifteen final seals required")
    barrier_path = root / "models" / "all-sealed.json"
    if source is None or barrier_path.exists():
        barrier = _read_json(barrier_path)
        if (barrier["ancestors"] != {f"block{b}": sha for b, sha in ancestors.items()}
                or barrier["final_models"] != {f"block{b}-{r}": sha for (b, r), sha in finals.items()}):
            raise ValueError("all-sealed manifest differs from model evidence")
    return ancestors, finals, dict(manifest_sha256=digest(manifest), ancestor_count=5,
        final_count=15, byte_verified=verified, manifest_only=declared,
        ancestors={f"block{b}": sha for b, sha in ancestors.items()},
        final_models={f"block{b}-{r}": sha for (b, r), sha in finals.items()},
        all_sealed_manifest_sha256=_digest(barrier_path) if barrier_path.exists() else None,
        model_deserialization_performed=False, optimizer_receipts_checked=False)


def _progress_evidence(root, expected, ancestors, finals):
    path = root / "progress.jsonl"
    if not path.exists():
        return dict(provided=False, barrier_verified=False, recorded_counts=None)
    events = [_json(line) for line in path.read_text().splitlines()]
    barriers = [i for i, e in enumerate(events) if e["event"] == "all_models_sealed"]
    if len(barriers) != 1:
        raise ValueError("one all-models-sealed barrier required")
    barrier = barriers[0]
    seen_ancestors, seen_finals, opened = {}, {}, set()
    counts = None
    wanted = {ident for w, _, ident in expected if w["phase"] == "evaluation"}
    expected_worlds = {ident: dict(w, role=r) for w, r, ident in expected if w["phase"] == "evaluation"}
    for i, event in enumerate(events):
        name = event["event"]
        if "counts" in event:
            counts = event["counts"]
            updates = [counts[k] for k in ("value_optimizer_steps", "total_optimizer_steps") if k in counts]
            if i >= barrier and (not updates or any(type(n) is not int or n != 11520 for n in updates)):
                raise ValueError("optimizer updates changed or absent after training barrier")
            if counts.get("actor_optimizer_steps", 0) != 0:
                raise ValueError("unexpected actor optimizer activity")
        if name in ("ancestor_bound", "final_sealed"):
            if i >= barrier:
                raise ValueError("model binding occurs after training barrier")
            key = event["block"] if name == "ancestor_bound" else (event["block"], event.get("role", event.get("method")))
            seen, bindings = (seen_ancestors, ancestors) if name == "ancestor_bound" else (seen_finals, finals)
            if key in seen or bindings.get(key) != event["sha256"]:
                raise ValueError("duplicate or wrong model seal event")
            seen[key] = event["sha256"]
        if name == "trajectory_started" and event["active"]["phase"] == "evaluation":
            active = event["active"]
            ident = _ident(active, active["role"])
            counters = event.get("counts", {})
            if (i <= barrier or ident in opened or expected_worlds.get(ident) != active
                    or not any(k in counters for k in ("value_optimizer_steps", "total_optimizer_steps"))):
                raise ValueError("evaluation opened before barrier or with wrong identity/counts")
            opened.add(ident)
    if ((seen_ancestors and seen_ancestors != ancestors) or (seen_finals and seen_finals != finals)
            or opened != wanted):
        raise ValueError("incomplete seal/barrier/evaluation event evidence")
    return dict(provided=True, barrier_verified=True, sha256=_digest(path),
                evaluation_starts=len(opened), recorded_counts=counts)


def _pair(a, ref):
    if ref["cost"] <= 0:
        raise ValueError("undefined relative savings with nonpositive reference cost")
    action_delta = np.asarray(a["actions"]) - np.asarray(ref["actions"])
    request_delta = np.asarray(a["requested_actions"]) - np.asarray(ref["requested_actions"])
    applied_delta = np.asarray(a["applied_actions"]) - np.asarray(ref["applied_actions"])
    world = a["world"]
    return dict(block=world["block"], replicate=world["replicate"], seed=world["seed"],
        savings=ref["cost"] - a["cost"], percent_savings=100 * (ref["cost"] - a["cost"]) / ref["cost"],
        extra_lost=a["lost"] - ref["lost"], extra_delivered=a["delivered"] - ref["delivered"],
        changed_action_boundaries=int((np.abs(action_delta).sum(axis=1) > 1e-9).sum()),
        action_l1_hours=float(np.abs(action_delta).sum()),
        requested_action_l1_hours=float(np.abs(request_delta).sum()),
        applied_action_l1_hours=float(np.abs(applied_delta).sum()),
        extra_requested_control_hours=a["total_requested_control_hours"] - ref["total_requested_control_hours"],
        extra_committed_control_hours=a["total_committed_control_hours"] - ref["total_committed_control_hours"],
        extra_applied_hours_including_tail=a["total_applied_hours_including_tail"] - ref["total_applied_hours_including_tail"],
        component_savings={k: ref["component_costs"][k] - a["component_costs"][k] for k in COST_COMPONENTS})


def _summarize(pairs, rng, resamples):
    result = summarize(pairs, rng, resamples=resamples)
    fields = ("changed_action_boundaries", "action_l1_hours", "requested_action_l1_hours",
              "applied_action_l1_hours", "extra_requested_control_hours",
              "extra_committed_control_hours", "extra_applied_hours_including_tail")
    for target, group in [(result["means"], pairs)] + [
            (block, [p for p in pairs if p["block"] == block["block"]]) for block in result["blocks"]]:
        target.update({k: float(np.mean([p[k] for p in group])) for k in fields})
        target["component_savings"] = {k: float(np.mean([p["component_savings"][k] for p in group]))
                                       for k in COST_COMPONENTS}
    result["harm"] = dict(cost_harm_worlds=sum(p["savings"] < 0 for p in pairs),
                          loss_harm_worlds=sum(p["extra_lost"] > 0 for p in pairs),
                          cost_harm_blocks=sum(b["savings"] < 0 for b in result["blocks"]),
                          loss_harm_blocks=sum(b["extra_lost"] > 0 for b in result["blocks"]))
    return result


def _criteria(conditions):
    return dict(all_persistent_blocks_cost_positive=all(b["savings"] > 0 for b in conditions["1"]["blocks"]),
        persistent_descriptive_cost_interval_above_zero=conditions["1"]["descriptive95"]["savings"][0] > 0,
        no_mean_extra_losses_in_any_condition=all(c["means"]["extra_lost"] <= 0 for c in conditions.values()))


def saved_tail_pairs(root, study):
    """Read local public-array artifacts, not neural models or new forecasts."""
    pairs = []
    files = list((root / "training-tails").glob("*.pkl.gz"))
    expected_files = {_ident(w, "plain_h8") + ".pkl.gz"
        for b in range(5) for w in worlds(study, "reference", b)}
    if {p.name for p in files} != expected_files:
        raise ValueError("missing or extra stored paired-tail files")
    for b in range(5):
        for world in worlds(study, "reference", b):
            path = root / "training-tails" / (_ident(world, "plain_h8") + ".pkl.gz")
            with gzip.open(path, "rb") as handle:
                records = pickle.load(handle)
            rows = {}
            expected = {(policy, epoch, candidate_index(world["index"], epoch), q)
                for policy in ("adaptive", "frozen_mpc")
                for epoch in capture_epochs(world["index"]) for q in (.1, .5, .9)}
            for r in records:
                key = (r["label_policy"], r["decision_epoch"], r["candidate"], r["quantile"])
                if key not in expected or key in rows:
                    raise ValueError("wrong or duplicate stored tail identity")
                start = r["decision_epoch"] + 8
                costs = np.asarray(r["costs"])
                wanted_sha = study["initial_models"]["sha256"][b] if key[0] == "frozen_mpc" else None
                if (r["epochs"] != list(range(start, 65)) or r["start_epoch"] != start
                        or costs.dtype != np.dtype("float64") or costs.shape != (64 - start,)
                        or not np.isfinite(costs).all() or (costs < 0).any()
                        or not math.isfinite(r["prefix_cost"]) or r["prefix_cost"] < 0
                        or r["continuation_sha256"] != wanted_sha):
                    raise ValueError("invalid stored policy-tail costs or provenance")
                rows[key] = r
            if set(rows) != expected:
                raise ValueError("incomplete stored policy-tail pairs")
            for key, rule in rows.items():
                if key[0] != "adaptive":
                    continue
                policy = rows[("frozen_mpc", *key[1:])]
                if (rule["prefix_cost"] != policy["prefix_cost"]
                        or not np.array_equal(rule["features"][0], policy["features"][0])):
                    raise ValueError("policy tails do not share the saved endpoint")
                a, p = math.fsum(rule["costs"]), math.fsum(policy["costs"])
                pairs.append(dict(block=b, world_index=world["index"], condition=world["condition"],
                    decision_epoch=key[1], candidate=key[2], quantile=key[3],
                    adaptive_cost=a, frozen_parent_mpc_cost=p, savings=a-p))
    return dict(pairs=pairs, pair_count=len(pairs), source="saved_public_model_training_tails",
        native_counterfactual_ground_truth=False, independent_test=False,
        all_candidate_ranking_available=False, new_scientific_calls=0)


def run_analysis(payload, study):
    """Read and return a JSON-serializable report; never write or launch work.

    Optional analysis_inputs.hardware is passed through as reported hardware,
    not probed. Limits are labeled planned, not actual measurements. Missing
    progress/tape/latency evidence is exposed in the report, never fabricated.
    """
    root = Path(payload).resolve()
    contract = numeric_contract(study)
    readout = study["readout"]
    if (readout["candidate"] != "policy_tail_td" or readout["primary_condition"] != 1
            or readout["primary_comparators"] != ["plain_h8", "existing_frozen", "adaptive_tail_td"]
            or readout["target_comparator"] != "policy_tail_mc" or readout["compute_reference"] != "plain_h16"
            or type(readout["bootstrap_resamples"]) is not int or readout["bootstrap_resamples"] != 2000
            or study["design"]["reference_behavior"] != "plain_h8"
            or study["value"]["evaluation_updates"] != 0 or readout["test_optimizer_steps"] != 0):
        raise ValueError("readout differs from the fixed six-role comparison")
    expected = list(expected_trajectories(study))
    names = {ident for _, _, ident in expected}
    if ({p.stem for p in (root / "summaries").glob("*.json")} != names
            or {p.name.removesuffix(".jsonl.gz") for p in (root / "raw").glob("*.jsonl.gz")} != names):
        raise ValueError("missing or extra planner-tail trajectories")
    if (root / "tapes").exists() and {p.stem for p in (root / "tapes").glob("*.json")} != {
            _ident(w, "exogenous") for w, _, _ in expected}:
        raise ValueError("missing or extra saved tapes")
    ancestors, finals, models = _model_evidence(root, study)
    progress = _progress_evidence(root, expected, ancestors, finals)
    rows = {}
    for world, role, ident in expected:
        row = read_saved_trajectory(root, world, role, ident)
        phase, b, c, j = (world[k] for k in ("phase", "block", "condition", "replicate"))
        wanted = (ancestors[b] if role == "existing_frozen" else finals.get((b, role))) if phase == "evaluation" else None
        if row["model_seal_sha256"] != wanted:
            raise ValueError("wrong model seal on trajectory: " + ident)
        rows[(phase, b, c, j, role)] = row
    for b in range(5):
        for world in worlds(study, "evaluation", b):
            group = [rows[("evaluation", b, world["condition"], world["replicate"], r)] for r in EVAL_ROLES]
            if any(len({r[k] for r in group}) != 1 for k in ("tape_sha256", "cohort_sha256", "enrolled")):
                raise ValueError("unpaired evaluation tape or patient cohort")

    rng, contrasts = np.random.default_rng(study["streams"]["bootstrap_seed"]), {}
    ordered = (readout["candidate"],) + tuple(r for r in EVAL_ROLES if r != readout["candidate"])
    for candidate, reference in combinations(ordered, 2):
        conditions = {}
        for c in range(3):
            pairs = [_pair(rows[("evaluation", b, c, j, candidate)], rows[("evaluation", b, c, j, reference)])
                     for b in range(5) for j in range(4)]
            conditions[str(c)] = _summarize(pairs, rng, readout["bootstrap_resamples"])
            conditions[str(c)].update(candidate=candidate, reference=reference)
        contrasts[candidate + "_vs_" + reference] = conditions
    primary = {r: _criteria(contrasts[readout["candidate"] + "_vs_" + r]) for r in readout["primary_comparators"]}
    latency = {}
    for phase in ("reference", "evaluation"):
        latency[phase] = {}
        for role in ("plain_h8",) if phase == "reference" else EVAL_ROLES:
            group = [r for r in rows.values() if r["world"]["phase"] == phase and r["role"] == role]
            latency[phase][role] = _latency([v for r in group for v in r["decision_wall_seconds"]], 48 * len(group))
    return dict(format="capacity-policy-tail-readout-v1", complete=True,
        completion_scope="raw_trajectory_matrix_and_model_bindings",
        trajectory_count=len(rows), raw_rows=64 * len(rows), reference_trajectories=120,
        evaluation_trajectories=360, paired_evaluation_worlds=60, trajectories=list(rows.values()),
        contrasts=contrasts, primary_criteria=primary,
        strong_baseline_development_signal=all(all(c.values()) for c in primary.values()),
        secondary_contrasts={"td_vs_mc": "policy_tail_td_vs_policy_tail_mc",
                             "td_vs_h16": "policy_tail_td_vs_plain_h16"},
        model_evidence=models, progress_evidence=progress,
        saved_tail_diagnostics=saved_tail_pairs(root, study),
        tape_artifacts_verified=all(r["tape_artifact_verified"] for r in rows.values()),
        decision_latency=latency, hardware=copy.deepcopy(study.get("analysis_inputs", {}).get("hardware")),
        compute=dict(planned_caps=contract["limits"], limits_are_not_measurements=True,
                     new_updates_from_seal_metadata=15 * 768, historical_ancestor_updates=5 * 1536,
                     recorded_progress_counts=progress["recorded_counts"], independent_budget_audit=False),
        interpretation="development_screen_not_multiplicity_adjusted_confirmation",
        reused_historical_training_blocks=True, new_independent_training_replication=False,
        public_forecast_returns_are_native_ground_truth=False, clinical_safety_established=False,
        deployment_online_adaptation=False, isolated_edge_effect=False,
        formal_confirmation=False, automatic_follow_on=False,
        fixed_parent_continuation_not_exact_updated_policy=True,
        sparse_candidate_collection_not_previous_96_tail_recipe=True)
