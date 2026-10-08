"""Read-only posthoc fixed-support supplement; never run a scientific backend."""

import copy
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from src.rl.capacity_family_analysis import COST_KEYS, pair_readout, reconcile_raw


FIXED_ROLE = "mdl2-fixed2"
OLD_ROLES = ("plain_h8", "value_td-graph-final", "sac-graph-final")
ROLES = (FIXED_ROLE,) + OLD_ROLES
PHASE = "confirmation_evaluation"
BLOCKS = 5
RESAMPLES = 2000
# The original family-selection bootstrap stream; contrast-specific seeds keep
# adding a comparator from advancing any previously reported pair's RNG stream.
BOOTSTRAP_SEED = 67190001
METRICS = ("cost", "lost", "elapsed", "requested_hours", "committed_hours", "applied_hours")
DELAYS = {"epoch_mean_reported_wait": "average_waiting_time",
          "epoch_mean_reported_turnaround": "average_turnaround_time"}


def _read(path):
    return json.loads(path.read_text())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-6)


def _key(row):
    w = row["world"]
    return w["block"], w["condition"], w["replicate"], row["role"]


def _index(rows, roles):
    expected = set(itertools.product(range(BLOCKS), range(3), range(8), roles))
    keyed = {}
    for row in rows:
        w = row["world"]
        if (w["phase"] != PHASE or type(w["seed"]) is not int
                or any(type(w[k]) is not int for k in ("block", "condition", "replicate"))):
            raise ValueError("invalid confirmation world identity/seed")
        key = _key(row)
        if key in keyed:
            raise ValueError("duplicate world/role")
        keyed[key] = row
    if set(keyed) != expected:
        raise ValueError("missing or extra confirmation world/role")
    return keyed


def _summaries(root, roles, *, new=False):
    paths = (sorted((root / "summaries").glob("*.json")) if new else
             sorted(p for role in roles for p in
                    (root / "summaries").glob(f"{PHASE}-*-{role}.json")))
    rows = [_read(p) for p in paths]
    _index(rows, roles)
    for row in rows:
        if (row["settled"] is not True or row["settlement"]["settled"] is not True
                or row["settlement"]["lost"] != row["lost"]
                or row["optimizer_updates_during_trajectory"] != 0
                or type(row["lost"]) is not int or row["lost"] < 0
                or not math.isfinite(row["cost"]) or row["cost"] <= 0
                or not math.isfinite(row["elapsed"]) or row["elapsed"] < 0
                or not isinstance(row["compute"], dict) or not row["compute"]
                or any(not math.isfinite(v) or v < 0 for v in row["compute"].values())):
            raise ValueError("invalid settled evaluation summary/compute")
        digest = row["tape_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid tape digest")
    return rows, {p.name: _sha(p) for p in paths}


def _old_records(summaries, saved_rows, comparison):
    rows = [r for r in saved_rows
            if r["world"]["phase"] == PHASE and r["role"] in OLD_ROLES]
    keyed = _index(rows, OLD_ROLES)
    prior = _index([r for r in comparison["raw_reconciliation"]
                    if r["world"]["phase"] == PHASE and r["role"] in OLD_ROLES], OLD_ROLES)
    for summary in summaries:
        saved = keyed[_key(summary)]
        if any(k not in saved or saved[k] != v for k, v in summary.items()):
            raise ValueError("saved old readout does not match summary keys")
        if (set(saved["components"]) != set(COST_KEYS)
                or any(not math.isfinite(v) or v < 0 for v in saved["components"].values())
                or not _close(math.fsum(saved["components"].values()), saved["cost"])
                or any(not math.isfinite(saved[k]) or saved[k] < 0 for k in METRICS)
                or len(saved["raw_sha256"]) != 64):
            raise ValueError("invalid saved old cost/hours reconciliation")
        bound = prior[_key(summary)]
        if (saved["raw_sha256"] != bound["raw_sha256"]
                or any(not _close(saved[k], bound[k]) for k in METRICS)
                or any(not _close(saved["components"][k], bound["components"][k]) for k in COST_KEYS)):
            raise ValueError("terminal readout differs from prior raw reconciliation")
    return rows


def _check_fixed_raw(root, rows):
    seen, extra = set(), {}
    for summary in rows:
        path = (root / summary["raw_path"]).resolve()
        if not path.is_relative_to(root / "raw") or path in seen:
            raise ValueError("invalid or reused new raw path")
        seen.add(path)
        counts = summary["compute"]
        if (summary.get("model_seal_sha256") is not None
                or counts.get("neural_forward_module_calls") != 0
                or counts.get("total_optimizer_steps") != 0
                or any(v for k, v in counts.items() if "optimizer" in k or k == "fit_batches")):
            raise ValueError("fixed reference cannot use a model or optimizer")
        patients, terminal, lost = {}, {}, set()
        delays = {k: [] for k in DELAYS}
        n = 0
        with gzip.open(path, "rt") as stream:
            for t, line in enumerate(stream):
                row = json.loads(line)
                n += 1
                if (row["epoch"] != t or row["world"] != summary["world"]
                        or row["role"] != FIXED_ROLE or set(row["components"]) != set(COST_KEYS)
                        or any(not math.isfinite(v) or v < 0 for v in row["components"].values())
                        or not _close(-row["reward"], row["cost"])):
                    raise ValueError("new raw identity/reward/components mismatch")
                receipt = row["info"]["support_public_receipt"]
                for key, field in DELAYS.items():
                    value = row["info"].get(field)
                    if value is not None:
                        if not math.isfinite(value) or value < 0:
                            raise ValueError("invalid recorded delay")
                        delays[key].append(value)
                action = [2.] * 4 if t < 48 else [0.] * 4
                applied = [2.] * 4 if 2 <= t < 50 else [0.] * 4
                for actual, expected in ((row["requested_hours"], action),
                        (row["executed_hours"], action), (receipt["raw_requested_hours"], action),
                        (receipt["committed_hours"], action), (receipt["applied_hours"], applied),
                        (receipt["ordinary_hours"], [4.] * 4)):
                    if np.shape(actual) != (4,) or not np.allclose(actual, expected, rtol=0., atol=2e-6):
                        raise ValueError("fixed support/ordinary hours/two-epoch lead mismatch")
                if receipt["epoch"] != t or receipt["known_at"] != t + 1:
                    raise ValueError("support receipt epoch mismatch")
                current = {p["patient_id"]: p for p in row["patient_records"]}
                if len(current) != len(row["patient_records"]) or not patients.keys() <= current.keys():
                    raise ValueError("duplicate or disappearing patient")
                for ident, p in current.items():
                    identity = (p["enrollment_epoch"], p["collection_site"])
                    if ident in patients and patients[ident] != identity:
                        raise ValueError("patient identity changed")
                    if ident in terminal and terminal[ident] != p["status"]:
                        raise ValueError("terminal patient reopened")
                    patients[ident] = identity
                    if p["status"] in ("lost", "delivered"):
                        terminal[ident] = p["status"]
                now_lost = {i for i, p in current.items() if p["status"] == "lost"}
                if (not lost <= now_lost or type(row["new_lost_patients"]) is not int
                        or row["new_lost_patients"] != len(now_lost - lost)):
                    raise ValueError("patient loss events mismatch")
                lost = now_lost
        settlement = summary["settlement"]
        if (n != 64 or len(terminal) != len(patients) or len(lost) != summary["lost"]
                or settlement["enrolled"] != len(patients)
                or settlement["delivered"] != len(patients) - len(lost)):
            raise ValueError("incomplete raw trajectory/patient settlement")
        extra[_key(summary)] = {k: float(np.mean(v)) if len(v) == 64 else None
                                for k, v in delays.items()}
        extra[_key(summary)]["delay_observed_epochs"] = {k: len(v) for k, v in delays.items()}
    if {p.resolve() for p in (root / "raw").glob("*.jsonl.gz")} != seen:
        raise ValueError("missing or extra new raw trajectory")
    return extra


def _contrast(rows, left, right):
    seed = int.from_bytes(hashlib.sha256(
        f"{BOOTSTRAP_SEED}:{BLOCKS}:{left}:{right}".encode()).digest()[:8], "big")
    return pair_readout(rows, left, right, BLOCKS, np.random.default_rng(seed), RESAMPLES)


def _preserved_pairs(comparison, rows):
    """Keep original primary H8 orientations and their original CI draws."""
    wanted = ((OLD_ROLES[1], "plain_h8"), (OLD_ROLES[2], "plain_h8"),
              ("sac-graph-final", "value_td-graph-final"))
    screens = [s["result"] for s in comparison["screens"]]
    output = []
    for left, right in wanted:
        candidates = screens if right == "plain_h8" else comparison["pairs"]
        saved = [p for p in candidates if (p["candidate"], p["reference"]) == (left, right)]
        if len(saved) != 1:
            raise ValueError("missing or duplicate prior pair readout")
        # Reproduce the original per-contrast seed, cross-check the bound saved
        # records, and return the original object so historical numbers do not churn.
        reproduced = _contrast(rows, left, right)
        for key in reproduced:
            if key in ("candidate", "reference"):
                continue
            if key == "metrics":
                for metric, values in reproduced[key].items():
                    for side, value in values.items():
                        if not _close(value, saved[0][key][metric][side]):
                            raise ValueError("prior pair metrics differ from bound summaries")
            elif key == "conditions":
                for actual, expected in zip(reproduced[key], saved[0][key]):
                    if actual.keys() != expected.keys() or any(
                            not np.allclose(actual[k], expected[k], rtol=1e-12, atol=1e-6)
                            for k in actual):
                        raise ValueError("prior pair condition readout mismatch")
                if len(saved[0][key]) != 3:
                    raise ValueError("prior pair missing conditions")
            elif not np.allclose(reproduced[key], saved[0][key], rtol=1e-12, atol=1e-6):
                raise ValueError("prior pair readout differs from original bootstrap")
        output.append(copy.deepcopy(saved[0]))
    return output


def _aggregate(rows):
    compute_keys = set(rows[0]["compute"])
    if any(set(r["compute"]) != compute_keys for r in rows):
        raise ValueError("inconsistent recorded compute fields within role")
    return dict(worlds=len(rows),
        **{key: float(np.mean([r[key] for r in rows])) for key in METRICS},
        components={k: float(np.mean([r["components"][k] for r in rows])) for k in COST_KEYS},
        compute_total={k: math.fsum(r["compute"][k] for r in rows) for k in sorted(compute_keys)},
        delay_metrics={k: dict(mean=float(np.mean([r[k] for r in rows]))
                              if all(r.get(k) is not None for r in rows) else None,
                              recorded_worlds=sum(r.get(k) is not None for r in rows)) for k in DELAYS})


def _harm(pair):
    costs, patients = np.asarray(pair["world_savings"]), np.asarray(pair["world_extra_losses"])
    return dict(candidate=pair["candidate"], reference=pair["reference"],
        cost_harmed_worlds=int((costs < 0).sum()), patient_harmed_worlds=int((patients > 0).sum()),
        blocks=[dict(block=b, cost_saving=float(costs[b].mean()),
                     extra_patient_losses=float(patients[b].mean()),
                     cost_harmed_worlds=int((costs[b] < 0).sum()),
                     patient_harmed_worlds=int((patients[b] > 0).sum())) for b in range(BLOCKS)],
        block_conditions=[dict(block=b, condition=c, cost_saving=float(costs[b, c].mean()),
                     extra_patient_losses=float(patients[b, c].mean()),
                     cost_harmed_worlds=int((costs[b, c] < 0).sum()),
                     patient_harmed_worlds=int((patients[b, c] > 0).sum()))
                     for b in range(BLOCKS) for c in range(3)])


def analyze(new_payload, old_payload):
    """Return four role rows and six paired readouts, without writing any files.

    Both inputs are payload directories. Only the new 120 raw ledgers are read;
    Old metrics reuse the independently verified terminal world readout (or
    comparison.json's raw reconciliation), bound to all 360 saved summaries.
    """
    new, old = Path(new_payload).resolve(), Path(old_payload).resolve()
    new_summaries, new_hashes = _summaries(new, (FIXED_ROLE,), new=True)
    old_summaries, old_hashes = _summaries(old, OLD_ROLES)
    comparison_path = old / "comparison.json"
    comparison = _read(comparison_path)
    source = old.parent / "terminal-readout/world-readout.json"
    if source.is_file():
        saved_rows = _read(source)
    else:
        source, saved_rows = comparison_path, comparison["raw_reconciliation"]
    old_rows = _old_records(old_summaries, saved_rows, comparison)
    keyed = _index(new_summaries + old_rows, ROLES)
    for b, c, j in itertools.product(range(BLOCKS), range(3), range(8)):
        group = [keyed[b, c, j, role] for role in ROLES]
        if any(r["world"] != group[0]["world"] or r["tape_sha256"] != group[0]["tape_sha256"]
               for r in group[1:]):
            raise ValueError("exact world/seed/tape pairing mismatch")
    extra = _check_fixed_raw(new, new_summaries)
    new_rows = reconcile_raw(new, new_summaries)
    for row in new_rows:
        row.update(extra[_key(row)])
    rows = new_rows + old_rows
    pairs = [_contrast(rows, role, FIXED_ROLE) for role in OLD_ROLES]
    pairs.extend(_preserved_pairs(comparison, old_rows))
    table = []
    for role in ROLES:
        role_rows = [r for r in rows if r["role"] == role]
        table.append(dict(role=role, **_aggregate(role_rows),
            conditions=[dict(condition=c, **_aggregate([r for r in role_rows if r["world"]["condition"] == c]))
                        for c in range(3)],
            blocks=[dict(block=b, **_aggregate([r for r in role_rows if r["world"]["block"] == b]))
                    for b in range(BLOCKS)]))
    return dict(format="capacity-fixed-reference-readout-v1", complete=True,
        table=table, pairs=pairs, pair_harm=[_harm(p) for p in pairs], raw_reconciliation=rows,
        new_trajectories=120, reused_trajectories=360, paired_worlds=120, new_raw_rows=7680,
        supplementary_primary_reference=FIXED_ROLE, prior_primary_reference="plain_h8",
        prior_primary_preserved=True, prior_pair_readouts_preserved=True,
        posthoc_supplement=True, new_independent_validation=False,
        bootstrap_resamples=RESAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
        shared_block_resampling_across_conditions=True, cross_condition_world_pairing=False,
        synthetic_only=True, formal_multiplicity_adjusted=False, automatic_winner=False,
        automatic_screen=False, graph_attribution=False, rl_attribution=False,
        baseline_assumed_weaker=False, clinical_safety=False, automatic_follow_on=False,
        analysis_scientific_calls=0, old_raw_reverified=False,
        elapsed_definition="whole trajectory including I/O, not per-decision latency",
        delay_definition="mean of epoch-reported metrics, not patient-level mean waiting/turnaround",
        compute_definition="saved evaluation counts only; excludes prior training",
        evidence=dict(old_metrics_source=str(source), old_metrics_sha256=_sha(source),
                      old_pairs_source=str(comparison_path), old_pairs_sha256=_sha(comparison_path),
                      old_summary_sha256=old_hashes, new_summary_sha256=new_hashes,
                      binding="all saved summary keys matched; prior raw verification reused"))
