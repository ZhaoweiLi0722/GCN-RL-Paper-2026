"""Saved JSON-only terminal cohort accounting. No simulator/model imports or calls."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

from src.rl.candidate_pilot_verification import count, identity_counts, json_hash


ACTIVE = ("waiting", "in_production", "in_transit", "finished")
STAGES = ("waiting", "production", "specimen_transit", "finished_return")
BASELINE = "own_frozen"
IMMUTABLE_PATIENT = ("patient_id", "specimen_id", "enrollment_epoch",
                     "collection_facility", "risk_type", "risk_multiplier",
                     "health_index", "deterioration_epoch")


def summarize_state(state, outcome, horizon):
    """Age is recorded active-patient age, not inferred time to resolution."""
    if state["scalars"]["t"] != horizon or outcome["steps"] != horizon:
        raise ValueError("wrong horizon")
    if json_hash(state) != outcome["final_state_sha256"]:
        raise ValueError("state/outcome binding differs")
    identities = identity_counts(state)
    if identities != outcome["terminal_compartments"]:
        raise ValueError("terminal compartment accounting differs")
    if identities["active"] != outcome["terminal_active"]:
        raise ValueError("active count differs")
    costs = outcome["components"]
    if (not math.isfinite(outcome["cost"]) or any(not math.isfinite(v) for v in costs.values())
            or not math.isclose(math.fsum(costs.values()), outcome["cost"], rel_tol=1e-12, abs_tol=1e-6)):
        raise ValueError("invalid cost accounting")
    cohorts, age, risk, statuses = {}, Counter(), Counter(), Counter()
    enrollment = []
    last_arrivals = 0
    for pid, p in state["patients"].items():
        epoch = count(p["enrollment_epoch"])
        if epoch >= horizon:
            raise ValueError("enrollment outside fixed window")
        status = p["status"]
        if status not in ACTIVE + ("lost", "delivered"):
            raise ValueError("unknown patient status")
        statuses[status] += 1
        key = str(epoch)
        bucket = cohorts.setdefault(key, {s: 0 for s in ACTIVE + ("lost", "delivered")})
        bucket[status] += 1
        enrollment.append([pid] + [p[k] for k in IMMUTABLE_PATIENT])
        if status in ACTIVE:
            age[str(count(p["age"]))] += 1
            risk[str(count(p["risk_type"]))] += 1
        if epoch == horizon - 1:
            last_arrivals += 1
            if status != "waiting" or p["age"] != 0:
                raise ValueError("final-step arrivals received unexpected processing")
    if last_arrivals != sum(count(v) for v in outcome["arrivals"][-1]):
        raise ValueError("last arrivals do not match enrollment records")
    active = identities["active"]
    if sum(age.values()) != active or active + identities["lost"] + identities["delivered"] != identities["enrolled"]:
        raise ValueError("patient conservation differs")
    return {
        "block": outcome["block"], "role": outcome["role"], "world": outcome["world_index"],
        "seed": outcome["seed"], "initial_state_sha256": outcome["initial_state_sha256"],
        "final_state_sha256": outcome["final_state_sha256"],
        "enrollment_signature": json_hash(sorted(enrollment)),
        "patient_registry_sha256": json_hash(state["patients"]),
        "cost": outcome["cost"], "components": costs, **identities,
        "final_step_new_waiting": last_arrivals,
        "older_active": active - last_arrivals,
        "active_age_histogram": dict(age), "active_risk_histogram": dict(risk),
        "enrollment_cohorts": cohorts,
    }


def compare(left, right):
    if (left["block"], left["world"], left["seed"], left["initial_state_sha256"]) != (
            right["block"], right["world"], right["seed"], right["initial_state_sha256"]):
        raise ValueError("paired starting world differs")
    fields = ("cost", "enrolled", "delivered", "lost", "active", "older_active", "final_step_new_waiting") + STAGES
    difference = {f: left[f] - right[f] for f in fields}
    if difference["enrolled"] != difference["delivered"] + difference["lost"] + difference["active"]:
        raise ValueError("paired conservation differs")
    return {"block": left["block"], "world": left["world"], "differences": difference,
            "same_full_final_state": left["final_state_sha256"] == right["final_state_sha256"],
            "same_patient_registry": left["patient_registry_sha256"] == right["patient_registry_sha256"],
            "same_enrollment_attributes": left["enrollment_signature"] == right["enrollment_signature"]}


def aggregate(records, fields):
    if not records:
        raise ValueError("empty aggregation")
    return {f: math.fsum(r[f] for r in records) / len(records) for f in fields}


def analyze(config, repo):
    root = repo / config["result_root"]
    closure = json.loads((repo / config["closure_index"]).read_text())
    if closure["terminal_status"] != "completed" or closure["exec_exit_code"] != 0:
        raise ValueError("only closed completed evidence admitted")
    consumed = {}

    def read(relative, expected=None):
        p = root / relative
        raw = p.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("saved input hash differs: " + relative)
        consumed[relative] = digest
        return json.loads(raw)

    receipt = read("launcher/archive-receipt.json", closure["receipt_sha256"]["launcher/archive-receipt.json"])
    if not receipt["local_archive_verified"]:
        raise ValueError("missing original archive verification")
    locked_files = receipt["archive"]["files"]
    episodes, lookup = [], {}
    for block in config["blocks"]:
        for role in config["roles"]:
            for world in range(config["worlds"]):
                base = f"episodes/evaluation/block{block}/{role}/world{world:02d}"
                paths = [f"{base}/{name}.json" for name in ("final_state", "outcome")]
                state, outcome = [read("payload/" + p, locked_files[p]) for p in paths]
                if (outcome["block"], outcome["role"], outcome["world_index"], outcome["split"]) != (block, role, world, "test"):
                    raise ValueError("wrong episode metadata")
                row = summarize_state(state, outcome, config["horizon"])
                episodes.append(row)
                lookup[block, role, world] = row
    fields = ("cost", "enrolled", "delivered", "lost", "active", "final_step_new_waiting", "older_active") + STAGES
    means = {role: aggregate([r for r in episodes if r["role"] == role], fields) for role in config["roles"]}
    contrasts = {}
    for role in config["roles"]:
        if role == BASELINE:
            continue
        pairs = [compare(lookup[b, role, w], lookup[b, BASELINE, w])
                 for b in config["blocks"] for w in range(config["worlds"])]
        contrasts[role + "_minus_own_frozen"] = {
            "pairs": pairs,
            "mean_differences": aggregate([p["differences"] for p in pairs], fields),
            "full_state_equal_pairs": sum(p["same_full_final_state"] for p in pairs),
            "patient_registry_equal_pairs": sum(p["same_patient_registry"] for p in pairs),
            "enrollment_attribute_equal_pairs": sum(p["same_enrollment_attributes"] for p in pairs),
        }
    # Descriptive histogram preserves every age/cohort; no outcome-selected cutoff.
    histograms = {}
    for role in config["roles"]:
        rows = [r for r in episodes if r["role"] == role]
        histograms[role] = {}
        for field in ("active_age_histogram", "active_risk_histogram"):
            h = Counter()
            for row in rows:
                h.update(row[field])
            histograms[role][field] = dict(sorted(h.items(), key=lambda x: int(x[0])))
    # All old sources stay byte-identical after the diagnostic read.
    for p, digest in consumed.items():
        if hashlib.sha256((root / p).read_bytes()).hexdigest() != digest:
            raise ValueError("input changed while reading: " + p)
    return {"format": "saved-terminal-obligation-diagnostic-v1", "post_hoc_development_only": True,
            "config": config, "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "source_sha256": consumed, "episodes": episodes, "means": means,
            "contrasts": contrasts, "histograms": histograms,
            "clinical_mortality_probabilities_estimated": False,
            "reward_coefficients_selected": False, "new_scientific_calls": 0,
            "data_role": "Previously observed development-test evidence, not independent confirmation or reward-training data"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("new report output required; old evidence never overwritten")
    report = analyze(config, repo)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"episodes": len(report["episodes"]), "source_files": len(report["source_sha256"]), "output": str(output)}))


if __name__ == "__main__":
    main()
