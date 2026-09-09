"""Read-only, standard-library audit; never imports or runs the simulator."""

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
CONFIG = "experiments/configs/intertemporal_residual_allocation_headroom_user_20260908.json"
CONFIG_HASH = "3469fd859561995da5e3e0bfaead28440c04df859e9e7a97403b4936c4397f4a"
EXECUTION_COMMIT = "400d64f764b4e96a0d93d0c4a2721c4f68b2f08b"
BASE = "baseline_graph_forecast"
METRICS = ("total_cost", "patients_lost", "completion_service_level")


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name):
    check(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), name)


def main():
    config = read(CONFIG)
    check(digest(CONFIG) == CONFIG_HASH, "execution config changed")
    out = Path(config["output_root"])
    status, claim, summary, inventory = [
        read(out / name) for name in (
            "status.json", "claim.json", "summary.json", "artifact_inventory.json"
        )
    ]
    check(status["status"] == "completed" and status["exit_code"] == 0, "terminal status")
    check((summary["state_count"], summary["candidate_count"], summary["discovery_row_count"], summary["validation_row_count"]) == (54, 81, 8748, 21870), "summary cardinalities")
    check(claim["execution_commit"] == EXECUTION_COMMIT, "execution commit")
    check(claim["config_sha256"] == CONFIG_HASH, "claim config hash")
    for item in (status, claim, summary):
        check(item["execution_approvals"] == {"howard": False, "zhaowei": True}, "approval record")
        check(not item["policy_training_performed"], "training performed")
        check(not item["formal_confirmation_performed"], "formal confirmation performed")
    check(not summary["observable_ranking_screen_authorized"], "downstream authorization")
    check(not summary["ddpg_training_authorized"], "DDPG authorization")
    for entry in inventory["files"]:
        check(digest(entry["path"]) == entry["sha256"], "artifact hash: " + entry["path"])
        if not Path(entry["path"]).is_relative_to(out):
            locked = subprocess.check_output(
                ["git", "show", EXECUTION_COMMIT + ":" + entry["path"]], cwd=ROOT
            )
            check(hashlib.sha256(locked).hexdigest() == entry["sha256"], "commit source hash")
    original = read(config["execution_exception"]["original_config"])
    metadata = {"name", "experimental_role", "execution_authorization", "execution_exception", "output_root"}
    check({k: v for k, v in config.items() if k not in metadata}
          == {k: v for k, v in original.items() if k not in metadata}, "scientific configuration changed")
    check(not (ROOT / original["output_root"]).exists(), "original output root was consumed")

    states = read(out / "state_manifest.json")["states"]
    scenarios = {"scheduled_rotation_forward", "scheduled_rotation_reverse", "scheduled_rotation_slower_noisy"}
    expected_states = {(scenario, seed, epoch) for scenario in scenarios
                       for seed in range(99700000, 99700003) for epoch in (4, 8, 12, 16, 24, 32)}
    check(len(states) == 54, "state count")
    check({(s["scenario"], s["generation_seed"], s["epoch"]) for s in states} == expected_states, "state coverage")
    check({s["state_index"] for s in states} == set(range(54)), "state indices")
    check(len({s["state_id"] for s in states}) == 54, "duplicate state IDs")
    check(len({s["observation_sha256"] for s in states}) == 54, "duplicate observations")
    state_map = {s["state_index"]: s for s in states}
    candidates = {BASE: (-1, "baseline", 0.0)}
    candidates.update({f"f{facility:02d}_{direction}_{fraction:.2f}": (facility, direction, fraction)
                       for facility in range(20) for direction in ("decrease", "increase")
                       for fraction in (0.05, 0.10)})
    phase_specs = {"discovery": (99800000, 2), "validation": (99900000, 5)}
    expected_keys = {(phase, i, candidate, start + i * count + rep)
                     for phase, (start, count) in phase_specs.items() for i in range(54)
                     for candidate in candidates for rep in range(count)}
    with (ROOT / out / "residual_headroom_rows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    check(len(rows) == status["persisted_rows"] == 30618, "row count")
    seen, grouped, worlds, mechanics = set(), defaultdict(list), defaultdict(list), {}
    text_fields = {"phase", "state_id", "scenario", "candidate", "direction", "allocation_signature", "rng_sha256"}
    mechanical_fields = ("realized_transfer_units", "residual_linf", "changed_facilities", "budget_error",
                         "request_total", "shared_budget", "allocation_signature", "requested_transfer_units", "reported_moved_units")
    for row in rows:
        for field, value in row.items():
            if field not in text_fields:
                check(math.isfinite(float(value)), "nonfinite CSV: " + field)
        check(all(math.isfinite(float(v)) for v in row["allocation_signature"].split(",")), "allocation nonfinite")
        phase, i, candidate, world = row["phase"], int(row["state_index"]), row["candidate"], int(row["world_seed"])
        key = (phase, i, candidate, world)
        check(key not in seen, "duplicate row")
        seen.add(key)
        check(key in expected_keys, "unexpected row or CRN")
        state = state_map[i]
        for field in ("state_id", "scenario", "generation_seed", "epoch"):
            check(row[field] == str(state[field]), "row-state mismatch: " + field)
        check((int(row["facility"]), row["direction"], float(row["transfer_budget_fraction"])) == candidates[candidate], "option mismatch")
        grouped[(phase, i, candidate)].append(row)
        worlds[(phase, i, world)].append(row)
        signature = tuple(row[field] for field in mechanical_fields)
        check(mechanics.setdefault((i, candidate), signature) == signature, "mechanics depend on world seed")
        check(abs(float(row["budget_error"])) <= 1e-8, "budget error")
    check(seen == expected_keys, "missing rows")
    for world_rows in worlds.values():
        check(len(world_rows) == 81 and len({r["rng_sha256"] for r in world_rows}) == 1, "paired RNG mismatch")
    means = {key: {field: mean(float(r[field]) for r in values) for field in METRICS}
             for key, values in grouped.items()}
    choice = {phase: {i: min(candidates, key=lambda c: (means[(phase, i, c)]["total_cost"], c))
                      for i in range(54)} for phase in phase_specs}
    comparisons = {}
    for label, phase in (("prospective_validation", "discovery"), ("optimistic_validation_oracle", "validation")):
        state_results = []
        recorded = {r["state_index"]: r for r in summary[label]["state_results"]}
        check(set(recorded) == set(range(54)), "summary state coverage")
        for i in range(54):
            b, s = means[("validation", i, BASE)], means[("validation", i, choice[phase][i])]
            saving = (b["total_cost"] - s["total_cost"]) / b["total_cost"]
            clinical = s["patients_lost"] <= b["patients_lost"] and s["completion_service_level"] + 1e-12 >= b["completion_service_level"]
            material = saving >= 0.005 and clinical
            check(recorded[i]["candidate"] == choice[phase][i], "summary selection")
            close(saving, recorded[i]["relative_saving"], "state saving")
            check(material == recorded[i]["material"] and clinical == recorded[i]["clinical_noninferior"], "state clinical gate")
            state_results.append((i, b, s, saving, clinical, material))
        base_sum = sum(r[1]["total_cost"] for r in state_results)
        saving = (base_sum - sum(r[2]["total_cost"] for r in state_results)) / base_sum
        material_count = sum(r[5] for r in state_results)
        clinical = (sum(r[2]["patients_lost"] for r in state_results) <= sum(r[1]["patients_lost"] for r in state_results)
                    and mean(r[2]["completion_service_level"] for r in state_results) + 1e-12 >= mean(r[1]["completion_service_level"] for r in state_results))
        close(saving, summary[label]["relative_saving"], "aggregate saving")
        close(material_count / 54, summary[label]["material_state_fraction"], "material fraction")
        check(clinical == summary[label]["clinical_noninferior"], "aggregate clinical gate")
        scenario_savings = {}
        for scenario in sorted(scenarios):
            subset = [r for r in state_results if state_map[r[0]]["scenario"] == scenario]
            btotal = sum(r[1]["total_cost"] for r in subset)
            scenario_savings[scenario] = (btotal - sum(r[2]["total_cost"] for r in subset)) / btotal
            close(scenario_savings[scenario], summary[label]["by_scenario"][scenario]["relative_saving"], "scenario saving")
        check(all(v > 0 for v in scenario_savings.values()) == summary[label]["positive_every_scenario"], "scenario positivity")
        comparisons[label] = {"relative_saving": saving, "material_state_count": material_count,
                              "positive_cost_saving_states": sum(r[3] > 0 for r in state_results),
                              "maximum_state_relative_saving": max(r[3] for r in state_results),
                              "clinical_noninferior": clinical, "by_scenario": scenario_savings}

    eligible = wins = material_pairs = signs = exact = positive_material_pairs = 0
    for i in range(54):
        dbase, vbase = (means[(p, i, BASE)]["total_cost"] for p in ("discovery", "validation"))
        selected = choice["discovery"][i]
        exact += selected == choice["validation"][i]
        if (dbase - means[("discovery", i, selected)]["total_cost"]) / dbase >= 0.005:
            eligible += 1
            wins += means[("validation", i, selected)]["total_cost"] < vbase
        for c in candidates:
            if c == BASE:
                continue
            d = (dbase - means[("discovery", i, c)]["total_cost"]) / dbase
            v = (vbase - means[("validation", i, c)]["total_cost"]) / vbase
            if abs(d) >= 0.005:
                material_pairs += 1
                positive_material_pairs += d > 0
                signs += d * v > 0
    label = summary["label_stability"]
    check((eligible, wins, material_pairs, signs) == (label["discovery_material_selected_states"], label["selected_validation_wins"], label["discovery_material_candidate_pairs"], label["pairwise_material_sign_matches"]), "stability counts")
    close(exact / 54, label["exact_best_action_agreement"], "exact choice agreement")
    close(wins / eligible if eligible else 0, label["selected_validation_win_fraction"], "selected win fraction")
    close(signs / material_pairs if material_pairs else 0, label["pairwise_material_sign_agreement"], "pairwise sign fraction")
    residuals = [(key, values) for key, values in mechanics.items() if key[1] != BASE]
    noncollapsed = [(key, values) for key, values in residuals if float(values[0]) > 1e-8]
    coverage = {(candidates[key[1]][0], candidates[key[1]][1]) for key, _ in noncollapsed}
    max_error = max(abs(float(values[3])) for _, values in residuals)
    m = summary["mechanics"]
    check((len(residuals), len(noncollapsed), len(coverage)) == (m["candidate_state_count"], m["noncollapsed_candidate_state_count"], m["covered_facility_directions"]), "mechanics counts")
    close(len(noncollapsed) / len(residuals), m["noncollapsed_candidate_fraction"], "mechanics fraction")
    close(max_error, m["maximum_budget_error"], "maximum budget error")
    r0 = len(noncollapsed) / len(residuals) >= 0.90 and len(coverage) == 40 and max_error <= 1e-8
    o, p = comparisons["optimistic_validation_oracle"], comparisons["prospective_validation"]
    r1 = o["relative_saving"] >= 0.005 and o["material_state_count"] / 54 >= 0.30 and o["clinical_noninferior"]
    r2 = (p["relative_saving"] >= 0.005 and p["material_state_count"] / 54 >= 0.30 and p["clinical_noninferior"]
          and all(v > 0 for v in p["by_scenario"].values()) and eligible > 0 and wins / eligible >= 0.70
          and material_pairs > 0 and signs / material_pairs >= 0.80)
    check((r0, r1, r2) == (m["passes"], summary["optimistic_gate_passes"], summary["prospective_gate_passes"]), "gate recomputation")
    check(summary["passes"] == (r0 and r1 and r2), "overall gate")
    check(summary["observable_ranking_screen_eligible"] == summary["passes"], "ranking eligibility")
    decision = "budget_neutral_residual_support_established" if summary["passes"] else "budget_neutral_residual_support_failed"
    check(summary["decision"] == status["decision"] == decision, "decision mismatch")
    check((ROOT / "results/residual_screen_20260908.stderr.log").stat().st_size == 0, "nonempty stderr")
    elapsed_wall = (datetime.fromisoformat(status["updated_at"]) - datetime.fromisoformat(status["started_at"])).total_seconds()
    print(json.dumps({"audit_passes": True, "execution_commit": EXECUTION_COMMIT,
                      "inventory_hashes_verified": len(inventory["files"]), "states": len(states),
                      "unique_rows": len(seen), "phase_rows": dict(Counter(r["phase"] for r in rows)),
                      "paired_worlds": len(worlds), "comparisons": comparisons,
                      "discovery_material_selections": eligible, "material_pairs": material_pairs,
                      "positive_material_pairs": positive_material_pairs, "exact_action_matches": exact,
                      "mechanics": m, "gates": {"R0": r0, "R1": r1, "R2": r2},
                      "decision": decision, "wall_timestamp_seconds": elapsed_wall,
                      "reported_monotonic_seconds": status["elapsed_seconds"],
                      "timing_discrepancy_seconds": elapsed_wall - status["elapsed_seconds"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
