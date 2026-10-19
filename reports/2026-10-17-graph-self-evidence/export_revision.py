"""Export completed records and provenance only; no scientific runtime imports."""

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREDECESSORS = (
    ("Native Value-TD A", "2026-10-10-native-value-td/reexecution1"),
    ("Original B CEM sequence", "2026-10-11-native-sequence-value-td"),
    ("Five-family development", "2026-10-11-native-learning-development"),
    ("Revised B structured TD", "2026-10-12-structured-native-candidates"),
    ("Candidate-Advantage v1", "2026-10-13-candidate-advantage"),
)


def read(relative, sources):
    data = (ROOT / relative).read_bytes()
    sources.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return json.loads(data)


def publish_json(name, data, check):
    encoded = json.dumps(data, indent=2, sort_keys=True) + "\n"
    path = HERE / name
    if check:
        assert path.read_text() == encoded, name + " differs from saved sources"
    else:
        path.write_text(encoded)


def reverse_display(pair):
    # Ratio-of-means gains require a new denominator when reversing a contrast.
    invert = lambda s: -s / (1 - s)
    lo, hi = pair["saving_fraction_ci95"]
    pl, ph = pair["extra_patient_losses_ci95"]
    return dict(candidate=pair["reference"], reference=pair["candidate"],
                saving_fraction=invert(pair["saving_fraction"]),
                saving_fraction_ci95=[invert(hi), invert(lo)],
                extra_patient_losses=-pair["extra_patient_losses"],
                extra_patient_losses_ci95=[-ph, -pl],
                source_pair=[pair["candidate"], pair["reference"]],
                transformation="Monotone endpoint conversion of saved intervals, no bootstrap")


def main(check):
    sources = []
    old = read("reports/2026-10-17-graph-self-evidence/evidence.json", sources)
    panel = old["panels"]["primary_72_worlds"]
    pairs = {(p["candidate"], p["reference"]): p for p in panel["pairs"]}
    m1 = read("specs/2026-10-15-gcn-self-attribution/frozen.json", sources)
    v2 = read("specs/2026-10-14-graph-local-residual/frozen.json", sources)
    recovered = read("specs/2026-10-15-gcn-self-attribution/parallel-recovery1/frozen.json", sources)
    worlds = [w for w in m1["streams"]["worlds"] if w["phase"] == "evaluation"]
    assert len(worlds) == 72
    assert pairs[("local_graph", "local_self")]["worlds"]
    identity = lambda w: (w["block"], w["condition"], w["replicate"])
    compared_worlds = [x["world"] for x in pairs[("local_graph", "local_self")]["worlds"]]
    assert len({identity(w) for w in compared_worlds}) == 72
    assert sorted(compared_worlds, key=identity) == sorted(worlds, key=identity)
    fresh = set(m1["streams"]["allocations"])
    assert len(fresh) == len(m1["streams"]["allocations"]) == 7634
    assert m1["seed_audit"]["passed"] and not m1["seed_audit"]["collisions"]
    assert m1["created_at_utc"] > v2["created_at_utc"]
    predecessors = []
    for label, spec in PREDECESSORS:
        path = f"specs/{spec}/frozen.json"
        packet = read(path, sources)
        allocations = set(packet["streams"]["allocations"])
        assert not fresh.intersection(allocations), label + " overlaps fresh M1 identities"
        assert packet["created_at_utc"] < v2["created_at_utc"]
        predecessors.append(dict(name=label, packet_path=path,
                                 created_at_utc=packet["created_at_utc"],
                                 packet_sha256=packet["packet_sha256"],
                                 learned_roles=packet["config"]["study"]["learned_roles"],
                                 fresh_m1_allocation_overlap=0))
    assert sum(len(p["learned_roles"]) for p in predecessors) == 14
    assert not fresh.intersection(v2["streams"]["allocations"])
    protocol_path = "specs/2026-10-15-gcn-self-attribution/protocol.md"
    protocol_bytes = (ROOT / protocol_path).read_bytes()
    protocol = dict(path=protocol_path, bytes=len(protocol_bytes),
                    sha256=hashlib.sha256(protocol_bytes).hexdigest())
    assert protocol["bytes"] == 6335
    assert protocol["sha256"] == m1["inputs"][protocol_path]["sha256"]
    # Recovery receipts preserve the original worlds; inspect their nested base packet.
    def has_same_worlds(obj):
        if isinstance(obj, dict):
            if obj.get("streams", {}).get("worlds") == m1["streams"]["worlds"]:
                return True
            return any(has_same_worlds(v) for v in obj.values())
        if isinstance(obj, list):
            return any(has_same_worlds(v) for v in obj)
        return False
    assert has_same_worlds(recovered), "Recovery must bind the same M1 world manifest"
    extras = dict(
        schema="graph-self-review-supplement-v1",
        source_manifest=sources,
        protocol=protocol,
        display_contrasts=[reverse_display(pairs[("local_self", "local_flat")]),
                           pairs[("structured_h8", "local_self")],
                           reverse_display(pairs[("structured_h8", "local_self")]),
                           reverse_display(pairs[("structured_h8", "local_flat")])],
        transit_table=[dict(role=r["role"],
                           specimen_arrivals=r["metrics_mean"]["flows.specimen_transfer_arrivals"],
                           transit_expired=r["metrics_mean"]["flows.patients_lost_transit_expired"])
                       for r in panel["role_summary_table"]],
        provenance=dict(old_graph_recipe_frozen_utc=v2["created_at_utc"],
                        m1_frozen_utc=m1["created_at_utc"],
                        m1_packet_sha256=m1["packet_sha256"],
                        namespace=m1["streams"]["namespace"],
                        fresh_allocation_count=len(fresh),
                        saved_historical_audit_file_count=len(m1["seed_audit"]["files"]),
                        saved_historical_audit_passed=True,
                        saved_historical_manifest_sha256=m1["seed_audit"]["manifest_sha256"],
                        no_overlap_with_predecessors_and_v2=True,
                        recovery_preserves_worlds=True,
                        comparison_uses_exact_declared_worlds=True,
                        evaluation_worlds=worlds,
                        limit="Supports nonreuse within the recorded lineage and saved seed audit; not proof about undocumented external work or independent source accessibility. These are development worlds, not a confirmatory holdout."),
        predecessor_inventory=dict(
            scope="Oct 10-14 Native lineage before Localv2 recipe freeze, not the full July-October project",
            campaign_design_count=5, configured_learned_arm_count=14,
            campaigns=predecessors,
            count_unit="A fixed learner architecture/target/planner arm. Seeds, recoveries and inherited reevaluated controls are not extra recipes.",
            zero_science_failures_counted_as_recipe_variants=False,
            later_diagnostics_counted_as_preselection=False,
            global_project_search_count="Not established; earlier DDPG and other campaigns exist. Do not use 14 as a global search-adjustment count."),
        scientific_runtime_calls=0, new_bootstrap=False, panels_pooled=False)
    publish_json("review-supplement.json", extras, check)

    rep_sources = []
    base = "results/native_graph_self_replication_20261017_eval_recovery1/payload/comparison.json"
    rep = read(base, rep_sources)
    terminal = read("specs/2026-10-17-graph-self-replication/evaluation-recovery1/terminal-readout.json", rep_sources)
    assert rep["complete"] and rep["paired_worlds"] == 120 and len(rep["pairs"]) == 6
    keys = ("pairs", "table", "condition_table", "claims", "initial_matches", "reconciliation",
            "training_accounting", "compute_counts", "training_blocks", "paired_worlds", "interval_method",
            "bootstrap_replicates", "development_only", "synthetic_only", "multiplicity_adjusted",
            "complete", "all_world_condition_block_harms_retained")
    exported = {k: rep[k] for k in keys}
    exported.update(schema="graph-self-five-block-saved-export-v1", source_manifest=rep_sources,
                    terminal_readout=terminal, new_scientific_calls=0,
                    archive_verified_again=False, old_panels_pooled=False)
    publish_json("replication-evidence.json", exported, check)
    print(json.dumps(dict(check=check, primary_worlds=len(worlds), predecessors=len(predecessors),
                          configured_predecessor_arms=14, replication_pairs=len(rep["pairs"]),
                          display_contrasts=[{k: p[k] for k in ("candidate", "reference", "saving_fraction",
                                                             "saving_fraction_ci95", "extra_patient_losses")}
                                             for p in extras["display_contrasts"]]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args().check)
