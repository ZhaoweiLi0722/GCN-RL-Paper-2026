"""Saved S1 arithmetic only; no model forward, checkpoint load or simulation."""

from collections import Counter
import json
import math
from pathlib import Path

from src.rl.candidate_pilot_verification import json_hash
from src.rl.dynamic_candidate_resource_verification import read_dynamic_raw_episodes
from src.utils.research_archive import inventory, sha256_file


def summarize_qualification(outcomes, config, streams):
    """Read necessary qualification conditions, never infer unscored paths pass."""
    qualification = [row for row in outcomes if row["split"] == "qualification"]
    count = config["qualification"]["fresh_worlds_per_block"]
    expected = {(b, role, w) for b in config["blocks"]
                for role in ("r4", "initializer_greedy") for w in range(count)}
    indexed = {(row["block"], row["role"], row["world_index"]): row for row in qualification}
    if set(indexed) != expected or len(qualification) != len(expected):
        raise ValueError("missing or duplicate saved qualification episode")
    reports = []
    metrics = ("cost", "losses", "completions", "terminal_active", "waiting_patient_steps", "expiry_losses")
    for block in config["blocks"]:
        pairs, hits, multi_hits, steps, multi = [], 0, 0, 0, 0
        for world in range(count):
            ref, own = (indexed[block, role, world] for role in ("r4", "initializer_greedy"))
            seed = streams["environment"][str(block)]["qualification"][world]
            if (ref["seed"] != seed or own["seed"] != seed
                    or ref["initial_state_sha256"] != own["initial_state_sha256"]):
                raise ValueError("saved qualification pair initial state or stream mismatch")
            pairs.append({"world_index": world, "seed": seed,
                          "initial_state_sha256": own["initial_state_sha256"],
                          "raw_sha256": {"r4": ref["raw_episode_sha256"], "initializer": own["raw_episode_sha256"]},
                          "differences_initializer_minus_r4": {k: own[k] - ref[k] for k in metrics},
                          "reference_cost": ref["cost"]})
            steps += own["steps"]
            hits += own["reference_class_choices"]
            for support, correction in zip(own["support_counts"], own["reference_corrections"]):
                if support > 1:
                    multi += 1
                    multi_hits += int(not correction["different_class"])
        if len({indexed[block, "initializer_greedy", w]["policy_sha256"] for w in range(count)}) != 1:
            raise ValueError("initializer changed across saved qualification worlds")
        means = {k: math.fsum(row["differences_initializer_minus_r4"][k] for row in pairs) / count for k in metrics}
        ref_cost = math.fsum(row["reference_cost"] for row in pairs) / count
        q = config["qualification"]
        own_path_pass = (hits / steps >= q["minimum_reference_agreement_each_path"]
                         and multi >= q["minimum_multiclass_rows_each_path"]
                         and multi_hits / multi >= q["minimum_multiclass_reference_agreement_each_path"])
        outcome_pass = all(means[k] <= 0 for k in ("cost", "losses", "terminal_active")) and means["completions"] >= 0
        reports.append({"block": block, "pairs": pairs, "mean_differences": means,
                        "relative_cost_change_percent": 100 * means["cost"] / ref_cost if ref_cost > 0 else None,
                        "recorded_own_path": {"rows": steps, "reference_choices": hits, "agreement": hits / steps,
                                              "multiclass_rows": multi, "multiclass_hits": multi_hits,
                                              "multiclass_agreement": multi_hits / multi if multi else None,
                                              "necessary_condition_passed": own_path_pass},
                        "outcome_necessary_condition_passed": outcome_pass,
                        "initializer_scoring_on_r4_path": "not_performed"})
    failed = any(not r["outcome_necessary_condition_passed"]
                 or not r["recorded_own_path"]["necessary_condition_passed"] for r in reports)
    return {"blocks": reports, "all_required_necessary_conditions_passed": not failed,
            "decision": "continuation_precluded_by_saved_necessary_condition" if failed else "full_qualification_unresolved",
            "full_qualification_replayed": False, "clinical_noninferiority_claim": False,
            "rl_performance_claim": False, "automatic_training_authorized": False}


def build_saved_readout(root, proposal_path, archive_manifest_path):
    root = Path(root).resolve()
    proposal_path, archive_manifest_path = Path(proposal_path), Path(archive_manifest_path)
    manifest = json.loads(archive_manifest_path.read_text())
    files = inventory(root)
    if files != manifest["files"] or len(files) != manifest["file_count"]:
        raise ValueError("failed attempt differs from preserved source inventory")
    if not proposal_path.resolve().is_relative_to(root):
        raise ValueError("use the archived proposal inside the hash-verified failed attempt")
    relative_proposal = proposal_path.resolve().relative_to(root).as_posix()
    if files.get(relative_proposal) != sha256_file(proposal_path):
        raise ValueError("proposal is not part of the preserved source inventory")
    packet = json.loads(proposal_path.read_text())
    config, streams = packet["scientific_config"], packet["streams"]
    headers = sorted(name for name in files if name.startswith("payload/episodes/") and name.endswith("/header.json"))
    index = []
    for name in headers:
        directory = Path(name).parent
        entry = {}
        for key, leaf in (("header", "header.json"), ("events", "events.jsonl"), ("final_state", "final_state.json")):
            relative = (directory / leaf).as_posix()
            entry[key] = {"path": relative, "bytes": (root / relative).stat().st_size, "sha256": files[relative]}
        index.append(entry)
    verified_files, outcomes = read_dynamic_raw_episodes(root, index, config)
    expected = {(b, "preflight", "preflight", 0) for b in config["blocks"]}
    expected.update((b, "demonstration", "r4", w) for b in config["blocks"]
                    for w in range(config["initialization"]["demonstration_episodes_per_block"]))
    expected.update((b, "qualification", role, w) for b in config["blocks"]
                    for role in config["qualification"]["controllers"]
                    for w in range(config["qualification"]["fresh_worlds_per_block"]))
    observed = {(r["block"], r["split"], r["role"], r["world_index"]) for r in outcomes}
    if observed != expected or len(outcomes) != len(expected):
        raise ValueError("saved closed-attempt episode matrix differs")
    for row in outcomes:
        split = "prototype_preflight" if row["split"] == "preflight" else row["split"]
        if row["seed"] != streams["environment"][str(row["block"])][split][row["world_index"]]:
            raise ValueError("recorded nonqualification stream mismatch")
    result = {"format": "s1-saved-only-readout-v1", "source_root": str(root),
              "proposal_file_sha256": sha256_file(proposal_path), "config_sha256": json_hash(config),
              "preservation_manifest_sha256": sha256_file(archive_manifest_path),
              "preserved_source_files": len(files), "source_inventory_sha256": json_hash(files),
              "raw_files": verified_files, "episodes": len(outcomes),
              "raw_rows": sum(row["steps"] for row in outcomes),
              "episodes_by_split": dict(Counter(row["split"] for row in outcomes)),
              "episode_outcomes": [{k: v for k, v in row.items() if k not in
                                    ("actions", "reference_corrections", "arrivals", "support_counts")} for row in outcomes],
              "qualification": summarize_qualification(outcomes, config, streams),
              "new_environment_steps": 0, "new_optimizer_steps": 0, "new_model_forwards": 0,
              "checkpoint_loads": 0, "source_unchanged": False}
    if inventory(root) != files:
        raise ValueError("failed attempt changed during saved-only readback")
    result["source_unchanged"] = True
    return result
