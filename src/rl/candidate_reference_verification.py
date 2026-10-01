"""P2 raw-data checks including duplicate comparator lineage and train coverage."""

import json
from pathlib import Path

from src.rl.candidate_pilot_verification import read_raw_episodes, verify_raw_bundle, json_hash


def behavioral_trace(root, entry):
    rows = [json.loads(line)["event"] for line in (root / entry["events"]["path"]).read_text().splitlines()]
    # Timing, trajectory IDs and model definitions differ, physics and requests may not.
    return json_hash([{"info": row["info"], "record": {k: row["audit"]["record"][k] for k in
        ("state", "next_state", "action", "raw_reward", "state_token", "next_state_token", "terminated", "truncated")}}
        for row in rows])


def verify_reference_bundle(root, index, config, streams):
    root = Path(root)
    report = verify_raw_bundle(root, index, config, streams)
    keyed = {(row["block"], row["representation"], row["role"], row["world_index"]): (row, entry)
             for row, entry in zip(report["outcomes"], index)}
    aliases = []
    for block in config["blocks"]:
        for world in range(config["evaluation"]["episodes_per_policy"]):
            reference, ref_entry = keyed[block, "reference", "r4", world]
            trace = behavioral_trace(root, ref_entry)
            for rep in config["representations"]:
                frozen, entry = keyed[block, rep["name"], "frozen", world]
                if (trace != behavioral_trace(root, entry) or any(frozen[k] != reference[k] for k in
                        ("initial_state_sha256", "final_state_sha256", "cost", "components", "losses",
                         "completions", "terminal_active", "route_count"))):
                    raise ValueError("untrained prior frozen controller differs from R4 closed loop")
                aliases.append({"block": block, "world": world, "frozen": rep["name"], "reference": "r4",
                                "behavioral_trace_sha256": trace, "exact_closed_loop_equal": True})
    training = []
    for block in config["blocks"]:
        for rep in config["representations"]:
            for role in ("ppo", "bc_continue"):
                scope = f"block{block}/{rep['name']}/{role}"
                path = root / "payload/models" / scope / "training.json"
                stored = json.loads(path.read_text())
                files, outcomes = read_raw_episodes(root, stored["raw_episodes"], config)
                if len(outcomes) != config[role]["episodes_per_model"] or stored["episodes"] != len(outcomes):
                    raise ValueError("incomplete prescribed training rows")
                for world, row in enumerate(outcomes):
                    if (row["split"] != "training" or row["block"] != block or row["representation"] != rep["name"]
                            or row["role"] != role or row["world_index"] != world
                            or row["seed"] != streams["environment"]["training"][str(block)][world]):
                        raise ValueError("wrong/duplicate training lineage or stream")
                total_steps = sum(r["steps"] for r in outcomes)
                training.append({"scope": scope, "files": files, "outcomes": outcomes, "steps": total_steps,
                    "nonreference_class_choices": sum(r["steps"] - r["reference_class_choices"] for r in outcomes),
                    "nonreference_choice_fraction": sum(r["steps"] - r["reference_class_choices"] for r in outcomes) / total_steps,
                    "route_count": sum(r["route_count"] for r in outcomes),
                    "singleton_steps": sum(k == 1 for r in outcomes for k in r["support_counts"]),
                    "sum_per_episode_distinct_executed_flows": sum(r["distinct_executed_net_flows"] for r in outcomes)})
    report.update(training=training, duplicate_frozen_reference_lineage=aliases,
                  independent_training_blocks=3, duplicate_baselines_increase_sample_size=False)
    report["analysis"]["limitations"].append("three frozen representations duplicate R4 by construction; no independent baseline replications")
    return report
