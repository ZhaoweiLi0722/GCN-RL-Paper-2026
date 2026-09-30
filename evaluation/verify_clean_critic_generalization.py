"""Independent raw arithmetic/chronology audit, plus saved-weight prediction replay."""

import argparse
import copy
import gzip
import json
import math
from pathlib import Path
import statistics

from evaluation.verify_replacement_fixed_window_pilot import close, digest, file_sha, read_trace


METRICS = ("total_cost", "patients_lost", "patients_completed",
           "completion_service_level", "manufacturing_loss_rate")


def load(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
        return json.load(handle)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(root):
    status, execution = load(root / "status.json"), load(root / "execution.json")
    spec = execution["config"]
    if status["status"] != "completed" or status["exit_code"] != 0 or status["seconds"] > 7200:
        raise ValueError("Bounded attempt did not complete")
    for name, sha in load(root / "artifact_inventory.json").items():
        if file_sha(root / name) != sha:
            raise ValueError(f"Inventory mismatch: {name}")
    streams = load(root / "seed_audit.json")["streams"]
    if len(streams) != 594 or len(set(streams.values())) != 594:
        raise ValueError("Stream uniqueness failed")
    records = rows(root / "outcomes.jsonl")
    if len(records) != 3456:
        raise ValueError("Incomplete logical records")
    states = {}
    actual_steps, alias_count = 0, 0
    worlds = {"train": set(), "test": set()}
    for seed in (60, 61, 62):
        for role, count in (("train", 4), ("test", 2)):
            for trajectory in range(count):
                name = f"seed{seed}/{role}{trajectory}"
                parent = read_trace(root / f"parents/{name}.jsonl.gz")
                if [r["t"] for r in parent] != list(range(52)):
                    raise ValueError("Parent trajectory incomplete")
                actual_steps += len(parent)
                worlds[role].add(streams[name])
                for step in (0, 13, 26, 39):
                    state_id = f"{name}/t{step}"
                    state = load(root / f"states/{state_id}.json.gz")
                    lineage = state["lineage"]
                    if (lineage["state_id"] != state_id or lineage["trajectory_seed"] != streams[name]
                            or lineage["snapshot_sha256"] != digest(state["state"])
                            or state["role"] != role or lineage["usage"] != "prospective_unseen"):
                        raise ValueError("Parent lineage mismatch")
                    public = state["public"]
                    if set(public) != {"observation", "requests", "reference_request"}:
                        raise ValueError("Public whitelist mismatch")
                    close(public["observation"][-1], step / 52)
                    if digest(public["observation"]) != parent[step]["observation_sha256"]:
                        raise ValueError("Snapshot does not belong to parent trace")
                    states[state_id] = state
    if worlds["train"] & worlds["test"] or len(worlds["train"]) != 12 or len(worlds["test"]) != 6:
        raise ValueError("Trajectory leakage")
    if len({s["lineage"]["environment_sha256"] for s in states.values()}) != 1:
        raise ValueError("Environment differs")
    lookup = {}
    for record in records:
        state_id, draw, action = record["state_id"], record["draw"], record["action_index"]
        key = state_id, draw, action
        if key in lookup or not 0 <= draw < 8 or not 0 <= action < 6:
            raise ValueError("Duplicate/invalid record")
        state = states[state_id]
        step = state["lineage"]["decision_step"]
        if (record["source_state_sha256"] != state["lineage"]["snapshot_sha256"]
                or record["continuation_policy_sha256"] != state["lineage"]["continuation_sha256"]
                or record["rng_seed"] != streams[f"{state_id}/draw{draw}"]
                or record["role"] != state["role"]):
            raise ValueError("Counterfactual lineage mismatch")
        own = read_trace(root / record["trace"])
        actual_steps += len(own)
        physical = copy.deepcopy(own[0]["post_state"])
        physical["scalars"].pop("cumulative_blocked_specimen_requests", None)
        identity = digest({"state": physical, "cost": own[0]["info"]["cost"],
                           "routes": own[0]["info"]["specimen_route_events"]})
        if identity != record["execution_identity"] or own[0]["request"] != state["public"]["requests"][action]:
            raise ValueError("Wrong execution/request")
        if record["tail_from"]:
            other = lookup[state_id, draw, record["representative_action"]]
            if (len(own) != 1 or other["execution_identity"] != identity or other["tail_from"]
                    or other["trace"] != record["tail_from"]):
                raise ValueError("Unsafe tail sharing")
            trace = own + read_trace(root / record["tail_from"])[1:]
            alias_count += 1
        else:
            trace = own
        if [r["t"] for r in trace] != list(range(step, 52)):
            raise ValueError("Wrong horizon")
        for i, row in enumerate(trace):
            close(math.fsum(row["cost_components"].values()), row["info"]["cost"])
            close(row["reward"], -row["info"]["cost"])
            close(row["scaled_reward"], row["reward"] * 1e-9)
            if (row["next_t"] != row["t"] + 1 or row["truncated"]
                    or row["objective_terminal"] != (i == len(trace) - 1)):
                raise ValueError("Incomplete/disconnected trace")
            if i and trace[i - 1]["next_observation_sha256"] != row["observation_sha256"]:
                raise ValueError("Observation lineage differs")
        outcome = record["outcome"]
        close(sum(r["info"]["cost"] for r in trace), outcome["total_cost"])
        close(sum(sum(r["info"]["patients_lost"]) for r in trace), outcome["patients_lost"])
        close(sum(sum(r["info"]["patients_completed"]) for r in trace), outcome["patients_completed"])
        close(sum(r["info"]["specimen_route_count"] for r in trace), outcome["route_count"])
        close(trace[-1]["info"]["completion_service_level"], outcome["completion_service_level"])
        close(trace[-1]["info"]["patient_ineligibility_during_manufacturing_rate"], outcome["manufacturing_loss_rate"])
        if outcome["step_count"] != 52 - step or not outcome["objective_terminal"] or outcome["truncated"]:
            raise ValueError("Invalid outcome endpoint")
        lookup[key] = record
    if actual_steps != status["environment_steps"] or actual_steps > 113256:
        raise ValueError("Actual budget mismatch")
    labels = {}
    for state_id, state in states.items():
        label = load(root / f"labels/{state_id}.json")
        means = []
        expected_records = [lookup[state_id, d, a] for d in range(8) for a in range(6)]
        if label["records_sha256"] != digest(expected_records):
            raise ValueError("Label source changed")
        for action in range(6):
            costs = [lookup[state_id, d, action]["outcome"]["total_cost"] for d in range(8)]
            deltas = [costs[d] - lookup[state_id, d, 0]["outcome"]["total_cost"] for d in range(8)]
            advantages = [-1e-9 * x for x in deltas]
            target = label["targets"][action]
            for a, b in zip(deltas, target["paired_cost_deltas"]):
                close(a, b)
            close(statistics.mean(advantages), target["mean_advantage"])
            close(statistics.stdev(advantages) / math.sqrt(8), target["advantage_mean_se"])
            means.append(statistics.mean(costs))
            close(means[-1], label["mean_costs"][action])
        labels[state_id] = label
    progress = rows(root / "progress.jsonl")
    reports = {}
    for seed in (60, 61, 62):
        directory = root / f"seed{seed}"
        seal = load(directory / "sealed_test_predictions.json")
        training = [f"seed{seed}/train{i}/t{t}" for i in range(4) for t in (0, 13, 26, 39)]
        testing = [f"seed{seed}/test{i}/t{t}" for i in range(2) for t in (0, 13, 26, 39)]
        if seal["state_ids"] != testing or seal["train_labels_sha256"] != digest([labels[x] for x in training]):
            raise ValueError("Sealed input lineage differs")
        if seal["public_test_sha256"] != digest([states[x]["public"] for x in testing]):
            raise ValueError("Sealed public inputs differ")
        if seal["checkpoint_sha256"] != file_sha(directory / "critic_final.pt"):
            raise ValueError("Final weights changed")
        marker = next(i for i, r in enumerate(progress) if r["phase"] == "test_predictions_sealed" and r["seed"] == seed)
        if progress[marker]["seal_sha256"] != file_sha(directory / "sealed_test_predictions.json"):
            raise ValueError("Seal hash differs")
        for i, row in enumerate(progress):
            if row["phase"] == "labels" and row["state_id"] in testing and i <= marker:
                raise ValueError("Test labels opened before prediction seal")
        constant_means = [statistics.mean(labels[x]["mean_costs"][a if states[x]["choices"][a]["map_reachable_certified"] else 0]
                                         for x in training) for a in range(6)]
        constant = min(range(6), key=lambda a: (constant_means[a], a))
        if constant != seal["training_selected_constant"]:
            raise ValueError("Constant baseline used wrong partition")
        fit_report = load(directory / "fit_summary.json")
        scale = max(math.sqrt(statistics.mean(
            sum(labels[x]["targets"][a]["mean_advantage"] ** 2 for a in range(6)
                if states[x]["choices"][a]["map_reachable_certified"])
            / sum(c["map_reachable_certified"] for c in states[x]["choices"]) for x in training)), 1e-9)
        if not math.isclose(scale, fit_report["target_scale"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Training-only scaling differs")
        if [r["update"] for r in rows(directory / "fit.jsonl")] != [1] + list(range(100, 1001, 100)):
            raise ValueError("Wrong optimizer progress")
        report = load(directory / "test_summary.json")
        correct = incorrect = ties = predicted_ties = 0
        for i, state_id in enumerate(testing):
            prediction = seal["advantage_predictions"][i]
            support = [a for a, c in enumerate(states[state_id]["choices"]) if c["map_reachable_certified"]]
            if prediction[0] != 0 or not all(math.isfinite(x) for x in prediction):
                raise ValueError("Invalid predictions")
            chosen = max(support, key=lambda a: (prediction[a], -a))
            if chosen != seal["critic_actions"][i]:
                raise ValueError("Action not selected from prediction alone")
            costs = labels[state_id]["mean_costs"]
            close(costs[chosen] - min(costs[a] for a in support), report["ranking"]["sampled_support_regret"][i])
            for j, left in enumerate(support):
                for right in support[j + 1:]:
                    truth, predicted = costs[right] - costs[left], prediction[left] - prediction[right]
                    if truth == 0:
                        ties += 1
                    elif predicted == 0:
                        predicted_ties += 1
                    elif truth * predicted > 0:
                        correct += 1
                    else:
                        incorrect += 1
            for baseline, action in (("frozen", 0), ("mdl2", 1), ("constant", constant)):
                action = action if action in support else 0
                if seal[f"{baseline}_actions"][i] != action:
                    raise ValueError("Baseline fallback differs")
                for metric in METRICS:
                    expected = [lookup[state_id, d, chosen]["outcome"][metric]
                                - lookup[state_id, d, action]["outcome"][metric] for d in range(8)]
                    for got, want in zip(report["states"][i]["contrasts"][baseline][metric], expected):
                        close(got, want)
        expected_counts = {"correct": correct, "incorrect": incorrect, "label_ties": ties,
                           "prediction_ties": predicted_ties, "comparable_pairs": correct + incorrect + predicted_ties}
        if any(report["ranking"][k] != v for k, v in expected_counts.items()):
            raise ValueError("Pairwise counts differ")
        expected_accuracy = correct / expected_counts["comparable_pairs"] if expected_counts["comparable_pairs"] else None
        if expected_accuracy != report["ranking"]["pairwise_accuracy"]:
            raise ValueError("Pairwise denominator differs")
        for trajectory in range(2):
            summary = report["trajectories"][trajectory]
            for baseline in ("frozen", "mdl2", "constant"):
                for metric in METRICS:
                    expected = statistics.mean(x for r in report["states"] if r["trajectory"] == trajectory
                                               for x in r["contrasts"][baseline][metric])
                    close(expected, summary["mean_contrasts"][baseline][metric])
            frozen = summary["mean_contrasts"]["frozen"]
            adverse = (frozen["patients_lost"] > 0 or frozen["patients_completed"] < 0
                       or frozen["completion_service_level"] < 0 or frozen["manufacturing_loss_rate"] > 0)
            if summary["adverse_clinical_mean_vs_frozen"] != adverse:
                raise ValueError("Clinical warning omitted")
        reports[str(seed)] = {"ranking": report["ranking"], "trajectories": report["trajectories"],
                              "training_constant": constant, "scale": scale}
    saved_summary = load(root / "summary.json")
    ts = [t for r in reports.values() for t in r["trajectories"]]
    cost = all(t["mean_contrasts"][b]["total_cost"] < 0 for t in ts for b in ("frozen", "constant"))
    ranking = all(r["ranking"]["pairwise_accuracy"] is not None
                  and r["ranking"]["pairwise_accuracy"] > .5 for r in reports.values())
    clinical = not any(t["adverse_clinical_mean_vs_frozen"] for t in ts)
    triage = {"all_trajectory_cost": cost, "all_policy_ranking": ranking,
              "clinical_directions": clinical, "followup_discussion_only": cost and ranking and clinical}
    if saved_summary["triage"] != triage or saved_summary["supervised_critic_updates"] != 3000:
        raise ValueError("Summary/triage differs")
    return {"verified_raw_arithmetic": True, "records": len(records), "states": len(states),
            "actual_steps": actual_steps, "aliased_continuations": alias_count,
            "policies": reports, "triage": triage, "online_benefit_claim": False,
            "limitations": ["six test trajectories", "dependent states and draws", "replacement frozen baselines",
                            "one-step changes then frozen continuation", "no clinical noninferiority"]}


def replay_predictions(root):
    import numpy as np
    import torch
    from evaluation.audit_replacement_policy_compatibility import assert_unchanged, verify_inputs
    from evaluation.run_replacement_fixed_window_pilot import load_model
    from src.rl.clean_critic_probe import advantage, fresh_critic, tensor_inputs
    from src.rl.critic_probe_contract import public_inputs
    spec = load(root / "execution.json")["config"]
    lock = load(Path(spec["input_lock_config"]))
    manifest, _ = verify_inputs(lock)
    report = {}
    for seed, init in zip(spec["seeds"], spec["critic_init_seeds"]):
        policy, _, payload = load_model(lock, manifest, seed, "mps")
        directory = root / f"seed{seed}"
        saved = torch.load(directory / "critic_final.pt", map_location="cpu", weights_only=False)
        initial = torch.load(directory / "critic_initial.pt", map_location="cpu", weights_only=False)
        model = fresh_critic(policy._agent, init)
        if not all(torch.equal(v.detach().cpu(), initial["critic"][k]) for k, v in model.state_dict().items()):
            raise ValueError("Fresh critic initialization differs")
        if saved["updates"] != 1000 or saved["actor_updates"] or saved["ddpg_updates"]:
            raise ValueError("Wrong saved training scope")
        if not all(float(x["step"]) == 1000 for x in saved["optimizer"]["state"].values()):
            raise ValueError("Wrong Adam step count")
        if "mps_rng" not in saved or "mps_rng" not in initial:
            raise ValueError("Missing MPS RNG preservation")
        model.load_state_dict(saved["critic"], strict=True)
        model.eval()
        seal = load(directory / "sealed_test_predictions.json")
        data = [load(root / f"states/{state_id}.json.gz") for state_id in seal["state_ids"]]
        public = [public_inputs(s["public"], s["lineage"]["decision_step"]) for s in data]
        with torch.no_grad():
            predicted = advantage(model, *tensor_inputs(policy._agent, public)).cpu().numpy() * saved["target_scale"]
        expected = np.array(seal["advantage_predictions"])
        np.testing.assert_allclose(predicted, expected, rtol=1e-5, atol=1e-8)
        assert_unchanged(policy, payload)
        if file_sha(directory / "frozen_actor_gate_source.pt") != policy.checkpoint_sha256:
            raise ValueError("Frozen source payload differs")
        report[str(seed)] = {"max_prediction_difference": float(np.max(np.abs(predicted - expected))),
                             "fresh_initialization_reproduced": True, "adam_steps": 1000}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-predictions", action="store_true")
    args = parser.parse_args()
    result = verify(args.root)
    if args.replay_predictions:
        result["saved_weight_prediction_replay"] = replay_predictions(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2))
