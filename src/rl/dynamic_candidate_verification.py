"""Independent raw-only verification for the proposed five-controller pilot.

No environment, policy, optimizer or checkpoint imports. The legacy reader
checks scalar accounting and file lineage; explicit new-role checks below close
its intentionally unchanged role-name guards. Nothing here authorizes execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from src.rl.candidate_pilot_verification import (
    CAUSES, finite, json_hash, read_raw_episodes, vector, verify_episode,
)


CONTROLLERS = ("own_frozen", "own_ppo", "own_bc_continue", "r4", "full_mdl2")
SECONDARIES = ("ppo_minus_bc_continue", "ppo_minus_r4", "ppo_minus_full_mdl2", "own_frozen_minus_r4")
CONTRASTS = (("ppo_minus_own_frozen", "own_ppo", "own_frozen"),
             (SECONDARIES[0], "own_ppo", "own_bc_continue"),
             (SECONDARIES[1], "own_ppo", "r4"),
             (SECONDARIES[2], "own_ppo", "full_mdl2"),
             (SECONDARIES[3], "own_frozen", "r4"))
EXECUTED = ("specimen_transfers", "capacity_transfers", "reagent_transfers", "replenishment")
METRICS = ("cost", "losses", "completions", "waiting_patient_steps", "expiry_losses",
           "terminal_active", "terminal_waiting", "terminal_production", "terminal_specimen_transit",
           "terminal_finished_return", "enrolled", "completion_service_level", "route_count",
           "blocked_requests", "inference_seconds")
PATIENT_DIRECTIONS = {"losses": 1, "completions": -1, "waiting_patient_steps": 1,
                      "expiry_losses": 1, "terminal_active": 1}
SPLIT_ROLES = {"test": CONTROLLERS, "qualification": ("r4", "initializer_greedy"),
               "preflight": ("preflight", *CONTROLLERS), "demonstration": ("r4",),
               "training": ("own_ppo", "own_bc_continue")}


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"positive integer {name} required")
    return value


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("explicit lowercase SHA256 lineage required")


def _validate_plan(config, streams):
    blocks, evaluation = config["blocks"], config["evaluation"]
    if (len(blocks) != 3 or any(type(b) is not int for b in blocks) or len(set(blocks)) != 3
            or config["representations"] != ["graph"]
            or tuple(config["final_controllers"]) != CONTROLLERS
            or tuple(evaluation["secondary_contrasts"]) != SECONDARIES):
        raise ValueError("single-graph, three-block five-controller schedule required")
    worlds = _positive_int(evaluation["fresh_paired_worlds_per_block"], "world count")
    if (evaluation["total_worlds"] != 3 * worlds or evaluation["total_episodes"] != 15 * worlds
            or evaluation["controllers_per_block"] != 5
            or evaluation["environment_steps"] != 15 * worlds * config["objective"]["horizon"]):
        raise ValueError("evaluation dimensions disagree with schedule")
    bootstrap = evaluation["bootstrap"]
    if (bootstrap["kind"] != "hierarchical_paired_block_then_world"
            or bootstrap["blocks_per_draw"] != 3 or bootstrap["worlds_per_sampled_block"] != worlds
            or evaluation["required_favorable_cost_blocks"] != 3
            or evaluation["proposed_practical_relative_cost_reduction_percent"] != 1.0
            or evaluation["learned_selection"] != "greedy_lexicographic_class_key_tie_break"
            or evaluation["primary"] != "paired_raw_total_cost_ppo_minus_own_frozen"
            or evaluation["nonpositive_relative_denominator"] != "undefined_not_passed"):
        raise ValueError("prespecified estimand/bootstrap/triage differs")
    _positive_int(bootstrap["draws"], "bootstrap draws")
    seed = streams["neural"]["analysis/bootstrap"]
    if type(seed) is not int or seed < 0:
        raise ValueError("explicit nonnegative bootstrap stream required")
    test_seeds = []
    for block in blocks:
        seeds = streams["environment"][str(block)]["test"]
        if len(seeds) != worlds or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("test stream dimensions differ")
        test_seeds.extend(seeds)
    if len(set(test_seeds)) != len(test_seeds):
        raise ValueError("test world starts are not unique across blocks")
    if config["objective"]["action_width"] != 4 * config["objective"]["num_facilities"]:
        raise ValueError("four-channel facility action layout required")
    if finite(config["objective"]["transfer_scale"]) <= 0:
        raise ValueError("positive specimen transfer scale required")
    return worlds


def _reread(root, record, *, jsonl=False):
    """Bind the role/action readback to exactly the already scalar-verified bytes."""
    path = root / record["path"]
    if (path.is_symlink() or path.resolve() != path.absolute()
            or not path.is_file() or not path.resolve().is_relative_to(root)):
        raise ValueError("redirected raw evidence during readback")
    raw = path.read_bytes()
    if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("raw file changed during independent readback")

    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate raw JSON key")
            result[key] = value
        return result

    def parse(value):
        return json.loads(value, object_pairs_hook=no_duplicate_keys)

    result = [parse(line) for line in raw.decode().splitlines()] if jsonl else parse(raw)

    def check(value):
        if isinstance(value, float):
            finite(value, signed=True)
        elif isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(result)
    return result


def _enrich(outcome, header, rows, final_state, config):
    role, split, n = header["role"], header["split"], config["objective"]["num_facilities"]
    if split not in SPLIT_ROLES or role not in SPLIT_ROLES[split]:
        raise ValueError("unexpected dynamic controller/split")
    expected_representation = "reference" if role in ("r4", "full_mdl2") else "graph"
    if header["representation"] != expected_representation:
        raise ValueError("controller representation mismatch")
    expected_selection = ("reference" if role == "r4" else "anchor" if role == "full_mdl2" else
                          "greedy" if role in ("own_frozen", "initializer_greedy") or split == "test" else "sample")
    if header["selection"] != expected_selection:
        raise ValueError("recorded controller selection mode differs")
    if any(not isinstance(header[k], str) or not header[k] for k in ("source_id", "trajectory_id")):
        raise ValueError("nonempty episode lineage required")
    _sha(header["policy_sha256"])
    if json_hash([header, rows, final_state]) != outcome["raw_episode_sha256"]:
        raise ValueError("raw episode changed between scalar and action verification")
    actions, waiting, corrections = [], 0, []
    for step, row in enumerate(rows):
        event = row["event"]
        record, decision = event["audit"]["record"], event["audit"]["decision"]
        evaluation, choice = decision["evaluation"], decision["choice"]
        bank, info = evaluation["candidates"], event["info"]
        request, reference, anchor = record["action"], bank["requests"][0], bank["requests"][1]
        # The old scalar verifier recognizes frozen/ppo/bc_continue/mdl2 only.
        # Check these new role names explicitly, including lexicographic ties.
        greedy = ((split == "test" and role.startswith("own_"))
                  or role in ("own_frozen", "initializer_greedy"))
        if greedy and choice["class_index"] != int(np.argmax(evaluation["log_probs"])):
            raise ValueError("dynamic learned test did not use canonical greedy class")
        if role == "r4" and request != reference:
            raise ValueError("R4 original request changed")
        if role == "full_mdl2" and (request != anchor or reference != anchor):
            raise ValueError("full MDL-2 requires the full-anchor collector, not a restricted anchor")
        step_waiting = sum(vector(info["waiting_patients"], n))
        waiting += step_waiting
        executed = {k: vector(info[k], n, signed=k != "replenishment") for k in EXECUTED}
        if any(sum(executed[k]) != 0 for k in EXECUTED[:3]):
            raise ValueError("executed facility-net transfer does not conserve flow")
        correction = [float(a) - float(b) for a, b in zip(request, reference)]
        corrections.append({"step_index": step, "request_delta": correction,
                            "request_l1": sum(abs(v) for v in correction),
                            "different_class": choice["class_index"] != bank["request_to_class"][0]})
        actions.append({"step_index": step, "requested": copy.deepcopy(request),
                        "specimen_requested_integer_net": copy.deepcopy(info["specimen_requested_integer_net"]),
                        "executed": executed, "reference_request": copy.deepcopy(reference),
                        "selected_class_key": copy.deepcopy(bank["class_keys"][choice["class_index"]]),
                        "inference_seconds": finite(row["inference_seconds"])})
    terminal = outcome["terminal_compartments"]
    for field, compartment in (("waiting_patients", "waiting"), ("in_production_patients", "production"),
                               ("specimen_in_transit", "specimen_transit")):
        if sum(vector(rows[-1]["event"]["info"][field], n)) != terminal[compartment]:
            raise ValueError("final reported queue differs from raw patient registry")
    # Waiting is end-of-epoch queue occupancy integrated over epochs, not a
    # completed-patient waiting-time estimator or a new cost component.
    return outcome | {
        "source_id": header["source_id"], "waiting_patient_steps": waiting,
        "waiting_definition": "sum_of_end_of_epoch_waiting_patients",
        "expiry_losses": sum(outcome["loss_causes"][k] for k in
                             ("patients_lost_waiting_expired", "patients_lost_transit_expired", "finished_expired")),
        **{f"terminal_{k}": terminal[k] for k in ("waiting", "production", "specimen_transit", "finished_return")},
        "actions": actions, "reference_corrections": corrections,
        "correction_reference": "full_mdl2" if role == "full_mdl2" else "contemporaneous_r4",
        "raw_header_sha256": json_hash(header),
        "compute_cost": {"environment_steps": len(rows), "evaluation_env_steps": len(rows) if split == "test" else 0,
                         "decision_path_seconds": outcome["inference_seconds"],
                         "training_compute": "not_in_episode_receipts_use_campaign_budget_ledger"},
    }


def _action_difference(left, right):
    result = {"requested_l1": 0.0, "requested_changed_steps": 0,
              "specimen_requested_integer_l1": 0, "executed_changed_steps": 0,
              "executed_l1": {k: 0 for k in EXECUTED}}
    for a, b in zip(left["actions"], right["actions"]):
        result["requested_l1"] += sum(abs(x - y) for x, y in zip(a["requested"], b["requested"]))
        result["requested_changed_steps"] += int(a["requested"] != b["requested"])
        result["specimen_requested_integer_l1"] += sum(abs(x - y) for x, y in zip(
            a["specimen_requested_integer_net"], b["specimen_requested_integer_net"]))
        result["executed_changed_steps"] += int(a["executed"] != b["executed"])
        for channel in EXECUTED:
            result["executed_l1"][channel] += sum(abs(x - y) for x, y in zip(a["executed"][channel], b["executed"][channel]))
    return result


def _paired_analysis(outcomes, config, streams, worlds):
    blocks, evaluation = config["blocks"], config["evaluation"]
    expected = {(b, role, w) for b in blocks for role in CONTROLLERS for w in range(worlds)}
    indexed = {}
    for row in outcomes:
        if any(type(row[k]) is not int for k in ("block", "world_index", "seed")):
            raise ValueError("integer evaluation lineage required")
        key = row["block"], row["role"], row["world_index"]
        if key not in expected or key in indexed or row["split"] != "test":
            raise ValueError("duplicate or unexpected evaluation outcome")
        if row["seed"] != streams["environment"][str(row["block"])]["test"][row["world_index"]]:
            raise ValueError("wrong test stream")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("missing prescribed evaluation outcomes")
    if len({r["trajectory_id"] for r in outcomes}) != len(outcomes):
        raise ValueError("duplicate evaluation trajectory lineage")
    for b in blocks:
        for w in range(worlds):
            if len({indexed[b, role, w]["initial_state_sha256"] for role in CONTROLLERS}) != 1:
                raise ValueError("paired controllers do not share the exact initial state")
        for role in CONTROLLERS:
            if len({indexed[b, role, w]["policy_sha256"] for w in range(worlds)}) != 1:
                raise ValueError("fixed final policy changed across test worlds")
    bootstrap = evaluation["bootstrap"]
    seed = streams["neural"]["analysis/bootstrap"]
    rng = np.random.default_rng(seed)
    # One shared draw design preserves paired worlds, blocks, metrics and
    # contrasts. Repeated copies of a block draw their worlds independently.
    block_draws = rng.integers(0, 3, size=(bootstrap["draws"], 3))
    world_draws = rng.integers(0, worlds, size=(bootstrap["draws"], 3, worlds))
    results = []
    for name, left, right in CONTRASTS:
        delta, baseline, block_rows, paired = [], [], [], []
        for b in blocks:
            differences, base, world_rows = [], [], []
            for w in range(worlds):
                lrow, rrow = indexed[b, left, w], indexed[b, right, w]
                d = {m: lrow[m] - rrow[m] for m in METRICS}
                differences.append([d[m] for m in METRICS])
                base.append(rrow["cost"])
                world_rows.append({"world_index": w, "seed": lrow["seed"],
                    "initial_state_sha256": lrow["initial_state_sha256"],
                    "left_raw_episode_sha256": lrow["raw_episode_sha256"],
                    "right_raw_episode_sha256": rrow["raw_episode_sha256"], "differences": d,
                    "component_differences": {k: lrow["components"][k] - rrow["components"][k] for k in lrow["components"]},
                    "loss_cause_differences": {k: lrow["loss_causes"][k] - rrow["loss_causes"][k] for k in CAUSES},
                    "action_differences": _action_difference(lrow, rrow)})
            mean = np.mean(differences, axis=0)
            denominator = float(np.mean(base))
            block_rows.append({"block": b, "mean_differences": dict(zip(METRICS, mean.tolist())),
                               "baseline_mean_cost": denominator,
                               "relative_cost_change_percent": 100 * float(mean[0]) / denominator if denominator > 0 else None})
            paired.append({"block": b, "worlds": world_rows})
            delta.append(differences)
            baseline.append(base)
        delta, baseline = np.asarray(delta, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
        boot_delta = delta[block_draws[..., None], world_draws].mean(axis=2)
        boot_baseline = baseline[block_draws[..., None], world_draws].mean(axis=2)
        interval = np.quantile(boot_delta.mean(axis=1), [.025, .975], axis=0)
        relative = [r["relative_cost_change_percent"] for r in block_rows]
        relative_mean = float(np.mean(relative)) if all(v is not None for v in relative) else None
        valid = np.all(boot_baseline > 0, axis=1)
        # Do not silently discard undefined resamples to manufacture an interval.
        relative_ci = (np.quantile((100 * boot_delta[:, :, 0] / boot_baseline).mean(axis=1),
                                   [.025, .975]).tolist() if bool(valid.all()) else None)
        adverse = [{"block": row["block"], "metric": m, "mean_difference": row["mean_differences"][m]}
                   for row in block_rows for m, direction in PATIENT_DIRECTIONS.items()
                   if direction * row["mean_differences"][m] > 0]
        means = dict(zip(METRICS, delta.mean(axis=(0, 1)).tolist()))
        results.append({"name": name, "left": left, "right": right, "blocks": block_rows, "paired_worlds": paired,
                        "equal_block_mean_differences": means,
                        "equal_block_relative_cost_change_percent": relative_mean,
                        "hierarchical_paired_ci95": {m: interval[:, i].tolist() for i, m in enumerate(METRICS)},
                        "relative_cost_change_ci95": relative_ci,
                        "undefined_relative_bootstrap_draws": int((~valid).sum()),
                        "adverse_patient_block_directions": adverse,
                        "cost_patient_tradeoff": means["cost"] < 0 and bool(adverse)})
    primary = results[0]
    relative = primary["equal_block_relative_cost_change_percent"]
    checks = {"all_three_primary_blocks_cost_favorable": all(r["mean_differences"]["cost"] < 0 for r in primary["blocks"]),
              "equal_block_relative_cost_reduction_at_least_one_percent": relative is not None and relative <= -1.0,
              "primary_relative_cost_defined": relative is not None,
              "primary_has_adverse_patient_direction": bool(primary["adverse_patient_block_directions"])}
    favorable = checks["all_three_primary_blocks_cost_favorable"] and checks["equal_block_relative_cost_reduction_at_least_one_percent"]
    decision = ("cost_patient_tradeoff" if primary["cost_patient_tradeoff"] else
                "promising_development_signal" if favorable else "limited_negative_or_inconclusive")
    return {"format": "dynamic-candidate-independent-analysis-v1", "evaluation_episodes": len(outcomes),
            "independent_training_blocks": 3, "primary": primary["name"], "contrasts": results,
            "triage_checks": checks, "decision": decision,
            "decision_scope": "primary PPO-versus-own-frozen development triage; all secondary tradeoffs reported separately",
            "bootstrap": {"kind": bootstrap["kind"], "draws": bootstrap["draws"], "seed": seed,
                          "blocks_per_draw": 3, "worlds_per_sampled_block": worlds,
                          "rng": "numpy.default_rng.PCG64", "numpy_version": np.__version__,
                          "quantile_method": "linear", "resample_design_sha256": json_hash([block_draws.tolist(), world_draws.tolist()]),
                          "descriptive_low_precision_three_blocks": True},
            "clinical_noninferiority_claim": False, "deployment_adaptation_claim": False,
            "clean_ddpg_superiority_claim": False, "isolated_graph_contribution_claim": False,
            "automatic_followon": False,
            "limitations": ["three training blocks; paired worlds are not independent trained policies",
                            "finite-window cost; unresolved terminal patients are not free benefit or deaths",
                            "queue occupancy integral is not mean completed-patient waiting time",
                            "same initial state does not imply event-level common random numbers after actions diverge",
                            "cross-controller action differences are closed-loop traces, not same-state counterfactual effects",
                            "full-anchor provenance depends on the source-frozen collector; no checkpoint inference is performed",
                            "training resource costs and all-model test seals must be verified by the campaign ledger/orchestrator"]}


def verify_dynamic_episode(header, rows, final_state, config):
    """Raw scalar/role verification for one complete recorded episode.

    Qualification is r4/initializer_greedy; preflight also permits sampled
    preflight/own_ppo/own_bc_continue. This function never admits a phase or
    evaluates a qualification threshold; the orchestrator owns those decisions.
    """
    scalar = verify_episode(header, rows, final_state, config)
    return _enrich(scalar, header, rows, final_state, config)


def read_dynamic_raw_episodes(root, index, config):
    """Read a partial phase or whole episode set without requiring the test matrix.

    Return (files, outcomes), like the legacy helper. Inputs are the existing
    EpisodeRecorder header/events/final_state records, never scalar summaries.
    """
    root = Path(root).resolve()
    files, scalar = read_raw_episodes(root, index, config)
    outcomes = []
    for entry, original in zip(index, scalar):
        header = _reread(root, entry["header"])
        rows = _reread(root, entry["events"], jsonl=True)
        final = _reread(root, entry["final_state"])
        outcomes.append(_enrich(original, header, rows, final, config) | {"raw_files": copy.deepcopy(entry)})
    return copy.deepcopy(files), outcomes


def verify_dynamic_raw_bundle(root, index, config, streams):
    """Verify complete raw test episodes, exact files, paired effects and triage.

    ``index`` uses the existing header/events/final_state file-record structure.
    The draft config additionally needs objective.transfer_scale and
    candidate_support.max_original_requests. No learner summaries are trusted.
    """
    worlds = _validate_plan(config, streams)
    if len(index) != config["evaluation"]["total_episodes"]:
        raise ValueError("missing or extra prescribed raw evaluation entries")
    files, outcomes = read_dynamic_raw_episodes(root, index, config)
    return {"format": "dynamic-candidate-raw-verification-v1", "files": copy.deepcopy(files),
            "index_sha256": json_hash(index), "config_sha256": json_hash(config), "streams_sha256": json_hash(streams),
            "outcomes": outcomes, "analysis": _paired_analysis(outcomes, config, streams, worlds),
            "verification": "raw file hashes, scalar accounting, dynamic role semantics and full paired schedule verified",
            "grants_scientific_execution_authorization": False}
