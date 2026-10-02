"""Independent, raw-only readout for the six-controller time-baseline trial.

No environment, model, checkpoint or optimizer is opened. The scalar/file and
continuous-resource readers remain unchanged; role checks and paired bootstrap
are explicitly versioned here. A readout never authorizes another experiment.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

from src.rl.candidate_pilot_verification import (
    CAUSES, finite, json_hash, read_raw_episodes, vector,
)
from src.rl.dynamic_candidate_resource_verification import executed_resources
from src.rl.dynamic_candidate_verification import (
    METRICS, PATIENT_DIRECTIONS, _action_difference, _reread, _sha,
)


TRAINING_ROLES = ("current_ppo", "time_baseline_ppo", "bc_continue")
LEARNED_ROLES = ("own_frozen", *TRAINING_ROLES)
CONTROLLERS = (*LEARNED_ROLES, "r4", "full_mdl2")
SPLIT_ROLES = {"training": TRAINING_ROLES, "preflight": CONTROLLERS, "test": CONTROLLERS}
CONTRASTS = (
    ("time_baseline_ppo_minus_own_frozen", "time_baseline_ppo", "own_frozen"),
    ("time_baseline_ppo_minus_current_ppo", "time_baseline_ppo", "current_ppo"),
    ("time_baseline_ppo_minus_bc_continue", "time_baseline_ppo", "bc_continue"),
    ("time_baseline_ppo_minus_r4", "time_baseline_ppo", "r4"),
    ("time_baseline_ppo_minus_full_mdl2", "time_baseline_ppo", "full_mdl2"),
    ("own_frozen_minus_r4", "own_frozen", "r4"),
)
SECONDARIES = tuple(row[0] for row in CONTRASTS[1:])
PRIMARY = "paired_raw_total_cost_time_baseline_ppo_minus_own_frozen"


def _decimal_seed(value):
    # Avoid float round trips for the 112-bit world seeds and reject JSON bools.
    if (not isinstance(value, str) or not value.isascii() or not value.isdecimal()
            or str(int(value)) != value):
        raise ValueError("canonical nonnegative decimal-string seed required")
    return int(value)


def _environment_schedule(config):
    proposal = config["time_baseline_proposal"]
    blocks = config["blocks"]
    if (blocks != [60, 61, 62] or any(type(b) is not int for b in blocks)
            or proposal["blocks"] != blocks
            or proposal["schema"] != "time-baseline-comparison-proposal-v1"
            or tuple(config["final_controllers"]) != CONTROLLERS
            or tuple(proposal["evaluation_controllers"]) != CONTROLLERS
            or tuple(proposal["training_arms"]) != TRAINING_ROLES):
        raise ValueError("three-block six-controller proposal required")
    rng = proposal["rng"]
    namespace = rng["namespace"]
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("explicit source namespace required")
    if rng["encoding"] != "sha256(namespace)[:12] big-endian <<16 plus ordinal; decimal strings on disk":
        raise ValueError("world seed encoding differs")
    lengths = {"layout": 1, "preflight": 1, "training": 32, "test": 12}
    if (proposal["episodes_per_training_arm_block"] != 32
            or proposal["evaluation_worlds_per_block"] != 12
            or proposal["preflight_worlds_per_block"] != 1):
        raise ValueError("prospective episode dimensions differ")
    if set(rng["roles_by_block"]) != {str(b) for b in blocks}:
        raise ValueError("world block inventory differs")
    base = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:12], "big") << 16
    schedule, ordinals = {}, []
    for block in blocks:
        roles = rng["roles_by_block"][str(block)]
        if set(roles) != set(lengths):
            raise ValueError("world split inventory differs")
        schedule[str(block)] = {}
        for split, length in lengths.items():
            bounds = roles[split]
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or any(type(v) is not int or v < 0 for v in bounds)
                    or bounds[1] - bounds[0] + 1 != length):
                raise ValueError("world ordinal range differs")
            selected = list(range(bounds[0], bounds[1] + 1))
            ordinals.extend(selected)
            schedule[str(block)][split] = [base + value for value in selected]
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("overlapping layout/preflight/training/test streams")
    return namespace, schedule


def _validate_plan(config, streams):
    namespace, schedule = _environment_schedule(config)
    evaluation, objective = config["evaluation"], config["objective"]
    if (config["representations"] != ["graph"]
            or evaluation["fresh_paired_worlds_per_block"] != 12
            or evaluation["total_episodes"] != 216
            or evaluation["required_favorable_cost_blocks"] != 3
            or evaluation["proposed_practical_relative_cost_reduction_percent"] != 1.0
            or evaluation["learned_selection"] != "greedy_lexicographic_class_key_tie_break"
            or evaluation["nonpositive_relative_denominator"] != "undefined_not_passed"):
        raise ValueError("prespecified six-controller dimensions/triage differs")
    # These metadata fields may be omitted by an adapter, but never contradict
    # the independently enumerated estimand and matrix when supplied.
    expected = {"controllers_per_block": 6, "total_worlds": 36,
                "environment_steps": 216 * objective["horizon"], "primary": PRIMARY,
                "secondary_contrasts": list(SECONDARIES)}
    for name, value in expected.items():
        if name in evaluation and evaluation[name] != value:
            raise ValueError(f"evaluation {name} differs from six-controller schedule")
    bootstrap = evaluation["bootstrap"]
    if (bootstrap["kind"] != "hierarchical_paired_block_then_world"
            or type(bootstrap["draws"]) is not int or bootstrap["draws"] != 10000
            or bootstrap["blocks_per_draw"] != 3 or bootstrap["worlds_per_sampled_block"] != 12):
        raise ValueError("prespecified paired bootstrap differs")
    if (objective["action_width"] != 4 * objective["num_facilities"]
            or finite(objective["transfer_scale"]) <= 0):
        raise ValueError("four-channel facility action layout required")
    if (streams["format"] != "time-baseline-streams-v1"
            or streams["namespace"] != namespace
            or streams["proposal_sha256"] != json_hash(config["time_baseline_proposal"])):
        raise ValueError("streams source/proposal binding differs")
    environment = streams["environment"]
    if set(environment) != set(schedule):
        raise ValueError("streams block inventory differs")
    for block, roles in schedule.items():
        if set(environment[block]) != set(roles):
            raise ValueError("streams split inventory differs")
        for split, seeds in roles.items():
            if [_decimal_seed(s) for s in environment[block][split]] != seeds:
                raise ValueError("wrong prescribed scenario seed stream")
    seed = _decimal_seed(streams["neural"]["analysis/bootstrap"])
    expected_seed = int.from_bytes(hashlib.sha256((namespace + "/analysis/bootstrap").encode()).digest()[:8], "big") % 2**63
    if seed != expected_seed:
        raise ValueError("wrong prescribed bootstrap seed")
    return 12, seed


def _enrich(outcome, header, rows, final_state, config, namespace, schedule):
    role, split, n = header["role"], header["split"], config["objective"]["num_facilities"]
    if split not in SPLIT_ROLES or role not in SPLIT_ROLES[split]:
        raise ValueError("unexpected time-baseline controller/split")
    if any(type(header[k]) is not int for k in ("block", "world_index", "seed")):
        raise ValueError("integer episode lineage required")
    block, world = str(header["block"]), header["world_index"]
    if (block not in schedule or not 0 <= world < len(schedule[block][split])
            or header["seed"] != schedule[block][split][world]):
        raise ValueError("wrong prescribed scenario seed or episode index")
    if (header["source_id"] != namespace or not isinstance(header["trajectory_id"], str)
            or not header["trajectory_id"]):
        raise ValueError("episode source namespace/trajectory differs")
    expected_representation = "graph" if role in LEARNED_ROLES else "reference"
    if header["representation"] != expected_representation:
        raise ValueError("controller representation mismatch")
    greedy = role in LEARNED_ROLES and (split == "test" or role == "own_frozen")
    selection = "reference" if role == "r4" else "anchor" if role == "full_mdl2" else "greedy" if greedy else "sample"
    if header["selection"] != selection:
        raise ValueError("recorded controller selection mode differs")
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
        if greedy and choice["class_index"] != int(np.argmax(evaluation["log_probs"])):
            raise ValueError("learned test did not use canonical greedy class")
        if role == "r4" and request != reference:
            raise ValueError("R4 original request changed")
        if role == "full_mdl2" and (request != anchor or reference != anchor):
            raise ValueError("full MDL-2 requires the full-anchor collector, not a restricted anchor")
        waiting += sum(vector(info["waiting_patients"], n))
        executed = executed_resources(info, n)
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
    return outcome | {
        "source_id": header["source_id"], "waiting_patient_steps": waiting,
        "waiting_definition": "sum_of_end_of_epoch_waiting_patients",
        "expiry_losses": sum(outcome["loss_causes"][k] for k in
                             ("patients_lost_waiting_expired", "patients_lost_transit_expired", "finished_expired")),
        **{f"terminal_{k}": terminal[k] for k in ("waiting", "production", "specimen_transit", "finished_return")},
        "actions": actions, "reference_corrections": corrections,
        "correction_reference": "full_mdl2" if role == "full_mdl2" else "contemporaneous_r4",
        "raw_header_sha256": json_hash(header),
        "target_receipt_diagnostics": {"status": "not_measured",
            "reason": "episode receipts do not establish the raw update targets or pre-normalization baseline diagnostics"},
        "compute_cost": {"environment_steps": len(rows), "evaluation_env_steps": len(rows) if split == "test" else 0,
                         "decision_path_seconds": outcome["inference_seconds"],
                         "training_compute": "not_in_episode_receipts_use_campaign_budget_ledger"},
    }


def _finite_report(value):
    if isinstance(value, float):
        finite(value, signed=True)
    elif isinstance(value, dict):
        for child in value.values():
            _finite_report(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite_report(child)


def read_time_baseline_raw_episodes(root, index, config):
    """Return (files, outcomes) for complete training/preflight/test episodes.

    The index contains header/events/final_state file records (path/bytes/sha256).
    A partial matrix is allowed; a partial episode, rewritten role, foreign source
    or wrong prescribed split seed is not. No side effects or numerical inference.
    """
    namespace, schedule = _environment_schedule(config)
    root = Path(root).resolve()
    paths = set()
    for entry in index:
        if set(entry) != {"header", "events", "final_state"}:
            raise ValueError("raw index entry inventory differs")
        for record in entry.values():
            path = record["path"]
            if (not isinstance(path, str) or not path or Path(path).is_absolute()
                    or Path(path).as_posix() != path or ".." in Path(path).parts or path in paths):
                raise ValueError("noncanonical or duplicate raw evidence path")
            paths.add(path)
            _sha(record["sha256"])
            if type(record["bytes"]) is not int or record["bytes"] <= 0:
                raise ValueError("positive raw byte count required")
    files, scalar = read_raw_episodes(root, index, config)
    outcomes, identities, slots = [], set(), set()
    for entry, original in zip(index, scalar):
        header = _reread(root, entry["header"])
        rows = _reread(root, entry["events"], jsonl=True)
        final = _reread(root, entry["final_state"])
        outcome = _enrich(original, header, rows, final, config, namespace, schedule)
        slot = tuple(outcome[k] for k in ("split", "block", "role", "world_index"))
        if outcome["trajectory_id"] in identities or slot in slots:
            raise ValueError("duplicate trajectory or episode slot")
        identities.add(outcome["trajectory_id"])
        slots.add(slot)
        outcomes.append(outcome | {"raw_files": copy.deepcopy(entry)})
    _finite_report(outcomes)
    return copy.deepcopy(files), outcomes


def _paired_analysis(outcomes, config, streams, worlds, seed):
    """Versioned copy of the frozen paired block/world bootstrap, with six roles."""
    blocks, evaluation = config["blocks"], config["evaluation"]
    expected = {(b, role, w) for b in blocks for role in CONTROLLERS for w in range(worlds)}
    indexed = {}
    for row in outcomes:
        key = row["block"], row["role"], row["world_index"]
        if key not in expected or key in indexed or row["split"] != "test":
            raise ValueError("duplicate or unexpected evaluation outcome")
        if (row["source_id"] != streams["namespace"]
                or row["seed"] != _decimal_seed(streams["environment"][str(row["block"])]["test"][row["world_index"]])):
            raise ValueError("wrong test source/stream")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("missing prescribed evaluation outcomes")
    for b in blocks:
        for w in range(worlds):
            if len({indexed[b, role, w]["initial_state_sha256"] for role in CONTROLLERS}) != 1:
                raise ValueError("paired controllers do not share the exact initial state")
        for role in CONTROLLERS:
            if len({indexed[b, role, w]["policy_sha256"] for w in range(worlds)}) != 1:
                raise ValueError("fixed final policy changed across test worlds")
    bootstrap = evaluation["bootstrap"]
    rng = np.random.default_rng(seed)
    # One shared design preserves pairing across metrics and contrasts. Repeated
    # copies of a sampled block independently draw their own within-block worlds.
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
        relative_ci = (np.quantile((100 * boot_delta[:, :, 0] / boot_baseline).mean(axis=1),
                                   [.025, .975]).tolist() if bool(valid.all()) else None)
        adverse = [{"block": row["block"], "metric": m, "mean_difference": row["mean_differences"][m]}
                   for row in block_rows for m, direction in PATIENT_DIRECTIONS.items()
                   if direction * row["mean_differences"][m] > 0]
        means = dict(zip(METRICS, delta.mean(axis=(0, 1)).tolist()))
        changed = sum(w["action_differences"]["requested_changed_steps"] for row in paired for w in row["worlds"])
        results.append({"name": name, "left": left, "right": right, "blocks": block_rows, "paired_worlds": paired,
                        "equal_block_mean_differences": means,
                        "equal_block_relative_cost_change_percent": relative_mean,
                        "hierarchical_paired_ci95": {m: interval[:, i].tolist() for i, m in enumerate(METRICS)},
                        "relative_cost_change_ci95": relative_ci,
                        "undefined_relative_bootstrap_draws": int((~valid).sum()),
                        "requested_changed_steps": changed,
                        "adverse_patient_block_directions": adverse,
                        "cost_patient_tradeoff": means["cost"] < 0 and bool(adverse)})
    primary, method, attribution = results[:3]
    relative = primary["equal_block_relative_cost_change_percent"]
    checks = {"all_three_primary_blocks_cost_favorable": all(r["mean_differences"]["cost"] < 0 for r in primary["blocks"]),
              "equal_block_relative_cost_reduction_at_least_one_percent": relative is not None and relative <= -1.0,
              "primary_relative_cost_defined": relative is not None,
              "primary_has_adverse_patient_direction": bool(primary["adverse_patient_block_directions"]),
              "primary_greedy_requests_changed": primary["requested_changed_steps"] > 0,
              "method_greedy_requests_changed": method["requested_changed_steps"] > 0}
    favorable = checks["all_three_primary_blocks_cost_favorable"] and checks["equal_block_relative_cost_reduction_at_least_one_percent"]
    close_reasons = []
    if not favorable:
        close_reasons.append("primary_development_screen_not_met")
    if not checks["primary_greedy_requests_changed"]:
        close_reasons.append("greedy_requests_unchanged_from_frozen")
    # The unchanged 1%/three-block primary screen is not a claim of method or
    # RL attribution: explicitly disclose a null/worse original-PPO comparator.
    if method["equal_block_mean_differences"]["cost"] >= 0:
        close_reasons.append("no_cost_benefit_over_current_ppo")
    if not checks["method_greedy_requests_changed"]:
        close_reasons.append("greedy_requests_unchanged_from_current_ppo")
    decision = ("cost_patient_tradeoff" if primary["cost_patient_tradeoff"] else
                "promising_development_signal" if favorable and not close_reasons else "limited_negative_or_inconclusive")
    return {"format": "time-baseline-independent-analysis-v1", "evaluation_episodes": len(outcomes),
            "independent_training_blocks": 3, "primary": primary["name"], "contrasts": results,
            "triage_checks": checks, "decision": decision,
            "primary_cost_screen_met": favorable,
            "decision_scope": "primary time-baseline-versus-frozen development screen; method and BC attribution are separate",
            "method_cost_direction": "favorable" if method["equal_block_mean_differences"]["cost"] < 0 else "null_or_unfavorable",
            "bc_attribution_cost_direction": "favorable" if attribution["equal_block_mean_differences"]["cost"] < 0 else "null_or_unfavorable",
            "baseline_route": {"close": bool(close_reasons), "reasons": close_reasons,
                "next": "reward_diagnosis_and_design_only" if close_reasons else "report_complete_comparison_for_discussion",
                "reward_training_authorized": False, "additional_baseline_attempt_authorized": False},
            "target_receipt_diagnostics": {"status": "not_measured",
                "reason": "requires separate raw training update-target receipts; not inferable from episode outcomes"},
            "bootstrap": {"kind": bootstrap["kind"], "draws": bootstrap["draws"], "seed": seed,
                          "blocks_per_draw": 3, "worlds_per_sampled_block": worlds,
                          "rng": "numpy.default_rng.PCG64", "numpy_version": np.__version__,
                          "quantile_method": "linear", "resample_design_sha256": json_hash([block_draws.tolist(), world_draws.tolist()]),
                          "descriptive_low_precision_three_blocks": True},
            "clinical_noninferiority_claim": False, "clinical_superiority_claim": False,
            "deployment_adaptation_claim": False, "clean_ddpg_superiority_claim": False,
            "isolated_graph_contribution_claim": False, "event_level_crn_equality_claim": False,
            "automatic_followon": False,
            "limitations": ["three training blocks; paired worlds are not independent trained policies",
                            "finite-window cost; unresolved terminal patients are not free benefit or deaths",
                            "queue occupancy integral is not mean completed-patient waiting time",
                            "same initial state does not imply event-level common random numbers after actions diverge",
                            "cross-controller action differences are closed-loop traces, not same-state counterfactual effects",
                            "raw episode reader verifies declared scenario seeds, not hidden scenario dynamics or source/runtime locks",
                            "full-anchor provenance depends on the source-frozen collector; no checkpoint inference is performed",
                            "training compute and all-model test seals require the campaign ledger/orchestrator",
                            "cost benefit versus frozen alone does not establish method benefit versus current PPO or attribution versus BC"]}


def verify_time_baseline_raw_bundle(root, index, config, streams):
    """Verify exactly 3 x 12 x 6 raw test episodes and return paired estimates.

    Reads only indexed JSON/JSONL evidence. Decimal seeds are parsed losslessly;
    training, preflight, optimizer receipts and learner summaries cannot replace
    final test entries. No missing world is dropped and no model is rescored.
    """
    worlds, seed = _validate_plan(config, streams)
    if len(index) != config["evaluation"]["total_episodes"]:
        raise ValueError("missing or extra prescribed raw evaluation entries")
    files, outcomes = read_time_baseline_raw_episodes(root, index, config)
    report = {"format": "time-baseline-raw-verification-v1", "files": copy.deepcopy(files),
              "index_sha256": json_hash(index), "config_sha256": json_hash(config), "streams_sha256": json_hash(streams),
              "outcomes": outcomes, "analysis": _paired_analysis(outcomes, config, streams, worlds, seed),
              "verification": "raw hashes, scalar accounting, continuous resources, six-role semantics and paired schedule verified",
              "scenario_seed_binding": {"source_id": streams["namespace"], "scenario": config["objective"]["scenario"],
                                        "all_declared_test_seeds_verified": True, "scenario_dynamics_reexecuted": False},
              "grants_scientific_execution_authorization": False}
    _finite_report(report)
    return report
