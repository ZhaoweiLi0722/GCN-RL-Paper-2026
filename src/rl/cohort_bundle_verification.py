"""Independent two-part raw reader. No environment or learned-policy execution.

The public MDL-2 formula is evaluated on recorded observations only. Statistical
arithmetic reuses the frozen six-role bootstrap, not its superseded 1% screen.
"""

import copy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import numpy as np

from src.baselines.heuristics import facility_net_action_from_state, heuristic_settings_for_policy
from src.rl.candidate_pilot_verification import (
    CAUSES, OPERATING, PATIENT, close, count, finite, identity_counts, json_hash,
    vector, verify_episode,
)
from src.rl.cohort_verification import verify_cohort_tail
from src.rl.dynamic_candidate_resource_verification import continuous_vector, executed_resources
from src.rl.dynamic_candidate_verification import _reread, _sha
from src.rl.preprocessing import facility_state_width
from src.rl.time_baseline_verification import _decimal_seed, _finite_report, _paired_analysis


TRAINING_ROLES = ("window_ppo", "cohort_ppo", "bc_continue")
LEARNED_ROLES = ("own_frozen", *TRAINING_ROLES)
CONTROLLERS = (*LEARNED_ROLES, "r4", "full_mdl2")
RESOURCE_FIELDS = ("reagents", "bioreactors", "reagent_transfer_pipeline",
                   "capacity_transfer_pipeline", "reagent_purchase_pipeline")


def _schedule(config):
    p, obj = config["cohort_proposal"], config["objective"]
    if (p["schema"] != "cohort-objective-proposal-v1" or config["blocks"] != [60, 61, 62]
            or p["blocks"] != config["blocks"] or config["representations"] != ["graph"]
            or tuple(p["training_arms"]) != TRAINING_ROLES
            or tuple(p["evaluation_controllers"]) != CONTROLLERS
            or tuple(config["final_controllers"]) != CONTROLLERS
            or p["episodes_per_training_arm_block"] != 32 or p["test_worlds_per_block"] != 12
            or p["preflight_worlds_per_block"] != 1 or obj["horizon"] != p["enrollment_steps"]
            or obj["action_width"] != 4 * obj["num_facilities"]
            or p["economic_endpoint"] != p["enrollment_steps"] + p["accounting_steps"]):
        raise ValueError("cohort six-controller dimensions differ")
    if any(type(p[k]) is not int or p[k] < 1 for k in
           ("enrollment_steps", "patient_resolution_steps", "accounting_steps")):
        raise ValueError("positive fixed cohort windows required")
    namespace = p["rng_namespace"]
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("explicit cohort namespace required")
    base = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:12], "big") << 16
    schedule = {}
    for i, b in enumerate(config["blocks"]):
        schedule[str(b)] = {"layout": [base + i], "preflight": [base + 3 + i],
            "training": list(range(base + 6 + i * 32, base + 6 + (i + 1) * 32)),
            "test": list(range(base + 102 + i * 12, base + 102 + (i + 1) * 12))}
    return namespace, schedule


def _validate_plan(config, streams):
    namespace, schedule = _schedule(config)
    if (streams["format"] != "cohort-objective-streams-v1"
            or streams["namespace"] != namespace
            or streams["proposal_sha256"] != json_hash(config["cohort_proposal"])
            or set(streams["environment"]) != set(schedule)):
        raise ValueError("cohort streams/proposal binding differs")
    for b, splits in schedule.items():
        if set(streams["environment"][b]) != set(splits):
            raise ValueError("cohort stream split inventory differs")
        for split, seeds in splits.items():
            if [_decimal_seed(s) for s in streams["environment"][b][split]] != seeds:
                raise ValueError("prescribed cohort seed differs")
    seed = _decimal_seed(streams["neural"]["analysis/bootstrap"])
    expected = int.from_bytes(hashlib.sha256((namespace + "/analysis/bootstrap").encode()).digest()[:8], "big") % 2**63
    boot = config["evaluation"]["bootstrap"]
    if (seed != expected or boot["kind"] != "hierarchical_paired_block_then_world"
            or type(boot["draws"]) is not int or boot["draws"] != 10000
            or boot["blocks_per_draw"] != 3 or boot["worlds_per_sampled_block"] != 12
            or config["evaluation"]["total_episodes"] != 216):
        raise ValueError("prespecified cohort bootstrap/evaluation differs")
    return seed


def _check_index(index):
    paths = set()
    for entry in index:
        if set(entry) != {"prefix", "tail"}:
            raise ValueError("complete two-part cohort index required")
        for part in entry.values():
            if set(part) != {"header", "events", "final_state"}:
                raise ValueError("raw part file inventory differs")
            for record in part.values():
                path = record["path"]
                if (not isinstance(path, str) or not path or Path(path).is_absolute()
                        or Path(path).as_posix() != path or ".." in Path(path).parts or path in paths):
                    raise ValueError("noncanonical or duplicate raw evidence path")
                _sha(record["sha256"])
                if type(record["bytes"]) is not int or record["bytes"] <= 0:
                    raise ValueError("positive raw byte count required")
                paths.add(path)


def _read_parts(root, entry):
    return tuple(tuple(_reread(root, part[k], jsonl=k == "events")
                       for k in ("header", "events", "final_state"))
                 for part in (entry["prefix"], entry["tail"]))


def _prefix(header, rows, final, config, namespace, schedule):
    # The shared scalar reader restricts patient/specimen counts only. Continuous
    # resource channels below use the corrected reader, never the old _enrich.
    out = verify_episode(header, rows, final, config)
    role, split = header["role"], header["split"]
    roles = TRAINING_ROLES if split == "training" else CONTROLLERS if split in ("test", "preflight") else ()
    if role not in roles or any(type(header[k]) is not int for k in ("block", "world_index", "seed")):
        raise ValueError("unexpected cohort controller/split or lineage")
    b, w = str(header["block"]), header["world_index"]
    if (b not in schedule or not 0 <= w < len(schedule[b][split])
            or header["seed"] != schedule[b][split][w] or header["source_id"] != namespace):
        raise ValueError("cohort episode source/seed differs")
    greedy = role in LEARNED_ROLES and (split == "test" or role == "own_frozen")
    selection = "reference" if role == "r4" else "anchor" if role == "full_mdl2" else "greedy" if greedy else "sample"
    if (header["selection"] != selection
            or header["representation"] != ("graph" if role in LEARNED_ROLES else "reference")):
        raise ValueError("cohort representation/selection differs")
    _sha(header["policy_sha256"])
    n = config["objective"]["num_facilities"]
    actions, waiting = [], 0
    for t, row in enumerate(rows):
        event = row["event"]
        audit, info = event["audit"], event["info"]
        evaluation, choice = audit["decision"]["evaluation"], audit["decision"]["choice"]
        request, bank = audit["record"]["action"], evaluation["candidates"]
        if greedy and choice["class_index"] != int(np.argmax(evaluation["log_probs"])):
            raise ValueError("learned cohort evaluation is not canonical greedy")
        if role == "r4" and request != bank["requests"][0]:
            raise ValueError("R4 request differs")
        if role == "full_mdl2" and not request == bank["requests"][0] == bank["requests"][1]:
            raise ValueError("full MDL-2 prefix request differs")
        waiting += sum(vector(info["waiting_patients"], n))
        actions.append(dict(step_index=t, requested=copy.deepcopy(request),
            specimen_requested_integer_net=vector(info["specimen_requested_integer_net"], n, signed=True),
            executed=executed_resources(info, n)))
    for field, compartment in (("waiting_patients", "waiting"), ("in_production_patients", "production"),
                               ("specimen_in_transit", "specimen_transit")):
        if sum(vector(rows[-1]["event"]["info"][field], n)) != out["terminal_compartments"][compartment]:
            raise ValueError("prefix final queue differs")
    return out | dict(source_id=namespace, actions=actions, waiting_patient_steps=waiting)


def _array(value, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("invalid resource stock shape/value")
    return result


def _equal_arrays(left, right):
    if np.shape(left) != np.shape(right) or not np.allclose(left, right, rtol=1e-12, atol=1e-8):
        raise ValueError("primitive resource flow differs")


def _snapshot_resources(state, fields, n):
    # The live engine always has a (1,n) zero procurement buffer when disabled;
    # its backward-compatible state_dict deliberately omits that buffer.
    return {k: state["arrays"][k] if k in state["arrays"] else [[0.] * n]
            for k in fields if k in state["arrays"] or k == "reagent_purchase_pipeline"}


def _resource_step(row, anchor, n):
    before, after, info = row["resources_before"], row["resources_after"], row["info"]
    required = set(RESOURCE_FIELDS[:4])
    if not required <= set(before) or set(before) != set(after) or set(before) - set(RESOURCE_FIELDS):
        raise ValueError("complete primitive resource inventory required")
    if (anchor.get("enable_overtime_control", False) or anchor.get("enable_stochastic_procurement", False)
            or anchor.get("reagent_purchase_lead_time", 0)):
        raise ValueError("resource reader requires unchanged no-overtime immediate-procurement contract")
    for state in (before, after):
        if "reagent_purchase_pipeline" in state and np.any(_array(state["reagent_purchase_pipeline"])):
            raise ValueError("unexpected procurement pipeline")
    executed = executed_resources(info, n)
    production = np.asarray(vector(info["production"], n), dtype=float)
    delayed = anchor["transfer_lead_time"] > 0
    report = {}
    for stock, pipeline, netkey, arrivalkey, capkey in (
            ("reagents", "reagent_transfer_pipeline", "reagent_transfers", "reagent_transfer_arrivals", "max_reagents"),
            ("bioreactors", "capacity_transfer_pipeline", "capacity_transfers", "capacity_transfer_arrivals", "max_idle_bioreactors")):
        old, new = _array(before[stock]), _array(after[stock])
        if stock == "bioreactors":
            if old.shape != (n, anchor["production_lead_time"]) or new.shape != old.shape:
                raise ValueError("reactor stage layout differs")
            _equal_arrays(new[:, 1:].sum(axis=1), vector(info["in_production_patients"], n))
            onhand, endpoint = old[:, 0], new[:, 0]
        else:
            if old.shape != (n,) or new.shape != (n,):
                raise ValueError("reagent stock layout differs")
            onhand, endpoint = old, new
        pipe, endpipe = _array(before[pipeline]), _array(after[pipeline])
        if pipe.shape != endpipe.shape or pipe.ndim != 2 or pipe.shape[1] != n:
            raise ValueError("transfer pipeline layout differs")
        arrivals = np.asarray(continuous_vector(info[arrivalkey], n))
        _equal_arrays(arrivals, pipe[0] if len(pipe) else np.zeros(n))
        net = np.asarray(executed[netkey])
        _equal_arrays(endpipe.sum(axis=0), pipe.sum(axis=0) - arrivals + (np.maximum(net, 0) if delayed else 0))
        cap = np.broadcast_to(_array(anchor[capkey]), (n,))
        arrival_overflow = np.maximum(onhand + arrivals - cap, 0)
        available = np.minimum(onhand + arrivals, cap)
        if (production > available + 1e-8).any():
            raise ValueError("production exceeds available resource")
        change = (np.asarray(executed["replenishment"]) - production if stock == "reagents"
                  else old[:, 1:].sum(axis=1) - new[:, 1:].sum(axis=1))
        unbounded = available + change + (np.minimum(net, 0) if delayed else net)
        if (unbounded < -1e-8).any():
            raise ValueError("negative resource balance")
        end_overflow = np.maximum(unbounded - cap, 0)
        _equal_arrays(endpoint, np.clip(unbounded, 0, cap))
        report[stock] = dict(arrival_clipped_overflow=arrival_overflow.tolist(),
                             step_clipped_overflow=end_overflow.tolist())
    return report


def _tail(header, prefix, final, rows, config, prefix_header, entry):
    p, n = config["cohort_proposal"], config["objective"]["num_facilities"]
    contract = {k: p[k] for k in ("enrollment_steps", "patient_resolution_steps", "accounting_steps")}
    contract["followup_rule"] = "full_mdl2_until_resolution_then_no_new_commitments"
    if (header["format"] != "cohort-tail-header-v1" or header["contract"] != contract
            or header["prefix"] != entry["prefix"]
            or any(header[k] != prefix_header[k] for k in
                   ("block", "representation", "role", "world_index", "seed", "trajectory_id", "source_id", "split"))):
        raise ValueError("tail header/contract/prefix lineage differs")
    anchor = header["anchor_config"]
    settings = heuristic_settings_for_policy("mdl2")
    if (anchor["num_facilities"] != n or anchor["episode_horizon"] != p["enrollment_steps"]
            or json_hash(dict(anchor_config=anchor, settings=asdict(settings))) !=
               prefix_header["session_manifest"]["producer_definition"]):
        raise ValueError("tail anchor source differs from original prefix producer")
    receipt = dict(prefix_final_sha256=header["prefix_final_sha256"], final_sha256=json_hash(final))
    out = verify_cohort_tail(prefix, final, rows, receipt, num_facilities=n,
                            **{k: p[k] for k in contract if k != "followup_rule"})
    closed = copy.deepcopy(prefix)
    for key in ("demand", "demand_forecast"):
        closed["arrays"][key] = [0.] * n
    closed["scalars"]["demand_forecast_error"] = 0.
    token, active = json_hash(closed), identity_counts(prefix)["active"]
    resources = _snapshot_resources(prefix, rows[0]["resources_before"], n)
    overlay = copy.deepcopy(anchor)
    overlay.update(demand_rates=[0.] * n, demand_rate_estimates=[0.] * n)
    causes, flows = dict.fromkeys(CAUSES, 0), []
    routes = blocked = 0
    for row in rows:
        _sha(row["before_state_sha256"])
        _sha(row["after_state_sha256"])
        if row["before_state_sha256"] != token or row["resources_before"] != resources:
            raise ValueError("broken tail state/resource chain")
        observation = np.asarray(row["public_observation"], dtype=np.float32)
        width, lead = facility_state_width(anchor), anchor["production_lead_time"]
        if observation.ndim != 1 or observation.size < n * width or not np.isfinite(observation).all():
            raise ValueError("finite flat public tail observation required")
        public = observation[:n * width].reshape(n, width)
        if np.any(public[:, 0]):
            raise ValueError("nonzero public post-window demand")
        if anchor.get("include_demand_forecast_state", False):
            if np.any(public[:, 3 + lead + int(anchor.get("include_supplier_state", False))]):
                raise ValueError("nonzero public post-window forecast")
        if (not np.array_equal(public[:, 2], np.asarray(resources["reagents"], dtype=np.float32))
                or not np.array_equal(public[:, 3:3 + lead], np.asarray(resources["bioreactors"], dtype=np.float32))):
            raise ValueError("public observation differs from resource stocks")
        expected = (facility_net_action_from_state(observation, overlay, settings=settings).tolist()
                    if active else [0.] * (3 * n) + [-1.] * n)
        if row["action"] != expected:
            raise ValueError("tail must use common zero-rate full MDL-2 or resolved idle request")
        info = row["info"]
        supplier = public[:, 3 + lead] if anchor.get("include_supplier_state", False) else np.ones(n)
        requested_orders = ((np.asarray(row["action"])[3 * n:] + 1.) / 2.) * np.asarray(anchor["max_reagent_replenishment"]) * supplier
        _equal_arrays(continuous_vector(info["replenishment"], n), requested_orders)
        if any(k.endswith("_cost") and k not in OPERATING + PATIENT +
               ("base_cost", "specimen_route_cost", "transshipment_cost") for k in info):
            raise ValueError("unknown tail cost component")
        close(info["specimen_route_cost"], info["specimen_transfer_cost"])
        close(info["transshipment_cost"], sum(info[k] for k in OPERATING[-3:]))
        loss = sum(vector(info["patients_lost"], n))
        per_cause = {k: sum(vector(info[k], n)) for k in CAUSES}
        if sum(per_cause.values()) != loss:
            raise ValueError("tail loss causes do not reconcile")
        for k, value in per_cause.items():
            causes[k] += value
        if count(info["identity_terminal_count"]) + row["active"] != out["final_compartments"]["enrolled"]:
            raise ValueError("tail terminal identity count differs")
        flows.append(_resource_step(row, anchor, n))
        routes += count(info["specimen_route_count"])
        blocked += count(info["blocked_specimen_requests"])
        token, active, resources = row["after_state_sha256"], row["active"], row["resources_after"]
    if token != json_hash(final) or resources != _snapshot_resources(final, resources, n):
        raise ValueError("tail final state/resource hash differs")
    close(finite(rows[-1]["info"]["completion_service_level"]),
          out["final_compartments"]["delivered"] / max(1, out["final_compartments"]["enrolled"]))
    return out | dict(loss_causes=causes, resource_flow=flows, route_count=routes, blocked_requests=blocked,
        clipping_disclosed=True, fractional_resources_preserved=True,
        resource_flow_scope="per-facility stocks, pipeline totals and arrivals; no simulator replay")


def read_cohort_raw_episodes(root, index, config):
    """Read complete episodes of any split; allow a partial matrix, never a tail."""
    root = Path(root).resolve()
    namespace, schedule = _schedule(config)
    _check_index(index)
    outcomes, files, slots, ids = [], [], set(), set()
    for entry in index:
        (header, rows, prefix), (tail_header, tail_rows, final) = _read_parts(root, entry)
        window = _prefix(header, rows, prefix, config, namespace, schedule)
        tail = _tail(tail_header, prefix, final, tail_rows, config, header, entry)
        slot = tuple(header[k] for k in ("split", "block", "role", "world_index"))
        if slot in slots or header["trajectory_id"] in ids:
            raise ValueError("duplicate trajectory or episode slot")
        slots.add(slot)
        ids.add(header["trajectory_id"])
        terminal = tail["final_compartments"]
        loss_causes = {k: window["loss_causes"][k] + tail["loss_causes"][k] for k in CAUSES}
        outcome = window | dict(window=copy.deepcopy(window), tail=tail,
            steps=window["steps"] + len(tail_rows), cost=math.fsum((window["cost"], tail["tail_cost"])),
            components={k: math.fsum((window["components"][k], tail["components"][k])) for k in OPERATING + PATIENT},
            losses=terminal["lost"], completions=terminal["delivered"], terminal_active=0,
            terminal_compartments=terminal, loss_causes=loss_causes,
            completion_service_level=terminal["delivered"] / max(1, terminal["enrolled"]),
            waiting_patient_steps=window["waiting_patient_steps"] + tail["tail_waiting_patient_steps"],
            expiry_losses=sum(loss_causes[k] for k in (CAUSES[2], CAUSES[4], CAUSES[7])),
            route_count=window["route_count"] + tail["route_count"],
            blocked_requests=window["blocked_requests"] + tail["blocked_requests"],
            final_state_sha256=json_hash(final), raw_episode_sha256=json_hash([header, rows, prefix, tail_header, tail_rows, final]),
            raw_files=copy.deepcopy(entry), average_turnaround_time=finite(tail_rows[-1]["info"]["average_turnaround_time"]),
            **{f"terminal_{k}": terminal[k] for k in ("waiting", "production", "specimen_transit", "finished_return")})
        files.extend(copy.deepcopy(record) for part in entry.values() for record in part.values())
        outcomes.append(outcome)
    _finite_report(outcomes)
    return files, outcomes


def _analysis(outcomes, config, streams, seed):
    # Adapt only verified analysis rows to reuse the exact existing estimator.
    # Never rewrite raw evidence, module globals, or retain its old decision rule.
    aliases = {"window_ppo": "current_ppo", "cohort_ppo": "time_baseline_ppo"}
    rows = [row | {"role": aliases.get(row["role"], row["role"])} for row in outcomes]
    old = _paired_analysis(rows, config, streams, 12, seed)
    inverse = {v: k for k, v in aliases.items()}
    contrasts = old["contrasts"]
    for row in contrasts:
        for field in ("left", "right"):
            row[field] = inverse.get(row[field], row[field])
        row["name"] = row["left"] + "_minus_" + row["right"]
    primary = contrasts[:3]
    cost = all(row["equal_block_mean_differences"]["cost"] < 0
               and all(b["mean_differences"]["cost"] < 0 for b in row["blocks"]) for row in primary)
    losses = all(b["mean_differences"]["losses"] <= 0 for row in primary for b in row["blocks"])
    return dict(format="cohort-independent-analysis-v1", evaluation_episodes=len(outcomes),
        independent_training_blocks=3, primary_contrasts=[r["name"] for r in primary],
        contrasts=contrasts, bootstrap=old["bootstrap"], primary_cost_screen_met=cost,
        observed_loss_direction_screen_met=losses, promotion_screen_met=cost and losses,
        decision="nominate_independent_confirmation" if cost and losses else "close_one_shot_route",
        clinical_noninferiority_claim=False, clinical_superiority_claim=False,
        event_level_crn_equality_claim=False, automatic_followon=False,
        practical_effect_threshold_applied=False,
        tradeoffs=[dict(contrast=r["name"], adverse_patient_block_directions=r["adverse_patient_block_directions"])
                  for r in contrasts if r["adverse_patient_block_directions"]],
        limitations=["descriptive intervals from three independent training blocks",
            "fixed economic endpoint, retained stocks without terminal credit",
            "prefix action differences are closed-loop traces, not same-state counterfactuals",
            "source/runtime locks, authorization and pre-test model seals belong to campaign verification"])


def verify_cohort_raw_bundle(root, index, config, streams):
    """Verify a 216-test bundle or the complete 522-episode campaign inventory.

    A mixed/partial campaign inventory is rejected. Target receipts and serialized
    training kernels are deliberately handled by verify_cohort_targets.
    """
    seed = _validate_plan(config, streams)
    files, outcomes = read_cohort_raw_episodes(root, index, config)
    splits = {row["split"] for row in outcomes}
    required = {"test"} if splits == {"test"} else {"training", "preflight", "test"}
    expected = {(split, b, role, w) for split in required for b in config["blocks"]
                for role in (TRAINING_ROLES if split == "training" else CONTROLLERS)
                for w in range({"training": 32, "preflight": 1, "test": 12}[split])}
    actual = {tuple(row[k] for k in ("split", "block", "role", "world_index")) for row in outcomes}
    if actual != expected:
        raise ValueError("missing/extra complete cohort split inventory")
    for split in required:
        subset = [r for r in outcomes if r["split"] == split]
        for b in config["blocks"]:
            for w in {r["world_index"] for r in subset if r["block"] == b}:
                if len({r["initial_state_sha256"] for r in subset if r["block"] == b and r["world_index"] == w}) != 1:
                    raise ValueError("controllers do not share exact paired starts")
    test = [r for r in outcomes if r["split"] == "test"]
    report = dict(format="cohort-raw-bundle-verification-v1", files=files, outcomes=outcomes,
        index_sha256=json_hash(index), config_sha256=json_hash(config), streams_sha256=json_hash(streams),
        analysis=_analysis(test, config, streams, seed),
        inventory={split: sum(r["split"] == split for r in outcomes) for split in sorted(required)},
        training_inventory_verified="training" in required, grants_scientific_execution_authorization=False)
    _finite_report(report)
    return report
