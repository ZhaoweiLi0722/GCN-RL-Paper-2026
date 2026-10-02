"""Independent raw readout for the paired-cost replacement package.

Only JSON evidence and metadata are consumed. No checkpoint, environment or
learner is constructed. Native role names are checked before shared arithmetic;
old campaign outcomes are never relabeled to satisfy this experiment's matrix.
"""

import copy
import math
from pathlib import Path

import numpy as np

from src.rl.candidate_pilot_verification import (
    CAUSES, OPERATING, PATIENT, close, count, finite, json_hash, vector, verify_episode,
)
from src.rl.cohort_bundle_verification import _check_index, _read_parts, _tail
from src.rl.dynamic_candidate_resource_verification import executed_resources
from src.rl.dynamic_candidate_verification import (
    METRICS as WINDOW_METRICS, PATIENT_DIRECTIONS, _action_difference, _sha,
)
from src.rl.paired_cohort_resources import stream_manifest
from src.rl.time_baseline_verification import _decimal_seed, _finite_report


LEARNED_ROLES = ("own_frozen", "paired_cost", "bc_continue", "saved_cohort_ppo")
CONTROLLERS = (*LEARNED_ROLES, "r4", "full_mdl2")
PRIMARY_CONTRASTS = ("paired_cost-minus-own_frozen", "paired_cost-minus-bc_continue")
SECONDARY_CONTRASTS = tuple("paired_cost-minus-" + r for r in ("saved_cohort_ppo", "r4", "full_mdl2"))
BOOTSTRAP = dict(kind="hierarchical_paired_block_then_world", draws=10000,
                 blocks_per_draw=3, worlds_per_sampled_block=12)
METRICS = (*WINDOW_METRICS, "average_turnaround_time", "turnaround_sum", "prefix_cost", "tail_cost")


def _validate_plan(config, streams, backend_config):
    if (config["schema"] != "paired-cohort-improvement-proposal-v1"
            or config["blocks"] != [60, 61, 62]
            or any(type(b) is not int for b in config["blocks"])
            or tuple(config["evaluation_controllers"]) != CONTROLLERS
            or tuple(config["primary_contrasts"]) != PRIMARY_CONTRASTS
            or tuple(config["secondary_contrasts"]) != SECONDARY_CONTRASTS
            or tuple(config["training_arms"]) != ("paired_cost", "bc_continue")):
        raise ValueError("native paired-cohort roles/contrasts/blocks required")
    for key, value in (("enrollment_steps", 52), ("economic_endpoint", 63),
                       ("test_worlds_per_block", 12), ("max_original_requests", 6)):
        if type(config[key]) is not int or config[key] != value:
            raise ValueError("fixed 52+11 step, twelve-world evaluation required")
    objective, tail = backend_config["objective"], backend_config["cohort_proposal"]
    n = objective["num_facilities"]
    if (backend_config["blocks"] != config["blocks"] or type(n) is not int or n < 1
            or objective["horizon"] != 52 or objective["action_width"] != 4 * n
            or backend_config["candidate_support"]["max_original_requests"] != 6
            or any(type(tail[k]) is not int or tail[k] != v for k, v in
                   (("enrollment_steps", 52), ("patient_resolution_steps", 8), ("accounting_steps", 11)))):
        raise ValueError("inherited raw objective/tail contract differs")
    if finite(objective["transfer_scale"]) <= 0:
        raise ValueError("positive specimen transfer scale required")
    # The proposal fixes the bootstrap in prose. An expanded runtime config may
    # repeat it, but cannot silently replace it with an inherited old estimator.
    bootstrap = config.get("evaluation", {}).get("bootstrap", BOOTSTRAP)
    if (bootstrap != BOOTSTRAP or any(type(bootstrap[k]) is not int for k in
                                    ("draws", "blocks_per_draw", "worlds_per_sampled_block"))):
        raise ValueError("prespecified 10000-draw hierarchical bootstrap differs")
    if set(streams["environment"]) != set(map(str, config["blocks"])):
        raise ValueError("exact stream block inventory required")
    layouts = {}
    for b, splits in streams["environment"].items():
        if len(splits["layout"]) != 1:
            raise ValueError("one immutable layout seed per block required")
        layouts[b] = str(_decimal_seed(splits["layout"][0]))
    expected = stream_manifest(config, layouts)
    for key in ("format", "namespace", "environment", "conditional_future", "neural", "bootstrap", "allocations"):
        if streams[key] != expected[key]:
            raise ValueError("prescribed paired-cohort stream schedule differs: " + key)
    return _decimal_seed(streams["bootstrap"])


def _header(header, config, streams):
    role = header["role"]
    if (role not in CONTROLLERS or header["split"] != "test"
            or any(type(header[k]) is not int for k in ("block", "world_index", "seed"))):
        raise ValueError("native evaluation role/test/integer lineage required")
    block, world = header["block"], header["world_index"]
    if (block not in config["blocks"] or not 0 <= world < config["test_worlds_per_block"]
            or header["source_id"] != streams["namespace"]
            or header["seed"] != _decimal_seed(streams["environment"][str(block)]["test"][world])):
        raise ValueError("evaluation source/world/seed differs from fresh schedule")
    selection = "greedy" if role in LEARNED_ROLES else "reference" if role == "r4" else "anchor"
    if (header["selection"] != selection
            or header["representation"] != ("graph" if role in LEARNED_ROLES else "reference")):
        raise ValueError("native controller representation/selection differs")
    if not isinstance(header["trajectory_id"], str) or not header["trajectory_id"]:
        raise ValueError("explicit trajectory identity required")
    _sha(header["policy_sha256"])


def _action(request, info, n, step):
    return dict(step_index=step, requested=copy.deepcopy(request),
                specimen_requested_integer_net=vector(info["specimen_requested_integer_net"], n, signed=True),
                executed=executed_resources(info, n))


def _tail_action(row, n, scale):
    request, info = row["action"], row["info"]
    if len(request) != 4 * n or any(abs(finite(v, signed=True)) > 1 for v in request):
        raise ValueError("invalid normalized tail request")
    scaled = np.asarray(request[:n], dtype=np.float64) * scale
    expected = (np.sign(scaled) * np.floor(np.abs(scaled) + .5)).astype(int).tolist()
    action = _action(request, info, n, 51 + row["index"])
    actual = action["executed"]["specimen_transfers"]
    routes = count(info["specimen_route_count"])
    if (action["specimen_requested_integer_net"] != expected
            or sum(max(v, 0) for v in actual) != routes
            or any((r >= 0 and not 0 <= a <= r) or (r < 0 and not r <= a <= 0)
                   for r, a in zip(expected, actual))):
        raise ValueError("tail requested/executed routing differs")
    inbound = sum(max(v, 0) for v in expected) - routes
    outbound = sum(max(-v, 0) for v in expected) - routes
    if (count(info["blocked_specimen_inbound_requests"]) != inbound
            or count(info["blocked_specimen_outbound_requests"]) != outbound
            or count(info["blocked_specimen_requests"]) != max(inbound, outbound)):
        raise ValueError("tail blocked requests differ")
    return action


def _turnaround(state, info, compartments):
    total = math.fsum(finite(p["age"]) for p in state["patients"].values() if p["status"] == "delivered")
    close(finite(state["scalars"]["cumulative_turnaround_time"]), total)
    mean = total / max(1, compartments["delivered"])
    close(finite(info["average_turnaround_time"]), mean)
    return dict(turnaround_sum=total, average_turnaround_time=mean)


def read_paired_cohort_raw_episodes(root, index, config, streams, *, backend_config):
    """Return ``(files, outcomes)`` from complete native test episodes.

    ``config`` is the new paired proposal; ``streams`` is the persisted decimal
    ``paired-cohort-streams-v1`` manifest, not its runtime integer translation.
    ``backend_config`` is inherited metadata (objective/candidate_support/
    cohort_proposal/blocks). ``index`` uses CohortRecorder's prefix/tail file
    records. A partial role/world matrix is allowed here, never a partial cohort.
    This function reads only the six files explicitly indexed per episode.
    """
    _validate_plan(config, streams, backend_config)
    root = Path(root).resolve()
    _check_index(index)
    outcomes, files, slots, identities = [], [], set(), set()
    n = backend_config["objective"]["num_facilities"]
    for entry in index:
        (header, rows, prefix), (tail_header, tail_rows, final) = _read_parts(root, entry)
        _header(header, config, streams)
        slot = tuple(header[k] for k in ("block", "role", "world_index"))
        if slot in slots or header["trajectory_id"] in identities:
            raise ValueError("duplicate trajectory or native evaluation slot")
        slots.add(slot)
        identities.add(header["trajectory_id"])
        window = verify_episode(header, rows, prefix, backend_config)
        actions, waiting = [], 0
        for t, row in enumerate(rows):
            audit, info = row["event"]["audit"], row["event"]["info"]
            evaluation, choice = audit["decision"]["evaluation"], audit["decision"]["choice"]
            request, bank = audit["record"]["action"], evaluation["candidates"]
            if header["role"] in LEARNED_ROLES and choice["class_index"] != int(np.argmax(evaluation["log_probs"])):
                raise ValueError("learned native evaluation must use canonical greedy tie rule")
            if header["role"] == "full_mdl2" and not request == bank["requests"][0] == bank["requests"][1]:
                raise ValueError("full MDL-2 prefix request differs")
            waiting += sum(vector(info["waiting_patients"], n))
            actions.append(_action(request, info, n, t))
        for field, compartment in (("waiting_patients", "waiting"), ("in_production_patients", "production"),
                                   ("specimen_in_transit", "specimen_transit")):
            if sum(vector(rows[-1]["event"]["info"][field], n)) != window["terminal_compartments"][compartment]:
                raise ValueError("prefix final queue differs")
        # Shared verifier checks each primitive; fsum retains float64 precision
        # when accumulating the full window rather than a rounded summary.
        window.update(cost=math.fsum(row["event"]["info"]["cost"] for row in rows),
            components={k: math.fsum(row["event"]["info"][k] for row in rows) for k in OPERATING + PATIENT},
            source_id=header["source_id"], actions=actions, waiting_patient_steps=waiting,
            **_turnaround(prefix, rows[-1]["event"]["info"], window["terminal_compartments"]))
        if (type(final["scalars"]["_episode_seed"]) is not int
                or final["scalars"]["_episode_seed"] != header["seed"]):
            raise ValueError("tail final world seed differs")
        tail = _tail(tail_header, prefix, final, tail_rows, backend_config, header, entry)
        tail["actions"] = [_tail_action(row, n, backend_config["objective"]["transfer_scale"]) for row in tail_rows]
        terminal = tail["final_compartments"]
        causes = {k: window["loss_causes"][k] + tail["loss_causes"][k] for k in CAUSES}
        outcome = window | dict(window=window, tail=tail, steps=63,
            prefix_cost=window["cost"], tail_cost=tail["tail_cost"],
            cost=math.fsum((window["cost"], tail["tail_cost"])),
            components={k: math.fsum((window["components"][k], tail["components"][k])) for k in OPERATING + PATIENT},
            losses=terminal["lost"], completions=terminal["delivered"], terminal_active=0,
            terminal_compartments=terminal, loss_causes=causes,
            completion_service_level=terminal["delivered"] / max(1, terminal["enrolled"]),
            waiting_patient_steps=waiting + tail["tail_waiting_patient_steps"],
            expiry_losses=sum(causes[k] for k in (CAUSES[2], CAUSES[4], CAUSES[7])),
            route_count=window["route_count"] + tail["route_count"],
            blocked_requests=window["blocked_requests"] + tail["blocked_requests"],
            actions=actions + tail["actions"],
            final_state_sha256=json_hash(final),
            raw_episode_sha256=json_hash([header, rows, prefix, tail_header, tail_rows, final]),
            raw_files=copy.deepcopy(entry),
            **_turnaround(final, tail_rows[-1]["info"], terminal),
            **{f"terminal_{k}": terminal[k] for k in ("waiting", "production", "specimen_transit", "finished_return")})
        outcomes.append(outcome)
        files.extend(copy.deepcopy(record) for part in entry.values() for record in part.values())
    _finite_report(outcomes)
    return files, outcomes


def _matrix(outcomes, config, streams):
    expected = {(b, r, w) for b in config["blocks"] for r in CONTROLLERS for w in range(12)}
    indexed = {}
    for row in outcomes:
        key = row["block"], row["role"], row["world_index"]
        if key in indexed or key not in expected or row["split"] != "test":
            raise ValueError("duplicate or unexpected evaluation outcome")
        if (row["source_id"] != streams["namespace"]
                or row["seed"] != _decimal_seed(streams["environment"][str(key[0])]["test"][key[2]])):
            raise ValueError("wrong test source/stream")
        if row["steps"] != 63 or len(row["actions"]) != 63 or len(row["window"]["actions"]) != 52:
            raise ValueError("whole63step action traces required")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("missing/extra complete 216-episode native evaluation inventory")
    for b in config["blocks"]:
        for w in range(12):
            if len({indexed[b, r, w]["initial_state_sha256"] for r in CONTROLLERS}) != 1:
                raise ValueError("controllers do not share exact paired starts")
        for r in CONTROLLERS:
            if len({indexed[b, r, w]["policy_sha256"] for w in range(12)}) != 1:
                raise ValueError("fixed final policy changed across test worlds")
    return indexed


def _mean_fields(rows, field, keys):
    return {k: math.fsum(r[field][k] for r in rows) / len(rows) for k in keys}


def _analysis(outcomes, config, streams, seed):
    indexed = _matrix(outcomes, config, streams)
    blocks = config["blocks"]
    rng = np.random.default_rng(seed)
    block_draws = rng.integers(0, 3, size=(10000, 3))
    world_draws = rng.integers(0, 12, size=(10000, 3, 12))
    contrasts = []
    for name in config["primary_contrasts"] + config["secondary_contrasts"]:
        left, right = name.split("-minus-")
        paired, block_rows, deltas, baselines = [], [], [], []
        for b in blocks:
            worlds, differences, baseline = [], [], []
            for w in range(12):
                lrow, rrow = indexed[b, left, w], indexed[b, right, w]
                diff = {m: lrow[m] - rrow[m] for m in METRICS}
                differences.append([diff[m] for m in METRICS])
                baseline.append(rrow["cost"])
                worlds.append(dict(world_index=w, seed=lrow["seed"],
                    initial_state_sha256=lrow["initial_state_sha256"],
                    left_raw_episode_sha256=lrow["raw_episode_sha256"], right_raw_episode_sha256=rrow["raw_episode_sha256"],
                    differences=diff,
                    component_differences={k: lrow["components"][k] - rrow["components"][k] for k in OPERATING + PATIENT},
                    loss_cause_differences={k: lrow["loss_causes"][k] - rrow["loss_causes"][k] for k in CAUSES},
                    action_differences=_action_difference(lrow, rrow),
                    prefix_action_differences=_action_difference(lrow["window"], rrow["window"]),
                    tail_action_differences=_action_difference(lrow["tail"], rrow["tail"])))
            mean, denominator = np.mean(differences, axis=0), float(np.mean(baseline))
            block_rows.append(dict(block=b, mean_differences=dict(zip(METRICS, mean.tolist())),
                baseline_mean_cost=denominator,
                component_differences=_mean_fields(worlds, "component_differences", OPERATING + PATIENT),
                loss_cause_differences=_mean_fields(worlds, "loss_cause_differences", CAUSES),
                relative_cost_change_percent=100 * float(mean[0]) / denominator if denominator > 0 else None))
            paired.append(dict(block=b, worlds=worlds))
            deltas.append(differences)
            baselines.append(baseline)
        delta, baseline = np.asarray(deltas, dtype=np.float64), np.asarray(baselines, dtype=np.float64)
        sampled = delta[block_draws[..., None], world_draws].mean(axis=2)
        sampled_base = baseline[block_draws[..., None], world_draws].mean(axis=2)
        interval = np.quantile(sampled.mean(axis=1), [.025, .975], axis=0)
        valid = np.all(sampled_base > 0, axis=1)
        relative_ci = (np.quantile((100 * sampled[:, :, 0] / sampled_base).mean(axis=1), [.025, .975]).tolist()
                       if bool(valid.all()) else None)
        relatives = [r["relative_cost_change_percent"] for r in block_rows]
        directions = PATIENT_DIRECTIONS | {"average_turnaround_time": 1}
        adverse = [dict(block=r["block"], metric=m, mean_difference=r["mean_differences"][m])
                   for r in block_rows for m, direction in directions.items() if direction * r["mean_differences"][m] > 0]
        changes = {scope + "_" + key: sum(w[scope + "_action_differences"][key] for p in paired for w in p["worlds"])
                   for scope in ("prefix", "tail") for key in ("requested_changed_steps", "executed_changed_steps")}
        means = dict(zip(METRICS, delta.mean(axis=(0, 1)).tolist()))
        contrasts.append(dict(name=name, left=left, right=right, blocks=block_rows, paired_worlds=paired,
            equal_block_mean_differences=means,
            equal_block_component_differences=_mean_fields(block_rows, "component_differences", OPERATING + PATIENT),
            equal_block_loss_cause_differences=_mean_fields(block_rows, "loss_cause_differences", CAUSES),
            equal_block_relative_cost_change_percent=float(np.mean(relatives)) if all(v is not None for v in relatives) else None,
            hierarchical_paired_ci95={m: interval[:, i].tolist() for i, m in enumerate(METRICS)},
            relative_cost_change_ci95=relative_ci, undefined_relative_bootstrap_draws=int((~valid).sum()),
            requested_changed_steps=changes["prefix_requested_changed_steps"] + changes["tail_requested_changed_steps"],
            executed_changed_steps=changes["prefix_executed_changed_steps"] + changes["tail_executed_changed_steps"],
            **changes, adverse_patient_block_directions=adverse, cost_patient_tradeoff=means["cost"] < 0 and bool(adverse)))
    primary = [r for r in contrasts if r["name"] in config["primary_contrasts"]]
    cost = all(b["mean_differences"]["cost"] < 0 for r in primary for b in r["blocks"])
    losses = all(b["mean_differences"]["losses"] <= 0 for r in primary for b in r["blocks"])
    unchanged = next(r for r in primary if r["right"] == "own_frozen")["prefix_requested_changed_steps"] == 0
    summaries = []
    for role in CONTROLLERS:
        per_block = []
        for b in blocks:
            rows = [indexed[b, role, w] for w in range(12)]
            per_block.append(dict(block=b, means={m: math.fsum(r[m] for r in rows) / 12 for m in METRICS},
                components=_mean_fields(rows, "components", OPERATING + PATIENT), loss_causes=_mean_fields(rows, "loss_causes", CAUSES)))
        summaries.append(dict(role=role, blocks=per_block, equal_block_means=_mean_fields(per_block, "means", METRICS),
            equal_block_components=_mean_fields(per_block, "components", OPERATING + PATIENT),
            equal_block_loss_causes=_mean_fields(per_block, "loss_causes", CAUSES)))
    return dict(format="paired-cohort-independent-comparison-v1", evaluation_episodes=216,
        independent_training_blocks=3, primary_contrasts=list(config["primary_contrasts"]),
        secondary_contrasts=list(config["secondary_contrasts"]), controllers=summaries, contrasts=contrasts,
        primary_cost_screen_met=cost, observed_loss_direction_screen_met=losses, promotion_screen_met=cost and losses,
        greedy_requests_unchanged_from_own_frozen=unchanged,
        decision="nominate_independent_confirmation" if cost and losses and not unchanged else "close_one_shot_mechanism",
        bootstrap=BOOTSTRAP | dict(seed=seed, rng="numpy.default_rng.PCG64", numpy_version=np.__version__,
            quantile_method="linear", resample_design_sha256=json_hash([block_draws.tolist(), world_draws.tolist()]),
            descriptive_low_precision_three_blocks=True),
        tradeoffs=[dict(contrast=r["name"], adverse_patient_block_directions=r["adverse_patient_block_directions"])
                  for r in contrasts if r["adverse_patient_block_directions"]],
        practical_effect_threshold_applied=False, clinical_noninferiority_claim=False, clinical_superiority_claim=False,
        equal_budget_ppo_superiority_claim=False, isolated_graph_contribution_claim=False,
        deployment_adaptation_claim=False, event_level_crn_equality_claim=False, automatic_followon=False,
        limitations=["descriptive intervals from three training blocks; worlds are not independent trained policies",
            "saved cohort PPO used different training data and compute; this tests a replacement learning package",
            "closed-loop requested/executed traces are not same-state counterfactual effects",
            "waiting occupancy is not completed-patient waiting time; retained resource stocks receive no terminal credit",
            "same initial state/seed is not event-aligned CRN after action-dependent random consumption",
            "source/runtime/model seals and historical seed overlap audit belong to the campaign binding",
            "simulator/optimizer/total wall-clock costs require the campaign ledger; raw inference time alone is not total compute"])


def verify_paired_cohort_raw_bundle(root, index, config, streams, *, backend_config):
    """Verify all 216 native test cohorts and report the fixed primary screen.

    Reads no historical outcomes, training targets or models. Model binding and
    the seal-before-test boundary are owned by the parent campaign. Historical
    freshness is not inferred from a namespace or a caller's boolean receipt.
    """
    seed = _validate_plan(config, streams, backend_config)
    files, outcomes = read_paired_cohort_raw_episodes(root, index, config, streams, backend_config=backend_config)
    report = dict(format="paired-cohort-raw-bundle-verification-v1", files=files, outcomes=outcomes,
        index_sha256=json_hash(index), config_sha256=json_hash(config), backend_config_sha256=json_hash(backend_config),
        streams_sha256=json_hash(streams), inventory={"test": len(outcomes)},
        analysis=_analysis(outcomes, config, streams, seed),
        scenario_seed_binding=dict(source_id=streams["namespace"], exact_declared_schedule_verified=True,
            historical_seed_freshness_verified_by_reader=False),
        training_inventory_verified=False, model_bindings_verified=False,
        compute_accounting=dict(status="requires_campaign_ledger", scientific_forward_calls=0, environment_calls=0, optimizer_calls=0),
        grants_scientific_execution_authorization=False)
    _finite_report(report)
    return report


def verify_paired_cohort_bundle(root, index, config, inherited_config, streams):
    """Runner API: new proposal, inherited scientific contract, new streams.

    The inherited config supplies only raw objective/candidate/tail dimensions;
    its historical roles, stream schedule and analysis settings are not used.
    Returns the raw verification report, including ``analysis.decision``.
    """
    return verify_paired_cohort_raw_bundle(root, index, config, streams, backend_config=inherited_config)
