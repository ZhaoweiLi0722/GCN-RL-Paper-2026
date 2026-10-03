"""Reconstruct immutable teacher evidence inside the caller's admitted budget.

No file loading, controller migration, native steps, or neural execution occurs
here. Only the filter is replayed; the caller supplies every accounting hook.
"""

import copy
from dataclasses import fields
import json
import math
from types import SimpleNamespace

import numpy as np

from src.baselines.capacity_completion_control import CompletionIntervalFilter
from src.env.patient_support_public import (
    PublicPatientRecord, PublicSupportOperations, PublicSupportServiceEvent,
)
from src.rl.capacity_ddpg_learner import CapacityTransition
from src.rl.capacity_pilot_runner import COST_KEYS, jsonable, trajectory_id
from src.rl.capacity_public_features import public_features
from src.rl.public_support_collector import PublicSupportInput
from src.rl.public_support_input import PublicSupportControlInput


def _canonical(value):
    return json.dumps(jsonable(value), sort_keys=True, allow_nan=False, separators=(",", ":"))


def _fields(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"invalid saved {label} fields")


def decode_public(value: dict) -> PublicSupportControlInput:
    """Decode only the public schema, validating it with the existing collector.

    The lightweight adapter supplies saved values, never a native environment.
    Nested dataclasses and all sequence fields are restored without aliasing.
    """
    value = copy.deepcopy(value)
    _fields(value, ("common", "operations"), "public input")
    common, operations = value["common"], value["operations"]
    _fields(common, (f.name for f in fields(PublicSupportInput)), "common input")
    _fields(operations, (f.name for f in fields(PublicSupportOperations)), "operations")
    patients = []
    for patient in operations["patients"]:
        _fields(patient, (f.name for f in fields(PublicPatientRecord)), "patient")
        patients.append(PublicPatientRecord(**patient))
    event = operations["last_service"]
    if event is not None:
        _fields(event, (f.name for f in fields(PublicSupportServiceEvent)), "service")
        event = PublicSupportServiceEvent(**event)
    operations["patients"], operations["last_service"] = tuple(patients), event
    typed_operations = PublicSupportOperations(**operations)
    adapter = SimpleNamespace(
        observation=lambda: common["base_observation"],
        public_capacity=lambda: dict(
            epoch=common["epoch"], site_ids=common["site_ids"],
            history=common["capacity_history"], pending_hours=common["pending_hours"],
            ready_waiting_counts=common["ready_waiting_counts"]),
        public_operations=lambda: typed_operations,
    )
    return PublicSupportControlInput.capture(adapter)


def _end_public(state, proposal, epoch):
    _fields(state, ("format", "filter", "lifecycle", "last_plan"), "controller")
    if state["format"] != "capacity-completion-control-recovery1":
        raise ValueError("saved controller recovery format mismatch")
    saved_filter = state["filter"]
    if (saved_filter["format"] != "completion-interval-filter-v1"
            or _canonical(saved_filter["proposal"]) != _canonical(proposal)
            or type(saved_filter["epoch"]) is not int or saved_filter["epoch"] != epoch
            or state["lifecycle"]["epoch"] != epoch):
        raise ValueError("saved controller proposal/epoch mismatch")
    view = decode_public(saved_filter["last_view"])
    if view.common.epoch != epoch:
        raise ValueError("saved final public epoch mismatch")
    return view


def _number(value, label, *, nonnegative=True):
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"invalid saved {label}")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"invalid saved {label}")
    return result


def _hours(values, size, label):
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != size:
        raise ValueError(f"invalid saved {label} dimensions")
    return tuple(_number(v, label) for v in values)


def _validate_rows(rows, end, proposal):
    system, design = proposal["synthetic_system"], proposal["design"]
    control, horizon = system["control_epochs"], system["host_total_horizon"]
    if (control, system["settlement_epochs"], horizon) != (48, 16, 64):
        raise ValueError("saved teacher requires the fixed 48+16 contract")
    if not isinstance(rows, list) or len(rows) not in (control - 1, horizon):
        raise ValueError("saved teacher requires exactly 47 or 64 ordered rows")
    required = {"epoch", "world", "role", "cost", "components", "reward",
                "requested_hours", "executed_hours", "public_input", "filter_summary",
                "filter_resets", "info", "patient_records", "service"}
    if any(not isinstance(row, dict) or not required.issubset(row) for row in rows):
        raise ValueError("missing saved teacher row fields")
    world = rows[0]["world"]
    _fields(world, ("phase", "block", "condition", "replicate", "seed"), "world")
    for key, upper in (("block", design["blocks"]), ("condition", design["conditions"]),
                       ("replicate", design["worlds_per_condition_per_phase_block"])):
        if type(world[key]) is not int or not 0 <= world[key] < upper:
            raise ValueError("invalid saved teacher world index")
    if (world["phase"] != "teacher_bc_critic_warmup" or type(world["seed"]) is not int
            or world["seed"] != 62600000 + 1000 * world["block"]
            + 10 * world["condition"] + world["replicate"]):
        raise ValueError("invalid saved teacher phase/seed")
    views, costs = [], []
    n = len(system["site_ids"])
    for epoch, row in enumerate(rows):
        if (type(row["epoch"]) is not int or row["epoch"] != epoch
                or _canonical(row["world"]) != _canonical(world) or row["role"] != "teacher"):
            raise ValueError("saved teacher order/world/role mismatch")
        view = decode_public(row["public_input"])
        if view.common.epoch != epoch or list(view.common.site_ids) != system["site_ids"]:
            raise ValueError("saved teacher public boundary mismatch")
        cost = _number(row["cost"], "cost")
        reward = _number(row["reward"], "reward", nonnegative=False)
        _fields(row["components"], COST_KEYS, "cost components")
        components = [_number(row["components"][k], "cost component") for k in COST_KEYS]
        if (not math.isclose(math.fsum(components), cost, rel_tol=1e-12, abs_tol=1e-6)
                or not math.isclose(-reward, cost, rel_tol=1e-12, abs_tol=1e-6)):
            raise ValueError("saved teacher cost/components/reward mismatch")
        requested = _hours(row["requested_hours"], n, "requested hours")
        executed = _hours(row["executed_hours"], n, "executed hours")
        caps = system["support"]["site_hour_caps"]
        if (any(x > cap + 1e-8 for x, cap in zip(executed, caps))
                or math.fsum(executed) > system["support"]["shared_hour_budget"] + 1e-8):
            raise ValueError("saved executed hours exceed capacity")
        if epoch >= control and (any(requested) or any(executed)):
            raise ValueError("saved fixed tail contains a new commitment")
        receipt = row["info"]["support_public_receipt"]
        if (receipt["epoch"] != epoch or receipt["known_at"] != epoch + 1
                or _hours(receipt["raw_requested_hours"], n, "receipt requested hours") != requested
                or _hours(receipt["committed_hours"], n, "receipt executed hours") != executed
                or _number(row["info"]["cost"], "info cost") != cost):
            raise ValueError("saved teacher action receipt mismatch")
        summaries = np.asarray(row["filter_summary"], dtype=float)
        if (summaries.shape != (n, 9) or not np.isfinite(summaries).all()
                or type(row["filter_resets"]) is not int or row["filter_resets"] < 0):
            raise ValueError("invalid saved filter summary/reset count")
        views.append(view)
        costs.append(cost)
    if list(end.common.site_ids) != system["site_ids"]:
        raise ValueError("saved final public sites mismatch")
    views.append(end)
    for row, next_view in zip(rows, views[1:]):
        if (_canonical(row["service"]) != _canonical(next_view.operations.last_service)
                or _canonical(row["patient_records"]) != _canonical(next_view.operations.patients)):
            raise ValueError("saved next-public service/patient mismatch")
        if next_view.common.epoch > control and any(next_view.operations.current_arrivals):
            raise ValueError("saved fixed tail contains new arrivals")
    return views, costs, trajectory_id(world, "teacher")


def reconstruct_teacher(rows: list, end_controller_state: dict, proposal, *,
                        before_receipt, before_filter, after_receipt,
                        filter_factory=None) -> list[CapacityTransition]:
    """Return 48 settled transitions or the 47 nonterminal prefix transitions.

    before_receipt() admits each epoch > 0 before filter.update; after_receipt()
    closes its prepaid chunk only after that update returns successfully.
    before_filter(payload) is forwarded unchanged to every filter primitive.
    Epoch zero is free. No failed update is retried or refunded here.

    filter_factory is for artificial tests; production defaults to the unchanged
    CompletionIntervalFilter(proposal, scientific=True). Inputs are not mutated.
    """
    if not all(callable(hook) for hook in (before_receipt, before_filter, after_receipt)):
        raise TypeError("all saved-reconstruction accounting hooks are required")
    if not isinstance(rows, list) or len(rows) not in (47, 64):
        raise ValueError("saved teacher requires exactly 47 or 64 ordered rows")
    end = _end_public(end_controller_state, proposal, len(rows))
    views, costs, identifier = _validate_rows(rows, end, proposal)
    factory = CompletionIntervalFilter if filter_factory is None else filter_factory
    interval_filter = factory(copy.deepcopy(proposal), scientific=True)
    states = []
    for epoch, view in enumerate(views):
        if epoch:
            before_receipt()
        interval_filter.update(view, before_filter=before_filter)
        if epoch:
            after_receipt()
            # Raw summaries were recorded after observing the next boundary.
            previous = rows[epoch - 1]
            if (_canonical(interval_filter.node_summaries()) != _canonical(previous["filter_summary"])
                    or len(interval_filter.reset_events) != previous["filter_resets"]):
                raise ValueError("reconstructed next-epoch filter summary mismatch")
        states.append(public_features(view, interval_filter, proposal))
    if _canonical(interval_filter.state_dict()) != _canonical(end_controller_state["filter"]):
        raise ValueError("reconstructed final filter state mismatch")
    complete = len(rows) == proposal["synthetic_system"]["host_total_horizon"]
    control = proposal["synthetic_system"]["control_epochs"]
    tail_cost = 0.0
    for cost in costs[control:]:
        tail_cost += cost  # Match the runner's ordered float64 accumulation.
    result = []
    for epoch in range(control if complete else len(rows)):
        terminal = complete and epoch == control - 1
        result.append(CapacityTransition.from_cost(
            state=states[epoch], executed_hours=rows[epoch]["executed_hours"],
            next_state=states[-1] if terminal else states[epoch + 1],
            done=terminal, world_id=identifier, control_cost=costs[epoch],
            settlement_cost=tail_cost if terminal else None, reward_divisor=100000.0))
    return result
