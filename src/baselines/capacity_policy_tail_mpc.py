"""Paired public-model continuations under a rule or a frozen MPC controller.

This collector is not an execution permit. Forecast costs are not native
patient counterfactual outcomes, and the fixed parent is not the updated policy.
"""

import copy
from dataclasses import fields, is_dataclass, replace
import math

import numpy as np

from src.baselines.capacity_completion_control import _admit
from src.baselines.capacity_forecast_recovery3 import PublicPatientForecast
from src.baselines.capacity_planner_tail_mpc import CapacityPlannerTailMPC
from src.rl.capacity_value_features import forecast_features


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _public_scalars(value):
    """Keep typed public records, normalizing NumPy scalars for filter replay."""
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return replace(value, **{f.name: _public_scalars(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, tuple):
        return tuple(_public_scalars(x) for x in value)
    if isinstance(value, list):
        return [_public_scalars(x) for x in value]
    if isinstance(value, dict):
        return {k: _public_scalars(v) for k, v in value.items()}
    return value


def collect_policy_tail_pair(view, reference, candidate_index, *, on_failure=None, **kwargs):
    boundary = {}
    try:
        return _collect_policy_tail_pair(view, reference, candidate_index, boundary=boundary, **kwargs)
    except BaseException:
        if on_failure is not None:
            saved = {k: copy.deepcopy(v) for k, v in boundary.items() if k != "controller"}
            if "controller" in boundary:
                saved["controller"] = boundary["controller"].state_dict()
            saved.update(format="capacity-policy-tail-failure-v1", automatic_resume=False,
                         primitive_may_be_partially_applied=True)
            on_failure(saved)
        raise


def _collect_policy_tail_pair(view, reference, candidate_index, *, boundary, value,
                             continuation_sha256, before_clone, before_prefix,
                             before_tail, before_query, before_filter,
                             before_plan=None, after_plan=None, after_filter=None,
                             forecast_factory=PublicPatientForecast,
                             controller_factory=CapacityPlannerTailMPC,
                             feature_adapter=forecast_features,
                             scientific=True):
    """Collect three quantile pairs after one prespecified H8 candidate prefix.

    The reference must already have observed the root, before recording its
    native operation. Only a cloned controller sees hypothetical receipts.
    Its frozen value weights are shared read-only, never copied or updated here.
    The caller binds the declared digest to the admitted model bytes.
    """
    epoch = view.common.epoch
    callbacks = (before_clone, before_prefix, before_tail, before_query, before_filter,
                 before_plan, after_plan, after_filter)
    if (type(epoch) is not int or not 0 <= epoch < 48
            or type(candidate_index) is not int or not 0 <= candidate_index < 16
            or value is None or not _sha(continuation_sha256)
            or scientific and any(not callable(cb) for cb in callbacks)):
        raise ValueError("bounded root, candidate, frozen value binding and admissions required")
    state = reference.state_dict()
    if (state["filter"]["epoch"] != epoch or state["lifecycle"]["epoch"] != epoch
            or state["lifecycle"]["recorded_epoch"] >= epoch):
        raise ValueError("root must be observed but not yet operated")
    candidates = reference.candidates(view)
    if len(candidates) != 16:
        raise ValueError("fixed sixteen-candidate support required")
    initial, switch = candidates[candidate_index]
    proposal, base_control = reference.proposal, reference.base_control
    quantiles = proposal["id_mpc"]["response_quantiles"]
    if quantiles != [.1, .5, .9]:
        raise ValueError("declared three response quantiles required")
    output = dict(adaptive=[], frozen_mpc=[])
    boundary.update(output=output, root_epoch=epoch, candidate=candidate_index,
                    continuation_sha256=continuation_sha256)
    for quantile in quantiles:
        boundary.update(phase="prefix", quantile=quantile)
        _admit(before_clone, scientific, dict(kind="prefix", epoch=epoch,
                                             candidate=candidate_index, quantile=quantile))
        model = forecast_factory(view, proposal, reference.filter, reference.lifecycle, quantile)
        controller = controller_factory(proposal, base_control=base_control,
            value=value, scientific=scientific, planning_horizon=8, capture_epochs=())
        restored = copy.deepcopy(state)
        # The branch controller cannot recursively collect another training tail.
        restored.update(capture_epochs=(), last_training_tails=[])
        controller.load_state_dict(restored)
        boundary.update(model=model, controller=controller)
        prefix_cost = 0.
        for offset in range(8):
            _admit(before_prefix, scientific, dict(epoch=model.epoch, model_epochs=1))
            public = _public_scalars(model.public_view())
            action = base_control(public, proposal, tail=model.epoch >= 48)
            controller.record_operation(public, action)
            hours = model.adaptive() if switch and offset >= 2 and model.epoch < 48 else initial
            cost = float(model.step(hours, base_control))
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("invalid public prefix cost")
            prefix_cost += cost
            controller.observe(_public_scalars(model.public_view()), before_filter=before_filter)
            _admit(after_filter, scientific, dict(epoch=model.epoch))
        for policy in ("adaptive", "frozen_mpc"):
            _admit(before_clone, scientific, dict(kind=policy, epoch=model.epoch,
                                                 candidate=candidate_index, quantile=quantile))
            tail = copy.deepcopy(model)
            records = dict(features=[], heuristics=[], costs=[], actions=[], plans=[])
            boundary.update(phase=policy, tail=tail, records=records)
            start = tail.epoch
            for t in range(start, 65):
                if tail.epoch != t:
                    raise ValueError("nonconsecutive public tail")
                features = np.asarray(feature_adapter(tail), dtype=np.float32)
                heuristic = 0. if t == 64 else float(tail.terminal_value())
                if features.shape != (4, 31) or not np.isfinite(features).all() or not math.isfinite(heuristic):
                    raise ValueError("invalid public tail features/heuristic")
                records["features"].append(features.copy())
                records["heuristics"].append(heuristic)
                if t == 64:
                    break
                public = _public_scalars(tail.public_view())
                if t >= 48:
                    hours = np.zeros(4)
                elif policy == "adaptive":
                    hours = tail.adaptive()
                else:
                    _admit(before_plan, scientific, dict(epoch=t))
                    hours = controller.act(public, role="id_mpc",
                        before_query=before_query, before_filter=before_filter)
                    _admit(after_plan, scientific, dict(epoch=t))
                hours = np.asarray(hours, dtype=np.float64)
                if hours.shape != (4,) or not np.isfinite(hours).all() or (hours < 0).any():
                    raise ValueError("invalid public continuation hours")
                if policy == "frozen_mpc":
                    controller.record_operation(public, base_control(public, proposal, tail=t >= 48))
                _admit(before_tail, scientific, dict(policy=policy, epoch=t, model_epochs=1))
                cost = float(tail.step(hours, base_control))
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError("invalid public continuation cost")
                if policy == "frozen_mpc":
                    controller.observe(_public_scalars(tail.public_view()), before_filter=before_filter)
                    _admit(after_filter, scientific, dict(epoch=tail.epoch))
                records["costs"].append(cost)
                records["actions"].append(hours.copy())
                records["plans"].append(copy.deepcopy(controller.last_plan)
                    if policy == "frozen_mpc" and t < 48 else None)
            records.update(features=np.stack(records["features"]),
                heuristics=np.asarray(records["heuristics"], dtype=np.float64),
                costs=np.asarray(records["costs"], dtype=np.float64),
                actions=np.stack(records["actions"]), start_epoch=start,
                epochs=list(range(start, 65)), decision_epoch=epoch,
                candidate=candidate_index, quantile=quantile, prefix_cost=prefix_cost,
                label_policy=policy, continuation_sha256=continuation_sha256 if policy == "frozen_mpc" else None,
                label_source="public_forecast_not_native",
                policy_is_fixed_parent_not_updated_student=policy == "frozen_mpc")
            output[policy].append(records)
    return output
