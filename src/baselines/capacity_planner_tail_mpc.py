"""Additive H8/H16 planner and shared forecast-tail capture, not a run permit."""

import copy

import numpy as np

from src.baselines.capacity_completion_control import _admit
from src.baselines.capacity_forecast_recovery3 import PublicPatientForecast
from src.baselines.capacity_value_mpc_recovery1 import CapacityValueMPCRecovery1
from src.rl.capacity_value_features import forecast_features


def forecast_tail(model, base_control, *, before_clone, before_step,
                  feature_adapter=forecast_features, scientific=True):
    """Continue an isolated public model; never treat these labels as native data."""
    start = model.epoch
    if type(start) is not int or not 8 <= start < 64:
        raise ValueError("forecast terminal start must be in 8..63")
    _admit(before_clone, scientific, dict(start_epoch=start, forecast_clones=1))
    tail = copy.deepcopy(model)
    features, heuristics, costs = [], [], []
    for epoch in range(start, 65):
        if tail.epoch != epoch:
            raise ValueError("forecast tail skipped an epoch")
        features.append(np.asarray(feature_adapter(tail), dtype=np.float32).copy())
        heuristics.append(0. if epoch == 64 else float(tail.terminal_value()))
        if epoch == 64:
            break
        _admit(before_step, scientific, dict(epoch=epoch, tail_model_epochs=1))
        hours = tail.adaptive() if epoch < 48 else np.zeros(4)
        costs.append(float(tail.step(hours, base_control)))
    x, h, c = np.stack(features), np.asarray(heuristics, dtype=np.float64), np.asarray(costs, dtype=np.float64)
    if (x.shape != (65 - start, 4, 31) or not np.isfinite(x).all()
            or not np.isfinite(h).all() or not np.isfinite(c).all() or (c < 0).any()):
        raise ValueError("invalid complete forecast tail")
    return dict(start_epoch=start, epochs=list(range(start, 65)), features=x,
                heuristics=h, costs=c, label_source="public_forecast_adaptive_not_native")


class CapacityPlannerTailMPC(CapacityValueMPCRecovery1):
    def __init__(self, *args, planning_horizon=8, capture_epochs=(),
                 before_tail_clone=None, before_tail_step=None,
                 forecast_factory=PublicPatientForecast,
                 feature_adapter=forecast_features, **kwargs):
        if type(planning_horizon) is not int or planning_horizon not in (8, 16):
            raise ValueError("only declared H8/H16 planning is supported")
        epochs = tuple(capture_epochs)
        if (len(epochs) != len(set(epochs)) or any(type(e) is not int or not 0 <= e < 48 for e in epochs)
                or epochs and planning_horizon != 8):
            raise ValueError("only H8 reference decisions may collect tails")
        # The immutable parent validates the original physical/H8 contract.
        # The H16 compute reference is explicit here, not a hidden config mutation.
        super().__init__(*args, **kwargs)
        if epochs and self.value is not None:
            raise ValueError("tail collection must use the plain reference controller")
        self.planning_horizon, self.capture_epochs = planning_horizon, epochs
        self.before_tail_clone, self.before_tail_step = before_tail_clone, before_tail_step
        self.forecast_factory, self.feature_adapter = forecast_factory, feature_adapter
        self.last_training_tails = []

    def act(self, view, *, role="id_mpc", before_query=None, before_filter=None):
        self.last_training_tails = []
        if view.common.epoch >= 48 or role == "fixed_allocation_reference":
            return super().act(view, role=role, before_query=before_query, before_filter=before_filter)
        if role != "id_mpc":
            raise ValueError("unknown controller role")
        self.observe(view, before_filter=before_filter)
        candidates, mpc = self.candidates(view), self.proposal["id_mpc"]
        totals, bases, features, terminal, captured = [], [], [], [], []
        for index, (initial, switch) in enumerate(candidates):
            for quantile in mpc["response_quantiles"]:
                model, total = None, 0.
                for step in range(self.planning_horizon):
                    _admit(before_query, self.scientific, dict(epoch=view.common.epoch,
                        candidate=index, quantile=quantile, horizon=step, model_epochs=1))
                    if model is None:
                        model = self.forecast_factory(view, self.proposal, self.filter, self.lifecycle, quantile)
                    hours = model.adaptive() if switch and step >= 2 and model.epoch < 48 else initial
                    total += model.step(hours, self.base_control)
                totals.append(total)
                bases.append(0. if model.epoch >= 64 else model.terminal_value())
                features.append(self.feature_adapter(model))
                terminal.append(model.epoch >= 64)
                if view.common.epoch in self.capture_epochs:
                    saved = forecast_tail(model, self.base_control,
                        before_clone=self.before_tail_clone, before_step=self.before_tail_step,
                        feature_adapter=self.feature_adapter, scientific=self.scientific)
                    saved.update(decision_epoch=view.common.epoch, candidate=index,
                                 quantile=quantile, prefix_cost=total)
                    captured.append(saved)
        residual = np.zeros(len(features)) if self.value is None else self.value.residuals(np.stack(features))
        residual = np.where(terminal, 0., residual)
        full = (np.asarray(totals) + np.asarray(bases) + residual).reshape(len(candidates), 3)
        scores = full @ np.asarray(mpc["summary_weights"])
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite terminal scores")
        best = min(range(len(scores)), key=scores.__getitem__)
        self.last_plan = dict(epoch=view.common.epoch, chosen=best, scores=scores.tolist(),
            predicted_prefix_costs=totals, heuristic_terminal_costs=bases,
            learned_terminal_residuals=residual.tolist(), learned_value=self.value is not None,
            model_epochs=len(candidates) * 3 * self.planning_horizon,
            planning_horizon=self.planning_horizon, approximate_public_forecast=True)
        self.last_training_tails = captured
        return candidates[best][0].copy()

    def state_dict(self):
        state = super().state_dict()
        state.update(format="capacity-planner-tail-mpc-v1",
                     planning_horizon=self.planning_horizon, capture_epochs=self.capture_epochs,
                     last_training_tails=copy.deepcopy(self.last_training_tails))
        return state

    def load_state_dict(self, state):
        if (state.get("format") != "capacity-planner-tail-mpc-v1"
                or state.get("planning_horizon") != self.planning_horizon
                or tuple(state.get("capture_epochs", ())) != self.capture_epochs):
            raise ValueError("planner horizon/capture binding mismatch")
        restored = copy.deepcopy(state)
        tails = restored.pop("last_training_tails")
        if not isinstance(tails, list):
            raise ValueError("invalid saved tail records")
        restored.pop("planning_horizon")
        restored.pop("capture_epochs")
        restored["format"] = "capacity-value-mpc-v1"
        super().load_state_dict(restored)
        self.last_training_tails = tails
