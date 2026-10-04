"""Value MPC using exact near-boundary forecast midpoint decisions."""

import copy

import numpy as np

from src.baselines.capacity_completion_control import _admit
from src.baselines.capacity_completion_control_recovery2 import CapacityCompletionControl
from src.baselines.capacity_forecast_recovery3 import PublicPatientForecast
from src.rl.capacity_value_features import forecast_features


class CapacityValueMPCRecovery1(CapacityCompletionControl):
    def __init__(self, *args, value=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = value

    def act(self, view, *, role="id_mpc", before_query=None, before_filter=None):
        if role == "fixed_allocation_reference" or view.common.epoch >= 48:
            return super().act(view, role=role, before_query=before_query, before_filter=before_filter)
        if role != "id_mpc":
            raise ValueError("new MPC only supports matched planner or uniform role")
        self.observe(view, before_filter=before_filter)
        mpc = self.proposal["id_mpc"]
        candidates = self.candidates(view)
        totals, bases, features, terminal = [], [], [], []
        for index, (initial, switch) in enumerate(candidates):
            for quantile in mpc["response_quantiles"]:
                model, total = None, 0.
                for step in range(mpc["predictive_horizon"]):
                    _admit(before_query, self.scientific, dict(epoch=view.common.epoch,
                           candidate=index, quantile=quantile, horizon=step, model_epochs=1))
                    if model is None:
                        model = PublicPatientForecast(view, self.proposal, self.filter, self.lifecycle, quantile)
                    hours = model.adaptive() if switch and step >= 2 and model.epoch < 48 else initial
                    total += model.step(hours, self.base_control)
                totals.append(total)
                bases.append(0. if model.epoch >= 64 else model.terminal_value())
                features.append(forecast_features(model))
                terminal.append(model.epoch >= 64)
        residual = np.zeros(len(features)) if self.value is None else self.value.residuals(np.stack(features))
        residual = np.where(terminal, 0., residual)
        full = (np.asarray(totals)+np.asarray(bases)+residual).reshape(len(candidates), 3)
        scores = full @ np.asarray(mpc["summary_weights"])
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite value-augmented MPC scores")
        best = min(range(len(scores)), key=scores.__getitem__)
        self.last_plan = dict(epoch=view.common.epoch, chosen=best, scores=scores.tolist(),
            predicted_prefix_costs=totals, heuristic_terminal_costs=bases,
            learned_terminal_residuals=residual.tolist(), learned_value=self.value is not None,
            model_epochs=len(candidates)*3*mpc["predictive_horizon"], approximate_public_forecast=True)
        return candidates[best][0].copy()

    def state_dict(self):
        state = super().state_dict()
        state["format"] = "capacity-value-mpc-v1"
        return state

    def load_state_dict(self, state):
        if state.get("format") != "capacity-value-mpc-v1":
            raise ValueError("value MPC boundary format mismatch")
        state = copy.deepcopy(state)
        state["format"] = "capacity-completion-control-recovery2"
        super().load_state_dict(state)
