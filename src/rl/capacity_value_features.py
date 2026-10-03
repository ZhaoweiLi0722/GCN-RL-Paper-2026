"""One public-belief feature schema for observed and forecast terminal states."""

import math

import numpy as np

from src.baselines.capacity_completion_control_recovery2 import PublicPatientForecast


FEATURE_NAMES = (
    "waiting", "ready", "unsupported", "production", "transit", "urgent",
    "mean_age", "max_specimen_age", "mean_survival", "work_low", "work_mean", "work_high",
    "reagents", "idle", "orders", "reagent_transfers", "capacity_transfers", "supplier",
    "pending0", "pending1", "previous_hours", "response",
    "production1", "production2", "production3", "production4", "production5",
    "epoch", "control_remaining", "total_remaining", "demand",
)


def forecast_features(model):
    """Never reads native patient health, residual work, tapes or future receipts."""
    s = model.proposal["synthetic_system"]
    rows = []
    for site in range(4):
        local = [p for p in model.patients.values()
                 if p.record["status"] not in ("lost", "delivered")
                 and (p.record["material_site"] if p.record["material_site"] is not None
                      else p.destination) == site]
        waiting = [p for p in local if p.record["status"] == "waiting"]
        pending = [p for p in local if not p.record["support_complete"]]
        n = max(1, len(local))
        row = [len(waiting)/100, sum(p.record["support_complete"] for p in waiting)/100,
               len(pending)/100, sum(p.record["status"] == "in_production" for p in local)/100,
               sum(p.record["status"] == "in_transit" for p in local)/100,
               sum(p.record["survival"] < s["patient"]["eligibility_threshold"]
                   + s["patient"]["urgency_margin"] for p in local)/100,
               sum(p.record["age"] for p in local)/n/64,
               max((p.record["specimen_age"] for p in local), default=0)/64,
               sum(p.record["survival"] for p in local)/n]
        row += [math.fsum(p.interval[0] for p in pending)/100,
                math.fsum(sum(p.interval)/2 for p in pending)/100,
                math.fsum(p.interval[1] for p in pending)/100]
        row += [model.reagents[site]/100, model.idle[site]/100,
                model.orders[:, site].sum()/100, model.reagent_transfers[:, site].sum()/100,
                model.capacity_transfers[:, site].sum()/100, model.supplier[site],
                model.pending[0, site]/8, model.pending[1, site]/8,
                model.previous[site]/8, model.eta[site]/1.5]
        row += [sum(p.record["status"] == "in_production" and p.stage == stage for p in local)/100
                for stage in range(1, 6)]
        row += [min(model.epoch, 64)/64, max(0, 48-model.epoch)/48,
                max(0, 64-model.epoch)/64, model.current_arrivals[site]/100]
        rows.append(row)
    values = np.asarray(rows, dtype=np.float32)
    if values.shape != (4, len(FEATURE_NAMES)) or not np.isfinite(values).all():
        raise ValueError("invalid public value feature schema")
    return values


def observed_features(view, controller):
    model = PublicPatientForecast(view, controller.proposal, controller.filter, controller.lifecycle, .5)
    return forecast_features(model), 0. if model.epoch >= 64 else model.terminal_value()
