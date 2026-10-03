"""Fixed, unfitted node features from the shared public capacity input only."""

import numpy as np

from src.rl.public_support_input import PublicSupportControlInput


def resource_adjacency(proposal):
    n = len(proposal["synthetic_system"]["site_ids"])
    adjacency = np.zeros((n, n), dtype=np.float32)
    for left, right in proposal["synthetic_system"]["transport_resource_information_edges"]:
        adjacency[left, right] = adjacency[right, left] = 1
    adjacency += np.eye(n, dtype=np.float32)
    degree = adjacency.sum(axis=1) ** np.float32(-.5)
    return degree[:, None] * adjacency * degree[None, :]


def public_features(view, interval_filter, proposal):
    """Do not pass a host, private info, patient health, tape or true progress."""
    if type(view) is not PublicSupportControlInput:
        raise TypeError("shared public input required")
    c, op = view.common, view.operations
    d = proposal["learner"]["feature_divisors"]
    n = len(op.site_ids)
    rows = []
    summaries = np.asarray(interval_filter.node_summaries(), dtype=np.float64)
    weights = np.asarray(interval_filter.weights, dtype=np.float64)
    for site in range(n):
        patients = [p for p in op.patients if p.material_site == site and p.status not in ("lost", "delivered")]
        waiting = [p for p in patients if p.status == "waiting"]
        patient = [len(patients) / d["counts"], len(waiting) / d["counts"],
                   sum(p.survival for p in waiting) / max(1, len(waiting)),
                   min((p.survival for p in waiting), default=1.),
                   sum(p.age for p in waiting) / max(1, len(waiting)) / d["age"],
                   max((p.specimen_age for p in waiting), default=0) / d["age"],
                   c.ready_waiting_counts[site] / d["counts"]]
        resources = [op.reagents[site] / d["counts"], op.supplier_available[site],
                     op.demand_forecast[site] / d["counts"], op.current_arrivals[site] / d["counts"]]
        resources += [x / d["counts"] for x in op.bioreactors[site]]
        for pipeline in (op.reagent_orders, op.reagent_transfers, op.capacity_transfers):
            resources += [stage[site] / d["counts"] for stage in pipeline]
        history = np.asarray(c.capacity_history[site], dtype=np.float64).copy()
        history[:, :3] /= d["hours"]
        history[:, 3:5] /= d["counts"]
        estimate = summaries[site].copy()
        estimate[[2, 6]] /= d["counts"]
        estimate[3:6] /= d["work"]
        estimate[8] /= d["age"]
        rows.append(patient + resources + history.ravel().tolist()
                    + [stage[site] / d["hours"] for stage in c.pending_hours]
                    + estimate.tolist() + weights[site].tolist()
                    + [min(c.epoch, 64) / d["age"], max(0, 48 - c.epoch) / d["control_time_remaining"]])
    result = np.asarray(rows, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != n or not np.isfinite(result).all():
        raise ValueError("nonfinite or inconsistent public feature dimensions")
    return result
