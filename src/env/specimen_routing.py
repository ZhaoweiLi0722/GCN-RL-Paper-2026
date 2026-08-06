"""Deterministic patient-indexed specimen-routing mechanics.

The facility-net specimen action is a signed desired net flow: negative values
offer lots for routing and positive values request lots. This module converts
that continuous vector into conserved integer patient-lot movements without
pooling, splitting, substitution, or additional random draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.env.patient_condition import PatientState, PatientStatus
from src.graph.edges import Edge


@dataclass(frozen=True)
class SpecimenTransit:
    """One identity-preserving specimen movement awaiting arrival."""

    patient_id: str
    origin_facility: int
    destination_facility: int
    remaining_epochs: int
    lead_time_epochs: int
    distance_miles: float
    transport_time_hours: float
    transfer_cost: float
    route_epoch: int


@dataclass(frozen=True)
class ProductReturnTransit:
    """Finished product returning to the patient's collection/infusion site."""

    patient_id: str
    manufacturing_facility: int
    infusion_facility: int
    remaining_epochs: int
    lead_time_epochs: int
    age_epochs: int = 0


@dataclass(frozen=True)
class RoutingExecution:
    """Executed integer flows and audit metadata for one decision epoch."""

    requested_integer_net: np.ndarray
    actual_net: np.ndarray
    edge_flows: np.ndarray
    immediate_arrivals: np.ndarray
    transits: tuple[SpecimenTransit, ...]
    events: tuple[dict[str, object], ...]
    blocked_requests: int
    blocked_inbound_requests: int
    blocked_outbound_requests: int


def round_facility_net_requests(requested_net: Sequence[float]) -> np.ndarray:
    """Round signed requests half away from zero to whole patient lots."""

    requested = np.asarray(requested_net, dtype=float)
    if requested.ndim != 1 or not np.all(np.isfinite(requested)):
        raise ValueError("specimen-routing requests must be a finite 1-D vector")
    magnitude = np.floor(np.abs(requested) + 0.5)
    return np.sign(requested) * magnitude


def execute_patient_indexed_routes(
    patient_queues: list[list[PatientState]],
    requested_net: Sequence[float],
    qualified_edges: Sequence[Edge],
    *,
    edge_priorities: Sequence[float] | None,
    edge_distances: Sequence[float],
    edge_transport_hours: Sequence[float],
    edge_transfer_costs: Sequence[float],
    lead_time_epochs: int,
    max_transfers_per_patient: int,
    route_epoch: int,
) -> RoutingExecution:
    """Execute deterministic, conserved, direct patient-lot routes.

    Endpoint matching is ordered by configured edge priority, receiver index,
    then donor index. Within a donor queue, patients are selected by lowest
    survival, greatest specimen age, enrollment epoch, then patient ID.
    """

    facility_count = len(patient_queues)
    rounded = round_facility_net_requests(requested_net)
    if rounded.shape != (facility_count,):
        raise ValueError(
            f"Expected {facility_count} specimen-routing requests, got {rounded.shape}"
        )
    if int(lead_time_epochs) < 0:
        raise ValueError("specimen routing lead time cannot be negative")
    if int(max_transfers_per_patient) < 1:
        raise ValueError("max_transfers_per_patient must be positive")

    edges = tuple(_normalize_edge(edge, facility_count) for edge in qualified_edges)
    if len(set(edges)) != len(edges):
        raise ValueError("qualified specimen edges must be unique")
    priorities = _finite_edge_values(edge_priorities, len(edges), default=0.0)
    distances = _finite_edge_values(edge_distances, len(edges), default=0.0)
    transport_hours = _finite_edge_values(
        edge_transport_hours,
        len(edges),
        default=0.0,
    )
    transfer_costs = _finite_edge_values(
        edge_transfer_costs,
        len(edges),
        default=0.0,
    )

    inbound_remaining = np.maximum(rounded, 0.0).astype(int)
    outbound_remaining = np.maximum(-rounded, 0.0).astype(int)
    original_inbound = inbound_remaining.copy()
    original_outbound = outbound_remaining.copy()
    edge_index = {edge: index for index, edge in enumerate(edges)}
    adjacency: dict[int, set[int]] = {}
    for source, destination in edges:
        adjacency.setdefault(source, set()).add(destination)
        adjacency.setdefault(destination, set()).add(source)

    pairs = [
        (receiver, donor)
        for receiver in range(facility_count)
        if inbound_remaining[receiver] > 0
        for donor in range(facility_count)
        if (
            outbound_remaining[donor] > 0
            and receiver != donor
            and receiver in adjacency.get(donor, set())
        )
    ]
    pairs.sort(
        key=lambda pair: (
            priorities[
                edge_index[(min(pair[0], pair[1]), max(pair[0], pair[1]))]
            ],
            pair[0],
            pair[1],
        )
    )

    actual_net = np.zeros(facility_count, dtype=float)
    edge_flows = np.zeros(len(edges), dtype=float)
    immediate_arrivals = np.zeros(facility_count, dtype=float)
    transits: list[SpecimenTransit] = []
    events: list[dict[str, object]] = []

    for receiver, donor in pairs:
        requested_count = min(
            int(inbound_remaining[receiver]),
            int(outbound_remaining[donor]),
        )
        if requested_count <= 0:
            continue
        candidates = sorted(
            (
                patient
                for patient in patient_queues[donor]
                if (
                    patient.status is PatientStatus.WAITING
                    and int(patient.transfer_count) < int(max_transfers_per_patient)
                )
            ),
            key=lambda patient: (
                float(patient.survival),
                -int(patient.specimen_age),
                int(patient.enrollment_epoch),
                str(patient.patient_id),
            ),
        )[:requested_count]
        if not candidates:
            continue

        selected_ids = {patient.patient_id for patient in candidates}
        patient_queues[donor] = [
            patient
            for patient in patient_queues[donor]
            if patient.patient_id not in selected_ids
        ]
        edge = (min(donor, receiver), max(donor, receiver))
        edge_position = edge_index[edge]
        direction = 1.0 if edge[0] == donor else -1.0

        for patient in candidates:
            if not patient.patient_id or patient.patient_id != patient.specimen_id:
                raise ValueError(
                    "patient/specimen identity must be initialized and identical before routing"
                )
            patient.transfer_count += 1
            patient.material_facility = (
                receiver if int(lead_time_epochs) == 0 else None
            )
            if int(lead_time_epochs) == 0:
                patient.status = PatientStatus.WAITING
                patient_queues[receiver].append(patient)
                immediate_arrivals[receiver] += 1.0
            else:
                patient.status = PatientStatus.IN_TRANSIT
                transits.append(
                    SpecimenTransit(
                        patient_id=patient.patient_id,
                        origin_facility=donor,
                        destination_facility=receiver,
                        remaining_epochs=int(lead_time_epochs),
                        lead_time_epochs=int(lead_time_epochs),
                        distance_miles=float(distances[edge_position]),
                        transport_time_hours=float(transport_hours[edge_position]),
                        transfer_cost=float(transfer_costs[edge_position]),
                        route_epoch=int(route_epoch),
                    )
                )
            events.append(
                {
                    "patient_id": patient.patient_id,
                    "specimen_id": patient.specimen_id,
                    "collection_facility": patient.collection_facility,
                    "origin_facility": donor,
                    "destination_facility": receiver,
                    "distance_miles": float(distances[edge_position]),
                    "lead_time_epochs": int(lead_time_epochs),
                    "transport_time_hours": float(transport_hours[edge_position]),
                    "transfer_cost": float(transfer_costs[edge_position]),
                    "route_epoch": int(route_epoch),
                    "transfer_count": int(patient.transfer_count),
                }
            )

        moved = len(candidates)
        actual_net[donor] -= float(moved)
        actual_net[receiver] += float(moved)
        edge_flows[edge_position] += direction * float(moved)
        inbound_remaining[receiver] -= moved
        outbound_remaining[donor] -= moved

    routed_count = int(len(events))
    blocked_inbound = int(original_inbound.sum()) - routed_count
    blocked_outbound = int(original_outbound.sum()) - routed_count
    return RoutingExecution(
        requested_integer_net=rounded.astype(float),
        actual_net=actual_net,
        edge_flows=edge_flows,
        immediate_arrivals=immediate_arrivals,
        transits=tuple(transits),
        events=tuple(events),
        blocked_requests=max(blocked_inbound, blocked_outbound),
        blocked_inbound_requests=blocked_inbound,
        blocked_outbound_requests=blocked_outbound,
    )


def _normalize_edge(edge: Edge, facility_count: int) -> Edge:
    source, destination = (int(edge[0]), int(edge[1]))
    if source == destination:
        raise ValueError("qualified specimen edges cannot be self loops")
    if not (0 <= source < facility_count and 0 <= destination < facility_count):
        raise ValueError(f"qualified specimen edge is out of range: {edge}")
    return (min(source, destination), max(source, destination))


def _finite_edge_values(
    values: Sequence[float] | None,
    length: int,
    *,
    default: float,
) -> tuple[float, ...]:
    normalized = (
        (float(default),) * int(length)
        if values is None
        else tuple(float(value) for value in values)
    )
    if len(normalized) != int(length) or not np.all(np.isfinite(normalized)):
        raise ValueError("routing edge metadata must be finite and match qualified edges")
    if any(value < 0.0 for value in normalized):
        raise ValueError("routing edge metadata cannot be negative")
    return normalized
