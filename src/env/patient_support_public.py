"""Versioned public patient/operation view for capacity-control preparation.

Current status and completion identity are proposed observable measurements.
Latent patient health, future shocks, response tapes and partial work are not.
Availability of these measurements in practice is a study assumption, not data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from src.env.patient_support_capacity import PatientSupportCapacityEnv


def _ids_by_site(rows):
    rows = tuple(tuple(row) for row in rows)
    ids = tuple(pid for row in rows for pid in row)
    if any(type(pid) is not str or not pid for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid or duplicated public patient identity")
    return rows


def _vector(values):
    values = tuple(float(x) for x in values)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("invalid public resource value")
    return values


def _matrix(values):
    result = tuple(_vector(row) for row in values)
    if len({len(row) for row in result}) > 1:
        raise ValueError("ragged public resource pipeline")
    return result


@dataclass(frozen=True)
class PublicPatientRecord:
    patient_id: str
    enrollment_epoch: int
    status: str
    collection_site: int | None
    material_site: int | None
    manufacturing_site: int | None
    age: int
    specimen_age: int
    survival: float
    support_complete: bool


@dataclass(frozen=True)
class PublicSupportServiceEvent:
    epoch: int
    known_at: int
    eligible_order: tuple[tuple[str, ...], ...]
    completed_ids: tuple[tuple[str, ...], ...]
    ordinary_hours: tuple[float, ...]
    applied_hours: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "eligible_order", _ids_by_site(self.eligible_order))
        object.__setattr__(self, "completed_ids", _ids_by_site(self.completed_ids))
        object.__setattr__(self, "ordinary_hours", _vector(self.ordinary_hours))
        object.__setattr__(self, "applied_hours", _vector(self.applied_hours))
        n = len(self.eligible_order)
        if (type(self.epoch) is not int or self.epoch < 0
                or type(self.known_at) is not int or self.known_at != self.epoch + 1
                or not n or any(len(x) != n for x in
                                (self.completed_ids, self.ordinary_hours, self.applied_hours))):
            raise ValueError("invalid service event boundary/dimension")
        for eligible, completed in zip(self.eligible_order, self.completed_ids):
            if completed != eligible[:len(completed)]:
                raise ValueError("completions must follow the actual eligible service order")


@dataclass(frozen=True)
class PublicSupportOperations:
    epoch: int
    site_ids: tuple[str, ...]
    patients: tuple[PublicPatientRecord, ...]
    waiting_order: tuple[tuple[str, ...], ...]
    reagents: tuple[float, ...]
    bioreactors: tuple[tuple[float, ...], ...]
    reagent_transfers: tuple[tuple[float, ...], ...]
    capacity_transfers: tuple[tuple[float, ...], ...]
    reagent_orders: tuple[tuple[float, ...], ...]
    supplier_available: tuple[float, ...]
    demand_forecast: tuple[float, ...]
    last_service: PublicSupportServiceEvent | None

    def __post_init__(self):
        object.__setattr__(self, "site_ids", tuple(self.site_ids))
        object.__setattr__(self, "patients", tuple(self.patients))
        object.__setattr__(self, "waiting_order", _ids_by_site(self.waiting_order))
        for field in ("reagents", "supplier_available", "demand_forecast"):
            object.__setattr__(self, field, _vector(getattr(self, field)))
        for field in ("bioreactors", "reagent_transfers", "capacity_transfers", "reagent_orders"):
            object.__setattr__(self, field, _matrix(getattr(self, field)))


def validate_public_operations(view):
    if type(view) is not PublicSupportOperations:
        raise TypeError("typed public operations required")
    n = len(view.site_ids)
    if (type(view.epoch) is not int or view.epoch < 0 or not n
            or any(type(x) is not str or not x for x in view.site_ids)
            or len(set(view.site_ids)) != n or len(view.waiting_order) != n):
        raise ValueError("invalid public operation boundary")
    waiting = _ids_by_site(view.waiting_order)
    by_id = {}
    for row in view.patients:
        if (type(row) is not PublicPatientRecord or type(row.patient_id) is not str or not row.patient_id
                or row.patient_id in by_id or row.status not in
                ("waiting", "in_transit", "in_production", "finished", "delivered", "lost")
                or type(row.enrollment_epoch) is not int or not 0 <= row.enrollment_epoch <= view.epoch
                or any(type(x) is not int or x < 0 for x in (row.age, row.specimen_age))
                or type(row.support_complete) is not bool
                or not math.isfinite(row.survival) or not 0 <= row.survival <= 1):
            raise ValueError("invalid public patient record")
        for site in (row.collection_site, row.material_site, row.manufacturing_site):
            if site is not None and (type(site) is not int or not 0 <= site < n):
                raise ValueError("invalid public patient location")
        by_id[row.patient_id] = row
    queued = {pid for row in waiting for pid in row}
    if queued != {p.patient_id for p in view.patients if p.status == "waiting"}:
        raise ValueError("public waiting identities are incomplete")
    for site, queue in enumerate(waiting):
        if any(by_id[pid].material_site != site for pid in queue):
            raise ValueError("public waiting location mismatch")
    for values in (view.reagents, view.supplier_available, view.demand_forecast):
        if len(_vector(values)) != n:
            raise ValueError("public facility vector mismatch")
    if len(_matrix(view.bioreactors)) != n:
        raise ValueError("public bioreactor facility dimension mismatch")
    for values in (view.reagent_transfers, view.capacity_transfers, view.reagent_orders):
        if any(len(row) != n for row in _matrix(values)):
            raise ValueError("public resource pipeline facility dimension mismatch")
    event = view.last_service
    if view.epoch == 0:
        if event is not None:
            raise ValueError("initial view must not contain future service")
    elif (type(event) is not PublicSupportServiceEvent or event.known_at != view.epoch
          or len(event.eligible_order) != n):
        raise ValueError("missing, stale or future service event")
    if event is not None:
        if any(pid not in by_id for queue in event.eligible_order for pid in queue):
            raise ValueError("service event references unknown public patient")
        if any(not by_id[pid].support_complete for queue in event.completed_ids for pid in queue):
            raise ValueError("completed support is absent from patient record")
    return view


class PublicPatientSupportCapacityEnv(PatientSupportCapacityEnv):
    """Additive v2; v1 and its historical fixtures remain unchanged."""

    support_format = "patient-support-capacity-v2-public-ids"

    def reset(self, seed=None):
        self._last_public_service = self._staged_public_service = None
        return super().reset(seed)

    def _start_production(self, production):
        eligible = tuple(tuple(p.patient_id for p in queue if not self.support.is_ready(p.patient_id))
                         for queue in self.patient_queues)
        started = super()._start_production(production)
        tasks = self.support.snapshot()["tasks"]
        completed = tuple(tuple(pid for pid in row if tasks[pid]["remaining"] == 0.0) for row in eligible)
        r = self._support_receipt
        self._staged_public_service = PublicSupportServiceEvent(
            r.epoch, r.known_at, eligible, completed, r.ordinary_hours, r.applied_hours)
        return started

    def step(self, action):
        try:
            result = super().step(action)
            self._last_public_service = self._staged_public_service
            return result
        finally:
            self._staged_public_service = None

    def public_operations(self):
        if self._pending_support_request is not None:
            raise RuntimeError("public operations only at completed decision boundaries")
        tasks = self.support.snapshot()["tasks"]
        records = tuple(PublicPatientRecord(
            p.patient_id, p.enrollment_epoch, p.status.value, p.collection_facility,
            p.material_facility, p.manufacturing_facility, p.age, p.specimen_age,
            float(p.survival), p.patient_id in tasks and tasks[p.patient_id]["remaining"] == 0.0)
            for _, p in sorted(self.patient_registry.items()))
        result = PublicSupportOperations(
            self.t, self.support_config.site_ids, records,
            tuple(tuple(p.patient_id for p in row) for row in self.patient_queues),
            _vector(self.reagents), _matrix(self.bioreactors), _matrix(self.reagent_transfer_pipeline),
            _matrix(self.capacity_transfer_pipeline), _matrix(self.reagent_purchase_pipeline),
            _vector(self.supplier_available), _vector(self.demand_forecast), self._last_public_service)
        return validate_public_operations(result)

    def state_dict(self):
        return {**super().state_dict(), "last_public_service":
                asdict(self._last_public_service) if self._last_public_service is not None else None}

    def _restore(self, state):
        raw = state["last_public_service"]
        event = PublicSupportServiceEvent(**raw) if raw is not None else None
        super()._restore(state)
        if (event is None and self.t != 0) or (event is not None and event.known_at != self.t):
            raise ValueError("restored public service clock mismatch")
        self._last_public_service = event
        self._staged_public_service = None
