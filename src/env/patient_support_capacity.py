"""Unregistered additive patient support-work adapter; not a pilot executor.

The existing patient model remains unchanged. This version rejects legacy
overtime because its precomputed borrowed-reactor accounting is incompatible
with reducing starts at the production hook. Support does not accelerate
culture, testing or release. Numerical assumptions require separate approval.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math

import numpy as np

from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.env.patient_condition import PatientStatus
from src.env.patient_support_work import PatientSupportConfig, PatientSupportWork, _canonical, _nonnegative


def validate_support_host_contract(config, support_config, response_tape):
    """Pure configuration validation, usable before constructing any environment."""
    if (type(support_config) is not PatientSupportConfig
            or config.base.enable_overtime_control or config.base.enable_overtime_fatigue
            or config.base.enable_intertemporal_overtime_commitment
            or config.base.num_facilities != len(support_config.site_ids)
            or config.base.episode_horizon != support_config.max_epochs):
        raise ValueError("matched sites/horizon and legacy overtime disabled are required")
    tape = tuple(_nonnegative(row, len(support_config.site_ids), "response tape") for row in response_tape)
    if len(tape) != support_config.max_epochs:
        raise ValueError("one private response per allowed epoch required")
    return tape


@dataclass(frozen=True)
class PatientSupportAction:
    base_action: tuple[float, ...]
    requested_hours: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "base_action", tuple(self.base_action))
        object.__setattr__(self, "requested_hours", tuple(self.requested_hours))


class PatientSupportCapacityEnv(PatientConditionCapacityEnv):
    """Private simulator state plus a separate causal public-capacity channel.

    Only `step` accepts the extended action. Existing base action_size and
    observations retain their meaning, so old runners must not use this class
    without an explicit collector adapter. Snapshots/tapes are never policy input.
    """

    support_format = "patient-support-capacity-v1"

    def __init__(self, config, *, support_config: PatientSupportConfig, response_tape, seed=None):
        self.support_config = support_config
        self._response_tape = validate_support_host_contract(config, support_config, response_tape)
        self._pending_support_request = None
        self._support_receipt = None
        super().__init__(config, seed)

    def reset(self, seed=None):
        obs = super().reset(seed)
        self.support = PatientSupportWork(self.support_config)
        self._pending_support_request = self._support_receipt = None
        return obs

    def public_capacity(self):
        t = self.support.history.next_decision_epoch
        if t != self.t:
            raise RuntimeError("host and support clocks differ")
        return {"epoch": t, "site_ids": self.support_config.site_ids,
                "history": self.support.history.node_history(decision_epoch=t),
                "pending_hours": self.support.history.pending_hours,
                "ready_waiting_counts": tuple(sum(self.support.is_ready(p.patient_id) for p in row)
                                              for row in self.patient_queues)}

    def _start_production(self, production):
        if self._pending_support_request is None or self._support_receipt is not None:
            raise RuntimeError("support production hook must occur exactly once per step")
        if (self.config.enable_overtime_control or self.config.enable_overtime_fatigue
                or self.config.enable_intertemporal_overtime_commitment):
            raise ValueError("legacy overtime accounting is unsupported")
        if (not isinstance(production, np.ndarray) or production.shape != (len(self.patient_queues),)
                or not production.flags.writeable or not np.isfinite(production).all()
                or (production < 0).any() or not np.equal(production, np.floor(production)).all()):
            raise ValueError("mutable integer-valued production vector required")
        for site, queue in enumerate(self.patient_queues):
            if any(p.status is not PatientStatus.WAITING or p.material_facility != site for p in queue):
                raise ValueError("host waiting queue identity/location mismatch")
        self._support_receipt = self.support.advance(
            epoch=self.t, waiting_ids=tuple(tuple(p.patient_id for p in row) for row in self.patient_queues),
            raw_hours=self._pending_support_request, response=self._response_tape[self.t])
        started = []
        remaining = []
        for site, queue in enumerate(self.patient_queues):
            selected = [p for p in queue if self.support.is_ready(p.patient_id)][:int(production[site])]
            selected_ids = {p.patient_id for p in selected}
            started.append(selected)
            remaining.append([p for p in queue if p.patient_id not in selected_ids])
        self.support.mark_started(p.patient_id for row in started for p in row)
        # Parent step reuses THIS array for reagent/reactor costs and counts.
        # Leaving requested starts here would consume resources for unready patients.
        production[:] = [len(row) for row in started]
        for site, row in enumerate(started):
            for patient in row:
                patient.status = PatientStatus.IN_PRODUCTION
                patient.material_facility = patient.manufacturing_facility = site
        self.patient_queues = remaining
        return started

    def step(self, action):
        if type(action) is not PatientSupportAction:
            raise TypeError("explicit PatientSupportAction required")
        if self._pending_support_request is not None:
            raise RuntimeError("nested support step")
        base = np.asarray(action.base_action, dtype=np.float64)
        if base.shape != (self.action_size,) or not np.isfinite(base).all():
            raise ValueError("finite base action with native dimension required")
        raw = _nonnegative(action.requested_hours, len(self.support_config.site_ids), "raw hours")
        if self.t != self.support.history.next_decision_epoch or self.t >= len(self._response_tape):
            raise ValueError("support clock exhausted or mismatched")
        before = self.state_dict()
        self._pending_support_request = raw
        try:
            obs, reward, done, info = super().step(base)
            receipt = self._support_receipt
            if receipt is None or self.t != receipt.known_at:
                raise RuntimeError("missing receipt or invalid host clock advancement")
            self.support.retire(p.patient_id for p in self.patient_registry.values()
                                if p.status is PatientStatus.LOST)
            ordinary = self.support.ordinary_cost()
            added = math.fsum((ordinary, receipt.labor_cost, receipt.switching_cost))
            total = float(info["cost"]) + added
            if not math.isfinite(total) or not math.isclose(-float(reward), float(info["cost"]),
                                                           rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("native raw cost/reward mismatch")
            info = {**info, "native_cost": float(info["cost"]), "cost": total,
                    "base_cost": float(info["base_cost"]) + added,
                    "support_ordinary_staff_cost": ordinary,
                    "support_flexible_labor_cost": receipt.labor_cost,
                    "support_switching_cost": receipt.switching_cost,
                    "support_public_receipt": asdict(receipt)}
            return obs, -total, done, info
        except Exception:
            self._restore(before)
            raise
        finally:
            self._pending_support_request = self._support_receipt = None

    def state_dict(self):
        if self._pending_support_request is not None:
            raise RuntimeError("snapshot only at completed step boundary")
        return {"format": self.support_format, "host_contract": _canonical(asdict(self.env_config)),
                "support_contract": _canonical(asdict(self.support_config)),
                "response_tape": copy.deepcopy(self._response_tape),
                "host": super().state_dict(), "support": self.support.snapshot()}

    def _restore(self, state):
        restored = PatientSupportWork.restore(state["support"], config=self.support_config)
        super().load_state_dict(state["host"])
        if self.t != restored.history.next_decision_epoch:
            raise ValueError("restored support and host clocks differ")
        tasks = restored.snapshot()["tasks"]
        for pid, row in tasks.items():
            if pid not in self.patient_registry:
                raise ValueError("support patient absent from host registry")
            status = self.patient_registry[pid].status
            if ((row["state"] == "waiting" and status not in (PatientStatus.WAITING, PatientStatus.IN_TRANSIT))
                    or (row["state"] == "retired" and status is not PatientStatus.LOST)
                    or (row["state"] == "started" and status in (PatientStatus.WAITING, PatientStatus.IN_TRANSIT))):
                raise ValueError("restored patient lifecycle mismatch")
        self.support = restored

    def load_state_dict(self, state):
        before = self.state_dict()
        if not isinstance(state, dict) or set(state) != set(before):
            raise ValueError("invalid support environment snapshot")
        for key in ("format", "host_contract", "support_contract", "response_tape"):
            if _canonical(state[key]) != _canonical(before[key]):
                raise ValueError(f"support environment contract mismatch: {key}")
        try:
            self._restore(state)
        except Exception:
            self._restore(before)
            raise

    def terminal_support_liabilities(self):
        return {**self.support.liabilities(),
                "active_patient_ids": sorted(p.patient_id for p in self.patient_registry.values()
                                             if p.status not in (PatientStatus.LOST, PatientStatus.DELIVERED)),
                "note": "Raw liabilities only; no terminal penalty or settlement is supplied here."}
