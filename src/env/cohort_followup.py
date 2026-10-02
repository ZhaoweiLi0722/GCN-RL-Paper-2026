"""Unregistered closed-enrollment continuation; never extends the learned prefix.

The mixin is tested against an invented engine without constructing a patient
environment. A later separately admitted campaign must supply the common public
controller, charge every call, and persist both raw prefix and tail receipts.
"""

import copy
from dataclasses import asdict, dataclass
import math

import numpy as np


def same_payload(left, right):
    if isinstance(left, np.ndarray):
        return (isinstance(right, np.ndarray) and left.dtype == right.dtype
                and np.array_equal(left, right))
    if isinstance(left, dict):
        return (isinstance(right, dict) and set(left) == set(right)
                and all(same_payload(left[k], right[k]) for k in left))
    if isinstance(left, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(same_payload(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


@dataclass(frozen=True)
class CohortTailSpec:
    enrollment_steps: int
    patient_resolution_steps: int
    accounting_steps: int
    followup_rule: str = "full_mdl2_until_resolution_then_no_new_commitments"

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (
                self.enrollment_steps, self.patient_resolution_steps, self.accounting_steps)):
            raise ValueError("positive integer prefix and tail bounds required")
        if self.accounting_steps < self.patient_resolution_steps:
            raise ValueError("accounting must cover patient resolution")
        if self.followup_rule != "full_mdl2_until_resolution_then_no_new_commitments":
            raise ValueError("unknown common continuation rule")


class ClosedCohortClockMixin:
    """Explicit prefix-to-tail switch with a fixed common accounting endpoint.

    Prefix state_dict/clock/step semantics are untouched. Tail restoration uses
    a separate envelope so an ordinary environment snapshot cannot omit closure.
    Failed tail transactions cannot be restored into another attempt. The outer
    campaign owns durable budgets and scientific admission; this is not a runner.
    """

    def __init__(self, *args, cohort_spec, enabled=False, **kwargs):
        if enabled is not True or type(cohort_spec) is not CohortTailSpec:
            raise ValueError("explicit cohort enablement and typed contract required")
        self.cohort_spec = cohort_spec
        self._cohort_closed = False
        self._cohort_ids = None
        self._cohort_enrolled = None
        self._cohort_resolution_step = None
        self._cohort_failure = None
        self._cohort_steps = 0
        self._cohort_costs = []
        self._cohort_prefix_state = None
        super().__init__(*args, **kwargs)
        if self.config.episode_horizon != cohort_spec.enrollment_steps:
            raise ValueError("prefix horizon must remain unchanged")
        if self.config.enable_overtime_control:
            raise ValueError("overtime is outside this continuation contract")

    def _cohort_live(self):
        if self._cohort_failure is not None:
            raise ValueError("failed cohort transaction is terminal")

    def _cohort_counts(self):
        audit = self.assert_identity_conservation()
        if not isinstance(audit, dict):
            raise ValueError("identity audit missing")
        active = audit.get("active_count")
        if type(active) is not int or not 0 <= active <= len(self.patient_registry):
            raise ValueError("integer conserved active count required")
        if self._cohort_ids is not None and (
                tuple(sorted(self.patient_registry)) != self._cohort_ids
                or self.cumulative_enrolled != self._cohort_enrolled):
            raise ValueError("enrollment or identities changed during closed follow-up")
        return active

    def close_enrollment(self):
        self._cohort_live()
        if self._cohort_closed or self.t != self.cohort_spec.enrollment_steps:
            raise ValueError("close exactly once at the completed prefix boundary")
        self._cohort_prefix_state = copy.deepcopy(super().state_dict())
        self._cohort_ids = tuple(sorted(self.patient_registry))
        self._cohort_enrolled = self.cumulative_enrolled
        self._cohort_closed = True
        if self._cohort_counts() == 0:
            self._cohort_resolution_step = 0
        self.demand = np.zeros_like(self.demand)
        self.demand_forecast = np.zeros_like(self.demand_forecast)
        self.demand_forecast_error = 0.0
        return copy.deepcopy(self._cohort_prefix_state)

    def step(self, action):
        self._cohort_live()
        if self._cohort_closed or self.t >= self.cohort_spec.enrollment_steps:
            raise ValueError("tail calls require step_followup and an external debit")
        return super().step(action)

    def _enroll_arrivals(self, demand):
        if self._cohort_closed and np.any(np.asarray(demand) != 0):
            raise ValueError("new referrals forbidden after enrollment closes")
        return super()._enroll_arrivals(demand)

    def _advance_clock(self):
        if not self._cohort_closed:
            return super()._advance_clock()
        self.t += 1
        # No new demand or forecast draws; supply evolution continues without
        # resetting its state/RNG. Prefix exogenous draws remain unchanged.
        self._advance_regional_supplier_disruptions()
        self.supplier_available = self._sample_supplier_available()
        self.demand = np.zeros_like(self.demand)
        self.demand_forecast = np.zeros_like(self.demand_forecast)
        self.demand_forecast_error = 0.0
        self._record_demand_observation()
        return self.t == self.cohort_spec.enrollment_steps + self.cohort_spec.accounting_steps

    def step_followup(self, action, *, before_step):
        self._cohort_live()
        if (not self._cohort_closed or self._cohort_steps >= self.cohort_spec.accounting_steps
                or self.t != self.cohort_spec.enrollment_steps + self._cohort_steps
                or not callable(before_step)):
            raise ValueError("live sequential tail and durable debit callback required")
        request = np.asarray(action, dtype=np.float64)
        if request.shape != (self.action_size,) or not np.isfinite(request).all():
            raise ValueError("finite full-precision request required")
        n = self.config.num_facilities
        if self.action_size != 4 * n:
            raise ValueError("four-channel resource contract required")
        if self._cohort_counts() == 0:
            idle = np.concatenate((np.zeros(3 * n), -np.ones(n)))
            if not np.array_equal(request, idle):
                raise ValueError("no new transfers or orders after patient resolution")
        before = self.followup_state_dict()
        debit_started = False
        try:
            debit_started = True
            before_step()
            raw, reward, done, info = super().step(request.copy())
            index = self._cohort_steps + 1
            if self.t != self.cohort_spec.enrollment_steps + index:
                raise ValueError("tail clock discontinuity")
            cost = float(info["cost"])
            if cost < 0 or not math.isfinite(cost) or not math.isfinite(float(reward)) or reward != -cost:
                raise ValueError("raw cost/reward mismatch")
            if bool(done) != (index == self.cohort_spec.accounting_steps):
                raise ValueError("fixed accounting endpoint required")
            active = self._cohort_counts()
            if index >= self.cohort_spec.patient_resolution_steps and active:
                raise ValueError("patient-resolution bound exceeded; no censoring-as-success")
            if done:
                for field in ("specimen_transfer_pipeline", "reagent_transfer_pipeline",
                              "capacity_transfer_pipeline", "reagent_purchase_pipeline"):
                    pipeline = np.asarray(getattr(self, field, ()), dtype=float)
                    if not np.isfinite(pipeline).all() or np.any(pipeline != 0):
                        raise ValueError("unsettled or invalid resource pipeline at accounting endpoint")
            if active == 0 and self._cohort_resolution_step is None:
                self._cohort_resolution_step = index
            self._cohort_steps = index
            self._cohort_costs.append(cost)
            event = dict(index=index, action=request.copy(), info=copy.deepcopy(info),
                         raw_reward=float(reward), cost=cost, active=active,
                         accounting_done=bool(done), resolution_step=self._cohort_resolution_step)
            return raw, reward, done, event
        except BaseException as error:
            failure = dict(error=repr(error), at_step=self._cohort_steps,
                           debit_started=debit_started, refunded=False)
            self._cohort_failure = failure
            try:
                failure["partial_environment"] = copy.deepcopy(super().state_dict())
            except BaseException as capture_error:
                failure["capture_error"] = repr(capture_error)
            try:
                super().load_state_dict(before["environment"])
            except BaseException as restore_error:
                failure["rollback_error"] = repr(restore_error)
            self._cohort_steps = before["steps"]
            self._cohort_costs = before["costs"]
            self._cohort_resolution_step = before["resolution_step"]
            raise

    def followup_state_dict(self):
        return dict(format="closed-cohort-clock-v1", contract=asdict(self.cohort_spec),
                    environment=copy.deepcopy(super().state_dict()), closed=self._cohort_closed,
                    prefix_state=copy.deepcopy(self._cohort_prefix_state), ids=self._cohort_ids,
                    enrolled=self._cohort_enrolled, resolution_step=self._cohort_resolution_step,
                    steps=self._cohort_steps, costs=list(self._cohort_costs),
                    failure=copy.deepcopy(self._cohort_failure))

    def load_followup_state_dict(self, state):
        """Only same-boundary restore; never rewind consumed tail work."""
        self._cohort_live()
        saved = copy.deepcopy(state)
        current = self.followup_state_dict()
        immutable = ("format", "contract", "closed", "ids", "enrolled", "resolution_step", "steps", "costs")
        if (set(saved) != set(current) or saved["failure"] is not None
                or any(saved[k] != current[k] for k in immutable)):
            raise ValueError("restore cannot change contract, progress, receipts or failure")
        # Prefix state is immutable provenance; compare structured values,
        # including RNG, without accepting ambiguous ndarray truth values.
        if not same_payload(saved["prefix_state"], current["prefix_state"]):
            raise ValueError("prefix provenance changed")
        # No reconstruction/resimulation is needed for a checkpoint readback.
        if not same_payload(saved["environment"], current["environment"]):
            raise ValueError("restore must be at the exact persisted environment boundary")
        super().load_state_dict(saved["environment"])

    def reconstruct_followup_from_prefix(self, state):
        """Restore an owned tail from its separately verified completed prefix.

        No calls are replayed and no RNG is reseeded. The outer collection owns
        receipt hashes and the durable nonrefundable budget. This method never
        clears a failure or rewinds an already active tail.
        """
        self._cohort_live()
        current, saved = self.followup_state_dict(), copy.deepcopy(state)
        if (self._cohort_closed or self.t != self.cohort_spec.enrollment_steps
                or not isinstance(saved, dict) or set(saved) != set(current)
                or saved["format"] != current["format"] or saved["contract"] != current["contract"]
                or saved["closed"] is not True or saved["failure"] is not None
                or not same_payload(saved["prefix_state"], current["environment"])):
            raise ValueError("exact completed prefix and live same-contract tail required")
        count, costs, resolved = saved["steps"], saved["costs"], saved["resolution_step"]
        if (type(count) is not int or not 0 <= count <= self.cohort_spec.accounting_steps
                or not isinstance(costs, list) or len(costs) != count
                or any(type(c) not in (int, float) or not math.isfinite(c) or c < 0 for c in costs)
                or (resolved is not None and (type(resolved) is not int or not 0 <= resolved <= count))
                or saved["ids"] != tuple(sorted(self.patient_registry))
                or type(saved["enrolled"]) is not int or saved["enrolled"] != self.cumulative_enrolled):
            raise ValueError("tail progress, identity or cost receipts differ")
        candidate = copy.deepcopy(self)
        candidate.close_enrollment()
        initial_active = candidate._cohort_counts()
        super(ClosedCohortClockMixin, candidate).load_state_dict(saved["environment"])
        if not same_payload(candidate.state_dict(), saved["environment"]):
            raise ValueError("tail environment/RNG did not roundtrip")
        candidate._cohort_steps, candidate._cohort_costs = count, costs
        candidate._cohort_resolution_step = resolved
        active = candidate._cohort_counts()
        if (candidate.t != self.cohort_spec.enrollment_steps + count
                or np.any(candidate.demand != 0) or np.any(candidate.demand_forecast != 0)
                or candidate.demand_forecast_error != 0
                or (active == 0) != (resolved is not None)
                or (initial_active == 0) != (resolved == 0)
                or (count >= self.cohort_spec.patient_resolution_steps and active)
                or (resolved is not None and resolved > self.cohort_spec.patient_resolution_steps)):
            raise ValueError("tail clock, enrollment closure or resolution boundary differs")
        if count == self.cohort_spec.accounting_steps:
            for field in ("specimen_transfer_pipeline", "reagent_transfer_pipeline",
                          "capacity_transfer_pipeline", "reagent_purchase_pipeline"):
                pipeline = np.asarray(getattr(candidate, field, ()), dtype=float)
                if not np.isfinite(pipeline).all() or np.any(pipeline != 0):
                    raise ValueError("restored final resource flow is not settled")
        if not same_payload(candidate.followup_state_dict(), saved):
            raise ValueError("full tail envelope did not roundtrip")
        self.__dict__.update(candidate.__dict__)

    def common_followup_request(self, anchor_config):
        """Public full MDL-2 under zero prospective demand; no learned actor."""
        self._cohort_live()
        if not self._cohort_closed or self._cohort_steps >= self.cohort_spec.accounting_steps:
            raise ValueError("common action is available only in the active tail")
        n = self.config.num_facilities
        if self._cohort_counts() == 0:
            return np.concatenate((np.zeros(3 * n), -np.ones(n)))
        if anchor_config.get("num_facilities") != n:
            raise ValueError("matching public anchor layout required")
        from src.baselines.heuristics import facility_net_action_from_state, heuristic_settings_for_policy
        overlay = copy.deepcopy(anchor_config)
        # Plain MDL-2 uses rate priors even when an observation forecast is zero.
        overlay["demand_rates"] = [0.] * n
        overlay["demand_rate_estimates"] = [0.] * n
        request = facility_net_action_from_state(self.observation(), overlay,
            settings=heuristic_settings_for_policy("mdl2"))
        return np.asarray(request, dtype=np.float64).copy()


def cohort_environment_class():
    """Lazy class factory, deliberately not registered in build_env."""
    from src.env.patient_capacity_planning import PatientConditionCapacityEnv

    class ClosedCohortPatientEnv(ClosedCohortClockMixin, PatientConditionCapacityEnv):
        pass

    return ClosedCohortPatientEnv
