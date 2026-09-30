"""Pure public-input and executed-step checks for prospective candidate collection.

No environment construction, step, reset, hidden-state mask or optimizer.
The caller owns simulator continuity/recovery; these checks cannot prove it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from src.env.specimen_routing import round_facility_net_requests
from src.rl.candidate_rollout import CandidateDecision, verify_behavior_evaluation
from src.rl.patient_replay_collector import PatientObservationProducer
from src.rl.prospective_adapter import pack_actor_state
from src.rl.routing_candidate_contract import (
    RoutingRequestSchema, build_request_candidates, validate_candidate_transition, validate_choice_precision,
)
from src.rl.validated_returns import OneStepRecord


OPERATING_COSTS = (
    "reagent_purchase_cost", "reagent_holding_cost", "reagent_shortage_cost",
    "bioreactor_holding_cost", "bioreactor_shortage_cost", "specimen_transfer_cost",
    "capacity_transfer_cost", "reagent_transfer_cost",
)
PATIENT_COSTS = ("patient_loss_cost", "expiry_cost", "urgency_cost")


def _number(value, name, *, nonnegative=True):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or (nonnegative and value < 0)):
        raise ValueError(f"invalid finite numeric {name}")
    return float(value)


def _count(value, name):
    value = _number(value, name)
    if not value.is_integer():
        raise ValueError(f"{name} must be an integer count")
    return int(value)


def _counts(value, name, n, *, signed=False):
    array = np.asarray(value, dtype=object)
    if array.shape != (n,):
        raise ValueError(f"{name} shape mismatch")
    result = tuple(_number(v, name, nonnegative=not signed) for v in array)
    if any(not v.is_integer() for v in result):
        raise ValueError(f"{name} must contain integer counts")
    return tuple(int(v) for v in result)


@dataclass(frozen=True)
class CollectionBoundary:
    episode_horizon: int
    max_steps: int
    horizon_end: str

    def __post_init__(self):
        for name in ("episode_horizon", "max_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("positive explicit collection and horizon limits required")
        if self.horizon_end not in ("terminal", "truncation"):
            raise ValueError("explicit horizon_end mapping required")

    def flags(self, step_index, environment_done):
        if (type(step_index) is not int or not 0 <= step_index < min(self.max_steps, self.episode_horizon)
                or type(environment_done) is not bool):
            raise ValueError("invalid step index or environment done flag")
        if environment_done != (step_index + 1 == self.episode_horizon):
            raise ValueError("environment done does not match the declared horizon")
        terminal = environment_done and self.horizon_end == "terminal"
        closed = environment_done or step_index + 1 == self.max_steps
        return terminal, closed and not terminal


def public_candidate_context(producer, raw, *, reference_request, option_requests,
                             state_token, max_candidates):
    """Use the existing raw producer and full MDL-2 anchor without hidden masks.

    Option generation and reference initialization are external, prospectively
    declared decisions. This function neither ranks nor searches options.
    All non-specimen groups must equal MDL-2, preventing a mixed-anchor context.
    """
    if type(producer) is not PatientObservationProducer:
        raise TypeError("audited PatientObservationProducer required")
    if producer.anchor_config["enable_specimen_routing"] is not True:
        raise ValueError("candidate routing requires the explicitly enabled routing channel")
    observation, anchor = producer.observe(raw)
    schema = RoutingRequestSchema(producer.contract.replay.action_schema_id,
                                   len(producer.contract.inputs.node_ids),
                                   producer.anchor_config["max_specimen_transfer"], max_candidates)
    bank = build_request_candidates({"state_token": state_token, "reference_request": reference_request,
                                     "anchor_request": anchor[0].tolist(), "option_requests": option_requests}, schema)
    return observation, bank


def candidate_submission(policy, observation, decision, *, current_state_token):
    """Check before execution and return the unchanged request in float64.

    The patient environment converts to NumPy float64. Do not route this through
    the historical float32 gate collector or reconstruct a category's center.
    """
    if not isinstance(decision, CandidateDecision):
        raise TypeError("sealed CandidateDecision required")
    if current_state_token != decision.choice.state_token:
        raise ValueError("candidate state token is stale")
    verify_behavior_evaluation(policy, observation, decision.evaluation)
    validate_choice_precision(decision.choice, decision.evaluation.candidates,
                              replay_dtype=decision.evaluation.inference_dtype)
    request = np.asarray(decision.choice.submitted_request, dtype=np.float64).copy()
    request.setflags(write=False)
    return request


@dataclass(frozen=True)
class CandidateStepAudit:
    decision: CandidateDecision
    record: OneStepRecord
    cost_components: tuple[tuple[str, float], ...]
    requested_integer_net: tuple[int, ...]
    actual_integer_net: tuple[int, ...]
    route_count: int
    blocked_requests: int
    patients_lost: tuple[int, ...]
    patients_completed: tuple[int, ...]
    active_patients: int
    other_active_patients: int
    unresolved_at_boundary: int | None
    candidate_class_count: int
    selected_request_differs_from_reference: bool
    terminal_cost_added: float = 0.


def audit_candidate_step(decision, submitted_request, *, next_observation, next_anchor,
                         next_state_token, raw_reward, environment_done, info, boundary,
                         source_id, trajectory_id, step_index):
    """Build an unscaled typed record from a reported real-step result.

    Post-step identity/route metadata is audit-only, never a neural feature.
    A horizon boundary reports unfinished identities but adds no invented cost
    or outcome. A failed audit does not rewind the caller's environment.
    """
    if not isinstance(decision, CandidateDecision) or not isinstance(boundary, CollectionBoundary):
        raise TypeError("sealed decision and explicit collection boundary required")
    evaluation = decision.evaluation
    contract, bank = evaluation.contract, evaluation.candidates
    if contract.replay.reward_kind != "absolute_environment":
        raise ValueError("patient negative-cost collector requires absolute environment rewards")
    if (not isinstance(submitted_request, np.ndarray) or submitted_request.dtype != np.float64
            or submitted_request.shape != (bank.schema.action_dim,)
            or tuple(submitted_request) != decision.choice.submitted_request):
        raise ValueError("execution must retain the exact submitted float64 request")
    if not isinstance(info, dict):
        raise TypeError("explicit step accounting dictionary required")
    known_costs = set(OPERATING_COSTS + PATIENT_COSTS + ("base_cost", "specimen_route_cost", "transshipment_cost"))
    if any(isinstance(k, str) and k.endswith("_cost") and k not in known_costs for k in info):
        raise ValueError("unsupported cost component; new scientific scope must be declared")
    components = tuple((k, _number(info[k], k)) for k in OPERATING_COSTS + PATIENT_COSTS)
    costs = dict(components)
    base, total = _number(info["base_cost"], "base_cost"), _number(info["cost"], "cost")
    for left, right in ((base, sum(costs[k] for k in OPERATING_COSTS)),
                        (total, base + sum(costs[k] for k in PATIENT_COSTS)),
                        (_number(info["specimen_route_cost"], "specimen_route_cost"), costs["specimen_transfer_cost"]),
                        (_number(info["transshipment_cost"], "transshipment_cost"),
                         sum(costs[k] for k in OPERATING_COSTS[-3:]))):
        if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError("step cost components or aliases do not reconcile")
    reward = _number(raw_reward, "raw_reward", nonnegative=False)
    if reward != -total:
        raise ValueError("raw reward is not the unscaled negative step cost")
    n = bank.schema.num_facilities
    requested = _counts(info["specimen_requested_integer_net"], "requested net", n, signed=True)
    expected = round_facility_net_requests(submitted_request[:n] * bank.schema.max_specimen_transfer)
    if requested != tuple(int(v) for v in expected):
        raise ValueError("reported integer request differs from original submitted action")
    actual = _counts(info["specimen_transfers"], "actual net", n, signed=True)
    routes = _count(info["specimen_route_count"], "route count")
    blocked = _count(info["blocked_specimen_requests"], "blocked requests")
    if (sum(actual) != 0 or sum(max(v, 0) for v in actual) != routes
            or any((r >= 0 and not 0 <= a <= r) or (r < 0 and not r <= a <= 0)
                   for r, a in zip(requested, actual))):
        raise ValueError("executed net flow is not a conserved subset of the request")
    # Match the existing decoder: headline blocked count is max(inbound, outbound), not their sum.
    inbound = sum(max(v, 0) for v in requested) - routes
    outbound = sum(max(-v, 0) for v in requested) - routes
    if (blocked != max(inbound, outbound)
            or _count(info["blocked_specimen_inbound_requests"], "blocked inbound") != inbound
            or _count(info["blocked_specimen_outbound_requests"], "blocked outbound") != outbound):
        raise ValueError("blocked-request count does not match unfilled integer requests")
    lost = _counts(info["patients_lost"], "patients lost", n)
    completed = _counts(info["patients_completed"], "patients completed", n)
    active = _count(info["identity_active_count"], "active identities")
    _count(info["identity_terminal_count"], "terminal identities")
    located = sum(sum(_counts(info[key], key, n)) for key in
                  ("waiting_patients", "in_production_patients", "specimen_in_transit"))
    if located > active:
        raise ValueError("reported active compartments exceed identity count")
    terminal, truncated = boundary.flags(step_index, environment_done)
    if next_state_token == decision.choice.state_token:
        raise ValueError("next-state lineage token did not advance")
    following = pack_actor_state(next_observation, next_anchor, contract)
    if str(following.dtype).removeprefix("torch.") != evaluation.inference_dtype or following.shape[0] != 1:
        raise ValueError("following observation inference precision/batch differs")
    record = OneStepRecord(contract.replay, source_id, "trajectory", trajectory_id, step_index,
                           decision.choice.state_token, next_state_token, evaluation.actor_state,
                           decision.choice.submitted_request, reward, tuple(following[0].detach().tolist()),
                           terminal, truncated)
    validate_candidate_transition(decision.choice, bank, record, contract.replay,
                                  replay_dtype=evaluation.inference_dtype)
    return CandidateStepAudit(decision, record, components, requested, actual, routes, blocked,
                               lost, completed, active, active - located,
                               active if terminal or truncated else None, len(bank.class_keys),
                               decision.choice.class_index != bank.reference_class)
