"""Opt-in specimen request classes, not a physical-feasibility oracle.

No environment object, patient registry, outcome labels, model or optimizer is
accepted. Equal classes mean equal integer decoder inputs and unchanged other
request groups, not a certificate of equal complete-policy returns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from numbers import Real
from typing import Mapping

import numpy as np

from src.env.specimen_routing import round_facility_net_requests
from src.rl.networks import require_torch, torch
from src.rl.validated_returns import OneStepRecord, ReplaySemantics


@dataclass(frozen=True)
class RoutingRequestSchema:
    action_schema_id: str
    num_facilities: int
    max_specimen_transfer: float
    max_candidates: int

    def __post_init__(self):
        if not isinstance(self.action_schema_id, str) or not self.action_schema_id.strip():
            raise ValueError("explicit submitted-action schema ID required")
        if type(self.num_facilities) is not int or self.num_facilities < 1:
            raise ValueError("positive integer facility count required")
        if type(self.max_candidates) is not int or self.max_candidates < 2:
            raise ValueError("candidate cap must retain reference and anchor")
        scale = self.max_specimen_transfer
        if (isinstance(scale, (bool, np.bool_)) or not isinstance(scale, Real)
                or not np.isfinite(scale) or not 0 < scale <= 2**31 - 1):
            raise ValueError("finite positive transfer scale within integer decoder range required")
        object.__setattr__(self, "max_specimen_transfer", float(scale))

    @property
    def action_dim(self):
        return 4 * self.num_facilities


def _request(value, width):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (width,):
        raise ValueError("facility-net request shape mismatch")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw):
        raise ValueError("request values must be real numbers, not flags or strings")
    result = tuple(float(v) for v in raw)
    if not all(np.isfinite(v) and -1 <= v <= 1 for v in result):
        raise ValueError("finite normalized requests required; no silent clipping")
    return result


def _decoder_key(request, schema):
    n = schema.num_facilities
    lots = round_facility_net_requests(
        np.asarray(request[:n], dtype=np.float64) * schema.max_specimen_transfer
    )
    return tuple(int(v) for v in lots) + tuple(request[n:])


@dataclass(frozen=True)
class RequestCandidates:
    schema: RoutingRequestSchema
    state_token: str
    requests: tuple[tuple[float, ...], ...]
    class_keys: tuple[tuple, ...] = field(init=False)
    members: tuple[tuple[int, ...], ...] = field(init=False)
    representatives: tuple[int, ...] = field(init=False)
    request_to_class: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        if not isinstance(self.schema, RoutingRequestSchema):
            raise TypeError("explicit RoutingRequestSchema required")
        if not isinstance(self.state_token, str) or not self.state_token.strip():
            raise ValueError("nonempty decision state token required")
        if not 2 <= len(self.requests) <= self.schema.max_candidates:
            raise ValueError("candidate count outside declared cap")
        requests = tuple(_request(row, self.schema.action_dim) for row in self.requests)
        n = self.schema.num_facilities
        if any(row[n:] != requests[0][n:] for row in requests[1:]):
            raise ValueError("specimen-only contract forbids differences in other request groups")
        groups = {}
        for index, row in enumerate(requests):
            groups.setdefault(_decoder_key(row, self.schema), []).append(index)
        # Canonical class order does not inherit arbitrary option enumeration.
        keys = tuple(sorted(groups))
        members = tuple(tuple(groups[key]) for key in keys)
        representatives = tuple(
            0 if 0 in group else 1 if 1 in group else min(group, key=lambda i: requests[i])
            for group in members
        )
        index_of = {key: i for i, key in enumerate(keys)}
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "class_keys", keys)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "representatives", representatives)
        object.__setattr__(self, "request_to_class", tuple(
            index_of[_decoder_key(row, self.schema)] for row in requests
        ))

    @property
    def reference_class(self):
        return self.request_to_class[0]

    @property
    def anchor_class(self):
        return self.request_to_class[1]

    @property
    def class_features(self):
        """Decoder features only; never substitute these for submitted actions."""
        n, scale = self.schema.num_facilities, self.schema.max_specimen_transfer
        return tuple(tuple(v / scale for v in key[:n]) + key[n:] for key in self.class_keys)

    @property
    def sha256(self):
        payload = {"format": "specimen-request-classes-v1", "schema": asdict(self.schema),
                   "state_token": self.state_token, "requests": self.requests,
                   "reference_index": 0, "anchor_index": 1}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()


def build_request_candidates(payload: Mapping, schema: RoutingRequestSchema):
    """Caller asserts decision-time provenance; a whitelist cannot prove it.

    The anchor is a specimen-slice comparator with the reference's other
    request groups, not necessarily the full MDL-2 policy action.
    """
    if not isinstance(payload, Mapping) or set(payload) != {
        "state_token", "reference_request", "anchor_request", "option_requests"
    }:
        raise ValueError("candidate payload whitelist rejects hidden state, outcomes and masks")
    options = payload["option_requests"]
    if not isinstance(options, (tuple, list, np.ndarray)):
        raise TypeError("option_requests must be an explicit sequence")
    return RequestCandidates(schema, payload["state_token"],
                             (payload["reference_request"], payload["anchor_request"], *options))


@dataclass(frozen=True)
class CandidateChoice:
    candidate_sha256: str
    state_token: str
    class_index: int
    submitted_request: tuple[float, ...]


def choose_candidate(candidates: RequestCandidates, class_index: int):
    if not isinstance(candidates, RequestCandidates):
        raise TypeError("validated RequestCandidates required")
    if type(class_index) is not int or not 0 <= class_index < len(candidates.class_keys):
        raise ValueError("class index outside canonical support")
    request = candidates.requests[candidates.representatives[class_index]]
    return CandidateChoice(candidates.sha256, candidates.state_token, class_index, request)


def class_distribution(logits, candidates: RequestCandidates):
    """One logit per unique request class; no unverified physical action mask."""
    require_torch()
    if not isinstance(candidates, RequestCandidates):
        raise TypeError("validated RequestCandidates required")
    if (not isinstance(logits, torch.Tensor) or logits.dtype not in (torch.float32, torch.float64)
            or tuple(logits.shape) != (len(candidates.class_keys),)
            or not torch.isfinite(logits).all().item()):
        raise ValueError("finite 1-D floating logits required, one per canonical class")
    centered = logits - logits.max()
    if not torch.isfinite(centered).all().item():
        raise ValueError("logit range overflows selected precision")
    return torch.distributions.Categorical(logits=centered)


def validate_choice_precision(choice, candidates, *, replay_dtype):
    """Check the selected request before collection, without inventing a transition."""
    if not isinstance(choice, CandidateChoice):
        raise TypeError("CandidateChoice required")
    if choice != choose_candidate(candidates, choice.class_index):
        raise ValueError("choice differs from candidate/state seal")
    dtype = np.dtype(replay_dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("replay precision must be float32 or float64")
    cast = np.asarray(choice.submitted_request, dtype=dtype).astype(np.float64)
    n = candidates.schema.num_facilities
    if _decoder_key(cast, candidates.schema)[:n] != _decoder_key(choice.submitted_request, candidates.schema)[:n]:
        raise ValueError("replay precision conversion changes specimen integer requests")


def validate_candidate_transition(choice, candidates, record, semantics, *, replay_dtype):
    """Check a collector receipt before existing typed replay consumes it.

    Does not reconstruct observations or certify their producer. No return
    labels or category IDs are converted into environment rewards/actions.
    """
    validate_choice_precision(choice, candidates, replay_dtype=replay_dtype)
    if not isinstance(record, OneStepRecord) or not isinstance(semantics, ReplaySemantics):
        raise TypeError("typed one-step record and expected replay semantics required")
    if (record.semantics != semantics or semantics.action_dim != candidates.schema.action_dim
            or semantics.action_schema_id != candidates.schema.action_schema_id):
        raise ValueError("candidate and replay action/reward contracts differ")
    if record.origin != "trajectory":
        raise ValueError("a selected behavior action requires an actual trajectory receipt")
    if record.state_token != choice.state_token or record.action != choice.submitted_request:
        raise ValueError("replay must retain the sealed state and original submitted request")
