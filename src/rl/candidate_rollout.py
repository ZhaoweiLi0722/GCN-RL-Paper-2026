"""Sealed categorical behavior receipts and closed-segment GAE preparation.

No optimizer, rollout driver or scientific evaluation. Reuses the existing PPO
GAE arithmetic after validating lineage and explicit terminal/truncation rules.
Seals detect mismatches, not dishonest provenance supplied by a collector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real

import numpy as np

from src.baselines.ppo import _compute_gae
from src.models.candidate_policy import CandidatePolicy
from src.rl.networks import require_torch, torch
from src.rl.prospective_adapter import ReplayInputContract, completed_segment_windows
from src.rl.routing_candidate_contract import (
    CandidateChoice, RequestCandidates, choose_candidate, class_distribution,
    validate_candidate_transition, validate_choice_precision,
)


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("finite real receipt value required")
    return float(value)


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("explicit SHA256 behavior-policy version required")


@dataclass(frozen=True)
class PolicyEvaluation:
    contract: ReplayInputContract
    candidates: RequestCandidates
    behavior_sha256: str
    policy_definition_sha256: str
    actor_state: tuple[float, ...]
    log_probs: tuple[float, ...]
    value: float
    inference_dtype: str

    def __post_init__(self):
        if not isinstance(self.contract, ReplayInputContract) or not isinstance(self.candidates, RequestCandidates):
            raise TypeError("explicit input and candidate contracts required")
        _sha(self.behavior_sha256)
        _sha(self.policy_definition_sha256)
        s, c = self.input_schema, self.candidates
        n, a = len(s.node_ids), len(s.action_names)
        if c.schema.num_facilities != n or c.schema.action_schema_id != s.definition_id + "/action" or a != 4 * n:
            raise ValueError("evaluation action/input schema mismatch")
        if self.inference_dtype not in ("float32", "float64"):
            raise ValueError("explicit receipt inference precision required")
        state = tuple(_number(v) for v in self.actor_state)
        log_probs = tuple(_number(v) for v in self.log_probs)
        width = n * len(s.node_feature_names) + len(s.global_feature_names) + n * n + a
        if len(state) != width or len(log_probs) != len(c.class_keys):
            raise ValueError("evaluation state/support shape mismatch")
        tolerance = 8 * np.finfo(self.inference_dtype).eps
        if any(v > tolerance for v in log_probs) or not math.isclose(
                math.fsum(math.exp(v) for v in log_probs), 1., rel_tol=0., abs_tol=tolerance):
            raise ValueError("stored class log probabilities must be normalized")
        expected_anchor = tuple(np.asarray(c.requests[1], dtype=self.inference_dtype).astype(float))
        if state[-a:] != expected_anchor:
            raise ValueError("observation anchor differs from candidate anchor")
        object.__setattr__(self, "actor_state", state)
        object.__setattr__(self, "log_probs", log_probs)
        object.__setattr__(self, "value", _number(self.value))

    @property
    def input_schema(self):
        return self.contract.inputs

    @property
    def sha256(self):
        payload = {"format": "candidate-evaluation-v1", "evaluation": asdict(self)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CandidateDecision:
    evaluation: PolicyEvaluation
    choice: CandidateChoice

    def __post_init__(self):
        if not isinstance(self.evaluation, PolicyEvaluation) or not isinstance(self.choice, CandidateChoice):
            raise TypeError("sealed policy evaluation and candidate choice required")
        if self.choice != choose_candidate(self.evaluation.candidates, self.choice.class_index):
            raise ValueError("choice differs from evaluated candidate support")

    @property
    def old_log_prob(self):
        return self.evaluation.log_probs[self.choice.class_index]


def evaluate_candidate_policy(policy, observation, candidates, contract):
    require_torch()
    if type(policy) is not CandidatePolicy:
        raise TypeError("explicit prospective CandidatePolicy required")
    if not isinstance(contract, ReplayInputContract) or contract.inputs != policy.schema:
        raise ValueError("explicit matching input/reward contract required")
    before = policy.snapshot_sha256()
    with torch.no_grad():
        scores = policy(observation, candidates)
        dist = class_distribution(scores.logits, candidates)
        result = PolicyEvaluation(contract, candidates, before, policy.definition_sha256(), scores.actor_state,
                                  tuple(dist.logits.tolist()), scores.value.item(),
                                  str(scores.logits.dtype).removeprefix("torch."))
    if policy.snapshot_sha256() != before:
        raise ValueError("behavior policy changed during evaluation")
    return result


def verify_behavior_evaluation(policy, observation, evaluation):
    if not isinstance(evaluation, PolicyEvaluation):
        raise TypeError("PolicyEvaluation required")
    if evaluate_candidate_policy(policy, observation, evaluation.candidates, evaluation.contract) != evaluation:
        raise ValueError("behavior evaluation does not reproduce from the supplied snapshot/input")


def reevaluate_candidate_decision(policy, observation, decision):
    """Differentiable new-policy likelihood on the unchanged collected support.

    Weights may change for a PPO update; observation/schema/support may not.
    The collector still owns verification of the original behavior snapshot.
    """
    if not isinstance(decision, CandidateDecision) or type(policy) is not CandidatePolicy:
        raise TypeError("explicit policy and sealed decision required")
    if policy.schema != decision.evaluation.input_schema:
        raise ValueError("new-policy input schema differs from collected input")
    if policy.definition_sha256() != decision.evaluation.policy_definition_sha256:
        raise ValueError("new-policy architecture/operator/precision definition differs from behavior")
    scores = policy(observation, decision.evaluation.candidates)
    if scores.actor_state != decision.evaluation.actor_state:
        raise ValueError("new-policy observation differs from collected input")
    dist = class_distribution(scores.logits, decision.evaluation.candidates)
    index = torch.tensor(decision.choice.class_index, device=scores.logits.device)
    return dist.log_prob(index), scores.value, dist.entropy()


def sample_candidate(evaluation, *, generator):
    """Use an explicit CPU RNG, never global sampling or a greedy training choice."""
    require_torch()
    if not isinstance(evaluation, PolicyEvaluation):
        raise TypeError("sealed PolicyEvaluation required")
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("explicit CPU sampling generator required")
    before = generator.get_state().clone()
    try:
        dtype = getattr(torch, evaluation.inference_dtype)
        weights = torch.tensor(evaluation.log_probs, dtype=dtype).exp()
        index = int(torch.multinomial(weights, 1, generator=generator).item())
        choice = choose_candidate(evaluation.candidates, index)
        validate_choice_precision(choice, evaluation.candidates, replay_dtype=evaluation.inference_dtype)
        return CandidateDecision(evaluation, choice)
    except Exception:
        generator.set_state(before)
        raise


@dataclass(frozen=True)
class PreparedCandidateSegment:
    decisions: tuple[CandidateDecision, ...]
    records: tuple
    bootstrap: PolicyEvaluation | None
    gae_lambda: float
    advantages: tuple[float, ...]
    returns: tuple[float, ...]


def prepare_candidate_segment(decisions, records, contract, *, behavior_sha256,
                              bootstrap, gae_lambda, max_steps):
    """One closed, contiguous behavior-policy segment; no shuffled replay.

    Bootstrap values are in once-scaled reward units, like collected values.
    A terminal boundary takes precedence over truncation. Segment-end truncation
    may bootstrap when declared, but cannot connect GAE to another segment.
    """
    if not isinstance(contract, ReplayInputContract):
        raise TypeError("explicit ReplayInputContract required")
    _sha(behavior_sha256)
    lam = _number(gae_lambda)
    if not 0 <= lam <= 1 or type(max_steps) is not int or max_steps < 1:
        raise ValueError("GAE lambda in [0,1] and positive segment cap required")
    decisions, records = tuple(decisions), tuple(records)
    if not 1 <= len(records) <= max_steps or len(decisions) != len(records):
        raise ValueError("segment count outside cap or missing decision receipts")
    completed_segment_windows(records, contract.replay, max_steps=1)
    precision = definition = None
    for decision, record in zip(decisions, records):
        if not isinstance(decision, CandidateDecision):
            raise TypeError("CandidateDecision required; old off-policy rows are not upgraded")
        e = decision.evaluation
        if e.contract != contract or e.behavior_sha256 != behavior_sha256:
            raise ValueError("mixed input/reward contracts or behavior-policy versions")
        if precision is not None and precision != e.inference_dtype:
            raise ValueError("mixed behavior inference precision")
        if definition is not None and definition != e.policy_definition_sha256:
            raise ValueError("mixed behavior policy definitions")
        precision = e.inference_dtype
        definition = e.policy_definition_sha256
        if record.state != e.actor_state:
            raise ValueError("observed state differs from evaluated policy input")
        validate_candidate_transition(decision.choice, e.candidates, record, contract.replay,
                                      replay_dtype=e.inference_dtype)
    last = records[-1]
    use_bootstrap = not last.terminated and contract.replay.bootstrap_on_truncation
    if use_bootstrap:
        if not isinstance(bootstrap, PolicyEvaluation):
            raise ValueError("truncation requires a sealed next-state value evaluation")
        if (bootstrap.behavior_sha256 != behavior_sha256 or bootstrap.contract != contract
                or bootstrap.policy_definition_sha256 != definition
                or bootstrap.inference_dtype != precision or bootstrap.actor_state != last.next_state
                or bootstrap.candidates.state_token != last.next_state_token):
            raise ValueError("bootstrap policy/input/state/precision mismatch")
        last_value = bootstrap.value
    else:
        if bootstrap is not None:
            raise ValueError("non-bootstrapping boundary must not supply a next value")
        last_value = 0.
    with np.errstate(over="ignore", invalid="ignore"):
        rewards = np.asarray([r.raw_reward * contract.replay.reward_scale for r in records], dtype=np.float32)
        values = np.asarray([d.evaluation.value for d in decisions], dtype=np.float32)
        if not np.isfinite(rewards).all() or not np.isfinite(values).all():
            raise ValueError("GAE input overflows the existing float32 core")
        dones = np.zeros(len(records), dtype=np.float32)
        dones[-1] = float(not use_bootstrap)
        advantages, returns = _compute_gae(rewards=rewards, dones=dones, values=values,
                                          last_value=last_value, gamma=contract.replay.gamma, gae_lambda=lam)
        if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
            raise ValueError("nonfinite GAE output")
    return PreparedCandidateSegment(decisions, records, bootstrap, lam,
                                    tuple(float(v) for v in advantages), tuple(float(v) for v in returns))
