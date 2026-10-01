"""Opt-in dynamic-policy likelihood adapters over the existing sealed receipts.

No collector, optimizer or environment launch is registered here. Historical
policy type guards remain unchanged; shared support/return contracts are reused.
"""

from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.rl.candidate_rollout import CandidateDecision, PolicyEvaluation
from src.rl.networks import require_torch, torch
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.routing_candidate_contract import class_distribution


def evaluate_dynamic_policy(policy, observation, candidates, contract):
    require_torch()
    if type(policy) is not DynamicCandidatePolicy:
        raise TypeError("explicit DynamicCandidatePolicy required")
    if not isinstance(contract, ReplayInputContract) or contract.inputs != policy.schema:
        raise ValueError("explicit matching input/reward contract required")
    before = policy.snapshot_sha256()
    with torch.no_grad():
        scores = policy(observation, candidates)
        distribution = class_distribution(scores.logits, candidates)
        result = PolicyEvaluation(
            contract, candidates, before, policy.definition_sha256(), scores.actor_state,
            tuple(distribution.logits.tolist()), scores.value.item(),
            str(scores.logits.dtype).removeprefix("torch."))
    if policy.snapshot_sha256() != before:
        raise ValueError("behavior policy changed during evaluation")
    return result


def verify_dynamic_evaluation(policy, observation, evaluation):
    if not isinstance(evaluation, PolicyEvaluation):
        raise TypeError("PolicyEvaluation required")
    if evaluate_dynamic_policy(policy, observation, evaluation.candidates,
                               evaluation.contract) != evaluation:
        raise ValueError("behavior evaluation does not reproduce from snapshot/input")


def reevaluate_dynamic_decision(policy, observation, decision):
    if type(policy) is not DynamicCandidatePolicy or not isinstance(decision, CandidateDecision):
        raise TypeError("explicit dynamic policy and sealed decision required")
    evaluation = decision.evaluation
    if policy.schema != evaluation.input_schema:
        raise ValueError("new-policy input schema differs")
    if policy.definition_sha256() != evaluation.policy_definition_sha256:
        raise ValueError("new-policy definition differs from behavior")
    scores = policy(observation, evaluation.candidates)
    if scores.actor_state != evaluation.actor_state:
        raise ValueError("new-policy observation differs from collected input")
    distribution = class_distribution(scores.logits, evaluation.candidates)
    index = torch.tensor(decision.choice.class_index, device=scores.logits.device)
    return distribution.log_prob(index), scores.value, distribution.entropy()
