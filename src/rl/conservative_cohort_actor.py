"""Unregistered, one-round adapter for two-round conservative improvement.

The caller owns authorization, the shared public examples, exact round-start
logits, and cross-round lineage. Round one forks the same original for both
arms; round two forks each arm's own committed round-one policy. Construction
never forwards a model, loads artifacts, or carries optimizer moments forward.
Inherited transactions retain terminal failures and non-refundable attempts.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.rl.dynamic_candidate_ppo import _no_compute_check
from src.rl.networks import torch
from src.rl.paired_cohort_actor import (
    PairedCohortActor, _actor_logits, _copy_example, _examples_digest,
    _objective_state, _sha256,
)
from src.rl.paired_cohort_objective import paired_cohort_objective
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.prospective_ddpg_kernel import state_digest


@dataclass(frozen=True)
class ConservativeCohortActorSettings:
    """Fixed numerical protocol; smaller update/context caps permit toy tests."""

    learning_rate: float = 3e-4
    betas: tuple[float, float] = (.9, .999)
    eps: float = 1e-8
    weight_decay: float = 0.
    max_grad_norm: float = .5
    cost_scale: float = 1e9
    max_optimizer_steps: int = 64
    context_count: int = 6
    future_replications: int = 4
    kl_coefficient: float = .05

    def __post_init__(self):
        exact = {"learning_rate": 3e-4, "eps": 1e-8, "weight_decay": 0.,
                 "max_grad_norm": .5, "cost_scale": 1e9, "kl_coefficient": .05}
        for name, expected in exact.items():
            value = getattr(self, name)
            if type(value) not in (int, float) or value != expected:
                raise ValueError(f"{name} differs from the fixed conservative protocol")
        if (type(self.betas) is not tuple or self.betas != (.9, .999)
                or any(type(v) not in (int, float) for v in self.betas)):
            raise ValueError("fixed Adam betas required")
        for name, cap in (("max_optimizer_steps", 64), ("context_count", 6)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f"{name} exceeds the fixed positive per-round cap")
        if type(self.future_replications) is not int or self.future_replications != 4:
            raise ValueError("exactly four future replications required")


def _reference_log_probabilities(logits, width):
    if (not isinstance(logits, torch.Tensor) or logits.layout != torch.strided
            or logits.device.type != "cpu" or logits.dtype != torch.float64
            or tuple(logits.shape) != (width,) or not torch.isfinite(logits).all().item()):
        raise ValueError("reference_logits must be finite CPU float64 [K] on the full support")
    log_q = torch.log_softmax(logits.detach(), dim=0)
    if not torch.isfinite(log_q).all().item():
        raise ValueError("reference log probabilities overflow; no clipping permitted")
    return log_q


def _copy_reference_logits(reference_logits, examples):
    if type(reference_logits) is not tuple or len(reference_logits) != len(examples):
        raise ValueError("reference_logits must be an immutable tuple for every public context")
    result = []
    for logits, example in zip(reference_logits, examples):
        _reference_log_probabilities(logits, len(example.class_keys))
        result.append(logits.detach().clone())
    return tuple(result)


def _conservative_loss(states, reference_logits, settings):
    """Uniform state mean of expected relative cost plus fixed KL(p || q)."""
    cost = paired_cohort_objective(states, cost_scale=settings.cost_scale).loss
    if type(reference_logits) is not tuple or len(reference_logits) != len(states):
        raise ValueError("one reference distribution per public context required")
    divergences = []
    for state, reference in zip(states, reference_logits):
        log_q = _reference_log_probabilities(reference, state.logits.numel())
        log_p = torch.log_softmax(state.logits, dim=0)
        divergences.append((torch.softmax(state.logits, dim=0) * (log_p - log_q)).sum())
    loss = cost + settings.kl_coefficient * torch.stack(divergences).mean()
    if not torch.isfinite(loss).item():
        raise ValueError("nonfinite conservative loss; no clipping permitted")
    return loss


class ConservativeCohortActor(PairedCohortActor):
    """One arm/block/round with a fresh Adam and immutable, supplied q logits.

    ``reference_logits`` follows example order and each bank's canonical class
    order. Supply detached float64 logits (or equivalent log probabilities)
    from this round's prototype, including for BC. Private copies are detached
    defensively. Their provenance cannot be certified without a forward, so
    the caller must establish it before construction. BC uses only R4 targets.

    Each new instance records exactly one round-start Adam reset. Updates and
    checkpoint restoration never reset Adam. The coordinator must reconcile
    external budgets across instances and supply the correct arm's prototype;
    creating a new adapter is not permission to retry a failed transaction.
    """

    def __init__(self, prototype, contract, settings, examples, *, dataset_sha256,
                 arm, reference_logits, enabled=False, round_index=1):
        # The locked parent's constructor requires its old settings type and
        # two-replication protocol. Only that setup is copied, not transactions.
        if (enabled is not True or type(prototype) is not DynamicCandidatePolicy
                or not isinstance(contract, ReplayInputContract) or prototype.schema != contract.inputs
                or type(settings) is not ConservativeCohortActorSettings):
            raise ValueError("explicit matching dynamic policy, contract and conservative settings required")
        if arm not in ("paired_cost", "bc_continue"):
            raise ValueError("explicit paired_cost or bc_continue arm required")
        if type(round_index) is not int or round_index not in (1, 2):
            raise ValueError("round_index must be one or two")
        _sha256(dataset_sha256)
        if type(examples) is not tuple or len(examples) != settings.context_count:
            raise ValueError("finite immutable tuple containing every configured context required")
        actor, critic = tuple(prototype.actor_parameters()), tuple(prototype.critic_parameters())
        if (not actor or not critic
                or {p.untyped_storage().data_ptr() for p in actor}.intersection(
                    p.untyped_storage().data_ptr() for p in critic)
                or {id(p) for p in (*actor, *critic)} != {id(p) for p in prototype.parameters()}
                or any(p.dtype != torch.float32 or p.device.type != "cpu" for p in (*actor, *critic))):
            raise ValueError("disjoint complete CPU float32 actor/critic ownership required")
        self.settings, self.contract, self.arm = settings, contract, arm
        self.round_index = round_index
        self._examples = tuple(_copy_example(e, contract.inputs, settings) for e in examples)
        tokens = [e.bank.state_token for e in self._examples]
        if len(tokens) != len(set(tokens)):
            raise ValueError("duplicate training context state tokens")
        self._reference_logits = _copy_reference_logits(reference_logits, self._examples)
        self.policy = copy.deepcopy(prototype)
        self._critic_sha256 = state_digest(prototype.critic.state_dict())
        initial_sha256 = prototype.snapshot_sha256()
        self._manifest = {
            "format": "conservative-cohort-actor-v1", "contract": asdict(contract),
            "settings": asdict(settings), "settings_sha256": state_digest(asdict(settings)),
            "arm": arm, "round_index": round_index, "total_rounds": 2,
            "policy": prototype.manifest(), "initial_policy_sha256": initial_sha256,
            "dataset_sha256": dataset_sha256, "examples_sha256": _examples_digest(self._examples),
            "reference_logits_sha256": state_digest(self._reference_logits),
            "reference_policy_sha256": initial_sha256,
            "reference_binding": "caller_supplied_exact_round_start_public_logits",
            "state_tokens": tokens, "trainable_owner": "actor", "state_weighting": "uniform_full_batch",
            "objective_version": "conservative-cohort-objective-v1",
            "subtract_reference_baseline": True, "kl_direction": "p||q",
            "bc_target": "bank.reference_class", "bc_uses_kl": False,
            "optimizer": {"kind": "Adam", "foreach": False, "amsgrad": False},
            "optimizer_reset_log": [{"round_index": round_index, "reason": "round_start",
                                     "reset_count": 1, "carried_moments": False}],
            "rng_scope": "python_numpy_torch_cpu_no_private_sampler",
        }
        self.manifest_sha256 = state_digest(self._manifest)
        self.optimizer = torch.optim.Adam(self.policy.actor_parameters(), lr=settings.learning_rate,
            betas=settings.betas, eps=settings.eps, weight_decay=settings.weight_decay,
            foreach=False, amsgrad=False)
        self._groups = copy.deepcopy(self.optimizer.state_dict()["param_groups"])
        self.steps, self.attempted_steps, self._history, self._failure = 0, 0, [], None
        self._set_modes()

    def _check_binding(self):
        super()._check_binding()
        if (self.round_index != self._manifest["round_index"]
                or state_digest(asdict(self.settings)) != self._manifest["settings_sha256"]
                or state_digest(self._reference_logits) != self._manifest["reference_logits_sha256"]
                or any(q.requires_grad or q.grad_fn is not None for q in self._reference_logits)):
            raise ValueError("round, settings or reference distribution binding differs")

    def loss(self, *, before_compute=_no_compute_check):
        """Actor-only forward; caller authorization is required for real models."""
        if self._failure is not None:
            raise ValueError("failed transaction is terminal")
        before_compute()
        self._check_binding()
        states = []
        for example in self._examples:
            before_compute()
            states.append(_objective_state(example, _actor_logits(self.policy, example).to(torch.float64)))
        before_compute()
        if self.arm == "paired_cost":
            return _conservative_loss(tuple(states), self._reference_logits, self.settings)
        loss = torch.stack([-torch.log_softmax(s.logits, dim=0)[s.reference_index] for s in states]).mean()
        if not torch.isfinite(loss).item():
            raise ValueError("nonfinite BC loss")
        return loss

    def state_dict(self):
        """Full parent boundary plus detached q; data remain rebound by hashes."""
        state = super().state_dict()
        state["reference_logits"] = copy.deepcopy(self._reference_logits)
        return state

    def _restore(self, state):
        if type(state) is not dict or "reference_logits" not in state:
            raise ValueError("checkpoint reference distribution binding missing")
        references = _copy_reference_logits(state["reference_logits"], self._examples)
        if state_digest(references) != self._manifest["reference_logits_sha256"]:
            raise ValueError("checkpoint reference distribution binding differs")
        parent_state = dict(state)
        del parent_state["reference_logits"]
        super()._restore(parent_state)
