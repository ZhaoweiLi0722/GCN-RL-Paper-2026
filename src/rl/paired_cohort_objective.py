"""All-candidate expected-cost arithmetic for one-step policy improvement.

This is not PPO or deployed online adaptation. It consumes supplied cost labels;
it does not generate them, load a model, sample an action or perform an update.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch


@dataclass(frozen=True)
class PairedCohortState:
    """One state's complete, already canonicalized candidate support.

    ``logits`` has shape [K] and ``raw_costs`` has shape [R, K], with R, K >= 1.
    Both must be finite float32/float64 tensors of the same dtype and device.
    Class keys are nonempty strings, finite Python numbers or nonempty tuples
    of those keys, matching the existing tuple-valued canonical class keys.

    ``replication_seed_ids`` is candidate-major [K][R]: each candidate declares
    the identical ordered sequence of unique, nonnegative integer or nonempty
    string seed identifiers. Identifiers are compared without coercion. These
    declarations cannot prove common exogenous streams or canonical action
    equivalence; the label producer must establish both independently.
    """

    logits: torch.Tensor
    raw_costs: torch.Tensor
    class_keys: tuple
    replication_seed_ids: tuple[tuple[int | str, ...], ...]
    reference_index: int


@dataclass(frozen=True)
class PairedCohortDiagnostics:
    """Detached copies; cost advantages are candidate minus reference (lower wins)."""

    class_keys: tuple
    replication_seed_ids: tuple[tuple[int | str, ...], ...]
    reference_index: int
    raw_mean_costs: torch.Tensor
    raw_cost_advantages: torch.Tensor
    scaled_cost_advantages: torch.Tensor
    objective_costs: torch.Tensor
    probabilities: torch.Tensor
    expected_raw_cost: torch.Tensor
    entropy: torch.Tensor


@dataclass(frozen=True)
class PairedCohortObjective:
    loss: torch.Tensor
    states: tuple[PairedCohortDiagnostics, ...]
    metadata: dict[str, object]


def _sequence(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be an explicit nonempty list or tuple")
    return tuple(value)


def _class_key(key):
    if type(key) is str and key.strip():
        return
    if type(key) is int or (type(key) is float and math.isfinite(key)):
        return
    if type(key) is tuple and key:
        for part in key:
            _class_key(part)
        return
    raise ValueError("class keys must be nonempty, immutable and finite")


def _finite(value, name):
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite; arithmetic overflow is not clipped")


def _validate_state(state, reference):
    if not isinstance(state, PairedCohortState):
        raise ValueError("each state must be a PairedCohortState")
    for name, tensor, ndim in (("logits", state.logits, 1), ("raw_costs", state.raw_costs, 2)):
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.device.type == "meta" or tensor.ndim != ndim
                or tensor.numel() == 0 or tensor.dtype not in (torch.float32, torch.float64)):
            raise ValueError(f"{name} must be a nonempty {ndim}-D float32/float64 tensor")
        _finite(tensor, name)
    logits, costs = state.logits, state.raw_costs
    if costs.shape[1] != logits.numel():
        raise ValueError("raw_costs columns must match logits; no broadcasting")
    if costs.dtype != logits.dtype or costs.device != logits.device:
        raise ValueError("raw_costs and logits must have the same dtype and device")
    if reference is not None and (logits.dtype != reference.dtype or logits.device != reference.device):
        raise ValueError("all states must share one dtype and device")
    if type(state.reference_index) is not int or not 0 <= state.reference_index < logits.numel():
        raise ValueError("an explicit reference_index within candidate support is required")

    keys = _sequence(state.class_keys, "class_keys")
    if len(keys) != logits.numel():
        raise ValueError("one class key per candidate is required")
    for key in keys:
        _class_key(key)
    if len(set(keys)) != len(keys):
        raise ValueError("class keys must be unique canonical candidates")

    declared = _sequence(state.replication_seed_ids, "replication_seed_ids")
    if len(declared) != logits.numel():
        raise ValueError("replication_seed_ids must declare seeds for every candidate")
    paired = []
    for row in declared:
        ids = _sequence(row, "candidate replication seed identifiers")
        if len(ids) != costs.shape[0]:
            raise ValueError("one seed identifier per cost replication is required")
        for seed in ids:
            if not ((type(seed) is int and seed >= 0) or (type(seed) is str and seed.strip())):
                raise ValueError("seed identifiers must be nonnegative integers or nonempty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("replication seed identifiers must be unique within each candidate")
        if paired and ids != paired[0]:
            raise ValueError("candidates must declare the same paired replication seeds in the same order")
        paired.append(ids)
    return keys, tuple(paired)


def paired_cohort_objective(states, *, cost_scale, subtract_reference_baseline=True):
    """Return a differentiable loss to minimize, with only logits carrying gradients.

    For N states, c[s,k] = mean_r raw_costs[s,r,k], b[s] = c[s,reference],
    and p[s] = softmax(logits[s]), the default is
    L = mean_s sum_k p[s,k] * (c[s,k] - b[s]) / cost_scale.
    Setting ``subtract_reference_baseline=False`` uses b[s] = 0 instead. The
    reference baseline depends on the state but not the selected action. It
    changes the loss value, not its exact-arithmetic logit gradient. Diagnostics
    always retain reference-relative advantages, even without subtraction.

    Every state has equal weight regardless of candidate or replication count.
    The positive scalar scale is supplied by config, never inferred or learned;
    callers must keep it fixed across calls. No entropy bonus, adaptive scaling,
    threshold, gate or quality decision is applied. Entropy uses natural logs.

    Labels are detached. This finite empirical objective does not establish
    label accuracy, independence of replications, correct rollout continuation,
    improved greedy behavior or improved full-policy/deployment performance.
    Validation checks current tensor contents on every call, without mutation.
    """
    states = _sequence(states, "states")
    if (isinstance(cost_scale, bool) or not isinstance(cost_scale, Real)
            or not math.isfinite(cost_scale) or cost_scale <= 0):
        raise ValueError("cost_scale must be a fixed positive finite real scalar, not a tensor")
    if type(subtract_reference_baseline) is not bool:
        raise ValueError("subtract_reference_baseline must be a bool")
    scale = float(cost_scale)
    losses, diagnostics = [], []
    reference = None
    for state in states:
        keys, seeds = _validate_state(state, reference)
        logits = state.logits
        if reference is None:
            scale_tensor = logits.new_tensor(scale)
            if not torch.isfinite(scale_tensor).item() or scale_tensor.item() <= 0:
                raise ValueError("cost_scale must be positive and finite in the input dtype")
            reference = logits

        raw_means = state.raw_costs.detach().mean(dim=0)
        raw_advantages = raw_means - raw_means[state.reference_index]
        scaled_advantages = raw_advantages / scale_tensor
        objective_costs = (scaled_advantages if subtract_reference_baseline
                           else raw_means / scale_tensor)
        for name, value in (("raw means", raw_means), ("raw cost advantages", raw_advantages),
                            ("scaled cost advantages", scaled_advantages), ("objective costs", objective_costs)):
            _finite(value, name)

        log_probabilities = torch.log_softmax(logits, dim=0)
        _finite(log_probabilities, "log probabilities")
        probabilities = torch.softmax(logits, dim=0)
        loss = (probabilities * objective_costs).sum()
        expected_raw_cost = (probabilities.detach() * raw_means).sum()
        entropy = -(probabilities.detach() * log_probabilities.detach()).sum()
        for name, value in (("state loss", loss), ("expected raw cost", expected_raw_cost), ("entropy", entropy)):
            _finite(value, name)
        losses.append(loss)
        diagnostics.append(PairedCohortDiagnostics(
            class_keys=keys, replication_seed_ids=seeds, reference_index=state.reference_index,
            raw_mean_costs=raw_means.clone(), raw_cost_advantages=raw_advantages.clone(),
            scaled_cost_advantages=scaled_advantages.clone(), objective_costs=objective_costs.clone(),
            probabilities=probabilities.detach().clone(), expected_raw_cost=expected_raw_cost.clone(),
            entropy=entropy.clone(),
        ))

    loss = torch.stack(losses).mean()
    _finite(loss, "uniform state mean loss")
    return PairedCohortObjective(loss, tuple(diagnostics), {
        "version": "paired-cohort-objective-v1",
        "method": "model_assisted_one_step_policy_improvement",
        "target_objective": "all_candidate_expected_raw_cohort_cost",
        "cost_scale": scale,
        "baseline": "reference_mean" if subtract_reference_baseline else "none",
        "state_weighting": "uniform",
        "pairing_validation": "declared_ordered_replication_seed_ids_only",
    })
