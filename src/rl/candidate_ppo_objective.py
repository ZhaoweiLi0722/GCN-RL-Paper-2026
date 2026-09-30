"""Checked categorical PPO objective, adapted from src/baselines/ppo.py.

Matches its clipped policy surrogate, unclipped value MSE, population advantage
normalization and entropy sign. This is loss arithmetic, not an update loop.
Shared candidate encoders receive gradients from the combined objective; that
differs from the historical agent's separate actor and critic networks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.rl.networks import require_torch, torch


def _vector(value, name, reference=None):
    require_torch()
    if (not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() == 0
            or value.dtype not in (torch.float32, torch.float64)):
        raise ValueError(f"{name} must be a nonempty one-dimensional floating tensor")
    if reference is not None and (value.shape != reference.shape or value.dtype != reference.dtype
                                   or value.device != reference.device):
        raise ValueError(f"{name} shape/dtype/device mismatch; no broadcasting")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")


def _coefficient(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return float(value)


def normalize_rollout_advantages(advantages, *, enabled):
    """Apply once to the full rollout before minibatching, never in-place."""
    _vector(advantages, "advantages")
    if type(enabled) is not bool:
        raise ValueError("advantage normalization choice must be explicit")
    result = advantages.detach().clone()
    if enabled and result.numel() > 1:
        result = (result - result.mean()) / (result.std(unbiased=False) + 1e-8)
    if not torch.isfinite(result).all().item():
        raise ValueError("nonfinite normalized advantages")
    return result


@dataclass(frozen=True)
class CandidatePPOLoss:
    total: object
    policy: object
    value: object
    entropy: object
    ratios: object
    clip_fraction: object


def candidate_ppo_loss(log_probs, values, entropies, old_log_probs, advantages, returns,
                       *, clip_ratio, value_loss_coef, entropy_coef):
    """Only current log-probabilities, values and entropies retain gradients.

    This helper does not recompute GAE, normalize minibatches, clip value loss,
    change action support, impose a KL threshold, or take an optimizer step.
    Categorical inputs must come from the sealed-support policy interface.
    """
    for name, value in (("log_probs", log_probs), ("values", values), ("entropies", entropies),
                        ("old_log_probs", old_log_probs), ("advantages", advantages), ("returns", returns)):
        _vector(value, name, log_probs if name != "log_probs" else None)
    clip = _coefficient(clip_ratio, "clip_ratio")
    value_coef = _coefficient(value_loss_coef, "value_loss_coef")
    entropy_weight = _coefficient(entropy_coef, "entropy_coef")
    if not 0 < clip < 1:
        raise ValueError("clip_ratio must be strictly between zero and one")
    if (log_probs > 0).any().item() or (old_log_probs > 0).any().item() or (entropies < 0).any().item():
        raise ValueError("categorical log probabilities must be nonpositive and entropies nonnegative")
    ratios = torch.exp(log_probs - old_log_probs.detach())
    if not torch.isfinite(ratios).all().item():
        raise ValueError("nonfinite importance ratios; refusing to hide overflow by clamping")
    fixed_advantages = advantages.detach()
    unclipped = ratios * fixed_advantages
    clipped = ratios.clamp(1 - clip, 1 + clip) * fixed_advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = torch.nn.functional.mse_loss(values, returns.detach())
    entropy = entropies.mean()
    total = policy_loss + value_coef * value_loss - entropy_weight * entropy
    if not all(torch.isfinite(v).all().item() for v in (unclipped, clipped, policy_loss, value_loss, total)):
        raise ValueError("nonfinite PPO loss")
    clip_fraction = ((ratios.detach() - 1).abs() > clip).to(log_probs.dtype).mean()
    return CandidatePPOLoss(total, policy_loss, value_loss, entropy, ratios.detach(), clip_fraction)
