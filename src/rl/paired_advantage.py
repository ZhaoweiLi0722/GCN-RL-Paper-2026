"""Online paired counterfactual-advantage utilities for DDPG critics."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.rl.action_projection import project_action
from src.rl.networks import torch


@dataclass(frozen=True)
class OnlinePairedAdvantageSettings:
    """Locked settings for an online behavior-versus-anchor critic target."""

    enabled: bool
    horizon: int
    loss_weight: float
    followup_policy: str
    reward_consistency_atol: float
    sign_tolerance: float


@dataclass(frozen=True)
class PairedRolloutResult:
    """Finite-horizon return difference from two exact-CRN env copies."""

    advantage: float
    anchor_first_reward: float
    behavior_first_reward: float
    behavior_reward_error: float
    rollout_steps: int


def online_paired_advantage_settings(
    config: dict[str, Any],
    *,
    residual_action_enabled: bool,
    online_reward_mode: str,
    online_reward_n_step_horizon: int,
    specimen_action_quantization_enabled: bool,
    action_mode: str,
    action_dim: int,
    num_facilities: int,
) -> OnlinePairedAdvantageSettings:
    """Parse and validate the single Stage F1 critic-correction contract."""

    raw = dict(config.get("online_paired_advantage_critic", {}))
    settings = OnlinePairedAdvantageSettings(
        enabled=bool(raw.get("enabled", False)),
        horizon=int(raw.get("horizon", online_reward_n_step_horizon)),
        loss_weight=float(raw.get("loss_weight", 1.0)),
        followup_policy=str(raw.get("followup_policy", "mdl2")).lower(),
        reward_consistency_atol=float(
            raw.get("reward_consistency_atol", 1e-6)
        ),
        sign_tolerance=float(raw.get("sign_tolerance", 1e-12)),
    )
    if settings.horizon < 1:
        raise ValueError("online_paired_advantage_critic.horizon must be positive")
    if not np.isfinite(settings.loss_weight) or settings.loss_weight < 0.0:
        raise ValueError(
            "online_paired_advantage_critic.loss_weight must be finite and "
            "nonnegative"
        )
    if (
        not np.isfinite(settings.reward_consistency_atol)
        or settings.reward_consistency_atol < 0.0
    ):
        raise ValueError(
            "online_paired_advantage_critic.reward_consistency_atol must be "
            "finite and nonnegative"
        )
    if not np.isfinite(settings.sign_tolerance) or settings.sign_tolerance < 0.0:
        raise ValueError(
            "online_paired_advantage_critic.sign_tolerance must be finite and "
            "nonnegative"
        )
    if settings.followup_policy != "mdl2":
        raise ValueError(
            "online_paired_advantage_critic.followup_policy must be 'mdl2'"
        )
    if not settings.enabled:
        return settings
    if settings.loss_weight <= 0.0:
        raise ValueError(
            "enabled online_paired_advantage_critic requires positive loss_weight"
        )
    if not residual_action_enabled:
        raise ValueError(
            "online_paired_advantage_critic requires residual_action.enabled"
        )
    if online_reward_mode != "n_step_anchor_relative":
        raise ValueError(
            "online_paired_advantage_critic requires "
            "residual_action.online_reward_mode='n_step_anchor_relative'"
        )
    if settings.horizon != int(online_reward_n_step_horizon):
        raise ValueError(
            "online_paired_advantage_critic.horizon must match the online "
            "n-step reward horizon"
        )
    if not specimen_action_quantization_enabled:
        raise ValueError(
            "online_paired_advantage_critic requires specimen action quantization"
        )
    if action_mode != "facility_net" or action_dim != 4 * int(num_facilities):
        raise ValueError(
            "online_paired_advantage_critic requires a facility-net action layout"
        )
    return settings


def quantize_specimen_action_np(
    action: np.ndarray,
    *,
    num_facilities: int,
    max_specimen_transfer: float,
) -> np.ndarray:
    """Return the executed patient-lot specimen slice in normalized units."""

    values = np.asarray(action, dtype=np.float32).reshape(-1)
    facilities = int(num_facilities)
    transfer_scale = float(max_specimen_transfer)
    if facilities < 1 or values.size < facilities:
        raise ValueError("Specimen action does not contain the facility slice")
    if not np.isfinite(transfer_scale) or transfer_scale <= 0.0:
        raise ValueError("max_specimen_transfer must be finite and positive")
    requested = values[:facilities].astype(np.float64) * transfer_scale
    rounded = np.sign(requested) * np.floor(np.abs(requested) + 0.5)
    return (rounded / transfer_scale).astype(np.float32)


def specimen_actions_are_distinct(
    behavior_action: np.ndarray,
    anchor_action: np.ndarray,
    *,
    num_facilities: int,
    max_specimen_transfer: float,
) -> bool:
    """Whether two actions differ on the integer patient-lot manifold."""

    behavior = quantize_specimen_action_np(
        behavior_action,
        num_facilities=num_facilities,
        max_specimen_transfer=max_specimen_transfer,
    )
    anchor = quantize_specimen_action_np(
        anchor_action,
        num_facilities=num_facilities,
        max_specimen_transfer=max_specimen_transfer,
    )
    return not np.array_equal(behavior, anchor)


def rollout_paired_anchor_advantage(
    *,
    anchor_env: Any,
    behavior_action: np.ndarray,
    anchor_action: np.ndarray,
    observed_behavior_reward: float,
    gamma: float,
    horizon: int,
    action_dim: int,
    base_action: Callable[[np.ndarray], np.ndarray],
    reward_consistency_atol: float,
) -> PairedRolloutResult:
    """Evaluate one action against MDL-2, then follow MDL-2 in both branches."""

    behavior_env = copy.deepcopy(anchor_env)
    anchor_state, anchor_reward, anchor_done, _anchor_info = anchor_env.step(
        np.asarray(anchor_action, dtype=np.float32)
    )
    behavior_state, behavior_reward, behavior_done, _behavior_info = (
        behavior_env.step(np.asarray(behavior_action, dtype=np.float32))
    )
    reward_error = abs(float(behavior_reward) - float(observed_behavior_reward))
    tolerance = float(reward_consistency_atol)
    if reward_error > tolerance:
        raise RuntimeError(
            "Paired behavior replay did not reproduce the observed reward: "
            f"error={reward_error:.12g}, tolerance={tolerance:.12g}"
        )

    advantage = float(behavior_reward) - float(anchor_reward)
    rollout_steps = 1
    for offset in range(1, int(horizon)):
        if behavior_done and anchor_done:
            break
        behavior_followup_reward = 0.0
        anchor_followup_reward = 0.0
        if not behavior_done:
            behavior_followup_action = project_action(
                base_action(np.asarray(behavior_state, dtype=np.float32)),
                env_state=behavior_env,
                action_space_info=action_dim,
            ).action
            (
                behavior_state,
                behavior_followup_reward,
                behavior_done,
                _behavior_info,
            ) = behavior_env.step(behavior_followup_action)
        if not anchor_done:
            anchor_followup_action = project_action(
                base_action(np.asarray(anchor_state, dtype=np.float32)),
                env_state=anchor_env,
                action_space_info=action_dim,
            ).action
            (
                anchor_state,
                anchor_followup_reward,
                anchor_done,
                _anchor_info,
            ) = anchor_env.step(anchor_followup_action)
        advantage += (float(gamma) ** offset) * (
            float(behavior_followup_reward) - float(anchor_followup_reward)
        )
        rollout_steps += 1

    return PairedRolloutResult(
        advantage=float(advantage),
        anchor_first_reward=float(anchor_reward),
        behavior_first_reward=float(behavior_reward),
        behavior_reward_error=float(reward_error),
        rollout_steps=int(rollout_steps),
    )


def paired_advantage_critic_loss(
    *,
    critic: Any,
    critic_states: Any,
    behavior_q: Any,
    anchor_actions: Any,
    targets: Any,
    masks: Any,
    critic_action_transform: Callable[[Any], Any],
    sign_tolerance: float,
) -> tuple[Any, dict[str, float]]:
    """Fit within-state Q differences only where a paired target exists."""

    valid = masks.reshape(-1) > 0.5
    valid_count = int(valid.sum().item())
    zero = behavior_q.sum() * 0.0
    if valid_count == 0:
        return zero, {
            "critic_online_paired_advantage_samples": 0.0,
            "critic_online_paired_advantage_batch_fraction": 0.0,
            "critic_online_paired_advantage_loss": 0.0,
            "critic_online_paired_advantage_sign_accuracy": 0.0,
            "critic_online_paired_advantage_target_mean": 0.0,
            "critic_online_paired_advantage_prediction_mean": 0.0,
        }

    anchor_q = critic(
        critic_states,
        critic_action_transform(anchor_actions),
    )
    predictions = (behavior_q - anchor_q)[valid]
    selected_targets = targets[valid]
    loss = torch.nn.functional.mse_loss(predictions, selected_targets)
    comparable = selected_targets.abs() > float(sign_tolerance)
    if bool(comparable.any().item()):
        sign_accuracy = (
            torch.sign(predictions[comparable])
            == torch.sign(selected_targets[comparable])
        ).to(dtype=torch.float32).mean()
        sign_accuracy_value = float(sign_accuracy.item())
    else:
        sign_accuracy_value = 0.0
    return loss, {
        "critic_online_paired_advantage_samples": float(valid_count),
        "critic_online_paired_advantage_batch_fraction": float(
            valid.to(dtype=torch.float32).mean().item()
        ),
        "critic_online_paired_advantage_loss": float(loss.item()),
        "critic_online_paired_advantage_sign_accuracy": sign_accuracy_value,
        "critic_online_paired_advantage_target_mean": float(
            selected_targets.mean().item()
        ),
        "critic_online_paired_advantage_prediction_mean": float(
            predictions.mean().item()
        ),
    }
