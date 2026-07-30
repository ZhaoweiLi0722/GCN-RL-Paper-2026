"""Conservative TD3 fine-tuning for the full flat AFR residual policy."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from src.baselines.flat_ddpg import FlatDDPGAgent
from src.rl.networks import torch


class ConservativeFlatResidualTD3Agent(FlatDDPGAgent):
    """Twin-critic TD3 backend for checkpoint-compatible flat ablations."""

    algorithm = "flat_residual_mdl2_network_td3_bc"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: dict[str, Any],
    ):
        super().__init__(state_dim, action_dim, config)
        self.policy_noise = max(float(config.get("policy_noise", 0.01)), 0.0)
        self.noise_clip = max(float(config.get("noise_clip", 0.02)), 0.0)
        self.policy_delay = max(
            int(config.get("policy_delay", self.actor_update_frequency)),
            1,
        )
        self.freeze_actor_updates = bool(
            config.get("freeze_actor_updates", False)
        )
        self.actor_gradient_clip = max(
            float(config.get("actor_gradient_clip", 5.0)),
            0.0,
        )
        self.critic_gradient_clip = max(
            float(config.get("critic_gradient_clip", 10.0)),
            0.0,
        )
        self.reference_policy_regularization_weight = max(
            float(
                config.get(
                    "reference_policy_regularization_weight",
                    0.0,
                )
            ),
            0.0,
        )
        self.actor_advantage_baseline = str(
            config.get("actor_advantage_baseline", "anchor")
        )
        if self.actor_advantage_baseline not in ("anchor", "reference"):
            raise ValueError(
                "actor_advantage_baseline must be 'anchor' or 'reference'"
            )
        self.reference_advantage_use_upper_confidence = bool(
            config.get(
                "reference_advantage_use_upper_confidence",
                True,
            )
        )

        self.critic1 = self.critic
        self.critic1_target = self.critic_target
        self.critic1_optimizer = self.critic_optimizer
        self.critic2 = copy.deepcopy(self.critic1).to(self.device)
        self._reset_module_parameters(self.critic2)
        self.critic2_target = copy.deepcopy(self.critic2).to(self.device)
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(),
            lr=float(config.get("critic_lr", 3e-4)),
        )
        self._capture_actor_reference()

    def update(self) -> dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self.total_updates += 1
        batch = self.replay_buffer.sample(self.batch_size)
        raw_states = torch.as_tensor(
            batch.states,
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            batch.actions,
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.as_tensor(
            batch.rewards,
            dtype=torch.float32,
            device=self.device,
        )
        raw_next_states = torch.as_tensor(
            batch.next_states,
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            batch.dones,
            dtype=torch.float32,
            device=self.device,
        )
        states = self._critic_states(raw_states)
        next_states = self._critic_states(raw_next_states)
        actor_inputs = self._actor_input_tensor(
            raw_states,
            normalized_states=states,
        )
        next_actor_inputs = self._actor_input_tensor(
            raw_next_states,
            normalized_states=next_states,
        )

        with torch.no_grad():
            next_network_actions = self.actor_target(next_actor_inputs)
            if self.policy_noise > 0.0:
                target_noise = (
                    torch.randn_like(next_network_actions)
                    * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                next_network_actions = (
                    next_network_actions + target_noise
                ).clamp(-1.0, 1.0)
            next_actions = self._compose_actions_tensor(
                raw_next_states,
                next_network_actions,
                apply_correction_gate=False,
            )
            target_q1 = self.critic1_target(
                next_states,
                next_actions,
            )
            target_q2 = self.critic2_target(
                next_states,
                next_actions,
            )
            target_q = torch.minimum(target_q1, target_q2)
            q_targets = rewards + self.gamma * (1.0 - dones) * target_q
            self._require_finite_tensor("TD3 target Q", q_targets)

        q1_expected = self.critic1(states, actions)
        q2_expected = self.critic2(states, actions)
        critic1_loss = torch.nn.functional.mse_loss(
            q1_expected,
            q_targets,
        )
        critic2_loss = torch.nn.functional.mse_loss(
            q2_expected,
            q_targets,
        )
        self._require_finite_tensor("TD3 critic1 loss", critic1_loss)
        self._require_finite_tensor("TD3 critic2 loss", critic2_loss)
        self._step_critic(
            critic1_loss,
            self.critic1_optimizer,
            self.critic1,
        )
        self._step_critic(
            critic2_loss,
            self.critic2_optimizer,
            self.critic2,
        )

        actor_update_due = (
            not self.freeze_actor_updates
            and self.total_updates > self.critic_warmup_updates
            and self.total_updates % self.policy_delay == 0
        )
        metrics = {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "critic_q1_mean": float(q1_expected.mean().item()),
            "critic_q2_mean": float(q2_expected.mean().item()),
            "critic_target_mean": float(q_targets.mean().item()),
            "critic_disagreement_mean": float(
                (q1_expected - q2_expected).abs().mean().item()
            ),
            "actor_updated": float(actor_update_due),
            "actor_warmup": float(
                self.total_updates <= self.critic_warmup_updates
            ),
        }
        target_update_due = self.total_updates % self.policy_delay == 0
        if not actor_update_due:
            if target_update_due:
                self._soft_update(self.critic1, self.critic1_target)
                self._soft_update(self.critic2, self.critic2_target)
            if self.freeze_actor_updates:
                metrics["actor_frozen"] = 1.0
            return metrics

        network_actions = self.actor(actor_inputs)
        actor_actions = self._compose_actions_tensor(
            raw_states,
            network_actions,
            apply_correction_gate=False,
        )
        actor_loss, advantage_metrics = self._actor_objective(
            states,
            raw_states,
            actor_actions,
        )
        reference_policy_loss_value = None
        if self.reference_policy_regularization_weight > 0.0:
            with torch.no_grad():
                reference_network_actions = self.actor_reference(
                    actor_inputs
                )
            reference_policy_loss = torch.nn.functional.mse_loss(
                network_actions,
                reference_network_actions,
            )
            actor_loss = (
                actor_loss
                + self.reference_policy_regularization_weight
                * reference_policy_loss
            )
            reference_policy_loss_value = float(
                reference_policy_loss.item()
            )

        policy_residuals = None
        residual_l2_loss_value = None
        if self.residual_action_enabled and self.residual_l2_weight > 0.0:
            policy_residuals = self._policy_residuals_tensor(
                raw_states,
                network_actions,
                apply_correction_gate=False,
            )
            residual_l2_loss = policy_residuals.pow(2).mean()
            actor_loss = (
                actor_loss + self.residual_l2_weight * residual_l2_loss
            )
            residual_l2_loss_value = float(residual_l2_loss.item())
        elif self.patient_service_proxy_actor_loss_enabled:
            policy_residuals = self._policy_residuals_tensor(
                raw_states,
                network_actions,
                apply_correction_gate=False,
            )

        proxy_metrics = {}
        if (
            self.patient_service_proxy_actor_loss_enabled
            and self.residual_action_enabled
            and policy_residuals is not None
        ):
            proxy_loss, proxy_metrics = (
                self._patient_service_proxy_actor_loss(
                    raw_states,
                    policy_residuals,
                )
            )
            actor_loss = actor_loss + proxy_loss

        imitation_loss_value = None
        if (
            self.imitation_regularization_weight > 0.0
            and self.imitation_states is not None
        ):
            imitation_loss = self._actor_imitation_loss()
            actor_loss = (
                actor_loss
                + self.imitation_regularization_weight * imitation_loss
            )
            imitation_loss_value = float(imitation_loss.item())

        self._require_finite_tensor("TD3 actor loss", actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.actor_gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm=self.actor_gradient_clip,
            )
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        metrics["actor_loss"] = float(actor_loss.item())
        metrics.update(advantage_metrics)
        metrics.update(proxy_metrics)
        metrics.update(self._actor_drift_metrics())
        if residual_l2_loss_value is not None:
            metrics["residual_l2_loss"] = residual_l2_loss_value
        if imitation_loss_value is not None:
            metrics["imitation_loss"] = imitation_loss_value
        if reference_policy_loss_value is not None:
            metrics["reference_policy_loss"] = (
                reference_policy_loss_value
            )
        return metrics

    def _critic_states(self, raw_states):
        if self.temporal_demand_encoder_enabled:
            return raw_states
        return self.observation_scaler.normalize_tensor(raw_states)

    def _actor_objective(
        self,
        states,
        raw_states,
        actor_actions,
    ):
        actor_q = self._twin_min_q(states, actor_actions)
        if (
            not self.anchor_advantage_actor_loss_enabled
            or not self.residual_action_enabled
        ):
            return -actor_q.mean(), {}

        with torch.no_grad():
            if self.actor_advantage_baseline == "reference":
                reference_actor_inputs = self._actor_input_tensor(
                    raw_states,
                    normalized_states=states,
                )
                reference_network_actions = self.actor_reference(
                    reference_actor_inputs
                )
                baseline_actions = self._compose_actions_tensor(
                    raw_states,
                    reference_network_actions,
                    apply_correction_gate=False,
                )
                baseline_q = (
                    self._twin_max_q(states, baseline_actions)
                    if self.reference_advantage_use_upper_confidence
                    else self._twin_min_q(states, baseline_actions)
                )
            else:
                baseline_actions = (
                    self._base_actions_from_states_tensor(raw_states)
                )
                baseline_q = self._twin_min_q(
                    states,
                    baseline_actions,
                )
        advantage = actor_q - baseline_q
        shifted = (
            advantage - self.anchor_advantage_margin
        ) / self.anchor_advantage_temperature
        actor_loss = -(
            self.anchor_advantage_temperature
            * torch.nn.functional.softplus(shifted)
        ).mean()

        negative_advantage_penalty = torch.zeros(
            (),
            dtype=advantage.dtype,
            device=advantage.device,
        )
        if self.anchor_advantage_negative_penalty_weight > 0.0:
            negative_shifted = (
                self.anchor_advantage_margin - advantage
            ) / self.anchor_advantage_temperature
            negative_advantage_penalty = (
                self.anchor_advantage_temperature
                * torch.nn.functional.softplus(negative_shifted)
            ).mean()
            actor_loss = (
                actor_loss
                + self.anchor_advantage_negative_penalty_weight
                * negative_advantage_penalty
            )

        metric_prefix = (
            "actor_reference"
            if self.actor_advantage_baseline == "reference"
            else "actor_anchor"
        )
        return actor_loss, {
            f"{metric_prefix}_advantage_mean": float(
                advantage.mean().item()
            ),
            f"{metric_prefix}_advantage_positive_fraction": float(
                (advantage > 0.0)
                .to(dtype=torch.float32)
                .mean()
                .item()
            ),
            f"{metric_prefix}_negative_advantage_penalty": float(
                negative_advantage_penalty.item()
            ),
        }

    def _twin_min_q(self, states, actions):
        return torch.minimum(
            self.critic1(states, actions),
            self.critic2(states, actions),
        )

    def _twin_max_q(self, states, actions):
        return torch.maximum(
            self.critic1(states, actions),
            self.critic2(states, actions),
        )

    def _step_critic(self, loss, optimizer, critic) -> None:
        optimizer.zero_grad()
        loss.backward()
        if self.critic_gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(),
                max_norm=self.critic_gradient_clip,
            )
        optimizer.step()

    def _capture_actor_reference(self) -> None:
        self.actor_reference = copy.deepcopy(self.actor).to(self.device)
        self.actor_reference.eval()
        for parameter in self.actor_reference.parameters():
            parameter.requires_grad_(False)
        self._actor_reference = {
            name: parameter.detach().clone()
            for name, parameter in self.actor.named_parameters()
        }

    def _actor_drift_metrics(self) -> dict[str, float]:
        squared_sum = 0.0
        parameter_count = 0
        max_abs = 0.0
        with torch.no_grad():
            for name, parameter in self.actor.named_parameters():
                difference = parameter - self._actor_reference[name]
                squared_sum += float(difference.pow(2).sum().item())
                parameter_count += int(difference.numel())
                max_abs = max(
                    max_abs,
                    float(difference.abs().max().item()),
                )
        rms = (
            (squared_sum / parameter_count) ** 0.5
            if parameter_count
            else 0.0
        )
        return {
            "actor_reference_drift_rms": float(rms),
            "actor_reference_drift_max_abs": float(max_abs),
        }

    def load_actor(self, path: str | Path) -> None:
        super().load_actor(path)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self._capture_actor_reference()

    def save(self, path: str | Path) -> None:
        super().save(path)
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        checkpoint["algorithm"] = self.algorithm
        checkpoint["critic1"] = self.critic1.state_dict()
        checkpoint["critic2"] = self.critic2.state_dict()
        checkpoint["total_updates"] = int(self.total_updates)
        torch.save(checkpoint, path)

    @staticmethod
    def _reset_module_parameters(module) -> None:
        for child in module.modules():
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()

    @staticmethod
    def _require_finite_tensor(name: str, value) -> None:
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} became non-finite")
