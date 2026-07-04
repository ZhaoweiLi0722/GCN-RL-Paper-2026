"""Proximal Policy Optimization baseline for continuous capacity planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.rl.action_projection import project_action
from src.rl.networks import nn, require_torch, torch


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPSILON = 1e-6


@dataclass
class PPOTransition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class PPOAgent:
    """Clipped-objective PPO agent with a Gaussian continuous actor."""

    algorithm = "ppo"

    def __init__(self, state_dim: int, action_dim: int, config: dict[str, Any]):
        require_torch()
        seed = int(config.get("seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(config.get("gamma", 0.99))
        self.gae_lambda = float(config.get("gae_lambda", 0.95))
        self.clip_ratio = float(config.get("clip_ratio", 0.2))
        self.entropy_coef = float(config.get("entropy_coef", 0.0))
        self.value_loss_coef = float(config.get("value_loss_coef", 0.5))
        self.max_grad_norm = float(config.get("max_grad_norm", 0.5))
        self.train_epochs = int(config.get("train_epochs", 10))
        self.minibatch_size = int(config.get("minibatch_size", 64))
        self.rollout_length = int(config.get("rollout_length", 1024))
        hidden_sizes = tuple(config.get("hidden_sizes", [256, 256]))
        device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)

        self.actor = PPOActor(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic = ValueNetwork(state_dim, hidden_sizes).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(config.get("actor_lr", 3e-4)))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(config.get("critic_lr", 1e-3)))
        self.rng = np.random.default_rng(seed)
        self.rollout: list[PPOTransition] = []
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._last_next_state: np.ndarray | None = None
        self._last_done = False

    def reset(self) -> None:
        return None

    def select_action(self, state: np.ndarray, explore: bool = True, env=None) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            if explore:
                action_tensor, log_prob_tensor, _entropy = self.actor.sample(state_tensor)
            else:
                action_tensor = self.actor.deterministic(state_tensor)
                log_prob_tensor = self.actor.log_prob(state_tensor, action_tensor)
            value_tensor = self.critic(state_tensor)
        self.actor.train()
        self.critic.train()
        self._last_log_prob = float(log_prob_tensor.cpu().numpy()[0, 0])
        self._last_value = float(value_tensor.cpu().numpy()[0, 0])
        action = action_tensor.cpu().numpy()[0]
        return project_action(action, env_state=env, action_space_info=self.action_dim).action

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.rollout.append(
            PPOTransition(
                state=np.asarray(state, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
                log_prob=self._last_log_prob,
                value=self._last_value,
            )
        )
        self._last_next_state = np.asarray(next_state, dtype=np.float32)
        self._last_done = bool(done)

    def update(self) -> dict[str, float]:
        if not self.rollout:
            return {}
        if len(self.rollout) < self.rollout_length and not self._last_done:
            return {}
        metrics = self._update_from_rollout()
        self.rollout.clear()
        self._last_next_state = None
        self._last_done = False
        return metrics

    def _update_from_rollout(self) -> dict[str, float]:
        states = torch.as_tensor(np.stack([item.state for item in self.rollout]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack([item.action for item in self.rollout]), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(
            np.asarray([[item.log_prob] for item in self.rollout], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        rewards = np.asarray([item.reward for item in self.rollout], dtype=np.float32)
        dones = np.asarray([item.done for item in self.rollout], dtype=np.float32)
        values = np.asarray([item.value for item in self.rollout], dtype=np.float32)

        last_value = 0.0
        if self._last_next_state is not None and not self._last_done:
            with torch.no_grad():
                next_tensor = torch.as_tensor(self._last_next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                last_value = float(self.critic(next_tensor).cpu().numpy()[0, 0])

        advantages, returns = _compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device).unsqueeze(-1)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(-1)
        if advantages_tensor.numel() > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std(unbiased=False) + 1e-8)

        batch_size = states.shape[0]
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropy_values: list[float] = []
        for _epoch in range(self.train_epochs):
            indices = self.rng.permutation(batch_size)
            for start in range(0, batch_size, self.minibatch_size):
                idx = torch.as_tensor(indices[start : start + self.minibatch_size], dtype=torch.long, device=self.device)
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages_tensor[idx]
                batch_returns = returns_tensor[idx]

                log_probs = self.actor.log_prob(batch_states, batch_actions)
                entropy = self.actor.entropy(batch_states).mean()
                ratio = torch.exp(log_probs - batch_old_log_probs)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.minimum(unclipped, clipped).mean() - self.entropy_coef * entropy

                values_pred = self.critic(batch_states)
                critic_loss = torch.nn.functional.mse_loss(values_pred, batch_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.value_loss_coef * critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropy_values.append(float(entropy.item()))

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropy_values)),
        }

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.algorithm,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            output_path,
        )

    def load_actor(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])


def _compute_gae(
    *,
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_value = float(last_value)
    for step in reversed(range(len(rewards))):
        nonterminal = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[step] = gae
        next_value = values[step]
    returns = advantages + values
    return advantages, returns


if torch is not None:

    class PPOActor(nn.Module):
        """Gaussian actor with tanh-squashed continuous actions."""

        def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
            super().__init__()
            self.backbone = _build_mlp(state_dim, hidden_sizes)
            last_dim = int(hidden_sizes[-1]) if hidden_sizes else state_dim
            self.mean = nn.Linear(last_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

        def forward(self, state):
            features = self.backbone(state)
            mean = self.mean(features)
            log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
            return mean, log_std

        def sample(self, state):
            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            pre_tanh = normal.rsample()
            action = torch.tanh(pre_tanh)
            log_prob = _squashed_log_prob(normal, pre_tanh, action)
            entropy = normal.entropy().sum(dim=-1, keepdim=True)
            return action, log_prob, entropy

        def deterministic(self, state):
            mean, _ = self.forward(state)
            return torch.tanh(mean)

        def log_prob(self, state, action):
            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            clipped_action = action.clamp(-1.0 + EPSILON, 1.0 - EPSILON)
            pre_tanh = torch.atanh(clipped_action)
            return _squashed_log_prob(normal, pre_tanh, clipped_action)

        def entropy(self, state):
            mean, log_std = self.forward(state)
            del mean
            std = log_std.exp()
            normal = torch.distributions.Normal(torch.zeros_like(std), std)
            return normal.entropy().sum(dim=-1, keepdim=True)


    class ValueNetwork(nn.Module):
        """State-value baseline network."""

        def __init__(self, state_dim: int, hidden_sizes: Sequence[int]):
            super().__init__()
            self.net = _build_mlp(state_dim, hidden_sizes, output_dim=1)

        def forward(self, state):
            return self.net(state)


    def _squashed_log_prob(normal, pre_tanh, action):
        log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + EPSILON)
        return log_prob.sum(dim=-1, keepdim=True)


    def _build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int | None = None):
        layers: list[nn.Module] = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(previous_dim, int(hidden_dim)))
            layers.append(nn.Tanh())
            previous_dim = int(hidden_dim)
        if output_dim is not None:
            layers.append(nn.Linear(previous_dim, int(output_dim)))
        return nn.Sequential(*layers)

else:

    class PPOActor:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class ValueNetwork:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()
