"""Soft Actor-Critic baseline for continuous capacity planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.rl.action_projection import project_action
from src.rl.networks import nn, require_torch, torch
from src.rl.replay_buffer import ReplayBuffer


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPSILON = 1e-6


class SACAgent:
    """Entropy-regularized off-policy actor-critic agent."""

    algorithm = "sac"

    def __init__(self, state_dim: int, action_dim: int, config: dict[str, Any]):
        require_torch()
        seed = int(config.get("seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.batch_size = int(config.get("batch_size", 256))
        hidden_sizes = tuple(config.get("hidden_sizes", [256, 256]))
        device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)

        self.actor = SquashedGaussianActor(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(config.get("actor_lr", 3e-4)))
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=float(config.get("critic_lr", 3e-4)))
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=float(config.get("critic_lr", 3e-4)))

        self.automatic_entropy_tuning = bool(config.get("automatic_entropy_tuning", True))
        if self.automatic_entropy_tuning:
            self.target_entropy = float(config.get("target_entropy", -float(action_dim)))
            self.log_alpha = torch.tensor(
                np.log(float(config.get("alpha", 0.2))),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(config.get("alpha_lr", 3e-4)))
        else:
            self.alpha = float(config.get("alpha", 0.2))
            self.log_alpha = None
            self.alpha_optimizer = None

        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=int(config.get("replay_buffer_size", 1000000)),
            seed=seed,
        )

    @property
    def current_alpha(self):
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp()
        return torch.tensor(self.alpha, dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        return None

    def select_action(self, state: np.ndarray, explore: bool = True, env=None) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            if explore:
                action, _log_prob, _mean_action = self.actor.sample(state_tensor)
            else:
                action = self.actor.deterministic(state_tensor)
            action_array = action.cpu().numpy()[0]
        self.actor.train()
        return project_action(action_array, env_state=env, action_space_info=self.action_dim).action

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.minimum(target_q1, target_q2) - self.current_alpha.detach() * next_log_probs
            q_targets = rewards + self.gamma * (1.0 - dones) * target_q

        q1_expected = self.critic1(states, actions)
        q2_expected = self.critic2(states, actions)
        critic1_loss = torch.nn.functional.mse_loss(q1_expected, q_targets)
        critic2_loss = torch.nn.functional.mse_loss(q2_expected, q_targets)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        sampled_actions, log_probs, _ = self.actor.sample(states)
        q1_pi = self.critic1(states, sampled_actions)
        q2_pi = self.critic2(states, sampled_actions)
        min_q_pi = torch.minimum(q1_pi, q2_pi)
        actor_loss = (self.current_alpha.detach() * log_probs - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics = {
            "actor_loss": float(actor_loss.item()),
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "alpha": float(self.current_alpha.detach().item()),
        }

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            metrics["alpha_loss"] = float(alpha_loss.item())

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        return metrics

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.algorithm,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
            },
            output_path,
        )

    def load_actor(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])

    def _soft_update(self, local_model, target_model) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


if torch is not None:

    class SquashedGaussianActor(nn.Module):
        """Gaussian actor with tanh-squashed continuous actions."""

        def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
            super().__init__()
            self.backbone = _build_mlp(state_dim, hidden_sizes)
            last_dim = int(hidden_sizes[-1]) if hidden_sizes else state_dim
            self.mean = nn.Linear(last_dim, action_dim)
            self.log_std = nn.Linear(last_dim, action_dim)

        def forward(self, state):
            features = self.backbone(state)
            mean = self.mean(features)
            log_std = self.log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
            return mean, log_std

        def sample(self, state):
            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            pre_tanh = normal.rsample()
            action = torch.tanh(pre_tanh)
            log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + EPSILON)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            return action, log_prob, torch.tanh(mean)

        def deterministic(self, state):
            mean, _ = self.forward(state)
            return torch.tanh(mean)


    class QNetwork(nn.Module):
        """State-action Q network."""

        def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
            super().__init__()
            self.net = _build_mlp(state_dim + action_dim, hidden_sizes, output_dim=1)

        def forward(self, state, action):
            return self.net(torch.cat((state, action), dim=-1))


    def _build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int | None = None):
        layers: list[nn.Module] = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(previous_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            previous_dim = int(hidden_dim)
        if output_dim is not None:
            layers.append(nn.Linear(previous_dim, int(output_dim)))
        return nn.Sequential(*layers)

else:

    class SquashedGaussianActor:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class QNetwork:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()
