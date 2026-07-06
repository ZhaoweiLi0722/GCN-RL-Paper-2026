"""Flat-state / MLP-DDPG baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.rl.action_projection import project_action
from src.rl.networks import MLPActor, MLPCritic, require_torch, resolve_torch_device, torch
from src.rl.noise import OUNoise
from src.rl.preprocessing import FixedObservationScaler, reward_scale_from_config
from src.rl.replay_buffer import ReplayBuffer


class FlatDDPGAgent:
    """DDPG agent that consumes a flattened environment observation."""

    algorithm = "flat_ddpg"

    def __init__(self, state_dim: int, action_dim: int, config: dict[str, Any]):
        require_torch()
        seed = int(config.get("seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.batch_size = int(config.get("batch_size", 128))
        self.reward_scale = reward_scale_from_config(config)
        self.observation_scaler = FixedObservationScaler.from_config(config, state_dim)
        hidden_sizes = tuple(config.get("hidden_sizes", [256, 256]))
        self.device = resolve_torch_device(config.get("device"))

        self.actor = MLPActor(state_dim, action_dim, hidden_sizes).to(self.device)
        self.actor_target = MLPActor(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic = MLPCritic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic_target = MLPCritic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(config.get("actor_lr", 1e-4)))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(config.get("critic_lr", 1e-3)))
        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=int(config.get("replay_buffer_size", 1000000)),
            seed=seed,
        )
        exploration = config.get("exploration_noise", {})
        self.noise = OUNoise(
            action_dim=action_dim,
            seed=seed,
            theta=float(exploration.get("theta", 0.15)),
            sigma=float(exploration.get("sigma", 0.2)),
        )

    def reset(self) -> None:
        self.noise.reset()

    def select_action(self, state: np.ndarray, explore: bool = True, env=None) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            state_tensor = self.observation_scaler.normalize_tensor(state_tensor)
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        if explore:
            action = action + self.noise.sample()
        return project_action(action, env_state=env, action_space_info=self.action_dim).action

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, float(reward) * self.reward_scale, next_state, done)

    def update(self) -> dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        states = self.observation_scaler.normalize_tensor(states)
        next_states = self.observation_scaler.normalize_tensor(next_states)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            q_targets = rewards + self.gamma * (1.0 - dones) * target_q

        q_expected = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item())}

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

    def _soft_update(self, local_model, target_model) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
