"""GNN-PPO: PPO backbone on a GCN encoder (Phase 6).

Mirrors ``src/baselines/ppo.py`` exactly — clipped surrogate objective, GAE,
value baseline — with the MLP actor/value nets replaced by a GCN Gaussian actor
and a GCN value head, and flat states converted to node features via
``flat_state_to_node_features``. Reuses the flat module's ``PPOTransition`` and
``_compute_gae`` so the on-policy bookkeeping stays identical. Learns from scratch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.ppo import PPOTransition, _compute_gae
from src.models.gcn import GCNGaussianActor, GCNValue
from src.models.graph_features import build_graph_spec, flat_state_to_node_features
from src.rl.action_projection import project_action
from src.rl.networks import require_torch, resolve_torch_device, torch
from src.rl.preprocessing import reward_scale_from_config


class GCNPPOAgent:
    """Clipped-objective PPO agent with a GCN-encoded Gaussian actor."""

    algorithm = "gcn_ppo"

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
        self.reward_scale = reward_scale_from_config(config)
        self.graph_spec = build_graph_spec(config, state_dim)
        self.device = resolve_torch_device(config.get("device"))

        gcn_hidden_sizes = tuple(config.get("gcn_hidden_sizes", [64, 64]))
        head_hidden_sizes = tuple(config.get("hidden_sizes", [256, 256]))
        include_global_context = bool(config.get("include_global_context", True))

        self.actor = GCNGaussianActor(
            self.graph_spec.node_feature_dim,
            self.graph_spec.num_facilities,
            self.graph_spec.num_nodes,
            action_dim,
            self.graph_spec.edge_index,
            gcn_hidden_sizes,
            head_hidden_sizes,
            include_global_context=include_global_context,
        ).to(self.device)
        self.critic = GCNValue(
            self.graph_spec.node_feature_dim,
            self.graph_spec.num_facilities,
            self.graph_spec.num_nodes,
            self.graph_spec.edge_index,
            gcn_hidden_sizes,
            head_hidden_sizes,
            include_global_context=include_global_context,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(config.get("actor_lr", 3e-4)))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(config.get("critic_lr", 1e-3)))
        self.rng = np.random.default_rng(seed)
        self.rollout: list[PPOTransition] = []
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._last_next_state: np.ndarray | None = None
        self._last_done = False

    def _nodes(self, states):
        return flat_state_to_node_features(states, self.graph_spec)

    def reset(self) -> None:
        return None

    def select_action(self, state: np.ndarray, explore: bool = True, env=None) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        node_features = self._nodes(state_tensor)
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            if explore:
                action_tensor, log_prob_tensor, _entropy = self.actor.sample(node_features)
            else:
                action_tensor = self.actor.deterministic(node_features)
                log_prob_tensor = self.actor.log_prob(node_features, action_tensor)
            value_tensor = self.critic(node_features)
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
                reward=float(reward) * self.reward_scale,
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
        node_features = self._nodes(states)
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
                last_value = float(self.critic(self._nodes(next_tensor)).cpu().numpy()[0, 0])

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

        batch_size = node_features.shape[0]
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropy_values: list[float] = []
        for _epoch in range(self.train_epochs):
            indices = self.rng.permutation(batch_size)
            for start in range(0, batch_size, self.minibatch_size):
                idx = torch.as_tensor(indices[start : start + self.minibatch_size], dtype=torch.long, device=self.device)
                batch_nodes = node_features[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages_tensor[idx]
                batch_returns = returns_tensor[idx]

                log_probs = self.actor.log_prob(batch_nodes, batch_actions)
                entropy = self.actor.entropy(batch_nodes).mean()
                ratio = torch.exp(log_probs - batch_old_log_probs)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.minimum(unclipped, clipped).mean() - self.entropy_coef * entropy

                values_pred = self.critic(batch_nodes)
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
