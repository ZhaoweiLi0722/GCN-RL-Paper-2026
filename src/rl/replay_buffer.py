"""Replay buffer for off-policy continuous-control algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Fixed-size NumPy replay buffer."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int, seed: int = 0):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.position] = np.asarray(state, dtype=np.float32)
        self.actions[self.position] = np.asarray(action, dtype=np.float32)
        self.rewards[self.position] = float(reward)
        self.next_states[self.position] = np.asarray(next_state, dtype=np.float32)
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.size < batch_size:
            raise ValueError(f"Cannot sample batch_size={batch_size}; buffer has {self.size} items")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return ReplayBatch(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size
