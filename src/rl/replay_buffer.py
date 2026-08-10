"""Replay buffer for off-policy continuous-control algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    one_step_rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    discount_multipliers: np.ndarray
    online_masks: np.ndarray
    online_fraction: float = 0.0


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
        self.one_step_rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.discount_multipliers = np.ones((capacity, 1), dtype=np.float32)
        self.online_mask = np.zeros(capacity, dtype=np.bool_)
        self.collecting_online = False
        self.position = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        *,
        discount_multiplier: float = 1.0,
        one_step_reward: float | None = None,
    ) -> None:
        self.states[self.position] = np.asarray(state, dtype=np.float32)
        self.actions[self.position] = np.asarray(action, dtype=np.float32)
        self.rewards[self.position] = float(reward)
        self.one_step_rewards[self.position] = float(
            reward if one_step_reward is None else one_step_reward
        )
        self.next_states[self.position] = np.asarray(next_state, dtype=np.float32)
        self.dones[self.position] = float(done)
        self.discount_multipliers[self.position] = float(discount_multiplier)
        self.online_mask[self.position] = bool(self.collecting_online)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def begin_online_collection(self) -> None:
        """Mark future transitions as online interaction data."""

        self.collecting_online = True

    def sample(
        self,
        batch_size: int,
        *,
        online_fraction: float | None = None,
    ) -> ReplayBatch:
        if self.size < batch_size:
            raise ValueError(f"Cannot sample batch_size={batch_size}; buffer has {self.size} items")
        if online_fraction is None:
            indices = self.rng.choice(
                self.size,
                size=batch_size,
                replace=False,
            )
        else:
            requested = float(online_fraction)
            if not 0.0 <= requested <= 1.0:
                raise ValueError("online_fraction must be in [0, 1]")
            online_indices = np.flatnonzero(self.online_mask[:self.size])
            offline_indices = np.flatnonzero(~self.online_mask[:self.size])
            online_count = min(
                int(round(batch_size * requested)),
                int(online_indices.size),
            )
            offline_count = min(
                batch_size - online_count,
                int(offline_indices.size),
            )
            remaining = batch_size - online_count - offline_count
            if remaining:
                extra_online = min(
                    remaining,
                    int(online_indices.size) - online_count,
                )
                online_count += extra_online
                remaining -= extra_online
            if remaining:
                offline_count += min(
                    remaining,
                    int(offline_indices.size) - offline_count,
                )
            chosen_online = (
                self.rng.choice(
                    online_indices,
                    size=online_count,
                    replace=False,
                )
                if online_count
                else np.empty(0, dtype=np.int64)
            )
            chosen_offline = (
                self.rng.choice(
                    offline_indices,
                    size=offline_count,
                    replace=False,
                )
                if offline_count
                else np.empty(0, dtype=np.int64)
            )
            indices = np.concatenate((chosen_online, chosen_offline))
            self.rng.shuffle(indices)
        sampled_online_fraction = float(
            self.online_mask[indices].mean()
        )
        return ReplayBatch(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            one_step_rewards=self.one_step_rewards[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices],
            discount_multipliers=self.discount_multipliers[indices],
            online_masks=self.online_mask[indices].reshape(-1, 1).copy(),
            online_fraction=sampled_online_fraction,
        )

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict[str, Any]:
        """Return the populated buffer and RNG state for exact resumption."""

        size = int(self.size)
        return {
            "capacity": int(self.capacity),
            "state_dim": int(self.states.shape[1]),
            "action_dim": int(self.actions.shape[1]),
            "position": int(self.position),
            "size": size,
            "states": self.states[:size].copy(),
            "actions": self.actions[:size].copy(),
            "rewards": self.rewards[:size].copy(),
            "one_step_rewards": self.one_step_rewards[:size].copy(),
            "next_states": self.next_states[:size].copy(),
            "dones": self.dones[:size].copy(),
            "discount_multipliers": self.discount_multipliers[:size].copy(),
            "online_mask": self.online_mask[:size].copy(),
            "collecting_online": bool(self.collecting_online),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`."""

        expected = (
            int(self.capacity),
            int(self.states.shape[1]),
            int(self.actions.shape[1]),
        )
        observed = (
            int(state["capacity"]),
            int(state["state_dim"]),
            int(state["action_dim"]),
        )
        if observed != expected:
            raise ValueError(
                "Replay-buffer shape mismatch: "
                f"checkpoint={observed}, current={expected}"
            )
        size = int(state["size"])
        position = int(state["position"])
        if not 0 <= size <= self.capacity:
            raise ValueError(f"Invalid replay-buffer size: {size}")
        if not 0 <= position < self.capacity:
            raise ValueError(f"Invalid replay-buffer position: {position}")
        arrays = {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "one_step_rewards": self.one_step_rewards,
            "next_states": self.next_states,
            "dones": self.dones,
            "discount_multipliers": self.discount_multipliers,
        }
        for name, target in arrays.items():
            if name == "discount_multipliers":
                default = np.ones((size, 1), dtype=np.float32)
            elif name == "one_step_rewards":
                default = state["rewards"]
            else:
                default = None
            values = np.asarray(state.get(name, default), dtype=np.float32)
            if values.shape != target[:size].shape:
                raise ValueError(
                    f"Replay-buffer {name} shape mismatch: "
                    f"checkpoint={values.shape}, current={target[:size].shape}"
                )
            target.fill(1.0 if name == "discount_multipliers" else 0.0)
            target[:size] = values
        online_mask = np.asarray(
            state.get("online_mask", np.zeros(size, dtype=np.bool_)),
            dtype=np.bool_,
        )
        if online_mask.shape != self.online_mask[:size].shape:
            raise ValueError(
                "Replay-buffer online_mask shape mismatch: "
                f"checkpoint={online_mask.shape}, "
                f"current={self.online_mask[:size].shape}"
            )
        self.online_mask.fill(False)
        self.online_mask[:size] = online_mask
        self.collecting_online = bool(
            state.get("collecting_online", False)
        )
        self.size = size
        self.position = position
        self.rng.bit_generator.state = state["rng_state"]
