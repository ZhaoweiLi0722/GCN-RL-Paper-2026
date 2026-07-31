"""Exploration noise processes."""

from __future__ import annotations

import numpy as np


class GaussianNoise:
    """Independent Gaussian exploration noise."""

    def __init__(self, action_dim: int, std: float = 0.1, seed: int = 0):
        self.action_dim = int(action_dim)
        self.std = float(std)
        self.rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        return self.rng.normal(0.0, self.std, size=self.action_dim).astype(np.float32)


class OUNoise:
    """Ornstein-Uhlenbeck exploration noise for deterministic policies."""

    def __init__(
        self,
        action_dim: int,
        seed: int = 0,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        self.action_dim = int(action_dim)
        self.mu = np.full(action_dim, mu, dtype=np.float32)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.normal(size=self.action_dim)
        self.state = (self.state + dx).astype(np.float32)
        return self.state

    def state_dict(self) -> dict:
        return {
            "action_dim": int(self.action_dim),
            "mu": self.mu.copy(),
            "theta": float(self.theta),
            "sigma": float(self.sigma),
            "state": self.state.copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["action_dim"]) != self.action_dim:
            raise ValueError("OU-noise action dimension does not match")
        if not np.array_equal(
            np.asarray(state["mu"], dtype=np.float32),
            self.mu,
        ):
            raise ValueError("OU-noise mean does not match")
        if float(state["theta"]) != self.theta:
            raise ValueError("OU-noise theta does not match")
        if float(state["sigma"]) != self.sigma:
            raise ValueError("OU-noise sigma does not match")
        restored = np.asarray(state["state"], dtype=np.float32)
        if restored.shape != self.state.shape:
            raise ValueError("OU-noise state shape does not match")
        self.state = restored.copy()
        self.rng.bit_generator.state = state["rng_state"]
