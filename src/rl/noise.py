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
