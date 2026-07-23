"""A linear-quadratic regulator (LQR) task with a known optimal policy.

Why LQR for verification: for a linear system with quadratic cost, the optimal
feedback policy and its cost are computable exactly from the discrete algebraic
Riccati equation (DARE). That gives a *precise*, dependency-free reference to
check a learned agent against — an agent that cannot approach the analytic
optimum on this easy task has a bug, and we find out in seconds rather than
after a full capacity-planning run.

The environment matches the interface the repo's agents expect:
``reset(seed=...) -> state`` and ``step(action) -> (next_state, reward, done,
info)``, with normalized actions in ``[-1, 1]`` (scaled internally to the real
control range).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def solve_discrete_are(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    max_iterations: int = 10000,
    tolerance: float = 1e-11,
) -> np.ndarray:
    """Solve the discrete-time algebraic Riccati equation by iteration.

    Returns the stabilizing solution ``P`` of
    ``P = Q + Aᵀ P A - Aᵀ P B (R + Bᵀ P B)⁻¹ Bᵀ P A``.

    Iteration keeps the dependency surface to NumPy only (no SciPy). For the
    small, controllable systems used in verification this converges quickly.
    """

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    P = Q.copy()
    for _ in range(max_iterations):
        BtP = B.T @ P
        S = R + BtP @ B
        K = np.linalg.solve(S, BtP @ A)
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(P_next - P)) < tolerance:
            return P_next
        P = P_next
    raise RuntimeError("DARE iteration did not converge; check controllability")


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Optimal LQR feedback gain ``K`` for control ``u = -K x``."""

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    P = solve_discrete_are(A, B, Q, R)
    S = R + B.T @ P @ B
    return np.linalg.solve(S, B.T @ P @ A)


@dataclass
class LinearQuadraticEnv:
    """Episodic LQR control task.

    Dynamics: ``x' = A x + B u + w``, ``w ~ N(0, process_noise² I)``.
    Cost per step: ``xᵀ Q x + uᵀ R u``; reward is the negative cost.
    The agent emits a normalized action ``a ∈ [-1, 1]ᵐ``; the applied control is
    ``u = action_scale · a``.
    """

    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    action_scale: float = 3.0
    process_noise: float = 0.05
    init_spread: float = 1.0
    horizon: int = 50
    state_clip: float = 1e3

    state_dim: int = field(init=False)
    action_dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=np.float64)
        self.B = np.asarray(self.B, dtype=np.float64)
        self.Q = np.asarray(self.Q, dtype=np.float64)
        self.R = np.asarray(self.R, dtype=np.float64)
        self.state_dim = int(self.A.shape[0])
        self.action_dim = int(self.B.shape[1])
        self._rng = np.random.default_rng(0)
        self._state = np.zeros(self.state_dim, dtype=np.float64)
        self._t = 0

    # Metadata used by the shared action projection helper.
    @property
    def action_size(self) -> int:
        return self.action_dim

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._state = self._rng.uniform(
            -self.init_spread, self.init_spread, size=self.state_dim
        )
        self._t = 0
        return self._state.astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if a.shape != (self.action_dim,):
            raise ValueError(f"Expected action of shape {(self.action_dim,)}, got {a.shape}")
        u = self.action_scale * a

        cost = float(self._state @ self.Q @ self._state + u @ self.R @ u)
        noise = self._rng.normal(0.0, self.process_noise, size=self.state_dim)
        next_state = self.A @ self._state + self.B @ u + noise
        next_state = np.clip(next_state, -self.state_clip, self.state_clip)

        self._state = next_state
        self._t += 1
        done = self._t >= self.horizon
        info = {"cost": cost}
        return next_state.astype(np.float32), -cost, done, info

    def optimal_gain(self) -> np.ndarray:
        """LQR gain for the *real* control; use with ``u = -K x``."""

        return lqr_gain(self.A, self.B, self.Q, self.R)

    def optimal_normalized_action(self, state: np.ndarray, gain: np.ndarray) -> np.ndarray:
        """Optimal control mapped back into the normalized ``[-1, 1]`` action."""

        u = -gain @ np.asarray(state, dtype=np.float64)
        return np.clip(u / self.action_scale, -1.0, 1.0).astype(np.float32)


def make_double_integrator(horizon: int = 50) -> LinearQuadraticEnv:
    """A small, controllable, mildly-unstable benchmark system.

    A double integrator needs active control to stay bounded, so a random policy
    diverges (high cost) while the LQR policy regulates it — a wide, reliable
    separation for verification.
    """

    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2)
    R = np.array([[0.1]])
    return LinearQuadraticEnv(A=A, B=B, Q=Q, R=R, horizon=horizon)
