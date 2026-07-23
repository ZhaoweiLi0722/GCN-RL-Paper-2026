"""Algorithm verification utilities.

A dependency-free, analytically-grounded control task used to sanity-check that
the repo's RL agents actually learn before they are trusted to produce paper
numbers (the V2/V3 verification gate in ``specs/tech-stack.md``).
"""

from src.verification.lqr_env import (
    LinearQuadraticEnv,
    lqr_gain,
    make_double_integrator,
    solve_discrete_are,
)

__all__ = [
    "LinearQuadraticEnv",
    "lqr_gain",
    "make_double_integrator",
    "solve_discrete_are",
]
