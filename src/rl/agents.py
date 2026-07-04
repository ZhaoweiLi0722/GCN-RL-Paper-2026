"""Lazy agent registry used by training and evaluation CLIs."""

from __future__ import annotations

from typing import Any


def available_algorithms() -> tuple[str, ...]:
    return ("flat_ddpg", "gcn_ddpg", "iso", "mdl1", "mdl2", "myo", "td3")


def get_agent_class(algorithm: str) -> Any:
    if algorithm == "flat_ddpg":
        from src.baselines.flat_ddpg import FlatDDPGAgent

        return FlatDDPGAgent
    if algorithm == "gcn_ddpg":
        from src.models.gcn_ddpg import GCNDDPGAgent

        return GCNDDPGAgent
    if algorithm == "td3":
        from src.baselines.td3 import TD3Agent

        return TD3Agent
    if algorithm in ("myo", "iso", "mdl1", "mdl2"):
        from src.baselines.heuristics import get_heuristic_class

        return get_heuristic_class(algorithm)
    raise ValueError(f"Unsupported algorithm: {algorithm}")
