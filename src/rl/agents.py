"""Lazy agent registry used by training and evaluation CLIs."""

from __future__ import annotations

from typing import Any


def available_algorithms() -> tuple[str, ...]:
    from src.baselines.heuristics import available_heuristics

    learned = ("flat_ddpg", "gcn_ddpg", "ppo", "sac", "td3")
    return tuple(sorted((*learned, *available_heuristics())))


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
    if algorithm == "sac":
        from src.baselines.sac import SACAgent

        return SACAgent
    if algorithm == "ppo":
        from src.baselines.ppo import PPOAgent

        return PPOAgent
    from src.baselines.heuristics import available_heuristics

    if algorithm in available_heuristics():
        from src.baselines.heuristics import get_heuristic_class

        return get_heuristic_class(algorithm)
    raise ValueError(f"Unsupported algorithm: {algorithm}")
