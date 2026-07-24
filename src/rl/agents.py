"""Lazy agent registry used by training and evaluation CLIs."""

from __future__ import annotations

from typing import Any


def available_algorithms() -> tuple[str, ...]:
    from src.baselines.heuristics import available_heuristics

    learned = (
        "flat_ddpg",
        "flat_residual_iso",
        "flat_residual_mdl2",
        "flat_residual_mdl2_replenish_ddpg_afd",
        "flat_residual_myo",
        "flat_residual_pmyo",
        "gcn_ddpg",
        "gcn_pure_ddpg",
        "gcn_ppo",
        "gcn_mdl2_shield_selector",
        "gcn_pmyo_shield_selector",
        "gcn_residual_iso",
        "gcn_residual_mdl2",
        "gcn_residual_mdl2_replenish_ddpg",
        "gcn_residual_mdl2_replenish_ddpg_afd",
        "gcn_residual_myo",
        "gcn_residual_pmyo",
        "gcn_residual_mdl2_replenish_td3",
        "gcn_residual_mdl2_replenish_td3_afd",
        "gcn_residual_mdl2_td3",
        "gcn_residual_mdl2_shield_td3",
        "gcn_residual_pmyo_risk_pressure_td3",
        "gcn_residual_pmyo_risk_replenish_td3",
        "gcn_residual_pmyo_rebalance_td3",
        "gcn_residual_pmyo_shield_td3",
        "gcn_residual_pmyo_td3",
        "gcn_residual_pmyo_transfer_td3_bc",
        "gcn_residual_pmyo_transfer_td3",
        "gcn_sac",
        "gcn_td3",
        "ppo",
        "sac",
        "td3",
    )
    return tuple(sorted((*learned, *available_heuristics())))


def get_agent_class(algorithm: str) -> Any:
    if algorithm in {
        "flat_ddpg",
        "flat_residual_iso",
        "flat_residual_mdl2",
        "flat_residual_mdl2_replenish_ddpg_afd",
        "flat_residual_myo",
        "flat_residual_pmyo",
    }:
        from src.baselines.flat_ddpg import FlatDDPGAgent

        return FlatDDPGAgent
    if algorithm == "gcn_ddpg":
        from src.models.gcn_ddpg import GCNDDPGAgent

        return GCNDDPGAgent
    if algorithm in {
        "gcn_pure_ddpg",
        "gcn_residual_iso",
        "gcn_residual_mdl2",
        "gcn_residual_mdl2_replenish_ddpg",
        "gcn_residual_mdl2_replenish_ddpg_afd",
        "gcn_residual_myo",
        "gcn_residual_pmyo",
    }:
        from src.models.gcn_ddpg import GCNDDPGAgent

        return GCNDDPGAgent
    if algorithm in {
        "gcn_td3",
        "gcn_residual_mdl2_replenish_td3",
        "gcn_residual_mdl2_replenish_td3_afd",
        "gcn_residual_mdl2_td3",
        "gcn_residual_mdl2_shield_td3",
        "gcn_residual_pmyo_risk_pressure_td3",
        "gcn_residual_pmyo_risk_replenish_td3",
        "gcn_residual_pmyo_shield_td3",
        "gcn_residual_pmyo_td3",
        "gcn_residual_pmyo_rebalance_td3",
        "gcn_residual_pmyo_transfer_td3_bc",
        "gcn_residual_pmyo_transfer_td3",
    }:
        from src.models.gcn_td3 import GCNTD3Agent

        return GCNTD3Agent
    if algorithm == "gcn_sac":
        from src.models.gcn_sac import GCNSACAgent

        return GCNSACAgent
    if algorithm == "gcn_ppo":
        from src.models.gcn_ppo import GCNPPOAgent

        return GCNPPOAgent
    if algorithm in {"gcn_mdl2_shield_selector", "gcn_pmyo_shield_selector"}:
        from src.models.gcn_shield_selector import GCNShieldSelectorAgent

        return GCNShieldSelectorAgent
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
