"""Graph ablation helpers for experiment configuration."""

from __future__ import annotations

from dataclasses import replace

from src.env.capacity_planning import CapacityPlanningConfig


def with_graph_ablation(
    config: CapacityPlanningConfig,
    *,
    remove_specimen_edges: bool = False,
    remove_capacity_edges: bool = False,
    remove_resource_edges: bool = False,
) -> CapacityPlanningConfig:
    """Return a copy of a config with selected sharing edge sets removed."""

    updates = {}
    if remove_specimen_edges:
        updates["specimen_edges"] = ()
    if remove_capacity_edges:
        updates["capacity_edges"] = ()
    if remove_resource_edges:
        updates["resource_edges"] = ()
    return replace(config, **updates)
