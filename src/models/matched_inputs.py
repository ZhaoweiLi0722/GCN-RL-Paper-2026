"""Opt-in common-information views; no historical agent uses this prototype.

The producer declares feature meaning, units, time availability and ordering.
This module checks that declaration and tensor layout, not its authenticity.
It implements no policy, feature aggregation, action projection or training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.rl.networks import nn, require_torch, torch


@dataclass(frozen=True)
class InputSchema:
    definition_id: str
    node_ids: tuple[str, ...]
    node_feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    action_names: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.definition_id, str) or not self.definition_id.strip():
            raise ValueError("explicit feature/action definition ID required")
        for name in ("node_ids", "node_feature_names", "global_feature_names", "action_names"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise ValueError(f"{name} must be an ordered tuple")
            if not values and name != "global_feature_names":
                raise ValueError(f"{name} must not be empty")
            if any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"{name} contains an invalid identifier")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicates")


@dataclass(frozen=True)
class ObservationBatch:
    schema: InputSchema
    nodes: Any
    globals: Any
    physical_links: Any


@dataclass(frozen=True)
class MatchedInputs:
    schema: InputSchema
    role: str
    message_mode: str
    nodes: Any
    context: Any
    flat: Any
    message_adjacency: Any


def _tensor(value, shape, name, reference=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor; implicit conversion is forbidden")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} shape mismatch; no implicit broadcasting")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")
    if reference is not None and (value.dtype != reference.dtype or value.device != reference.device):
        raise ValueError(f"{name} dtype/device mismatch")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")


def build_matched_inputs(
    observation: ObservationBatch,
    expected_schema: InputSchema,
    anchor_actions,
    *,
    role: str,
    message_mode: str,
    action=None,
    detach_proposal: bool | None = None,
) -> MatchedInputs:
    """Expose identical numerical information in graph and flat layouts.

    Actor: nodes, globals, physical links, anchor.
    Critic: the same plus candidate action in the declared action coordinates.
    Gate: the same plus pre-gate proposed action minus anchor; detach is explicit.
    Message ablation changes only the neural operator, not physical metadata.
    """
    require_torch()
    if not isinstance(expected_schema, InputSchema) or not isinstance(observation, ObservationBatch):
        raise TypeError("explicit ObservationBatch and expected InputSchema required")
    if observation.schema != expected_schema:
        raise ValueError("feature/action schema or ordering mismatch")
    if role not in ("actor", "critic", "gate"):
        raise ValueError("unknown input role")
    if message_mode not in ("physical", "self_only"):
        raise ValueError("unknown neural message mode")
    nodes = observation.nodes
    if not isinstance(nodes, torch.Tensor) or nodes.ndim != 3 or nodes.shape[0] < 1:
        raise ValueError("nodes must be a nonempty (batch, node, feature) tensor")
    batch, count = nodes.shape[0], len(expected_schema.node_ids)
    width = len(expected_schema.action_names)
    _tensor(nodes, (batch, count, len(expected_schema.node_feature_names)), "nodes")
    _tensor(observation.globals, (batch, len(expected_schema.global_feature_names)), "globals", nodes)
    _tensor(observation.physical_links, (batch, count, count), "physical_links", nodes)
    _tensor(anchor_actions, (batch, width), "anchor_actions", nodes)
    links = observation.physical_links
    if (links < 0).any().item() or not torch.equal(links, links.transpose(1, 2)):
        raise ValueError("physical_links must be nonnegative and symmetric")
    if torch.count_nonzero(links.diagonal(dim1=1, dim2=2)).item():
        raise ValueError("physical_links must exclude neural self-loops")
    parts = [observation.globals, links.reshape(batch, -1), anchor_actions]
    if role == "actor":
        if action is not None or detach_proposal is not None:
            raise ValueError("actor cannot consume its own proposal or gate detach setting")
    else:
        _tensor(action, (batch, width), "action", nodes)
        if role == "critic":
            if detach_proposal is not None:
                raise ValueError("critic action must not use gate detach setting")
            parts.append(action)
        else:
            if type(detach_proposal) is not bool:
                raise ValueError("gate detach_proposal policy must be explicit")
            delta = action - anchor_actions
            parts.append(delta.detach() if detach_proposal else delta)

    context = torch.cat(parts, dim=1)
    _tensor(context, context.shape, "assembled context", nodes)
    identity = torch.eye(count, dtype=nodes.dtype, device=nodes.device).expand(batch, -1, -1)
    if message_mode == "physical":
        # Batched equivalent of the existing GCN's symmetric self-loop normalization.
        adjacency = links + identity
        degree = adjacency.sum(dim=2)
        if not torch.isfinite(degree).all().item():
            raise ValueError("message degree overflow")
        inverse = degree.rsqrt()
        adjacency = inverse[:, :, None] * adjacency * inverse[:, None, :]
    else:
        adjacency = identity.clone()
    nodes = nodes.clone()
    return MatchedInputs(expected_schema, role, message_mode, nodes, context,
                         torch.cat((nodes.reshape(batch, -1), context), dim=1), adjacency)


def parameter_inventory(*, actor, critic, gate=None) -> dict[str, Any]:
    """Count explicitly supplied deployment/training components, not target copies.

    Both all and currently trainable unique parameters are reported. Sharing is
    deduplicated by Parameter identity across components, not just Module identity.
    This is not an architecture matcher or a compute/latency equivalence check.
    """
    require_torch()
    components = {"actor": actor, "critic": critic, "gate": gate}
    counts, unique = {}, {}
    for name, module in components.items():
        if module is None and name == "gate":
            counts[name] = 0
            continue
        if not isinstance(module, nn.Module):
            raise TypeError(f"{name} must be an explicit neural module")
        parameters = {id(p): p for p in module.parameters()}
        counts[name] = sum(p.numel() for p in parameters.values())
        unique.update(parameters)
    total = sum(p.numel() for p in unique.values())
    return {"components": counts, "total_unique": total,
            "trainable_unique": sum(p.numel() for p in unique.values() if p.requires_grad),
            "shared_duplicate_numel": sum(counts.values()) - total}
