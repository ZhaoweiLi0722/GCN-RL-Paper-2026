"""Opt-in common-input forward agent. Frozen engineering weights, no optimizer.

This is not a registered algorithm, fitted model, or replacement for old DDPG.
Graph and flat heads share raw information, not equal parameter counts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib

import numpy as np

from src.models.gcn import GraphConvolution
from src.models.matched_inputs import ObservationBatch, build_matched_inputs, parameter_inventory
from src.rl.networks import MLPActor, MLPCritic, MLPCorrectionGate, nn, require_torch, torch


@dataclass(frozen=True)
class ForwardDecision:
    proposal: np.ndarray
    gate: np.ndarray
    request: np.ndarray


if torch is not None:
    class _CommonHead(nn.Module):
        def __init__(self, schema, role, architecture, hidden_width):
            super().__init__()
            self.schema, self.role = schema, role
            n, f, a = len(schema.node_ids), len(schema.node_feature_names), len(schema.action_names)
            self.encoder = GraphConvolution(f, f) if architecture == "graph" else None
            state_width = n * f + len(schema.global_feature_names) + n * n + a
            hidden = (hidden_width,)
            if role == "actor":
                self.head = MLPActor(state_width, a, hidden)
            elif role == "critic":
                self.head = MLPCritic(state_width, a, hidden)
            else:
                self.head = MLPCorrectionGate(state_width + a, hidden, output_dim=a)

        def forward(self, view):
            if view.schema != self.schema or view.role != self.role:
                raise ValueError("head input contract mismatch")
            nodes = view.nodes
            if self.encoder is not None:
                nodes = torch.relu(self.encoder(nodes, view.message_adjacency))
            flat = torch.cat((nodes.flatten(1), view.context), dim=1)
            a = len(self.schema.action_names)
            if self.role == "critic":
                return self.head(flat[:, :-a], flat[:, -a:])
            output = self.head(flat)
            return torch.sigmoid(output) if self.role == "gate" else output


    class ProspectiveForwardAgent(nn.Module):
        def __init__(self, contract, *, enabled=False, architecture, message_mode,
                     hidden_width, residual_scale, seed):
            super().__init__()
            if enabled is not True:
                raise ValueError("prospective agent must be explicitly enabled")
            if architecture not in ("graph", "flat") or message_mode not in ("physical", "self_only"):
                raise ValueError("unknown architecture or message mode")
            if type(hidden_width) is not int or hidden_width < 1:
                raise ValueError("hidden_width must be positive integer")
            if not np.isfinite(residual_scale) or not 0 <= residual_scale <= 1:
                raise ValueError("invalid residual scale")
            self.contract, self.message_mode = contract, message_mode
            # Isolate initialization from other experiments' global torch RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.actor = _CommonHead(contract.inputs, "actor", architecture, hidden_width)
                self.critic = _CommonHead(contract.inputs, "critic", architecture, hidden_width)
                self.gate = _CommonHead(contract.inputs, "gate", architecture, hidden_width)
            self.target_actor = copy.deepcopy(self.actor)
            self.target_critic = copy.deepcopy(self.critic)
            self.target_gate = copy.deepcopy(self.gate)
            scales = torch.zeros(len(contract.inputs.action_names), dtype=torch.float32)
            scales[:len(contract.inputs.node_ids)] = residual_scale
            self.register_buffer("residual_scales", scales)
            self.requires_grad_(False)
            self.eval()

        def _parts(self, actor_view):
            schema = self.contract.inputs
            if (actor_view.schema != schema or actor_view.role != "actor"
                    or actor_view.message_mode != self.message_mode):
                raise ValueError("agent actor-view contract mismatch")
            n, g = len(schema.node_ids), len(schema.global_feature_names)
            obs = ObservationBatch(schema, actor_view.nodes, actor_view.context[:, :g],
                                   actor_view.context[:, g:g + n * n].reshape(-1, n, n))
            return obs, actor_view.context[:, g + n * n:]

        @torch.no_grad()
        def policy_tensors(self, actor_view, *, target=False):
            obs, anchor = self._parts(actor_view)
            actor = self.target_actor if target else self.actor
            gate_head = self.target_gate if target else self.gate
            proposal = torch.clamp(anchor + self.residual_scales * actor(actor_view), -1, 1)
            gate_view = build_matched_inputs(obs, self.contract.inputs, anchor, role="gate",
                                             message_mode=self.message_mode, action=proposal,
                                             detach_proposal=True)
            gate = gate_head(gate_view)
            request = torch.clamp(anchor + gate * (proposal - anchor), -1, 1)
            if not all(torch.isfinite(t).all().item() for t in (proposal, gate, request)):
                raise ValueError("nonfinite policy output")
            return proposal, gate, request

        @torch.no_grad()
        def select_action(self, observation, anchor):
            view = build_matched_inputs(observation, self.contract.inputs, anchor,
                                        role="actor", message_mode=self.message_mode)
            if view.flat.shape[0] != 1:
                raise ValueError("collector action selection expects exactly one state")
            proposal, gate, request = self.policy_tensors(view)
            return ForwardDecision(*(t[0].cpu().numpy().copy() for t in (proposal, gate, request)))

        @torch.no_grad()
        def replay_forward(self, batch):
            for view in (batch.current_actor, batch.current_critic, batch.next_actor):
                if view.schema != self.contract.inputs or view.message_mode != self.message_mode:
                    raise ValueError("replay neural input contract mismatch")
            for sample in batch.samples:
                if sample.semantics != self.contract.replay:
                    raise ValueError("replay semantics mismatch")
            current_q = self.critic(batch.current_critic)
            _, _, next_request = self.policy_tensors(batch.next_actor, target=True)
            obs, anchor = self._parts(batch.next_actor)
            target_view = build_matched_inputs(obs, self.contract.inputs, anchor, role="critic",
                                               message_mode=self.message_mode, action=next_request)
            next_q = self.target_critic(target_view)
            if not torch.isfinite(current_q).all().item():
                raise ValueError("nonfinite current critic output")
            return current_q, next_q, batch.targets(next_q)

        def inventory(self):
            return parameter_inventory(actor=self.actor, critic=self.critic, gate=self.gate)

        def weights_digest(self):
            digest = hashlib.sha256()
            for name, tensor in self.state_dict().items():
                digest.update(name.encode())
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.detach().cpu().numpy().tobytes())
            return digest.hexdigest()
else:
    class ProspectiveForwardAgent:
        def __init__(self, *args, **kwargs):
            require_torch()
