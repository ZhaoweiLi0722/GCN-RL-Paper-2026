"""Opt-in CPU candidate scorer/value baseline, not a registered PPO learner.

Graph, self-only and flat controls share public numerical inputs and heads.
Flat and graph encoders do not have matched parameter counts. No optimizer,
environment, hidden-state mask or historical checkpoint is used here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from src.models.gcn import GraphConvolution
from src.models.matched_inputs import InputSchema, build_matched_inputs
from src.rl.networks import nn, require_torch, torch
from src.rl.routing_candidate_contract import RequestCandidates


@dataclass(frozen=True)
class CandidateScores:
    logits: object
    value: object
    actor_state: tuple[float, ...]


if torch is not None:
    class CandidatePolicy(nn.Module):
        def __init__(self, schema, *, enabled=False, architecture, message_mode,
                     encoder_width, head_width, seed):
            super().__init__()
            if enabled is not True:
                raise ValueError("candidate policy must be explicitly enabled")
            if not isinstance(schema, InputSchema):
                raise TypeError("explicit InputSchema required")
            if architecture not in ("graph", "flat") or message_mode not in ("physical", "self_only"):
                raise ValueError("unknown architecture or message mode")
            if architecture == "flat" and message_mode != "self_only":
                raise ValueError("flat encoder has no physical message operator")
            for name, value in (("encoder_width", encoder_width), ("head_width", head_width)):
                if type(value) is not int or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError("explicit nonnegative CPU initialization seed required")
            n, f, a = len(schema.node_ids), len(schema.node_feature_names), len(schema.action_names)
            if a != 4 * n:
                raise ValueError("candidate policy requires the four-group facility-net schema")
            self.schema, self.architecture, self.message_mode = schema, architecture, message_mode
            self.encoder_width, self.head_width = encoder_width, head_width
            base_width = n * encoder_width + len(schema.global_feature_names) + n * n + 2 * a
            # Seed CPU initialization only; preserve the caller's global CPU RNG.
            with torch.random.fork_rng(devices=[]), torch.device("cpu"):
                torch.random.default_generator.manual_seed(seed)
                self.encoder = (GraphConvolution(f, encoder_width) if architecture == "graph"
                                else nn.Linear(n * f, n * encoder_width))
                self.score_head = nn.Sequential(nn.Linear(base_width + a + 2, head_width),
                                                nn.Tanh(), nn.Linear(head_width, 1))
                self.value_head = nn.Sequential(nn.Linear(base_width, head_width),
                                                nn.Tanh(), nn.Linear(head_width, 1))
            self.to(dtype=torch.float32, device="cpu")

        def manifest(self):
            return {"format": "candidate-policy-v1", "schema": asdict(self.schema),
                    "architecture": self.architecture, "message_mode": self.message_mode,
                    "encoder_width": self.encoder_width, "head_width": self.head_width,
                    "dtype": str(next(self.parameters()).dtype), "device": str(next(self.parameters()).device)}

        def definition_sha256(self):
            return hashlib.sha256(json.dumps(self.manifest(), sort_keys=True).encode()).hexdigest()

        def snapshot_sha256(self):
            digest = hashlib.sha256(json.dumps(self.manifest(), sort_keys=True).encode())
            for name, tensor in self.state_dict().items():
                digest.update(name.encode())
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.detach().cpu().numpy().tobytes())
            return digest.hexdigest()

        def forward(self, observation, candidates):
            if not isinstance(candidates, RequestCandidates):
                raise TypeError("validated request candidates required")
            if (candidates.schema.action_schema_id != self.schema.definition_id + "/action"
                    or candidates.schema.num_facilities != len(self.schema.node_ids)):
                raise ValueError("candidate and policy action schemas differ")
            parameter = next(self.parameters())
            if (parameter.device.type != "cpu" or parameter.dtype not in (torch.float32, torch.float64)
                    or any(p.device != parameter.device or p.dtype != parameter.dtype for p in self.parameters())):
                raise ValueError("engineering policy supports uniform CPU float32/float64 only")
            anchor = torch.tensor([candidates.requests[1]], dtype=parameter.dtype, device="cpu")
            view = build_matched_inputs(observation, self.schema, anchor,
                                         role="actor", message_mode=self.message_mode)
            if view.nodes.shape[0] != 1:
                raise ValueError("one decision state required; no implicit ragged batching")
            if self.architecture == "graph":
                encoded = self.encoder(view.nodes, view.message_adjacency).flatten(1)
            else:
                encoded = self.encoder(view.nodes.flatten(1))
            reference = torch.tensor([candidates.requests[0]], dtype=parameter.dtype, device="cpu")
            base = torch.cat((torch.tanh(encoded), view.context, reference), dim=1)
            features = torch.tensor(candidates.class_features, dtype=parameter.dtype, device="cpu")
            roles = torch.tensor([[i == candidates.reference_class, i == candidates.anchor_class]
                                  for i in range(len(candidates.class_keys))], dtype=parameter.dtype)
            per_class = torch.cat((base.expand(features.shape[0], -1), features, roles), dim=1)
            logits = self.score_head(per_class).squeeze(1)
            value = self.value_head(base).reshape(())
            if not torch.isfinite(logits).all().item() or not torch.isfinite(value).item():
                raise ValueError("nonfinite candidate score or value")
            return CandidateScores(logits, value, tuple(view.flat[0].detach().tolist()))
else:
    class CandidatePolicy:
        def __init__(self, *args, **kwargs):
            require_torch()
