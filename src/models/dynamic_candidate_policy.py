"""Opt-in dynamic request scorer with an independent public-state value model.

Only canonical class features are scored; original submitted requests remain
owned by RequestCandidates. Graph and flat models share information and head
widths, not parameter counts. This module contains no optimizer or environment.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from numbers import Real

from src.models.candidate_policy import CandidateScores
from src.models.gcn import GraphConvolution
from src.models.matched_inputs import InputSchema, build_matched_inputs
from src.rl.networks import nn, require_torch, torch
from src.rl.routing_candidate_contract import RequestCandidates


if torch is not None:
    class _FiniteScalarHead(nn.Sequential):
        def forward(self, inputs):
            for layer in self:
                inputs = layer(inputs)
                if not torch.isfinite(inputs).all().item():
                    raise ValueError("nonfinite scalar head activation")
            return inputs


    class _PublicStateEncoder(nn.Module):
        def __init__(self, schema, architecture, encoder_width):
            super().__init__()
            n, f = len(schema.node_ids), len(schema.node_feature_names)
            self.architecture = architecture
            self.projection = (GraphConvolution(f, encoder_width) if architecture == "graph"
                               else nn.Linear(n * f, n * encoder_width))

        def forward(self, view, reference):
            if self.architecture == "graph":
                encoded = self.projection(view.nodes, view.message_adjacency).flatten(1)
            else:
                encoded = self.projection(view.nodes.flatten(1))
            if not torch.isfinite(encoded).all().item():
                raise ValueError("nonfinite public-state encoding")
            return torch.cat((torch.tanh(encoded), view.context, reference), dim=1)


    class _DynamicActor(nn.Module):
        def __init__(self, schema, architecture, encoder_width, head_width,
                     base_width, initial_reference_bias):
            super().__init__()
            self.encoder = _PublicStateEncoder(schema, architecture, encoder_width)
            self.score_head = _FiniteScalarHead(
                nn.Linear(base_width + len(schema.action_names) + 2, head_width),
                nn.Tanh(), nn.Linear(head_width, 1),
            )
            self.reference_bias = nn.Parameter(torch.tensor(initial_reference_bias, dtype=torch.float32))

        def forward(self, view, reference, features, roles):
            base = self.encoder(view, reference)
            per_class = torch.cat((base.expand(features.shape[0], -1), features, roles), dim=1)
            return self.score_head(per_class).squeeze(1) + self.reference_bias * roles[:, 0]


    class _DynamicCritic(nn.Module):
        def __init__(self, schema, architecture, encoder_width, head_width, base_width):
            super().__init__()
            self.encoder = _PublicStateEncoder(schema, architecture, encoder_width)
            self.value_head = _FiniteScalarHead(
                nn.Linear(base_width, head_width), nn.Tanh(), nn.Linear(head_width, 1),
            )

        def forward(self, view, reference):
            return self.value_head(self.encoder(view, reference)).reshape(())


    def _initialize(factory, seed):
        # Reset after conversion so ambient default dtype cannot change the start.
        # The second seed reproduces ordinary float32 Linear construction order.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            module = factory().to(device="cpu", dtype=torch.float32)
            torch.random.default_generator.manual_seed(seed)
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    child.reset_parameters()
        return module


    class DynamicCandidatePolicy(nn.Module):
        """One shared scalar scorer per current class, plus action-independent V(s).

        Actor and critic initialization seeds are independent. The actor uses
        PyTorch's default Linear initialization; the critic uses that rule then
        zeros its final output layer. The reference preference is an ordinary
        trainable actor parameter, not a fixed probability prior or output bound.
        """

        def __init__(self, schema, *, enabled=False, architecture, message_mode,
                     encoder_width, head_width, actor_seed, critic_seed,
                     initial_reference_bias):
            super().__init__()
            if enabled is not True:
                raise ValueError("dynamic candidate policy must be explicitly enabled")
            if not isinstance(schema, InputSchema):
                raise TypeError("explicit InputSchema required")
            if architecture not in ("graph", "flat") or message_mode not in ("physical", "self_only"):
                raise ValueError("unknown architecture or message mode")
            if architecture == "flat" and message_mode != "self_only":
                raise ValueError("flat encoder has no physical message operator")
            for name, value in (("encoder_width", encoder_width), ("head_width", head_width)):
                if type(value) is not int or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
            for name, value in (("actor_seed", actor_seed), ("critic_seed", critic_seed)):
                if type(value) is not int or not 0 <= value < 2**63:
                    raise ValueError(f"{name} must be an explicit nonnegative CPU initialization seed")
            if isinstance(initial_reference_bias, bool) or not isinstance(initial_reference_bias, Real):
                raise ValueError("initial_reference_bias must be a finite real scalar")
            try:
                bias = float(initial_reference_bias)
            except (OverflowError, ValueError) as exc:
                raise ValueError("initial_reference_bias must be finite in float32") from exc
            if not math.isfinite(bias) or abs(bias) > torch.finfo(torch.float32).max:
                raise ValueError("initial_reference_bias must be finite in float32")
            n, a = len(schema.node_ids), len(schema.action_names)
            if a != 4 * n:
                raise ValueError("dynamic candidate policy requires the four-group facility-net schema")
            self.schema, self.architecture, self.message_mode = schema, architecture, message_mode
            self.encoder_width, self.head_width = encoder_width, head_width
            self.actor_seed, self.critic_seed = actor_seed, critic_seed
            self.initial_reference_bias = bias
            base_width = n * encoder_width + len(schema.global_feature_names) + n * n + 2 * a
            self.actor = _initialize(
                lambda: _DynamicActor(schema, architecture, encoder_width, head_width, base_width, bias),
                actor_seed,
            )
            self.critic = _initialize(
                lambda: _DynamicCritic(schema, architecture, encoder_width, head_width, base_width),
                critic_seed,
            )
            self._initial_reference_bias_float32 = self.actor.reference_bias.item()
            with torch.no_grad():
                self.critic.value_head[-1].weight.zero_()
                self.critic.value_head[-1].bias.zero_()

        def actor_parameters(self):
            yield from self.actor.parameters()

        def critic_parameters(self):
            yield from self.critic.parameters()

        def _parameter_reference(self):
            parameters = tuple(self.parameters())
            reference = parameters[0]
            if (reference.device.type != "cpu" or reference.dtype not in (torch.float32, torch.float64)
                    or any(p.device != reference.device or p.dtype != reference.dtype for p in parameters)):
                raise ValueError("dynamic policy supports uniform CPU float32/float64 only")
            if any(not torch.isfinite(p).all().item() for p in parameters):
                raise ValueError("nonfinite dynamic policy parameter")
            return reference

        def manifest(self):
            parameter = self._parameter_reference()
            actor_count = sum(p.numel() for p in self.actor_parameters())
            critic_count = sum(p.numel() for p in self.critic_parameters())
            return {
                "format": "dynamic-candidate-policy-v1", "schema": asdict(self.schema),
                "architecture": self.architecture, "message_mode": self.message_mode,
                "encoder_width": self.encoder_width, "head_width": self.head_width,
                "dtype": str(parameter.dtype), "device": str(parameter.device),
                "initialization": {
                    "actor_seed": self.actor_seed, "critic_seed": self.critic_seed,
                    "construction_dtype": "torch.float32", "construction_device": "cpu",
                    "linear_rule": "torch.nn.Linear.reset_parameters_in_module_order",
                    "actor": "random_encoder_and_score_head",
                    "critic": "independent_random_encoder_and_head_then_zero_final_weight_and_bias",
                    "initial_reference_bias": self.initial_reference_bias,
                    "initial_reference_bias_float32": self._initial_reference_bias_float32,
                    "torch_version": str(torch.__version__),
                },
                "reference_preference": "trainable_actor_scalar_times_reference_indicator",
                "outputs": "linear_unbounded_score_and_state_value_no_fixed_prior_or_clamp",
                "information": {
                    "state": "public_nodes_globals_physical_links_anchor_and_reference_requests",
                    "actor_extra": "canonical_class_features_reference_and_anchor_indicators",
                    "critic_extra": "none_no_candidate_action_selected_action_or_support_pooling",
                    "actor_state": "matched_actor_flat_includes_anchor_excludes_reference",
                    "request_precision": "network_features_cast_only_original_requests_retained_by_bank",
                },
                "parameter_counts": {"actor": actor_count, "critic": critic_count,
                                     "total": actor_count + critic_count, "shared": 0},
                "graph_flat_parameter_counts_matched": False,
            }

        def definition_sha256(self):
            return hashlib.sha256(json.dumps(self.manifest(), sort_keys=True, allow_nan=False).encode()).hexdigest()

        def snapshot_sha256(self):
            digest = hashlib.sha256(json.dumps(self.manifest(), sort_keys=True, allow_nan=False).encode())
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
            parameter = self._parameter_reference()
            anchor = torch.tensor([candidates.requests[1]], dtype=parameter.dtype, device="cpu")
            # Both models use actor-role public inputs; critic-role adds an action.
            view = build_matched_inputs(observation, self.schema, anchor,
                                        role="actor", message_mode=self.message_mode)
            if view.nodes.shape[0] != 1:
                raise ValueError("one decision state required; no implicit ragged batching")
            reference = torch.tensor([candidates.requests[0]], dtype=parameter.dtype, device="cpu")
            features = torch.tensor(candidates.class_features, dtype=parameter.dtype, device="cpu")
            roles = torch.tensor([[i == candidates.reference_class, i == candidates.anchor_class]
                                  for i in range(len(candidates.class_keys))],
                                 dtype=parameter.dtype, device="cpu")
            if not torch.isfinite(features).all().item():
                raise ValueError("nonfinite candidate features")
            logits = self.actor(view, reference, features, roles)
            value = self.critic(view, reference)
            if not torch.isfinite(logits).all().item() or not torch.isfinite(value).item():
                raise ValueError("nonfinite candidate score or value")
            return CandidateScores(logits, value, tuple(view.flat[0].detach().tolist()))
else:
    class DynamicCandidatePolicy:
        def __init__(self, *args, **kwargs):
            require_torch()
