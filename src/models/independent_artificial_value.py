"""Independent public-state baseline for the fixed-bank artificial diagnostic."""

import copy
import math

from src.models.matched_inputs import build_matched_inputs
from src.rl.networks import nn, require_torch, torch


if torch is not None:
    class IndependentArtificialValue(nn.Module):
        def __init__(self, schema, units, *, output_gain):
            super().__init__()
            if type(output_gain) not in (int, float) or not math.isfinite(output_gain) or output_gain <= 0:
                raise ValueError("positive finite value output gain required")
            self.schema, self.units = schema, copy.deepcopy(units)
            self.output_gain = float(output_gain)
            n, f, a = len(schema.node_ids), len(schema.node_feature_names), len(schema.action_names)
            sizes = {"node_divisors": f, "global_divisors": len(schema.global_feature_names), "action_divisors": a}
            for name, size in sizes.items():
                if len(units[name]) != size or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                                                 for v in units[name]):
                    raise ValueError("explicit matching positive neural units required")
            if type(units["link_divisor"]) not in (int, float) or not math.isfinite(units["link_divisor"]) or units["link_divisor"] <= 0:
                raise ValueError("invalid link divisor")
            # Allocate zero weights directly: constructing this baseline consumes no RNG.
            self.weight = nn.Parameter(torch.zeros(n*f + len(schema.global_feature_names) + n*n + 2*a,
                                                   dtype=torch.float32, device="cpu"))
            self.bias = nn.Parameter(torch.zeros((), dtype=torch.float32, device="cpu"))

        def forward(self, observation, bank):
            p, u = self.weight, self.units
            if p.dtype != torch.float32 or p.device.type != "cpu" or self.bias.dtype != p.dtype or self.bias.device != p.device:
                raise ValueError("uniform CPU float32 value parameters required")
            if bank.schema.action_schema_id != self.schema.definition_id + "/action":
                raise ValueError("bank action schema mismatch")
            make = lambda x: torch.tensor(x, dtype=p.dtype, device=p.device)
            anchor = make([bank.requests[1]])
            public = build_matched_inputs(observation, self.schema, anchor, role="actor", message_mode="self_only")
            if public.nodes.shape[0] != 1 or public.nodes.dtype != p.dtype:
                raise ValueError("one CPU float32 public context required")
            features = torch.cat(((public.nodes/make(u["node_divisors"])).flatten(1),
                observation.globals/make(u["global_divisors"]),
                observation.physical_links.flatten(1)/u["link_divisor"],
                anchor/make(u["action_divisors"]), make([bank.requests[0]])/make(u["action_divisors"])), dim=1)
            result = self.output_gain*(features @ self.weight + self.bias)
            if not torch.isfinite(features).all() or not torch.isfinite(result).all():
                raise ValueError("nonfinite value input/output")
            return result.squeeze(0)
else:
    class IndependentArtificialValue:
        def __init__(self, *args, **kwargs):
            require_torch()
