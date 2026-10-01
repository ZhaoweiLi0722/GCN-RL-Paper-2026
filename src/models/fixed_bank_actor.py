"""Unregistered actor-only positive control for a fixed, public candidate bank."""

from dataclasses import asdict
import hashlib
import json
import math

from src.models.candidate_policy import CandidateScores
from src.models.gcn import GraphConvolution
from src.models.matched_inputs import InputSchema, build_matched_inputs
from src.rl.networks import nn, torch, require_torch
from src.rl.routing_candidate_contract import RequestCandidates


if torch is not None:
    class FixedBankActor(nn.Module):
        def __init__(self, schema, bank, *, architecture, message_mode,
                     encoder_width, seed, units, initial_reference_logit):
            super().__init__()
            if not isinstance(schema, InputSchema) or not isinstance(bank, RequestCandidates):
                raise TypeError("explicit public schema and validated bank required")
            if architecture not in ("graph", "flat") or message_mode not in ("physical", "self_only"):
                raise ValueError("invalid encoder")
            if architecture == "flat" and message_mode != "self_only":
                raise ValueError("flat has no physical message operator")
            if type(seed) is not int or not 0 <= seed < 2**63 or type(encoder_width) is not int or encoder_width < 1:
                raise ValueError("explicit valid width and initialization seed required")
            n, f, a = len(schema.node_ids), len(schema.node_feature_names), len(schema.action_names)
            if a != 4*n or bank.schema.num_facilities != n or bank.schema.action_schema_id != schema.definition_id + "/action":
                raise ValueError("bank and observation schemas differ")
            if set(units) != {"node_divisors", "global_divisors", "action_divisors", "link_divisor", "logit_gain"}:
                raise ValueError("explicit fixed neural units required")
            sizes = {"node_divisors": f, "global_divisors": len(schema.global_feature_names), "action_divisors": a}
            for key, size in sizes.items():
                if not isinstance(units[key], (list, tuple)) or len(units[key]) != size:
                    raise ValueError("unit dimensions differ from schema")
            constants = [v for key in sizes for v in units[key]] + [units["link_divisor"], units["logit_gain"], initial_reference_logit]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in constants):
                raise ValueError("finite positive units and initial tie-break required")
            if not 8*torch.finfo(torch.float32).eps <= initial_reference_logit <= .001:
                raise ValueError("initial tie-break must be small and resolvable")
            self.schema, self.architecture, self.message_mode = schema, architecture, message_mode
            self.encoder_width, self.initial_reference_logit = encoder_width, initial_reference_logit
            self.units = {key: tuple(value) if key in sizes else value for key, value in units.items()}
            self.bank_contract = self._bank_contract(bank)
            width = n*encoder_width + len(schema.global_feature_names) + n*n + 2*a
            with torch.random.fork_rng(devices=[]), torch.device("cpu"):
                torch.random.default_generator.manual_seed(seed)
                self.encoder = GraphConvolution(f, encoder_width) if architecture == "graph" else nn.Linear(n*f, n*encoder_width)
                self.actor_head = nn.Linear(width, len(bank.class_keys))
                with torch.no_grad():
                    self.actor_head.weight.zero_()
                    self.actor_head.bias.zero_()
                    self.actor_head.bias[bank.reference_class] = initial_reference_logit / units["logit_gain"]
            self.to(dtype=torch.float32, device="cpu")

        @staticmethod
        def _bank_contract(bank):
            return {"schema": asdict(bank.schema), "keys": bank.class_keys,
                    "reference": bank.requests[0], "anchor": bank.requests[1],
                    "representative_requests": tuple(bank.requests[i] for i in bank.representatives)}

        def manifest(self):
            return {"format": "fixed-bank-actor-positive-control-v1", "schema": asdict(self.schema),
                    "architecture": self.architecture, "message_mode": self.message_mode,
                    "encoder_width": self.encoder_width, "units": self.units, "bank": self.bank_contract,
                    "initial_reference_logit": self.initial_reference_logit,
                    "fixed_prior": False, "critic": False, "scientific_runner_integrated": False,
                    "dtype": str(next(self.parameters()).dtype), "device": str(next(self.parameters()).device)}

        def snapshot_sha256(self):
            digest = hashlib.sha256(json.dumps(self.manifest(), sort_keys=True).encode())
            for name, tensor in self.state_dict().items():
                digest.update(name.encode())
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.detach().cpu().numpy().tobytes())
            return digest.hexdigest()

        def forward(self, observation, candidates):
            if not isinstance(candidates, RequestCandidates):
                raise TypeError("validated candidates required")
            if self._bank_contract(candidates) != self.bank_contract:
                raise ValueError("this positive control requires its fixed candidate bank")
            p = next(self.parameters())
            if p.device.type != "cpu" or p.dtype not in (torch.float32, torch.float64) or any(
                    x.dtype != p.dtype or x.device != p.device for x in self.parameters()):
                raise ValueError("uniform CPU float32/float64 required")
            make = lambda x: torch.tensor(x, dtype=p.dtype, device="cpu")
            anchor = make([candidates.requests[1]])
            raw = build_matched_inputs(observation, self.schema, anchor, role="actor", message_mode=self.message_mode)
            if raw.nodes.shape[0] != 1 or raw.nodes.dtype != p.dtype:
                raise ValueError("one model-precision decision required")
            u = self.units
            nodes = raw.nodes / make(u["node_divisors"])
            context = torch.cat((observation.globals / make(u["global_divisors"]),
                observation.physical_links.flatten(1)/u["link_divisor"], anchor/make(u["action_divisors"]),
                make([candidates.requests[0]])/make(u["action_divisors"])), dim=1)
            if not torch.isfinite(nodes).all() or not torch.isfinite(context).all():
                raise ValueError("nonfinite normalized public input")
            encoded = self.encoder(nodes, raw.message_adjacency).flatten(1) if self.architecture == "graph" else self.encoder(nodes.flatten(1))
            logits = u["logit_gain"]*self.actor_head(torch.cat((torch.tanh(encoded), context), dim=1)).flatten()
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite actor output")
            return CandidateScores(logits, logits.new_zeros(()), tuple(raw.flat[0].detach().tolist()))
else:
    class FixedBankActor:
        def __init__(self, *args, **kwargs):
            require_torch()
