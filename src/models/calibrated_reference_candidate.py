"""Engineering-only calibrated scorer; deliberately unregistered by collectors.

Fixed divisors change neural units, never submitted requests or raw receipts.
The old P2 implementation and its checkpoints remain untouched.
"""

from dataclasses import asdict, dataclass
import math

from src.models.candidate_policy import CandidateScores
from src.models.matched_inputs import build_matched_inputs
from src.models.reference_prior_candidate import ReferencePriorCandidatePolicy
from src.rl.networks import require_torch, torch
from src.rl.routing_candidate_contract import RequestCandidates


@dataclass(frozen=True)
class CandidateCalibration:
    node_divisors: tuple
    global_divisors: tuple
    action_divisors: tuple
    link_divisor: float
    logit_gain: float
    value_gain: float

    def __post_init__(self):
        for name in ("node_divisors", "global_divisors", "action_divisors"):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError("ordered immutable divisor tuples required")
        values = (*self.node_divisors, *self.global_divisors, *self.action_divisors,
                  self.link_divisor, self.logit_gain, self.value_gain)
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("finite positive calibration constants required")

    def validate(self, schema):
        if (len(self.node_divisors) != len(schema.node_feature_names)
                or len(self.global_divisors) != len(schema.global_feature_names)
                or len(self.action_divisors) != len(schema.action_names)):
            raise ValueError("calibration dimensions must match the explicit schema")


if torch is not None:
    class CalibratedReferenceCandidatePolicy(ReferencePriorCandidatePolicy):
        def __init__(self, schema, *, calibration, **kwargs):
            if not isinstance(calibration, CandidateCalibration):
                raise TypeError("explicit CandidateCalibration required")
            calibration.validate(schema)
            self.calibration = calibration
            super().__init__(schema, **kwargs)

        def manifest(self):
            return super().manifest() | {
                "format": "calibrated-reference-candidate-engineering-v1",
                "calibration": asdict(self.calibration),
                "neural_units": "fixed_divisors_no_clipping_no_fitted_statistics",
                "adjacency": "original_unscaled_physical_operator",
                "receipt_and_submission": "original_unscaled_public_values",
                "scientific_runner_integrated": False,
            }

        def forward(self, observation, candidates):
            if not isinstance(candidates, RequestCandidates):
                raise TypeError("validated request candidates required")
            if (candidates.schema.action_schema_id != self.schema.definition_id + "/action"
                    or candidates.schema.num_facilities != len(self.schema.node_ids)):
                raise ValueError("candidate and policy schemas differ")
            p = next(self.parameters())
            if (p.device.type != "cpu" or p.dtype not in (torch.float32, torch.float64)
                    or any(v.dtype != p.dtype or v.device != p.device for v in self.parameters())):
                raise ValueError("uniform CPU float32/float64 required")
            make = lambda values: torch.tensor(values, dtype=p.dtype, device="cpu")
            c = self.calibration
            anchor = make([candidates.requests[1]])
            raw = build_matched_inputs(observation, self.schema, anchor,
                                       role="actor", message_mode=self.message_mode)
            if raw.nodes.shape[0] != 1 or raw.nodes.dtype != p.dtype:
                raise ValueError("one decision with model-matching precision required")
            nodes = raw.nodes / make(c.node_divisors)
            action_units = make(c.action_divisors)
            context = torch.cat((observation.globals / make(c.global_divisors),
                                 observation.physical_links.flatten(1) / c.link_divisor,
                                 anchor / action_units), dim=1)
            reference = make([candidates.requests[0]]) / action_units
            features = make(candidates.class_features) / action_units
            if not all(torch.isfinite(x).all().item() for x in (nodes, context, reference, features)):
                raise ValueError("calibrated input overflow")
            encoded = (self.encoder(nodes, raw.message_adjacency).flatten(1)
                       if self.architecture == "graph" else self.encoder(nodes.flatten(1)))
            base = torch.cat((torch.tanh(encoded), context, reference), dim=1)
            roles = make([[i == candidates.reference_class, i == candidates.anchor_class]
                          for i in range(len(candidates.class_keys))])
            per_class = torch.cat((base.expand(features.shape[0], -1), features, roles), dim=1)
            residual = self.score_head(per_class).squeeze(1)
            logits = c.logit_gain * (residual - residual[candidates.reference_class])
            k = len(candidates.class_keys)
            if k > 1:
                prior = torch.full_like(logits, math.log(self.nonreference_mass / (k-1)))
                prior[candidates.reference_class] = math.log1p(-self.nonreference_mass)
                logits = logits + prior
            value = c.value_gain * self.value_head(base).reshape(())
            if not torch.isfinite(logits).all().item() or not torch.isfinite(value).item():
                raise ValueError("nonfinite calibrated output")
            return CandidateScores(logits, value, tuple(raw.flat[0].detach().tolist()))
else:
    class CalibratedReferenceCandidatePolicy:
        def __init__(self, *args, **kwargs):
            require_torch()
