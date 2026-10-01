"""Explicit reference-prior candidate policy; no environment or fitting.

Zero output layers make the initial residual/value exactly zero. The prior
preserves the reference greedy request, not its deterministic sampling law.
"""

from __future__ import annotations

import math

from src.models.candidate_policy import CandidatePolicy, CandidateScores
from src.rl.networks import require_torch, torch


if torch is not None:
    class ReferencePriorCandidatePolicy(CandidatePolicy):
        def __init__(self, schema, *, nonreference_mass, enabled=False, architecture,
                     message_mode, encoder_width, head_width, seed):
            # Keep the two-class prior distinct from a tie/point mass in float32.
            margin = 8 * torch.finfo(torch.float32).eps
            if (type(nonreference_mass) not in (int, float)
                    or not math.isfinite(nonreference_mass)
                    or not margin <= nonreference_mass <= .5 - margin):
                raise ValueError("explicit numerically resolvable nonreference mass in (0, .5) required")
            super().__init__(schema, enabled=enabled, architecture=architecture,
                             message_mode=message_mode, encoder_width=encoder_width,
                             head_width=head_width, seed=seed)
            self.nonreference_mass = float(nonreference_mass)
            with torch.no_grad():
                for head in (self.score_head, self.value_head):
                    head[-1].weight.zero_()
                    head[-1].bias.zero_()

        def manifest(self):
            return super().manifest() | {
                "format": "reference-prior-candidate-v1",
                "initialization": "zero_scorer_and_value_output_layers",
                "reference_prior": {
                    "nonreference_mass": self.nonreference_mass,
                    "allocation": "uniform_over_unique_nonreference_request_classes",
                    "singleton": "probability_one",
                    "application": "fixed_log_prior_plus_learned_residual",
                },
            }

        def forward(self, observation, candidates):
            residual = super().forward(observation, candidates)
            k = len(candidates.class_keys)
            if k == 1:
                prior = torch.zeros_like(residual.logits)
            else:
                prior = torch.full_like(residual.logits, math.log(self.nonreference_mass / (k - 1)))
                prior[candidates.reference_class] = math.log1p(-self.nonreference_mass)
            return CandidateScores(residual.logits + prior, residual.value, residual.actor_state)
else:
    class ReferencePriorCandidatePolicy:
        def __init__(self, *args, **kwargs):
            require_torch()
