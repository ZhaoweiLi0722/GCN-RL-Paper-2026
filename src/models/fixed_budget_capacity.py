"""Versioned fixed-total redistribution, not the legacy raw-hours projection.

Only the encoder/head are inherited. Checkpoint owners must persist and check
``actor.metadata()``: tensor keys/shapes alone intentionally do not identify
the action semantics. This module never samples noise, fits, or loads a model.
"""

from __future__ import annotations

from dataclasses import asdict
import math

from src.models.capacity_ddpg import CapacityActor, CapacityModelConfig, validate_features
from src.rl.networks import require_torch, torch


FORMAT = "fixed-budget-capacity-logits-v1"
SITES = 4
TOTAL_HOURS = 8.0
CENTER_HOURS = 2.0
HOUR_BOUNDS = (0.5, 3.5)
NATIVE_SUM_TOLERANCE = 4 * math.ulp(TOTAL_HOURS)


def _validate_logits(logits):
    require_torch()
    if (not isinstance(logits, torch.Tensor) or logits.device.type != "cpu"
            or logits.dtype != torch.float32 or logits.ndim not in (1, 2)
            or logits.shape[-1] != SITES or logits.numel() == 0
            or not torch.isfinite(logits).all().item()):
        raise ValueError("finite CPU float32 [4] or [batch, 4] logits required")


def _hours64(logits, noise=None):
    _validate_logits(logits)
    values = logits.to(dtype=torch.float64)
    if noise is not None:
        _validate_logits(noise)
        if noise.shape != logits.shape:
            raise ValueError("noise must have exactly the logits shape; no broadcasting")
        # Addition in double also keeps finite extreme float32 inputs finite.
        values = values + noise.to(dtype=torch.float64)
    bounded = torch.tanh(values)
    return CENTER_HOURS + bounded - bounded.mean(dim=-1, keepdim=True)


def project_fixed_budget_logits(logits, noise=None):
    """Differentiable physical hours for a critic, returned as CPU float32.

The map is 2 + tanh(z) - mean(tanh(z)), with total eight in exact
arithmetic and site closure [0.5, 3.5], not the full capped simplex.
Supplied noise is added to logits before tanh; omit it for critic actions.
Float32 rounding is subject to the existing critic's admissibility tolerance.
The map is permutation-equivariant, NOT common-logit-shift-invariant.
"""
    return _hours64(logits, noise=noise).to(dtype=torch.float32)


def native_fixed_budget_hours(logits, noise=None):
    """Map one already-computed CPU float32 logit vector to four native doubles.

No actor forward is performed. Noise, when supplied, is a same-shape float32
logit perturbation; the caller owns its Gaussian sigma/stream. Returned values
are detached physical requests, not the differentiable critic representation.
The last component is closed only after the first three returned doubles exist.
``fsum`` must not exceed eight and may undershoot by at most four float64 ulps;
the normal four-site closure sums to exactly eight. No tolerance is changed in
the existing simulator or live projection.
"""
    _validate_logits(logits)
    if logits.ndim != 1:
        raise ValueError("native mapping requires one [4] vector; select the batch row explicitly")
    with torch.no_grad():
        first = tuple(float(x) for x in _hours64(logits, noise=noise)[:3].tolist())
    hours = first + (TOTAL_HOURS - math.fsum(first),)
    total = math.fsum(hours)
    if (any(not math.isfinite(x) or not HOUR_BOUNDS[0] <= x <= HOUR_BOUNDS[1] for x in hours)
            or total > TOTAL_HOURS or TOTAL_HOURS - total > NATIVE_SUM_TOLERANCE
            or sum(hours) > TOTAL_HOURS):
        raise ValueError("native fixed-budget closure outside the physical contract")
    return hours


class FixedBudgetCapacityActor(CapacityActor):
    """CapacityActor-shaped GCN with unclipped logits and a fixed-total map.

Initialization is inherited. The learner owns zeroing the final affine for
an exact [2, 2, 2, 2] start; this class performs no initialization forwards.
Noise-free projection is used for critic/actor gradients. Behavior exploration
supplies iid Gaussian sigma0.25 *logit* noise through project_noisy or
project_native(noise=...), never noise on the already-mapped physical hours.
"""

    format = FORMAT
    forward_output = "unclipped_logits"
    action_parameterization = "2_plus_tanh_logits_minus_site_mean_tanh"

    def __init__(self, config: CapacityModelConfig, *, adjacency=None):
        if (type(config) is not CapacityModelConfig or config.num_nodes != SITES
                or config.shared_hour_budget != TOTAL_HOURS
                or config.site_hour_caps != (4.0,) * SITES
                or config.raw_hour_offset != CENTER_HOURS):
            raise ValueError("four-site capacity config with caps4, budget8 and offset2 required")
        super().__init__(config, adjacency=adjacency)

    def metadata(self):
        """Required enclosing-checkpoint identity; not an extra tensor state key."""
        return dict(format=self.format, model_config=asdict(self.config),
            forward_output=self.forward_output, action_parameterization=self.action_parameterization,
            total_hours=TOTAL_HOURS, site_hour_bounds=list(HOUR_BOUNDS),
            full_capped_simplex=False, map_compute_dtype="float64", critic_output_dtype="float32",
            native_output_dtype="float64", native_closure="last_equals_8_minus_fsum_first_three",
            native_sum_tolerance=NATIVE_SUM_TOLERANCE, native_strict_upper_bound=TOTAL_HOURS,
            noise_location="logits_before_tanh", noise_addition_dtype="float64",
            prospective_noise_sigma=0.25, zero_final_affine_owned_by_caller=True,
            permutation_equivariant=True, common_logit_shift_invariant=False,
            legacy_raw_hours_state_semantics_compatible=False)

    def forward(self, features):
        validate_features(features, self.config)
        logits = self.head(self.encoder(features)).squeeze(-1)
        _validate_logits(logits)
        return logits

    @staticmethod
    def project(logits):
        return project_fixed_budget_logits(logits)

    @staticmethod
    def project_noisy(logits, noise):
        return project_fixed_budget_logits(logits, noise=noise)

    @staticmethod
    def project_native(logits, noise=None):
        return native_fixed_budget_hours(logits, noise=noise)
