"""Patient-condition dynamics for the cell-therapy capacity-planning network.

Individual patients wait for autologous manufacturing while their health
deteriorates. This module ports the SimPAC mechanism in an MVP-lean, CPU-fast
form: a per-patient survival scalar decaying between two exponential
survival-curve bounds (a healthy upper bound and a frail lower bound), with a
per-patient health index interpolating between them and a stochastic
deterioration shock that accelerates decay. A patient whose survival falls below
an eligibility threshold is lost (ineligible for treatment).

Parameters are literature-inspired and fully config-driven; the concrete
magnitudes are tuned in the Phase 7 pilot (see the Phase 4 feature spec).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class PatientStatus(enum.Enum):
    """Lifecycle status of a patient in the network."""

    WAITING = "waiting"
    IN_PRODUCTION = "in_production"
    DELIVERED = "delivered"
    LOST = "lost"


@dataclass
class PatientState:
    """One patient's condition state.

    ``survival`` is recomputed from ``age`` by :class:`PatientConditionModel`;
    ``health_index`` (H in [0, 1], higher = healthier) and
    ``deterioration_epoch`` (the age at which decay accelerates) are drawn once at
    enrollment and held fixed.
    """

    health_index: float
    deterioration_epoch: float
    enrollment_epoch: int
    age: int = 0
    survival: float = 1.0
    status: PatientStatus = PatientStatus.WAITING


@dataclass(frozen=True)
class PatientConditionConfig:
    """Config for the patient survival model. All rates are per epoch (week)."""

    healthy_decay_rate: float = 0.01
    frail_decay_rate: float = 0.06
    weibull_shape: float = 1.5
    weibull_scale: float = 8.0
    post_shock_multiplier: float = 2.0
    eligibility_threshold: float = 0.75


class PatientConditionModel:
    """Draws and advances per-patient survival trajectories."""

    def __init__(self, config: PatientConditionConfig | None = None):
        self.config = config or PatientConditionConfig()
        if self.config.frail_decay_rate < self.config.healthy_decay_rate:
            raise ValueError("frail_decay_rate must be >= healthy_decay_rate")
        if not 0.0 <= self.config.eligibility_threshold <= 1.0:
            raise ValueError("eligibility_threshold must be in [0, 1]")

    def enroll(self, rng: np.random.Generator, epoch: int) -> PatientState:
        """Create a new waiting patient with a drawn health index and shock time."""

        health_index = float(rng.uniform(0.0, 1.0))
        deterioration_epoch = float(
            self.config.weibull_scale * rng.weibull(self.config.weibull_shape)
        )
        patient = PatientState(
            health_index=health_index,
            deterioration_epoch=deterioration_epoch,
            enrollment_epoch=int(epoch),
        )
        patient.survival = self.survival_at(0, health_index, deterioration_epoch)
        return patient

    def survival_at(self, age: float, health_index: float, deterioration_epoch: float) -> float:
        """Survival in [0, 1] for a given waiting age.

        Decay uses an effective time that doubles (``post_shock_multiplier``)
        past the deterioration epoch, so the frail/healthy exponential bounds
        drop faster after the shock.
        """

        if age <= deterioration_epoch:
            effective_time = age
        else:
            effective_time = deterioration_epoch + self.config.post_shock_multiplier * (
                age - deterioration_epoch
            )
        upper = float(np.exp(-self.config.healthy_decay_rate * effective_time))
        lower = float(np.exp(-self.config.frail_decay_rate * effective_time))
        survival = lower + health_index * (upper - lower)
        return float(np.clip(survival, 0.0, 1.0))

    def advance(self, patient: PatientState, epochs: int = 1) -> PatientState:
        """Age a waiting patient by ``epochs`` and update its survival in place."""

        patient.age += int(epochs)
        patient.survival = self.survival_at(
            patient.age, patient.health_index, patient.deterioration_epoch
        )
        return patient

    def is_eligible(self, patient: PatientState) -> bool:
        """Whether the patient still meets the survival eligibility threshold."""

        return patient.survival >= self.config.eligibility_threshold
