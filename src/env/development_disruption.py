"""Opt-in synthetic disruption pilot; never used by frozen experiment paths."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np

from src.env.patient_capacity_planning import PatientConditionCapacityEnv


@dataclass(frozen=True)
class DisruptionProfile:
    change_step: int
    blocked_capacity_fraction: tuple[float, ...]
    shifted_lead_probabilities: tuple[float, ...]

    def validate(self, config) -> None:
        fractions = np.asarray(self.blocked_capacity_fraction, dtype=float)
        probs = np.asarray(self.shifted_lead_probabilities, dtype=float)
        if self.change_step < 0 or self.change_step >= config.episode_horizon:
            raise ValueError("change_step must be inside the episode")
        if fractions.shape != (config.num_facilities,) or not np.isfinite(fractions).all():
            raise ValueError("one finite capacity fraction is required per facility")
        if ((fractions < 0) | (fractions > 1)).any():
            raise ValueError("capacity fractions must be in [0, 1]")
        if not config.enable_stochastic_procurement:
            raise ValueError("pilot requires the paired stochastic procurement interface")
        if len(probs) != len(config.reagent_lead_time_probabilities):
            raise ValueError("lead distributions must have identical support width")
        if not np.isfinite(probs).all() or (probs < 0).any() or not np.isclose(probs.sum(), 1):
            raise ValueError("invalid shifted lead probabilities")
        if config.procurement_lead_override is not None or config.enable_overtime_control:
            raise ValueError("pilot does not combine lead overrides or overtime")


class DevelopmentDisruptionEnv(PatientConditionCapacityEnv):
    """Persistent loss of idle units, or delayed newly placed procurement.

    A capacity outage quarantines up to floor(fraction * initial fleet) units
    per affected site, as they become idle. Active treatments are never
    interrupted. Quarantined units cannot be transferred, remain physically
    accounted for and still incur holding cost. No recovery is modeled.
    """

    def __init__(self, config, profile: DisruptionProfile, seed=None):
        profile.validate(config.base)
        self.disruption_profile = profile
        self.blocked_idle = np.zeros(config.base.num_facilities)
        super().__init__(config, seed=seed)

    def reset(self, seed=None):
        self.blocked_idle = np.zeros(self.env_config.base.num_facilities)
        self.last_procurement_leads = np.zeros(self.env_config.base.num_facilities, dtype=int)
        super().reset(seed)
        self._quarantine_idle()
        return self.observation()

    def _quarantine_idle(self):
        if self.t < self.disruption_profile.change_step:
            return
        target = np.floor(
            self.initial_idle_bioreactors
            * np.asarray(self.disruption_profile.blocked_capacity_fraction)
        )
        newly_blocked = np.minimum(
            np.maximum(target - self.blocked_idle, 0), self.bioreactors[:, 0]
        )
        self.bioreactors[:, 0] -= newly_blocked
        self.blocked_idle += newly_blocked

    def _advance_clock(self):
        done = super()._advance_clock()
        self._quarantine_idle()
        return done

    def _draw_procurement_leads(self):
        probabilities = self.config.reagent_lead_time_probabilities
        if self.t >= self.disruption_profile.change_step:
            probabilities = self.disruption_profile.shifted_lead_probabilities
        # Match the parent method's draw count even for zero-quantity orders.
        self.last_procurement_leads = self.rng.choice(len(probabilities), size=self.config.num_facilities, p=probabilities)
        return self.last_procurement_leads.copy()

    def step(self, action):
        held = self.blocked_idle.copy()
        observation, reward, done, info = super().step(action)
        charge = float(held.sum() * self.config.costs.bioreactor_holding)
        for field in ("cost", "base_cost", "bioreactor_holding_cost"):
            info[field] = float(info[field]) + charge
        info["blocked_capacity_before_action"] = held
        info["blocked_capacity_after_step"] = self.blocked_idle.copy()
        info["blocked_capacity_holding_cost"] = charge
        info["procurement_leads"] = self.last_procurement_leads.copy()
        return observation, reward - charge, done, info

    def state_dict(self):
        state = super().state_dict()
        state["development_disruption"] = {
            "profile": asdict(self.disruption_profile),
            "blocked_idle": self.blocked_idle.copy(),
            "last_procurement_leads": self.last_procurement_leads.copy(),
        }
        return state

    def load_state_dict(self, state):
        payload = state.get("development_disruption", {})
        if payload.get("profile") != asdict(self.disruption_profile):
            raise ValueError("disruption profile mismatch")
        blocked = np.asarray(payload.get("blocked_idle"), dtype=float)
        target = np.floor(self.initial_idle_bioreactors * np.asarray(self.disruption_profile.blocked_capacity_fraction))
        if blocked.shape != target.shape or not np.isfinite(blocked).all() or ((blocked < 0) | (blocked > target)).any():
            raise ValueError("invalid blocked capacity state")
        super().load_state_dict(copy.deepcopy(state))
        self.blocked_idle = blocked.copy()
        self.last_procurement_leads = np.asarray(payload["last_procurement_leads"], dtype=int).copy()
