"""No scientific seeds: public information and checkpoint contracts."""

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from src.env.capacity_planning import CapacityPlanningConfig, CapacityPlanningEnv
from src.env.public_demand_information import PublicDemandForecaster


class PublicForecastTest(unittest.TestCase):
    def make(self, adaptive=True):
        return PublicDemandForecaster((2.0, 3.0), window=2, horizon=2, adaptive=adaptive)

    def observe(self, model, epoch, arrivals, exposure=(1.0, 1.0)):
        model.observe(epoch=epoch, arrivals=arrivals, announced_exposure=exposure)

    def issue(self, model, epoch):
        return model.issue(epoch=epoch, future_announced_exposure=((1.0, 1.0), (2.0, 1.0)))

    def test_issue_requires_observation_at_that_epoch(self):
        model = self.make()
        with self.assertRaises(ValueError):
            self.issue(model, 0)
        self.observe(model, 0, (4, 2))
        with self.assertRaises(ValueError):
            self.issue(model, 1)

    def test_rolling_estimate_and_announced_exposure(self):
        model = self.make()
        self.observe(model, 0, (8, 2), (2, 1))
        self.observe(model, 1, (2, 4))
        self.assertEqual(self.issue(model, 1).total, (9, 6))
        self.observe(model, 2, (6, 8))
        self.assertEqual(self.issue(model, 2).total, (12, 12))

    def test_frozen_and_adaptive_share_history_not_estimate(self):
        frozen, adaptive = self.make(False), self.make(True)
        for model in (frozen, adaptive):
            self.observe(model, 0, (10, 10))
        self.assertEqual(frozen.state_dict()["history"], adaptive.state_dict()["history"])
        self.assertEqual(self.issue(frozen, 0).total, (6, 6))
        self.assertEqual(self.issue(adaptive, 0).total, (30, 20))

    def test_issued_forecast_does_not_change_after_new_outcome(self):
        model = self.make()
        self.observe(model, 0, (1, 2))
        issued = self.issue(model, 0)
        self.observe(model, 1, (30, 40))
        self.assertEqual(issued.total, (3, 4))
        self.assertNotEqual(issued.total, self.issue(model, 1).total)

    def test_observations_cannot_skip_repeat_or_go_back(self):
        model = self.make()
        with self.assertRaises(ValueError):
            self.observe(model, 1, (1, 2))
        self.observe(model, 0, (1, 2))
        for epoch in (0, -1, 2, True):
            with self.assertRaises(ValueError):
                self.observe(model, epoch, (1, 2))

    def test_nonfinite_negative_and_invalid_exposure_rejected(self):
        for arrivals, exposure in (((float("nan"), 1), (1, 1)), ((-1, 1), (1, 1)), ((1, 1), (0, 1)), ((1,), (1, 1))):
            with self.assertRaises(ValueError):
                self.observe(self.make(), 0, arrivals, exposure)

    def test_checkpoint_roundtrip_after_truncation_and_continuation(self):
        model, restored = self.make(), self.make()
        for epoch in range(5):
            self.observe(model, epoch, (epoch, 2 * epoch))
        restored.load_state_dict(json.loads(json.dumps(model.state_dict())))
        self.assertEqual(self.issue(model, 4), self.issue(restored, 4))
        for item in (model, restored):
            self.observe(item, 5, (7, 8))
        self.assertEqual(model.state_dict(), restored.state_dict())
        self.assertEqual(self.issue(model, 5), self.issue(restored, 5))

    def test_checkpoint_rejects_mode_future_epoch_and_hidden_fields(self):
        model = self.make()
        self.observe(model, 0, (1, 2))
        original = model.state_dict()
        for key, value in (("adaptive", False), ("last_epoch", 1), ("future_regime", [9, 9]), ("version", True)):
            bad = copy.deepcopy(original)
            bad[key] = value
            with self.assertRaises(ValueError):
                model.load_state_dict(bad)
            self.assertEqual(original, model.state_dict())

    def test_input_and_checkpoint_buffers_are_not_aliased(self):
        model = self.make()
        arrivals = [1, 2]
        self.observe(model, 0, arrivals)
        arrivals[0] = 999
        state = model.state_dict()
        state["history"][0]["arrivals"][0] = 999
        self.assertEqual(self.issue(model, 0).total, (3, 4))

    def test_finite_per_step_values_cannot_overflow_horizon_total(self):
        model = PublicDemandForecaster((1e308,), window=1, horizon=2, adaptive=False)
        model.observe(epoch=0, arrivals=(0,), announced_exposure=(1,))
        with self.assertRaises(ValueError):
            model.issue(epoch=0, future_announced_exposure=((1,), (1,)))

    def test_current_legacy_forecast_and_new_public_contract_differ(self):
        values = json.loads(Path("experiments/configs/online_adaptation_information_mechanics.json").read_text())["env"]
        values.update(demand_rates=(2.0, 2.0), include_demand_forecast_state=True,
                      demand_forecast_horizon=2, demand_forecast_error=0.0,
                      enable_scheduled_referral_waves=True, scheduled_referral_clusters=((0,), (1,)),
                      demand_regime_change_step=2)
        for source in ("effective_rate", "prior_estimate"):
            with self.subTest(source=source):
                a = CapacityPlanningEnv(CapacityPlanningConfig(**dict(values, demand_forecast_source=source, demand_regime_final_multipliers=(4, 1))), seed=123)
                b = CapacityPlanningEnv(CapacityPlanningConfig(**dict(values, demand_forecast_source=source, demand_regime_final_multipliers=(1, 4))), seed=123)
                np.testing.assert_array_equal(a.demand, b.demand)
                self.assertFalse(np.array_equal(a.demand_forecast, b.demand_forecast))
                public = []
                for env in (a, b):
                    model = self.make()
                    self.observe(model, env.t, env.demand)
                    public.append(self.issue(model, env.t))
                self.assertEqual(*public)


if __name__ == "__main__":
    unittest.main()
