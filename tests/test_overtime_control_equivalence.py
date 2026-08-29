"""Flag-off equivalence for continuous overtime control (spec 2026-08-29).

Invariant 1 of specs/2026-08-29-continuous-overtime-control/requirements.md:
with the overtime flags disabled the environment must behave bit-identically
to the pre-overtime environment. Because the flag-off code path is the
unchanged legacy path, the strongest executable check is the paired one: a
flag-ON environment driven with a permanently zero overtime block must
reproduce the flag-OFF environment exactly (rewards, shared info entries, and
state arrays), while flag-OFF must expose the legacy action/observation sizes
and no overtime artifacts.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from src.baselines.heuristics import MeanDemandLookahead2Policy
from src.env.patient_capacity_planning import (
    PatientConditionCapacityEnv,
    PatientEnvConfig,
)

_OVERTIME_INFO_KEYS = {
    "overtime_fraction",
    "overtime_surge",
    "overtime_production",
    "overtime_outstanding",
    "overtime_cost",
}

_SEED = 20260829


def _make_envs() -> tuple[PatientConditionCapacityEnv, PatientConditionCapacityEnv]:
    config_off = PatientEnvConfig()
    config_on = dataclasses.replace(
        config_off,
        base=dataclasses.replace(config_off.base, enable_overtime_control=True),
    )
    env_off = PatientConditionCapacityEnv(config_off, seed=_SEED)
    env_on = PatientConditionCapacityEnv(config_on, seed=_SEED)
    env_off.reset(seed=_SEED)
    env_on.reset(seed=_SEED)
    return env_off, env_on


def _zero_overtime(action_4n: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate(
        [np.asarray(action_4n, dtype=np.float32), np.full(n, -1.0, dtype=np.float32)]
    )


class OvertimeEquivalenceTest(unittest.TestCase):
    def _assert_step_equal(self, out_off, out_on) -> None:
        obs_off, reward_off, done_off, info_off = out_off
        obs_on, reward_on, done_on, info_on = out_on
        del obs_off, obs_on  # widths differ by design; state arrays are compared
        self.assertEqual(reward_off, reward_on)
        self.assertEqual(done_off, done_on)
        self.assertFalse(_OVERTIME_INFO_KEYS & set(info_off))
        self.assertTrue(_OVERTIME_INFO_KEYS <= set(info_on))
        for key, value in info_off.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(
                    value, info_on[key], err_msg=f"info[{key!r}] diverged"
                )
            else:
                self.assertEqual(value, info_on[key], f"info[{key!r}] diverged")

    def _assert_state_equal(self, env_off, env_on) -> None:
        for name in ("specimens", "reagents", "bioreactors", "demand", "supplier_available"):
            np.testing.assert_array_equal(
                np.asarray(getattr(env_off, name)),
                np.asarray(getattr(env_on, name)),
                err_msg=f"state {name} diverged",
            )
        self.assertEqual(
            env_off.rng.bit_generator.state, env_on.rng.bit_generator.state
        )
        self.assertEqual(
            sorted(env_off.patient_registry), sorted(env_on.patient_registry)
        )

    def _run_episode(self, action_fn) -> None:
        env_off, env_on = _make_envs()
        n = env_off.config.num_facilities
        done = False
        while not done:
            action_4n = action_fn(env_off)
            out_off = env_off.step(np.asarray(action_4n, dtype=np.float32))
            out_on = env_on.step(_zero_overtime(action_4n, n))
            self._assert_step_equal(out_off, out_on)
            done = out_off[2]
        self._assert_state_equal(env_off, env_on)

    def test_noop_episode_equivalent(self) -> None:
        self._run_episode(lambda env: env.noop_action())

    def test_random_episode_equivalent(self) -> None:
        rng = np.random.default_rng(_SEED)
        size = 4 * PatientEnvConfig().base.num_facilities
        self._run_episode(lambda env: rng.uniform(-1.0, 1.0, size=size))

    def test_mdl2_episode_equivalent(self) -> None:
        policy = MeanDemandLookahead2Policy()
        # The policy pads the overtime block itself on the flag-on env; here the
        # flag-off action is reused so both arms see identical base blocks.
        self._run_episode(
            lambda env: policy.select_action(env.observation(), env=env)
        )

    def test_flag_off_surface_is_legacy(self) -> None:
        env_off, env_on = _make_envs()
        n = env_off.config.num_facilities
        self.assertEqual(env_off.action_size, 4 * n)
        self.assertEqual(env_on.action_size, 5 * n)
        # +3 overtime features per facility, appended inside the base block.
        self.assertEqual(
            env_on.features_per_facility, env_off.features_per_facility + 3
        )
        self.assertNotIn("enable_overtime_control", env_off.state_dict())
        self.assertTrue(env_on.state_dict()["enable_overtime_control"])


if __name__ == "__main__":
    unittest.main()
