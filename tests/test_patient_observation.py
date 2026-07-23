"""Tests for the patient observation & graph summary (Phase 4, task group 5)."""

from __future__ import annotations

import unittest

import numpy as np

from src.env.capacity_planning import CapacityPlanningConfig
from src.env.patient_capacity_planning import (
    PatientConditionCapacityEnv,
    PatientEnvConfig,
)


def _small_base(**overrides) -> CapacityPlanningConfig:
    params = dict(
        num_facilities=2,
        production_lead_time=2,
        episode_horizon=20,
        demand_rates=(3.0, 3.0),
        initial_reagents=(100.0, 100.0),
        initial_idle_bioreactors=(10.0, 10.0),
        max_reagent_replenishment=(100.0, 100.0),
        action_mode="facility_net",
        include_supplier_state=True,
        supplier_disruption_rate=0.0,
    )
    params.update(overrides)
    return CapacityPlanningConfig(**params)


def _env(**cfg) -> PatientConditionCapacityEnv:
    return PatientConditionCapacityEnv(PatientEnvConfig(base=_small_base(), **cfg), seed=0)


class PatientObservationTests(unittest.TestCase):
    def test_observation_size_accounts_for_summary(self) -> None:
        env = _env()  # default 4 buckets -> summary_width = 3 + 4 = 7
        self.assertEqual(env.summary_width, 7)
        expected = env.base_observation_size + env.config.num_facilities * env.summary_width
        obs = env.reset(seed=0)
        self.assertEqual(env.observation_size, expected)
        self.assertEqual(obs.shape, (expected,))

    def test_observation_width_is_fixed_across_steps(self) -> None:
        env = _env()
        env.reset(seed=1)
        widths = set()
        for _ in range(10):
            obs, *_ = env.step(env.noop_action())
            widths.add(obs.shape[0])
        self.assertEqual(widths, {env.observation_size})

    def test_bucket_count_is_config_driven(self) -> None:
        env = _env(survival_bucket_edges=(0.80, 0.85, 0.90, 0.95, 0.98))  # 6 buckets
        self.assertEqual(env.summary_width, 3 + 6)
        obs = env.reset(seed=0)
        self.assertEqual(obs.shape[0], env.observation_size)

    def test_summary_histogram_sums_to_waiting_count(self) -> None:
        env = _env()
        env.reset(seed=2)
        for _ in range(6):
            env.step(env.noop_action())
        summary = env._patient_summary()
        for i in range(env.config.num_facilities):
            waiting = summary[i, 0]
            hist_sum = summary[i, 3:].sum()
            self.assertAlmostEqual(waiting, hist_sum)
            self.assertAlmostEqual(waiting, float(len(env.patient_queues[i])))

    def test_mean_survival_matches_queue(self) -> None:
        env = _env()
        env.reset(seed=3)
        for _ in range(5):
            env.step(env.noop_action())
        summary = env._patient_summary()
        for i, queue in enumerate(env.patient_queues):
            if queue:
                expected = float(np.mean([p.survival for p in queue]))
                self.assertAlmostEqual(summary[i, 1], expected, places=6)
            else:
                self.assertEqual(summary[i, 1], 0.0)

    def test_empty_queue_summary_is_zero(self) -> None:
        env = _env()
        env.reset(seed=0)  # queues start empty (initial_specimens = 0)
        summary = env._patient_summary()
        np.testing.assert_array_equal(summary, np.zeros_like(summary))

    def test_graph_node_features_include_summary(self) -> None:
        env = _env()
        env.reset(seed=0)
        base_dim = env.graph_observation()["node_features"].shape[1] - env.summary_width
        for _ in range(5):
            env.step(env.noop_action())
        graph = env.graph_observation()
        nf = graph["node_features"]
        self.assertEqual(nf.shape[1], base_dim + env.summary_width)
        # Central capacity hub (last row) carries no patients -> summary part is 0.
        if env.config.include_central_capacity_hub:
            np.testing.assert_array_equal(nf[-1, base_dim:], np.zeros(env.summary_width))

    def test_rejects_non_monotonic_edges(self) -> None:
        with self.assertRaises(ValueError):
            PatientConditionCapacityEnv(
                PatientEnvConfig(base=_small_base(), survival_bucket_edges=(0.9, 0.85)),
                seed=0,
            )


if __name__ == "__main__":
    unittest.main()
