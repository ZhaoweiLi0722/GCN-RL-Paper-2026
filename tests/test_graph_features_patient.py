"""Patient-aware graph plumbing (Phase 6, group 2).

The select-action path builds node features from ``env.graph_observation()``;
the replay path rebuilds them from the flat ``env.observation()`` vector. They
must agree, or the GCN agents learn on features they never act on.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.models.graph_features import build_graph_spec
from src.rl.config import load_config
from src.rl.experiment import build_env

DEV_CONFIG = "experiments/configs/2_clinic_patient_condition.json"       # no hub
HUB_CONFIG = "experiments/configs/20_clinic_patient_condition.json"      # central hub


def _spec_and_env(path: str):
    env = build_env({"env": load_config(path)}, seed=0)
    spec = build_graph_spec({"env": load_config(path)}, state_dim=env.observation_size)
    return spec, env


class PatientGraphSpecTests(unittest.TestCase):
    def test_spec_dims_match_patient_env(self) -> None:
        for path in (DEV_CONFIG, HUB_CONFIG):
            with self.subTest(path=path):
                spec, env = _spec_and_env(path)
                # Summary width = 3 scalars + (len(edges)+1) histogram buckets.
                self.assertEqual(spec.patient_summary_width, env.summary_width)
                node_width = env.graph_observation()["node_features"].shape[1]
                self.assertEqual(spec.node_feature_dim, node_width)

    def test_expected_state_dim_accepts_patient_observation(self) -> None:
        # build_graph_spec used to hard-raise on the appended summary block.
        for path in (DEV_CONFIG, HUB_CONFIG):
            with self.subTest(path=path):
                _spec, env = _spec_and_env(path)  # would raise on mismatch
                self.assertEqual(
                    env.observation_size,
                    env.base_observation_size + env.config.num_facilities * env.summary_width,
                )


class PatientNodeFeatureEquivalenceTests(unittest.TestCase):
    def test_select_path_equals_replay_path(self) -> None:
        try:
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")
        from src.models.graph_features import flat_state_to_node_features

        for path in (DEV_CONFIG, HUB_CONFIG):
            with self.subTest(path=path):
                spec, env = _spec_and_env(path)
                # Advance a few steps so queues are non-trivial (histogram populated).
                state = env.reset(seed=3)
                rng = np.random.default_rng(3)
                for _ in range(4):
                    state, _r, done, _info = env.step(rng.uniform(-1.0, 1.0, env.action_size))
                    if done:
                        break
                flat = torch.as_tensor(env.observation(), dtype=torch.float32)
                rebuilt = flat_state_to_node_features(flat, spec).numpy()[0]
                expected = env.graph_observation()["node_features"]
                np.testing.assert_allclose(rebuilt, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
