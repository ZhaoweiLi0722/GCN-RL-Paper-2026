from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
import unittest

import numpy as np

from src.env.capacity_planning import CapacityPlanningEnv, make_20_clinic_config
from src.rl.residual_options import residual_option_actions_from_env
from src.rl.structured_exploration import StructuredSpecimenExplorer

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from src.baselines.flat_ddpg import FlatDDPGAgent
    from src.models.gcn_ddpg import GCNDDPGAgent


OPTIONS = (
    {"group": "specimen_transfer", "epsilon": 0.05, "sign": -1.0},
    {"group": "specimen_transfer", "epsilon": 0.05, "sign": 1.0},
    {"group": "specimen_transfer", "epsilon": 0.10, "sign": -1.0},
    {"group": "specimen_transfer", "epsilon": 0.10, "sign": 1.0},
)


class _OptionEnv:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_facilities=3)
        self.demand = np.asarray([8.0, 3.0, 1.0])
        self.demand_forecast = np.asarray([7.0, 4.0, 1.0])
        self.specimens = np.asarray([2.0, 7.0, 11.0])
        self.reagents = np.asarray([15.0, 12.0, 8.0])
        self.bioreactors = np.asarray(
            [[5.0, 0.0], [4.0, 0.0], [3.0, 0.0]]
        )
        self.at_risk_counts = np.asarray([1.0, 0.0, 2.0])
        self.near_expiry_counts = np.asarray([0.0, 1.0, 0.0])

    def _pending_transfer_arrivals(self):
        zeros = np.zeros(3, dtype=float)
        return zeros, zeros, zeros


def _settings(*, enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "selection_probability": 1.0,
        "selection_mode": "uniform",
        "apply_after_correction_gate": True,
        "seed_offset": 91,
        "options": list(OPTIONS),
    }


class StructuredSpecimenExplorerTests(unittest.TestCase):
    def test_only_specimen_slice_is_replaced_by_legal_options(self) -> None:
        env = _OptionEnv()
        explorer = StructuredSpecimenExplorer(
            action_dim=12,
            num_facilities=3,
            seed=7,
            settings=_settings(),
        )
        policy = np.linspace(-0.4, 0.4, 12, dtype=np.float32)
        anchor = np.zeros(12, dtype=np.float32)
        legal = residual_option_actions_from_env(
            anchor,
            env,
            explorer.option_specs,
        )
        legal_specimen = [action[:3] for action in legal]

        for _ in range(40):
            behavior = explorer.apply(policy, anchor, env=env)
            np.testing.assert_array_equal(behavior[3:], policy[3:])
            self.assertTrue(
                any(
                    np.array_equal(behavior[:3], candidate)
                    for candidate in legal_specimen
                )
            )
            explorer.record_projected_action(policy, behavior)

        summary = explorer.summary()
        self.assertEqual(summary["total_decisions"], 40)
        self.assertEqual(summary["total_selections"], 40)
        self.assertGreater(summary["total_correction_selections"], 0)
        self.assertGreater(summary["total_behaviorally_distinct"], 0)
        self.assertEqual(
            set(summary["total_option_counts"]),
            set(explorer.option_labels),
        )

    def test_state_round_trip_restores_exact_next_choices(self) -> None:
        env = _OptionEnv()
        first = StructuredSpecimenExplorer(
            action_dim=12,
            num_facilities=3,
            seed=11,
            settings=_settings(),
        )
        policy = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
        anchor = np.zeros(12, dtype=np.float32)
        for _ in range(7):
            behavior = first.apply(policy, anchor, env=env)
            first.record_projected_action(policy, behavior)
        state = first.state_dict()

        expected = []
        for _ in range(10):
            behavior = first.apply(policy, anchor, env=env)
            first.record_projected_action(policy, behavior)
            expected.append(
                (
                    behavior.copy(),
                    dict(first.last_decision),
                )
            )

        restored = StructuredSpecimenExplorer(
            action_dim=12,
            num_facilities=3,
            seed=999,
            settings=_settings(),
        )
        restored.load_state_dict(state)
        for expected_action, expected_decision in expected:
            behavior = restored.apply(policy, anchor, env=env)
            restored.record_projected_action(policy, behavior)
            np.testing.assert_array_equal(behavior, expected_action)
            self.assertEqual(restored.last_decision, expected_decision)

    def test_disabled_explorer_preserves_policy_without_consuming_options(self) -> None:
        env = _OptionEnv()
        explorer = StructuredSpecimenExplorer(
            action_dim=12,
            num_facilities=3,
            seed=3,
            settings=_settings(enabled=False),
        )
        policy = np.linspace(-0.5, 0.5, 12, dtype=np.float32)
        behavior = explorer.apply(
            policy,
            np.zeros(12, dtype=np.float32),
            env=env,
        )
        explorer.record_projected_action(policy, behavior)

        np.testing.assert_array_equal(behavior, policy)
        summary = explorer.summary()
        self.assertEqual(summary["total_decisions"], 1)
        self.assertEqual(summary["total_selections"], 0)
        self.assertEqual(sum(summary["total_option_counts"].values()), 0)

    def test_enabled_explorer_requires_live_environment(self) -> None:
        explorer = StructuredSpecimenExplorer(
            action_dim=12,
            num_facilities=3,
            seed=3,
            settings=_settings(),
        )
        with self.assertRaisesRegex(ValueError, "live environment"):
            explorer.apply(
                np.zeros(12, dtype=np.float32),
                np.zeros(12, dtype=np.float32),
                env=None,
            )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DDPGStructuredExplorationIntegrationTests(unittest.TestCase):
    def test_gcn_and_flat_agents_share_specimen_only_behavior_contract(self) -> None:
        for agent_class in (GCNDDPGAgent, FlatDDPGAgent):
            with self.subTest(agent=agent_class.__name__):
                env = CapacityPlanningEnv(
                    make_20_clinic_config(episode_horizon=2),
                    seed=19,
                )
                config = self._agent_config(env)
                agent = agent_class(
                    env.observation_size,
                    env.action_size,
                    config,
                )
                env.reset(seed=19)
                env.specimens = np.linspace(
                    0.0,
                    95.0,
                    env.config.num_facilities,
                )
                state = env.observation()
                agent.reset()
                deterministic = agent.select_action(
                    state,
                    explore=False,
                    env=env,
                )
                anchor = agent._base_action_from_state_np(state)
                legal = residual_option_actions_from_env(
                    anchor,
                    env,
                    agent.structured_specimen_explorer.option_specs,
                )
                legal_specimen = [action[:20] for action in legal]

                for _ in range(30):
                    action = agent.select_action(
                        state,
                        explore=True,
                        env=env,
                    )
                    np.testing.assert_allclose(
                        action[20:],
                        deterministic[20:],
                        atol=1e-7,
                    )
                    self.assertTrue(
                        any(
                            np.allclose(action[:20], candidate, atol=1e-7)
                            for candidate in legal_specimen
                        )
                    )

                summary = agent.structured_exploration_summary()
                self.assertEqual(summary["total_decisions"], 30)
                self.assertEqual(summary["total_selections"], 30)
                self.assertGreater(
                    summary["total_behaviorally_distinct"],
                    0,
                )

    @staticmethod
    def _agent_config(env: CapacityPlanningEnv) -> dict:
        return {
            "seed": 19,
            "device": "cpu",
            "batch_size": 2,
            "gcn_hidden_sizes": [8],
            "actor_hidden_sizes": [8],
            "critic_hidden_sizes": [8],
            "hidden_sizes": [8],
            "actor_readout_mode": "network_residual",
            "exploration_noise": {"theta": 0.15, "sigma": 0.0},
            "env": asdict(env.config),
            "residual_action": {
                "enabled": True,
                "base_policy": "mdl2",
                "zero_init_actor": True,
                "scale": 0.1,
                "group_scales": {
                    "specimen_transfer": 0.1,
                    "reagent_transfer": 0.0,
                    "capacity_transfer": 0.0,
                    "replenishment": 0.0,
                },
                "structured_exploration": _settings(),
            },
        }


if __name__ == "__main__":
    unittest.main()
