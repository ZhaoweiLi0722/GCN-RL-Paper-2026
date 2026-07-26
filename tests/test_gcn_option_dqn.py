"""Tests for the graph residual-option Double-DQN policy."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.baselines.heuristics import (
    get_heuristic_class,
    heuristic_settings_for_policy,
)
from src.rl.agents import available_algorithms, get_agent_class
from src.rl.config import load_config
from src.rl.experiment import build_env
from src.rl.residual_options import (
    make_explicit_residual_option_specs,
    make_residual_option_specs,
    residual_option_actions_from_env,
    residual_option_actions_from_state,
    residual_option_labels_from_advantages,
    residual_option_labels_from_actions,
)


def _config(
    seed: int = 0,
    algorithm: str = "gcn_residual_mdl2_option_dqn_afd",
) -> dict:
    return {
        "algorithm": algorithm,
        "seed": seed,
        "env": load_config("experiments/configs/2_clinic_patient_condition.json"),
        "gcn_hidden_sizes": [16, 8],
        "hidden_sizes": [16],
        "batch_size": 2,
        "replay_capacity": 100,
        "residual_option": {
            "anchor_policy": "mdl2",
            "epsilons": [0.32, 0.64, 1.0],
            "candidate_groups": [
                "replenishment_uniform",
                "reagent_transfer",
                "combined_transfer",
                "reagent_replenishment",
                "combined_network",
            ],
            "candidate_signs": [-1.0, 1.0],
            "hidden_sizes": [16],
            "target_update_interval": 1,
            "imitation_regularization_weight": 0.1,
            "imitation_batch_size": 2,
        },
    }


class ResidualOptionTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from src.rl.networks import torch
        except Exception:  # pragma: no cover
            self.skipTest("torch not available")
        if torch is None:  # pragma: no cover
            self.skipTest("torch not available")
        self.config = _config()
        self.env = build_env(self.config, seed=0)
        self.specs = make_residual_option_specs(
            self.config["residual_option"]["epsilons"],
            self.config["residual_option"]["candidate_groups"],
            self.config["residual_option"]["candidate_signs"],
        )

    def test_option_set_has_fixed_anchor_plus_thirty_corrections(self) -> None:
        self.assertEqual(len(self.specs), 31)
        self.assertTrue(self.specs[0].is_anchor)

    def test_explicit_option_set_preserves_requested_order(self) -> None:
        specs = make_explicit_residual_option_specs(
            [
                {"group": "combined_transfer", "epsilon": 0.32, "sign": -1.0},
                {
                    "group": "reagent_replenishment",
                    "epsilon": 1.0,
                    "sign": 1.0,
                },
            ]
        )

        self.assertEqual(len(specs), 3)
        self.assertTrue(specs[0].is_anchor)
        self.assertEqual(specs[1].group, "combined_transfer")
        self.assertEqual(specs[2].epsilon, 1.0)

    def test_cached_actions_recover_their_option_labels(self) -> None:
        config = dict(self.config)
        config["env"] = dict(self.config["env"])
        config["env"]["transfer_lead_time"] = 2
        config["env"]["include_transfer_pipeline_state"] = True
        env = build_env(config, seed=4)
        env.demand = np.asarray([12.0, 2.0])
        env.demand_forecast = np.asarray([10.0, 3.0])
        env.specimens = np.asarray([18.0, 3.0])
        env.reagents = np.asarray([2.0, 24.0])
        env.bioreactors[:, 0] = np.asarray([1.0, 9.0])
        env.reagent_transfer_pipeline[-1] = np.asarray([7.0, 1.0])
        env.capacity_transfer_pipeline[-1] = np.asarray([0.0, 3.0])
        state = env.observation()
        anchor_cls = get_heuristic_class("mdl2")
        anchor = anchor_cls(
            state_dim=env.observation_size,
            action_dim=env.action_size,
            config={},
        )
        anchor_action = anchor.select_action(state, explore=False, env=env)
        candidates = residual_option_actions_from_env(
            anchor_action,
            env,
            self.specs,
        )
        state_candidates = residual_option_actions_from_state(
            state,
            config["env"],
            heuristic_settings_for_policy("mdl2", {}),
            self.specs,
        )
        np.testing.assert_allclose(candidates, state_candidates, atol=1e-6)
        unique_indices: list[int] = []
        unique_actions: list[np.ndarray] = []
        for index, candidate in enumerate(candidates):
            if any(np.allclose(candidate, existing) for existing in unique_actions):
                continue
            unique_indices.append(index)
            unique_actions.append(candidate)
            if len(unique_indices) == 4:
                break
        self.assertEqual(len(unique_indices), 4)
        selected = np.asarray(unique_indices, dtype=np.int64)

        labels = residual_option_labels_from_actions(
            np.repeat(state.reshape(1, -1), selected.size, axis=0),
            np.asarray(unique_actions),
            config["env"],
            heuristic_settings_for_policy("mdl2", {}),
            self.specs,
        )

        np.testing.assert_array_equal(labels, selected)

    def test_dense_advantages_produce_materiality_filtered_labels(self) -> None:
        advantages = np.asarray(
            [
                [0.0, 600_000.0, 2_000_000.0],
                [0.0, 400_000.0, -1.0],
            ],
            dtype=np.float32,
        )
        feasible = np.asarray(
            [
                [True, True, False],
                [True, True, True],
            ]
        )

        labels = residual_option_labels_from_advantages(
            advantages,
            feasible,
            min_advantage=500_000.0,
        )

        np.testing.assert_array_equal(labels, [1, 0])

    def test_agent_can_distill_an_option_subset_from_full_cache_metadata(self) -> None:
        config = _config()
        config["residual_option"] = dict(config["residual_option"])
        config["residual_option"]["explicit_options"] = [
            {"group": "combined_transfer", "epsilon": 0.32, "sign": -1.0},
            {"group": "reagent_transfer", "epsilon": 1.0, "sign": 1.0},
        ]
        agent = get_agent_class("gcn_residual_mdl2_option_dqn_afd")(
            self.env.observation_size,
            self.env.action_size,
            config,
        )
        full_specs = make_residual_option_specs(
            [0.32, 0.64, 1.0],
            config["residual_option"]["candidate_groups"],
            config["residual_option"]["candidate_signs"],
        )
        advantages = np.zeros((2, len(full_specs)), dtype=np.float32)
        feasible = np.ones_like(advantages, dtype=bool)
        selected_full_index = next(
            index
            for index, option in enumerate(full_specs)
            if option.group == "reagent_transfer"
            and option.epsilon == 1.0
            and option.sign == 1.0
        )
        advantages[0, selected_full_index] = 2_000_000.0
        demos = {
            "option_advantages": advantages,
            "option_feasible": feasible,
            "option_groups": np.asarray([option.group for option in full_specs]),
            "option_epsilons": np.asarray(
                [option.epsilon for option in full_specs],
                dtype=np.float32,
            ),
            "option_signs": np.asarray(
                [option.sign for option in full_specs],
                dtype=np.float32,
            ),
        }

        labels = agent.demonstration_option_labels(demos)

        self.assertNotEqual(int(labels[0]), 0)
        self.assertEqual(int(labels[1]), 0)

    def test_agent_pretrains_updates_and_round_trips(self) -> None:
        agent_cls = get_agent_class("gcn_residual_mdl2_option_dqn_afd")
        agent = agent_cls(
            self.env.observation_size,
            self.env.action_size,
            self.config,
        )
        state = self.env.reset(seed=8)
        self.assertEqual(agent._select_option_index(state, explore=False), 0)
        anchor = get_heuristic_class("mdl2")(
            state_dim=self.env.observation_size,
            action_dim=self.env.action_size,
            config={},
        )
        anchor_action = anchor.select_action(state, explore=False, env=self.env)
        candidates = residual_option_actions_from_env(
            anchor_action,
            self.env,
            self.specs,
        )
        states = np.repeat(state.reshape(1, -1), 4, axis=0)
        actions = np.asarray([candidates[index] for index in (0, 1, 10, 30)])
        summary = agent.fit_action_batch(
            states,
            actions,
            {
                "epochs": 2,
                "batch_size": 2,
                "seed": 9,
                "retain_for_regularization": True,
            },
        )
        self.assertEqual(summary["samples"], 4)
        self.assertTrue(np.isfinite(summary["final_loss"]))
        diagnostics = agent.option_diagnostics()
        self.assertEqual(diagnostics["total_selections"], 0)

        for _step in range(3):
            action = agent.select_action(state, explore=True, env=self.env)
            next_state, reward, done, _info = self.env.step(action)
            agent.observe(state, action, reward, next_state, done)
            state = next_state
        diagnostics = agent.option_diagnostics()
        self.assertEqual(diagnostics["total_selections"], 3)
        losses = agent.update()
        self.assertTrue(np.isfinite(losses["q_loss"]))

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "option_dqn.pt"
            agent.save(checkpoint)
            restored = agent_cls(
                self.env.observation_size,
                self.env.action_size,
                self.config,
            )
            restored.load_actor(checkpoint)
            restored_action = restored.select_action(state, explore=False, env=self.env)
        self.assertEqual(restored_action.shape, (self.env.action_size,))
        self.assertTrue(np.all(restored_action >= -1.0))
        self.assertTrue(np.all(restored_action <= 1.0))

    def test_anchor_relative_reward_is_zero_for_the_anchor_under_exact_crn(self) -> None:
        agent_cls = get_agent_class("gcn_residual_mdl2_option_dqn_afd")
        config = _config()
        config["residual_option"] = dict(config["residual_option"])
        config["residual_option"]["online_reward_mode"] = (
            "one_step_anchor_relative"
        )
        agent = agent_cls(
            self.env.observation_size,
            self.env.action_size,
            config,
        )
        state = self.env.reset(seed=21)
        action = agent.select_action(state, explore=False, env=self.env)
        context = agent.capture_training_reward_context(state, action, self.env)

        next_state, reward, done, info = self.env.step(action)
        relative_reward = agent.transform_training_reward(
            state,
            action,
            reward,
            next_state,
            done,
            info,
            context,
        )

        self.assertEqual(agent._pending_option_index, 0)
        self.assertAlmostEqual(relative_reward, 0.0, places=6)
        diagnostics = agent.option_diagnostics()
        self.assertEqual(diagnostics["anchor_relative_reward_count"], 1)
        self.assertAlmostEqual(
            diagnostics["anchor_relative_reward_mean_abs"],
            0.0,
            places=6,
        )

    def test_environment_reward_mode_preserves_delayed_credit_signal(self) -> None:
        agent_cls = get_agent_class("gcn_residual_mdl2_option_dqn_afd")
        agent = agent_cls(
            self.env.observation_size,
            self.env.action_size,
            self.config,
        )
        state = self.env.reset(seed=22)
        action = agent.select_action(state, explore=False, env=self.env)
        context = agent.capture_training_reward_context(state, action, self.env)
        next_state, reward, done, info = self.env.step(action)

        transformed = agent.transform_training_reward(
            state,
            action,
            reward,
            next_state,
            done,
            info,
            context,
        )

        self.assertEqual(context, {})
        self.assertEqual(transformed, reward)
        self.assertEqual(
            agent.option_diagnostics()["anchor_relative_reward_count"],
            0,
        )

    def test_agent_fits_dense_option_advantages(self) -> None:
        agent_cls = get_agent_class("gcn_residual_mdl2_option_dqn_afd")
        agent = agent_cls(
            self.env.observation_size,
            self.env.action_size,
            self.config,
        )
        state = self.env.reset(seed=11)
        anchor = get_heuristic_class("mdl2")(
            state_dim=self.env.observation_size,
            action_dim=self.env.action_size,
            config={},
        )
        anchor_action = anchor.select_action(state, explore=False, env=self.env)
        candidates = residual_option_actions_from_env(
            anchor_action,
            self.env,
            self.specs,
        )
        states = np.repeat(state.reshape(1, -1), 3, axis=0)
        actions = np.asarray([candidates[index] for index in (0, 1, 2)])
        advantages = np.zeros((3, len(self.specs)), dtype=np.float32)
        advantages[0, 1] = 2.0e9
        advantages[1, 2] = 1.5e9
        advantages[2, 0] = 0.0
        advantages[2, 1:] = -1.0e9
        demonstrations = {
            "option_advantages": advantages,
            "option_feasible": np.ones_like(advantages, dtype=bool),
            "option_groups": np.asarray(
                [option.group for option in self.specs],
            ),
            "option_epsilons": np.asarray(
                [option.epsilon for option in self.specs],
                dtype=np.float32,
            ),
            "option_signs": np.asarray(
                [option.sign for option in self.specs],
                dtype=np.float32,
            ),
        }

        summary = agent.fit_action_batch(
            states,
            actions,
            {
                "epochs": 2,
                "batch_size": 2,
                "seed": 12,
                "retain_for_regularization": True,
                "demonstrations": demonstrations,
            },
        )

        self.assertEqual(summary["target_mode"], "soft_advantage")
        self.assertEqual(summary["samples"], 3)
        self.assertTrue(np.isfinite(summary["final_loss"]))
        self.assertIsNotNone(agent.imitation_targets)
        self.assertIsNone(agent.imitation_labels)

    def test_teacher_best_gate_labels_only_the_selected_option(self) -> None:
        config = _config()
        config["residual_option"] = dict(config["residual_option"])
        config["residual_option"]["explicit_options"] = [
            {"group": "reagent_transfer", "epsilon": 0.32, "sign": 1.0},
            {"group": "combined_transfer", "epsilon": 0.32, "sign": -1.0},
        ]
        config["residual_option"]["correction_gate_target_mode"] = "teacher_best"
        agent_cls = get_agent_class("gcn_residual_mdl2_option_dqn_afd")
        agent = agent_cls(
            self.env.observation_size,
            self.env.action_size,
            config,
        )
        specs = make_explicit_residual_option_specs(
            config["residual_option"]["explicit_options"]
        )
        state = self.env.reset(seed=15)
        anchor = get_heuristic_class("mdl2")(
            state_dim=self.env.observation_size,
            action_dim=self.env.action_size,
            config={},
        )
        anchor_action = anchor.select_action(state, explore=False, env=self.env)
        candidates = residual_option_actions_from_env(
            anchor_action,
            self.env,
            specs,
        )
        states = np.repeat(state.reshape(1, -1), 2, axis=0)
        actions = np.asarray([candidates[1], candidates[0]])
        advantages = np.asarray(
            [
                [0.0, 2.0e9, 1.0e9],
                [0.0, -1.0e9, -2.0e9],
            ],
            dtype=np.float32,
        )
        demonstrations = {
            "option_advantages": advantages,
            "option_feasible": np.ones_like(advantages, dtype=bool),
            "option_groups": np.asarray(
                [option.group for option in specs],
            ),
            "option_epsilons": np.asarray(
                [option.epsilon for option in specs],
                dtype=np.float32,
            ),
            "option_signs": np.asarray(
                [option.sign for option in specs],
                dtype=np.float32,
            ),
        }

        agent.fit_action_batch(
            states,
            actions,
            {
                "epochs": 1,
                "batch_size": 2,
                "seed": 16,
                "retain_for_regularization": True,
                "demonstrations": demonstrations,
            },
        )

        gate_labels = agent.imitation_gate_labels.detach().cpu().numpy()
        np.testing.assert_array_equal(
            gate_labels,
            np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        )

    def test_gate_probability_can_select_the_correction_option(self) -> None:
        from src.rl.networks import torch

        config = _config()
        config["residual_option"] = dict(config["residual_option"])
        config["residual_option"]["explicit_options"] = [
            {"group": "reagent_transfer", "epsilon": 0.32, "sign": 1.0},
            {"group": "combined_transfer", "epsilon": 0.32, "sign": -1.0},
        ]
        config["residual_option"]["correction_selection_mode"] = (
            "gate_probability"
        )
        config["residual_option"]["correction_gate_threshold"] = 0.5
        agent_cls = get_agent_class("gcn_residual_mdl2_option_dqn_afd")
        agent = agent_cls(
            self.env.observation_size,
            self.env.action_size,
            config,
        )
        agent._q_and_gate_values = lambda _network, _states: (
            torch.tensor([[0.0, 2.0, 1.0]], device=agent.device),
            torch.tensor([[0.1, 3.0]], device=agent.device),
        )
        state = self.env.reset(seed=17)

        selected = agent._select_option_index(state, explore=False)

        self.assertEqual(selected, 2)

    def test_algorithm_is_registered(self) -> None:
        self.assertIn(
            "gcn_residual_mdl2_option_dqn_afd",
            available_algorithms(),
        )
        self.assertIn(
            "flat_residual_mdl2_option_dqn_afd",
            available_algorithms(),
        )

    def test_matched_flat_option_agent_uses_mlp_encoder(self) -> None:
        config = _config(algorithm="flat_residual_mdl2_option_dqn_afd")
        agent_cls = get_agent_class("flat_residual_mdl2_option_dqn_afd")
        agent = agent_cls(
            self.env.observation_size,
            self.env.action_size,
            config,
        )
        state = self.env.reset(seed=14)

        action = agent.select_action(state, explore=False, env=self.env)

        self.assertFalse(agent.use_graph_encoder)
        self.assertIsNone(agent.graph_spec)
        self.assertEqual(action.shape, (self.env.action_size,))


if __name__ == "__main__":
    unittest.main()
