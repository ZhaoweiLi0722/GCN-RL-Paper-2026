import unittest

import numpy as np

from evaluation.audit_formal_replay_contract import METRICS, loss_profile, reference_windows, teacher_profile


class FormalReplayContractTests(unittest.TestCase):
    def test_windows_stop_at_terminals(self):
        demos = {"transition_dones": np.array([False, True, False, False, False, True])}
        self.assertEqual(reference_windows(demos), [(0, 2), (1, 2), (2, 6), (3, 6), (4, 6), (5, 6)])

    def test_loader_does_not_implicitly_flush_nonterminal_tail(self):
        demos = {"transition_dones": np.zeros(5, dtype=bool)}
        self.assertEqual(reference_windows(demos), [(0, 4), (1, 5)])

    def teacher(self):
        return {
            "states": np.array([[0], [1], [2]], dtype=np.float32),
            "transition_states": np.array([[0], [1], [2]], dtype=np.float32),
            "transition_next_states": np.array([[1], [9], [3]], dtype=np.float32),
            "transition_rewards": np.array([-3, -2, -1], dtype=np.float32),
            "transition_dones": np.array([False, False, True]),
            "option_groups": np.array(["anchor", "specimen_transfer", "reagent_transfer"]),
            "option_epsilons": np.array([0, .1, .1]), "option_signs": np.array([0, 1, 1]),
            "option_advantages": np.array([[0, 1, 2], [0, 3, 9], [0, -1, -2]], dtype=np.float32),
            "option_feasible": np.array([[True, True, True], [True, True, False], [True, True, True]]),
            "improved_mask": np.array([True, True, False]),
        }

    def test_teacher_support_respects_feasibility(self):
        p = teacher_profile(self.teacher())
        self.assertEqual(p["best_option_outside_anchor_or_specimen"], 1)
        self.assertEqual(p["best_option_groups"], {"anchor": 1, "specimen_transfer": 1, "reagent_transfer": 1})
        self.assertEqual(p["nonterminal_state_discontinuities"], 1)
        self.assertFalse(p["label_horizon_serialized_in_cache"])

    def test_terminal_adjacency_is_not_marked_as_broken(self):
        d = self.teacher()
        d["transition_dones"][1] = True
        self.assertEqual(teacher_profile(d)["nonterminal_state_discontinuities"], 0)

    def test_nonfinite_and_infeasible_teacher_rejected(self):
        d = self.teacher()
        d["transition_rewards"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            teacher_profile(d)
        d = self.teacher()
        d["option_feasible"][0] = False
        with self.assertRaisesRegex(ValueError, "feasibility"):
            teacher_profile(d)

    def rows(self):
        return [{"episode": str(i), "online_rl_updates": "52", "online_rl_actor_updated_mean": "0.5",
                 **{f"online_rl_{k}_mean": "0.01" for k in METRICS}} for i in range(100)]

    def test_update_counts_and_loss_aggregation(self):
        p = loss_profile(self.rows())
        self.assertEqual(p["online_critic_updates"], 5200)
        self.assertEqual(p["online_actor_updates"], 2600)
        self.assertAlmostEqual(p["metrics"]["actor_loss"]["episode_mean_average"], .01)

    def test_missing_duplicate_and_nonfinite_episodes_rejected(self):
        for mutate in (lambda r: r.pop(), lambda r: r[0].update(episode="1"),
                       lambda r: r[0].update(online_rl_actor_loss_mean="nan"),
                       lambda r: r[0].update(online_rl_updates="51")):
            rows = self.rows()
            mutate(rows)
            with self.assertRaises(ValueError):
                loss_profile(rows)


if __name__ == "__main__":
    unittest.main()
