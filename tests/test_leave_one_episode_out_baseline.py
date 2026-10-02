"""Artificial scalar receipts only; no research data, tensors or fitting."""

import unittest

from src.rl.leave_one_episode_out_baseline import leave_one_episode_out_targets


class TimeBaselineTest(unittest.TestCase):
    def build(self, rows, **overrides):
        count = len(rows)
        args = dict(trajectory_ids=tuple("episode%d" % i for i in range(count)),
                    environment_seeds=tuple(range(count)), behavior_sha256s=("a"*64,)*count,
                    terminal_flags=(True,)*count)
        args.update(overrides)
        return leave_one_episode_out_targets(rows, **args)

    def test_exact_other_episode_average(self):
        result = self.build([[-3, -1], [-6, -2], [-9, -3], [-12, -4]])
        self.assertEqual(result.baselines[0], (-9, -3))
        self.assertEqual(result.advantages[0], (6, 2))
        self.assertEqual(result.returns[0], (-3, -1))

    def test_own_target_cannot_change_own_baseline(self):
        a = self.build([[-3, -1], [-6, -2]])
        b = self.build([[1e12, -1e12], [-6, -2]])
        self.assertEqual(a.baselines[0], b.baselines[0])

    def test_time_common_component_cancels(self):
        rows = [[-3, -1], [-6, -2], [-9, -3], [-12, -4]]
        shifted = [[row[0]-100, row[1]-10] for row in rows]
        for original, transformed in zip(self.build(rows).advantages,
                                         self.build(shifted).advantages):
            for before, after in zip(original, transformed):
                self.assertAlmostEqual(before, after, places=12)

    def test_permutation_equivariance(self):
        rows = [[-3, -1], [-6, -2], [-9, -3]]
        self.assertEqual(self.build(rows).advantages[::-1], self.build(rows[::-1]).advantages)

    def test_bad_shapes_and_values_rejected(self):
        for rows in ([], [[1]], [[1], [2, 3]], [[float("nan")], [1]], [[True], [1]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.build(rows)

    def test_mixed_behavior_and_open_episode_rejected(self):
        with self.assertRaises(ValueError):
            self.build([[1], [2]], behavior_sha256s=("a"*64, "b"*64))
        with self.assertRaises(ValueError):
            self.build([[1], [2]], terminal_flags=(True, False))

    def test_duplicate_or_missing_provenance_rejected(self):
        for kwargs in (dict(environment_seeds=(1, 1)), dict(trajectory_ids=("x", "x")),
                       dict(terminal_flags=(True,)), dict(environment_seeds=(True, 2))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build([[1], [2]], **kwargs)


if __name__ == "__main__":
    unittest.main()
