"""Handcrafted arrays only: no models, environments, checkpoints or optimizers."""

import unittest

import numpy as np

from src.rl.capacity_planner_tail_targets import mc_rows, td_rows


def tail(length=56):
    epochs = np.arange(64 - length, 65, dtype=np.int64)
    features = np.broadcast_to(epochs[:, None, None], (length + 1, 4, 31)).astype(np.float32)
    return dict(features=features, heuristics=1000. + epochs.astype(np.float64) * 100.,
                costs=np.arange(1, length + 1, dtype=np.float64) * 10., epochs=epochs)


def frozen_values(data, value=0.):
    return np.full(len(data["costs"]) + 1, value, dtype=np.float64)


class TailTargetTests(unittest.TestCase):
    def test_every_valid_tail_length_and_eight_step_boundary(self):
        for length in range(1, 57):
            with self.subTest(length=length):
                data = tail(length)
                ends = np.minimum(np.arange(length) + 8, length)
                done = ends == length
                frozen = np.arange(length + 1, dtype=np.float64) * 123.
                rows = td_rows(**data, frozen_residuals=frozen)
                mc = mc_rows(**data)
                sums = np.asarray([sum(data["costs"][t:end]) for t, end in enumerate(ends)])
                suffixes = np.asarray([sum(data["costs"][t:]) for t in range(length)])
                expected_td = [(sums[t]
                                + (0. if done[t] else data["heuristics"][end] + frozen[end])
                                - data["heuristics"][t]) / 1e6 for t, end in enumerate(ends)]
                expected_mc = (suffixes - data["heuristics"][:-1]) / 1e6
                np.testing.assert_array_equal(rows["targets"], expected_td)
                np.testing.assert_array_equal(mc["targets"], expected_mc)
                np.testing.assert_array_equal(rows["states"], data["features"][:-1])
                np.testing.assert_array_equal(mc["states"], rows["states"])
                np.testing.assert_array_equal(rows["next_states"], data["features"][ends])
                np.testing.assert_array_equal(mc["next_states"], np.repeat(data["features"][-1:], length, axis=0))
                np.testing.assert_array_equal(rows["done"], done)
                self.assertEqual(np.count_nonzero(rows["done"]), min(8, length))
                np.testing.assert_array_equal(mc["done"], np.ones(length, dtype=bool))
                np.testing.assert_array_equal(rows["base"], data["heuristics"][:-1] / 1e6)
                np.testing.assert_array_equal(rows["observed_cost"], sums / 1e6)
                np.testing.assert_array_equal(mc["observed_cost"], suffixes / 1e6)
                np.testing.assert_array_equal(rows["bootstrap_base"],
                                              np.where(done, 0., data["heuristics"][ends] / 1e6))
                np.testing.assert_array_equal(rows["bootstrap_residual"], np.where(done, 0., frozen[ends] / 1e6))
                for key in ("bootstrap_base", "bootstrap_residual"):
                    np.testing.assert_array_equal(mc[key], np.zeros(length))
                for result in (rows, mc):
                    self.assertEqual(result["states"].shape, (length, 4, 31))
                    self.assertEqual(result["next_states"].shape, (length, 4, 31))
                    self.assertEqual(result["states"].dtype, np.float32)
                    self.assertEqual(result["next_states"].dtype, np.float32)
                    self.assertEqual(result["done"].dtype, np.bool_)
                    for key in ("base", "bootstrap_base", "observed_cost", "bootstrap_residual", "targets"):
                        self.assertEqual(result[key].dtype, np.float64)
                        self.assertEqual(result[key].shape, (length,))
                np.testing.assert_array_equal(rows["targets"][done], mc["targets"][done])
                indices = np.asarray([0, length // 2, length - 1])
                np.testing.assert_array_equal(rows["states"][indices], mc["states"][indices])

    def test_costs_and_target_subtraction_keep_float64_precision(self):
        data = tail(9)
        data["costs"][:] = .25
        data["costs"][0] = 2. ** 40 + .125
        data["heuristics"][:] = 0.
        data["heuristics"][0] = 2. ** 40
        td = td_rows(**data, frozen_residuals=frozen_values(data, .5))
        mc = mc_rows(**data)
        self.assertEqual(td["targets"][0], (.125 + 7 * .25 + .5) / 1e6)
        self.assertEqual(mc["targets"][0], (.125 + 8 * .25) / 1e6)

    def test_late_small_costs_are_not_lost_to_prefix_sum_cancellation(self):
        data = tail()
        data["costs"][:] = .25
        data["costs"][0] = 1e20
        data["heuristics"][:] = 0.
        td = td_rows(**data, frozen_residuals=frozen_values(data))
        mc = mc_rows(**data)
        np.testing.assert_array_equal(td["observed_cost"][1:],
                                      np.minimum(8, np.arange(55, 0, -1)) * .25 / 1e6)
        np.testing.assert_array_equal(mc["targets"][1:], np.arange(55, 0, -1) * .25 / 1e6)

    def test_terminal_heuristic_and_residual_are_ignored(self):
        for length in (1, 3, 8, 9, 56):
            with self.subTest(length=length):
                data = tail(length)
                frozen = frozen_values(data, 7.)
                before_td = td_rows(**data, frozen_residuals=frozen)
                before_mc = mc_rows(**data)
                data["heuristics"][-1] = np.finfo(np.float64).max
                after_td = td_rows(**data, frozen_residuals=frozen)
                after_mc = mc_rows(**data)
                np.testing.assert_array_equal(before_td["targets"], after_td["targets"])
                np.testing.assert_array_equal(before_mc["targets"], after_mc["targets"])
                frozen[-1] = np.finfo(np.float64).max
                ignored = td_rows(**data, frozen_residuals=frozen)
                np.testing.assert_array_equal(ignored["targets"], before_td["targets"])

    def test_residual_is_explicit_unscaled_and_snapshotted(self):
        data = tail()
        frozen = frozen_values(data, 2e6)
        before = {key: value.copy() for key, value in data.items()}
        rows = td_rows(**data, frozen_residuals=frozen)
        zero = td_rows(**data, frozen_residuals=frozen_values(data))
        np.testing.assert_allclose(rows["targets"] - zero["targets"], np.where(rows["done"], 0., 2.))
        self.assertEqual(rows["bootstrap_residual"][0], 2.)
        np.testing.assert_array_equal(frozen, frozen_values(data, 2e6))
        for key in data:
            np.testing.assert_array_equal(data[key], before[key])
        expected = {key: value.copy() for key, value in rows.items() if isinstance(value, np.ndarray)}
        frozen[...] = -9e6
        for value in data.values():
            value[...] = 0
        for key, value in expected.items():
            np.testing.assert_array_equal(rows[key], value)
        with self.assertRaises(TypeError):
            td_rows(**before)

    def test_readonly_inputs_lists_and_owned_outputs(self):
        data = tail()
        for value in data.values():
            value.flags.writeable = False
        frozen = frozen_values(data, -10.)
        frozen.flags.writeable = False
        rows = td_rows(**data, frozen_residuals=frozen)
        direct = mc_rows(**data)
        lists = {key: value.tolist() for key, value in data.items()}
        listed = td_rows(**lists, frozen_residuals=frozen.tolist())
        for key, value in rows.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, listed[key])
                for source in data.values():
                    self.assertFalse(np.shares_memory(value, source))
        self.assertFalse(np.shares_memory(rows["states"], direct["states"]))
        rows["states"][:] = -1
        self.assertEqual(data["features"][0, 0, 0], 8.)

    def test_metadata_binds_all_supplied_tail_data_for_both_targets(self):
        data = tail()
        td = td_rows(**data, frozen_residuals=frozen_values(data, 5.))
        mc = mc_rows(**data)
        digest = td["metadata"]["source_data_sha256"]
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, mc["metadata"]["source_data_sha256"])
        self.assertEqual(td["metadata"]["start_epoch"], 8)
        self.assertEqual(td["metadata"]["td_horizon"], 8)
        self.assertEqual(mc["metadata"]["terminal_epoch"], 64)
        self.assertEqual(mc["metadata"]["tail_length"], 56)
        for key in ("features", "heuristics", "costs"):
            changed = tail()
            changed[key].flat[-1] += 1
            self.assertNotEqual(digest, td_rows(**changed, frozen_residuals=frozen_values(changed, 5.))["metadata"]["source_data_sha256"])

    def test_invalid_lengths_and_shapes_rejected(self):
        for length in (0, 57, 64):
            self.assert_invalid(tail(length))
        for field in ("features", "heuristics", "costs", "epochs"):
            with self.subTest(field=field):
                data = tail()
                data[field] = data[field][:-1]
                self.assert_invalid(data)
        for shape in ((57, 4, 30), (57, 3, 31), (57, 4, 31, 1), (57, 124)):
            data = tail()
            data["features"] = np.zeros(shape)
            self.assert_invalid(data)
        for field in ("costs", "heuristics", "epochs"):
            data = tail()
            data[field] = data[field][:, None]
            self.assert_invalid(data)

    def test_epoch_path_must_be_integer_contiguous_and_end_at_64(self):
        for defect in ("gap", "duplicate", "reverse", "shift", "fraction", "float", "bool", "string", "nan"):
            with self.subTest(defect=defect):
                data = tail(8)
                if defect == "gap":
                    data["epochs"][3] += 2
                elif defect == "duplicate":
                    data["epochs"][3] = data["epochs"][2]
                elif defect == "reverse":
                    data["epochs"] = data["epochs"][::-1]
                elif defect == "shift":
                    data["epochs"] -= 1
                elif defect in ("fraction", "float", "nan"):
                    data["epochs"] = data["epochs"].astype(float)
                    if defect != "float":
                        data["epochs"][0] = .5 if defect == "fraction" else np.nan
                else:
                    data["epochs"] = data["epochs"].astype(bool if defect == "bool" else str)
                self.assert_invalid(data)

    def test_nonfinite_values_negative_costs_and_feature_overflow_rejected(self):
        for field in ("features", "heuristics", "costs"):
            for invalid in (np.nan, np.inf, -np.inf):
                with self.subTest(field=field, invalid=invalid):
                    data = tail()
                    data[field].flat[-1] = invalid
                    self.assert_invalid(data)
        data = tail()
        data["costs"][-1] = -.01
        self.assert_invalid(data)
        data = tail()
        data["features"] = data["features"].astype(np.float64)
        data["features"].flat[-1] = np.finfo(np.float64).max
        self.assert_invalid(data)

    def test_nonnumeric_complex_and_ragged_arrays_rejected(self):
        for field in ("features", "heuristics", "costs"):
            for dtype in (str, complex, bool, object):
                with self.subTest(field=field, dtype=dtype):
                    data = tail()
                    data[field] = data[field].astype(dtype)
                    self.assert_invalid(data)
            data = tail()
            data[field] = [[1.], [2., 3.]]
            self.assert_invalid(data)

    def test_invalid_residuals_rejected_even_at_terminal(self):
        for length in (1, 8, 56):
            for residual in (np.nan, np.inf, -np.inf, [0.], 0., None, "0", 1j, True):
                with self.subTest(length=length, residual=residual), self.assertRaises(ValueError):
                    td_rows(**tail(length), frozen_residuals=residual)
            for invalid in (np.nan, np.inf, -np.inf):
                frozen = frozen_values(tail(length))
                frozen[-1] = invalid
                with self.assertRaises(ValueError):
                    td_rows(**tail(length), frozen_residuals=frozen)
            for shape in ((length,), (length + 2,), (length + 1, 1)):
                with self.assertRaises(ValueError):
                    td_rows(**tail(length), frozen_residuals=np.zeros(shape))

    def test_float64_sum_and_target_overflow_rejected(self):
        data = tail()
        data["costs"][:] = np.finfo(np.float64).max
        self.assert_invalid(data)
        data = tail()
        data["heuristics"][8] = np.finfo(np.float64).max
        with self.assertRaisesRegex(ValueError, "overflows float64"):
            td_rows(**data, frozen_residuals=frozen_values(data, np.finfo(np.float64).max))
        data = tail(1)
        data["costs"][0] = np.finfo(np.float64).max
        data["heuristics"][0] = -np.finfo(np.float64).max
        self.assert_invalid(data)

    def assert_invalid(self, data):
        with self.assertRaises(ValueError):
            td_rows(**data, frozen_residuals=frozen_values(data))
        with self.assertRaises(ValueError):
            mc_rows(**data)


if __name__ == "__main__":
    unittest.main()
