"""Artificial records only: no scientific models, forwards, environments or fits."""

import copy
import unittest

import numpy as np

from src.rl.capacity_native_tail_targets import (
    TAIL_SCHEMA, admit_tails, mc_rows, source_data_sha256, td_rows,
    validate_metadata, validate_record, validate_records,
)
from src.rl.capacity_planner_tail_targets import mc_rows as planner_mc, td_rows as planner_td


ANCESTOR = "a" * 64
TAPE = "b" * 64
ARRAY_KEYS = ("epochs", "features", "heuristics", "costs")
ROW_KEYS = ("states", "next_states", "base", "bootstrap_base", "observed_cost",
            "done", "bootstrap_residual", "targets")


def branch(root=0, candidate=0):
    epochs = np.arange(root + 8, 65, dtype=np.int64)
    features = np.broadcast_to(epochs[:, None, None], (len(epochs), 4, 31)).astype(np.float32)
    return dict(format=TAIL_SCHEMA, root_epoch=root, candidate=candidate,
                start_epoch=root + 8, label_policy="frozen_mpc", continuation_sha256=ANCESTOR,
                source_state_sha256=f"{root + 1:064x}", tape_sha256=TAPE,
                provenance="native_patient_branch", epochs=epochs, features=features,
                heuristics=1000. + epochs.astype(np.float64) * 100.,
                costs=np.arange(1, len(epochs), dtype=np.float64) * 10.)


def world(index=0):
    return [branch(index + 24 * slot, (index + 8 * slot) % 16) for slot in range(2)]


def frozen(record, value=7.):
    return np.full(len(record["epochs"]), value, dtype=np.float64)


def with_diagnostics(record):
    return dict(record, prefix_costs=np.ones(8, dtype=np.float64),
                total_branch_cost=float(np.sum(record["costs"]) + 8.),
                settlement=dict(settled=True, lost=0, delivered=0,
                                support=dict(outstanding=0.), live_ids=[]),
                policy_is_fixed_parent_not_updated_student=True,
                feature_distribution="native_observed_not_forecast_endpoint")


class NativeTailTargetTests(unittest.TestCase):
    def test_collector_audit_bundle_is_owned_hash_bound_and_not_target_cost(self):
        core = branch()
        record = with_diagnostics(core)
        snapshot = validate_record(record)
        digest = source_data_sha256(record)
        self.assertNotEqual(digest, source_data_sha256(core))
        np.testing.assert_array_equal(mc_rows(record)["targets"], mc_rows(core)["targets"])
        for field, value in (("prefix_costs", np.full(8, 2., dtype=np.float64)),
                             ("total_branch_cost", record["total_branch_cost"] + 8.),
                             ("settlement", dict(settled=True, lost=1, delivered=0))):
            changed = copy.deepcopy(record)
            changed[field] = value
            self.assertNotEqual(digest, source_data_sha256(changed))
        record["prefix_costs"][:] = 999.
        record["settlement"]["support"]["outstanding"] = 999.
        self.assertEqual(snapshot["prefix_costs"][0], 1.)
        self.assertEqual(snapshot["settlement"]["support"]["outstanding"], 0.)

    def test_incomplete_nonfinite_and_conflicting_collector_diagnostics_rejected(self):
        for field, value in (("prefix_costs", np.ones(8, dtype=np.float32)),
                             ("prefix_costs", np.full(8, np.nan)),
                             ("prefix_costs", -np.ones(8)), ("total_branch_cost", np.inf),
                             ("total_branch_cost", True), ("total_branch_cost", -1.),
                             ("settlement", dict(settled=False)),
                             ("settlement", dict(settled=True, costs=[np.inf])),
                             ("settlement", {"settled": True, 1: 0}),
                             ("feature_distribution", "forecast_endpoint"),
                             ("policy_is_fixed_parent_not_updated_student", False)):
            record = with_diagnostics(branch())
            record[field] = value
            self.assert_invalid(record)
        record = with_diagnostics(branch())
        del record["settlement"]
        self.assert_invalid(record)

    def test_exact_planner_formula_parity_at_every_native_root(self):
        for root in range(48):
            with self.subTest(root=root):
                record = branch(root, root % 16)
                arrays = {key: record[key] for key in ARRAY_KEYS}
                residuals = np.arange(len(record["epochs"]), dtype=np.float64) * 123.
                direct = mc_rows(record)
                td = td_rows(record, frozen_residuals=residuals)
                for rows, expected in ((direct, planner_mc(**arrays)),
                                       (td, planner_td(**arrays, frozen_residuals=residuals))):
                    for key in ROW_KEYS:
                        np.testing.assert_array_equal(rows[key], expected[key])
                        dtype = np.float32 if key in ("states", "next_states") else (
                            np.bool_ if key == "done" else np.float64)
                        self.assertEqual(rows[key].dtype, dtype)
                    self.assertEqual(rows["metadata"]["format"], "capacity-native-tail-targets-v1")
                    self.assertEqual(rows["metadata"]["feature_source"], "public_native_state")
                    self.assertEqual(rows["metadata"]["record"]["root_epoch"], root)
                self.assertEqual(np.count_nonzero(td["done"]), 8)
                np.testing.assert_array_equal(td["targets"][-8:], direct["targets"][-8:])
                self.assertEqual(td["metadata"]["source_data_sha256"],
                                 direct["metadata"]["source_data_sha256"])

    def test_float64_precision_and_late_small_costs(self):
        record = branch()
        record["costs"][:] = .25
        record["costs"][0] = 2. ** 40 + .125
        record["heuristics"][:] = 0.
        record["heuristics"][0] = 2. ** 40
        td = td_rows(record, frozen_residuals=frozen(record, .5))
        direct = mc_rows(record)
        self.assertEqual(td["targets"][0], (.125 + 7 * .25 + .5) / 1e6)
        self.assertEqual(direct["targets"][0], (.125 + 55 * .25) / 1e6)
        record["costs"][0] = 1e20
        td = td_rows(record, frozen_residuals=frozen(record, 0.))
        direct = mc_rows(record)
        np.testing.assert_array_equal(td["observed_cost"][1:],
                                      np.minimum(8, np.arange(55, 0, -1)) * .25 / 1e6)
        np.testing.assert_array_equal(direct["targets"][1:], np.arange(55, 0, -1) * .25 / 1e6)

    def test_terminal_bootstraps_ignored_but_raw_snapshot_required(self):
        record = branch(47)
        residuals = frozen(record, 2e6)
        before = td_rows(record, frozen_residuals=residuals)
        direct = mc_rows(record)
        zero = td_rows(record, frozen_residuals=frozen(record, 0.))
        self.assertEqual(before["bootstrap_residual"][0], 2.)
        np.testing.assert_allclose(before["targets"] - zero["targets"],
                                   np.where(before["done"], 0., 2.))
        record["heuristics"][-1] = np.finfo(np.float64).max
        residuals[-1] = np.finfo(np.float64).max
        np.testing.assert_array_equal(td_rows(record, frozen_residuals=residuals)["targets"],
                                      before["targets"])
        np.testing.assert_array_equal(mc_rows(record)["targets"], direct["targets"])
        with self.assertRaises(TypeError):
            td_rows(record)

    def test_exact_two_branch_schedule_for_every_world(self):
        for index in range(24):
            with self.subTest(index=index):
                records = world(index)
                snapshots = validate_records(records, world_index=index, continuation_sha256=ANCESTOR,
                                             tape_sha256=TAPE)
                self.assertEqual([r["root_epoch"] for r in snapshots], [index, index + 24])
                for kind in ("mc", "td8"):
                    values = [frozen(r, slot * 1e6) for slot, r in enumerate(records)]
                    admitted = admit_tails(records, world_index=index, continuation_sha256=ANCESTOR,
                                           target_kind=kind,
                                           frozen_residuals=values if kind == "td8" else None)
                    expected = [mc_rows(r) if kind == "mc" else td_rows(r, frozen_residuals=v)
                                for r, v in zip(records, values)]
                    for key in ROW_KEYS:
                        np.testing.assert_array_equal(admitted[key],
                                                      np.concatenate([r[key] for r in expected]))
                    self.assertEqual(len(admitted["targets"]), 88 - 2 * index)
                    self.assertEqual(admitted["source_hashes"], [source_data_sha256(r) for r in records])
                    self.assertEqual(admitted["metadata"]["tails_per_world"], 2)

    def test_input_order_preserves_residual_branch_alignment(self):
        records = world(19)[::-1]
        values = [frozen(r, v) for r, v in zip(records, (2e6, -3e6))]
        rows = admit_tails(records, world_index=19, continuation_sha256=ANCESTOR,
                          target_kind="td8", frozen_residuals=values)
        self.assertEqual([r["record"]["root_epoch"] for r in rows["metadata"]["records"]], [43, 19])
        np.testing.assert_array_equal(rows["targets"], np.concatenate([
            td_rows(r, frozen_residuals=v)["targets"] for r, v in zip(records, values)]))

    def test_metadata_only_validation_for_checkpoint_binding(self):
        metadata = [{k: v for k, v in r.items() if k not in ARRAY_KEYS} for r in world(23)]
        validated = validate_metadata(metadata, world_index=23, continuation_sha256=ANCESTOR)
        self.assertEqual(list(validated), metadata)
        metadata[0]["provenance"] = "forecast"
        self.assertEqual(validated[0]["provenance"], "native_patient_branch")
        with self.assertRaises(ValueError):
            validate_metadata(metadata, world_index=23, continuation_sha256=ANCESTOR)
        with self.assertRaises(ValueError):
            validate_metadata(world(23), world_index=23, continuation_sha256=ANCESTOR)

    def test_copy_ownership_readonly_strided_and_list_epochs(self):
        records = world()
        records[0]["epochs"] = records[0]["epochs"].tolist()
        records[1]["features"] = np.asfortranarray(records[1]["features"])
        for record in records:
            for key in ARRAY_KEYS:
                if isinstance(record[key], np.ndarray):
                    record[key].flags.writeable = False
        values = [frozen(r) for r in records]
        for value in values:
            value.flags.writeable = False
        saved = copy.deepcopy(records)
        validated = validate_record(records[1])
        rows = admit_tails(records, world_index=0, continuation_sha256=ANCESTOR,
                          target_kind="td8", frozen_residuals=values)
        before = copy.deepcopy(rows)
        for key in ARRAY_KEYS:
            self.assertFalse(np.shares_memory(validated[key], records[1][key]))
            self.assertTrue(validated[key].flags.c_contiguous)
        for row_key in ROW_KEYS:
            for record, snapshot in zip(records, saved):
                for key in ARRAY_KEYS:
                    np.testing.assert_array_equal(record[key], snapshot[key])
                    self.assertFalse(np.shares_memory(rows[row_key], record[key]))
            for value in values:
                self.assertFalse(np.shares_memory(rows[row_key], value))
        for record in records:
            record["root_epoch"] = 99
            for key in ARRAY_KEYS:
                if isinstance(record[key], np.ndarray):
                    record[key].flags.writeable = True
                    record[key][...] = 0
        for value in values:
            value.flags.writeable = True
            value[:] = -9e6
        for key in ROW_KEYS:
            np.testing.assert_array_equal(rows[key], before[key])
        self.assertEqual(rows["metadata"], before["metadata"])
        rows["states"][:] = -99
        self.assertFalse(np.any(rows["next_states"] == -99))

    def test_stable_source_hash_canonicalizes_layout_endianness_and_keys(self):
        record = branch()
        expected = source_data_sha256(record)
        changed = dict(reversed(list(record.items())))
        for key in ARRAY_KEYS:
            values = record[key]
            changed[key] = np.asfortranarray(values.astype(values.dtype.newbyteorder(">")))
        self.assertEqual(source_data_sha256(changed), expected)
        for key in ARRAY_KEYS:
            values = record[key]
            backing = np.repeat(values, 2, axis=0)
            changed[key] = backing[::2]
        self.assertEqual(source_data_sha256(changed), expected)
        changed["epochs"] = tuple(int(e) for e in record["epochs"])
        self.assertEqual(source_data_sha256(changed), expected)
        self.assertEqual(len(expected), 64)
        self.assertEqual(expected, mc_rows(record)["metadata"]["source_data_sha256"])
        self.assertEqual(expected, td_rows(record, frozen_residuals=frozen(record, 2e6))[
            "metadata"]["source_data_sha256"])

    def test_source_hash_binds_every_variable_metadata_field_and_array(self):
        record = branch()
        digest = source_data_sha256(record)
        for field in ("continuation_sha256", "source_state_sha256", "tape_sha256", "candidate"):
            changed = copy.deepcopy(record)
            changed[field] = 1 if field == "candidate" else "c" * 64
            self.assertNotEqual(digest, source_data_sha256(changed), field)
        for field in ("features", "heuristics", "costs"):
            changed = copy.deepcopy(record)
            changed[field].flat[-1] += 1
            self.assertNotEqual(digest, source_data_sha256(changed), field)
        self.assertNotEqual(digest, source_data_sha256(branch(1)))

    def test_forecast_records_cannot_be_admitted_by_relabeling(self):
        for field, value in (("format", "capacity-policy-tail-record-v1"),
                             ("format", "capacity-planner-tail-record-v1"),
                             ("provenance", "public_forecast_not_native"),
                             ("provenance", "predicted_endpoint"), ("label_policy", "adaptive")):
            record = branch()
            record[field] = value
            self.assert_invalid(record)
        for field, value in (("quantile", .5), ("label_source", "public_forecast_not_native"),
                             ("decision_epoch", 0), ("unexpected", np.ones(3))):
            record = branch()
            record[field] = value
            self.assert_invalid(record)

    def test_missing_fields_and_invalid_record_types(self):
        for key in branch():
            record = branch()
            del record[key]
            self.assert_invalid(record)
        for record in (None, [], "native", 0):
            self.assert_invalid(record)

    def test_metadata_integer_and_string_types_are_strict(self):
        for field in ("root_epoch", "candidate", "start_epoch"):
            for value in (True, 0., "0", None, np.int64(0), np.array(0), -1, 64):
                record = branch()
                record[field] = value
                self.assert_invalid(record)
        for field in ("format", "label_policy", "provenance"):
            for value in (None, True, 0, [branch()[field]], np.array(branch()[field])):
                record = branch()
                record[field] = value
                self.assert_invalid(record)

    def test_sha256_validation_and_expected_context(self):
        for field in ("continuation_sha256", "source_state_sha256", "tape_sha256"):
            for value in (None, 0, True, "", "a" * 63, "A" * 64, "g" * 64, b"a" * 64):
                record = branch()
                record[field] = value
                self.assert_invalid(record)
        for kwargs in (dict(continuation_sha256="c" * 64), dict(tape_sha256="c" * 64),
                       dict(continuation_sha256=True), dict(tape_sha256="bad")):
            for builder in (validate_record, mc_rows):
                with self.assertRaises(ValueError):
                    builder(branch(), **kwargs)
            with self.assertRaises(ValueError):
                td_rows(branch(), frozen_residuals=frozen(branch()), **kwargs)

    def test_wrong_horizon_shapes_epoch_types_and_noncontiguous_paths(self):
        for field in ARRAY_KEYS:
            for transform in (lambda a: a[:-1], lambda a: a[None], lambda a: a[::-1]):
                record = branch()
                record[field] = transform(record[field])
                # Reversal is only invalid for the chronological epoch path.
                if field != "epochs" and record[field].shape == branch()[field].shape:
                    continue
                self.assert_invalid(record)
        for dtype in (np.float64, bool, str, object):
            record = branch()
            record["epochs"] = record["epochs"].astype(dtype)
            self.assert_invalid(record)
        for value in (True, 8., "8", None, [8]):
            record = branch()
            record["epochs"] = record["epochs"].tolist()
            record["epochs"][0] = value
            self.assert_invalid(record)
        for index, value in ((0, 9), (3, 10), (-1, 63), (-1, 65)):
            record = branch()
            record["epochs"][index] = value
            self.assert_invalid(record)
        record = branch()
        record["start_epoch"] += 1
        self.assert_invalid(record)
        for shape in ((57, 124), (57, 4, 30), (57, 3, 31), (57, 4, 31, 1)):
            record = branch()
            record["features"] = np.zeros(shape, dtype=np.float32)
            self.assert_invalid(record)

    def test_strict_float_arrays_nonfinite_and_negative_costs(self):
        for field in ("features", "heuristics", "costs"):
            wrong_float = np.float64 if field == "features" else np.float32
            for dtype in (wrong_float, np.float16, np.int64, bool, str, complex, object):
                record = branch()
                record[field] = record[field].astype(dtype)
                self.assert_invalid(record)
            record = branch()
            record[field] = record[field].tolist()
            self.assert_invalid(record)
            for value in (np.nan, np.inf, -np.inf):
                record = branch()
                record[field].flat[-1] = value
                self.assert_invalid(record)
        record = branch()
        record["costs"][-1] = -.01
        self.assert_invalid(record)

    def test_td_residual_type_shape_and_terminal_finiteness(self):
        record = branch()
        for value in (None, 0., True, "0", lambda x: x, frozen(record).tolist(),
                      frozen(record).astype(np.float32), frozen(record).astype(complex),
                      frozen(record).astype(object), frozen(record)[:-1], frozen(record)[:, None]):
            with self.assertRaises(ValueError):
                td_rows(record, frozen_residuals=value)
        for value in (np.nan, np.inf, -np.inf):
            residuals = frozen(record)
            residuals[-1] = value
            with self.assertRaises(ValueError):
                td_rows(record, frozen_residuals=residuals)

    def test_float64_arithmetic_overflow_is_rejected(self):
        record = branch()
        record["costs"][:] = np.finfo(np.float64).max
        self.assert_invalid_targets(record)
        record = branch()
        record["costs"][0] = np.finfo(np.float64).max
        record["heuristics"][0] = -np.finfo(np.float64).max
        self.assert_invalid_targets(record)
        record = branch()
        record["heuristics"][8] = np.finfo(np.float64).max
        with self.assertRaisesRegex(ValueError, "overflows float64"):
            td_rows(record, frozen_residuals=frozen(record, np.finfo(np.float64).max))
        records = world()
        records[-1]["costs"][:] = np.finfo(np.float64).max
        with self.assertRaisesRegex(ValueError, "overflows float64"):
            admit_tails(records, world_index=0, continuation_sha256=ANCESTOR)

    def test_duplicate_wrong_root_candidate_tape_and_ancestor_rejected(self):
        invalid = [[branch(), branch()], [branch(), branch(23, 8)],
                   [branch(), branch(24, 9)], [branch(), branch(0, 8)]]
        for field in ("continuation_sha256", "tape_sha256"):
            records = world()
            records[1][field] = "c" * 64
            invalid.append(records)
        for records in invalid:
            with self.assertRaises(ValueError):
                validate_records(records, world_index=0, continuation_sha256=ANCESTOR)
            with self.assertRaises(ValueError):
                admit_tails(records, world_index=0, continuation_sha256=ANCESTOR)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_records([branch(), branch()], world_index=0, continuation_sha256=ANCESTOR)
        with self.assertRaisesRegex(ValueError, "tape_sha256 mismatch"):
            admit_tails(world(), world_index=0, continuation_sha256=ANCESTOR, tape_sha256="c" * 64)

    def test_two_branches_only_not_three_quantiles(self):
        for records in (None, {}, iter(world()), [], world()[:1], world() + [branch()], world() * 3):
            with self.assertRaises(ValueError):
                admit_tails(records, world_index=0, continuation_sha256=ANCESTOR)
        for value in (None, True, -1, 24, 0., "0", np.int64(0)):
            with self.assertRaises(ValueError):
                admit_tails(world(), world_index=value, continuation_sha256=ANCESTOR)
        for value in (None, True, "bad"):
            with self.assertRaises(ValueError):
                admit_tails(world(), world_index=0, continuation_sha256=value)

    def test_batch_target_kind_and_residual_alignment_required(self):
        records = world()
        for kind in (None, True, "td", "forecast", np.array("mc")):
            with self.assertRaises(ValueError):
                admit_tails(records, world_index=0, continuation_sha256=ANCESTOR, target_kind=kind)
        values = [frozen(r) for r in records]
        for bad in (None, [], values[:1], values + values, values[::-1], [values[0], None]):
            with self.assertRaises(ValueError):
                admit_tails(records, world_index=0, continuation_sha256=ANCESTOR,
                            target_kind="td8", frozen_residuals=bad)
        with self.assertRaises(ValueError):
            admit_tails(records, world_index=0, continuation_sha256=ANCESTOR,
                        target_kind="mc", frozen_residuals=values)

    def assert_invalid(self, record):
        with self.assertRaises(ValueError):
            validate_record(record)
        with self.assertRaises(ValueError):
            source_data_sha256(record)
        self.assert_invalid_targets(record)

    def assert_invalid_targets(self, record):
        with self.assertRaises(ValueError):
            mc_rows(record)
        with self.assertRaises(ValueError):
            td_rows(record, frozen_residuals=frozen(branch()))


if __name__ == "__main__":
    unittest.main()
