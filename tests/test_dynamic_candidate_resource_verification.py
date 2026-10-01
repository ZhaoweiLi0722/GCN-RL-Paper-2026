"""Invented raw receipts only; no environment, model, checkpoint or optimizer.

Continuous net-flow conservation uses abs(fsum(values)) <=
1e-12 * fsum(abs(value) for value in values) + 1e-8. This is a validation
tolerance, never permission to round or replace the recorded resource values.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from src.rl.candidate_pilot_verification import CAUSES, json_hash
from src.rl import dynamic_candidate_verification as legacy
from src.rl.dynamic_candidate_resource_verification import (
    continuous_vector,
    executed_resources,
    read_dynamic_raw_episodes,
    verify_dynamic_episode,
    verify_dynamic_raw_bundle,
)
from src.rl.dynamic_candidate_saved_readout import summarize_qualification
from tests.test_dynamic_candidate_verification import (
    fixture_config,
    raw_episode,
    write_bundle,
)


CHANNELS = ("specimen_transfers", "capacity_transfers", "reagent_transfers", "replenishment")
TRANSFERS = ("capacity_transfers", "reagent_transfers")
FRACTIONAL = {
    "specimen_transfers": [0, 0],
    "capacity_transfers": [3.0000019669532776, -3.0000019669532776],
    "reagent_transfers": [-8.600000560283661, 8.600000560283661],
    "replenishment": [0.12500000000000003, 1.0000000000000002],
}


def fractional_resources(header, rows, final):
    for row in rows:
        row["event"]["info"].update(copy.deepcopy(FRACTIONAL))


def resource_info(n=2, **overrides):
    return {channel: [0] * n for channel in CHANNELS} | overrides


class ContinuousVectorTests(unittest.TestCase):
    def test_signed_fractional_values_are_preserved_exactly(self):
        values = [3.0000019669532776, -8.600000560283661, -0.0, 5e-324]
        for supplied in (values, tuple(values)):
            with self.subTest(container=type(supplied).__name__):
                result = continuous_vector(supplied, 4, signed=True)
                self.assertEqual(result, values)
                self.assertEqual([v.hex() for v in result], [v.hex() for v in values])
        self.assertEqual(continuous_vector([0, 1, .125], 3), [0, 1, .125])

    def test_default_unsigned_rejects_even_tiny_negative_values(self):
        for value in (-1, -.125, -5e-324):
            with self.subTest(value=value), self.assertRaises(ValueError):
                continuous_vector([value, 0], 2)

    def test_wrong_width_and_nonvector_inputs_fail(self):
        for values in ([], [0], [0, 0, 0], [[0], [0]], None, 0, "00", {0: 0, 1: 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                continuous_vector(values, 2, signed=True)

    def test_nonfinite_boolean_and_nonnumeric_elements_fail(self):
        for signed in (False, True):
            for value in (math.nan, math.inf, -math.inf, True, False, "0.5", None):
                with self.subTest(signed=signed, value=value), self.assertRaises(ValueError):
                    continuous_vector([value, 0], 2, signed=signed)


class ExecutedResourceTests(unittest.TestCase):
    def test_mixed_integer_and_continuous_channels_are_lossless(self):
        info = copy.deepcopy(FRACTIONAL)
        info["specimen_transfers"] = [2, -2]
        before = copy.deepcopy(info)
        result = executed_resources(info, 2)
        self.assertEqual(result, before)
        self.assertTrue(all(type(v) is int for v in result["specimen_transfers"]))
        for channel in CHANNELS[1:]:
            self.assertEqual([v.hex() for v in result[channel]], [v.hex() for v in before[channel]])
        self.assertEqual(info, before)

    def test_legal_cancellation_drift_is_not_rounded_or_rebalanced(self):
        values = [.1, .2, -.3]
        self.assertNotEqual(math.fsum(values), 0)
        for channel in TRANSFERS:
            with self.subTest(channel=channel):
                info = resource_info(3, **{channel: values})
                self.assertEqual(executed_resources(info, 3)[channel], values)

    def test_absolute_and_throughput_relative_tolerances_are_additive(self):
        cases = (
            ([1., -1., 5e-9], True),
            ([1., -1., 2e-8], False),
            ([1., -1., 1.0001e-8], True),
            ([1., -1., 1.0003e-8], False),
            ([1e8, -1e8, .00015], True),
            ([1e8, -1e8, .0003], False),
        )
        for channel in TRANSFERS:
            for values, accepted in cases:
                with self.subTest(channel=channel, values=values):
                    tolerance = 1e-12 * math.fsum(abs(v) for v in values) + 1e-8
                    self.assertEqual(abs(math.fsum(values)) <= tolerance, accepted)
                    info = resource_info(3, **{channel: values})
                    if accepted:
                        self.assertEqual(executed_resources(info, 3)[channel], values)
                    else:
                        with self.assertRaises(ValueError):
                            executed_resources(info, 3)

    def test_fsum_detects_imbalance_hidden_by_naive_cancellation(self):
        # Binary64 sum loses .5 here, crossing the documented throughput bound.
        values = [1e16, 20000.5, -1e16]
        tolerance = 1e-12 * math.fsum(abs(v) for v in values) + 1e-8
        naive_net = 0.
        for value in values:
            naive_net += value
        self.assertLessEqual(abs(naive_net), tolerance)
        self.assertGreater(abs(math.fsum(values)), tolerance)
        for channel in TRANSFERS:
            for ordered in (values, list(reversed(values)), [values[0], values[2], values[1]]):
                with self.subTest(channel=channel, ordered=ordered), self.assertRaises(ValueError):
                    executed_resources(resource_info(3, **{channel: ordered}), 3)

    def test_all_channels_require_correct_width_and_finite_nonboolean_values(self):
        for channel in CHANNELS:
            for values in ([0], [0, 0, 0], [math.nan, 0], [math.inf, 0],
                           [-math.inf, 0], [True, 0], [False, 0], ["0", 0]):
                with self.subTest(channel=channel, values=values), self.assertRaises(ValueError):
                    executed_resources(resource_info(**{channel: values}), 2)

    def test_specimen_stays_integer_and_exactly_conserved(self):
        for values in ([.5, -.5], [1.0000000000000002, -1.0000000000000002],
                       [5e-324, -5e-324], [1, 0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                executed_resources(resource_info(specimen_transfers=values), 2)

    def test_replenishment_is_nonnegative_not_a_net_flow(self):
        values = [3.0000019669532776, .125]
        self.assertEqual(executed_resources(resource_info(replenishment=values), 2)["replenishment"], values)
        for value in (-.25, -5e-324):
            with self.subTest(value=value), self.assertRaises(ValueError):
                executed_resources(resource_info(replenishment=[value, 1.]), 2)

    def test_missing_channels_fail_closed(self):
        for channel in CHANNELS:
            info = resource_info()
            del info[channel]
            with self.subTest(channel=channel), self.assertRaises(KeyError):
                executed_resources(info, 2)


class DynamicResourceVerificationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.cfg, self.streams = fixture_config()

    def episode(self, role="own_ppo"):
        header, rows, final = raw_episode(60, role, 0, 123)
        fractional_resources(header, rows, final)
        return header, rows, final

    def test_episode_preserves_exact_resources_hashes_and_input(self):
        header, rows, final = self.episode()
        before = copy.deepcopy((header, rows, final, self.cfg))
        outcome = verify_dynamic_episode(header, rows, final, self.cfg)
        self.assertEqual(outcome["raw_episode_sha256"], json_hash([header, rows, final]))
        self.assertEqual(outcome["raw_header_sha256"], json_hash(header))
        self.assertEqual((outcome["cost"], outcome["losses"], outcome["completions"]), (100, 1, 1))
        for action in outcome["actions"]:
            self.assertEqual(action["executed"], FRACTIONAL)
            self.assertEqual(action["specimen_requested_integer_net"], [2, -2])
        self.assertEqual((header, rows, final, self.cfg), before)
        json.dumps(outcome, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "non-integer raw vector"):
            legacy.verify_dynamic_episode(header, rows, final, self.cfg)

    def test_integer_fixture_episode_and_raw_bundle_output_parity(self):
        for role in legacy.CONTROLLERS:
            with self.subTest(role=role):
                header, rows, final = raw_episode(60, role, 0, 123)
                old = legacy.verify_dynamic_episode(header, rows, final, self.cfg)
                new = verify_dynamic_episode(header, rows, final, self.cfg)
                self.assertEqual(new, old)
        index = write_bundle(self.root, self.cfg, self.streams)
        old = legacy.verify_dynamic_raw_bundle(self.root, index, self.cfg, self.streams)
        new = verify_dynamic_raw_bundle(self.root, index, self.cfg, self.streams)
        for key in ("files", "index_sha256", "config_sha256", "streams_sha256", "analysis"):
            self.assertEqual(new[key], old[key])
        self.assertEqual(new["outcomes"], old["outcomes"])
        old_files, old_outcomes = legacy.read_dynamic_raw_episodes(self.root, index, self.cfg)
        new_files, new_outcomes = read_dynamic_raw_episodes(self.root, index, self.cfg)
        self.assertEqual((new_files, new_outcomes), (old_files, old_outcomes))
        for record in new_files:
            raw = (self.root / record["path"]).read_bytes()
            self.assertEqual(record["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(record["bytes"], len(raw))
        self.assertNotEqual(new["format"], old["format"])
        self.assertNotEqual(new["verification"], old["verification"])
        self.assertFalse(new["grants_scientific_execution_authorization"])

    def test_patient_and_specimen_vectors_remain_integer(self):
        fields = ("patients_lost", "patients_completed", "demand", "waiting_patients",
                  "in_production_patients", "specimen_in_transit",
                  "specimen_requested_integer_net", "specimen_transfers", *CAUSES)
        for field in fields:
            for value in (.25, True):
                header, rows, final = self.episode()
                rows[0]["event"]["info"][field][0] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    verify_dynamic_episode(header, rows, final, self.cfg)
        for field in ("identity_active_count", "identity_terminal_count", "specimen_route_count",
                      "blocked_specimen_requests", "blocked_specimen_inbound_requests",
                      "blocked_specimen_outbound_requests"):
            header, rows, final = self.episode()
            rows[0]["event"]["info"][field] = .5
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_dynamic_episode(header, rows, final, self.cfg)

    def test_raw_patient_fractions_fail_even_with_valid_file_hashes(self):
        for field in ("patients_lost", "patients_completed", "demand", "waiting_patients",
                      "in_production_patients", "specimen_in_transit", *CAUSES):
            def mutate(header, rows, final):
                fractional_resources(header, rows, final)
                rows[0]["event"]["info"][field][0] = .25

            index = write_bundle(self.root, self.cfg, self.streams, mutate=mutate)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "non-integer"):
                read_dynamic_raw_episodes(self.root, index[:1], self.cfg)

        def fractional_registry(header, rows, final):
            fractional_resources(header, rows, final)
            final["scalars"]["cumulative_enrolled"] = 3.25
            rows[-1]["event"]["audit"]["record"]["next_state_token"] = json_hash(final)

        index = write_bundle(self.root, self.cfg, self.streams, mutate=fractional_registry)
        with self.assertRaisesRegex(ValueError, "non-integer"):
            read_dynamic_raw_episodes(self.root, index[:1], self.cfg)

    def test_valid_role_split_and_selection_contracts_with_continuous_resources(self):
        cases = [(role, "test", None) for role in legacy.CONTROLLERS]
        cases += [("initializer_greedy", "qualification", "greedy"),
                  ("r4", "qualification", "reference"), ("r4", "demonstration", "reference")]
        cases += [(role, "preflight", "reference" if role == "r4" else
                   "anchor" if role == "full_mdl2" else "greedy" if role == "own_frozen" else "sample")
                  for role in ("preflight", *legacy.CONTROLLERS)]
        cases += [(role, "training", "sample") for role in ("own_ppo", "own_bc_continue")]
        for role, split, selection in cases:
            fixture_role = "own_frozen" if role == "initializer_greedy" else "own_ppo" if role == "preflight" else role
            header, rows, final = self.episode(fixture_role)
            header.update(role=role, split=split)
            if selection is not None:
                header["selection"] = selection
            if selection == "sample":
                rows[0]["event"]["audit"]["decision"]["evaluation"]["log_probs"] = [math.log(.9), math.log(.1)]
            with self.subTest(role=role, split=split):
                outcome = verify_dynamic_episode(header, rows, final, self.cfg)
                self.assertEqual((outcome["role"], outcome["split"]), (role, split))
                self.assertEqual(outcome["compute_cost"]["evaluation_env_steps"], 2 if split == "test" else 0)
                self.assertEqual(outcome["actions"][0]["executed"], FRACTIONAL)

    def test_invalid_roles_greedy_selection_lineage_and_accounting_still_fail(self):
        mutations = (
            lambda h, r, f: h.update(role="unknown"),
            lambda h, r, f: h.update(split="qualification"),
            lambda h, r, f: h.update(representation="flat"),
            lambda h, r, f: h.update(selection="sample"),
            lambda h, r, f: h.update(source_id=""),
            lambda h, r, f: h.update(policy_sha256="b" * 64),
            lambda h, r, f: r[0]["event"]["audit"]["record"].update(trajectory_id="other"),
            lambda h, r, f: r[0]["event"]["audit"]["record"].update(state_token="other"),
            lambda h, r, f: r[0]["event"]["audit"]["decision"]["evaluation"].update(log_probs=[math.log(.5)] * 2),
            lambda h, r, f: r[0]["event"]["info"].update(cost=999),
            lambda h, r, f: f["patients"]["p2"].update(specimen_id="p1"),
        )
        for number, mutate in enumerate(mutations):
            header, rows, final = self.episode()
            mutate(header, rows, final)
            with self.subTest(mutation=number), self.assertRaises(ValueError):
                verify_dynamic_episode(header, rows, final, self.cfg)

    def test_partial_qualification_raw_reader_preserves_fractional_evidence(self):
        def qualification(header, rows, final):
            fractional_resources(header, rows, final)
            if header["role"] == "own_frozen":
                header.update(role="initializer_greedy", split="qualification")
            elif header["role"] == "r4":
                header["split"] = "qualification"

        index = write_bundle(self.root, self.cfg, self.streams, mutate=qualification)
        selected = [index[0], index[3]]
        files, outcomes = read_dynamic_raw_episodes(self.root, selected, self.cfg)
        self.assertEqual(len(files), 6)
        self.assertEqual([row["role"] for row in outcomes], ["initializer_greedy", "r4"])
        for entry, outcome in zip(selected, outcomes):
            self.assertEqual(outcome["raw_files"], entry)
            self.assertEqual(outcome["split"], "qualification")
            self.assertEqual(outcome["compute_cost"]["evaluation_env_steps"], 0)
            self.assertEqual(outcome["actions"][0]["executed"], FRACTIONAL)

    def test_full_fractional_bundle_is_versioned_lossless_and_read_only(self):
        index = write_bundle(self.root, self.cfg, self.streams, mutate=fractional_resources)
        before = copy.deepcopy((index, self.cfg, self.streams))
        raw_bytes = {path: path.read_bytes() for path in self.root.iterdir()}
        report = verify_dynamic_raw_bundle(self.root, index, self.cfg, self.streams)
        self.assertNotEqual(report["format"], "dynamic-candidate-raw-verification-v1")
        self.assertEqual(len(report["files"]), 90)
        self.assertEqual(len(report["outcomes"]), 30)
        self.assertEqual({row["role"] for row in report["outcomes"]}, set(legacy.CONTROLLERS))
        for row in report["outcomes"]:
            self.assertTrue(all(action["executed"] == FRACTIONAL for action in row["actions"]))
        self.assertEqual(report["index_sha256"], json_hash(index))
        self.assertFalse(report["grants_scientific_execution_authorization"])
        self.assertFalse(report["analysis"]["automatic_followon"])
        self.assertEqual((index, self.cfg, self.streams), before)
        self.assertEqual({path: path.read_bytes() for path in self.root.iterdir()}, raw_bytes)
        json.dumps(report, allow_nan=False)

    def test_invalid_recorded_resources_fail_episode_and_raw_readback(self):
        cases = [(channel, values) for channel in CHANNELS
                 for values in ([0], [0, 0, 0], [math.nan, 0], [math.inf, 0], [True, 0])]
        cases += [("specimen_transfers", [.25, -.25]), ("replenishment", [-5e-324, 0])]
        cases += [(channel, [1., -.5]) for channel in TRANSFERS]
        for channel, values in cases:
            def mutate(header, rows, final):
                fractional_resources(header, rows, final)
                rows[0]["event"]["info"][channel] = values

            with self.subTest(channel=channel, values=values):
                header, rows, final = self.episode()
                mutate(header, rows, final)
                with self.assertRaises(ValueError):
                    verify_dynamic_episode(header, rows, final, self.cfg)
                index = write_bundle(self.root, self.cfg, self.streams, mutate=mutate)
                with self.assertRaises(ValueError):
                    read_dynamic_raw_episodes(self.root, index[:1], self.cfg)

    def test_fractional_bundle_still_enforces_schedule_and_unique_lineage(self):
        index = write_bundle(self.root, self.cfg, self.streams, mutate=fractional_resources)
        for invalid in (index[:-1], index + [index[0]], [index[0]] + index[:-1]):
            with self.subTest(entries=len(invalid)), self.assertRaises(ValueError):
                verify_dynamic_raw_bundle(self.root, invalid, self.cfg, self.streams)

        def duplicate_lineage(header, rows, final):
            fractional_resources(header, rows, final)
            header["trajectory_id"] = "invented/duplicate"
            for row in rows:
                row["event"]["audit"]["record"]["trajectory_id"] = header["trajectory_id"]

        index = write_bundle(self.root, self.cfg, self.streams, mutate=duplicate_lineage)
        with self.assertRaisesRegex(ValueError, "lineage"):
            verify_dynamic_raw_bundle(self.root, index, self.cfg, self.streams)

    def test_fractional_raw_reader_rejects_tampered_bytes_and_duplicate_json_keys(self):
        index = write_bundle(self.root, self.cfg, self.streams, mutate=fractional_resources)
        record = index[0]["events"]
        path = self.root / record["path"]
        raw = path.read_bytes()
        path.write_bytes(raw + b" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            read_dynamic_raw_episodes(self.root, index[:1], self.cfg)
        path.write_bytes(raw)
        record = index[0]["header"]
        path = self.root / record["path"]
        raw = b'{"split":"training",' + path.read_bytes()[1:]
        path.write_bytes(raw)
        record.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError, "duplicate raw JSON key"):
            read_dynamic_raw_episodes(self.root, index[:1], self.cfg)


class SavedQualificationReadoutTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.streams = fixture_config()
        self.outcomes = []
        for block in self.cfg["blocks"]:
            for world, seed in enumerate(self.streams["environment"][str(block)]["qualification"]):
                for role in ("r4", "initializer_greedy"):
                    fixture_role = "own_frozen" if role == "initializer_greedy" else role
                    header, rows, final = raw_episode(block, fixture_role, world, seed)
                    header.update(role=role, split="qualification")
                    fractional_resources(header, rows, final)
                    self.outcomes.append(verify_dynamic_episode(header, rows, final, self.cfg))

    def assert_unscored_and_unauthorized(self, report):
        for block in report["blocks"]:
            self.assertEqual(block["initializer_scoring_on_r4_path"], "not_performed")
        for field in ("full_qualification_replayed", "clinical_noninferiority_claim",
                      "rl_performance_claim", "automatic_training_authorized"):
            self.assertIs(report[field], False)

    def test_passing_necessary_conditions_leave_unscored_r4_path_unresolved(self):
        before = copy.deepcopy((self.outcomes, self.cfg, self.streams))
        report = summarize_qualification(self.outcomes, self.cfg, self.streams)
        self.assertTrue(report["all_required_necessary_conditions_passed"])
        self.assertEqual(report["decision"], "full_qualification_unresolved")
        for block in report["blocks"]:
            self.assertTrue(block["recorded_own_path"]["necessary_condition_passed"])
            self.assertTrue(block["outcome_necessary_condition_passed"])
            self.assertEqual(block["recorded_own_path"]["agreement"], 1.)
        self.assert_unscored_and_unauthorized(report)
        self.assertEqual((self.outcomes, self.cfg, self.streams), before)

    def test_each_own_path_necessary_condition_can_preclude_continuation(self):
        for failure in ("overall_agreement", "no_multiclass_rows", "multiclass_agreement"):
            outcomes = copy.deepcopy(self.outcomes)
            own = [row for row in outcomes if row["role"] == "initializer_greedy"
                   and row["block"] == self.cfg["blocks"][0]]
            for row in own:
                if failure == "no_multiclass_rows":
                    row["support_counts"] = [1] * row["steps"]
                elif failure == "overall_agreement":
                    row["reference_class_choices"] -= 1
                    row["reference_corrections"][0]["different_class"] = True
                else:
                    # A high overall match rate cannot hide misses on multi-class rows.
                    row.update(steps=100, reference_class_choices=99,
                               support_counts=[1] * 98 + [2, 2],
                               reference_corrections=[{"different_class": False} for _ in range(99)]
                               + [{"different_class": True}])
            with self.subTest(failure=failure):
                report = summarize_qualification(outcomes, self.cfg, self.streams)
                block = report["blocks"][0]
                self.assertFalse(block["recorded_own_path"]["necessary_condition_passed"])
                self.assertTrue(block["outcome_necessary_condition_passed"])
                self.assertFalse(report["all_required_necessary_conditions_passed"])
                self.assertEqual(report["decision"], "continuation_precluded_by_saved_necessary_condition")
                if failure == "no_multiclass_rows":
                    self.assertIsNone(block["recorded_own_path"]["multiclass_agreement"])
                elif failure == "multiclass_agreement":
                    self.assertEqual(block["recorded_own_path"]["agreement"], .99)
                    self.assertEqual(block["recorded_own_path"]["multiclass_agreement"], .5)
                self.assert_unscored_and_unauthorized(report)

    def test_each_adverse_required_outcome_in_one_block_precludes_continuation(self):
        for metric, delta in (("cost", 1), ("losses", 1), ("terminal_active", 1), ("completions", -1)):
            outcomes = copy.deepcopy(self.outcomes)
            own = next(row for row in outcomes if row["role"] == "initializer_greedy")
            own[metric] += delta
            with self.subTest(metric=metric):
                report = summarize_qualification(outcomes, self.cfg, self.streams)
                self.assertTrue(report["blocks"][0]["recorded_own_path"]["necessary_condition_passed"])
                self.assertFalse(report["blocks"][0]["outcome_necessary_condition_passed"])
                self.assertTrue(all(row["outcome_necessary_condition_passed"] for row in report["blocks"][1:]))
                self.assertEqual(report["blocks"][0]["mean_differences"][metric], delta / 2)
                self.assertFalse(report["all_required_necessary_conditions_passed"])
                self.assertEqual(report["decision"], "continuation_precluded_by_saved_necessary_condition")
                self.assert_unscored_and_unauthorized(report)

    def test_exact_agreement_threshold_is_necessary_but_not_full_qualification(self):
        for row in self.outcomes:
            if row["role"] == "initializer_greedy":
                row.update(steps=20, reference_class_choices=19, support_counts=[2] * 20,
                           reference_corrections=[{"different_class": False} for _ in range(19)]
                           + [{"different_class": True}])
        report = summarize_qualification(self.outcomes, self.cfg, self.streams)
        self.assertTrue(report["all_required_necessary_conditions_passed"])
        self.assertEqual(report["decision"], "full_qualification_unresolved")
        for block in report["blocks"]:
            self.assertEqual(block["recorded_own_path"]["agreement"], .95)
            self.assertEqual(block["recorded_own_path"]["multiclass_agreement"], .95)
        self.assert_unscored_and_unauthorized(report)

    def test_missing_duplicate_unpaired_or_policy_changed_outcomes_fail_closed(self):
        for invalid in (self.outcomes[:-1], self.outcomes + [self.outcomes[0]],
                        [self.outcomes[0]] + self.outcomes[:-1]):
            with self.subTest(entries=len(invalid)), self.assertRaises(ValueError):
                summarize_qualification(invalid, self.cfg, self.streams)
        for field, value in (("seed", 123), ("initial_state_sha256", "a" * 64),
                             ("policy_sha256", "b" * 64)):
            outcomes = copy.deepcopy(self.outcomes)
            own = next(row for row in outcomes if row["role"] == "initializer_greedy")
            own[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_qualification(outcomes, self.cfg, self.streams)

    def test_zero_reference_cost_leaves_relative_change_undefined_and_no_authority(self):
        for row in self.outcomes:
            row["cost"] = 0.
        report = summarize_qualification(self.outcomes, self.cfg, self.streams)
        self.assertTrue(all(row["relative_cost_change_percent"] is None for row in report["blocks"]))
        self.assertEqual(report["decision"], "full_qualification_unresolved")
        self.assert_unscored_and_unauthorized(report)
        json.dumps(report, allow_nan=False)


class SavedReadoutIntegrationTests(unittest.TestCase):
    def test_complete_saved_only_boundary_and_preservation(self):
        from src.rl.dynamic_candidate_saved_readout import build_saved_readout
        from src.utils.research_archive import inventory

        cfg, streams = fixture_config()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "failed"
            root.mkdir()
            proposal = root / "proposal.json"
            proposal.write_text(json.dumps({"scientific_config": cfg, "streams": streams}))
            descriptors = []
            for block in cfg["blocks"]:
                descriptors.append((block, "preflight", "preflight", "prototype_preflight", 0))
                descriptors.extend((block, "demonstration", "r4", "demonstration", w)
                                   for w in range(cfg["initialization"]["demonstration_episodes_per_block"]))
                descriptors.extend((block, "qualification", role, "qualification", w)
                                   for role in ("r4", "initializer_greedy")
                                   for w in range(cfg["qualification"]["fresh_worlds_per_block"]))
            for block, split, role, stream, world in descriptors:
                fixture_role = "r4" if role == "r4" else "own_frozen"
                seed = streams["environment"][str(block)][stream][world]
                header, rows, final = raw_episode(block, fixture_role, world, seed)
                header.update(role=role, split=split, selection="reference" if role == "r4"
                              else "sample" if role == "preflight" else "greedy")
                fractional_resources(header, rows, final)
                folder = root / "payload/episodes" / f"{split}-{block}-{role}-{world}"
                folder.mkdir(parents=True)
                (folder / "header.json").write_text(json.dumps(header))
                (folder / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                (folder / "final_state.json").write_text(json.dumps(final))
            files = inventory(root)
            manifest = base / "preservation.json"
            manifest.write_text(json.dumps({"files": files, "file_count": len(files)}))
            report = build_saved_readout(root, proposal, manifest)
            self.assertEqual((report["episodes"], report["raw_rows"]), (39, 78))
            self.assertEqual(report["episodes_by_split"], {"demonstration": 24, "preflight": 3, "qualification": 12})
            self.assertTrue(report["source_unchanged"])
            self.assertEqual(report["qualification"]["decision"], "full_qualification_unresolved")
            for key in ("new_environment_steps", "new_optimizer_steps", "new_model_forwards", "checkpoint_loads"):
                self.assertEqual(report[key], 0)
            self.assertEqual(inventory(root), files)
            external = base / "unbound-proposal.json"
            external.write_bytes(proposal.read_bytes())
            with self.assertRaisesRegex(ValueError, "archived proposal"):
                build_saved_readout(root, external, manifest)
            proposal.write_text("{}")
            with self.assertRaisesRegex(ValueError, "preserved source inventory"):
                build_saved_readout(root, proposal, manifest)


if __name__ == "__main__":
    unittest.main()
