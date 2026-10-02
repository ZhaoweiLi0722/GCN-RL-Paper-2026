"""Invented, persisted 52+11-step JSON only; no science or optimizer calls."""

import copy
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.baselines.heuristics import heuristic_settings_for_policy
from src.rl.candidate_pilot_verification import CAUSES, json_hash
from src.rl.paired_cohort_comparison import (
    BOOTSTRAP, CONTROLLERS, LEARNED_ROLES, PRIMARY_CONTRASTS, SECONDARY_CONTRASTS,
    _analysis, read_paired_cohort_raw_episodes, verify_paired_cohort_bundle,
)
from src.rl.paired_cohort_resources import stream_manifest
from tests.test_cohort_bundle_verification import (
    ACTION, fixture_config as old_fixture_config, invented_episode as short_episode, save_episode,
)


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def no_scientific_execution():
    from src.rl.networks import torch

    targets = [f"src.env.{module}.{cls}.{method}"
               for module, cls in (("capacity_planning", "CapacityPlanningEnv"),
                                   ("patient_capacity_planning", "PatientConditionCapacityEnv"))
               for method in ("__init__", "reset", "step")]
    if torch is not None:
        targets += ["torch.load", "torch.nn.Module._call_impl", "torch.optim.Adam.step", "torch.optim.SGD.step"]
    with ExitStack() as stack:
        guards = [stack.enter_context(patch(target, side_effect=AssertionError("no scientific execution")))
                  for target in targets]
        yield
        for guard in guards:
            guard.assert_not_called()


def fixture_config():
    proposal = json.loads((ROOT / "experiments/configs/paired_cohort_improvement_20261002.json").read_text())
    old, _ = old_fixture_config()
    backend = {k: copy.deepcopy(old[k]) for k in ("blocks", "objective", "candidate_support")}
    backend["objective"]["horizon"] = 52
    backend["cohort_proposal"] = dict(enrollment_steps=52, patient_resolution_steps=8, accounting_steps=11)
    streams = stream_manifest(proposal, {str(b): 100 + b for b in proposal["blocks"]})
    return proposal, streams, backend


def invented_episode(config, streams, backend, block=60, role="paired_cost", world=0, cost=99.5,
                     changed=True, executed_changed=False):
    # The reusable factory creates synthetic primitives with the NEW role from
    # the outset. No old recorded episode is loaded or relabeled.
    small = copy.deepcopy(backend)
    small["objective"]["horizon"] = 2
    small["cohort_proposal"] = dict(enrollment_steps=2, patient_resolution_steps=2, accounting_steps=3)
    (header, rows, prefix), (tail_header, tail, final) = short_episode(
        small, streams, block, role, world, cost=cost)
    rows = [copy.deepcopy(rows[0]) for _ in range(51)] + [copy.deepcopy(rows[-1])]
    for state in (header["initial_state"], prefix, final):
        for patient in state["patients"].values():
            patient["age"] = 2 if patient["status"] == "delivered" else 1
        state["scalars"]["cumulative_turnaround_time"] = 0. if state is header["initial_state"] else 2.
    prefix["scalars"]["t"], final["scalars"]["t"] = 52, 63
    anchor = tail_header["anchor_config"]
    anchor["episode_horizon"] = 52
    header["session_manifest"]["producer_definition"] = json_hash(dict(
        anchor_config=anchor, settings=asdict(heuristic_settings_for_policy("mdl2"))))
    previous = json_hash(header["initial_state"])
    for t, row in enumerate(rows):
        audit, info = row["event"]["audit"], row["event"]["info"]
        token = json_hash(prefix) if t == 51 else json_hash(dict(invented_prefix=t + 1))
        audit["record"].update(step_index=t, state_token=previous, next_state_token=token,
            state=[t], next_state=[t + 1], terminated=t == 51, raw_reward=-cost / 52)
        info.update(cost=cost / 52, base_cost=cost / 52, reagent_purchase_cost=cost / 52,
                    demand=[int(t == 0), 0], average_turnaround_time=2. if t == 51 else 0.)
        if changed and role == "paired_cost":
            audit["record"]["action"] = [0.] * 8
            audit["decision"]["choice"].update(class_index=0, submitted_request=[0.] * 8)
            audit["decision"]["evaluation"]["log_probs"] = [math.log(.9), math.log(.1)]
            info.update(specimen_requested_integer_net=[0, 0], blocked_specimen_requests=0,
                        blocked_specimen_inbound_requests=0, blocked_specimen_outbound_requests=0)
        if executed_changed:
            info["capacity_transfers"] = [.25, -.25]
        previous = token
    tail = tail[:2] + [copy.deepcopy(tail[-1]) for _ in range(9)]
    closed = copy.deepcopy(prefix)
    closed["arrays"].update(demand=[0., 0.], demand_forecast=[0., 0.])
    closed["scalars"]["demand_forecast_error"] = 0.
    previous = json_hash(closed)
    for i, row in enumerate(tail, 1):
        token = json_hash(final) if i == 11 else json_hash(dict(invented_tail=i))
        row.update(index=i, before_state_sha256=previous, after_state_sha256=token, accounting_done=i == 11)
        row["info"].update(specimen_requested_integer_net=[0, 0], blocked_specimen_inbound_requests=0,
                           blocked_specimen_outbound_requests=0)
        previous = token
    tail_header.update(contract=backend["cohort_proposal"] | dict(
        followup_rule="full_mdl2_until_resolution_then_no_new_commitments"), prefix_final_sha256=json_hash(prefix))
    return (header, rows, prefix), (tail_header, tail, final)


class RawComparisonTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config, self.streams, self.backend = fixture_config()
        action = patch("src.rl.cohort_bundle_verification.facility_net_action_from_state", return_value=np.asarray(ACTION))
        self.action = action.start()
        self.addCleanup(action.stop)
        guard = patch("src.env.capacity_planning.CapacityPlanningEnv.__init__", side_effect=AssertionError("no environment"))
        self.env = guard.start()
        self.addCleanup(guard.stop)

    def read(self, values=None, **kwargs):
        values = values or invented_episode(self.config, self.streams, self.backend, **kwargs)
        entry = save_episode(self.root, "single", values)
        return read_paired_cohort_raw_episodes(self.root, [entry], self.config, self.streams, backend_config=self.backend)

    def test_full63_raw_primitives_clinical_counts_and_continuous_resources(self):
        files, rows = self.read()
        row = rows[0]
        self.assertEqual(len(files), 6)
        self.assertEqual((row["steps"], len(row["actions"]), len(row["window"]["actions"]), len(row["tail"]["actions"])),
                         (63, 63, 52, 11))
        self.assertEqual([a["step_index"] for a in row["actions"]], list(range(63)))
        self.assertEqual((row["cost"], row["prefix_cost"], row["tail_cost"]), (110.5, 99.5, 11.))
        self.assertEqual(sum(row["components"].values()), row["cost"])
        self.assertEqual((row["losses"], row["completions"], row["terminal_active"], row["enrolled"]), (2, 1, 0, 3))
        self.assertEqual(sum(row["loss_causes"].values()), 2)
        self.assertEqual(row["waiting_patient_steps"], 154)
        self.assertEqual((row["average_turnaround_time"], row["turnaround_sum"]), (2., 2.))
        self.assertEqual(row["actions"][0]["executed"]["capacity_transfers"], [.5, -.5])
        self.assertEqual(row["tail"]["retained_reagents"], [1.75, 2.])
        self.assertEqual(row["tail"]["resource_flow"][0]["reagents"]["arrival_clipped_overflow"], [.25, 0.])
        self.assertFalse(row["tail"]["terminal_stock_valued"])
        self.env.assert_not_called()
        json.dumps(rows, allow_nan=False)

    def test_native_roles_are_preserved_and_old_roles_rejected(self):
        for role in CONTROLLERS:
            with self.subTest(role=role):
                self.assertEqual(self.read(role=role)[1][0]["role"], role)
        for role in ("cohort_ppo", "window_ppo", "current_ppo", "time_baseline_ppo", "own_ppo"):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "native evaluation role"):
                self.read(role=role)

    def test_all_four_learned_roles_use_canonical_greedy_and_lexical_ties(self):
        for role in LEARNED_ROLES:
            for probabilities in ([math.log(.9), math.log(.1)], [math.log(.5)] * 2):
                values = invented_episode(self.config, self.streams, self.backend, role=role, changed=False)
                values[0][1][0]["event"]["audit"]["decision"]["evaluation"]["log_probs"] = probabilities
                with self.subTest(role=role), self.assertRaisesRegex(ValueError, "greedy"):
                    self.read(values)
        values = invented_episode(self.config, self.streams, self.backend)
        values[0][1][0]["event"]["audit"]["decision"]["evaluation"]["log_probs"] = [math.log(.5)] * 2
        self.assertEqual(self.read(values)[1][0]["actions"][0]["requested"], [0.] * 8)

    def test_bad_raw_cost_patient_resource_chain_and_turnaround_fail(self):
        mutations = [
            lambda v: v[0][1][0]["event"]["info"].update(patient_loss_cost=1.),
            lambda v: v[1][1][0]["info"].update(patient_loss_cost=1.),
            lambda v: v[1][1][0]["info"].update(patients_lost_waiting_ineligible=[0, 0]),
            lambda v: v[1][2]["scalars"].update(cumulative_lost=99),
            lambda v: v[1][2]["scalars"].update(_episode_seed=123),
            lambda v: v[1][1][0].update(before_state_sha256="f" * 64),
            lambda v: v[1][1][0]["resources_after"]["reagents"].__setitem__(0, 1.8),
            lambda v: v[0][1][0]["event"]["info"].update(capacity_transfers=[.5, -.25]),
            lambda v: v[1][1][0]["info"].update(specimen_requested_integer_net=[1, -1]),
            lambda v: v[1][1][-1]["info"].update(average_turnaround_time=99.),
            lambda v: v[0][1][-1]["event"]["info"].update(average_turnaround_time=99.),
            lambda v: v[1][1][0]["info"].update(cost=float("nan")),
        ]
        for i, mutate in enumerate(mutations):
            values = invented_episode(self.config, self.streams, self.backend)
            mutate(values)
            with self.subTest(i=i), self.assertRaises(ValueError):
                self.read(values)

    def test_partial_reordered_extra_and_foreign_tail_rejected(self):
        for mutate in (lambda v: v[0][1].pop(), lambda v: v[1][1].pop(),
                       lambda v: v[1][1].reverse(), lambda v: v[1][1].append(v[1][1][-1]),
                       lambda v: v[1][0].update(role="saved_cohort_ppo"),
                       lambda v: v[1][0]["contract"].update(accounting_steps=10)):
            values = invented_episode(self.config, self.streams, self.backend)
            mutate(values)
            with self.assertRaises(ValueError):
                self.read(values)

    def test_exact_decimal_schedule_fresh_analysis_seed_and_config_comparisons(self):
        original = copy.deepcopy(self.streams)
        mutations = [lambda s: s["environment"]["60"]["test"].__setitem__(0, "01"),
                     lambda s: s["environment"]["60"]["test"].__setitem__(0, int(s["environment"]["60"]["test"][0])),
                     lambda s: s.update(bootstrap="1"),
                     lambda s: s.update(namespace="old-campaign"),
                     lambda s: s["conditional_future"].pop(next(iter(s["conditional_future"])))]
        for mutate in mutations:
            self.streams = copy.deepcopy(original)
            mutate(self.streams)
            with self.assertRaises(ValueError):
                read_paired_cohort_raw_episodes(self.root, [], self.config, self.streams, backend_config=self.backend)
        self.streams = original
        values = invented_episode(self.config, self.streams, self.backend)
        values[0][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed"):
            self.read(values)
        self.config["primary_contrasts"] = ["paired_cost-minus-r4", "paired_cost-minus-bc_continue"]
        with self.assertRaisesRegex(ValueError, "contrasts"):
            self.read()

    def test_protocol_dimensions_and_bootstrap_cannot_be_overridden(self):
        for key, value in (("enrollment_steps", 2), ("economic_endpoint", 62), ("test_worlds_per_block", 11)):
            cfg = copy.deepcopy(self.config)
            cfg[key] = value
            with self.assertRaises(ValueError):
                read_paired_cohort_raw_episodes(self.root, [], cfg, self.streams, backend_config=self.backend)
        self.config["evaluation"] = dict(bootstrap=BOOTSTRAP | dict(draws=99))
        with self.assertRaisesRegex(ValueError, "10000"):
            self.read()

    def test_source_hash_duplicate_slot_path_alias_and_symlink(self):
        values = invented_episode(self.config, self.streams, self.backend)
        entry = save_episode(self.root, "a", values)
        path = self.root / entry["tail"]["events"]["path"]
        raw = path.read_bytes()
        path.write_bytes(raw + b" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            read_paired_cohort_raw_episodes(self.root, [entry], self.config, self.streams, backend_config=self.backend)
        path.write_bytes(raw)
        other = save_episode(self.root, "b", copy.deepcopy(values))
        with self.assertRaisesRegex(ValueError, "duplicate trajectory"):
            read_paired_cohort_raw_episodes(self.root, [entry, other], self.config, self.streams, backend_config=self.backend)
        alias = copy.deepcopy(entry)
        alias["tail"]["events"]["path"] = "./" + alias["tail"]["events"]["path"]
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            read_paired_cohort_raw_episodes(self.root, [alias], self.config, self.streams, backend_config=self.backend)
        path.unlink()
        path.symlink_to(self.root / other["tail"]["events"]["path"])
        with self.assertRaisesRegex(ValueError, "redirected"):
            read_paired_cohort_raw_episodes(self.root, [entry], self.config, self.streams, backend_config=self.backend)

    def test_raw_role_and_metadata_are_never_modified(self):
        values = invented_episode(self.config, self.streams, self.backend, role="saved_cohort_ppo")
        entry = save_episode(self.root, "single", values)
        before = copy.deepcopy((entry, self.config, self.streams, self.backend))
        files = [self.root / r["path"] for part in entry.values() for r in part.values()]
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
        read_paired_cohort_raw_episodes(self.root, [entry], self.config, self.streams, backend_config=self.backend)
        self.assertEqual(before, (entry, self.config, self.streams, self.backend))
        self.assertEqual(hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in files])

    def test_reader_never_loads_models_forwards_steps_or_updates(self):
        with no_scientific_execution():
            self.read()

    def test_duplicate_json_keys_are_rejected_even_with_matching_byte_hash(self):
        values = invented_episode(self.config, self.streams, self.backend)
        entry = save_episode(self.root, "single", values)
        record = entry["prefix"]["header"]
        path = self.root / record["path"]
        raw = path.read_bytes()[:-1] + b', "role": "paired_cost"}'
        path.write_bytes(raw)
        record.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError, "duplicate raw JSON key"):
            read_paired_cohort_raw_episodes(self.root, [entry], self.config, self.streams, backend_config=self.backend)


class CompleteComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.config, cls.streams, cls.backend = fixture_config()
        # Carry old role/schedule/analysis metadata through the actual runner
        # signature. It must never be used to validate the new role records.
        cls.inherited, _ = old_fixture_config()
        cls.inherited["objective"] = copy.deepcopy(cls.backend["objective"])
        cls.inherited["cohort_proposal"].update(cls.backend["cohort_proposal"])
        cls.index = []
        for b in cls.config["blocks"]:
            for role in CONTROLLERS:
                for w in range(12):
                    values = invented_episode(cls.config, cls.streams, cls.backend, b, role, w,
                        cost=99.5 if role == "paired_cost" else 100., executed_changed=role == "paired_cost")
                    cls.index.append(save_episode(cls.root, f"{b}/{role}/{w}", values))
        with patch("src.rl.cohort_bundle_verification.facility_net_action_from_state", return_value=np.asarray(ACTION)), \
                no_scientific_execution():
            with patch("src.rl.cohort_bundle_verification._schedule", side_effect=AssertionError("no old schedule")), \
                    patch("src.rl.cohort_bundle_verification._validate_plan", side_effect=AssertionError("no old plan")):
                cls.report = verify_paired_cohort_bundle(cls.root, cls.index, cls.config, cls.inherited, cls.streams)

    def analyze(self, rows=None):
        return _analysis(self.report["outcomes"] if rows is None else rows, self.config, self.streams, int(self.streams["bootstrap"]))

    def test_complete_native_matrix_and_fixed_nomination_screen(self):
        report, analysis = self.report, self.report["analysis"]
        self.assertEqual(report["inventory"], {"test": 216})
        self.assertEqual(len(report["files"]), 1296)
        self.assertEqual(analysis["primary_contrasts"], list(PRIMARY_CONTRASTS))
        self.assertEqual(analysis["secondary_contrasts"], list(SECONDARY_CONTRASTS))
        self.assertEqual({r["role"] for r in report["outcomes"]}, set(CONTROLLERS))
        self.assertTrue(analysis["promotion_screen_met"])
        self.assertEqual(analysis["decision"], "nominate_independent_confirmation")
        self.assertFalse(analysis["practical_effect_threshold_applied"])
        self.assertEqual(analysis["bootstrap"]["draws"], 10000)
        self.assertEqual(analysis["bootstrap"]["seed"], int(self.streams["bootstrap"]))
        self.assertTrue(analysis["bootstrap"]["descriptive_low_precision_three_blocks"])
        for contrast in analysis["contrasts"]:
            self.assertEqual(contrast["hierarchical_paired_ci95"]["cost"], [-.5, -.5])
        first = analysis["contrasts"][0]
        self.assertEqual(first["prefix_requested_changed_steps"], 52 * 36)
        self.assertEqual(first["prefix_executed_changed_steps"], 52 * 36)
        self.assertEqual(first["tail_requested_changed_steps"], 0)
        self.assertEqual(first["tail_executed_changed_steps"], 0)
        self.assertEqual(report["compute_accounting"]["optimizer_calls"], 0)
        self.assertFalse(report["grants_scientific_execution_authorization"])
        self.assertFalse(report["scenario_seed_binding"]["historical_seed_freshness_verified_by_reader"])
        self.assertEqual(analysis, self.analyze(report["outcomes"][::-1]))
        json.dumps(report, allow_nan=False)

    def test_no_secondary_can_rescue_either_primary_or_one_bad_block(self):
        for reference in ("own_frozen", "bc_continue"):
            rows = copy.deepcopy(self.report["outcomes"])
            for row in rows:
                if row["block"] == 60 and row["role"] == reference:
                    row["cost"] = 90.
            analysis = self.analyze(rows)
            self.assertFalse(analysis["primary_cost_screen_met"])
            self.assertFalse(analysis["promotion_screen_met"])
            self.assertTrue(all(c["equal_block_mean_differences"]["cost"] < 0 for c in analysis["contrasts"][2:]))

    def test_any_primary_loss_increase_fails_but_waiting_and_expiry_stay_visible(self):
        rows = copy.deepcopy(self.report["outcomes"])
        for row in rows:
            if row["role"] == "paired_cost" and row["block"] == 61:
                row["waiting_patient_steps"] += 3
                row["expiry_losses"] += 1
        report = self.analyze(rows)
        self.assertTrue(report["promotion_screen_met"])
        self.assertEqual({v["metric"] for v in report["contrasts"][0]["adverse_patient_block_directions"]},
                         {"waiting_patient_steps", "expiry_losses"})
        for row in rows:
            if row["role"] == "paired_cost" and row["block"] == 61:
                row["losses"] += 1
        report = self.analyze(rows)
        self.assertTrue(report["primary_cost_screen_met"])
        self.assertFalse(report["observed_loss_direction_screen_met"])
        self.assertFalse(report["promotion_screen_met"])

    def test_missing_unpaired_or_changed_policy_rejected(self):
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.analyze(self.report["outcomes"][:-1])
        for field, value, message in (("initial_state_sha256", "f" * 64, "paired starts"),
                                      ("policy_sha256", "f" * 64, "policy changed")):
            rows = copy.deepcopy(self.report["outcomes"])
            rows[0][field] = value
            with self.assertRaisesRegex(ValueError, message):
                self.analyze(rows)

    def test_requested_and_executed_changes_are_separate_and_unchanged_closes(self):
        rows = copy.deepcopy(self.report["outcomes"])
        lookup = {(r["block"], r["role"], r["world_index"]): r for r in rows}
        for row in rows:
            if row["role"] == "paired_cost":
                reference = lookup[row["block"], "own_frozen", row["world_index"]]
                for a, b in zip(row["actions"], reference["actions"]):
                    a["executed"] = copy.deepcopy(b["executed"])
        analysis = self.analyze(rows)
        self.assertGreater(analysis["contrasts"][0]["requested_changed_steps"], 0)
        self.assertEqual(analysis["contrasts"][0]["executed_changed_steps"], 0)
        for row in rows:
            if row["role"] == "paired_cost":
                reference = lookup[row["block"], "own_frozen", row["world_index"]]
                row["actions"] = copy.deepcopy(reference["actions"])
                row["window"]["actions"] = copy.deepcopy(reference["window"]["actions"])
                row["cost"] = reference["cost"]
        analysis = self.analyze(rows)
        self.assertTrue(analysis["greedy_requests_unchanged_from_own_frozen"])
        self.assertFalse(analysis["promotion_screen_met"])
        self.assertEqual(analysis["decision"], "close_one_shot_mechanism")

    def test_bootstrap_resamples_blocks_then_worlds_with_one_shared_design(self):
        rows = copy.deepcopy(self.report["outcomes"])
        delta = np.asarray([[float(100 * b + w) for w in range(12)] for b in range(3)])
        for row in rows:
            if row["role"] == "paired_cost":
                row["cost"] = 111. + delta[row["block"] - 60, row["world_index"]]
        analysis = self.analyze(rows)
        rng = np.random.default_rng(int(self.streams["bootstrap"]))
        blocks = rng.integers(0, 3, size=(10000, 3))
        worlds = rng.integers(0, 12, size=(10000, 3, 12))
        expected = np.quantile(delta[blocks[..., None], worlds].mean(axis=2).mean(axis=1), [.025, .975]).tolist()
        self.assertEqual(analysis["contrasts"][0]["hierarchical_paired_ci95"]["cost"], expected)
        self.assertEqual(analysis["bootstrap"]["resample_design_sha256"], json_hash([blocks.tolist(), worlds.tolist()]))
        self.assertEqual(analysis["contrasts"][0]["equal_block_mean_differences"]["cost"], float(delta.mean()))

    def test_zero_relative_denominator_is_undefined_not_infinite(self):
        rows = copy.deepcopy(self.report["outcomes"])
        for row in rows:
            if row["role"] == "own_frozen":
                row["cost"] = 0.
        contrast = self.analyze(rows)["contrasts"][0]
        self.assertIsNone(contrast["relative_cost_change_ci95"])
        self.assertIsNone(contrast["equal_block_relative_cost_change_percent"])
        self.assertEqual(contrast["undefined_relative_bootstrap_draws"], 10000)


if __name__ == "__main__":
    unittest.main()
