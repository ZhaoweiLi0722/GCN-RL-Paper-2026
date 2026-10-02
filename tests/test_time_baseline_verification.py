"""Invented JSON registries only: no patient environment, model or optimizer."""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.rl.candidate_pilot_verification import json_hash
from src.rl.time_baseline_plan import time_baseline_stream_manifest
from src.rl.time_baseline_verification import (
    CONTROLLERS, LEARNED_ROLES, PRIMARY, SECONDARIES, TRAINING_ROLES,
    read_time_baseline_raw_episodes, verify_time_baseline_raw_bundle,
)
from tests.test_dynamic_candidate_verification import fixture_config as old_fixture_config, raw_episode


ROOT = Path(__file__).resolve().parents[1]


def fixture_config():
    config, _ = old_fixture_config(worlds=12)
    proposal = json.loads((ROOT / "specs/2026-10-02-time-baseline-comparison/proposal.json").read_text())
    config["time_baseline_proposal"] = proposal
    config["final_controllers"] = list(CONTROLLERS)
    config["evaluation"].update(controllers_per_block=6, total_episodes=216,
                                environment_steps=432, primary=PRIMARY,
                                secondary_contrasts=list(SECONDARIES))
    config["evaluation"]["bootstrap"]["draws"] = 10000
    return config, time_baseline_stream_manifest(proposal)


def episode(config, streams, block=60, role="time_baseline_ppo", world=0,
            split="test", cost=100, changed=False):
    # Start with an invented registry factory, supplying the actual new role
    # before serialization. No recorded evidence or historical output is used.
    seed = int(streams["environment"][str(block)][split][world])
    header, rows, final = raw_episode(block, role, world, seed, cost)
    source = streams["namespace"]
    trajectory = f"invented/{split}/{block}/{role}/{world}"
    header.update(source_id=source, trajectory_id=trajectory, split=split,
                  representation="graph" if role in LEARNED_ROLES else "reference")
    if split != "test" and role in TRAINING_ROLES:
        header["selection"] = "sample"
    for row in rows:
        audit, info = row["event"]["audit"], row["event"]["info"]
        audit["record"].update(source_id=source, trajectory_id=trajectory)
        info.update(capacity_transfers=[.5, -.5], reagent_transfers=[-.125, .125],
                    replenishment=[1.25, 0.])
        if changed:
            audit["record"]["action"] = [0.] * 8
            audit["decision"]["choice"].update(class_index=0, submitted_request=[0.] * 8)
            audit["decision"]["evaluation"]["log_probs"] = [math.log(.9), math.log(.1)]
            info.update(specimen_requested_integer_net=[0, 0], blocked_specimen_requests=0,
                        blocked_specimen_inbound_requests=0, blocked_specimen_outbound_requests=0)
    return header, rows, final


def save_episode(root, prefix, header, rows, final):
    def save(suffix, value, jsonl=False):
        raw = ("".join(json.dumps(row) + "\n" for row in value) if jsonl else json.dumps(value)).encode()
        name = prefix + suffix
        (root / name).write_bytes(raw)
        return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return {"header": save("-header.json", header), "events": save("-events.jsonl", rows, True),
            "final_state": save("-final.json", final)}


def write_bundle(root, config, streams, *, costs=None, mutate=None, changed=True):
    index = []
    for block in config["blocks"]:
        for world in range(12):
            for role in CONTROLLERS:
                cost = costs(block, role, world) if costs else {
                    "time_baseline_ppo": 90, "current_ppo": 95, "bc_continue": 98}.get(role, 100)
                values = episode(config, streams, block, role, world, cost=cost,
                                 changed=changed and role == "time_baseline_ppo")
                if mutate:
                    mutate(*values)
                index.append(save_episode(root, f"{block}-{role}-{world}", *values))
    return index


class TimeBaselineVerificationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.config, self.streams = fixture_config()

    def verify(self, **kwargs):
        index = write_bundle(self.root, self.config, self.streams, **kwargs)
        return verify_time_baseline_raw_bundle(self.root, index, self.config, self.streams)

    def read_one(self, values=None, **kwargs):
        values = values or episode(self.config, self.streams, **kwargs)
        index = [save_episode(self.root, "single", *values)]
        return read_time_baseline_raw_episodes(self.root, index, self.config)

    def test_full_matrix_contrasts_costs_resource_precision_and_no_claims(self):
        report = self.verify()
        self.assertEqual(len(report["outcomes"]), 216)
        self.assertEqual(len(report["files"]), 648)
        analysis = report["analysis"]
        self.assertEqual(analysis["primary"], "time_baseline_ppo_minus_own_frozen")
        self.assertEqual([r["name"] for r in analysis["contrasts"]][1:], list(SECONDARIES))
        self.assertEqual([r["equal_block_mean_differences"]["cost"] for r in analysis["contrasts"]],
                         [-10., -5., -8., -10., -10., 0.])
        primary = analysis["contrasts"][0]
        self.assertEqual(primary["hierarchical_paired_ci95"]["cost"], [-10., -10.])
        self.assertEqual(primary["relative_cost_change_ci95"], [-10., -10.])
        self.assertEqual(primary["requested_changed_steps"], 72)
        self.assertTrue(analysis["primary_cost_screen_met"])
        self.assertEqual(analysis["decision"], "promising_development_signal")
        self.assertFalse(analysis["baseline_route"]["close"])
        row = report["outcomes"][0]
        self.assertEqual(row["actions"][0]["requested"][0], .0125)
        self.assertEqual(row["actions"][0]["executed"]["capacity_transfers"], [.5, -.5])
        self.assertEqual(row["actions"][0]["executed"]["reagent_transfers"], [-.125, .125])
        self.assertEqual(row["actions"][0]["executed"]["replenishment"], [1.25, 0.])
        self.assertEqual(row["waiting_patient_steps"], 4)
        self.assertEqual(row["terminal_waiting"], 1)
        self.assertEqual(row["target_receipt_diagnostics"]["status"], "not_measured")
        self.assertEqual(analysis["target_receipt_diagnostics"]["status"], "not_measured")
        for key in ("clinical_noninferiority_claim", "clinical_superiority_claim", "deployment_adaptation_claim",
                    "clean_ddpg_superiority_claim", "isolated_graph_contribution_claim", "automatic_followon",
                    "event_level_crn_equality_claim"):
            self.assertFalse(analysis[key])
        self.assertFalse(report["grants_scientific_execution_authorization"])
        self.assertFalse(analysis["baseline_route"]["reward_training_authorized"])
        self.assertFalse(analysis["baseline_route"]["additional_baseline_attempt_authorized"])
        self.assertEqual(report["config_sha256"], json_hash(self.config))
        self.assertGreater(row["seed"], 2**64)
        self.assertEqual(report["scenario_seed_binding"]["source_id"], self.streams["namespace"])
        json.dumps(report, allow_nan=False)

    def test_all_four_learned_test_roles_enforce_greedy_including_lexical_ties(self):
        for role in LEARNED_ROLES:
            for probabilities in ([math.log(.9), math.log(.1)], [math.log(.5)] * 2):
                values = episode(self.config, self.streams, role=role)
                values[1][0]["event"]["audit"]["decision"]["evaluation"]["log_probs"] = probabilities
                with self.subTest(role=role, probabilities=probabilities), self.assertRaisesRegex(ValueError, "greedy"):
                    self.read_one(values)
        row = self.read_one(role="time_baseline_ppo", changed=True)[1][0]
        self.assertEqual(row["reference_corrections"][0]["request_l1"], .025)

    def test_partial_preflight_training_roles_accept_sampling_not_greedy(self):
        for split, roles in (("preflight", CONTROLLERS), ("training", TRAINING_ROLES)):
            for role in roles:
                values = episode(self.config, self.streams, role=role, split=split)
                if role in TRAINING_ROLES:
                    for row in values[1]:
                        row["event"]["audit"]["decision"]["evaluation"]["log_probs"] = [math.log(.9), math.log(.1)]
                files, outcomes = self.read_one(values)
                self.assertEqual(len(files), 3)
                self.assertEqual((outcomes[0]["role"], outcomes[0]["split"]), (role, split))
                self.assertEqual(outcomes[0]["compute_cost"]["evaluation_env_steps"], 0)

    def test_no_extra_roles_splits_or_selection_modes(self):
        for role in ("r4", "full_mdl2", "own_frozen", "own_ppo", "initializer_greedy"):
            values = episode(self.config, self.streams, role=role, split="training")
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "controller/split"):
                self.read_one(values)
        for split in ("validation", "qualification", "holdout", "demonstration"):
            values = episode(self.config, self.streams)
            values[0]["split"] = split
            with self.assertRaisesRegex(ValueError, "controller/split"):
                self.read_one(values)
        for role in LEARNED_ROLES:
            values = episode(self.config, self.streams, role=role)
            values[0]["selection"] = "sample"
            with self.assertRaisesRegex(ValueError, "selection mode"):
                self.read_one(values)

    def test_source_namespace_seed_index_and_representation_are_bound(self):
        for kind in ("source", "seed", "world", "block", "representation", "empty_trajectory", "float_seed"):
            h, rows, final = episode(self.config, self.streams)
            if kind == "source":
                h["source_id"] = "foreign"
                for row in rows:
                    row["event"]["audit"]["record"]["source_id"] = "foreign"
            elif kind in ("seed", "float_seed"):
                h["seed"] = h["seed"] + 1 if kind == "seed" else float(h["seed"])
                h["initial_state"]["scalars"]["_episode_seed"] = h["seed"]
                final["scalars"]["_episode_seed"] = h["seed"]
                rows[0]["event"]["audit"]["record"]["state_token"] = json_hash(h["initial_state"])
                rows[-1]["event"]["audit"]["record"]["next_state_token"] = json_hash(final)
            elif kind == "empty_trajectory":
                h["trajectory_id"] = ""
                for row in rows:
                    row["event"]["audit"]["record"]["trajectory_id"] = ""
            else:
                field, value = {"world": ("world_index", -1), "block": ("block", 99),
                                "representation": ("representation", "flat")}[kind]
                h[field] = value
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.read_one((h, rows, final))

    def test_continuous_conservation_but_integer_patients_and_specimens(self):
        corrupt = (("capacity_transfers", [.5, -.4]), ("reagent_transfers", [-.125, .2]),
                   ("replenishment", [-.1, 0]), ("specimen_transfers", [.5, -.5]),
                   ("specimen_requested_integer_net", [1.5, -1.5]), ("patients_completed", [.5, 0]),
                   ("waiting_patients", [1.5, 0]), ("patients_lost", [0., .1]),
                   ("replenishment", [float("inf"), 0]), ("capacity_transfers", [float("nan"), 0]))
        for field, value in corrupt:
            values = episode(self.config, self.streams)
            values[1][0]["event"]["info"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.read_one(values)
        values = episode(self.config, self.streams)
        values[1][0]["event"]["info"]["capacity_transfers"] = [.1, -.1 + 1e-10]
        self.assertEqual(self.read_one(values)[1][0]["actions"][0]["executed"]["capacity_transfers"], [.1, -.1 + 1e-10])

    def test_raw_cost_identity_original_float64_and_queue_mismatch_rejected(self):
        for kind in ("cost", "identity", "request", "queue", "partial", "seconds", "terminal_reward"):
            values = episode(self.config, self.streams)
            h, rows, final = values
            if kind == "cost":
                rows[0]["event"]["info"]["cost"] += 1
            elif kind == "identity":
                final["patients"]["p2"]["specimen_id"] = "p0"
            elif kind == "request":
                rows[0]["event"]["audit"]["record"]["action"][0] = float(np.float32(.0125))
            elif kind == "queue":
                rows[-1]["event"]["info"]["waiting_patients"] = [0, 0]
            elif kind == "partial":
                rows.pop()
            elif kind == "seconds":
                rows[0]["inference_seconds"] = -1
            else:
                rows[-1]["event"]["audit"]["terminal_cost_added"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.read_one(values)

    def test_full_anchor_is_not_restricted_and_r4_need_not_be_greedy(self):
        values = episode(self.config, self.streams, role="r4")
        for row in values[1]:
            row["event"]["audit"]["decision"]["evaluation"]["log_probs"] = [math.log(.9), math.log(.1)]
        self.assertEqual(self.read_one(values)[1][0]["role"], "r4")
        values = episode(self.config, self.streams, role="full_mdl2")
        self.assertEqual(self.read_one(values)[1][0]["actions"][0]["requested"][-2:], [.5, .5])
        for row in values[1]:
            evaluation = row["event"]["audit"]["decision"]["evaluation"]
            bank = evaluation["candidates"]
            bank["requests"][0] = [.0125, -.0125, 0., 0., 0., 0., .5, .5]
            bank["class_keys"].append([2, -2, 0., 0., 0., 0., .5, .5])
            bank["request_to_class"] = [1, 0]
            evaluation["log_probs"] = [math.log(.9), math.log(.1)]
        with self.assertRaisesRegex(ValueError, "full-anchor"):
            self.read_one(values)

    def test_missing_duplicate_extra_non_test_and_rehashed_duplicate_slots(self):
        index = write_bundle(self.root, self.config, self.streams)
        for candidate in (index[:-1], index + [index[0]], [index[0]] + index[:-1]):
            with self.assertRaises(ValueError):
                verify_time_baseline_raw_bundle(self.root, candidate, self.config, self.streams)
        duplicate = save_episode(self.root, "same-slot-new-files", *episode(self.config, self.streams, role="own_frozen"))
        with self.assertRaisesRegex(ValueError, "duplicate trajectory or episode slot"):
            read_time_baseline_raw_episodes(self.root, [index[0], duplicate], self.config)
        index[0] = save_episode(self.root, "preflight", *episode(self.config, self.streams, role="own_frozen", split="preflight"))
        with self.assertRaisesRegex(ValueError, "unexpected evaluation"):
            verify_time_baseline_raw_bundle(self.root, index, self.config, self.streams)

    def test_changed_policy_and_unpaired_initial_state_rejected(self):
        def changed_policy(h, rows, final):
            if h["world_index"] == 0:
                h["policy_sha256"] = "b" * 64
                for row in rows:
                    row["event"]["audit"]["decision"]["evaluation"]["behavior_sha256"] = "b" * 64
        def unpaired(h, rows, final):
            if h["role"] == "time_baseline_ppo":
                h["initial_state"]["invented_public_marker"] = 1
                rows[0]["event"]["audit"]["record"]["state_token"] = json_hash(h["initial_state"])
        for mutate, message in ((changed_policy, "policy changed"), (unpaired, "exact initial state")):
            with self.assertRaisesRegex(ValueError, message):
                self.verify(mutate=mutate)

    def test_config_thresholds_names_and_draws_cannot_drift(self):
        bad = (("required_favorable_cost_blocks", 2), ("proposed_practical_relative_cost_reduction_percent", .5),
               ("total_episodes", 180), ("fresh_paired_worlds_per_block", 13), ("primary", "old_primary"),
               ("secondary_contrasts", ["old_contrast"]), ("learned_selection", "sample"),
               ("controllers_per_block", 5), ("environment_steps", 999))
        for key, value in bad:
            cfg = copy.deepcopy(self.config)
            cfg["evaluation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_time_baseline_raw_bundle(self.root, [], cfg, self.streams)
        for key, value in (("draws", 80), ("kind", "unpaired"), ("blocks_per_draw", 2), ("worlds_per_sampled_block", 10)):
            cfg = copy.deepcopy(self.config)
            cfg["evaluation"]["bootstrap"][key] = value
            with self.assertRaisesRegex(ValueError, "bootstrap"):
                verify_time_baseline_raw_bundle(self.root, [], cfg, self.streams)

    def test_decimal_stream_seeds_namespace_hash_and_schedule_rejected(self):
        for value in (1, True, 1., "01", "+1", "1e3", "-1", " 1", "\u0661"):
            streams = copy.deepcopy(self.streams)
            streams["environment"]["60"]["test"][0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "decimal-string"):
                verify_time_baseline_raw_bundle(self.root, [], self.config, streams)
        for key in ("source", "hash", "bootstrap", "test_seed", "training_seed"):
            streams = copy.deepcopy(self.streams)
            if key == "source":
                streams["namespace"] = "foreign"
            elif key == "hash":
                streams["proposal_sha256"] = "a" * 64
            elif key == "bootstrap":
                streams["neural"]["analysis/bootstrap"] = "123"
            else:
                split = "test" if key == "test_seed" else "training"
                streams["environment"]["60"][split][0] = streams["environment"]["61"][split][0]
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_time_baseline_raw_bundle(self.root, [], self.config, streams)

    def test_equal_block_relative_estimator_not_pooled_ratio(self):
        def costs(block, role, world):
            baseline = {60: 100, 61: 200, 62: 1000}[block]
            return baseline - 10 if role == "time_baseline_ppo" else baseline
        primary = self.verify(costs=costs)["analysis"]["contrasts"][0]
        self.assertAlmostEqual(primary["equal_block_relative_cost_change_percent"], (-10 - 5 - 1) / 3)
        self.assertNotAlmostEqual(primary["equal_block_relative_cost_change_percent"], -30 / 1300 * 100)

    def test_10000_shared_hierarchical_draws_match_independent_loop(self):
        def costs(block, role, world):
            base = 100 + world * 20
            return base + (block - 59) * (world * 4 - 3) if role == "time_baseline_ppo" else base
        analysis = self.verify(costs=costs)["analysis"]
        rng = np.random.default_rng(int(self.streams["neural"]["analysis/bootstrap"]))
        blocks = rng.integers(0, 3, size=(10000, 3))
        worlds = rng.integers(0, 12, size=(10000, 3, 12))
        deltas, relatives = [], []
        for draw in range(10000):
            means, ratios = [], []
            for slot in range(3):
                b = self.config["blocks"][blocks[draw, slot]]
                ds = [costs(b, "time_baseline_ppo", int(w)) - costs(b, "own_frozen", int(w)) for w in worlds[draw, slot]]
                bs = [costs(b, "own_frozen", int(w)) for w in worlds[draw, slot]]
                means.append(sum(ds) / 12)
                ratios.append(100 * sum(ds) / sum(bs))
            deltas.append(sum(means) / 3)
            relatives.append(sum(ratios) / 3)
        primary = analysis["contrasts"][0]
        np.testing.assert_allclose(primary["hierarchical_paired_ci95"]["cost"], np.quantile(deltas, [.025, .975]))
        np.testing.assert_allclose(primary["relative_cost_change_ci95"], np.quantile(relatives, [.025, .975]))
        self.assertEqual(analysis["bootstrap"]["resample_design_sha256"], json_hash([blocks.tolist(), worlds.tolist()]))

    def test_null_weak_inconsistent_and_zero_denominators_close_route(self):
        for costs in (lambda b, r, w: 100, lambda b, r, w: 99.5 if r == "time_baseline_ppo" else 100,
                      lambda b, r, w: (101 if b == 60 else 90) if r == "time_baseline_ppo" else 100,
                      lambda b, r, w: 0):
            result = self.verify(costs=costs)["analysis"]
            self.assertFalse(result["primary_cost_screen_met"])
            self.assertTrue(result["baseline_route"]["close"])
            self.assertEqual(result["baseline_route"]["next"], "reward_diagnosis_and_design_only")
            self.assertFalse(result["baseline_route"]["reward_training_authorized"])
        primary = result["contrasts"][0]
        self.assertIsNone(primary["equal_block_relative_cost_change_percent"])
        self.assertIsNone(primary["relative_cost_change_ci95"])
        self.assertEqual(primary["undefined_relative_bootstrap_draws"], 10000)

    def test_unchanged_actions_close_even_with_apparently_lower_raw_cost(self):
        analysis = self.verify(changed=False)["analysis"]
        self.assertTrue(analysis["primary_cost_screen_met"])
        self.assertTrue(analysis["baseline_route"]["close"])
        self.assertIn("greedy_requests_unchanged_from_frozen", analysis["baseline_route"]["reasons"])
        self.assertIn("greedy_requests_unchanged_from_current_ppo", analysis["baseline_route"]["reasons"])
        self.assertEqual(analysis["decision"], "limited_negative_or_inconclusive")

    def test_primary_screen_does_not_hide_method_null_or_bc_tradeoff(self):
        costs = lambda b, r, w: 90 if r in ("time_baseline_ppo", "current_ppo") else 80 if r == "bc_continue" else 100
        analysis = self.verify(costs=costs)["analysis"]
        self.assertTrue(analysis["primary_cost_screen_met"])
        self.assertEqual(analysis["method_cost_direction"], "null_or_unfavorable")
        self.assertEqual(analysis["bc_attribution_cost_direction"], "null_or_unfavorable")
        self.assertIn("no_cost_benefit_over_current_ppo", analysis["baseline_route"]["reasons"])
        self.assertEqual(analysis["contrasts"][2]["equal_block_mean_differences"]["cost"], 10.)

    def test_adverse_patient_directions_are_preserved_including_secondary_only(self):
        def adverse(h, rows, final):
            if h["role"] == "time_baseline_ppo" and h["block"] == 60:
                final["patients"]["p0"]["status"] = "lost"
                final["scalars"].update(cumulative_lost=2, cumulative_served=0)
                rows[-1]["event"]["info"].update(patients_lost=[2, 0], patients_completed=[0, 0],
                    patients_lost_waiting_ineligible=[2, 0], completion_service_level=0.)
                rows[-1]["event"]["audit"]["record"]["next_state_token"] = json_hash(final)
        analysis = self.verify(mutate=adverse)["analysis"]
        self.assertEqual(analysis["decision"], "cost_patient_tradeoff")
        self.assertTrue(analysis["triage_checks"]["primary_has_adverse_patient_direction"])
        self.assertEqual({r["metric"] for r in analysis["contrasts"][0]["adverse_patient_block_directions"]}, {"losses", "completions"})
        self.assertFalse(analysis["clinical_superiority_claim"])
        self.assertTrue(analysis["contrasts"][2]["cost_patient_tradeoff"])

    def test_raw_byte_hash_redirect_duplicate_keys_and_path_alias_rejected(self):
        entry = save_episode(self.root, "tamper", *episode(self.config, self.streams))
        path = self.root / entry["events"]["path"]
        raw = path.read_bytes()
        path.write_bytes(raw + b" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            read_time_baseline_raw_episodes(self.root, [entry], self.config)
        path.write_bytes(raw)
        link = self.root / "link.jsonl"
        link.symlink_to(path)
        bad = copy.deepcopy(entry)
        bad["events"]["path"] = link.name
        with self.assertRaisesRegex(ValueError, "redirected"):
            read_time_baseline_raw_episodes(self.root, [bad], self.config)
        for name in ("./" + entry["events"]["path"], "../outside.jsonl", str(path)):
            bad["events"]["path"] = name
            with self.assertRaisesRegex(ValueError, "noncanonical"):
                read_time_baseline_raw_episodes(self.root, [bad], self.config)
        header = self.root / entry["header"]["path"]
        duplicate = b'{"split":"training",' + header.read_bytes()[1:]
        header.write_bytes(duplicate)
        entry["header"].update(bytes=len(duplicate), sha256=hashlib.sha256(duplicate).hexdigest())
        with self.assertRaisesRegex(ValueError, "duplicate raw JSON key"):
            read_time_baseline_raw_episodes(self.root, [entry], self.config)

    def test_read_only_input_unchanged_and_index_permutation_keeps_analysis(self):
        index = write_bundle(self.root, self.config, self.streams)
        before = copy.deepcopy((index, self.config, self.streams))
        first = verify_time_baseline_raw_bundle(self.root, index, self.config, self.streams)
        second = verify_time_baseline_raw_bundle(self.root, list(reversed(index)), self.config, self.streams)
        self.assertEqual(first["analysis"], second["analysis"])
        self.assertEqual(before, (index, self.config, self.streams))
        self.assertNotEqual(first["index_sha256"], second["index_sha256"])
        for record in first["files"]:
            raw = (self.root / record["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["sha256"])


if __name__ == "__main__":
    unittest.main()
