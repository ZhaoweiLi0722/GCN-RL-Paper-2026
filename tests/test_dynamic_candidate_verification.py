"""Invented raw registries/receipts only: no simulator, model or optimizer."""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.rl.candidate_pilot_verification import CAUSES, OPERATING, PATIENT, json_hash
from src.rl.dynamic_candidate_resources import dynamic_stream_manifest
from src.rl.dynamic_candidate_verification import (
    CONTROLLERS, read_dynamic_raw_episodes, verify_dynamic_episode, verify_dynamic_raw_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def fixture_config(worlds=2):
    cfg = json.loads((ROOT / "specs/2026-10-01-adaptive-paper-delivery/pilot-budget-draft.json").read_text())
    streams = dynamic_stream_manifest(cfg)
    cfg["objective"].update(horizon=2, num_facilities=2, action_width=8, transfer_scale=120)
    cfg["candidate_support"] = {"max_original_requests": 6}
    cfg["evaluation"].update(fresh_paired_worlds_per_block=worlds, total_worlds=3 * worlds,
                             total_episodes=15 * worlds, environment_steps=30 * worlds)
    cfg["evaluation"]["bootstrap"].update(draws=80, worlds_per_sampled_block=worlds)
    for block in cfg["blocks"]:
        streams["environment"][str(block)]["test"] = streams["environment"][str(block)]["test"][:worlds]
    return cfg, streams


def patient(pid, status="waiting"):
    return {"patient_id": pid, "specimen_id": pid, "status": status,
            "material_facility": 0, "manufacturing_facility": None}


def raw_episode(block, role, world, seed, total_cost=100):
    initial = {"scalars": {"t": 0, "_episode_seed": seed, "cumulative_enrolled": 2,
                           "cumulative_lost": 0, "cumulative_served": 0},
               "patients": {"p0": patient("p0"), "p1": patient("p1")},
               "patient_queues": [["p0", "p1"], []], "in_production_patients": [[[], []], [[], []]],
               "specimen_transits": [], "product_return_transits": []}
    final = copy.deepcopy(initial)
    final["scalars"].update(t=2, cumulative_enrolled=3, cumulative_lost=1, cumulative_served=1)
    final["patients"] = {"p0": patient("p0", "delivered"), "p1": patient("p1", "lost"), "p2": patient("p2")}
    final["patient_queues"] = [["p2"], []]
    lineage = f"invented/{block}/{role}/{world}"
    policy = hashlib.sha256(f"invented-policy/{block}/{role}".encode()).hexdigest()
    header = {"initial_state": initial, "seed": seed, "source_id": "invented-only", "trajectory_id": lineage,
              "block": block, "representation": "graph" if role.startswith("own_") else "reference", "role": role,
              "world_index": world, "policy_sha256": policy, "split": "test",
              "selection": "reference" if role == "r4" else "anchor" if role == "full_mdl2" else "greedy"}
    request = [.0125, -.0125, 0., 0., 0., 0., 0., 0.]
    anchor = [0.] * 8
    if role == "full_mdl2":
        request = [0., 0., 0., 0., 0., 0., .5, .5]
        anchor = request.copy()
        bank = {"class_keys": [request.copy()], "requests": [request, anchor], "request_to_class": [0, 0]}
        selected, log_probs, requested, blocked = 0, [0.], [0, 0], 0
    else:
        bank = {"class_keys": [[0.] * 8, [2, -2, 0., 0., 0., 0., 0., 0.]],
                "requests": [request, anchor], "request_to_class": [1, 0]}
        selected, log_probs, requested, blocked = 1, [math.log(.1), math.log(.9)], [2, -2], 2
    rows = []
    for step in range(2):
        info = {k: 0. for k in OPERATING + PATIENT}
        info.update(reagent_purchase_cost=total_cost / 2, cost=total_cost / 2, base_cost=total_cost / 2,
                    specimen_route_cost=0., transshipment_cost=0., patients_lost=[step, 0],
                    patients_completed=[step, 0], demand=[1 - step, 0], identity_active_count=3 - 2 * step,
                    identity_terminal_count=2 * step, waiting_patients=[3 - 2 * step, 0],
                    in_production_patients=[0, 0], specimen_in_transit=[0, 0],
                    specimen_requested_integer_net=requested, specimen_transfers=[0, 0], specimen_route_count=0,
                    blocked_specimen_requests=blocked, blocked_specimen_inbound_requests=blocked,
                    blocked_specimen_outbound_requests=blocked, completion_service_level=step / 3,
                    capacity_transfers=[0, 0], reagent_transfers=[0, 0], replenishment=[1, 0])
        info.update({k: [0, 0] for k in CAUSES})
        info["patients_lost_waiting_ineligible"] = [step, 0]
        record = {"semantics": {"reward_kind": "absolute_environment", "reward_scale": 1e-9, "gamma": 1.},
                  "origin": "trajectory", "source_id": "invented-only", "trajectory_id": lineage, "step_index": step,
                  "state_token": json_hash(initial) if step == 0 else "intermediate",
                  "next_state_token": "intermediate" if step == 0 else json_hash(final),
                  "state": [step], "next_state": [step + 1], "terminated": step == 1, "truncated": False,
                  "raw_reward": -total_cost / 2, "action": request.copy()}
        rows.append({"inference_seconds": .001, "event": {"info": info, "audit": {"record": record,
                     "terminal_cost_added": 0., "decision": {"evaluation": {"candidates": copy.deepcopy(bank),
                     "behavior_sha256": policy, "log_probs": log_probs.copy(), "inference_dtype": "float32"},
                     "choice": {"class_index": selected, "submitted_request": request.copy()}}}}})
    return header, rows, final


def write_bundle(root, cfg, streams, *, costs=None, mutate=None):
    index = []

    def save(name, value, jsonl=False):
        text = "".join(json.dumps(row) + "\n" for row in value) if jsonl else json.dumps(value)
        raw = text.encode()
        (root / name).write_bytes(raw)
        return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    for block in cfg["blocks"]:
        for world, seed in enumerate(streams["environment"][str(block)]["test"]):
            for role in CONTROLLERS:
                cost = costs(block, role, world) if costs else (90 if role == "own_ppo" else 100)
                header, rows, final = raw_episode(block, role, world, seed, cost)
                if mutate:
                    mutate(header, rows, final)
                prefix = f"{block}-{role}-{world}"
                index.append({"header": save(prefix + "-header.json", header),
                              "events": save(prefix + "-events.jsonl", rows, True),
                              "final_state": save(prefix + "-final.json", final)})
    return index


class DynamicCandidateVerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.cfg, self.streams = fixture_config()

    def verify(self, **kwargs):
        index = write_bundle(self.root, self.cfg, self.streams, **kwargs)
        return verify_dynamic_raw_bundle(self.root, index, self.cfg, self.streams)

    def test_complete_raw_schedule_metrics_hashes_and_primary(self):
        report = self.verify()
        self.assertEqual(len(report["files"]), 90)
        self.assertEqual(len(report["outcomes"]), 30)
        row = report["outcomes"][0]
        self.assertEqual((row["waiting_patient_steps"], row["terminal_waiting"], row["expiry_losses"]), (4, 1, 0))
        self.assertEqual(row["actions"][0]["requested"][0], .0125)
        self.assertEqual(row["actions"][0]["specimen_requested_integer_net"], [2, -2])
        self.assertEqual(row["compute_cost"]["decision_path_seconds"], .002)
        self.assertEqual(row["compute_cost"]["evaluation_env_steps"], 2)
        self.assertEqual(report["analysis"]["decision"], "promising_development_signal")
        self.assertEqual(len(report["analysis"]["contrasts"]), 5)
        primary = report["analysis"]["contrasts"][0]
        self.assertEqual(primary["equal_block_relative_cost_change_percent"], -10.)
        self.assertEqual(primary["hierarchical_paired_ci95"]["cost"], [-10., -10.])
        self.assertEqual(primary["relative_cost_change_ci95"], [-10., -10.])
        self.assertEqual(len(primary["paired_worlds"]), 3)
        self.assertFalse(report["analysis"]["clinical_noninferiority_claim"])
        self.assertFalse(report["analysis"]["deployment_adaptation_claim"])
        self.assertFalse(report["analysis"]["clean_ddpg_superiority_claim"])
        self.assertFalse(report["analysis"]["automatic_followon"])
        self.assertFalse(report["grants_scientific_execution_authorization"])
        self.assertEqual(report["config_sha256"], json_hash(self.cfg))

    def test_default_three_by_twelve_by_five_schedule_is_supported(self):
        self.cfg, self.streams = fixture_config(worlds=12)
        report = self.verify()
        self.assertEqual(report["analysis"]["evaluation_episodes"], 180)
        self.assertEqual(len(report["files"]), 540)

    def test_equal_block_relative_change_not_pooled_ratio(self):
        def costs(block, role, world):
            base = {60: 100, 61: 200, 62: 1000}[block]
            return base - 10 if role == "own_ppo" else base
        result = self.verify(costs=costs)["analysis"]["contrasts"][0]
        self.assertAlmostEqual(result["equal_block_relative_cost_change_percent"], (-10 - 5 - 1) / 3)
        self.assertNotAlmostEqual(result["equal_block_relative_cost_change_percent"], -30 / 1300 * 100)

    def test_hierarchical_bootstrap_reproduced_by_independent_loop(self):
        def costs(block, role, world):
            base = 100 + world * 20
            return base + (block - 59) * (world * 4 - 3) if role == "own_ppo" else base
        report = self.verify(costs=costs)
        analysis = report["analysis"]
        primary = analysis["contrasts"][0]
        rng = np.random.default_rng(self.streams["neural"]["analysis/bootstrap"])
        bdraw = rng.integers(0, 3, size=(80, 3))
        wdraw = rng.integers(0, 2, size=(80, 3, 2))
        deltas, relatives = [], []
        for draw in range(80):
            means, ratios = [], []
            for slot in range(3):
                b = self.cfg["blocks"][bdraw[draw, slot]]
                ds = [costs(b, "own_ppo", int(w)) - costs(b, "own_frozen", int(w)) for w in wdraw[draw, slot]]
                bs = [costs(b, "own_frozen", int(w)) for w in wdraw[draw, slot]]
                means.append(sum(ds) / 2)
                ratios.append(100 * sum(ds) / sum(bs))
            deltas.append(sum(means) / 3)
            relatives.append(sum(ratios) / 3)
        np.testing.assert_allclose(primary["hierarchical_paired_ci95"]["cost"], np.quantile(deltas, [.025, .975]))
        np.testing.assert_allclose(primary["relative_cost_change_ci95"], np.quantile(relatives, [.025, .975]))
        self.assertEqual(analysis, self.verify(costs=costs)["analysis"])

    def test_new_learned_role_names_enforce_greedy_and_tie_break(self):
        for role in CONTROLLERS[:3]:
            def bad(header, rows, final):
                if header["role"] == role:
                    rows[0]["event"]["audit"]["decision"]["evaluation"]["log_probs"] = [math.log(.5)] * 2
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "canonical greedy"):
                self.verify(mutate=bad)

    def test_canonical_greedy_correction_is_counted_not_forced_to_reference(self):
        def correction(header, rows, final):
            if header["role"] != "own_ppo":
                return
            for row in rows:
                audit, info = row["event"]["audit"], row["event"]["info"]
                audit["record"]["action"] = [0.] * 8
                audit["decision"]["choice"].update(class_index=0, submitted_request=[0.] * 8)
                audit["decision"]["evaluation"]["log_probs"] = [math.log(.5)] * 2
                info.update(specimen_requested_integer_net=[0, 0], blocked_specimen_requests=0,
                            blocked_specimen_inbound_requests=0, blocked_specimen_outbound_requests=0)
        report = self.verify(mutate=correction)
        ppo = next(r for r in report["outcomes"] if r["role"] == "own_ppo")
        self.assertTrue(all(r["different_class"] for r in ppo["reference_corrections"]))
        self.assertEqual(ppo["reference_corrections"][0]["request_l1"], .025)
        d = report["analysis"]["contrasts"][0]["paired_worlds"][0]["worlds"][0]["action_differences"]
        self.assertEqual(d["requested_changed_steps"], 2)
        self.assertEqual(d["specimen_requested_integer_l1"], 8)
        self.assertEqual(d["executed_changed_steps"], 0)

    def test_full_mdl2_is_not_a_restricted_anchor_and_r4_need_not_be_greedy(self):
        def r4_nongreedy(header, rows, final):
            if header["role"] == "r4":
                for row in rows:
                    row["event"]["audit"]["decision"]["evaluation"]["log_probs"] = [math.log(.9), math.log(.1)]
        report = self.verify(mutate=r4_nongreedy)
        full = next(r for r in report["outcomes"] if r["role"] == "full_mdl2")
        self.assertEqual(full["actions"][0]["requested"][-2:], [.5, .5])
        def restricted(header, rows, final):
            if header["role"] == "full_mdl2":
                # Internally valid bank with a distinct R4 specimen request.
                for row in rows:
                    bank = row["event"]["audit"]["decision"]["evaluation"]["candidates"]
                    bank["requests"][0] = [.0125, -.0125, 0., 0., 0., 0., .5, .5]
                    bank["class_keys"].append([2, -2, 0., 0., 0., 0., .5, .5])
                    bank["request_to_class"] = [1, 0]
                    row["event"]["audit"]["decision"]["evaluation"]["log_probs"] = [math.log(.9), math.log(.1)]
        with self.assertRaisesRegex(ValueError, "full-anchor"):
            self.verify(mutate=restricted)

    def test_null_weak_inconsistent_and_undefined_relative_results(self):
        for costs in (lambda b, r, w: 100, lambda b, r, w: 99.5 if r == "own_ppo" else 100,
                      lambda b, r, w: (101 if b == 60 else 90) if r == "own_ppo" else 100,
                      lambda b, r, w: 0):
            result = self.verify(costs=costs)["analysis"]
            self.assertEqual(result["decision"], "limited_negative_or_inconclusive")
        primary = result["contrasts"][0]
        self.assertIsNone(primary["equal_block_relative_cost_change_percent"])
        self.assertIsNone(primary["relative_cost_change_ci95"])
        self.assertEqual(primary["undefined_relative_bootstrap_draws"], 80)
        self.assertFalse(result["triage_checks"]["primary_relative_cost_defined"])

    def test_cost_improvement_with_patient_direction_is_reported_as_tradeoff(self):
        def adverse(header, rows, final):
            if header["role"] != "own_ppo" or header["block"] != 60:
                return
            final["patients"]["p0"]["status"] = "lost"
            final["scalars"].update(cumulative_lost=2, cumulative_served=0)
            last = rows[-1]["event"]
            last["info"].update(patients_lost=[2, 0], patients_completed=[0, 0],
                                 patients_lost_waiting_ineligible=[2, 0], completion_service_level=0.)
            last["audit"]["record"]["next_state_token"] = json_hash(final)
        result = self.verify(mutate=adverse)["analysis"]
        self.assertEqual(result["decision"], "cost_patient_tradeoff")
        self.assertTrue(result["triage_checks"]["primary_has_adverse_patient_direction"])
        self.assertFalse(result["clinical_noninferiority_claim"])

    def test_secondary_costs_are_reported_without_adding_success_gates(self):
        costs = lambda b, r, w: 90 if r == "own_ppo" else 80 if r == "own_bc_continue" else 100
        result = self.verify(costs=costs)["analysis"]
        self.assertEqual(result["decision"], "promising_development_signal")
        self.assertEqual(result["contrasts"][1]["equal_block_mean_differences"]["cost"], 10)
        self.assertIn("primary", result["decision_scope"])
        self.assertEqual(result["contrasts"][4]["name"], "own_frozen_minus_r4")

    def test_missing_duplicate_and_wrong_schedule_rejected(self):
        index = write_bundle(self.root, self.cfg, self.streams)
        for candidate in (index[:-1], index + [index[0]], [index[0]] + index[:-1]):
            with self.assertRaises(ValueError):
                verify_dynamic_raw_bundle(self.root, candidate, self.cfg, self.streams)
        for key, value in (("blocks_per_draw", 2), ("worlds_per_sampled_block", 3), ("kind", "unpaired")):
            cfg = copy.deepcopy(self.cfg)
            cfg["evaluation"]["bootstrap"][key] = value
            with self.assertRaises(ValueError):
                verify_dynamic_raw_bundle(self.root, index, cfg, self.streams)

    def test_policy_pairing_stream_and_lineage_corruption_rejected(self):
        def changed_policy(h, r, f):
            if h["world_index"] == 0:
                h["policy_sha256"] = "b" * 64
                for row in r:
                    row["event"]["audit"]["decision"]["evaluation"]["behavior_sha256"] = h["policy_sha256"]
        def wrong_seed(h, r, f):
            h["seed"] += 500
            h["initial_state"]["scalars"]["_episode_seed"] = h["seed"]
            f["scalars"]["_episode_seed"] = h["seed"]
            r[0]["event"]["audit"]["record"]["state_token"] = json_hash(h["initial_state"])
            r[-1]["event"]["audit"]["record"]["next_state_token"] = json_hash(f)
        def unpaired(h, r, f):
            if h["role"] == "own_ppo":
                h["initial_state"]["invented_public_marker"] = 1
                r[0]["event"]["audit"]["record"]["state_token"] = json_hash(h["initial_state"])
        for mutate in (changed_policy, wrong_seed, unpaired,
                       lambda h, r, f: h.update(representation="flat"),
                       lambda h, r, f: h.update(split="training")):
            with self.assertRaises(ValueError):
                self.verify(mutate=mutate)

    def test_raw_arithmetic_identity_nonfinite_action_and_compute_corruption_rejected(self):
        mutations = (lambda h, r, f: r[0]["event"]["info"].update(cost=999),
                     lambda h, r, f: f["patients"]["p2"].update(specimen_id="p1"),
                     lambda h, r, f: r[0].update(inference_seconds=float("inf")),
                     lambda h, r, f: r[0]["event"]["audit"]["record"]["action"].__setitem__(0, float(np.float32(.0125))),
                     lambda h, r, f: r[0]["event"]["info"].update(capacity_transfers=[.5, -.5]),
                     lambda h, r, f: r[0]["event"]["info"].update(reagent_transfers=[1, 0]),
                     lambda h, r, f: r[0]["event"]["info"].update(replenishment=[-1, 0]))
        for mutation in mutations:
            with self.assertRaises(ValueError):
                self.verify(mutate=mutation)

    def test_raw_bytes_duplicate_path_symlink_and_mid_read_changes_rejected(self):
        index = write_bundle(self.root, self.cfg, self.streams)
        target = self.root / index[0]["events"]["path"]
        original = target.read_bytes()
        target.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_dynamic_raw_bundle(self.root, index, self.cfg, self.streams)
        target.write_bytes(original)
        redirected = self.root / "redirected.jsonl"
        redirected.symlink_to(target)
        bad = copy.deepcopy(index)
        bad[0]["events"]["path"] = redirected.name
        with self.assertRaisesRegex(ValueError, "redirected"):
            verify_dynamic_raw_bundle(self.root, bad, self.cfg, self.streams)
        from src.rl.dynamic_candidate_verification import read_raw_episodes
        def alter_after_scalar(*args):
            result = read_raw_episodes(*args)
            target.write_bytes(original + b" ")
            return result
        with patch("src.rl.dynamic_candidate_verification.read_raw_episodes", side_effect=alter_after_scalar):
            with self.assertRaisesRegex(ValueError, "changed during"):
                verify_dynamic_raw_bundle(self.root, index, self.cfg, self.streams)

    def test_index_order_does_not_change_analysis_and_input_is_not_mutated(self):
        index = write_bundle(self.root, self.cfg, self.streams)
        before = copy.deepcopy((index, self.cfg, self.streams))
        result = verify_dynamic_raw_bundle(self.root, index, self.cfg, self.streams)
        reversed_result = verify_dynamic_raw_bundle(self.root, list(reversed(index)), self.cfg, self.streams)
        self.assertEqual(result["analysis"], reversed_result["analysis"])
        self.assertEqual(before, (index, self.cfg, self.streams))
        json.dumps(result, allow_nan=False)

    def test_qualification_single_episode_api_and_partial_raw_reread(self):
        def qualification(header, rows, final):
            if header["role"] == "own_frozen":
                header.update(role="initializer_greedy", split="qualification")
            elif header["role"] == "r4":
                header["split"] = "qualification"
        index = write_bundle(self.root, self.cfg, self.streams, mutate=qualification)
        # Partial phase readback does not need the complete final-test matrix.
        selected = [index[0], index[3]]
        files, outcomes = read_dynamic_raw_episodes(self.root, selected, self.cfg)
        self.assertEqual(len(files), 6)
        self.assertEqual([r["role"] for r in outcomes], ["initializer_greedy", "r4"])
        self.assertTrue(all(r["split"] == "qualification" for r in outcomes))
        self.assertTrue(all(r["compute_cost"]["evaluation_env_steps"] == 0 for r in outcomes))
        h, rows, f = raw_episode(60, "own_frozen", 0, 123)
        h.update(role="initializer_greedy", split="qualification")
        self.assertEqual(verify_dynamic_episode(h, rows, f, self.cfg)["cost"], 100)
        rows[0]["event"]["audit"]["decision"]["evaluation"]["log_probs"] = [math.log(.9), math.log(.1)]
        with self.assertRaisesRegex(ValueError, "canonical greedy"):
            verify_dynamic_episode(h, rows, f, self.cfg)

    def test_preflight_sampling_training_and_demonstration_role_boundaries(self):
        for role in ("preflight", *CONTROLLERS):
            h, rows, f = raw_episode(60, "own_ppo" if role == "preflight" else role, 0, 123)
            h.update(role=role, split="preflight")
            if role in ("preflight", "own_ppo", "own_bc_continue"):
                # Sampled preflight need not choose the highest probability.
                h["selection"] = "sample"
                rows[0]["event"]["audit"]["decision"]["evaluation"]["log_probs"] = [math.log(.9), math.log(.1)]
            self.assertEqual(verify_dynamic_episode(h, rows, f, self.cfg)["split"], "preflight")
        for role in ("own_ppo", "own_bc_continue"):
            h, rows, f = raw_episode(60, role, 0, 123)
            h.update(split="training", selection="sample")
            rows[0]["event"]["audit"]["decision"]["evaluation"]["log_probs"] = [math.log(.9), math.log(.1)]
            self.assertEqual(verify_dynamic_episode(h, rows, f, self.cfg)["role"], role)
        h, rows, f = raw_episode(60, "r4", 0, 123)
        h["split"] = "demonstration"
        self.assertEqual(verify_dynamic_episode(h, rows, f, self.cfg)["role"], "r4")
        for split in ("training", "validation", "holdout"):
            h["split"] = split
            with self.assertRaisesRegex(ValueError, "controller/split"):
                verify_dynamic_episode(h, rows, f, self.cfg)

    def test_terminal_waiting_registry_disagreement_and_missing_execution_channel_fail(self):
        h, rows, f = raw_episode(60, "own_ppo", 0, 123)
        rows[-1]["event"]["info"]["waiting_patients"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "queue differs"):
            verify_dynamic_episode(h, rows, f, self.cfg)
        h, rows, f = raw_episode(60, "own_ppo", 0, 123)
        del rows[0]["event"]["info"]["capacity_transfers"]
        with self.assertRaises(KeyError):
            verify_dynamic_episode(h, rows, f, self.cfg)

    def test_recorded_selection_cannot_claim_sample_for_greedy_evaluation(self):
        h, rows, f = raw_episode(60, "own_ppo", 0, 123)
        h["selection"] = "sample"
        with self.assertRaisesRegex(ValueError, "selection mode"):
            verify_dynamic_episode(h, rows, f, self.cfg)

    def test_rehashed_duplicate_json_keys_do_not_bypass_strict_reader(self):
        index = write_bundle(self.root, self.cfg, self.streams)
        record = index[0]["header"]
        path = self.root / record["path"]
        raw = b'{"split":"training",' + path.read_bytes()[1:]
        path.write_bytes(raw)
        record.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError, "duplicate raw JSON key"):
            verify_dynamic_raw_bundle(self.root, index, self.cfg, self.streams)


if __name__ == "__main__":
    unittest.main()
