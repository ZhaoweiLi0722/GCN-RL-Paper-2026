"""Artificial scalar/JSON fixtures only; no scientific inputs or execution."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, validate_budget_plan
from src.rl.time_baseline_plan import (
    time_baseline_budget_plan, time_baseline_serial_sections, time_baseline_stream_manifest,
)


def artificial_proposal():
    rows = [
        ("runtime_binding", 0, 0, 0, 0, 0, 300),
        ("same_start_preflight", 18, 936, 72, 0, 0, 600),
        ("current_ppo", 96, 4992, 0, 384, 384, 2250),
        ("time_baseline_ppo", 96, 4992, 0, 384, 384, 2250),
        ("bc_continue", 96, 4992, 0, 384, 0, 1200),
        ("all_model_seal", 0, 0, 0, 0, 0, 120),
        ("final_evaluation", 216, 11232, 0, 0, 0, 1620),
        ("raw_verification", 0, 0, 0, 0, 0, 600),
        ("payload_archive", 0, 0, 0, 0, 0, 900),
        ("supervisor_closure", 0, 0, 0, 0, 0, 600),
    ]
    return {
        "schema": "time-baseline-comparison-proposal-v1",
        "scientific_execution_authorized": False,
        "blocks": [60, 61, 62],
        "training_arms": ["current_ppo", "time_baseline_ppo", "bc_continue"],
        "evaluation_controllers": ["own_frozen", "current_ppo", "time_baseline_ppo",
                                   "bc_continue", "r4", "full_mdl2"],
        "episodes_per_training_arm_block": 32, "episodes_per_rollout": 4,
        "rollouts_per_training_arm_block": 8, "epochs_per_rollout": 4,
        "minibatch_sizes": [64, 64, 64, 16], "evaluation_worlds_per_block": 12,
        "preflight_worlds_per_block": 1, "clone_steps_per_preflight_controller": 4,
        "initial_historical_loads": 6, "layout_builds": 3, "episode_builds": 522,
        "preflight_clone_instances": 18, "same_start_learned_forks": 12,
        "scheduled_restore_envelope_reads": 234,
        "phase_budgets": [dict(zip(("id", "episodes", "trajectory", "clone", "actor", "critic", "seconds"), row))
                          for row in rows],
        "totals": {"episodes": 522, "trajectory": 27144, "clone": 72, "environment": 27216,
                   "actor": 1152, "critic": 768, "optimizer": 1920, "environment_builds": 525,
                   "phase_seconds": 10440, "global_seconds": 10800},
        "per_owner": {
            "ppo_each_arm_block": {"episodes": 32, "trajectory": 1664, "actor": 128, "critic": 128, "seconds": 750},
            "bc_each_block": {"episodes": 32, "trajectory": 1664, "actor": 128, "critic": 0, "seconds": 400},
            "evaluation_each_controller_block": {"episodes": 12, "trajectory": 624, "seconds": 90}},
        "rng": {
            "namespace": "artificial-time-baseline-fixture",
            "encoding": "sha256(namespace)[:12] big-endian <<16 plus ordinal; decimal strings on disk",
            "roles_by_block": {
                "60": {"layout": [0, 0], "preflight": [3, 3], "training": [6, 37], "test": [102, 113]},
                "61": {"layout": [1, 1], "preflight": [4, 4], "training": [38, 69], "test": [114, 125]},
                "62": {"layout": [2, 2], "preflight": [5, 5], "training": [70, 101], "test": [126, 137]}},
            "paired_environment_starts": True, "paired_future_events_after_divergence_claimed": False,
            "freshness_verified": False},
    }


class TimeBaselinePlanTests(unittest.TestCase):
    def assert_rejected(self, proposal):
        for function in (time_baseline_budget_plan, time_baseline_serial_sections, time_baseline_stream_manifest):
            with self.subTest(api=function.__name__), self.assertRaises(ValueError):
                function(proposal)

    def test_exact_plan_arithmetic_and_json_shape(self):
        proposal = artificial_proposal()
        plan = time_baseline_budget_plan(proposal)
        self.assertEqual(plan["limits"], {"trajectory": 27144, "clone": 72,
                                         "actor": 1152, "critic": 768, "seconds": 10800})
        self.assertEqual(plan["draft_sha256"], digest(proposal))
        self.assertEqual(len(plan["sections"]), 33)
        self.assertEqual(sum(row["seconds"] for row in plan["phases"].values()), 10440)
        for phase, caps in plan["phases"].items():
            scopes = [row for row in plan["sections"].values() if row["phase"] == phase]
            self.assertEqual({key: sum(row[key] for row in scopes) for key in caps}, caps)
        validate_budget_plan(json.loads(json.dumps(plan, sort_keys=True)))

    def test_per_owner_caps_and_no_extra_fit_gate(self):
        plan = time_baseline_budget_plan(artificial_proposal())
        self.assertEqual(plan["sections"]["time_baseline_ppo/block60"], {
            "phase": "time_baseline_ppo", "trajectory": 1664, "clone": 0,
            "actor": 128, "critic": 128, "seconds": 750})
        self.assertEqual(plan["sections"]["bc_continue/block62"]["critic"], 0)
        self.assertEqual(plan["sections"]["bc_continue/block62"]["seconds"], 400)
        for block in (60, 61, 62):
            for controller in artificial_proposal()["evaluation_controllers"]:
                self.assertEqual(plan["sections"][f"final_evaluation/block{block}/{controller}"], {
                    "phase": "final_evaluation", "trajectory": 624, "clone": 0,
                    "actor": 0, "critic": 0, "seconds": 90})
        self.assertFalse(any("fit" in phase or "qualification" in phase for phase in plan["phases"]))

    def test_serial_order_survives_sorted_json_and_seals_all_arms_first(self):
        proposal = artificial_proposal()
        order = time_baseline_serial_sections(proposal)
        expected = ["runtime_input_binding", "same_start_preflight"]
        expected += [f"{arm}/block{block}" for arm in proposal["training_arms"] for block in proposal["blocks"]]
        expected += ["all_model_seal"]
        expected += [f"final_evaluation/block{block}/{controller}" for block in proposal["blocks"]
                     for controller in proposal["evaluation_controllers"]]
        expected += ["raw_verification", "payload_archive", "supervisor_closure"]
        self.assertEqual(order, expected)
        self.assertEqual(time_baseline_serial_sections(json.loads(json.dumps(proposal, sort_keys=True))), expected)

    def test_pure_repeatable_detached_outputs(self):
        proposal = artificial_proposal()
        original = copy.deepcopy(proposal)
        first = time_baseline_stream_manifest(proposal)
        self.assertEqual(first, time_baseline_stream_manifest(proposal))
        plan = time_baseline_budget_plan(proposal)
        first["ordinals"]["60"]["test"][0] = -1
        first["environment"]["60"]["training"].clear()
        plan["sections"]["same_start_preflight"]["trajectory"] = -1
        self.assertEqual(proposal, original)
        self.assertEqual(time_baseline_stream_manifest(proposal)["ordinals"]["60"]["test"], [102, 113])

    def test_unknown_missing_duplicate_controllers_and_blocks(self):
        for field in ("training_arms", "evaluation_controllers", "blocks"):
            for operation in ("missing", "duplicate", "unknown"):
                proposal = artificial_proposal()
                if operation == "missing":
                    proposal[field].pop()
                elif operation == "duplicate":
                    proposal[field][-1] = proposal[field][0]
                else:
                    proposal[field][-1] = "unknown"
                with self.subTest(field=field, operation=operation):
                    self.assert_rejected(proposal)

    def test_unknown_missing_duplicate_and_reordered_phases(self):
        for operation in ("missing", "duplicate", "unknown", "reordered"):
            proposal = artificial_proposal()
            rows = proposal["phase_budgets"]
            if operation == "missing":
                rows.pop()
            elif operation == "duplicate":
                rows.append(copy.deepcopy(rows[0]))
            elif operation == "unknown":
                rows[-1]["id"] = "extra_fit"
            else:
                rows.reverse()
            with self.subTest(operation=operation):
                self.assert_rejected(proposal)

    def test_every_declared_total_and_owner_arithmetic(self):
        original = artificial_proposal()
        for key in original["totals"]:
            proposal = artificial_proposal()
            proposal["totals"][key] += 1
            if key == "global_seconds":
                proposal["totals"][key] = 10439
            with self.subTest(total=key):
                self.assert_rejected(proposal)
        for owner, row in original["per_owner"].items():
            for key in row:
                proposal = artificial_proposal()
                proposal["per_owner"][owner][key] += 1
                with self.subTest(owner=owner, resource=key):
                    self.assert_rejected(proposal)

    def test_every_phase_resource_and_episode_mismatch(self):
        for index, row in enumerate(artificial_proposal()["phase_budgets"]):
            for key in ("episodes", "trajectory", "clone", "actor", "critic"):
                proposal = artificial_proposal()
                proposal["phase_budgets"][index][key] += 1
                with self.subTest(phase=row["id"], key=key):
                    self.assert_rejected(proposal)

    def test_build_load_fork_and_restore_envelope_arithmetic(self):
        for key in ("initial_historical_loads", "layout_builds", "episode_builds", "preflight_clone_instances",
                    "same_start_learned_forks", "scheduled_restore_envelope_reads"):
            proposal = artificial_proposal()
            proposal[key] += 1
            with self.subTest(key=key):
                self.assert_rejected(proposal)

    def test_missing_fields_are_not_silently_defaulted(self):
        for field in ("totals", "rng", "per_owner", "phase_budgets", "episodes_per_rollout"):
            proposal = artificial_proposal()
            del proposal[field]
            self.assert_rejected(proposal)
        for path in (("totals", "actor"), ("per_owner", "bc_each_block"), ("rng", "roles_by_block")):
            proposal = artificial_proposal()
            del proposal[path[0]][path[1]]
            self.assert_rejected(proposal)
        proposal = artificial_proposal()
        del proposal["phase_budgets"][0]["clone"]
        self.assert_rejected(proposal)

    def test_protocol_dimensions_and_noninteger_caps_rejected(self):
        for key in ("episodes_per_training_arm_block", "episodes_per_rollout", "rollouts_per_training_arm_block",
                    "epochs_per_rollout", "evaluation_worlds_per_block", "preflight_worlds_per_block",
                    "clone_steps_per_preflight_controller"):
            proposal = artificial_proposal()
            proposal[key] += 1
            self.assert_rejected(proposal)
        for value in (True, 128.0, "128", -1, None):
            proposal = artificial_proposal()
            proposal["per_owner"]["ppo_each_arm_block"]["actor"] = value
            self.assert_rejected(proposal)
        proposal = artificial_proposal()
        proposal["minibatch_sizes"] = [64, 64, 63, 17]
        self.assert_rejected(proposal)

    def test_seed_encoding_counts_and_decimal_json_roundtrip(self):
        proposal = artificial_proposal()
        manifest = time_baseline_stream_manifest(proposal)
        base = int.from_bytes(hashlib.sha256(b"artificial-time-baseline-fixture").digest()[:12], "big") << 16
        environment = manifest["environment"]
        self.assertEqual(environment["60"]["layout"], [str(base)])
        self.assertEqual(environment["62"]["test"][-1], str(base + 137))
        seeds = [seed for roles in environment.values() for values in roles.values() for seed in values]
        self.assertEqual(sorted(map(int, seeds)), list(range(base, base + 138)))
        self.assertEqual(len(manifest["neural"]), 19)
        self.assertEqual(manifest["unique_worlds"], 138)
        self.assertEqual(manifest["episode_uses"], 522)
        uses = sum(len(values) * manifest["uses_per_world"][role]
                   for roles in environment.values() for role, values in roles.items())
        self.assertEqual(uses, 522)
        self.assertEqual(manifest["layout_builds"], 3)
        for path, seed in manifest["neural"].items():
            expected = int.from_bytes(hashlib.sha256((proposal["rng"]["namespace"] + "/" + path).encode()).digest()[:8], "big") % 2**63
            self.assertEqual(seed, str(expected))
        self.assertTrue(all(type(seed) is str and seed.isdecimal() for seed in seeds + list(manifest["neural"].values())))
        self.assertEqual(json.loads(json.dumps(manifest)), manifest)

    def test_training_pairing_preflight_test_and_neural_purpose_separation(self):
        manifest = time_baseline_stream_manifest(artificial_proposal())
        all_seeds = []
        for block, roles in manifest["environment"].items():
            for values in roles.values():
                all_seeds.extend(values)
            bindings = manifest["neural_bindings"][block]
            train, preflight = bindings["training"], bindings["preflight"]
            self.assertEqual(len({row["sampling_seed"] for row in train.values()}), 1)
            self.assertEqual(train["current_ppo"]["shuffle_seed"], train["time_baseline_ppo"]["shuffle_seed"])
            self.assertNotEqual(train["current_ppo"]["shuffle_seed"], train["bc_continue"]["shuffle_seed"])
            train_paths = {path for row in train.values() for path in row.values()}
            preflight_paths = {path for row in preflight.values() for path in row.values()}
            self.assertTrue(train_paths.isdisjoint(preflight_paths))
            self.assertTrue((train_paths | preflight_paths) <= set(manifest["neural"]))
            for start in range(0, 32, 4):
                self.assertEqual(len(set(roles["training"][start:start + 4])), 4)
        all_seeds.extend(manifest["neural"].values())
        self.assertEqual(len(all_seeds), len(set(all_seeds)))
        self.assertIn("analysis/bootstrap", manifest["neural"])
        self.assertFalse(any("initialization" in key for key in manifest["neural"]))

    def test_ordinal_omissions_overlaps_gaps_and_unknown_roles_rejected(self):
        for operation in ("block", "role", "unknown", "overlap", "gap", "truncated", "bool", "overflow", "test_swap"):
            proposal = artificial_proposal()
            ranges = proposal["rng"]["roles_by_block"]
            if operation == "block":
                del ranges["61"]
            elif operation == "role":
                del ranges["60"]["preflight"]
            elif operation == "unknown":
                ranges["60"]["holdout"] = [138, 138]
            elif operation == "overlap":
                ranges["61"]["training"] = [6, 37]
            elif operation == "gap":
                ranges["62"]["test"] = [127, 138]
            elif operation == "truncated":
                ranges["60"]["training"].pop()
            elif operation == "bool":
                ranges["60"]["layout"] = [False, False]
            elif operation == "overflow":
                ranges["62"]["test"] = [65536, 65547]
            else:
                ranges["60"]["preflight"], ranges["60"]["test"] = ranges["60"]["test"], ranges["60"]["preflight"]
            with self.subTest(operation=operation):
                self.assert_rejected(proposal)

    def test_freshness_and_execution_never_inferred_from_input_flags(self):
        proposal = artificial_proposal()
        proposal["rng"]["freshness_verified"] = True
        proposal["scientific_execution_authorized"] = True
        manifest = time_baseline_stream_manifest(proposal)
        for key in ("seed_freshness_verified", "scientific_execution_authorized", "consumed",
                    "paired_future_events_after_divergence_claimed"):
            self.assertIs(manifest[key], False)
        for key, value in (("namespace", ""), ("encoding", "other"), ("paired_environment_starts", False),
                           ("paired_future_events_after_divergence_claimed", True)):
            proposal = artificial_proposal()
            proposal["rng"][key] = value
            self.assert_rejected(proposal)

    def test_internal_collision_raises_without_reseeding(self):
        proposal = artificial_proposal()
        with patch("src.rl.time_baseline_plan.hashlib.sha256") as hash_function:
            hash_function.return_value.digest.return_value = bytes(32)
            with self.assertRaisesRegex(ValueError, "cross-purpose seed collision"):
                time_baseline_stream_manifest(proposal)
        self.assertEqual(proposal, artificial_proposal())

    def test_dynamic_budget_accepts_plan_and_charges_outer_startup(self):
        plan = time_baseline_budget_plan(artificial_proposal())
        with tempfile.TemporaryDirectory() as directory:
            budget = DynamicCandidateBudget(Path(directory) / "fixture.jsonl", plan, enabled=True,
                                            clock=lambda: 301.0, started=0.0)
            try:
                with self.assertRaises(TimeoutError):
                    budget.begin("runtime_input_binding")
                self.assertTrue(budget.failed)
            finally:
                budget.close()

    def test_dynamic_owner_caps_do_not_borrow_critic_or_other_arm_calls(self):
        plan = time_baseline_budget_plan(artificial_proposal())
        with tempfile.TemporaryDirectory() as directory:
            budget = DynamicCandidateBudget(Path(directory) / "fixture.jsonl", plan, enabled=True, clock=lambda: 0.0)
            try:
                budget.begin("time_baseline_ppo/block60")
                for _ in range(128):
                    budget.debit_optimizer("actor")  # Ledger fixture; no optimizer exists.
                with self.assertRaises(RuntimeError):
                    budget.debit_optimizer("actor")
                self.assertEqual(budget.counts["optimizer"], 128)
                self.assertNotIn("global:critic", budget.owner_counts)
            finally:
                budget.close()

    def test_import_has_no_model_environment_or_optimizer_dependencies(self):
        script = """
import builtins
real_import = builtins.__import__
def scalar_only(name, *args, **kwargs):
    blocked = ('torch', 'numpy', 'src.models', 'src.baselines', 'src.env')
    if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked):
        raise AssertionError('non-scalar import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = scalar_only
import src.rl.time_baseline_plan
"""
        result = subprocess.run([sys.executable, "-B", "-S", "-c", script],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
