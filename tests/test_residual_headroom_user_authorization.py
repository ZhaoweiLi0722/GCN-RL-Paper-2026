"""The dated user exception preserves provenance and the frozen science."""

from __future__ import annotations

import csv
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation.screen_intertemporal_residual_allocation_headroom import (
    DEFAULT_CONFIG,
    candidate_specs,
    describe,
    execution_approvals,
    load_json,
    require_execution_authorization,
    run,
    sha256,
    validate_config,
    write_csv,
)
from evaluation.screen_intertemporal_shared_capacity_episodes import scenario_env
from evaluation.screen_intertemporal_shared_capacity_states import make_policy
from src.rl.experiment import build_env


CONFIG = Path(
    "experiments/configs/intertemporal_residual_allocation_headroom_user_20260908.json"
)
MODULE = "evaluation.screen_intertemporal_residual_allocation_headroom"


class UserExceptionTest(unittest.TestCase):
    def test_original_still_requires_howard(self) -> None:
        with self.assertRaisesRegex(PermissionError, "howard"):
            require_execution_authorization(load_json(DEFAULT_CONFIG))
        self.assertFalse(describe(load_json(DEFAULT_CONFIG))["execution_authorized"])

    def test_exception_is_authorized_without_claiming_howard_approval(self) -> None:
        config = load_json(CONFIG)
        require_execution_authorization(config)
        description = describe(config)
        self.assertTrue(description["execution_authorized"])
        self.assertEqual(execution_approvals(config), {"zhaowei": True, "howard": False})
        self.assertEqual(description["expected_discovery_rows"], 8748)
        self.assertEqual(description["expected_validation_rows"], 21870)

    def test_each_scientific_change_is_rejected(self) -> None:
        changes = (
            ("lookahead", None, 13),
            ("gate", "minimum_prospective_relative_saving", 0.006),
            ("gate", "require_clinical_noninferiority", False),
            ("generation_seeds", "start", 99700001),
            ("discovery_worlds", "count_per_state", 3),
            ("action_library", "transfer_budget_fractions", [0.05, 0.20]),
        )
        for field, child, value in changes:
            with self.subTest(field=field, child=child):
                config = load_json(CONFIG)
                if child is None:
                    config[field] = value
                else:
                    config[field][child] = value
                with self.assertRaisesRegex(PermissionError, "scientific settings"):
                    require_execution_authorization(config)

    def test_exception_cannot_mark_howard_approved(self) -> None:
        config = load_json(CONFIG)
        config["execution_authorization"]["howard"]["approved"] = True
        with self.assertRaisesRegex(PermissionError, "Howard as unapproved"):
            require_execution_authorization(config)

    def test_missing_actual_user_approval_is_rejected(self) -> None:
        config = load_json(CONFIG)
        config["execution_authorization"]["zhaowei"]["approved"] = "true"
        with self.assertRaises(PermissionError):
            require_execution_authorization(config)

    def test_record_or_original_hash_mismatch_is_rejected(self) -> None:
        config = load_json(CONFIG)
        with mock.patch(f"{MODULE}.sha256", return_value="not-the-frozen-hash"):
            with self.assertRaisesRegex(PermissionError, "evidence hash mismatch"):
                require_execution_authorization(config)

    def test_exception_cannot_choose_another_record_or_output(self) -> None:
        config = load_json(CONFIG)
        config["execution_exception"]["record"] = "other.md"
        with self.assertRaisesRegex(PermissionError, "unrecognized"):
            require_execution_authorization(config)
        config = load_json(CONFIG)
        config["output_root"] = "results/not-authorized"
        with self.assertRaisesRegex(PermissionError, "single-use root"):
            require_execution_authorization(config)


class PersistenceTest(unittest.TestCase):
    def test_real_environment_end_to_end_smoke_uses_only_fixture_seeds(self) -> None:
        config = load_json(DEFAULT_CONFIG)
        source = validate_config(config)
        env_config = scenario_env(source, source["scenarios"][0])
        env = build_env({"env": env_config}, seed=123)
        behavior = make_policy({"budget_fraction": 1.0, "smoothing": 0.25})
        for _ in range(4):
            env.step(behavior.select_action(env.observation(), env=env))
        state = {
            "state_index": 0, "state_id": "fixture-only", "scenario": "fixture",
            "generation_seed": 123, "epoch": int(env.t),
            "env_config": env_config, "snapshot": env.state_dict(),
        }
        all_candidates = candidate_specs(config, env.config.num_facilities)
        config["discovery_worlds"] = {"start": 456, "count_per_state": 1}
        config["validation_worlds"] = {"start": 1456, "count_per_state": 1}
        config["lookahead"] = 3
        expected = {"state_count": 1, "expected_discovery_rows": 2, "expected_validation_rows": 2}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fixture-result"
            config["output_root"] = str(output)
            with mock.patch(f"{MODULE}.require_execution_authorization"), mock.patch(
                f"{MODULE}.generate_states", return_value=([state], [{"state_id": "fixture-only"}])
            ), mock.patch(f"{MODULE}.describe", return_value=expected), mock.patch(
                f"{MODULE}.candidate_specs", return_value=(all_candidates[0], all_candidates[-1])
            ), contextlib.redirect_stdout(io.StringIO()):
                result = run(config, config_path=DEFAULT_CONFIG)
            with (output / "residual_headroom_rows.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertEqual({int(row["world_seed"]) for row in rows}, {456, 1456})
            self.assertEqual(load_json(output / "status.json")["persisted_rows"], 4)
            self.assertEqual(load_json(output / "status.json")["exit_code"], 0)
            self.assertFalse(result["ddpg_training_authorized"])
            self.assertFalse(result["observable_ranking_screen_authorized"])
            self.assertTrue(result["collaborator_review_pending"])
            for item in load_json(output / "artifact_inventory.json")["files"]:
                self.assertEqual(sha256(Path(item["path"])), item["sha256"])

    def test_csv_append_preserves_rows_and_one_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            write_csv([{"world": 1, "cost": 2.0}], path)
            write_csv([{"world": 2, "cost": 3.0}], path, append=True)
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows, [{"world": "1", "cost": "2.0"}, {"world": "2", "cost": "3.0"}])

    def test_failed_execution_preserves_claim_and_partial_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "test-only"
            config = load_json(DEFAULT_CONFIG)
            config["output_root"] = str(output)

            def fail(*args, **kwargs):
                self.assertTrue((output / "claim.json").is_file())
                write_csv([{"world": 123, "cost": 1.0}], output / "partial.csv")
                raise RuntimeError("synthetic failure before any scientific rollout")

            with mock.patch(f"{MODULE}.require_execution_authorization"), mock.patch(
                f"{MODULE}._execute_screen", side_effect=fail
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    run(config, config_path=DEFAULT_CONFIG)
            status = load_json(output / "status.json")
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["exit_code"], 1)
            self.assertTrue((output / "partial.csv").is_file())
            self.assertFalse((output / "summary.json").exists())

    def test_completed_execution_has_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "test-only"
            config = load_json(DEFAULT_CONFIG)
            config["output_root"] = str(output)
            with mock.patch(f"{MODULE}.require_execution_authorization"), mock.patch(
                f"{MODULE}._execute_screen", return_value={"decision": "unit-test"}
            ):
                run(config, config_path=DEFAULT_CONFIG)
            status = load_json(output / "status.json")
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["exit_code"], 0)
            self.assertFalse(status["execution_approvals"]["howard"])

    def test_existing_root_blocks_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_json(DEFAULT_CONFIG)
            config["output_root"] = tmp
            with mock.patch(f"{MODULE}.require_execution_authorization"), mock.patch(
                f"{MODULE}._execute_screen"
            ) as execute:
                with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                    run(config, config_path=DEFAULT_CONFIG)
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
