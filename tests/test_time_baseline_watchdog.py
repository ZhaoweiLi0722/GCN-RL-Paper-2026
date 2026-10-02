"""Invented signed ledgers and mocked processes; no scientific child is run."""

import copy
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, call, patch

from src.rl.candidate_pilot_resources import digest
from src.rl.time_baseline_plan import time_baseline_budget_plan
from src.rl.time_baseline_watchdog import TimeBaselineLedgerDeadline, supervise_time_baseline
from src.utils.research_clock import CLOCK_ID


MODULE = "src.rl.time_baseline_watchdog"


def tiny_plan():
    def limits(seconds):
        return {"trajectory": 0, "clone": 0, "actor": 0, "critic": 0, "seconds": seconds}
    phases = {"runtime_binding": limits(2), "invented_work": limits(10), "supervisor_closure": limits(4)}
    sections = {"runtime_input_binding": limits(2) | {"phase": "runtime_binding"},
                "invented_work/a": limits(5) | {"phase": "invented_work"},
                "invented_work/b": limits(5) | {"phase": "invented_work"},
                "supervisor_closure": limits(4) | {"phase": "supervisor_closure"}}
    return {"format": "dynamic-candidate-budget-plan-v1", "draft_sha256": "invented-only",
            "limits": limits(20), "phases": phases, "sections": sections}


def claim(plan, started=100.):
    def combine(row):
        return {"environment": row["trajectory"] + row["clone"],
                "optimizer": row["actor"] + row["critic"], "seconds": row["seconds"]}
    return {"event": "claim", "started": started, "clock_id": CLOCK_ID,
            "dynamic_plan": copy.deepcopy(plan), "limits": combine(plan["limits"]),
            "sections": {name: combine(row) | {"phase": row["phase"]} for name, row in plan["sections"].items()},
            "phase_limits": {kind: {name: combine(row)[kind] for name, row in plan["phases"].items()}
                             for kind in ("environment", "optimizer")}}


def complete_rows(plan):
    rows, phase_spent = [claim(plan)], {}
    for index, section in enumerate(plan["sections"]):
        start = 100. + index
        elapsed = .5 if section == "supervisor_closure" else 1.
        phase = plan["sections"][section]["phase"]
        phase_spent[phase] = phase_spent.get(phase, 0.) + elapsed
        rows.extend([{"event": "begin", "section": section, "clock": start},
                     {"event": "finish", "section": section, "clock": start + elapsed,
                      "seconds": elapsed, "phase_seconds": phase_spent[phase]}])
    return rows


def signed_bytes(rows):
    previous, lines = "0" * 64, []
    for sequence, row in enumerate(rows):
        payload = copy.deepcopy(row) | {"sequence": sequence, "previous": previous}
        previous = digest(payload)
        lines.append(json.dumps(payload | {"sha256": previous}, sort_keys=True).encode() + b"\n")
    return b"".join(lines)


class TimeBaselineDeadlineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root, self.plan = Path(directory.name), tiny_plan()
        self.path = self.root / "budget.jsonl"

    def watcher(self, rows=None):
        if rows is not None:
            self.path.write_bytes(signed_bytes(rows))
        return TimeBaselineLedgerDeadline(self.path, self.plan, 100.)

    def test_new_positive_closure_deadline_survives_finish_in_single_poll(self):
        watcher = self.watcher(complete_rows(self.plan))
        watcher.poll()
        self.assertEqual(watcher.closure_deadline, 107.)
        self.assertIsNone(watcher.active)
        self.assertIsNone(watcher.phase_deadline)
        self.assertIsNone(watcher.initial_deadline)
        self.assertEqual(watcher.closed, set(self.plan["sections"]))
        self.assertEqual(watcher.deadline(120.), 107.)
        self.assertEqual(watcher.deadline(106.), 106.)
        self.assertEqual(watcher.phase_spent, {"runtime_binding": 1., "invented_work": 2., "supervisor_closure": .5})
        watcher.poll()
        self.assertEqual(watcher.deadline(120.), 107.)

    def test_partial_lines_and_incremental_closure_preserve_chain_and_deadline(self):
        lines = signed_bytes(complete_rows(self.plan)).splitlines(keepends=True)
        self.path.write_bytes(b"".join(lines[:-2]) + lines[-2][:17])
        watcher = self.watcher()
        watcher.poll()
        self.assertIsNone(watcher.closure_deadline)
        self.assertTrue(watcher.partial)
        with self.path.open("ab") as handle:
            handle.write(lines[-2][17:])
        watcher.poll()
        self.assertEqual(watcher.closure_deadline, 107.)
        self.assertEqual(watcher.active, "supervisor_closure")
        with self.path.open("ab") as handle:
            handle.write(lines[-1])
        watcher.poll()
        self.assertEqual(watcher.deadline(120.), 107.)
        self.assertFalse(watcher.partial)
        self.assertFalse(watcher.time_baseline_partial)

    def test_global_setup_owner_and_aggregate_phase_deadlines(self):
        watcher = self.watcher()
        watcher.poll()
        self.assertEqual(watcher.deadline(120.), 102.)
        rows = complete_rows(self.plan)
        # Up through the first work owner: owner cap=106, aggregate cap=111.
        self.path.write_bytes(signed_bytes(rows[:4]))
        watcher.poll()
        self.assertEqual(watcher.phase_deadline, 111.)
        self.assertEqual(watcher.deadline(120.), 106.)
        with self.path.open("ab") as handle:
            handle.write(signed_bytes(rows)[len(signed_bytes(rows[:4])):len(signed_bytes(rows[:6]))])
        watcher.poll()
        self.assertEqual(watcher.phase_deadline, 111.)
        self.assertEqual(watcher.deadline(120.), 107.)
        self.assertEqual(watcher.deadline(103.), 103.)

    def test_setup_cannot_be_restarted_after_late_child_begin(self):
        rows = [claim(self.plan), {"event": "begin", "section": "runtime_input_binding", "clock": 102.5}]
        with self.assertRaisesRegex(TimeoutError, "setup"):
            self.watcher(rows).poll()

    def test_plan_origin_claim_and_aggregate_elapsed_mismatches_fail(self):
        cases = []
        rows = complete_rows(self.plan)
        rows[0]["started"] = 99.
        cases.append((rows, "origin"))
        rows = complete_rows(self.plan)
        rows[0]["dynamic_plan"]["draft_sha256"] = "different"
        cases.append((rows, "plan/origin"))
        for key in ("limits", "sections", "phase_limits"):
            rows = complete_rows(self.plan)
            if key == "limits":
                rows[0][key]["seconds"] += 1
            elif key == "sections":
                rows[0][key]["invented_work/a"]["seconds"] += 1
            else:
                rows[0][key]["environment"]["invented_work"] += 1
            cases.append((rows, "deadline claim"))
        for key in ("seconds", "phase_seconds"):
            rows = complete_rows(self.plan)
            rows[4][key] += .1
            cases.append((rows, "phase time"))
        for rows, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.watcher(rows).poll()

    def test_regressed_clock_hash_corruption_scope_overrun_and_clock_id(self):
        rows = complete_rows(self.plan)
        rows[3]["clock"] = 100.5
        with self.assertRaisesRegex(ValueError, "regressed"):
            self.watcher(rows).poll()
        rows = complete_rows(self.plan)
        rows[0]["clock_id"] = "injected-test-clock-not-cross-process"
        with self.assertRaisesRegex(ValueError, "shared cross-process"):
            self.watcher(rows).poll()
        rows = complete_rows(self.plan)[:5]
        rows[-1].update(clock=106.1, seconds=5.1, phase_seconds=5.1)
        with self.assertRaisesRegex(ValueError, "scope deadline"):
            self.watcher(rows).poll()
        raw = signed_bytes(complete_rows(self.plan)).replace(b'"sequence": 3', b'"sequence": 4', 1)
        self.path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "chain mismatch"):
            self.watcher().poll()

    def test_positive_exact_closure_binding_and_immutable_admitted_plan(self):
        for kind in ("old_name", "zero_closure", "wrong_phase", "invalid_origin"):
            plan, started = copy.deepcopy(self.plan), 100.
            if kind == "old_name":
                plan["sections"]["supervisor_dispatch_terminal_closure"] = plan["sections"].pop("supervisor_closure")
            elif kind == "zero_closure":
                plan["sections"]["supervisor_closure"]["seconds"] = 0
                plan["phases"]["supervisor_closure"]["seconds"] = 0
            elif kind == "wrong_phase":
                plan["phases"]["wrong"] = plan["phases"].pop("supervisor_closure")
                plan["sections"]["supervisor_closure"]["phase"] = "wrong"
            else:
                started = float("nan")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                TimeBaselineLedgerDeadline(self.path, plan, started)
        watcher = self.watcher()
        self.plan["sections"]["supervisor_closure"]["seconds"] = 999
        self.assertEqual(watcher.plan["sections"]["supervisor_closure"]["seconds"], 4)

    def test_real_proposal_is_accepted_without_executing_it(self):
        proposal = json.loads((Path(__file__).parents[1] /
            "specs/2026-10-02-time-baseline-comparison/proposal.json").read_text())
        plan = time_baseline_budget_plan(proposal)
        watcher = TimeBaselineLedgerDeadline(self.path, plan, 100.)
        self.assertEqual(watcher.initial_deadline, 400.)
        self.assertEqual(watcher.plan["sections"]["supervisor_closure"]["seconds"], 600)
        self.assertFalse(self.path.exists())


class TimeBaselineSupervisorTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root, self.plan = Path(directory.name), tiny_plan()
        self.clock = [100.]
        self.popen = self.enter_patch("subprocess.Popen")
        self.killpg = self.enter_patch("os.killpg")
        self.sleep = self.enter_patch("time.sleep")
        self.enter_patch("shared_monotonic", side_effect=lambda: self.clock[0])

    def enter_patch(self, name, **kwargs):
        guard = patch(MODULE + "." + name, **kwargs)
        self.addCleanup(guard.stop)
        return guard.start()

    def run_supervisor(self, **kwargs):
        options = {"cwd": self.root, "launcher": self.root, "plan": self.plan,
                   "started": 100., "poll_seconds": .1, "termination_grace_seconds": .2}
        return supervise_time_baseline(["invented-nonexecuting-child"], **(options | kwargs))

    def child(self, *, rows=None, code=0, running=False):
        process = Mock()
        process.pid = 432123
        process.returncode = None if running else code
        process.poll.side_effect = lambda: process.returncode

        def wait(timeout=None):
            if process.returncode is None:
                process.returncode = -signal.SIGTERM
            return process.returncode
        process.wait.side_effect = wait

        def launch(*args, **kwargs):
            if rows is not None:
                (self.root / "budget.jsonl").write_bytes(signed_bytes(rows))
                self.clock[0] = max(self.clock[0], max(row.get("clock", row.get("started", 0)) for row in rows))
            return process
        self.popen.side_effect = launch
        return process

    def receipt(self):
        return json.loads((self.root / "supervisor.json").read_text())

    def test_complete_mock_child_reaped_and_durable_receipt_preserves_closure(self):
        child = self.child(rows=complete_rows(self.plan))
        result = self.run_supervisor()
        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "child_exited")
        self.assertEqual(result["final_deadline"], 107.)
        self.assertEqual(result["closure_deadline"], 107.)
        self.assertEqual(result, self.receipt())
        self.assertEqual(result["last_ledger_sequence"], 9)
        self.assertFalse(result["automatic_retry"])
        self.popen.assert_called_once()
        self.assertTrue(self.popen.call_args.kwargs["start_new_session"])
        child.wait.assert_called_once_with()
        self.killpg.assert_not_called()
        self.assertTrue((self.root / "stdout.log").exists())
        self.assertTrue((self.root / "stderr.log").exists())

    def test_zero_exit_with_missing_budget_is_failure(self):
        child = self.child()
        result = self.run_supervisor()
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "missing_budget_claim")
        self.assertEqual(result, self.receipt())
        child.wait.assert_called_once_with()
        self.killpg.assert_not_called()

    def test_nonzero_exit_incomplete_scopes_and_partial_final_line(self):
        self.child(rows=complete_rows(self.plan), code=7)
        result = self.run_supervisor()
        self.assertEqual(result["reason"], "child_failed")
        self.assertEqual(result["exit_code"], 7)
        self.assertFalse(result["passed"])

    def test_incomplete_scopes_do_not_count_as_completion(self):
        self.child(rows=complete_rows(self.plan)[:3])
        result = self.run_supervisor()
        self.assertEqual(result["reason"], "incomplete_budget_scopes")
        self.assertFalse(result["passed"])

    def test_partial_final_line_does_not_count_as_completion(self):
        child = self.child()
        def launch(*args, **kwargs):
            (self.root / "budget.jsonl").write_bytes(signed_bytes(complete_rows(self.plan)) + b'{"event":')
            self.clock[0] = 104.
            return child
        self.popen.side_effect = launch
        result = self.run_supervisor()
        self.assertEqual(result["reason"], "incomplete_final_budget_line")
        self.assertFalse(result["passed"])

    def test_setup_overrun_prevents_child_launch_and_records_failure(self):
        self.clock[0] = 102.
        result = self.run_supervisor()
        self.popen.assert_not_called()
        self.killpg.assert_not_called()
        self.assertEqual(result["reason"], "watchdog_or_launch_error")
        self.assertIn("setup deadline", result["error"])
        self.assertIsNone(result["pid"])
        self.assertEqual(result, self.receipt())

    def test_launch_failure_is_durable_and_not_retried(self):
        self.popen.side_effect = OSError("invented launch failure")
        result = self.run_supervisor()
        self.popen.assert_called_once()
        self.killpg.assert_not_called()
        self.assertIn("invented launch failure", result["error"])
        self.assertFalse(result["passed"])
        self.assertEqual(result, self.receipt())

    def test_timeout_terminates_only_owned_new_group_and_reaps(self):
        child = self.child(running=True)
        self.sleep.side_effect = lambda _: self.clock.__setitem__(0, 102.)
        result = self.run_supervisor()
        self.assertEqual(result["reason"], "wall_clock_deadline")
        self.killpg.assert_called_once_with(child.pid, signal.SIGTERM)
        child.wait.assert_called_once_with(timeout=.2)
        self.assertFalse(result["forced_kill"])
        self.assertEqual(result["exit_code"], -signal.SIGTERM)
        self.assertEqual(result, self.receipt())
        self.popen.assert_called_once()

    def test_forced_kill_after_bounded_grace_has_no_retry_and_reaps(self):
        child = self.child(running=True)
        def wait(timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("invented-child", timeout)
            child.returncode = -signal.SIGKILL
            return child.returncode
        child.wait.side_effect = wait
        self.sleep.side_effect = lambda _: self.clock.__setitem__(0, 102.)
        result = self.run_supervisor()
        self.assertEqual(self.killpg.call_args_list, [call(child.pid, signal.SIGTERM), call(child.pid, signal.SIGKILL)])
        self.assertEqual(child.wait.call_args_list, [call(timeout=.2), call()])
        self.assertTrue(result["forced_kill"])
        self.assertEqual(result["exit_code"], -signal.SIGKILL)
        self.assertFalse(result["automatic_retry"])
        self.assertEqual(result, self.receipt())

    def test_exit_race_process_lookup_error_still_reaps(self):
        child = self.child(running=True)
        self.sleep.side_effect = lambda _: self.clock.__setitem__(0, 102.)
        self.killpg.side_effect = ProcessLookupError("invented exit race")
        result = self.run_supervisor()
        child.wait.assert_called_once_with(timeout=.2)
        self.assertFalse(result["passed"])
        self.assertEqual(result, self.receipt())

    def test_bad_live_ledger_kills_owned_child_and_records_error(self):
        rows = [claim(self.plan)]
        rows[0]["started"] = 99.
        child = self.child(rows=rows, running=True)
        result = self.run_supervisor()
        self.assertEqual(result["reason"], "watchdog_or_launch_error")
        self.assertIn("origin", result["error"])
        self.killpg.assert_called_once_with(child.pid, signal.SIGTERM)
        child.wait.assert_called_once_with(timeout=.2)
        self.assertEqual(result, self.receipt())

    def test_finished_closure_still_times_out_a_lingering_child(self):
        child = self.child(rows=complete_rows(self.plan), running=True)
        self.sleep.side_effect = lambda _: self.clock.__setitem__(0, 107.)
        result = self.run_supervisor()
        self.assertEqual(result["reason"], "wall_clock_deadline")
        self.assertEqual(result["closure_deadline"], 107.)
        self.assertEqual(result["final_deadline"], 107.)
        self.killpg.assert_called_once_with(child.pid, signal.SIGTERM)
        child.wait.assert_called_once_with(timeout=.2)

    def test_receipt_write_overrun_appends_failure_without_overwriting_original(self):
        from src.rl.candidate_pilot_recording import write_json_once
        self.child(rows=complete_rows(self.plan))
        def slow_receipt(path, data):
            write_json_once(path, data)
            self.clock[0] = 107.1
        self.enter_patch("write_json_once", side_effect=slow_receipt)
        result = self.run_supervisor()
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "supervisor_receipt_exceeded_deadline")
        self.assertTrue(self.receipt()["passed"])
        failure = json.loads((self.root / "supervisor-overrun.json").read_text())
        self.assertFalse(failure["passed"])
        self.assertFalse(failure["automatic_retry"])
        self.assertEqual(failure["closure_deadline"], 107.)
        self.assertEqual(failure["last_ledger_sha256"], self.receipt()["last_ledger_sha256"])

    def test_existing_evidence_prevents_relaunch_and_is_not_overwritten(self):
        for name in ("stdout.log", "stderr.log", "supervisor.json", "budget.jsonl", "supervisor-overrun.json"):
            directory = self.root / name.replace(".", "-")
            directory.mkdir()
            path = directory / name
            path.write_text("invented existing evidence")
            with self.subTest(name=name), self.assertRaises(FileExistsError):
                self.run_supervisor(launcher=directory)
            self.assertEqual(path.read_text(), "invented existing evidence")
        self.popen.assert_not_called()

    def test_bad_argument_timing_rejected_before_launch(self):
        for kwargs in ({"started": float("nan")}, {"started": True}, {"poll_seconds": 0},
                       {"poll_seconds": float("inf")}, {"termination_grace_seconds": 3},
                       {"termination_grace_seconds": -1}, {"termination_grace_seconds": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_supervisor(**kwargs)
        self.popen.assert_not_called()
        self.assertFalse((self.root / "supervisor.json").exists())


if __name__ == "__main__":
    unittest.main()
