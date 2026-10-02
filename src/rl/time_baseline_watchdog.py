"""Single owned-child watchdog for the versioned time-baseline schedule.

Reuse the immutable dynamic ledger's hash chain, owner/setup/phase clocks and
deadline minimum. Only the new terminal section binding is added. This utility
does not admit scientific work, retry it, or load a model/environment.
"""

import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from src.rl.candidate_pilot_recording import write_json_once
from src.rl.dynamic_candidate_resources import _combined
from src.rl.dynamic_candidate_watchdog import DynamicLedgerDeadline
from src.utils.research_clock import CLOCK_ID, shared_monotonic


CLOSURE = "supervisor_closure"


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


class TimeBaselineLedgerDeadline(DynamicLedgerDeadline):
    """Add the exact new closure name without patching or renaming a ledger.

    The closure phase has exactly one owner. Its begin clock fixes a deadline
    that survives finish, including when one poll reads the entire ledger.
    """

    def __init__(self, path, plan, started):
        if not _finite_number(started) or started < 0:
            raise ValueError("finite nonnegative supervisor origin required")
        super().__init__(path, copy.deepcopy(plan), started)
        sections, phases = self.plan["sections"], self.plan["phases"]
        if (CLOSURE not in sections or sections[CLOSURE]["phase"] != CLOSURE
                or "supervisor_dispatch_terminal_closure" in sections
                or [name for name, row in sections.items() if row["phase"] == CLOSURE] != [CLOSURE]
                or sections[CLOSURE]["seconds"] <= 0 or phases[CLOSURE]["seconds"] <= 0
                or sections["runtime_input_binding"]["seconds"] <= 0
                or self.plan["limits"]["seconds"] <= 0):
            raise ValueError("positive setup and exact single supervisor_closure owner required")
        self.time_baseline_partial = b""

    def poll(self):
        offset = self.offset
        super().poll()
        if self.claim is not None:
            sections = {name: _combined(row) | {"phase": row["phase"]}
                        for name, row in self.plan["sections"].items()}
            phases = {kind: {name: _combined(row)[kind] for name, row in self.plan["phases"].items()}
                      for kind in ("environment", "optimizer")}
            if (self.claim["limits"] != _combined(self.plan["limits"])
                    or self.claim["sections"] != sections or self.claim["phase_limits"] != phases):
                raise ValueError("child deadline claim differs from admitted plan")
        if self.offset == offset:
            return
        with self.path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read(self.offset - offset)
        lines = (self.time_baseline_partial + raw).split(b"\n")
        self.time_baseline_partial = lines.pop()
        for line in lines:
            row = json.loads(line)
            if row["event"] == "begin" and row["section"] == CLOSURE:
                self.closure_deadline = row["clock"] + min(
                    self.plan["sections"][CLOSURE]["seconds"], self.plan["phases"][CLOSURE]["seconds"])


def supervise_time_baseline(command, *, cwd, launcher, plan, started,
                            poll_seconds=.1, termination_grace_seconds=1.):
    """Supervise one new session, retain deadlines through receipt I/O, no retry.

    The admission wrapper owns authorization, source/runtime locks and exclusive
    attempt claiming. Only a group created by this Popen call is ever signaled.
    Poll scheduling and the bounded termination grace can delay process cleanup;
    the receipt records actual elapsed time and never treats an overrun as pass.
    """
    if (not isinstance(command, list) or not command
            or any(not isinstance(value, str) or not value for value in command)
            or not _finite_number(started) or started < 0
            or not _finite_number(poll_seconds) or not 0 < poll_seconds <= 1
            or not _finite_number(termination_grace_seconds) or not 0 <= termination_grace_seconds <= 2):
        raise ValueError("explicit bounded supervisor arguments required")
    launcher = Path(launcher)
    watcher = TimeBaselineLedgerDeadline(launcher / "budget.jsonl", plan, started)
    plan = watcher.plan
    deadline = started + plan["limits"]["seconds"]
    paths = [launcher / name for name in ("stdout.log", "stderr.log", "supervisor.json")]
    evidence = paths + [watcher.path, launcher / "supervisor-overrun.json"]
    if any(path.exists() or path.is_symlink() for path in evidence):
        raise FileExistsError("supervisor evidence must be fresh; no retry or ledger adoption")
    child, reason, error, forced = None, None, None, False
    launcher.mkdir(parents=True, exist_ok=True)
    with paths[0].open("xb") as stdout, paths[1].open("xb") as stderr:
        try:
            now = shared_monotonic()
            if now < started:
                raise ValueError("supervisor origin is in the future")
            if now >= watcher.deadline(deadline):
                raise TimeoutError("setup deadline exceeded before child")
            child = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True)
            while child.poll() is None:
                watcher.poll()
                if shared_monotonic() >= watcher.deadline(deadline):
                    reason = "wall_clock_deadline"
                    break
                time.sleep(poll_seconds)
            if reason is None:
                watcher.poll()
                reason = "child_exited" if child.returncode == 0 else "child_failed"
                if watcher.claim is None:
                    reason = "missing_budget_claim"
                elif watcher.partial:
                    reason = "incomplete_final_budget_line"
                elif set(watcher.closed) != set(plan["sections"]) or watcher.active is not None:
                    reason = "incomplete_budget_scopes"
                if shared_monotonic() > watcher.deadline(deadline):
                    reason = "wall_clock_deadline"
        except BaseException as exc:
            reason, error = "watchdog_or_launch_error", repr(exc)
        finally:
            if child is not None:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=termination_grace_seconds)
                    except subprocess.TimeoutExpired:
                        forced = True
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
                else:
                    child.wait()
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
    result = {"format": "time-baseline-supervisor-v1", "command": list(command), "cwd": str(cwd),
        "pid": None if child is None else child.pid, "ppid": os.getpid(),
        "exit_code": None if child is None else child.returncode, "reason": reason, "error": error,
        "forced_kill": forced, "clock_id": CLOCK_ID, "started": started,
        "elapsed_seconds": shared_monotonic() - started, "maximum_seconds": plan["limits"]["seconds"],
        "poll_seconds": poll_seconds, "termination_grace_seconds": termination_grace_seconds,
        "final_deadline": watcher.deadline(deadline), "closure_deadline": watcher.closure_deadline,
        "active_scope": watcher.active, "last_ledger_sequence": watcher.sequence,
        "last_ledger_sha256": watcher.previous, "automatic_retry": False,
        "passed": reason == "child_exited" and child is not None and child.returncode == 0}
    write_json_once(paths[2], result)
    if result["passed"] and shared_monotonic() > watcher.deadline(deadline):
        # Preserve the original receipt and append the authoritative overrun
        # failure rather than overwriting evidence after its own fsync deadline.
        result["passed"] = False
        result["reason"] = "supervisor_receipt_exceeded_deadline"
        write_json_once(launcher / "supervisor-overrun.json", {
            "format": "time-baseline-supervisor-overrun-v1", "status": "failed", "passed": False,
            "reason": result["reason"], "automatic_retry": False, "final_deadline": watcher.deadline(deadline),
            "closure_deadline": watcher.closure_deadline, "last_ledger_sha256": watcher.previous})
    return result
