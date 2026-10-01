"""Independent single-child deadlines for the dynamic owner's ledger.

Historical watchdog code is locked. Reuse its incremental chain reader while
adding aggregate phase/setup/closure deadlines for this new schedule.
"""

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_watchdog import LedgerDeadline
from src.rl.dynamic_candidate_resources import validate_budget_plan
from src.utils.research_clock import CLOCK_ID, shared_monotonic


class DynamicLedgerDeadline(LedgerDeadline):
    def __init__(self, path, plan, started):
        super().__init__(path)
        validate_budget_plan(plan)
        self.plan, self.origin = plan, started
        self.initial_deadline = started + plan["sections"]["runtime_input_binding"]["seconds"]
        self.phase_deadline = self.closure_deadline = None
        self.phase_spent, self.dynamic_active, self.dynamic_start, self.extra_partial = {}, None, None, b""

    def poll(self):
        offset = self.offset
        super().poll()
        if self.offset == offset:
            return
        with self.path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read(self.offset - offset)
        lines = (self.extra_partial + raw).split(b"\n")
        self.extra_partial = lines.pop()
        for line in lines:
            row = json.loads(line)
            if row["event"] == "claim":
                if row.get("dynamic_plan") != self.plan or row["started"] != self.origin:
                    raise ValueError("child ledger differs from the admitted plan/origin")
                continue
            now = row["clock"]
            if self.initial_deadline is not None and now > self.initial_deadline:
                raise TimeoutError("input binding including supervisor setup exceeded its cap")
            if row["event"] == "begin":
                self.dynamic_active, self.dynamic_start = row["section"], now
                phase = self.plan["sections"][self.dynamic_active]["phase"]
                self.phase_deadline = now + self.plan["phases"][phase]["seconds"] - self.phase_spent.get(phase, 0.)
                if self.dynamic_active == "supervisor_dispatch_terminal_closure":
                    self.closure_deadline = min(self.phase_deadline,
                        now + self.plan["sections"][self.dynamic_active]["seconds"])
            else:
                if self.phase_deadline is None or now > self.phase_deadline:
                    raise TimeoutError("aggregate dynamic phase deadline exceeded")
                if row["event"] == "finish":
                    phase = self.plan["sections"][self.dynamic_active]["phase"]
                    elapsed = now - self.dynamic_start
                    total = self.phase_spent.get(phase, 0.) + elapsed
                    if row["seconds"] != elapsed or row["phase_seconds"] != total:
                        raise ValueError("aggregate phase time receipt differs")
                    self.phase_spent[phase] = total
                    if self.dynamic_active == "runtime_input_binding":
                        self.initial_deadline = None
                    self.dynamic_active, self.dynamic_start, self.phase_deadline = None, None, None

    def deadline(self, supervisor_deadline):
        values = [super().deadline(supervisor_deadline)]
        values.extend(v for v in (self.initial_deadline, self.phase_deadline, self.closure_deadline) if v is not None)
        return min(values)


def supervise_dynamic(command, *, cwd, launcher, plan, started, poll_seconds=.1, termination_grace_seconds=1.):
    """Supervise and reap exactly the one owned process group, with no retry."""
    if (not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command)
            or not math.isfinite(started) or not 0 < poll_seconds <= 1
            or not 0 <= termination_grace_seconds <= 2):
        raise ValueError("explicit bounded supervisor arguments required")
    launcher = Path(launcher)
    watcher = DynamicLedgerDeadline(launcher / "budget.jsonl", plan, started)
    deadline = started + plan["limits"]["seconds"]
    paths = [launcher / name for name in ("stdout.log", "stderr.log", "supervisor.json")]
    if any(p.exists() or p.is_symlink() for p in paths):
        raise FileExistsError("supervisor evidence must be fresh")
    child, reason, error, forced = None, None, None, False
    launcher.mkdir(parents=True, exist_ok=True)
    with paths[0].open("xb") as stdout, paths[1].open("xb") as stderr:
        try:
            if shared_monotonic() >= watcher.deadline(deadline):
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
            if child is not None and child.poll() is None:
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
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
    result = {"format": "dynamic-candidate-supervisor-v1", "command": command, "cwd": str(cwd),
        "pid": None if child is None else child.pid, "ppid": os.getpid(),
        "exit_code": None if child is None else child.returncode, "reason": reason, "error": error,
        "forced_kill": forced, "clock_id": CLOCK_ID, "started": started,
        "elapsed_seconds": shared_monotonic() - started, "maximum_seconds": plan["limits"]["seconds"],
        "poll_seconds": poll_seconds, "termination_grace_seconds": termination_grace_seconds,
        "final_deadline": watcher.deadline(deadline), "last_ledger_sequence": watcher.sequence,
        "last_ledger_sha256": watcher.previous, "automatic_retry": False,
        "passed": reason == "child_exited" and child is not None and child.returncode == 0}
    write_json_once(paths[2], result)
    if result["passed"] and shared_monotonic() > watcher.deadline(deadline):
        result["passed"] = False
        result["reason"] = "supervisor_receipt_exceeded_deadline"
        write_json_once(launcher / "supervisor-overrun.json", {
            "status": "failed", "reason": result["reason"], "automatic_retry": False})
    return result
