"""Outer single-child deadline watchdog, independent of learner/native calls.

The scientific entrypoint must enforce source/config locks and exclusive claim
before binding this utility. This module does not authorize or retry a campaign.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from src.rl.candidate_pilot_resources import digest


class LedgerDeadline:
    """Read a live append-only ledger, tolerating only an unfinished last line."""

    def __init__(self, path):
        self.path = Path(path)
        self.offset, self.partial, self.sequence, self.previous = 0, b"", 0, "0" * 64
        self.claim, self.active, self.scope_deadline = None, None, None
        self.last_clock, self.inode, self.closed = None, None, set()

    def poll(self):
        if not self.path.exists():
            if self.offset:
                raise ValueError("budget ledger disappeared")
            return
        stat = self.path.stat()
        identity = stat.st_dev, stat.st_ino
        if self.path.is_symlink() or stat.st_size < self.offset or (self.inode is not None and self.inode != identity):
            raise ValueError("budget ledger replaced, truncated or symlinked")
        self.inode = identity
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
        self.offset += len(chunk)
        lines = (self.partial + chunk).split(b"\n")
        self.partial = lines.pop()
        for line in lines:
            row = json.loads(line)
            seal = row.pop("sha256")
            if row["sequence"] != self.sequence or row["previous"] != self.previous or digest(row) != seal:
                raise ValueError("live budget chain mismatch")
            event = row["event"]
            if self.sequence == 0:
                if event != "claim":
                    raise ValueError("missing initial budget claim")
                self.claim, self.last_clock = row, row["started"]
                if not math.isfinite(self.last_clock) or self.claim["limits"]["seconds"] <= 0:
                    raise ValueError("invalid initial budget deadline")
            else:
                now = row["clock"]
                if not math.isfinite(now) or now < self.last_clock:
                    raise ValueError("live budget clock regressed")
                self.last_clock = now
                if event == "begin":
                    if self.active is not None or row["section"] in self.closed:
                        raise ValueError("concurrent/repeated live scope")
                    self.active = row["section"]
                    self.scope_deadline = now + self.claim["sections"][self.active]["seconds"]
                elif event in ("debit", "finish"):
                    if self.active is None or row["section"] != self.active:
                        raise ValueError("live event outside scope")
                    if now > self.scope_deadline:
                        raise ValueError("live event exceeded scope deadline")
                    if event == "finish":
                        self.closed.add(self.active)
                        self.active = self.scope_deadline = None
                else:
                    raise ValueError("unknown live budget event")
            self.previous, self.sequence = seal, self.sequence + 1

    def deadline(self, supervisor_deadline):
        values = [supervisor_deadline]
        if self.claim is not None:
            values.append(self.claim["started"] + self.claim["limits"]["seconds"])
        if self.scope_deadline is not None:
            values.append(self.scope_deadline)
        return min(values)


def supervise(command, *, cwd, stdout_path, stderr_path, ledger_path, report_path,
              maximum_seconds, poll_seconds=.1, termination_grace_seconds=1.):
    """Run once; reap the exact owned process group on timeout/monitor error.

    Deadline detection can be delayed by OS scheduling and the polling interval;
    termination has the reported bounded grace. Every simulator/optimizer call
    still requires the child's pre-debit. There is no retry or resume branch.
    """
    if (not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command)
            or not math.isfinite(maximum_seconds) or maximum_seconds <= 0
            or not 0 < poll_seconds <= 1 or not 0 <= termination_grace_seconds <= 2):
        raise ValueError("explicit bounded command and watchdog timing required")
    paths = list(map(Path, (stdout_path, stderr_path, report_path)))
    if len(set(map(str, paths))) != 3 or any(p.exists() for p in paths):
        raise FileExistsError("watchdog evidence paths must be fresh and distinct")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    watcher, child, reason, failure, forced = LedgerDeadline(ledger_path), None, None, None, False
    with paths[0].open("xb") as stdout, paths[1].open("xb") as stderr:
        try:
            child = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True)
            while child.poll() is None:
                watcher.poll()
                if time.monotonic() >= watcher.deadline(started + maximum_seconds):
                    reason = "wall_clock_deadline"
                    break
                time.sleep(poll_seconds)
            if reason is None:
                watcher.poll()
                reason = "child_exited" if child.returncode == 0 else "child_failed"
                if watcher.partial:
                    reason = "incomplete_final_budget_line"
                if time.monotonic() > watcher.deadline(started + maximum_seconds):
                    reason = "wall_clock_deadline"
        except BaseException as exc:
            reason, failure = "watchdog_or_launch_error", repr(exc)
        finally:
            if child is not None and child.poll() is None:
                # New session ownership makes this group specific to this launch.
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
    result = {"format": "candidate-pilot-supervisor-v1", "command": command, "cwd": str(cwd),
              "pid": None if child is None else child.pid, "ppid": os.getpid(),
              "exit_code": None if child is None else child.returncode, "reason": reason,
              "error": failure, "forced_kill": forced, "elapsed_seconds": time.monotonic() - started,
              "maximum_seconds": maximum_seconds, "poll_seconds": poll_seconds,
              "termination_grace_seconds": termination_grace_seconds,
              "last_ledger_sequence": watcher.sequence, "last_ledger_sha256": watcher.previous,
              "active_scope": watcher.active, "automatic_retry": False,
              "passed": reason == "child_exited" and child is not None and child.returncode == 0}
    with paths[2].open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    return result
