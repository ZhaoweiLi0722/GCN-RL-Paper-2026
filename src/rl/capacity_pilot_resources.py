"""Single-use durable counters for the approved capacity comparison.

Prediction/filter work is prepaid in fixed chunks, then marked completed.
An interrupted chunk remains consumed with its exact completion count unknown;
it is never refunded or mislabeled as fully executed. Native and optimizer
calls have individual pre-invocation receipts. No scientific operation lives here.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import resource
import sys

from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_adaptation_campaign import numeric_contract
from src.utils.research_clock import shared_monotonic


class CapacityPilotBudget:
    format = "capacity-pilot-budget-v1"

    def __init__(self, root, proposal, *, clock=shared_monotonic, started=None):
        self.root, self.proposal, self.clock = Path(root), copy.deepcopy(proposal), clock
        self.contract = numeric_contract(proposal)
        self.limits = self.contract["limits"]
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {k: dict.fromkeys(self.limits, 0) for k in self.contract["phase_limits"]}
        self.started = self.last_clock = clock() if started is None else started
        self.times = dict.fromkeys(proposal["proposed_budget"]["time_seconds"], 0.0)
        self.owner, self.phase, self.failed = "admission_lock_binding", None, False
        self.chunks, self.executed_chunks = {}, dict(planner_total_model_epochs=0, estimator_hypothesis_transitions=0)
        self.previous, self.sequence = "0" * 64, 0
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root / "budget.jsonl").open("x", encoding="utf8")
        self._append({"event": "claim", "pid": os.getpid(), "started": self.started,
                      "proposal_sha256": digest(proposal), "limits": self.limits})

    def _append(self, payload):
        row = dict(payload, sequence=self.sequence, previous=self.previous)
        seal = digest(row)
        try:
            self.handle.write(json.dumps(dict(row, sha256=seal), sort_keys=True, allow_nan=False) + "\n")
            self.handle.flush()
            os.fsync(self.handle.fileno())
        except BaseException:
            self.failed = True
            raise
        self.sequence += 1
        self.previous = seal

    def check(self, *, storage=False):
        if self.failed:
            raise RuntimeError("closed capacity budget")
        now = self.clock()
        if not math.isfinite(now) or now < self.last_clock:
            self.failed = True
            raise RuntimeError("invalid budget clock")
        self.times[self.owner] += now - self.last_clock
        self.last_clock = now
        caps = self.proposal["proposed_budget"]
        if now - self.started > caps["time_seconds"]["global"] or self.times[self.owner] > caps["time_seconds"][self.owner]:
            self.failed = True
            raise TimeoutError("declared phase/global wall cap including recording exhausted")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > caps["rss_cap_bytes"]:
            self.failed = True
            raise MemoryError("declared RSS cap exhausted")
        if storage:
            files = [p for p in self.root.parent.rglob("*") if p.is_file()]
            archived = [p for p in files if "archives" in p.relative_to(self.root.parent).parts]
            total = sum(p.stat().st_size for p in files)
            archive_bytes = sum(p.stat().st_size for p in archived)
            if (len(files) > caps["file_cap"] or total-archive_bytes > caps["raw_artifact_cap_bytes"]
                    or archive_bytes > caps["archive_cap_bytes"] or total > caps["combined_disk_cap_bytes"]):
                self.failed = True
                raise RuntimeError("raw storage/file cap exhausted")
        return now

    def enter(self, owner, phase=None):
        now = self.check(storage=True)
        if owner not in self.times or owner == "global" or phase is not None and phase not in self.phase_counts:
            raise ValueError("unknown budget owner/phase")
        if self.chunks:
            raise RuntimeError("unfinished query reservation at phase boundary")
        self.owner, self.phase = owner, phase
        self._append(dict(event="enter", owner=owner, phase=phase, clock=now, owner_times=dict(self.times)))

    def debit(self, charges):
        now = self.check()
        if self.phase is None:
            raise RuntimeError("no scientific phase admitted")
        phase_caps = self.contract["phase_limits"][self.phase]
        for k, amount in charges.items():
            if (k not in self.limits or type(amount) is not int or amount <= 0
                    or self.counts[k] + amount > self.limits[k]
                    or k in phase_caps and self.phase_counts[self.phase][k] + amount > phase_caps[k]):
                self.failed = True
                raise RuntimeError(f"declared counter exhausted: {k}")
        self._append(dict(event="debit", charges=charges, phase=self.phase, owner=self.owner, clock=now))
        for k, amount in charges.items():
            self.counts[k] += amount
            self.phase_counts[self.phase][k] += amount

    def chunk_call(self, kind, total, width):
        """Called BEFORE each real primitive; logs prepaid fixed chunks."""
        self.check()
        if kind not in self.executed_chunks or type(total) is not int or total <= 0:
            raise ValueError("explicit model/filter chunk kind required")
        if kind not in self.chunks:
            self.debit({kind: total})
            self.chunks[kind] = {"reserved": total, "dispatched": 0}
        chunk = self.chunks[kind]
        if total != chunk["reserved"] or type(width) is not int or width <= 0 or chunk["dispatched"] + width > total:
            self.failed = True
            raise ValueError("invalid prepaid query progress")
        chunk["dispatched"] += width

    def finish_chunk(self, kind):
        row = self.chunks[kind]
        if row["dispatched"] != row["reserved"]:
            raise ValueError("incomplete primitive batch cannot be declared finished")
        self.check()
        self._append(dict(event="query_chunk_completed", kind=kind, amount=row["reserved"], phase=self.phase))
        self.executed_chunks[kind] += row["reserved"]
        del self.chunks[kind]

    def snapshot(self):
        return copy.deepcopy(dict(format=self.format, counts=self.counts, phases=self.phase_counts,
            times=self.times, elapsed=self.last_clock-self.started, owner=self.owner, phase=self.phase,
            failed=self.failed, pending_chunks=self.chunks, verified_completed_chunk_calls=self.executed_chunks,
            ledger_sha256=self.previous, events=self.sequence))

    def close(self):
        self.handle.close()
        self.failed = True
