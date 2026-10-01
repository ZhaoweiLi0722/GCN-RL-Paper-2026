"""Prospective streams and durable, non-refundable P1 resource accounting."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from src.utils.research_clock import CLOCK_ID, INJECTED_CLOCK_ID, shared_monotonic


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def stream_manifest(config):
    rng = config["rng"]
    namespace = rng["namespace"]
    base = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:12], "big") << 16
    ranges = rng["ordinal_ranges"]
    blocks = config["blocks"]
    counts = {"demonstration": config["initialization"].get("demonstration_episodes_per_block", 0),
              "qualification": config["initialization"]["qualification_episodes_per_block"],
              "training": config["ppo"]["episodes_per_model"],
              "test": config["evaluation"]["episodes_per_policy"]}
    environment = {}
    for role, (start, end) in ranges.items():
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < 2**16:
            raise ValueError("invalid explicit stream ordinal range")
        seeds = [base + i for i in range(start, end + 1)]
        if role == "preflight":
            environment[role] = seeds
        else:
            count = counts[role]
            if len(seeds) != count * len(blocks):
                raise ValueError("stream ordinal range/count mismatch")
            environment[role] = {str(b): seeds[i * count:(i + 1) * count] for i, b in enumerate(blocks)}
    neural = {}
    for template in rng["neural_role_paths"]:
        paths = [template] if "{" not in template else [
            template.format(b=b, representation=r["name"]) for b in blocks for r in config["representations"]]
        for path in paths:
            if path in neural:
                raise ValueError("duplicate neural stream role")
            neural[path] = int.from_bytes(hashlib.sha256((namespace + "/" + path).encode()).digest()[:8], "big") % 2**63
    result = {"format": "candidate-pilot-streams-v1", "namespace": namespace, "environment": environment,
              "model_initialization": dict(zip(map(str, blocks), config["policy_init_seeds"])), "neural": neural}
    values = numeric_leaves(result)
    # Deliberate arm pairing refers to these unique streams; it is not duplicated here.
    if len(values) != len(set(values)):
        raise ValueError("cross-purpose stream collision")
    return result


def numeric_leaves(data):
    if type(data) is int:
        return [data]
    if type(data) is float and math.isfinite(data) and data.is_integer():
        return [int(data)]
    if isinstance(data, str) and data.isascii() and data.isdecimal():
        return [int(data)]
    if isinstance(data, dict):
        return [v for item in data.values() for v in numeric_leaves(item)]
    if isinstance(data, (tuple, list)):
        return [v for item in data for v in numeric_leaves(item)]
    return []


def audit_stream_collisions(manifest, files):
    """Conservative numeric scan, not just filenames or namespace strings.

    The caller supplies the complete tracked-config/prior-local-seed-manifest
    inventory, excluding this prospective packet. Parse failures are errors,
    not silent exclusions. This does not prove coverage of unavailable files.
    """
    wanted = set(numeric_leaves(manifest))
    records, collisions = [], []
    for filename in sorted(set(map(str, files))):
        path = Path(filename)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or symlinked seed evidence: {path}")
        raw = path.read_bytes()
        if path.suffix == ".jsonl":
            data = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        elif path.suffix == ".json":
            data = json.loads(raw)
        else:
            raise ValueError(f"unhandled seed evidence format: {path}")
        hits = sorted(wanted.intersection(numeric_leaves(data)))
        records.append({"path": filename, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        if hits:
            collisions.append({"path": filename, "values": hits})
    if not records:
        raise ValueError("empty collision evidence inventory")
    return {"format": "candidate-stream-collision-audit-v1", "manifest_sha256": digest(manifest),
            "files": records, "collisions": collisions, "passed": not collisions}


def budget_sections(config):
    caps = config["caps"]
    time_limits = caps["seconds"]
    sections = {}
    prior = config.get("pilot_profile") == "p2_reference_prior"
    phases = [("preflight_including_clones", "preflight_total"), ("qualification", "qualification_total")]
    if not prior:
        phases.insert(1, ("demonstrations", "demonstrations_total"))
    for phase, time_name in phases:
        sections[phase] = {"phase": phase, "environment": caps["environment_steps"][phase],
                           "optimizer": 0, "seconds": time_limits[time_name]}
    for block in config["blocks"]:
        for representation in config["representations"]:
            key = f"block{block}/{representation['name']}"
            if not prior:
                sections[key + "/initialization"] = {"phase": "initialization", "environment": 0,
                    "optimizer": config["initialization"]["optimizer_steps_per_model"],
                    "seconds": time_limits["initialization_per_model"]}
            for role in ("ppo", "bc_continue"):
                sections[key + "/" + role] = {"phase": role,
                    "environment": caps["per_continuation_model_environment_steps"],
                    "optimizer": config[role]["max_optimizer_steps"],
                    "seconds": time_limits["continuation_per_model"]}
            for role in config["candidate_roles"]:
                sections[key + "/" + role + "/evaluation"] = {"phase": "evaluation",
                    "environment": caps["per_evaluation_policy_environment_steps"], "optimizer": 0,
                    "seconds": time_limits["evaluation_per_policy"]}
        for role in config["reference_roles"]:
            sections[f"block{block}/{role}/evaluation"] = {"phase": "evaluation",
                "environment": caps["per_evaluation_policy_environment_steps"], "optimizer": 0,
                "seconds": time_limits["evaluation_per_policy"]}
    return sections


class PilotBudget:
    """Single-owner append-only ledger. No refund, resume or counter-reset API.

    A debit must return successfully before the operation starts. A partial
    write or timeout poisons this owner; scientific failure must close the run.
    Session checkpoint restoration never restores this external budget.
    """

    def __init__(self, path, config, *, clock=None):
        self.path, self.clock = Path(path), shared_monotonic if clock is None else clock
        self.clock_id = CLOCK_ID if clock is None else INJECTED_CLOCK_ID
        caps = config["caps"]
        self.limits = {"environment": caps["maximum_environment_steps"],
                       "optimizer": caps["maximum_optimizer_steps"], "seconds": caps["maximum_seconds"]}
        self.phase_limits = {"environment": caps["environment_steps"], "optimizer": caps["optimizer_steps"]}
        self.sections = budget_sections(config)
        for row in [self.limits, *self.sections.values()]:
            if any(type(row[k]) is not int or row[k] < 0 for k in ("environment", "optimizer", "seconds")):
                raise ValueError("explicit nonnegative integer resource limits required")
        self.started = self.last_clock = self.clock()
        if not math.isfinite(self.started):
            raise ValueError("invalid monotonic clock")
        self.active, self.section_started, self.closed = None, None, []
        self.counts = {"environment": 0, "optimizer": 0}
        self.phases, self.scopes = {}, {}
        self.previous, self.sequence, self.failed = "0" * 64, 0, False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x", encoding="utf-8")
        self._append({"event": "claim", "limits": self.limits, "sections": self.sections,
                      "phase_limits": self.phase_limits, "started": self.started, "clock_id": self.clock_id})

    def _append(self, event):
        if self.failed:
            raise RuntimeError("budget owner is failed/closed")
        row = event | {"sequence": self.sequence, "previous": self.previous}
        sealed = row | {"sha256": digest(row)}
        try:
            self.handle.write(json.dumps(sealed, sort_keys=True, allow_nan=False) + "\n")
            self.handle.flush()
            os.fsync(self.handle.fileno())
        except Exception:
            self.failed = True
            raise
        self.previous, self.sequence = sealed["sha256"], self.sequence + 1

    def check(self):
        if self.failed:
            raise RuntimeError("budget owner is failed/closed")
        now = self.clock()
        invalid = (not math.isfinite(now) or now < self.last_clock or now - self.started > self.limits["seconds"])
        if self.active is not None:
            invalid |= now - self.section_started > self.sections[self.active]["seconds"]
        if invalid:
            self.failed = True
            raise TimeoutError("declared wall-clock budget exceeded or clock regressed")
        self.last_clock = now
        return now

    def begin(self, section):
        now = self.check()
        if self.active is not None or section not in self.sections or section in self.closed:
            raise ValueError("unknown/repeated section or concurrent budget scope")
        self._append({"event": "begin", "section": section, "clock": now})
        self.active, self.section_started = section, now

    def debit(self, kind):
        now = self.check()
        if self.active is None or kind not in self.counts:
            raise ValueError("active section and explicit resource kind required")
        section, limits = self.active, self.sections[self.active]
        phase = limits["phase"]
        scope_key, phase_key = f"{section}:{kind}", f"{phase}:{kind}"
        count = self.counts[kind] + 1
        scope_count, phase_count = self.scopes.get(scope_key, 0) + 1, self.phases.get(phase_key, 0) + 1
        if (count > self.limits[kind] or scope_count > limits[kind]
                or phase_count > self.phase_limits[kind].get(phase, 0)):
            self.failed = True
            raise RuntimeError("declared resource cap exceeded; attempt closed")
        self._append({"event": "debit", "section": section, "phase": phase, "resource": kind,
                      "total": count, "scope_total": scope_count, "phase_total": phase_count, "clock": now})
        self.counts[kind], self.scopes[scope_key], self.phases[phase_key] = count, scope_count, phase_count

    def finish(self):
        now = self.check()
        if self.active is None:
            raise ValueError("no active budget scope")
        self._append({"event": "finish", "section": self.active, "clock": now,
                      "seconds": now - self.section_started})
        self.closed.append(self.active)
        self.active, self.section_started = None, None

    def snapshot(self):
        return copy.deepcopy({"counts": self.counts, "phases": self.phases, "scopes": self.scopes,
                              "active": self.active, "closed": self.closed, "ledger_sha256": self.previous,
                              "events": self.sequence, "failed": self.failed})

    def close(self):
        self.handle.close()
        self.failed = True


def read_ledger(path):
    """Independent chain and arithmetic check; cannot resume an experiment."""
    previous, rows, totals, scopes, phases = "0" * 64, [], {}, {}, {}
    active, closed, started_scope, clock = None, set(), None, None
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        seal = row.pop("sha256")
        if (type(row["sequence"]) is not int or row["sequence"] != len(rows)
                or row["previous"] != previous or digest(row) != seal):
            raise ValueError("budget ledger chain mismatch")
        event = row["event"]
        if not rows:
            if event != "claim":
                raise ValueError("missing budget claim")
            claim, clock = row, row["started"]
        else:
            now = row.get("clock")
            if (type(now) not in (int, float) or not math.isfinite(now) or now < clock
                    or now - claim["started"] > claim["limits"]["seconds"]):
                raise ValueError("budget ledger clock invalid")
            clock = now
            if event == "begin":
                section = row["section"]
                if active is not None or section in closed or section not in claim["sections"]:
                    raise ValueError("overlapping/unknown/repeated ledger scope")
                active, started_scope = section, now
            elif event in ("debit", "finish"):
                if active is None or row["section"] != active:
                    raise ValueError("ledger event outside active section")
                if now - started_scope > claim["sections"][active]["seconds"]:
                    raise ValueError("ledger section time cap exceeded")
                if event == "finish":
                    if row["seconds"] != now - started_scope:
                        raise ValueError("ledger elapsed time mismatch")
                    closed.add(active)
                    active = None
            else:
                raise ValueError("unknown/repeated ledger claim/event")
        if event == "debit":
            kind, section, phase = row["resource"], row["section"], row["phase"]
            spec = claim["sections"][section]
            if kind not in ("environment", "optimizer") or phase != spec["phase"]:
                raise ValueError("ledger resource/phase mismatch")
            totals[kind] = totals.get(kind, 0) + 1
            scopes[section, kind] = scopes.get((section, kind), 0) + 1
            phases[phase, kind] = phases.get((phase, kind), 0) + 1
            if (row["total"] != totals[kind] or row["scope_total"] != scopes[section, kind]
                    or row["phase_total"] != phases[phase, kind]
                    or totals[kind] > claim["limits"][kind] or scopes[section, kind] > spec[kind]
                    or phases[phase, kind] > claim["phase_limits"][kind].get(phase, 0)):
                raise ValueError("budget ledger arithmetic mismatch")
        rows.append(row)
        previous = seal
    if not rows or rows[0]["event"] != "claim":
        raise ValueError("missing budget claim")
    return {"events": len(rows), "last_sha256": previous, "counts": totals}
