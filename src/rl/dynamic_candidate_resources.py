"""Opt-in dynamic-policy stream planning and non-refundable owner budgets.

This module is an engineering adapter, not a scientific launch authorization.
The optional counterfactual arm is intentionally unsupported by this version.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from src.rl.candidate_pilot_resources import PilotBudget, digest, numeric_leaves, read_ledger
from src.utils.research_clock import CLOCK_ID, INJECTED_CLOCK_ID, shared_monotonic


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("explicit nonnegative integer limits required")
    return value


def dynamic_stream_manifest(draft):
    """Allocate prospective streams without consuming them or claiming freshness."""
    proposal = draft["rng_proposal"]
    namespace, blocks = proposal["namespace"], draft["blocks"]
    if not isinstance(namespace, str) or not namespace or len(set(blocks)) != len(blocks):
        raise ValueError("unique blocks and nonempty namespace required")
    expected = {
        "prototype_preflight": 1, "fork_preflight": 1,
        "demonstration": draft["initialization"]["demonstration_episodes_per_block"],
        "qualification": draft["qualification"]["fresh_worlds_per_block"],
        "training": draft["continuation"]["episodes_per_arm_per_block"],
        "test": draft["evaluation"]["fresh_paired_worlds_per_block"],
    }
    ranges = proposal["block_ordinal_ranges"]
    if set(ranges) != set(map(str, blocks)):
        raise ValueError("stream block inventory mismatch")
    base = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:12], "big") << 16
    environment, ordinals = {}, []
    for block in blocks:
        scoped = ranges[str(block)]
        if set(scoped) != set(expected):
            raise ValueError("stream role inventory mismatch")
        environment[str(block)] = {}
        for role, count in expected.items():
            start, end = scoped[role]
            if (_integer(start) > _integer(end) or end >= 2**16
                    or end - start + 1 != _integer(count)):
                raise ValueError("stream ordinal count/range mismatch")
            values = list(range(start, end + 1))
            ordinals.extend(values)
            environment[str(block)][role] = [base + v for v in values]
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("overlapping environment stream roles")
    paths = [p.format(b=b) for b in blocks for p in proposal["role_paths_per_block"]]
    paths += proposal["global_role_paths"]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate neural stream role")
    neural = {p: int.from_bytes(hashlib.sha256((namespace + "/" + p).encode()).digest()[:8],
                               "big") % 2**63 for p in paths}
    values = numeric_leaves(environment) + list(neural.values())
    if len(values) != len(set(values)):
        raise ValueError("cross-purpose seed collision")
    uses = proposal["uses_per_world"]
    if set(uses) != set(expected) or any(_integer(v) == 0 for v in uses.values()):
        raise ValueError("explicit paired-world multiplicities required")
    episodes = len(blocks) * sum(expected[k] * uses[k] for k in expected)
    if (len(ordinals) != draft["totals"]["unique_world_start_allocations"]
            or episodes != draft["totals"]["main_full_episodes"]):
        raise ValueError("stream allocation disagrees with declared world/episode totals")
    return {"format": "dynamic-candidate-streams-v1", "namespace": namespace,
            "environment": environment, "neural": neural,
            "unique_worlds": len(ordinals),
            "episode_uses": episodes,
            "uses_per_world": copy.deepcopy(uses), "seed_freshness_verified": False,
            "consumed": False, "scientific_execution_authorized": False}


def _phase_limits(row):
    return {"trajectory": _integer(row["trajectory_steps"]),
            "clone": _integer(row["mandatory_clone_steps"]),
            "actor": _integer(row["actor_adam_calls"]),
            "critic": _integer(row["critic_adam_calls"]),
            "seconds": _integer(row["seconds"])}


def dynamic_budget_plan(draft):
    """Translate the enumerated draft, including per-block/controller ceilings."""
    if draft["optional_counterfactual"]["include_optional_counterfactual"] is not False:
        raise ValueError("optional arm needs a separately implemented prospective plan")
    phases = {}
    for row in draft["phase_budgets"]:
        if row["id"] in phases:
            raise ValueError("duplicate phase")
        phases[row["id"]] = _phase_limits(row)
    sections = {name: row | {"phase": name} for name, row in phases.items()}
    owner = draft["owner_budgets"]
    bindings = [("initialization_fit", "initialization_fit_each_block"),
                ("ppo_continuation", "ppo_each_block"),
                ("bc_continuation", "bc_each_block"),
                ("final_evaluation", "evaluation_each_controller_each_block")]
    for phase, key in bindings:
        del sections[phase]
        caps = owner[key]
        roles = draft["final_controllers"] if phase == "final_evaluation" else [None]
        for block in draft["blocks"]:
            for role in roles:
                name = f"{phase}/block{block}" + (f"/{role}" if role is not None else "")
                if name in sections:
                    raise ValueError("duplicate owner scope")
                sections[name] = {"phase": phase, "trajectory": caps.get("environment_steps", 0),
                                  "clone": 0, "actor": caps.get("actor_adam_calls", 0),
                                  "critic": caps.get("critic_adam_calls", 0), "seconds": caps["seconds"]}
    totals = draft["totals"]
    limits = {"trajectory": totals["main_trajectory_steps"], "clone": totals["mandatory_restore_clone_steps"],
              "actor": totals["main_actor_adam_calls"], "critic": totals["main_critic_adam_calls"],
              "seconds": totals["global_elapsed_seconds"]}
    if (limits["trajectory"] + limits["clone"] != totals["main_environment_steps"]
            or limits["actor"] + limits["critic"] != totals["main_optimizer_calls"]
            or sum(p["seconds"] for p in phases.values()) != totals["main_phase_seconds"]):
        raise ValueError("declared aggregate resource totals disagree")
    result = {"format": "dynamic-candidate-budget-plan-v1", "draft_sha256": digest(draft),
              "limits": limits, "phases": phases, "sections": sections}
    validate_budget_plan(result)
    return result


def validate_budget_plan(plan):
    keys = {"trajectory", "clone", "actor", "critic", "seconds"}
    if (plan.get("format") != "dynamic-candidate-budget-plan-v1"
            or not plan["sections"] or not plan["phases"] or set(plan["limits"]) != keys):
        raise ValueError("invalid dynamic budget plan")
    for row in [plan["limits"], *plan["phases"].values()]:
        if set(row) != keys:
            raise ValueError("budget resource inventory mismatch")
        for value in row.values():
            _integer(value)
    for row in plan["sections"].values():
        if set(row) != keys | {"phase"} or row["phase"] not in plan["phases"]:
            raise ValueError("budget section inventory mismatch")
        for key in keys:
            _integer(row[key])
    for phase, limits in plan["phases"].items():
        scopes = [r for r in plan["sections"].values() if r["phase"] == phase]
        if not scopes or any(sum(r[k] for r in scopes) != limits[k] for k in keys):
            raise ValueError("owner allocations do not partition the phase budget")
    for key in keys:
        total = sum(r[key] for r in plan["phases"].values())
        if total > plan["limits"][key] or (key != "seconds" and total != plan["limits"][key]):
            raise ValueError("phase allocations disagree with global budget")


def _combined(row):
    return {"environment": row["trajectory"] + row["clone"],
            "optimizer": row["actor"] + row["critic"], "seconds": row["seconds"]}


class DynamicCandidateBudget(PilotBudget):
    """Reuse the durable ledger with separate operation owners and phase clocks."""

    def __init__(self, path, plan, *, enabled=False, clock=None, started=None):
        if enabled is not True:
            raise ValueError("dynamic candidate budget requires explicit opt-in")
        validate_budget_plan(plan)
        self.plan = copy.deepcopy(plan)
        self.path, self.clock = Path(path), shared_monotonic if clock is None else clock
        self.clock_id = CLOCK_ID if clock is None else INJECTED_CLOCK_ID
        self.limits = _combined(plan["limits"])
        self.sections = {name: _combined(row) | {"phase": row["phase"]}
                         for name, row in plan["sections"].items()}
        self.phase_limits = {kind: {phase: _combined(row)[kind] for phase, row in plan["phases"].items()}
                             for kind in ("environment", "optimizer")}
        self.last_clock = self.clock()
        self.started = self.last_clock if started is None else started
        if (not math.isfinite(self.started) or not math.isfinite(self.last_clock)
                or self.started > self.last_clock):
            raise ValueError("invalid monotonic clock")
        self.active, self.section_started, self.closed = None, None, []
        self.counts = {"environment": 0, "optimizer": 0}
        self.phases, self.scopes, self.owner_counts, self.phase_seconds = {}, {}, {}, {}
        self.previous, self.sequence, self.failed = "0" * 64, 0, False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x", encoding="utf-8")
        self._append({"event": "claim", "limits": self.limits, "sections": self.sections,
                      "phase_limits": self.phase_limits, "started": self.started,
                      "clock_id": self.clock_id, "dynamic_plan": self.plan})

    def check(self):
        now = super().check()
        if self.active is not None:
            phase = self.sections[self.active]["phase"]
            elapsed = self.phase_seconds.get(phase, 0.0) + now - self.section_started
            if elapsed > self.plan["phases"][phase]["seconds"]:
                self.failed = True
                raise TimeoutError("aggregate phase wall-clock cap exceeded")
        return now

    def begin(self, section):
        if section == "runtime_input_binding" and not self.closed and self.active is None:
            now = self.check()
            if now - self.started > self.sections[section]["seconds"]:
                self.failed = True
                raise TimeoutError("supervisor setup consumed the initial phase cap")
            self._append({"event": "begin", "section": section, "clock": self.started})
            self.active, self.section_started = section, self.started
            return
        super().begin(section)

    def _capacity(self, owners):
        now = self.check()
        if self.active is None or not owners or any(o not in ("trajectory", "clone", "actor", "critic") for o in owners):
            raise ValueError("active section and declared operation owners required")
        section, phase = self.active, self.sections[self.active]["phase"]
        for owner in set(owners):
            required = owners.count(owner)
            for prefix, row in [("global", self.plan["limits"]),
                                (f"phase/{phase}", self.plan["phases"][phase]),
                                (f"section/{section}", self.plan["sections"][section])]:
                if self.owner_counts.get(f"{prefix}:{owner}", 0) + required > row[owner]:
                    self.failed = True
                    raise RuntimeError("declared owner cap exceeded; attempt closed")
        return now

    def check_minibatch(self):
        """Admission only; each later actor/critic call still needs its own debit."""
        self._capacity(["actor", "critic"])

    def debit(self, kind):
        raise ValueError("use debit_environment or debit_optimizer with an explicit owner")

    def _debit_owner(self, owner):
        now = self._capacity([owner])
        section, phase = self.active, self.sections[self.active]["phase"]
        kind = "environment" if owner in ("trajectory", "clone") else "optimizer"
        count = self.counts[kind] + 1
        scope_key, phase_key = f"{section}:{kind}", f"{phase}:{kind}"
        scope_count, phase_count = self.scopes.get(scope_key, 0) + 1, self.phases.get(phase_key, 0) + 1
        keys = [f"global:{owner}", f"phase/{phase}:{owner}", f"section/{section}:{owner}"]
        owner_totals = {key: self.owner_counts.get(key, 0) + 1 for key in keys}
        self._append({"event": "debit", "section": section, "phase": phase, "resource": kind,
                      "total": count, "scope_total": scope_count, "phase_total": phase_count,
                      "clock": now, "operation_owner": owner, "owner_totals": owner_totals})
        self.counts[kind], self.scopes[scope_key], self.phases[phase_key] = count, scope_count, phase_count
        self.owner_counts.update(owner_totals)

    def debit_environment(self, owner):
        if owner not in ("trajectory", "clone"):
            raise ValueError("environment owner must be trajectory or clone")
        self._debit_owner(owner)

    def debit_optimizer(self, owner):
        if owner not in ("actor", "critic"):
            raise ValueError("optimizer owner must be actor or critic")
        self._debit_owner(owner)

    def finish(self):
        now = self.check()
        if self.active is None:
            raise ValueError("no active budget scope")
        phase = self.sections[self.active]["phase"]
        elapsed = now - self.section_started
        total = self.phase_seconds.get(phase, 0.0) + elapsed
        self._append({"event": "finish", "section": self.active, "clock": now,
                      "seconds": elapsed, "phase_seconds": total})
        self.phase_seconds[phase] = total
        self.closed.append(self.active)
        self.active, self.section_started = None, None

    def snapshot(self):
        return super().snapshot() | {"owner_counts": copy.deepcopy(self.owner_counts),
                                    "phase_seconds": copy.deepcopy(self.phase_seconds),
                                    "plan_sha256": digest(self.plan)}


def read_dynamic_ledger(path):
    """Independent raw-ledger validation, including owners and clone accounting."""
    result = read_ledger(path)
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    claim = rows[0]
    plan = claim["dynamic_plan"]
    validate_budget_plan(plan)
    sections = {k: _combined(v) | {"phase": v["phase"]} for k, v in plan["sections"].items()}
    phases = {k: {p: _combined(v)[k] for p, v in plan["phases"].items()} for k in ("environment", "optimizer")}
    if claim["limits"] != _combined(plan["limits"]) or claim["sections"] != sections or claim["phase_limits"] != phases:
        raise ValueError("dynamic plan disagrees with ledger claim")
    owners, seconds, start, active = {}, {}, None, None
    for row in rows[1:]:
        if row["event"] == "begin":
            active, start = row["section"], row["clock"]
        phase = plan["sections"][active]["phase"]
        if seconds.get(phase, 0.0) + row["clock"] - start > plan["phases"][phase]["seconds"]:
            raise ValueError("aggregate phase clock exceeded")
        if row["event"] == "debit":
            owner = row.get("operation_owner")
            allowed = ("trajectory", "clone") if row["resource"] == "environment" else ("actor", "critic")
            if owner not in allowed:
                raise ValueError("missing or inconsistent operation owner")
            expected = {}
            for prefix, cap in [("global", plan["limits"]),
                                (f"phase/{phase}", plan["phases"][phase]),
                                (f"section/{active}", plan["sections"][active])]:
                key = f"{prefix}:{owner}"
                value = owners.get(key, 0) + 1
                if value > cap[owner]:
                    raise ValueError("owner ledger cap exceeded")
                owners[key] = expected[key] = value
            if (row.get("owner_totals") != expected
                    or any(type(value) is not int for value in row["owner_totals"].values())):
                raise ValueError("owner ledger arithmetic mismatch")
        elif row["event"] == "finish":
            seconds[phase] = seconds.get(phase, 0.0) + row["seconds"]
            if row.get("phase_seconds") != seconds[phase]:
                raise ValueError("aggregate phase elapsed mismatch")
            active = None
    return result | {"owner_counts": owners, "phase_seconds": seconds,
                     "closed": [r["section"] for r in rows if r["event"] == "finish"], "active": active,
                     "plan_sha256": digest(plan)}


def assert_dynamic_budget_ancestor(snapshot, budget):
    """Restore model/collector state only; all later durable spend stays charged."""
    from src.rl.candidate_pilot_driver import assert_budget_ancestor

    if type(budget) is not DynamicCandidateBudget:
        raise TypeError("dynamic owner budget required")
    extra = {"owner_counts", "phase_seconds", "plan_sha256"}
    if not isinstance(snapshot, dict) or set(snapshot) != set(budget.snapshot()):
        raise ValueError("dynamic checkpoint budget fields differ")
    base = {k: v for k, v in snapshot.items() if k not in extra}
    assert_budget_ancestor(base, budget)
    checked, live = read_dynamic_ledger(budget.path), budget.snapshot()
    if any(checked[k] != live[k] for k in extra):
        raise ValueError("dynamic owner differs from durable ledger")
    rows = [json.loads(line) for line in budget.path.read_text().splitlines()]
    owners, seconds = {}, {}
    for row in rows[:snapshot["events"]]:
        if row["event"] == "debit":
            owners.update(row["owner_totals"])
        elif row["event"] == "finish":
            phase = budget.plan["sections"][row["section"]]["phase"]
            seconds[phase] = row["phase_seconds"]
    if (snapshot["owner_counts"] != owners or snapshot["phase_seconds"] != seconds
            or snapshot["plan_sha256"] != digest(budget.plan)):
        raise ValueError("checkpoint owner counters/phase clocks are not a ledger prefix")
