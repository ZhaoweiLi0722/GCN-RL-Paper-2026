"""No science: reconcile saved metadata and fail-closed remaining budgets."""

import copy
import json
from pathlib import Path

import unittest
from tempfile import TemporaryDirectory

from src.rl.capacity_value_comparison_recovery1_resources import (
    CapacityValueComparisonRecovery1Budget, PARTIAL, PHASES, remainder)

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    frozen = json.loads((ROOT/"specs/2026-10-04-value-mpc-comparison/frozen.json").read_text())
    oldroot = ROOT/"results/capacity_value_comparison_20261004/launcher"
    old = json.loads((oldroot/"child-failure.json").read_text())["budget"]
    terminal = json.loads((oldroot/"terminal.json").read_text())
    return frozen["config"], old, terminal


def contract():
    p,b,t = inputs()
    return p, remainder(p["value_comparison_study"], b, t,
                       dict(files=1, total_bytes=100, raw_bytes=100, archive_bytes=0))


def check_remaining_saved_counter_arithmetic():
    _,r = contract()
    wanted = dict(trajectories=528, native_steps=33819, total_native_operations=34875,
        value_optimizer_steps=12320, total_optimizer_steps=12320,
        neural_forward_module_calls=27692, planner_total_decisions=22475,
        planner_candidate_rollouts=1078800, planner_total_model_epochs=8630400,
        estimator_hypothesis_transitions=3381900, initial_seals=8, final_seals=10)
    assert {k:r["limits"][k] for k in wanted} == wanted
    assert r["seconds"]["global"] == 29855
    assert r["retained_interrupted_reservation"]["planner_total_model_epochs"]["dispatched"] == 13
    assert all(sum(p[k] for p in r["phase_limits"].values()) == v for k,v in r["limits"].items())


def check_reject_different_failure_or_refund():
    p,b,t = inputs()
    b["pending_chunks"]["planner_total_model_epochs"]["reserved"] = 13
    with unittest.TestCase().assertRaises(ValueError):
        remainder(p["value_comparison_study"],b,t,{})


def check_zero_budget_preserves_original_config_and_partial_time(tmp_path):
    p,r = contract()
    original = copy.deepcopy(p)
    budget = CapacityValueComparisonRecovery1Budget(tmp_path/"launcher",p,r,clock=lambda: 100.)
    try:
        assert not any(budget.counts.values())
        assert p == original
        assert budget.proposal["proposed_budget"]["combined_disk_cap_bytes"] == 8589934592-100
        budget.enter(PHASES[1],PHASES[1])
        budget.job(PARTIAL+"-remaining","world")
        assert budget.job_times[budget.job_id] == r["partial_job_seconds"]
        assert budget.job_caps["world"] == 120
    finally:
        budget.close()


def check_phase_cap_cannot_borrow(tmp_path):
    p,r = contract()
    budget = CapacityValueComparisonRecovery1Budget(tmp_path/"launcher",p,r,clock=lambda: 100.)
    try:
        budget.enter(PHASES[0],PHASES[0])
        with unittest.TestCase().assertRaises(RuntimeError):
            budget.debit({"final_seals": 1})
        assert budget.failed
    finally:
        budget.close()


def check_partial_time_cannot_restart(tmp_path):
    p,r = contract()
    clock = [100.]
    budget = CapacityValueComparisonRecovery1Budget(tmp_path/"launcher",p,r,clock=lambda: clock[0])
    try:
        budget.enter(PHASES[1],PHASES[1])
        budget.job(PARTIAL+"-remaining","world")
        clock[0] += 120-r["partial_job_seconds"]+1
        with unittest.TestCase().assertRaises(TimeoutError):
            budget.check()
    finally:
        budget.close()


class RemainingResourcesTests(unittest.TestCase):
    def test_arithmetic(self):
        check_remaining_saved_counter_arithmetic()

    def test_boundary(self):
        check_reject_different_failure_or_refund()

    def test_zero_start(self):
        with TemporaryDirectory() as path:
            check_zero_budget_preserves_original_config_and_partial_time(Path(path))

    def test_phase_cap(self):
        with TemporaryDirectory() as path:
            check_phase_cap_cannot_borrow(Path(path))

    def test_partial_time(self):
        with TemporaryDirectory() as path:
            check_partial_time_cannot_restart(Path(path))


if __name__ == "__main__":
    unittest.main()
