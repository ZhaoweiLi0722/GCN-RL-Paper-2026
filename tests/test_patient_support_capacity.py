"""Artificial records and fake native methods only; no patient environments."""

import copy
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.env.patient_capacity_planning import PatientConditionCapacityEnv, PatientEnvConfig
from src.env.patient_condition import PatientStatus
from src.env.patient_support_capacity import (
    PatientSupportAction, PatientSupportCapacityEnv, validate_support_host_contract,
)
from src.env.patient_support_work import PatientSupportConfig, PatientSupportWork
from src.env.service_effort_mechanics import ServiceEffortConfig


def contract(ordinary=(0.0, 0.0), *, lead=1, work=1.0):
    return PatientSupportConfig(ServiceEffortConfig((2.0, 2.0), 2.0, lead, 3.0, 1.0, 0.5),
                                ("A", "B"), ordinary, work, 3, 10)


class WorkLedgerTests(unittest.TestCase):
    def test_fractional_work_survives_and_completes_indivisible_patient(self):
        ledger = PatientSupportWork(contract())
        r0 = ledger.advance(epoch=0, waiting_ids=(("p",), ()), raw_hours=(0.375, 0), response=(1, 1))
        self.assertEqual(r0.completed_jobs, (0, 0))
        r1 = ledger.advance(epoch=1, waiting_ids=(("p",), ()), raw_hours=(0.625, 0), response=(1, 1))
        self.assertEqual(r1.completed_jobs, (0, 0))
        self.assertEqual(ledger.snapshot()["tasks"]["p"]["remaining"], 0.625)
        r2 = ledger.advance(epoch=2, waiting_ids=(("p",), ()), raw_hours=(0, 0), response=(1, 1))
        self.assertEqual(r2.completed_jobs, (1, 0))
        ledger.mark_started(("p",))
        self.assertEqual(ledger.snapshot()["delivered_by_site"], [1.0, 0.0])

    def test_delayed_hours_are_paid_even_idle(self):
        ledger = PatientSupportWork(contract(lead=2))
        for t in range(3):
            r = ledger.advance(epoch=t, waiting_ids=((), ()), raw_hours=(0.5, 0), response=(1, 1))
            self.assertEqual(r.applied_hours, (0.5, 0.0) if t == 2 else (0.0, 0.0))
            self.assertEqual(r.labor_cost, 1.75)
            self.assertEqual(r.completed_jobs, (0, 0))

    def test_work_moves_with_patient_not_site_or_transit_time(self):
        ledger = PatientSupportWork(contract((0.25, 0.5)))
        ledger.advance(epoch=0, waiting_ids=(("p",), ()), raw_hours=(0, 0), response=(1, 1))
        ledger.advance(epoch=1, waiting_ids=((), ()), raw_hours=(0, 0), response=(1, 1))
        self.assertEqual(ledger.snapshot()["tasks"]["p"]["remaining"], 0.75)
        ledger.advance(epoch=2, waiting_ids=((), ("p",)), raw_hours=(0, 0), response=(1, 1))
        self.assertEqual(ledger.snapshot()["tasks"]["p"]["remaining"], 0.25)
        self.assertEqual(ledger.snapshot()["delivered_by_site"], [0.25, 0.5])

    def test_retirement_keeps_unfinished_work_and_forbids_resurrection(self):
        ledger = PatientSupportWork(contract((0.25, 0)))
        ledger.advance(epoch=0, waiting_ids=(("p",), ()), raw_hours=(0, 0), response=(1, 1))
        ledger.retire(("p",))
        self.assertEqual(ledger.liabilities()["retired_unfinished_work"], 0.75)
        self.assertFalse(ledger.liabilities()["settled"])
        before = ledger.snapshot()
        with self.assertRaises(ValueError):
            ledger.advance(epoch=1, waiting_ids=(("p",), ()), raw_hours=(0, 0), response=(1, 1))
        self.assertEqual(before, ledger.snapshot())

    def test_duplicate_id_and_invalid_response_fail_atomically(self):
        ledger = PatientSupportWork(contract())
        before = ledger.snapshot()
        for queues, response in ((("p",), ("p",)), (1, 1)), ((("p",), ()), (float("nan"), 1)):
            with self.assertRaises(ValueError):
                ledger.advance(epoch=0, waiting_ids=queues, raw_hours=(1, 0), response=response)
            self.assertEqual(before, ledger.snapshot())
        with self.assertRaises(ValueError):
            ledger.mark_started(("absent",))

    def test_json_restore_continues_exactly_with_pipeline_and_history(self):
        cfg = contract((0.125, 0), lead=2)
        original = PatientSupportWork(cfg)
        original.advance(epoch=0, waiting_ids=(("p",), ()), raw_hours=(0.25, 0), response=(1, 1))
        snapshot = json.loads(json.dumps(original.snapshot()))
        restored = PatientSupportWork.restore(snapshot, config=cfg)
        snapshot["tasks"]["p"]["remaining"] = 0.0
        for ledger in (original, restored):
            ledger.advance(epoch=1, waiting_ids=(("p",), ()), raw_hours=(0.5, 0), response=(1, 1))
            ledger.advance(epoch=2, waiting_ids=(("p",), ()), raw_hours=(0, 0), response=(1, 1))
        self.assertEqual(original.snapshot(), restored.snapshot())

    def test_snapshot_rejects_contract_and_conservation_damage(self):
        cfg = contract((0.25, 0))
        ledger = PatientSupportWork(cfg)
        ledger.advance(epoch=0, waiting_ids=(("p",), ()), raw_hours=(0, 0), response=(1, 1))
        damaged = ledger.snapshot()
        damaged["tasks"]["p"]["remaining"] = 0.0
        with self.assertRaises(ValueError):
            PatientSupportWork.restore(damaged, config=cfg)
        with self.assertRaises(ValueError):
            PatientSupportWork.restore(ledger.snapshot(), config=replace(cfg, work_per_patient=2.0))

    def test_public_receipt_omits_latent_work_and_response(self):
        ledger = PatientSupportWork(contract((0.25, 0)))
        r = ledger.advance(epoch=0, waiting_ids=(("p",), ()), raw_hours=(0, 0), response=(0.8, 1))
        self.assertNotIn("response", r.__dict__)
        self.assertNotIn("remaining", r.__dict__)
        self.assertEqual(r.completed_jobs, (0, 0))
        self.assertEqual(ledger.ordinary_cost(), 0.75)

    def test_site_permutation_equivariance(self):
        a, b = PatientSupportWork(contract((0.25, 0.5))), PatientSupportWork(contract((0.5, 0.25)))
        for t in range(3):
            a.advance(epoch=t, waiting_ids=(("p",), ("q",)), raw_hours=(0.2, 0.4), response=(0.8, 0.5))
            b.advance(epoch=t, waiting_ids=(("q",), ("p",)), raw_hours=(0.4, 0.2), response=(0.5, 0.8))
        self.assertEqual(a.snapshot()["tasks"], b.snapshot()["tasks"])
        self.assertEqual(a.snapshot()["delivered_by_site"], b.snapshot()["delivered_by_site"][::-1])


def patient(pid, site):
    return SimpleNamespace(patient_id=pid, status=PatientStatus.WAITING, material_facility=site,
                           manufacturing_facility=None, age=0)


HOST_FIELDS = ("t", "patient_registry", "patient_queues", "reagents", "bioreactors", "fake_rng")


def host_snapshot(env):
    return copy.deepcopy({k: getattr(env, k) for k in HOST_FIELDS})


def host_restore(env, state):
    for key, value in copy.deepcopy(state).items():
        setattr(env, key, value)


def fake_host_step(env, action):
    """Native-shaped bookkeeping fixture; no patient model or stochastic step."""
    production = np.minimum.reduce((np.array([len(q) for q in env.patient_queues], dtype=float),
                                    env.reagents, env.bioreactors)).copy()
    started = env._start_production(production)
    for p in env.patient_registry.values():
        p.age += 1
    env.reagents -= production
    env.bioreactors -= production
    env.fake_rng.append(len(started[0]))
    env.t += 1
    return np.zeros(1), -7.0, False, {"cost": 7.0, "base_cost": 2.0,
                                     "production": production.copy(), "patients_started": production.copy()}


class NativeAdapterMockTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch.object(PatientConditionCapacityEnv, "__init__", side_effect=AssertionError("no real env"))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        for name, method in (("state_dict", host_snapshot), ("load_state_dict", host_restore),
                             ("step", fake_host_step)):
            patched = patch.object(PatientConditionCapacityEnv, name, method)
            patched.start()
            self.addCleanup(patched.stop)

    def fake_env(self, ordinary=(0.5, 0.0)):
        env = object.__new__(PatientSupportCapacityEnv)
        env.env_config = PatientEnvConfig()
        env.config = SimpleNamespace(enable_overtime_control=False, enable_overtime_fatigue=False,
                                     enable_intertemporal_overtime_commitment=False)
        env.support_config = contract(ordinary)
        env.support = PatientSupportWork(env.support_config)
        env._response_tape = ((1.0, 1.0),) * env.support_config.max_epochs
        env._pending_support_request = env._support_receipt = None
        env.t, env.action_size = 0, 8
        env.patient_queues = [[patient("p", 0), patient("q", 0)], []]
        env.patient_registry = {p.patient_id: p for row in env.patient_queues for p in row}
        env.reagents = np.array([2.0, 2.0])
        env.bioreactors = np.array([2.0, 2.0])
        env.fake_rng = []
        return env

    def action(self, hours=(0, 0)):
        return PatientSupportAction((0.0,) * 8, hours)

    def test_pure_configuration_binding_never_constructs_native_env(self):
        cfg = PatientEnvConfig()
        cfg = replace(cfg, base=replace(cfg.base, num_facilities=2, episode_horizon=10))
        tape = ((1.0, 1.0),) * 10
        self.assertEqual(validate_support_host_contract(cfg, contract(), tape), tape)
        for bad in (replace(cfg.base, enable_overtime_control=True), replace(cfg.base, enable_overtime_fatigue=True),
                    replace(cfg.base, num_facilities=3), replace(cfg.base, episode_horizon=9)):
            with self.assertRaises(ValueError):
                validate_support_host_contract(replace(cfg, base=bad), contract(), tape)

    def test_original_waiting_aging_hook_sees_unready_patient(self):
        env = self.fake_env()
        env._pending_support_request = (0, 0)
        env.patient_registry["p"].specimen_age = env.patient_registry["q"].specimen_age = 0
        env.patient_registry["p"].survival = env.patient_registry["q"].survival = 1.0
        env.config.num_facilities = 2
        def advance(p):
            p.age += 1
        env.patient_model = SimpleNamespace(advance=advance, is_eligible=lambda p: True)
        starts = np.array([2.0, 0.0])
        env._start_production(starts)
        lost, expired = PatientConditionCapacityEnv._age_and_gate_patients(env)
        self.assertEqual([p.age for p in env.patient_queues[0]], [1, 1])
        self.assertEqual([p.specimen_age for p in env.patient_queues[0]], [1, 1])
        self.assertEqual(lost.sum() + expired.sum(), 0.0)
        self.assertEqual(starts.tolist(), [0, 0])

    def test_unready_patients_age_without_consuming_native_resources(self):
        env = self.fake_env()
        _, reward, _, info = env.step(self.action())
        self.assertEqual(info["production"].tolist(), [0, 0])
        self.assertEqual(env.reagents.tolist(), [2, 2])
        self.assertEqual(env.bioreactors.tolist(), [2, 2])
        self.assertEqual([p.age for p in env.patient_queues[0]], [1, 1])
        self.assertEqual(reward, -(7.0 + 1.5))
        _, _, _, info = env.step(self.action())
        self.assertEqual(info["patients_started"].tolist(), [1, 0])
        self.assertEqual(env.reagents.tolist(), [1, 2])
        self.assertEqual(env.bioreactors.tolist(), [1, 2])
        self.assertEqual([p.patient_id for p in env.patient_queues[0]], ["q"])
        self.assertIs(env.patient_registry["p"].status, PatientStatus.IN_PRODUCTION)
        self.assertEqual(env.patient_registry["p"].age, 2)

    def test_charges_use_booked_not_realized_effort_and_no_hidden_info(self):
        env = self.fake_env((0, 0))
        _, reward, _, info = env.step(self.action((0.375, 0)))
        self.assertEqual(info["support_flexible_labor_cost"], 1.265625)
        self.assertEqual(info["support_switching_cost"], 0.1875)
        self.assertEqual(reward, -info["cost"])
        self.assertNotIn("response_tape", info)
        self.assertNotIn("tasks", info)
        self.assertEqual(env.public_capacity()["pending_hours"], ((0.375, 0.0),))

    def test_native_failure_rolls_back_all_state_including_fake_rng(self):
        env = self.fake_env()
        before = env.state_dict()
        def failing_step(env, action):
            fake_host_step(env, action)
            raise RuntimeError("mock disk failure")
        with patch.object(PatientConditionCapacityEnv, "step", failing_step):
            with self.assertRaisesRegex(RuntimeError, "mock disk"):
                env.step(self.action((0.25, 0)))
        self.assertEqual(env.support.snapshot(), before["support"])
        self.assertEqual(env.t, 0)
        self.assertEqual(env.fake_rng, [])
        self.assertEqual([p.age for p in env.patient_queues[0]], [0, 0])
        self.assertIsNone(env._pending_support_request)
        self.assertEqual(env.reagents.tolist(), [2, 2])

    def test_restore_exactly_and_reject_tape_clock_and_id_mismatch(self):
        env = self.fake_env()
        env.step(self.action((0.125, 0)))
        saved = env.state_dict()
        clone = self.fake_env()
        clone.load_state_dict(saved)
        for obj in (env, clone):
            obj.step(self.action())
        self.assertEqual(env.support.snapshot(), clone.support.snapshot())
        self.assertEqual(env.reagents.tolist(), clone.reagents.tolist())
        self.assertEqual(env.fake_rng, clone.fake_rng)
        for damage in ("clock", "tape", "identity"):
            broken = copy.deepcopy(saved)
            if damage == "clock":
                broken["host"]["t"] = 7
            elif damage == "tape":
                broken["response_tape"] = ((2.0, 2.0),) * 10
            else:
                del broken["host"]["patient_registry"]["p"]
            with self.assertRaises(ValueError):
                clone.load_state_dict(broken)
            self.assertEqual(clone.t, 2)
            self.assertEqual(env.support.snapshot(), clone.support.snapshot())

    def test_invalid_action_and_legacy_overtime_fail_without_change(self):
        env = self.fake_env()
        for action in (np.zeros(8), PatientSupportAction((float("nan"),) * 8, (0, 0)), self.action((-1, 0))):
            with self.assertRaises((ValueError, TypeError)):
                env.step(action)
            self.assertEqual(env.t, 0)
        env.config.enable_overtime_control = True
        with self.assertRaises(ValueError):
            env.step(self.action())
        self.assertEqual(env.t, 0)

    def test_terminal_obligations_include_untracked_new_arrivals(self):
        env = self.fake_env()
        env.step(self.action())
        new = patient("new-arrival", 1)
        env.patient_registry[new.patient_id] = new
        env.patient_queues[1].append(new)
        liabilities = env.terminal_support_liabilities()
        self.assertEqual(liabilities["active_patient_ids"], ["new-arrival", "p", "q"])
        self.assertFalse(liabilities["settled"])
        self.assertEqual(liabilities["unfinished_work"], 1.5)


if __name__ == "__main__":
    unittest.main()
