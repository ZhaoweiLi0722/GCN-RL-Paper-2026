"""Six role names share artificial public records, not six fitted controllers."""

import copy
import unittest

from src.rl.public_support_collector import (
    CAPACITY_CONTROL_ROLES, PublicSupportInput, collect_support_step,
)


def capacity(epoch=0):
    return {"epoch": epoch, "site_ids": ("A", "B"), "history": (((0.0,) * 6,),) * 2,
            "pending_hours": ((0.0, 0.0),), "ready_waiting_counts": (0, 0)}


class GuardedFakeHost:
    def __init__(self):
        self.public = capacity()
        self.calls = 0

    def observation(self):
        return [2.0, 3.0]

    def public_capacity(self):
        return copy.deepcopy(self.public)

    def step(self, action):
        self.calls += 1
        self.public["epoch"] += 1
        return [2, 3], -7.123456789123, False, {"cost": 7.123456789123, "private": "must_not_escape"}

    def __getattr__(self, key):
        raise AssertionError(f"privileged/unexpected host access: {key}")


class PublicCollectorTests(unittest.TestCase):
    def test_all_six_roles_receive_identical_information_and_own_host(self):
        payloads = []
        for role in CAPACITY_CONTROL_ROLES:
            host = GuardedFakeHost()
            before = PublicSupportInput.capture(host)
            records = []
            result = collect_support_step(host, before=before, base_action=(0,) * 8,
                                          requested_hours=(0.375, 0), record_raw=records.append)
            payloads.append(result)
            self.assertEqual(host.calls, 1, role)
            self.assertEqual(result.action.requested_hours, (0.375, 0))
            self.assertEqual(result.raw_cost, 7.123456789123)
            self.assertNotIn("private", result.__dict__)
            self.assertEqual(records[0]["private"], "must_not_escape")
        self.assertTrue(all(p == payloads[0] for p in payloads))

    def test_captured_records_do_not_alias_mutable_sources(self):
        host = GuardedFakeHost()
        result = PublicSupportInput.capture(host)
        host.public["epoch"] = 7
        self.assertEqual(result.epoch, 0)
        with self.assertRaises(ValueError):
            collect_support_step(host, before=result, base_action=(0,) * 8,
                                 requested_hours=(0, 0), record_raw=lambda r: None)
        self.assertEqual(host.calls, 0)

    def test_privileged_fields_and_invalid_features_rejected(self):
        for key, value in (("response_tape", (1, 2)), ("epoch", -1),
                           ("ready_waiting_counts", (0.5, 0)), ("pending_hours", ((float("nan"), 0),))):
            host = GuardedFakeHost()
            host.public[key] = value
            with self.assertRaises(ValueError):
                PublicSupportInput.capture(host)

    def test_post_step_error_does_not_hide_consumed_call(self):
        host = GuardedFakeHost()
        before = PublicSupportInput.capture(host)
        original = host.step
        def broken(action):
            obs, _, done, info = original(action)
            return obs, 99.0, done, info
        host.step = broken
        with self.assertRaises(ValueError):
            collect_support_step(host, before=before, base_action=(0,) * 8,
                                 requested_hours=(0, 0), record_raw=lambda r: None)
        self.assertEqual(host.calls, 1)


if __name__ == "__main__":
    unittest.main()
