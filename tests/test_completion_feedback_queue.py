"""Bounded hypothetical service and causal completion-filter contracts."""

from dataclasses import asdict, replace
import math
import unittest

from src.baselines.completion_response_filter import CompletionResponseFilter
from src.env.completion_feedback_queue import (
    CompletionConfig, CompletionJob, advance, booked_backlog_action, closed,
    completion_probability, initial_observation, unresolved_ledger,
)
from src.env.service_effort_mechanics import ServiceEffortConfig


def config(releases=(0,), slots=1, max_closure=64):
    effort = ServiceEffortConfig((1, 1), 1, 1, 2, 0.5, 0.25)
    jobs = tuple(CompletionJob(f"s{site}j{k}", site, t, 2) for site in (0, 1) for k, t in enumerate(releases))
    return CompletionConfig(effort, jobs, slots, 1, max_closure)


def exposed_receipt(event=False):
    cfg = config()
    state, _, _ = advance(initial_observation(cfg), (0.5, 0.5), (1, 1), (0.9, 0.9), cfg)
    return advance(state, (0, 0), (1, 1), (0 if event else 0.99,) * 2, cfg)[1]


def estimator(refresh=0):
    return CompletionResponseFilter((0.5, 1, 1.5), (1/3, 1/3, 1/3), refresh)


class CompletionQueueTests(unittest.TestCase):
    def test_probability_matches_declared_cdf(self):
        self.assertEqual(completion_probability(1, 0), 0)
        self.assertAlmostEqual(completion_probability(1, 0.5), 1-math.exp(-0.5))
        self.assertLess(completion_probability(1, 0.25), completion_probability(1, 0.5))
        self.assertLess(completion_probability(0.5, 0.5), completion_probability(1.5, 0.5))

    def test_probability_and_config_reject_invalid_values(self):
        for rate, hours in ((0, 1), (-1, 1), (1, -1), (math.nan, 1), (1, math.inf), (True, 1)):
            with self.subTest(rate=rate, hours=hours), self.assertRaises(ValueError):
                completion_probability(rate, hours)
        cfg = config()
        for change in ({"downstream_slots": 0}, {"holding_cost": -1}, {"jobs": cfg.jobs + cfg.jobs}, {"max_closure_steps": 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(cfg, **change)

    def test_invalid_actions_and_tapes_leave_original_state_unchanged(self):
        cfg = config()
        state = initial_observation(cfg)
        for action in ((1, 1), (-0.1, 0), (True, 0), (math.nan, 0), (1.1, 0), (0,)):
            with self.subTest(action=action), self.assertRaises(ValueError):
                advance(state, action, (1, 1), (0, 0), cfg)
        for response, uniform in (((1,), (0, 0)), ((1, 1), (1, 0)), ((1, 0), (0, 0))):
            with self.assertRaises(ValueError):
                advance(state, (0, 0), response, uniform, cfg)
        self.assertEqual(state, initial_observation(cfg))

    def test_request_is_delayed_one_interval(self):
        cfg = config()
        first, receipt, _ = advance(initial_observation(cfg), (1, 0), (1, 1), (0, 0), cfg)
        self.assertEqual(receipt.applied_hours, (0, 0))
        self.assertEqual(receipt.completed, (False, False))
        second, receipt, _ = advance(first, (0, 0), (1, 1), (0, 0), cfg)
        self.assertEqual(receipt.completed, (True, False))
        self.assertEqual(second.stages, ("waiting", "support"))
        self.assertEqual(receipt.known_at, second.epoch)

    def test_response_at_delivery_not_purchase_drives_completion(self):
        cfg = config()
        state, _, _ = advance(initial_observation(cfg), (0.5, 0.5), (1, 1), (0.3, 0.3), cfg)
        _, slow, _ = advance(state, (0, 0), (0.5, 1.5), (0.3, 0.3), cfg)
        _, fast, _ = advance(state, (0, 0), (1.5, 0.5), (0.3, 0.3), cfg)
        self.assertEqual(slow.completed, (False, True))
        self.assertEqual(fast.completed, (True, False))

    def test_fifo_only_one_support_completion_per_site_interval(self):
        cfg = config((0, 0))
        state, _, _ = advance(initial_observation(cfg), (0.5, 0.5), (100, 100), (0, 0), cfg)
        state, receipt, _ = advance(state, (0, 0), (100, 100), (0, 0), cfg)
        self.assertEqual(receipt.eligible_tasks, ("s0j0", "s1j0"))
        self.assertEqual(state.stages, ("waiting", "support", "waiting", "support"))

    def test_downstream_duration_and_capacity_cannot_be_bypassed(self):
        cfg = config()
        state = initial_observation(cfg)
        state, _, _ = advance(state, (0.5, 0.5), (100, 100), (0, 0), cfg)
        state, _, _ = advance(state, (0, 0), (100, 100), (0, 0), cfg)
        state, _, cost = advance(state, (0, 0), (100, 100), (0, 0), cfg)
        self.assertEqual(state.stages, ("active", "waiting"))
        self.assertEqual(cost["downstream_finished"], [])
        self.assertEqual(state.downstream_remaining, (1, 0))
        state, _, cost = advance(state, (0, 0), (100, 100), (0, 0), cfg)
        self.assertEqual(state.stages, ("done", "waiting"))
        self.assertEqual(cost["downstream_finished"], ["s0j0"])

    def test_public_observation_does_not_leak_private_probability_or_rate(self):
        cfg = config()
        state, _, _ = advance(initial_observation(cfg), (0.5, 0.5), (1, 1), (0.99, 0.99), cfg)
        a = advance(state, (0, 0), (0.5, 1.5), (0.99, 0.99), cfg)
        b = advance(state, (0, 0), (1.5, 0.5), (0.99, 0.99), cfg)
        self.assertEqual(a, b)
        keys = set(asdict(a[0])) | set(asdict(a[1]))
        self.assertTrue(keys.isdisjoint({"response", "rate", "probability", "uniform", "remaining_work", "change_epoch"}))

    def test_idle_no_effort_and_censored_are_distinct(self):
        cfg = config((2,))
        state, receipt, _ = advance(initial_observation(cfg), (0.5, 0.5), (1, 1), (0.99, 0.99), cfg)
        self.assertEqual(receipt.observation_kind, ("idle", "idle"))
        state, receipt, _ = advance(state, (0, 0), (1, 1), (0.99, 0.99), cfg)
        self.assertEqual(receipt.applied_hours, (0.5, 0.5))
        self.assertEqual(receipt.exposure_hours, (0, 0))
        _, receipt, _ = advance(state, (0, 0), (1, 1), (0.99, 0.99), cfg)
        self.assertEqual(receipt.observation_kind, ("no_effort", "no_effort"))
        self.assertEqual(exposed_receipt().observation_kind, ("right_censored", "right_censored"))

    def test_recurring_bookings_change_events_not_only_cost_offsets(self):
        paths = []
        for releases in ((0, 0), (0, 8)):
            cfg = config(releases, slots=4)
            state, events = initial_observation(cfg), []
            for _ in range(12):
                state, receipt, _ = advance(state, (0.5, 0.5), (1, 1), (0, 0), cfg)
                events.append(receipt.completed)
            paths.append(events)
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(paths[0][2][0])
        self.assertFalse(paths[1][2][0])
        self.assertTrue(paths[1][8][0])

    def test_booking_rule_only_uses_public_jobs_and_backlog(self):
        cfg = config((2,))
        state = initial_observation(cfg)
        self.assertEqual(booked_backlog_action(state, cfg), (0, 0))
        state, _, _ = advance(state, (0, 0), (1, 1), (0.9, 0.9), cfg)
        self.assertEqual(booked_backlog_action(state, cfg), (0.5, 0.5))

    def test_complete_ledger_and_commitment_accounting(self):
        cfg = config((0, 8))
        state, purchased, applied, total = initial_observation(cfg), 0, 0, 0
        for _ in range(32):
            if closed(state):
                break
            state, receipt, cost = advance(state, booked_backlog_action(state, cfg), (1, 1), (0, 0), cfg)
            purchased += sum(receipt.requested_hours)
            applied += sum(receipt.applied_hours)
            total += cost["total"]
            self.assertAlmostEqual(purchased, applied + sum(state.pending_hours))
        self.assertTrue(closed(state))
        self.assertEqual(unresolved_ledger(state, cfg)["unfinished_jobs"], [])
        self.assertGreater(total, 0)

    def test_unfinished_tasks_never_count_as_closed(self):
        cfg = config()
        state = initial_observation(cfg)
        for _ in range(10):
            state, _, _ = advance(state, (0.5, 0.5), (1, 1), (0.999999, 0.999999), cfg)
        self.assertFalse(closed(state))
        self.assertEqual(len(unresolved_ledger(state, cfg)["unfinished_jobs"]), 2)


class CompletionFilterTests(unittest.TestCase):
    def test_completion_uses_interval_probability_not_density(self):
        filt = estimator()
        receipt = replace(exposed_receipt(True), epoch=0, known_at=1)
        posterior = filt.observe(receipt)[0]
        likelihoods = [1-math.exp(-rate*0.5) for rate in filt.rates]
        expected = [x/sum(likelihoods) for x in likelihoods]
        for actual, wanted in zip(posterior, expected):
            self.assertAlmostEqual(actual, wanted)
        self.assertGreater(filt.mean[0], 1)
        self.assertTrue(all(0 < p < 1 for p in posterior))

    def test_unfinished_exposure_uses_survival_not_zero_rate(self):
        filt = estimator()
        receipt = replace(exposed_receipt(False), epoch=0, known_at=1)
        posterior = filt.observe(receipt)[0]
        likelihoods = [math.exp(-rate*0.5) for rate in filt.rates]
        for actual, wanted in zip(posterior, [x/sum(likelihoods) for x in likelihoods]):
            self.assertAlmostEqual(actual, wanted)
        self.assertTrue(0.5 < filt.mean[0] < 1)

    def test_zero_exposure_does_not_supply_a_rate_observation(self):
        filt = estimator()
        cfg = config((8,))
        _, receipt, _ = advance(initial_observation(cfg), (0.5, 0.5), (1, 1), (0, 0), cfg)
        self.assertEqual(filt.observe(receipt), (filt.prior, filt.prior))
        self.assertEqual(filt.exposed_intervals, (0, 0))

    def test_no_exposure_still_allows_declared_prediction_refresh(self):
        filt = estimator(refresh=0.05)
        filt.observe(replace(exposed_receipt(True), epoch=0, known_at=1))
        old = filt.posterior[0]
        receipt = replace(exposed_receipt(), epoch=1, known_at=2, applied_hours=(0,0), exposure_hours=(0,0),
                          completed=(False,False), observation_kind=("no_effort","no_effort"))
        new = filt.observe(receipt)[0]
        for actual, p, base in zip(new, old, filt.prior):
            self.assertAlmostEqual(actual, 0.95*p+0.05*base)
        self.assertEqual(filt.exposed_intervals, (1, 1))

    def test_invalid_second_site_is_atomic(self):
        filt = estimator()
        before = filt.posterior, filt.exposed_intervals, filt.last_epoch
        receipt = replace(exposed_receipt(), epoch=0, known_at=1, observation_kind=("right_censored", "completed"))
        with self.assertRaises(ValueError):
            filt.observe(receipt)
        self.assertEqual((filt.posterior, filt.exposed_intervals, filt.last_epoch), before)

    def test_rejects_out_of_order_and_future_known_receipts(self):
        filt = estimator()
        for receipt in (exposed_receipt(), replace(exposed_receipt(), epoch=0, known_at=2)):
            with self.assertRaises(ValueError):
                filt.observe(receipt)
        self.assertEqual(filt.last_epoch, -1)

    def test_long_censored_sequence_stays_finite(self):
        filt = estimator(refresh=0.05)
        for t in range(1000):
            filt.observe(replace(exposed_receipt(), epoch=t, known_at=t+1))
        self.assertTrue(all(math.isfinite(x) and x > 0 for row in filt.posterior for x in row))
        for row in filt.posterior:
            self.assertAlmostEqual(sum(row), 1)


if __name__ == "__main__":
    unittest.main()
