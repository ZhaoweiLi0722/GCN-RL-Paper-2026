"""Zero-execution accounting checks using the actual prospective config."""

import copy
import json
from pathlib import Path
import unittest

from src.rl.paired_cohort_plan import branch_plan, context_keys, maximum_accounting


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] /
            "experiments/configs/paired_cohort_improvement_20261002.json").read_text())

    def test_real_config_arithmetic_and_no_mutation(self):
        before = copy.deepcopy(self.config)
        result = maximum_accounting(self.config)
        for key, value in self.config["maximum"].items():
            self.assertEqual(result["contexts" if key == "context_states" else key], value)
        self.assertEqual(result["phase_seconds"], 13740)
        self.assertEqual(self.config, before)
        self.assertFalse(self.config["scientific_execution_authorized"])

    def test_full_remaining_horizon_and_once_only_first_action(self):
        counts = {key: 6 for key in context_keys(self.config)}
        plan = branch_plan(self.config, counts)
        self.assertEqual(len(plan), 432)
        self.assertEqual({p.environment_calls for p in plan}, {59, 43, 27})
        self.assertTrue(all(p.tail_calls == 11 for p in plan))
        self.assertEqual(len({(p.block, p.cohort, p.after_prefix_steps, p.replication, p.candidate)
                              for p in plan}), 432)

    def test_alias_collapses_do_not_refund_calls_or_add_states(self):
        counts = {key: 1 for key in context_keys(self.config)}
        plan = branch_plan(self.config, counts)
        self.assertEqual(len(plan), 72)
        self.assertEqual(sum(p.environment_calls for p in plan), 3096)
        self.assertEqual(maximum_accounting(self.config)["branch_calls"], 18576)

    def test_invalid_support_or_context_fails(self):
        counts = {key: 6 for key in context_keys(self.config)}
        first = next(iter(counts))
        for value in (0, 7, True, 2.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                branch_plan(self.config, counts | {first: value})
        del counts[first]
        with self.assertRaises(ValueError):
            branch_plan(self.config, counts)
        for times in ([4, 4], [52], [36, 4], [True]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                context_keys(self.config | {"context_after_prefix_steps": times})

    def test_invalid_budget_and_duplicate_roles_fail(self):
        for change in ({"global_seconds": 100}, {"economic_endpoint": 52},
                       {"blocks": [60, 60]}, {"future_replications": True},
                       {"evaluation_controllers": ["r4", "r4"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                maximum_accounting(self.config | change)


if __name__ == "__main__":
    unittest.main()
