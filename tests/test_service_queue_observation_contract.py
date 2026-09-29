"""Observation projection must not retain hidden work or mutate raw evidence."""

import copy
import unittest

from evaluation.check_service_queue_boundary import make_config, take_step
from evaluation.check_service_queue_observation_contract import project_edge
from src.env.service_queue_network import initial_observation
from tests.test_service_queue_network import fixture


class ObservationContractTests(unittest.TestCase):
    def edge(self):
        f = fixture()
        config = make_config(f, "booked_flow", "shared_bottleneck")
        state = initial_observation(config)
        _, _, row = take_step(state, (0.5, 0.5), config, f["families"][0]["worlds"][0])
        return {"step": row, "cost": row["cost"]["total"]}

    def test_completion_view_hides_continuous_work_and_censoring(self):
        edge = self.edge()
        original = copy.deepcopy(edge)
        projected = project_edge(edge, "completion_events")
        self.assertEqual(edge, original)
        for key in ("before", "after"):
            self.assertNotIn("remaining_work", projected["step"][key])
            self.assertIn("stages", projected["step"][key])
        for key in ("available_work", "delivered_work", "observation_kind"):
            self.assertNotIn(key, projected["step"]["receipt"])
        self.assertEqual(projected["cost"], edge["cost"])

    def test_no_feedback_hides_reward_and_all_measured_state(self):
        projected = project_edge(self.edge(), "no_feedback")
        self.assertEqual(projected["step"], {"before": {"epoch": 0}, "after": {"epoch": 1}})

    def test_full_view_unchanged_and_unknown_mode_rejected(self):
        edge = self.edge()
        self.assertEqual(project_edge(edge, "exact_progress"), edge)
        with self.assertRaises(ValueError):
            project_edge(edge, "oracle")


if __name__ == "__main__":
    unittest.main()
