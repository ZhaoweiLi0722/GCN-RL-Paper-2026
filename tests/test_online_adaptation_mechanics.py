"""Hand-solvable information-set checks and unchanged patient-env fixtures."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.check_online_adaptation_mechanics import PatientScenarioFixture, run
from src.utils.finite_scenario_tree import solve_fixture


class TableModel:
    def __init__(self, reveal=True):
        self.reveal = reveal

    def information_key(self, state):
        world, actions = state
        return (actions, world if actions and self.reveal else "not_yet_observed")

    def transition(self, state, action):
        world, actions = state
        cost = 0 if (action == "left") == (world == 0) else 2
        return cost, (world, actions + (action,))


class ScenarioTreeTest(unittest.TestCase):
    def solve(self, model=None, **changes):
        kwargs = dict(worlds=((0.5, (0, ())), (0.5, (1, ()))), actions=("left", "right"), horizon=2)
        kwargs.update(changes)
        return solve_fixture(model or TableModel(), **kwargs)

    def test_hand_solution_clairvoyant_zero_nonanticipative_one_openloop_two(self):
        result = self.solve()
        self.assertEqual(result["clairvoyant_cost"], 0)
        self.assertEqual(result["nonanticipative_cost"], 1)
        self.assertEqual(result["best_open_loop_cost"], 2)
        self.assertEqual(result["root_information_sets"], 1)
        self.assertEqual(result["root_actions"], ["left"])
        self.assertEqual(result["policy_replay_cost"], 1)

    def test_no_information_negative_control(self):
        result = self.solve(TableModel(reveal=False))
        self.assertEqual(result["nonanticipative_cost"], result["best_open_loop_cost"])

    def test_world_order_has_no_effect(self):
        forward = self.solve()
        reverse = self.solve(worlds=((0.5, (1, ())), (0.5, (0, ()))))
        self.assertEqual(forward, reverse)

    def test_invalid_probabilities_duplicate_worlds_and_excess_budget(self):
        for kwargs in (dict(worlds=((0.4, (0, ())), (0.4, (1, ())))), dict(worlds=((1.0, (0, ())), (0.0, (1, ())))),
                       dict(worlds=((0.5, (0, ())), (0.5, (0, ())))), dict(horizon=7), dict(actions=("left", "left"))):
            with self.assertRaises(ValueError):
                self.solve(**kwargs)

    def test_nonfinite_transition_rejected(self):
        model = TableModel()
        model.transition = lambda state, action: (float("nan"), state)
        with self.assertRaises(ValueError):
            self.solve(model)

    def test_finite_transition_costs_cannot_overflow_aggregate(self):
        model = TableModel()
        model.transition = lambda state, action: (1e308, (state[0], state[1] + (action,)))
        with self.assertRaises(ValueError):
            self.solve(model, horizon=3)


class PatientFixtureTest(unittest.TestCase):
    def config(self):
        return json.loads(Path("experiments/configs/online_adaptation_information_mechanics.json").read_text())

    def test_future_world_is_hidden_until_observed(self):
        model = PatientScenarioFixture(self.config())
        roots = [(0, ()), (1, ())]
        self.assertEqual(model.information_key(roots[0]), model.information_key(roots[1]))
        successors = [model.transition(state, "balanced")[1] for state in roots]
        self.assertNotEqual(model.information_key(successors[0]), model.information_key(successors[1]))

    def test_branching_does_not_mutate_root_and_restore_is_exact(self):
        model, restored = PatientScenarioFixture(self.config()), PatientScenarioFixture(self.config())
        root = (0, ())
        before = model.information_key(root)
        model.transition(root, "left")
        model.transition(root, "right")
        self.assertEqual(before, model.information_key(root))
        self.assertEqual(model.transition(root, "balanced"), restored.transition(root, "balanced"))
        self.assertEqual(model.records[-1], restored.records[-1])

    def test_forbids_legacy_information_channels_and_scientific_seed(self):
        for key in ("include_demand_forecast_state", "include_demand_history_state", "include_demand_sequence_state"):
            config = self.config()
            config["env"][key] = True
            with self.assertRaises(ValueError):
                PatientScenarioFixture(config)
        config = self.config()
        config["seed"] = 91100000
        with self.assertRaises(ValueError):
            PatientScenarioFixture(config)

    def test_two_step_end_to_end_and_refuse_overwrite(self):
        config = self.config()
        config["horizon"] = config["env"]["episode_horizon"] = 2
        for world in config["worlds"]:
            world["arrivals"] = world["arrivals"][:3]
        with tempfile.TemporaryDirectory() as temporary:
            path, output = Path(temporary) / "config.json", Path(temporary) / "result"
            path.write_text(json.dumps(config))
            result = run(path, output)
            self.assertFalse(result["research_performance_evidence"])
            self.assertEqual(json.loads((output / "status.json").read_text())["exit_code"], 0)
            with self.assertRaises(FileExistsError):
                run(path, output)


if __name__ == "__main__":
    unittest.main()
