import copy
import json
from pathlib import Path
import unittest

from evaluation.rebuild_frozen_baselines import build_config, require_finite, validate_pretrain_summary


class FrozenBaselineRebuildTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(Path("experiments/configs/frozen_baseline_rebuild_20260929.json").read_text())
        self.source = json.loads(Path(self.spec["source_config"]).read_text())

    def test_config_preserves_science_except_declared_rebuild_fields(self):
        original = copy.deepcopy(self.source)
        for seed in self.spec["seeds"]:
            config = build_config(self.spec, self.source, seed, Path("payload"))
            self.assertEqual(config["online_episodes"], 0)
            self.assertEqual(config["pretrain_epochs"], 300)
            self.assertEqual(config["offline_updates"], 500)
            self.assertEqual(config["scenario_by_seed"], original["scenario_by_seed"])
            overrides = copy.deepcopy(config["config_overrides"])
            self.assertIn(f"seed{seed}", overrides.pop("preonline_training_state_path"))
            self.assertEqual(overrides["critic_teacher_advantage_calibration"]["updates"], 0)
            overrides["critic_teacher_advantage_calibration"]["updates"] = 100
            self.assertEqual(overrides, original["config_overrides"])
        self.assertEqual(self.source, original)

    def test_rejects_changed_budget_or_invented_claim(self):
        changes = {"online_episodes": 1, "pretrain_epochs": 301, "offline_updates": 501,
                   "seeds": [60], "maximum_environment_steps_per_seed": 1041,
                   "maximum_seconds": 3601, "scientific_evaluation_authorized": True,
                   "historical_reproduction_claim": True}
        for key, value in changes.items():
            with self.subTest(key=key):
                spec = dict(self.spec, **{key: value})
                with self.assertRaises(ValueError):
                    build_config(spec, self.source, 60, Path("payload"))

    def test_no_cpu_fallback(self):
        self.source["config_overrides"]["device"] = "cpu"
        with self.assertRaises(ValueError):
            build_config(self.spec, self.source, 60, Path("payload"))

    def test_nested_nonfinite_metrics_rejected(self):
        for value in (float("inf"), float("nan"), float("-inf")):
            with self.assertRaises(ValueError):
                require_finite({"pretrain": [{"loss": value}]})
        require_finite({"loss": 0.12, "path": "pretrain.pt"})

    def test_summary_uses_public_distillation_prefix_not_local_search(self):
        run = {"online_episodes": 0, "scenario_episode_counts": {}, "pretrain": {
            "offline_rl_updates": 500, "advantage_distillation_epochs": 300,
            "advantage_distillation_demonstration_source": "cache"}}
        validate_pretrain_summary(run, updates=500, epochs=300)
        run["pretrain"]["local_search_demonstration_source"] = run["pretrain"].pop(
            "advantage_distillation_demonstration_source")
        with self.assertRaisesRegex(RuntimeError, "Teacher cache"):
            validate_pretrain_summary(run, updates=500, epochs=300)

    def test_online_run_and_short_update_budget_rejected(self):
        for key, value in (("online_episodes", 1), ("scenario_episode_counts", {"nominal": 1})):
            run = {"online_episodes": 0, "scenario_episode_counts": {}, "pretrain": {}}
            run[key] = value
            with self.assertRaises(RuntimeError):
                validate_pretrain_summary(run, updates=500, epochs=300)
        with self.assertRaises(RuntimeError):
            validate_pretrain_summary({"online_episodes": 0, "scenario_episode_counts": {},
                                       "pretrain": {"offline_rl_updates": 499}}, updates=500, epochs=300)


if __name__ == "__main__":
    unittest.main()
