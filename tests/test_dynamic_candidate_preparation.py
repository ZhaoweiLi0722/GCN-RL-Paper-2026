"""Source/config preparation only, with model-count algebra and disabled launch."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl.dynamic_candidate_factory import dynamic_template
from src.rl.dynamic_candidate_preparation import seed_evidence_inventory, prepare_proposal, reject_scientific_execution
from src.rl.dynamic_candidate_resources import dynamic_stream_manifest
from src.rl.dynamic_candidate_specification import build_dynamic_proposal, static_model_counts
from src.rl.networks import torch
from tests.test_dynamic_candidate_session import FakeProducer


ROOT = Path(__file__).resolve().parents[1]


class DynamicCandidatePreparationTests(unittest.TestCase):
    def setUp(self):
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("numerical update forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)

    def test_algebra_matches_artificial_model_without_any_forward(self):
        producer = FakeProducer()
        config = build_dynamic_proposal(ROOT)
        counts = static_model_counts(2, 4, 1, 16, 32)
        config["model_proposal"].update(actor_parameter_count=counts["actor"], critic_parameter_count=counts["critic"])
        template = dynamic_template(producer, config, dynamic_stream_manifest(config), 60)
        self.assertEqual(template.policy.manifest()["parameter_counts"], counts)
        config["model_proposal"]["actor_parameter_count"] += 1
        with self.assertRaisesRegex(ValueError, "static model dimensions"):
            dynamic_template(producer, config, dynamic_stream_manifest(config), 60)

    def test_real_dimensions_algebra_only(self):
        self.assertEqual(static_model_counts(20, 28, 1, 16, 32),
                         {"actor": 31346, "critic": 28721, "total": 60067, "shared": 0})

    def test_disabled_launch_does_not_construct_or_load(self):
        with patch("src.rl.experiment.build_env", side_effect=AssertionError("patient build forbidden")) as env, \
                patch.object(torch, "load", side_effect=AssertionError("checkpoint load forbidden")) as load:
            with self.assertRaises(PermissionError):
                reject_scientific_execution()
            env.assert_not_called()
            load.assert_not_called()

    def test_dirty_freeze_stops_before_reference_inspection(self):
        with patch("src.rl.dynamic_candidate_preparation.git", side_effect=["codex/september-research-integration", " M x"]), \
                patch("src.rl.dynamic_candidate_preparation.inspect_dynamic_inputs") as inspect:
            with self.assertRaisesRegex(ValueError, "clean committed"):
                prepare_proposal(ROOT, freeze=True)
            inspect.assert_not_called()

    def test_inventory_includes_old_campaigns_and_skips_unrelated_episode_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            names = ["experiments/configs/old.json", "results/old/seed_manifest.json",
                     "results/old/effective_config.json", "reports/old/streams.jsonl",
                     "specs/prior/execution_manifest.json", "results/old/episode01.json",
                     "specs/2026-10-01-adaptive-paper-delivery/frozen-proposal/config.json"]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
            with patch("src.rl.dynamic_candidate_preparation.git", return_value=names[0]):
                actual = {path.relative_to(root).as_posix() for path in seed_evidence_inventory(root)}
            self.assertEqual(actual, set(names[:5]))

    def test_unhandled_seed_format_does_not_silently_disappear(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            (root / "results/seed_manifest.csv").write_text("seed\n123\n")
            with patch("src.rl.dynamic_candidate_preparation.git", return_value=""):
                with self.assertRaisesRegex(ValueError, "unhandled"):
                    seed_evidence_inventory(root)

    def test_only_whitespace_repository_placeholder_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "experiments/configs/.gitkeep"
            marker.parent.mkdir(parents=True)
            marker.write_bytes(b"")
            with patch("src.rl.dynamic_candidate_preparation.git", return_value="experiments/configs/.gitkeep"):
                self.assertEqual(seed_evidence_inventory(root), [])
                marker.write_bytes(b"\n")
                self.assertEqual(seed_evidence_inventory(root), [])
                marker.write_text("seed=1\n")
                with self.assertRaisesRegex(ValueError, "unhandled tracked config"):
                    seed_evidence_inventory(root)


if __name__ == "__main__":
    unittest.main()
