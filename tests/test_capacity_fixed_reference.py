"""Structural supplement checks without native or neural execution."""

import json
from pathlib import Path
import tempfile
import unittest

from src.rl.capacity_fixed_reference import Budget, fixed_hours, validate_config, write_once


CONFIG = json.loads((Path(__file__).resolve().parents[1] /
    "experiments/configs/capacity_fixed_reference_20261008.json").read_text())


class FixedReferenceTests(unittest.TestCase):
    def test_exact_user_scope(self):
        validate_config(CONFIG)
        for key, value in (("fixed_hours", [0., 0., 0., 0.]), ("max_trajectories", 121),
                           ("neural_forwards", 1), ("automatic_retry", True)):
            with self.assertRaises(PermissionError):
                validate_config(dict(CONFIG, **{key: value}))

    def test_control_and_settlement_schedule(self):
        self.assertEqual([fixed_hours(t, CONFIG) for t in range(48)], [(2.,) * 4] * 48)
        self.assertEqual([fixed_hours(t, CONFIG) for t in range(48, 64)], [(0.,) * 4] * 16)
        self.assertEqual(sum(sum(fixed_hours(t, CONFIG)) for t in range(64)), 384)
        for epoch in (-1, 64):
            with self.assertRaises(ValueError):
                fixed_hours(epoch, CONFIG)

    def test_native_counts_and_no_learning(self):
        with tempfile.TemporaryDirectory() as folder:
            b = Budget(folder, CONFIG)
            for _ in range(120):
                b.begin({})
                b.native("construction")
                b.native("construction_reset")
                for t in range(64):
                    b.epoch = t
                    b.native("step")
            self.assertEqual(b.counts["total_native_operations"], 7920)
            self.assertEqual(b.counts["control_steps"], 5760)
            self.assertEqual(b.counts["tail_steps"], 1920)
            for key in ("total_optimizer_steps", "neural_forward_module_calls", "native_clones"):
                self.assertEqual(b.counts[key], 0)
            with self.assertRaises(RuntimeError):
                b.begin({})
            with self.assertRaises(RuntimeError):
                b.native("step")

    def test_write_once_preserves_existing_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evidence.json"
            write_once(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                write_once(path, {"x": 2})
            self.assertEqual(json.loads(path.read_text()), {"x": 1})
