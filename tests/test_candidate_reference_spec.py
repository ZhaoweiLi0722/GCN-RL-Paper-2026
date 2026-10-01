"""P2 fixed scope, stream and historical-inventory checks without simulation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl import candidate_reference_spec as spec
from src.rl.candidate_pilot_driver import pilot_jobs
from src.rl.candidate_pilot_resources import budget_sections, stream_manifest, numeric_leaves

ROOT = Path(__file__).resolve().parents[1]


class ReferenceSpecTests(unittest.TestCase):
    def test_fixed_delta_caps_and_unique_streams(self):
        cfg = spec.configuration(ROOT)
        original = json.loads((ROOT / spec.PROPOSAL).read_text())
        for key in ("objective", "reference", "representations", "model", "candidate_support", "ppo", "bc_continue",
                    "optimizer", "evaluation", "blocks", "candidate_roles", "reference_roles", "ddpg"):
            self.assertEqual(cfg[key], original[key], key)
        scopes = budget_sections(cfg)
        self.assertEqual(sum(r["environment"] for r in scopes.values()), 51480)
        self.assertEqual(sum(r["optimizer"] for r in scopes.values()), 2304)
        self.assertEqual(len(pilot_jobs(cfg)), 56)
        self.assertFalse(any("initialization" in s or "demonstration" in s for s in scopes))
        streams = stream_manifest(cfg)
        self.assertEqual(len(numeric_leaves(streams["environment"])), 150)
        self.assertEqual(len(streams["neural"]), 28)
        self.assertEqual(len(numeric_leaves(streams)), 181)
        self.assertFalse(set(numeric_leaves(streams)).intersection(numeric_leaves(stream_manifest(original))))
        self.assertFalse(cfg["scientific_execution_authorized"])
        self.assertEqual(cfg["initialization"]["nonreference_mass"], .1)

    def test_inventory_includes_old_p1_r1_results_configs_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            names = [spec.PROPOSAL, "experiments/configs/candidate_return_pilot_20260930_recovery1_execution.json",
                     "results/candidate_return_pilot_20260930/launcher/seed.json",
                     "results/candidate_return_pilot_20260930_recovery1/launcher/seed.json",
                     "reports/2026-09-30-candidate-pilot-integration/readiness.json"]
            excluded = [spec.DESIGN, spec.EFFECTIVE, spec.OUTPUT + "/prospective-seeds.json",
                        "reports/2026-10-01-reference-prior-integration/readiness.json"]
            for name in names + excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
            with patch.object(spec.subprocess, "check_output", return_value="\0".join(names[:2] + excluded[:2]).encode()):
                self.assertEqual(set(spec.historical_files(root)), {root / name for name in names})
                (root / "results/old-streams.csv").write_text('seed\n1\n')
                with self.assertRaisesRegex(ValueError, "unhandled"):
                    spec.historical_files(root)
