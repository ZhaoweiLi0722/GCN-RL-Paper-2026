"""Locked execution-contract tests for attribution Recovery 3."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "experiments" / "configs"


class Recovery3ContractTests(unittest.TestCase):
    def test_training_and_gate_scientific_configs_match_recovery2(self) -> None:
        pairs = (
            (
                "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2.json",
                "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery3.json",
            ),
            (
                "multiscenario_gcn_ddpg_attribution_recovery2_stability_gate.json",
                "multiscenario_gcn_ddpg_attribution_recovery3_stability_gate.json",
            ),
        )
        for recovery2_name, recovery3_name in pairs:
            with self.subTest(config=recovery3_name):
                recovery2 = self._load_config(recovery2_name)
                recovery3 = self._load_config(recovery3_name)
                recovery2.pop("name")
                recovery3.pop("name")
                self.assertEqual(recovery3, recovery2)

    def test_evaluation_scientific_configs_match_recovery2(self) -> None:
        pairs = (
            (
                "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2_eval.json",
                "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery3_eval.json",
            ),
            (
                "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2_pretrain_eval.json",
                "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery3_pretrain_eval.json",
            ),
        )
        for recovery2_name, recovery3_name in pairs:
            with self.subTest(config=recovery3_name):
                recovery2 = self._load_config(recovery2_name)
                recovery3 = self._load_config(recovery3_name)
                for config in (recovery2, recovery3):
                    config.pop("name")
                    config.pop("training_manifest")
                    config.pop("output_root")
                self.assertEqual(recovery3, recovery2)

    def test_runner_locks_replay_stress_evidence(self) -> None:
        runner = (
            ROOT / "scripts" / "run_multiscenario_ddpg_attribution_pilot_recovery3.ps1"
        ).read_text(encoding="utf-8")

        self.assertIn('--calls", [string]$AnchorStressCalls', runner)
        self.assertIn("$AnchorStressCalls = 150000", runner)
        self.assertIn(
            "daf90359a4018206f232ee68b8c533e0ea758e0981b32e436672e33f743f8c75",
            runner,
        )
        self.assertIn(
            "88f045e9a334253d85d75a8cef679f4a552ec5aaf45b7a510abe5c9b18cbbda8",
            runner,
        )

    @staticmethod
    def _load_config(name: str) -> dict:
        return json.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
