import json
from pathlib import Path
import unittest


CONFIG_ROOT = Path("experiments/configs")


def _load(name: str) -> dict:
    return json.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))


class MultiscenarioDDPGRecovery2Tests(unittest.TestCase):
    def test_official_training_science_matches_recovery1(self):
        recovery1 = _load(
            "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery1.json"
        )
        recovery2 = _load(
            "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2.json"
        )
        recovery1.pop("name")
        recovery2.pop("name")
        recovery2["config_overrides"].pop("checkpoint_interval")
        recovery2["config_overrides"].pop(
            "training_state_checkpoint_interval"
        )

        self.assertEqual(recovery2, recovery1)

    def test_official_evaluation_science_matches_recovery1(self):
        pairs = (
            (
                "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery1_eval.json",
                "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2_eval.json",
            ),
            (
                "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery1_pretrain_eval.json",
                "multiscenario_gcn_flat_ddpg_attribution_cuda_pilot_recovery2_pretrain_eval.json",
            ),
        )
        for recovery1_name, recovery2_name in pairs:
            with self.subTest(config=recovery2_name):
                recovery1 = _load(recovery1_name)
                recovery2 = _load(recovery2_name)
                for config in (recovery1, recovery2):
                    config.pop("name")
                    config.pop("training_manifest")
                    config.pop("output_root")
                self.assertEqual(recovery2, recovery1)

    def test_stability_gate_is_isolated_and_diagnostic_only(self):
        gate = _load(
            "multiscenario_gcn_ddpg_attribution_recovery2_stability_gate.json"
        )

        self.assertTrue(gate["diagnostic_only"])
        self.assertEqual(gate["online_episodes"], 20)
        self.assertEqual(gate["pretrain_epochs"], 0)
        self.assertEqual(gate["seeds"], [0])
        self.assertEqual(len(gate["algorithms"]), 1)
        self.assertEqual(gate["algorithms"][0]["seeds"], [0])
        self.assertIn(
            "recovery1",
            gate["config_overrides"]["initial_checkpoint"],
        )

    def test_recovery_runner_never_uses_force(self):
        runner = Path(
            "scripts/run_multiscenario_ddpg_attribution_pilot_recovery2.ps1"
        ).read_text(encoding="utf-8")
        resume = Path(
            "scripts/resume_multiscenario_ddpg_attribution_recovery2_run.ps1"
        ).read_text(encoding="utf-8")

        self.assertNotIn("--force", runner)
        self.assertNotIn("--force", resume)
        self.assertIn("non-paper GCN seed 0 online stability gate", runner)
        self.assertIn("training_state.pt", resume)

    def test_process_gate_excludes_current_process_and_ancestors(self):
        for script_name in (
            "run_multiscenario_ddpg_attribution_pilot_recovery2.ps1",
            "resume_multiscenario_ddpg_attribution_recovery2_run.ps1",
        ):
            with self.subTest(script=script_name):
                script = (Path("scripts") / script_name).read_text(
                    encoding="utf-8"
                )
                self.assertIn("Get-IndependentRelatedProcesses", script)
                self.assertIn("ParentProcessId", script)
                self.assertIn("ExcludedProcessIds", script)
                self.assertIn("CurrentProcessId ([int]$PID)", script)
                self.assertIn(
                    "(?:run|resume)_multiscenario_ddpg_attribution",
                    script,
                )


if __name__ == "__main__":
    unittest.main()
