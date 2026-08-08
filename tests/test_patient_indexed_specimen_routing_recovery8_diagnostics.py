"""Regression tests for the Recovery 8 Windows native-crash diagnostics."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from evaluation import diagnose_patient_indexed_specimen_routing_recovery8 as diagnostic
from src.env.capacity_planning import CapacityPlanningEnv, make_20_clinic_config
from src.rl.training_state import _agent_state_dict, training_contract_sha256


torch = diagnostic.torch


def _replay_payload(states: np.ndarray, action_dim: int = 80) -> dict:
    rows = int(states.shape[0])
    return {
        "states": np.array(states, dtype=np.float32, copy=True),
        "actions": np.zeros((rows, action_dim), dtype=np.float32),
        "rewards": np.zeros((rows, 1), dtype=np.float32),
        "next_states": np.array(states, dtype=np.float32, copy=True),
        "dones": np.zeros((rows, 1), dtype=np.float32),
    }


def _checkpoint(states: np.ndarray, action_dim: int = 80) -> dict:
    return {
        "format_version": 1,
        "algorithm": "gcn_residual_mdl2_network_ddpg_afd",
        "seed": 0,
        "agent": {
            "replay_buffer": _replay_payload(states, action_dim),
            "total_updates": 52,
        },
        "training": {"next_episode": 1, "global_step": 52},
    }


def _agent_config(env: CapacityPlanningEnv) -> dict:
    return {
        "algorithm": "gcn_residual_mdl2_network_ddpg_afd",
        "seed": 0,
        "device": "cpu",
        "batch_size": 2,
        "replay_buffer_size": 32,
        "gcn_hidden_sizes": [8],
        "actor_hidden_sizes": [16],
        "critic_hidden_sizes": [16],
        "actor_readout_mode": "network_residual",
        "normalize_observations": False,
        "residual_action": {
            "enabled": True,
            "base_policy": "mdl2",
            "include_base_action_features": True,
            "zero_init_actor": True,
            "scale": 1.0,
            "group_scales": {
                "specimen_transfer": 1.0,
                "reagent_transfer": 1.0,
                "capacity_transfer": 1.0,
                "replenishment": 0.0,
            },
            "center_groups": (
                "specimen_transfer",
                "reagent_transfer",
                "capacity_transfer",
            ),
        },
        "imitation_pretrain": {
            "regularization_weight": 0.25,
            "regularization_batch_size": 2,
        },
        "env": asdict(env.config),
    }


class Recovery8PureStressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = CapacityPlanningEnv(
            make_20_clinic_config(episode_horizon=3),
            seed=808,
        )
        self.state = self.env.reset(seed=808)
        self.states = np.stack((self.state, self.state), axis=0).astype(np.float32)

    def test_complete_edges_and_facility_action_stress_are_stable(self) -> None:
        config = {
            "env": asdict(self.env.config),
            "residual_action": {"base_policy": "mdl2"},
        }
        checkpoint = _checkpoint(self.states)

        edge_result = diagnostic.stress_complete_edges(config, checkpoint, 250)
        action_result = diagnostic.stress_facility_action(config, checkpoint, 250)

        self.assertEqual(edge_result["edge_count"], 190)
        self.assertEqual(action_result["action_dim"], 80)
        self.assertEqual(action_result["replay_rows"], 2)

    def test_immutable_replay_states_are_owned_and_read_only(self) -> None:
        checkpoint = _checkpoint(self.states)
        original = diagnostic.sha256_array(
            checkpoint["agent"]["replay_buffer"]["states"]
        )

        states = diagnostic.immutable_replay_states(checkpoint)

        self.assertTrue(states.flags.c_contiguous)
        self.assertTrue(states.flags.owndata)
        self.assertFalse(states.flags.writeable)
        with self.assertRaises(ValueError):
            states[0, 0] = 99.0
        self.assertEqual(
            diagnostic.sha256_array(
                checkpoint["agent"]["replay_buffer"]["states"]
            ),
            original,
        )

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_tensor_conversion_is_an_independent_contiguous_copy(self) -> None:
        source = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]

        converted = diagnostic.independent_contiguous_numpy(source)

        self.assertTrue(converted.flags.c_contiguous)
        self.assertTrue(converted.flags.owndata)
        self.assertFalse(np.shares_memory(converted, source.numpy()))
        source[0, 0] = -1.0
        self.assertNotEqual(float(converted[0, 0]), -1.0)

    def test_training_contract_must_match_frozen_checkpoint(self) -> None:
        config = {"algorithm": "locked", "seed": 0, "device": "cuda"}
        checkpoint = {
            "training_contract_sha256": training_contract_sha256(config)
        }
        self.assertEqual(
            diagnostic.validate_training_contract(checkpoint, config),
            checkpoint["training_contract_sha256"],
        )
        with self.assertRaisesRegex(ValueError, "scientific contract mismatch"):
            diagnostic.validate_training_contract(
                checkpoint,
                {**config, "seed": 1},
            )

    def test_locked_smoke_config_reconstruction_has_no_output_side_effect(self) -> None:
        source = Path(
            "experiments/configs/patient_indexed_specimen_routing_smoke_routing.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / "teacher_cache.npz"
            np.savez(
                teacher,
                demand_history_window=np.asarray(12, dtype=np.int64),
            )
            payload["teacher_cache"] = str(teacher)
            payload["output_root"] = str(root / "must_not_be_created")
            smoke_config = root / "smoke.json"
            smoke_config.write_text(json.dumps(payload), encoding="utf-8")

            config, metadata = diagnostic.build_locked_smoke_agent_config(
                smoke_config,
                algorithm="gcn_residual_mdl2_network_ddpg_afd",
                seed=0,
            )

            self.assertEqual(config["algorithm"], payload["algorithms"][0]["name"])
            self.assertEqual(config["device"], "cuda")
            self.assertEqual(config["num_episodes"], 5)
            self.assertEqual(config["env"]["num_facilities"], 20)
            self.assertEqual(metadata["teacher_cache"], str(teacher))
            self.assertFalse((root / "must_not_be_created").exists())

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_batched_base_action_stress_keeps_replay_immutable(self) -> None:
        config = _agent_config(self.env)
        checkpoint = _checkpoint(self.states)
        before = diagnostic.replay_array_hashes(
            checkpoint["agent"]["replay_buffer"]
        )

        result = diagnostic.stress_batched_base_action(
            config,
            checkpoint,
            3,
        )

        self.assertEqual(result["batch_size"], 2)
        self.assertEqual(
            diagnostic.replay_array_hashes(
                checkpoint["agent"]["replay_buffer"]
            ),
            before,
        )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class Recovery8ActorCriticStressTests(unittest.TestCase):
    def test_cpu_update_stress_exercises_imitation_path(self) -> None:
        env = CapacityPlanningEnv(
            make_20_clinic_config(episode_horizon=3),
            seed=809,
        )
        config = _agent_config(env)
        agent = diagnostic.get_agent_class(config["algorithm"])(
            env.observation_size,
            env.action_size,
            config,
        )
        states = []
        state = env.reset(seed=809)
        for _ in range(4):
            action = np.zeros(env.action_size, dtype=np.float32)
            next_state, reward, done, _ = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            states.append(np.array(state, copy=True))
            state = next_state
        agent.imitation_states = torch.as_tensor(
            np.stack(states),
            dtype=torch.float32,
            device="cpu",
        )
        agent.imitation_actions = torch.zeros(
            (len(states), env.action_size),
            dtype=torch.float32,
            device="cpu",
        )
        agent.imitation_node_features = None
        agent.imitation_weights = None
        checkpoint = {
            "format_version": 1,
            "algorithm": config["algorithm"],
            "seed": 0,
            "agent": _agent_state_dict(agent),
            "training": {"next_episode": 1},
        }
        before = diagnostic.replay_array_hashes(
            checkpoint["agent"]["replay_buffer"]
        )

        result = diagnostic.stress_actor_critic_update(
            config,
            checkpoint,
            2,
            device="cpu",
        )

        self.assertEqual(result["final_total_updates"], 2)
        self.assertIn("actor_loss", result["metric_ranges"])
        self.assertIn("critic_loss", result["metric_ranges"])
        self.assertIn("imitation_loss", result["metric_ranges"])
        self.assertEqual(
            diagnostic.replay_array_hashes(
                checkpoint["agent"]["replay_buffer"]
            ),
            before,
        )


class Recovery8WindowsRunnerContractTests(unittest.TestCase):
    def test_recovery7_launch_chain_is_permanently_frozen(self) -> None:
        for path in (
            "scripts/start_patient_indexed_specimen_routing_phase.ps1",
            "scripts/invoke_patient_indexed_specimen_routing_phase_detached.ps1",
            "scripts/run_patient_indexed_specimen_routing.ps1",
        ):
            source = Path(path).read_text(encoding="utf-8")
            self.assertIn("Recovery 7 is permanently frozen", source)
            self.assertIn("0xC0000005/BEX64", source)
            self.assertIn("Retry, resume", source)

    def test_each_diagnostic_layer_is_an_independent_python_process(self) -> None:
        runner = Path(
            "scripts/run_patient_indexed_specimen_routing_recovery8_diagnostics.ps1"
        ).read_text(encoding="utf-8")
        for layer in diagnostic.LAYERS:
            self.assertIn(layer, runner)
        self.assertIn("foreach ($Entry in $LayerIterations.GetEnumerator())", runner)
        self.assertIn("Start-Process", runner)
        self.assertIn("$Process.WaitForExit()", runner)
        self.assertIn("child_pid", runner)
        self.assertIn("observed_ppid", runner)
        self.assertIn("command_line", runner)
        self.assertIn('"-X", "faulthandler"', runner)
        self.assertIn("Get-WinEvent", runner)
        self.assertIn("Windows Error Reporting", runner)
        self.assertIn("CrashDumps", runner)
        self.assertIn("C:\\Windows\\Minidump", runner)
        self.assertIn("signed = $Signed", runner)
        self.assertIn("unsigned = [uint32]$Unsigned", runner)
        self.assertIn('hex = ("0x{0:X8}" -f $Unsigned)', runner)

    def test_runner_rehashes_frozen_inputs_and_cannot_train(self) -> None:
        runner = Path(
            "scripts/run_patient_indexed_specimen_routing_recovery8_diagnostics.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("Get-FrozenInputManifest", runner)
        self.assertIn("Assert-FrozenInputManifest", runner)
        self.assertIn("frozen_inputs.before.json", runner)
        self.assertIn("frozen_inputs.after.json", runner)
        self.assertIn("recovery7_outputs_read_only = $true", runner)
        self.assertIn("recovery7_outputs_reused_for_training = $false", runner)
        self.assertIn("formal_training_permitted = $false", runner)
        self.assertIn("smoke_requires_separate_locked_commit = $true", runner)
        self.assertIn("pilot_permitted = $false", runner)
        for forbidden in (
            '"-m", "evaluation.train_multiscenario_network_residual"',
            "--resume-training-state",
            "--force",
            "Stop-Process",
            "Remove-Item",
            "Copy-Item",
        ):
            self.assertNotIn(forbidden, runner)

    def test_detached_launcher_is_single_use_and_nonblocking(self) -> None:
        launcher = Path(
            "scripts/start_patient_indexed_specimen_routing_recovery8_diagnostics.ps1"
        ).read_text(encoding="utf-8")
        wrapper = Path(
            "scripts/invoke_patient_indexed_specimen_routing_recovery8_diagnostics_detached.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("Refusing to reuse existing Recovery 8 root", launcher)
        self.assertIn("Start-Process", launcher)
        self.assertIn("-RedirectStandardOutput", launcher)
        self.assertIn("-RedirectStandardError", launcher)
        self.assertIn("-PassThru", launcher)
        self.assertNotIn("-Wait", launcher)
        self.assertIn("PythonExecutableBase64", launcher)
        self.assertIn("PythonExecutableBase64", wrapper)
        self.assertIn("finally", wrapper)
        self.assertIn("exit $ExitCode", wrapper)

    def test_diagnostic_module_does_not_change_scientific_sources(self) -> None:
        source = Path(
            "evaluation/diagnose_patient_indexed_specimen_routing_recovery8.py"
        ).read_bytes()
        self.assertEqual(
            hashlib.sha256(source).hexdigest(),
            diagnostic.sha256_file(
                "evaluation/diagnose_patient_indexed_specimen_routing_recovery8.py"
            ),
        )
        text = source.decode("utf-8")
        self.assertNotIn("train_off_policy_agent(", text)
        self.assertNotIn("save_off_policy_training_state(", text)
        self.assertNotIn("agent.save(", text)


if __name__ == "__main__":
    unittest.main()
