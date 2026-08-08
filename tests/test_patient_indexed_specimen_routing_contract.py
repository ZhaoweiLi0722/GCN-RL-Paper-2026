"""Cross-module and experiment-contract tests for specimen routing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from evaluation.compare_patient_indexed_specimen_routing import (
    compare_attribution,
)
from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    make_training_config,
    resolve_budget,
    select_scenarios,
)
from evaluation.train_multiscenario_network_residual import (
    actor_checkpoint_drift,
    agent_parameter_count,
)
from src.rl.agents import get_agent_class
from src.rl.experiment import build_env
from src.rl.networks import torch
from src.rl.residual_options import (
    make_explicit_residual_option_specs,
    residual_option_actions_from_patterns,
)


PLAN_PATH = (
    "experiments/configs/"
    "patient_indexed_specimen_routing_benchmark.json"
)
GCN = "gcn_residual_mdl2_network_ddpg_afd"
FLAT = "flat_residual_mdl2_network_ddpg_afd"
RESULT_ROOT = "results/patient_indexed_specimen_routing_recovery5/"
FROZEN_TEACHER_CONFIG = (
    "patient_indexed_specimen_routing_teacher_routing.json"
)
FROZEN_TEACHER_ROOT = (
    "results/patient_indexed_specimen_routing_recovery4/teachers/routing"
)


def _scenario(plan: dict, name: str) -> dict:
    return next(item for item in plan["scenarios"] if item["name"] == name)


def _config(plan: dict, algorithm: str, scenario: dict) -> dict:
    budget = resolve_budget(plan, "routing_smoke")
    config = make_training_config(
        plan,
        "routing_smoke",
        budget,
        algorithm,
        scenario,
        0,
    )
    config["device"] = "cpu"
    return config


class RoutingExperimentContractTests(unittest.TestCase):
    def test_routing_primary_plan_and_optional_controls_are_explicit(self) -> None:
        plan = load_benchmark_plan(PLAN_PATH)
        self.assertEqual(resolve_budget(plan, "routing_smoke")["num_episodes"], 5)
        pilot = resolve_budget(plan, "routing_pilot")
        self.assertEqual(pilot["seeds"], [0, 1, 2])
        self.assertEqual(pilot["num_episodes"], 100)
        self.assertEqual(pilot["checkpoint_interval"], 5)
        self.assertEqual(plan["parameter_matching"]["gcn_count"], 563397)
        self.assertEqual(plan["parameter_matching"]["flat_count"], 564137)
        self.assertLess(
            plan["parameter_matching"]["relative_gap"],
            plan["parameter_matching"]["maximum_relative_gap"],
        )

        default_names = {
            scenario["name"] for scenario in select_scenarios(plan, None)
        }
        self.assertEqual(
            default_names,
            {
                "routing_nominal_history",
                "routing_abrupt_regime_shift",
                "routing_regional_drift",
                "routing_compound_regional_stress",
            },
        )
        self.assertTrue(all(name.startswith("routing_") for name in default_names))

        routing = _scenario(plan, "routing_nominal_history")["env_overrides"]
        control = _scenario(plan, "no_routing_nominal_history")["env_overrides"]
        self.assertTrue(routing["enable_specimen_routing"])
        self.assertFalse(control["enable_specimen_routing"])
        self.assertTrue(routing["include_specimen_routing_state"])
        self.assertTrue(control["include_specimen_routing_state"])
        self.assertEqual(routing["specimen_edges"], control["specimen_edges"])
        self.assertEqual(routing["specimen_routing_lead_time_epochs"], 1)
        self.assertEqual(routing["finished_product_return_lead_time_epochs"], 0)
        self.assertEqual(
            _scenario(plan, "routing_nominal_lead0_sensitivity")["env_overrides"][
                "specimen_routing_lead_time_epochs"
            ],
            0,
        )
        self.assertEqual(
            _scenario(plan, "routing_nominal_return1_sensitivity")["env_overrides"][
                "finished_product_return_lead_time_epochs"
            ],
            1,
        )
        routing_teacher = json.loads(
            Path(
                "experiments/configs/"
                "patient_indexed_specimen_routing_teacher_routing.json"
            ).read_text(encoding="utf-8")
        )
        control_teacher = json.loads(
            Path(
                "experiments/configs/"
                "patient_indexed_specimen_routing_teacher_no_routing.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            control_teacher["experimental_role"],
            "optional_no_routing_supplement_only",
        )
        self.assertTrue(control_teacher["requires_separate_approval"])
        self.assertEqual(routing_teacher["seed"], control_teacher["seed"])
        self.assertEqual(
            routing_teacher["lookahead_seed"],
            control_teacher["lookahead_seed"],
        )
        self.assertEqual(
            routing_teacher["explicit_options"],
            control_teacher["explicit_options"],
        )
        afd = plan["algorithm_settings"][GCN]["config_overrides"][
            "advantage_distillation_pretrain"
        ]
        for key in ("epsilons", "candidate_groups", "candidate_signs"):
            self.assertIn(key, routing_teacher)
            self.assertEqual(routing_teacher[key], afd[key])
            self.assertEqual(control_teacher[key], afd[key])

    @unittest.skipIf(torch is None, "PyTorch is required for model matching")
    def test_gcn_flat_and_control_contracts_are_shape_and_parameter_matched(self) -> None:
        plan = load_benchmark_plan(PLAN_PATH)
        routing_scenario = _scenario(plan, "routing_nominal_history")
        control_scenario = _scenario(plan, "no_routing_nominal_history")

        agents = {}
        environments = {}
        for arm, scenario in (("routing", routing_scenario), ("control", control_scenario)):
            for algorithm in (GCN, FLAT):
                config = _config(plan, algorithm, scenario)
                env = build_env(config, seed=0)
                agent = get_agent_class(algorithm)(
                    env.observation_size,
                    env.action_size,
                    config,
                )
                environments[(arm, algorithm)] = env
                agents[(arm, algorithm)] = agent

        dimensions = {
            (env.observation_size, env.action_size)
            for env in environments.values()
        }
        self.assertEqual(dimensions, {(561, 80)})
        self.assertTrue(hasattr(agents[("routing", GCN)].actor, "specimen_edge_head"))
        self.assertTrue(hasattr(agents[("control", GCN)].actor, "specimen_edge_head"))

        routing_gcn = agent_parameter_count(agents[("routing", GCN)])
        routing_flat = agent_parameter_count(agents[("routing", FLAT)])
        self.assertEqual(routing_gcn, 563397)
        self.assertEqual(routing_flat, 564137)
        self.assertEqual(
            routing_gcn,
            agent_parameter_count(agents[("control", GCN)]),
        )
        self.assertEqual(
            routing_flat,
            agent_parameter_count(agents[("control", FLAT)]),
        )
        self.assertLess(
            abs(routing_gcn - routing_flat) / max(routing_gcn, routing_flat),
            0.01,
        )

        for key, agent in agents.items():
            env = environments[key]
            state = env.reset(seed=1234)
            action = agent.select_action(state, explore=False, env=env)
            self.assertEqual(action.shape, (80,))
            self.assertTrue(np.isfinite(action).all())

    def test_specimen_residual_options_are_conserved_and_shared(self) -> None:
        specs = make_explicit_residual_option_specs(
            [
                {"group": "specimen_transfer", "epsilon": 0.5, "sign": 1.0},
                {"group": "combined_routing_network", "epsilon": 0.5, "sign": 1.0},
            ]
        )
        anchor = np.zeros(12, dtype=np.float32)
        specimen = np.asarray((-1.0, 0.25, 0.75), dtype=np.float32)
        resource = np.asarray((-0.5, 0.0, 0.5), dtype=np.float32)
        capacity = np.asarray((0.5, -0.5, 0.0), dtype=np.float32)
        actions = residual_option_actions_from_patterns(
            anchor,
            resource,
            capacity,
            specs,
            specimen_pattern=specimen,
        )
        self.assertEqual(len(actions), 3)
        np.testing.assert_allclose(actions[1][:3], 0.5 * specimen)
        self.assertAlmostEqual(float(actions[1][:3].sum()), 0.0)
        np.testing.assert_allclose(actions[2][:3], 0.5 * specimen)
        self.assertGreater(float(np.abs(actions[2][3:9]).sum()), 0.0)

    def test_new_configs_never_reference_legacy_results(self) -> None:
        paths = sorted(
            Path("experiments/configs").glob(
                "patient_indexed_specimen_routing_*.json"
            )
        )
        self.assertGreaterEqual(len(paths), 10)

        def leaves(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from leaves(item)
            elif isinstance(value, list):
                for item in value:
                    yield from leaves(item)

        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for value in leaves(payload):
                if value.startswith("results/"):
                    frozen_teacher_reference = (
                        path.name == FROZEN_TEACHER_CONFIG
                        and value
                        in {
                            FROZEN_TEACHER_ROOT,
                            f"{FROZEN_TEACHER_ROOT}/teacher_cache.npz",
                        }
                    )
                    self.assertTrue(
                        value.startswith(RESULT_ROOT)
                        or frozen_teacher_reference,
                        f"{path}: {value}",
                    )

    @unittest.skipIf(torch is None, "PyTorch is required for actor drift")
    def test_actor_drift_is_measured_against_frozen_pretrain(self) -> None:
        actor = torch.nn.Linear(2, 1, bias=False)
        agent = SimpleNamespace(actor=actor)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pretrain.pt"
            torch.save({"actor": actor.state_dict()}, checkpoint)
            with torch.no_grad():
                actor.weight.add_(0.25)
            drift = actor_checkpoint_drift(agent, checkpoint)
        self.assertGreater(drift["rms"], 0.0)
        self.assertAlmostEqual(drift["max_abs"], 0.25)
        self.assertEqual(drift["parameter_count"], 2.0)

    def test_attribution_integrates_routing_primary_cells(self) -> None:
        def write_root(
            root: Path,
            *,
            scenario: str,
            gcn_cost: float,
            flat_cost: float,
            anchor_cost: float,
        ) -> None:
            root.mkdir(parents=True)
            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            for algorithm, cost in ((GCN, gcn_cost), (FLAT, flat_cost)):
                run = root / algorithm / "seed0"
                run.mkdir(parents=True)
                row = {
                    "algorithm": algorithm,
                    "training_seed": 0,
                    "evaluation_seed": 8400000,
                    "replication": 0,
                    "scenario": scenario,
                    "total_cost": cost,
                }
                with (run / "holdout_rows.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                if algorithm == GCN:
                    anchor = dict(row)
                    anchor["algorithm"] = "mdl2"
                    anchor["total_cost"] = anchor_cost
                    with (run / "holdout_anchor_rows.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(anchor))
                        writer.writeheader()
                        writer.writerow(anchor)

        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            routing_final = temporary / "routing_final"
            routing_pretrain = temporary / "routing_pretrain"
            write_root(
                routing_final,
                scenario="routing_nominal_history",
                gcn_cost=80.0,
                flat_cost=90.0,
                anchor_cost=95.0,
            )
            write_root(
                routing_pretrain,
                scenario="routing_nominal_history",
                gcn_cost=85.0,
                flat_cost=95.0,
                anchor_cost=95.0,
            )
            output = temporary / "attribution.json"
            config = {
                "routing_final_root": str(routing_final),
                "routing_pretrain_root": str(routing_pretrain),
                "algorithms": [GCN, FLAT],
                "metrics": {"total_cost": "lower"},
                "bootstrap_resamples": 20,
                "bootstrap_seed": 1,
                "output_path": str(output),
            }
            result = compare_attribution(config)
            self.assertTrue(result["decision"]["gcn_vs_flat_total_cost_supported"])
            self.assertTrue(result["decision"]["gcn_vs_mdl2_total_cost_supported"])
            self.assertTrue(
                result["decision"]["gcn_final_vs_pretrain_total_cost_supported"]
            )
            self.assertEqual(
                result["decision"]["interpretation"],
                "Under patient-indexed specimen routing, the preregistered "
                "total-cost comparisons support GCN residual control over both "
                "matched flat residual control and MDL-2.",
            )
            self.assertTrue(output.is_file())
            with self.assertRaises(FileExistsError):
                compare_attribution(config)

    def test_locked_runner_is_staged_and_non_destructive(self) -> None:
        runner = Path(
            "scripts/run_patient_indexed_specimen_routing.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("codex/patient-indexed-specimen-routing", runner)
        self.assertIn("ce9b6274419c8e0e7adf800f434e47d96c18c1dc", runner)
        self.assertIn(
            r'$ResultRoot = "results\patient_indexed_specimen_routing_recovery5"',
            runner,
        )
        self.assertIn(
            r'$SupersededResultRoot = "results\patient_indexed_specimen_routing_recovery4"',
            runner,
        )
        self.assertIn("0628af8bbfa584b344d145eaff1234e1e49b122a", runner)
        self.assertIn(
            "launcher process-gate false positive",
            runner,
        )
        self.assertIn(
            "8e2d82058532bef73bbb3b59325c0162eaf9aaf76846b5387be919b36597b6f8",
            runner,
        )
        self.assertIn("ControlProcessIds", runner)
        self.assertIn("$ExplicitControlProcessIds", runner)
        self.assertIn("$IsPythonWorkload", runner)
        self.assertIn("$IsDetachedRoutingLauncher", runner)
        self.assertNotIn(
            '$_.CommandLine -match "patient_indexed_specimen_routing"',
            runner,
        )
        self.assertIn("superseded_outputs_reused = $false", runner)
        for phase in ("Preflight", "ImportTeacher", "Smoke", "Pilot", "Evaluate"):
            self.assertIn(f'"{phase}"', runner)
        self.assertNotIn('"Validate"', runner)
        self.assertIn(
            "evaluation.verify_patient_indexed_specimen_routing_validation",
            runner,
        )
        self.assertIn("verify frozen Mac validation evidence", runner)
        self.assertNotIn('"-m", "unittest"', runner)
        self.assertNotIn('"-m", "compileall"', runner)
        self.assertNotIn(
            "evaluation.validate_patient_indexed_specimen_routing",
            runner,
        )
        self.assertNotIn('"Teachers"', runner)
        self.assertIn("evaluation.headroom_teacher_bundle", runner)
        self.assertIn('"extract"', runner)
        self.assertNotIn('"evaluation.network_residual_headroom"', runner)
        self.assertIn("-ApprovePilot", runner)
        self.assertNotIn("teacher_no_routing.json", runner)
        self.assertNotIn("smoke_no_routing.json", runner)
        self.assertNotIn("pilot_no_routing.json", runner)
        self.assertNotIn("pilot_no_routing_eval.json", runner)
        self.assertIn("learned_runs = 2", runner)
        self.assertIn("learned_runs = 6", runner)
        for forbidden in (
            "--force",
            "Remove-Item",
            "Stop-Process",
            "Compress-Archive",
            "Copy-Item",
            "resume-training-state",
        ):
            self.assertNotIn(forbidden, runner)

    def test_recovery5_launcher_detaches_and_preserves_control_chain(self) -> None:
        launcher = Path(
            "scripts/start_patient_indexed_specimen_routing_phase.ps1"
        ).read_text(encoding="utf-8")
        wrapper = Path(
            "scripts/invoke_patient_indexed_specimen_routing_phase_detached.ps1"
        ).read_text(encoding="utf-8")

        self.assertIn("Start-Process", launcher)
        self.assertIn("-RedirectStandardOutput", launcher)
        self.assertIn("-RedirectStandardError", launcher)
        self.assertIn("-PassThru", launcher)
        self.assertNotIn("-Wait", launcher)
        self.assertIn("launcher-logs", launcher)
        self.assertIn("patient_indexed_specimen_routing_recovery5", launcher)
        self.assertIn("TeacherBundle", launcher)
        self.assertIn("TeacherBundle", wrapper)
        self.assertIn("Get-ControlProcessIds", launcher)
        self.assertIn("SerializedControlProcessIds", launcher)
        self.assertIn("ControlProcessIds", launcher)
        self.assertIn("ControlProcessIds", wrapper)
        self.assertIn("control_process_ids", launcher)
        self.assertIn("control_process_ids", wrapper)
        self.assertIn("PID=$($Process.Id)", launcher)
        self.assertIn("run_patient_indexed_specimen_routing.ps1", wrapper)
        self.assertIn("exit $ExitCode", wrapper)
        self.assertIn("status.json", launcher)
        self.assertIn("Resolve-RoutingPython", launcher)
        self.assertLess(
            launcher.index("$PythonProbe = Resolve-RoutingPython"),
            launcher.index("$ResultRoot = Join-Path"),
        )
        self.assertIn("Python 3.11.9", launcher)
        self.assertIn('python_version -ne "3.11.9"', launcher)
        self.assertIn('numpy_version -ne "2.0.2"', launcher)
        self.assertIn('cuda_device_name -notmatch "(?i)RTX\\s*4090"', launcher)
        self.assertIn(r'C:\gcnrl\.venv\Scripts\python.exe', launcher)
        self.assertIn("requested_python", launcher)
        self.assertIn("python_sha256", launcher)
        self.assertIn('"Preflight"', launcher)
        self.assertIn('"Preflight"', wrapper)
        self.assertNotIn('"Validate"', launcher)
        self.assertNotIn('"Validate"', wrapper)
        self.assertNotIn("Missing requested Python executable", launcher)
        for forbidden in (
            "--force",
            "Remove-Item",
            "Stop-Process",
            "Copy-Item",
            "git reset",
            "git clean",
        ):
            self.assertNotIn(forbidden, launcher)
            self.assertNotIn(forbidden, wrapper)


if __name__ == "__main__":
    unittest.main()
