"""Contract tests for the routing-primary Stage C TD3 screen."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from evaluation.run_patient_indexed_specimen_routing_stage_c_td3 import (
    validate_scientific_contract,
    verify_locked_assets,
)
from evaluation.run_full_benchmark import algorithm_config_overrides
from src.rl.agents import get_agent_class


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / (
    "experiments/configs/"
    "patient_indexed_specimen_routing_stage_c_td3_execution.json"
)
DDPG_PATH = ROOT / (
    "experiments/configs/"
    "patient_indexed_specimen_routing_mac_mps_ddpg_confirmation_100.json"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class StageCTD3ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_json(SPEC_PATH)
        self.training = load_json(ROOT / self.spec["training_config"])
        self.ddpg = load_json(DDPG_PATH)

    def test_locked_assets_and_scientific_contract(self) -> None:
        current = Path.cwd()
        try:
            if current.resolve() != ROOT.resolve():
                import os

                os.chdir(ROOT)
            verified = verify_locked_assets(self.spec)
            validate_scientific_contract(self.spec)
        finally:
            if Path.cwd().resolve() != current.resolve():
                import os

                os.chdir(current)
        self.assertEqual(
            verified[
                "experiments/evidence/"
                "patient_indexed_specimen_routing_stage_c/"
                "teacher_train_dagger.npz"
            ],
            "9ba2ac0873c0f68e6ecc4b443e0eace8"
            "230f8e151cceb485fafe218a7ac78d92",
        )

    def test_td3_changes_backbone_not_routing_contract(self) -> None:
        for key in (
            "plan",
            "budget",
            "reference_scenario",
            "scenarios",
            "demand_history_window",
            "online_episodes",
            "pretrain_epochs",
            "offline_updates",
            "teacher_cache_sha256",
        ):
            self.assertEqual(self.training[key], self.ddpg[key])
        td3 = self.training["config_overrides"]
        ddpg = self.ddpg["config_overrides"]
        for key in (
            "batch_size",
            "actor_lr",
            "actor_update_frequency",
            "critic_warmup_updates",
            "update_frequency",
            "specimen_action_quantization",
            "critic_teacher_advantage_calibration",
            "pretrain_reference_actor_loss",
            "online_advantage_self_imitation",
            "imitation_pretrain",
            "advantage_distillation_pretrain",
            "residual_action",
            "train_randomization",
        ):
            self.assertEqual(td3[key], ddpg[key])
        self.assertEqual(td3["policy_delay"], 2)
        self.assertEqual(td3["policy_noise"], 0.01)
        self.assertEqual(td3["noise_clip"], 0.02)

    def test_fresh_development_streams_and_symmetric_seed_budget(self) -> None:
        self.assertEqual(self.spec["required_accelerator"], "mps")
        self.assertEqual(self.training["config_overrides"]["device"], "mps")
        self.assertEqual(self.spec["training_seeds"], [20, 21, 22])
        self.assertTrue(
            set(self.spec["development_crn_seeds"]).isdisjoint(
                self.spec["forbidden_crn_seeds"]
            )
        )
        entries = self.training["algorithms"]
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["seeds"], entries[1]["seeds"])
        self.assertEqual(entries[0]["seeds"], [20, 21, 22])

    def test_td3_parameter_match_is_prespecified_below_one_percent(self) -> None:
        plan = load_json(ROOT / self.training["plan"])
        matching = plan["stage_c_td3_parameter_matching"]
        self.assertLessEqual(
            matching["relative_gap"],
            matching["maximum_relative_gap"],
        )
        self.assertEqual(
            algorithm_config_overrides(
                plan,
                "flat_residual_mdl2_network_td3_bc",
            )["hidden_sizes"],
            [284, 200, 132],
        )

    def test_registry_uses_matched_td3_agents(self) -> None:
        self.assertEqual(
            get_agent_class("gcn_residual_mdl2_network_td3_bc").__name__,
            "ConservativeGCNResidualTD3Agent",
        )
        self.assertEqual(
            get_agent_class("flat_residual_mdl2_network_td3_bc").__name__,
            "ConservativeFlatResidualTD3Agent",
        )

    def test_stage_spec_digest_is_stable_and_not_self_referential(self) -> None:
        digest = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
        self.assertEqual(len(digest), 64)
        locked_paths = {entry["path"] for entry in self.spec["locked_files"]}
        self.assertNotIn(str(SPEC_PATH.relative_to(ROOT)), locked_paths)


if __name__ == "__main__":
    unittest.main()
