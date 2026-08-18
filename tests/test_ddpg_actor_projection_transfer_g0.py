import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluation.audit_ddpg_actor_projection_transfer import (
    FORMAL_HOLDOUT_SEED,
    classify_actor_transfer,
    diagnostic_crn_seed_sequence,
    smoke_config,
    summarize_cross_arm,
    training_scale_diagnostics,
    transfer_metrics,
    validate_config,
)


class DDPGActorProjectionTransferG0Tests(unittest.TestCase):
    def test_locked_config_uses_fresh_nonoverlapping_crns(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = (
            root
            / "experiments/configs/"
            "patient_indexed_specimen_routing_ddpg_actor_projection_"
            "transfer_g0.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))

        validate_config(config)
        used = diagnostic_crn_seed_sequence(config)

        self.assertEqual(len(used), len(set(used)))
        self.assertNotIn(FORMAL_HOLDOUT_SEED, used)
        self.assertFalse(set(used) & set(config["forbidden_crn_seeds"]))
        self.assertEqual(config["audit"]["device"], "cpu")
        self.assertEqual(config["stage_g0_gate"]["primary_horizon"], "remaining")

        smoke = smoke_config(config)
        validate_config(smoke)
        self.assertEqual(smoke["audit"]["horizons"], [4])
        self.assertEqual(smoke["stage_g0_gate"]["primary_horizon"], "4")

    def test_transfer_metrics_separates_critic_gradient_and_policy(self) -> None:
        rows = []
        for step, true_best, critic_best, gradient_best, policy_index in (
            (0, 1, 1, 0, 1),
            (1, 2, 0, 2, -1),
        ):
            advantages = [0.0, 2_000_000.0, 3_000_000.0]
            if true_best == 1:
                advantages[1], advantages[2] = advantages[2], advantages[1]
            for index, advantage in enumerate(advantages):
                rows.append(
                    {
                        "training_seed": 60,
                        "step": step,
                        "candidate_index": index,
                        "true_cost_advantage": advantage,
                        "true_best_index": true_best,
                        "critic_best_index": critic_best,
                        "gradient_best_index": gradient_best,
                        "policy_candidate_index": policy_index,
                    }
                )

        metrics = transfer_metrics(rows, material_improvement=1_000_000.0)

        self.assertEqual(metrics["states"], 2)
        self.assertEqual(metrics["material_headroom_fraction"], 1.0)
        self.assertEqual(metrics["critic_top1_true_accuracy"], 0.5)
        self.assertEqual(metrics["gradient_top1_true_accuracy"], 0.5)
        self.assertEqual(metrics["policy_top1_true_accuracy"], 0.5)
        self.assertEqual(metrics["policy_legal_candidate_fraction"], 0.5)

    def test_cross_arm_quantization_collapse_is_conditional_on_actor_change(self) -> None:
        records = {}
        for step in (0, 1):
            common = {
                "network_action": np.array([0.0, 0.0]),
                "policy_action": np.array([0.0, 0.0]),
                "policy_executed_action_id": "same",
                "critic_best_index": 0,
                "gradient_best_index": 0,
                "policy_candidate_index": 0,
            }
            records[(60, step, "control", "pretrain")] = dict(common)
            records[(60, step, "candidate", "pretrain")] = dict(common)
            records[(60, step, "control", "final")] = dict(common)
            candidate = dict(common)
            if step == 0:
                candidate["network_action"] = np.array([0.01, 0.0])
            else:
                candidate["network_action"] = np.array([0.02, 0.0])
                candidate["policy_executed_action_id"] = "different"
            records[(60, step, "candidate", "final")] = candidate

        result = summarize_cross_arm(
            records,
            seeds=(60,),
            steps=(0, 1),
            variants=("pretrain", "final"),
        )

        self.assertEqual(result["pretrain"]["network_action_difference_fraction"], 0.0)
        self.assertEqual(result["final"]["network_action_difference_fraction"], 1.0)
        self.assertEqual(result["final"]["executed_action_difference_fraction"], 0.5)
        self.assertEqual(
            result["final"]["incremental_quantization_collapse_fraction"],
            0.5,
        )

    def test_classification_is_a_mechanism_gate_not_training_authorization(self) -> None:
        gates = {
            "primary_horizon": "remaining",
            "minimum_headroom_state_fraction": 0.10,
            "minimum_candidate_critic_top1_accuracy": 0.40,
            "minimum_candidate_critic_gain": 0.05,
            "minimum_critic_gradient_accuracy_gap": 0.10,
            "maximum_incremental_executed_action_difference_fraction": 0.20,
            "minimum_incremental_quantization_collapse_fraction": 0.50,
        }

        def classify(
            *,
            control_critic: float,
            candidate_critic: float,
            gradient: float,
            execution: float,
            collapse: float,
            headroom: float = 0.8,
        ) -> dict:
            base = {
                "material_headroom_fraction": headroom,
                "critic_top1_true_accuracy": control_critic,
                "gradient_top1_true_accuracy": gradient,
            }
            candidate = dict(base)
            candidate["critic_top1_true_accuracy"] = candidate_critic
            metrics = {
                "control": {"final": {"remaining": base}},
                "candidate": {"final": {"remaining": candidate}},
            }
            cross_arm = {
                "final": {
                    "executed_action_difference_fraction": execution,
                    "incremental_quantization_collapse_fraction": collapse,
                }
            }
            return classify_actor_transfer(
                metrics,
                cross_arm=cross_arm,
                gates=gates,
            )

        critic_failure = classify(
            control_critic=0.30,
            candidate_critic=0.35,
            gradient=0.20,
            execution=0.05,
            collapse=0.9,
        )
        gradient_failure = classify(
            control_critic=0.30,
            candidate_critic=0.60,
            gradient=0.30,
            execution=0.05,
            collapse=0.9,
        )
        projection_failure = classify(
            control_critic=0.30,
            candidate_critic=0.60,
            gradient=0.55,
            execution=0.05,
            collapse=0.9,
        )

        self.assertEqual(
            critic_failure["classification"],
            "paired_critic_did_not_generalize",
        )
        self.assertEqual(
            gradient_failure["classification"],
            "straight_through_gradient_bottleneck",
        )
        self.assertEqual(
            projection_failure["classification"],
            "actor_projection_transfer_bottleneck",
        )
        for decision in (
            critic_failure,
            gradient_failure,
            projection_failure,
        ):
            self.assertFalse(decision["training_authorized"])
            self.assertFalse(decision["formal_confirmation_authorized"])

    def test_training_scale_reader_accepts_persisted_large_fields(self) -> None:
        field = "online_rl_critic_loss_mean"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "gcn" / "seed60"
            run_root.mkdir(parents=True)
            config_path = run_root / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            with (run_root / "training.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=[field, "payload"])
                writer.writeheader()
                writer.writerow({field: "0.5", "payload": "x" * 140_000})
            runs = {
                "control": {
                    60: {
                        "config": str(config_path),
                        "actor_drift_from_pretrain": {
                            "rms": 0.0,
                            "max_abs": 0.0,
                            "parameter_count": 1,
                        },
                    }
                }
            }

            result = training_scale_diagnostics(runs, algorithm="gcn")

        self.assertEqual(
            result["control"]["60"][field],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
