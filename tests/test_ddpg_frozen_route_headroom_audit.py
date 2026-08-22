from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from evaluation.audit_frozen_specimen_route_headroom import (
    is_validated_opportunity,
    summarize_headroom,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "experiments/configs/"
    "patient_indexed_specimen_routing_ddpg_frozen_route_headroom_audit.json"
)


class FrozenSpecimenRouteHeadroomAuditTests(unittest.TestCase):
    def test_committed_config_has_disjoint_development_contract(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        validate_config(config)

        self.assertNotEqual(
            config["discovery_seed"],
            config["validation_seed"],
        )
        self.assertNotIn(91_100_000, (
            config["discovery_seed"],
            config["validation_seed"],
        ))

    def test_validation_stream_cannot_overlap_discovery_or_forbidden_seed(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        overlap = copy.deepcopy(config)
        overlap["validation_seed"] = overlap["discovery_seed"]
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            validate_config(overlap)

        forbidden = copy.deepcopy(config)
        forbidden["validation_seed"] = 91_100_000
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_config(forbidden)

    def test_summary_counts_only_independently_validated_safe_rows(self) -> None:
        rows = [
            self._row(
                seed=40,
                step=0,
                group="specimen_transfer",
                epsilon=0.10,
                sign=1.0,
                validation_cost_improvement=3_000_000.0,
            ),
            self._row(
                seed=40,
                step=1,
                group="specimen_transfer",
                epsilon=0.05,
                sign=-1.0,
                validation_cost_improvement=-1.0,
            ),
            self._row(
                seed=41,
                step=0,
                group="mdl2",
                epsilon=0.0,
                sign=0.0,
                validation_cost_improvement=500_000.0,
            ),
            self._row(
                seed=41,
                step=1,
                group="frozen",
                epsilon=0.0,
                sign=0.0,
                validation_cost_improvement=0.0,
            ),
        ]
        config = {
            "checkpoint_variant": "pretrain",
            "live_seed": 1,
            "discovery_seed": 2,
            "discovery_replications": 3,
            "validation_seed": 4,
            "validation_replications": 5,
            "max_steps": 2,
        }

        summary = summarize_headroom(
            rows,
            config=config,
            provenance=[],
        )

        self.assertEqual(summary["states"], 4)
        self.assertEqual(summary["discovery_opportunities"], 3)
        self.assertEqual(summary["validated_opportunities"], 2)
        self.assertEqual(summary["validated_specimen_opportunities"], 1)
        self.assertEqual(summary["validated_specimen_above_2m"], 1)
        self.assertTrue(is_validated_opportunity(rows[0]))
        self.assertFalse(is_validated_opportunity(rows[1]))
        self.assertIn("must not be summed", summary["interpretation_limit"])

    @staticmethod
    def _row(
        *,
        seed: int,
        step: int,
        group: str,
        epsilon: float,
        sign: float,
        validation_cost_improvement: float,
    ) -> dict:
        return {
            "training_seed": seed,
            "scenario": f"scenario_{seed}",
            "step": step,
            "selected_group": group,
            "selected_epsilon": epsilon,
            "selected_sign": sign,
            "validation_cost_improvement": validation_cost_improvement,
            "validation_completion_service_level_delta": 0.0,
            "validation_patients_lost_delta": 0.0,
            "validation_manufacturing_ineligibility_delta": 0.0,
        }


if __name__ == "__main__":
    unittest.main()
