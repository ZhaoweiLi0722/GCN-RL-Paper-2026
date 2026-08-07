from __future__ import annotations

import unittest
from unittest.mock import patch

from evaluation.diagnose_patient_registry_lookahead import (
    DiagnosticLocation,
    assert_registry_types,
    invalid_registry_values,
    replay_window,
)
from src.env.capacity_planning import CapacityPlanningConfig
from src.env.patient_capacity_planning import (
    PatientConditionCapacityEnv,
    PatientEnvConfig,
)


def _diagnostic_env() -> PatientConditionCapacityEnv:
    base = CapacityPlanningConfig(
        num_facilities=2,
        production_lead_time=2,
        episode_horizon=3,
        demand_rates=(0.0, 0.0),
        initial_specimens=(1.0, 0.0),
        initial_reagents=(0.0, 0.0),
        initial_idle_bioreactors=(0.0, 0.0),
        max_specimens=(5.0, 5.0),
        max_reagents=(5.0, 5.0),
        max_idle_bioreactors=(2.0, 2.0),
        max_reagent_replenishment=(0.0, 0.0),
        max_specimen_transfer=1.0,
        action_mode="facility_net",
        specimen_edges=((0, 1),),
        resource_edges=((0, 1),),
        capacity_edges=((0, 1),),
    )
    return PatientConditionCapacityEnv(PatientEnvConfig(base=base), seed=11)


class PatientRegistryLookaheadDiagnosticTests(unittest.TestCase):
    def test_invalid_registry_values_reports_type_without_dereference(self) -> None:
        env = _diagnostic_env()
        patient_id = next(iter(env.patient_registry))
        env.patient_registry[patient_id] = patient_id  # type: ignore[assignment]

        invalid = invalid_registry_values(env)

        self.assertEqual(invalid[0]["patient_id"], patient_id)
        self.assertEqual(invalid[0]["value_type"], "str")
        with self.assertRaisesRegex(
            RuntimeError,
            "PATIENT_REGISTRY_TYPE_CORRUPTION",
        ):
            assert_registry_types(
                env,
                DiagnosticLocation(0, 0, 0, 0, 0, "test"),
            )

    def test_replay_window_checks_deepcopy_and_each_transition(self) -> None:
        env = _diagnostic_env()
        config = {
            "seed": 101,
            "max_steps": 3,
            "state_probe_rollouts": 1,
            "lookahead": 2,
            "lookahead_replications": 1,
            "lookahead_seed": 500,
            "epsilons": [],
            "candidate_groups": [],
            "candidate_signs": [-1.0, 1.0],
            "anchor_policy": "mdl2",
        }
        with patch(
            "evaluation.diagnose_patient_registry_lookahead.load_env_config",
            return_value={},
        ), patch(
            "evaluation.diagnose_patient_registry_lookahead.build_env",
            return_value=env,
        ):
            result = replay_window(
                config,
                rollout=0,
                step_start=0,
                step_end=0,
            )

        self.assertEqual(result["evaluated_candidate_replications"], 1)
        self.assertEqual(result["lookahead_transitions"], 2)


if __name__ == "__main__":
    unittest.main()
