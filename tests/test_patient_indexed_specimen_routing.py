"""Contract tests for patient-indexed, identity-preserving specimen routing."""

from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from src.env.capacity_planning import CapacityPlanningConfig
from src.env.patient_capacity_planning import (
    PatientConditionCapacityEnv,
    PatientEnvConfig,
)
from src.env.patient_condition import (
    PatientConditionConfig,
    PatientState,
    PatientStatus,
)
from src.env.specimen_routing import (
    execute_patient_indexed_routes,
    round_facility_net_requests,
)


def _base(**overrides) -> CapacityPlanningConfig:
    params = {
        "num_facilities": 3,
        "production_lead_time": 2,
        "episode_horizon": 8,
        "demand_rates": (0.0, 0.0, 0.0),
        "initial_specimens": (2.0, 0.0, 0.0),
        "initial_reagents": (0.0, 10.0, 0.0),
        "initial_idle_bioreactors": (0.0, 2.0, 0.0),
        "max_specimens": (20.0, 20.0, 20.0),
        "max_reagents": (20.0, 20.0, 20.0),
        "max_idle_bioreactors": (4.0, 4.0, 4.0),
        "max_reagent_replenishment": (0.0, 0.0, 0.0),
        "max_specimen_transfer": 2.0,
        "max_reagent_transfer": 0.0,
        "max_bioreactor_transfer": 0.0,
        "action_mode": "facility_net",
        "include_supplier_state": True,
        "supplier_disruption_rate": 0.0,
        "clinic_coordinates": (
            (33.7490, -84.3880),
            (33.9519, -83.3576),
            (32.0809, -81.0912),
        ),
        "specimen_edges": ((0, 1), (1, 2)),
        "resource_edges": ((0, 1), (1, 2)),
        "capacity_edges": ((0, 1), (1, 2)),
        "information_edges": ((0, 1), (1, 2)),
    }
    params.update(overrides)
    return CapacityPlanningConfig(**params)


def _env(
    *,
    base: CapacityPlanningConfig | None = None,
    enabled: bool = True,
    lead_time: int = 1,
    return_lead_time: int = 0,
    material_shelf_life: int = 10,
    transit_loss_probability: float = 0.0,
    patient: PatientConditionConfig | None = None,
    seed: int = 0,
) -> PatientConditionCapacityEnv:
    return PatientConditionCapacityEnv(
        PatientEnvConfig(
            base=base or _base(),
            patient=patient
            or PatientConditionConfig(
                healthy_decay_rate=0.0,
                frail_decay_rate=0.0,
            ),
            material_shelf_life=material_shelf_life,
            enable_specimen_routing=enabled,
            include_specimen_routing_state=True,
            specimen_routing_lead_time_epochs=lead_time,
            max_specimen_transfers_per_patient=1,
            specimen_transit_loss_probability=transit_loss_probability,
            finished_product_return_lead_time_epochs=return_lead_time,
        ),
        seed=seed,
    )


def _route_action(
    env: PatientConditionCapacityEnv,
    donor: int,
    receiver: int,
    lots: int,
) -> np.ndarray:
    action = env.noop_action()
    normalized = float(lots) / float(env.config.max_specimen_transfer)
    action[donor] = -normalized
    action[receiver] = normalized
    return action


def _patient(identifier: str, *, survival: float, specimen_age: int) -> PatientState:
    return PatientState(
        health_index=1.0,
        deterioration_epoch=100.0,
        enrollment_epoch=0,
        patient_id=identifier,
        specimen_id=identifier,
        collection_facility=0,
        material_facility=0,
        survival=survival,
        specimen_age=specimen_age,
    )


class SpecimenRoutingExecutorTests(unittest.TestCase):
    def test_signed_rounding_is_half_away_from_zero(self) -> None:
        np.testing.assert_array_equal(
            round_facility_net_requests((-1.5, -0.5, 0.49, 0.5, 1.5)),
            np.asarray((-2.0, -1.0, 0.0, 1.0, 2.0)),
        )

    def test_integer_flow_is_conserved_and_tie_breaking_is_deterministic(self) -> None:
        patients = [
            _patient("p-high", survival=0.95, specimen_age=4),
            _patient("p-low-old", survival=0.80, specimen_age=5),
            _patient("p-low-young", survival=0.80, specimen_age=2),
        ]
        queues = [patients.copy(), [], []]

        result = execute_patient_indexed_routes(
            queues,
            (-1.6, 0.6, 1.0),
            ((0, 1), (0, 2)),
            edge_priorities=(1.0, 0.0),
            edge_distances=(10.0, 20.0),
            edge_transport_hours=(1.0, 2.0),
            edge_transfer_costs=(100.0, 200.0),
            lead_time_epochs=0,
            max_transfers_per_patient=1,
            route_epoch=7,
        )

        np.testing.assert_array_equal(result.requested_integer_net, (-2, 1, 1))
        np.testing.assert_array_equal(result.actual_net, (-2, 1, 1))
        self.assertEqual(float(result.actual_net.sum()), 0.0)
        self.assertEqual(queues[2][0].patient_id, "p-low-old")
        self.assertEqual(queues[1][0].patient_id, "p-low-young")
        self.assertEqual(queues[0][0].patient_id, "p-high")
        self.assertEqual(len({event["patient_id"] for event in result.events}), 2)
        self.assertTrue(all(patient.transfer_count == 1 for patient in queues[1] + queues[2]))

    def test_only_waiting_once_and_qualified_edges_are_eligible(self) -> None:
        already_moved = _patient("moved", survival=0.7, specimen_age=1)
        already_moved.transfer_count = 1
        producing = _patient("producing", survival=0.6, specimen_age=1)
        producing.status = PatientStatus.IN_PRODUCTION
        waiting = _patient("waiting", survival=0.8, specimen_age=1)
        queues = [[already_moved, producing, waiting], [], []]

        result = execute_patient_indexed_routes(
            queues,
            (-3.0, 1.0, 2.0),
            ((0, 1),),
            edge_priorities=None,
            edge_distances=(5.0,),
            edge_transport_hours=(1.0,),
            edge_transfer_costs=(10.0,),
            lead_time_epochs=0,
            max_transfers_per_patient=1,
            route_epoch=0,
        )

        self.assertEqual(tuple(event["patient_id"] for event in result.events), ("waiting",))
        self.assertEqual(result.blocked_inbound_requests, 2)
        self.assertEqual(result.blocked_outbound_requests, 2)
        self.assertEqual([patient.patient_id for patient in queues[2]], [])
        self.assertCountEqual(
            [patient.patient_id for queue in queues for patient in queue],
            ["moved", "producing", "waiting"],
        )

    def test_patient_and_specimen_identity_are_write_once(self) -> None:
        patient = _patient("stable", survival=1.0, specimen_age=0)
        with self.assertRaises(AttributeError):
            patient.patient_id = "replacement"
        with self.assertRaises(AttributeError):
            patient.specimen_id = "replacement"
        with self.assertRaises(ValueError):
            PatientState(
                health_index=1.0,
                deterioration_epoch=1.0,
                enrollment_epoch=0,
                patient_id="patient-a",
                specimen_id="specimen-b",
            )


class SpecimenRoutingEnvironmentTests(unittest.TestCase):
    def test_lead_one_arrives_before_next_epoch_production(self) -> None:
        env = _env(lead_time=1)
        env.reset(seed=11)
        routed_id = env.patient_queues[0][0].patient_id

        _obs, _reward, _done, first = env.step(_route_action(env, 0, 1, 1))

        self.assertEqual(first["transferred_patient_ids"], (routed_id,))
        self.assertEqual(first["specimen_route_count"], 1.0)
        self.assertEqual(first["patients_started"].sum(), 0.0)
        self.assertEqual(env.patient_registry[routed_id].status, PatientStatus.IN_TRANSIT)
        self.assertEqual(env.patient_registry[routed_id].age, 1)
        self.assertEqual(env.patient_registry[routed_id].specimen_age, 1)

        _obs, _reward, _done, second = env.step(env.noop_action())

        self.assertEqual(second["specimen_transfer_arrivals"][1], 1.0)
        self.assertEqual(second["patients_started"][1], 1.0)
        self.assertEqual(env.patient_registry[routed_id].manufacturing_facility, 1)
        self.assertEqual(env.patient_registry[routed_id].transfer_count, 1)
        env.assert_identity_conservation()

    def test_direct_route_cannot_ping_pong_or_move_in_production(self) -> None:
        waiting_base = _base(
            initial_specimens=(1.0, 0.0, 0.0),
            initial_reagents=(0.0, 0.0, 0.0),
            initial_idle_bioreactors=(0.0, 0.0, 0.0),
        )
        env = _env(base=waiting_base, lead_time=0)
        env.reset(seed=2)
        patient_id = env.patient_queues[0][0].patient_id
        env.step(_route_action(env, 0, 1, 1))
        _obs, _reward, _done, info = env.step(_route_action(env, 1, 0, 1))
        self.assertEqual(info["specimen_route_count"], 0.0)
        self.assertGreaterEqual(info["blocked_specimen_requests"], 1.0)
        self.assertEqual(env.patient_registry[patient_id].material_facility, 1)
        self.assertEqual(env.patient_registry[patient_id].transfer_count, 1)

        producing_env = _env(lead_time=0)
        producing_env.reset(seed=3)
        producing_id = producing_env.patient_queues[0][0].patient_id
        producing_env.step(_route_action(producing_env, 0, 1, 1))
        self.assertEqual(
            producing_env.patient_registry[producing_id].status,
            PatientStatus.IN_PRODUCTION,
        )
        _obs, _reward, _done, info = producing_env.step(
            _route_action(producing_env, 1, 0, 1)
        )
        self.assertNotIn(producing_id, info["transferred_patient_ids"])

    def test_transit_expiry_ineligibility_and_transport_loss_are_separate(self) -> None:
        expiry_env = _env(
            base=_base(initial_specimens=(1.0, 0.0, 0.0)),
            material_shelf_life=1,
            lead_time=1,
        )
        expiry_env.reset(seed=5)
        _obs, _reward, _done, expiry = expiry_env.step(
            _route_action(expiry_env, 0, 1, 1)
        )
        self.assertEqual(expiry["transit_expiry"].sum(), 1.0)
        self.assertEqual(expiry["transit_loss"].sum(), 0.0)
        self.assertEqual(expiry["patients_lost_waiting_expired"].sum(), 0.0)

        waiting_env = _env(
            base=_base(
                initial_reagents=(0.0, 0.0, 0.0),
                initial_idle_bioreactors=(0.0, 0.0, 0.0),
            ),
            material_shelf_life=1,
            lead_time=1,
        )
        waiting_env.reset(seed=5)
        _obs, _reward, _done, waiting = waiting_env.step(
            waiting_env.noop_action()
        )
        self.assertEqual(waiting["patients_lost_waiting_expired"].sum(), 2.0)
        self.assertEqual(waiting["transit_expiry"].sum(), 0.0)

        ineligible_env = _env(
            lead_time=1,
            patient=PatientConditionConfig(
                healthy_decay_rate=1.0,
                frail_decay_rate=1.0,
                eligibility_threshold=0.9,
            ),
        )
        ineligible_env.reset(seed=5)
        _obs, _reward, _done, ineligible = ineligible_env.step(
            _route_action(ineligible_env, 0, 1, 1)
        )
        self.assertEqual(ineligible["patients_lost_transit_ineligible"].sum(), 1.0)
        self.assertEqual(ineligible["transit_expiry"].sum(), 0.0)

        loss_env = _env(lead_time=1, transit_loss_probability=1.0)
        loss_env.reset(seed=5)
        _obs, _reward, _done, loss = loss_env.step(
            _route_action(loss_env, 0, 1, 1)
        )
        self.assertEqual(loss["transit_loss"].sum(), 1.0)
        self.assertEqual(loss["transit_expiry"].sum(), 0.0)

    def test_destination_manufacturing_and_return_preserve_patient_identity(self) -> None:
        base = _base(initial_specimens=(1.0, 0.0, 0.0))
        env = _env(base=base, lead_time=0, return_lead_time=1)
        env.reset(seed=9)
        patient_id = env.patient_queues[0][0].patient_id

        env.step(_route_action(env, 0, 1, 1))
        env.step(env.noop_action())
        _obs, _reward, _done, delivered = env.step(env.noop_action())

        patient = env.patient_registry[patient_id]
        self.assertEqual(patient.patient_id, patient.specimen_id)
        self.assertEqual(patient.collection_facility, 0)
        self.assertEqual(patient.manufacturing_facility, 1)
        self.assertEqual(patient.status, PatientStatus.DELIVERED)
        self.assertEqual(delivered["patients_completed"][0], 1.0)
        self.assertEqual(
            delivered["finished_product_return_assumption"],
            "return_to_collection_facility",
        )
        env.assert_identity_conservation()

    def test_route_cost_and_diagnostics_are_finite_and_auditable(self) -> None:
        env = _env(lead_time=1)
        env.reset(seed=4)
        _obs, _reward, _done, info = env.step(_route_action(env, 0, 1, 2))

        for key in (
            "specimen_route_count",
            "specimen_route_distance_miles",
            "specimen_route_time_hours",
            "specimen_route_cost",
            "blocked_specimen_requests",
        ):
            self.assertTrue(np.isfinite(float(info[key])), key)
        self.assertEqual(len(info["specimen_route_events"]), 2)
        for event in info["specimen_route_events"]:
            self.assertEqual(event["collection_facility"], 0)
            self.assertEqual(event["origin_facility"], 0)
            self.assertEqual(event["destination_facility"], 1)
            self.assertGreater(event["distance_miles"], 0.0)
            self.assertEqual(event["lead_time_epochs"], 1)
            self.assertGreater(event["transfer_cost"], 0.0)

    def test_routing_uses_no_exogenous_rng_and_preserves_paired_crn(self) -> None:
        crn_base = _base(
            demand_rates=(1.2, 0.8, 0.4),
            initial_specimens=(2.0, 0.0, 0.0),
            initial_idle_bioreactors=(0.0, 0.0, 0.0),
            initial_reagents=(0.0, 0.0, 0.0),
        )
        routing = _env(base=crn_base, enabled=True, lead_time=1, seed=21)
        control = _env(base=crn_base, enabled=False, lead_time=1, seed=21)
        routing.reset(seed=8001)
        control.reset(seed=8001)
        action = _route_action(routing, 0, 1, 1)

        for _ in range(4):
            _or, _rr, _dr, route_info = routing.step(action)
            _oc, _rc, _dc, control_info = control.step(action)
            np.testing.assert_array_equal(route_info["demand"], control_info["demand"])
            self.assertEqual(
                json.dumps(routing.rng.bit_generator.state, sort_keys=True),
                json.dumps(control.rng.bit_generator.state, sort_keys=True),
            )
            self.assertEqual(set(routing.patient_registry), set(control.patient_registry))
            for patient_id in routing.patient_registry:
                route_patient = routing.patient_registry[patient_id]
                control_patient = control.patient_registry[patient_id]
                self.assertEqual(route_patient.health_index, control_patient.health_index)
                self.assertEqual(
                    route_patient.deterioration_epoch,
                    control_patient.deterioration_epoch,
                )

    def test_full_state_round_trip_preserves_transit_ids_rng_and_next_step(self) -> None:
        base = _base(
            initial_idle_bioreactors=(0.0, 0.0, 0.0),
            initial_reagents=(0.0, 0.0, 0.0),
            demand_rates=(0.5, 0.2, 0.1),
        )
        original = _env(base=base, lead_time=1, seed=31)
        original.reset(seed=3131)
        original.step(_route_action(original, 0, 1, 1))
        snapshot = original.state_dict()

        restored = _env(base=base, lead_time=1, seed=999)
        restored.reset(seed=999)
        restored.load_state_dict(snapshot)

        self.assertEqual(
            [transit.patient_id for transit in original.specimen_transits],
            [transit.patient_id for transit in restored.specimen_transits],
        )
        self.assertEqual(set(original.patient_registry), set(restored.patient_registry))
        self.assertEqual(
            json.dumps(original.rng.bit_generator.state, sort_keys=True),
            json.dumps(restored.rng.bit_generator.state, sort_keys=True),
        )
        next_original = original.step(original.noop_action())
        next_restored = restored.step(restored.noop_action())
        np.testing.assert_array_equal(next_original[0], next_restored[0])
        self.assertEqual(next_original[1], next_restored[1])
        self.assertEqual(next_original[2], next_restored[2])
        np.testing.assert_array_equal(
            next_original[3]["waiting_patients"],
            next_restored[3]["waiting_patients"],
        )

    def test_disabled_mode_matches_frozen_ce9b627_trajectory(self) -> None:
        base = CapacityPlanningConfig(
            num_facilities=2,
            production_lead_time=2,
            episode_horizon=4,
            demand_rates=(0.0, 0.0),
            initial_specimens=(2.0, 1.0),
            initial_reagents=(0.0, 0.0),
            initial_idle_bioreactors=(0.0, 0.0),
            max_specimens=(10.0, 10.0),
            max_reagents=(10.0, 10.0),
            max_idle_bioreactors=(2.0, 2.0),
            max_reagent_replenishment=(0.0, 0.0),
            max_specimen_transfer=2.0,
            action_mode="facility_net",
            include_supplier_state=True,
            supplier_disruption_rate=0.0,
        )
        env = PatientConditionCapacityEnv(
            PatientEnvConfig(
                base=base,
                patient=PatientConditionConfig(
                    healthy_decay_rate=0.0,
                    frail_decay_rate=0.0,
                ),
                material_shelf_life=10,
            ),
            seed=0,
        )
        observation = env.reset(seed=123)
        rows = [{"obs": observation.tolist()}]
        action = env.noop_action()
        action[:2] = (-1.0, 1.0)
        for _ in range(4):
            observation, reward, done, info = env.step(action)
            rows.append(
                {
                    "obs": observation.tolist(),
                    "reward": reward,
                    "done": done,
                    "waiting": info["waiting_patients"].tolist(),
                    "lost": info["patients_lost"].tolist(),
                    "routes": info["specimen_transfers"].tolist(),
                }
            )
        payload = json.dumps(
            rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "8fa138df33d5f280d89fbf5dcc055d10906101eac32530b371c21486d1d88529",
        )


if __name__ == "__main__":
    unittest.main()
