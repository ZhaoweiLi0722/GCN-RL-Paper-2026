"""Invented JSON only; the public action formula is mocked, never simulated."""

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.baselines.heuristics import heuristic_settings_for_policy
from src.rl.candidate_pilot_verification import CAUSES, OPERATING, PATIENT, json_hash
from src.rl.cohort_bundle_verification import (
    CONTROLLERS, TRAINING_ROLES, _analysis, _schedule, read_cohort_raw_episodes, verify_cohort_raw_bundle,
)
from tests.test_time_baseline_verification import episode as prefix_episode, fixture_config as prefix_config


ROOT = Path(__file__).resolve().parents[1]
ACTION = [0., 0., -.125, .125, 0., 0., -1., -1.]
IDLE = [0.] * 6 + [-1.] * 2


def fixture_config():
    cfg, _ = prefix_config()
    cfg.pop("time_baseline_proposal")
    p = json.loads((ROOT / "specs/2026-10-02-terminal-obligation/proposal.json").read_text())
    # Deliberately small invented accounting windows, not a patient experiment.
    p.update(enrollment_steps=2, patient_resolution_steps=2, accounting_steps=3, economic_endpoint=5)
    cfg.update(cohort_proposal=p, final_controllers=list(CONTROLLERS))
    namespace, env = _schedule(cfg)
    streams = dict(format="cohort-objective-streams-v1", namespace=namespace, proposal_sha256=json_hash(p),
        environment={b: {k: [str(s) for s in seeds] for k, seeds in splits.items()} for b, splits in env.items()},
        neural={"analysis/bootstrap": str(int.from_bytes(hashlib.sha256(
            (namespace + "/analysis/bootstrap").encode()).digest()[:8], "big") % 2**63)})
    return cfg, streams


def invented_episode(config, streams, b=60, role="cohort_ppo", world=0, split="test", cost=100):
    header, rows, prefix = prefix_episode(config, streams, b, role, world, split, cost)
    header["representation"] = "reference" if role in ("r4", "full_mdl2") else "graph"
    if split != "test" and role in TRAINING_ROLES:
        header["selection"] = "sample"
    trajectory = f"{split}/block{b}/graph/{role}/episode{world:02d}"
    header["trajectory_id"] = trajectory
    anchor = dict(num_facilities=2, episode_horizon=2, production_lead_time=2,
        transfer_lead_time=2, max_reagents=[2., 2.], max_idle_bioreactors=[3., 3.],
        max_reagent_replenishment=[2., 2.], max_reagent_transfer=2., max_bioreactor_transfer=2.,
        max_specimen_transfer=120., demand_rates=[9., 9.], demand_rate_estimates=[7., 7.])
    header["session_manifest"] = dict(producer_definition=json_hash(dict(anchor_config=anchor,
        settings=asdict(heuristic_settings_for_policy("mdl2")))))
    for state in (header["initial_state"], prefix):
        for patient in state["patients"].values():
            patient.update(enrollment_epoch=0, collection_facility=0, health_index=1.,
                           deterioration_epoch=10, risk_type="invented", risk_multiplier=1.)
    resource = dict(reagents=[1.75, 2.], bioreactors=[[1.5, 0.], [1., 0.]],
        reagent_transfer_pipeline=[[.5, 0.], [0., 0.]], capacity_transfer_pipeline=[[0., 0.], [0., 0.]])
    prefix["arrays"] = copy.deepcopy(resource) | dict(demand=[2., 0.], demand_forecast=[3., 0.],
                                                       specimen_transfer_pipeline=[[0., 0.]])
    prefix["scalars"].update(demand_forecast_error=1.)
    for row in rows:
        row["event"]["audit"]["record"]["trajectory_id"] = trajectory
        row["event"]["audit"]["decision"]["evaluation"]["value"] = 0.
    rows[0]["event"]["audit"]["record"]["state_token"] = json_hash(header["initial_state"])
    rows[-1]["event"]["audit"]["record"]["next_state_token"] = json_hash(prefix)
    final = copy.deepcopy(prefix)
    final["patients"]["p2"]["status"] = "lost"
    final["patient_queues"] = [[], []]
    final["scalars"].update(t=5, cumulative_lost=2, demand_forecast_error=0.)
    final["arrays"].update(demand=[0., 0.], demand_forecast=[0., 0.],
                          reagent_transfer_pipeline=[[0., 0.], [0., 0.]])
    closed = copy.deepcopy(prefix)
    closed["arrays"].update(demand=[0., 0.], demand_forecast=[0., 0.])
    closed["scalars"]["demand_forecast_error"] = 0.
    token, tail = json_hash(closed), []
    for i in range(3):
        info = dict.fromkeys(OPERATING + PATIENT, 0.)
        info.update(reagent_holding_cost=1., base_cost=1., cost=1., specimen_route_cost=0., transshipment_cost=0.,
            demand=[0, 0], patients_lost=[int(i == 0), 0], patients_completed=[0, 0],
            waiting_patients=[0, 0], in_production_patients=[0, 0], specimen_in_transit=[0, 0],
            identity_active_count=0, identity_terminal_count=3, production=[0, 0],
            replenishment=[0., 0.], reagent_transfer_arrivals=[.5, 0.] if i == 0 else [0., .25] if i == 1 else [0., 0.],
            capacity_transfer_arrivals=[0., 0.], reagent_transfers=[-.25, .25] if i == 0 else [0., 0.],
            capacity_transfers=[0., 0.], specimen_transfers=[0, 0], specimen_route_count=0,
            blocked_specimen_requests=0, average_turnaround_time=2., completion_service_level=1 / 3)
        info.update({k: [0, 0] for k in CAUSES})
        info["patients_lost_waiting_ineligible"] = [int(i == 0), 0]
        after = copy.deepcopy(resource)
        after["reagent_transfer_pipeline"] = [[0., .25], [0., 0.]] if i == 0 else [[0., 0.], [0., 0.]]
        after_token = json_hash(final) if i == 2 else json_hash({"invented_tail_step": i + 1})
        observation = [0., float(i == 0), resource["reagents"][0], 1.5, 0., 0., 0., 2., 1., 0.]
        tail.append(dict(index=i + 1, action=ACTION.copy() if i == 0 else IDLE.copy(), info=info,
            cost=1., raw_reward=-1., active=0, resolution_step=1, accounting_done=i == 2,
            before_state_sha256=token, after_state_sha256=after_token, public_observation=observation,
            resources_before=copy.deepcopy(resource), resources_after=copy.deepcopy(after)))
        token, resource = after_token, after
    contract = {k: config["cohort_proposal"][k] for k in ("enrollment_steps", "patient_resolution_steps", "accounting_steps")}
    contract["followup_rule"] = "full_mdl2_until_resolution_then_no_new_commitments"
    tail_header = {k: header[k] for k in ("block", "representation", "role", "world_index", "seed", "trajectory_id", "source_id", "split")}
    tail_header.update(format="cohort-tail-header-v1", contract=contract,
                       prefix_final_sha256=json_hash(prefix), anchor_config=anchor)
    return (header, rows, prefix), (tail_header, tail, final)


def save(root, name, value, jsonl=False):
    raw = ("".join(json.dumps(r) + "\n" for r in value) if jsonl else json.dumps(value)).encode()
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return dict(path=str(name), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save_episode(root, name, values):
    result = {}
    for part, payload in zip(("prefix", "tail"), values):
        if part == "tail":
            payload[0]["prefix"] = result["prefix"]
        result[part] = {key: save(root, f"{name}/{part}/{filename}", val, key == "events")
            for key, filename, val in zip(("header", "events", "final_state"),
                                         ("header.json", "events.jsonl", "final_state.json"), payload)}
    return result


class CohortBundleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cfg, self.streams = fixture_config()
        self.action = patch("src.rl.cohort_bundle_verification.facility_net_action_from_state", return_value=np.asarray(ACTION))
        self.mock_action = self.action.start()
        self.addCleanup(self.action.stop)

    def read(self, values=None):
        values = values or invented_episode(self.cfg, self.streams)
        entry = save_episode(self.root, "single", values)
        return read_cohort_raw_episodes(self.root, [entry], self.cfg)

    def bundle(self, full=False, cost=None, mutate=None):
        index = []
        for split in (("training", "preflight", "test") if full else ("test",)):
            for b in self.cfg["blocks"]:
                for role in TRAINING_ROLES if split == "training" else CONTROLLERS:
                    for w in range({"training": 32, "preflight": 1, "test": 12}[split]):
                        c = cost(b, role, w) if cost else (99.5 if role == "cohort_ppo" else 100.)
                        values = invented_episode(self.cfg, self.streams, b, role, w, split, c)
                        if mutate:
                            mutate(values)
                        index.append(save_episode(self.root, f"{split}/{b}/{role}/{w}", values))
        return index

    def test_raw_primitives_fractional_clipping_and_zero_rate_overlay(self):
        files, outcomes = self.read()
        self.assertEqual(len(files), 6)
        row = outcomes[0]
        self.assertEqual((row["cost"], row["window"]["cost"], row["tail"]["tail_cost"]), (103., 100., 3.))
        self.assertEqual((row["losses"], row["completions"], row["terminal_active"]), (2, 1, 0))
        flow = row["tail"]["resource_flow"]
        self.assertEqual(flow[0]["reagents"]["arrival_clipped_overflow"], [.25, 0.])
        self.assertEqual(flow[1]["reagents"]["arrival_clipped_overflow"], [0., .25])
        self.assertEqual(row["tail"]["retained_reagents"], [1.75, 2.])
        overlay = self.mock_action.call_args.args[1]
        self.assertEqual((overlay["demand_rates"], overlay["demand_rate_estimates"]), ([0., 0.], [0., 0.]))
        self.assertEqual(self.mock_action.call_count, 1)

    def test_tail_action_hash_resource_and_cost_tampering_rejected(self):
        changes = [lambda v: v[1][1][0].update(action=IDLE.copy()),
            lambda v: v[1][1][1].update(action=[0.] * 8),
            lambda v: v[1][1][0].update(before_state_sha256="f" * 64),
            lambda v: v[1][1][0]["resources_after"]["reagents"].__setitem__(0, 1.8),
            lambda v: v[1][1][0]["info"].update(reagent_transfers=[-.2, .25]),
            lambda v: v[1][1][0]["info"].update(patient_loss_cost=10.),
            lambda v: v[1][1][0]["public_observation"].__setitem__(0, 1.),
            lambda v: v[1][0]["anchor_config"].update(demand_rates=[0., 0.]),
            lambda v: v[1][1].pop()]
        for i, change in enumerate(changes):
            values = invented_episode(self.cfg, self.streams)
            change(values)
            with self.subTest(i=i), self.assertRaises(ValueError):
                self.read(values)

    def test_prefix_fractional_cost_transfers_and_replenishment_preserved(self):
        values = invented_episode(self.cfg, self.streams, cost=100.25)
        row = self.read(values)[1][0]["window"]
        self.assertEqual(row["cost"], 100.25)
        resource = row["actions"][0]["executed"]
        self.assertEqual(resource["capacity_transfers"], [.5, -.5])
        self.assertEqual(resource["reagent_transfers"], [-.125, .125])
        self.assertEqual(resource["replenishment"], [1.25, 0.])
        values[0][1][0]["event"]["info"]["capacity_transfers"] = [.5, -.25]
        with self.assertRaisesRegex(ValueError, "conserve flow"):
            self.read(values)

    def test_live_zero_procurement_buffer_and_integral_float_counters(self):
        values = invented_episode(self.cfg, self.streams)
        header, prefix_rows, prefix = values[0]
        tail_header, tail, final = values[1]
        for state in (header["initial_state"], prefix, final):
            for key in ("cumulative_enrolled", "cumulative_lost", "cumulative_served"):
                state["scalars"][key] = float(state["scalars"][key])
        prefix_rows[0]["event"]["audit"]["record"]["state_token"] = json_hash(header["initial_state"])
        prefix_rows[-1]["event"]["audit"]["record"]["next_state_token"] = json_hash(prefix)
        tail_header["prefix_final_sha256"] = json_hash(prefix)
        closed = copy.deepcopy(prefix)
        closed["arrays"].update(demand=[0., 0.], demand_forecast=[0., 0.])
        closed["scalars"]["demand_forecast_error"] = 0.
        tail[0]["before_state_sha256"], tail[-1]["after_state_sha256"] = json_hash(closed), json_hash(final)
        for row in tail:
            for key in ("resources_before", "resources_after"):
                row[key]["reagent_purchase_pipeline"] = [[0., 0.]]
        self.assertEqual(self.read(values)[1][0]["enrolled"], 3)
        values[0][0]["initial_state"]["scalars"]["cumulative_enrolled"] = 2.5
        with self.assertRaises(ValueError):
            self.read(values)

    def test_source_byte_hash_path_alias_and_duplicate_slot_rejected(self):
        values = invented_episode(self.cfg, self.streams)
        entry = save_episode(self.root, "a", values)
        path = self.root / entry["tail"]["events"]["path"]
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaises(ValueError):
            read_cohort_raw_episodes(self.root, [entry], self.cfg)
        path.write_bytes(original)
        duplicate = save_episode(self.root, "b", invented_episode(self.cfg, self.streams))
        with self.assertRaisesRegex(ValueError, "duplicate trajectory"):
            read_cohort_raw_episodes(self.root, [entry, duplicate], self.cfg)
        entry["tail"]["events"]["path"] = "./" + entry["tail"]["events"]["path"]
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            read_cohort_raw_episodes(self.root, [entry], self.cfg)

    def test_complete_six_controller_matrix_no_inherited_one_percent_gate(self):
        index = self.bundle()
        report = verify_cohort_raw_bundle(self.root, index, self.cfg, self.streams)
        self.assertEqual(report["inventory"], {"test": 216})
        self.assertEqual(len(report["files"]), 1296)
        self.assertTrue(report["analysis"]["promotion_screen_met"])
        self.assertFalse(report["analysis"]["practical_effect_threshold_applied"])
        self.assertFalse(report["training_inventory_verified"])
        self.assertEqual(report["analysis"]["contrasts"][0]["hierarchical_paired_ci95"]["cost"], [-.5, -.5])
        self.assertFalse(report["grants_scientific_execution_authorization"])
        reversed_report = verify_cohort_raw_bundle(self.root, index[::-1], self.cfg, self.streams)
        self.assertEqual(report["analysis"], reversed_report["analysis"])
        # Direction-screen unit check on invented analysis rows only.
        rows = copy.deepcopy(report["outcomes"])
        for row in rows:
            if row["role"] == "cohort_ppo" and row["block"] == 60:
                row["losses"] += 1
        screen = _analysis(rows, self.cfg, self.streams, int(self.streams["neural"]["analysis/bootstrap"]))
        self.assertTrue(screen["primary_cost_screen_met"])
        self.assertFalse(screen["observed_loss_direction_screen_met"])
        self.assertFalse(screen["promotion_screen_met"])

    def test_full_training_preflight_inventory_and_missing_entries(self):
        index = self.bundle(full=True)
        report = verify_cohort_raw_bundle(self.root, index, self.cfg, self.streams)
        self.assertEqual(report["inventory"], {"preflight": 18, "test": 216, "training": 288})
        self.assertTrue(report["training_inventory_verified"])
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_cohort_raw_bundle(self.root, index[1:], self.cfg, self.streams)

    def test_paired_start_policy_stream_and_all_three_cost_directions(self):
        def unpaired(values):
            h, rows, _ = values[0]
            if h["role"] == "cohort_ppo":
                h["initial_state"]["marker"] = 1
                rows[0]["event"]["audit"]["record"]["state_token"] = json_hash(h["initial_state"])
        with self.assertRaisesRegex(ValueError, "paired starts"):
            verify_cohort_raw_bundle(self.root, self.bundle(mutate=unpaired), self.cfg, self.streams)
        costs = lambda b, r, w: 80 if r == "bc_continue" else 90 if r == "cohort_ppo" else 100
        report = verify_cohort_raw_bundle(self.root, self.bundle(cost=costs), self.cfg, self.streams)
        self.assertFalse(report["analysis"]["promotion_screen_met"])
        streams = copy.deepcopy(self.streams)
        streams["environment"]["60"]["test"][0] = "01"
        with self.assertRaisesRegex(ValueError, "decimal-string"):
            verify_cohort_raw_bundle(self.root, [], self.cfg, streams)


if __name__ == "__main__":
    unittest.main()
