"""Hand-calculated raw JSON fixtures; no simulator, torch or scientific fit."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.rl.candidate_pilot_verification import (
    CAUSES, OPERATING, PATIENT, identity_counts, json_hash, paired_analysis, verify_episode, verify_raw_bundle,
)
from src.rl.candidate_pilot_resources import stream_manifest
from tests.test_candidate_pilot_resources import config


def patient(pid, status="waiting"):
    return {"patient_id": pid, "specimen_id": pid, "status": status,
            "material_facility": 0, "manufacturing_facility": None}


def raw_fixture():
    cfg = config()
    cfg["objective"].update(horizon=2, num_facilities=2, action_width=8)
    initial = {"scalars": {"t": 0, "_episode_seed": 123, "cumulative_enrolled": 2,
                           "cumulative_lost": 0, "cumulative_served": 0},
               "patients": {"p0": patient("p0"), "p1": patient("p1")},
               "patient_queues": [["p0", "p1"], []], "in_production_patients": [[[], []], [[], []]],
               "specimen_transits": [], "product_return_transits": []}
    final = copy.deepcopy(initial)
    final["scalars"].update(t=2, cumulative_enrolled=3, cumulative_lost=1, cumulative_served=1)
    final["patients"] = {"p0": patient("p0", "delivered"), "p1": patient("p1", "lost"), "p2": patient("p2")}
    final["patient_queues"] = [["p2"], []]
    header = {"initial_state": initial, "seed": 123, "source_id": "invented-only", "trajectory_id": "fake/0",
              "block": 60, "representation": "graph", "role": "ppo", "world_index": 0,
              "policy_sha256": "a" * 64, "split": "test"}
    # Half away from zero: +/-1.5 lots must become +/-2, not a truncation.
    request = [.0125, -.0125, 0., 0., 0., 0., 0., 0.]
    bank = {"class_keys": [[2, -2, 0., 0., 0., 0., 0., 0.]], "requests": [request, request],
            "request_to_class": [0, 0]}
    rows = []
    for step in range(2):
        info = {k: 1. for k in OPERATING + PATIENT}
        info.update(cost=11., base_cost=8., specimen_route_cost=1., transshipment_cost=3.,
            patients_lost=[step, 0], patients_completed=[step, 0], demand=[1 - step, 0],
            identity_active_count=3 - 2 * step, identity_terminal_count=2 * step,
            waiting_patients=[3 - 2 * step, 0], in_production_patients=[0, 0], specimen_in_transit=[0, 0],
            specimen_requested_integer_net=[2, -2], specimen_transfers=[0, 0], specimen_route_count=0,
            blocked_specimen_requests=2, blocked_specimen_inbound_requests=2, blocked_specimen_outbound_requests=2,
            completion_service_level=step / 3)
        info.update({k: [0, 0] for k in CAUSES})
        info["patients_lost_waiting_ineligible"] = [step, 0]
        record = {"semantics": {"reward_kind": "absolute_environment", "reward_scale": 1e-9, "gamma": 1.},
                  "origin": "trajectory", "source_id": "invented-only", "trajectory_id": "fake/0", "step_index": step,
                  "state_token": json_hash(initial) if step == 0 else "intermediate",
                  "next_state_token": "intermediate" if step == 0 else json_hash(final),
                  "state": [step], "next_state": [step + 1], "terminated": step == 1, "truncated": False,
                  "raw_reward": -11., "action": request}
        rows.append({"inference_seconds": .001, "event": {"info": info, "audit": {"record": record,
                     "terminal_cost_added": 0., "decision": {"evaluation": {"candidates": bank, "behavior_sha256": "a" * 64,
                     "log_probs": [0.], "inference_dtype": "float32"},
                     "choice": {"class_index": 0, "submitted_request": request}}}}})
    return cfg, header, rows, final


def analysis_fixture():
    cfg = config()
    streams = stream_manifest(cfg)
    # Fewer artificial worlds, not changed scientific config or live execution.
    cfg["evaluation"].update(episodes_per_policy=2, total_episodes=66)
    policies = [(r["name"], role) for r in cfg["representations"] for role in cfg["candidate_roles"]]
    policies += [("reference", role) for role in cfg["reference_roles"]]
    outcomes = []
    for b in cfg["blocks"]:
        for rep, role in policies:
            for world in range(2):
                outcomes.append({"block": b, "representation": rep, "role": role, "world_index": world,
                    "seed": streams["environment"]["test"][str(b)][world], "split": "test",
                    "trajectory_id": f"fake/{b}/{rep}/{role}/{world}", "cost": 90. if (rep, role) == ("graph", "ppo") else 100.,
                    "losses": 1, "completions": 8, "terminal_active": 1, "enrolled": 10,
                    "initial_state_sha256": f"invented/{b}/{world}", "policy_sha256": f"model/{b}/{rep}/{role}"})
    return cfg, streams, outcomes


class CandidatePilotVerificationTests(unittest.TestCase):
    def test_raw_components_ids_losses_completions_and_rounding_reconcile(self):
        cfg, header, rows, final = raw_fixture()
        result = verify_episode(header, rows, final, cfg)
        self.assertEqual(result["cost"], 22.)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["completions"], 1)
        self.assertEqual(result["terminal_active"], 1)
        self.assertEqual(result["initial_enrolled"], 2)
        self.assertEqual(result["enrolled"], 3)
        self.assertEqual(result["blocked_requests"], 4)
        self.assertEqual(result["reference_class_choices"], 2)
        self.assertEqual(result["inference_seconds"], .002)
        self.assertEqual(result["arrivals"], [[1, 0], [0, 0]])
        self.assertEqual(result["components"]["patient_loss_cost"], 2.)

    def test_finished_expiry_is_a_separate_cause_not_silently_dropped(self):
        cfg, header, rows, final = raw_fixture()
        rows[-1]["event"]["info"]["patients_lost_waiting_ineligible"] = [0, 0]
        rows[-1]["event"]["info"]["finished_expired"] = [1, 0]
        result = verify_episode(header, rows, final, cfg)
        self.assertEqual(result["loss_causes"]["finished_expired"], 1)

    def test_raw_cost_reward_seed_termination_support_and_nan_corruption_rejected(self):
        mutations = [lambda h, r, f: r[0]["event"]["info"].update(cost=12.),
                     lambda h, r, f: r[0]["event"]["audit"]["record"].update(raw_reward=-1.),
                     lambda h, r, f: h.update(seed=999),
                     lambda h, r, f: r[0]["event"]["audit"]["record"].update(terminated=True),
                     lambda h, r, f: r[0]["event"]["audit"]["record"].update(step_index=1),
                     lambda h, r, f: r[0].update(inference_seconds=float("nan")),
                     lambda h, r, f: r[0]["event"]["info"].update(specimen_requested_integer_net=[1, -1]),
                     lambda h, r, f: r[0]["event"]["audit"]["decision"]["choice"].update(class_index=1),
                     lambda h, r, f: r[0]["event"]["audit"]["decision"]["evaluation"].update(behavior_sha256="b" * 64),
                     lambda h, r, f: r[0]["event"]["info"].update(hidden_cost=3.),
                     lambda h, r, f: r.pop()]
        for change in mutations:
            cfg, header, rows, final = raw_fixture()
            change(header, rows, final)
            with self.assertRaises(ValueError):
                verify_episode(header, rows, final, cfg)

    def test_duplicate_missing_or_substituted_identity_rejected(self):
        for mutate in (lambda s: s["patient_queues"][0].append("p0"),
                       lambda s: s["patient_queues"][0].remove("p0"),
                       lambda s: s["patients"]["p0"].update(specimen_id="p1"),
                       lambda s: s["scalars"].update(cumulative_enrolled=3)):
            _, header, _, _ = raw_fixture()
            state = header["initial_state"]
            mutate(state)
            with self.assertRaises(ValueError):
                identity_counts(state)

    def test_prespecified_primary_all_secondary_intervals_and_positive_triage(self):
        cfg, streams, rows = analysis_fixture()
        result = paired_analysis(rows, cfg, streams)
        self.assertEqual(result["decision"], "promising_for_discussion_only")
        self.assertEqual(len(result["contrasts"]), 12)
        primary = result["contrasts"][0]
        self.assertEqual(primary["grand_mean_cost_difference"], -10.)
        self.assertEqual(primary["grand_relative_cost_gain_percent"], 10.)
        self.assertEqual(primary["descriptive_t_ci95_df2"], [-10., -10.])
        self.assertEqual(primary["blocks"][0]["paired_world_bootstrap_ci95"], [-10., -10.])
        self.assertEqual(result, paired_analysis(rows, cfg, streams))
        self.assertFalse(result["automatic_followon"])
        self.assertFalse(result["clean_ddpg_superiority_claim"])

    def test_cost_benefit_with_clinical_harm_is_tradeoff_even_in_one_block(self):
        cfg, streams, rows = analysis_fixture()
        for row in rows:
            if row["block"] == 60 and row["representation"] == "graph" and row["role"] == "ppo":
                row["completions"] -= 1
        result = paired_analysis(rows, cfg, streams)
        self.assertEqual(result["decision"], "cost_clinical_tradeoff")
        self.assertFalse(result["triage_checks"]["no_adverse_clinical_block_direction"])

    def test_inconsistent_or_no_cost_gain_cannot_advance(self):
        cfg, streams, rows = analysis_fixture()
        for row in rows:
            if row["block"] == 60 and row["representation"] == "graph" and row["role"] == "ppo":
                row["cost"] = 120.
        result = paired_analysis(rows, cfg, streams)
        self.assertEqual(result["decision"], "limited_negative_or_inconclusive")
        self.assertFalse(result["triage_checks"]["all_blocks_lower_cost"])

    def test_missing_duplicate_wrong_stream_changed_policy_or_unpaired_state_rejected(self):
        for change in (lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0])),
                       lambda rows: rows[0].update(seed=999), lambda rows: rows[0].update(split="training"),
                       lambda rows: rows[0].update(policy_sha256="changed"),
                       lambda rows: rows[0].update(initial_state_sha256="unpaired"),
                       lambda rows: rows[0].update(cost=0)):
            cfg, streams, rows = analysis_fixture()
            change(rows)
            with self.assertRaises(ValueError):
                paired_analysis(rows, cfg, streams)

    def test_bundle_reopens_all_raw_files_and_detects_changed_bytes(self):
        cfg, streams, outcomes = analysis_fixture()
        cfg["objective"].update(horizon=2, num_facilities=2, action_width=8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            index = []
            def save(name, data, jsonl=False):
                path = root / name
                text = "".join(json.dumps(row) + "\n" for row in data) if jsonl else json.dumps(data)
                path.write_text(text)
                return {"path": name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for i, outcome in enumerate(outcomes):
                _, header, rows, final = raw_fixture()
                header.update({k: outcome[k] for k in ("block", "representation", "role", "world_index", "seed", "trajectory_id")})
                header["initial_state"]["scalars"]["_episode_seed"] = outcome["seed"]
                final["scalars"]["_episode_seed"] = outcome["seed"]
                rows[0]["event"]["audit"]["record"]["state_token"] = json_hash(header["initial_state"])
                rows[-1]["event"]["audit"]["record"]["next_state_token"] = json_hash(final)
                for row in rows:
                    row["event"]["audit"]["record"]["trajectory_id"] = outcome["trajectory_id"]
                index.append({"header": save(f"{i}-header.json", header), "events": save(f"{i}-events.jsonl", rows, True),
                              "final_state": save(f"{i}-final.json", final)})
            report = verify_raw_bundle(root, index, cfg, streams)
            self.assertEqual(len(report["files"]), 198)
            self.assertEqual(len(report["outcomes"]), 66)
            self.assertEqual(report["analysis"]["decision"], "limited_negative_or_inconclusive")
            with (root / index[0]["events"]["path"]).open("a") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_raw_bundle(root, index, cfg, streams)


if __name__ == "__main__":
    unittest.main()
