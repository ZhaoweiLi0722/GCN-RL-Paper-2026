"""Production-shaped invented raw data only; no simulator/model/optimizer calls."""

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from src.rl.candidate_pilot_verification import CAUSES, OPERATING, PATIENT, json_hash
from src.rl.paired_cohort_verification import (
    assemble_paired_cohort_labels, raw_json, verify_paired_cohort_branch,
)
from tests.test_candidate_pilot_verification import patient


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "experiments/configs/paired_cohort_improvement_20261002.json").read_text())


def context_seed(block, cohort):
    return block * 100 + cohort


def future_seed(block, cohort, start, replication):
    return 100000 + block * 1000 + cohort * 10 + [4, 20, 36].index(start) * 2 + replication


def make_patient(pid, status="waiting", epoch=0, age=2):
    return patient(pid, status) | dict(enrollment_epoch=epoch, collection_facility=0,
        health_index=.5, deterioration_epoch=3., risk_type=1, risk_multiplier=1., age=age)


def resources():
    return dict(reagents=[3.5000000000000004, 2.125], bioreactors=[[1.25, 0.], [2.5, 0.]],
        reagent_transfer_pipeline=[[0., 0.]], capacity_transfer_pipeline=[[0., 0.]],
        reagent_purchase_pipeline=[[0., 0.]])


def make_context(block=60, cohort=0, start=4):
    cfg = config()
    cid = f"context/{block}/{cohort}/{start}"
    state = dict(scalars=dict(t=start, _episode_seed=context_seed(block, cohort), cumulative_enrolled=4,
        cumulative_lost=1, cumulative_served=1, cumulative_turnaround_time=3., demand_forecast_error=.5),
        patients={"done": make_patient("done", "delivered", age=3), "lost": make_patient("lost", "lost"),
                  "p2": make_patient("p2"), "p3": make_patient("p3")},
        patient_queues=[["p2", "p3"], []], in_production_patients=[[[], []], [[], []]],
        specimen_transits=[], product_return_transits=[],
        arrays=resources() | dict(specimen_transfer_pipeline=[[0., 0.]], demand=[1., 0.], demand_forecast=[1.5, 0.]),
        rng_state=np.random.PCG64(context_seed(block, cohort)).state)
    tail = [0., 0., 0., 0., -1., -1.]
    x = .012500000000000002
    requests = [[0., 0., *tail], [x, -x, *tail], [-x, x, *tail], [.0001, -.0001, *tail]]
    keys = [[-2, 2, *tail], [0, 0, *tail], [2, -2, *tail]]
    token = json_hash(state)
    bank = dict(schema=dict(action_schema_id="invented-public/action", num_facilities=2,
        max_specimen_transfer=120., max_candidates=6), state_token=token, requests=requests,
        class_keys=keys, members=[[2], [0, 3], [1]], representatives=[2, 0, 1], request_to_class=[1, 2, 0, 1])
    result = dict(format="paired-cohort-context-v1", context_id=cid, block=block, cohort=cohort,
        after_prefix_steps=start, source_id=cfg["rng_namespace"], reference_sha256=json_hash({"reference": block}),
        options={"invented": True}, environment=state, environment_sha256=token,
        cohort_spec=dict(enrollment_steps=52, patient_resolution_steps=8, accounting_steps=11,
            followup_rule="full_mdl2_until_resolution_then_no_new_commitments"),
        public_example=dict(split="training", identity=cid, actor_state=[0.] * 7 + np.asarray(requests[1], dtype=np.float32).tolist(),
                            candidates=bank),
        anchor_config=dict(num_facilities=2, episode_horizon=52, production_lead_time=2,
            max_specimen_transfer=120., transfer_lead_time=1))
    result["context_sha256"] = json_hash(result)
    return result


def close_state(prefix):
    state = copy.deepcopy(prefix)
    state["arrays"].update(demand=[0., 0.], demand_forecast=[0., 0.])
    state["scalars"]["demand_forecast_error"] = 0.
    return state


def refresh(payload):
    rows = payload["rows"]
    payload["manifest"]["initial_state_sha256"] = json_hash(raw_json(payload["initial_state"]))
    payload["receipt"] = dict(manifest=copy.deepcopy(payload["manifest"]), steps=len(rows),
        remaining_raw_cost=math.fsum(float(row["info"]["cost"]) for row in rows),
        prefix_raw_cost=math.fsum(float(row["info"]["cost"]) for row in rows if row["stage"] == "prefix"),
        tail_raw_cost=math.fsum(float(row["info"]["cost"]) for row in rows if row["stage"] == "tail"),
        raw_rows_sha256=json_hash(raw_json(rows)), final_state_sha256=json_hash(raw_json(payload["final_state"])),
        terminal_active=0, included_sunk_cost=False)


def branch(context=None, k=0, replication=0):
    context = make_context() if context is None else context
    start, b, c = (context[key] for key in ("after_prefix_steps", "block", "cohort"))
    seed = future_seed(b, c, start, replication)
    initial = copy.deepcopy(context["environment"])
    initial["rng_state"] = np.random.PCG64(seed).state
    prefix = copy.deepcopy(initial)
    prefix["scalars"].update(t=52, cumulative_enrolled=5, cumulative_served=2, cumulative_turnaround_time=6.)
    prefix["patients"]["p2"].update(status="delivered", age=3)
    prefix["patients"]["p4"] = make_patient("p4", epoch=start, age=1)
    prefix["patient_queues"] = [["p3", "p4"], []]
    after_resources = resources()
    after_resources["reagents"] = [3.8750000000000004, 1.875]
    after_resources["bioreactors"] = [[1.125, 0.], [2.625, 0.]]
    prefix["arrays"].update(copy.deepcopy(after_resources))
    final = close_state(prefix)
    final["scalars"].update(t=63, cumulative_lost=2, cumulative_served=3, cumulative_turnaround_time=8.)
    final["patients"]["p3"]["status"] = "lost"
    final["patients"]["p4"].update(status="delivered", age=2)
    final["patient_queues"] = [[], []]
    bank = context["public_example"]["candidates"]
    request = copy.deepcopy(bank["requests"][bank["representatives"][k]])
    manifest = dict(format="paired-cohort-branch-v1", branch_id=f"branch/{context['context_id']}/{k}/{replication}",
        **{key: context[key] for key in ("context_sha256", "source_id", "context_id", "block", "cohort",
                                         "after_prefix_steps", "reference_sha256")},
        endpoint=63, candidate_index=k, first_request=request, replication=replication, future_seed=str(seed),
        initial_state_sha256=json_hash(initial), future_rng_only=True, event_aligned_crn_after_divergence_claimed=False)
    rows, token, stocks = [], json_hash(initial), resources()
    for i, t in enumerate(range(start, 63)):
        if t == 52:
            token = json_hash(close_state(prefix))
        action = copy.deepcopy(request if i == 0 else bank["requests"][0])
        info = dict.fromkeys(OPERATING + PATIENT, 0.)
        base, residual = (1e9, k * .25 + replication * .125) if i == 0 else (1., 0.)
        cost = math.fsum((base, residual))
        active = 2 if t < 52 else 0
        info.update(reagent_holding_cost=base, urgency_cost=residual, base_cost=base, cost=cost,
            specimen_route_cost=0., transshipment_cost=0., demand=[int(i == 0), 0],
            patients_lost=[int(t == 52), 0], patients_completed=[int(i == 0 or t == 52), 0],
            identity_active_count=active, identity_terminal_count=3 if t < 52 else 5,
            waiting_patients=[active, 0], in_production_patients=[0, 0], specimen_in_transit=[0, 0], production=[0, 0],
            specimen_transfers=[0, 0], specimen_route_count=0,
            capacity_transfers=[-.125, .125] if i == 0 else [0., 0.],
            reagent_transfers=[.25, -.25] if i == 0 else [0., 0.],
            replenishment=[.125, 0.] if i == 0 else [0., 0.],
            average_turnaround_time=3. if t < 52 else 8 / 3, completion_service_level=.4 if t < 52 else .6)
        info.update({key: [0, 0] for key in CAUSES})
        info["patients_lost_waiting_ineligible"] = [int(t == 52), 0]
        info["specimen_requested_integer_net"] = [int(v) for v in np.sign(action[:2]) * np.floor(np.abs(action[:2]) * 120 + .5)]
        inbound = sum(max(v, 0) for v in info["specimen_requested_integer_net"])
        outbound = sum(max(-v, 0) for v in info["specimen_requested_integer_net"])
        info.update(blocked_specimen_requests=max(inbound, outbound), blocked_specimen_inbound_requests=inbound,
                    blocked_specimen_outbound_requests=outbound)
        after = (json_hash(prefix) if t == 51 else json_hash(final) if t == 62
                 else json_hash(dict(invented_step=t + 1, branch=manifest["branch_id"])))
        row = dict(stage="prefix" if t < 52 else "tail", absolute_step=t, branch_step=i,
            action_source="candidate" if i == 0 else "fixed_r4" if t < 52 else "common_tail",
            action=action, action_dtype="float64", info=info, cost=cost, raw_reward=-cost,
            public_observation=[0., 2., 3.5, 1.25, 0., 0., 0., 2.125, 2.5, 0.],
            before_state_sha256=token, after_state_sha256=after,
            resources_before=copy.deepcopy(stocks), resources_after=copy.deepcopy(after_resources))
        if t >= 52:
            row.update(index=t - 51, active=0, resolution_step=1, accounting_done=t == 62)
        rows.append(row)
        token, stocks = after, after_resources
    payload = dict(manifest=manifest, initial_state=initial, prefix_final=prefix, final_state=final, rows=rows)
    refresh(payload)
    return context, payload


def verify(context, payload, **changes):
    m = payload["manifest"]
    options = dict(config=config(), expected_context_seed=context_seed(context["block"], context["cohort"]),
                   expected_future_seed=future_seed(context["block"], context["cohort"], context["after_prefix_steps"], m["replication"]))
    return verify_paired_cohort_branch(context, **payload, **(options | changes))


class PairedCohortVerificationTests(unittest.TestCase):
    def test_full_record_with_128bit_context_and_future_seeds(self):
        from src.rl.paired_cohort_resources import stream_manifest
        streams = stream_manifest(config(), {str(b): str(b) for b in (60, 61, 62)})
        world = int(streams["environment"]["60"]["context"][0])
        future = int(streams["conditional_future"]["block60/cohort0/after4"][0])
        with patch(__name__ + ".context_seed", return_value=world), \
                patch(__name__ + ".future_seed", return_value=future):
            context, payload = branch()
            result = verify(context, payload)
        self.assertGreater(world, 2**63)
        self.assertGreater(future, 2**63)
        self.assertEqual(result["future_seed"], str(future))

    def test_complete_midprefix_branch_uses_captured_not_zero_patient_counts(self):
        for start in (4, 20, 36):
            context, payload = branch(make_context(start=start))
            original = copy.deepcopy((context, payload))
            result = verify(context, payload)
            self.assertEqual(result["steps"], 63 - start)
            self.assertEqual(result["remaining_raw_cost"], 1e9 + 62 - start)
            self.assertEqual(result["tail_raw_cost"], 11.)
            self.assertEqual((result["remaining_losses"], result["remaining_completions"], result["remaining_enrolled"]), (1, 2, 1))
            self.assertEqual((result["initial_compartments"]["lost"], result["initial_compartments"]["delivered"]), (1, 1))
            self.assertEqual(result["remaining_turnaround_sum"], 5.)
            self.assertEqual(result["waiting_patient_steps"], (52 - start) * 2)
            self.assertFalse(result["full_new_episode_outcome"])
            self.assertFalse(result["included_sunk_cost"])
            self.assertFalse(result["controller_requests_independently_replayed"])
            self.assertEqual((context, payload), original)
            json.dumps(result, allow_nan=False)

    def test_binary64_cost_residual_and_fractional_resources_survive(self):
        context, payload = branch(k=2, replication=1)
        result = verify(context, payload)
        self.assertEqual(result["remaining_raw_cost"], 1e9 + 58 + .625)
        self.assertNotEqual(float(np.float32(result["remaining_raw_cost"])), result["remaining_raw_cost"])
        self.assertEqual(result["actions"][0]["requested"], payload["manifest"]["first_request"])
        self.assertEqual(result["actions"][0]["executed"]["capacity_transfers"], [-.125, .125])
        self.assertEqual(result["tail"]["retained_reagents"], [3.8750000000000004, 1.875])

    def test_decoded_numpy_evidence_hash_matches_json_without_checkpoint_imports(self):
        context, payload = branch()
        for row in payload["rows"]:
            row["action"] = np.asarray(row["action"], dtype=np.float64)
            row.pop("action_dtype")
            row["info"]["capacity_transfers"] = np.asarray(row["info"]["capacity_transfers"], dtype=np.float64)
        refresh(payload)
        decoded = verify(context, payload)
        self.assertEqual(decoded["raw_rows_sha256"], payload["receipt"]["raw_rows_sha256"])
        with self.assertRaisesRegex(ValueError, "action_dtype"):
            verify(context, raw_json(payload))
        payload["rows"][0]["action"] = payload["rows"][0]["action"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "float64"):
            verify(context, payload)

    def test_existing_tail_reader_is_used_with_exact_11_rows(self):
        import src.rl.paired_cohort_verification as module
        context, payload = branch()
        with patch.object(module, "verify_cohort_tail", wraps=module.verify_cohort_tail) as reader:
            verify(context, payload)
        self.assertEqual(len(reader.call_args.args[2]), 11)
        self.assertEqual(reader.call_args.kwargs["patient_resolution_steps"], 8)

    def test_raw_hash_receipt_manifest_and_seed_corruption_fail(self):
        mutations = [lambda c, p: p["receipt"].update(raw_rows_sha256="0" * 64),
            lambda c, p: p["receipt"].update(final_state_sha256="0" * 64),
            lambda c, p: p["receipt"].update(included_sunk_cost=True),
            lambda c, p: p["manifest"].update(future_seed="1"),
            lambda c, p: c.update(context_sha256="0" * 64),
            lambda c, p: p["manifest"].update(reference_sha256="0" * 64),
            lambda c, p: p["receipt"].update(remaining_raw_cost=p["receipt"]["remaining_raw_cost"] + 100),
            lambda c, p: p["initial_state"]["patients"]["p2"].update(health_index=.9),
            lambda c, p: p["initial_state"]["rng_state"].update(has_uint32=1),
            lambda c, p: p["final_state"]["scalars"].update(_episode_seed=100)]
        for i, change in enumerate(mutations):
            context, payload = branch()
            change(context, payload)
            with self.subTest(i=i), self.assertRaises(ValueError):
                verify(context, payload)
        for value in (True, 1.5, "001", "+1"):
            context, payload = branch()
            with self.subTest(seed=value), self.assertRaises(ValueError):
                verify(context, payload, expected_future_seed=value)

    def test_cost_reward_counts_controller_chain_and_tail_tampering_with_fresh_hashes(self):
        mutations = [lambda p: p["rows"][0]["info"].update(base_cost=0),
            lambda p: p["rows"][0]["info"].update(hidden_cost=1),
            lambda p: p["rows"][0].update(raw_reward=0),
            lambda p: p["rows"][0]["info"].update(identity_terminal_count=1),
            lambda p: p["rows"][0]["info"].update(patients_completed=[0, 0]),
            lambda p: p["rows"][0].update(action_source="fixed_r4"),
            lambda p: p["rows"][1].update(action_source="candidate"),
            lambda p: p["rows"][1].update(absolute_step=99),
            lambda p: p["rows"][1].update(before_state_sha256="0" * 64),
            lambda p: p["rows"][-11].update(before_state_sha256=json_hash(p["prefix_final"])),
            lambda p: p["rows"][-11]["info"].update(demand=[1, 0]),
            lambda p: p["rows"][-11].update(resolution_step=0),
            lambda p: p["rows"][-2].update(action=[0.] * 8),
            lambda p: p["rows"][-1].update(accounting_done=False),
            lambda p: p["rows"][-11]["info"].update(patients_lost_waiting_ineligible=[0, 0]),
            lambda p: p["rows"].pop(),
            lambda p: p["rows"].insert(0, copy.deepcopy(p["rows"][0]))]
        for i, change in enumerate(mutations):
            context, payload = branch()
            change(payload)
            refresh(payload)
            with self.subTest(i=i), self.assertRaises(ValueError):
                verify(context, payload)

    def test_original_first_action_not_float32_surrogate_or_alias(self):
        for change in (lambda p: p["rows"][0].update(action=np.asarray(p["rows"][0]["action"], dtype=np.float32).tolist()),
                       lambda p: p["manifest"].update(candidate_index=1),
                       lambda p: p["rows"][0].update(action_dtype="float32")):
            context, payload = branch(k=2)
            change(payload)
            refresh(payload)
            with self.assertRaises(ValueError):
                verify(context, payload)
        context, payload = branch(k=1)
        alias = context["public_example"]["candidates"]["requests"][3]
        payload["manifest"]["first_request"] = alias
        payload["rows"][0]["action"] = alias
        refresh(payload)
        with self.assertRaisesRegex(ValueError, "representative"):
            verify(context, payload)

    def test_patient_and_specimen_integrality_resource_fractions_and_continuity(self):
        for key in ("demand", "production", "patients_lost", "patients_completed", "waiting_patients",
                    "in_production_patients", "specimen_in_transit", "specimen_requested_integer_net", "specimen_transfers", *CAUSES):
            context, payload = branch()
            payload["rows"][0]["info"][key][0] = .25
            refresh(payload)
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify(context, payload)
        for change in (lambda p: p["rows"][0]["info"].update(capacity_transfers=[.25, -.125]),
                       lambda p: p["rows"][0]["info"].update(replenishment=[-.25, 0]),
                       lambda p: p["rows"][1]["resources_before"]["reagents"].__setitem__(0, 8.),
                       lambda p: p["rows"][0]["resources_before"]["bioreactors"].__setitem__(0, [True, 0.])):
            context, payload = branch()
            change(payload)
            refresh(payload)
            with self.assertRaises(ValueError):
                verify(context, payload)

    def test_registry_attribute_resolution_and_terminal_pipeline_tamper(self):
        for change in (lambda p: p["final_state"]["patients"]["done"].update(status="lost"),
                       lambda p: p["prefix_final"]["patients"]["p2"].update(health_index=.6),
                       lambda p: p["final_state"]["arrays"]["specimen_transfer_pipeline"][0].__setitem__(0, .5),
                       lambda p: p["final_state"]["patients"].pop("p3"),
                       lambda p: p["final_state"]["scalars"].update(cumulative_turnaround_time=99)):
            context, payload = branch()
            change(payload)
            refresh(payload)
            with self.assertRaises(ValueError):
                verify(context, payload)

    def test_arrival_timing_cannot_be_shifted_while_retaining_total_enrollment(self):
        context, payload = branch()
        payload["rows"][0]["info"]["demand"] = [0, 0]
        payload["rows"][1]["info"]["demand"] = [1, 0]
        refresh(payload)
        with self.assertRaisesRegex(ValueError, "enrollment epochs"):
            verify(context, payload)

    def test_canonical_class_bool_and_manifest_bool_identity_are_not_integers(self):
        context, payload = branch()
        context["public_example"]["candidates"]["class_keys"][1][0] = False
        body = dict(context)
        body.pop("context_sha256")
        context["context_sha256"] = json_hash(body)
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            verify(context, payload)
        context, payload = branch()
        payload["manifest"]["cohort"] = False
        refresh(payload)
        with self.assertRaisesRegex(ValueError, "manifest"):
            verify(context, payload)

    def test_verifier_imports_no_model_environment_torch_or_optimizer(self):
        code = '''import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith(("src.env", "src.models", "src.baselines")):
        raise AssertionError("forbidden import: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import src.rl.paired_cohort_verification
'''
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)


class PairedCohortLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = config()
        cls.contexts = [make_context(b, c, t) for b in (60, 61, 62) for c in range(4) for t in (4, 20, 36)]
        cls.context_seeds = {str(b): [str(context_seed(b, c)) for c in range(4)] for b in (60, 61, 62)}
        cls.future_seeds = {ctx["context_id"]: [str(future_seed(ctx["block"], ctx["cohort"], ctx["after_prefix_steps"], r))
                           for r in range(2)] for ctx in cls.contexts}
        cls.branches = [branch(ctx, k, r)[1] for ctx in cls.contexts for k in range(3) for r in range(2)]

    def assemble(self, **changes):
        args = dict(contexts=self.contexts, branches=self.branches, config=self.config,
                    context_seeds=self.context_seeds, future_seeds=self.future_seeds)
        return assemble_paired_cohort_labels(**(args | changes))

    def test_exact_complete_matrix_float64_ordering_hash_and_no_hidden_actor_inputs(self):
        result = self.assemble(branches=list(reversed(self.branches)), contexts=list(reversed(self.contexts)))
        self.assertEqual(len(result["labels"]), 36)
        self.assertEqual(len(result["branch_outcomes"]), 216)
        label = result["labels"][0]
        self.assertEqual(label["raw_costs"].dtype, np.float64)
        self.assertEqual(label["raw_costs"].shape, (2, 3))
        self.assertFalse(label["raw_costs"].flags.writeable)
        np.testing.assert_array_equal(label["raw_costs"] - label["raw_costs"][0, 0], [[0., .25, .5], [.125, .375, .625]])
        self.assertEqual(label["reference_index"], 1)
        self.assertEqual(label["replication_seed_ids"], (tuple(self.future_seeds[label["context_id"]]),) * 3)
        self.assertEqual(set(label["public_example"]), {"split", "identity", "actor_state", "candidates"})
        body = dict(result)
        seal = body.pop("dataset_sha256")
        self.assertEqual(seal, json_hash(raw_json(body)))
        self.assertFalse(result["full_new_episode_outcome"])

    def test_missing_duplicate_foreign_context_or_seed_inventory(self):
        bad = copy.deepcopy(self.contexts)
        bad[0]["cohort"] = 4
        seeds = copy.deepcopy(self.future_seeds)
        seeds[next(iter(seeds))][1] = seeds[next(iter(seeds))][0]
        for changes in (dict(contexts=self.contexts[:-1]), dict(contexts=[*self.contexts, self.contexts[0]]),
                        dict(contexts=bad), dict(future_seeds=seeds), dict(future_seeds={}), dict(context_seeds={})):
            with self.subTest(fields=tuple(changes)), self.assertRaises(ValueError):
                self.assemble(**changes)

    def test_missing_duplicate_forbidden_class_or_replication(self):
        for mode in ("missing", "duplicate", "class", "replication", "wrong_seed"):
            # Put the invalid record first so rejection does not reverify an
            # unchanged full matrix except for the single missing-slot case.
            item = copy.deepcopy(self.branches[0])
            if mode in ("class", "replication", "wrong_seed"):
                item["manifest"][{"class": "candidate_index", "replication": "replication", "wrong_seed": "future_seed"}[mode]] = "7" if mode == "wrong_seed" else 99
                refresh(item)
                records = [item, *self.branches[1:]]
            elif mode == "duplicate":
                records = [item, item, *self.branches[1:]]
            else:
                records = self.branches[:-1]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.assemble(branches=records)


if __name__ == "__main__":
    unittest.main()
