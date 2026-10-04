"""Handwritten raw data and inert byte seals only; no scientific backend calls."""

from contextlib import contextmanager
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.rl import capacity_planner_tail_analysis as analysis
from src.rl.capacity_planner_tail_design import EVAL_ROLES, TRAIN_ROLES


ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT / "experiments/configs/capacity_planner_tail_20261004.json").read_text())
UNIT_COST = dict(plain_h8=1000., existing_frozen=980., observed_td=970.,
                 planner_tail_td=900., planner_tail_mc=905., plain_h16=890.)
ACTION = dict(plain_h8=[1., 2., 2., 1.], existing_frozen=[1.5, 1.5, 2., 1.],
              observed_td=[1., 1., 1., 1.], planner_tail_td=[1., 2., 1., 2.],
              planner_tail_mc=[1., 2., 1., 2.], plain_h16=[2., 2., 2., 2.])


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_raw(path, rows):
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_raw(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


@contextmanager
def preserved(*paths):
    before = {path: path.read_bytes() if path.exists() else None for path in paths}
    try:
        yield
    finally:
        for path, contents in before.items():
            if contents is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_bytes(contents)


def make_fixture(root):
    """480 artificial 64-row ledgers, not simulated trajectories."""
    study = copy.deepcopy(STUDY)
    study["analysis_inputs"] = dict(hardware={"description": "handcrafted fixture; no measured hardware"})
    for directory in ("raw", "summaries", "models", "tapes"):
        (root / directory).mkdir()
    ancestors, finals, manifest, events = {}, {}, dict(
        format="capacity-planner-tail-model-manifest-v1", ancestors=[], final_models=[]), []
    for b in range(5):
        name = f"block{b}-ancestor"
        raw = ("inert ancestor bytes " + str(b)).encode()
        sha = hashlib.sha256(raw).hexdigest()
        ancestors[b] = sha
        study["initial_models"]["sha256"][b] = sha
        (root / "models" / (name + ".pt")).write_bytes(raw)
        record = dict(sha256=sha, updates=1536, source_path="fixture-only-not-a-real-checkpoint",
                      historical_provenance="handcrafted, no model was loaded")
        write_json(root / "models" / (name + ".json"), record)
        manifest["ancestors"].append(dict(record, block=b))
        events.append(dict(event="ancestor_bound", block=b, sha256=sha))
        for role in TRAIN_ROLES:
            name = f"block{b}-{role}-final"
            raw = ("inert final bytes " + name).encode()
            sha = hashlib.sha256(raw).hexdigest()
            finals[(b, role)] = sha
            (root / "models" / (name + ".pt")).write_bytes(raw)
            record = dict(sha256=sha, updates=768, ancestor_sha256=ancestors[b], method=role,
                          config=dict(study["value"], method=role, architecture="graph", max_new_updates=768))
            write_json(root / "models" / (name + ".json"), record)
            manifest["final_models"].append(dict(record, block=b, role=role))
            events.append(dict(event="final_sealed", block=b, role=role, sha256=sha))
    write_json(root / "models" / "all-sealed.json", dict(
        ancestors={f"block{b}": sha for b, sha in ancestors.items()},
        final_models={f"block{b}-{r}": sha for (b, r), sha in finals.items()}))
    events.append(dict(event="all_models_sealed", counts=dict(value_optimizer_steps=11520, actor_optimizer_steps=0)))
    expected = list(analysis.expected_trajectories(study))
    for world, role, ident in expected:
        rows = []
        for t in range(64):
            action = ACTION[role] if t < 48 else [0.] * 4
            previous = ACTION[role] if 0 <= t - 1 < 48 else [0.] * 4
            applied = ACTION[role] if 0 <= t - 2 < 48 else [0.] * 4
            cost = UNIT_COST[role] + t + world["block"] + world["condition"] + world["replicate"]
            labor = 10. * sum(action)
            switch = 2. * sum(abs(a - b) for a, b in zip(action, previous))
            components = dict.fromkeys(analysis.COST_COMPONENTS, 0.)
            components.update(reagent_purchase_cost=cost - labor - switch,
                              support_flexible_labor_cost=labor, support_switching_cost=switch)
            record = dict(patient_id=f"patient-{world['seed']}", enrollment_epoch=0, collection_site=0,
                          status="delivered" if t == 63 else "waiting")
            receipt = dict(epoch=t, known_at=t + 1, raw_requested_hours=action,
                           committed_hours=action, applied_hours=applied, labor_cost=labor, switching_cost=switch)
            plan = dict(decision_wall_seconds=(t + 1) * .001 * (2 if role == "plain_h16" else 1)) if t < 48 else None
            rows.append(dict(epoch=t, world=world, role=role, cost=cost, reward=-cost,
                             components=components, patient_records=[record], new_lost_patients=0,
                             cumulative=dict(enrolled=1, lost=0, delivered=int(t == 63)),
                             requested_hours=action, executed_hours=action,
                             info=dict(support_public_receipt=receipt), plan=plan))
        write_raw(root / "raw" / (ident + ".jsonl.gz"), rows)
        tape = dict(world=world, source="handcrafted keyed data, not a native tape")
        write_json(root / "tapes" / (analysis._ident(world, "exogenous") + ".json"), tape)
        b = world["block"]
        model = (ancestors[b] if role == "existing_frozen" else finals.get((b, role))) if world["phase"] == "evaluation" else None
        updates = 1536 if role == "existing_frozen" else 768 if role in TRAIN_ROLES else None
        summary = dict(world=world, role=role, raw_path="raw/" + ident + ".jsonl.gz",
                       cost=sum(r["cost"] for r in rows), lost=0, delivered=1, enrolled=1,
                       tape_sha256=analysis.digest(tape), model_seal_sha256=model,
                       value_updates_before_trajectory=updates, optimizer_updates_during_trajectory=0,
                       settled=True, settlement=dict(settled=True, enrolled=1, lost=0, delivered=1,
                           live_ids=[], pending_obligations=0, resource_conservation=True))
        write_json(root / "summaries" / (ident + ".json"), summary)
        if world["phase"] == "evaluation":
            events.append(dict(event="trajectory_started", active=dict(world, role=role),
                               counts=dict(value_optimizer_steps=11520, actor_optimizer_steps=0)))
    (root / "progress.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return study, expected, ancestors, finals, manifest


class PlannerTailAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name).resolve()
        cls.study, cls.expected, cls.ancestors, cls.finals, cls.manifest = make_fixture(cls.root)
        cls.report = analysis.run_analysis(cls.root, cls.study)

    def one(self, role="planner_tail_td"):
        return next((w, r, ident) for w, r, ident in self.expected if w["phase"] == "evaluation" and r == role)

    def test_complete_six_role_matrix_and_all_five_block_contrasts(self):
        report = self.report
        self.assertEqual(report["trajectory_count"], 480)
        self.assertEqual(report["raw_rows"], 30720)
        self.assertEqual(report["reference_trajectories"], 120)
        self.assertEqual(report["evaluation_trajectories"], 360)
        self.assertEqual(report["paired_evaluation_worlds"], 60)
        self.assertEqual(len(report["contrasts"]), 15)
        self.assertEqual(set(report["primary_criteria"]), {"plain_h8", "existing_frozen", "observed_td"})
        self.assertTrue(report["strong_baseline_development_signal"])
        for conditions in report["contrasts"].values():
            self.assertEqual(set(conditions), {"0", "1", "2"})
            for condition in conditions.values():
                self.assertEqual(condition["n_worlds"], 20)
                self.assertEqual(condition["independent_blocks"], 5)
                self.assertEqual([b["block"] for b in condition["blocks"]], list(range(5)))
                self.assertEqual(len(condition["pairs"]), 20)
        primary = report["contrasts"]["planner_tail_td_vs_plain_h8"]["1"]
        self.assertEqual(primary["means"]["savings"], 6400.)
        self.assertEqual(primary["descriptive95"]["savings"], [6400., 6400.])
        self.assertEqual(sum(primary["means"]["component_savings"].values()), 6400.)
        self.assertEqual(primary["means"]["extra_lost"], 0.)
        self.assertEqual(primary["means"]["changed_action_boundaries"], 48)
        self.assertEqual(primary["means"]["action_l1_hours"], 96.)
        self.assertEqual(primary["means"]["extra_requested_control_hours"], 0.)
        self.assertEqual(primary["harm"]["cost_harm_worlds"], 0)
        self.assertLess(report["contrasts"]["planner_tail_td_vs_plain_h16"]["1"]["means"]["savings"], 0.)
        self.assertEqual(report["contrasts"]["planner_tail_td_vs_plain_h16"]["1"]["harm"]["cost_harm_worlds"], 20)
        self.assertFalse(report["formal_confirmation"])
        self.assertFalse(report["clinical_safety_established"])
        json.dumps(report, allow_nan=False)

    def test_latency_quantiles_and_distinct_requested_committed_applied_receipts(self):
        latency = self.report["decision_latency"]["evaluation"]["plain_h16"]
        self.assertEqual(latency["observed_decisions"], 60 * 48)
        self.assertTrue(latency["complete"])
        self.assertAlmostEqual(latency["median_seconds"], .049)
        self.assertAlmostEqual(latency["p95_seconds"], .092)
        self.assertAlmostEqual(latency["max_seconds"], .096)
        world, role, ident = self.one()
        row = analysis.read_saved_trajectory(self.root, world, role, ident)
        self.assertEqual(row["total_requested_control_hours"], 288.)
        self.assertEqual(row["total_committed_control_hours"], 288.)
        self.assertEqual(row["total_applied_hours_including_tail"], 288.)
        self.assertEqual(row["applied_actions"][:2], [[0.] * 4] * 2)
        self.assertEqual(row["applied_actions"][48:50], [ACTION[role]] * 2)
        self.assertTrue(row["labor_receipt_identity_checked"])

    def test_five_block_bootstrap_and_patient_harm_prevent_false_screen(self):
        world, role, ident = self.one()
        base = analysis.read_saved_trajectory(self.root, world, role, ident)
        pairs = []
        for b in range(5):
            for j in range(4):
                a, ref = copy.deepcopy(base), copy.deepcopy(base)
                a["world"].update(block=b, replicate=j)
                ref["cost"] += 100. if b == 4 else 0.
                if b == 4 and j == 3:
                    a["lost"] += 1
                    a["delivered"] -= 1
                pairs.append(analysis._pair(a, ref))
        one = analysis._summarize(pairs, np.random.default_rng(10), 2000)
        two = analysis._summarize(pairs, np.random.default_rng(10), 2000)
        self.assertEqual(one, two)
        self.assertEqual(one["means"]["savings"], 20.)
        self.assertGreater(one["descriptive95"]["savings"][1], 0.)
        self.assertEqual(one["harm"]["loss_harm_worlds"], 1)
        criteria = analysis._criteria(dict.fromkeys(("0", "1", "2"), one))
        self.assertFalse(criteria["all_persistent_blocks_cost_positive"])
        self.assertFalse(criteria["no_mean_extra_losses_in_any_condition"])
        with self.assertRaises(ValueError):
            analysis._summarize(pairs[:-1], np.random.default_rng(10), 2000)

    def test_complete_model_byte_evidence_and_progress_barrier(self):
        models = self.report["model_evidence"]
        self.assertEqual(models["final_count"], 15)
        self.assertEqual(models["ancestor_count"], 5)
        self.assertEqual(len(models["byte_verified"]), 20)
        self.assertEqual(models["manifest_only"], [])
        self.assertFalse(models["model_deserialization_performed"])
        self.assertTrue(self.report["progress_evidence"]["barrier_verified"])
        self.assertEqual(self.report["progress_evidence"]["evaluation_starts"], 360)
        self.assertTrue(self.report["tape_artifacts_verified"])
        self.assertEqual(self.report["compute"]["new_updates_from_seal_metadata"], 11520)
        self.assertEqual(self.report["compute"]["historical_ancestor_updates"], 7680)
        self.assertTrue(self.report["compute"]["limits_are_not_measurements"])

    def test_hash_only_supplied_manifest_is_not_called_byte_verified(self):
        study = copy.deepcopy(self.study)
        study["analysis_inputs"]["model_manifest"] = self.manifest
        _, _, evidence = analysis._model_evidence(self.root, study)
        self.assertEqual(evidence["byte_verified"], [])
        self.assertEqual(len(evidence["manifest_only"]), 20)
        for field in ("ancestors", "final_models"):
            with self.subTest(field=field):
                broken = copy.deepcopy(study)
                broken["analysis_inputs"]["model_manifest"][field].pop()
                with self.assertRaises(ValueError):
                    analysis._model_evidence(self.root, broken)
                broken = copy.deepcopy(study)
                broken["analysis_inputs"]["model_manifest"][field].append(
                    broken["analysis_inputs"]["model_manifest"][field][0])
                with self.assertRaises(ValueError):
                    analysis._model_evidence(self.root, broken)

    def test_model_bytes_update_counts_ancestry_and_all_sealed_map_are_checked(self):
        pt = self.root / "models/block0-planner_tail_td-final.pt"
        meta = pt.with_suffix(".json")
        barrier = self.root / "models/all-sealed.json"
        with preserved(pt):
            pt.write_bytes(b"tampered fixture bytes")
            with self.assertRaisesRegex(ValueError, "model byte"):
                analysis._model_evidence(self.root, self.study)
        for field, value in (("updates", 2304), ("ancestor_sha256", "0" * 64), ("method", "observed_td")):
            with self.subTest(field=field), preserved(meta):
                record = json.loads(meta.read_text())
                record[field] = value
                write_json(meta, record)
                with self.assertRaises(ValueError):
                    analysis._model_evidence(self.root, self.study)
        with preserved(meta):
            record = json.loads(meta.read_text())
            record["config"]["lr"] *= 10
            write_json(meta, record)
            with self.assertRaisesRegex(ValueError, "method/config"):
                analysis._model_evidence(self.root, self.study)
        extra = self.root / "models/extra-final.pt"
        with preserved(extra):
            extra.write_bytes(b"unselected extra model fixture")
            with self.assertRaisesRegex(ValueError, "missing or extra"):
                analysis._model_evidence(self.root, self.study)
        with preserved(barrier):
            record = json.loads(barrier.read_text())
            record["final_models"].pop("block4-planner_tail_mc")
            write_json(barrier, record)
            with self.assertRaisesRegex(ValueError, "all-sealed"):
                analysis._model_evidence(self.root, self.study)

    def test_progress_missing_is_explicit_but_malformed_or_early_barrier_fails(self):
        path = self.root / "progress.jsonl"
        original = [json.loads(line) for line in path.read_text().splitlines()]
        for defect in ("early_test", "late_seal", "update", "missing_test", "wrong_seed", "wrong_seal", "two_barriers"):
            with self.subTest(defect=defect), preserved(path):
                events = copy.deepcopy(original)
                if defect == "early_test":
                    events.insert(0, events.pop())
                elif defect == "late_seal":
                    events.append(events.pop(1))
                elif defect == "update":
                    events[-1]["counts"]["value_optimizer_steps"] += 1
                elif defect == "missing_test":
                    events.pop()
                elif defect == "wrong_seed":
                    events[-1]["active"]["seed"] += 1
                elif defect == "wrong_seal":
                    events[0]["sha256"] = "0" * 64
                else:
                    events.append(dict(event="all_models_sealed"))
                path.write_text("".join(json.dumps(e) + "\n" for e in events))
                with self.assertRaises(ValueError):
                    analysis._progress_evidence(self.root, self.expected, self.ancestors, self.finals)
        with preserved(path):
            path.unlink()
            evidence = analysis._progress_evidence(self.root, self.expected, self.ancestors, self.finals)
            self.assertFalse(evidence["provided"])
            self.assertFalse(evidence["barrier_verified"])
        with preserved(path):
            events = [e for e in original if e["event"] not in ("ancestor_bound", "final_sealed")]
            path.write_text("".join(json.dumps(e) + "\n" for e in events))
            self.assertTrue(analysis._progress_evidence(self.root, self.expected, self.ancestors, self.finals)["barrier_verified"])

    def test_raw_cost_patient_and_labor_tampering_is_rejected(self):
        world, role, ident = self.one()
        path = self.root / "raw" / (ident + ".jsonl.gz")
        original = read_raw(path)
        mutations = {
            "cost": lambda rows: rows[1].update(cost=-1.),
            "reward": lambda rows: rows[1].update(reward=0.),
            "components": lambda rows: rows[1]["components"].update(reagent_purchase_cost=0.),
            "patient_count": lambda rows: rows[1]["cumulative"].update(enrolled=3),
            "patient_identity": lambda rows: rows[1]["patient_records"][0].update(collection_site=2),
            "terminal_reopened": lambda rows: rows[1]["patient_records"][0].update(status="delivered"),
            "duplicate_patient": lambda rows: rows[1]["patient_records"].append(rows[1]["patient_records"][0]),
            "loss_event": lambda rows: rows[1].update(new_lost_patients=1),
            "receipt_request": lambda rows: rows[1]["info"]["support_public_receipt"].update(raw_requested_hours=[0.] * 4),
            "receipt_committed": lambda rows: rows[1]["info"]["support_public_receipt"].update(committed_hours=[0.] * 4),
            "receipt_epoch": lambda rows: rows[1]["info"]["support_public_receipt"].update(known_at=1),
            "applied_lag": lambda rows: rows[2]["info"]["support_public_receipt"].update(applied_hours=[0.] * 4),
            "illegal_execute": lambda rows: rows[1].update(executed_hours=[5., 0., 0., 0.]),
            "settlement_request": lambda rows: rows[-1].update(requested_hours=[1.] * 4),
            "latency": lambda rows: rows[1]["plan"].update(decision_wall_seconds=-1.),
            "short": lambda rows: rows.pop(),
        }
        for defect, mutate in mutations.items():
            with self.subTest(defect=defect), preserved(path):
                rows = copy.deepcopy(original)
                mutate(rows)
                write_raw(path, rows)
                with self.assertRaises(ValueError):
                    analysis.read_saved_trajectory(self.root, world, role, ident)

    def test_summary_reconciliation_and_new_versus_historical_update_counts(self):
        for role, expected_updates in (("existing_frozen", 1536), ("planner_tail_td", 768)):
            world, _, ident = self.one(role)
            path = self.root / "summaries" / (ident + ".json")
            original = json.loads(path.read_text())
            self.assertEqual(original["value_updates_before_trajectory"], expected_updates)
            for field, value in (("cost", original["cost"] + 1.), ("delivered", 3),
                                 ("optimizer_updates_during_trajectory", 1), ("value_updates_before_trajectory", 2304),
                                 ("raw_path", "incorrect")):
                with self.subTest(role=role, field=field), preserved(path):
                    write_json(path, dict(original, **{field: value}))
                    with self.assertRaises(ValueError):
                        analysis.read_saved_trajectory(self.root, world, role, ident)

    def test_missing_latency_and_tape_evidence_never_become_zero_measurements(self):
        world, role, ident = self.one()
        path = self.root / "raw" / (ident + ".jsonl.gz")
        tape = self.root / "tapes" / (analysis._ident(world, "exogenous") + ".json")
        with preserved(path, tape):
            rows = read_raw(path)
            for row in rows:
                row["plan"] = None
            write_raw(path, rows)
            tape.unlink()
            result = analysis.read_saved_trajectory(self.root, world, role, ident)
            self.assertIsNone(result["decision_latency"]["median_seconds"])
            self.assertEqual(result["decision_latency"]["observed_decisions"], 0)
            self.assertFalse(result["decision_latency"]["complete"])
            self.assertFalse(result["tape_artifact_verified"])
            rows[0]["plan"] = dict(decision_wall_seconds=.1)
            write_raw(path, rows)
            result = analysis.read_saved_trajectory(self.root, world, role, ident)
            self.assertEqual(result["decision_latency"]["observed_decisions"], 1)
            self.assertFalse(result["decision_latency"]["complete"])

    def test_missing_extra_and_old_four_role_or_three_block_designs_rejected(self):
        path = self.root / "summaries" / (self.expected[-1][2] + ".json")
        with preserved(path):
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing or extra"):
                analysis.run_analysis(self.root, self.study)
        extra = self.root / "summaries" / "extra.json"
        with preserved(extra):
            write_json(extra, {})
            with self.assertRaisesRegex(ValueError, "missing or extra"):
                analysis.run_analysis(self.root, self.study)
        for field, value in (("eval_roles", list(EVAL_ROLES[:4])), ("blocks", 3)):
            study = copy.deepcopy(self.study)
            study["design"][field] = value
            with self.assertRaises(ValueError):
                analysis.run_analysis(self.root, study)

    def test_pairing_and_frozen_model_binding_are_not_just_filename_checks(self):
        world, role, ident = self.one()
        raw = self.root / "raw" / (ident + ".jsonl.gz")
        summary = self.root / "summaries" / (ident + ".json")
        with preserved(raw):
            rows = read_raw(raw)
            for row in rows:
                row["patient_records"][0]["patient_id"] = "different-paired-patient"
            write_raw(raw, rows)
            with self.assertRaisesRegex(ValueError, "unpaired evaluation"):
                analysis.run_analysis(self.root, self.study)
        for field in ("model_seal_sha256", "tape_sha256"):
            with self.subTest(field=field), preserved(summary):
                record = json.loads(summary.read_text())
                record[field] = "0" * 64
                write_json(summary, record)
                with self.assertRaises(ValueError):
                    analysis.run_analysis(self.root, self.study)

    def test_nonpositive_reference_cost_has_no_fabricated_percent_savings(self):
        world, role, ident = self.one()
        row = analysis.read_saved_trajectory(self.root, world, role, ident)
        reference = copy.deepcopy(row)
        reference["cost"] = 0.
        with self.assertRaisesRegex(ValueError, "undefined relative"):
            analysis._pair(row, reference)

    def test_tape_world_cannot_be_swapped_even_with_matching_summary_hash(self):
        world, role, ident = self.one()
        tape = self.root / "tapes" / (analysis._ident(world, "exogenous") + ".json")
        summary = self.root / "summaries" / (ident + ".json")
        with preserved(tape, summary):
            data = json.loads(tape.read_text())
            data["world"]["seed"] += 1
            write_json(tape, data)
            metadata = json.loads(summary.read_text())
            metadata["tape_sha256"] = analysis.digest(data)
            write_json(summary, metadata)
            with self.assertRaisesRegex(ValueError, "tape world/hash"):
                analysis.read_saved_trajectory(self.root, world, role, ident)


if __name__ == "__main__":
    unittest.main()
