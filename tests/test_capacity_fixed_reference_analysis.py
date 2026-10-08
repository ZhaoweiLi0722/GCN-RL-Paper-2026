"""Artificial saved ledgers only: no environment, model, or optimizer calls."""

from contextlib import contextmanager
import copy
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.rl import capacity_fixed_reference_analysis as analysis


def write_json(path, data):
    path.write_text(json.dumps(data))


def write_raw(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


@contextmanager
def preserved(path):
    data = path.read_bytes() if path.exists() else None
    try:
        yield
    finally:
        if data is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(data)


def fixture(new, old):
    for root in (new, old):
        (root / "summaries").mkdir(parents=True)
    (new / "raw").mkdir()
    saved = []
    for b, c, j in itertools.product(range(5), range(3), range(8)):
        world = dict(phase=analysis.PHASE, block=b, condition=c, replicate=j,
                     seed=67400000 + b * 1000 + j * 3 + c)
        tape = hashlib.sha256(json.dumps(world, sort_keys=True).encode()).hexdigest()
        for ri, role in enumerate(analysis.ROLES):
            name = f"{analysis.PHASE}-b{b}-c{c}-j{j}-{role}"
            unit = 1000. + b * 20 + c * 7 + j + ri * (j - 3)
            cost, lost = unit * 64, (b + j + ri) % 2
            components = dict.fromkeys(analysis.COST_KEYS, 0.)
            components["reagent_purchase_cost"] = cost
            summary = dict(world=world, role=role, cost=cost, lost=lost,
                raw_path=f"raw/{name}.jsonl.gz", tape_sha256=tape, elapsed=ri + .5,
                settled=True, settlement=dict(settled=True, lost=lost, enrolled=2, delivered=2-lost),
                model_seal_sha256=None, optimizer_updates_during_trajectory=0,
                compute=dict(native_steps=64, control_steps=48, tail_steps=16,
                             neural_forward_module_calls=48 if ri > 1 else 0, total_optimizer_steps=0))
            root = new if role == analysis.FIXED_ROLE else old
            write_json(root / "summaries" / (name + ".json"), summary)
            if role != analysis.FIXED_ROLE:
                saved.append(dict(summary, components=components, requested_hours=300. + ri,
                    committed_hours=300. + ri, applied_hours=300. + ri, raw_sha256="a" * 64))
                continue
            raw = []
            for t in range(64):
                action = [2.] * 4 if t < 48 else [0.] * 4
                applied = [2.] * 4 if 2 <= t < 50 else [0.] * 4
                patients = [dict(patient_id=f"{world['seed']}-{i}", enrollment_epoch=0,
                    collection_site=i, status=("lost" if i < lost else "delivered") if t == 63 else "waiting")
                    for i in range(2)]
                receipt = dict(epoch=t, known_at=t+1, raw_requested_hours=action,
                               committed_hours=action, applied_hours=applied, ordinary_hours=[4.] * 4)
                raw.append(dict(epoch=t, world=world, role=role, cost=unit, reward=-unit,
                    components={k: v/64 for k, v in components.items()}, new_lost_patients=lost if t == 63 else 0,
                    patient_records=patients, requested_hours=action, executed_hours=action,
                    info=dict(support_public_receipt=receipt, average_waiting_time=2., average_turnaround_time=3.)))
            write_raw(root / summary["raw_path"], raw)
    comparison = dict(raw_reconciliation=saved,
        screens=[dict(role=r, result=analysis._contrast(saved, r, "plain_h8")) for r in analysis.OLD_ROLES[1:]],
        pairs=[analysis._contrast(saved, "sac-graph-final", "value_td-graph-final")])
    write_json(old / "comparison.json", comparison)
    return comparison


class FixedReferenceAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name).resolve()
        cls.new, cls.old = cls.root / "new", cls.root / "old"
        cls.comparison = fixture(cls.new, cls.old)
        cls.report = analysis.analyze(cls.new, cls.old)

    def test_four_roles_six_pairs_and_explicit_supplement_limits(self):
        report = self.report
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        self.assertEqual([r["role"] for r in report["table"]], list(analysis.ROLES))
        self.assertEqual(len(report["raw_reconciliation"]), 480)
        self.assertEqual(len({frozenset((p["candidate"], p["reference"])) for p in report["pairs"]}), 6)
        self.assertEqual(report["new_raw_rows"], 7680)
        self.assertEqual(report["table"][0]["requested_hours"], 384.)
        self.assertEqual(report["table"][0]["applied_hours"], 384.)
        self.assertEqual(report["table"][0]["compute_total"]["native_steps"], 7680)
        self.assertEqual(report["table"][0]["delay_metrics"]["epoch_mean_reported_wait"]["mean"], 2.)
        self.assertIsNone(report["table"][1]["delay_metrics"]["epoch_mean_reported_wait"]["mean"])
        self.assertEqual(report["supplementary_primary_reference"], "mdl2-fixed2")
        self.assertEqual(report["prior_primary_reference"], "plain_h8")
        self.assertTrue(report["posthoc_supplement"])
        for flag in ("new_independent_validation", "automatic_winner", "automatic_screen",
                     "graph_attribution", "rl_attribution", "baseline_assumed_weaker", "old_raw_reverified"):
            self.assertFalse(report[flag])
        self.assertEqual(len(report["evidence"]["old_summary_sha256"]), 360)
        for pair, harm in zip(report["pairs"], report["pair_harm"]):
            self.assertEqual(np.shape(pair["world_savings"]), (5, 3, 8))
            self.assertEqual(harm["cost_harmed_worlds"], int((np.asarray(pair["world_savings"]) < 0).sum()))
            self.assertEqual(len(harm["block_conditions"]), 15)

    def test_old_primary_intervals_preserved_and_read_only(self):
        expected = [s["result"] for s in self.comparison["screens"]] + self.comparison["pairs"]
        self.assertEqual(self.report["pairs"][3:], expected)
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        real_reconcile = analysis.reconcile_raw
        with patch.object(analysis, "reconcile_raw", wraps=real_reconcile) as call:
            analysis.analyze(self.new, self.old)
        call.assert_called_once()
        self.assertEqual(call.call_args.args[0], self.new.resolve())
        self.assertEqual(len(call.call_args.args[1]), 120)
        self.assertFalse((self.old / "raw").exists())
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_bootstrap_delegates_2000_shared_block_independent_world_draws(self):
        class RecordingRng:
            def __init__(self):
                self.shapes = []
                self.rng = np.random.default_rng(7)

            def integers(self, low, high, size):
                self.shapes.append(size)
                return self.rng.integers(low, high, size)

        rng = RecordingRng()
        with patch.object(analysis.np.random, "default_rng", return_value=rng):
            analysis._contrast(self.report["raw_reconciliation"], "plain_h8", analysis.FIXED_ROLE)
        self.assertEqual(rng.shapes, [(2000, 5, 1, 1), (2000, 5, 3, 8)])

    def test_missing_duplicate_and_wrong_pairing_rejected(self):
        path = next((self.new / "summaries").glob("*.json"))
        original = json.loads(path.read_text())
        with preserved(path):
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing or extra"):
                analysis.analyze(self.new, self.old)
        duplicate = path.with_name("duplicate.json")
        with preserved(duplicate):
            write_json(duplicate, original)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                analysis.analyze(self.new, self.old)
        for field in ("seed", "tape_sha256"):
            with self.subTest(field=field), preserved(path):
                changed = copy.deepcopy(original)
                if field == "seed":
                    changed["world"][field] += 1
                else:
                    changed[field] = "0" * 64
                write_json(path, changed)
                with self.assertRaisesRegex(ValueError, "pairing"):
                    analysis.analyze(self.new, self.old)

    def test_reused_old_summary_and_ci_tampering_rejected(self):
        path = next((self.old / "summaries").glob("*.json"))
        original = json.loads(path.read_text())
        for field, value in (("elapsed", 999.), ("cost", 999.), ("extra_metadata", "changed")):
            with self.subTest(field=field), preserved(path):
                write_json(path, dict(original, **{field: value}))
                with self.assertRaisesRegex(ValueError, "summary keys"):
                    analysis.analyze(self.new, self.old)
        path = self.old / "comparison.json"
        with preserved(path):
            changed = copy.deepcopy(self.comparison)
            changed["screens"][0]["result"]["cost_ci95"][0] += 1.
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "bootstrap"):
                analysis.analyze(self.new, self.old)

    def test_new_fixed_support_reward_patient_and_epoch_contract(self):
        path = next((self.new / "raw").glob("*.jsonl.gz"))
        with gzip.open(path, "rt") as stream:
            original = [json.loads(line) for line in stream]
        mutations = {
            "reward": lambda r: r[0].update(reward=0.),
            "component": lambda r: r[0]["components"].update(reagent_purchase_cost=0.),
            "ordinary": lambda r: r[0]["info"]["support_public_receipt"].update(ordinary_hours=[0.] * 4),
            "lag": lambda r: r[1]["info"]["support_public_receipt"].update(applied_hours=[2.] * 4),
            "fixed": lambda r: r[4].update(requested_hours=[1.] * 4),
            "settlement": lambda r: r[48].update(executed_hours=[2.] * 4),
            "patient": lambda r: r[0]["patient_records"].append(r[0]["patient_records"][0]),
            "loss": lambda r: r[0].update(new_lost_patients=1),
            "identity": lambda r: r[1]["patient_records"][0].update(collection_site=3),
            "epoch": lambda r: r[1].update(epoch=0),
            "short": lambda r: r.pop(),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), preserved(path):
                changed = copy.deepcopy(original)
                mutate(changed)
                write_raw(path, changed)
                with self.assertRaises(ValueError):
                    analysis.analyze(self.new, self.old)

    def test_model_activity_in_fixed_reference_rejected(self):
        path = next((self.new / "summaries").glob("*.json"))
        original = json.loads(path.read_text())
        for counter in ("neural_forward_module_calls", "total_optimizer_steps"):
            with self.subTest(counter=counter), preserved(path):
                changed = copy.deepcopy(original)
                changed["compute"][counter] = 1
                write_json(path, changed)
                with self.assertRaisesRegex(ValueError, "model or optimizer"):
                    analysis.analyze(self.new, self.old)

    def test_independent_terminal_readout_reused_and_bound(self):
        directory = self.old.parent / "terminal-readout"
        directory.mkdir(exist_ok=True)
        path = directory / "world-readout.json"
        with preserved(path):
            saved = copy.deepcopy(self.comparison["raw_reconciliation"])
            for row in saved:
                row.update(epoch_mean_reported_wait=4., epoch_mean_reported_turnaround=5.)
            write_json(path, saved)
            report = analysis.analyze(self.new, self.old)
            self.assertEqual(report["evidence"]["old_metrics_source"], str(path))
            self.assertEqual(report["table"][1]["delay_metrics"]["epoch_mean_reported_wait"]["mean"], 4.)
            saved[0]["raw_sha256"] = "b" * 64
            write_json(path, saved)
            with self.assertRaisesRegex(ValueError, "prior raw reconciliation"):
                analysis.analyze(self.new, self.old)


if __name__ == "__main__":
    unittest.main()
