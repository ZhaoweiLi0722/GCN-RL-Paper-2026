from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluation.verify_g1_decision_cost import D, check, histogram, stats

from evaluation.audit_g1_decision_cost import (
    audit, bins, compare_metrics, decisions, describe, ranking_metrics,
    stability, summarize, validate_tables,
)


def fixture():
    old = {"training_seeds": [60, 61, 62],
           "scenario_by_training_seed": {str(s): "recorded_" + str(s) for s in (60, 61, 62)},
           "dataset": {"decision_steps": [0, 1], "horizons": [4, "remaining"],
                       "primary_horizon": "remaining", "max_steps_per_episode": 2,
                       "discovery_replications": 3, "validation_replications": 5,
                       "discovery_rollout_seed": 103000000, "validation_rollout_seed": 107000000,
                       "explicit_options": [{"group": "specimen_transfer", "sign": s, "epsilon": e}
                                            for e in (.05, .1) for s in (1, -1)]}}
    labels = ["mdl2", "specimen_transfer:+1x0.05", "specimen_transfer:-1x0.05",
              "specimen_transfer:+1x0.1", "specimen_transfer:-1x0.1"]
    ids = [(s, t) for s in old["training_seeds"] for t in (0, 1)]
    arrays = {"states": np.zeros((6, 2)), "actions": np.zeros((6, 5, 1)),
              "training_seeds": np.array([s for s, _ in ids]), "steps": np.array([t for _, t in ids]),
              "horizons": np.array(["4", "remaining"]), "labels": np.array(labels),
              "discovery_advantages": np.zeros((6, 2, 5)), "validation_advantages": np.zeros((6, 2, 5))}
    costs, preds = [], []
    for ix, (seed, step) in enumerate(ids):
        for hi, horizon in enumerate(("4", "remaining")):
            for i, label in enumerate(labels):
                row = {"training_seed": str(seed), "step": str(step), "configured_horizon": horizon,
                       "horizon": str(2 - step), "candidate_index": str(i), "candidate_label": label,
                       "scenario": old["scenario_by_training_seed"][str(seed)], "executed_action_id": str(i)}
                for stream, base, rep in (("discovery", 103000000, 3), ("validation", 107000000, 5)):
                    row.update({stream + "_total_cost": str(100 + i), stream + "_cost_advantage": str(-i),
                                stream + "_seed_start": str(base + (seed - 60) * 1000000 + step * 100 + hi * 10),
                                stream + "_replications": str(rep)})
                    arrays[stream + "_advantages"][ix, hi, i] = -i
                costs.append(row)
        for stream in ("discovery", "validation"):
            for i, label in enumerate(labels):
                preds.append({"held_out_seed": str(seed), "step": str(step), "stream": stream,
                              "configured_horizon": "remaining", "candidate_index": str(i),
                              "candidate_label": label, "scenario": "recorded_" + str(seed),
                              "training_seeds": "|".join(str(s) for s in (60, 61, 62) if s != seed),
                              "true_cost_advantage": str(-i), "baseline_q_advantage": str(i),
                              "fitted_q_advantage": str(-i)})
    return costs, preds, arrays, old


class DecisionCostTest(unittest.TestCase):
    def test_valid_join_and_measured_choices(self):
        records = validate_tables(*fixture())
        rows = decisions(records, "remaining", 1e-9)
        self.assertEqual(len(rows), 6)
        result = summarize(rows, [250000, 1000000], 1e-9)
        self.assertEqual(result["selectors"]["frozen_critic"]["cost_minus_mdl2"]["mean"], 4)
        self.assertEqual(result["fitted_minus_frozen_cost"]["mean"], -4)
        self.assertEqual(result["selectors"]["fitted_offline_critic"]["validation_top1_count"], 6)

    def test_duplicate_missing_and_extra_rows(self):
        for change in (lambda c: c.append(c[0]), lambda c: c.pop(),
                       lambda c: c[0].update(training_seed="99")):
            c, p, a, o = fixture()
            change(c)
            with self.assertRaises(ValueError):
                validate_tables(c, p, a, o)

    def test_recorded_contract_mutations(self):
        for key, value in (("scenario", "other"), ("horizon", "99"), ("candidate_label", "bad"),
                           ("discovery_seed_start", "1"), ("validation_replications", "3"),
                           ("validation_total_cost", "nan"), ("discovery_cost_advantage", "7")):
            with self.subTest(key=key):
                c, p, a, o = fixture()
                c[0][key] = value
                with self.assertRaises(ValueError):
                    validate_tables(c, p, a, o)

    def test_npz_corruption(self):
        for change in (lambda a: a["steps"].__setitem__(1, 0),
                       lambda a: a["states"].__setitem__((0, 0), np.inf),
                       lambda a: a["validation_advantages"].__setitem__((0, 0, 1), 3)):
            c, p, a, o = fixture()
            change(a)
            with self.assertRaises(ValueError):
                validate_tables(c, p, a, o)

    def test_fold_and_prediction_integrity(self):
        for key, value in (("training_seeds", "60|61"), ("training_seeds", "61|61|62"),
                           ("true_cost_advantage", "4"), ("configured_horizon", "4"),
                           ("baseline_q_advantage", "inf"), ("fitted_q_advantage", "1")):
            c, p, a, o = fixture()
            p[0][key] = value
            with self.assertRaises(ValueError):
                validate_tables(c, p, a, o)

    def test_actions_distinct_and_same_between_horizons(self):
        for index, value in ((1, "0"), (5, "new")):
            c, p, a, o = fixture()
            c[index]["executed_action_id"] = value
            with self.assertRaises(ValueError):
                validate_tables(c, p, a, o)

    def test_ranking_signs_and_pairwise(self):
        records = validate_tables(*fixture())
        bad = ranking_metrics(records, "baseline", "validation", "remaining", 1, 1, 1e-9)
        good = ranking_metrics(records, "fitted", "validation", "remaining", 1, 1, 1e-9)
        self.assertEqual(bad["top1_accuracy"], 0)
        self.assertEqual(bad["pairwise_accuracy"], 0)
        self.assertEqual(good["top1_accuracy"], 1)
        self.assertEqual(good["pairwise_accuracy"], 1)
        self.assertEqual(good["pairwise_comparisons"], 60)

    def test_exact_label_ties_and_first_index(self):
        records = validate_tables(*fixture())
        row = records[0]
        row["costs"]["remaining"]["validation"] = np.array([100., 100., 101., 102., 103.])
        row["predictions"]["fitted"] = [0, 0, -1, -2, -3]
        result = decisions([row], "remaining", 1e-9)[0]
        self.assertEqual(result["selectors"]["fitted_offline_critic"]["action_index"], 0)
        self.assertTrue(result["label_best_sets_agree"])
        row["advantages"]["remaining"]["validation"][0] = -1e-10
        row["advantages"]["remaining"]["validation"][1] = 0
        self.assertEqual(stability([row], "remaining", 1)["best_action_agreement"], 0)

    def test_cost_not_top1_is_separate(self):
        records = validate_tables(*fixture())
        row = records[0]
        row["costs"]["remaining"]["validation"] = np.array([2000000., 1000000., 500000., 0., .1])
        result = summarize(decisions([row], "remaining", 1e-9), [250000, 1000000], 1e-9)
        frozen = result["selectors"]["frozen_critic"]
        self.assertEqual(frozen["validation_top1_count"], 0)
        self.assertEqual(frozen["top1_error_excess_bins"]["above_tolerance_to_250000"], 1)
        self.assertLess(frozen["cost_minus_mdl2"]["mean"], 0)

    def test_bins_empty_and_thresholds(self):
        b = bins([0, 1e-9, .1, 250000, 250001, 1000000, 1000001], [250000, 1000000], 1e-9)
        self.assertEqual(list(b.values()), [2, 2, 2, 1])
        self.assertIsNone(describe([], 1e-9)["mean"])
        self.assertEqual(sum(bins([], [250000, 1000000], 1e-9).values()), 0)

    def test_historical_metric_disagreement_rejected(self):
        with self.assertRaises(ValueError):
            compare_metrics({"x": 1}, {"x": 2})
        with self.assertRaises(ValueError):
            compare_metrics({"x": 1}, {"y": 1})
        compare_metrics({"x": None}, {"x": None})

    def test_hash_mismatch_stops_before_reading_data(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "x"
            path.write_text("synthetic")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                audit({"inputs": {"cost_rows": {"path": "x", "sha256": "wrong"}}}, Path(name))

    def test_decimal_stats_and_bins(self):
        self.assertEqual(stats([D("3"), D("-1"), D("2")])["median"], D("2"))
        self.assertEqual(stats([D("3"), D("-1")])["mean"], D("1"))
        self.assertEqual(histogram([D("0"), D("250000"), D("1000000"), D("1000001")]),
                         bins([0, 250000, 1000000, 1000001], [250000, 1000000], 1e-9))

    def test_decimal_verifier_detects_numeric_and_identity_errors(self):
        check(D("1.0000001"), D("1"))
        with self.assertRaises(AssertionError):
            check(D("1.01"), D("1"))
        with self.assertRaises(AssertionError):
            check({"action_index": 0}, {"action_index": 1})


if __name__ == "__main__":
    unittest.main()
