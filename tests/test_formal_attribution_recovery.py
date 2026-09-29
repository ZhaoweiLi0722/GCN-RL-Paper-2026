"""Fail-closed identity checks for the immutable formal cost projection."""

import tempfile
import unittest
from pathlib import Path

from evaluation.recover_formal_attribution_rows import ALGORITHMS, SCENARIOS, require_hash, sha256, validate_rows


class FormalRecoveryTests(unittest.TestCase):
    def fixture(self, anchor=False, algorithm=ALGORITHMS[0]):
        rows = [dict(algorithm="mdl2" if anchor else algorithm, seed=str(crn), training_seed="10",
                     evaluation_seed=str(crn), replication=str(r), scenario=s,
                     graph_ablation="full_graph" if algorithm == ALGORITHMS[0] else "flat_state_no_graph",
                     steps="52", total_cost=str(100 + r))
                for s, crn in SCENARIOS.items() for r in range(100)]
        means = {s: {"candidate_cost_mean": 149.5, "anchor_cost_mean": 149.5} for s in SCENARIOS}
        return rows, means

    def check(self, rows, means, **kwargs):
        return validate_rows(rows, algorithm=kwargs.get("algorithm", ALGORITHMS[0]),
                             training_seed=10, anchor=kwargs.get("anchor", False), per_scenario=means)

    def test_complete_projection_for_both_algorithms_and_arms(self):
        for algo in ALGORITHMS:
            for anchor in (False, True):
                rows, means = self.fixture(anchor, algo)
                self.assertEqual(self.check(rows, means, algorithm=algo, anchor=anchor)["rows"], 400)

    def test_duplicate_and_missing_replications_rejected(self):
        rows, means = self.fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check(rows + [rows[0]], means)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.check(rows[1:], means)
        rows[-1]["replication"] = "100"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.check(rows, means)

    def test_wrong_identity_rejected(self):
        for field, value in (("evaluation_seed", "0"), ("seed", "0"), ("training_seed", "11"),
                             ("algorithm", "mdl2"), ("graph_ablation", "none"), ("steps", "51")):
            with self.subTest(field=field):
                rows, means = self.fixture()
                rows[0][field] = value
                with self.assertRaisesRegex(ValueError, "identity"):
                    self.check(rows, means)

    def test_unknown_scenario_and_missing_field_rejected(self):
        rows, means = self.fixture()
        rows[0]["scenario"] = "unknown"
        with self.assertRaisesRegex(ValueError, "scenario"):
            self.check(rows, means)
        del rows[0]["steps"]
        with self.assertRaisesRegex(ValueError, "field"):
            self.check(rows, means)

    def test_nonfinite_and_summary_drift_rejected(self):
        for value in ("nan", "inf", "-inf"):
            rows, means = self.fixture()
            rows[0]["total_cost"] = value
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.check(rows, means)
        rows, means = self.fixture()
        rows[0]["total_cost"] = "101"
        with self.assertRaisesRegex(ValueError, "summary mean"):
            self.check(rows, means)

    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture"
            path.write_bytes(b"immutable fixture")
            self.assertEqual(require_hash(path, sha256(path)), sha256(path))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                require_hash(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
