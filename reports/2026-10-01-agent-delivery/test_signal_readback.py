"""Hand-calculable invented receipts only; no optimization or scientific rerun."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("signal_readback", HERE / "signal_readback.py")
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def fixture(actions=None):
    config = {"cues": [-1, 1], "training_times": [0], "request_values": [0, 1],
              "negative_cue_winner_request": 0, "positive_cue_winner_request": 1,
              "invented_return": {"intercept": 0, "remaining_time_slope": 0,
                                  "wrong_action_penalty": -2},
              "representations": ["flat"], "initialization_seeds": [1],
              "sampling": {"context_repetitions": 2, "rollouts_per_fit": 1,
                           "normalize_advantages_once_per_rollout": True}}
    # Reversed manifest order proves winner is not assumed to be config index.
    manifest = {"actor_manifest": {"bank": {"representative_requests": [[1], [0]]},
                                   "schema": {"action_names": ["request"]}}}
    actions = [1, 1, 0, 0] if actions is None else actions
    rewards = [0 if action == winner else -2 for action, winner in zip(actions, [1, 0, 1, 0])]
    values = [-3, 1, -3, 1]
    residuals = [reward - value for reward, value in zip(rewards, values)]
    mu = math.fsum(residuals) / 4
    scale = math.sqrt(math.fsum((x - mu) ** 2 for x in residuals) / 4) + 1e-8
    receipt = {"context_indices": [0, 1, 0, 1], "actions": actions,
               "old_log_probs": [-math.log(2)] * 4, "old_values": values,
               "returns": rewards, "advantages": [(x - mu) / scale for x in residuals],
               "behavior_logits": [[0, 0] for _ in range(4)]}
    manifest["counts"] = {"observations": 4}
    manifest["coverage"] = {f"{c}:{a}": sum(c == ci and a == ai for ci, ai in
                                            zip(receipt["context_indices"], actions))
                            for c in range(2) for a in range(2)}
    return config, manifest, receipt


def rows_for(config, manifest, receipt):
    contexts, count, gap = reader.context_spec(config, manifest)
    return reader.read_rows(receipt, config, contexts, count, gap)[0], contexts


class SignalArithmeticTests(unittest.TestCase):
    def test_hand_variance_equal_groups(self):
        result = reader.variance_partition([3, -3, 1, -1], [0, 1, 0, 1])
        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["total"], 5)
        self.assertEqual(result["between_context"], 4)
        self.assertEqual(result["within_context"], 1)
        self.assertEqual(result["between_share"], 0.8)

    def test_unequal_groups_are_observation_weighted(self):
        result = reader.variance_partition([0, 2, 5], [0, 0, 1])
        self.assertAlmostEqual(result["between_context"], 32 / 9)
        self.assertAlmostEqual(result["within_context"], 2 / 3)
        self.assertAlmostEqual(result["total"], 38 / 9)

    def test_constant_variance_share_is_undefined(self):
        result = reader.variance_partition([2, 2], [0, 1])
        self.assertIsNone(result["between_share"])
        self.assertEqual(result["total"], 0)

    def test_hand_score_oracle_and_manifest_order(self):
        rows, contexts = rows_for(*fixture())
        self.assertEqual([c["winner"] for c in contexts], [1, 0])
        self.assertEqual([row["raw_score"] for row in rows], [1.5, 1.5, -0.5, -0.5])
        self.assertAlmostEqual(reader.mean([row["raw_score"] for row in rows]), 0.5)
        self.assertAlmostEqual(reader.mean([row["A_score"] for row in rows]),
                               0.5 / (math.sqrt(5) + 1e-8))
        for row in rows:
            self.assertEqual(row["oracle_expected"], 0.5)
            self.assertEqual(row["oracle_centered_score"], 0.5)
            self.assertAlmostEqual(row["raw_score"], row["oracle_centered_score"]
                                   + row["baseline_offset_score"])

    def test_sparse_sign_reversal_not_oracle_reversal(self):
        rows, contexts = rows_for(*fixture([0, 0, 0, 0]))
        table = reader.context_table(rows, contexts)
        self.assertEqual(table[0][2], 0)
        self.assertEqual(table[0][8], -0.5)
        self.assertEqual(table[0][12], 0.5)
        self.assertEqual(table[1][2], 2)
        self.assertEqual(table[1][8], -0.5)

    def test_no_observation_is_null_not_zero_effect(self):
        _, contexts = rows_for(*fixture())
        table = reader.context_table([], contexts)
        self.assertEqual(table[0][1:4], [0, 0, 0])
        self.assertIsNone(table[0][8])
        self.assertIsNone(table[0][9])

    def test_extreme_logits_softmax_stable(self):
        p, norm = reader.distribution([1000, 1000])
        self.assertEqual(p, [0.5, 0.5])
        self.assertAlmostEqual(norm, 1000 + math.log(2))

    def test_nonuniform_exact_expected_direction_cancels_baseline(self):
        config, manifest, receipt = fixture()
        receipt["behavior_logits"] = [[math.log(3), 0] for _ in range(4)]
        receipt["old_log_probs"] = [math.log(0.75 if action == 0 else 0.25)
                                    for action in receipt["actions"]]
        rows, _ = rows_for(config, manifest, receipt)
        for context, winner_probability in ((0, 0.25), (1, 0.75)):
            group = [row for row in rows if row["context"] == context]
            exact = math.fsum(row["raw_score"] * (winner_probability if row["win"]
                                                 else 1 - winner_probability) for row in group)
            self.assertAlmostEqual(exact, 0.375)
            self.assertAlmostEqual(group[0]["oracle_expected"], exact)

    def test_shape_finite_and_indices_fail_closed(self):
        cases = [("actions", [0]), ("actions", [True, 1, 0, 0]),
                 ("actions", [2, 1, 0, 0]), ("context_indices", [-1, 1, 0, 1]),
                 ("context_indices", [0, 2, 0, 1]),
                 ("returns", [math.nan, -2, -2, 0]),
                 ("old_values", [math.inf, 1, -3, 1]),
                 ("behavior_logits", [[0], [0, 0], [0, 0], [0, 0]]),
                 ("behavior_logits", [[math.inf, 0]] * 4)]
        for field, replacement in cases:
            with self.subTest(field=field, replacement=replacement):
                config, manifest, receipt = fixture()
                receipt[field] = replacement
                with self.assertRaises(ValueError):
                    rows_for(config, manifest, receipt)

    def test_incorrect_advantage_or_reward_rejected(self):
        for field in ("advantages", "returns", "old_log_probs"):
            config, manifest, receipt = fixture()
            receipt[field][0] += 0.1
            with self.assertRaises(ValueError):
                rows_for(config, manifest, receipt)

    def test_invalid_support_or_nonunique_winner_rejected(self):
        config, manifest, _ = fixture()
        manifest["actor_manifest"]["bank"]["representative_requests"] = [[0], [0]]
        with self.assertRaises(ValueError):
            reader.context_spec(config, manifest)
        config, manifest, _ = fixture()
        config["invented_return"]["wrong_action_penalty"] = 0
        with self.assertRaises(ValueError):
            reader.context_spec(config, manifest)

    def test_packet_only_consumes_allowed_files(self):
        config, manifest, receipt = fixture()
        payloads = {"config.json": config, "flat-1/training.json": manifest,
                    "flat-1/rollout-00/samples.json": receipt}
        reads = []

        def fake_load(path, root, hashes):
            name = path.relative_to(root).as_posix()
            reads.append(name)
            self.assertIn(name, payloads)
            hashes[name] = "synthetic"
            return copy.deepcopy(payloads[name])

        with patch.object(reader, "load_json", side_effect=fake_load):
            result = reader.analyze_packet(Path("synthetic"), {})
        self.assertEqual(set(reads), set(payloads))
        self.assertEqual(result["summary"]["observations"], 4)
        self.assertEqual(result["summary"]["consumed_files"], 3)
        stats = result["fits"]["flat-1"]["summary"]
        self.assertEqual(stats["residual_variance"]["between_share"], 0.8)
        self.assertEqual(stats["minimum_context_winner_observations"], 1)
        self.assertEqual(stats["nonpositive_A_context_batches"], 0)

    def test_write_once_and_consumed_hash(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            path = root / "synthetic.json"
            reader.write_once(path, {"test": 1})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                reader.write_once(path, {"test": 2})
            self.assertEqual(path.read_bytes(), before)
            hashes = {}
            self.assertEqual(reader.load_json(path, root, hashes), {"test": 1})
            self.assertEqual(hashes, {"synthetic.json": reader.hashlib.sha256(before).hexdigest()})
            with self.assertRaises(ValueError):
                reader.write_once(root / "invalid.json", {"x": math.nan})
            self.assertFalse((root / "invalid.json").exists())

    def test_fit_pooling_distinguishes_between_rollout_value_drift(self):
        config, manifest, receipt = fixture()
        config["sampling"]["rollouts_per_fit"] = 2
        manifest["counts"]["observations"] = 8
        manifest["coverage"] = {key: 2 * value for key, value in manifest["coverage"].items()}
        later = copy.deepcopy(receipt)
        later["old_values"] = [value - 10 for value in later["old_values"]]
        payloads = {"config.json": config, "flat-1/training.json": manifest,
                    "flat-1/rollout-00/samples.json": receipt,
                    "flat-1/rollout-01/samples.json": later}

        def fake_load(path, root, hashes):
            name = path.relative_to(root).as_posix()
            hashes[name] = "synthetic"
            return copy.deepcopy(payloads[name])

        with patch.object(reader, "load_json", side_effect=fake_load):
            stats = reader.analyze_packet(Path("synthetic"), {})["fits"]["flat-1"]["summary"]
        self.assertEqual(stats["residual_variance"]["total"], 30)
        self.assertEqual(stats["residual_variance"]["between_context"], 4)
        self.assertEqual(stats["residual_variance"]["within_context"], 26)
        self.assertEqual(stats["pooled_within_rollout_variance"]["total"], 5)
        self.assertEqual(stats["pooled_within_rollout_variance"]["between_share"], 0.8)

    def test_main_failure_preserves_error_receipt(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            with patch.object(reader, "HERE", root), patch.object(
                    reader, "analyze_packet", side_effect=ValueError("invented failure")):
                with self.assertRaisesRegex(ValueError, "invented failure"):
                    reader.main()
            errors = list(root.glob("signal-readback-error-*.json"))
            self.assertEqual(len(errors), 1)
            self.assertEqual(json.loads(errors[0].read_text())["status"], "error")
            self.assertFalse((root / "signal-readback.json").exists())


if __name__ == "__main__":
    unittest.main()
