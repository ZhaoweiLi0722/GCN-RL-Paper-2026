"""Independent target arithmetic from invented JSON only, without model imports."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.rl.candidate_pilot_driver import file_record
from src.rl.time_baseline_target_verification import verify_time_baseline_targets


class TargetVerificationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = dict(blocks=[60], continuation=dict(episodes_per_rollout=2,
            rollouts_per_arm_per_block=1, episodes_per_arm_per_block=2),
            objective=dict(horizon=2, gamma=1., gae_lambda=1., reward_scale=1.))
        self.streams = dict(namespace="invented-targets", environment={"60": {"training": ["11", "12"]}})
        self.index, self.phases = [], {}
        for role, method in (("current_ppo", "collected_value"), ("time_baseline_ppo", "leave_one_episode_out_time")):
            ids = []
            for e, rewards in enumerate(([-1., -2.], [-3., -4.])):
                trajectory = f"training/block60/graph/{role}/episode{e:02d}"
                ids.append(trajectory)
                header = dict(block=60, role=role, split="training", world_index=e,
                    source_id="invented-targets", trajectory_id=trajectory, seed=11 + e, policy_sha256="a" * 64)
                rows = [dict(event={"audit": {
                    "record": dict(source_id="invented-targets", trajectory_id=trajectory, step_index=t,
                                   truncated=False, terminated=t == 1, raw_reward=r),
                    "decision": {"evaluation": dict(value=0., behavior_sha256="a" * 64, inference_dtype="float32")}}})
                    for t, r in enumerate(rewards)]
                self.index.append(dict(header=self.save(f"{role}-{e}-header.json", header),
                                       events=self.save(f"{role}-{e}-events.jsonl", rows, jsonl=True)))
            baseline = [[0., 0.], [0., 0.]] if role == "current_ppo" else [[-7., -4.], [-3., -2.]]
            advantages = [[-3., -2.], [-7., -4.]] if role == "current_ppo" else [[4., 2.], [-4., -2.]]
            receipt = dict(trajectory_ids=ids, environment_seeds=[11, 12], behavior_sha256s=["a" * 64] * 2,
                terminal_flags=[True, True], method=method, returns=[[-3., -2.], [-7., -4.]],
                collected_values=[[0., 0.], [0., 0.]], baselines=baseline, advantages=advantages,
                raw_rewards=[[-1., -2.], [-3., -4.]], segment_sha256s=["b" * 64, "c" * 64])
            phase = dict(job=f"{role}/block60", updates=[dict(update=1, rollout_steps=4, target_receipt=receipt)])
            self.phases[role] = phase
            self.save(f"payload/phases/{role}/block60.json", phase)

    def save(self, name, value, jsonl=False):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(v) + "\n" for v in value) if jsonl else json.dumps(value)
        path.write_text(text)
        return file_record(self.root, name)

    def check(self):
        return verify_time_baseline_targets(self.root, self.index, self.config, self.streams)

    def test_hand_computed_targets_baselines_and_variance(self):
        report = self.check()
        self.assertEqual(len(report["rollouts"]), 2)
        left, right = report["rollouts"]
        self.assertEqual(left["collected_value_mse"], 19.5)
        self.assertEqual(right["collected_value_mse"], 19.5)
        self.assertEqual(right["applied_baseline_mse"], 10.)
        self.assertEqual(right["time_mean_advantage_variance"], 0.)
        self.assertFalse(right["gradient_variance_measured"])
        self.assertFalse(report["segment_digest_payload_reconstruction_verified"])

    def test_tampered_target_self_inclusion_or_role_rejected(self):
        role = "time_baseline_ppo"
        original = self.phases[role]
        for field, value in (("baselines", [[-5., -3.], [-5., -3.]]),
                             ("returns", [[-3., -2.], [-8., -4.]]),
                             ("environment_seeds", [11, 11]), ("method", "collected_value"),
                             ("raw_rewards", [[-1., -2.], [-30., -4.]])):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed["updates"][0]["target_receipt"][field] = value
                self.save(f"payload/phases/{role}/block60.json", changed)
                with self.assertRaises(ValueError):
                    self.check()
        self.save(f"payload/phases/{role}/block60.json", original)
        self.check()

    def test_missing_duplicate_and_raw_byte_change_rejected(self):
        original = self.index
        for changed in (original[:-1], original + [original[0]]):
            self.index = changed
            with self.assertRaises(ValueError):
                self.check()
        self.index = original
        path = self.root / original[0]["events"]["path"]
        path.write_text(path.read_text() + "\n")
        with self.assertRaises(ValueError):
            self.check()

    def test_wrong_update_order_and_mixed_behavior_rejected(self):
        self.phases["current_ppo"]["updates"][0]["update"] = 2
        self.save("payload/phases/current_ppo/block60.json", self.phases["current_ppo"])
        with self.assertRaises(ValueError):
            self.check()
        self.phases["current_ppo"]["updates"][0]["update"] = 1
        self.save("payload/phases/current_ppo/block60.json", self.phases["current_ppo"])
        entry = self.index[0]
        rows = [json.loads(x) for x in (self.root / entry["events"]["path"]).read_text().splitlines()]
        rows[0]["event"]["audit"]["decision"]["evaluation"]["behavior_sha256"] = "d" * 64
        entry["events"] = self.save(entry["events"]["path"], rows, jsonl=True)
        with self.assertRaises(ValueError):
            self.check()
