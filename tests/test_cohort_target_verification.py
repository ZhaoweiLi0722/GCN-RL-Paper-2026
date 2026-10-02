"""Invented checkpoint dictionaries only; no real checkpoint/model/fit access."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl.candidate_pilot_verification import json_hash
from src.rl.cohort_target_verification import _load_container, verify_cohort_targets
from tests.test_cohort_bundle_verification import fixture_config, invented_episode, save, save_episode


class CohortTargetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cfg, self.streams = fixture_config()
        self.cfg["blocks"] = [60]
        self.cfg["continuation"].update(episodes_per_rollout=2, rollouts_per_arm_per_block=1,
                                         episodes_per_arm_per_block=2)
        self.cfg["objective"].update(reward_scale=1.)
        self.index, self.containers, self.phases, self.targets = [], {}, {}, {}
        for role, objective in (("window_ppo", "window"), ("cohort_ppo", "cohort")):
            batch, origins, consumed = [], [], []
            for e in range(2):
                values = invented_episode(self.cfg, self.streams, role=role, world=e, split="training", cost=2 * (e + 1))
                h, rows, prefix = values[0]
                th, tail, final = values[1]
                for row in rows:
                    row["event"]["audit"]["record"]["semantics"]["reward_scale"] = 1.
                name = f"episodes/{role}/{e}"
                entry = save_episode(self.root, name, values)
                self.index.append(entry)
                trajectory = h["trajectory_id"]
                raw = [-float(e + 1)] * 2
                training = raw.copy()
                if objective == "cohort":
                    training[-1] -= 3.
                target = dict(format="cohort-objective-target-v1", objective=objective,
                    trajectory_id=trajectory, split="training", raw_prefix_rewards=raw,
                    training_raw_rewards=training, window_cost=2. * (e + 1), tail_cost=3.,
                    cohort_cost=2. * (e + 1) + 3., added_terminal_charge=3. if objective == "cohort" else 0.,
                    environment_reward_unchanged=True, reward_scale_applied=False, bootstrap=0., tail_has_learned_actions=False)
                closure = dict(trajectory_id=trajectory, environment_seed=h["seed"], source_id=h["source_id"],
                    split="training", tail_costs=[1.] * 3, terminal_active=0,
                    prefix_state_sha256=json_hash(prefix), final_state_sha256=json_hash(final), tail_rows_sha256=json_hash(tail))
                self.targets[role, e] = target
                save(self.root, f"{name}/tail/training-target.json", target)
                save(self.root, f"{name}/tail/training-lineage.json", closure)
                original = dict(failure=None, index=2, environment=prefix,
                    events=[r["event"] for r in rows], manifest=h["session_manifest"])
                collected = dict(format="cohort-collection-v1", failure=None, objective=objective, split="training",
                    trajectory_id=trajectory, prefix_live=None, prefix_snapshot=copy.deepcopy(original),
                    prefix_receipt=entry["prefix"], prefix_costs=[float(e + 1)] * 2, tail_events=tail,
                    followup=dict(failure=None, closed=True, steps=3, costs=[1.] * 3, environment=final,
                                  prefix_state=prefix, contract=th["contract"]))
                self.put_container(f"{name}/prefix/collector.pt", original)
                self.put_container(f"{name}/tail/collector.pt", collected)
                records = [r["event"]["audit"]["record"] for r in rows]
                decisions = [r["event"]["audit"]["decision"] for r in rows]
                old_returns = [sum(raw), raw[-1]]
                segment = dict(decisions=decisions, records=records, bootstrap=None, gae_lambda=1.,
                               advantages=old_returns, returns=old_returns)
                batch.append(dict(raw=raw, returns=[sum(training), training[-1]], target=target,
                                  closure=closure, sha=json_hash(segment), policy=h["policy_sha256"]))
                origins.append(dict(trajectory_id=trajectory, environment_seed=h["seed"]))
                consumed.extend([[h["source_id"], trajectory, t] for t in range(2)])
            receipt = dict(method="collected_value", objective=objective,
                trajectory_ids=[r["trajectory_id"] for r in origins], environment_seeds=[r["environment_seed"] for r in origins],
                behavior_sha256s=[r["policy"] for r in batch], terminal_flags=[True] * 2,
                raw_rewards=[r["raw"] for r in batch], collected_values=[[0., 0.]] * 2, baselines=[[0., 0.]] * 2,
                returns=[r["returns"] for r in batch], advantages=[r["returns"] for r in batch],
                segment_sha256s=[r["sha"] for r in batch], cohort_receipts=[r["target"] for r in batch],
                closures=[r["closure"] for r in batch])
            manifest = dict(format="cohort-ppo-kernel-v1", objective=objective, target_method="collected_value",
                mode="online", accounting_steps=3, episode_horizon=2, episodes_per_rollout=2,
                training_source_id=self.streams["namespace"], training_manifest=origins)
            state = dict(manifest=manifest, manifest_sha256=json_hash(manifest), failure=None, pending=[], pending_cohorts=[],
                         history=[4], consumed=consumed, target_history=[copy.deepcopy(receipt)])
            self.put_container(f"payload/models/block60/graph/{role}/final.pt", state)
            phase = dict(job=f"{role}/block60", indexes=copy.deepcopy(self.index[-2:]),
                         updates=[dict(update=1, rollout_steps=4, target_receipt=receipt)])
            self.phases[role] = phase
            save(self.root, f"payload/phases/{role}/block60.json", phase)
        bc = []
        for e in range(2):
            values = invented_episode(self.cfg, self.streams, role="bc_continue", world=e, split="training")
            entry = save_episode(self.root, f"episodes/bc_continue/{e}", values)
            self.index.append(entry)
            bc.append(entry)
        save(self.root, "payload/phases/bc_continue/block60.json",
             dict(job="bc_continue/block60", indexes=bc, updates=[{"invented_imitation_receipt": True}]))
        self.loader = patch("src.rl.cohort_target_verification._load_container",
                            side_effect=lambda root, rec: copy.deepcopy(self.containers[rec["path"]]))
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def put_container(self, path, state):
        self.containers[path] = state
        save(self.root, path, {"invented_container_placeholder": path})

    def check(self):
        return verify_cohort_targets(self.root, self.index, self.cfg, self.streams)

    def test_hand_computed_once_only_targets_and_segment_hashes(self):
        report = self.check()
        self.assertEqual(len(report["rollouts"]), 2)
        self.assertEqual(report["rollouts"][0]["collected_value_mse"], 6.25)
        self.assertEqual(report["rollouts"][1]["collected_value_mse"], 28.75)
        self.assertTrue(report["segment_digest_payload_reconstruction_verified"])
        self.assertTrue(report["once_only_suffix_charge_verified"])
        self.assertFalse(report["checkpoint_inference_performed"])
        self.assertEqual(self.phases["cohort_ppo"]["updates"][0]["target_receipt"]["returns"], [[-5., -4.], [-7., -5.]])

    def test_double_charge_scale_and_closure_hash_rejected(self):
        original = self.targets["cohort_ppo", 0]
        for field, value in (("training_raw_rewards", [-1., -7.]), ("added_terminal_charge", 6.),
                             ("reward_scale_applied", True), ("bootstrap", 1.)):
            target = copy.deepcopy(original)
            target[field] = value
            save(self.root, "episodes/cohort_ppo/0/tail/training-target.json", target)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "suffix target"):
                self.check()
        save(self.root, "episodes/cohort_ppo/0/tail/training-target.json", original)
        path = "episodes/cohort_ppo/0/tail/training-lineage.json"
        closure = json.loads((self.root / path).read_text())
        closure["tail_rows_sha256"] = "a" * 64
        save(self.root, path, closure)
        with self.assertRaisesRegex(ValueError, "closure hash"):
            self.check()

    def test_raw_snapshot_mutation_and_missing_tail_rejected(self):
        key = "episodes/window_ppo/0/tail/collector.pt"
        original = copy.deepcopy(self.containers[key])
        self.containers[key]["prefix_snapshot"]["events"][0]["audit"]["record"]["raw_reward"] = -50.
        with self.assertRaisesRegex(ValueError, "snapshot"):
            self.check()
        self.containers[key] = original
        self.containers[key]["followup"]["steps"] = 2
        with self.assertRaisesRegex(ValueError, "followup"):
            self.check()

    def test_kernel_objective_pending_consumed_and_target_drift_rejected(self):
        key = "payload/models/block60/graph/cohort_ppo/final.pt"
        original = copy.deepcopy(self.containers[key])
        for mutate in (lambda s: s["manifest"].update(objective="window"),
                       lambda s: s.update(pending_cohorts=[{}]),
                       lambda s: s["consumed"].pop(),
                       lambda s: s["target_history"][0]["returns"][0].__setitem__(0, -8.),
                       lambda s: s["target_history"][0]["segment_sha256s"].__setitem__(0, "f" * 64)):
            self.containers[key] = copy.deepcopy(original)
            mutate(self.containers[key])
            with self.assertRaises(ValueError):
                self.check()
        self.containers[key] = original
        self.check()

    def test_missing_duplicate_phase_order_and_raw_hash_rejected(self):
        original = self.index
        for index in (original[:-1], original + [original[0]]):
            self.index = index
            with self.assertRaises(ValueError):
                self.check()
        self.index = original
        self.phases["window_ppo"]["updates"][0]["update"] = 2
        save(self.root, "payload/phases/window_ppo/block60.json", self.phases["window_ppo"])
        with self.assertRaisesRegex(ValueError, "target history"):
            self.check()

    def test_serialization_boundary_mocked_no_model_load_or_forward(self):
        # Exercise the actual safe decoder with an invented in-memory envelope.
        rec = save(self.root, "invented-envelope.pt", {"placeholder": True})
        state = {"only": "artificial"}
        with patch("torch.load", return_value={"state": state, "sha256": json_hash(state)}) as load:
            self.assertEqual(_load_container(self.root, rec), state)
            self.assertTrue(load.call_args.kwargs["weights_only"])
            self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
        with patch("torch.load", return_value={"state": state, "sha256": "f" * 64}):
            with self.assertRaisesRegex(ValueError, "checksum"):
                _load_container(self.root, rec)


if __name__ == "__main__":
    unittest.main()
