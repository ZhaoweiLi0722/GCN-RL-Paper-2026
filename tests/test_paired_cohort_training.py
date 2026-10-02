"""Actual metadata + invented serialized rows/owner adapters, never science."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.rl.candidate_imitation import public_example
from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_recording import normalized
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.paired_cohort_recording import PairedBranchRecorder
from src.rl.paired_cohort_resources import budget_plan, stream_manifest, runtime_streams
from src.rl.paired_cohort_training import actor_examples, block_dataset, branch_verifier, collect_block, fit_block
from src.rl.prospective_patient_session import encode_arrays
from tests.test_paired_cohort_actor import prototype, example, contract
from tests.test_paired_cohort_resources import config
from tests.test_paired_cohort_verification import branch, make_context, raw_json


class RawBranchFixture:
    """Replay authored records without constructing or stepping any simulator."""
    def __init__(self, context, payload):
        self.context, self.payload = copy.deepcopy((context, payload))
        self.manifest = self.payload["manifest"]
        self.rows, self.failure, self.prefix_final = [], None, None
        self.env = self

    @property
    def closed(self):
        return len(self.rows) == len(self.payload["rows"])

    def state_dict(self):
        # The JSON states are sufficient for this fake writer/dispatcher test.
        return copy.deepcopy(self.payload["final_state"] if self.closed else self.payload["initial_state"])

    def receipt(self):
        if not self.closed:
            raise ValueError("not complete")
        return copy.deepcopy(self.payload["receipt"])

    def step(self, *, before_step):
        before_step()
        row = copy.deepcopy(self.payload["rows"][len(self.rows)])
        self.record(self, row)
        self.rows.append(row)
        if row["absolute_step"] == 51:
            self.prefix_final = copy.deepcopy(self.payload["prefix_final"])


def dataset():
    rows = []
    for i in range(12):
        value = example(i)
        rows.append(dict(context_id=f"invented/{i}", block=60, cohort=i // 3,
            after_prefix_steps=[4, 20, 36][i % 3], public_example=public_example(
                value.observation, value.bank, contract(), split="training", identity=f"invented/{i}"),
            raw_costs=value.raw_costs.tolist(), replication_seed_ids=value.replication_seed_ids))
    complete = normalized(dict(format="paired-cohort-label-dataset-v1", labels=rows))
    complete["dataset_sha256"] = digest(complete)
    return block_dataset(complete, 60)


class FakeActor:
    def __init__(self, prototype, contract, settings, examples, *, dataset_sha256, arm, enabled):
        if not enabled or len(examples) != 12 or settings.max_optimizer_steps != 128:
            raise ValueError("wrong finite fake job")
        self.steps, self.arm = 0, arm
        self.binding = dataset_sha256

    def state_dict(self):
        return dict(invented_fixture=True, steps=self.steps, arm=self.arm, dataset_sha256=self.binding)

    def update(self, *, before_optimizer_step, before_compute):
        before_compute()
        before_optimizer_step("actor")
        self.steps += 1
        return dict(optimizer_steps=self.steps, critic_optimizer_steps=0, invented_counter_only=True)


class PairedCohortTrainingTests(unittest.TestCase):
    def setUp(self):
        for cls in (torch.optim.Adam, torch.optim.SGD, torch.optim.AdamW):
            guard = patch.object(cls, "step", side_effect=AssertionError("real optimizer forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_independent_json_to_actor_preserves_raw_costs_and_original_support(self):
        data = dataset()
        values, sha = actor_examples(json.loads(json.dumps(data)), contract())
        self.assertEqual(sha, data["dataset_sha256"])
        self.assertEqual(len(values), 12)
        self.assertEqual(values[0].raw_costs.dtype, torch.float64)
        self.assertEqual(values[0].raw_costs[0, 1] - values[0].raw_costs[0, 0], 2.)
        self.assertEqual(values[0].bank, example(0).bank)
        data["states"][0]["public_example"]["split"] = "test"
        body = dict(data)
        body.pop("dataset_sha256")
        data["dataset_sha256"] = digest(body)
        with self.assertRaises(ValueError):
            actor_examples(data, contract())

    def test_two_finite_fit_jobs_reuse_models_and_charge_only_fake_actor_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = DynamicCandidateBudget(root / "ledger.jsonl", budget_plan(config()), enabled=True, clock=lambda: 0.)
            self.addCleanup(budget.close)
            model = prototype()
            sha = model.snapshot_sha256()
            for arm, phase in (("paired_cost", "paired_actor"), ("bc_continue", "bc_actor")):
                budget.begin(phase + "/block60")
                owner, record = fit_block(root, config(), 60, arm, model, contract(), dataset(), budget,
                                          admit=lambda _: None, owner_type=FakeActor)
                self.assertEqual(owner.steps, 128)
                self.assertEqual(load_envelope(root / record["path"])["steps"], 128)
                self.assertEqual(len(list((root / record["path"]).parent.glob("step*.pt"))), 128)
                budget.finish()
            budget.close()
            ledger = read_dynamic_ledger(root / "ledger.jsonl")
            self.assertEqual(ledger["counts"]["optimizer"], 256)
            self.assertEqual(ledger["counts"].get("environment", 0), 0)
            self.assertEqual(model.snapshot_sha256(), sha)

    def test_write_once_branch_reader_path_and_partial_boundary(self):
        context, payload = branch()
        cfg = config()
        streams = dict(environment={"60": {"context": [context["environment"]["scalars"]["_episode_seed"]]}},
            conditional_future={"block60/cohort0/after4": [payload["manifest"]["future_seed"], "999"]})
        with tempfile.TemporaryDirectory() as temp:
            run = RawBranchFixture(context, payload)
            recorder = PairedBranchRecorder(temp, "branch", run)
            run.record = recorder.append
            saved = recorder.snapshot()
            run.step(before_step=lambda: None)
            with self.assertRaises(ValueError):
                recorder.assert_boundary(saved)
            while not run.closed:
                run.step(before_step=lambda: None)
            index = recorder.finish(run, verifier=branch_verifier(cfg, streams))
            self.assertEqual(index["result"]["steps"], 59)
            self.assertEqual(index["result"]["remaining_losses"], 1)
            self.assertTrue((Path(temp) / "branch/final.pt").is_file())
            with self.assertRaises(ValueError):
                recorder.finish(run, verifier=branch_verifier(cfg, streams))

    def test_real_stream_metadata_through_complete_fake_block_branch_entry(self):
        cfg = config()
        streams = stream_manifest(cfg, {str(b): str(b) for b in cfg["blocks"]})
        def world_seed(b, c):
            return int(streams["environment"][str(b)]["context"][c])
        def future_seed(b, c, t, r):
            return int(streams["conditional_future"][f"block{b}/cohort{c}/after{t}"][r])
        with patch("tests.test_paired_cohort_verification.context_seed", world_seed), \
                patch("tests.test_paired_cohort_verification.future_seed", future_seed):
            contexts = {(c, t): make_context(60, c, t) for c in range(4) for t in (4, 20, 36)}
            def factory(template, producer_factory, reference, context, **kw):
                kw["before_clone"]()
                _, payload = branch(context, kw["candidate_index"], kw["replication"])
                self.assertEqual(payload["manifest"]["future_seed"], str(kw["future_seed"]))
                payload["manifest"]["branch_id"] = kw["branch_id"]
                payload["receipt"]["manifest"] = copy.deepcopy(payload["manifest"])
                return RawBranchFixture(context, payload)
            with tempfile.TemporaryDirectory() as temp:
                budget = DynamicCandidateBudget(Path(temp) / "ledger.jsonl", budget_plan(cfg), enabled=True, clock=lambda: 0.)
                self.addCleanup(budget.close)
                budget.begin("paired_branches/block60")
                counts = {}
                def admit(operation):
                    counts[operation] = counts.get(operation, 0) + 1
                index = collect_block(temp, cfg, streams, 60, contexts, None, None, None, budget,
                    admit=admit, verifier=branch_verifier(cfg, streams), branch_factory=factory)
                budget.finish()
                budget.close()
                self.assertEqual(len(index), 72)  # Three canonical classes, not six padded aliases.
                self.assertEqual(counts["conditional_branch_clone"], 72)
                self.assertEqual(counts["branch_step"], 3096)
                self.assertEqual(read_dynamic_ledger(Path(temp) / "ledger.jsonl")["counts"]["environment"], 3096)

    def test_branch_publication_failure_preserves_partial_and_forbids_second_pass(self):
        context, payload = branch()
        with tempfile.TemporaryDirectory() as temp:
            run = RawBranchFixture(context, payload)
            recorder = PairedBranchRecorder(temp, "branch", run)
            run.record = recorder.append
            run.step(before_step=lambda: None)
            recorder.close_partial()
            self.assertTrue((Path(temp) / "branch/events.jsonl").read_text())
            with self.assertRaises(FileExistsError):
                PairedBranchRecorder(temp, "branch", RawBranchFixture(context, payload))


if __name__ == "__main__":
    unittest.main()
