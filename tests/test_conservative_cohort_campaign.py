"""Actual persisted metadata through the native dispatcher using only fake science."""

import copy
from functools import partial
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.conservative_cohort_binding import prepare, bind_inputs
from src.rl.conservative_cohort_campaign import ConservativeCohortCampaign
from src.rl.conservative_cohort_plan import budget_plan
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.prospective_ddpg_kernel import state_digest
from src.utils.research_archive import inventory
from tests.test_conservative_cohort_plan import ROOT
from tests.test_paired_cohort_campaign import Owner, Backend, FakeAdmission


class CampaignTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.packet = prepare(ROOT)
        self.admit = FakeAdmission()
        self.budget = DynamicCandidateBudget(self.root / "launcher/budget.jsonl",
            budget_plan(self.packet["config"]), enabled=True, clock=lambda: 0.)
        self.addCleanup(self.budget.close)
        for cls, name in ((PatientConditionCapacityEnv, "__init__"), (PatientConditionCapacityEnv, "step"),
                          (torch.optim.Adam, "step"), (torch.optim.SGD, "step")):
            guard = patch.object(cls, name, side_effect=AssertionError("real science prohibited"))
            spy = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(spy.assert_not_called)
        self.loads, self.fit_starts = [], {}
        records = self.packet["inherited"]["model_inputs"]
        for b in self.packet["config"]["blocks"]:
            source = records[f"block{b}/initializer"]["path"]
            self.packet["qualified_blocks"][str(b)] = dict(passed=True,
                kernel_sha256=state_digest(dict(invented=source)))
        def load(path):
            name = path.relative_to(ROOT).as_posix()
            self.assertIn(name, {r["path"] for r in records.values()})
            self.loads.append(name)
            return dict(invented=name)
        self.binder = partial(bind_inputs, backend_type=Backend, loader=load,
            restore=lambda s, *a: Owner(s), wrap=lambda owner, seed: Owner(owner.state_dict()),
            record=lambda root, name: next(copy.deepcopy(r) for r in records.values() if r["path"] == name))

    def episode(self, root, backend_cfg, cfg, streams, backend, model, budget, **kw):
        admit, b, w, split = (kw[k] for k in ("admit", "block", "world", "split"))
        admit("episode_build")
        self.assertIs(type(kw["seed"]), int)
        if split == "training":
            self.assertEqual(kw["selection"], "greedy")
            if w >= 2:
                self.assertEqual(model.value["arm"], "paired_cost")
                self.assertEqual(model.value["round"], 0)
        if split == "test":
            self.assertTrue((root / "payload/model-seals.json").is_file())
            self.assertEqual(len(self.fit_starts), 12)
        if kw["clone"]:
            admit("preflight_clone")
        for _ in range(63):
            admit("episode_step")
            budget.debit_environment("trajectory")
            if kw["clone"]:
                admit("clone_step")
                budget.debit_environment("clone")
        contexts = {(w, t): dict(invented=True, block=b, cohort=w, after_prefix_steps=t)
                    for t in cfg["context_after_prefix_steps"]} if split == "training" else {}
        path = root / f"payload/episodes/{kw['name']}/fixture.json"
        write_json_once(path, dict(block=b, world=w, split=split, role=kw["role"], seed=kw["seed"]))
        return dict(index={"fixture": file_record(root, path)}, contexts=contexts,
                    environment=Owner(dict(fake_environment=b)))

    def collect(self, root, cfg, streams, b, rnd, contexts, template, producer, reference, continuation, budget, **kw):
        self.assertEqual(len(contexts), 6)
        if rnd:
            self.assertEqual(continuation.value["round"], 0)
        indexes = []
        for c, t in contexts:
            for r in range(4):
                for k in range(3):
                    kw["admit"]("conditional_branch_clone")
                    for _ in range(63 - t):
                        kw["admit"]("branch_step")
                        budget.debit_environment("clone")
                    indexes.append(dict(result=dict(environment_calls=63 - t), fixture=[b, rnd, c, t, r, k]))
        return indexes

    def assemble(self, contexts, entries, cfg, streams, b, rnd, owner_hash):
        self.assertEqual(len(entries), 72)
        out = dict(block=b, round=rnd, states=list(contexts.values()), continuation=owner_hash)
        out["dataset_sha256"] = digest(out)
        return out

    def fit(self, root, cfg, b, rnd, arm, prototype, contract, dataset, budget, **kw):
        self.fit_starts[b, rnd, arm] = copy.deepcopy(prototype.state_dict())
        self.assertEqual(dataset["round"], rnd)
        if rnd:
            self.assertEqual(prototype.value["arm"], arm)
            self.assertEqual(prototype.value["round"], 0)
        kw["admit"]("actor_fork")
        for _ in range(6):
            kw["admit"]("round_start_logits")
        for _ in range(64):
            kw["admit"]("actor_update")
            budget.debit_optimizer("actor")
        owner = Owner(dict(block=b, round=rnd, arm=arm, updates=64))
        path = root / f"payload/models/round{rnd}/block{b}/{arm}/final.pt"
        save_envelope(path, owner.state_dict())
        return owner, file_record(root, path)

    def verify(self, root, index, cfg, inherited, streams):
        self.assertEqual(len(index), 180)
        return dict(engineering_fixture=True, decision="invented_not_science")

    def campaign(self, **changes):
        args = dict(engineering=True, binder=self.binder, episode=self.episode, collect=self.collect,
            fit=self.fit, assemble=self.assemble, reader=lambda *a: lambda **kw: None,
            wrapper=lambda owner, seed: Owner(owner.state_dict()), recycle=lambda env, context: env,
            verifier=self.verify, archive=lambda source, dest: dict(engineering_fixture=True, files=inventory(source)))
        args.update(changes)
        return ConservativeCohortCampaign(getattr(self, "workspace", ROOT), self.root, self.packet, self.budget, self.admit, **args)

    def test_complete_serial_chain_and_round_lineage_no_science(self):
        run = self.campaign()
        result = run.run()
        self.assertTrue(result["completed"])
        self.assertEqual(len(self.loads), 3)
        self.assertEqual(len(run.contexts), 36)
        self.assertEqual(len(run.branch_indexes), 432)
        self.assertEqual(len(run.seals), 9)
        for b in (60, 61, 62):
            self.assertEqual(self.fit_starts[b, 0, "paired_cost"], self.fit_starts[b, 0, "bc_continue"])
        ledger = read_dynamic_ledger(self.budget.path)
        self.assertEqual(ledger["counts"], dict(environment=31050, optimizer=768))
        self.assertEqual(ledger["owner_counts"].get("global:critic", 0), 0)
        with self.assertRaises(ValueError):
            run.run()
        self.assertTrue((self.root / "launcher/launcher-archive.json").is_file())

    def test_failure_never_spends_the_rest_or_opens_tests(self):
        def fail(*args, **kw):
            args[6].debit_environment("trajectory")
            raise RuntimeError("invented environment failure")
        run = self.campaign(episode=fail)
        with self.assertRaises(RuntimeError):
            run.run()
        self.assertEqual(self.budget.counts["environment"], 1)
        self.assertFalse((self.root / "payload/model-seals.json").exists())
        with self.assertRaises(ValueError):
            run.run()


if __name__ == "__main__":
    unittest.main()
