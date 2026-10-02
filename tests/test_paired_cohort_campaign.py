"""The native serial campaign with real saved JSON metadata and fake backends.

Every world, model load and update below is an authored fixture; real patient
construction and optimizer steps are guarded. No scientific result is read.
"""

import copy
from functools import partial
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.rl.candidate_patient_session import load_envelope, save_envelope
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, read_dynamic_ledger
from src.rl.paired_cohort_binding import bind_inputs
from src.rl.paired_cohort_campaign import PairedCohortCampaign
from src.rl.paired_cohort_resources import inherited_metadata, stream_manifest, budget_plan
from src.rl.prospective_ddpg_kernel import state_digest
from src.utils.research_archive import inventory
from tests.test_paired_cohort_resources import ROOT, config


def packet():
    old = json.loads((ROOT / "specs/2026-10-02-terminal-obligation/frozen.json").read_text())
    previous = json.loads((ROOT / "specs/2026-10-02-cohort-evaluation-recovery2/frozen.json").read_text())
    cfg = config()
    inherited = inherited_metadata(cfg, old, previous)
    return dict(config=cfg, inherited=inherited, streams=stream_manifest(cfg, inherited["layout_seeds"]),
        initializer_config=old["original_config"], initializer_streams=old["initializer_streams"],
        inherited_scientific_config=old["scientific_config"], qualified_blocks=old["qualified_blocks"],
        engineering_fixture=True)


class Owner:
    def __init__(self, value):
        self.value = copy.deepcopy(value)
        self.contract = "invented-contract"
        self.policy = self
    def state_dict(self):
        return copy.deepcopy(self.value)
    def load_state_dict(self, value):
        self.value = copy.deepcopy(value)
    def snapshot_sha256(self):
        return state_digest(self.value)


class FakeAdmission:
    engineering_fixture = True
    def __init__(self):
        self.counts = {}
    def __call__(self, operation):
        self.counts[operation] = self.counts.get(operation, 0) + 1


class Backend:
    engineering_fixture = True
    def __init__(self, workspace, config, streams, *, admit_real_calls):
        self.config, self.streams, self.admit, self.contexts = config, streams, admit_real_calls, {}
    def prepare(self, block, seed):
        if type(seed) is not int:
            raise ValueError("native runtime decimal conversion missing")
        self.admit("input_hash_verification")
        self.admit("input_hash_verification")
        for op in ("reference_and_layout_build", "reference_checkpoint_load", "layout_environment_build"):
            self.admit(op)
        self.contexts[block] = (None, None, "invented-reference")
        return dict(engineering_fixture=True, block=block, seed=str(seed))
    def producer(self, block):
        return Owner({"invented": block})
    def branch_producer(self, block, env):
        raise AssertionError("fake branch factory should not request native producer")
    def state_dict(self):
        return dict(engineering_fixture=True, blocks=list(self.contexts))


class PairedCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.p = packet()
        self.admit = FakeAdmission()
        self.budget = DynamicCandidateBudget(self.root / "launcher/budget.jsonl", budget_plan(self.p["config"]),
                                             enabled=True, clock=lambda: 0.)
        self.addCleanup(self.budget.close)
        self.loaded = []
        for cls, method in ((PatientConditionCapacityEnv, "__init__"), (PatientConditionCapacityEnv, "step"),
                            (torch.optim.Adam, "step"), (torch.optim.SGD, "step"), (torch.optim.AdamW, "step")):
            guard = patch.object(cls, method, side_effect=AssertionError("real science forbidden"))
            spy = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(spy.assert_not_called)
        records = self.p["inherited"]["model_inputs"]
        def load(path):
            relative = path.relative_to(ROOT).as_posix()
            if relative not in {r["path"] for r in records.values()}:
                raise AssertionError("undeclared fake load")
            self.loaded.append(relative)
            return dict(invented_model=relative)
        for b in self.p["config"]["blocks"]:
            source = records[f"block{b}/initializer"]["path"]
            self.p["qualified_blocks"][str(b)] = dict(passed=True,
                kernel_sha256=state_digest(dict(invented_model=source)))
        def check(pathroot, path):
            return copy.deepcopy(next(r for r in records.values() if r["path"] == path))
        self.bind = partial(bind_inputs, backend_type=Backend, loader=load,
            initializer_restore=lambda s, *args: Owner(s), ppo_restore=Owner,
            wrap=lambda owner, seed: Owner(owner.state_dict()), record=check)

    def fake_episode(self, root, backend_config, cfg, streams, backend, model, budget, **kw):
        admit, block, world, split = (kw[k] for k in ("admit", "block", "world", "split"))
        admit("episode_build")
        if split == "test":
            self.assertTrue((root / "payload/model-seals.json").is_file())
            expected = int(self.p["streams"]["environment"][str(block)]["test"][world])
            self.assertEqual(kw["seed"], expected)
        if kw["clone"]:
            admit("preflight_clone")
        for i in range(63):
            admit("episode_step")
            budget.debit_environment("trajectory")
            if kw["clone"]:
                admit("clone_step")
                budget.debit_environment("clone")
        contexts = {(world, t): dict(invented=True, block=block, cohort=world, after_prefix_steps=t,
            context_id=f"block{block}/cohort{world}/after{t}") for t in cfg["context_after_prefix_steps"]} if split == "training" else {}
        path = root / f"payload/episodes/{kw['name']}/fixture.json"
        write_json_once(path, {k: kw[k] for k in ("block", "role", "world", "split", "seed")})
        return dict(index=dict(fixture=file_record(root, path)), contexts=contexts,
                    environment=Owner(dict(invented_environment=block, t=63)))

    def fake_collect(self, root, cfg, streams, block, contexts, template, producer, reference, budget, **kw):
        self.assertEqual(len(contexts), 12)
        entries = []
        for (cohort, t), context in contexts.items():
            for replication in range(2):
                for candidate in range(3):
                    kw["admit"]("conditional_branch_clone")
                    for _ in range(63 - t):
                        kw["admit"]("branch_step")
                        budget.debit_environment("clone")
                    entries.append(dict(context=context, replication=replication, candidate=candidate, steps=63 - t))
        return entries

    def fake_assemble(self, contexts, branches, **kw):
        self.assertEqual(len(contexts), 36)
        self.assertEqual(len(branches), 216)
        result = dict(format="paired-cohort-label-dataset-v1", labels=contexts, branch_outcomes=branches)
        result["dataset_sha256"] = digest(result)
        return result

    def fake_fit(self, root, cfg, block, arm, prototype, contract, dataset, budget, **kw):
        self.assertEqual(len(dataset["states"]), 12)
        self.assertFalse((root / "payload/model-seals.json").exists())
        self.assertEqual(len(list((root / "payload/episodes/test").glob("**/*"))), 0)
        kw["admit"]("actor_fork")
        for _ in range(128):
            kw["admit"]("actor_update")
            budget.debit_optimizer("actor")
        owner = Owner(dict(invented_model=arm, block=block, steps=128))
        path = root / f"payload/models/block{block}/{arm}/final.pt"
        save_envelope(path, owner.state_dict())
        return owner, file_record(root, path)

    def verify(self, root, index, cfg, inherited, streams):
        self.assertEqual(len(index), 216)
        rows = [json.loads((root / row["fixture"]["path"]).read_text()) for row in index]
        expected = {(b, r, w) for b in cfg["blocks"] for r in cfg["evaluation_controllers"] for w in range(12)}
        self.assertEqual({(r["block"], r["role"], r["world"]) for r in rows}, expected)
        return dict(engineering_fixture=True, analysis=dict(decision="invented_not_science"))

    def campaign(self, **changes):
        args = dict(engineering=True, binder=self.bind, episode=self.fake_episode, collect=self.fake_collect,
            assemble=self.fake_assemble, branch_reader=lambda root, rows: rows, fit=self.fake_fit,
            wrapper=lambda owner, seed: Owner(owner.state_dict()), recycle=lambda env, context: env,
            verifier=self.verify, archive=lambda source, dest: dict(files=inventory(source), engineering_fixture=True))
        args.update(changes)
        return PairedCohortCampaign(ROOT, self.root, self.p, self.budget, self.admit, **args)

    def test_complete_native_phase_dispatch_with_real_metadata_and_no_science(self):
        run = self.campaign()
        state = run.run()
        self.assertIsNone(state["active"])
        self.assertEqual(len(self.loaded), 6)
        self.assertEqual(len(set(self.loaded)), 6)
        self.assertEqual(len(run.models), 12)
        self.assertEqual(len(run.contexts), 36)
        self.assertEqual(len(run.fitted), 6)
        self.assertEqual(len(run.evaluation_indexes), 216)
        self.assertEqual(self.admit.counts["episode_build"], 231)
        self.assertEqual(self.admit.counts["preflight_clone"], 3)
        self.assertEqual(self.admit.counts["conditional_branch_clone"], 216)
        ledger = read_dynamic_ledger(self.budget.path)
        self.assertEqual(ledger["counts"], dict(environment=24030, optimizer=768))
        self.assertEqual(ledger["owner_counts"].get("global:critic", 0), 0)
        saved = run.checkpoint()
        run.restore_boundary(saved)
        self.assertEqual(state_digest(saved), state_digest(run.checkpoint()))
        closure = json.loads((self.root / "launcher/closure.json").read_text())
        self.assertTrue(closure["engineering_fixture"])
        self.assertFalse(closure["automatic_followon"])

    def test_failure_is_terminal_and_spend_is_preserved(self):
        def failure(*args, **kwargs):
            kwargs["admit"]("episode_step")
            args[6].debit_environment("trajectory")
            raise RuntimeError("invented episode failure")
        run = self.campaign(episode=failure)
        with self.assertRaisesRegex(RuntimeError, "episode failure"):
            run.run()
        self.assertEqual(self.budget.counts["environment"], 1)
        self.assertIsNotNone(run.sequence.failure)
        self.assertTrue((self.root / "launcher/campaign-failure.pt").is_file())
        self.assertFalse((self.root / "payload/model-seals.json").exists())
        with self.assertRaises(ValueError):
            run.restore_boundary(run.checkpoint())

    def test_whole_fake_campaign_uses_exact_native_admission_and_owner_caps(self):
        from src.rl import paired_cohort_execution as execution
        from tests.test_paired_cohort_execution import synthetic_packet, reseal

        workspace = self.root / "synthetic-workspace"
        native = synthetic_packet(workspace)
        native.update(self.p)
        native["workspace"] = str(workspace.resolve())
        self.p = reseal(native)
        self.root = workspace / execution.RUN
        self.budget = execution.PairedBudget(self.root / "launcher/budget.jsonl",
            self.p["budget_plan"], enabled=True)
        self.addCleanup(self.budget.close)
        auth = execution.authorization(self.p)
        claim = dict(pid=os.getppid(), packet_sha256=self.p["packet_sha256"],
            authorization_sha256=digest(auth), started=self.budget.started,
            clock_id=execution.CLOCK_ID, head="synthetic-not-authority")
        write_json_once(self.root / "launcher/claim.json", claim)
        self.admit = execution.PairedAdmission(workspace, self.p, auth, claim, self.budget)
        self.addCleanup(self.admit.close)
        run = self.campaign(engineering=False)
        run.run()
        actual = execution.read_admission(self.admit.path, self.p)
        expected = execution.operation_limits(self.p["config"])
        for b in self.p["config"]["blocks"]:
            expected[f"paired_branches/block{b}"] = dict(conditional_branch_clone=72, branch_step=3096)
        self.assertEqual(actual["by_section"], {job: caps for job, caps in expected.items() if caps})
        self.assertEqual(actual["jobs"], list(expected))
        self.assertEqual(self.budget.counts, dict(environment=24030, optimizer=768))

    def test_restore_rejects_changed_durable_evidence(self):
        run = self.campaign()
        before = run.checkpoint()
        run.sequence.begin("binding")
        self.budget.begin("binding")
        with self.assertRaisesRegex(ValueError, "boundary differs"):
            run.restore_boundary(before)

    def test_mismatched_qualification_rejects_before_second_model_load(self):
        self.p["qualified_blocks"]["60"]["kernel_sha256"] = "0" * 64
        run = self.campaign()
        with self.assertRaisesRegex(ValueError, "qualification"):
            run.run()
        self.assertEqual(len(self.loaded), 1)
        self.assertEqual(self.budget.counts.get("environment", 0), 0)


if __name__ == "__main__":
    unittest.main()
