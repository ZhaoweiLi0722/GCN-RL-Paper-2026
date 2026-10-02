"""Persisted-metadata regressions with no research model or environment I/O."""

import copy
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from src.rl import cohort_evaluation_recovery as original
from src.rl import cohort_evaluation_recovery2 as recovery


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "specs/2026-10-02-cohort-evaluation-recovery/frozen.json"
CONTRACT = {"fixture": "no-scientific-model"}


def environment_seeds(streams):
    for block, splits in streams["environment"].items():
        for split, seeds in splits.items():
            for world, seed in enumerate(seeds):
                yield block, split, world, seed


class FakeBudget:
    def __init__(self, path, plan, **kwargs):
        self.path, self.plan = Path(path), plan
        self.started = kwargs.get("started", 0)
        self.active, self.closed = None, False
        self.counts = {"environment": 0, "optimizer": 0}
        self.sections = []
        self.debit_optimizer = Mock(side_effect=AssertionError("no optimizer"))

    def begin(self, section):
        assert self.active is None
        self.active = section
        self.sections.append(section)

    def finish(self):
        assert self.active is not None
        self.active = None

    def check(self):
        assert not self.closed

    def debit_environment(self, kind):
        assert kind == "trajectory" and self.active.startswith("final_evaluation/")
        self.counts["environment"] += 1

    def snapshot(self):
        return dict(counts=dict(self.counts), active=self.active)

    def close(self):
        self.closed = True


class FakeAdmission:
    def __init__(self):
        self.counts = {}
        self.caps = dict(checkpoint_load=12, reference_checkpoint_load=3,
                         layout_environment_build=3, episode_build=205)

    def __call__(self, operation):
        assert operation in self.caps
        self.counts[operation] = self.counts.get(operation, 0) + 1
        assert self.counts[operation] <= self.caps[operation]


class FakeOwner:
    def __init__(self, state):
        self.contract = CONTRACT
        self.state = copy.deepcopy(state)
        self.fit = Mock(side_effect=AssertionError("no fit"))
        self.update = Mock(side_effect=AssertionError("no update"))

    def state_dict(self):
        return copy.deepcopy(self.state)


class FakeBackend:
    def __init__(self, root, config, streams, *, admit_real_calls):
        assert all(type(seed) is int for *_, seed in environment_seeds(streams))
        self.root, self.config, self.streams = Path(root), config, streams
        self.admission = admit_real_calls
        self.layouts, self.sessions = [], []

    def prepare(self, block, seed):
        assert type(seed) is int
        assert seed == self.streams["environment"][str(block)]["layout"][0]
        self.admission("layout_environment_build")
        self.admission("reference_checkpoint_load")
        self.layouts.append((block, seed))
        return dict(block=block, seed=seed, fake=True)

    def producer(self, block):
        assert block in [row[0] for row in self.layouts]
        return SimpleNamespace(contract=CONTRACT, anchor_config={})

    def session(self, block, owner, seed, **metadata):
        assert type(seed) is int
        assert seed in self.streams["environment"][str(block)]["test"]
        self.admission("episode_build")
        self.sessions.append((block, seed, metadata))
        return SimpleNamespace(learner=owner, producer=self.producer(block), environment_seed=seed)


class FakeRecorder:
    def __init__(self, root, path, prefix, config, **metadata):
        assert type(metadata["seed"]) is int
        assert metadata["seed"] == prefix.environment_seed
        self.metadata = metadata

    def finish_prefix(self, *args):
        return {}

    def record_prefix(self, *args):
        pass

    def record_tail(self, *args):
        pass

    def finish(self, run):
        assert run.closed and run.steps == 63
        assert all(type(seed) is int and seed == self.metadata["seed"] for seed in run.seeds)
        return dict(fake=True, step_seeds=run.seeds, **self.metadata)

    def close_partial(self):
        pass


class FakeCollection:
    def __init__(self, prefix, **kwargs):
        assert kwargs["objective"] == "none" and kwargs["split"] == "test"
        self.steps, self.closed = 0, False
        self.seed, self.seeds = prefix.environment_seed, []

    def step(self, *, before_step):
        before_step()
        assert type(self.seed) is int
        self.seeds.append(self.seed)
        self.steps += 1
        self.closed = self.steps == 63


class CohortEvaluationRecovery2Tests(unittest.TestCase):
    def setUp(self):
        self.frozen_bytes = FROZEN.read_bytes()
        self.packet = json.loads(self.frozen_bytes)
        self.saved_packet = copy.deepcopy(self.packet)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(TemporaryDirectory(prefix="cohort-recovery2-test-"))).resolve()
        self.output = self.root / "mock-output"
        self.writes = {}
        for module in (recovery, original):
            self.stack.enter_context(patch.object(module, "write_json_once", side_effect=self.write_json))
        self.copy_files = self.stack.enter_context(patch.object(recovery, "copy_verified"))
        self.stack.enter_context(patch("builtins.print"))
        self.torch_load = self.stack.enter_context(patch("torch.load", side_effect=AssertionError("no checkpoint I/O")))
        self.budget = FakeBudget(self.output / "budget.jsonl", self.packet["budget_plan"])
        self.admission = FakeAdmission()
        self.backend_type = Mock(side_effect=FakeBackend)
        self.owner_loader = Mock(side_effect=lambda path: {"fake_source": str(path)})
        self.restore_owner = Mock(side_effect=FakeOwner)

    def tearDown(self):
        self.assertEqual(self.packet, self.saved_packet)
        self.assertEqual(FROZEN.read_bytes(), self.frozen_bytes)
        self.torch_load.assert_not_called()
        self.budget.debit_optimizer.assert_not_called()
        self.assertFalse(list(self.root.rglob("*")), "all output stays in doubles")

    def write_json(self, path, value):
        path = Path(path)
        self.assertTrue(path.is_relative_to(self.root))
        self.assertNotIn(path, self.writes, "write-once output duplicated")
        self.writes[path] = copy.deepcopy(value)

    def assert_runtime_streams(self, streams, frozen=None):
        frozen = self.packet["streams"] if frozen is None else frozen
        expected = copy.deepcopy(frozen)
        for block, split, world, seed in environment_seeds(frozen):
            expected["environment"][block][split][world] = int(seed)
        self.assertEqual(streams, expected)
        self.assertTrue(all(type(seed) is int for *_, seed in environment_seeds(streams)))

    def bind(self):
        self.budget.begin("runtime_input_binding")
        return recovery.bind_inputs(self.root, self.output, self.packet, self.budget, self.admission,
                                    backend_type=self.backend_type, owner_loader=self.owner_loader,
                                    restore_owner=self.restore_owner)

    def assert_binding(self, backend, models, runtime_packet):
        self.assert_runtime_streams(backend.streams)
        self.assert_runtime_streams(runtime_packet["streams"])
        self.assertEqual(runtime_packet, self.packet | {"streams": backend.streams})
        self.assertIsNot(runtime_packet, self.packet)
        expected_layouts = [(b, int(self.packet["streams"]["environment"][str(b)]["layout"][0]))
                            for b in original.BLOCKS]
        self.assertEqual(backend.layouts, expected_layouts)
        keys = [f"block{b}/graph/{role}" for b in original.BLOCKS for role in original.ROLES[:4]]
        self.assertEqual(set(models), set(keys))
        self.assertEqual(len({id(owner) for owner in models.values()}), 12)
        self.assertEqual(self.owner_loader.call_args_list,
                         [call(self.root / self.packet["models"][key]["path"]) for key in keys])
        self.assertEqual(self.restore_owner.call_count, 12)
        for key, owner in models.items():
            self.assertEqual(owner.state_dict(), {"fake_source": str(self.root / self.packet["models"][key]["path"])})
            owner.fit.assert_not_called()
            owner.update.assert_not_called()

    def test_real_frozen_seeds_are_lossless_and_deep_copied(self):
        seeds = [seed for *_, seed in environment_seeds(self.packet["streams"])]
        self.assertEqual(len(seeds), 138)
        self.assertTrue(all(type(seed) is str and int(seed) > 2 ** 64 for seed in seeds))
        converted = recovery.runtime_streams(self.packet["streams"])
        self.assert_runtime_streams(converted)
        converted["environment"]["60"]["test"][0] += 1
        converted["neural"]["analysis/bootstrap"] = "changed-only-in-copy"
        converted["ordinals"]["60"]["test"][0] = -1
        self.assertEqual(self.packet, self.saved_packet)

    def test_runtime_streams_accepts_canonical_zero(self):
        streams = copy.deepcopy(self.packet["streams"])
        streams["environment"]["60"]["test"][0] = "0"
        saved = copy.deepcopy(streams)
        converted = recovery.runtime_streams(streams)
        self.assert_runtime_streams(converted, saved)
        self.assertEqual(streams, saved)

    def test_runtime_streams_rejects_ambiguous_or_malformed_seeds(self):
        invalid = [True, False, 0, 1, 2 ** 112, 1.0, 1.5, float("inf"), None, [], {}, "", " 1", "1 ",
                   "+1", "01", "-0", "-1", "1.0", "1e3", "0x10", "1_000", "\u0661", "\uff11"]
        for split in ("layout", "preflight", "test", "training"):
            for value in invalid:
                with self.subTest(split=split, seed=value):
                    streams = copy.deepcopy(self.packet["streams"])
                    streams["environment"]["62"][split][0] = value
                    before = copy.deepcopy(streams)
                    with self.assertRaises((TypeError, ValueError)):
                        recovery.runtime_streams(streams)
                    self.assertEqual(streams, before)

    def test_bind_inputs_loads_each_owner_once_and_passes_runtime_ints(self):
        backend, models, runtime_packet = self.bind()
        self.backend_type.assert_called_once()
        self.assert_binding(backend, models, runtime_packet)
        self.assertEqual(self.admission.counts, dict(checkpoint_load=12,
                         reference_checkpoint_load=3, layout_environment_build=3))
        self.assertEqual(self.budget.counts, {"environment": 0, "optimizer": 0})
        self.copy_files.assert_not_called()

    def test_bind_inputs_rejects_bad_seed_before_backend_or_model_loading(self):
        packet = copy.deepcopy(self.packet)
        packet["streams"]["environment"]["62"]["training"][-1] = "01"
        before = copy.deepcopy(packet)
        self.budget.begin("runtime_input_binding")
        with self.assertRaises(ValueError):
            recovery.bind_inputs(self.root, self.output, packet, self.budget, self.admission,
                                 backend_type=self.backend_type, owner_loader=self.owner_loader,
                                 restore_owner=self.restore_owner)
        self.assertEqual(packet, before)
        self.backend_type.assert_not_called()
        self.owner_loader.assert_not_called()
        self.restore_owner.assert_not_called()
        self.assertEqual(self.admission.counts, {})
        self.assertEqual(self.writes, {})

    @staticmethod
    def fake_campaign(*args, **kwargs):
        return original.RecoveryCampaign(*args, recorder=FakeRecorder, collection=FakeCollection, **kwargs)

    def test_real_metadata_mock_evaluation_records_exact_205_integer_seeds(self):
        backend, models, runtime_packet = self.bind()

        campaign = recovery.run_evaluations(self.output, runtime_packet, self.budget, backend, models,
                                            campaign_type=self.fake_campaign)
        self.assert_evaluations(campaign, backend, models, runtime_packet)
        self.assertEqual(self.budget.sections, ["runtime_input_binding"] +
                         [f"final_evaluation/block{b}/{role}" for b in original.BLOCKS for role in original.ROLES])
        self.assertIsNone(self.budget.active)

    def assert_evaluations(self, campaign, backend, models, runtime_packet):
        self.assert_binding(backend, models, runtime_packet)
        self.assertIs(campaign.packet, runtime_packet)
        self.assertEqual(campaign.index[:11], self.packet["reused_index"])
        self.assertEqual(len(campaign.index), 216)
        expected = [(b, role, world, int(seed))
                    for b in original.BLOCKS for role in original.ROLES
                    for world, seed in enumerate(self.packet["streams"]["environment"][str(b)]["test"])
                    if not (b == 60 and role == "own_frozen" and world < 11)]
        recorded = [(r["block"], r["role"], r["world_index"], r["seed"]) for r in campaign.index[11:]]
        self.assertEqual(recorded, expected)
        self.assertEqual(len(recorded), 205)
        self.assertTrue(all(type(row[-1]) is int for row in recorded))
        for row in campaign.index[11:]:
            self.assertEqual(row["step_seeds"], [row["seed"]] * 63)
            self.assertTrue(all(type(seed) is int for seed in row["step_seeds"]))
        self.assertEqual([(b, seed) for b, seed, _ in backend.sessions], [(b, seed) for b, _, _, seed in expected])
        self.assertEqual(len({meta["trajectory"] for _, _, meta in backend.sessions}), 205)
        self.assertEqual(self.admission.counts, self.admission.caps)
        self.assertEqual(self.budget.counts, {"environment": 12915, "optimizer": 0})

    def test_mock_child_uses_real_binding_and_runtime_packet_without_research_io(self):
        self.output = self.root / recovery.RUN
        self.budget.path = self.output / "launcher/budget.jsonl"
        auth = {"approved": True, "fake": True}
        claim = {"head": "mock-commit", "started": 10.0}
        self.budget.started = claim["started"]

        def read_claim(path, *args, **kwargs):
            self.assertEqual(path, self.output / "launcher/claim.json")
            return json.dumps(claim)

        self.stack.enter_context(patch.object(Path, "read_text", autospec=True, side_effect=read_claim))
        approved = self.stack.enter_context(patch.object(recovery, "approved", return_value=(self.packet, auth)))
        self.stack.enter_context(patch.object(recovery, "git", return_value=claim["head"]))
        budget_type = self.stack.enter_context(patch.object(recovery, "DynamicCandidateBudget", return_value=self.budget))
        admission_type = self.stack.enter_context(patch.object(recovery, "Recovery2Admission", return_value=self.admission))
        self.stack.enter_context(patch("src.rl.cohort_backend.CohortPatientBackend", self.backend_type))
        self.stack.enter_context(patch.object(recovery, "load_envelope", self.owner_loader))
        self.stack.enter_context(patch.object(recovery, "restore_evaluation_owner", self.restore_owner))
        binding = self.stack.enter_context(patch.object(recovery, "bind_inputs", wraps=recovery.bind_inputs))
        run_evaluations = recovery.run_evaluations
        completed = []

        def run_with_doubles(*args):
            campaign = run_evaluations(*args, campaign_type=self.fake_campaign)
            completed.append(campaign)
            return campaign

        evaluation = self.stack.enter_context(patch.object(recovery, "run_evaluations", side_effect=run_with_doubles))
        verifier = self.stack.enter_context(patch.object(original, "verify_cohort_raw_bundle",
            return_value={"analysis": {"decision": "mock-only-no-research-result"}}))
        verify_bindings = self.stack.enter_context(patch.object(recovery, "verify_bindings"))
        self.stack.enter_context(patch.object(recovery, "read_dynamic_ledger",
                                             side_effect=lambda path: {"counts": dict(self.budget.counts)}))
        self.stack.enter_context(patch.object(recovery, "inventory", return_value=[]))
        archive = self.stack.enter_context(patch.object(recovery, "create_archive", return_value={"files": []}))

        result = recovery.child(self.root)
        self.assertEqual(result, 0, self.writes.get(self.output / "launcher/child-failure.json"))
        approved.assert_called_once_with(self.root)
        budget_type.assert_called_once_with(self.budget.path, self.packet["budget_plan"],
                                            enabled=True, started=claim["started"])
        admission_type.assert_called_once_with(self.root, self.packet, auth, claim, self.budget)
        binding.assert_called_once_with(self.root, self.output, self.packet, self.budget, self.admission)
        evaluation.assert_called_once()
        output, runtime_packet, budget, backend, models = evaluation.call_args.args
        self.assertEqual(output, self.output)
        self.assertIs(budget, self.budget)
        self.assertEqual(len(completed), 1)
        campaign = completed[0]
        self.assert_evaluations(campaign, backend, models, runtime_packet)
        verifier.assert_called_once_with(self.output, campaign.index, self.packet["scientific_config"],
                                         self.packet["streams"])
        self.assertEqual(verify_bindings.call_args_list, [call(self.root, self.packet)] * 2)
        archive.assert_called_once_with(self.output / "payload", self.output / "archives/completed-payload.tar.gz")
        self.assertEqual(self.writes[self.output / "payload/evaluation-index.json"], campaign.index)
        self.assertEqual(self.writes[self.output / "launcher/closure.json"]["evaluations"], 216)
        self.assertEqual(self.writes[self.output / "launcher/closure.json"]["new_training_jobs"], 0)
        self.assertEqual(self.budget.sections, ["runtime_input_binding"] +
                         [f"final_evaluation/block{b}/{role}" for b in original.BLOCKS for role in original.ROLES] +
                         ["raw_verification", "payload_archive", "supervisor_closure"])
        self.assertEqual(set(self.budget.sections), set(self.packet["budget_plan"]["sections"]))
        self.assertTrue(self.budget.closed)
        self.assertIsNone(self.budget.active)
        self.assertTrue(self.copy_files.called)
        for args in self.copy_files.call_args_list:
            self.assertTrue(Path(args.args[1]).is_relative_to(self.output))


if __name__ == "__main__":
    unittest.main()
