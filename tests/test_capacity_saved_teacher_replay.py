"""Artificial empty public records and fake filters; zero neural/native work."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.rl import capacity_saved_teacher_replay as replay


ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = json.loads((ROOT / "specs/2026-10-03-dynamic-capacity-adaptation/pilot-proposal.json").read_text())
ZERO = [0.0] * 4


def saved_public(epoch):
    committed = 2.0 if 0 < epoch <= 48 else 0.0
    last = [committed, 0.0, 4.0, 0.0, 0.0, 1.0] if epoch else [0.0] * 6
    service = None if not epoch else dict(
        epoch=epoch - 1, known_at=epoch, eligible_order=[[] for _ in range(4)],
        completed_ids=[[] for _ in range(4)], ordinary_hours=[4.0] * 4, applied_hours=ZERO[:])
    return dict(
        common=dict(epoch=epoch, site_ids=["S0", "S1", "S2", "S3"],
                    base_observation=[float(epoch)],
                    capacity_history=[[[0.0] * 6 for _ in range(7)] + [last[:]] for _ in range(4)],
                    pending_hours=[ZERO[:], ZERO[:]], ready_waiting_counts=[0] * 4),
        operations=dict(epoch=epoch, site_ids=["S0", "S1", "S2", "S3"], patients=[],
                        waiting_order=[[] for _ in range(4)], reagents=[8.0] * 4,
                        bioreactors=[[8.0, 0.0, 0.0, 0.0, 0.0] for _ in range(4)],
                        reagent_transfers=[ZERO[:]], capacity_transfers=[ZERO[:]],
                        reagent_orders=[ZERO[:], ZERO[:]], supplier_available=[1.0] * 4,
                        demand_forecast=[1.5] * 4, current_arrivals=ZERO[:], last_service=service))


def summaries(epoch):
    return [[float(epoch + 1)] + [0.0] * 8 for _ in range(4)]


def filter_state(epoch, proposal):
    return dict(format="completion-interval-filter-v1", proposal=copy.deepcopy(proposal),
                epoch=epoch, last_view=saved_public(epoch), weights=[[0.2] * 5 for _ in range(4)],
                boxes=[[{} for _ in range(5)] for _ in range(4)], location={}, closed=[],
                patients={}, reset_events=[])


def fixture(length=64, *, condition=0, replicate=0):
    world = dict(phase="teacher_bc_critic_warmup", block=1, condition=condition,
                 replicate=replicate, seed=62601000 + 10 * condition + replicate)
    rows = []
    for epoch in range(length):
        cost = float(epoch + 1)
        hours = [2.0] * 4 if epoch < 48 else ZERO[:]
        components = {key: 0.0 for key in replay.COST_KEYS}
        components[replay.COST_KEYS[0]] = cost
        rows.append(dict(epoch=epoch, world=world.copy(), role="teacher", cost=cost,
                         reward=-cost, components=components, requested_hours=hours[:],
                         executed_hours=hours[:], public_input=saved_public(epoch),
                         filter_summary=summaries(epoch + 1), filter_resets=0,
                         patient_records=[], service=saved_public(epoch + 1)["operations"]["last_service"],
                         info=dict(cost=cost, support_public_receipt=dict(
                             epoch=epoch, known_at=epoch + 1, raw_requested_hours=hours[:],
                             committed_hours=hours[:]))))
    end = dict(format="capacity-completion-control-recovery1", filter=filter_state(length, PROPOSAL),
               lifecycle=dict(epoch=length), last_plan=None)
    return rows, end


class FakeFilter:
    def __init__(self, proposal, *, scientific):
        if scientific is not True:
            raise AssertionError("reconstruction must request scientific accounting")
        self.proposal = copy.deepcopy(proposal)
        self.epoch = -1
        self.last_view = None
        self.reset_events = []
        self.weights = np.full((4, 5), 0.2)

    def update(self, view, *, before_filter):
        epoch = view.common.epoch
        if epoch != self.epoch + 1:
            raise AssertionError("fake filter received a repeated or skipped receipt")
        if epoch:
            for site in range(4):
                for old in range(5):
                    for new in range(5):
                        before_filter(dict(epoch=epoch - 1, site=site, old_response_index=old,
                                           new_response_index=new, hypothesis_transitions=1))
        self.epoch, self.last_view = epoch, replay.jsonable(view)

    def node_summaries(self):
        return np.asarray(summaries(self.epoch))

    def state_dict(self):
        state = filter_state(self.epoch, self.proposal)
        state["last_view"] = self.last_view
        return state


class Hooks:
    def __init__(self):
        self.receipts = self.finished = self.primitives = 0
        self.pending = 0

    def before(self):
        if self.pending:
            raise AssertionError("previous chunk was not closed")
        self.receipts += 1
        self.pending = 100

    def primitive(self, payload):
        if self.pending <= 0 or payload["hypothesis_transitions"] != 1:
            raise AssertionError("primitive was not admitted")
        self.pending -= 1
        self.primitives += 1

    def after(self):
        if self.pending:
            raise AssertionError("chunk closed before the last primitive")
        self.finished += 1


class SavedTeacherReplayTests(unittest.TestCase):
    def run_replay(self, rows, end, hooks=None, **kwargs):
        hooks = hooks or Hooks()
        result = replay.reconstruct_teacher(
            rows, end, PROPOSAL, before_receipt=hooks.before, before_filter=hooks.primitive,
            after_receipt=hooks.after, filter_factory=kwargs.pop("filter_factory", FakeFilter), **kwargs)
        return result, hooks

    def test_decode_nested_tuples_and_no_aliasing(self):
        raw = saved_public(1)
        before = copy.deepcopy(raw)
        public = replay.decode_public(raw)
        self.assertIsInstance(public.common.site_ids, tuple)
        self.assertIsInstance(public.common.capacity_history[0][0], tuple)
        self.assertIsInstance(public.operations.patients, tuple)
        self.assertIsInstance(public.operations.last_service.eligible_order[0], tuple)
        self.assertEqual(raw, before)
        raw["common"]["capacity_history"][0][-1][0] = 900
        self.assertEqual(public.common.capacity_history[0][-1][0], 2.0)

    def test_complete_terminal_tail_and_same_epoch_features(self):
        rows, end = fixture()
        result, hooks = self.run_replay(rows, end)
        self.assertEqual((len(result), hooks.receipts, hooks.finished, hooks.primitives), (48, 64, 64, 6400))
        for epoch, transition in enumerate(result):
            self.assertEqual(transition.state[0][-16], epoch + 1)
            self.assertEqual(transition.state[0][-2], epoch / 64)
            expected_next = 64 if epoch == 47 else epoch + 1
            self.assertEqual(transition.next_state[0][-16], expected_next + 1)
            self.assertEqual(transition.next_state[0][-2], expected_next / 64)
            self.assertEqual(transition.done, epoch == 47)
        tail = sum(float(epoch + 1) for epoch in range(48, 64))
        self.assertEqual(result[-1].settlement_cost, tail)
        self.assertEqual(result[-1].full_cost, 48 + tail)
        self.assertEqual(result[-1].reward, -(48 + tail) / 100000)
        self.assertTrue(all(t.settlement_cost is None for t in result[:-1]))

    def test_partial_last_next_state_comes_from_failure_view(self):
        rows, end = fixture(47, condition=2, replicate=3)
        result, hooks = self.run_replay(rows, end)
        self.assertEqual((len(result), hooks.receipts, hooks.finished, hooks.primitives), (47, 47, 47, 4700))
        self.assertEqual(result[-1].next_state[0][-16], 48)
        self.assertEqual(result[-1].next_state[0][-2], 47 / 64)
        self.assertTrue(all(not t.done and t.settlement_cost is None for t in result))
        self.assertEqual(result[-1].world_id, "teacher_bc_critic_warmup-b1-c2-j3-teacher")

    def test_block1_exact_751_receipts_zero_neural_or_native(self):
        hooks = Hooks()
        with patch("src.rl.capacity_ddpg_learner.CapacityDDPGLearner.__init__", side_effect=AssertionError("neural")), \
                patch("src.rl.capacity_native.CapacityPilotEnv.__init__", side_effect=AssertionError("native")):
            for replicate in range(4):
                for condition in range(3):
                    length = 47 if (condition, replicate) == (2, 3) else 64
                    self.run_replay(*fixture(length, condition=condition, replicate=replicate), hooks=hooks)
        self.assertEqual((hooks.receipts, hooks.finished, hooks.primitives), (751, 751, 75100))

    def test_default_filter_binding_is_scientific_and_epoch_zero_free(self):
        rows, end = fixture(47)
        hooks = Hooks()
        with patch.object(replay, "CompletionIntervalFilter", FakeFilter):
            replay.reconstruct_teacher(rows, end, PROPOSAL, before_receipt=hooks.before,
                                       before_filter=hooks.primitive, after_receipt=hooks.after)
        self.assertEqual(hooks.receipts, 47)

    def test_independent_reconstruction_and_no_mutation(self):
        rows, end = fixture(47)
        original = copy.deepcopy((rows, end, PROPOSAL))
        first, _ = self.run_replay(rows, end)
        second, _ = self.run_replay(rows, end)
        self.assertEqual(first, second)
        self.assertEqual((rows, end, PROPOSAL), original)
        self.assertIsNot(first, second)
        rows[0]["executed_hours"][0] = 0
        self.assertEqual(first[0].executed_hours[0], 2)

    def test_jsonable_comparison_accepts_tuple_snapshot_sequences(self):
        rows, end = fixture(47)
        end["filter"]["boxes"] = tuple(tuple(box for box in site) for site in end["filter"]["boxes"])
        end["filter"]["weights"] = np.asarray(end["filter"]["weights"])
        result, _ = self.run_replay(rows, end)
        self.assertEqual(len(result), 47)

    def test_corrupt_rows_fail_before_any_filter_work(self):
        cases = {
            "order": lambda r, e: r[1].update(epoch=0),
            "world": lambda r, e: r[1]["world"].update(seed=62601001),
            "role": lambda r, e: r[0].update(role="learner"),
            "cost": lambda r, e: r[0].update(cost=float("nan")),
            "component": lambda r, e: r[0]["components"].update(patient_loss_cost=2),
            "missing_component": lambda r, e: r[0]["components"].pop("patient_loss_cost"),
            "reward": lambda r, e: r[0].update(reward=0),
            "request": lambda r, e: r[0]["requested_hours"].__setitem__(0, float("inf")),
            "receipt": lambda r, e: r[0]["info"]["support_public_receipt"].update(known_at=2),
            "missing_public": lambda r, e: r[0].pop("public_input"),
            "public_epoch": lambda r, e: r[0]["public_input"]["common"].update(epoch=1),
            "end_epoch": lambda r, e: e["filter"].update(epoch=46),
            "proposal": lambda r, e: e["filter"]["proposal"]["learner"].update(gamma=0.5),
            "format": lambda r, e: e.update(format="capacity-completion-control-v1"),
        }
        for name, corrupt in cases.items():
            with self.subTest(name=name):
                rows, end = fixture(47)
                corrupt(rows, end)
                hooks = Hooks()
                with self.assertRaises((ValueError, KeyError)):
                    self.run_replay(rows, end, hooks)
                self.assertEqual(hooks.receipts, 0)

    def test_missing_tail_and_nonzero_tail_commitments_rejected(self):
        for length in (0, 46, 48, 63, 65):
            with self.subTest(length=length), self.assertRaises(ValueError):
                self.run_replay(*fixture(length))
        rows, end = fixture()
        rows[48]["requested_hours"][0] = 1
        with self.assertRaisesRegex(ValueError, "fixed tail"):
            self.run_replay(rows, end)

    def test_wrong_next_summary_and_corrupted_final_filter_fail(self):
        rows, end = fixture(47)
        rows[0]["filter_summary"] = summaries(0)
        with self.assertRaisesRegex(ValueError, "next-epoch"):
            self.run_replay(rows, end)
        rows, end = fixture(47)
        end["filter"]["closed"] = ["corrupt"]
        with self.assertRaisesRegex(ValueError, "final filter"):
            self.run_replay(rows, end)

    def test_failed_primitive_does_not_finish_chunk_or_mutate_inputs(self):
        rows, end = fixture(47)
        before = copy.deepcopy((rows, end))
        hooks = Hooks()

        def fail(payload):
            hooks.primitive(payload)
            if hooks.primitives == 14:
                raise RuntimeError("artificial primitive failure")

        with self.assertRaisesRegex(RuntimeError, "primitive failure"):
            replay.reconstruct_teacher(rows, end, PROPOSAL, before_receipt=hooks.before,
                                       before_filter=fail, after_receipt=hooks.after, filter_factory=FakeFilter)
        self.assertEqual((hooks.receipts, hooks.primitives, hooks.finished, hooks.pending), (1, 14, 0, 86))
        self.assertEqual((rows, end), before)

    def test_after_receipt_runs_after_filter_return_not_last_admission(self):
        rows, end = fixture(47)
        hooks = Hooks()
        instances = []

        def factory(*args, **kwargs):
            instance = FakeFilter(*args, **kwargs)
            instances.append(instance)
            return instance

        def after():
            self.assertEqual(instances[0].epoch, hooks.receipts)
            hooks.after()

        replay.reconstruct_teacher(rows, end, PROPOSAL, before_receipt=hooks.before,
                                   before_filter=hooks.primitive, after_receipt=after, filter_factory=factory)

    def test_missing_hooks_rejected(self):
        rows, end = fixture(47)
        with self.assertRaises(TypeError):
            replay.reconstruct_teacher(rows, end, PROPOSAL, before_receipt=None,
                                       before_filter=lambda _: None, after_receipt=lambda: None)


if __name__ == "__main__":
    unittest.main()
