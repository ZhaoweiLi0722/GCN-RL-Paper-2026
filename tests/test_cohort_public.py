"""Invented arrays/tensors only: no patient objects, scientific loads or fits."""

import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from src.env.capacity_planning import CapacityPlanningConfig
from src.env.cohort_followup import ClosedCohortClockMixin, CohortTailSpec
from src.env.patient_capacity_planning import PatientConditionCapacityEnv, PatientEnvConfig
from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.rl import cohort_public, patient_replay_collector
from src.rl.candidate_imitation import ImitationSettings
from src.rl.candidate_ppo_kernel import CandidatePPOSettings
from src.rl.cohort_ppo import CohortPPOKernel
from src.rl.cohort_public import CohortObservationProducer, CohortPrefixSession
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.dynamic_candidate_session import DynamicCandidateSession, dynamic_context_from_public
from src.rl.networks import torch
from src.rl.patient_replay_collector import PatientObservationProducer, evidence_digest
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import encode_arrays
from src.rl.time_baseline_collection import TimeBaselineSession
from tests.test_dynamic_candidate_session import FakeReference, OPTIONS, fake_info


SOURCE = "invented-cohort-public"
MANIFEST = tuple(dict(trajectory_id=f"training/invented/{i}", environment_seed=101 + i) for i in range(4))


def config(horizon=2):
    return PatientEnvConfig(base=CapacityPlanningConfig(
        num_facilities=2, production_lead_time=1, episode_horizon=horizon,
        action_mode="facility_net", include_time_state=True,
        demand_rates=(1., 2.), max_specimen_transfer=4., max_bioreactor_transfer=4.,
        max_reagent_transfer=4., max_reagent_replenishment=(4., 4.),
        specimen_edges=((0, 1),), resource_edges=((0, 1),),
        capacity_edges=((0, 1),), information_edges=((0, 1),)),
        enable_specimen_routing=True, include_specimen_routing_state=True)


class ArrayLayout:
    """Unrelated fake engine; uses only config dataclasses and invented arrays."""

    def __init__(self, env_config, seed=101):
        self.env_config, self.config, self._episode_seed = env_config, env_config.base, seed
        self.t, self.actions = 0, []
        for key in ("specimen_edges", "resource_edges", "capacity_edges", "information_edges"):
            setattr(self, key, getattr(self.config, key))
        self.features_per_facility, self.summary_width, self.action_size = 4, 14, 8
        self.observation_size = 37
        self.raw = np.concatenate((np.array([1., 5., 2., 2., 2., 1., 6., 6.]), np.zeros(29))).astype(np.float32)
        self.patient_registry, self.cumulative_enrolled = {}, 0.
        self.demand, self.demand_forecast = np.ones(2), np.ones(2)

    def observation(self):
        return self.raw.copy()

    def assert_identity_conservation(self):
        return dict(active_count=0)

    def state_dict(self):
        return dict(t=self.t, raw=self.raw.copy(), actions=copy.deepcopy(self.actions), seed=self._episode_seed)

    def load_state_dict(self, state):
        self.t, self.raw = state["t"], state["raw"].copy()
        self.actions, self._episode_seed = copy.deepcopy(state["actions"]), state["seed"]

    def step(self, action):
        if action.dtype != np.float64 or action.flags.writeable or action.shape != (8,):
            raise AssertionError("original immutable float64 request required")
        self.actions.append(action.copy())
        self.t += 1
        self.raw[-1] = self.t / self.config.episode_horizon
        info = fake_info(action)
        return self.observation(), -info["cost"], self.t == self.config.episode_horizon, info


class ArrayCohort(ClosedCohortClockMixin, ArrayLayout):
    pass


def fixture(horizon=2, accounting=3):
    cfg = config(horizon)
    # Only the declared type bindings are replaced. The production producer's
    # constructor, packing, anchor and exact-type guard all run unchanged.
    layout_env = ArrayLayout(cfg)
    layout = PatientObservationProducer(layout_env, enabled=True, gamma=1., reward_scale=1e-9,
                                        message_graph="specimen_routes")
    env = ArrayCohort(cfg, cohort_spec=CohortTailSpec(horizon, 2, accounting), enabled=True)
    return layout_env, layout, env, CohortObservationProducer(env, layout, enabled=True)


def kernel(producer, kind="cohort", dtype=None, sampling_seed=41):
    model = DynamicCandidatePolicy(producer.contract.inputs, enabled=True, architecture="graph",
        message_mode="physical", encoder_width=3, head_width=5, actor_seed=31, critic_seed=37,
        initial_reference_bias=0.).to(dtype=torch.float32 if dtype is None else dtype)
    horizon = producer.env.config.episode_horizon
    settings = CandidatePPOSettings(.001, .2, .5, .01, .5, 1., True, 1, 2, 2 * horizon, 2, 8 * horizon)
    common = dict(enabled=True, sampling_seed=sampling_seed, shuffle_seed=43)
    if kind in ("cohort", "window"):
        return CohortPPOKernel(model, producer.contract, settings, mode="online", objective=kind,
            accounting_steps=producer.env.cohort_spec.accounting_steps, training_manifest=MANIFEST,
            training_source_id=SOURCE, episode_horizon=horizon, episodes_per_rollout=2, **common)
    if kind in ("frozen", "legacy_online"):
        return DynamicCandidatePPOKernel(model, producer.contract, settings,
            mode="frozen" if kind == "frozen" else "online", **common)
    return DynamicCandidateImitationKernel(model, producer.contract,
        ImitationSettings(.001, .5, 2, 4, 2, "demonstration" if kind == "initialization" else "training"),
        **common)


def collector(kind="cohort", *, dtype=None, boundary=False, horizon=2, sampling_seed=41, **changes):
    _, _, env, producer = fixture(horizon)
    learner = kernel(producer, kind, dtype, sampling_seed)
    args = dict(enabled=True, options=OPTIONS, trajectory_id=MANIFEST[0]["trajectory_id"],
                split="training", selection="sample", source_id=SOURCE, environment_seed=101)
    return CohortPrefixSession(env, producer, FakeReference(producer, boundary=boundary), learner,
                               **(args | changes))


@unittest.skipIf(torch is None, "torch unavailable")
class CohortPublicTests(unittest.TestCase):
    def setUp(self):
        for owner, method in ((PatientConditionCapacityEnv, "__init__"),
                              (PatientConditionCapacityEnv, "step"),
                              (PatientConditionCapacityEnv, "observation"),
                              (torch.optim.Adam, "step"), (torch.optim.SGD, "step"),
                              (DynamicCandidateImitationKernel, "fit"), (CohortPPOKernel, "update")):
            guard = patch.object(owner, method, side_effect=AssertionError("real environment/update forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)
        for module in (patient_replay_collector, cohort_public):
            binding = patch.object(module, "PatientConditionCapacityEnv", ArrayLayout)
            binding.start()
            self.addCleanup(binding.stop)

    def finish(self, run):
        while not run.closed:
            run.step(before_step=lambda: None)

    def test_original_layout_roundtrip_support_and_time_denominator_52(self):
        original, layout, env, producer = fixture(52, 11)
        original.t = env.t = 51
        original.raw[-1] = env.raw[-1] = 51 / 52
        obs, anchor = producer.observe(env.observation())
        old_obs, old_anchor = layout.observe(original.observation())
        self.assertEqual(producer.contract, layout.contract)
        self.assertEqual(producer.contract.replay.reward_scale, 1e-9)
        self.assertEqual(producer.manifest["normalized_time_denominator"], 52)
        self.assertEqual(float(obs.globals[0, 0]), float(np.float32(51 / 52)))
        self.assertNotEqual(float(obs.globals[0, 0]), float(np.float32(51 / 63)))
        for a, b in ((obs.nodes, old_obs.nodes), (obs.globals, old_obs.globals),
                     (obs.physical_links, old_obs.physical_links), (anchor, old_anchor)):
            self.assertTrue(torch.equal(a, b))
        np.testing.assert_array_equal(producer.unpack_raw(obs)[0], env.observation())
        reference = FakeReference(producer)
        _, actual = dynamic_context_from_public(producer, reference, env.observation(), OPTIONS, "same")
        _, expected = dynamic_context_from_public(layout, reference, original.observation(), OPTIONS, "same")
        self.assertEqual(actual, expected)
        before = evidence_digest(producer._layout_definition())
        layout.anchor_config["demand_rates"] = (999., 999.)
        self.assertEqual(before, evidence_digest(producer._layout_definition()))

    def test_no_old_exact_type_guard_is_widened(self):
        _, layout, env, _ = fixture()
        with self.assertRaisesRegex(TypeError, "declared patient"):
            PatientObservationProducer(env, enabled=True, gamma=1., reward_scale=1e-9)
        with self.assertRaisesRegex(ValueError, "config/topology"):
            layout.check_environment(env)
        run = collector()
        args = dict(enabled=True, options=OPTIONS, trajectory_id=run.trajectory_id,
                    split="training", selection="sample", source_id=SOURCE)
        with self.assertRaisesRegex(TypeError, "explicit dynamic"):
            DynamicCandidateSession(run.env, run.producer, run.reference, run.learner, **args)
        with self.assertRaisesRegex(ValueError, "versioned kernel"):
            TimeBaselineSession(run.env, run.producer, run.reference, run.learner,
                                environment_seed=101, **args)

    def test_bridge_rejects_noncohort_opt_out_and_wrong_layout(self):
        original, layout, env, producer = fixture()
        with self.assertRaises(ValueError):
            CohortObservationProducer(env, layout)
        with self.assertRaises(TypeError):
            CohortObservationProducer(original, layout, enabled=True)
        with self.assertRaises(TypeError):
            CohortObservationProducer(env, object(), enabled=True)
        with self.assertRaises(ValueError):
            producer.check_environment(copy.deepcopy(env))
        with self.assertRaises(TypeError):
            producer.observe(env.observation().astype(np.float64))
        with self.assertRaises(ValueError):
            producer.observe(env.observation()[:-1])
        raw = env.observation()
        raw[0] = np.nan
        with self.assertRaises(ValueError):
            producer.observe(raw)

    def test_config_topology_dimensions_clock_and_layout_drift_rejected(self):
        mutations = (
            lambda e, p: setattr(e, "env_config", replace(e.env_config, weight_expiry=1.)),
            lambda e, p: setattr(e, "config", replace(e.config, episode_horizon=63)),
            lambda e, p: setattr(e, "resource_edges", ()),
            lambda e, p: setattr(e, "specimen_edges", ()),
            lambda e, p: setattr(e, "features_per_facility", 5),
            lambda e, p: setattr(e, "summary_width", 13),
            lambda e, p: setattr(e, "observation_size", 38),
            lambda e, p: setattr(e, "action_size", 10),
            lambda e, p: setattr(e, "cohort_spec", CohortTailSpec(2, 2, 4)),
            lambda e, p: setattr(e, "t", 3),
            lambda e, p: setattr(e, "_cohort_failure", {"invented": True}),
            lambda e, p: p.anchor_config.update(demand_rates=(99., 99.)),
            lambda e, p: p._layout.links.fill(0.),
            lambda e, p: setattr(p, "contract", replace(p.contract,
                inputs=replace(p.contract.inputs, global_feature_names=("tail_time",)))),
        )
        for i, mutate in enumerate(mutations):
            _, _, env, producer = fixture()
            mutate(env, producer)
            with self.subTest(case=i), self.assertRaises(ValueError):
                producer.observe(env.observation())
        _, layout, env, _ = fixture()
        layout.contract = replace(layout.contract,
                                  inputs=replace(layout.contract.inputs, global_feature_names=("tail_time",)))
        with self.assertRaisesRegex(ValueError, "schema"):
            CohortObservationProducer(env, layout, enabled=True)

    def test_changed_anchor_and_schema_ids_cannot_be_admitted_as_original_layout(self):
        for change in ("anchor", "settings", "schema", "reward"):
            _, layout, env, _ = fixture()
            if change == "anchor":
                layout.anchor_config["max_specimen_transfer"] = 8.
            elif change == "settings":
                layout.settings = replace(layout.settings, use_patient_priority=True)
            elif change == "schema":
                layout.contract = replace(layout.contract,
                    inputs=replace(layout.contract.inputs, definition_id="invented-forged-schema"),
                    replay=replace(layout.contract.replay, state_schema_id="invented-forged-schema/actor-flat",
                                   action_schema_id="invented-forged-schema/action"))
            else:
                layout.contract = replace(layout.contract,
                    replay=replace(layout.contract.replay, reward_definition_id="invented-other-cost"))
            with self.subTest(change=change), self.assertRaises(ValueError):
                CohortObservationProducer(env, layout, enabled=True)

    def test_interrupted_and_complete_restore_all_supported_modes_no_updates(self):
        modes = [(kind, "sample", "training") for kind in ("cohort", "window", "bc")]
        modes += [(kind, mode, "test") for kind in ("cohort", "window", "bc", "frozen")
                  for mode in ("greedy", "reference", "anchor")]
        modes += [("frozen", "sample", "preflight")]
        for kind, selection, split in modes:
            with self.subTest(kind=kind, selection=selection, split=split):
                args = dict(selection=selection, split=split)
                left, restored = collector(kind, **args), collector(kind, **args)
                initial_weights = left.learner.policy.snapshot_sha256()
                left.step(before_step=lambda: None)
                restored.load_state_dict(left.state_dict())
                self.assertIs(restored.producer.env, restored.env)
                self.assertEqual(state_digest(restored.state_dict()), state_digest(left.state_dict()))
                a = left.step(before_step=lambda: None)
                b = restored.step(before_step=lambda: None)
                self.assertEqual(state_digest(encode_arrays(a)), state_digest(encode_arrays(b)))
                finished = collector(kind, **args)
                finished.load_state_dict(left.state_dict())
                self.assertTrue(finished.closed)
                self.assertEqual(state_digest(left.state_dict()), state_digest(finished.state_dict()))
                self.assertEqual(left.learner.policy.snapshot_sha256(), initial_weights)
                self.assertEqual(left.manifest["format"], "cohort-prefix-session-v1")
                if split == "training":
                    self.assertEqual(left.segment(1.), restored.segment(1.))
                else:
                    with self.assertRaises(ValueError):
                        left.segment(1.)

    def test_reference_and_anchor_requests_are_exact_and_distinct(self):
        reference = collector("frozen", split="test", selection="reference")
        anchor = collector("frozen", split="test", selection="anchor")
        expected_reference = reference.reference.act(reference.env.observation())
        expected_anchor = anchor.producer.observe(anchor.env.observation())[1][0].tolist()
        a = reference.step(before_step=lambda: None)["audit"]["record"]["action"]
        b = anchor.step(before_step=lambda: None)["audit"]["record"]["action"]
        self.assertEqual(a, tuple(expected_reference))
        self.assertEqual(b, tuple(expected_anchor))
        self.assertNotEqual(a, b)

    def test_bc_evaluation_without_sampler_restores_but_cannot_sample(self):
        args = dict(sampling_seed=None, split="test", selection="greedy")
        run, restored = collector("bc", **args), collector("bc", **args)
        run.step(before_step=lambda: None)
        restored.load_state_dict(run.state_dict())
        self.assertEqual(state_digest(run.state_dict()), state_digest(restored.state_dict()))
        self.assertIsNone(restored.initial_rng)
        with self.assertRaisesRegex(ValueError, "private generator"):
            collector("bc", sampling_seed=None)

    def test_float64_boundary_requests_not_replaced_with_float32(self):
        args = dict(split="test", selection="reference", boundary=True)
        run = collector("frozen", dtype=torch.float64, **args)
        original = run.reference.act(run.env.observation())
        event = run.step(before_step=lambda: None)
        self.assertEqual(event["audit"]["record"]["action"], tuple(original))
        self.assertEqual(run.env.actions[0].dtype, np.float64)
        self.assertNotEqual(tuple(original), tuple(original.astype(np.float32).astype(np.float64)))
        restored = collector("frozen", dtype=torch.float64, **args)
        restored.load_state_dict(run.state_dict())
        self.assertEqual(state_digest(run.state_dict()), state_digest(restored.state_dict()))
        unsafe, charges = collector("frozen", **args), []
        with self.assertRaisesRegex(ValueError, "precision conversion"):
            unsafe.step(before_step=lambda: charges.append(1))
        self.assertEqual(charges, [])

    def test_closed_enrollment_forbids_inference_snapshot_and_prefix_reopening(self):
        run = collector()
        self.finish(run)
        saved, segment = run.state_dict(), run.segment(1.)
        run.env.close_enrollment()
        with patch.object(run.learner.policy, "forward", side_effect=AssertionError("tail inference forbidden")) as forward:
            for call in (lambda: run.step(before_step=lambda: None),
                         lambda: run.producer.observe(run.env.observation()),
                         lambda: run._context(run.env.observation(), run.expected_token),
                         run.state_dict, lambda: run.load_state_dict(saved)):
                with self.assertRaises(ValueError):
                    call()
            self.assertEqual(segment, run.segment(1.))
            forward.assert_not_called()
        self.assertTrue(run.env._cohort_closed)
        restored = collector()
        restored.load_state_dict(saved)
        self.assertTrue(restored.closed)
        self.assertFalse(restored.env._cohort_closed)

    def test_owned_copy_restore_validates_loaded_clock_not_discarded_tail_clock(self):
        run = collector()
        self.finish(run)
        saved = run.state_dict()
        run.env.close_enrollment()
        # Mirror only the dispatcher's private-copy reset, not a tail transition.
        owned = copy.deepcopy(run)
        for name in ("_cohort_closed", "_cohort_ids", "_cohort_enrolled",
                     "_cohort_resolution_step", "_cohort_prefix_state"):
            setattr(owned.env, name, False if name == "_cohort_closed" else None)
        owned.env._cohort_steps, owned.env._cohort_costs = 0, []
        owned.env.t = 5
        owned.load_state_dict(saved)
        self.assertEqual(owned.env.t, 2)
        self.assertEqual(state_digest(owned.state_dict()), state_digest(saved))
        self.assertTrue(run.env._cohort_closed)

    def test_full_52_step_fake_prefix_packs_endpoint_before_closure(self):
        run = collector("frozen", horizon=52, split="test", selection="anchor")
        self.finish(run)
        self.assertEqual(len(run.events), 52)
        self.assertEqual(run.producer.normalized_time_denominator, 52)
        self.assertEqual(run.env.raw[-1], 1.)
        self.assertTrue(run.events[-1]["audit"]["record"]["terminated"])
        self.assertFalse(run.env._cohort_closed)
        run.producer.observe(run.env.observation())
        saved = run.state_dict()
        run.env.close_enrollment()
        self.assertEqual(saved["index"], 52)
        with self.assertRaises(ValueError):
            run.producer.observe(run.env.observation())

    def test_seed_manifest_modes_and_target_horizons_fail_closed(self):
        for changes in (dict(enabled=False), dict(environment_seed=True), dict(environment_seed=999),
                        dict(split="holdout"), dict(split="demonstration"), dict(split="qualification"),
                        dict(split="test", selection="sample"), dict(selection="reference"),
                        dict(selection="greedy"), dict(source_id="wrong"), dict(trajectory_id="wrong"),
                        dict(options=[dict(group="reagent_transfer", epsilon=.5, sign=1)])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                collector(**changes)
        for kind in ("frozen", "legacy_online", "initialization"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                collector(kind)
        for field in ("accounting_steps", "episode_horizon"):
            run = collector()
            setattr(run.learner, field, 99)
            with self.assertRaisesRegex(ValueError, "horizon"):
                CohortPrefixSession(run.env, run.producer, run.reference, run.learner,
                    enabled=True, options=OPTIONS, trajectory_id=MANIFEST[0]["trajectory_id"],
                    source_id=SOURCE, environment_seed=101, split="training", selection="sample")

    def test_manifest_reward_and_objective_tamper_restore_atomically(self):
        run = collector()
        run.step(before_step=lambda: None)
        before = state_digest(run.state_dict())
        for field in ("format", "environment_seed", "kernel_manifest_sha256", "cohort_producer", "reward"):
            saved = run.state_dict()
            if field == "reward":
                saved["events"][0]["audit"]["record"]["raw_reward"] += 1.
            elif field == "cohort_producer":
                saved["manifest"][field]["normalized_time_denominator"] = 63
            else:
                saved["manifest"][field] = "forged"
            with self.subTest(field=field), self.assertRaises(ValueError):
                run.load_state_dict(saved)
            self.assertEqual(state_digest(run.state_dict()), before)
        with self.assertRaisesRegex(ValueError, "manifest"):
            collector("window").load_state_dict(run.state_dict())

    def test_debit_failure_latches_without_refund_and_cannot_retry(self):
        run, charges = collector(), []
        saved = run.state_dict()
        before_environment = evidence_digest(run.env.state_dict())

        def fail():
            charges.append("spent")
            raise RuntimeError("invented debit failure")

        with self.assertRaisesRegex(RuntimeError, "debit failure"):
            run.step(before_step=fail)
        self.assertEqual(charges, ["spent"])
        self.assertEqual(before_environment, evidence_digest(run.env.state_dict()))
        self.assertTrue(run.state_dict()["failure"]["debit_callback_started"])
        for call in (lambda: run.step(before_step=lambda: charges.append("retry")),
                     lambda: run.load_state_dict(saved),
                     lambda: collector().load_state_dict(run.state_dict())):
            with self.assertRaisesRegex(ValueError, "failed session"):
                call()
        self.assertEqual(charges, ["spent"])

    def test_prefix_alone_cannot_enter_cohort_kernel_and_raw_rewards_stay_raw(self):
        run = collector()
        self.finish(run)
        segment = run.segment(1.)
        with self.assertRaisesRegex(ValueError, "complete training-only tail"):
            run.learner.add_segment(segment)
        self.assertEqual(run.learner.pending, [])
        self.assertEqual(tuple(r.raw_reward for r in segment.records),
                         tuple(-e["info"]["cost"] for e in run.events))
        self.assertTrue(segment.records[-1].terminated)
        self.assertFalse(segment.records[-1].truncated)
        self.assertEqual(run.learner.total_optimizer_steps, 0)


if __name__ == "__main__":
    unittest.main()
