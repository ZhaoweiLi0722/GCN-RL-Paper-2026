"""Pure configs, invented model tensors and text seals, never scientific fits."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.rl.cohort_factory import cohort_config, fork_cohort_initializer
from src.rl.cohort_objective_plan import cohort_stream_manifest
from src.rl.cohort_sequence import CohortSequence
from src.rl.dynamic_candidate_factory import dynamic_template, dynamic_initializer
from src.rl.dynamic_candidate_resources import dynamic_stream_manifest
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from tests.test_cohort_objective_plan import fixture
from tests.test_dynamic_candidate_factory import proposal as original_config
from tests.test_dynamic_candidate_session import FakeProducer
from tests.test_time_baseline_campaign import saved_design


class CohortFactorySequenceTests(unittest.TestCase):
    def test_pure_binding_preserves_mechanisms_and_splits_objective_from_raw_reward(self):
        base, p = fixture()
        original = saved_design()
        saved = copy.deepcopy((original, base, p))
        cfg = cohort_config(original, base, p)
        self.assertEqual((original, base, p), saved)
        for field in ("objective", "optimizer", "reference", "candidate_support", "model_proposal", "initialization"):
            self.assertEqual(cfg[field], original[field])
        self.assertFalse(cfg["scientific_execution_authorized"])
        self.assertEqual(cfg["totals"]["main_environment_steps"], 33186)
        self.assertEqual(cfg["totals"]["mandatory_restore_clone_steps"], 144)
        self.assertEqual(cfg["totals"]["prefix_parity_steps"], 156)
        self.assertEqual(cfg["totals"]["main_optimizer_calls"], 1920)
        self.assertEqual(cfg["evaluation"]["environment_steps"], 216 * 63)
        self.assertNotIn("time_baseline_proposal", cfg)

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_artificial_equal_weight_forks_no_inherited_moments_or_rng_aliases(self):
        base, p = fixture()
        original = original_config()
        cfg = cohort_config(original, base, p)
        streams = cohort_stream_manifest(base, p)
        old_streams = dynamic_stream_manifest(original)
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("no fit")) as sentinel:
            init = dynamic_initializer(dynamic_template(FakeProducer(), original, old_streams, 60),
                                       original, old_streams, 60)
            receipt = dict(passed=True, kernel_sha256=state_digest(init.state_dict()))
            with self.assertRaises(ValueError):
                fork_cohort_initializer(init, receipt, cfg, streams, 60)
            # Artificial metadata fixture, not qualification evidence or fitting.
            init.steps = cfg["initialization"]["actor_adam_calls_per_block"]
            receipt["kernel_sha256"] = state_digest(init.state_dict())
            forks = fork_cohort_initializer(init, receipt, cfg, streams, 60)
            preflight = fork_cohort_initializer(init, receipt, cfg, streams, 60, split="preflight")
            self.assertEqual(len(forks), 4)
            self.assertEqual({k.policy.snapshot_sha256() for k in forks.values()}, {init.policy.snapshot_sha256()})
            self.assertEqual({state_digest(k.sampling_rng.get_state()) for k in forks.values()},
                             {state_digest(forks["own_frozen"].sampling_rng.get_state())})
            self.assertNotEqual(state_digest(preflight["cohort_ppo"].sampling_rng.get_state()),
                                state_digest(forks["cohort_ppo"].sampling_rng.get_state()))
            self.assertEqual(forks["cohort_ppo"].objective, "cohort")
            self.assertEqual(forks["window_ppo"].objective, "window")
            for k in (forks["cohort_ppo"], forks["window_ppo"]):
                self.assertEqual((k.episode_horizon, k.accounting_steps), (52, 11))
            self.assertEqual(len({id(k.sampling_rng) for k in forks.values()}), 4)
            sentinel.assert_not_called()

    def test_fixed_33_section_order_and_all_twelve_models_before_test(self):
        base, p = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sequence = CohortSequence(base, p, root, enabled=True)
            self.assertEqual(len(sequence.jobs), 33)
            self.assertEqual(len(sequence.models), 12)
            with self.assertRaises(ValueError):
                sequence.begin("final_evaluation/block60/cohort_ppo")
            for i, job in enumerate(sequence.jobs):
                sequence.begin(job)
                if job == "all_model_seal":
                    with self.assertRaises(ValueError):
                        sequence.seal_models({})
                    paths = {}
                    for j, name in enumerate(sequence.models):
                        path = root / f"invented-model{j}.txt"
                        path.write_text(name)
                        paths[name] = path
                    sequence.seal_models(paths)
                evidence = root / f"invented-evidence{i}.txt"
                evidence.write_text(job)
                sequence.finish([evidence])
                restored = CohortSequence(base, p, root, enabled=True)
                restored.load_state_dict(sequence.state_dict())
                self.assertEqual(restored.state_dict(), sequence.state_dict())
            self.assertIsNone(sequence.next_job)
            Path(next(iter(paths.values()))).write_text("tampered")
            with self.assertRaises(ValueError):
                sequence.check_seals()

    def test_failure_is_terminal_and_completed_phase_cannot_rewind(self):
        base, p = fixture()
        with tempfile.TemporaryDirectory() as directory:
            sequence = CohortSequence(base, p, directory, enabled=True)
            old = sequence.state_dict()
            sequence.begin(sequence.next_job)
            evidence = Path(directory).resolve() / "invented.txt"
            evidence.write_text("fixture")
            sequence.finish([evidence])
            with self.assertRaises(ValueError):
                sequence.load_state_dict(old)
            sequence.fail("invented failure")
            with self.assertRaises(ValueError):
                sequence.load_state_dict(old)


if __name__ == "__main__":
    unittest.main()
