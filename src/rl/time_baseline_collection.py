"""Additive collector/continuation admission for the versioned target kernel.

Historical exact-type guards remain intact. The inherited transition, failure,
raw-request precision, checkpoint and nonrefundable-budget code is unchanged.
"""

import copy
from dataclasses import asdict

import numpy as np

from src.rl.candidate_collection_boundary import CollectionBoundary
from src.rl.dynamic_candidate_continuation import DynamicCandidateContinuation
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget
from src.rl.dynamic_candidate_session import DynamicCandidateSession, evidence_digest, make_explicit_residual_option_specs
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.time_baseline_ppo import TimeBaselinePPOKernel


class TimeBaselineSession(DynamicCandidateSession):
    def __init__(self, env, producer, reference, learner, *, enabled=False, options,
                 trajectory_id, split, selection, source_id, environment_seed):
        if (enabled is not True or type(learner) is not TimeBaselinePPOKernel
                or split not in ("preflight", "training", "test")
                or selection not in ("sample", "greedy", "reference", "anchor")):
            raise ValueError("explicit versioned kernel and collection contract required")
        if (type(environment_seed) is not int or environment_seed < 0
                or (split == "training" and selection != "sample")
                or (split == "test" and selection == "sample")):
            raise ValueError("collection mode or environment seed conflicts with split")
        if learner._failure is not None:
            raise ValueError("failed kernel is terminal")
        if (not isinstance(getattr(producer, "contract", None), ReplayInputContract)
                or any(not callable(getattr(producer, method, None))
                       for method in ("observe", "unpack_raw", "check_environment"))):
            raise TypeError("explicit public producer required")
        if not callable(getattr(reference, "act", None)):
            raise TypeError("explicit public reference required")
        reference_id = getattr(reference, "checkpoint_sha256", None)
        if (not isinstance(reference_id, str) or len(reference_id) != 64
                or any(c not in "0123456789abcdef" for c in reference_id)):
            raise ValueError("explicit reference SHA256 required")
        option_specs = make_explicit_residual_option_specs(options)
        if any(spec.group != "specimen_transfer" or not np.isfinite(spec.epsilon) for spec in option_specs[1:]):
            raise ValueError("finite specimen-only options required")
        if not all(isinstance(v, str) and v for v in (trajectory_id, source_id)):
            raise ValueError("explicit source and trajectory identity required")
        if learner.contract != producer.contract or env.t != 0:
            raise ValueError("compatible learner and fresh episode required")
        if split == "training":
            expected = {r["trajectory_id"]: r["environment_seed"] for r in learner.training_manifest}
            if (source_id != learner.training_source_id or expected.get(trajectory_id) != environment_seed
                    or learner.mode != "online"):
                raise ValueError("episode does not match fixed training manifest")
        producer.check_environment(env)
        if env.config.episode_horizon != learner.episode_horizon:
            raise ValueError("episode horizon differs from target contract")
        self.env, self.producer, self.reference, self.learner = env, producer, reference, learner
        self.options, self.split, self.selection = copy.deepcopy(options), split, selection
        self.trajectory_id, self.source_id = trajectory_id, source_id
        self.boundary = CollectionBoundary(env.config.episode_horizon, env.config.episode_horizon, "terminal")
        self.initial_token = evidence_digest(env.state_dict())
        self.expected_token = self.initial_token
        self.initial_weights = learner.policy.snapshot_sha256()
        self.initial_rng = learner.sampling_rng.get_state().clone()
        self.expected_rng = state_digest(self.initial_rng)
        self.events, self.examples, self.index = [], [], 0
        self._failure, self.last_inference_seconds = None, None
        self.manifest = dict(format="time-baseline-session-v1", contract=asdict(producer.contract),
                             initial_token=self.initial_token, initial_weights=self.initial_weights,
                             initial_rng_sha256=state_digest(self.initial_rng),
                             reference=reference_id, options=copy.deepcopy(options),
                             trajectory_id=trajectory_id, source_id=source_id, split=split,
                             selection=selection, horizon=env.config.episode_horizon,
                             inference_dtype=str(self.dtype), submission_dtype="float64_original_request",
                             replay_precision="integer_request_must_survive_inference_dtype",
                             failure_policy="terminal_session_evidence_never_resumable_no_refunds",
                             environment_seed=environment_seed,
                             kernel_manifest_sha256=learner.manifest_sha256,
                             producer_definition=evidence_digest(dict(
                                 anchor_config=producer.anchor_config, settings=asdict(producer.settings))))


class TimeBaselineContinuation(DynamicCandidateContinuation):
    def __init__(self, kernel, budget, settings, *, enabled=False, scope,
                 seeds, horizon, session_factory):
        if (enabled is not True or type(kernel) is not TimeBaselinePPOKernel
                or type(budget) is not DynamicCandidateBudget or budget.active != scope
                or kernel.mode != "online"):
            raise ValueError("explicit new kernel and active durable owner required")
        expected = dict(episodes_per_model=len(kernel.training_manifest),
                        episodes_per_rollout=kernel.episodes_per_rollout,
                        max_updates=kernel.settings.max_updates, epochs=kernel.settings.epochs, gae_lambda=1.)
        if (settings != expected or horizon != kernel.episode_horizon or not callable(session_factory)
                or tuple(seeds) != tuple(r["environment_seed"] for r in kernel.training_manifest)):
            raise ValueError("exact kernel training contract required")
        self.kernel, self.budget, self.factory = kernel, budget, session_factory
        self.settings, self.role, self.scope, self.seeds = copy.deepcopy(settings), "ppo", scope, tuple(seeds)
        self.horizon = horizon
        self.manifest = dict(format="time-baseline-continuation-v1", role="ppo", scope=scope,
                             seeds=self.seeds, settings=self.settings, horizon=horizon,
                             initial_kernel_sha256=state_digest(kernel.state_dict()))
        self.episode, self.active, self.examples, self.receipts, self.last_closed = 0, None, [], [], None
        self.failure = None

    def start_episode(self):
        session = super().start_episode()
        expected = self.kernel.training_manifest[self.episode]
        if (type(session) is not TimeBaselineSession or session.trajectory_id != expected["trajectory_id"]
                or session.manifest["environment_seed"] != expected["environment_seed"]):
            raise ValueError("collector must bind the exact declared episode seed and identity")
        return session
