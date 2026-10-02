"""Additive reference-training prefix and admitted native environment binding."""

import copy
from dataclasses import asdict, replace

import numpy as np

from src.env.cohort_followup import CohortTailSpec, cohort_environment_class
from src.rl.candidate_collection_boundary import CollectionBoundary
from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.cohort_backend import CohortPatientBackend
from src.rl.cohort_public import CohortObservationProducer, CohortPrefixSession
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.patient_replay_collector import evidence_digest
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.residual_options import make_explicit_residual_option_specs


class ReferenceContextSession(CohortPrefixSession):
    """Training-only R4 contexts; frozen scorer records receipts, never fits.

    This constructor deliberately does not relabel training as preflight or
    mutate any historical sample-only guard. Transition/restore checks are reused.
    """

    def __init__(self, env, producer, reference, learner, *, enabled=False, options,
                 trajectory_id, source_id, environment_seed):
        if enabled is not True or type(producer) is not CohortObservationProducer:
            raise ValueError("explicit episode-bound reference training session required")
        producer.check_environment(env)
        if (type(learner) is not DynamicCandidatePPOKernel or learner.mode != "frozen"
                or learner._failure is not None or learner.contract != producer.contract
                or type(environment_seed) is not int or environment_seed < 0
                or env.t != 0 or env._episode_seed != environment_seed):
            raise ValueError("unchanged frozen owner and fresh declared reference world required")
        ref = getattr(reference, "checkpoint_sha256", None)
        if (not callable(getattr(reference, "act", None)) or not isinstance(ref, str)
                or len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref)
                or not all(isinstance(v, str) and v for v in (trajectory_id, source_id))):
            raise ValueError("reference and context provenance required")
        specs = make_explicit_residual_option_specs(options)
        if any(s.group != "specimen_transfer" or not np.isfinite(s.epsilon) for s in specs[1:]):
            raise ValueError("unchanged finite specimen support required")
        self.env, self.producer, self.reference, self.learner = env, producer, reference, learner
        self.options, self.split, self.selection = copy.deepcopy(options), "training", "reference"
        self.trajectory_id, self.source_id = trajectory_id, source_id
        self.boundary = CollectionBoundary(env.config.episode_horizon, env.config.episode_horizon, "terminal")
        self.initial_token = self.expected_token = evidence_digest(env.state_dict())
        self.initial_weights = learner.policy.snapshot_sha256()
        self.initial_rng = None if learner.sampling_rng is None else learner.sampling_rng.get_state().clone()
        self.expected_rng = state_digest(self.initial_rng)
        self.events, self.examples, self.index = [], [], 0
        self._failure = self.last_inference_seconds = None
        self.manifest = dict(format="paired-reference-context-session-v1", contract=asdict(producer.contract),
            initial_token=self.initial_token, initial_weights=self.initial_weights,
            initial_rng_sha256=self.expected_rng, reference=ref, options=copy.deepcopy(options),
            trajectory_id=trajectory_id, source_id=source_id, split=self.split, selection=self.selection,
            horizon=env.config.episode_horizon, inference_dtype=str(self.dtype),
            submission_dtype="float64_original_request", replay_precision="integer_request_must_survive_inference_dtype",
            failure_policy="terminal_session_evidence_never_resumable_no_refunds", environment_seed=environment_seed,
            kernel_manifest_sha256=learner.manifest_sha256,
            producer_definition=evidence_digest(dict(anchor_config=producer.anchor_config, settings=asdict(producer.settings))),
            cohort_producer=copy.deepcopy(producer.manifest), target_admission="external_branch_labels_only")

    def segment(self, *args, **kwargs):
        raise ValueError("reference contexts cannot enter a PPO rollout")


class PairedCohortBackend(CohortPatientBackend):
    def session(self, block, kernel, seed, *, trajectory, split, selection):
        if split != "training":
            return super().session(block, kernel, seed, trajectory=trajectory, split=split, selection=selection)
        self._admit("episode_build")
        if (selection != "reference" or block not in self.contexts
                or seed not in self.streams["environment"][str(block)]["context"]
                or self.episode_builds >= self.config["totals"]["fresh_episode_builds"]):
            raise ValueError("one prescribed fresh R4 context cohort required")
        self.episode_builds += 1
        before = global_rng_state()
        try:
            from src.env.patient_capacity_planning import patient_env_config_from_dict
            from src.rl.experiment import apply_graph_ablation
            from src.rl.frozen_value_probe import assert_scenario
            runtime, layout, reference = self.contexts[block]
            cfg = dict(runtime.get("env", {}))
            ablation = cfg.pop("graph_ablation", runtime.get("graph_ablation", "full_graph"))
            scenario = cfg.pop("scenario_name", runtime.get("scenario", "default"))
            typed = patient_env_config_from_dict(cfg)
            typed = replace(typed, base=apply_graph_ablation(typed.base, ablation))
            spec = CohortTailSpec(**self.config["cohort_proposal"])
            env = cohort_environment_class()(typed, seed=seed, cohort_spec=spec, enabled=True)
            env.scenario_name, env.graph_ablation = scenario, ablation
            assert_scenario(env, runtime, self.config["objective"]["scenario"])
            return ReferenceContextSession(env, CohortObservationProducer(env, layout, enabled=True), reference,
                kernel, enabled=True, options=self.config["candidate_support"]["options"], trajectory_id=trajectory,
                source_id=self.streams["namespace"], environment_seed=seed)
        except BaseException as error:
            self.failure = dict(operation="reference_context_session", error=repr(error))
            raise
        finally:
            restore_global_rng(before)

    def branch_producer(self, block, env):
        return CohortObservationProducer(env, self.contexts[block][1], enabled=True)
