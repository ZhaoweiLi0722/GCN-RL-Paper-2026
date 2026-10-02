"""Opt-in public prefix bridge; no environment builds, registration or fitting.

Supply a PatientObservationProducer from the already-budgeted original layout
environment, then bind one bridge to each explicitly constructed cohort episode.
The original schema, anchor, graph and raw request support are reused verbatim.
This module owns only prefix collection: freeze its snapshot before enrollment
closure and use the separate cohort dispatcher for tail persistence/recovery.
"""

import copy
from dataclasses import asdict

import numpy as np

from src.baselines.heuristics import heuristic_settings_for_policy
from src.env.cohort_followup import ClosedCohortClockMixin, CohortTailSpec
from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.rl.candidate_collection_boundary import CollectionBoundary
from src.rl.cohort_ppo import CohortPPOKernel
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.dynamic_candidate_session import DynamicCandidateSession
from src.rl.patient_replay_collector import PatientObservationProducer, _edge_sets, evidence_digest
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.residual_options import make_explicit_residual_option_specs


class CohortObservationProducer:
    """Episode-bound composition of an original, independently built layout.

    check_environment/observe allow the open prefix endpoint for terminal record
    packing and historical receipt replay, but never a closed-enrollment state.
    No layout environment is constructed or retained by this bridge.
    """

    def __init__(self, env, layout_producer, *, enabled=False):
        if enabled is not True:
            raise ValueError("explicit cohort public bridge enablement required")
        if type(layout_producer) is not PatientObservationProducer:
            raise TypeError("original PatientObservationProducer layout required")
        if (not isinstance(env, ClosedCohortClockMixin)
                or not isinstance(env, PatientConditionCapacityEnv)
                or type(env.cohort_spec) is not CohortTailSpec):
            raise TypeError("explicit closed-cohort patient subclass required")
        self.env = env
        self._layout = copy.deepcopy(layout_producer)
        self.contract = self._layout.contract
        self.anchor_config = self._layout.anchor_config
        self.settings = self._layout.settings
        self.config_digest = self._layout.config_digest
        self.cohort_spec = env.cohort_spec
        self.normalized_time_denominator = self.anchor_config["episode_horizon"]
        self._validate_layout()
        self._layout_sha256 = evidence_digest(self._layout_definition())
        self.check_environment(env)
        self.manifest = dict(
            format="cohort-public-producer-v1", cohort_spec=asdict(self.cohort_spec),
            layout_sha256=self._layout_sha256, contract=asdict(self.contract),
            normalized_time_denominator=self.normalized_time_denominator,
            endpoint_observation="open_prefix_only_no_closed_enrollment_inference")

    def _layout_definition(self):
        layout = self._layout
        return dict(contract=asdict(layout.contract), config_digest=layout.config_digest,
                    anchor_config=layout.anchor_config, settings=asdict(layout.settings),
                    edges=layout.edges, message_graph=layout.message_graph, links=layout.links,
                    base_width=layout.base_width, summary_width=layout.summary_width,
                    raw_width=layout.raw_width)

    def _validate_layout(self):
        layout, base = self._layout, self.anchor_config
        if not isinstance(self.contract, ReplayInputContract):
            raise TypeError("original replay input contract required")
        schema, replay = self.contract.inputs, self.contract.replay
        patient = asdict(self.env.env_config)
        expected_anchor = patient.pop("base")
        expected_anchor.update(patient, env_type="patient_condition")
        if base != expected_anchor or self.settings != heuristic_settings_for_policy("mdl2"):
            raise ValueError("original config-derived full MDL-2 anchor required")
        definition = "patient-raw-request-v1/" + evidence_digest(dict(
            env=layout.config_digest, edges=layout.edges, anchor=asdict(layout.settings),
            message_graph="specimen_routes"))
        n = base["num_facilities"]
        actions = tuple(f"{group}_{i}" for group in
                        ("specimen_net", "reagent_net", "capacity_net", "replenishment")
                        for i in range(n))
        if (layout.message_graph != "specimen_routes" or base["action_mode"] != "facility_net"
                or base.get("enable_specimen_routing") is not True
                or base["enable_overtime_control"] or base["include_on_order_state"]
                or base["enable_stochastic_procurement"] or base["reagent_purchase_lead_time"]
                or self.normalized_time_denominator != self.cohort_spec.enrollment_steps
                or schema.definition_id != definition
                or replay.state_schema_id != definition + "/actor-flat"
                or replay.action_schema_id != definition + "/action"
                or replay.reward_kind != "absolute_environment"
                or replay.reward_definition_id != "patient-negative-total-cost/" + layout.config_digest
                or schema.node_ids != tuple(f"facility_{i}" for i in range(n))
                or schema.action_names != actions
                or schema.global_feature_names != (("normalized_time",) if base["include_time_state"] else ())
                or len(schema.node_feature_names) != layout.base_width + layout.summary_width
                or layout.raw_width != n * (layout.base_width + layout.summary_width) + int(base["include_time_state"])
                or replay.action_dim != 4 * n or replay.state_dim != layout.raw_width + n * n + 4 * n):
            raise ValueError("unchanged specimen-graph prefix observation/action schema required")
        links = np.zeros((n, n), dtype=np.float32)
        for a, b in layout.edges[0]:
            links[a, b] = links[b, a] = 1.
        if layout.links.dtype != np.float32 or not np.array_equal(layout.links, links):
            raise ValueError("original specimen message topology required")

    def check_environment(self, env):
        layout = self._layout
        if (env is not self.env or env.cohort_spec != self.cohort_spec
                or evidence_digest(asdict(env.env_config)) != self.config_digest
                or env.config != env.env_config.base or _edge_sets(env) != layout.edges
                or env.features_per_facility != layout.base_width
                or env.summary_width != layout.summary_width
                or env.observation_size != layout.raw_width
                or env.action_size != self.contract.replay.action_dim
                or env.config.episode_horizon != self.normalized_time_denominator):
            raise ValueError("cohort environment config/topology/schema or binding changed")
        if (self.contract != layout.contract or self.anchor_config != layout.anchor_config
                or self.settings != layout.settings or self.config_digest != layout.config_digest
                or evidence_digest(self._layout_definition()) != self._layout_sha256):
            raise ValueError("original public layout changed")
        if (env._cohort_closed is not False or env._cohort_failure is not None
                or env._cohort_steps != 0 or env._cohort_prefix_state is not None
                or env._cohort_ids is not None or env._cohort_enrolled is not None
                or env._cohort_resolution_step is not None or env._cohort_costs
                or type(env.t) is not int or not 0 <= env.t <= self.cohort_spec.enrollment_steps):
            raise ValueError("closed, failed or non-prefix cohort cannot supply public inference")

    def observe(self, raw):
        self.check_environment(self.env)
        return self._layout.observe(raw)

    def unpack_raw(self, observation):
        return self._layout.unpack_raw(observation)


class CohortPrefixSession(DynamicCandidateSession):
    """Versioned constructor over the unchanged dynamic transition/restore code.

    CohortPPOKernel covers both objective arms. Existing training-split BC and
    frozen DynamicCandidatePPOKernel cover BC and evaluation/reference/anchor
    controls. No legacy online PPO or initializer fit path is admitted here.
    BC's ordered training inventory remains the outer coordinator's authority.
    """

    def __init__(self, env, producer, reference, learner, *, enabled=False, options,
                 trajectory_id, split, selection, source_id, environment_seed):
        if (enabled is not True or split not in ("preflight", "training", "test")
                or selection not in ("sample", "greedy", "reference", "anchor")):
            raise ValueError("explicit cohort prefix collection contract required")
        if (type(environment_seed) is not int or environment_seed < 0
                or (split == "training" and selection != "sample")
                or (split == "test" and selection == "sample")
                or (selection == "anchor" and split not in ("preflight", "test"))):
            raise ValueError("collection mode or environment seed conflicts with split")
        if type(producer) is not CohortObservationProducer:
            raise TypeError("episode-bound CohortObservationProducer required")
        producer.check_environment(env)
        if type(learner) not in (CohortPPOKernel, DynamicCandidateImitationKernel, DynamicCandidatePPOKernel):
            raise TypeError("explicit cohort PPO, continuation BC or frozen kernel required")
        if learner._failure is not None:
            raise ValueError("failed kernel is terminal")
        if type(learner) is DynamicCandidatePPOKernel and (learner.mode != "frozen" or split == "training"):
            raise ValueError("legacy dynamic kernel is allowed only as a frozen nontraining control")
        if type(learner) is DynamicCandidateImitationKernel and learner.settings.allowed_split != "training":
            raise ValueError("only existing continuation BC is admitted; no initializer path")
        if (learner.contract != producer.contract or env.t != 0
                or env._episode_seed != environment_seed):
            raise ValueError("compatible learner and fresh episode with exact seed required")
        if selection == "sample" and learner.sampling_rng is None:
            raise ValueError("sampling collection requires a private generator")
        reference_id = getattr(reference, "checkpoint_sha256", None)
        if not callable(getattr(reference, "act", None)):
            raise TypeError("explicit public reference required")
        if (not isinstance(reference_id, str) or len(reference_id) != 64
                or any(c not in "0123456789abcdef" for c in reference_id)):
            raise ValueError("explicit reference SHA256 required")
        specs = make_explicit_residual_option_specs(options)
        if any(spec.group != "specimen_transfer" or not np.isfinite(spec.epsilon) for spec in specs[1:]):
            raise ValueError("finite specimen-only options required")
        if not all(isinstance(v, str) and v for v in (trajectory_id, source_id)):
            raise ValueError("explicit source and trajectory identity required")
        if type(learner) is CohortPPOKernel:
            if (env.config.episode_horizon != learner.episode_horizon
                    or env.cohort_spec.accounting_steps != learner.accounting_steps):
                raise ValueError("cohort prefix/tail horizon differs from target contract")
            if split == "training":
                expected = {r["trajectory_id"]: r["environment_seed"] for r in learner.training_manifest}
                if (source_id != learner.training_source_id or expected.get(trajectory_id) != environment_seed
                        or learner.mode != "online"):
                    raise ValueError("episode does not match fixed training manifest")
        self.env, self.producer, self.reference, self.learner = env, producer, reference, learner
        self.options, self.split, self.selection = copy.deepcopy(options), split, selection
        self.trajectory_id, self.source_id = trajectory_id, source_id
        self.boundary = CollectionBoundary(env.config.episode_horizon, env.config.episode_horizon, "terminal")
        self.initial_token = evidence_digest(env.state_dict())
        self.expected_token = self.initial_token
        self.initial_weights = learner.policy.snapshot_sha256()
        self.initial_rng = None if learner.sampling_rng is None else learner.sampling_rng.get_state().clone()
        self.expected_rng = state_digest(self.initial_rng)
        self.events, self.examples, self.index = [], [], 0
        self._failure, self.last_inference_seconds = None, None
        kernel_manifest = (state_digest(learner.manifest) if type(learner) is DynamicCandidateImitationKernel
                           else learner.manifest_sha256)
        self.manifest = dict(
            format="cohort-prefix-session-v1", contract=asdict(producer.contract),
            initial_token=self.initial_token, initial_weights=self.initial_weights,
            initial_rng_sha256=state_digest(self.initial_rng), reference=reference_id,
            options=copy.deepcopy(options), trajectory_id=trajectory_id, source_id=source_id,
            split=split, selection=selection, horizon=env.config.episode_horizon,
            inference_dtype=str(self.dtype), submission_dtype="float64_original_request",
            replay_precision="integer_request_must_survive_inference_dtype",
            failure_policy="terminal_session_evidence_never_resumable_no_refunds",
            environment_seed=environment_seed, kernel_manifest_sha256=kernel_manifest,
            producer_definition=evidence_digest(dict(anchor_config=producer.anchor_config,
                                                      settings=asdict(producer.settings))),
            cohort_producer=copy.deepcopy(producer.manifest))

    def _require_live_kernel(self):
        super()._require_live_kernel()
        if self.env._cohort_failure is not None:
            raise ValueError("failed cohort is terminal; prefix cannot restore to retry")

    def state_dict(self):
        if self.env._cohort_closed:
            raise ValueError("freeze prefix snapshot before enrollment closure; tail uses separate persistence")
        return super().state_dict()

    def _restore(self, saved):
        # Check before the inherited loader can replace the environment snapshot.
        # The outer dispatcher may explicitly reset a private copy's cohort
        # flags first; the inherited loader validates its restored prefix clock.
        self._require_live_kernel()
        if self.env._cohort_closed:
            raise ValueError("closed cohort requires separate tail recovery, not prefix reopening")
        super()._restore(saved)
