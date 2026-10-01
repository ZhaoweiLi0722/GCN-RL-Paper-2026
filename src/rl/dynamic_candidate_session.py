"""Opt-in dynamic-policy episode sessions; no registration or research launcher.

This additive collector retains the audited session's receipts, cost checks and
atomic array/envelope encoding. It does not alter legacy type or split guards.
Kernel terminal failures are never cleared by session rollback or restoration.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import time

import numpy as np

from src.rl.candidate_collection_boundary import CollectionBoundary, audit_candidate_step
from src.rl.candidate_imitation import decode_example, public_example
from src.rl.candidate_patient_session import restore_decision, save_envelope, load_envelope
from src.rl.candidate_rollout import CandidateDecision, prepare_candidate_segment, sample_candidate
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.dynamic_candidate_rollout import evaluate_dynamic_policy, verify_dynamic_evaluation
from src.rl.networks import torch
from src.rl.patient_replay_collector import evidence_digest
from src.rl.prospective_adapter import ReplayInputContract, _decode_actor_state, pack_actor_state
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import encode_arrays, decode_arrays, restore_record
from src.rl.residual_options import make_explicit_residual_option_specs, residual_option_actions_from_state
from src.rl.routing_candidate_contract import (
    RoutingRequestSchema, build_request_candidates, choose_candidate, validate_choice_precision,
)


def _cast_observation(observation, dtype):
    return replace(observation, nodes=observation.nodes.to(dtype=dtype),
                   globals=observation.globals.to(dtype=dtype),
                   physical_links=observation.physical_links.to(dtype=dtype))


def dynamic_context_from_public(producer, reference, raw, options, token, *, full_anchor=False):
    """Structural producer boundary, independently validated, with no old type bypass."""
    if producer.anchor_config["enable_specimen_routing"] is not True:
        raise ValueError("candidate routing must be explicitly enabled")
    observation, anchor = producer.observe(raw)
    packed = pack_actor_state(observation, anchor, producer.contract)
    if packed.shape[0] != 1:
        raise ValueError("one public observation required")
    choices = residual_option_actions_from_state(
        raw, producer.anchor_config, producer.settings, make_explicit_residual_option_specs(options))
    if not np.array_equal(choices[0], anchor[0].detach().cpu().numpy()):
        raise ValueError("public producer anchor differs from full MDL-2 reconstruction")
    reference_request = choices[0] if full_anchor else reference.act(raw)
    bank = build_request_candidates(
        {"state_token": token, "reference_request": reference_request,
         "anchor_request": anchor[0].tolist(), "option_requests": choices[1:]},
        RoutingRequestSchema(producer.contract.replay.action_schema_id,
                             len(producer.contract.inputs.node_ids),
                             producer.anchor_config["max_specimen_transfer"], len(options) + 2))
    return observation, bank


def dynamic_candidate_submission(policy, observation, decision, *, current_state_token):
    """Verify the dynamic receipt; never replace the original float64 action."""
    if not isinstance(decision, CandidateDecision):
        raise TypeError("sealed CandidateDecision required")
    if current_state_token != decision.choice.state_token:
        raise ValueError("candidate state token is stale")
    verify_dynamic_evaluation(policy, observation, decision.evaluation)
    validate_choice_precision(decision.choice, decision.evaluation.candidates, replay_dtype=np.float64)
    # Existing segment/sampler contracts also require decoder stability at model precision.
    validate_choice_precision(decision.choice, decision.evaluation.candidates,
                              replay_dtype=decision.evaluation.inference_dtype)
    request = np.asarray(decision.choice.submitted_request, dtype=np.float64).copy()
    request.setflags(write=False)
    return request


def _decode_session_example(example, contract, dtype):
    observation, bank = decode_example(example, contract)
    if dtype == torch.float64:
        observation, anchor = _decode_actor_state(
            torch.tensor([example["actor_state"]], dtype=dtype), contract)
        if not torch.equal(anchor, torch.tensor([bank.requests[1]], dtype=dtype)):
            raise ValueError("example anchor differs from double-precision support")
        pack_actor_state(observation, anchor, contract)
    return observation, bank


def _failure_evidence(value):
    """Keep nonfinite numerical failures inspectable in the finite envelope format."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all().item():
            return {"nonfinite_tensor": {"dtype": str(tensor.dtype), "shape": tuple(tensor.shape),
                    "bytes_hex": tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes().hex()}}
        return tensor.clone()
    if isinstance(value, np.ndarray):
        return _failure_evidence(encode_arrays(value))
    if isinstance(value, np.generic):
        return _failure_evidence(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_float": value.hex()}
    if isinstance(value, dict):
        return {key: _failure_evidence(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_failure_evidence(item) for item in value)
    return copy.deepcopy(value)


def _capture_failure(getter):
    try:
        value = _failure_evidence(getter())
        state_digest(value)
        return value
    except BaseException as error:
        return {"capture_error": {"error_type": type(error).__name__, "message": str(error)}}


class DynamicCandidateSession:
    """One explicitly enabled episode; collection never performs an update.

    The producer must expose contract, anchor_config, settings, observe,
    unpack_raw and check_environment. Its public inputs are validated below;
    the caller remains responsible for their causal provenance.
    """

    def __init__(self, env, producer, reference, learner, *, enabled=False, options,
                 trajectory_id, split, selection, source_id):
        if enabled is not True or selection not in ("sample", "greedy", "reference", "anchor"):
            raise ValueError("explicit session enablement/selection required")
        if split not in ("preflight", "demonstration", "qualification", "training", "test"):
            raise ValueError("explicit session split required")
        if ((selection == "anchor" and split not in ("preflight", "test"))
                or (selection == "sample" and split not in ("preflight", "training"))
                or (split == "demonstration" and selection != "reference")
                or (split == "qualification" and selection not in ("reference", "greedy"))
                or (split == "training" and selection != "sample")):
            raise ValueError("collection mode conflicts with data split")
        if type(learner) not in (DynamicCandidatePPOKernel, DynamicCandidateImitationKernel):
            raise TypeError("explicit dynamic PPO or imitation kernel required")
        if learner._failure is not None:
            raise ValueError("failed kernel is terminal")
        if (not isinstance(getattr(producer, "contract", None), ReplayInputContract)
                or any(not callable(getattr(producer, method, None))
                       for method in ("observe", "unpack_raw", "check_environment"))):
            raise TypeError("explicit public producer contract and methods required")
        if not callable(getattr(reference, "act", None)):
            raise TypeError("explicit public reference required")
        reference_id = getattr(reference, "checkpoint_sha256", None)
        if (not isinstance(reference_id, str) or len(reference_id) != 64
                or any(c not in "0123456789abcdef" for c in reference_id)):
            raise ValueError("explicit reference SHA256 required")
        option_specs = make_explicit_residual_option_specs(options)
        if any(spec.group != "specimen_transfer" or not np.isfinite(spec.epsilon) for spec in option_specs[1:]):
            raise ValueError("finite specimen-only options required")
        if selection == "sample" and learner.sampling_rng is None:
            raise ValueError("sampling collection requires a private generator")
        if not all(isinstance(v, str) and v for v in (trajectory_id, source_id)):
            raise ValueError("explicit source and trajectory identity required")
        if learner.contract != producer.contract or env.t != 0:
            raise ValueError("compatible learner and fresh episode required")
        producer.check_environment(env)
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
        self._failure = None
        self.last_inference_seconds = None  # Timing is observational, not restorable stochastic state.
        self.manifest = {"format": "dynamic-candidate-session-v2", "contract": asdict(producer.contract),
                         "initial_token": self.initial_token, "initial_weights": self.initial_weights,
                         "initial_rng_sha256": state_digest(self.initial_rng),
                         "reference": reference.checkpoint_sha256, "options": copy.deepcopy(options),
                         "trajectory_id": trajectory_id, "source_id": source_id, "split": split,
                         "selection": selection, "horizon": env.config.episode_horizon,
                         "inference_dtype": str(self.dtype),
                         "submission_dtype": "float64_original_request",
                         "replay_precision": "integer_request_must_survive_inference_dtype",
                         "failure_policy": "terminal_session_evidence_never_resumable_no_refunds",
                         "producer_definition": evidence_digest({
                             "anchor_config": producer.anchor_config,
                             "settings": asdict(producer.settings)})}

    @property
    def dtype(self):
        return next(self.learner.policy.parameters()).dtype

    def _require_live_kernel(self):
        if self._failure is not None:
            raise ValueError("failed session is terminal; cannot restore to retry")
        if self.learner._failure is not None:
            raise ValueError("failed kernel is terminal; session cannot restore to retry")

    def _observe(self, raw):
        observation, anchor = self.producer.observe(raw)
        pack_actor_state(observation, anchor, self.learner.contract)
        return _cast_observation(observation, self.dtype), anchor.to(dtype=self.dtype)

    def _context(self, raw, token):
        observation, bank = dynamic_context_from_public(
            self.producer, self.reference, raw, self.options, token,
            full_anchor=self.selection == "anchor")
        return _cast_observation(observation, self.dtype), bank

    @property
    def closed(self):
        return self.index == self.boundary.episode_horizon

    def step(self, *, before_step):
        self._require_live_kernel()
        if not callable(before_step):
            raise TypeError("external nonrefundable before_step callback required")
        if self.closed or self.env.t != self.index:
            raise ValueError("closed or discontinuous episode")
        self.producer.check_environment(self.env)
        before_env = self.env.state_dict()
        if evidence_digest(before_env) != self.expected_token or self.learner.policy.snapshot_sha256() != self.initial_weights:
            raise ValueError("environment or episode behavior snapshot drift")
        sampler = self.learner.sampling_rng
        before_rng = None if sampler is None else sampler.get_state().clone()
        if state_digest(before_rng) != self.expected_rng:
            raise ValueError("sampling RNG drift between steps")
        before_policy = copy.deepcopy(self.learner.policy.state_dict())
        before_index, before_token, before_timing = self.index, self.expected_token, self.last_inference_seconds
        stage, debit_started, debit_acknowledged, step_started = "inference", False, False, False
        decision = action = step_result = None
        try:
            inference_start = time.perf_counter()
            raw = self.env.observation()
            obs, bank = self._context(raw, self.expected_token)
            evaluation = evaluate_dynamic_policy(self.learner.policy, obs, bank, self.learner.contract)
            if self.selection == "sample":
                decision = self.learner.decide(obs, bank)
            else:
                index = (bank.reference_class if self.selection == "reference" else bank.anchor_class
                         if self.selection == "anchor" else int(np.argmax(evaluation.log_probs)))
                decision = CandidateDecision(evaluation, choose_candidate(bank, index))
            if decision.evaluation != evaluation:
                raise ValueError("decision differs from evaluated public support")
            action = dynamic_candidate_submission(self.learner.policy, obs, decision, current_state_token=self.expected_token)
            inference_seconds = time.perf_counter() - inference_start
            example = public_example(obs, bank, self.learner.contract,
                                     split="test" if self.split == "preflight" else self.split,
                                     identity=f"{self.trajectory_id}/{self.index}")
            stage, debit_started = "before_step", True
            before_step()  # External durable budget is never refunded by rollback.
            debit_acknowledged = True
            self._require_live_kernel()
            stage, step_started = "environment_step", True
            next_raw, reward, done, info = self.env.step(action)
            step_result = {"next_raw": next_raw, "reward": reward, "done": done, "info": info}
            stage = "step_audit"
            if self.env.t != self.index + 1 or not np.array_equal(next_raw, self.env.observation()):
                raise ValueError("post-step environment/observation discontinuity")
            obs2, anchor2 = self._observe(next_raw)
            token = evidence_digest(self.env.state_dict())
            audit = audit_candidate_step(decision, action, next_observation=obs2, next_anchor=anchor2,
                                         next_state_token=token, raw_reward=reward, environment_done=done,
                                         info=info, boundary=self.boundary, source_id=self.source_id,
                                         trajectory_id=self.trajectory_id, step_index=self.index)
            event = {"audit": asdict(audit), "info": copy.deepcopy(info)}
            state_digest(encode_arrays(event))
            stage = "receipt_publication"
            returned = copy.deepcopy(event)
            next_rng = state_digest(None if sampler is None else sampler.get_state())
            self.events.append(event)
            self.examples.append(example)
            self.expected_token, self.index = token, self.index + 1
            self.expected_rng = next_rng
            self.last_inference_seconds = inference_seconds
            return returned
        except BaseException as error:
            # Latch first: evidence capture or rollback failure must never permit a retry.
            self._failure = {
                "error_type": type(error).__name__, "message": str(error), "stage": stage,
                "step_index": before_index, "debit_callback_started": debit_started,
                "debit_callback_acknowledged": debit_acknowledged, "environment_step_started": step_started,
                "resource_authority": "external_non_refundable_ledger",
                "scope": "session_terminal_campaign_failure_owned_by_caller",
                "partial_transaction": {}, "rollback_errors": {},
            }
            captures = {"environment": self.env.state_dict, "kernel": self.learner.state_dict,
                        "sampling_rng": lambda: None if sampler is None else sampler.get_state(),
                        "decision": lambda: None if decision is None else asdict(decision),
                        "submitted_request": lambda: action, "step_result": lambda: step_result}
            for name, getter in captures.items():
                self._failure["partial_transaction"][name] = _capture_failure(getter)
            rollback = {"environment": lambda: self.env.load_state_dict(before_env),
                        "policy": lambda: self.learner.policy.load_state_dict(before_policy),
                        "sampling_rng": lambda: None if sampler is None else sampler.set_state(before_rng)}
            for name, restore in rollback.items():
                try:
                    restore()
                except BaseException as rollback_error:
                    self._failure["rollback_errors"][name] = {
                        "error_type": type(rollback_error).__name__, "message": str(rollback_error)}
            del self.events[before_index:]
            del self.examples[before_index:]
            self.index, self.expected_token = before_index, before_token
            self.expected_rng = state_digest(before_rng)
            self.last_inference_seconds = before_timing
            raise

    def segment(self, gae_lambda):
        self._require_live_kernel()
        if not self.closed or self.selection != "sample" or self.split != "training":
            raise ValueError("only a complete sampled training episode can train")
        decisions = [restore_decision(e["audit"]["decision"], self.learner.contract) for e in self.events]
        records = [restore_record(e["audit"]["record"]) for e in self.events]
        return prepare_candidate_segment(decisions, records, self.learner.contract,
                                         behavior_sha256=self.initial_weights, bootstrap=None,
                                         gae_lambda=gae_lambda, max_steps=self.boundary.episode_horizon)

    def state_dict(self):
        failed = self._failure is not None or self.learner._failure is not None
        environment = _capture_failure(self.env.state_dict) if failed else self.env.state_dict()
        kernel = _capture_failure(self.learner.state_dict) if failed else self.learner.state_dict()
        return encode_arrays({"manifest": self.manifest, "environment": environment,
                              "kernel": kernel, "events": self.events,
                              "examples": self.examples, "index": self.index, "token": self.expected_token,
                              "initial_rng": self.initial_rng, "failure": self._failure})

    def _restore(self, saved):
        self._require_live_kernel()
        if set(saved) != {"manifest", "environment", "kernel", "events", "examples", "index", "token", "initial_rng", "failure"}:
            raise ValueError("session state fields differ")
        if saved["failure"] is not None:
            raise ValueError("failed session evidence cannot resume")
        state_digest(saved)
        state = decode_arrays(saved)
        if state["manifest"] != self.manifest:
            raise ValueError("session manifest mismatch")
        if state_digest(state["initial_rng"]) != self.manifest["initial_rng_sha256"]:
            raise ValueError("initial sampling RNG differs")
        self.learner.load_state_dict(state["kernel"])
        self.env.load_state_dict(state["environment"])
        self.producer.check_environment(self.env)
        if evidence_digest(self.env.state_dict()) != state["token"]:
            raise ValueError("environment checkpoint did not roundtrip")
        index = state["index"]
        if type(index) is not int or not 0 <= index <= self.boundary.episode_horizon or self.env.t != index:
            raise ValueError("session cursor mismatch")
        if len(state["events"]) != index or len(state["examples"]) != index:
            raise ValueError("session receipts missing")
        if self.learner.policy.snapshot_sha256() != self.initial_weights:
            raise ValueError("episode weights differ")
        token = self.initial_token
        previous = None
        sampler = None if self.initial_rng is None else torch.Generator(device="cpu")
        if sampler is not None:
            sampler.set_state(self.initial_rng)
        for i, (event, example) in enumerate(zip(state["events"], state["examples"])):
            decision = restore_decision(event["audit"]["decision"], self.learner.contract)
            record = restore_record(event["audit"]["record"])
            obs, bank = _decode_session_example(example, self.learner.contract, self.dtype)
            verify_dynamic_evaluation(self.learner.policy, obs, decision.evaluation)
            raw = self.producer.unpack_raw(obs)[0]
            public_raw = raw.astype(self.env.observation().dtype)
            if not np.array_equal(raw, public_raw):
                raise ValueError("restored public input loses raw observation precision")
            raw = public_raw
            _, rebuilt = self._context(raw, token)
            if rebuilt != bank:
                raise ValueError("restored candidates differ from public-state construction")
            if self.selection == "sample":
                expected = sample_candidate(decision.evaluation, generator=sampler)
            else:
                selected = (bank.reference_class if self.selection == "reference" else bank.anchor_class
                            if self.selection == "anchor" else int(np.argmax(decision.evaluation.log_probs)))
                expected = CandidateDecision(decision.evaluation, choose_candidate(bank, selected))
            obs2, anchor2 = _decode_actor_state(torch.tensor([record.next_state], dtype=self.dtype),
                                               self.learner.contract)
            rebuilt_audit = audit_candidate_step(
                decision, np.array(record.action, dtype=np.float64), next_observation=obs2, next_anchor=anchor2,
                next_state_token=record.next_state_token, raw_reward=record.raw_reward,
                environment_done=i + 1 == self.boundary.episode_horizon, info=event["info"],
                boundary=self.boundary, source_id=self.source_id, trajectory_id=self.trajectory_id, step_index=i)
            if (record.state_token != token or record.step_index != i or record.trajectory_id != self.trajectory_id
                    or record.source_id != self.source_id or record.origin != "trajectory"
                    or record.action != decision.choice.submitted_request
                    or record.state != decision.evaluation.actor_state
                    or record.raw_reward != -event["info"]["cost"]
                    or record.terminated != (i + 1 == self.boundary.episode_horizon) or record.truncated
                    or (previous is not None and previous != record.state)
                    or example["actor_state"] != record.state
                    or example["identity"] != f"{self.trajectory_id}/{i}"
                    or example["split"] != ("test" if self.split == "preflight" else self.split)
                    or example["candidates"] != asdict(decision.evaluation.candidates)
                    or decision != expected
                    or state_digest(encode_arrays(asdict(rebuilt_audit))) != state_digest(encode_arrays(event["audit"]))):
                raise ValueError("receipt lineage/behavior/reward mismatch")
            token, previous = record.next_state_token, record.next_state
        obs, anchor = self._observe(self.env.observation())
        packed = tuple(pack_actor_state(obs, anchor, self.learner.contract)[0].tolist())
        if token != state["token"] or (previous is not None and previous != packed):
            raise ValueError("restored endpoint mismatch")
        actual_rng = None if self.learner.sampling_rng is None else self.learner.sampling_rng.get_state()
        replayed_rng = None if sampler is None else sampler.get_state()
        if state_digest(actual_rng) != state_digest(replayed_rng):
            raise ValueError("sampling RNG does not reproduce from episode receipts")
        self.events, self.examples = copy.deepcopy(state["events"]), copy.deepcopy(state["examples"])
        self.index, self.expected_token = index, token
        self.expected_rng = state_digest(actual_rng)

    def load_state_dict(self, saved):
        self._require_live_kernel()
        candidate = copy.deepcopy(self)
        candidate._restore(copy.deepcopy(saved))
        self.__dict__.update(candidate.__dict__)

    def save(self, path):
        state = self.state_dict()
        if self._failure is None and self.learner._failure is None:
            copy.deepcopy(self)._restore(state)
        save_envelope(path, state)
