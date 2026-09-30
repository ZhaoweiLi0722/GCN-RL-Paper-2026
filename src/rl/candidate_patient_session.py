"""Audited public candidate collection and whole episode/kernel checkpoints."""

from __future__ import annotations

import copy
from dataclasses import asdict
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from src.rl.candidate_collection_boundary import CollectionBoundary, audit_candidate_step, candidate_submission, public_candidate_context
from src.rl.candidate_imitation import decode_example, public_example
from src.rl.candidate_rollout import (CandidateDecision, PolicyEvaluation, evaluate_candidate_policy,
                                     prepare_candidate_segment, sample_candidate, verify_behavior_evaluation)
from src.rl.networks import torch
from src.rl.patient_replay_collector import evidence_digest
from src.rl.prospective_adapter import _decode_actor_state, pack_actor_state
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import encode_arrays, decode_arrays, restore_record
from src.rl.residual_options import make_explicit_residual_option_specs, residual_option_actions_from_state
from src.rl.routing_candidate_contract import CandidateChoice, RequestCandidates, RoutingRequestSchema, choose_candidate


def context_from_public(producer, reference, raw, options, token, *, full_anchor=False):
    choices = residual_option_actions_from_state(raw, producer.anchor_config, producer.settings,
                                                 make_explicit_residual_option_specs(options))
    # A deterministic MDL-2 comparator must submit MDL-2, even when R4 aliases it.
    # Its receipt uses MDL-2 as reference; it never supplies training examples.
    reference_request = choices[0] if full_anchor else reference.act(raw)
    return public_candidate_context(producer, raw, reference_request=reference_request,
                                    option_requests=choices[1:], state_token=token,
                                    max_candidates=len(options) + 2)


def restore_decision(data, contract):
    evaluation = dict(data["evaluation"])
    if evaluation.pop("contract") != asdict(contract):
        raise ValueError("restored receipt contract differs")
    bank = evaluation.pop("candidates")
    candidates = RequestCandidates(RoutingRequestSchema(**bank["schema"]), bank["state_token"], bank["requests"])
    if state_digest(asdict(candidates)) != state_digest(bank):
        raise ValueError("restored request support differs")
    return CandidateDecision(PolicyEvaluation(contract=contract, candidates=candidates, **evaluation),
                             CandidateChoice(**data["choice"]))


def save_envelope(path, state):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    payload = {"state": state, "sha256": state_digest(state)}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="candidate-state-", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp)
        with open(temp, "rb") as handle:
            os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        os.unlink(temp)


def load_envelope(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if set(data) != {"state", "sha256"} or state_digest(data["state"]) != data["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    return data["state"]


class CandidatePatientSession:
    def __init__(self, env, producer, reference, learner, *, enabled=False, options,
                 trajectory_id, split, selection, source_id):
        if enabled is not True or selection not in ("sample", "greedy", "reference", "anchor"):
            raise ValueError("explicit session enablement/selection required")
        if split not in ("preflight", "demonstration", "qualification", "training", "test"):
            raise ValueError("explicit session split required")
        if ((selection == "anchor" and split not in ("preflight", "test"))
                or (selection == "sample" and split not in ("preflight", "training"))
                or (split in ("demonstration", "qualification") and selection != "reference")
                or (split == "training" and selection != "sample")):
            raise ValueError("collection mode conflicts with data split")
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
        self.last_inference_seconds = None  # Timing is observational, not restorable stochastic state.
        self.manifest = {"format": "candidate-patient-session-v1", "contract": asdict(producer.contract),
                         "initial_token": self.initial_token, "initial_weights": self.initial_weights,
                         "initial_rng_sha256": state_digest(self.initial_rng),
                         "reference": reference.checkpoint_sha256, "options": copy.deepcopy(options),
                         "trajectory_id": trajectory_id, "source_id": source_id, "split": split,
                         "selection": selection, "horizon": env.config.episode_horizon}

    @property
    def closed(self):
        return self.index == self.boundary.episode_horizon

    def step(self, *, before_step):
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
        try:
            inference_start = time.perf_counter()
            raw = self.env.observation()
            obs, bank = context_from_public(self.producer, self.reference, raw, self.options,
                                             self.expected_token, full_anchor=self.selection == "anchor")
            evaluation = evaluate_candidate_policy(self.learner.policy, obs, bank, self.learner.contract)
            if self.selection == "sample":
                decision = self.learner.decide(obs, bank)
            else:
                index = (bank.reference_class if self.selection == "reference" else bank.anchor_class
                         if self.selection == "anchor" else int(np.argmax(evaluation.log_probs)))
                decision = CandidateDecision(evaluation, choose_candidate(bank, index))
            action = candidate_submission(self.learner.policy, obs, decision, current_state_token=self.expected_token)
            inference_seconds = time.perf_counter() - inference_start
            example = public_example(obs, bank, self.learner.contract,
                                     split="test" if self.split == "preflight" else self.split,
                                     identity=f"{self.trajectory_id}/{self.index}")
            before_step()  # External durable budget is never refunded by rollback.
            next_raw, reward, done, info = self.env.step(action)
            if self.env.t != self.index + 1 or not np.array_equal(next_raw, self.env.observation()):
                raise ValueError("post-step environment/observation discontinuity")
            obs2, anchor2 = self.producer.observe(next_raw)
            token = evidence_digest(self.env.state_dict())
            audit = audit_candidate_step(decision, action, next_observation=obs2, next_anchor=anchor2,
                                         next_state_token=token, raw_reward=reward, environment_done=done,
                                         info=info, boundary=self.boundary, source_id=self.source_id,
                                         trajectory_id=self.trajectory_id, step_index=self.index)
            event = {"audit": asdict(audit), "info": copy.deepcopy(info)}
            state_digest(encode_arrays(event))
        except Exception:
            self.env.load_state_dict(before_env)
            if sampler is not None:
                sampler.set_state(before_rng)
            raise
        self.events.append(event)
        self.examples.append(example)
        self.expected_token, self.index = token, self.index + 1
        self.expected_rng = state_digest(None if sampler is None else sampler.get_state())
        self.last_inference_seconds = inference_seconds
        return copy.deepcopy(event)

    def segment(self, gae_lambda):
        if not self.closed or self.selection != "sample" or self.split != "training":
            raise ValueError("only a complete sampled training episode can train")
        decisions = [restore_decision(e["audit"]["decision"], self.learner.contract) for e in self.events]
        records = [restore_record(e["audit"]["record"]) for e in self.events]
        return prepare_candidate_segment(decisions, records, self.learner.contract,
                                         behavior_sha256=self.initial_weights, bootstrap=None,
                                         gae_lambda=gae_lambda, max_steps=self.boundary.episode_horizon)

    def state_dict(self):
        return encode_arrays({"manifest": self.manifest, "environment": self.env.state_dict(),
                              "kernel": self.learner.state_dict(), "events": self.events,
                              "examples": self.examples, "index": self.index, "token": self.expected_token,
                              "initial_rng": self.initial_rng})

    def _restore(self, saved):
        if set(saved) != {"manifest", "environment", "kernel", "events", "examples", "index", "token", "initial_rng"}:
            raise ValueError("session state fields differ")
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
            obs, bank = decode_example(example, self.learner.contract)
            verify_behavior_evaluation(self.learner.policy, obs, decision.evaluation)
            raw = self.producer.unpack_raw(obs)[0]
            _, rebuilt = context_from_public(self.producer, self.reference, raw, self.options, token,
                                               full_anchor=self.selection == "anchor")
            if rebuilt != bank:
                raise ValueError("restored candidates differ from public-state construction")
            if self.selection == "sample":
                expected = sample_candidate(decision.evaluation, generator=sampler)
            else:
                selected = (bank.reference_class if self.selection == "reference" else bank.anchor_class
                            if self.selection == "anchor" else int(np.argmax(decision.evaluation.log_probs)))
                expected = CandidateDecision(decision.evaluation, choose_candidate(bank, selected))
            obs2, anchor2 = _decode_actor_state(torch.tensor([record.next_state], dtype=torch.float32),
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
        obs, anchor = self.producer.observe(self.env.observation())
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
        candidate = copy.deepcopy(self)
        candidate._restore(copy.deepcopy(saved))
        self.__dict__.update(candidate.__dict__)

    def save(self, path):
        state = self.state_dict()
        copy.deepcopy(self)._restore(state)
        save_envelope(path, state)
